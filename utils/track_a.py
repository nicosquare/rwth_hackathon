"""Track A: information experiments on the validation folds only.

Run from the repository root with ``uv run python -m utils.track_a``.

Each variant changes one thing relative to the Level 0 recipe (global LightGBM,
L2, 4M sampled training rows) and is scored on the pooled and per-fold
validation rows.  The test window is never touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.level0 import (
    LGB_PARAMS, MAX_TRAIN_ROWS, ROOT, VALIDATION_FOLDS, FOLDS, cached_designs,
    baseline_predictions, household_board, portfolio_metrics, subset, summarise,
)
from utils.modeling import DAY_AHEAD


OUTPUT_DIR = ROOT / "artifacts" / "track_a"
REFERENCE = "daily_weekly_ensemble"
GROUPS = {
    "holidays": ["is_holiday", "lag_d2_is_holiday", "lag_d7_is_holiday", "christmas_period"],
    "morning": ["morning_trend", "temp_morning_delta"],
    "portfolio": ["pf_level_4h", "pf_level_1d", "pf_trend_1d_7d", "pf_morning_trend"],
}
NEW_COLUMNS = [c for cols in GROUPS.values() for c in cols]


def bias_corrected(evaluate: dict, prediction: np.ndarray, strength: float,
                   window: tuple[int, int] = (2, 8), min_days: int = 3) -> np.ndarray:
    """Scale each delivery day by the recent portfolio actual/forecast ratio.

    Only days D-8..D-2 inside the evaluated (out-of-sample) window are used:
    D-2 is the latest day fully observed before the 10:00 UTC D-1 cutoff.
    """
    days = evaluate["index"].delivery_day.to_numpy()
    totals = (pd.DataFrame({"day": days, "actual": evaluate["y"], "pred": prediction})
              .groupby("day")[["actual", "pred"]].sum())
    factors = {}
    for day in totals.index:
        lo, hi = day - pd.Timedelta(days=window[1]), day - pd.Timedelta(days=window[0])
        past = totals.loc[(totals.index >= lo) & (totals.index <= hi)]
        ratio = past.actual.sum() / past.pred.sum() if len(past) >= min_days else 1.0
        factors[day] = 1 + strength * (ratio - 1)
    return prediction * pd.Series(days).map(factors).to_numpy()


def score(evaluate: dict, preds: dict[str, np.ndarray], fold: str) -> pd.DataFrame:
    board = household_board(evaluate["index"], evaluate["y"], preds, REFERENCE)
    households = summarise(board, list(preds), REFERENCE).set_index("model")
    portfolio = portfolio_metrics(evaluate["index"], evaluate["y"], preds).set_index("model")
    return households.join(portfolio[["wape", "bias", "mae"]]).assign(fold=fold).reset_index()


def run(output_dir: Path = OUTPUT_DIR, quick: bool = False) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {**LGB_PARAMS, **({"n_estimators": 150} if quick else {})}
    max_rows = 1_000_000 if quick else MAX_TRAIN_ROWS
    t0 = time.perf_counter()
    _, history, _ = cached_designs("trackA_v1")
    base_columns = [c for c in history["X"].columns if c not in NEW_COLUMNS]
    variants = {  # name -> (feature columns, sample-weight power of the scale)
        "base": (base_columns, 0),
        **{f"+{g}": (base_columns + cols, 0) for g, cols in GROUPS.items()},
        "+all": (base_columns + NEW_COLUMNS, 0),
        "+all, kWh weight": (base_columns + NEW_COLUMNS, 1),
        "+all, kWh^2 weight": (base_columns + NEW_COLUMNS, 2),
    }
    delivery = history["index"].delivery_day
    rows, evaluated = [], {}
    for fold in VALIDATION_FOLDS:
        start, end = FOLDS[fold]
        train = subset(history, (delivery < DAY_AHEAD.cutoff(start).normalize()).to_numpy())
        evaluate = subset(history, ((delivery >= start) & (delivery < end)).to_numpy())
        target = train["y"] / train["scale"]
        d1, w1 = train["X"].latest_same_slot.to_numpy(), train["X"].lag_d7.to_numpy()
        weight = float(np.clip(np.dot(target - w1, d1 - w1) / np.dot(d1 - w1, d1 - w1), 0, 1))
        preds = {REFERENCE: baseline_predictions(evaluate["X"], evaluate["scale"], weight)[REFERENCE]}
        sample = np.sort(np.random.default_rng(42).choice(len(target), min(max_rows, len(target)), replace=False))
        for name, (columns, power) in variants.items():
            started = time.perf_counter()
            sample_weight = train["scale"][sample] ** power if power else None
            model = fit_lgb_weighted(train["X"].iloc[sample][columns], target[sample], params, sample_weight)
            preds[name] = np.maximum(model.predict(evaluate["X"][columns]), 0) * evaluate["scale"]
            print(f"  {fold} {name}: {time.perf_counter() - started:.0f}s", flush=True)
        for strength in (.5, 1.0):
            preds[f"+all, bias corr {strength:g}"] = bias_corrected(evaluate, preds["+all"], strength)
        rows.append(score(evaluate, preds, fold))
        evaluated[fold] = (evaluate, preds)
        del train

    pooled = {k: np.concatenate([evaluated[f][0][k] for f in VALIDATION_FOLDS]) for k in ("y", "scale")}
    pooled["index"] = pd.concat([evaluated[f][0]["index"] for f in VALIDATION_FOLDS], ignore_index=True)
    pooled_preds = {n: np.concatenate([evaluated[f][1][n] for f in VALIDATION_FOLDS]) for n in evaluated[VALIDATION_FOLDS[0]][1]}
    rows.append(score(pooled, pooled_preds, "validation"))
    results = pd.concat(rows, ignore_index=True)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps({
        "lightgbm": params, "max_train_rows": max_rows, "groups": GROUPS,
        "feature_set": "trackA_v1", "execution_mode": "quick" if quick else "full",
        "note": "validation folds only; test untouched",
    }, indent=2) + "\n")
    cols = ["median_household_nmae", f"share_households_beating_{REFERENCE}", "wape", "bias"]
    print(results.pivot(index="model", columns="fold", values=cols).round(4).to_string())
    print(f"complete in {time.perf_counter() - t0:.0f}s")
    return results


def fit_lgb_weighted(X: pd.DataFrame, target: np.ndarray, params: dict, sample_weight: np.ndarray | None):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective="l2", **params)
    categorical = [c for c in ("pv_status", "building_type", "heatpump_type") if c in X.columns]
    model.fit(X, target, sample_weight=sample_weight, categorical_feature=categorical)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir, args.quick)


if __name__ == "__main__":
    main()
