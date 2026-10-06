"""Level 0: one global household model that works for most households.

Run from the repository root with ``uv run python -m utils.level0``.

Differences to the earlier model ladder (removed; see commit 542b2eb) that matter
for household accuracy:

* Real day-ahead information set (``utils.modeling.DAY_AHEAD``): the forecast
  for UTC delivery day D uses only data before 10:00 UTC on D-1, ahead of the
  12:00 local EPEX gate closure.
* Target-aligned seasonal lags.  For a target slot the model sees the same
  quarter-hour on previous days and weeks, where known at the cutoff (D-1 only
  for slots before 10:00 UTC; D-2 and older always).  The ladder only used
  origin-anchored scalars, which hide the recent load shape from the model.
* Scale-free learning.  Every target and lag is divided by the household's mean
  load over the 28 days before the data cutoff, so one global estimator serves small
  and large households alike and is not dominated by the biggest consumers.
* Expanding-window folds over all history since autumn 2020.  Two validation
  folds (summer and winter) select the model; the test fold is scored once.
* Household-first scoring.  Each household is scored on identical rows for all
  models and we report the distribution of per-household skill, not only the
  portfolio total.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge

from utils.data import load_household_metadata, load_weather_data, scan_meter_data
from utils.modeling import DAY_AHEAD, regression_metrics


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "level0"
# The global model needs no fixed cohort, so training uses every household-day
# from autumn 2020 onwards (two extra winters compared with the model ladder).
HIST_START = pd.Timestamp("2020-09-01", tz="UTC")
FIRST_DAY = HIST_START + pd.Timedelta(days=35)  # 4 weeks of weekly lags + margin
END = pd.Timestamp("2024-02-28", tz="UTC")
# Expanding-window folds: each stage trains on all delivery days whose targets
# were already observed at the bid cutoff of its first evaluated day.
# Validation covers a full year so model selection sees every season, and each
# fold has at least one winter in training.  The test window matches notebook 02.
FOLDS = {
    "val_summer": (pd.Timestamp("2022-05-29", tz="UTC"), pd.Timestamp("2022-11-28", tz="UTC")),
    "val_winter": (pd.Timestamp("2022-11-28", tz="UTC"), pd.Timestamp("2023-05-29", tz="UTC")),
    "test": (pd.Timestamp("2023-05-29", tz="UTC"), END),
}
VALIDATION_FOLDS = ("val_summer", "val_winter")
SCALE_DAYS = 28
MAX_TRAIN_ROWS = 4_000_000
LGB_PARAMS = {
    "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
    "min_child_samples": 200, "subsample": 0.7, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 42,
    # Shared server (load ~100): a bounded thread count beats n_jobs=-1, which
    # starts one thread per core and contends with everyone else's jobs.
    "n_jobs": min(16, len(os.sched_getaffinity(0))), "verbosity": -1,
}


# --------------------------------------------------------------------------- data

def load_arrays() -> dict[str, object]:
    """Dense (time, household) arrays for all 410 households; NaN = no reading."""
    raw = (scan_meter_data()
           .filter((pl.col("timestamp_utc") >= HIST_START.to_pydatetime()) &
                   (pl.col("timestamp_utc") < END.to_pydatetime()))
           .select("Household_ID", "timestamp_utc", "kWh_received_Total")
           .collect())
    ids = sorted(raw["Household_ID"].unique().to_list())
    n_steps = int((END - HIST_START) / pd.Timedelta("15min"))
    matrix = np.full((n_steps, len(ids)), np.nan, dtype=np.float32)
    hh = pd.Categorical(raw["Household_ID"].to_pandas(), categories=ids).codes
    ts = pd.to_datetime(raw["timestamp_utc"].to_pandas(), utc=True)
    pos = ((ts - HIST_START) / pd.Timedelta("15min")).astype(int).to_numpy()
    matrix[pos, hh] = raw["kWh_received_Total"].to_numpy().astype(np.float32)

    meta = load_household_metadata().to_pandas().set_index("Household_ID").reindex(ids)
    static_fields = {
        "pv_status": meta["PV_Status"].fillna("unknown"),
        "building_type": meta["Survey_Building_Type"].fillna("unknown").astype(str),
        "heatpump_type": meta["Survey_HeatPump_Installation_Type"].fillna("unknown").astype(str),
        "living_area": pd.to_numeric(meta["Survey_Building_LivingArea"], errors="coerce"),
        "residents": pd.to_numeric(meta["Survey_Building_Residents"], errors="coerce"),
    }
    static = np.stack([pd.Categorical(v).codes.astype(np.float32) if v.dtype == object
                       else v.to_numpy(dtype=np.float32) for v in static_fields.values()], axis=1)
    static[static < 0] = np.nan

    # Past-only temperature at the household's station (aggregated up to the data cutoff).
    weather = load_weather_data().to_pandas()
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    hourly = pd.date_range(HIST_START, END, freq="1h", inclusive="left")
    temp = np.full((len(hourly), len(ids)), np.nan, dtype=np.float32)
    stations = meta["Weather_ID"].astype(str).to_numpy()
    for station in np.unique(stations):
        series = (weather.loc[weather.Weather_ID.astype(str) == station]
                  .set_index("timestamp_utc").reindex(hourly)["Temperature_avg_hourly"])
        temp[:, stations == station] = pd.to_numeric(series, errors="coerce").to_numpy(np.float32)[:, None]

    return {"ids": ids, "matrix": matrix, "static": static,
            "static_names": list(static_fields), "pv_status": meta["PV_Status"].fillna("unknown").to_numpy(),
            "temperature": temp}


def day_index(ts: pd.Timestamp) -> int:
    return int((ts - HIST_START) / pd.Timedelta("1D"))


def days_between(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.arange(day_index(start), day_index(end))


# ----------------------------------------------------------------------- features

def _window_mean(cs: np.ndarray, cnt: np.ndarray, end: np.ndarray, length: int,
                 min_share: float = .5) -> np.ndarray:
    """Mean of matrix rows [end - length, end) per household, NaN if too sparse."""
    total = cs[end] - cs[end - length]
    count = cnt[end] - cnt[end - length]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count < min_share * length] = np.nan
    return out


def build_design(arrays: dict, days: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows ordered (day, horizon, household). Returns features, target and scale.

    ``days`` are delivery days.  Only observations before the day-ahead data
    cutoff (``DAY_AHEAD``: 10:00 UTC on D-1) are used; any lag that would fall
    after the cutoff is NaN rather than silently shifted.
    """
    m = arrays["matrix"]; n_hh = m.shape[1]; D = len(days)
    filled = np.nan_to_num(m, nan=0.0).astype(np.float64)
    cs = np.vstack([np.zeros((1, n_hh)), np.cumsum(filled, axis=0)])
    cnt = np.vstack([np.zeros((1, n_hh)), np.cumsum(np.isfinite(m), axis=0)])
    delivery_start = days * 96
    cutoff = delivery_start - DAY_AHEAD.gap_steps                     # first unavailable step
    scale = _window_mean(cs, cnt, cutoff, SCALE_DAYS * 96)            # (D, N)
    scale = np.where(scale > 1e-3, scale, np.nan)
    pos = delivery_start[:, None] + np.arange(96)[None, :]            # (D, 96)

    def lag(k: int) -> np.ndarray:                                    # (D, 96, N)
        values = m[pos - k]
        return np.where(((pos - k) < cutoff[:, None])[:, :, None], values, np.nan)

    s3 = scale[:, None, :]
    feats: dict[str, np.ndarray] = {}
    # Same slot on D-1 is only known for slots before the cutoff hour (NaN otherwise);
    # D-2 and older are always known, so slot statistics use days 2..8.
    day_lags = np.stack([lag(96 * k) for k in range(1, 9)])            # (8, D, 96, N)
    week_lags = np.stack([lag(672 * k) for k in range(1, 5)])          # (4, D, 96, N)
    for k in (1, 2, 3, 7):
        feats[f"lag_d{k}"] = day_lags[k - 1] / s3
    feats["lag_w2"] = week_lags[1] / s3
    feats["latest_same_slot"] = np.where(np.isfinite(day_lags[0]), day_lags[0], day_lags[1]) / s3
    with np.errstate(invalid="ignore"):
        feats["slot_mean_7d"] = np.nanmean(day_lags[1:], axis=0) / s3
        feats["slot_median_7d"] = np.nanmedian(day_lags[1:], axis=0) / s3
        feats["slot_std_7d"] = np.nanstd(day_lags[1:], axis=0) / s3
        feats["slot_mean_4w"] = np.nanmean(week_lags, axis=0) / s3
    # Hourly-smoothed same slot on D-2 (trailing 1 h, always before the cutoff).
    feats["lag_d2_hour"] = np.mean(np.stack([lag(192 + j) for j in range(4)]), axis=0) / s3
    # Recent level up to the cutoff, relative to the 28-day scale.
    for name, length in (("level_4h", 16), ("level_1d", 96), ("level_7d", 672)):
        feats[name] = np.broadcast_to((_window_mean(cs, cnt, cutoff, length) / scale)[:, None, :], (D, 96, n_hh))
    feats["log_scale"] = np.broadcast_to(np.log(scale)[:, None, :], (D, 96, n_hh))

    temp = arrays["temperature"]
    cutoff_hour = cutoff // 4
    temp_24h = np.stack([np.nanmean(temp[h - 24:h], axis=0) for h in cutoff_hour])
    temp_7d = np.stack([np.nanmean(temp[h - 168:h], axis=0) for h in cutoff_hour])
    feats["temp_past_24h"] = np.broadcast_to(temp_24h[:, None, :], (D, 96, n_hh))
    feats["temp_trend"] = np.broadcast_to((temp_24h - temp_7d)[:, None, :], (D, 96, n_hh))

    ts = HIST_START + pd.to_timedelta(pos.reshape(-1) * 15, unit="min")
    local = ts.tz_convert("Europe/Berlin")
    calendar = {
        "lead_steps": np.tile(np.arange(96) + DAY_AHEAD.gap_steps, D),
        "local_slot": local.hour * 4 + local.minute // 15,
        "weekday": local.weekday,
        "doy_sin": np.sin(2 * np.pi * local.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * local.dayofyear / 365.25),
    }
    for name, values in calendar.items():
        feats[name] = np.broadcast_to(np.asarray(values, dtype=np.float32).reshape(D, 96)[:, :, None], (D, 96, n_hh))
    feats["lag_day_is_weekend_mismatch"] = np.broadcast_to(
        ((np.asarray(local.weekday).reshape(D, 96) >= 5) !=
         (np.asarray((local - pd.Timedelta("2D")).weekday).reshape(D, 96) >= 5)).astype(np.float32)[:, :, None],
        (D, 96, n_hh))
    for j, name in enumerate(arrays["static_names"]):
        feats[name] = np.broadcast_to(arrays["static"][None, None, :, j], (D, 96, n_hh))

    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in feats.items()})
    y = m[pos].reshape(-1)
    scale_rows = np.broadcast_to(s3, (D, 96, n_hh)).reshape(-1)
    return X, y, scale_rows


def row_index(arrays: dict, days: np.ndarray) -> pd.DataFrame:
    n_hh = len(arrays["ids"])
    delivery_day = HIST_START + pd.to_timedelta(days, unit="D")
    return pd.DataFrame({
        "delivery_day": np.repeat(delivery_day, 96 * n_hh),
        "horizon_step": np.tile(np.repeat(np.arange(1, 97), n_hh), len(days)),
        "hh": np.tile(np.arange(n_hh), 96 * len(days)),
    })


def prepare(arrays: dict, days: np.ndarray, chunk: int = 60) -> dict[str, object]:
    """Design rows for ``days`` that can be scored, built in chunks to bound memory.

    A row is kept only where the target, the scale and both persistence lags
    exist, so every model, baselines included, is scored on identical rows.
    """
    parts = []
    for i in range(0, len(days), chunk):
        block = days[i:i + chunk]
        with warnings.catch_warnings():  # all-NaN look-back windows are expected; they stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            X, y, scale = build_design(arrays, block)
        mask = (np.isfinite(y) & np.isfinite(scale) & X.latest_same_slot.notna().to_numpy()
                & X.lag_d7.notna().to_numpy())
        parts.append((X[mask].reset_index(drop=True), y[mask], scale[mask],
                      row_index(arrays, block)[mask].reset_index(drop=True)))
    return {"X": pd.concat([p[0] for p in parts], ignore_index=True),
            "y": np.concatenate([p[1] for p in parts]),
            "scale": np.concatenate([p[2] for p in parts]),
            "index": pd.concat([p[3] for p in parts], ignore_index=True)}


def subset(data: dict, rows: np.ndarray) -> dict[str, object]:
    return {"X": data["X"][rows].reset_index(drop=True), "y": data["y"][rows],
            "scale": data["scale"][rows], "index": data["index"][rows].reset_index(drop=True)}


# ------------------------------------------------------------------------- models

def baseline_predictions(X: pd.DataFrame, scale: np.ndarray, weight: float) -> dict[str, np.ndarray]:
    # "previous_day" is the latest same slot known at bid time: D-1 for slots
    # before the cutoff hour, D-2 otherwise.
    d1, w1 = X.latest_same_slot.to_numpy() * scale, X.lag_d7.to_numpy() * scale
    return {"previous_day": d1, "previous_week": w1,
            "daily_weekly_ensemble": weight * d1 + (1 - weight) * w1,
            "slot_mean_7d": X.slot_mean_7d.to_numpy() * scale,
            "slot_median_7d": X.slot_median_7d.to_numpy() * scale}


def household_board(index: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray],
                    reference: str) -> pd.DataFrame:
    """Per-household MAE for every model on identical rows, plus skill vs reference."""
    frame = pd.DataFrame({"hh": index.hh.to_numpy(), "actual": y})
    for name, p in preds.items():
        frame[name] = np.abs(p - y)
    out = frame.groupby("hh").agg(n=("actual", "size"), mean_load=("actual", "mean"),
                                  **{name: (name, "mean") for name in preds}).reset_index()
    for name in preds:
        out[f"skill_{name}"] = 1 - out[name] / out[reference]
        out[f"nmae_{name}"] = out[name] / out.mean_load
    return out


def summarise(board: pd.DataFrame, models: list[str], reference: str) -> pd.DataFrame:
    rows = []
    for name in models:
        rows.append({"model": name,
                     "median_household_nmae": board[f"nmae_{name}"].median(),
                     "p90_household_nmae": board[f"nmae_{name}"].quantile(.9),
                     "mean_household_mae": board[name].mean(),
                     f"median_skill_vs_{reference}": board[f"skill_{name}"].median(),
                     f"share_households_beating_{reference}": (board[f"skill_{name}"] > 0).mean(),
                     "households": len(board)})
    return pd.DataFrame(rows).sort_values("median_household_nmae")


def portfolio_metrics(index: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Sum over households present on identical rows; same households for all models."""
    keys = index[["delivery_day", "horizon_step"]]
    rows = []
    for name, p in preds.items():
        frame = keys.assign(actual=y, prediction=p)
        agg = frame.groupby(["delivery_day", "horizon_step"]).sum().reset_index()
        rows.append({"model": name, **regression_metrics(agg)})
    return pd.DataFrame(rows).sort_values("mae")


def fit_lgb(X: pd.DataFrame, target: np.ndarray, objective: str, params: dict) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective=objective, **params)
    model.fit(X, target, categorical_feature=["pv_status", "building_type", "heatpump_type"])
    return model


def fit_ridge(X: pd.DataFrame, target: np.ndarray, columns: list[str]) -> tuple[Ridge, pd.Series]:
    fill = X[columns].median()
    model = Ridge(alpha=1.0).fit(X[columns].fillna(fill), target)
    return model, fill


RIDGE_COLUMNS = ["latest_same_slot", "lag_d2", "lag_d3", "lag_d7", "lag_w2", "slot_mean_7d",
                 "slot_median_7d", "slot_mean_4w", "lag_d2_hour", "level_1d", "level_7d", "level_4h"]


# --------------------------------------------------------------------------- run

def run_fold(train: dict, evaluate: dict, objectives: tuple[str, ...], params: dict,
             max_rows: int, seed: int = 42) -> tuple[dict[str, np.ndarray], float, pd.DataFrame | None]:
    """Fit every candidate on ``train`` and predict ``evaluate`` (kWh)."""
    target = train["y"] / train["scale"]
    # Ensemble weight by clipped least squares on the training rows only.
    d1, w1 = train["X"].latest_same_slot.to_numpy(), train["X"].lag_d7.to_numpy()
    delta = d1 - w1
    weight = float(np.clip(np.dot(target - w1, delta) / np.dot(delta, delta), 0, 1))
    preds = baseline_predictions(evaluate["X"], evaluate["scale"], weight)

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(target), size=min(max_rows, len(target)), replace=False))
    X_fit, y_fit = train["X"].iloc[rows], target[rows]
    ridge, fill = fit_ridge(X_fit, y_fit, RIDGE_COLUMNS)
    preds["ridge_global"] = np.maximum(ridge.predict(evaluate["X"][RIDGE_COLUMNS].fillna(fill)), 0) * evaluate["scale"]
    importance = None
    for objective in objectives:
        started = time.perf_counter()
        model = fit_lgb(X_fit, y_fit, objective, params)
        preds[f"lgb_global_{objective}"] = np.maximum(model.predict(evaluate["X"]), 0) * evaluate["scale"]
        print(f"      lgb {objective}: {len(rows):,} rows, {time.perf_counter() - started:.0f}s", flush=True)
        if objective == "l2":
            importance = pd.DataFrame({"feature": X_fit.columns,
                                       "gain": model.booster_.feature_importance("gain")})
    return preds, weight, importance


def run(output_dir: Path = ARTIFACT_DIR, quick: bool = False,
        objectives: tuple[str, ...] = ("l2", "l1")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {**LGB_PARAMS, **({"n_estimators": 150} if quick else {})}
    max_rows = 1_000_000 if quick else MAX_TRAIN_ROWS
    t0 = time.perf_counter()
    print("[1/4] loading arrays", flush=True)
    arrays = load_arrays()
    ids, pv = np.asarray(arrays["ids"]), arrays["pv_status"]

    print("[2/4] building design rows", flush=True)
    # A row's features only use data before its own cutoff, so building once and
    # slicing rows by delivery day is equivalent to building per fold.
    history = prepare(arrays, days_between(FIRST_DAY, FOLDS["test"][0]))
    test_data = prepare(arrays, days_between(*FOLDS["test"]))
    delivery = history["index"].delivery_day
    print(f"      {len(history['y']):,} pre-test rows, {len(test_data['y']):,} test rows "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    results, weights, importance = {}, {}, None
    for step, (fold, (start, end)) in enumerate(FOLDS.items(), 1):
        print(f"[3/4] fold {fold}: train < {DAY_AHEAD.cutoff(start).normalize().date()}, evaluate {start.date()} -> {end.date()}", flush=True)
        # Targets of training rows must be observed before the first evaluated
        # cutoff (D-1 10:00 UTC), so the day before the fold is excluded.
        train = subset(history, (delivery < DAY_AHEAD.cutoff(start).normalize()).to_numpy())
        evaluate = test_data if fold == "test" else subset(history, ((delivery >= start) & (delivery < end)).to_numpy())
        preds, weights[fold], fold_importance = run_fold(train, evaluate, objectives, params, max_rows)
        if fold == "test":
            importance = fold_importance
        results[fold] = (evaluate, preds)
        del train

    print(f"[4/4] scoring ({time.perf_counter() - t0:.0f}s)", flush=True)
    reference = "daily_weekly_ensemble"

    def score(evaluate, preds, label):
        board = household_board(evaluate["index"], evaluate["y"], preds, reference)
        board.insert(0, "Household_ID", ids[board.hh])
        board.insert(1, "PV_Status", pv[board.hh])
        return (board.drop(columns="hh").assign(fold=label),
                summarise(board, list(preds), reference).assign(fold=label),
                portfolio_metrics(evaluate["index"], evaluate["y"], preds).assign(fold=label))

    val_eval = {k: np.concatenate([results[f][0][k] for f in VALIDATION_FOLDS])
                for k in ("y", "scale")}
    val_eval["index"] = pd.concat([results[f][0]["index"] for f in VALIDATION_FOLDS], ignore_index=True)
    val_preds = {name: np.concatenate([results[f][1][name] for f in VALIDATION_FOLDS])
                 for name in results[VALIDATION_FOLDS[0]][1]}
    scored = {fold: score(*results[fold], fold) for fold in FOLDS}
    scored["validation"] = score(val_eval, val_preds, "validation")

    # Procurement sums household forecasts, and only mean forecasts add up to the
    # portfolio mean.  The L1 (median) model wins household MAE but under-forecasts
    # the portfolio, so it is reported as a diagnostic and is not eligible.
    val_summary = scored["validation"][1].set_index("model")
    eligible = [name for name in ("ridge_global", "lgb_global_l2") if name in val_summary.index]
    winner = min(eligible, key=lambda name: val_summary.loc[name, "median_household_nmae"])

    summary = pd.concat([v[1] for v in scored.values()], ignore_index=True)
    portfolio = pd.concat([v[2] for v in scored.values()], ignore_index=True)
    for fold in ("validation", *FOLDS):
        print(f"--- {fold}")
        print(scored[fold][1].drop(columns="fold").round(3).to_string(index=False))
        print(scored[fold][2].drop(columns="fold").round(3).to_string(index=False))
    print(f"validation-selected model: {winner}")

    board = scored["test"][0]
    by_pv = pd.concat([summarise(g, [winner, reference, "previous_day"], reference)
                       .assign(PV_Status=status) for status, g in board.groupby("PV_Status")])
    test_eval, test_preds = results["test"]
    by_horizon = (pd.DataFrame({"horizon_step": test_eval["index"].horizon_step.to_numpy(),
                                **{n: np.abs(test_preds[n] - test_eval["y"])
                                   for n in (winner, reference, "previous_day")}})
                  .groupby("horizon_step").mean().reset_index())
    print(by_pv.round(3).to_string(index=False))

    pd.concat([v[0] for k, v in scored.items() if k in ("validation", "test")], ignore_index=True) \
        .to_parquet(output_dir / "household_metrics.parquet", index=False)
    summary.to_csv(output_dir / "household_summary.csv", index=False)
    portfolio.to_csv(output_dir / "portfolio_metrics.csv", index=False)
    by_pv.to_csv(output_dir / "pv_summary.csv", index=False)
    by_horizon.to_csv(output_dir / "horizon_mae.csv", index=False)
    if importance is not None:
        importance.sort_values("gain", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps({
        "contract": DAY_AHEAD.to_dict(), "history_start": HIST_START.isoformat(), "first_delivery_day": FIRST_DAY.isoformat(),
        "folds": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in FOLDS.items()},
        "fold_scheme": "expanding window: train on delivery days fully observed before the first evaluated bid cutoff",
        "scale_days": SCALE_DAYS, "lightgbm": params, "max_train_rows": max_rows,
        "ensemble_previous_day_weight": weights, "selected_model": winner,
        "selection_rule": "median household nMAE on pooled validation folds among mean-forecast models",
        "households_in_data": len(ids), "execution_mode": "quick" if quick else "full",
    }, indent=2) + "\n")
    print(f"complete in {time.perf_counter() - t0:.0f}s: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--objectives", nargs="+", default=["l2", "l1"], choices=["l2", "l1"])
    args = parser.parse_args()
    run(args.output_dir, args.quick, tuple(args.objectives))


if __name__ == "__main__":
    main()
