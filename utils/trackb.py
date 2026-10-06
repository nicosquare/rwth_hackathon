"""Track B: model-capacity experiments on the frozen Level 0 feature set.

Run from the repository root, one step at a time (each step saves its
predictions, so steps are resumable and independent):

    uv run python -m utils.trackb --step baselines   # baselines, ridge, lgb_4m (reference)
    uv run python -m utils.trackb --step b1          # GBMs at scale (LightGBM / XGBoost / CatBoost)
    uv run python -m utils.trackb --step b2          # Optuna search for the best GBM library
    uv run python -m utils.trackb --step b3          # tabular MLP
    uv run python -m utils.seq_model                 # B4 sequence model
    uv run python -m utils.trackb --step b5          # non-negative combination
    uv run python -m utils.trackb --step score       # validation_summary.csv / test_summary.csv

Every model uses the frozen ``baseline_v1`` design rows, the Level 0 folds and
the day-ahead contract: each fold trains on delivery days fully observed before
its first bid cutoff.  Predictions are saved as float32 kWh arrays in cache row
order under ``artifacts/trackb/preds/<fold>/<model>.npy``.  Model selection uses
the validation folds only; test metrics are written to a separate table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,4")  # shared server: idle GPUs only

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from utils.level0 import (
    FOLDS, LGB_PARAMS, RIDGE_COLUMNS, VALIDATION_FOLDS, baseline_predictions,
    cached_designs, fit_ridge, household_board, portfolio_metrics, subset, summarise,
)
from utils.modeling import DAY_AHEAD


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "trackb"
PRED_DIR = OUT / "preds"
FEATURE_SET = "baseline_v1"
THREADS = 16
CATEGORICAL = ["pv_status", "building_type", "heatpump_type"]
REFERENCE = "daily_weekly_ensemble"
BASELINE_MODEL = "lgb_4m"            # like-for-like reproduction of Level 0's lgb_global_l2
ALL_FOLDS = (*VALIDATION_FOLDS, "test")


# --------------------------------------------------------------------- data/io

_CACHE: dict[str, object] = {}


def designs() -> tuple[dict, dict, dict]:
    if not _CACHE:
        _CACHE["arrays"], _CACHE["history"], _CACHE["test"] = cached_designs(FEATURE_SET)
    return _CACHE["arrays"], _CACHE["history"], _CACHE["test"]


def fold_data(fold: str) -> tuple[dict, dict]:
    """(train, evaluate) rows with the Level 0 boundary rule."""
    _, history, test = designs()
    start, end = FOLDS[fold]
    delivery = history["index"].delivery_day
    train = subset(history, (delivery < DAY_AHEAD.cutoff(start).normalize()).to_numpy())
    evaluate = test if fold == "test" else subset(history, ((delivery >= start) & (delivery < end)).to_numpy())
    return train, evaluate


def eval_rows(fold: str) -> dict:
    """Evaluation rows of a fold (index, y, scale) without the training copy."""
    _, history, test = designs()
    if fold == "test":
        return test
    start, end = FOLDS[fold]
    delivery = history["index"].delivery_day
    return subset(history, ((delivery >= start) & (delivery < end)).to_numpy())


def save_pred(fold: str, name: str, values: np.ndarray) -> None:
    (PRED_DIR / fold).mkdir(parents=True, exist_ok=True)
    np.save(PRED_DIR / fold / f"{name}.npy", np.asarray(values, dtype=np.float32))


def load_preds(fold: str) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p) for p in sorted((PRED_DIR / fold).glob("*.npy"))}


def log_runtime(step: str, name: str, fold: str, seconds: float, **extra) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runtimes.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "model": name, "fold": fold,
                            "seconds": round(seconds, 1), **extra}) + "\n")


def sample_rows(n: int, cap: int | None, seed: int) -> np.ndarray:
    if cap is None or cap >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))


# ---------------------------------------------------------------------- models
# Every fitter maps (X_fit, y_fit_scaled, X_eval, seed) -> scaled predictions.

def fit_predict_lgb(X_fit, y_fit, X_eval, seed=42, params=None):
    model = lgb.LGBMRegressor(objective="l2", **{**LGB_PARAMS, "random_state": seed, **(params or {})})
    model.fit(X_fit, y_fit, categorical_feature=CATEGORICAL)
    return model.predict(X_eval)


XGB_PARAMS = {
    "device": "cuda", "tree_method": "hist", "objective": "reg:squarederror",
    "n_estimators": 600, "learning_rate": 0.05, "grow_policy": "lossguide", "max_leaves": 63,
    "max_depth": 0, "min_child_weight": 200, "subsample": 0.7, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "max_bin": 256, "nthread": THREADS,
}


def fit_predict_xgb(X_fit, y_fit, X_eval, seed=42, params=None):
    import xgboost as xgb
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed, **(params or {})})
    model.fit(X_fit.to_numpy(np.float32), y_fit)
    return model.predict(X_eval.to_numpy(np.float32))


CAT_PARAMS = {
    "task_type": "GPU", "devices": "0", "loss_function": "RMSE", "iterations": 1500,
    "learning_rate": 0.08, "depth": 8, "border_count": 254, "bootstrap_type": "Bernoulli",
    "subsample": 0.7, "l2_leaf_reg": 3.0, "thread_count": THREADS, "verbose": 0,
}


def fit_predict_cat(X_fit, y_fit, X_eval, seed=42, params=None):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(**{**CAT_PARAMS, "random_seed": seed, **(params or {})})
    model.fit(X_fit.to_numpy(np.float32), y_fit)
    return model.predict(X_eval.to_numpy(np.float32))


FITTERS = {"lgb": fit_predict_lgb, "xgb": fit_predict_xgb, "cat": fit_predict_cat}


def run_gbm(step: str, name: str, library: str, cap: int | None, seed: int = 42,
            params: dict | None = None, folds=ALL_FOLDS) -> None:
    for fold in folds:
        train, evaluate = fold_data(fold)
        rows = sample_rows(len(train["y"]), cap, seed if cap is None else 42)
        X_fit = train["X"].iloc[rows]
        y_fit = (train["y"] / train["scale"])[rows]
        started = time.perf_counter()
        pred = FITTERS[library](X_fit, y_fit, evaluate["X"], seed=seed, params=params)
        seconds = time.perf_counter() - started
        save_pred(fold, name, np.maximum(pred, 0) * evaluate["scale"])
        log_runtime(step, name, fold, seconds, train_rows=len(rows))
        print(f"  {name:<18} {fold:<10} {len(rows):>10,} rows  {seconds:7.0f}s", flush=True)
        del train, evaluate, X_fit


# ------------------------------------------------------------------------ steps

def step_baselines() -> None:
    """Persistence baselines, global Ridge and the Level 0 reference (4M-row LightGBM)."""
    for fold in ALL_FOLDS:
        train, evaluate = fold_data(fold)
        target = train["y"] / train["scale"]
        d1, w1 = train["X"].latest_same_slot.to_numpy(), train["X"].lag_d7.to_numpy()
        delta = d1 - w1
        weight = float(np.clip(np.dot(target - w1, delta) / np.dot(delta, delta), 0, 1))
        for name, pred in baseline_predictions(evaluate["X"], evaluate["scale"], weight).items():
            save_pred(fold, name, pred)
        rows = sample_rows(len(target), 4_000_000, 42)
        ridge, fill = fit_ridge(train["X"].iloc[rows], target[rows], RIDGE_COLUMNS)
        save_pred(fold, "ridge_global",
                  np.maximum(ridge.predict(evaluate["X"][RIDGE_COLUMNS].fillna(fill)), 0) * evaluate["scale"])
        print(f"  baselines {fold} done (ensemble weight {weight:.3f})", flush=True)
        del train, evaluate
    run_gbm("baselines", BASELINE_MODEL, "lgb", cap=4_000_000)


def step_b1() -> None:
    run_gbm("b1", "lgb_all", "lgb", cap=None)
    run_gbm("b1", "xgb_all", "xgb", cap=None)
    run_gbm("b1", "cat_all", "cat", cap=None)


def step_b1_seeds(library: str, base: str) -> None:
    """Two extra seeds for the best B1 library, then the seed average."""
    for seed in (7, 1234):
        run_gbm("b1", f"{base}_s{seed}", library, cap=None, seed=seed)
    for fold in ALL_FOLDS:
        preds = load_preds(fold)
        save_pred(fold, f"{base}_seedavg",
                  np.mean([preds[base], preds[f"{base}_s7"], preds[f"{base}_s1234"]], axis=0))


# ------------------------------------------------------------------------ B2

def _validation_objective(preds_by_fold: dict[str, np.ndarray], reference: dict) -> float:
    """Mean of (median household nMAE, portfolio WAPE), each relative to the reference model."""
    idx, y, p = [], [], []
    for fold in VALIDATION_FOLDS:
        ev = reference["eval"][fold]
        idx.append(ev["index"]); y.append(ev["y"]); p.append(preds_by_fold[fold])
    index, y, p = pd.concat(idx, ignore_index=True), np.concatenate(y), np.concatenate(p)
    board = household_board(index, y, {"cand": p, REFERENCE: reference["blend"]}, REFERENCE)
    nmae = board["nmae_cand"].median()
    wape = portfolio_metrics(index, y, {"cand": p}).wape.item()
    return 0.5 * nmae / reference["nmae"] + 0.5 * wape / reference["wape"]


def step_b2(library: str, n_trials: int, search_rows: int) -> None:
    import optuna
    evals, trains = {}, {}
    for fold in VALIDATION_FOLDS:
        train, evaluate = fold_data(fold)
        rows = sample_rows(len(train["y"]), search_rows, 42)
        trains[fold] = (train["X"].iloc[rows].reset_index(drop=True), (train["y"] / train["scale"])[rows])
        evals[fold] = evaluate
        del train
    blend = np.concatenate([load_preds(f)[REFERENCE] for f in VALIDATION_FOLDS])
    base = {f: load_preds(f)[BASELINE_MODEL] for f in VALIDATION_FOLDS}
    reference = {"eval": evals, "blend": blend, "nmae": 1.0, "wape": 1.0}
    index = pd.concat([evals[f]["index"] for f in VALIDATION_FOLDS], ignore_index=True)
    y = np.concatenate([evals[f]["y"] for f in VALIDATION_FOLDS])
    p = np.concatenate([base[f] for f in VALIDATION_FOLDS])
    reference["nmae"] = household_board(index, y, {"b": p, REFERENCE: blend}, REFERENCE)["nmae_b"].median()
    reference["wape"] = portfolio_metrics(index, y, {"b": p}).wape.item()

    def suggest(trial):
        if library == "xgb":
            return {"learning_rate": trial.suggest_float("learning_rate", .02, .2, log=True),
                    "max_leaves": trial.suggest_int("max_leaves", 31, 511, log=True),
                    "min_child_weight": trial.suggest_float("min_child_weight", 20, 2000, log=True),
                    "subsample": trial.suggest_float("subsample", .5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 300, 2000, log=True)}
        if library == "lgb":
            return {"learning_rate": trial.suggest_float("learning_rate", .02, .2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
                    "subsample": trial.suggest_float("subsample", .5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 300, 2000, log=True)}
        return {"learning_rate": trial.suggest_float("learning_rate", .03, .3, log=True),
                "depth": trial.suggest_int("depth", 6, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", .1, 30, log=True),
                "subsample": trial.suggest_float("subsample", .5, 1.0),
                "iterations": trial.suggest_int("iterations", 500, 3000, log=True)}

    def objective(trial):
        params = suggest(trial)
        started = time.perf_counter()
        preds = {}
        for fold in VALIDATION_FOLDS:
            X_fit, y_fit = trains[fold]
            pred = FITTERS[library](X_fit, y_fit, evals[fold]["X"], seed=42, params=params)
            preds[fold] = np.maximum(pred, 0) * evals[fold]["scale"]
        value = _validation_objective(preds, reference)
        print(f"  trial {trial.number:3d}: {value:.4f} ({time.perf_counter() - started:.0f}s) {params}", flush=True)
        return value

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    # Trial 0 = current defaults so the search is anchored on the reference.
    defaults = {"xgb": {k: XGB_PARAMS[k] for k in ("learning_rate", "max_leaves", "min_child_weight",
                                                    "subsample", "colsample_bytree", "reg_lambda", "n_estimators")},
                "lgb": {k: LGB_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_child_samples",
                                                    "subsample", "colsample_bytree", "reg_lambda", "n_estimators")},
                "cat": {k: CAT_PARAMS[k] for k in ("learning_rate", "depth", "l2_leaf_reg", "subsample",
                                                    "iterations")}}[library]
    study.enqueue_trial(defaults)
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    OUT.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(OUT / f"b2_{library}_trials.csv", index=False)
    best = study.best_params
    (OUT / f"b2_{library}_best.json").write_text(json.dumps(
        {"best_params": best, "best_value": study.best_value, "default_value": study.trials[0].value,
         "search_rows": search_rows, "n_trials": n_trials,
         "objective": "0.5*median_household_nMAE/ref + 0.5*portfolio_WAPE/ref on pooled validation (ref=lgb_4m)",
         "seconds": round(time.perf_counter() - started)}, indent=2) + "\n")
    print(f"best {study.best_value:.4f} (defaults {study.trials[0].value:.4f}): {best}", flush=True)
    del trains, evals
    run_gbm("b2", f"{library}_tuned", library, cap=None, params=best)


# ------------------------------------------------------------------------ B3

def mlp_matrices(train_X: pd.DataFrame, eval_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Standardised numeric features + missing indicators + one-hot categoricals.

    Statistics (mean/std/fill) come from the training rows only.
    """
    numeric = [c for c in train_X.columns if c not in CATEGORICAL]
    mean = train_X[numeric].mean()
    std = train_X[numeric].std().replace(0, 1).fillna(1)
    with_missing = [c for c in numeric if train_X[c].isna().any()]

    def encode(X: pd.DataFrame) -> np.ndarray:
        parts = [((X[numeric] - mean) / std).fillna(0).to_numpy(np.float32),
                 X[with_missing].isna().to_numpy(np.float32)]
        for c in CATEGORICAL:
            codes = X[c].fillna(-1).to_numpy()
            parts.append(np.stack([(codes == v).astype(np.float32) for v in (-1, 0, 1, 2)], axis=1))
        return np.concatenate(parts, axis=1)

    return encode(train_X), encode(eval_X)


def train_mlp(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, seed: int = 42,
              epochs: int = 8, batch: int = 8192, holdout_share: float = .05) -> tuple[np.ndarray, dict]:
    import torch
    from torch import nn
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    # Early stopping on the last 5% of training rows (rows are in delivery-day order,
    # so the holdout is the most recent training period, still before the cutoff).
    n_hold = int(len(y_fit) * holdout_share)
    Xt = torch.from_numpy(X_fit[:-n_hold]).to(device); yt = torch.from_numpy(y_fit[:-n_hold].astype(np.float32)).to(device)
    Xh = torch.from_numpy(X_fit[-n_hold:]).to(device); yh = torch.from_numpy(y_fit[-n_hold:].astype(np.float32)).to(device)
    width = 512
    model = nn.Sequential(
        nn.Linear(X_fit.shape[1], width), nn.GELU(), nn.Dropout(.1),
        nn.Linear(width, width), nn.GELU(), nn.Dropout(.1),
        nn.Linear(width, width // 2), nn.GELU(),
        nn.Linear(width // 2, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(yt) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), device=device, generator=gen)
        for i in range(len(yt) // batch):
            j = perm[i * batch:(i + 1) * batch]
            loss = nn.functional.mse_loss(model(Xt[j]).squeeze(1), yt[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            hold = float(torch.cat([((model(Xh[k:k + 65536]).squeeze(1) - yh[k:k + 65536]) ** 2)
                                    for k in range(0, len(yh), 65536)]).mean())
        history.append(hold)
        print(f"    epoch {epoch}: holdout mse {hold:.4f}", flush=True)
        if hold < best:
            best, best_state = hold, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    Xe = torch.from_numpy(X_eval)
    with torch.no_grad():
        pred = torch.cat([model(Xe[k:k + 262144].to(device)).squeeze(1).cpu()
                          for k in range(0, len(Xe), 262144)]).numpy()
    return pred, {"holdout_mse": history, "best_epoch": int(np.argmin(history))}


def step_b3() -> None:
    for fold in ALL_FOLDS:
        train, evaluate = fold_data(fold)
        started = time.perf_counter()
        X_fit, X_eval = mlp_matrices(train["X"], evaluate["X"])
        pred, info = train_mlp(X_fit, (train["y"] / train["scale"]).astype(np.float32), X_eval)
        seconds = time.perf_counter() - started
        save_pred(fold, "mlp", np.maximum(pred, 0) * evaluate["scale"])
        log_runtime("b3", "mlp", fold, seconds, train_rows=len(X_fit), **info)
        print(f"  mlp {fold}: {seconds:.0f}s, best epoch {info['best_epoch']}", flush=True)
        del train, evaluate, X_fit, X_eval


# ------------------------------------------------------------------------ B5

def step_b5(members: list[str]) -> None:
    """Non-negative stacking weights; cross-fitted across the two validation folds.

    Validation numbers for the stack are honest: weights fitted on val_summer
    score val_winter and vice versa.  The weights used for the test fold are
    fitted on both validation folds pooled.
    """
    vals = {f: load_preds(f) for f in VALIDATION_FOLDS}
    ys = {f: eval_rows(f)["y"] for f in VALIDATION_FOLDS}

    def fit(folds):
        A = np.concatenate([np.stack([vals[f][m] for m in members], axis=1) for f in folds]).astype(np.float64)
        b = np.concatenate([ys[f] for f in folds]).astype(np.float64)
        w, _ = nnls(A, b)
        return w

    weights = {}
    for target, source in (("val_summer", "val_winter"), ("val_winter", "val_summer")):
        w = fit([source]); weights[f"fit_on_{source}"] = dict(zip(members, w.round(4)))
        save_pred(target, "stack", np.stack([vals[target][m] for m in members], axis=1) @ w)
        save_pred(target, "mean_ensemble", np.mean([vals[target][m] for m in members], axis=0))
    w = fit(list(VALIDATION_FOLDS)); weights["fit_on_validation"] = dict(zip(members, w.round(4)))
    test = load_preds("test")
    save_pred("test", "stack", np.stack([test[m] for m in members], axis=1) @ w)
    save_pred("test", "mean_ensemble", np.mean([test[m] for m in members], axis=0))
    (OUT / "b5_weights.json").write_text(json.dumps({"members": members, "weights": weights}, indent=2,
                                                    default=float) + "\n")
    print(json.dumps(weights, indent=2, default=float))


# --------------------------------------------------------------------- diagnose

def _mse(pred, y) -> float:
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(np.mean((pred[ok] - y[ok]) ** 2))


def _outlier_share(pred, y, top: float = .999) -> float:
    """Share of squared error from rows whose target is above the given quantile."""
    ok = np.isfinite(pred) & np.isfinite(y)
    err, y = (pred[ok] - y[ok]) ** 2, y[ok]
    return float(err[y > np.quantile(y, top)].sum() / err.sum())


def step_diagnose(fold: str = "val_summer") -> None:
    """Reference MSEs on the networks' own early-stopping slices (scaled units)."""
    import torch
    from utils.seq_model import GpuData, fit_predict, history_grid, training_pairs
    from utils.level0 import HIST_START
    report = {}

    # --- B3 slice: last 5% of the fold's training rows (cache order = delivery-day order).
    train, _ = fold_data(fold)
    target = (train["y"] / train["scale"]).astype(np.float32)
    n_hold = int(len(target) * .05)
    cut = len(target) - n_hold
    y_hold = target[cut:]
    hold_days = train["index"].delivery_day.iloc[cut:]
    sm = train["X"].slot_mean_7d.to_numpy()[cut:]
    rows = sample_rows(cut, 4_000_000, 42)
    lgb_pred = fit_predict_lgb(train["X"].iloc[rows], target[rows], train["X"].iloc[cut:])
    X_fit, X_hold = mlp_matrices(train["X"], train["X"].iloc[cut:])
    batch = X_fit[:8192]
    mlp_pred, info = train_mlp(X_fit, target, X_hold)
    report["b3_slice"] = {
        "rows": int(n_hold), "days": f"{hold_days.min().date()}..{hold_days.max().date()}",
        "target_quantiles_50_99_999_max": [float(np.quantile(y_hold, q)) for q in (.5, .99, .999)] + [float(y_hold.max())],
        "mse_constant_train_mean": _mse(np.full_like(y_hold, target[:cut].mean()), y_hold),
        "mse_slot_mean_7d": _mse(sm, y_hold), "slot_mean_7d_coverage": float(np.isfinite(sm).mean()),
        "mse_lgb_4m": _mse(lgb_pred, y_hold),
        "mse_mlp_retrained": _mse(mlp_pred, y_hold), "mlp_logged_holdout_curve": info["holdout_mse"],
        "outlier_share_top0.1pct_constant": _outlier_share(np.full_like(y_hold, target[:cut].mean()), y_hold),
        "outlier_share_top0.1pct_lgb": _outlier_share(lgb_pred, y_hold),
        "mlp_batch_feature_mean_range": [float(batch.mean(0).min()), float(batch.mean(0).max())],
        "mlp_batch_feature_std_range": [float(batch.std(0).min()), float(batch.std(0).max())],
        "mlp_constant_columns_in_batch": int((batch.std(0) == 0).sum()), "mlp_n_columns": int(batch.shape[1]),
    }
    print(json.dumps(report["b3_slice"], indent=2), flush=True)
    del X_fit, X_hold

    # --- B4 slice: last 42 training days, all observed slots of those (day, household) pairs.
    arrays, history, _ = designs()
    data = GpuData(arrays, torch.device("cuda:0"))
    start, _ = FOLDS[fold]
    last_day = int((DAY_AHEAD.cutoff(start).normalize() - HIST_START) / pd.Timedelta("1D"))
    d, h = training_pairs(data, last_day)
    hold = d >= d.max() - 41
    m = arrays["matrix"]
    scale = data.scale_np[d, h]
    T = m[d[:, None] * 96 + np.arange(96), h[:, None]] / scale[:, None]
    const = np.nanmean(T[~hold])
    values, known = history_grid(m, d[hold], h[hold])
    with np.errstate(invalid="ignore"):
        sm_seq = (values[:, 1:8] * known[:, 1:8]).sum(1) / known[:, 1:8].sum(1) / scale[hold][:, None]
    cnn_pred, cnn_info = fit_predict(data, d, h, d[hold], h[hold])
    # Batch sanity: target alignment and input statistics.
    dt, ht = torch.from_numpy(d[hold][:512]).cuda(), torch.from_numpy(h[hold][:512]).cuda()
    grid, mask, ctx, log_scale, tgt = data.batch(dt, ht)
    aligned = np.allclose(tgt.cpu().numpy(), T[hold][:512], equal_nan=True, rtol=1e-5)
    # LightGBM on the cache rows of the same 42 days (trained on cache rows before them).
    day_idx = ((history["index"].delivery_day - HIST_START) / pd.Timedelta("1D")).to_numpy()
    hold_rows = (day_idx >= d.max() - 41) & (day_idx < last_day)
    fit_rows = day_idx < d.max() - 41
    ht_y = history["y"][hold_rows] / history["scale"][hold_rows]
    fr = np.flatnonzero(fit_rows); fr = fr[sample_rows(len(fr), 4_000_000, 42)]
    lgb_seq = fit_predict_lgb(history["X"].iloc[fr], history["y"][fr] / history["scale"][fr],
                              history["X"][hold_rows])
    # Map CNN predictions onto those cache rows for a like-for-like comparison.
    key = pd.MultiIndex.from_arrays([d[hold], h[hold]])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([day_idx[hold_rows].astype(np.int64),
                                                      history["index"].hh.to_numpy()[hold_rows]]))
    slot = history["index"].horizon_step.to_numpy()[hold_rows] - 1
    cnn_on_rows = np.where(pos >= 0, cnn_pred[np.maximum(pos, 0), slot], np.nan)
    report["b4_slice"] = {
        "pairs": int(hold.sum()), "observed_targets": int(np.isfinite(T[hold]).sum()),
        "target_quantiles_50_99_999_max": [float(np.nanquantile(T[hold], q)) for q in (.5, .99, .999)]
                                          + [float(np.nanmax(T[hold]))],
        "mse_constant_train_mean": _mse(np.full_like(T[hold], const), T[hold]),
        "mse_slot_mean_7d": _mse(sm_seq, T[hold]),
        "mse_cnn_retrained": _mse(cnn_pred, T[hold]), "cnn_logged_holdout_curve": cnn_info["holdout_mse"],
        "batch_target_aligned_with_matrix": bool(aligned),
        "batch_grid_mean_std": [float(grid.mean()), float(grid.std())],
        "batch_mask_share": float(mask.mean()),
        "batch_d1_mask_share_slots_ge_40": float(mask[:, 0, 40:].mean()),
        "batch_context_mean_range": [float(ctx.mean(0).min()), float(ctx.mean(0).max())],
        "cache_rows_in_slice": int(hold_rows.sum()),
        "on_cache_rows": {"mse_constant": _mse(np.full_like(ht_y, const), ht_y),
                          "mse_slot_mean_7d": _mse(history["X"].slot_mean_7d.to_numpy()[hold_rows], ht_y),
                          "mse_lgb_4m": _mse(lgb_seq, ht_y), "mse_cnn": _mse(cnn_on_rows, ht_y),
                          "cnn_mapped_share": float((pos >= 0).mean())},
    }
    print(json.dumps(report["b4_slice"], indent=2), flush=True)
    (OUT / "diagnose.json").write_text(json.dumps(report, indent=2) + "\n")


# ---------------------------------------------------------------------- scoring

def _score(index, y, preds) -> pd.DataFrame:
    board = household_board(index, y, preds, REFERENCE)
    hh = summarise(board, list(preds), REFERENCE).set_index("model")
    port = portfolio_metrics(index, y, preds).set_index("model")[["wape", "bias", "mae"]] \
        .rename(columns=lambda c: f"portfolio_{c}")
    return hh.join(port).reset_index()


def step_score() -> None:
    folds = {f: (eval_rows(f), load_preds(f)) for f in ALL_FOLDS}
    common = set.intersection(*(set(p) for _, p in folds.values() if p))
    tables = []
    for fold in VALIDATION_FOLDS:
        ev, p = folds[fold]
        tables.append(_score(ev["index"], ev["y"], {m: p[m] for m in sorted(common)}).assign(fold=fold))
    index = pd.concat([folds[f][0]["index"] for f in VALIDATION_FOLDS], ignore_index=True)
    y = np.concatenate([folds[f][0]["y"] for f in VALIDATION_FOLDS])
    pooled = {m: np.concatenate([folds[f][1][m] for f in VALIDATION_FOLDS]) for m in sorted(common)}
    tables.insert(0, _score(index, y, pooled).assign(fold="validation"))
    validation = pd.concat(tables, ignore_index=True)
    validation.to_csv(OUT / "validation_summary.csv", index=False)
    ev, p = folds["test"]
    test = _score(ev["index"], ev["y"], {m: p[m] for m in sorted(common)}).assign(
        fold="test (not used for selection)")
    test.to_csv(OUT / "test_summary.csv", index=False)
    cols = ["model", "median_household_nmae", f"share_households_beating_{REFERENCE}",
            "portfolio_wape", "portfolio_bias"]
    for name, table in [("validation (pooled)", validation[validation.fold == "validation"]),
                        ("val_summer", validation[validation.fold == "val_summer"]),
                        ("val_winter", validation[validation.fold == "val_winter"]),
                        ("test - not used for selection", test)]:
        print(f"--- {name}")
        print(table[cols].sort_values("portfolio_wape").round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                        choices=["baselines", "b1", "b1_seeds", "b2", "b3", "b5", "score", "diagnose"])
    parser.add_argument("--library", default="xgb", choices=["lgb", "xgb", "cat"])
    parser.add_argument("--base", default="xgb_all")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--search-rows", type=int, default=2_000_000)
    parser.add_argument("--members", nargs="+", default=["xgb_tuned", "mlp", "seq_cnn", "ridge_global"])
    args = parser.parse_args()
    started = time.perf_counter()
    if args.step == "baselines": step_baselines()
    elif args.step == "b1": step_b1()
    elif args.step == "b1_seeds": step_b1_seeds(args.library, args.base)
    elif args.step == "b2": step_b2(args.library, args.trials, args.search_rows)
    elif args.step == "b3": step_b3()
    elif args.step == "b5": step_b5(args.members)
    elif args.step == "score": step_score()
    elif args.step == "diagnose": step_diagnose()
    print(f"step {args.step} complete in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
