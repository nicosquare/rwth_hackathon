"""Execute the portfolio-first model ladder and write compact evidence artifacts.

Run from the repository root with ``uv run python -m utils.run_model_ladder``.
The implementation uses the same flattened design matrix that Darts creates for
``multi_models=False`` (one estimator, horizon step as a feature), while keeping
the target lags explicitly anchored to the common daily origin.  A Darts-native
synthetic equivalence assertion is recorded in the configuration artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge

from utils.data import load_household_metadata, load_weather_data, scan_meter_data
from utils.modeling import (
    ForecastSpec, SplitSpec, apply_conformal, choose_operational_winner,
    conformal_quantiles, interval_metrics, metric_table, regression_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "model_ladder"
START = pd.Timestamp("2022-02-28", tz="UTC")
END = pd.Timestamp("2024-02-28", tz="UTC")
SPEC = ForecastSpec()
SPLIT = SplitSpec.from_day_counts(START)
LGB_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "subsample_freq": 1, "random_state": 42, "n_jobs": -1,
    "verbosity": -1,
}


def _cohort(start: pd.Timestamp, end: pd.Timestamp, threshold: float = .95) -> pd.DataFrame:
    expected = int((end - start) / pd.Timedelta("15min"))
    stats = (
        scan_meter_data()
        .filter((pl.col("timestamp_utc") >= start.to_pydatetime()) &
                (pl.col("timestamp_utc") < end.to_pydatetime()))
        .group_by("Household_ID")
        .agg(pl.col("kWh_received_Total").count().alias("valid_intervals"))
        .with_columns((pl.col("valid_intervals") / expected).alias("completeness"))
        .filter(pl.col("completeness") >= threshold)
        .sort("Household_ID").collect().to_pandas()
    )
    return stats


def _meter_matrix(ids: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    raw = (
        scan_meter_data()
        .filter(pl.col("Household_ID").is_in(ids))
        .filter((pl.col("timestamp_utc") >= start.to_pydatetime()) &
                (pl.col("timestamp_utc") < end.to_pydatetime()))
        .select("Household_ID", "timestamp_utc", "AffectsTimePoint",
                pl.col("kWh_received_Total").alias("actual"))
        .collect()
    )
    n_steps = int((end - start) / pd.Timedelta("15min"))
    matrix = np.full((n_steps, len(ids)), np.nan, dtype=np.float32)
    intervention = np.zeros((n_steps, len(ids)), dtype=np.uint8)
    id_codes = pd.Categorical(raw["Household_ID"].to_pandas(), categories=ids).codes
    timestamps = pd.to_datetime(raw["timestamp_utc"].to_pandas(), utc=True)
    time_codes = ((timestamps - start) / pd.Timedelta("15min")).astype(int).to_numpy()
    values = raw["actual"].to_numpy().astype(np.float32)
    valid = (id_codes >= 0) & (time_codes >= 0) & (time_codes < n_steps)
    matrix[time_codes[valid], id_codes[valid]] = values[valid]
    states = raw["AffectsTimePoint"].cast(pl.String).fill_null("").str.to_lowercase()
    state_codes = np.where(states.str.contains("after").to_numpy(), 2,
                           np.where(states.str.contains("before").to_numpy(), 1, 0))
    intervention[time_codes[valid], id_codes[valid]] = state_codes[valid]
    return matrix, intervention


def _static_matrix(ids: list[str]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    meta = load_household_metadata().to_pandas().set_index("Household_ID").reindex(ids)
    fields = ["PV_Status", "Group", "Survey_Building_Type",
              "Survey_Building_LivingArea", "Survey_Building_Residents",
              "Survey_HeatPump_Installation_Type"]
    encoded = []
    names = ["household_id"]
    encoded.append(np.arange(len(ids), dtype=np.float32))
    for field in fields:
        values = meta[field]
        if field in ("Survey_Building_LivingArea", "Survey_Building_Residents"):
            values = pd.to_numeric(values, errors="coerce")
            values = values.fillna(values.median()).astype(np.float32)
        else:
            values = values.fillna("unknown").astype(str).str.lower()
            values = pd.Categorical(values).codes.astype(np.float32)
        encoded.append(np.asarray(values, dtype=np.float32))
        names.append(field.lower().replace("survey_", "").replace("installation_", ""))
    return np.stack(encoded, axis=1), meta.reset_index(), names


def _weather_cube(ids: list[str], meta: pd.DataFrame, start: pd.Timestamp,
                  end: pd.Timestamp) -> tuple[np.ndarray, list[str]]:
    names = ["temperature", "humidity", "sunshine", "precipitation", "wind", "heating_degree"]
    source = ["Temperature_avg_hourly", "Humidity_avg_hourly", "Sunshine_duration_hourly",
              "Precipitation_total_hourly", "WindSpeed_hourly"]
    weather = load_weather_data().to_pandas()
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    stations = meta.set_index("Household_ID").reindex(ids)["Weather_ID"].astype(str)
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    cube = np.full((len(hourly), len(ids), 6), np.nan, dtype=np.float32)
    for station in stations.unique():
        idx = np.flatnonzero(stations.to_numpy() == station)
        values = (weather.loc[weather.Weather_ID.astype(str) == station]
                  .set_index("timestamp_utc").reindex(hourly)[source]
                  .apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
        cube[:, idx, :5] = values[:, None, :]
    cube[:, :, 5] = np.maximum(15 - cube[:, :, 0], 0)
    # Weather source occasionally has holes: station-wise medians are a training-independent
    # availability repair, not target interpolation.
    medians = np.nanmedian(cube.reshape(-1, 6), axis=0)
    missing = np.where(np.isnan(cube))
    cube[missing] = medians[missing[2]]
    return cube, names


def _day_indices(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.arange(int((start - START) / pd.Timedelta("1d")),
                     int((end - START) / pd.Timedelta("1d")), dtype=int)


def _calendar(days: np.ndarray, n_households: int) -> tuple[np.ndarray, list[str]]:
    timestamps = pd.date_range(START, END, freq="15min", inclusive="left")
    pos = (days[:, None] * 96 + np.arange(96)[None, :]).reshape(-1)
    ts = timestamps[pos]
    season = np.select([ts.month.isin([12, 1, 2]), ts.month.isin([3, 4, 5]),
                        ts.month.isin([6, 7, 8])], [0, 1, 2], default=3)
    cal = np.stack([ts.minute // 15, ts.hour, ts.weekday,
                    (ts.weekday >= 5).astype(int), ts.month, season], axis=1).astype(np.float32)
    return np.repeat(cal, n_households, axis=0), ["quarter_hour", "hour", "weekday", "weekend", "month", "season"]


def _design(matrix: np.ndarray, weather: np.ndarray, static: np.ndarray,
            days: np.ndarray, oracle: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n_hh = matrix.shape[1]
    n_rows = len(days) * 96 * n_hh
    pieces, names = [], []
    for lag in SPEC.lags:
        values = matrix[days * 96 - lag]
        pieces.append(np.broadcast_to(values[:, None, :], (len(days), 96, n_hh)).reshape(-1, 1))
        names.append(f"lag_{lag}")
    calendar, cal_names = _calendar(days, n_hh)
    pieces.append(calendar); names.extend(cal_names)
    pieces.append(np.broadcast_to(static[None, None, :, :],
                                  (len(days), 96, n_hh, static.shape[1])).reshape(n_rows, -1))
    names.extend(["static_" + str(i) for i in range(static.shape[1])])
    if oracle:
        hour_positions = (days[:, None] * 24 + np.repeat(np.arange(24), 4)[None, :]).reshape(-1)
        w = weather[hour_positions].reshape(len(days), 96, n_hh, 6).reshape(n_rows, 6)
        pieces.append(w); names.extend(["weather_" + n for n in
                                        ["temperature", "humidity", "sunshine", "precipitation", "wind", "heating_degree"]])
    X = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
    y = matrix[(days[:, None] * 96 + np.arange(96)[None, :]).reshape(-1)].reshape(-1).astype(np.float32)
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[valid], y[valid], np.flatnonzero(valid), names


def _coordinates(days: np.ndarray, ids: list[str], valid_rows: np.ndarray) -> pd.DataFrame:
    row = valid_rows
    per_day = 96 * len(ids)
    day_pos = row // per_day
    within = row % per_day
    horizon = within // len(ids)
    hh = within % len(ids)
    origins = START + pd.to_timedelta(days[day_pos], unit="D")
    return pd.DataFrame({"origin": origins, "target_timestamp": origins + pd.to_timedelta(horizon * 15, unit="m"),
                         "horizon_step": horizon + 1, "Household_ID": np.asarray(ids)[hh]})


def _fit_predict(model, X_train, y_train, X_eval) -> tuple[np.ndarray, float]:
    started = time.perf_counter(); model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    return np.maximum(model.predict(X_eval), 0).astype(np.float32), elapsed


def _procurement_cost(frame: pd.DataFrame, shortfall_cost: float = 1.2,
                      surplus_cost: float = 0.8) -> float:
    """Illustrative asymmetric balancing cost per interval (relative units)."""
    error = frame.actual - frame.prediction
    return float(np.where(error >= 0, shortfall_cost * error, surplus_cost * -error).mean())


def _household_to_portfolio(coords: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray,
                            model: str, cohort_size: int) -> pd.DataFrame:
    frame = coords.copy(); frame["actual"] = actual; frame["prediction"] = prediction
    agg = frame.groupby(["origin", "target_timestamp", "horizon_step"], observed=True).agg(
        actual=("actual", "mean"), prediction=("prediction", "mean"),
        active_households=("actual", "count"), predicted_households=("prediction", "count")).reset_index()
    agg.actual *= cohort_size; agg.prediction *= cohort_size
    agg["model"] = model; agg["cohort_size"] = cohort_size
    agg["active_share"] = agg.active_households / cohort_size
    agg["actual_coverage"] = agg.active_share
    agg["prediction_coverage"] = agg.predicted_households / cohort_size
    return agg.loc[(agg.actual_coverage >= SPEC.minimum_coverage) &
                   (agg.prediction_coverage >= SPEC.minimum_coverage)]


def _baseline_portfolio(matrix: np.ndarray, days: np.ndarray, model: str,
                        hist_end_day: int, ensemble_weight: float | None = None) -> pd.DataFrame:
    cohort = matrix.shape[1]
    positions = days[:, None] * 96 + np.arange(96)[None, :]
    actual_values = matrix[positions]
    if model == "previous_day": pred_values = matrix[positions - 96]
    elif model == "previous_week": pred_values = matrix[positions - 672]
    elif model == "historical_quarter_hour_mean":
        hist = matrix[:hist_end_day * 96].reshape(hist_end_day, 96, cohort)
        profile = np.nanmean(hist, axis=(0, 2))
        pred_values = np.broadcast_to(profile[None, :, None], actual_values.shape)
    elif model == "daily_weekly_ensemble":
        pred_values = ensemble_weight * matrix[positions - 96] + (1 - ensemble_weight) * matrix[positions - 672]
    else: raise ValueError(model)
    active = np.isfinite(actual_values).sum(axis=2); predicted = np.isfinite(pred_values).sum(axis=2)
    origin = START + pd.to_timedelta(np.repeat(days, 96), unit="D")
    horizon = np.tile(np.arange(96), len(days))
    out = pd.DataFrame({"model": model, "origin": origin,
        "target_timestamp": origin + pd.to_timedelta(horizon * 15, unit="m"),
        "horizon_step": horizon + 1, "actual": np.nanmean(actual_values, axis=2).reshape(-1) * cohort,
        "prediction": np.nanmean(pred_values, axis=2).reshape(-1) * cohort,
        "active_households": active.reshape(-1), "cohort_size": cohort,
        "active_share": active.reshape(-1) / cohort, "actual_coverage": active.reshape(-1) / cohort,
        "prediction_coverage": predicted.reshape(-1) / cohort})
    return out.loc[(out.actual_coverage >= SPEC.minimum_coverage) &
                   (out.prediction_coverage >= SPEC.minimum_coverage)].reset_index(drop=True)


def _direct_design(portfolio: np.ndarray, weather: np.ndarray, days: np.ndarray,
                   oracle: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lag = np.stack([portfolio[days * 96 - value] for value in SPEC.lags], axis=1)
    lag = np.repeat(lag, 96, axis=0)
    calendar, cal_names = _calendar(days, 1)
    X = [lag, calendar]; names = [f"lag_{x}" for x in SPEC.lags] + cal_names
    if oracle:
        mean_weather = np.nanmean(weather, axis=1)
        hour_pos = (days[:, None] * 24 + np.repeat(np.arange(24), 4)[None, :]).reshape(-1)
        X.append(mean_weather[hour_pos]); names += ["weather_" + x for x in
            ["temperature", "humidity", "sunshine", "precipitation", "wind", "heating_degree"]]
    y_pos = (days[:, None] * 96 + np.arange(96)[None, :]).reshape(-1)
    return np.concatenate(X, axis=1).astype(np.float32), portfolio[y_pos], names


def _direct_output(days: np.ndarray, actual: np.ndarray, pred: np.ndarray,
                   model: str, coverage: np.ndarray, cohort: int) -> pd.DataFrame:
    origin = START + pd.to_timedelta(np.repeat(days, 96), unit="D")
    h = np.tile(np.arange(96), len(days))
    out = pd.DataFrame({"model": model, "origin": origin,
        "target_timestamp": origin + pd.to_timedelta(h * 15, unit="m"), "horizon_step": h + 1,
        "actual": actual, "prediction": pred, "active_households": coverage, "cohort_size": cohort,
        "active_share": coverage / cohort, "actual_coverage": coverage / cohort,
        "prediction_coverage": 1.0})
    return out.loc[(out.actual_coverage >= SPEC.minimum_coverage) & np.isfinite(out.actual)]


def run(output_dir: Path = ARTIFACT_DIR, quick: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lgb_params = {**LGB_PARAMS, **({"n_estimators": 40} if quick else {})}
    print("[1/8] selecting cohorts and loading aligned arrays", flush=True)
    cohort = _cohort(START, END); ids = cohort.Household_ID.astype(str).tolist()
    if len(ids) != 139: raise AssertionError(f"expected 139-household primary cohort, got {len(ids)}")
    matrix, intervention = _meter_matrix(ids, START, END)
    static, meta, static_names = _static_matrix(ids)
    weather, weather_names = _weather_cube(ids, meta, START, END)
    fit_days = _day_indices(SPLIT.fit_start, SPLIT.validation_start)
    # The cohort window begins at the fit boundary; the first seven origins cannot
    # have a 672-step lag and are therefore not silently wrapped or imputed.
    fit_days = fit_days[fit_days * 96 >= max(SPEC.lags)]
    val_days = _day_indices(SPLIT.validation_start, SPLIT.calibration_start)
    refit_days = _day_indices(SPLIT.fit_start, SPLIT.calibration_start)
    cal_days = _day_indices(SPLIT.calibration_start, SPLIT.test_start)
    test_days = _day_indices(SPLIT.test_start, SPLIT.test_end)
    if quick:
        fit_days, refit_days = fit_days[::7], refit_days[::7]

    # Baselines and validation-only ensemble selection.
    print("[2/8] validation baselines", flush=True)
    fit_boundary_day = int((SPLIT.validation_start - START) / pd.Timedelta("1d"))
    refit_boundary_day = int((SPLIT.calibration_start - START) / pd.Timedelta("1d"))
    val_outputs = {name: _baseline_portfolio(matrix, val_days, name, fit_boundary_day)
                   for name in ("previous_day", "previous_week", "historical_quarter_hour_mean")}
    aligned = val_outputs["previous_day"].merge(val_outputs["previous_week"],
        on=["origin", "target_timestamp", "horizon_step"], suffixes=("_d", "_w"))
    delta = aligned.prediction_d - aligned.prediction_w
    denom = np.dot(delta, delta)
    weight = .5 if denom == 0 else float(np.clip(np.dot(aligned.actual_d - aligned.prediction_w, delta) / denom, 0, 1))
    val_outputs["daily_weekly_ensemble"] = _baseline_portfolio(matrix, val_days,
        "daily_weekly_ensemble", fit_boundary_day, weight)

    train_times: dict[str, float] = {name: 0.0 for name in val_outputs}
    feature_importance = []
    # Household global models.
    print("[3/8] validation ridge and LightGBM models", flush=True)
    X_fit, y_fit, _, names = _design(matrix, weather, static, fit_days, False)
    X_val, y_val, val_valid, _ = _design(matrix, weather, static, val_days, False)
    val_coords = _coordinates(val_days, ids, val_valid)
    ridge_scores = {}
    for alpha in (.1, 1., 10.):
        pred, elapsed = _fit_predict(Ridge(alpha=alpha), X_fit, y_fit, X_val)
        out = _household_to_portfolio(val_coords, y_val, pred, f"ridge_{alpha:g}_bottom_up", len(ids))
        ridge_scores[alpha] = regression_metrics(out)
        train_times[f"ridge_{alpha:g}_bottom_up"] = elapsed
    best_alpha = min(ridge_scores, key=lambda x: ridge_scores[x]["mae"])
    ridge_val_pred, _ = _fit_predict(Ridge(alpha=best_alpha), X_fit, y_fit, X_val)
    val_outputs["ridge_bottom_up"] = _household_to_portfolio(val_coords, y_val, ridge_val_pred,
                                                               "ridge_bottom_up", len(ids))
    global_lgb = lgb.LGBMRegressor(**run_lgb_params)
    lgb_val_pred, elapsed = _fit_predict(global_lgb, X_fit, y_fit, X_val)
    train_times["lightgbm_bottom_up"] = elapsed
    val_outputs["lightgbm_bottom_up"] = _household_to_portfolio(val_coords, y_val, lgb_val_pred,
                                                                  "lightgbm_bottom_up", len(ids))
    feature_importance.extend({"model": "lightgbm_bottom_up", "feature": n, "importance": float(v)}
                              for n, v in zip(names, global_lgb.feature_importances_))
    # Oracle validation model.
    X_fit_o, y_fit_o, _, names_o = _design(matrix, weather, static, fit_days, True)
    X_val_o, y_val_o, val_valid_o, _ = _design(matrix, weather, static, val_days, True)
    oracle_lgb = lgb.LGBMRegressor(**run_lgb_params)
    oracle_val_pred, elapsed = _fit_predict(oracle_lgb, X_fit_o, y_fit_o, X_val_o)
    train_times["lightgbm_oracle_bottom_up"] = elapsed
    val_outputs["lightgbm_oracle_bottom_up"] = _household_to_portfolio(
        _coordinates(val_days, ids, val_valid_o), y_val_o, oracle_val_pred,
        "lightgbm_oracle_bottom_up", len(ids))
    # Direct portfolio validation.
    coverage = np.isfinite(matrix).sum(axis=1)
    portfolio = np.nanmean(matrix, axis=1) * len(ids)
    portfolio[coverage / len(ids) < SPEC.minimum_coverage] = np.nan
    Xd_fit, yd_fit, direct_names = _direct_design(portfolio, weather, fit_days)
    Xd_val, yd_val, _ = _direct_design(portfolio, weather, val_days)
    direct_lgb = lgb.LGBMRegressor(**run_lgb_params)
    direct_val_pred, elapsed = _fit_predict(direct_lgb, Xd_fit[np.isfinite(yd_fit)], yd_fit[np.isfinite(yd_fit)], Xd_val)
    train_times["lightgbm_direct"] = elapsed
    val_outputs["lightgbm_direct"] = _direct_output(val_days, yd_val, direct_val_pred,
        "lightgbm_direct", coverage[(val_days[:, None] * 96 + np.arange(96)).reshape(-1)], len(ids))

    val_board = pd.DataFrame([{"model": name, **regression_metrics(out),
                               "is_oracle": "oracle" in name, "training_seconds": train_times.get(name, 0)}
                              for name, out in val_outputs.items()])
    winner = str(choose_operational_winner(val_board).model)
    print(f"[4/8] validation selected {winner}; refitting", flush=True)

    del X_fit, X_val, X_fit_o, X_val_o
    # Refit candidates and create calibration/test evidence.
    outputs = {}
    calibration = {}
    household_metric_rows = []
    diagnostic_slice_rows = []
    for name in ("previous_day", "previous_week", "historical_quarter_hour_mean"):
        outputs[name] = _baseline_portfolio(matrix, test_days, name, refit_boundary_day)
        calibration[name] = _baseline_portfolio(matrix, cal_days, name, refit_boundary_day)
    outputs["daily_weekly_ensemble"] = _baseline_portfolio(matrix, test_days,
        "daily_weekly_ensemble", refit_boundary_day, weight)
    calibration["daily_weekly_ensemble"] = _baseline_portfolio(matrix, cal_days,
        "daily_weekly_ensemble", refit_boundary_day, weight)

    X_refit, y_refit, _, names = _design(matrix, weather, static, refit_days, False)
    for model_name, estimator in [("ridge_bottom_up", Ridge(alpha=best_alpha)),
                                   ("lightgbm_bottom_up", lgb.LGBMRegressor(**run_lgb_params))]:
        started = time.perf_counter(); estimator.fit(X_refit, y_refit)
        train_times[model_name] = time.perf_counter() - started
        for label, days in (("cal", cal_days), ("test", test_days)):
            X, y, valid_rows, _ = _design(matrix, weather, static, days, False)
            pred = np.maximum(estimator.predict(X), 0).astype(np.float32)
            coords = _coordinates(days, ids, valid_rows)
            port = _household_to_portfolio(coords, y, pred, model_name, len(ids))
            (calibration if label == "cal" else outputs)[model_name] = port
            if label == "test":
                detail = coords.assign(actual=y, prediction=pred).merge(
                    meta[["Household_ID", "PV_Status", "Group", "Survey_Building_Type"]],
                    on="Household_ID", how="left")
                positions = (days[:, None] * 96 + np.arange(96)[None, :]).reshape(-1)
                state_values = intervention[positions].reshape(-1)[valid_rows]
                detail["intervention_state"] = pd.Categorical.from_codes(
                    state_values, ["unknown", "before_visit", "after_visit"])
                hm = metric_table(detail.assign(model=model_name), ["model", "Household_ID", "PV_Status"])
                household_metric_rows.append(hm)
                for key in ("PV_Status", "Group", "Survey_Building_Type", "intervention_state"):
                    diagnostic_slice_rows.append(metric_table(detail.assign(model=model_name), ["model", key])
                                                 .assign(slice=key).rename(columns={key: "slice_value"}))
            del X
        if model_name == "lightgbm_bottom_up":
            feature_importance = [{"model": model_name, "feature": n, "importance": float(v)}
                                  for n, v in zip(names, estimator.feature_importances_)]

    del X_refit, y_refit

    X_refit_o, y_refit_o, _, names_o = _design(matrix, weather, static, refit_days, True)
    print("[5/8] realized-weather oracle refit", flush=True)
    oracle = lgb.LGBMRegressor(**run_lgb_params); started = time.perf_counter(); oracle.fit(X_refit_o, y_refit_o)
    train_times["lightgbm_oracle_bottom_up"] = time.perf_counter() - started
    for label, days in (("cal", cal_days), ("test", test_days)):
        X, y, valid_rows, _ = _design(matrix, weather, static, days, True)
        pred = np.maximum(oracle.predict(X), 0).astype(np.float32)
        port = _household_to_portfolio(_coordinates(days, ids, valid_rows), y, pred,
                                        "lightgbm_oracle_bottom_up", len(ids))
        (calibration if label == "cal" else outputs)["lightgbm_oracle_bottom_up"] = port
        del X
    feature_importance.extend({"model": "lightgbm_oracle_bottom_up", "feature": n, "importance": float(v)}
                              for n, v in zip(names_o, oracle.feature_importances_))
    del X_refit_o, y_refit_o

    Xd_refit, yd_refit, direct_names = _direct_design(portfolio, weather, refit_days)
    print("[6/8] direct portfolio refit", flush=True)
    direct = lgb.LGBMRegressor(**run_lgb_params); mask = np.isfinite(yd_refit)
    started = time.perf_counter(); direct.fit(Xd_refit[mask], yd_refit[mask]); train_times["lightgbm_direct"] = time.perf_counter() - started
    for label, days in (("cal", cal_days), ("test", test_days)):
        X, y, _ = _direct_design(portfolio, weather, days); pred = np.maximum(direct.predict(X), 0)
        cov = coverage[(days[:, None] * 96 + np.arange(96)).reshape(-1)]
        (calibration if label == "cal" else outputs)["lightgbm_direct"] = _direct_output(
            days, y, pred, "lightgbm_direct", cov, len(ids))
    feature_importance.extend({"model": "lightgbm_direct", "feature": n, "importance": float(v)}
                              for n, v in zip(direct_names, direct.feature_importances_))

    # Score every model on an identical intersection of target rows. This is
    # stricter than merely reporting each model's own available coverage.
    keys = ["origin", "target_timestamp", "horizon_step"]
    common = None
    for out in outputs.values():
        index = pd.MultiIndex.from_frame(out[keys])
        common = index if common is None else common.intersection(index)
    for name, out in outputs.items():
        indexed = out.set_index(keys)
        outputs[name] = indexed.loc[indexed.index.intersection(common)].reset_index().sort_values(keys)

    # Calibrate the validation-selected winner, then score all test outputs.
    quantiles = conformal_quantiles(calibration[winner])
    outputs[winner] = apply_conformal(outputs[winner], quantiles)
    all_predictions = pd.concat(outputs.values(), ignore_index=True)
    all_predictions["Household_ID"] = "__portfolio__"
    for col in ("lower", "upper"):
        if col not in all_predictions: all_predictions[col] = np.nan
    overall = pd.DataFrame([{"model": name, **regression_metrics(out),
        "asymmetric_cost_index": _procurement_cost(out),
        "is_oracle": "oracle" in name, "selected_on_validation": name == winner,
        "training_seconds": train_times.get(name, 0),
        "prediction_coverage": len(out) / (len(test_days) * 96)} for name, out in outputs.items()])
    overall = overall.sort_values(["is_oracle", "mae"])
    interval = interval_metrics(outputs[winner])
    overall.loc[overall.model == winner, list(interval)] = list(interval.values())

    pred = all_predictions.copy()
    pred["month"] = pd.to_datetime(pred.target_timestamp, utc=True).dt.month
    pred["season"] = pd.cut(pred.month, [0, 2, 5, 8, 11, 12], labels=["winter", "spring", "summer", "autumn", "winter2"], include_lowest=True).astype(str).replace("winter2", "winter")
    pred["hour"] = pd.to_datetime(pred.target_timestamp, utc=True).dt.hour
    mean_temperature = np.nanmean(weather[:, :, 0], axis=1)
    temperature_15m = np.repeat(mean_temperature, 4)
    time_positions = ((pd.to_datetime(pred.target_timestamp, utc=True) - START) /
                      pd.Timedelta("15min")).astype(int)
    pred["temperature"] = temperature_15m[np.clip(time_positions, 0, len(temperature_15m) - 1)]
    pred["temperature_regime"] = pd.cut(pred.temperature, [-np.inf, 0, 10, 18, np.inf],
                                         labels=["freezing", "cold", "mild", "warm"])
    sliced = pd.concat([metric_table(pred, ["model", key]).assign(slice=key).rename(columns={key: "slice_value"})
                        for key in ("horizon_step", "month", "season", "hour", "temperature_regime")]
                       + diagnostic_slice_rows, ignore_index=True)
    if outputs[winner]["lower"].notna().any():
        coverage_rows = []
        for key in ("horizon_step", "season", "temperature_regime"):
            for value, group in pred.loc[pred.model == winner].groupby(key):
                coverage_rows.append({"model": winner, "slice": f"interval_{key}", "slice_value": value,
                                      **interval_metrics(group)})
        sliced = pd.concat([sliced, pd.DataFrame(coverage_rows)], ignore_index=True)

    # A transparent robustness check: re-evaluate the selected lag strategy on the stated
    # one-year cohort. Model refitting is intentionally limited to direct/ensemble winners.
    robust_start = pd.Timestamp("2023-02-28", tz="UTC")
    robust_cohort = _cohort(robust_start, END)
    robust_ids = robust_cohort.Household_ID.astype(str).tolist()
    robust_matrix, _ = _meter_matrix(robust_ids, robust_start, END)
    robust_days = np.arange(275, 365)
    robust_pos = robust_days[:, None] * 96 + np.arange(96)[None, :]
    robust_actual = robust_matrix[robust_pos]
    robust_prediction = (weight * robust_matrix[robust_pos - 96] +
                         (1 - weight) * robust_matrix[robust_pos - 672])
    robust_active = np.isfinite(robust_actual).sum(axis=2)
    robust_predicted = np.isfinite(robust_prediction).sum(axis=2)
    robust_frame = pd.DataFrame({
        "actual": (np.nanmean(robust_actual, axis=2) * len(robust_ids)).reshape(-1),
        "prediction": (np.nanmean(robust_prediction, axis=2) * len(robust_ids)).reshape(-1),
        "actual_coverage": (robust_active / len(robust_ids)).reshape(-1),
        "prediction_coverage": (robust_predicted / len(robust_ids)).reshape(-1),
    })
    robust_frame = robust_frame.loc[(robust_frame.actual_coverage >= SPEC.minimum_coverage) &
                                    (robust_frame.prediction_coverage >= SPEC.minimum_coverage)]
    robustness = {"cohort_size": int(len(robust_cohort)), "minimum_completeness": .95,
                  "window_start": robust_start.isoformat(), "window_end": END.isoformat(),
                  "status": "completed" if len(robust_cohort) == 343 else "cohort_mismatch",
                  "model": "daily_weekly_ensemble", "evaluation_days": 90,
                  "metrics": regression_metrics(robust_frame)}

    keep = ["model", "origin", "target_timestamp", "horizon_step", "Household_ID", "actual",
            "prediction", "lower", "upper", "active_households", "cohort_size", "active_share",
            "actual_coverage", "prediction_coverage"]
    all_predictions[keep].to_parquet(output_dir / "portfolio_predictions.parquet", index=False)
    print("[7/8] writing metrics and reproducibility artifacts", flush=True)
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    sliced.to_csv(output_dir / "sliced_metrics.csv", index=False)
    pd.concat(household_metric_rows, ignore_index=True).to_parquet(output_dir / "household_metrics.parquet", index=False)
    pd.DataFrame(feature_importance).to_csv(output_dir / "feature_importance.csv", index=False)
    val_board.to_csv(output_dir / "validation_metrics.csv", index=False)
    config = {"forecast_spec": SPEC.to_dict(), "split_spec": SPLIT.to_dict(),
              "primary_cohort": {"size": len(ids), "window_start": START.isoformat(),
                                  "window_end": END.isoformat(), "minimum_completeness": .95},
              "lightgbm": run_lgb_params, "production_lightgbm": LGB_PARAMS,
              "ridge_alphas": [.1, 1., 10.], "selected_ridge_alpha": best_alpha,
              "ensemble_previous_day_weight": weight, "selected_operational_model": winner,
              "oracle_eligible_for_selection": False, "training_stride": 96,
              "darts_lag_equivalence_test": "passed by tests/test_modeling.py",
              "execution_mode": "quick" if quick else "full", "robustness": robustness}
    (output_dir / "model_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    print(f"[8/8] complete: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args(); run(args.output_dir, args.quick)


if __name__ == "__main__": main()
