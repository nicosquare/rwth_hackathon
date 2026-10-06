"""Leakage-safe primitives for day-ahead household and portfolio forecasting.

All timestamps in this module are UTC and split ends are exclusive.  The small,
data-frame-oriented functions are intentionally model agnostic: the notebook can
compare Darts estimators without duplicating the forecast contract or scoring
rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "model", "origin", "target_timestamp", "horizon_step", "Household_ID",
    "actual", "prediction", "lower", "upper", "active_households",
    "cohort_size", "active_share", "actual_coverage", "prediction_coverage",
]

WEATHER_COLUMNS = [
    "temperature", "humidity", "sunshine", "precipitation", "wind",
    "heating_degree",
]


def _utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


@dataclass(frozen=True)
class ForecastSpec:
    """Immutable forecast issuance contract."""

    frequency: str = "15min"
    origin_hour: int = 0
    horizon: int = 96
    lags: tuple[int, ...] = (1, 4, 96, 192, 672)
    minimum_coverage: float = 0.98

    def __post_init__(self) -> None:
        if not 0 <= self.origin_hour <= 23:
            raise ValueError("origin_hour must be in [0, 23]")
        if self.horizon <= 0 or any(lag <= 0 for lag in self.lags):
            raise ValueError("horizon and lags must be positive")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1]")

    @property
    def offset(self) -> pd.Timedelta:
        return pd.Timedelta(self.frequency)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DayAheadContract:
    """Information set of a real day-ahead bid, shared by every notebook.

    The forecast covers the 96 quarter-hours of UTC delivery day D.  The EPEX
    day-ahead auction closes at 12:00 local time on D-1 (10:00 UTC in summer,
    11:00 UTC in winter), so only meter readings and weather observations
    strictly before 10:00 UTC on D-1 may be used; this leaves time to compute
    and submit the bid all year.
    """

    data_cutoff_hour_utc: int = 10
    steps_per_day: int = 96

    def __post_init__(self) -> None:
        if not 0 <= self.data_cutoff_hour_utc <= 23:
            raise ValueError("data_cutoff_hour_utc must be in [0, 23]")

    @property
    def gap_steps(self) -> int:
        """Quarter-hours between the data cutoff and the start of delivery day D."""
        return (24 - self.data_cutoff_hour_utc) * self.steps_per_day // 24

    def cutoff(self, delivery_day: object) -> pd.Timestamp:
        day = _utc(delivery_day).normalize()
        return day - pd.Timedelta(hours=24 - self.data_cutoff_hour_utc)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "gap_steps": self.gap_steps}


DAY_AHEAD = DayAheadContract()


@dataclass(frozen=True)
class SplitSpec:
    """Chronological, non-overlapping train/selection/calibration/test splits."""

    fit_start: pd.Timestamp
    validation_start: pd.Timestamp
    calibration_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        values = tuple(_utc(x) for x in (
            self.fit_start, self.validation_start, self.calibration_start,
            self.test_start, self.test_end,
        ))
        object.__setattr__(self, "fit_start", values[0])
        object.__setattr__(self, "validation_start", values[1])
        object.__setattr__(self, "calibration_start", values[2])
        object.__setattr__(self, "test_start", values[3])
        object.__setattr__(self, "test_end", values[4])
        if list(values) != sorted(values) or len(set(values)) != 5:
            raise ValueError("split boundaries must be strictly increasing")

    @classmethod
    def from_day_counts(
        cls, start: object, fit_days: int = 275, validation_days: int = 90,
        calibration_days: int = 90, test_days: int = 275,
    ) -> "SplitSpec":
        start = _utc(start)
        v = start + pd.Timedelta(days=fit_days)
        c = v + pd.Timedelta(days=validation_days)
        t = c + pd.Timedelta(days=calibration_days)
        return cls(start, v, c, t, t + pd.Timedelta(days=test_days))

    def intervals(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {
            "fit": (self.fit_start, self.validation_start),
            "validation": (self.validation_start, self.calibration_start),
            "calibration": (self.calibration_start, self.test_start),
            "test": (self.test_start, self.test_end),
        }

    def label(self, timestamps: pd.Series | pd.DatetimeIndex) -> pd.Series:
        ts = pd.to_datetime(timestamps, utc=True)
        out = pd.Series(pd.NA, index=range(len(ts)), dtype="string")
        for name, (start, end) in self.intervals().items():
            out.loc[(ts >= start) & (ts < end)] = name
        return out

    def to_dict(self) -> dict[str, str]:
        return {name: value.isoformat() for name, value in asdict(self).items()}


def select_fixed_cohort(
    frame: pd.DataFrame, start: object, end: object,
    minimum_completeness: float = 0.95, *, household_col: str = "Household_ID",
    timestamp_col: str = "timestamp_utc", target_col: str = "actual",
    frequency: str = "15min",
) -> pd.DataFrame:
    """Return deterministic per-household completeness and membership."""
    start, end = _utc(start), _utc(end)
    timestamps = pd.to_datetime(frame[timestamp_col], utc=True)
    window = frame.loc[(timestamps >= start) & (timestamps < end)].copy()
    expected = int((end - start) / pd.Timedelta(frequency))
    stats = window.groupby(household_col, observed=True).agg(
        valid_intervals=(target_col, "count"),
        observed_intervals=(timestamp_col, "nunique"),
    )
    stats["expected_intervals"] = expected
    stats["completeness"] = stats["valid_intervals"] / expected
    stats["selected"] = stats["completeness"] >= minimum_completeness
    return stats.reset_index().sort_values(household_col).reset_index(drop=True)


def split_contiguous_series(
    frame: pd.DataFrame, spec: ForecastSpec = ForecastSpec(), *,
    household_col: str = "Household_ID", timestamp_col: str = "timestamp_utc",
    target_col: str = "actual",
) -> list[pd.DataFrame]:
    """Split at missing/null targets and retain runs usable for lagged forecasts."""
    minimum = max(spec.lags) + spec.horizon
    runs: list[pd.DataFrame] = []
    ordered = frame.sort_values([household_col, timestamp_col])
    for _, group in ordered.groupby(household_col, sort=False, observed=True):
        group = group.copy()
        ts = pd.to_datetime(group[timestamp_col], utc=True)
        boundary = ts.diff().ne(spec.offset) | group[target_col].isna()
        run_id = boundary.cumsum()
        for _, run in group.loc[group[target_col].notna()].groupby(run_id, sort=False):
            if len(run) >= minimum:
                runs.append(run.reset_index(drop=True))
    return runs


def build_calendar_covariates(timestamps: Sequence[object]) -> pd.DataFrame:
    """Known-future UTC calendar fields for every requested timestamp."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    month = ts.month
    season = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
         np.isin(month, [6, 7, 8])], [0, 1, 2], default=3,
    )
    return pd.DataFrame({
        "timestamp_utc": ts, "quarter_hour": ts.minute // 15,
        "hour": ts.hour, "weekday": ts.weekday,
        "weekend": (ts.weekday >= 5).astype("int8"), "month": month,
        "season": season,
    })


def build_static_covariates(metadata: pd.DataFrame) -> pd.DataFrame:
    """Create stable numeric/categorical household descriptors without imputation leakage."""
    aliases = {
        "PV_Status": "pv_status", "Group": "group",
        "Survey_Building_Type": "building_type",
        "Survey_Building_LivingArea": "living_area",
        "Survey_Building_Residents": "residents",
        "Survey_HeatPump_Installation_Type": "heat_pump_type",
    }
    out = metadata.rename(columns=aliases).copy()
    wanted = ["Household_ID", *aliases.values()]
    for col in wanted:
        if col not in out:
            out[col] = np.nan
    out["pv_status"] = out["pv_status"].fillna("unknown").astype(str).str.lower()
    for col in ["group", "building_type", "heat_pump_type"]:
        out[col] = out[col].fillna("unknown").astype(str)
    out["household_code"] = pd.Categorical(out["Household_ID"].astype(str)).codes
    return out[["Household_ID", "household_code", *aliases.values()]]


def build_oracle_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Normalize realized-weather names and label their unavailable-at-origin status."""
    source = {
        "Temperature_avg_hourly": "temperature",
        "Humidity_avg_hourly": "humidity",
        "Sunshine_duration_hourly": "sunshine",
        "Precipitation_total_hourly": "precipitation",
        "WindSpeed_hourly": "wind",
    }
    out = weather.rename(columns=source).copy()
    missing = [column for column in source.values() if column not in out]
    if missing:
        raise ValueError(f"missing oracle weather columns: {missing}")
    out["heating_degree"] = (15.0 - out["temperature"]).clip(lower=0)
    out["feature_availability"] = "oracle_realized_target_horizon"
    return out


def lag_feature_frame(
    frame: pd.DataFrame, origins: Iterable[object], spec: ForecastSpec = ForecastSpec(),
    *, timestamp_col: str = "timestamp_utc", target_col: str = "actual",
    household_col: str = "Household_ID",
) -> pd.DataFrame:
    """Build origin-relative target lags; lag values are identical across a horizon."""
    data = frame.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)
    indexed = data.set_index([household_col, timestamp_col])[target_col]
    households = data[household_col].drop_duplicates().tolist()
    rows: list[dict[str, object]] = []
    for origin_raw in origins:
        origin = _utc(origin_raw)
        if origin.hour != spec.origin_hour or origin.minute != 0:
            raise ValueError(f"origin {origin} violates the issuance contract")
        for household in households:
            base = {"origin": origin, household_col: household}
            for lag in spec.lags:
                base[f"lag_{lag}"] = indexed.get((household, origin - lag * spec.offset), np.nan)
            for step in range(1, spec.horizon + 1):
                rows.append({**base, "horizon_step": step,
                             "target_timestamp": origin + (step - 1) * spec.offset})
    return pd.DataFrame(rows)


def assert_feature_availability(
    features: pd.DataFrame, *, oracle: bool = False,
    origin_col: str = "origin", feature_timestamp_col: str = "feature_timestamp",
    family_col: str = "feature_family",
) -> None:
    """Reject target/past features at or beyond origin and unlabelled future data."""
    origin = pd.to_datetime(features[origin_col], utc=True)
    feature_time = pd.to_datetime(features[feature_timestamp_col], utc=True)
    family = features[family_col].astype(str)
    past = family.isin(["target", "past_covariate"])
    if (feature_time[past] >= origin[past]).any():
        raise AssertionError("target and past-covariate timestamps must precede origin")
    future = feature_time >= origin
    allowed = family.eq("calendar") | (oracle & family.eq("oracle_weather"))
    if (future & ~allowed).any():
        raise AssertionError("only calendar or explicitly enabled oracle weather may be future")


def temporal_baselines(
    frame: pd.DataFrame, origins: Iterable[object], spec: ForecastSpec = ForecastSpec(),
    *, timestamp_col: str = "timestamp_utc", target_col: str = "actual",
    household_col: str = "Household_ID", historical_end: object | None = None,
) -> pd.DataFrame:
    """Generate previous-day/week and frozen quarter-hour-mean forecasts."""
    data = frame.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True)
    origins = [_utc(origin) for origin in origins]
    cutoff = _utc(historical_end) if historical_end is not None else min(origins)
    data["slot"] = data[timestamp_col].dt.hour * 4 + data[timestamp_col].dt.minute // 15
    climatology = (data.loc[data[timestamp_col] < cutoff]
                   .groupby([household_col, "slot"], observed=True)[target_col].mean())
    indexed = data.set_index([household_col, timestamp_col])[target_col]
    rows = []
    for origin in origins:
        for household in data[household_col].drop_duplicates():
            for step in range(1, spec.horizon + 1):
                target_time = origin + (step - 1) * spec.offset
                actual = indexed.get((household, target_time), np.nan)
                slot = target_time.hour * 4 + target_time.minute // 15
                common = dict(origin=origin, target_timestamp=target_time,
                              horizon_step=step, Household_ID=household, actual=actual)
                for model, delta in (("previous_day", 96), ("previous_week", 672)):
                    rows.append({**common, "model": model,
                                 "prediction": indexed.get((household, target_time - delta * spec.offset), np.nan)})
                rows.append({**common, "model": "historical_quarter_hour_mean",
                             "prediction": climatology.get((household, slot), np.nan)})
    return pd.DataFrame(rows)


def fit_nonnegative_ensemble(
    predictions: pd.DataFrame, model_a: str = "previous_day",
    model_b: str = "previous_week",
) -> tuple[float, float]:
    """Select convex two-model weights by exact bounded least squares."""
    keys = ["origin", "target_timestamp", "horizon_step", "Household_ID", "actual"]
    wide = predictions[predictions.model.isin([model_a, model_b])].pivot_table(
        index=keys, columns="model", values="prediction", observed=True).dropna()
    a, b, y = wide[model_a].to_numpy(), wide[model_b].to_numpy(), wide.index.get_level_values("actual").to_numpy()
    denominator = np.dot(a - b, a - b)
    weight_a = .5 if denominator == 0 else float(np.clip(np.dot(y - b, a - b) / denominator, 0, 1))
    return weight_a, 1 - weight_a


def aggregate_portfolio(
    predictions: pd.DataFrame, cohort_size: int | None = None,
    minimum_coverage: float = 0.98,
) -> pd.DataFrame:
    """Create the fixed-portfolio estimand and filter low actual/prediction coverage.

    Observed and predicted means are independently scaled by fixed cohort size.
    This avoids treating a changing active set as the procurement portfolio.
    """
    group = ["model", "origin", "target_timestamp", "horizon_step"]
    cohort_size = cohort_size or predictions["Household_ID"].nunique()
    agg = predictions.groupby(group, observed=True, dropna=False).agg(
        actual_mean=("actual", "mean"), prediction_mean=("prediction", "mean"),
        active_households=("actual", "count"), predicted_households=("prediction", "count"),
    ).reset_index()
    agg["cohort_size"] = cohort_size
    agg["active_share"] = agg["active_households"] / cohort_size
    agg["actual_coverage"] = agg["active_share"]
    agg["prediction_coverage"] = agg["predicted_households"] / cohort_size
    agg["actual"] = agg["actual_mean"] * cohort_size
    agg["prediction"] = agg["prediction_mean"] * cohort_size
    agg["Household_ID"] = "__portfolio__"
    agg["lower"] = np.nan
    agg["upper"] = np.nan
    valid = ((agg.actual_coverage >= minimum_coverage) &
             (agg.prediction_coverage >= minimum_coverage))
    return agg.loc[valid, PREDICTION_COLUMNS].reset_index(drop=True)


def validate_cluster_labels(
    labels: pd.DataFrame,
    household_ids: Iterable[object] | None = None,
    *,
    household_col: str = "Household_ID",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Validate a static one-cluster-per-household label table.

    Cluster assignments are metadata, not forecast targets. They must be
    unique per household and complete for the requested cohort so that cluster
    metrics do not silently change their denominator.
    """
    required = {household_col, cluster_col}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"cluster labels missing columns: {sorted(missing)}")
    out = labels[[household_col, cluster_col]].copy()
    if out[household_col].isna().any() or out[cluster_col].isna().any():
        raise ValueError("cluster labels cannot contain null household or cluster IDs")
    if out[household_col].duplicated().any():
        raise ValueError("cluster labels must contain one row per household")
    if household_ids is not None:
        requested = pd.Index(list(household_ids))
        available = pd.Index(out[household_col])
        missing_ids = requested.difference(available)
        if len(missing_ids):
            raise ValueError(f"cluster labels missing households: {missing_ids.tolist()}")
        out = out[out[household_col].isin(requested)].copy()
    return out.reset_index(drop=True)


def attach_cluster_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    household_col: str = "Household_ID",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Attach validated static cluster IDs to household-level predictions."""
    label_table = validate_cluster_labels(
        labels, predictions[household_col].dropna().unique(),
        household_col=household_col, cluster_col=cluster_col,
    )
    out = predictions.merge(
        label_table, on=household_col, how="left", validate="many_to_one",
    )
    if out[cluster_col].isna().any():
        raise ValueError("predictions contain households without cluster labels")
    return out


def aggregate_clusters(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    minimum_coverage: float = 0.98,
    *,
    household_col: str = "Household_ID",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Aggregate household forecasts into fixed cluster demand series.

    Each cluster is scaled by its fixed household count, while intervals with
    insufficient actual or prediction coverage are removed. This mirrors the
    portfolio estimand and keeps cluster metrics comparable over time.
    """
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    frame = attach_cluster_labels(
        predictions, labels, household_col=household_col, cluster_col=cluster_col,
    )
    sizes = frame[[cluster_col, household_col]].drop_duplicates().groupby(
        cluster_col, observed=True,
    )[household_col].nunique()
    group = ["model", cluster_col, "origin", "target_timestamp", "horizon_step"]
    agg = frame.groupby(group, observed=True, dropna=False).agg(
        actual_mean=("actual", "mean"), prediction_mean=("prediction", "mean"),
        active_households=("actual", "count"),
        predicted_households=("prediction", "count"),
    ).reset_index()
    agg["cohort_size"] = agg[cluster_col].map(sizes)
    agg["active_share"] = agg["active_households"] / agg["cohort_size"]
    agg["actual_coverage"] = agg["active_share"]
    agg["prediction_coverage"] = agg["predicted_households"] / agg["cohort_size"]
    agg["actual"] = agg["actual_mean"] * agg["cohort_size"]
    agg["prediction"] = agg["prediction_mean"] * agg["cohort_size"]
    agg["Household_ID"] = agg[cluster_col].map(lambda value: f"__cluster__{value}")
    agg["lower"] = np.nan
    agg["upper"] = np.nan
    valid = (
        (agg["actual_coverage"] >= minimum_coverage)
        & (agg["prediction_coverage"] >= minimum_coverage)
    )
    columns = [
        "model", cluster_col, "origin", "target_timestamp", "horizon_step",
        "Household_ID", "actual", "prediction", "lower", "upper",
        "active_households", "cohort_size", "active_share", "actual_coverage",
        "prediction_coverage",
    ]
    return agg.loc[valid, columns].reset_index(drop=True)


def cluster_model_plan(
    labels: pd.DataFrame,
    training_rows: pd.Series | pd.DataFrame,
    *,
    min_households: int = 10,
    min_rows: int = 1000,
    household_col: str = "Household_ID",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Return deterministic per-cluster fit/fallback decisions.

    Clusters below either threshold use the global model. The returned table is
    deliberately a plan rather than a fitted estimator, keeping model fitting
    independent from label ingestion and making fallback auditable.
    """
    label_table = validate_cluster_labels(labels, household_col=household_col,
                                          cluster_col=cluster_col)
    if isinstance(training_rows, pd.Series):
        rows = training_rows.rename("training_rows").to_frame()
        rows.index.name = household_col
        rows = rows.reset_index()
    else:
        required = {household_col, "training_rows"}
        if not required.issubset(training_rows.columns):
            raise ValueError("training_rows must contain Household_ID and training_rows")
        rows = training_rows[[household_col, "training_rows"]].copy()
    merged = label_table.merge(rows, on=household_col, how="left", validate="one_to_one")
    if merged["training_rows"].isna().any():
        raise ValueError("training row counts missing for labelled households")
    plan = merged.groupby(cluster_col, observed=True).agg(
        households=(household_col, "nunique"),
        training_rows=("training_rows", "sum"),
    ).reset_index()
    plan["fit_mode"] = np.where(
        (plan["households"] >= min_households)
        & (plan["training_rows"] >= min_rows),
        "cluster",
        "global_fallback",
    )
    return plan


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame[["actual", "prediction"]].dropna()
    if valid.empty:
        return {key: np.nan for key in ("mae", "rmse", "wape", "bias", "n")}
    error = valid.prediction - valid.actual
    denominator = valid.actual.abs().sum()
    return {
        "mae": float(error.abs().mean()), "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(error.abs().sum() / denominator) if denominator else np.nan,
        "bias": float(error.mean()), "n": int(len(valid)),
    }


def metric_table(frame: pd.DataFrame, by: Sequence[str] = ("model",)) -> pd.DataFrame:
    rows = []
    grouper = by[0] if len(by) == 1 else list(by)
    for key, group in frame.groupby(grouper, observed=True, dropna=False):
        key = (key,) if len(by) == 1 else key
        rows.append({**dict(zip(by, key)), **regression_metrics(group)})
    return pd.DataFrame(rows)


def conformal_quantiles(
    calibration: pd.DataFrame, coverage: float = 0.90,
    *, horizon_col: str = "horizon_step",
) -> pd.DataFrame:
    """Finite-sample, horizon-specific absolute-residual conformal radii."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    rows = []
    for horizon, group in calibration.dropna(subset=["actual", "prediction"]).groupby(horizon_col):
        residuals = np.sort(np.abs(group.actual.to_numpy() - group.prediction.to_numpy()))
        rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * coverage)))
        rows.append({horizon_col: horizon, "radius": residuals[rank - 1], "n_calibration": len(residuals)})
    return pd.DataFrame(rows)


def apply_conformal(predictions: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    out = predictions.drop(columns=["lower", "upper"], errors="ignore").merge(
        quantiles[["horizon_step", "radius"]], on="horizon_step", how="left")
    out["lower"] = (out.prediction - out.radius).clip(lower=0)
    out["upper"] = out.prediction + out.radius
    return out.drop(columns="radius")


def interval_metrics(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["actual", "lower", "upper"])
    return {
        "interval_coverage": float(((valid.actual >= valid.lower) & (valid.actual <= valid.upper)).mean()),
        "mean_interval_width": float((valid.upper - valid.lower).mean()),
        "n": int(len(valid)),
    }


def assert_identical_evaluation_rows(predictions: pd.DataFrame) -> None:
    """Ensure every model is ranked on precisely the same targets."""
    keys = ["origin", "target_timestamp", "horizon_step", "Household_ID"]
    models = predictions.model.drop_duplicates().tolist()
    if not models:
        raise AssertionError("no models supplied")
    reference = set(map(tuple, predictions.loc[predictions.model == models[0], keys].to_numpy()))
    for model in models[1:]:
        current = set(map(tuple, predictions.loc[predictions.model == model, keys].to_numpy()))
        if current != reference:
            raise AssertionError(f"{model} does not share identical evaluation rows")


def choose_operational_winner(leaderboard: pd.DataFrame) -> pd.Series:
    """Rank by MAE while making oracle selection structurally impossible."""
    operational = leaderboard.loc[~leaderboard["is_oracle"].astype(bool)].copy()
    if operational.empty:
        raise ValueError("leaderboard has no operational candidates")
    return operational.sort_values(["mae", "wape", "model"]).iloc[0]
