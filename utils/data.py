"""Reusable, leakage-conscious data preparation for the hackathon analyses."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METER_DIR = DATA_DIR / "15min"
WEATHER_DIR = DATA_DIR / "weather_data_hourly"
METADATA_DIR = DATA_DIR / "smart_meter_meta_data"

METER_COLUMNS = {
    "Household_ID": pl.String,
    "AffectsTimePoint": pl.String,
    "Group": pl.String,
    "Timestamp": pl.String,
    "kWh_received_Total": pl.Float64,
    "kWh_received_HeatPump": pl.Float64,
    "kWh_received_Other": pl.Float64,
}


def _parse_utc(column: str = "Timestamp") -> pl.Expr:
    return (
        pl.col(column)
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias("timestamp_utc")
    )


def scan_meter_data(data_dir: Path = METER_DIR) -> pl.LazyFrame:
    """Lazily scan all household meter files using a stable shared schema."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No meter CSV files found in {data_dir}")

    # Source files use two different physical orders for Group/AffectsTimePoint.
    # Selecting by name before concatenation prevents positional schema conflicts.
    scans = [
        pl.scan_csv(
            path,
            separator=";",
            schema_overrides=METER_COLUMNS,
            null_values=[""],
        ).select(*METER_COLUMNS)
        for path in files
    ]
    return pl.concat(scans, how="vertical_relaxed").with_columns(_parse_utc()).drop(
        "Timestamp"
    )


def load_household_metadata(metadata_dir: Path = METADATA_DIR) -> pl.DataFrame:
    """Load household, survey, and availability metadata without losing unknown PV flags."""
    households = pl.read_csv(
        metadata_dir / "households.csv",
        separator=";",
        schema_overrides={"Household_ID": pl.String, "Weather_ID": pl.String},
        null_values=[""],
    ).with_columns(
        pl.when(pl.col("Installation_HasPVSystem").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("Installation_HasPVSystem"))
        .then(pl.lit("true"))
        .otherwise(pl.lit("false"))
        .alias("PV_Status")
    )

    survey = pl.read_csv(
        metadata_dir / "meta_data.csv",
        separator=";",
        schema_overrides={"Household_ID": pl.String},
        null_values=[""],
    )
    availability = pl.read_csv(
        metadata_dir / "smart_meter_data_15min_overview.csv",
        separator=";",
        schema_overrides={"Household_ID": pl.String},
        null_values=[""],
    )
    return households.join(survey, on="Household_ID", how="left").join(
        availability, on="Household_ID", how="left"
    )


def load_weather_data(data_dir: Path = WEATHER_DIR) -> pl.DataFrame:
    """Load and concatenate hourly weather observations from all stations."""
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        frames.append(
            pl.read_csv(
                path,
                separator=";",
                schema_overrides={"Weather_ID": pl.String, "Timestamp": pl.String},
                null_values=[""],
            )
            .with_columns(_parse_utc())
            .drop("Timestamp")
        )
    if not frames:
        raise FileNotFoundError(f"No weather CSV files found in {data_dir}")
    return pl.concat(frames, how="diagonal_relaxed")


def _calendar_columns(timestamp_column: str = "timestamp_local") -> list[pl.Expr]:
    """Calendar fields used consistently by interval and hourly analyses."""
    month = pl.col(timestamp_column).dt.month()
    return [
        pl.col(timestamp_column).dt.date().alias("local_date"),
        pl.col(timestamp_column).dt.hour().alias("local_hour"),
        pl.col(timestamp_column).dt.weekday().alias("local_weekday"),
        month.alias("local_month"),
        (pl.col(timestamp_column).dt.weekday() >= 6).alias("is_weekend"),
        (pl.col(timestamp_column).dt.hour() * 4 + pl.col(timestamp_column).dt.minute() // 15)
        .cast(pl.UInt8)
        .alias("local_slot"),
        pl.when(month.is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(month.is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(month.is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("autumn"))
        .alias("season"),
    ]


def _analysis_household_fields(metadata: pl.DataFrame) -> pl.DataFrame:
    return metadata.select(
        "Household_ID",
        "Weather_ID",
        "PV_Status",
        "Survey_Building_Type",
        "Survey_Building_LivingArea",
        "Survey_Building_Residents",
        "Survey_HeatPump_Installation_Type",
    )


def _analysis_weather_fields(weather: pl.DataFrame) -> pl.DataFrame:
    return weather.select(
        "Weather_ID",
        "timestamp_utc",
        "Temperature_avg_hourly",
        "Humidity_avg_hourly",
        "Precipitation_total_hourly",
        "Sunshine_duration_hourly",
        "WindSpeed_hourly",
    )


def build_interval_panel(
    meter: pl.LazyFrame | None = None,
    metadata: pl.DataFrame | None = None,
    weather: pl.DataFrame | None = None,
    household_ids: list[str] | pl.Series | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Build an enriched 15-minute panel, optionally restricted before collection."""
    meter = meter if meter is not None else scan_meter_data()
    metadata = metadata if metadata is not None else load_household_metadata()
    weather = weather if weather is not None else load_weather_data()

    if household_ids is not None:
        ids = household_ids.to_list() if isinstance(household_ids, pl.Series) else household_ids
        meter = meter.filter(pl.col("Household_ID").is_in(ids))
    if start is not None:
        meter = meter.filter(pl.col("timestamp_utc") >= start)
    if end is not None:
        meter = meter.filter(pl.col("timestamp_utc") <= end)

    intervals = meter.collect().with_columns(
        pl.col("timestamp_utc").dt.truncate("1h").alias("weather_timestamp_utc"),
        pl.col("timestamp_utc")
        .dt.convert_time_zone("Europe/Berlin")
        .alias("timestamp_local"),
    )
    weather_fields = _analysis_weather_fields(weather).rename(
        {"timestamp_utc": "weather_timestamp_utc"}
    )
    return (
        intervals.join(_analysis_household_fields(metadata), on="Household_ID", how="left")
        .join(weather_fields, on=["Weather_ID", "weather_timestamp_utc"], how="left")
        .with_columns(*_calendar_columns())
    )


def daily_meter_quality(meter: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Summarize timestamp and value completeness for each observed UTC day."""
    meter = meter if meter is not None else scan_meter_data()
    return (
        meter.with_columns(pl.col("timestamp_utc").dt.date().alias("date_utc"))
        .group_by("Household_ID", "date_utc")
        .agg(
            pl.len().alias("timestamp_rows"),
            pl.col("kWh_received_Total").count().alias("valid_total_values"),
            pl.col("kWh_received_HeatPump").count().alias("valid_heatpump_values"),
            pl.col("kWh_received_Other").count().alias("valid_other_values"),
        )
        .with_columns(
            (
                (pl.col("timestamp_rows") == 96)
                & (pl.col("valid_total_values") == 96)
            ).alias("day_complete")
        )
        .collect()
    )


def meter_gap_table(meter: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Return gaps between observed timestamps; null-valued rows are not timestamp gaps."""
    meter = meter if meter is not None else scan_meter_data()
    return (
        meter.select("Household_ID", "timestamp_utc")
        .sort("Household_ID", "timestamp_utc")
        .with_columns(pl.col("timestamp_utc").diff().over("Household_ID").alias("delta"))
        .filter(pl.col("delta") > pl.duration(minutes=15))
        .with_columns(
            (pl.col("delta").dt.total_minutes() / 15 - 1)
            .cast(pl.Int64)
            .alias("missing_intervals"),
            pl.col("timestamp_utc").alias("gap_end"),
        )
        .with_columns(
            (
                pl.col("gap_end")
                - pl.duration(minutes=15) * (pl.col("missing_intervals") + 1)
            ).alias("previous_observation")
        )
        .collect()
    )


def build_hourly_panel(
    meter: pl.LazyFrame | None = None,
    metadata: pl.DataFrame | None = None,
    weather: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a household-hour panel and explicitly flag incomplete hours."""
    meter = meter if meter is not None else scan_meter_data()
    metadata = metadata if metadata is not None else load_household_metadata()
    weather = weather if weather is not None else load_weather_data()

    hourly = (
        meter.with_columns(pl.col("timestamp_utc").dt.truncate("1h").alias("timestamp_utc"))
        .group_by("Household_ID", "timestamp_utc", "Group", "AffectsTimePoint")
        .agg(
            pl.len().alias("interval_rows"),
            pl.col("kWh_received_Total").count().alias("valid_total_intervals"),
            pl.col("kWh_received_HeatPump").count().alias("valid_heatpump_intervals"),
            pl.col("kWh_received_Other").count().alias("valid_other_intervals"),
            pl.col("kWh_received_Total").sum().alias("hourly_kwh_raw"),
            pl.col("kWh_received_HeatPump").sum().alias("hourly_heatpump_kwh_raw"),
            pl.col("kWh_received_Other").sum().alias("hourly_other_kwh_raw"),
        )
        .with_columns(
            (
                (pl.col("interval_rows") == 4)
                & (pl.col("valid_total_intervals") == 4)
            ).alias("hour_complete")
        )
        .with_columns(
            pl.when(pl.col("hour_complete"))
            .then(pl.col("hourly_kwh_raw"))
            .otherwise(None)
            .alias("hourly_kwh"),
            pl.when(pl.col("valid_heatpump_intervals") == 4)
            .then(pl.col("hourly_heatpump_kwh_raw"))
            .otherwise(None)
            .alias("hourly_heatpump_kwh"),
            pl.when(pl.col("valid_other_intervals") == 4)
            .then(pl.col("hourly_other_kwh_raw"))
            .otherwise(None)
            .alias("hourly_other_kwh"),
        )
        .collect()
    )

    panel = hourly.join(_analysis_household_fields(metadata), on="Household_ID", how="left").join(
        _analysis_weather_fields(weather), on=["Weather_ID", "timestamp_utc"], how="left"
    )
    return panel.with_columns(
        pl.col("timestamp_utc")
        .dt.convert_time_zone("Europe/Berlin")
        .alias("timestamp_local")
    ).with_columns(*_calendar_columns())


def add_exact_lags(
    panel: pl.DataFrame,
    lags: tuple[int, ...] = (1, 4, 96, 192, 672),
    target: str = "kWh_received_Total",
) -> pl.DataFrame:
    """Attach exact timestamp lags; gaps never collapse into adjacent row shifts."""
    result = panel
    source = panel.select("Household_ID", "timestamp_utc", target)
    for lag in lags:
        shifted = source.select(
            "Household_ID",
            (pl.col("timestamp_utc") + pl.duration(minutes=15 * lag)).alias(
                "timestamp_utc"
            ),
            pl.col(target).alias(f"lag_{lag}"),
        )
        result = result.join(shifted, on=["Household_ID", "timestamp_utc"], how="left")
    return result


def temporal_baseline_predictions(
    panel: pl.DataFrame, test_days: int = 28
) -> tuple[pl.DataFrame, datetime]:
    """Create leakage-safe rolling persistence and fixed-training climatology forecasts."""
    end = panel.select(pl.col("timestamp_utc").max()).item()
    if end is None:
        raise ValueError("Cannot evaluate baselines on an empty panel")
    test_start = end - timedelta(days=test_days) + timedelta(minutes=15)

    lagged = add_exact_lags(panel, lags=(96, 672))
    historical_mean = (
        panel.filter(pl.col("timestamp_utc") < test_start)
        .group_by("Household_ID", "local_slot")
        .agg(
            pl.col("kWh_received_Total")
            .mean()
            .alias("prediction_historical_tod")
        )
    )
    predictions = (
        lagged.filter(pl.col("timestamp_utc") >= test_start)
        .join(historical_mean, on=["Household_ID", "local_slot"], how="left")
        .rename(
            {
                "lag_96": "prediction_previous_day",
                "lag_672": "prediction_previous_week",
            }
        )
    )
    return predictions, test_start


def stable_cohort(
    panel: pl.DataFrame, window_days: int = 180, minimum_completeness: float = 0.95
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return a trailing-window panel and per-household stable-cohort diagnostics."""
    end = panel.select(pl.col("timestamp_utc").max()).item()
    if end is None:
        raise ValueError("Cannot select a stable cohort from an empty panel")
    start = end - timedelta(days=window_days) + timedelta(hours=1)
    expected_hours = window_days * 24
    window = panel.filter(pl.col("timestamp_utc").is_between(start, end))
    cohort = (
        window.group_by("Household_ID")
        .agg(
            pl.col("hour_complete").sum().alias("complete_hours"),
            pl.col("timestamp_utc").n_unique().alias("observed_hours"),
        )
        .with_columns(
            (pl.col("complete_hours") / expected_hours).alias("completeness")
        )
        .with_columns(
            (pl.col("completeness") >= minimum_completeness).alias("is_stable")
        )
        .sort("completeness", descending=True)
    )
    stable_ids = cohort.filter("is_stable").get_column("Household_ID")
    return window.filter(pl.col("Household_ID").is_in(stable_ids)), cohort


def build_portfolio(panel: pl.DataFrame) -> pl.DataFrame:
    """Aggregate a panel while retaining its changing observation denominator."""
    cohort_size = panel.get_column("Household_ID").n_unique()
    return (
        panel.group_by("timestamp_utc")
        .agg(
            pl.col("hourly_kwh").sum().alias("portfolio_kwh"),
            pl.col("hourly_kwh").mean().alias("mean_kwh_per_active_household"),
            pl.col("hourly_kwh").count().alias("active_households"),
            pl.col("Temperature_avg_hourly").mean().alias("temperature_c"),
        )
        .with_columns(
            pl.lit(cohort_size).alias("cohort_size"),
            (pl.col("active_households") / cohort_size).alias("active_share"),
        )
        .sort("timestamp_utc")
    )
