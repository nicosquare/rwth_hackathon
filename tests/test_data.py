from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from utils.data import temporal_baseline_predictions
from utils.modeling import DAY_AHEAD


def test_previous_day_baseline_only_uses_slots_observed_before_the_cutoff():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=15 * i) for i in range(96 * 20)]
    panel = pl.DataFrame({
        "Household_ID": ["a"] * len(ts), "timestamp_utc": ts,
        "kWh_received_Total": [float(i) for i in range(len(ts))],  # value == step index
        "local_slot": [i % 96 for i in range(len(ts))],
    })
    predictions, test_start = temporal_baseline_predictions(panel, test_days=3)
    assert test_start.hour == 0 and test_start.minute == 0
    gap = predictions.with_columns(
        (pl.col("kWh_received_Total") - pl.col("prediction_previous_day")).alias("gap"),
        (pl.col("timestamp_utc").dt.hour() < DAY_AHEAD.data_cutoff_hour_utc).alias("before_cutoff_hour"),
    )
    assert set(gap.filter("before_cutoff_hour")["gap"]) == {96.0}
    assert set(gap.filter(~pl.col("before_cutoff_hour"))["gap"]) == {192.0}
