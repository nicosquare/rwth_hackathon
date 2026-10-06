from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.modeling import (
    ForecastSpec, SplitSpec, aggregate_portfolio, apply_conformal,
    aggregate_clusters, attach_cluster_labels, cluster_model_plan,
    assert_feature_availability, choose_operational_winner, conformal_quantiles,
    fit_nonnegative_ensemble, interval_metrics, regression_metrics,
    split_contiguous_series,
    validate_cluster_labels,
)


def test_splits_are_strict_and_match_contract():
    split = SplitSpec.from_day_counts("2022-02-28")
    assert split.validation_start == pd.Timestamp("2022-11-30", tz="UTC")
    assert split.calibration_start == pd.Timestamp("2023-02-28", tz="UTC")
    assert split.test_start == pd.Timestamp("2023-05-29", tz="UTC")
    assert split.test_end == pd.Timestamp("2024-02-28", tz="UTC")
    intervals = list(split.intervals().values())
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))


def test_gap_segmentation_never_interpolates():
    spec = ForecastSpec(horizon=2, lags=(1, 2))
    first = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    second = pd.date_range("2024-01-02", periods=4, freq="15min", tz="UTC")
    frame = pd.DataFrame({"Household_ID": "A", "timestamp_utc": first.append(second), "actual": 1.0})
    runs = split_contiguous_series(frame, spec)
    assert [len(run) for run in runs] == [5, 4]


def test_feature_availability_rejects_leakage_and_allows_labelled_oracle():
    origin = pd.Timestamp("2024-01-02", tz="UTC")
    safe = pd.DataFrame({"origin": [origin, origin],
        "feature_timestamp": [origin - pd.Timedelta(minutes=15), origin],
        "feature_family": ["target", "calendar"]})
    assert_feature_availability(safe)
    bad = safe.copy(); bad.loc[0, "feature_timestamp"] = origin
    with pytest.raises(AssertionError): assert_feature_availability(bad)
    oracle = pd.DataFrame({"origin": [origin], "feature_timestamp": [origin],
                           "feature_family": ["oracle_weather"]})
    with pytest.raises(AssertionError): assert_feature_availability(oracle)
    assert_feature_availability(oracle, oracle=True)


def test_portfolio_fixed_scaling_and_coverage_filter():
    rows = pd.DataFrame({"model": "m", "origin": pd.Timestamp("2024-01-01", tz="UTC"),
        "target_timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "horizon_step": 1,
        "Household_ID": ["a", "b", "c", "d"], "actual": [1, 3, 5, np.nan],
        "prediction": [2, 4, 6, np.nan]})
    kept = aggregate_portfolio(rows, cohort_size=4, minimum_coverage=.75)
    assert kept.iloc[0].actual == 12 and kept.iloc[0].prediction == 16
    assert kept.iloc[0].active_share == .75
    assert aggregate_portfolio(rows, cohort_size=4, minimum_coverage=.98).empty


def test_cluster_labels_are_static_and_cluster_aggregation_has_fixed_denominators():
    labels = pd.DataFrame({
        "Household_ID": ["a", "b", "c", "d"],
        "cluster_id": ["warm", "warm", "cold", "cold"],
    })
    predictions = pd.DataFrame({
        "model": "global",
        "origin": pd.Timestamp("2024-01-01", tz="UTC"),
        "target_timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        "horizon_step": 1,
        "Household_ID": ["a", "b", "c", "d"],
        "actual": [1., 3., 5., np.nan],
        "prediction": [2., 4., 6., np.nan],
    })
    attached = attach_cluster_labels(predictions, labels)
    assert set(attached["cluster_id"]) == {"warm", "cold"}
    kept = aggregate_clusters(predictions, labels, minimum_coverage=.5)
    cold = kept.loc[kept.cluster_id == "cold"].iloc[0]
    assert cold.actual == 10 and cold.prediction == 12
    assert cold.cohort_size == 2 and cold.active_share == .5
    strict = aggregate_clusters(predictions, labels, minimum_coverage=.98)
    assert set(strict.cluster_id) == {"warm"}


def test_cluster_labels_and_small_clusters_use_explicit_global_fallback():
    labels = pd.DataFrame({
        "Household_ID": ["a", "b", "c"],
        "cluster_id": ["large", "large", "small"],
    })
    with pytest.raises(ValueError):
        validate_cluster_labels(labels.iloc[[0, 0]])
    plan = cluster_model_plan(
        labels,
        pd.DataFrame({"Household_ID": ["a", "b", "c"], "training_rows": [600, 600, 500]}),
        min_households=2,
        min_rows=1000,
    ).set_index("cluster_id")
    assert plan.loc["large", "fit_mode"] == "cluster"
    assert plan.loc["small", "fit_mode"] == "global_fallback"


def test_metrics_ensemble_and_conformal():
    frame = pd.DataFrame({"actual": [1., 2.], "prediction": [2., 2.]})
    metrics = regression_metrics(frame)
    assert metrics["mae"] == .5 and metrics["wape"] == pytest.approx(1 / 3)
    base = pd.DataFrame({"origin": [1, 1, 1, 1], "target_timestamp": [1, 1, 2, 2],
        "horizon_step": [1, 1, 2, 2], "Household_ID": ["a"] * 4,
        "actual": [1., 1., 2., 2.], "model": ["previous_day", "previous_week"] * 2,
        "prediction": [1., 3., 2., 4.]})
    assert fit_nonnegative_ensemble(base) == (1.0, 0.0)
    calibration = pd.DataFrame({"horizon_step": [1] * 10,
        "actual": np.arange(10.), "prediction": np.arange(10.) + np.arange(10.)})
    q = conformal_quantiles(calibration, .9)
    assert q.iloc[0].radius == 9
    pred = pd.DataFrame({"horizon_step": [1], "prediction": [3.], "actual": [4.]})
    bounded = apply_conformal(pred, q)
    assert bounded.iloc[0].lower == 0 and interval_metrics(bounded)["interval_coverage"] == 1


def test_oracle_cannot_win_operational_selection():
    board = pd.DataFrame({"model": ["oracle", "safe"], "mae": [0., 1.],
                          "wape": [0., .1], "is_oracle": [True, False]})
    assert choose_operational_winner(board).model == "safe"


def test_darts_single_model_lags_are_relative_to_common_origin():
    from darts import TimeSeries
    from darts.models import RegressionModel
    from sklearn.linear_model import Ridge

    values = np.arange(1000, dtype=float)
    series = TimeSeries.from_times_and_values(
        pd.date_range("2024-01-01", periods=len(values), freq="15min"),
        values,
        columns=["load"],
    )
    model = RegressionModel(
        lags=[-1, -4, -96, -192, -672], output_chunk_length=96,
        multi_models=False, model=Ridge(),
    )
    X, y, _ = model._create_lagged_data(
        [series], None, None, max_samples_per_ts=1, stride=96,
    )
    # The output chunk is 904..999. All five features are anchored at its
    # common origin (904), rather than moving with the 96 output timestamps.
    np.testing.assert_array_equal(X[0], [232, 712, 808, 900, 903])
    assert y[0] == 999
