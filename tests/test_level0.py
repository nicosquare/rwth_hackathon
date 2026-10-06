from __future__ import annotations

import numpy as np
import pandas as pd

from utils.level0 import build_design
from utils.modeling import DAY_AHEAD


def _arrays(n_days: int = 60, n_hh: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "ids": [str(i) for i in range(n_hh)],
        "matrix": rng.gamma(2.0, .2, size=(n_days * 96, n_hh)).astype(np.float32),
        "static": rng.normal(size=(n_hh, 5)).astype(np.float32),
        "static_names": ["pv_status", "building_type", "heatpump_type", "living_area", "residents"],
        "pv_status": np.array(["true", "false", "unknown"][:n_hh]),
        "temperature": rng.normal(5, 3, size=(n_days * 24, n_hh)).astype(np.float32),
    }


def test_day_ahead_contract_cuts_data_at_10_utc_on_previous_day():
    assert DAY_AHEAD.cutoff("2023-06-15") == pd.Timestamp("2023-06-14 10:00", tz="UTC")
    assert DAY_AHEAD.gap_steps == 56


def test_features_use_only_data_before_the_day_ahead_cutoff():
    arrays = _arrays()
    day = 45
    X, y, scale = build_design(arrays, np.array([day]))

    cutoff_step = day * 96 - DAY_AHEAD.gap_steps
    blanked = {**arrays, "matrix": arrays["matrix"].copy(), "temperature": arrays["temperature"].copy()}
    blanked["matrix"][cutoff_step:] = np.nan
    blanked["temperature"][cutoff_step // 4:] = np.nan
    X_blank, y_blank, scale_blank = build_design(blanked, np.array([day]))

    pd.testing.assert_frame_equal(X, X_blank)
    np.testing.assert_array_equal(scale, scale_blank)
    assert np.isfinite(y).all() and np.isnan(y_blank).all()


def test_same_slot_lags_are_target_aligned_and_respect_the_cutoff():
    arrays = _arrays()
    day = 45
    X, _, scale = build_design(arrays, np.array([day]))
    n_hh = len(arrays["ids"])
    slot = np.repeat(np.arange(96), n_hh)          # rows are ordered (day, slot, household)
    hh = np.tile(np.arange(n_hh), 96)
    d1 = arrays["matrix"][day * 96 - 96 + slot, hh]
    d2 = arrays["matrix"][day * 96 - 192 + slot, hh]
    known_d1 = slot < 96 - DAY_AHEAD.gap_steps      # D-1 slots before 10:00 UTC

    lag_d1 = X.lag_d1.to_numpy() * scale
    np.testing.assert_allclose(lag_d1[known_d1], d1[known_d1], rtol=1e-5)
    assert np.isnan(lag_d1[~known_d1]).all()
    np.testing.assert_allclose(X.lag_d2.to_numpy() * scale, d2, rtol=1e-5)
    np.testing.assert_allclose(X.latest_same_slot.to_numpy() * scale,
                               np.where(known_d1, d1, d2), rtol=1e-5)
