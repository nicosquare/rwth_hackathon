from __future__ import annotations

import numpy as np
import pytest

from utils.modeling import DAY_AHEAD
from utils.seq_model import HISTORY_DAYS, history_grid, sample_inputs


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


def test_sequence_inputs_use_only_data_before_the_day_ahead_cutoff():
    arrays = _arrays()
    day = 45
    days, hh = np.array([day, day, day]), np.array([0, 1, 2])
    reference = sample_inputs(arrays, days, hh)

    cutoff_step = day * 96 - DAY_AHEAD.gap_steps
    blanked = {**arrays, "matrix": arrays["matrix"].copy(), "temperature": arrays["temperature"].copy()}
    blanked["matrix"][cutoff_step:] = np.nan
    blanked["temperature"][cutoff_step // 4:] = np.nan
    after = sample_inputs(blanked, days, hh)

    for key in reference:
        np.testing.assert_array_equal(reference[key], after[key], err_msg=key)


def test_history_grid_is_slot_aligned_and_masks_the_cutoff():
    arrays = _arrays()
    day, hh = 45, 1
    values, known = history_grid(arrays["matrix"], np.array([day]), np.array([hh]))
    m = arrays["matrix"]
    for k in (1, 2, 7, HISTORY_DAYS):
        expected = m[(day - k) * 96 + np.arange(96), hh]
        visible = known[0, k - 1]
        np.testing.assert_array_equal(values[0, k - 1][visible], expected[visible])
    first_unknown = 96 - DAY_AHEAD.gap_steps  # D-1 slots from 10:00 UTC are not yet observed
    assert known[0, 0, :first_unknown].all() and not known[0, 0, first_unknown:].any()
    assert known[0, 1:].all()


def test_gpu_gather_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    arrays = _arrays()
    m = arrays["matrix"].copy()
    m[1000:1100, 0] = np.nan  # missing readings must be masked identically
    days, hh = np.array([40, 45, 50]), np.array([0, 1, 2])
    ref_values, ref_known = history_grid(m, days, hh)
    values, known = history_grid(torch.from_numpy(m), torch.from_numpy(days), torch.from_numpy(hh), xp=torch)
    np.testing.assert_array_equal(values.numpy(), ref_values)
    np.testing.assert_array_equal(known.numpy(), ref_known)


@pytest.mark.parametrize("arch", ["cnn", "gru"])
def test_sequence_models_output_all_96_slots(arch):
    torch = pytest.importorskip("torch")
    from utils.seq_model import build_model
    model = build_model(n_context=29, arch=arch)
    grid, mask = torch.randn(4, HISTORY_DAYS, 96), torch.ones(4, HISTORY_DAYS, 96)
    out = model(grid, mask, torch.randn(4, 29), torch.zeros(4))
    assert out.shape == (4, 96) and torch.isfinite(out).all()
