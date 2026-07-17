from __future__ import annotations

import numpy as np

from src.evaluation.buffered_crop_benchmark import (
    cosine_core_window,
    external_sponge_mask,
    prepare_buffered_case,
    run_buffered_case,
)


def test_cosine_core_window_is_exact_zero_at_edges() -> None:
    window = cosine_core_window((64, 64), 8)
    assert np.array_equal(window[[0, -1], :], np.zeros((2, 64)))
    assert np.array_equal(window[:, [0, -1]], np.zeros((64, 2)))
    assert np.all(window[7:-7, 7:-7] == 1.0)


def test_buffered_case_preserves_core_bathymetry_and_has_zero_exterior_source() -> None:
    bathymetry = -np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / 4096.0
    eta0 = np.ones((64, 64), dtype=np.float64)
    prepared = prepare_buffered_case(
        bathymetry,
        eta0,
        buffer_cells=16,
        source_taper_cells=8,
    )
    crop = prepared["crop"]
    assert prepared["bathymetry"].shape == (96, 96)
    assert np.array_equal(prepared["bathymetry"][crop], bathymetry)
    assert np.count_nonzero(prepared["eta0"][:16]) == 0
    assert np.count_nonzero(prepared["eta0"][-16:]) == 0
    assert np.count_nonzero(prepared["eta0"][:, :16]) == 0
    assert np.count_nonzero(prepared["eta0"][:, -16:]) == 0


def test_external_sponge_never_enters_central_crop() -> None:
    mask = external_sponge_mask((128, 128), buffer_cells=32, min_factor=0.8)
    assert np.array_equal(mask[32:96, 32:96], np.ones((64, 64)))
    assert float(np.min(mask)) == 0.8


def test_fixed_sponge_width_must_fit_inside_the_exterior_buffer() -> None:
    record = {
        "qualified_id": "train:scenario_000001",
    }
    with np.testing.assert_raises_regex(ValueError, "fit entirely"):
        run_buffered_case(
            record,
            solver_name="swe_hydrostatic",
            total_grid=96,
            sponge_width_cells=17,
        )
