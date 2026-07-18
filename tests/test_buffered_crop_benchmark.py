from __future__ import annotations

import numpy as np

from src.data_gen.common_time_v2 import candidate_requested_times
from src.evaluation import buffered_crop_benchmark
from src.evaluation.buffered_crop_benchmark import (
    cosine_core_window,
    external_sponge_mask,
    prepare_buffered_case,
    run_buffered_case,
    run_buffered_case_detailed,
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


def test_buffered_runner_uses_exact_canonical_requested_times(monkeypatch) -> None:
    from src.evaluation import common_time_v2_level_a

    shape = (6, 6)
    bathymetry = -np.ones(shape, dtype=np.float32)
    source = np.zeros(shape, dtype=np.float32)
    arrays = {"rest_depth": np.ones(shape, dtype=np.float32)}
    record = {
        "qualified_id": "train:scenario_000001",
        "input_fingerprint": "input",
        "bathymetry_type": "canyon",
        "source_type": "dipole",
    }

    monkeypatch.setattr(
        common_time_v2_level_a,
        "_load_canary_arrays",
        lambda _record: (
            bathymetry,
            source,
            np.asarray([0.5], dtype=np.float32),
            0.5,
            arrays,
        ),
    )

    class FakeSolver:
        def reset_operator_diagnostics(self) -> None:
            pass

        def set_bathymetry(self, values: np.ndarray) -> None:
            self.bathymetry = values

        def set_initial_condition(self, *args, **kwargs) -> None:
            pass

        def get_operator_diagnostics(self) -> dict[str, int]:
            return {}

    captured: dict[str, object] = {}

    def fake_make_solver(*args, **kwargs):
        captured["solver_target_cfl"] = kwargs["target_cfl"]
        return FakeSolver()

    monkeypatch.setattr(buffered_crop_benchmark, "_make_solver", fake_make_solver)
    monkeypatch.setattr(
        buffered_crop_benchmark,
        "external_sponge_mask",
        lambda values, **kwargs: np.ones(values, dtype=np.float64),
    )
    def fake_simulation(*args, requested_times, **kwargs):
        captured["requested_times"] = requested_times.copy()
        captured["simulation_target_cfl"] = kwargs["target_cfl"]
        states = np.zeros((requested_times.size, 3, 8, 8), dtype=np.float64)
        diagnostics = {"post_step_cfl": np.asarray([0.4], dtype=np.float64)}
        return (
            states,
            requested_times.copy(),
            np.asarray([0.001], dtype=np.float64),
            diagnostics,
        )

    monkeypatch.setattr(
        buffered_crop_benchmark, "_simulate_one_local", fake_simulation
    )
    _row, _trajectory, details = run_buffered_case_detailed(
        record,
        solver_name="swe_hydrostatic",
        total_grid=8,
        core_grid=6,
        source_taper_cells=2,
        sponge_width_cells=1,
    )

    expected = candidate_requested_times()
    np.testing.assert_array_equal(captured["requested_times"], expected)
    np.testing.assert_array_equal(details["requested_times"], expected)
    assert captured["solver_target_cfl"] == 0.45
    assert captured["simulation_target_cfl"] == 0.45
    assert details["target_cfl"] == 0.45
    assert captured["requested_times"][-1] == np.float64(0.175)
    assert captured["requested_times"][-1] != (
        np.arange(1, 51, dtype=np.float64) * np.float64(0.0035)
    )[-1]

    row, _trajectory, details = run_buffered_case_detailed(
        record,
        solver_name="swe_hydrostatic",
        total_grid=8,
        core_grid=6,
        source_taper_cells=2,
        sponge_width_cells=1,
        target_cfl=0.225,
    )
    assert captured["solver_target_cfl"] == 0.225
    assert captured["simulation_target_cfl"] == 0.225
    assert row["target_cfl"] == 0.225
    assert details["target_cfl"] == 0.225
