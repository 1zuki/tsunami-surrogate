from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import candidate_requested_times
from src.data_gen.preprocess import TsunamiPreprocessor
from src.data_gen.simulate_dataset import (
    _extract_requested_states_from_bracket,
    _simulate_one_local,
)
from src.solver.boussinesq import BoussinesqSolver


class _HealthFakeSolver:
    def __init__(self) -> None:
        self.state = np.zeros((3, 2, 2), dtype=np.float32)
        self.dt = 0.3
        self.h = self.state[0]
        self.hu = self.state[1]
        self.hv = self.state[2]
        self.dry_tolerance = 1.0e-6

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def compute_velocity(self):
        return np.zeros_like(self.h), np.zeros_like(self.h)

    def compute_cfl(self, *, dt: float) -> float:
        return float(dt * 2.0)

    def step(self, *, dt: float, auto_dt: bool) -> None:
        assert not auto_dt
        self.state = self.state + np.float32(dt)
        self.h = self.state[0]
        self.hu = self.state[1]
        self.hv = self.state[2]


def test_requested_mode_collects_natural_health_without_dense_state_recording() -> None:
    solver = _HealthFakeSolver()
    requested = np.asarray([0.1, 0.2, 0.3, 0.55], dtype=np.float64)
    trajectory, timestamps, dt_history, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=1,
        save_every=99,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
        requested_times=requested,
        max_natural_steps=3,
        collect_natural_step_health=True,
    )
    assert trajectory.shape[0] == requested.size
    np.testing.assert_array_equal(timestamps, requested)
    np.testing.assert_array_equal(dt_history, [0.3, 0.3])
    for key in (
        "natural_health_step_indices",
        "left_natural_step_times",
        "right_natural_step_times",
        "proposed_dt",
        "pre_step_cfl",
        "post_step_cfl",
        "finite_state_flag",
        "swe_min_depth",
        "swe_max_speed",
        "swe_dry_cell_count",
    ):
        assert diagnostics[key].shape == (2,)
    assert diagnostics["proposed_dt"].dtype == np.float64
    assert diagnostics["final_natural_timestamp"][0] == pytest.approx(0.6)


def test_interpolated_outputs_never_feed_back_into_natural_solver() -> None:
    solver_a = _HealthFakeSolver()
    solver_b = _HealthFakeSolver()
    _, _, dt_a, _ = _simulate_one_local(
        solver=solver_a,
        n_steps=1,
        save_every=1,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
        requested_times=np.asarray([0.05, 0.15, 0.6], dtype=np.float64),
        max_natural_steps=3,
    )
    _, _, dt_b, _ = _simulate_one_local(
        solver=solver_b,
        n_steps=1,
        save_every=1,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
        requested_times=np.asarray([0.2, 0.4, 0.6], dtype=np.float64),
        max_natural_steps=3,
    )
    np.testing.assert_array_equal(dt_a, dt_b)
    np.testing.assert_array_equal(solver_a.get_state(), solver_b.get_state())


def test_swe_state_then_eta_matches_direct_eta_interpolation() -> None:
    bathymetry = np.asarray([[-1.0, -0.5]], dtype=np.float32)
    left = np.asarray([[[1.1, 0.7]], [[0.2, 0.3]], [[0.0, 0.1]]], dtype=np.float32)
    right = np.asarray([[[1.5, 0.9]], [[0.4, 0.5]], [[0.2, 0.3]]], dtype=np.float32)
    requested = np.asarray([0.25, 0.75], dtype=np.float64)
    states, _ = _extract_requested_states_from_bracket(
        left_state=left,
        right_state=right,
        left_time=0.0,
        right_time=1.0,
        requested_times=requested,
        right_natural_step_index=1,
    )
    eta_from_state = states[:, 0] + bathymetry[None]
    weights = requested[:, None, None]
    eta_left = left[0] + bathymetry
    eta_right = right[0] + bathymetry
    direct_eta = eta_left[None] * (1.0 - weights) + eta_right[None] * weights
    np.testing.assert_allclose(eta_from_state, direct_eta, rtol=1e-6, atol=1e-6)


def _preprocess_config(tmp_path: Path, variable: str = "eta") -> Path:
    config = {
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "manifest_path": str(tmp_path / "manifest.jsonl"),
        "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 1},
        "input": {
            "use_bathymetry": True,
            "use_source": True,
            "use_initial_depth": True,
        },
        "target": {
            "mode": "multi_step",
            "variable": variable,
            "forecast_steps": 2,
            "stride": 1,
        },
        "normalization": {"method": "standardize", "channels": {"trajectory": False}},
        "saving": {"sharded": False},
    }
    path = tmp_path / f"preprocess-{variable}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_eta_primary_preprocessing_does_not_require_full_state(tmp_path: Path) -> None:
    preprocessor = TsunamiPreprocessor(str(_preprocess_config(tmp_path, "eta")))
    sample = {
        "bathymetry": np.zeros((2, 2), dtype=np.float32),
        "source_field": np.ones((2, 2), dtype=np.float32),
        "initial_depth": np.ones((2, 2), dtype=np.float32),
        "trajectory_eta": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
        "meta": {"schema_id": "tsunami-surrogate.common-time-v2.eta-sample.v1"},
    }
    _, target = preprocessor.build_example(sample)
    assert target.shape == (2, 2, 2)
    np.testing.assert_array_equal(target[0], sample["trajectory_eta"][0])

    state_preprocessor = TsunamiPreprocessor(str(_preprocess_config(tmp_path, "state")))
    with pytest.raises(KeyError, match="trajectory is required"):
        state_preprocessor.build_example(sample)


@pytest.mark.slow
def test_real_boussinesq_clean_mechanics_reaches_0175() -> None:
    nx = ny = 12
    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=1.0 / nx,
        dy=1.0 / ny,
        dt=0.001,
        g=1.0,
        cfl=0.35,
        alpha=1.0 / 3.0,
        min_depth=1.0e-3,
        boundary="periodic",
        mode="linear_constant_depth",
        use_sponge=False,
        filter_strength=0.0,
        linear_solver_tol=1.0e-8,
        linear_solver_max_iter=200,
        check_finite=True,
    )
    bathymetry = np.full((nx, ny), -1.0, dtype=np.float32)
    x = np.arange(nx, dtype=np.float64)[:, None] / nx
    y = np.arange(ny, dtype=np.float64)[None, :] / ny
    eta0 = (1.0e-4 * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)).astype(
        np.float32
    )
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
    trajectory, timestamps, dt_history, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=0.35,
        include_initial_state=False,
        requested_times=candidate_requested_times(),
        max_natural_steps=1000,
        collect_natural_step_health=True,
    )
    assert trajectory.shape == (50, 2, nx, ny)
    np.testing.assert_array_equal(timestamps, candidate_requested_times())
    assert np.isfinite(trajectory).all()
    assert dt_history.size == diagnostics["total_natural_steps"][0]
    assert not np.any(diagnostics["filter_enabled"])
    assert np.sum(diagnostics["cg_failed_count"]) == 0
