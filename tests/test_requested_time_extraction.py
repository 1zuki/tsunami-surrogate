from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_gen import simulate_dataset as simulate_module
from src.data_gen.simulate_dataset import (
    _extract_requested_states_from_bracket,
    _run_fde_rollout,
    _simulate_one_local,
)


class _AffineFakeSolver:
    def __init__(
        self,
        state: np.ndarray,
        *,
        dt: float = 0.1,
        suggested_dts: list[float] | None = None,
        slope: np.ndarray | None = None,
    ) -> None:
        self.state = np.asarray(state, dtype=np.float32).copy()
        self.dt = float(dt)
        self._suggested_dts = list(suggested_dts or [])
        self.slope = np.ones_like(self.state) if slope is None else np.asarray(slope)
        self.step_dts: list[float] = []
        self.suggested_target_cfls: list[float] = []

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def suggest_dt(self, *, target_cfl: float) -> float:
        self.suggested_target_cfls.append(float(target_cfl))
        if not self._suggested_dts:
            raise RuntimeError("No fake suggested timestep remains")
        return float(self._suggested_dts.pop(0))

    def step(self, *, dt: float, auto_dt: bool) -> None:
        assert not auto_dt
        natural_dt = float(dt)
        self.step_dts.append(natural_dt)
        self.state = np.asarray(
            self.state.astype(np.float64) + self.slope.astype(np.float64) * natural_dt,
            dtype=np.float32,
        )


class _FactoryFakeSolver(_AffineFakeSolver):
    def __init__(self, *, state_channels: int, dt: float = 0.1) -> None:
        super().__init__(np.zeros((state_channels, 2, 2), dtype=np.float32), dt=dt)

    def set_bathymetry(self, bathymetry: np.ndarray) -> None:
        self.bathymetry = np.asarray(bathymetry, dtype=np.float32).copy()

    def set_initial_condition(self, first: np.ndarray, **kwargs: np.ndarray) -> None:
        first_values = np.asarray(first, dtype=np.float32)
        if self.state.shape[0] == 3:
            self.state = np.stack(
                (
                    first_values,
                    np.asarray(kwargs["hu0"], dtype=np.float32),
                    np.asarray(kwargs["hv0"], dtype=np.float32),
                ),
                axis=0,
            )
        else:
            self.state = np.stack(
                (first_values, np.asarray(kwargs["eta_t0"], dtype=np.float32)),
                axis=0,
            )


def _requested_rollout(
    solver: _AffineFakeSolver,
    requested_times: np.ndarray,
    *,
    auto_dt: bool = False,
    max_natural_steps: int = 10,
):
    return _simulate_one_local(
        solver=solver,
        n_steps=1,
        save_every=99,
        auto_dt=auto_dt,
        target_cfl=0.37,
        include_initial_state=True,
        requested_times=requested_times,
        max_natural_steps=max_natural_steps,
    )


def test_bracket_extractor_is_exact_for_affine_multicomponent_states() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    slope = np.linspace(0.5, 2.0, 12, dtype=np.float32).reshape(3, 2, 2)
    left_time = 0.2
    right_time = 0.8
    left = base + np.float32(left_time) * slope
    right = base + np.float32(right_time) * slope
    requested = np.asarray([0.35, 0.6], dtype=np.float64)

    states, provenance = _extract_requested_states_from_bracket(
        left_state=left,
        right_state=right,
        left_time=left_time,
        right_time=right_time,
        requested_times=requested,
        right_natural_step_index=4,
    )

    expected = np.stack([base + np.float32(t) * slope for t in requested])
    np.testing.assert_allclose(states, expected, rtol=1.0e-6, atol=1.0e-6)
    assert states.dtype == np.float32
    assert np.array_equal(provenance["requested_timestamps"], requested)
    np.testing.assert_allclose(
        provenance["interpolation_weights"],
        (requested - left_time) / (right_time - left_time),
    )
    assert np.array_equal(
        provenance["left_natural_timestamps"], np.full(2, left_time)
    )
    assert np.array_equal(
        provenance["right_natural_timestamps"], np.full(2, right_time)
    )
    np.testing.assert_allclose(provenance["bracket_widths"], np.full(2, 0.6))
    assert not np.any(provenance["exact_knot"])
    assert np.array_equal(provenance["natural_step_indices"], np.full(2, 4))
    for key in (
        "requested_timestamps",
        "left_natural_timestamps",
        "right_natural_timestamps",
        "interpolation_weights",
        "bracket_widths",
    ):
        assert provenance[key].dtype == np.float64


def test_bracket_extractor_handles_multiple_requests_and_canonical_knots() -> None:
    left = np.zeros((2, 1, 1), dtype=np.float32)
    right = np.full((2, 1, 1), 10.0, dtype=np.float32)
    requested = np.asarray([1.0, 1.25, 1.75, 2.0], dtype=np.float64)

    states, provenance = _extract_requested_states_from_bracket(
        left_state=left,
        right_state=right,
        left_time=1.0,
        right_time=2.0,
        requested_times=requested,
        right_natural_step_index=8,
    )

    np.testing.assert_allclose(states[:, 0, 0, 0], [0.0, 2.5, 7.5, 10.0])
    assert np.array_equal(provenance["exact_knot"], [True, False, False, True])
    assert np.array_equal(provenance["interpolation_weights"], [0.0, 0.25, 0.75, 0.0])
    assert np.array_equal(provenance["bracket_widths"], [0.0, 1.0, 1.0, 0.0])
    assert np.array_equal(provenance["natural_step_indices"], [7, 8, 8, 8])
    assert provenance["left_natural_timestamps"][0] == requested[0]
    assert provenance["right_natural_timestamps"][0] == requested[0]
    assert provenance["left_natural_timestamps"][-1] == requested[-1]
    assert provenance["right_natural_timestamps"][-1] == requested[-1]


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"right_state": np.zeros((3, 1), dtype=np.float32)}, "identical shapes"),
        ({"left_time": 1.0, "right_time": 1.0}, "strictly increasing"),
        ({"requested_times": np.asarray([], dtype=np.float64)}, "non-empty 1-D"),
        ({"requested_times": np.asarray([0.5, 0.4])}, "strictly increasing"),
        ({"requested_times": np.asarray([0.1, 0.5])}, "beyond"),
        ({"requested_times": np.asarray([0.5, 1.1])}, "beyond"),
        ({"requested_times": np.asarray([np.nan])}, "finite"),
    ],
)
def test_bracket_extractor_rejects_invalid_inputs(changes, match: str) -> None:
    kwargs = {
        "left_state": np.zeros((2, 1), dtype=np.float32),
        "right_state": np.ones((2, 1), dtype=np.float32),
        "left_time": 0.25,
        "right_time": 1.0,
        "requested_times": np.asarray([0.5], dtype=np.float64),
        "right_natural_step_index": 2,
    }
    kwargs.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        _extract_requested_states_from_bracket(**kwargs)


def test_requested_rollout_uses_unmodified_adaptive_steps_and_time_completion() -> None:
    initial = np.zeros((2, 1, 1), dtype=np.float32)
    slope = np.asarray([[[2.0]], [[-1.0]]], dtype=np.float32)
    solver = _AffineFakeSolver(
        initial,
        suggested_dts=[0.3, 0.4, 0.5],
        slope=slope,
    )
    requested = np.asarray([0.1, 0.2, 0.3, 0.55, 0.65], dtype=np.float64)

    trajectory, timestamps, dt_history, diagnostics = _requested_rollout(
        solver,
        requested,
        auto_dt=True,
        max_natural_steps=20,
    )

    assert np.array_equal(solver.step_dts, [0.3, 0.4])
    assert np.array_equal(solver.suggested_target_cfls, [0.37, 0.37])
    assert np.array_equal(timestamps, requested)
    assert timestamps.dtype == np.float64
    assert np.array_equal(dt_history, np.asarray([0.3, 0.4], dtype=np.float64))
    np.testing.assert_allclose(trajectory[:, 0, 0, 0], 2.0 * requested)
    np.testing.assert_allclose(trajectory[:, 1, 0, 0], -requested)
    assert trajectory.dtype == np.float32
    assert np.array_equal(diagnostics["natural_step_indices"], [1, 1, 1, 2, 2])
    assert np.array_equal(diagnostics["exact_knot"], [False, False, True, False, False])
    assert np.array_equal(diagnostics["total_natural_steps"], [2])
    assert np.array_equal(diagnostics["natural_dt_history"], dt_history)


def test_requested_rollout_exhaustion_and_invalid_dt_fail_without_result() -> None:
    solver = _AffineFakeSolver(np.zeros((1, 1, 1), dtype=np.float32), dt=0.1)
    with pytest.raises(
        RuntimeError,
        match=r"emitted=1/2.*next_missing_requested_time=0.5.*cap=2",
    ):
        _requested_rollout(
            solver,
            np.asarray([0.05, 0.5], dtype=np.float64),
            max_natural_steps=2,
        )

    invalid_solver = _AffineFakeSolver(
        np.zeros((1, 1, 1), dtype=np.float32), dt=np.nan
    )
    with pytest.raises(RuntimeError, match="finite and positive"):
        _requested_rollout(
            invalid_solver,
            np.asarray([0.1], dtype=np.float64),
            max_natural_steps=2,
        )


@pytest.mark.parametrize(
    "requested, cap, match",
    [
        (np.asarray([], dtype=np.float64), 2, "non-empty 1-D"),
        (np.asarray([[0.1]], dtype=np.float64), 2, "non-empty 1-D"),
        (np.asarray([0.0], dtype=np.float64), 2, "strictly positive"),
        (np.asarray([0.2, 0.1], dtype=np.float64), 2, "strictly increasing"),
        (np.asarray([0.1, 0.1], dtype=np.float64), 2, "strictly increasing"),
        (np.asarray([np.inf], dtype=np.float64), 2, "finite"),
        (np.asarray([0.1], dtype=np.float64), None, "positive integer"),
        (np.asarray([0.1], dtype=np.float64), 0, "positive integer"),
    ],
)
def test_requested_rollout_validates_grid_and_step_cap(requested, cap, match) -> None:
    solver = _AffineFakeSolver(np.zeros((1, 1, 1), dtype=np.float32), dt=0.1)
    with pytest.raises(ValueError, match=match):
        _simulate_one_local(
            solver=solver,
            n_steps=1,
            save_every=1,
            auto_dt=False,
            target_cfl=0.4,
            include_initial_state=True,
            requested_times=requested,
            max_natural_steps=cap,
        )


def test_requested_rollout_rejects_dense_modes_and_orphan_cap() -> None:
    requested = np.asarray([0.05], dtype=np.float64)
    for kwargs in (
        {"record_every_step": True},
        {"record_every_step": True, "dense_diagnostics": True},
    ):
        solver = _AffineFakeSolver(np.zeros((1, 1, 1), dtype=np.float32), dt=0.1)
        with pytest.raises(ValueError, match="does not yet support"):
            _simulate_one_local(
                solver=solver,
                n_steps=1,
                save_every=1,
                auto_dt=False,
                target_cfl=0.4,
                include_initial_state=True,
                requested_times=requested,
                max_natural_steps=2,
                **kwargs,
            )

    solver = _AffineFakeSolver(np.zeros((1, 1, 1), dtype=np.float32), dt=0.1)
    with pytest.raises(ValueError, match="requires requested_times"):
        _simulate_one_local(
            solver=solver,
            n_steps=1,
            save_every=1,
            auto_dt=False,
            target_cfl=0.4,
            include_initial_state=True,
            max_natural_steps=2,
        )


def test_legacy_no_grid_behavior_and_dtypes_remain_unchanged() -> None:
    solver = _AffineFakeSolver(np.zeros((1, 1, 1), dtype=np.float32), dt=0.1)
    trajectory, timestamps, dt_history, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=4,
        save_every=2,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
    )

    np.testing.assert_allclose(trajectory[:, 0, 0, 0], [0.0, 0.2, 0.4])
    assert np.array_equal(timestamps, np.asarray([0.0, 0.2, 0.4], dtype=np.float32))
    assert np.array_equal(dt_history, np.asarray([0.0, 0.1, 0.1], dtype=np.float32))
    assert trajectory.dtype == np.float32
    assert timestamps.dtype == np.float32
    assert dt_history.dtype == np.float32
    assert diagnostics == {}
    assert solver.step_dts == [0.1, 0.1, 0.1, 0.1]


@pytest.mark.parametrize(
    "fde_name, channels",
    [
        ("swe_hydrostatic", 3),
        ("swe_muscl_hr", 3),
        ("boussinesq", 2),
    ],
)
def test_run_fde_rollout_forwards_requested_mode_and_converts_elevation(
    monkeypatch: pytest.MonkeyPatch,
    fde_name: str,
    channels: int,
) -> None:
    factory_name = {
        "swe_hydrostatic": "_make_hydrostatic_solver_from_cfg",
        "swe_muscl_hr": "_make_muscl_solver_from_cfg",
        "boussinesq": "_make_boussinesq_solver_from_cfg",
    }[fde_name]
    created: list[_FactoryFakeSolver] = []

    def factory(_cfg):
        solver = _FactoryFakeSolver(state_channels=channels)
        created.append(solver)
        return solver

    monkeypatch.setattr(simulate_module, factory_name, factory)
    bathymetry = np.full((2, 2), -1.0, dtype=np.float32)
    eta0 = np.full((2, 2), 0.2, dtype=np.float32)
    h0 = np.full((2, 2), 1.2, dtype=np.float32)
    dataset = SimpleNamespace(
        n_steps=1,
        save_every=5,
        auto_dt=False,
        target_cfl=0.45,
        include_initial_state=True,
    )
    requested = np.asarray([0.05], dtype=np.float64)

    result = _run_fde_rollout(
        fde_name,
        {},
        dataset,
        bathymetry,
        eta0,
        h0,
        requested_times=requested,
        max_natural_steps=2,
    )

    assert len(created) == 1
    assert created[0].step_dts == [0.1]
    assert result.trajectory.shape == (1, channels, 2, 2)
    assert np.array_equal(result.timestamps, requested)
    assert result.timestamps.dtype == np.float64
    expected_eta = np.full((1, 2, 2), 0.25, dtype=np.float32)
    np.testing.assert_allclose(result.trajectory_eta, expected_eta)
    assert result.diagnostics is not None
    assert np.array_equal(result.diagnostics["total_natural_steps"], [1])
