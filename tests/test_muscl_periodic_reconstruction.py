from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver


def _solver(
    *,
    boundary: str | tuple[str, str] = "periodic",
    limiter: str = "minmod",
    nx: int = 16,
    ny: int = 4,
) -> MUSCLHRShallowWaterSolver:
    return MUSCLHRShallowWaterSolver(
        nx=nx,
        ny=ny,
        dx=1.0 / nx,
        dy=1.0 / ny,
        dt=1.0e-4,
        boundary=boundary,
        use_sponge=False,
        reconstruction_limiter=limiter,
    )


def _minmod(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    same = np.sign(forward) == np.sign(backward)
    return np.where(
        same,
        np.sign(forward) * np.minimum(np.abs(forward), np.abs(backward)),
        0.0,
    )


def _smooth_state(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=np.float64)[:, None] / nx
    y = np.arange(ny, dtype=np.float64)[None, :] / ny
    eta = 1.0e-3 * np.cos(2.0 * np.pi * x) * np.ones((1, ny))
    eta += 2.0e-4 * np.sin(2.0 * np.pi * y) * np.ones((nx, 1))
    h = 1.0 + eta
    hu = np.sqrt(9.81) * eta
    return -np.ones((nx, ny), dtype=np.float64), np.stack(
        [h, hu, np.zeros_like(h)], axis=0
    )


def test_periodic_constant_state_is_preserved() -> None:
    solver = _solver()
    solver.set_bathymetry(-np.ones((solver.nx, solver.ny)))
    solver.set_initial_condition(
        np.ones((solver.nx, solver.ny)),
        hu0=np.full((solver.nx, solver.ny), 0.1),
        hv0=np.full((solver.nx, solver.ny), -0.05),
    )
    before = solver.get_state().copy()
    solver.step(dt=1.0e-4)
    np.testing.assert_allclose(solver.get_state(), before, rtol=0.0, atol=2.0e-15)


def test_periodic_slopes_use_wrapped_neighbors_at_both_seams() -> None:
    solver = _solver(nx=8, ny=6)
    field = np.arange(48, dtype=np.float64).reshape(8, 6) ** 2

    expected_x = _minmod(
        np.roll(field, -1, axis=0) - field,
        field - np.roll(field, 1, axis=0),
    )
    expected_y = _minmod(
        np.roll(field, -1, axis=1) - field,
        field - np.roll(field, 1, axis=1),
    )
    np.testing.assert_array_equal(solver._slope_x(field), expected_x)
    np.testing.assert_array_equal(solver._slope_y(field), expected_y)
    np.testing.assert_array_equal(solver._slope_x(field)[[0, -1]], expected_x[[0, -1]])
    np.testing.assert_array_equal(
        solver._slope_y(field)[:, [0, -1]], expected_y[:, [0, -1]]
    )


@pytest.mark.parametrize("limiter", ["minmod", "unlimited"])
def test_periodic_interfaces_use_reconstructed_opposite_faces(limiter: str) -> None:
    solver = _solver(nx=8, ny=4, limiter=limiter)
    bathymetry, state = _smooth_state(solver.nx, solver.ny)
    solver.set_bathymetry(bathymetry)
    rec = solver._reconstructed_faces(state[0], state[1], state[2], bathymetry)
    interface_states = solver._reconstructed_x_interface_states(
        rec, state[0], state[1], state[2], bathymetry
    )
    left = [values[0, 0] for values in interface_states[:4]]
    right = [values[-1, 0] for values in interface_states[4:]]
    np.testing.assert_allclose(
        left,
        [rec["h_e"][-1, 0], rec["hu_e"][-1, 0], rec["hv_e"][-1, 0], rec["b_e"][-1, 0]],
    )
    np.testing.assert_allclose(
        right,
        [rec["h_w"][0, 0], rec["hu_w"][0, 0], rec["hv_w"][0, 0], rec["b_w"][0, 0]],
    )


@pytest.mark.parametrize("limiter", ["minmod", "unlimited"])
def test_periodic_update_is_translation_invariant(limiter: str) -> None:
    bathymetry, state = _smooth_state(16, 4)
    shifted_bathymetry = np.roll(bathymetry, 3, axis=0)
    shifted_state = np.roll(state, 3, axis=1)
    reference = _solver(nx=16, ny=4, limiter=limiter)
    shifted = _solver(nx=16, ny=4, limiter=limiter)
    for solver, b, values in (
        (reference, bathymetry, state),
        (shifted, shifted_bathymetry, shifted_state),
    ):
        solver.set_bathymetry(b)
        solver.set_state(values)
        solver.step(dt=1.0e-4)
    np.testing.assert_allclose(
        shifted.get_state(),
        np.roll(reference.get_state(), 3, axis=1),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_periodic_update_conserves_water_depth() -> None:
    solver = _solver(nx=24, ny=6)
    bathymetry, state = _smooth_state(solver.nx, solver.ny)
    solver.set_bathymetry(bathymetry)
    solver.set_state(state)
    initial = float(np.sum(solver.h, dtype=np.float64))
    for _ in range(5):
        solver.step(dt=5.0e-5)
    assert float(np.sum(solver.h, dtype=np.float64)) == pytest.approx(
        initial, rel=0.0, abs=3.0e-13
    )


class _LegacyNonPeriodicMUSCL(MUSCLHRShallowWaterSolver):
    def _slope_x(self, field: np.ndarray) -> np.ndarray:
        slope = np.zeros_like(field)
        slope[1:-1] = self._minmod(field[2:] - field[1:-1], field[1:-1] - field[:-2])
        return slope

    def _slope_y(self, field: np.ndarray) -> np.ndarray:
        slope = np.zeros_like(field)
        slope[:, 1:-1] = self._minmod(
            field[:, 2:] - field[:, 1:-1], field[:, 1:-1] - field[:, :-2]
        )
        return slope


@pytest.mark.parametrize("boundary", ["open", "reflective"])
def test_nonperiodic_outputs_match_legacy_reconstruction(boundary: str) -> None:
    common = dict(
        nx=12,
        ny=5,
        dx=1.0 / 12,
        dy=0.2,
        dt=1.0e-4,
        boundary=boundary,
        use_sponge=False,
    )
    observed = MUSCLHRShallowWaterSolver(**common)
    legacy = _LegacyNonPeriodicMUSCL(**common)
    bathymetry, state = _smooth_state(12, 5)
    for solver in (observed, legacy):
        solver.set_bathymetry(bathymetry)
        solver.set_state(state)
        solver.step(dt=1.0e-4)
    np.testing.assert_array_equal(observed.get_state(), legacy.get_state())


def test_invalid_reconstruction_limiter_fails_closed() -> None:
    with pytest.raises(ValueError, match="reconstruction_limiter"):
        _solver(limiter="not-a-limiter")
