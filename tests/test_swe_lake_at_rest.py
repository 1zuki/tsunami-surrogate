from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver
from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver


def _lake_at_rest_case(solver_cls: type[HydrostaticShallowWaterSolver]) -> tuple[float, float, float]:
    nx = 40
    ny = 40
    dx = 1.0 / nx
    dy = 1.0 / ny

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # fully submerged bathymetry to avoid wet/dry edge effects in this balance test
    b = -1.0 - 0.2 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.03)
    eta0 = 0.0
    h0 = eta0 - b

    solver = solver_cls(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=1e-3,
        boundary="reflective",
        use_sponge=False,
        cfl=0.25,
    )
    solver.set_bathymetry(b)
    solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

    eta_before = solver.compute_free_surface().copy()
    for _ in range(12):
        dt = solver.suggest_dt(target_cfl=0.2)
        solver.step(dt=dt, auto_dt=False)

    eta_after = solver.compute_free_surface()
    eta_drift = float(np.max(np.abs(eta_after - eta_before)))
    hu_max = float(np.max(np.abs(solver.hu)))
    hv_max = float(np.max(np.abs(solver.hv)))
    return eta_drift, hu_max, hv_max


def test_hydrostatic_lake_at_rest_is_balanced() -> None:
    eta_drift, hu_max, hv_max = _lake_at_rest_case(HydrostaticShallowWaterSolver)
    assert eta_drift < 1e-8
    assert hu_max < 1e-7
    assert hv_max < 1e-7


def test_muscl_hr_lake_at_rest_is_balanced() -> None:
    eta_drift, hu_max, hv_max = _lake_at_rest_case(MUSCLHRShallowWaterSolver)
    assert eta_drift < 1e-8
    assert hu_max < 1e-7
    assert hv_max < 1e-7


def _build_bathymetry(nx: int, ny: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if kind == "flat":
        b = -np.ones((nx, ny), dtype=float)
    elif kind == "slope":
        # keep fully wet while adding a directional slope.
        b = -(1.0 + 0.25 * (X - 0.5) + 0.10 * (Y - 0.5))
    elif kind == "multi_gaussian":
        b = (
            -1.0
            - 0.18 * np.exp(-((X - 0.33) ** 2 + (Y - 0.65) ** 2) / 0.015)
            - 0.12 * np.exp(-((X - 0.72) ** 2 + (Y - 0.28) ** 2) / 0.02)
        )
    else:
        raise ValueError(f"unknown bathymetry kind: {kind}")

    return b, X, Y


def _run_muscl_dynamic_case(
    nx: int,
    ny: int,
    n_steps: int,
    target_cfl: float,
    bathy_kind: str,
    boundary: str,
) -> tuple[float, float]:
    dx = 1.0 / nx
    dy = 1.0 / ny
    bathymetry, X, Y = _build_bathymetry(nx, ny, bathy_kind)

    eta0 = 0.02 * np.exp(-((X - 0.35) ** 2 + (Y - 0.58) ** 2) / (2.0 * 0.06 ** 2))
    h0 = np.maximum(-bathymetry + eta0, 0.0)

    solver = MUSCLHRShallowWaterSolver(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=5e-4,
        cfl=0.45,
        boundary=boundary,
        use_sponge=False,
        max_velocity=30.0,
    )
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

    min_h_seen = float(np.min(h0))
    max_speed_seen = 0.0

    for _ in range(n_steps):
        dt = solver.suggest_dt(target_cfl=target_cfl)
        assert np.isfinite(dt)
        assert dt > 0.0

        solver.step(dt=dt, auto_dt=False)
        state = solver.get_state()
        assert np.isfinite(state).all()

        h = state[0]
        min_h_seen = min(min_h_seen, float(np.min(h)))
        assert min_h_seen >= -1e-8

        u, v = solver.compute_velocity()
        speed = np.hypot(u, v)
        step_max_speed = float(np.max(speed))
        max_speed_seen = max(max_speed_seen, step_max_speed)
        assert step_max_speed <= solver.max_velocity * 1.05 + 1e-8

    eta = solver.compute_free_surface()
    dynamic_signal = float(np.max(np.abs(eta - np.mean(eta))))
    assert dynamic_signal > 1e-6

    return min_h_seen, max_speed_seen


@pytest.mark.parametrize(
    "nx,ny,n_steps,target_cfl,bathy_kind,boundary",
    [
        (32, 32, 45, 0.35, "flat", "periodic"),
        (64, 64, 35, 0.40, "slope", "open"),
        (64, 64, 35, 0.45, "multi_gaussian", "periodic"),
    ],
)
@pytest.mark.slow
def test_muscl_hr_dynamic_pulse_finite_state(
    nx: int,
    ny: int,
    n_steps: int,
    target_cfl: float,
    bathy_kind: str,
    boundary: str,
) -> None:
    min_h_seen, max_speed_seen = _run_muscl_dynamic_case(
        nx=nx,
        ny=ny,
        n_steps=n_steps,
        target_cfl=target_cfl,
        bathy_kind=bathy_kind,
        boundary=boundary,
    )
    assert min_h_seen >= -1e-8
    assert max_speed_seen <= 30.0 * 1.05 + 1e-8


RUN_LONG_MUSCL_TESTS = os.environ.get("TSUNAMI_LONG_TESTS", "0") == "1"


@pytest.mark.skipif(
    not RUN_LONG_MUSCL_TESTS,
    reason="set TSUNAMI_LONG_TESTS=1 to run long MUSCL-HR stress variants",
)
@pytest.mark.parametrize(
    "nx,ny,n_steps,target_cfl,bathy_kind,boundary",
    [
        (32, 32, 120, 0.35, "flat", "periodic"),
        (64, 64, 120, 0.40, "slope", "open"),
        (128, 128, 100, 0.45, "multi_gaussian", "periodic"),
    ],
)
@pytest.mark.slow
def test_muscl_hr_dynamic_pulse_finite_state_long(
    nx: int,
    ny: int,
    n_steps: int,
    target_cfl: float,
    bathy_kind: str,
    boundary: str,
) -> None:
    min_h_seen, max_speed_seen = _run_muscl_dynamic_case(
        nx=nx,
        ny=ny,
        n_steps=n_steps,
        target_cfl=target_cfl,
        bathy_kind=bathy_kind,
        boundary=boundary,
    )
    assert min_h_seen >= -1e-8
    assert max_speed_seen <= 30.0 * 1.05 + 1e-8
