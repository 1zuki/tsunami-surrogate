from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
