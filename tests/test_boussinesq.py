from __future__ import annotations

import numpy as np

from src.solver.boussinesq import BoussinesqSolver


def test_boussinesq_initial_state_shape() -> None:
    solver = BoussinesqSolver(
        nx=8,
        ny=6,
        dx=1.0,
        dy=1.0,
        dt=0.01,
        boundary="open",
        use_sponge=False,
    )

    bathymetry = -np.ones((8, 6), dtype=float)
    eta0 = np.zeros((8, 6), dtype=float)

    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(eta0)

    state = solver.get_state()
    assert state.shape == (2, 8, 6)
    assert np.allclose(state[0], eta0)
    assert np.allclose(state[1], 0.0)


def test_boussinesq_flat_surface_remains_flat_after_one_step() -> None:
    solver = BoussinesqSolver(
        nx=8,
        ny=8,
        dx=1.0,
        dy=1.0,
        dt=0.01,
        boundary="open",
        use_sponge=False,
    )

    bathymetry = -np.ones((8, 8), dtype=float)
    eta0 = np.zeros((8, 8), dtype=float)

    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(eta0)
    solver.step()

    state = solver.get_state()
    assert np.isfinite(state).all()
    assert np.allclose(state[0], 0.0)
    assert np.allclose(state[1], 0.0)


def test_boussinesq_one_step_with_gaussian_is_finite() -> None:
    nx, ny = 16, 16
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta0 = 0.01 * np.exp(-40.0 * (X * X + Y * Y))

    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=1.0 / nx,
        dy=1.0 / ny,
        dt=1e-4,
        boundary="open",
        use_sponge=False,
    )

    bathymetry = -np.ones((nx, ny), dtype=float)
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(eta0)
    solver.step()

    state = solver.get_state()
    assert state.shape == (2, nx, ny)
    assert np.isfinite(state).all()


def test_boussinesq_suggest_dt_is_positive() -> None:
    solver = BoussinesqSolver(
        nx=8,
        ny=8,
        dx=0.1,
        dy=0.1,
        dt=0.01,
        boundary="open",
        use_sponge=False,
    )
    solver.set_bathymetry(-np.ones((8, 8), dtype=float))

    dt = solver.suggest_dt(target_cfl=0.25)
    assert dt > 0.0
    assert np.isfinite(dt)
