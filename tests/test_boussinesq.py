from __future__ import annotations

import numpy as np

from src.solver.boussinesq import BoussinesqSolver, simulate_rollout


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


def test_boussinesq_zero_source_rest_stays_zero() -> None:
    nx, ny = 10, 12
    rng = np.random.default_rng(2)
    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=0.1,
        dy=0.1,
        dt=1e-3,
        boundary="periodic",
    )

    bathymetry = -rng.uniform(0.5, 2.0, size=(nx, ny))
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(np.zeros((nx, ny)), eta_t0=np.zeros((nx, ny)))
    solver.run(8, return_history=False)

    state = solver.get_state()
    assert np.isfinite(state).all()
    assert np.allclose(state[0], 0.0)
    assert np.allclose(state[1], 0.0)


def test_boussinesq_alpha_zero_matches_face_flux_wave_operator() -> None:
    nx, ny = 16, 12
    dx = 1.0 / nx
    dy = 1.0 / ny
    depth = 1.7
    g = 9.81
    rng = np.random.default_rng(3)
    eta = rng.standard_normal((nx, ny)) * 1e-3

    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=1e-4,
        g=g,
        alpha=0.0,
        boundary="periodic",
        use_sponge=False,
        mode="linear_constant_depth",
    )
    solver.set_bathymetry(-depth * np.ones((nx, ny), dtype=float))

    expected_laplacian = (
        (np.roll(eta, -1, axis=0) - 2.0 * eta + np.roll(eta, 1, axis=0)) / (dx * dx)
        + (np.roll(eta, -1, axis=1) - 2.0 * eta + np.roll(eta, 1, axis=1)) / (dy * dy)
    )
    expected = g * depth * expected_laplacian

    assert np.allclose(solver.rhs(eta), expected)
    assert np.allclose(solver.solve_acceleration(eta), expected)


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


def test_boussinesq_flat_bottom_gaussian_remains_stable_without_checkerboard() -> None:
    nx, ny = 32, 32
    dx = 1.0 / nx
    dy = 1.0 / ny
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta0 = 0.01 * np.exp(-80.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=5e-4,
        boundary="periodic",
        mode="linear_constant_depth",
        use_sponge=False,
    )

    solver.set_bathymetry(-np.ones((nx, ny), dtype=float))
    solver.set_initial_condition(eta0)
    solver.run(40, return_history=False)

    state = solver.get_state()
    checker = (-1.0) ** np.indices((nx, ny)).sum(axis=0)
    checker_fraction = abs(float(np.sum(state[0] * checker))) / max(float(np.sum(np.abs(state[0]))), 1e-30)

    assert np.isfinite(state).all()
    assert float(np.max(np.abs(state[0]))) < 0.05
    assert checker_fraction < 0.05


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


def test_boussinesq_fourier_mode_dispersion_constant_depth_periodic() -> None:
    nx, ny = 64, 16
    dx = 1.0 / nx
    dy = 1.0 / ny
    depth = 1.25
    alpha = 1.0 / 3.0
    g = 9.81
    mode = 2
    x = np.arange(nx) * dx
    eta = 1e-3 * np.cos(2.0 * np.pi * mode * x)[:, None] * np.ones((1, ny))

    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=1e-4,
        g=g,
        alpha=alpha,
        boundary="periodic",
        use_sponge=False,
        mode="linear_constant_depth",
        linear_solver_tol=1e-12,
        linear_solver_max_iter=200,
    )
    solver.set_bathymetry(-depth * np.ones((nx, ny), dtype=float))

    acceleration = solver.solve_acceleration(eta)
    omega2_measured = -float(np.sum(acceleration * eta) / np.sum(eta * eta))

    k = 2.0 * np.pi * mode
    omega2_continuous = g * depth * k * k / (1.0 + alpha * depth * depth * k * k)
    k2_discrete = (2.0 * np.sin(0.5 * k * dx) / dx) ** 2
    omega2_discrete = g * depth * k2_discrete / (1.0 + alpha * depth * depth * k2_discrete)

    assert np.isclose(omega2_measured, omega2_continuous, rtol=1e-2)
    assert np.isclose(omega2_measured, omega2_discrete, rtol=1e-8)
    assert solver.last_cg_iterations > 0
    assert solver.last_cg_converged
    assert solver.last_cg_final_residual <= solver.linear_solver_tol * solver.last_cg_initial_residual * 1.01



def test_boussinesq_simulate_rollout_converts_depth_to_eta() -> None:
    nx, ny = 12, 10
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta0 = 0.02 * np.exp(-20.0 * (X * X + Y * Y))
    bathymetry = -np.ones((nx, ny), dtype=float)
    initial_depth = -bathymetry + eta0
    source = np.zeros((nx, ny), dtype=float)

    sample = np.stack([bathymetry, source, initial_depth], axis=0)
    rollout = simulate_rollout(
        sample,
        n_steps=2,
        record_every=1,
        include_initial_state=True,
        boundary="periodic",
        use_sponge=False,
        alpha=0.0,
        auto_dt=False,
        dt=1e-4,
        channel_map={"bathymetry": 0, "source": None, "initial_depth": 2, "initial_surface": None},
    )

    assert rollout.shape == (3, nx, ny)
    assert np.isfinite(rollout).all()
    assert np.allclose(rollout[0], eta0.astype(np.float32))


def test_boussinesq_simulate_rollout_uses_source_as_default_eta0() -> None:
    nx, ny = 24, 24
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    bathymetry = -np.ones((nx, ny), dtype=np.float32)
    source = np.exp(-((X - 0.4) ** 2 + (Y - 0.6) ** 2) / (2 * 0.08 ** 2)).astype(np.float32)
    initial_depth = np.ones((nx, ny), dtype=np.float32)
    sample = np.stack([bathymetry, source, initial_depth], axis=0)

    frames = simulate_rollout(
        sample,
        n_steps=8,
        record_every=1,
        include_initial_state=True,
        output_field="eta",
        source_scale=0.6,
        alpha=0.0,
        boundary="periodic",
        use_sponge=False,
        filter_strength=0.0,
        auto_dt=True,
        target_cfl=0.35,
    )

    assert frames.shape[0] >= 2
    assert np.isfinite(frames).all()
    assert np.allclose(frames[0], 0.6 * source, atol=1e-6)
    assert not np.allclose(frames[1], frames[0])


def test_boussinesq_simulate_rollout_depth_fallback_when_source_missing() -> None:
    nx, ny = 18, 18
    bathymetry = -np.full((nx, ny), 1.25, dtype=np.float32)
    initial_depth = np.full((nx, ny), 1.4, dtype=np.float32)
    sample = np.stack([bathymetry, np.zeros_like(bathymetry), initial_depth], axis=0)

    frames = simulate_rollout(
        sample,
        n_steps=1,
        record_every=1,
        include_initial_state=True,
        output_field="eta",
        source_scale=1.0,
        boundary="open",
        use_sponge=False,
        filter_strength=0.0,
        channel_map={"bathymetry": 0, "source": None, "initial_depth": 2, "initial_surface": None},
    )

    expected_eta0 = initial_depth + bathymetry
    assert np.isfinite(frames).all()
    assert np.allclose(frames[0], expected_eta0, atol=1e-6)


def test_boussinesq_simulate_rollout_sea_level_conversion_and_depth_output() -> None:
    nx, ny = 12, 10
    bathymetry = -np.full((nx, ny), 1.2, dtype=np.float32)
    initial_depth = np.full((nx, ny), 1.5, dtype=np.float32)
    sample = np.stack([bathymetry, np.zeros_like(bathymetry), initial_depth], axis=0)

    sea_level_offset = 0.2
    frames = simulate_rollout(
        sample,
        n_steps=1,
        record_every=1,
        include_initial_state=True,
        output_field="depth",
        sea_level_offset=sea_level_offset,
        boundary="periodic",
        use_sponge=False,
        filter_strength=0.0,
        alpha=0.0,
        channel_map={"bathymetry": 0, "source": None, "initial_depth": 2, "initial_surface": None},
    )

    # eta0 = h0 + b - sea_level_offset => 1.5 - 1.2 - 0.2 = 0.1
    # H   = -b + sea_level_offset      => 1.2 + 0.2 = 1.4
    # depth = H + eta = 1.5
    assert np.isfinite(frames).all()
    assert np.allclose(frames[0], initial_depth, atol=1e-6)


def test_boussinesq_simulate_rollout_eta_overrides_and_default_bathymetry() -> None:
    nx, ny = 14, 14
    source = np.ones((nx, ny), dtype=np.float32) * 0.25
    # no bathymetry channel: source only
    sample = source[None, ...]
    eta0 = np.full((nx, ny), 0.03, dtype=np.float32)
    eta_t0 = np.full((nx, ny), -0.01, dtype=np.float32)

    frames = simulate_rollout(
        sample,
        n_steps=0,
        include_initial_state=True,
        output_field="state",
        default_depth=2.0,
        eta0=eta0,
        eta_t0=eta_t0,
        channel_map={"bathymetry": None, "source": 0, "initial_depth": None, "initial_surface": None},
        use_sponge=False,
        filter_strength=0.0,
    )

    assert frames.shape == (1, 2, nx, ny)
    assert np.isfinite(frames).all()
    assert np.allclose(frames[0, 0], eta0, atol=1e-6)
    assert np.allclose(frames[0, 1], eta_t0, atol=1e-6)
