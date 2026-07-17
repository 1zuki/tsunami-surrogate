from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.solver.boundary_conditions import (
    pad_scalar_field,
    radiation_boundary_state_x,
)
from src.solver.boussinesq import BoussinesqSolver
from src.solver.hydrostatic_swe import ShallowWaterSolver
from src.solver.operator_time import build_sponge_mask


def test_left_radiation_state_preserves_outgoing_characteristic() -> None:
    depth = 1.7
    elevation = 2.0e-3
    wave_speed = np.sqrt(9.81 * depth)
    observed = radiation_boundary_state_x(
        depth + elevation,
        -wave_speed * elevation,
        0.0,
        -depth,
        side="left",
        g=9.81,
        dry_tolerance=1.0e-6,
    )
    np.testing.assert_allclose(
        observed, [depth + elevation, -wave_speed * elevation, 0.0, -depth]
    )


def test_left_radiation_state_removes_incoming_characteristic() -> None:
    depth = 0.8
    elevation = 3.0e-3
    wave_speed = np.sqrt(9.81 * depth)
    observed = radiation_boundary_state_x(
        depth + elevation,
        wave_speed * elevation,
        0.0,
        -depth,
        side="left",
        g=9.81,
        dry_tolerance=1.0e-6,
    )
    np.testing.assert_allclose(observed, [depth, 0.0, 0.0, -depth], atol=1.0e-17)


def test_radiation_state_uses_local_bathymetry_wave_speed() -> None:
    elevation = 1.0e-3
    shallow = radiation_boundary_state_x(
        0.5 + elevation,
        0.0,
        0.0,
        -0.5,
        side="left",
        g=9.81,
        dry_tolerance=1.0e-6,
    )
    deep = radiation_boundary_state_x(
        2.0 + elevation,
        0.0,
        0.0,
        -2.0,
        side="left",
        g=9.81,
        dry_tolerance=1.0e-6,
    )
    assert abs(deep[1] / shallow[1]) == pytest.approx(2.0)


def test_radiation_padding_fails_instead_of_degrading_to_open() -> None:
    with pytest.raises(ValueError, match="characteristic face states"):
        pad_scalar_field(np.ones((4, 4)), "radiation", "open")


def test_boussinesq_rejects_swe_radiation_mode() -> None:
    with pytest.raises(ValueError, match="only for SWE"):
        BoussinesqSolver(
            nx=8,
            ny=4,
            dx=0.125,
            dy=0.25,
            dt=1.0e-3,
            boundary=("radiation", "open"),
            use_sponge=False,
        )


def test_cosine_sponge_is_explicit_x_only_and_legacy_default_is_unchanged() -> None:
    quadratic = build_sponge_mask(
        nx=32,
        ny=4,
        width=6,
        min_factor=0.8,
        axes="x",
        profile="quadratic",
    )
    cosine = build_sponge_mask(
        nx=32,
        ny=4,
        width=6,
        min_factor=0.8,
        axes="x",
        profile="cosine",
    )
    assert np.all(quadratic[6:-6] == 1.0)
    assert np.all(cosine[6:-6] == 1.0)
    assert np.all(cosine[:, 0] == cosine[:, -1])
    assert cosine[0, 0] == pytest.approx(0.8)
    assert not np.array_equal(cosine, quadratic)

    default = ShallowWaterSolver(
        nx=32,
        ny=4,
        dx=1.0 / 32,
        dy=0.25,
        dt=1.0e-3,
        boundary="open",
        use_sponge=True,
        sponge_width=6,
        sponge_min_factor=0.8,
        sponge_axes="x",
    )
    np.testing.assert_array_equal(default.sponge_mask, quadratic)


def test_radiation_solver_remains_finite_for_small_outgoing_packet() -> None:
    solver = ShallowWaterSolver(
        nx=32,
        ny=4,
        dx=1.0 / 32,
        dy=0.25,
        dt=1.0e-4,
        boundary=("radiation", "open"),
        use_sponge=False,
    )
    x = np.arange(32)[:, None] / 32
    eta = 1.0e-5 * np.exp(-0.5 * ((x - 0.2) / 0.04) ** 2) * np.ones((1, 4))
    solver.set_bathymetry(-np.ones((32, 4)))
    solver.set_initial_condition(
        1.0 + eta,
        hu0=-np.sqrt(9.81) * eta,
        hv0=np.zeros_like(eta),
    )
    for _ in range(10):
        solver.step(dt=1.0e-4)
    assert np.isfinite(solver.get_state()).all()
