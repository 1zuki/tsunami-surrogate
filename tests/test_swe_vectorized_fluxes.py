from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver


def _solver() -> HydrostaticShallowWaterSolver:
    return HydrostaticShallowWaterSolver(
        nx=7,
        ny=5,
        dx=1.0 / 7.0,
        dy=0.2,
        dt=1.0e-4,
        boundary="radiation",
        use_sponge=False,
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_vectorized_hydrostatic_fluxes_match_scalar_faces(axis: str) -> None:
    solver = _solver()
    rng = np.random.default_rng(20260726)
    shape = (6, 4)
    h_left = rng.uniform(0.0, 1.5, shape)
    h_right = rng.uniform(0.0, 1.5, shape)
    h_left[0, 0] = 0.0
    h_right[1, 0] = 0.5 * solver.dry_tolerance
    hu_left = rng.normal(0.0, 0.03, shape) * h_left
    hv_left = rng.normal(0.0, 0.03, shape) * h_left
    hu_right = rng.normal(0.0, 0.03, shape) * h_right
    hv_right = rng.normal(0.0, 0.03, shape) * h_right
    b_left = rng.uniform(-1.1, 0.1, shape)
    b_right = rng.uniform(-1.1, 0.1, shape)
    states = (
        h_left,
        hu_left,
        hv_left,
        b_left,
        h_right,
        hu_right,
        hv_right,
        b_right,
    )

    base, left_correction, right_correction = (
        solver._hydrostatic_interface_fluxes(*states, axis=axis)
    )
    scalar_face = solver._hydro_face_x if axis == "x" else solver._hydro_face_y
    correction_channel = 1 if axis == "x" else 2

    for index in np.ndindex(shape):
        values = tuple(float(field[index]) for field in states)
        expected_for_left_cell = scalar_face(
            *values, use_left_correction=True
        )
        expected_for_right_cell = scalar_face(
            *values, use_left_correction=False
        )
        observed_for_left_cell = base[(slice(None), *index)].copy()
        observed_for_right_cell = base[(slice(None), *index)].copy()
        observed_for_left_cell[correction_channel] += left_correction[index]
        observed_for_right_cell[correction_channel] += right_correction[index]
        np.testing.assert_array_equal(
            observed_for_left_cell, expected_for_left_cell
        )
        np.testing.assert_array_equal(
            observed_for_right_cell, expected_for_right_cell
        )
