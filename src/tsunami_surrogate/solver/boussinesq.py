from __future__ import annotations

import numpy as np
from .shallow_water import toy_shallow_water_solver, laplacian


def toy_boussinesq_solver(source: np.ndarray, bathymetry: np.ndarray, steps: int = 30, dt: float = 0.15) -> np.ndarray:
    """Toy dispersive correction around the shallow-water test solver."""
    base = toy_shallow_water_solver(source, bathymetry, steps=steps, dt=dt)
    return (base - 0.03 * laplacian(base)).astype(np.float32)
