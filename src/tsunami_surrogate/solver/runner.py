from __future__ import annotations

from typing import Literal
import numpy as np
from .shallow_water import toy_shallow_water_solver
from .boussinesq import toy_boussinesq_solver


def run_solver(source: np.ndarray, bathymetry: np.ndarray, solver: Literal['shallow_water', 'boussinesq'] = 'shallow_water', **kwargs) -> np.ndarray:
    if solver == 'shallow_water':
        return toy_shallow_water_solver(source, bathymetry, **kwargs)
    if solver == 'boussinesq':
        return toy_boussinesq_solver(source, bathymetry, **kwargs)
    raise ValueError(f'Unknown solver: {solver}')
