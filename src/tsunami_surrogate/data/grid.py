from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int
    dx: float
    dy: float
    domain_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)

    @classmethod
    def square(cls, resolution: int, bounds=(0.0, 1.0, 0.0, 1.0)) -> 'GridSpec':
        x0, x1, y0, y1 = bounds
        return cls(resolution, resolution, (x1 - x0) / (resolution - 1), (y1 - y0) / (resolution - 1), bounds)

    def mesh(self):
        x0, x1, y0, y1 = self.domain_bounds
        xs = np.linspace(x0, x1, self.nx, dtype=np.float32)
        ys = np.linspace(y0, y1, self.ny, dtype=np.float32)
        return np.meshgrid(xs, ys, indexing='xy')
