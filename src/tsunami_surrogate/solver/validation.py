from __future__ import annotations

import numpy as np


def assert_finite_field(field: np.ndarray, name: str = 'field') -> None:
    if not np.isfinite(field).all():
        raise ValueError(f'{name} contains NaN or Inf values')
