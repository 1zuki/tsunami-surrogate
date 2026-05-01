from __future__ import annotations

from typing import Iterable, Dict
import torch
from .uncertainty import coverage

_Z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96}


def interval_calibration(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, levels: Iterable[float]) -> Dict[str, float]:
    out = {}
    for level in levels:
        z = _Z.get(float(level), 1.96)
        cov = coverage(mean, variance, target, z=z)
        out[f'coverage_{level}'] = cov
        out[f'calibration_error_{level}'] = abs(cov - float(level))
    return out
