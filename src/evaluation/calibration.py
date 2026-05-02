from __future__ import annotations

import math
from typing import Dict, Iterable

import torch


def interval_calibration(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    levels: Iterable[float],
) -> Dict[str, float]:
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    err = target - mean
    out: Dict[str, float] = {}
    for level in levels:
        p = float(level)
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        z = math.sqrt(2.0) * torch.erfinv(torch.tensor(p, dtype=mean.dtype, device=mean.device)).item()
        inside = ((err >= -z * std) & (err <= z * std)).float().mean().item()
        out[f"coverage_{int(round(p * 100))}"] = float(inside)
    nll = 0.5 * (torch.log(torch.clamp(variance, min=1e-12)) + (err * err) / torch.clamp(variance, min=1e-12))
    out["nll"] = float(nll.mean().item())
    return out

