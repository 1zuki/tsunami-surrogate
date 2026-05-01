from __future__ import annotations

import torch


def prediction_interval(mean: torch.Tensor, variance: torch.Tensor, z: float = 1.96):
    std = torch.sqrt(variance.clamp_min(1e-8))
    return mean - z * std, mean + z * std


def coverage(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor, z: float = 1.96) -> float:
    lo, hi = prediction_interval(mean, variance, z)
    inside = (target >= lo) & (target <= hi)
    return float(inside.float().mean().detach().cpu())


def error_uncertainty_correlation(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> float:
    error = torch.abs(mean - target).flatten()
    unc = variance.flatten()
    if error.numel() < 2:
        return 0.0
    error = error - error.mean()
    unc = unc - unc.mean()
    corr = (error * unc).mean() / (error.std().clamp_min(1e-8) * unc.std().clamp_min(1e-8))
    return float(corr.detach().cpu())
