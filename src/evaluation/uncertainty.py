from __future__ import annotations

import torch


def error_uncertainty_correlation(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> float:
    std = torch.sqrt(torch.clamp(variance, min=1e-12)).reshape(-1).float()
    err = torch.abs(target - mean).reshape(-1).float()
    if std.numel() < 2:
        return 0.0
    std = std - std.mean()
    err = err - err.mean()
    denom = torch.sqrt((std * std).sum()) * torch.sqrt((err * err).sum())
    if float(denom) < 1e-12:
        return 0.0
    corr = (std * err).sum() / denom
    return float(corr.item())

