from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)


def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    enable_mc_dropout(model)
    preds = []
    for _ in range(n_samples):
        preds.append(model(x))
    stacked = torch.stack(preds, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0), stacked


@torch.no_grad()
def ensemble_predict(models: Iterable[nn.Module], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = [model(x) for model in models]
    stacked = torch.stack(preds, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0), stacked


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = std.pow(2).clamp_min(eps)
    return 0.5 * (torch.log(var) + (target - mean).pow(2) / var).mean()


def interval_coverage(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, z_value: float) -> torch.Tensor:
    lower = mean - z_value * std
    upper = mean + z_value * std
    inside = ((target >= lower) & (target <= upper)).float()
    return inside.mean()
