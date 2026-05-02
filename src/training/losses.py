from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + eps)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_ft = torch.fft.rfft2(pred)
    target_ft = torch.fft.rfft2(target)
    return torch.mean(torch.abs(pred_ft - target_ft) ** 2)


def coastal_weighted_mse(pred: torch.Tensor, target: torch.Tensor, coastal_mask: torch.Tensor | None = None, weight: float = 4.0) -> torch.Tensor:
    if coastal_mask is None:
        return F.mse_loss(pred, target)
    while coastal_mask.dim() < pred.dim():
        coastal_mask = coastal_mask.unsqueeze(1)
    weights = 1.0 + (weight - 1.0) * coastal_mask.to(pred.device)
    return torch.mean(weights * (pred - target) ** 2)


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(log_var + (target - mean) ** 2 / torch.exp(log_var).clamp_min(1e-8))


def build_loss(name: str):
    if name == 'mse':
        return lambda pred, target, batch=None: F.mse_loss(pred, target)
    if name == 'relative_l2':
        return lambda pred, target, batch=None: relative_l2(pred, target)
    if name == 'spectral':
        return lambda pred, target, batch=None: spectral_loss(pred, target)
    if name == 'coastal_weighted_mse':
        return lambda pred, target, batch=None: coastal_weighted_mse(pred, target, batch['x'][:, 2:3] if batch is not None else None)
    raise ValueError(f'Unknown loss: {name}')
