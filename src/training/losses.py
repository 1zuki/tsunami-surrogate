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


def horizon_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_weight: float = 0.8,
    max_weight: float = 1.6,
    power: float = 1.5,
) -> torch.Tensor:
    # Expected common shape: [B, T, H, W]. Fallback to plain MSE for non-temporal targets.
    if pred.dim() < 4 or pred.shape[1] <= 1:
        return F.mse_loss(pred, target)

    t = pred.shape[1]
    w = torch.linspace(0.0, 1.0, t, device=pred.device, dtype=pred.dtype)
    w = min_weight + (max_weight - min_weight) * (w ** power)

    # error^2 reduced over all dimensions except timestep, then weighted over time.
    sq = (pred - target) ** 2
    reduce_dims = [0] + list(range(2, sq.dim()))
    per_t = sq.mean(dim=reduce_dims)
    loss = (per_t * w).sum() / w.sum().clamp_min(1e-8)
    return loss


def build_loss(name: str, train_cfg: dict | None = None):
    train_cfg = train_cfg or {}
    if name == 'mse':
        return lambda pred, target, batch=None: F.mse_loss(pred, target)
    if name == 'relative_l2':
        return lambda pred, target, batch=None: relative_l2(pred, target)
    if name == 'spectral':
        return lambda pred, target, batch=None: spectral_loss(pred, target)
    if name == 'coastal_weighted_mse':
        return lambda pred, target, batch=None: coastal_weighted_mse(pred, target, batch['x'][:, 2:3] if batch is not None else None)
    if name == 'horizon_weighted_mse':
        min_w = float(train_cfg.get("horizon_min_weight", 0.8))
        max_w = float(train_cfg.get("horizon_max_weight", 1.6))
        power = float(train_cfg.get("horizon_power", 1.5))
        return lambda pred, target, batch=None: horizon_weighted_mse(
            pred,
            target,
            min_weight=min_w,
            max_weight=max_w,
            power=power,
        )

    raise ValueError(f'Unknown loss: {name}')
