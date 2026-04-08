from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def spatial_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def temporal_gradient(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:, ...] - x[:, :-1, ...]


class CompositeLoss(nn.Module):
    def __init__(self, config: Dict[str, float]):
        super().__init__()
        self.mse_weight = float(config.get("mse_weight", 1.0))
        self.l1_weight = float(config.get("l1_weight", 0.0))
        self.gradient_weight = float(config.get("gradient_weight", 0.0))
        self.temporal_weight = float(config.get("temporal_weight", 0.0))
        self.mass_weight = float(config.get("mass_weight", 0.0))
        self.spectral_weight = float(config.get("spectral_weight", 0.0))
        self.epsilon = float(config.get("epsilon", 1e-6))
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        loss = pred.new_tensor(0.0)
        log: Dict[str, float] = {}

        if self.mse_weight > 0:
            mse = self.mse(pred, target)
            loss = loss + self.mse_weight * mse
            log["mse"] = float(mse.detach())

        if self.l1_weight > 0:
            l1 = self.l1(pred, target)
            loss = loss + self.l1_weight * l1
            log["l1"] = float(l1.detach())

        if self.gradient_weight > 0:
            pdx, pdy = spatial_gradient(pred)
            tdx, tdy = spatial_gradient(target)
            grad = self.l1(pdx, tdx) + self.l1(pdy, tdy)
            loss = loss + self.gradient_weight * grad
            log["gradient"] = float(grad.detach())

        if self.temporal_weight > 0 and pred.shape[1] > 1:
            temp = self.l1(temporal_gradient(pred), temporal_gradient(target))
            loss = loss + self.temporal_weight * temp
            log["temporal"] = float(temp.detach())

        if self.mass_weight > 0:
            p_mass = pred.sum(dim=(-1, -2))
            t_mass = target.sum(dim=(-1, -2))
            mass = self.l1(p_mass, t_mass)
            loss = loss + self.mass_weight * mass
            log["mass"] = float(mass.detach())

        if self.spectral_weight > 0:
            p_ft = torch.fft.rfft2(pred, norm="ortho")
            t_ft = torch.fft.rfft2(target, norm="ortho")
            spec = (p_ft.abs() - t_ft.abs()).abs().mean()
            loss = loss + self.spectral_weight * spec
            log["spectral"] = float(spec.detach())

        log["total"] = float(loss.detach())
        return loss, log
