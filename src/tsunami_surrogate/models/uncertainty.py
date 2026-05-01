from __future__ import annotations

import torch
from torch import nn
from .fno2d import FNO2D


class ProbabilisticFNO2D(nn.Module):
    """FNO head that predicts mean and log-variance for heteroscedastic regression."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.core = FNO2D(in_channels=in_channels, out_channels=out_channels * 2, **kwargs)

    def forward(self, x: torch.Tensor):
        raw = self.core(x)
        mean, log_var = torch.split(raw, self.out_channels, dim=1)
        log_var = torch.clamp(log_var, -10.0, 6.0)
        return mean, log_var

    def predict(self, x: torch.Tensor):
        mean, log_var = self.forward(x)
        return {'mean': mean, 'variance': torch.exp(log_var), 'log_variance': log_var}
