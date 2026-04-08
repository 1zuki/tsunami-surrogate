from __future__ import annotations

import torch
import torch.nn as nn


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CNNForecaster(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64, n_blocks: int = 8, dropout: float = 0.1, use_grid: bool = True):
        super().__init__()
        self.use_grid = use_grid
        input_dim = in_channels + (2 if use_grid else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8] * ((n_blocks + 3) // 4)
        self.blocks = nn.Sequential(*[ResidualDilatedBlock(hidden_channels, d, dropout) for d in dilations[:n_blocks]])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def _get_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gx = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        gy = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat([x, self._get_grid(x)], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
