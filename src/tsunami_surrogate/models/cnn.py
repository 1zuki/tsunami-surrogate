from __future__ import annotations

from torch import nn


class CNNBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)
