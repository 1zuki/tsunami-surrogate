from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetForecaster(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, depth: int = 4, dropout: float = 0.1, use_grid: bool = True):
        super().__init__()
        self.use_grid = use_grid
        input_dim = in_channels + (2 if use_grid else 0)
        self.depth = depth

        chs = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = input_dim
        for ch in chs:
            self.encoders.append(DoubleConv(prev, ch, dropout))
            self.pools.append(nn.MaxPool2d(2))
            prev = ch

        self.bottleneck = DoubleConv(chs[-1], chs[-1] * 2, dropout)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = chs[-1] * 2
        for ch in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(current, ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(ch * 2, ch, dropout))
            current = ch
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _get_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gx = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        gy = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat([x, self._get_grid(x)], dim=1)
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        return self.head(x)
