from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, size_x, size_y = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            size_x,
            size_y // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mx = min(self.modes_x, size_x)
        my = min(self.modes_y, size_y // 2 + 1)
        if mx > 0 and my > 0:
            out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
            out_ft[:, :, -mx:, :my] = self.compl_mul2d(x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my])
        x = torch.fft.irfft2(out_ft, s=(size_x, size_y), norm="ortho")
        return x


class FNOBlock(nn.Module):
    def __init__(self, channels: int, modes_x: int, modes_y: int, dropout: float = 0.1):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return y


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        modes_x: int = 12,
        modes_y: int = 12,
        n_layers: int = 4,
        padding: int = 4,
        dropout: float = 0.1,
        use_grid: bool = True,
    ):
        super().__init__()
        self.use_grid = use_grid
        self.padding = padding
        input_dim = in_channels + (2 if use_grid else 0)
        self.lift = nn.Conv2d(input_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(hidden_channels, modes_x, modes_y, dropout) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def _get_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        grid_x = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        grid_y = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([grid_x, grid_y], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat([x, self._get_grid(x)], dim=1)
        x = self.lift(x)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        x = self.project(x)
        return x
