from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights_x = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )
        self.weights_y = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul1d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bimn,iom->bomn", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        height = x.size(-2)
        width = x.size(-1)

        x_ft = torch.fft.rfft(x, dim=-2)
        out_x_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height // 2 + 1,
            width,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes1, x_ft.size(-2))
        out_x_ft[:, :, :m1, :] = self.compl_mul1d(
            x_ft[:, :, :m1, :], self.weights_x[:, :, :m1]
        )
        out_x = torch.fft.irfft(out_x_ft, n=height, dim=-2)

        y_ft = torch.fft.rfft(x, dim=-1)
        out_y_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m2 = min(self.modes2, y_ft.size(-1))
        out_y_ft[:, :, :, :m2] = self.compl_mul1d(
            y_ft[:, :, :, :m2].transpose(-1, -2), self.weights_y[:, :, :m2]
        ).transpose(-1, -2)
        out_y = torch.fft.irfft(out_y_ft, n=width, dim=-1)

        return out_x + out_y


class FFNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = FactorizedSpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FFNO2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        depth: int = 4,
        padding: int = 6,
        use_grid: bool = True,
    ):
        super().__init__()
        self.use_grid = use_grid
        self.padding = padding

        lifted_channels = in_channels + (2 if use_grid else 0)

        self.lift = nn.Conv2d(lifted_channels, width, 1)
        self.blocks = nn.ModuleList(
            [FFNOBlock(width, modes1, modes2) for _ in range(depth)]
        )
        self.proj1 = nn.Conv2d(width, width * 2, 1)
        self.proj2 = nn.Conv2d(width * 2, out_channels, 1)

    def get_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        grid_y = (
            torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)
            .view(1, 1, h, 1)
            .repeat(b, 1, 1, w)
        )
        grid_x = (
            torch.linspace(0, 1, w, device=x.device, dtype=x.dtype)
            .view(1, 1, 1, w)
            .repeat(b, 1, h, 1)
        )

        return torch.cat([grid_x, grid_y], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat([x, self.get_grid(x)], dim=1)

        x = self.lift(x)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = F.gelu(self.proj1(x))

        return self.proj2(x)
