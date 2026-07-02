from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, [0, pad_w, 0, pad_h])

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)

    return torch.cat([ll, lh, hl, hh], dim=1), (h, w)


def _haar_iwt2(coeffs: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    ll, lh, hl, hh = torch.chunk(coeffs, 4, dim=1)
    h2, w2 = ll.shape[-2:]
    out = torch.empty(
        ll.shape[0],
        ll.shape[1],
        h2 * 2,
        w2 * 2,
        dtype=ll.dtype,
        device=ll.device,
    )
    out[..., 0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
    out[..., 0::2, 1::2] = 0.5 * (ll - lh + hl - hh)
    out[..., 1::2, 0::2] = 0.5 * (ll + lh - hl - hh)
    out[..., 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)

    h, w = shape
    return out[..., :h, :w]


class HaarWaveletConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = int(kernel_size) // 2
        self.wavelet = nn.Conv2d(
            in_channels * 4,
            out_channels * 4,
            kernel_size=int(kernel_size),
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs, shape = _haar_dwt2(x)
        coeffs = self.wavelet(coeffs)

        return _haar_iwt2(coeffs, shape)


class WNOBlock(nn.Module):
    def __init__(self, width: int, wavelet_kernel_size: int = 3):
        super().__init__()
        self.wavelet = HaarWaveletConv2d(width, width, kernel_size=wavelet_kernel_size)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.wavelet(x) + self.pointwise(x))


class WNO2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        depth: int = 4,
        padding: int = 6,
        use_grid: bool = True,
        wavelet_kernel_size: int = 3,
        **_: int,
    ):
        super().__init__()
        self.use_grid = use_grid
        self.padding = padding

        lifted_channels = in_channels + (2 if use_grid else 0)

        self.lift = nn.Conv2d(lifted_channels, width, 1)
        self.blocks = nn.ModuleList(
            [WNOBlock(width, wavelet_kernel_size) for _ in range(depth)]
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
