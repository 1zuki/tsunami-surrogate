from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64, dropout: float = 0.1, use_grid: bool = True):
        super().__init__()
        self.use_grid = use_grid
        self.out_channels = out_channels
        input_dim = in_channels + (2 if use_grid else 0)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.cell = ConvLSTMCell(hidden_channels, hidden_channels)
        self.feedback = nn.Conv2d(1, hidden_channels, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def _get_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gx = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        gy = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            x = torch.cat([x, self._get_grid(x)], dim=1)
        base = self.encoder(x)
        b, ch, h, w = base.shape
        h_state = torch.zeros((b, ch, h, w), device=x.device, dtype=x.dtype)
        c_state = torch.zeros_like(h_state)
        feedback = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
        outputs = []
        for _ in range(self.out_channels):
            inp = base + self.feedback(feedback)
            h_state, c_state = self.cell(inp, h_state, c_state)
            frame = self.head(h_state)
            outputs.append(frame)
            feedback = frame
        return torch.cat(outputs, dim=1)
