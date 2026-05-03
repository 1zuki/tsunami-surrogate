from __future__ import annotations

from torch import nn
from .fno2d import FNO2D


class TemporalFNO2D(nn.Module):
    """simple temporal wrapper that predicts T output frames as channels
    for more advanced work, replace this with recurrent decoding or a 3D FNO
    """

    def __init__(self, in_channels: int, field_channels: int, time_steps: int, **kwargs):
        super().__init__()
        self.time_steps = time_steps
        self.field_channels = field_channels
        self.core = FNO2D(in_channels=in_channels, out_channels=field_channels * time_steps, **kwargs)

    def forward(self, x):
        y = self.core(x)
        b, _, h, w = y.shape
        return y.view(b, self.time_steps, self.field_channels, h, w)
