from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, width=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, width)
        self.enc2 = DoubleConv(width, width * 2)
        self.dec1 = DoubleConv(width * 3, width)
        self.out = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        up = F.interpolate(e2, size=e1.shape[-2:], mode='bilinear', align_corners=False)

        return self.out(self.dec1(torch.cat([e1, up], dim=1)))
