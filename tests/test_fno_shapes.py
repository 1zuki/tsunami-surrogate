import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tsunami_surrogate.models.fno2d import FNO2D


def test_fno_shape_same_resolution():
    model = FNO2D(in_channels=3, out_channels=1, modes1=6, modes2=6, width=12, depth=2)
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 16, 16)


def test_fno_shape_cross_resolution():
    model = FNO2D(in_channels=3, out_channels=1, modes1=6, modes2=6, width=12, depth=2)
    x = torch.randn(2, 3, 24, 24)
    y = model(x)
    assert y.shape == (2, 1, 24, 24)
