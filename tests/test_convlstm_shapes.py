import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.convlstm import ConvLSTMBaseline


def test_convlstm_shape_same_resolution():
    model = ConvLSTMBaseline(
        in_channels=3,
        out_channels=10,
        hidden_channels=12,
        num_layers=2,
        kernel_size=3,
    )
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    assert y.shape == (2, 10, 16, 16)


def test_convlstm_shape_cross_resolution():
    model = ConvLSTMBaseline(
        in_channels=3,
        out_channels=12,
        hidden_channels=10,
        num_layers=1,
        kernel_size=3,
    )
    x = torch.randn(2, 3, 24, 20)
    y = model(x)
    assert y.shape == (2, 12, 24, 20)
