import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from src.models.ensemble import EnsemblePredictor


class IdentityModel(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias
    def forward(self, x):
        return x[:, :1] + self.bias


def test_ensemble_aggregation():
    ens = EnsemblePredictor([IdentityModel(0.0), IdentityModel(2.0)])
    x = torch.zeros(4, 3, 8, 8)
    out = ens(x)
    assert out['mean'].shape == (4, 1, 8, 8)
    assert torch.allclose(out['mean'], torch.ones_like(out['mean']))
    assert torch.all(out['variance'] > 0)


def test_ensemble_requires_at_least_two_members():
    try:
        EnsemblePredictor([IdentityModel(0.0)])
        assert False, "expected ValueError for single-member ensemble"
    except ValueError as e:
        assert "at least 2 members" in str(e)
