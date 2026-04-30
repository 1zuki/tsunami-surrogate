import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tsunami_surrogate.training.losses import relative_l2, spectral_loss, gaussian_nll


def test_losses_finite():
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    assert torch.isfinite(relative_l2(pred, target))
    assert torch.isfinite(spectral_loss(pred, target))
    assert torch.isfinite(gaussian_nll(pred, torch.zeros_like(pred), target))
