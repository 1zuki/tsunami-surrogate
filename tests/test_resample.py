import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tsunami_surrogate.utils.resample import resize_field, project_coarse_to_fine, restrict_fine_to_coarse


def test_resize():
    x = torch.randn(2, 3, 16, 16)
    y = resize_field(x, (32, 32))
    assert y.shape == (2, 3, 32, 32)
    z = restrict_fine_to_coarse(y, factor=2)
    assert z.shape == x.shape
    f = project_coarse_to_fine(x, factor=2)
    assert f.shape == y.shape
