from __future__ import annotations

import numpy as np


def standardize_channels(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    std = x.std(axis=(0, 2, 3), keepdims=True) + eps
    return (x - mean) / std, {'mean': mean.squeeze().tolist(), 'std': std.squeeze().tolist()}
