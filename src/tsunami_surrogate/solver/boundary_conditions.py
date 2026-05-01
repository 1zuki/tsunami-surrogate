from __future__ import annotations

import numpy as np


def reflective_boundary(field: np.ndarray) -> np.ndarray:
    out = field.copy()
    out[..., 0, :] = out[..., 1, :]
    out[..., -1, :] = out[..., -2, :]
    out[..., :, 0] = out[..., :, 1]
    out[..., :, -1] = out[..., :, -2]
    return out


def sponge_mask(h: int, w: int, width: int = 4) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(width):
        val = (i + 1) / (width + 1)
        mask[i, :] *= val
        mask[-i-1, :] *= val
        mask[:, i] *= val
        mask[:, -i-1] *= val
    return mask
