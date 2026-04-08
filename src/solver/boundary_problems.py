from __future__ import annotations

import numpy as np


def pad_state(U: np.ndarray, boundary: str = "transmissive") -> np.ndarray:
    """Pad a state tensor U with one ghost cell on each side.

    U must have shape [3, H, W] containing [h, hu, hv].
    """
    if U.ndim != 3 or U.shape[0] != 3:
        raise ValueError(f"Expected U with shape [3, H, W], got {U.shape}")

    boundary = boundary.lower()
    if boundary not in {"transmissive", "open", "reflective", "periodic"}:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    if boundary in {"transmissive", "open"}:
        return np.pad(U, ((0, 0), (1, 1), (1, 1)), mode="edge")
    if boundary == "periodic":
        return np.pad(U, ((0, 0), (1, 1), (1, 1)), mode="wrap")

    padded = np.pad(U, ((0, 0), (1, 1), (1, 1)), mode="edge")
    # Left/right reflect hu.
    padded[1, 1:-1, 0] = -padded[1, 1:-1, 1]
    padded[1, 1:-1, -1] = -padded[1, 1:-1, -2]
    padded[0, 1:-1, 0] = padded[0, 1:-1, 1]
    padded[0, 1:-1, -1] = padded[0, 1:-1, -2]
    padded[2, 1:-1, 0] = padded[2, 1:-1, 1]
    padded[2, 1:-1, -1] = padded[2, 1:-1, -2]

    # Bottom/top reflect hv.
    padded[2, 0, 1:-1] = -padded[2, 1, 1:-1]
    padded[2, -1, 1:-1] = -padded[2, -2, 1:-1]
    padded[0, 0, 1:-1] = padded[0, 1, 1:-1]
    padded[0, -1, 1:-1] = padded[0, -2, 1:-1]
    padded[1, 0, 1:-1] = padded[1, 1, 1:-1]
    padded[1, -1, 1:-1] = padded[1, -2, 1:-1]
    return padded


def sponge_mask(ny: int, nx: int, width: int = 4, strength: float = 0.12) -> np.ndarray:
    """Create a multiplicative damping mask near the domain boundary."""
    mask = np.ones((ny, nx), dtype=np.float32)
    if width <= 0:
        return mask
    for i in range(width):
        factor = 1.0 - strength * (1.0 - i / max(width - 1, 1))
        mask[i, :] *= factor
        mask[-1 - i, :] *= factor
        mask[:, i] *= factor
        mask[:, -1 - i] *= factor
    return np.clip(mask, 0.0, 1.0)
