from __future__ import annotations

import numpy as np
from .boundary_conditions import reflective_boundary, sponge_mask


def laplacian(u: np.ndarray) -> np.ndarray:
    return (
        np.roll(u, 1, axis=-2) + np.roll(u, -1, axis=-2) +
        np.roll(u, 1, axis=-1) + np.roll(u, -1, axis=-1) - 4 * u
    )


def toy_shallow_water_solver(source: np.ndarray, bathymetry: np.ndarray, steps: int = 30, dt: float = 0.15) -> np.ndarray:
    """Toy wave-like solver for framework testing.

    This is not a production tsunami solver. It produces smooth wave propagation
    targets for pipeline validation and unit tests.
    """
    eta_prev = source.astype(np.float32)
    eta = source.astype(np.float32)
    h, w = source.shape
    damp = sponge_mask(h, w, width=max(2, h // 16))
    depth_speed = np.clip(np.abs(bathymetry) / (np.abs(bathymetry).max() + 1e-6), 0.1, 1.0)
    c2 = 0.18 + 0.82 * depth_speed
    for _ in range(steps):
        nxt = 2 * eta - eta_prev + (dt ** 2) * c2 * laplacian(eta)
        nxt = reflective_boundary(nxt[None, ...])[0]
        nxt *= damp
        eta_prev, eta = eta, nxt.astype(np.float32)
    return eta.astype(np.float32)
