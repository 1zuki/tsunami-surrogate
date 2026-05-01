from __future__ import annotations

import numpy as np


def generate_bathymetry(resolution: int, rng: np.random.Generator, bathymetry_id: int = 0) -> np.ndarray:
    y = np.linspace(0, 1, resolution, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, resolution, dtype=np.float32)[None, :]
    shelf = -4500 + 4300 * np.exp(-((y - 0.85) ** 2) / 0.05)
    ridge = 600 * np.sin(2 * np.pi * (x * (bathymetry_id % 3 + 1))) * np.exp(-((y - 0.45) ** 2) / 0.08)
    noise = 80 * rng.normal(size=(resolution, resolution)).astype(np.float32)
    return (shelf + ridge + noise).astype(np.float32)
