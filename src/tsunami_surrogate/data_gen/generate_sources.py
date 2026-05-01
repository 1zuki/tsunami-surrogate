from __future__ import annotations

import numpy as np


def gaussian_source(resolution: int, center, amplitude: float, sigma: float) -> np.ndarray:
    xs = np.linspace(0, 1, resolution, dtype=np.float32)
    ys = np.linspace(0, 1, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    cx, cy = center
    return (amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def sample_source(resolution: int, rng: np.random.Generator, source_id: int):
    centers = [(0.2, 0.25), (0.35, 0.35), (0.5, 0.25), (0.65, 0.35), (0.8, 0.25), (0.25, 0.55), (0.5, 0.5), (0.75, 0.55)]
    center = centers[source_id % len(centers)]
    amplitude = float(rng.uniform(0.5, 3.0))
    sigma = float(rng.uniform(0.04, 0.11))
    return gaussian_source(resolution, center, amplitude, sigma), center, amplitude, sigma
