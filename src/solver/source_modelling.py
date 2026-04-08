from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


SOURCE_NAME_TO_ID = {
    "gaussian": 0,
    "dipole": 1,
    "ring": 2,
    "okada_like": 3,
}


def make_grid(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    xx, yy = np.meshgrid(x, y)
    return xx.astype(np.float32), yy.astype(np.float32)


def _rotate(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    x = xx - x0
    y = yy - y0
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return xr, yr


def gaussian_source(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, amplitude: float, sigma_x: float, sigma_y: float, angle_deg: float = 0.0) -> np.ndarray:
    xr, yr = _rotate(xx, yy, x0, y0, math.radians(angle_deg))
    field = amplitude * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return field.astype(np.float32)


def dipole_source(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, amplitude: float, sigma_x: float, sigma_y: float, offset: float, angle_deg: float = 0.0) -> np.ndarray:
    angle = math.radians(angle_deg)
    dx = offset * math.cos(angle)
    dy = offset * math.sin(angle)
    uplift = gaussian_source(xx, yy, x0 + dx, y0 + dy, amplitude, sigma_x, sigma_y, angle_deg)
    subsidence = gaussian_source(xx, yy, x0 - dx, y0 - dy, 0.8 * amplitude, 1.05 * sigma_x, 1.05 * sigma_y, angle_deg)
    return (uplift - subsidence).astype(np.float32)


def ring_source(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, amplitude: float, radius: float, sigma: float) -> np.ndarray:
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    field = amplitude * np.exp(-0.5 * ((r - radius) / max(sigma, 1e-6)) ** 2)
    return field.astype(np.float32)


def okada_like_source(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, amplitude: float, sigma_x: float, sigma_y: float, offset: float, angle_deg: float = 0.0) -> np.ndarray:
    """A compact uplift/subsidence pattern inspired by coseismic deformation.

    This is not a full Okada solution. It is a smooth synthetic source designed to mimic
    the paired uplift/subsidence structure often seen in coseismic seafloor displacement.
    """
    primary = gaussian_source(xx, yy, x0, y0, amplitude, sigma_x, sigma_y, angle_deg)
    dipole = dipole_source(xx, yy, x0, y0, 0.9 * amplitude, 0.8 * sigma_x, 1.2 * sigma_y, offset, angle_deg + 10.0)
    return (0.65 * primary + 0.75 * dipole).astype(np.float32)


def random_source(config: dict, rng: np.random.Generator, nx: int, ny: int, dx: float = 1.0, dy: float = 1.0) -> tuple[np.ndarray, Dict[str, float]]:
    source_cfg = config.get("source", {})
    xx, yy = make_grid(nx, ny, dx=dx, dy=dy)

    source_types = list(source_cfg.get("types", ["gaussian", "dipole", "ring", "okada_like"]))
    source_type = str(rng.choice(source_types))

    amp_min, amp_max = source_cfg.get("amplitude_range", [0.05, 0.35])
    sig_min, sig_max = source_cfg.get("sigma_range", [1.5, 5.0])
    off_min, off_max = source_cfg.get("offset_range", [1.0, 4.0])
    ring_min, ring_max = source_cfg.get("ring_radius_range", [2.0, 6.0])

    amplitude = float(rng.uniform(amp_min, amp_max))
    sigma_x = float(rng.uniform(sig_min, sig_max))
    sigma_y = float(rng.uniform(sig_min, sig_max))
    angle = float(rng.uniform(0.0, 180.0))
    offset = float(rng.uniform(off_min, off_max))
    radius = float(rng.uniform(ring_min, ring_max))

    x0 = float(rng.uniform(-0.30 * nx * dx, 0.30 * nx * dx))
    y0 = float(rng.uniform(-0.30 * ny * dy, 0.30 * ny * dy))

    if source_type == "gaussian":
        field = gaussian_source(xx, yy, x0, y0, amplitude, sigma_x, sigma_y, angle)
    elif source_type == "dipole":
        field = dipole_source(xx, yy, x0, y0, amplitude, sigma_x, sigma_y, offset, angle)
    elif source_type == "ring":
        field = ring_source(xx, yy, x0, y0, amplitude, radius, 0.5 * (sigma_x + sigma_y))
    elif source_type == "okada_like":
        field = okada_like_source(xx, yy, x0, y0, amplitude, sigma_x, sigma_y, offset, angle)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    metadata = {
        "source_type": float(SOURCE_NAME_TO_ID[source_type]),
        "source_center_x": x0,
        "source_center_y": y0,
        "source_amplitude": amplitude,
        "source_sigma_x": sigma_x,
        "source_sigma_y": sigma_y,
        "source_angle": angle,
        "source_offset": offset,
        "source_radius": radius,
    }
    return field.astype(np.float32), metadata
