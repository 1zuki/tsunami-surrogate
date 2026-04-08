from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from src.solver.source_modelling import make_grid
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import plot_fields


def generate_random_bathymetry(config: dict, rng: np.random.Generator, nx: int, ny: int, dx: float = 1.0, dy: float = 1.0) -> tuple[np.ndarray, Dict[str, float]]:
    bath_cfg = config.get("bathymetry", {})
    xx, yy = make_grid(nx, ny, dx=dx, dy=dy)
    x_norm = xx / max(np.max(np.abs(xx)), 1.0)
    y_norm = yy / max(np.max(np.abs(yy)), 1.0)

    min_depth = float(bath_cfg.get("min_depth", 1.0))
    max_depth = float(bath_cfg.get("max_depth", 4.0))
    slope_scale = float(bath_cfg.get("slope_scale", 0.6))
    noise_scale = float(bath_cfg.get("noise_scale", 0.15))
    amp_scale = float(bath_cfg.get("feature_amplitude_scale", 0.35))
    nmin, nmax = bath_cfg.get("num_features", [2, 5])
    sig_min, sig_max = bath_cfg.get("feature_sigma_range", [0.08, 0.28])
    smooth_min, smooth_max = bath_cfg.get("smoothing_sigma_range", [1.0, 3.0])

    base_depth = float(rng.uniform(min_depth, max_depth))
    depth = np.full((ny, nx), base_depth, dtype=np.float32)

    slope_x = float(rng.uniform(-slope_scale, slope_scale))
    slope_y = float(rng.uniform(-slope_scale, slope_scale))
    depth += slope_x * x_norm + slope_y * y_norm

    n_features = int(rng.integers(nmin, nmax + 1))
    for _ in range(n_features):
        x0 = float(rng.uniform(-0.7, 0.7))
        y0 = float(rng.uniform(-0.7, 0.7))
        sigma_x = float(rng.uniform(sig_min, sig_max))
        sigma_y = float(rng.uniform(sig_min, sig_max))
        angle = float(rng.uniform(0.0, 180.0))
        amp = float(rng.uniform(-amp_scale, amp_scale)) * (max_depth - min_depth)
        ca = np.cos(np.deg2rad(angle))
        sa = np.sin(np.deg2rad(angle))
        xr = ca * (x_norm - x0) + sa * (y_norm - y0)
        yr = -sa * (x_norm - x0) + ca * (y_norm - y0)
        bump = amp * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
        depth += bump.astype(np.float32)

    noise = rng.normal(size=(ny, nx)).astype(np.float32)
    smooth_sigma = float(rng.uniform(smooth_min, smooth_max))
    noise = gaussian_filter(noise, sigma=smooth_sigma)
    noise = noise / (np.std(noise) + 1e-6)
    depth += noise_scale * (max_depth - min_depth) * noise
    depth = np.clip(depth, min_depth, max_depth).astype(np.float32)

    bathymetry = -depth
    grad_y, grad_x = np.gradient(depth, dy, dx)
    roughness = float(np.sqrt(grad_x**2 + grad_y**2).std())
    meta = {
        "mean_depth": float(depth.mean()),
        "std_depth": float(depth.std()),
        "roughness": roughness,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "num_features": float(n_features),
    }
    return bathymetry.astype(np.float32), meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and preview a random bathymetry field.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="data/bathymetry/preview.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sim = config.get("simulation", {})
    bathy, meta = generate_random_bathymetry(config, rng, int(sim.get("nx", 32)), int(sim.get("ny", 32)), float(sim.get("dx", 1.0)), float(sim.get("dy", 1.0)))
    plot_fields({"Bathymetry": bathy}, args.output, title=str(meta), cmap=config.get("visualization", {}).get("cmap_bathy", "viridis"))


if __name__ == "__main__":
    main()
