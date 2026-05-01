from __future__ import annotations

from typing import List, Tuple
import json
import numpy as np
from tsunami_surrogate.data_gen.generate_bathymetry import generate_bathymetry
from tsunami_surrogate.data_gen.generate_sources import sample_source
from tsunami_surrogate.solver.runner import run_solver


def simulate_dataset(num_samples: int, resolution: int, seed: int = 42, solver: str = 'shallow_water') -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    rng = np.random.default_rng(seed)
    xs, ys, metadata = [], [], []
    for i in range(num_samples):
        source_id = int(i % 8)
        bathymetry_id = int((i // 8) % 4)
        bathy = generate_bathymetry(resolution, rng, bathymetry_id)
        source, center, amplitude, sigma = sample_source(resolution, rng, source_id)
        target = run_solver(source, bathy, solver=solver, steps=25)
        # Channels: source map, normalized bathymetry, coastal mask
        bathy_norm = (bathy - bathy.mean()) / (bathy.std() + 1e-6)
        coastal_mask = (bathy > -500).astype(np.float32)
        x = np.stack([source, bathy_norm, coastal_mask], axis=0).astype(np.float32)
        y = target[None, ...].astype(np.float32)
        xs.append(x)
        ys.append(y)
        metadata.append({
            'sample_id': i,
            'source_id': source_id,
            'source_amplitude': amplitude,
            'source_location': center,
            'source_sigma': sigma,
            'bathymetry_id': bathymetry_id,
            'grid_resolution': resolution,
            'time_horizon': 25,
            'solver': solver,
        })
    return np.stack(xs), np.stack(ys), metadata


def save_npz(path: str, x: np.ndarray, y: np.ndarray, metadata: List[dict]) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, metadata=json.dumps(metadata))
