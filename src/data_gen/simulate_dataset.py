from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from src.data_gen.dataset import DatasetStats, compute_stats
from src.data_gen.generate_bathymetry import generate_random_bathymetry
from src.solver import build_solver
from src.solver.source_modelling import random_source
from src.utils.config import load_config, save_yaml
from src.utils.seed import set_seed
from src.utils.visualization import plot_prediction_vs_truth


def _simulate_one(index: int, config: dict) -> Dict[str, np.ndarray]:
    seed = int(config.get("project", {}).get("seed", 42)) + index
    rng = np.random.default_rng(seed)
    sim = config.get("simulation", {})
    nx, ny = int(sim.get("nx", 32)), int(sim.get("ny", 32))
    dx, dy = float(sim.get("dx", 1.0)), float(sim.get("dy", 1.0))
    nt = int(sim.get("nt", 20))

    bathy, bath_meta = generate_random_bathymetry(config, rng, nx, ny, dx, dy)
    disturbance, src_meta = random_source(config, rng, nx, ny, dx, dy)
    solver = build_solver(config)
    wave = solver.simulate(bathy, disturbance, nt=nt, save_every=int(sim.get("save_every", 1)))

    result = {
        "bathymetry": bathy.astype(np.float32),
        "disturbance": disturbance.astype(np.float32),
        "wave": wave.astype(np.float32),
    }
    result.update({k: np.asarray(v, dtype=np.float32) for k, v in {**bath_meta, **src_meta}.items()})
    result["source_center"] = np.asarray([src_meta["source_center_x"], src_meta["source_center_y"]], dtype=np.float32)
    return result


def _stack_results(items: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    items = list(items)
    keys = items[0].keys()
    stacked: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [item[key] for item in items]
        stacked[key] = np.stack(values, axis=0).astype(np.float32)
    return stacked


def _save_npz(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    if n >= 3:
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)
    elif n == 2:
        n_train, n_val, n_test = 1, 0, 1
    else:
        n_train, n_val, n_test = n, 0, 0

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def _subset(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[indices] for k, v in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic tsunami dataset.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--train-name", type=str, default=None)
    parser.add_argument("--val-name", type=str, default=None)
    parser.add_argument("--test-name", type=str, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("project", {})["seed"] = args.seed
    if args.nx is not None:
        config.setdefault("simulation", {})["nx"] = args.nx
    if args.ny is not None:
        config.setdefault("simulation", {})["ny"] = args.ny
    if args.n_samples is not None:
        config.setdefault("dataset_generation", {})["n_samples"] = args.n_samples

    set_seed(int(config.get("project", {}).get("seed", 42)), deterministic=bool(config.get("project", {}).get("deterministic", False)))

    paths = config.get("paths", {})
    dataset_dir = Path(args.dataset_dir or paths.get("dataset_dir", "data/synthetic/default"))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_name = args.train_name or Path(paths.get("train_file", dataset_dir / "train.npz")).name
    val_name = args.val_name or Path(paths.get("val_file", dataset_dir / "val.npz")).name
    test_name = args.test_name or Path(paths.get("test_file", dataset_dir / "test.npz")).name
    stats_path = dataset_dir / Path(paths.get("stats_file", dataset_dir / "stats.yaml")).name

    gen_cfg = config.get("dataset_generation", {})
    n_samples = int(gen_cfg.get("n_samples", 512))
    num_workers = int(gen_cfg.get("num_workers", 0))
    preview_every = int(gen_cfg.get("preview_every", 0))

    if num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_simulate_one, range(n_samples), [config] * n_samples))
    else:
        results = [_simulate_one(i, config) for i in range(n_samples)]

    arrays = _stack_results(results)
    train_idx, val_idx, test_idx = _split_indices(
        n_samples,
        float(gen_cfg.get("train_ratio", 0.8)),
        float(gen_cfg.get("val_ratio", 0.1)),
        int(config.get("project", {}).get("seed", 42)),
    )

    train_arrays = _subset(arrays, train_idx)
    val_arrays = _subset(arrays, val_idx)
    test_arrays = _subset(arrays, test_idx)

    train_path = dataset_dir / train_name
    val_path = dataset_dir / val_name
    test_path = dataset_dir / test_name
    _save_npz(train_path, train_arrays)
    _save_npz(val_path, val_arrays)
    _save_npz(test_path, test_arrays)

    stats = compute_stats(train_path, input_keys=["bathymetry", "disturbance"], target_key="wave")
    stats.save(stats_path)

    save_yaml(
        {
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "nx": int(config.get("simulation", {}).get("nx", 32)),
            "ny": int(config.get("simulation", {}).get("ny", 32)),
            "nt": int(config.get("simulation", {}).get("nt", 20)),
        },
        dataset_dir / "dataset_summary.yaml",
    )

    if preview_every > 0:
        figure_dir = Path(config.get("paths", {}).get("figure_dir", "figures"))
        figure_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(preview_every, len(test_arrays["wave"]))):
            plot_prediction_vs_truth(
                test_arrays["bathymetry"][i],
                test_arrays["disturbance"][i],
                test_arrays["wave"][i],
                test_arrays["wave"][i],
                figure_dir / f"dataset_preview_{i}.png",
            )

    print(f"Saved dataset to {dataset_dir}")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
