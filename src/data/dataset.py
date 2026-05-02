from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.seed import make_torch_generator, make_worker_init_fn


def _as_nchw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, None, :, :].astype(np.float32)
    if arr.ndim == 4:
        return arr.astype(np.float32)
    raise ValueError(f"Expected [N,H,W] or [N,C,H,W], got {arr.shape}")


def _to_nchw_if_single(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, None, :, :].astype(np.float32)
    if arr.ndim == 4:
        return arr.astype(np.float32)
    if arr.ndim == 5 and arr.shape[2] == 1:
        return arr[:, :, 0, :, :].astype(np.float32)
    return arr.astype(np.float32)


def save_npz(path: str | Path, x: np.ndarray, y: np.ndarray, metadata: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = {}
    np.savez_compressed(path, x=np.asarray(x), y=np.asarray(y), metadata=np.array([metadata], dtype=object))


def make_toy_dataset(
    num_samples: int = 16,
    resolution: int = 32,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    h = w = int(resolution)
    yv, xv = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
    xs = []
    ys = []
    source_ids = []
    for _ in range(int(num_samples)):
        cx, cy = rng.uniform(-0.5, 0.5, size=2)
        amp = rng.uniform(0.6, 2.2)
        sigma = rng.uniform(0.08, 0.22)
        source = amp * np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2.0 * sigma**2))
        bathy = -1.0 + 0.15 * np.sin(2.0 * np.pi * xv) * np.cos(2.0 * np.pi * yv)
        init_depth = np.maximum(-bathy + source, 0.0)
        target = 0.7 * source + 0.15 * np.roll(source, 1, axis=0) + 0.15 * np.roll(source, -1, axis=1)
        x = np.stack([bathy, source, init_depth], axis=0)
        y = target[None, ...]
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
        source_ids.append(f"src_{int((cx > 0) + 2 * (cy > 0))}")
    return (
        np.stack(xs, axis=0),
        np.stack(ys, axis=0),
        {"source_id": np.asarray(source_ids, dtype=object)},
    )


@dataclass
class LoadedArrays:
    x: np.ndarray
    y: np.ndarray
    source_id: np.ndarray


def _load_arrays(path: str | Path) -> LoadedArrays:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        candidate = path / "eval_dataset.npz"
        if candidate.exists():
            path = candidate
        else:
            npz_candidates = sorted(path.glob("*.npz"))
            if not npz_candidates:
                raise FileNotFoundError(f"No .npz found in directory: {path}")
            path = npz_candidates[0]

    with np.load(path, allow_pickle=True) as data:
        if "x" in data and "y" in data:
            x = _as_nchw(data["x"])
            y = _to_nchw_if_single(data["y"])
        elif "inputs" in data and "targets" in data:
            x = _as_nchw(data["inputs"])
            y = _to_nchw_if_single(data["targets"])
        else:
            raise KeyError(f"Unsupported dataset keys in {path}. Expected x/y or inputs/targets.")

        if "source_id" in data:
            source_id = np.asarray(data["source_id"])
        elif "sample_id" in data:
            source_id = np.asarray(data["sample_id"])
        else:
            source_id = np.asarray([f"sample_{i:06d}" for i in range(x.shape[0])], dtype=object)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched sample count: x={x.shape[0]} y={y.shape[0]}")

    return LoadedArrays(x=x.astype(np.float32), y=y.astype(np.float32), source_id=source_id)


class TsunamiDataset(Dataset):
    def __init__(self, path: str | Path):
        arrays = _load_arrays(path)
        self.x = torch.from_numpy(arrays.x)
        self.y = torch.from_numpy(arrays.y)
        self.source_id = arrays.source_id

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "source_id": str(self.source_id[idx]),
            "sample_id": str(self.source_id[idx]),
        }


def _split_indices(n: int, split_cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_type = str(split_cfg.get("type", "iid")).lower()
    if split_type != "iid":
        # For now, fallback to deterministic IID-style split for unsupported modes.
        split_type = "iid"
    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(min(n_train, n), 0)
    n_val = min(max(n_val, 0), max(n - n_train - 1, 0)) if n >= 2 else 0
    n_test_start = n_train + n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_test_start]
    test_idx = perm[n_test_start:]
    if test_idx.size == 0 and val_idx.size > 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]
    if test_idx.size == 0 and train_idx.size > 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, val_idx, test_idx


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(seed),
        generator=make_torch_generator(seed),
    )


def create_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    if not data_cfg:
        raise KeyError("Config must contain `data` or `dataset` section.")
    path = data_cfg.get("path")
    if path is None:
        raise KeyError("data.path is required.")

    dataset = TsunamiDataset(path)
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", data_cfg.get("seed", 42)))
    split_cfg = data_cfg.get("split", {"type": "iid"})
    train_idx, val_idx, test_idx = _split_indices(len(dataset), split_cfg, seed)

    loaders: Dict[str, DataLoader] = {}
    if train_idx.size > 0:
        loaders["train"] = _make_loader(
            Subset(dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )
    if val_idx.size > 0:
        loaders["val"] = _make_loader(
            Subset(dataset, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 1,
            num_workers=num_workers,
        )
    if test_idx.size > 0:
        loaders["test"] = _make_loader(
            Subset(dataset, test_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 2,
            num_workers=num_workers,
        )
    return loaders

