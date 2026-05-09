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


@dataclass
class LoadedArrays:
    x: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    source_id: np.ndarray
    source_type: np.ndarray
    bathymetry_type: np.ndarray
    source_strength: np.ndarray
    scenario_id: np.ndarray
    solver_name: np.ndarray


def _string_array(data: Any, n: int, default: str) -> np.ndarray:
    arr = np.asarray(data) if data is not None else np.asarray([], dtype=object)
    if arr.size == 0:
        return np.asarray([default] * n, dtype=object)

    arr = arr.reshape(-1)
    out = np.asarray([str(v) for v in arr], dtype=object)

    if out.shape[0] != n:
        return np.asarray([default] * n, dtype=object)

    return out


def _float_array(data: Any, n: int, default: float = np.nan) -> np.ndarray:
    arr = np.asarray(data) if data is not None else np.asarray([], dtype=float)
    if arr.size == 0:
        return np.full((n,), float(default), dtype=np.float32)

    arr = arr.reshape(-1).astype(np.float32)
    if arr.shape[0] != n:
        return np.full((n,), float(default), dtype=np.float32)

    return arr


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

        n = x.shape[0]

        sample_id = _string_array(
            data["sample_id"] if "sample_id" in data else None,
            n,
            default="",
        )
        if not np.any(sample_id != ""):
            sample_id = np.asarray([f"sample_{i:06d}" for i in range(n)], dtype=object)

        source_type = _string_array(
            data["source_type"] if "source_type" in data else None,
            n,
            default="unknown",
        )
        source_id = _string_array(
            data["source_id"] if "source_id" in data else source_type,
            n,
            default="unknown",
        )
        bathymetry_type = _string_array(
            data["bathymetry_type"] if "bathymetry_type" in data else None,
            n,
            default="unknown",
        )
        source_strength = _float_array(
            data["source_strength"] if "source_strength" in data else None,
            n,
            default=np.nan,
        )
        scenario_id = _string_array(
            data["scenario_id"] if "scenario_id" in data else sample_id,
            n,
            default="",
        )
        solver_name = _string_array(
            data["solver_name"] if "solver_name" in data else None,
            n,
            default="unknown",
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched sample count: x={x.shape[0]} y={y.shape[0]}")

    return LoadedArrays(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        sample_id=sample_id,
        source_id=source_id,
        source_type=source_type,
        bathymetry_type=bathymetry_type,
        source_strength=source_strength,
        scenario_id=scenario_id,
        solver_name=solver_name,
    )


class TsunamiDataset(Dataset):
    def __init__(self, path: str | Path):
        arrays = _load_arrays(path)
        self.x = torch.from_numpy(arrays.x)
        self.y = torch.from_numpy(arrays.y)
        self.sample_id = arrays.sample_id
        self.source_id = arrays.source_id
        self.source_type = arrays.source_type
        self.bathymetry_type = arrays.bathymetry_type
        self.source_strength = arrays.source_strength
        self.scenario_id = arrays.scenario_id
        self.solver_name = arrays.solver_name

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "sample_id": str(self.sample_id[idx]),
            "source_id": str(self.source_id[idx]),
            "source_type": str(self.source_type[idx]),
            "bathymetry_type": str(self.bathymetry_type[idx]),
            "source_strength": float(self.source_strength[idx]),
            "scenario_id": str(self.scenario_id[idx]),
            "solver_name": str(self.solver_name[idx]),
        }


def _split_indices(n: int, split_cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_type = str(split_cfg.get("type", "iid")).lower()

    if split_type != "iid":
        # fallback to deterministic IID-style split for unsupported modes
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

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", data_cfg.get("seed", 42)))

    loaders: Dict[str, DataLoader] = {}

    # Preferred mode: use explicit pre-split datasets if provided.
    split_paths = {
        "train": data_cfg.get("train_path"),
        "val": data_cfg.get("val_path"),
        "test": data_cfg.get("test_path"),
    }
    if any(v is not None for v in split_paths.values()):
        for split_name, split_path in split_paths.items():
            if split_path is None:
                continue
            split_dataset = TsunamiDataset(split_path)
            loaders[split_name] = _make_loader(
                split_dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                seed=seed + {"train": 0, "val": 1, "test": 2}[split_name],
                num_workers=num_workers,
            )
        if not loaders:
            raise ValueError("Explicit split-path mode was requested, but no split dataset could be loaded.")
        return loaders

    path = data_cfg.get("path")
    if path is None:
        raise KeyError("data.path is required when explicit train_path/val_path/test_path are not provided.")

    # Auto-detect common pre-split folder layout:
    # data/processed/{train,val,test}/eval_dataset.npz
    root = Path(path)
    if root.is_dir():
        pre_split = {
            "train": root / "train",
            "val": root / "val",
            "test": root / "test",
        }
        if any(p.exists() for p in pre_split.values()):
            for split_name, split_path in pre_split.items():
                if not split_path.exists():
                    continue
                split_dataset = TsunamiDataset(split_path)
                loaders[split_name] = _make_loader(
                    split_dataset,
                    batch_size=batch_size,
                    shuffle=(split_name == "train"),
                    seed=seed + {"train": 0, "val": 1, "test": 2}[split_name],
                    num_workers=num_workers,
                )
            if loaders:
                return loaders

    # Backward-compatible mode: one dataset path + random split.
    dataset = TsunamiDataset(path)
    split_cfg = data_cfg.get("split", {"type": "iid"})
    train_idx, val_idx, test_idx = _split_indices(len(dataset), split_cfg, seed)

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
