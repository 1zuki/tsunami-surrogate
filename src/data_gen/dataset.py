from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import save_yaml, load_yaml


@dataclass
class DatasetStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: float
    target_std: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_mean": self.input_mean.astype(float).tolist(),
            "input_std": self.input_std.astype(float).tolist(),
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DatasetStats":
        return cls(
            input_mean=np.asarray(data["input_mean"], dtype=np.float32),
            input_std=np.asarray(data["input_std"], dtype=np.float32),
            target_mean=float(data["target_mean"]),
            target_std=float(data["target_std"]),
        )

    def save(self, path: str | Path) -> None:
        save_yaml(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "DatasetStats":
        return cls.from_dict(load_yaml(path))


def load_npz_dict(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compute_stats(file_path: str | Path, input_keys: Iterable[str], target_key: str) -> DatasetStats:
    arrays = load_npz_dict(file_path)
    inputs = np.stack([arrays[key] for key in input_keys], axis=1).astype(np.float32)
    target = arrays[target_key].astype(np.float32)
    input_mean = inputs.mean(axis=(0, 2, 3))
    input_std = inputs.std(axis=(0, 2, 3)) + 1e-6
    target_mean = float(target.mean())
    target_std = float(target.std() + 1e-6)
    return DatasetStats(input_mean=input_mean, input_std=input_std, target_mean=target_mean, target_std=target_std)


def normalize_inputs(x: np.ndarray, stats: DatasetStats) -> np.ndarray:
    return ((x - stats.input_mean[:, None, None]) / stats.input_std[:, None, None]).astype(np.float32)


def denormalize_inputs(x: np.ndarray, stats: DatasetStats) -> np.ndarray:
    return (x * stats.input_std[:, None, None] + stats.input_mean[:, None, None]).astype(np.float32)


def normalize_targets(y: np.ndarray, stats: DatasetStats) -> np.ndarray:
    return ((y - stats.target_mean) / stats.target_std).astype(np.float32)


def denormalize_targets(y: np.ndarray, stats: DatasetStats) -> np.ndarray:
    return (y * stats.target_std + stats.target_mean).astype(np.float32)


class TsunamiDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        input_keys: List[str],
        target_key: str,
        stats: Optional[DatasetStats] = None,
        normalize_input: bool = True,
        normalize_target: bool = True,
        augment: Optional[Dict[str, object]] = None,
        return_meta: bool = True,
    ):
        self.file_path = str(file_path)
        self.arrays = load_npz_dict(file_path)
        self.input_keys = list(input_keys)
        self.target_key = target_key
        self.stats = stats
        self.normalize_input = normalize_input and stats is not None
        self.normalize_target = normalize_target and stats is not None
        self.augment = augment or {}
        self.return_meta = return_meta
        self.length = int(self.arrays[self.target_key].shape[0])
        self.metadata_keys = [
            k for k, v in self.arrays.items() if k not in set(self.input_keys + [self.target_key]) and v.shape[0] == self.length
        ]

    def __len__(self) -> int:
        return self.length

    def _apply_augmentation(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.augment.get("horizontal_flip", False) and np.random.rand() < 0.5:
            x = np.flip(x, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()
        if self.augment.get("vertical_flip", False) and np.random.rand() < 0.5:
            x = np.flip(x, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
        noise_std = float(self.augment.get("noise_std", 0.0) or 0.0)
        if noise_std > 0:
            x = x + np.random.normal(scale=noise_std, size=x.shape).astype(np.float32)
        return x, y

    def __getitem__(self, idx: int):
        x = np.stack([self.arrays[key][idx] for key in self.input_keys], axis=0).astype(np.float32)
        y = self.arrays[self.target_key][idx].astype(np.float32)

        if self.normalize_input:
            x = normalize_inputs(x, self.stats)
        if self.normalize_target:
            y = normalize_targets(y, self.stats)
        if self.augment:
            x, y = self._apply_augmentation(x, y)

        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)

        if not self.return_meta:
            return xt, yt

        meta = {k: torch.from_numpy(np.asarray(self.arrays[k][idx])) for k in self.metadata_keys}
        return xt, yt, meta
