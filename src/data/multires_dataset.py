from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import _as_nchw, _to_nchw_if_single
from src.utils.resample import resize_field


class MultiResolutionDataset(Dataset):
    def __init__(self, path: str | Path, resolutions: Iterable[int]):
        path = Path(path)
        if path.is_dir():
            candidate = path / "eval_dataset.npz"
            if candidate.exists():
                path = candidate
            else:
                files = sorted(path.glob("*.npz"))
                if not files:
                    raise FileNotFoundError(f"No .npz found in: {path}")
                path = files[0]

        with np.load(path, allow_pickle=True) as data:
            if "x" in data and "y" in data:
                x = _as_nchw(data["x"])
                y = _to_nchw_if_single(data["y"])
            elif "inputs" in data and "targets" in data:
                x = _as_nchw(data["inputs"])
                y = _to_nchw_if_single(data["targets"])
            else:
                raise KeyError("Expected x/y or inputs/targets in multi-resolution dataset.")

        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.resolutions = [int(r) for r in resolutions]
        if not self.resolutions:
            raise ValueError("resolutions must not be empty")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        out = {}
        for res in self.resolutions:
            out[f"x_{res}"] = resize_field(x, (res, res))
            out[f"y_{res}"] = resize_field(y, (res, res))
        return out

