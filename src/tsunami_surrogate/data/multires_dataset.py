from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any
import torch
from torch.utils.data import Dataset
from .dataset import TsunamiDataset
from tsunami_surrogate.utils.resample import resize_field


class MultiResolutionDataset(Dataset):
    """Returns each sample at multiple target resolutions."""

    def __init__(self, path: str | Path, resolutions: Sequence[int]):
        self.base = TsunamiDataset(path)
        self.resolutions = list(resolutions)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.base[idx]
        out = {'metadata': sample['metadata'], 'index': sample['index']}
        for res in self.resolutions:
            size = (int(res), int(res))
            out[f'x_{res}'] = resize_field(sample['x'], size)
            out[f'y_{res}'] = resize_field(sample['y'], size)
        return out
