import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.model_io import validate_model_io_channels


class _DictDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


def test_validate_model_io_channels_passes():
    x = torch.zeros(4, 3, 8, 8)
    y = torch.zeros(4, 5, 8, 8)
    loaders = {"train": DataLoader(_DictDataset(x, y), batch_size=2)}
    cfg = {"model": {"in_channels": 3, "out_channels": 5}}
    validate_model_io_channels(cfg, loaders)


def test_validate_model_io_channels_fails_for_in_channels():
    x = torch.zeros(4, 4, 8, 8)
    y = torch.zeros(4, 5, 8, 8)
    loaders = {"train": DataLoader(_DictDataset(x, y), batch_size=2)}
    cfg = {"model": {"in_channels": 3, "out_channels": 5}}
    with pytest.raises(ValueError, match="in_channels"):
        validate_model_io_channels(cfg, loaders)
