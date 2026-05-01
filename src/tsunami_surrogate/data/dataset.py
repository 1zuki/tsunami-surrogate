from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .splits import make_split


class TsunamiDataset(Dataset):
    """Loads `.npz` tensors with x, y, and optional metadata.

    x shape: [N,C_in,H,W]
    y shape: [N,C_out,H,W]
    metadata: JSON string or object array of dict-like metadata.
    """

    def __init__(self, path: str | Path, indices: Optional[Sequence[int]] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'Dataset not found: {self.path}')
        data = np.load(self.path, allow_pickle=True)
        self.x = torch.from_numpy(data['x']).float()
        self.y = torch.from_numpy(data['y']).float()
        self.metadata = self._read_metadata(data)
        self.indices = list(indices) if indices is not None else list(range(len(self.x)))

    def _read_metadata(self, data) -> List[Dict[str, Any]]:
        if 'metadata' not in data:
            return [{'sample_id': i} for i in range(len(data['x']))]
        raw = data['metadata']
        if raw.shape == ():
            return json.loads(str(raw.item()))
        out = []
        for item in raw.tolist():
            if isinstance(item, str):
                out.append(json.loads(item))
            elif isinstance(item, dict):
                out.append(item)
            else:
                out.append(dict(item))
        return out

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        return {'x': self.x[idx], 'y': self.y[idx], 'metadata': self.metadata[idx], 'index': idx}


def create_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = cfg.get('data', cfg)
    full = TsunamiDataset(data_cfg['path'])
    splits = make_split(len(full.x), full.metadata, data_cfg.get('split', {'type': 'iid'}))
    loaders = {}
    for name, indices in splits.items():
        ds = TsunamiDataset(data_cfg['path'], indices=indices)
        loaders[name] = DataLoader(
            ds,
            batch_size=int(data_cfg.get('batch_size', 8)),
            shuffle=(name == 'train'),
            num_workers=int(data_cfg.get('num_workers', 0)),
        )
    return loaders
