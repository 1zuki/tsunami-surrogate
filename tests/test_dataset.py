import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.dataset import TsunamiDataset, create_dataloaders, save_npz


def test_dataset_shape(tmp_path):
    path = tmp_path / 'toy.npz'
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 3, 16, 16), dtype=np.float32)
    y = rng.standard_normal((10, 1, 16, 16), dtype=np.float32)
    meta = {"source_id": np.asarray([f"sample_{i:03d}" for i in range(10)], dtype=object)}
    save_npz(str(path), x, y, meta)
    ds = TsunamiDataset(path)
    sample = ds[0]
    assert sample['x'].shape == (3, 16, 16)
    assert sample['y'].shape == (1, 16, 16)
    loaders = create_dataloaders({'data': {'path': str(path), 'batch_size': 2, 'split': {'type': 'iid'}}})
    assert set(loaders) == {'train', 'val', 'test'}


def test_create_dataloaders_presplit_dirs(tmp_path):
    root = tmp_path / "processed"
    rng = np.random.default_rng(1)

    def _write_split(name: str, n: int) -> None:
        split_dir = root / name
        split_dir.mkdir(parents=True, exist_ok=True)
        x = rng.standard_normal((n, 3, 8, 8), dtype=np.float32)
        y = rng.standard_normal((n, 1, 8, 8), dtype=np.float32)
        save_npz(split_dir / "eval_dataset.npz", x, y)

    _write_split("train", 6)
    _write_split("val", 4)
    _write_split("test", 5)

    loaders = create_dataloaders({"data": {"path": str(root), "batch_size": 2}})
    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) == 6
    assert len(loaders["val"].dataset) == 4
    assert len(loaders["test"].dataset) == 5
