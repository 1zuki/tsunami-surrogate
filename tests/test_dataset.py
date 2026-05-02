import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import TsunamiDataset, create_dataloaders, make_toy_dataset, save_npz


def test_dataset_shape(tmp_path):
    path = tmp_path / 'toy.npz'
    x, y, meta = make_toy_dataset(num_samples=10, resolution=16, seed=0)
    save_npz(str(path), x, y, meta)
    ds = TsunamiDataset(path)
    sample = ds[0]
    assert sample['x'].shape == (3, 16, 16)
    assert sample['y'].shape == (1, 16, 16)
    loaders = create_dataloaders({'data': {'path': str(path), 'batch_size': 2, 'split': {'type': 'iid'}}})
    assert set(loaders) == {'train', 'val', 'test'}
