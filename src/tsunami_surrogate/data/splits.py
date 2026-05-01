from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np


def _shuffle(indices: Sequence[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.array(indices, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def iid_split(n: int, train: float = 0.7, val: float = 0.15, test: float = 0.15, seed: int = 42) -> Dict[str, List[int]]:
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError('train + val + test must sum to 1')
    idx = _shuffle(range(n), seed)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    return {
        'train': idx[:n_train].tolist(),
        'val': idx[n_train:n_train + n_val].tolist(),
        'test': idx[n_train + n_val:].tolist(),
    }


def source_ood_split(metadata: List[dict], holdout_source_ids: Sequence[int], val_fraction: float = 0.15, seed: int = 42) -> Dict[str, List[int]]:
    holdout = set(holdout_source_ids)
    test = [i for i, m in enumerate(metadata) if int(m.get('source_id', -1)) in holdout]
    train_pool = [i for i, m in enumerate(metadata) if int(m.get('source_id', -1)) not in holdout]
    train_pool = _shuffle(train_pool, seed).tolist()
    n_val = max(1, int(round(len(train_pool) * val_fraction))) if train_pool else 0
    return {'train': train_pool[n_val:], 'val': train_pool[:n_val], 'test': test}


def amplitude_ood_split(metadata: List[dict], threshold: float, val_fraction: float = 0.15, seed: int = 42) -> Dict[str, List[int]]:
    test = [i for i, m in enumerate(metadata) if float(m.get('source_amplitude', 0.0)) >= threshold]
    train_pool = [i for i, m in enumerate(metadata) if float(m.get('source_amplitude', 0.0)) < threshold]
    train_pool = _shuffle(train_pool, seed).tolist()
    n_val = max(1, int(round(len(train_pool) * val_fraction))) if train_pool else 0
    return {'train': train_pool[n_val:], 'val': train_pool[:n_val], 'test': test}


def make_split(n: int, metadata: List[dict], cfg: dict) -> Dict[str, List[int]]:
    split_type = cfg.get('type', 'iid')
    seed = int(cfg.get('seed', 42))
    if split_type == 'iid':
        return iid_split(n, cfg.get('train', 0.7), cfg.get('val', 0.15), cfg.get('test', 0.15), seed)
    if split_type == 'source_ood':
        return source_ood_split(metadata, cfg.get('holdout_source_ids', []), cfg.get('val_fraction', 0.15), seed)
    if split_type == 'amplitude_ood':
        return amplitude_ood_split(metadata, cfg.get('threshold', 2.5), cfg.get('val_fraction', 0.15), seed)
    raise ValueError(f'Unknown split type: {split_type}')
