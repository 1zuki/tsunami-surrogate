from __future__ import annotations

from collections import defaultdict
import torch
from tsunami_surrogate.training.metrics import compute_metrics


@torch.no_grad()
def evaluate_by_regime(model, loader, device, key: str = 'source_id'):
    model.eval()
    sums = defaultdict(lambda: {'n': 0, 'mae': 0.0, 'rmse': 0.0, 'rel_l2': 0.0, 'max_error': 0.0})
    for batch in loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        for i, meta in enumerate(batch['metadata']):
            group = str(meta.get(key, 'unknown')) if isinstance(meta, dict) else 'unknown'
            metrics = compute_metrics(pred[i:i+1], y[i:i+1])
            sums[group]['n'] += 1
            for k, v in metrics.items():
                sums[group][k] += v
    return {g: {k: (v / vals['n'] if k != 'n' else v) for k, v in vals.items()} for g, vals in sums.items()}
