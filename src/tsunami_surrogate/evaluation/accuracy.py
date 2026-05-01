from __future__ import annotations

from typing import Dict
import torch
from tsunami_surrogate.training.metrics import compute_metrics


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> Dict[str, float]:
    model.eval()
    sums = {'mae': 0.0, 'rmse': 0.0, 'rel_l2': 0.0, 'max_error': 0.0}
    n = 0
    for batch in loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        metrics = compute_metrics(pred, y)
        bs = x.size(0)
        for k, v in metrics.items():
            sums[k] += v * bs
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}
