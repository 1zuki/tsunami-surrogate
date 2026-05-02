from __future__ import annotations

from typing import Dict
import torch
from tqdm import tqdm
from .metrics import compute_metrics


def _model_output(model, x):
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get('mean', next(iter(out.values())))
    return out


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip: float | None = None) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc='train', leave=False):
        x, y = batch['x'].to(device), batch['y'].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = _model_output(model, x)
        loss = loss_fn(pred, y, batch)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return {'loss': total_loss / max(1, n)}


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metric_sums = {'mae': 0.0, 'rmse': 0.0, 'rel_l2': 0.0, 'max_error': 0.0}
    n = 0
    for batch in tqdm(loader, desc='eval', leave=False):
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = _model_output(model, x)
        loss = loss_fn(pred, y, batch)
        metrics = compute_metrics(pred, y)
        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        for k, v in metrics.items():
            metric_sums[k] += v * bs
        n += bs
    out = {k: v / max(1, n) for k, v in metric_sums.items()}
    out['loss'] = total_loss / max(1, n)
    return out
