from __future__ import annotations

from typing import Dict
import torch
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable
from .metrics import MetricAccumulator


def _require_finite_tensor(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"Nonfinite {label} encountered")


def _require_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _require_finite_tensor(parameter.grad, f"gradient for {name}")


def _require_finite_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        _require_finite_tensor(parameter.detach(), f"parameter {name}")


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
        if pred.shape != y.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: pred={tuple(pred.shape)} target={tuple(y.shape)}. "
                "Check preprocess target horizon/channel settings vs model out_channels."
            )
        _require_finite_tensor(pred, "training prediction")
        _require_finite_tensor(y, "training target")
        loss = loss_fn(pred, y, batch)
        _require_finite_tensor(loss, "training loss")
        loss.backward()
        _require_finite_gradients(model)
    
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            _require_finite_gradients(model)
    
        optimizer.step()
        _require_finite_parameters(model)
        total_loss += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    
    return {'loss': total_loss / max(1, n)}


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metrics_acc = MetricAccumulator()
    n = 0
    
    for batch in tqdm(loader, desc='eval', leave=False):
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = _model_output(model, x)
        if pred.shape != y.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: pred={tuple(pred.shape)} target={tuple(y.shape)}. "
                "Check preprocess target horizon/channel settings vs model out_channels."
        )
        _require_finite_tensor(pred, "evaluation prediction")
        _require_finite_tensor(y, "evaluation target")
        loss = loss_fn(pred, y, batch)
        _require_finite_tensor(loss, "evaluation loss")
        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        metrics_acc.update(pred, y)
        n += bs

    out = metrics_acc.compute()
    out['loss'] = total_loss / max(1, n)
    
    return out
