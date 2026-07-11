from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from src.training.metrics import MetricAccumulator
from src.evaluation.target_scaling import apply_target_denorm


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)

    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))

    return out


@torch.no_grad()
def evaluate_accuracy(
    model,
    loader,
    device,
    target_denorm: Optional[Tuple[float, float]] = None,
    batch_transform: Optional[
        Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
) -> Dict[str, float]:
    model.eval()
    metrics_acc = MetricAccumulator()

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if batch_transform is not None:
            x, y = batch_transform(x, y)

        pred = _model_output(model, x)
        pred_eval = apply_target_denorm(pred, target_denorm)
        y_eval = apply_target_denorm(y, target_denorm)
        metrics_acc.update(pred_eval, y_eval)

    return metrics_acc.compute()
