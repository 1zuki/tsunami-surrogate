from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from src.training.metrics import compute_metrics
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
) -> Dict[str, float]:
    model.eval()
    sums = {"mae": 0.0, "rmse": 0.0, "rel_l2": 0.0, "max_error": 0.0}
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = _model_output(model, x)
        pred_eval = apply_target_denorm(pred, target_denorm)
        y_eval = apply_target_denorm(y, target_denorm)
        metrics = compute_metrics(pred_eval, y_eval)
        bs = x.size(0)

        for key, value in metrics.items():
            sums[key] += float(value) * bs

        n += bs

    return {key: value / max(1, n) for key, value in sums.items()}
