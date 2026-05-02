from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from src.training.metrics import compute_metrics


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


@torch.no_grad()
def evaluate_by_regime(model, loader, device, key: str = "source_id") -> Dict[str, Dict[str, float]]:
    model.eval()
    sums = defaultdict(lambda: {"mae": 0.0, "rmse": 0.0, "rel_l2": 0.0, "max_error": 0.0, "n": 0})
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = _model_output(model, x)
        metrics = compute_metrics(pred, y)
        labels = batch.get(key, ["unknown"] * x.size(0))
        for i in range(x.size(0)):
            label = str(labels[i])
            sums[label]["mae"] += float(metrics["mae"])
            sums[label]["rmse"] += float(metrics["rmse"])
            sums[label]["rel_l2"] += float(metrics["rel_l2"])
            sums[label]["max_error"] += float(metrics["max_error"])
            sums[label]["n"] += 1

    out: Dict[str, Dict[str, float]] = {}
    for label, row in sums.items():
        n = max(1, int(row["n"]))
        out[label] = {
            "mae": row["mae"] / n,
            "rmse": row["rmse"] / n,
            "rel_l2": row["rel_l2"] / n,
            "max_error": row["max_error"] / n,
            "n": float(row["n"]),
        }
    return out

