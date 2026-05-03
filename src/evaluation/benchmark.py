from __future__ import annotations

import time
from typing import Dict

import torch


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)

    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))

    return out


@torch.no_grad()
def benchmark_inference(
    model,
    loader,
    device,
    warmup_steps: int = 5,
    timed_steps: int = 20,
) -> Dict[str, float]:
    model.eval()
    batches = []

    for batch in loader:
        batches.append(batch)
        if len(batches) >= max(1, timed_steps):
            break

    if not batches:
        return {"mean_seconds": 0.0, "samples_per_second": 0.0, "timed_steps": 0}

    for i in range(max(0, warmup_steps)):
        batch = batches[i % len(batches)]
        _model_output(model, batch["x"].to(device))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    total_samples = 0

    for i in range(max(1, timed_steps)):
        batch = batches[i % len(batches)]
        x = batch["x"].to(device)
        _model_output(model, x)
        total_samples += int(x.size(0))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    mean_seconds = elapsed / max(1, timed_steps)

    return {
        "mean_seconds": float(mean_seconds),
        "samples_per_second": float(total_samples / max(elapsed, 1e-12)),
        "timed_steps": int(timed_steps),
    }
