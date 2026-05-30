from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict

import torch

from src.utils.precision import PrecisionConfig, autocast_context


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
    max_batches: int | None = None,
    precision_cfg: PrecisionConfig | None = None,
) -> Dict[str, float]:
    model.eval()
    batches = []
    timed_steps = max(1, int(timed_steps))
    warmup_steps = max(0, int(warmup_steps))
    limit = timed_steps if max_batches is None else max(1, int(max_batches))

    for batch in loader:
        batches.append(batch)
        if len(batches) >= limit:
            break

    if not batches:
        return {
            "time_total_s": 0.0,
            "time_per_batch_mean_s": 0.0,
            "time_per_sample_mean_s": 0.0,
            "samples_per_second": 0.0,
            "num_batches_timed": 0,
            "num_samples_timed": 0,
            "num_warmup": warmup_steps,
            "num_repeats": timed_steps,
            # backward-compatible aliases:
            "mean_seconds": 0.0,
            "timed_steps": 0,
        }

    input_dtype = precision_cfg.input_dtype if precision_cfg is not None else None

    for i in range(warmup_steps):
        batch = batches[i % len(batches)]
        x = batch["x"].to(device=device, non_blocking=(device.type == "cuda"))
        if input_dtype is not None:
            x = x.to(dtype=input_dtype)
        with autocast_context(device, precision_cfg) if precision_cfg is not None else nullcontext():
            _model_output(model, x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    total_samples = 0

    for i in range(timed_steps):
        batch = batches[i % len(batches)]
        x = batch["x"].to(device=device, non_blocking=(device.type == "cuda"))
        if input_dtype is not None:
            x = x.to(dtype=input_dtype)
        with autocast_context(device, precision_cfg) if precision_cfg is not None else nullcontext():
            _model_output(model, x)
        total_samples += int(x.size(0))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    time_per_batch = elapsed / float(max(1, timed_steps))
    time_per_sample = elapsed / float(max(1, total_samples))

    return {
        "time_total_s": float(elapsed),
        "time_per_batch_mean_s": float(time_per_batch),
        "time_per_sample_mean_s": float(time_per_sample),
        "samples_per_second": float(total_samples / max(elapsed, 1e-12)),
        "num_batches_timed": int(timed_steps),
        "num_samples_timed": int(total_samples),
        "num_warmup": int(warmup_steps),
        "num_repeats": int(timed_steps),
        # backward-compatible aliases:
        "mean_seconds": float(time_per_batch),
        "timed_steps": int(timed_steps),
    }
