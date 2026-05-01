from __future__ import annotations

import time
import torch


@torch.no_grad()
def benchmark_inference(model, loader, device, warmup_steps: int = 5, timed_steps: int = 20):
    model.eval()
    batches = list(loader)
    if not batches:
        raise ValueError('Loader is empty')
    for i in range(warmup_steps):
        batch = batches[i % len(batches)]
        _ = model(batch['x'].to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    samples = 0
    for i in range(timed_steps):
        batch = batches[i % len(batches)]
        x = batch['x'].to(device)
        _ = model(x)
        samples += x.size(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        'seconds': elapsed,
        'samples': samples,
        'latency_ms_per_sample': 1000 * elapsed / max(1, samples),
        'throughput_samples_per_second': samples / max(elapsed, 1e-9),
    }
