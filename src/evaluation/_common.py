from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_gen.dataset import DatasetStats, TsunamiDataset, compute_stats, denormalize_inputs
from src.models import build_model
from src.solver import build_solver
from src.training.metrics import average_metric_dicts, compute_metrics_np, compute_metrics_torch
from src.utils.config import load_config, maybe_resolve_device, save_yaml
from src.utils.seed import make_generator, seed_worker


def load_stats_from_config(config: dict) -> DatasetStats:
    stats_path = Path(config.get("paths", {}).get("stats_file", "data/synthetic/default/stats.yaml"))
    if stats_path.exists():
        return DatasetStats.load(stats_path)
    train_file = config.get("paths", {}).get("train_file")
    stats = compute_stats(train_file, input_keys=["bathymetry", "disturbance"], target_key="wave")
    stats.save(stats_path)
    return stats


def make_eval_loader(config: dict, split: str = "test", return_meta: bool = True) -> DataLoader:
    stats = load_stats_from_config(config)
    path_key = f"{split}_file"
    file_path = config.get("paths", {}).get(path_key)
    eval_cfg = config.get("evaluation", {})
    dataset = TsunamiDataset(
        file_path=file_path,
        input_keys=["bathymetry", "disturbance"],
        target_key="wave",
        stats=stats,
        normalize_input=bool(config.get("normalization", {}).get("normalize_inputs", True)),
        normalize_target=bool(config.get("normalization", {}).get("normalize_targets", True)),
        augment=None,
        return_meta=return_meta,
    )
    return DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        pin_memory=False,
        worker_init_fn=seed_worker if int(eval_cfg.get("num_workers", 0)) > 0 else None,
        generator=make_generator(int(config.get("project", {}).get("seed", 42))),
    )


def load_checkpoint_and_model(config_path: str, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
    config = load_config(config_path)
    device_name = maybe_resolve_device(device or str(config.get("project", {}).get("device", "auto")))
    device_obj = torch.device(device_name)

    if checkpoint_path is None:
        checkpoint_path = str(Path(config.get("paths", {}).get("checkpoint_dir", "results/default/checkpoints")) / "best.pt")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state.get("config"), dict):
        config = state["config"]
    model = build_model(config)
    model.load_state_dict(state["model_state"])
    model.to(device_obj)
    model.eval()

    if "stats" in state and isinstance(state["stats"], dict):
        stats = DatasetStats.from_dict(state["stats"])
    else:
        stats = load_stats_from_config(config)
    return config, model, stats, device_obj, state


def denormalize_batch(y: torch.Tensor, stats: DatasetStats, enabled: bool) -> torch.Tensor:
    if not enabled:
        return y
    return y * stats.target_std + stats.target_mean


@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device, stats: DatasetStats, normalize_targets: bool, max_batches: Optional[int] = None):
    all_metrics = []
    cached_examples = []
    preds_np = []
    targets_np = []
    metas = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred_den = denormalize_batch(pred, stats, normalize_targets)
        y_den = denormalize_batch(y, stats, normalize_targets)
        all_metrics.append(compute_metrics_torch(pred_den, y_den))
        preds_np.append(pred_den.cpu().numpy())
        targets_np.append(y_den.cpu().numpy())
        if meta is not None:
            metas.append({k: v.cpu().numpy() for k, v in meta.items()})
        if len(cached_examples) < 4:
            cached_examples.append((x.cpu().numpy(), y_den.cpu().numpy(), pred_den.cpu().numpy(), None if meta is None else {k: v.cpu().numpy() for k, v in meta.items()}))

    merged_metrics = average_metric_dicts(all_metrics)
    pred_arr = np.concatenate(preds_np, axis=0) if preds_np else np.empty((0,))
    target_arr = np.concatenate(targets_np, axis=0) if targets_np else np.empty((0,))
    return merged_metrics, pred_arr, target_arr, metas, cached_examples


def benchmark_model(model, loader: DataLoader, device: torch.device, warmup_batches: int = 2, timed_batches: int = 10) -> Dict[str, float]:
    times = []
    model.eval()
    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch[0].to(device)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(timed_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch[0].to(device)
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    arr = np.asarray(times, dtype=np.float64)
    batch_size = int(loader.batch_size or 1)
    return {
        "mean_batch_seconds": float(arr.mean()),
        "std_batch_seconds": float(arr.std()),
        "mean_sample_seconds": float(arr.mean() / batch_size),
        "samples_per_second": float(batch_size / arr.mean()),
    }


def benchmark_solver(config: dict, loader: DataLoader, stats: DatasetStats, n_samples: int = 8) -> Dict[str, float]:
    solver = build_solver(config)
    times = []
    nt = int(config.get("simulation", {}).get("nt", 20))
    count = 0
    for batch in loader:
        x = batch[0].cpu().numpy()
        bathy = []
        disturbance = []
        for sample in x:
            sample = denormalize_inputs(sample, stats)
            bathy.append(sample[0])
            disturbance.append(sample[1])
        bathy = np.asarray(bathy, dtype=np.float32)
        disturbance = np.asarray(disturbance, dtype=np.float32)
        for b in range(bathy.shape[0]):
            start = time.perf_counter()
            _ = solver.simulate(bathy[b], disturbance[b], nt=nt)
            times.append(time.perf_counter() - start)
            count += 1
            if count >= n_samples:
                arr = np.asarray(times, dtype=np.float64)
                return {
                    "mean_sample_seconds": float(arr.mean()),
                    "std_sample_seconds": float(arr.std()),
                    "samples_per_second": float(1.0 / arr.mean()),
                }
    arr = np.asarray(times, dtype=np.float64) if times else np.asarray([np.nan])
    return {
        "mean_sample_seconds": float(arr.mean()),
        "std_sample_seconds": float(arr.std()),
        "samples_per_second": float(1.0 / arr.mean()) if np.isfinite(arr.mean()) else float("nan"),
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
