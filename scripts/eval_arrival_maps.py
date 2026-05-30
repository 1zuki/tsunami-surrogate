#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.target_scaling import load_target_denorm, resolve_eval_dataset_path
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _first_arrival_index(abs_eta: np.ndarray, threshold_abs: float) -> np.ndarray:
    crossed = abs_eta >= float(threshold_abs)
    has_cross = crossed.any(axis=0)
    first = np.argmax(crossed, axis=0).astype(np.int32)
    first[~has_cross] = -1
    return first


def _ensure_time_grid(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x[None, ...]
    raise ValueError(f"Expected [T,H,W] or [H,W], got {x.shape}")


def _accumulator(h: int, w: int) -> Dict[str, np.ndarray]:
    return {
        "count_samples": np.zeros((h, w), dtype=np.float64),
        "count_valid_pair": np.zeros((h, w), dtype=np.float64),
        "count_valid_pred": np.zeros((h, w), dtype=np.float64),
        "count_valid_target": np.zeros((h, w), dtype=np.float64),
        "sum_first_pred": np.zeros((h, w), dtype=np.float64),
        "sum_first_target": np.zeros((h, w), dtype=np.float64),
        "sum_abs_diff_steps": np.zeros((h, w), dtype=np.float64),
    }


def _update_acc(acc: Dict[str, np.ndarray], pred: np.ndarray, target: np.ndarray, threshold_fraction: float) -> Tuple[float, np.ndarray]:
    pred_t = _ensure_time_grid(pred)
    tgt_t = _ensure_time_grid(target)
    t = min(pred_t.shape[0], tgt_t.shape[0])
    pred_t = pred_t[:t]
    tgt_t = tgt_t[:t]

    shared_peak = float(max(np.max(np.abs(pred_t)), np.max(np.abs(tgt_t)), 0.0))
    thr = float(threshold_fraction) * shared_peak

    first_pred = _first_arrival_index(np.abs(pred_t), threshold_abs=thr)
    first_tgt = _first_arrival_index(np.abs(tgt_t), threshold_abs=thr)
    valid = (first_pred >= 0) & (first_tgt >= 0)
    valid_pred = first_pred >= 0
    valid_tgt = first_tgt >= 0
    diff = np.full(first_pred.shape, np.nan, dtype=np.float64)
    diff[valid] = np.abs(first_pred[valid] - first_tgt[valid]).astype(np.float64)

    acc["count_samples"] += 1.0
    acc["count_valid_pair"] += valid.astype(np.float64)
    acc["count_valid_pred"] += valid_pred.astype(np.float64)
    acc["count_valid_target"] += valid_tgt.astype(np.float64)
    acc["sum_first_pred"] += np.where(valid_pred, first_pred, 0.0)
    acc["sum_first_target"] += np.where(valid_tgt, first_tgt, 0.0)
    acc["sum_abs_diff_steps"] += np.where(valid, diff, 0.0)

    valid_vals = diff[np.isfinite(diff)]
    sample_mean = float(np.mean(valid_vals)) if valid_vals.size > 0 else float("nan")
    return sample_mean, diff


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate model-vs-target arrival-time maps on a processed test split.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--arrival-threshold-fraction", type=float, default=0.05)
    p.add_argument("--output", type=str, default=None, help="Optional summary json path.")
    p.add_argument("--maps-output", type=str, default=None, help="Optional arrival maps npz path.")
    args = p.parse_args()

    if args.arrival_threshold_fraction < 0.0:
        raise ValueError("--arrival-threshold-fraction must be >= 0")

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    data_cfg = dict(cfg.get("data", {}))
    if eval_cfg.get("dataset_path"):
        data_cfg["test_path"] = eval_cfg["dataset_path"]
    if "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]
    cfg["data"] = data_cfg
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))

    device = resolve_device(cfg.get("device", "auto"))
    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError("No test dataloader found. Set eval.dataset_path or data.test_path.")
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))

    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    target_denorm = None
    if report_physical:
        resolved_path = resolve_eval_dataset_path(cfg, split="test")
        if resolved_path is not None:
            target_denorm = load_target_denorm(resolved_path)

    acc = None
    sample_mean_diffs: list[float] = []
    sample_max_diffs: list[float] = []
    total_samples = 0

    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = _model_output(model, x)
        if target_denorm is not None:
            offset, scale = float(target_denorm[0]), float(target_denorm[1])
            pred = pred * scale + offset
            y = y * scale + offset

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        bs = int(pred_np.shape[0])
        total_samples += bs

        for i in range(bs):
            p_i = np.asarray(pred_np[i], dtype=np.float64)
            y_i = np.asarray(y_np[i], dtype=np.float64)
            p_i = _ensure_time_grid(p_i)
            y_i = _ensure_time_grid(y_i)
            if p_i.shape[1:] != y_i.shape[1:]:
                continue
            if acc is None:
                h, w = p_i.shape[1], p_i.shape[2]
                acc = _accumulator(h, w)
            assert acc is not None
            sample_mean, diff = _update_acc(acc, p_i, y_i, threshold_fraction=float(args.arrival_threshold_fraction))
            sample_mean_diffs.append(sample_mean)
            finite = diff[np.isfinite(diff)]
            sample_max_diffs.append(float(np.max(finite)) if finite.size > 0 else float("nan"))

    if acc is None:
        raise RuntimeError("Could not compare any model-vs-target samples for arrival maps.")

    cnt_samples = np.maximum(acc["count_samples"], 1.0)
    cnt_valid_pair = np.maximum(acc["count_valid_pair"], 1.0)
    cnt_valid_pred = np.maximum(acc["count_valid_pred"], 1.0)
    cnt_valid_target = np.maximum(acc["count_valid_target"], 1.0)

    coverage_map = acc["count_valid_pair"] / cnt_samples
    mean_pred_arrival = np.where(acc["count_valid_pred"] > 0.0, acc["sum_first_pred"] / cnt_valid_pred, -1.0)
    mean_target_arrival = np.where(acc["count_valid_target"] > 0.0, acc["sum_first_target"] / cnt_valid_target, -1.0)
    mean_abs_diff = np.where(acc["count_valid_pair"] > 0.0, acc["sum_abs_diff_steps"] / cnt_valid_pair, np.nan)

    eval_output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not eval_output_dir or eval_output_dir == "experiments/eval":
        eval_output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"

    summary_path = Path(args.output) if args.output else Path(eval_output_dir) / "arrival_map_model_vs_target.json"
    maps_path = Path(args.maps_output) if args.maps_output else Path(eval_output_dir) / "arrival_map_model_vs_target.npz"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        maps_path,
        coverage_map=coverage_map.astype(np.float32),
        mean_arrival_step_pred=mean_pred_arrival.astype(np.float32),
        mean_arrival_step_target=mean_target_arrival.astype(np.float32),
        mean_abs_diff_steps=mean_abs_diff.astype(np.float32),
        valid_pair_count=acc["count_valid_pair"].astype(np.float32),
        sample_count=acc["count_samples"].astype(np.float32),
    )

    flat_valid = mean_abs_diff[np.isfinite(mean_abs_diff)]
    summary: Dict[str, Any] = {
        "evaluation_type": "arrival_map_model_vs_target",
        "arrival_threshold_fraction": float(args.arrival_threshold_fraction),
        "target_units": "physical" if target_denorm is not None else "normalized",
        "target_offset": float(target_denorm[0]) if target_denorm is not None else None,
        "target_scale": float(target_denorm[1]) if target_denorm is not None else None,
        "num_samples_seen": int(total_samples),
        "arrival_map_shape": [int(coverage_map.shape[0]), int(coverage_map.shape[1])],
        "arrival_valid_fraction_mean": float(np.mean(coverage_map)),
        "arrival_mean_abs_diff_steps_map_mean": float(np.mean(flat_valid)) if flat_valid.size > 0 else float("nan"),
        "arrival_mean_abs_diff_steps_map_p90": float(np.percentile(flat_valid, 90)) if flat_valid.size > 0 else float("nan"),
        "arrival_mean_abs_diff_steps_map_max": float(np.max(flat_valid)) if flat_valid.size > 0 else float("nan"),
        "sample_mean_abs_diff_steps": float(np.nanmean(np.asarray(sample_mean_diffs, dtype=np.float64))),
        "sample_max_abs_diff_steps": float(np.nanmax(np.asarray(sample_max_diffs, dtype=np.float64))),
        "maps_path": str(maps_path),
    }
    save_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
