#!/usr/bin/env python
"""Evaluate a Window-FNO rollout checkpoint on multiple full-trajectory suites."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_window_rollout import rollout_trajectory
from src.data.dataset import create_dataloaders
from src.evaluation.target_scaling import load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.training.metrics import MetricAccumulator
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything


def _suite_list(eval_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> list[dict[str, Any]]:
    direct = eval_cfg.get("window_suites")
    if direct:
        return list(direct)
    for key in ("generalization", "real_resolution", "cross_resolution"):
        block = eval_cfg.get(key, cfg.get(key, {}))
        if isinstance(block, dict) and block.get("suites"):
            return list(block.get("suites", []))
    return []


def _suite_loader(cfg: Dict[str, Any], dataset_path: str, batch_size: int):
    local_cfg = dict(cfg)
    data_cfg = {
        "windowed": False,
        "test_path": dataset_path,
        "batch_size": int(batch_size),
        "num_workers": 0,
    }
    local_cfg["data"] = data_cfg
    loaders = create_dataloaders(local_cfg)
    if "test" not in loaders:
        raise KeyError(f"No test loader for {dataset_path}")

    return loaders["test"]


@torch.no_grad()
def _evaluate_suite(
    model, loader, device, K: int, include_source: bool, use_prev: bool, target_denorm
):
    glob = MetricAccumulator()
    glob_phys = MetricAccumulator() if target_denorm is not None else None
    pf_sq = None
    pf_tsq = None
    n_samples = 0
    t_infer = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        B, T = y.shape[0], y.shape[1]
        target = y[:, 1:]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = rollout_trajectory(
            model, x, y[:, 0], T, K, include_source, use_prev, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer += time.perf_counter() - t0

        glob.update(pred, target)
        if glob_phys is not None:
            offset, scale = target_denorm
            glob_phys.update(
                pred * float(scale) + float(offset),
                target * float(scale) + float(offset),
            )

        diff = (pred - target).to(torch.float64)
        tgt = target.to(torch.float64)
        red = [0] + list(range(2, diff.dim()))
        if pf_sq is None:
            pf_sq = torch.zeros(target.shape[1], dtype=torch.float64, device=device)
            pf_tsq = torch.zeros(target.shape[1], dtype=torch.float64, device=device)
        pf_sq += (diff * diff).sum(dim=red)
        pf_tsq += (tgt * tgt).sum(dim=red)
        n_samples += B

    metrics = {k: float(v) for k, v in glob.compute().items()}
    if glob_phys is not None:
        metrics.update(
            {f"{k}_physical": float(v) for k, v in glob_phys.compute().items()}
        )
        metrics["target_offset"] = float(target_denorm[0])
        metrics["target_scale"] = float(target_denorm[1])
    rel = (pf_sq.sqrt() / (pf_tsq.sqrt() + 1e-8)).detach().cpu().tolist()
    mid = len(rel) // 2
    metrics.update(
        {
            "num_samples": int(n_samples),
            "num_predicted_frames": int(len(rel)),
            "first_frame_rel_l2": float(rel[0]) if rel else None,
            "mid_frame_rel_l2": float(rel[mid]) if rel else None,
            "final_frame_rel_l2": float(rel[-1]) if rel else None,
            "time_per_sample_rollout_s": float(t_infer / max(1, n_samples)),
            "samples_per_second_rollout": float(n_samples / max(t_infer, 1e-12)),
        }
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    suites = _suite_list(eval_cfg, cfg)
    if not suites:
        raise ValueError(
            "No suites configured. Set eval.window_suites or eval.{generalization,real_resolution,cross_resolution}.suites."
        )

    data_cfg = dict(cfg.get("data", {}))
    K = int(data_cfg.get("window_K", 5))
    include_source = bool(data_cfg.get("window_include_source", True))
    use_prev = bool(data_cfg.get("window_prev", True))
    device = resolve_device(cfg.get("device", "auto"))

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    for i, suite in enumerate(suites):
        suite_cfg = suite if isinstance(suite, dict) else {}
        label = str(suite_cfg.get("label", f"suite_{i}"))
        path = str(suite_cfg.get("path", "")).strip()
        if not path:
            raise KeyError(f"suite {label!r} has no path")
        batch_size = int(
            suite_cfg.get(
                "batch_size", eval_cfg.get("batch_size", data_cfg.get("batch_size", 8))
            )
        )
        loader = _suite_loader(cfg, path, batch_size)
        try:
            denorm = (
                load_target_denorm(path)
                if bool(eval_cfg.get("report_physical_metrics", True))
                else None
            )
        except Exception:
            denorm = None
        metrics = _evaluate_suite(
            model, loader, device, K, include_source, use_prev, denorm
        )
        rows.append({"label": label, "dataset_path": path, **metrics})
        print(
            f"[window-suite] {label}: rel_l2={metrics['rel_l2']:.4f} final={metrics['final_frame_rel_l2']:.4f}"
        )

    output_dir = (
        str(eval_cfg.get("output_dir", "")).strip()
        or f"{cfg.get('output_dir', 'experiments/default')}/eval"
    )
    summary = {
        "evaluation_type": "window_rollout_suites",
        "checkpoint": args.checkpoint,
        "window_K": int(K),
        "seeded_with_true_first_frame": True,
        "rows": rows,
    }
    save_json(summary, f"{output_dir}/window_rollout_suites.json")


if __name__ == "__main__":
    main()
