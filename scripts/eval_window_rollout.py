#!/usr/bin/env python
"""Autoregressive rollout evaluation for the windowed FNO (fno_window5_*).

The windowed model maps [bathymetry, source, eta_t, eta_{t-1}] -> next K eta frames.
This script reconstructs the full eta[1:T] trajectory by rolling the model forward in
chunks of K, seeded from the given first frame y[0], then scores it against the SAME
test targets and metric conventions as scripts/eval_accuracy.py for an apples-to-apples
comparison with the direct FNO.

Outputs (same schema as the direct-FNO evals so it slots into the comparison):
  experiments/<out>/eval/metrics.json   global rel-L2/mae/rmse/max_error (+ _physical)
  experiments/<out>/eval/perframe.json  per-frame rel-L2/mae/rmse over the rollout
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.target_scaling import (
    apply_target_denorm,
    load_target_denorm,
    resolve_eval_dataset_path,
)
from src.evaluation.window_rollout import rollout_trajectory
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.training.metrics import MetricAccumulator
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything

@torch.no_grad()
def _run(
    model,
    test_loader,
    device,
    K,
    include_source,
    use_prev,
    target_denorm,
    cfg,
    eval_cfg,
    args,
    resolved,
):
    glob = MetricAccumulator()
    glob_phys = MetricAccumulator() if target_denorm is not None else None
    pf_sums = None  # per-frame sums, lazily sized to T-1
    n_samples = 0
    t_infer = 0.0
    t_samples = 0

    for batch in test_loader:
        x = batch["x"].to(device)  # [B, 3, H, W]
        y = batch["y"].to(device)  # [B, T, H, W]
        B, T = y.shape[0], y.shape[1]
        y0 = y[:, 0]
        target = y[:, 1:]  # [B, T-1, H, W]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = rollout_trajectory(model, x, y0, T, K, include_source, use_prev, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer += time.perf_counter() - t0
        t_samples += B

        pe = apply_target_denorm(pred, None)
        te = apply_target_denorm(target, None)
        glob.update(pe, te)
        if glob_phys is not None:
            glob_phys.update(
                apply_target_denorm(pred, target_denorm),
                apply_target_denorm(target, target_denorm),
            )

        # per-frame (normalized units), accumulate sums over frames 1..T-1
        if pf_sums is None:
            F = target.shape[1]
            pf_sums = {
                "sq": torch.zeros(F, dtype=torch.float64, device=device),
                "abs": torch.zeros(F, dtype=torch.float64, device=device),
                "tsq": torch.zeros(F, dtype=torch.float64, device=device),
                "n": torch.zeros(F, dtype=torch.float64, device=device),
            }
        diff = (pred - target).to(torch.float64)
        t64 = target.to(torch.float64)
        red = [0] + list(range(2, diff.dim()))
        pf_sums["sq"] += (diff * diff).sum(dim=red)
        pf_sums["abs"] += diff.abs().sum(dim=red)
        pf_sums["tsq"] += (t64 * t64).sum(dim=red)
        pf_sums["n"] += torch.tensor(
            B * diff.shape[2] * diff.shape[3], dtype=torch.float64, device=device
        )
        n_samples += B

    metrics = glob.compute()
    metrics["evaluation_type"] = "conditional_seeded_window_rollout"
    metrics["num_samples"] = float(n_samples)
    if resolved is not None:
        metrics["dataset_path"] = str(resolved)
    metrics["rollout_chunks"] = int(((target.shape[1]) + K - 1) // K)
    metrics["window_K"] = int(K)
    metrics["seeded_with_true_first_frame"] = True
    metrics["num_predicted_frames"] = int(target.shape[1])
    metrics["time_per_sample_rollout_s"] = t_infer / max(1, t_samples)
    metrics["samples_per_second_rollout"] = t_samples / max(t_infer, 1e-12)
    metrics["config_path"] = str(args.config)
    metrics["checkpoint"] = str(args.checkpoint)
    if glob_phys is not None:
        for k, v in glob_phys.compute().items():
            metrics[f"{k}_physical"] = v
        metrics["target_offset"] = float(target_denorm[0])
        metrics["target_scale"] = float(target_denorm[1])

    eps = 1e-8
    rel = (pf_sums["sq"].sqrt() / (pf_sums["tsq"].sqrt() + eps)).cpu().tolist()
    mae_pf = (pf_sums["abs"] / pf_sums["n"].clamp_min(1)).cpu().tolist()
    rmse_pf = (pf_sums["sq"] / pf_sums["n"].clamp_min(1)).sqrt().cpu().tolist()
    perframe = {
        "evaluation_type": "window_rollout_perframe",
        "config_path": args.config,
        "checkpoint": args.checkpoint,
        "dataset_path": str(resolved) if resolved is not None else "",
        "num_samples": float(n_samples),
        "num_frames": len(rel),
        "frame_offset": 1,  # per-frame index f corresponds to eta frame f+1 (frame 0 is the seed)
        "seeded_with_true_first_frame": True,
        "per_frame": {"rel_l2": rel, "mae": mae_pf, "rmse": rmse_pf},
    }

    out_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not out_dir or out_dir == "experiments/eval":
        out_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    metrics_output = (
        Path(args.output) if args.output else Path(out_dir) / "metrics.json"
    )
    perframe_output = (
        Path(args.perframe_output)
        if args.perframe_output
        else Path(out_dir) / "perframe.json"
    )
    save_json(metrics, metrics_output)
    save_json(perframe, perframe_output)
    print(
        f"[rollout] rel_l2={metrics['rel_l2']:.4f} "
        f"last_frame_rel_l2={rel[-1]:.4f} first={rel[0]:.4f} "
        f"t/sample={metrics['time_per_sample_rollout_s'] * 1e3:.3f}ms -> {out_dir}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--perframe-output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))

    data_cfg = dict(cfg.get("data", {}))
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    K = int(data_cfg.get("window_K", 5))
    include_source = bool(data_cfg.get("window_include_source", True))
    use_prev = bool(data_cfg.get("window_prev", True))

    # Load the BASE (non-windowed) test set: we need full trajectories, not windows.
    base_data_cfg = {
        "test_path": data_cfg.get("test_path"),
        "batch_size": data_cfg.get("batch_size", 8),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "windowed": False,
    }
    if "dataset_path" in eval_cfg:
        base_data_cfg["test_path"] = eval_cfg["dataset_path"]
    if "batch_size" in eval_cfg:
        base_data_cfg["batch_size"] = eval_cfg["batch_size"]
    cfg_base = dict(cfg)
    cfg_base["data"] = base_data_cfg
    loaders = create_dataloaders(cfg_base)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError("No test dataloader; set eval.dataset_path or data.test_path.")

    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    resolved = resolve_eval_dataset_path(cfg, split="test")
    target_denorm = None
    if bool(eval_cfg.get("report_physical_metrics", True)) and resolved is not None:
        try:
            target_denorm = load_target_denorm(resolved)
        except Exception:
            target_denorm = None

    _run(
        model,
        test_loader,
        device,
        K,
        include_source,
        use_prev,
        target_denorm,
        cfg,
        eval_cfg,
        args,
        resolved,
    )


if __name__ == "__main__":
    main()
