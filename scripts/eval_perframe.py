#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.utils.seed import seed_everything


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _empty_sums(num_frames: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "sum_abs_err": torch.zeros(num_frames, dtype=torch.float64, device=device),
        "sum_sq_err": torch.zeros(num_frames, dtype=torch.float64, device=device),
        "sum_target_sq": torch.zeros(num_frames, dtype=torch.float64, device=device),
        "n_elements": torch.zeros(num_frames, dtype=torch.float64, device=device),
    }


def _update_sums(
    sums: Dict[str, torch.Tensor], pred: torch.Tensor, target: torch.Tensor
) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim < 3:
        raise ValueError(
            f"Expected batched trajectory tensor with frame/channel dim at axis 1, got {tuple(pred.shape)}"
        )

    diff = (pred - target).to(torch.float64)
    target64 = target.to(torch.float64)
    reduce_dims = tuple(i for i in range(diff.ndim) if i != 1)
    per_frame_elements = pred.numel() // int(pred.shape[1])

    sums["sum_abs_err"] += torch.abs(diff).sum(dim=reduce_dims)
    sums["sum_sq_err"] += (diff * diff).sum(dim=reduce_dims)
    sums["sum_target_sq"] += (target64 * target64).sum(dim=reduce_dims)
    sums["n_elements"] += float(per_frame_elements)


def _compute_curve(
    sums: Dict[str, torch.Tensor], eps: float = 1e-8
) -> list[Dict[str, float]]:
    sum_abs = sums["sum_abs_err"].cpu()
    sum_sq = sums["sum_sq_err"].cpu()
    target_sq = sums["sum_target_sq"].cpu()
    n_elements = torch.clamp(sums["n_elements"].cpu(), min=1.0)

    rows: list[Dict[str, float]] = []
    for i in range(int(sum_sq.numel())):
        rows.append(
            {
                "frame": int(i),
                "mae": float(sum_abs[i] / n_elements[i]),
                "rmse": float(torch.sqrt(sum_sq[i] / n_elements[i])),
                "rel_l2": float(
                    torch.sqrt(sum_sq[i]) / (torch.sqrt(target_sq[i]) + eps)
                ),
            }
        )
    return rows


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate dataset-level per-frame error curves."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))

    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    data_cfg = dict(cfg.get("data", {}))
    dataset_cfg = cfg.get("dataset", {})
    if not data_cfg and isinstance(dataset_cfg, dict):
        dataset_path = dataset_cfg.get("path")
        if dataset_path:
            data_cfg["test_path"] = dataset_path
        if "batch_size" in dataset_cfg:
            data_cfg["batch_size"] = dataset_cfg["batch_size"]

    dataset_path = eval_cfg.get("dataset_path")
    if dataset_path:
        data_cfg["test_path"] = dataset_path
    if "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]
    cfg["data"] = data_cfg

    device = resolve_device(cfg.get("device", "auto"))
    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError(
            "No test dataloader found. Set eval.dataset_path or data.test_path."
        )
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))

    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    resolved_dataset_path = resolve_eval_dataset_path(cfg, split="test")
    target_denorm = None
    if (
        bool(eval_cfg.get("report_physical_metrics", True))
        and resolved_dataset_path is not None
    ):
        try:
            target_denorm = load_target_denorm(resolved_dataset_path)
        except Exception:
            target_denorm = None

    sums: Dict[str, torch.Tensor] | None = None
    physical_sums: Dict[str, torch.Tensor] | None = None

    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = _model_output(model, x)
        if sums is None:
            sums = _empty_sums(int(y.shape[1]), device)
        _update_sums(sums, pred, y)

        if target_denorm is not None:
            if physical_sums is None:
                physical_sums = _empty_sums(int(y.shape[1]), device)
            _update_sums(
                physical_sums,
                apply_target_denorm(pred, target_denorm),
                apply_target_denorm(y, target_denorm),
            )

    if sums is None:
        raise RuntimeError("No batches were evaluated.")

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    output_path = (
        Path(args.output) if args.output else Path(output_dir) / "perframe.json"
    )

    out: Dict[str, Any] = {
        "evaluation_type": "per_frame_error",
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(resolved_dataset_path)
        if resolved_dataset_path is not None
        else "",
        "num_samples": int(_dataset_num_samples(test_loader)),
        "num_frames": int(sums["sum_sq_err"].numel()),
        "per_frame": _compute_curve(sums),
    }
    if physical_sums is not None and target_denorm is not None:
        out["target_offset"] = float(target_denorm[0])
        out["target_scale"] = float(target_denorm[1])
        out["per_frame_physical"] = _compute_curve(physical_sums)

    print(out)
    save_json(out, output_path)


if __name__ == "__main__":
    main()
