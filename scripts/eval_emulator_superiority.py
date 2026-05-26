#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.target_scaling import load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _solver_metric_mean(payload: Dict[str, Any], metric: str) -> float:
    agg = payload.get("aggregate_metrics", {})
    if metric not in agg:
        raise KeyError(f"Metric '{metric}' not found in solver comparison aggregate_metrics")
    row = agg[metric]
    if not isinstance(row, dict) or "mean" not in row:
        raise KeyError(f"Metric '{metric}' in solver comparison has no 'mean'")
    return float(row["mean"])


def main() -> None:
    p = argparse.ArgumentParser(description="Compute emulator-superiority ratio against solver-vs-solver error.")
    p.add_argument("--config", required=True, help="YAML config for ratio evaluation")
    args = p.parse_args()

    cfg = load_config(args.config)
    model_cfg_path = str(cfg.get("model_config", "")).strip()
    checkpoint_path = str(cfg.get("checkpoint", "")).strip()
    solver_compare_path = Path(str(cfg.get("solver_compare_path", "")).strip())
    eval_cfg = dict(cfg.get("evaluation", {}))
    ratio_cfg = dict(cfg.get("ratio", {}))

    if not model_cfg_path:
        raise KeyError("config requires model_config")
    if not checkpoint_path:
        raise KeyError("config requires checkpoint")
    if not solver_compare_path:
        raise KeyError("config requires solver_compare_path")
    if not solver_compare_path.exists():
        raise FileNotFoundError(solver_compare_path)

    dataset_path = str(eval_cfg.get("dataset_path", "")).strip()
    if not dataset_path:
        raise KeyError("config requires evaluation.dataset_path")
    
    batch_size = int(eval_cfg.get("batch_size", 8))
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))

    numerator_metric = str(ratio_cfg.get("numerator_metric", "rmse_physical"))
    denominator_metric = str(ratio_cfg.get("denominator_metric", "rmse"))
    output_path = Path(str(cfg.get("output_path", "results/emulator_superiority.json")))

    model_cfg = load_config(model_cfg_path)
    data_cfg = dict(model_cfg.get("data", {}))
    data_cfg["test_path"] = dataset_path
    data_cfg["batch_size"] = batch_size
    model_cfg["data"] = data_cfg

    device = resolve_device(model_cfg.get("device", "auto"))
    loaders = create_dataloaders(model_cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError("Could not create test loader for evaluation.dataset_path")

    validate_model_io_channels(model_cfg, loaders, preferred_splits=("test",))

    model = build_model(model_cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    metrics = evaluate_accuracy(model, test_loader, device)
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["num_samples"] = float(_dataset_num_samples(test_loader))
    metrics["dataset_path"] = str(dataset_path)

    denorm = load_target_denorm(dataset_path) if report_physical else None
    if denorm is not None:
        phys = evaluate_accuracy(model, test_loader, device, target_denorm=denorm)
        for k, v in phys.items():
            metrics[f"{k}_physical"] = float(v)

        metrics["target_offset"] = float(denorm[0])
        metrics["target_scale"] = float(denorm[1])

    with solver_compare_path.open("r", encoding="utf-8") as f:
        solver_payload = json.load(f)

    solver_mean = _solver_metric_mean(solver_payload, denominator_metric)

    if numerator_metric not in metrics:
        raise KeyError(
            f"numerator_metric '{numerator_metric}' not found in model metrics. "
            f"Available={sorted(metrics.keys())}"
        )
    numerator = float(metrics[numerator_metric])
    ratio = float(numerator / solver_mean) if abs(solver_mean) > 0 else float("inf")

    out = {
        "evaluation_type": "emulator_superiority_ratio",
        "model_config": model_cfg_path,
        "checkpoint": checkpoint_path,
        "model_metrics": metrics,
        "solver_compare_path": str(solver_compare_path),
        "solver_denominator_metric": denominator_metric,
        "solver_denominator_mean": solver_mean,
        "emulator_numerator_metric": numerator_metric,
        "emulator_numerator_value": numerator,
        "ratio": ratio,
        "interpretation": "ratio < 1 means emulator error is lower than solver-A vs solver-B disagreement.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, output_path)
    print(out)


if __name__ == "__main__":
    main()
