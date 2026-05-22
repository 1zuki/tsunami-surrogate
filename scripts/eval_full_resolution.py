#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

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


def _suite_test_loader(cfg: Dict[str, Any], dataset_path: str, batch_size: int):
    local_cfg = dict(cfg)
    local_data = dict(local_cfg.get("data", {}))
    local_data["test_path"] = dataset_path
    local_data["batch_size"] = batch_size
    local_cfg["data"] = local_data

    loaders = create_dataloaders(local_cfg)
    test_loader = loaders.get("test")

    if test_loader is None:
        raise KeyError(f"No test dataloader could be created for suite dataset_path: {dataset_path}")
    
    validate_model_io_channels(local_cfg, loaders, preferred_splits=("test",))
    return test_loader


def _dataset_size_from_loader(loader) -> int:
    ds = getattr(loader, "dataset", None)
    
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a single trained model across multiple real-resolution processed datasets."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    rr_cfg = eval_cfg.get("real_resolution", cfg.get("real_resolution", {}))
    suites = list(rr_cfg.get("suites", []))
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))

    if not suites:
        raise ValueError(
            "No real-resolution suites configured. "
            "Add `eval.real_resolution.suites` with `label` + `path` entries."
        )

    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    result_rows: List[Dict[str, Any]] = []

    for i, suite in enumerate(suites):
        suite_cfg = suite if isinstance(suite, dict) else {}
        label = str(suite_cfg.get("label", f"suite_{i}")).strip()
        dataset_path = str(suite_cfg.get("path", "")).strip()

        if not dataset_path:
            raise KeyError(f"real_resolution.suites[{i}] is missing required key: path")
        
        batch_size = int(
            suite_cfg.get("batch_size", eval_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 8)))
        )

        loader = _suite_test_loader(cfg, dataset_path=dataset_path, batch_size=batch_size)
        metrics = evaluate_accuracy(model, loader, device)
        row: Dict[str, Any] = {
            "label": label,
            "dataset_path": dataset_path,
            "num_samples": _dataset_size_from_loader(loader),
            **{k: float(v) for k, v in metrics.items()},
        }

        if report_physical:
            denorm = load_target_denorm(Path(dataset_path))
            if denorm is not None:
                metrics_phys = evaluate_accuracy(model, loader, device, target_denorm=denorm)
                row.update({f"{k}_physical": float(v) for k, v in metrics_phys.items()})
                row["target_offset"] = float(denorm[0])
                row["target_scale"] = float(denorm[1])

        result_rows.append(row)

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"

    summary: Dict[str, Any] = {"rows": result_rows}
    print(summary)
    save_json(summary, f"{output_dir}/real_resolution.json")


if __name__ == "__main__":
    main()
