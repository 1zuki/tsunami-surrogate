#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.target_scaling import load_target_denorm, resolve_eval_dataset_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)

    args = p.parse_args()
    cfg = load_config(args.config)

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
            "No test dataloader found. Set `eval.dataset_path` (preferred) or `data.test_path` "
            "to a valid evaluation dataset."
        )
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate_accuracy(model, test_loader, device)

    resolved_dataset_path = resolve_eval_dataset_path(cfg, split="test")
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))
    target_denorm = None

    if report_physical and resolved_dataset_path is not None:
        try:
            target_denorm = load_target_denorm(resolved_dataset_path)
        except Exception:
            target_denorm = None

    if target_denorm is not None:
        physical_metrics = evaluate_accuracy(model, test_loader, device, target_denorm=target_denorm)
        metrics.update({f"{k}_physical": v for k, v in physical_metrics.items()})
        metrics["target_offset"] = float(target_denorm[0])
        metrics["target_scale"] = float(target_denorm[1])

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"

    print(metrics)
    save_json(metrics, f"{output_dir}/metrics.json")


if __name__ == "__main__":
    main()
