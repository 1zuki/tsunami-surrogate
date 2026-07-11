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
from src.utils.seed import seed_everything
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.normalization_bridge import (
    load_evaluation_normalization_bridge,
)
from src.evaluation.target_scaling import load_target_denorm, resolve_eval_dataset_path


def _dataset_num_samples(loader) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

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

    resolved_dataset_path = resolve_eval_dataset_path(cfg, split="test")
    bridge_cfg = eval_cfg.get("normalization_bridge")
    normalization_bridge = None
    if bridge_cfg is not None:
        if not isinstance(bridge_cfg, dict):
            raise TypeError("eval.normalization_bridge must be a mapping")
        if resolved_dataset_path is None:
            raise ValueError(
                "eval.normalization_bridge requires a resolvable evaluation dataset"
            )
        dataset_stats_path = bridge_cfg.get("dataset_stats_path")
        model_stats_path = bridge_cfg.get("model_stats_path")
        if not dataset_stats_path or not model_stats_path:
            raise KeyError(
                "eval.normalization_bridge requires dataset_stats_path and "
                "model_stats_path"
            )
        normalization_bridge = load_evaluation_normalization_bridge(
            dataset_path=resolved_dataset_path,
            dataset_stats_path=dataset_stats_path,
            model_stats_path=model_stats_path,
        )

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
    batch_transform = (
        normalization_bridge.transform if normalization_bridge is not None else None
    )
    metrics = evaluate_accuracy(
        model, test_loader, device, batch_transform=batch_transform
    )
    metrics["num_samples"] = float(_dataset_num_samples(test_loader))

    if resolved_dataset_path is not None:
        metrics["dataset_path"] = str(resolved_dataset_path)
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))
    target_denorm = None

    if normalization_bridge is not None:
        target_denorm = normalization_bridge.model_target_denorm
    elif report_physical and resolved_dataset_path is not None:
        try:
            target_denorm = load_target_denorm(resolved_dataset_path)
        except Exception:
            target_denorm = None

    if report_physical and target_denorm is not None:
        physical_metrics = evaluate_accuracy(
            model,
            test_loader,
            device,
            target_denorm=target_denorm,
            batch_transform=batch_transform,
        )
        metrics.update({f"{k}_physical": v for k, v in physical_metrics.items()})
        metrics["target_offset"] = float(target_denorm[0])
        metrics["target_scale"] = float(target_denorm[1])
    if normalization_bridge is not None:
        metrics["normalization_bridge"] = normalization_bridge.metadata()

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"

    print(metrics)
    save_json(metrics, f"{output_dir}/metrics.json")


if __name__ == "__main__":
    main()
