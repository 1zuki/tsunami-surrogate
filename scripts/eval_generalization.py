#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from typing import Any, Dict
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.generalization_suite import evaluate_by_regime
from src.evaluation.target_scaling import load_target_denorm, resolve_eval_dataset_path
from src.utils.io import save_json


def _build_test_loader(cfg: Dict[str, Any], test_path: str, batch_size: int):
    local_cfg = dict(cfg)
    local_data = dict(local_cfg.get("data", {}))
    local_data["test_path"] = test_path
    local_data["batch_size"] = batch_size
    local_cfg["data"] = local_data
    loaders = create_dataloaders(local_cfg)
    test_loader = loaders.get("test")

    if test_loader is None:
        raise KeyError(f"No test dataloader could be built for suite path: {test_path}")
    validate_model_io_channels(local_cfg, loaders, preferred_splits=("test",))
    
    return test_loader


def _attach_physical_metrics(
    base: Dict[str, Dict[str, float]],
    physical: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in base.items()}
    for label, row in physical.items():
        if label not in out:
            out[label] = {}
        for key, value in row.items():
            if key == "n":
                out[label].setdefault("n", float(value))
            else:
                out[label][f"{key}_physical"] = float(value)
    return out


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
    
    if eval_cfg.get("dataset_path"):
        data_cfg["test_path"] = eval_cfg["dataset_path"]
    if "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]
    
    cfg["data"] = data_cfg

    device = resolve_device(cfg.get('device', 'auto'))
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    generalization_cfg = eval_cfg.get("generalization", cfg.get("generalization", {}))
    group_key_default = generalization_cfg.get("group_key", "source_id")
    suites = list(generalization_cfg.get("suites", []))
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))

    if suites:
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for i, suite in enumerate(suites):
            suite_cfg = suite if isinstance(suite, dict) else {}
            label = str(suite_cfg.get("label", f"suite_{i}"))
            suite_path = suite_cfg.get("path")
            if not suite_path:
                raise KeyError(f"generalization.suites[{i}] is missing required key: path")
            batch_size = int(
                suite_cfg.get(
                    "batch_size",
                    eval_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 8)),
                )
            )
            key = str(suite_cfg.get("group_key", group_key_default))
            test_loader = _build_test_loader(cfg, str(suite_path), batch_size)

            suite_result = evaluate_by_regime(model, test_loader, device, key=key)
            if report_physical:
                denorm = load_target_denorm(str(suite_path))
                if denorm is not None:
                    suite_physical = evaluate_by_regime(model, test_loader, device, key=key, target_denorm=denorm)
                    suite_result = _attach_physical_metrics(suite_result, suite_physical)

            result[label] = suite_result
    else:
        loaders = create_dataloaders(cfg)
        test_loader = loaders.get("test")
        if test_loader is None:
            raise KeyError("No test dataloader found. Set eval.dataset_path or data.test_path.")
        validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))

        result = evaluate_by_regime(model, test_loader, device, key=group_key_default)
        if report_physical:
            resolved_dataset_path = resolve_eval_dataset_path(cfg, split="test")
            if resolved_dataset_path is not None:
                denorm = load_target_denorm(resolved_dataset_path)
                if denorm is not None:
                    physical = evaluate_by_regime(model, test_loader, device, key=group_key_default, target_denorm=denorm)
                    result = _attach_physical_metrics(result, physical)

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    print(result)
    save_json(result, f"{output_dir}/ood_by_source.json")


if __name__ == '__main__':
    main()
