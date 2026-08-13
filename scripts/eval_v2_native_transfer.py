#!/usr/bin/env python
"""Evaluate the completed native MUSCL-HR models across v2 grid resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

from torch.utils.data import DataLoader
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import _make_dataset
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.normalization_bridge import (
    load_evaluation_normalization_bridge,
)
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything


def _load_specs(contract_path: str) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation suite contract must be a mapping")
    paper = payload.get("paper_evidence")
    native = paper.get("native_transfer") if isinstance(paper, Mapping) else None
    if not isinstance(native, Mapping):
        raise ValueError("Evaluation suite has no native-transfer contract")
    grids = [int(value) for value in native.get("grids", [])]
    configs = [str(value) for value in native.get("configs", [])]
    checkpoints = [str(value) for value in native.get("checkpoints", [])]
    datasets = [str(value) for value in native.get("datasets", [])]
    if grids != [32, 64, 128] or not (
        len(grids) == len(configs) == len(checkpoints) == len(datasets)
    ):
        raise ValueError("Native-transfer contract must define grids 32/64/128")
    return [
        {
            "grid": grid,
            "config": config,
            "checkpoint": checkpoint,
            "dataset": dataset,
            "stats": str(Path(dataset).parent / "normalization_stats.json"),
        }
        for grid, config, checkpoint, dataset in zip(
            grids,
            configs,
            checkpoints,
            datasets,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/eval/final_v2_suite.yaml",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(18)
    device = resolve_device(args.device)
    specs = _load_specs(args.contract)
    rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, float]] = {}

    for train_spec in specs:
        train_grid = int(train_spec["grid"])
        config_path = str(train_spec["config"])
        checkpoint_path = str(train_spec["checkpoint"])
        cfg = load_config(config_path)
        cfg["device"] = args.device
        model = build_model(cfg).to(device).eval()
        load_checkpoint(checkpoint_path, model, map_location=device)
        matrix[str(train_grid)] = {}

        for test_spec in specs:
            test_grid = int(test_spec["grid"])
            dataset_path = str(test_spec["dataset"])
            dataset = _make_dataset(dataset_path)
            loader = DataLoader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=0,
            )
            bridge = load_evaluation_normalization_bridge(
                dataset_path=dataset_path,
                dataset_stats_path=str(test_spec["stats"]),
                model_stats_path=str(train_spec["stats"]),
            )
            normalized = evaluate_accuracy(
                model,
                loader,
                device,
                batch_transform=bridge.transform,
            )
            physical = evaluate_accuracy(
                model,
                loader,
                device,
                target_denorm=bridge.model_target_denorm,
                batch_transform=bridge.transform,
            )
            row = {
                "train_grid": int(train_grid),
                "test_grid": int(test_grid),
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "dataset_path": dataset_path,
                "num_samples": int(len(dataset)),
                "metrics": normalized,
                "metrics_physical": physical,
                "normalization_bridge": bridge.metadata(),
            }
            rows.append(row)
            matrix[str(train_grid)][str(test_grid)] = float(physical["rel_l2"])

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "evaluation_type": "v2_native_resolution_transfer_matrix",
        "contract_path": str(args.contract),
        "reference": "muscl_hr",
        "grids": [int(spec["grid"]) for spec in specs],
        "configs": [str(spec["config"]) for spec in specs],
        "checkpoints": [str(spec["checkpoint"]) for spec in specs],
        "dataset_paths": [str(spec["dataset"]) for spec in specs],
        "matrix_rel_l2_physical": matrix,
        "rows": rows,
        "normalization_policy": (
            "rebase each evaluation dataset into the checkpoint training "
            "normalization, then compare in physical benchmark-scale units"
        ),
    }
    save_json(result, args.output)
    print(f"[v2-native-transfer] rows={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
