#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from src.utils.seed import seed_everything


GRIDS = (32, 64, 128)


def _config_path(grid: int) -> Path:
    return ROOT / "configs" / "model" / f"fno_res{grid}_hydrostatic.yaml"


def _checkpoint_path(grid: int) -> Path:
    return ROOT / "experiments" / f"fno_res{grid}_hydrostatic" / "best.pt"


def _dataset_path(grid: int) -> Path:
    return ROOT / "data" / "processed_crossres" / f"res{grid}" / "test"


def _dataset_num_samples(loader) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _loader_for(cfg: Dict[str, Any], dataset_path: Path, batch_size: int):
    local_cfg = dict(cfg)
    local_cfg["data"] = {
        "test_path": str(dataset_path),
        "batch_size": int(batch_size),
        "num_workers": 0,
    }
    loaders = create_dataloaders(local_cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError(
            f"No test dataloader for native-resolution dataset: {dataset_path}"
        )
    validate_model_io_channels(local_cfg, loaders, preferred_splits=("test",))
    return test_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the existing hydrostatic FNO native-resolution checkpoints on the 32/64/128 suites."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output",
        default="results/native_resolution_transfer_matrix_fno_hydrostatic.json",
    )
    args = parser.parse_args()

    seed_everything(42)
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, float]] = {}
    diagonal: list[dict[str, Any]] = []

    for train_grid in GRIDS:
        cfg_path = _config_path(train_grid)
        ckpt_path = _checkpoint_path(train_grid)
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

        cfg = load_config(str(cfg_path))
        cfg["device"] = args.device
        seed_everything(int(cfg.get("seed", 42)))
        model = build_model(cfg).to(device)
        load_checkpoint(str(ckpt_path), model, map_location=device)

        eval_cfg = cfg.get("eval", {})
        batch_size = int(
            eval_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 64))
        )
        train_key = f"res{train_grid}"
        matrix[train_key] = {}

        for test_grid in GRIDS:
            dataset_path = _dataset_path(test_grid)
            if not dataset_path.is_dir():
                raise FileNotFoundError(dataset_path)

            loader = _loader_for(cfg, dataset_path, batch_size=batch_size)
            target_denorm = load_target_denorm(dataset_path)
            metrics = evaluate_accuracy(model, loader, device)
            row: dict[str, Any] = {
                "train_grid": int(train_grid),
                "test_grid": int(test_grid),
                "checkpoint": str(ckpt_path.relative_to(ROOT)),
                "config_path": str(cfg_path.relative_to(ROOT)),
                "dataset_path": str(dataset_path.relative_to(ROOT)),
                "num_samples": int(_dataset_num_samples(loader)),
                **{k: float(v) for k, v in metrics.items()},
            }
            if target_denorm is not None:
                metrics_phys = evaluate_accuracy(
                    model, loader, device, target_denorm=target_denorm
                )
                row.update({f"{k}_physical": float(v) for k, v in metrics_phys.items()})
                row["target_offset"] = float(target_denorm[0])
                row["target_scale"] = float(target_denorm[1])

            rows.append(row)
            matrix[train_key][f"res{test_grid}"] = float(row["rel_l2"])
            if train_grid == test_grid:
                diagonal.append(row)

    summary = {
        "evaluation_type": "native_resolution_transfer_matrix",
        "model": "fno_hydrostatic",
        "solvers": ["swe_hydrostatic"],
        "grids": list(GRIDS),
        "note": "Hydrostatic-only native-resolution auxiliary diagnostic; MUSCL-HR and Boussinesq native-resolution matrices are intentionally not included.",
        "matrix_rel_l2": matrix,
        "diagonal": diagonal,
        "rows": rows,
    }
    save_json(summary, ROOT / args.output)
    print(summary)


if __name__ == "__main__":
    main()
