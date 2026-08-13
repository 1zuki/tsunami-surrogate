#!/usr/bin/env python
"""Fit and evaluate ensemble variance calibration using only current v2 data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_ensemble_calibration import (
    _evaluate_loader,
    _fit_gamma,
    _load_ensemble,
    _loader_for_path,
    _nominal_levels,
)
from scripts.eval_v2_slices import _loader, _parse_suite, _select_indices
from src.data.dataset import _make_dataset
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--val-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.checkpoint) < 2:
        raise ValueError("Calibration requires at least two ensemble members")
    cfg = load_config(args.config)
    cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)
    checkpoints = [Path(path) for path in args.checkpoint]
    ensemble = _load_ensemble(cfg, checkpoints, device)
    val_loader = _loader_for_path(
        cfg,
        "val",
        Path(args.val_dataset),
        int(args.batch_size),
    )
    test_loader = _loader_for_path(
        cfg,
        "test",
        Path(args.test_dataset),
        int(args.batch_size),
    )
    levels = _nominal_levels()
    fit = _fit_gamma(
        ensemble,
        val_loader,
        device,
        eps=1.0e-12,
        max_samples=None,
    )
    datasets = {
        "validation": _evaluate_loader(
            ensemble,
            val_loader,
            device,
            levels=levels,
            gamma=float(fit["gamma"]),
            eps=1.0e-12,
            max_samples=None,
        ),
        "test": _evaluate_loader(
            ensemble,
            test_loader,
            device,
            levels=levels,
            gamma=float(fit["gamma"]),
            eps=1.0e-12,
            max_samples=None,
        ),
    }
    if args.suite:
        test_dataset = _make_dataset(args.test_dataset)
        for raw in args.suite:
            label, key, value = _parse_suite(raw)
            indices = _select_indices(test_dataset, key, value)
            datasets[label] = _evaluate_loader(
                ensemble,
                _loader(test_dataset, indices, int(args.batch_size)),
                device,
                levels=levels,
                gamma=float(fit["gamma"]),
                eps=1.0e-12,
                max_samples=None,
            )
    result = {
        "evaluation_type": "v2_ensemble_calibration",
        "config_path": str(args.config),
        "checkpoints": [str(path) for path in checkpoints],
        "calibration_fit_split": "validation",
        "val_dataset": str(args.val_dataset),
        "test_dataset": str(args.test_dataset),
        "dataset_path": str(args.test_dataset),
        "nominal_levels": levels,
        "fit": fit,
        "datasets": datasets,
        "device": str(device),
        "batch_size": int(args.batch_size),
    }
    save_json(result, args.output)
    print(f"[v2-calibration] members={len(checkpoints)} -> {args.output}")


if __name__ == "__main__":
    main()
