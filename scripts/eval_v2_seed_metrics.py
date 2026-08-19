#!/usr/bin/env python
"""Measure paired test-scenario metrics across the completed ensemble seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.target_scaling import load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything


def _metrics_by_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, np.ndarray]:
    diff = (prediction - target).to(torch.float64)
    target = target.to(torch.float64)
    dims = tuple(range(1, diff.ndim))
    mse = diff.square().mean(dim=dims).cpu().numpy()
    return {
        "mae": diff.abs().mean(dim=dims).cpu().numpy(),
        "rmse": np.sqrt(mse),
        "rel_l2": (
            torch.linalg.vector_norm(diff.flatten(start_dim=1), dim=1)
            / (torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1) + 1.0e-12)
        )
        .cpu()
        .numpy(),
        "max_error": diff.abs().amax(dim=dims).cpu().numpy(),
    }


def _ci(values: np.ndarray, seed: int, resamples: int) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    means = np.mean(values[indices], axis=1, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
    }


def _validate_training_seeds(
    member_rows: list[dict[str, Any]],
    expected_seeds: list[int] | None,
) -> list[int]:
    seeds = [int(row["training_seed"]) for row in member_rows]
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"Checkpoint training seed is missing: {seeds}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Checkpoint training seeds are not unique: {seeds}")
    if expected_seeds is not None and seeds != expected_seeds:
        raise ValueError(
            f"Checkpoint training seeds do not match the requested order: "
            f"{seeds} != {expected_seeds}"
        )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--label", default="seed_stability")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.checkpoint) < 2:
        raise ValueError("Seed metrics require at least two checkpoints")
    cfg = load_config(args.config)
    cfg["device"] = args.device
    cfg["data"] = {
        "test_path": args.dataset,
        "batch_size": int(args.batch_size),
        "num_workers": 0,
    }
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)
    denorm = load_target_denorm(args.dataset)
    if denorm is None:
        raise ValueError("Seed metrics require target normalization statistics")
    loaders = create_dataloaders(cfg)
    loader = loaders.get("test")
    if loader is None:
        raise KeyError("Could not build the seed-metrics test loader")

    member_rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoint:
        model = build_model(cfg).to(device).eval()
        payload = load_checkpoint(checkpoint, model, map_location=device)
        metric_values = {key: [] for key in ("mae", "rmse", "rel_l2", "max_error")}
        with torch.no_grad():
            for batch in loader:
                prediction = model(batch["x"].to(device))
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                elif isinstance(prediction, dict):
                    prediction = prediction.get("mean", next(iter(prediction.values())))
                target = batch["y"].to(device)
                prediction = prediction * float(denorm[1]) + float(denorm[0])
                target = target * float(denorm[1]) + float(denorm[0])
                values = _metrics_by_sample(prediction, target)
                for key, array in values.items():
                    metric_values[key].extend(float(value) for value in array)
        member_rows.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "training_seed": int(
                    payload.get("config", {}).get(
                        "seed",
                        payload.get("seed", -1),
                    )
                ),
                "metrics": {
                    key: _ci(
                        np.asarray(values, dtype=np.float64),
                        args.bootstrap_seed + index,
                        args.bootstrap_resamples,
                    )
                    for index, (key, values) in enumerate(metric_values.items())
                },
            }
        )

    training_seeds = _validate_training_seeds(member_rows, args.expected_seeds)
    seed_summary: dict[str, Any] = {}
    for metric in ("mae", "rmse", "rel_l2", "max_error"):
        values = np.asarray(
            [float(row["metrics"][metric]["mean"]) for row in member_rows],
            dtype=np.float64,
        )
        seed_summary[metric] = {
            "member_mean": float(np.mean(values)),
            "member_std": float(np.std(values, ddof=1)),
            "member_min": float(np.min(values)),
            "member_max": float(np.max(values)),
        }

    result = {
        "evaluation_type": "v2_seed_stability",
        "label": str(args.label),
        "reference": args.reference,
        "config_path": str(args.config),
        "dataset_path": str(args.dataset),
        "num_samples": int(len(loader.dataset)),
        "member_count": len(member_rows),
        "training_seeds": training_seeds,
        "checkpoints": [str(path) for path in args.checkpoint],
        "bootstrap": {
            "seed": int(args.bootstrap_seed),
            "resamples": int(args.bootstrap_resamples),
            "confidence_level": 0.95,
        },
        "members": member_rows,
        "seed_summary": seed_summary,
        "interpretation": (
            "Scenario-bootstrap intervals describe test-scenario variation "
            "within each fixed checkpoint. The member standard deviation "
            "describes variation across the supplied training seeds; it is "
            "not a population-level confidence interval by itself."
        ),
    }
    save_json(result, args.output)
    print(f"[v2-seed-metrics] members={len(member_rows)} -> {args.output}")


if __name__ == "__main__":
    main()
