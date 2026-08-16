#!/usr/bin/env python
"""Summarize paired per-scenario statistics for the direct Hydrostatic models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_SPECS = (
    ("fno", "FNO"),
    ("ffno", "F-FNO"),
    ("convlstm", "ConvLSTM"),
    ("ufno", "U-FNO"),
    ("fno_modes8", "FNO-modes8"),
    ("fno_modes20", "FNO-modes20"),
    ("wno", "WNO"),
    ("unet", "U-Net"),
    ("cnn", "CNN"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray, bootstrap_values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "ci95_lower": float(np.quantile(bootstrap_values, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap_values, 0.975)),
    }


def _read_metrics(path: Path) -> tuple[list[tuple[str, ...]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No per-scenario rows found in {path}")

    required = {
        "sample_id",
        "scenario_id",
        "source_type",
        "bathymetry_type",
        "source_strength",
        "rel_l2",
    }
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    identities = [
        (
            str(row["sample_id"]),
            str(row["scenario_id"]),
            str(row["source_type"]),
            str(row["bathymetry_type"]),
            str(row["source_strength"]),
        )
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate scenario identities found in {path}")

    values = np.asarray([float(row["rel_l2"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite relative L2 values found in {path}")
    return identities, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-run",
        default="evaluation_runs/final-v2-paper-full-r1",
        help="Validated evaluation-run root containing direct/* per-scenario CSVs.",
    )
    parser.add_argument(
        "--baseline",
        choices=[key for key, _ in MODEL_SPECS],
        default="fno",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")

    run_root = Path(args.evaluation_run)
    completion_path = run_root / "completion_manifest.json"
    with completion_path.open(encoding="utf-8") as handle:
        completion = json.load(handle)
    if completion.get("status") != "validated":
        raise ValueError(f"Evaluation run is not validated: {completion_path}")

    declared_hashes = {
        str(row["path"]): str(row["sha256"])
        for row in completion.get("artifacts", [])
        if isinstance(row, dict) and "path" in row and "sha256" in row
    }

    identities: list[tuple[str, ...]] | None = None
    values_by_model: dict[str, np.ndarray] = {}
    source_artifacts: list[dict[str, Any]] = []
    for model_key, _ in MODEL_SPECS:
        relative_path = f"direct/{model_key}/physics_diagnostics_per_sample.csv"
        path = run_root / relative_path
        declared_hash = declared_hashes.get(relative_path)
        if declared_hash is None:
            raise KeyError(f"Completion manifest does not declare {relative_path}")
        observed_hash = _sha256(path)
        if observed_hash != declared_hash:
            raise ValueError(
                f"Artifact hash mismatch for {path}: {observed_hash} != {declared_hash}"
            )

        model_identities, values = _read_metrics(path)
        if identities is None:
            identities = model_identities
        elif model_identities != identities:
            raise ValueError(f"Scenario ordering differs for model {model_key}")
        values_by_model[model_key] = values
        source_artifacts.append(
            {
                "model": model_key,
                "path": relative_path,
                "sha256": observed_hash,
            }
        )

    if identities is None:
        raise RuntimeError("No model metrics were loaded")

    baseline = values_by_model[args.baseline]
    rng = np.random.default_rng(int(args.bootstrap_seed))
    indices = rng.integers(
        0,
        len(identities),
        size=(int(args.bootstrap_resamples), len(identities)),
    )

    model_rows: list[dict[str, Any]] = []
    for model_key, label in MODEL_SPECS:
        values = values_by_model[model_key]
        bootstrap_means = np.mean(values[indices], axis=1, dtype=np.float64)
        paired_delta = values - baseline
        bootstrap_delta = np.mean(
            paired_delta[indices],
            axis=1,
            dtype=np.float64,
        )
        model_rows.append(
            {
                "model": model_key,
                "label": label,
                "mean_per_scenario_relative_l2": _summary(
                    values,
                    bootstrap_means,
                ),
                "paired_difference_vs_baseline": _summary(
                    paired_delta,
                    bootstrap_delta,
                ),
                "scenario_fraction_better_than_baseline": float(
                    np.mean(values < baseline)
                ),
            }
        )

    result = {
        "evaluation_type": "direct_model_paired_scenario_bootstrap",
        "evaluation_run": str(run_root),
        "run_id": str(completion.get("run_id", run_root.name)),
        "num_scenarios": len(identities),
        "metric": "denormalized per-scenario relative_l2",
        "aggregation": "mean of per-scenario relative_l2 values",
        "baseline": args.baseline,
        "bootstrap": {
            "method": "paired nonparametric scenario bootstrap",
            "seed": int(args.bootstrap_seed),
            "resamples": int(args.bootstrap_resamples),
            "confidence_level": 0.95,
        },
        "source_artifacts": source_artifacts,
        "models": model_rows,
        "interpretation": (
            "Intervals quantify variation from resampling the shared test scenarios "
            "for fixed selected checkpoints. They do not quantify training-seed "
            "variation or establish population-level architecture rankings."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[direct-model-stats] models={len(model_rows)} "
        f"scenarios={len(identities)} -> {output_path}"
    )


if __name__ == "__main__":
    main()
