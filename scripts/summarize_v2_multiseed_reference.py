#!/usr/bin/env python
"""Summarize seed-to-seed variation in v2 cross-reference diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    if payload.get("evaluation_type") != "v2_cross_reference_discrepancy":
        raise ValueError(f"Unexpected evaluation type: {path}")
    if not isinstance(payload.get("training_seed"), int):
        raise ValueError(f"Missing integer training_seed: {path}")
    return payload


def _seed_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("Multi-seed summaries require at least two seeds")
    return {
        "seed_mean": float(np.mean(array, dtype=np.float64)),
        "seed_sample_std": float(np.std(array, ddof=1)),
        "seed_min": float(np.min(array)),
        "seed_max": float(np.max(array)),
    }


def summarize_payloads(
    payloads: list[dict[str, Any]],
    source_paths: list[str],
) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("At least two cross-reference artifacts are required")

    seeds = [int(payload["training_seed"]) for payload in payloads]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training_seed values must be unique")

    first = payloads[0]
    expected_common_time = first.get("common_time_v2")
    expected_datasets = first.get("dataset_paths")
    expected_samples = int(first.get("num_samples", -1))

    keyed_rows: list[dict[tuple[str, str], dict[str, Any]]] = []
    for payload in payloads:
        if payload.get("common_time_v2") != expected_common_time:
            raise ValueError("Common-time contracts differ across seeds")
        if payload.get("dataset_paths") != expected_datasets:
            raise ValueError("Dataset paths differ across seeds")
        if int(payload.get("num_samples", -1)) != expected_samples:
            raise ValueError("Scenario counts differ across seeds")
        rows = {
            (str(row["model_solver"]), str(row["benchmark_solver"])): row
            for row in payload.get("directions", [])
        }
        keyed_rows.append(rows)

    expected_keys = set(keyed_rows[0])
    if not expected_keys:
        raise ValueError("Cross-reference artifacts contain no directions")
    if any(set(rows) != expected_keys for rows in keyed_rows[1:]):
        raise ValueError("Cross-reference directions differ across seeds")

    directions: list[dict[str, Any]] = []
    for model_solver, benchmark_solver in sorted(expected_keys):
        rows = [mapping[(model_solver, benchmark_solver)] for mapping in keyed_rows]
        rho_values = [float(row["rho"]["point_estimate"]) for row in rows]
        numerator_values = [
            float(row["numerator"]["global_field_rmse"]) for row in rows
        ]
        control_values = [
            float(row["same_reference_control"]["global_field_rmse"]) for row in rows
        ]
        denominator_values = [
            float(row["denominator_solver_gap"]["global_field_rmse"]) for row in rows
        ]
        directions.append(
            {
                "model_solver": model_solver,
                "benchmark_solver": benchmark_solver,
                "rho": {
                    **_seed_stats(rho_values),
                    "by_seed": [
                        {
                            "seed": seed,
                            "point_estimate": value,
                            "scenario_ci_lower": float(row["rho"]["ci_lower"]),
                            "scenario_ci_upper": float(row["rho"]["ci_upper"]),
                        }
                        for seed, value, row in zip(seeds, rho_values, rows)
                    ],
                },
                "numerator_global_field_rmse": _seed_stats(numerator_values),
                "same_reference_global_field_rmse": _seed_stats(control_values),
                "denominator_solver_gap_global_field_rmse": {
                    "value": float(np.mean(denominator_values, dtype=np.float64)),
                    "max_abs_seed_difference": float(
                        np.max(
                            np.abs(
                                np.asarray(denominator_values, dtype=np.float64)
                                - denominator_values[0]
                            )
                        )
                    ),
                },
            }
        )

    return {
        "evaluation_type": "v2_multiseed_cross_reference_summary",
        "training_seeds": seeds,
        "seed_count": len(seeds),
        "num_samples": expected_samples,
        "dataset_paths": expected_datasets,
        "common_time_v2": expected_common_time,
        "source_artifacts": source_paths,
        "directions": directions,
        "interpretation": (
            "seed_sample_std is the sample standard deviation across the "
            "supplied independently trained checkpoints. Per-seed intervals "
            "remain paired test-scenario bootstrap intervals. With three "
            "seeds, neither quantity is presented as a population-level "
            "architecture confidence interval or physical-superiority claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payloads = [_load(path) for path in args.input]
    result = summarize_payloads(payloads, [str(path) for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[multiseed-reference-summary] seeds={result['training_seeds']} "
        f"directions={len(result['directions'])} -> {args.output}"
    )


if __name__ == "__main__":
    main()
