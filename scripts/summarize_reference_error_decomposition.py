#!/usr/bin/env python
"""Recover reference-error identities from the frozen global squared errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("evaluation_runs/final-v2-full-nonspeed-20260912-r6"),
    )
    parser.add_argument(
        "--pooled",
        type=Path,
        default=Path(
            "evaluation_runs/pooled-reference-ablation-r6/primary_equal_updates.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    completion_path = args.run / "completion_manifest.json"
    completion = json.loads(completion_path.read_text())
    if completion["status"] != "validated":
        raise ValueError("Expected the completed archived evaluation")
    relative = "paper_evidence/reference_analysis/cross_reference.json"
    expected = [
        row["sha256"] for row in completion["artifacts"] if row["path"] == relative
    ]
    cross_path = args.run / relative
    if expected != [digest(cross_path)]:
        raise ValueError(
            "Cross-reference field errors differ from the archived inventory"
        )
    cross = json.loads(cross_path.read_text())
    if cross["num_samples"] != 2500 or len(cross["directions"]) != 6:
        raise ValueError("Expected all six directions on the paired test set")
    rows = []
    directions = set()
    for row in cross["directions"]:
        pair = (row["model_solver"], row["benchmark_solver"])
        if pair in directions or pair[0] == pair[1] or row["num_samples"] != 2500:
            raise ValueError("Invalid reference direction or population")
        directions.add(pair)
        error = row["same_reference_control"]["global_field_rmse"] ** 2
        gap = row["denominator_solver_gap"]["global_field_rmse"] ** 2
        target = row["numerator"]["global_field_rmse"] ** 2
        if not np.isfinite([error, gap, target]).all() or min(error, gap, target) <= 0:
            raise ValueError("Invalid squared-error measurements")
        rho_squared = target / gap
        normalized_error = error / gap
        alignment = rho_squared - 1 - normalized_error
        # The archived field accumulator subtracts in float32 before its
        # float64 reduction, while the paired-ratio path casts before
        # subtraction. Preserve this measured arithmetic-path difference.
        ratio_residual = rho_squared - row["rho"]["point_estimate"] ** 2
        if not np.isclose(
            rho_squared,
            row["rho"]["point_estimate"] ** 2,
            atol=0,
            rtol=8 * np.finfo(np.float32).eps,
        ):
            raise ValueError("Stored ratio does not agree with its field norms")
        rows.append(
            {
                "model_solver": pair[0],
                "benchmark_solver": pair[1],
                "normalized_surrogate_error": normalized_error,
                "normalized_alignment": alignment,
                "rho_squared": rho_squared,
                "stored_ratio_squared_residual": ratio_residual,
            }
        )
    pooled = json.loads(args.pooled.read_text())
    if pooled["num_paired_scenarios"] != 2500:
        raise ValueError("Pooled comparison uses another test population")
    members = [
        row
        for name, row in pooled["models"].items()
        if name.startswith("anonymous_equal_s")
    ]
    if len(members) != 3:
        raise ValueError("Expected the three anonymous training seeds")
    for row in members:
        if digest(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError(
                "Anonymous checkpoint no longer matches its evaluation record"
            )
    errors = np.asarray(
        [row["equal_solver_rmse"]["global_field_rmse"] for row in members]
    )
    centroid_rms = pooled["oracle_centroid"]["equal_solver_rmse"]["global_field_rmse"]
    excess_mse = np.mean(errors**2) - centroid_rms**2
    if excess_mse < 0:
        raise ValueError("Anonymous error is below the equal-reference centroid error")
    source_files = {
        str(path.resolve()): digest(path)
        for path in (completion_path, cross_path, args.pooled)
    }
    result = {
        "interpretation": "Algebraic decomposition of stored equal-weight global squared errors. The alignment is inferred from the norm identity; neither component identifies physical truth or separately measures numerical mechanisms.",
        "source_files": source_files,
        "script_sha256": digest(Path(__file__)),
        "cross_reference": rows,
        "anonymous_reference_centroid": {
            "reference_centroid_rms": centroid_rms,
            "member_equal_reference_rmse": errors.tolist(),
            "member_rmse_mean": float(np.mean(errors)),
            "member_rmse_sample_std": float(np.std(errors, ddof=1)),
            "model_to_centroid_rms_from_mean_mse": float(np.sqrt(excess_mse)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
