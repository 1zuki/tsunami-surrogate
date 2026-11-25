#!/usr/bin/env python
"""Compare two frozen independent-comparator grids on their common output cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsunami_surrogate.evaluation.common_time_level_a import validate_checksums
from tsunami_surrogate.evaluation.established_solver_validation import (
    _load_external_result,
    _load_external_run_manifest,
    _validate_external_checksums,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(left: np.ndarray, reference: np.ndarray) -> dict:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "per_time_rmse": np.sqrt(np.mean(difference**2, axis=(1, 2))).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-bundle", type=Path, required=True)
    parser.add_argument("--coarse-external", type=Path, required=True)
    parser.add_argument("--fine-bundle", type=Path, required=True)
    parser.add_argument("--fine-external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundles = {}
    manifests = {}
    source_files = {}
    for name in ("coarse", "fine"):
        bundle = getattr(args, f"{name}_bundle")
        external = getattr(args, f"{name}_external")
        validate_checksums(bundle)
        frozen = json.loads((bundle / "frozen_contract.json").read_text())
        _validate_external_checksums(external, frozen)
        manifests[name] = _load_external_run_manifest(external, frozen)
        bundles[name] = frozen
        for path in (
            bundle / "frozen_contract.json",
            bundle / "SHA256SUMS.txt",
            external / "RUN_MANIFEST.json",
            external / "SHA256SUMS.txt",
        ):
            source_files[str(path.resolve())] = _digest(path)
    coarse = bundles["coarse"]
    fine = bundles["fine"]
    if fine["refinement"]["parent_bundle"] != coarse["bundle_hash"]:
        raise ValueError("Comparator grids do not share a frozen parent")
    if fine["requested_times"] != coarse["requested_times"]:
        raise ValueError("Comparator grids have different requested times")
    factor = fine["refinement"]["factor"]
    times = np.asarray(coarse["requested_times"], dtype=np.float64)
    requirement_maps = {
        name: {row["case_id"]: row for row in frozen["external_results"]}
        for name, frozen in bundles.items()
    }
    if set(requirement_maps["coarse"]) != set(requirement_maps["fine"]):
        raise ValueError("Comparator case rosters differ")
    rows = []
    for case_id in requirement_maps["coarse"]:
        with (
            np.load(
                args.coarse_bundle / "cases" / case_id / "input.npz", allow_pickle=False
            ) as parent,
            np.load(
                args.fine_bundle / "cases" / case_id / "input.npz", allow_pickle=False
            ) as refined,
        ):
            for key in ("bathymetry", "eta0", "initial_depth", "hu0", "hv0"):
                expected = np.repeat(
                    np.repeat(parent[key], factor, axis=0), factor, axis=1
                )
                if not np.array_equal(expected, refined[key]):
                    raise ValueError(
                        f"Initial cell averages differ for {case_id}/{key}"
                    )
            if not np.array_equal(
                parent["domain_bounds"], refined["domain_bounds"]
            ) or not np.array_equal(
                parent["output_crop"] * factor, refined["output_crop"]
            ):
                raise ValueError("Physical domain or output crop changed")
        fields = {}
        for name in bundles:
            requirement = requirement_maps[name][case_id]
            path = getattr(args, f"{name}_external") / requirement["relative_path"]
            fields[name], _ = _load_external_result(
                path, requirement, times, manifests[name]
            )
            source_files[str(path.resolve())] = _digest(path)
        row = {
            "case_id": case_id,
            "geoclaw_coarse_vs_fine": _metrics(fields["coarse"], fields["fine"]),
            "archived_baseline_comparisons": {},
        }
        for solver in coarse["source_config"]["inhouse_references"]:
            path = args.coarse_bundle / "cases" / case_id / f"inhouse_{solver}.npz"
            with np.load(path, allow_pickle=False) as payload:
                trajectory = payload["eta"].astype(np.float64)
            source_files[str(path.resolve())] = _digest(path)
            comparisons = {
                name: _metrics(trajectory, field) for name, field in fields.items()
            }
            comparisons["rmse_fraction_change"] = (
                comparisons["fine"]["rmse"] / comparisons["coarse"]["rmse"] - 1
            )
            row["archived_baseline_comparisons"][solver] = comparisons
        rows.append(row)
    if any(_digest(Path(path)) != expected for path, expected in source_files.items()):
        raise RuntimeError("A comparator input changed during summary computation")
    result = {
        "scope": "Three fixed benchmark scenarios, two independent-code grids, common 64-cell output grid",
        "interpretation": "Nested fixed-raster inputs on the same physical domain. Neither independent-comparator grid is physical truth; reference norms change with resolution, so absolute RMSE is the primary comparison.",
        "coarse_shape": coarse["cases"][0]["external_domain"]["shape"],
        "fine_shape": fine["cases"][0]["external_domain"]["shape"],
        "requested_times": times.tolist(),
        "rows": rows,
        "source_files": source_files,
        "script_sha256": _digest(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
