#!/usr/bin/env python
"""Freeze a nested GeoClaw grid while preserving the physical domain and inputs.

The refined cells repeat frozen parent cell averages. This isolates numerical
resolution for a fixed raster initial condition; it does not restore unresolved
bathymetry or establish a continuum exact solution.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsunami_surrogate.data_gen.common_time_v2 import hash_array, stable_hash_payload
from tsunami_surrogate.evaluation.common_time_v2_level_a import validate_checksums
from tsunami_surrogate.evaluation.established_solver_validation import _write_checksums
from tsunami_surrogate.utils.hashing import sha256_file


def _refine_arrays(arrays: dict[str, np.ndarray], factor: int) -> dict[str, np.ndarray]:
    if type(factor) is not int or factor < 2:
        raise ValueError("Refinement factor must be an integer of at least two")
    result = {key: value.copy() for key, value in arrays.items()}
    shape = arrays["bathymetry"].shape
    for key in ("bathymetry", "eta0", "initial_depth", "hu0", "hv0"):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid parent array: {key}")
        result[key] = np.repeat(np.repeat(arrays[key], factor, axis=0), factor, axis=1)
    result["output_crop"] = arrays["output_crop"] * factor
    return result


def prepare(parent: Path, output: Path, factor: int) -> Path:
    parent = parent.resolve()
    validate_checksums(parent)
    parent_file = parent / "frozen_contract.json"
    original = json.loads(parent_file.read_text())
    if (
        original.get("artifact_kind")
        != "corrected-production-geoclaw-swe-frozen-contract"
    ):
        raise ValueError("Expected the frozen benchmark GeoClaw comparison as parent")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frozen = copy.deepcopy(original)
    frozen["artifact_kind"] = "nested-grid-geoclaw-swe-comparison"
    frozen.pop("bundle_hash")
    frozen["refinement"] = {
        "parent_bundle": original["bundle_hash"],
        "parent_manifest_sha256": sha256_file(parent_file),
        "factor": factor,
        "input_policy": "repeat each parent cell average on a fixed physical domain",
        "interpretation": "Independent-code grid sensitivity, not physical truth or a formal convergence proof",
        "preparation_script_sha256": sha256_file(Path(__file__)),
    }
    refined_inputs = {}
    for case in frozen["cases"]:
        case_id = case["case_id"]
        with np.load(
            parent / "cases" / case_id / "input.npz", allow_pickle=False
        ) as payload:
            arrays = _refine_arrays(
                {key: payload[key] for key in payload.files}, factor
            )
        parent_case_hash = case.pop("case_hash")
        case["parent_case_hash"] = parent_case_hash
        case["input_role"] = (
            "parent benchmark metadata; execution arrays are described by refined_initial_arrays"
        )
        # The in-house fields remain the archived baseline, on the same 64-cell
        # output grid. They must not be described as refined in-house solutions.
        case["inhouse_role"] = "unchanged archived baseline fields"
        domain = case["external_domain"]
        domain["shape"] = list(arrays["bathymetry"].shape)
        domain["output_crop"] = arrays["output_crop"].tolist()
        domain["publication_reduction"] = [
            value * factor for value in domain["publication_reduction"]
        ]
        case["refined_initial_arrays"] = {
            key: hash_array(arrays[key])
            for key in ("bathymetry", "eta0", "initial_depth", "hu0", "hv0")
        }
        case["case_hash"] = stable_hash_payload(
            artifact_kind="nested-geoclaw-case",
            payload=case,
            schema_id=frozen["schema_id"],
        )
        arrays["case_hash"] = np.asarray(case["case_hash"])
        refined_inputs[case_id] = arrays
        requirement = next(
            row for row in frozen["external_results"] if row["case_id"] == case_id
        )
        requirement.update(
            case_hash=case["case_hash"],
            computational_shape=domain["shape"],
            output_crop=domain["output_crop"],
            publication_reduction=domain["publication_reduction"],
        )
    # The original source configuration remains an explicit parent snapshot;
    # the refined experiment is described by the new domain records above.
    frozen["source_config_role"] = (
        "parent benchmark settings; external grid overridden by refinement records"
    )
    frozen["bundle_hash"] = stable_hash_payload(
        artifact_kind="nested-geoclaw-bundle",
        payload=frozen,
        schema_id=frozen["schema_id"],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_contract.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    (output / "external_results_manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "tsunami-surrogate.corrected-production-geoclaw-external-result.v1",
                "bundle_hash": frozen["bundle_hash"],
                "results": frozen["external_results"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for case in frozen["cases"]:
        case_id = case["case_id"]
        directory = output / "cases" / case_id
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "input.npz", **refined_inputs[case_id])
        for solver in frozen["source_config"]["inhouse_references"]:
            with np.load(
                parent / "cases" / case_id / f"inhouse_{solver}.npz", allow_pickle=False
            ) as payload:
                fields = {key: payload[key] for key in payload.files}
            fields["case_hash"] = np.asarray(case["case_hash"])
            np.savez_compressed(directory / f"inhouse_{solver}.npz", **fields)
    (output / "README.md").write_text(
        "# Nested-grid independent numerical comparison\n\n"
        f"The external grid is refined by {factor} on the same physical domain. "
        "Initial conserved fields and bathymetry repeat parent cell averages. "
        "Archived in-house comparison fields retain their original grid. "
        "The external fields are restricted to the same 64-cell output grid. "
        "This is a numerical sensitivity study; neither grid is physical truth.\n"
    )
    _write_checksums(output)
    validate_checksums(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    print(prepare(args.parent, args.output, args.factor))
