#!/usr/bin/env python
"""Summarize the frozen production-like GeoClaw compatibility canaries.

This is a post-hoc sensitivity diagnostic. It records which setup dimensions
differ and summarizes the retained comparison metrics; it does not assign a
unique causal explanation to the observed discrepancies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_HASH = (
    "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
)
DEFAULT_CONTRACT = (
    ROOT
    / "artifacts/common_time_v2/level_b_minimum"
    / BUNDLE_HASH
    / "frozen_contract.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "artifacts/common_time_v2/level_b_minimum_evaluation"
    / BUNDLE_HASH
    / "comparison_rows.json"
)
DEFAULT_EXTERNAL_RUN = (
    ROOT
    / "artifacts/common_time_v2/level_b_minimum_external"
    / BUNDLE_HASH
    / "RUN_MANIFEST.json"
)
DEFAULT_OUTPUT = ROOT / "paper/figures/geoclaw_discrepancy_diagnostic.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _range(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array, dtype=np.float64)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _common_or_fail(values: list[Any], label: str) -> Any:
    canonical = [json.dumps(value, sort_keys=True) for value in values]
    if len(set(canonical)) != 1:
        raise ValueError(f"Production canaries do not share one {label}")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--external-run", type=Path, default=DEFAULT_EXTERNAL_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    contract = _load_json(args.contract)
    rows = _load_json(args.evaluation)
    external_run = _load_json(args.external_run)
    if not isinstance(contract, dict) or not isinstance(rows, list):
        raise ValueError("Unexpected established-solver artifact structure")
    if contract.get("bundle_hash") != BUNDLE_HASH:
        raise ValueError("Frozen contract bundle hash changed")
    if external_run.get("bundle_hash") != BUNDLE_HASH:
        raise ValueError("External-run bundle hash changed")

    cases = [
        case
        for case in contract.get("cases", [])
        if case.get("category") == "production_input"
    ]
    production_rows = [
        row for row in rows if row.get("category") == "production_input"
    ]
    if len(cases) != 3 or len(production_rows) != 6:
        raise ValueError(
            f"Expected 3 production canaries and 6 comparisons, got "
            f"{len(cases)} and {len(production_rows)}"
        )

    inhouse_domain = _common_or_fail(
        [case["inhouse_domain"] for case in cases],
        "in-house domain setup",
    )
    external_domains = [case["external_domain"] for case in cases]
    external_domain = dict(external_domains[0])
    shared_external_keys = (
        "boundary",
        "bounds",
        "dx",
        "output_crop",
        "shape",
        "sponge",
    )
    for key in shared_external_keys:
        _common_or_fail(
            [domain[key] for domain in external_domains],
            f"external domain {key}",
        )
    external_domain["round_trip_time_bound_by_canary"] = {
        str(case["case_id"]): float(case["external_domain"]["round_trip_time_bound"])
        for case in cases
    }

    summary_by_solver: dict[str, Any] = {}
    for solver in ("swe_hydrostatic", "swe_muscl_hr"):
        solver_rows = [
            row for row in production_rows if row.get("inhouse_solver") == solver
        ]
        if len(solver_rows) != 3:
            raise ValueError(f"Expected three production rows for {solver}")
        summary_by_solver[solver] = {
            metric: _range([float(row[metric]) for row in solver_rows])
            for metric in (
                "trajectory_relative_l2",
                "per_time_scaled_l2_p95_active",
                "field_norm_ratio",
                "field_cosine_similarity",
                "boundary_band_relative_l2",
                "interior_relative_l2",
            )
        }

    output = {
        "schema_id": "tsunami-surrogate.geoclaw-discrepancy-diagnostic.v1",
        "bundle_hash": BUNDLE_HASH,
        "source_artifacts": [
            {
                "path": args.contract.resolve().relative_to(ROOT).as_posix(),
                "sha256": _sha256(args.contract),
            },
            {
                "path": args.evaluation.resolve().relative_to(ROOT).as_posix(),
                "sha256": _sha256(args.evaluation),
            },
            {
                "path": args.external_run.resolve().relative_to(ROOT).as_posix(),
                "sha256": _sha256(args.external_run),
            },
        ],
        "canaries": [
            {
                "case_id": str(case["case_id"]),
                "case_hash": str(case["case_hash"]),
                "qualified_id": str(case["source"]["qualified_id"]),
                "input_fingerprint": str(case["source"]["input_fingerprint"]),
            }
            for case in cases
        ],
        "shared_comparison_target": {
            "requested_time_count": len(contract["requested_times"]),
            "requested_time_horizon": float(contract["requested_times"][-1]),
            "publication_crop_shape": [64, 64],
            "cell_size": float(inhouse_domain["dx"]),
        },
        "frozen_setup_difference": {
            "inhouse": inhouse_domain,
            "geoclaw": external_domain,
            "geoclaw_numerics": {
                key: external_run["execution"][key]
                for key in (
                    "spatial_order",
                    "limiter",
                    "use_fwaves",
                    "dimensional_split",
                    "source_split",
                    "cfl_desired",
                    "cfl_max",
                )
            },
            "initial_state_embedding": (
                "The contract stores distinct 96x96 in-house and 192x192 "
                "external bathymetry/initial-state arrays around the common "
                "64x64 comparison crop."
            ),
        },
        "metrics_by_inhouse_solver": summary_by_solver,
        "causal_interpretation": {
            "status": "not_isolated",
            "supported_observations": [
                (
                    "MUSCL-HR has lower trajectory discrepancy than the "
                    "Hydrostatic reference on all three retained canaries."
                ),
                (
                    "Both boundary-band and interior discrepancies are "
                    "material, so the retained metrics do not support a "
                    "boundary-only explanation."
                ),
                (
                    "Field cosine similarities remain high while field-norm "
                    "ratios are below one, which is consistent with a mixture "
                    "of shape and amplitude/dissipation differences."
                ),
            ],
            "unisolated_dimensions": [
                "computational-domain extent",
                "outer-boundary treatment",
                "sponge treatment",
                "spatial reconstruction and flux/source treatment",
                "time integration and CFL policy",
                "initial-state embedding outside the publication crop",
            ],
            "excluded_conclusion": (
                "These frozen compatibility canaries cannot identify one "
                "unique cause or establish either implementation as physical "
                "truth."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "[geoclaw-diagnostic] "
        f"canaries={len(cases)} rows={len(production_rows)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
