#!/usr/bin/env python3
"""Run a non-decisional finer-grid diagnostic for the frozen Level A MUSCL result."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.common_time_v2_level_a import _observed_order, _run_mode


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _limiter_record(row: dict[str, Any]) -> dict[str, Any]:
    operator = row["operator"]
    total = int(operator.get("muscl_limiter_total_count", 0))
    limited = int(operator.get("muscl_limiter_limited_count", 0))
    zeroed = int(operator.get("muscl_limiter_zeroed_count", 0))
    denominator = max(total, 1)
    return {
        "grid": int(row["grid"]),
        "limited_count": limited,
        "limited_fraction": limited / denominator,
        "total_count": total,
        "zeroed_count": zeroed,
        "zeroed_fraction": zeroed / denominator,
    }


def _classification(errors: list[float], orders: list[float | None]) -> str:
    if not all(math.isfinite(value) for value in errors):
        return "implementation defect"
    if not all(coarse > fine for coarse, fine in zip(errors, errors[1:])):
        return "implementation defect"
    final_order = orders[-1]
    if final_order is not None and final_order >= 1.5:
        return "non-asymptotic behavior"
    return "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-rows", type=Path, required=True)
    parser.add_argument("--historical-aggregate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=256)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    historical = [
        row
        for row in _load_jsonl(args.historical_rows)
        if row.get("component") == "analytical_mode"
        and row.get("solver") == "swe_muscl_hr"
        and row.get("analytical_role") == "spatial"
    ]
    historical.sort(key=lambda row: int(row["grid"]))
    if [int(row["grid"]) for row in historical] != [32, 64, 128]:
        raise ValueError("Expected frozen MUSCL spatial rows at grids 32, 64, and 128")

    aggregate = json.loads(args.historical_aggregate.read_text(encoding="utf-8"))
    temporal = next(
        gate
        for gate in aggregate["gates"]
        if gate["gate"] == "temporal_refinement_swe_muscl_hr"
    )

    analytical = config["analytical"]
    started = time.perf_counter()
    finer = _run_mode(
        "swe_muscl_hr",
        nx=args.grid,
        ny=int(analytical["transverse_cells"]),
        mode=1,
        cfl=float(analytical["spatial_refinement_cfl"]["swe_muscl_hr"]),
        amplitude=float(analytical["amplitude"]),
    )
    runtime_s = time.perf_counter() - started
    finer.pop("_trajectory_eta", None)
    finer["analytical_role"] = "spatial_non_decisional_diagnostic"

    combined = [*historical, finer]
    errors = [float(row["field_relative_l2"]) for row in combined]
    orders = [_observed_order(coarse, fine) for coarse, fine in zip(errors, errors[1:])]
    limiter = [_limiter_record(row) for row in combined]
    result = {
        "schema_id": "tsunami-surrogate.level-a-muscl-spatial-diagnostic.v1",
        "decision_role": "non_decisional_diagnostic",
        "frozen_result_unchanged": True,
        "threshold_unchanged": 1.5,
        "grids": [int(row["grid"]) for row in combined],
        "field_relative_l2_errors": errors,
        "pairwise_observed_orders": orders,
        "limiter_activation": limiter,
        "effect_separation": {
            "temporal": {
                "assessment": "below the frozen roundoff floor at the tested refinement",
                "frozen_gate": temporal,
            },
            "boundary": {
                "assessment": "excluded by the periodic boundary and disabled sponge",
                "boundary": "periodic",
                "sponge_enabled": False,
            },
            "limiter": {
                "assessment": "active on every grid; association is recorded but is not proof of causation",
                "activation": limiter,
            },
        },
        "classification": _classification(errors, orders),
        "finer_grid_row": finer,
        "runtime_seconds": runtime_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
