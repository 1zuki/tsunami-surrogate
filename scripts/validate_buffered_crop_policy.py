#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.buffered_crop_benchmark import (
    SOLVERS,
    load_inventory_records,
    run_buffered_case,
    write_result,
)
from src.evaluation.common_time_v2_level_a import _load_canary_arrays
from src.evaluation.finite_horizon_boundary_study import comparison_metrics


DEFAULT_INVENTORY = Path(
    "artifacts/common_time_v2/h0/"
    "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b/"
    "h0_input_inventory.jsonl"
)
DEFAULT_V5_FREEZE = Path(
    "artifacts/common_time_v2/boundary_contract_study/finite_horizon_v5/"
    "STATIC_FREEZE.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/common_time_v2/buffered_crop_validation_v1/result.json"
)
METRIC_THRESHOLDS = {
    "relative_l2": "shared_crop_relative_l2",
    "interior_relative_l2": "interior_relative_l2",
    "amplitude_relative_error": "amplitude_relative_error",
    "phase_correlation_loss": "phase_correlation_loss",
}


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "absolute_rms",
        "relative_l2",
        "interior_absolute_rms",
        "interior_relative_l2",
        "boundary_absolute_rms",
        "boundary_relative_l2",
        "amplitude_relative_error",
        "phase_correlation_loss",
    )
    return {
        key: {
            "maximum": float(max(float(row[key]) for row in rows)),
            "median": float(np.median([float(row[key]) for row in rows])),
            "final": float(rows[-1][key]),
        }
        for key in keys
    }


def _run_task(args: tuple[dict[str, Any], str, int, int]) -> dict[str, Any]:
    record, solver, grid, sponge_width = args
    row, trajectory = run_buffered_case(
        record,
        solver_name=solver,
        total_grid=grid,
        source_taper_cells=8,
        sponge_min_factor=0.8,
        sponge_width_cells=sponge_width,
    )
    return {"row": row, "trajectory": trajectory}


def _load_frozen_thresholds(path: Path) -> dict[str, float]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(freeze, Mapping):
        raise ValueError("frozen v5 study record must be a mapping")
    frozen = freeze["thresholds_frozen_before_selected_case_execution"]
    return {
        metric: float(frozen[name]) for metric, name in METRIC_THRESHOLDS.items()
    }


def run_validation(
    record: Mapping[str, Any],
    *,
    grids: Sequence[int],
    sponge_width: int,
    thresholds: Mapping[str, float],
    workers: int,
) -> dict[str, Any]:
    ordered_grids = sorted({int(value) for value in grids})
    if len(ordered_grids) < 3 or ordered_grids[0] != 96:
        raise ValueError("validation requires 96 and at least two larger control grids")
    if workers < 1:
        raise ValueError("workers must be positive")
    total = len(SOLVERS) * len(ordered_grids)
    tasks = [
        (dict(record), solver, grid, sponge_width)
        for grid in ordered_grids
        for solver in SOLVERS
    ]
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            _record, solver, grid, _width = futures[future]
            result = future.result()
            results.append(result)
            completed += 1
            print(
                f"[{completed}/{total}] {solver} grid={grid} finished in "
                f"{result['row']['health']['runtime_s']:.2f}s",
                flush=True,
            )

    trajectories = {
        (str(item["row"]["solver"]), int(item["row"]["total_grid"])): np.asarray(
            item["trajectory"], dtype=np.float64
        )
        for item in results
    }
    run_rows = [item["row"] for item in results]
    largest_grid = ordered_grids[-1]
    control_grid = ordered_grids[-2]
    comparisons: list[dict[str, Any]] = []
    assessments: dict[str, Any] = {}
    absolute_floor = 1.0e-7
    control_fraction = 0.25
    for solver in SOLVERS:
        reference = trajectories[(solver, largest_grid)]
        solver_comparisons: dict[int, dict[str, Any]] = {}
        for grid in ordered_grids[:-1]:
            metrics = comparison_metrics(
                trajectories[(solver, grid)],
                reference,
                boundary_band_cells=12,
                absolute_floor=absolute_floor,
            )
            summary = _metric_summary(metrics)
            solver_comparisons[grid] = summary
            comparisons.append(
                {
                    "solver": solver,
                    "candidate_grid": grid,
                    "comparison_grid": largest_grid,
                    "metrics": summary,
                }
            )

        candidate_maxima = {
            metric: float(solver_comparisons[96][metric]["maximum"])
            for metric in thresholds
        }
        control_maxima = {
            metric: float(solver_comparisons[control_grid][metric]["maximum"])
            for metric in thresholds
        }
        passed_by_metric = {
            metric: candidate_maxima[metric] <= float(limit)
            for metric, limit in thresholds.items()
        }
        control_allowances = {
            metric: control_fraction * float(limit)
            for metric, limit in thresholds.items()
        }
        control_adequate_by_metric = {
            metric: control_maxima[metric] <= control_allowances[metric]
            for metric in thresholds
        }
        health_rows = [row for row in run_rows if row["solver"] == solver]
        health_ok = all(
            bool(row["health"]["finite"])
            and int(row["health"]["cg_failure_count"]) == 0
            for row in health_rows
        )
        reference_adequate = all(control_adequate_by_metric.values())
        assessments[solver] = {
            "candidate_grid": 96,
            "control_grid": control_grid,
            "comparison_grid": largest_grid,
            "candidate_maxima": candidate_maxima,
            "thresholds": dict(thresholds),
            "passed_by_metric": passed_by_metric,
            "control_maxima": control_maxima,
            "control_allowances": control_allowances,
            "control_adequate_by_metric": control_adequate_by_metric,
            "reference_adequate": reference_adequate,
            "solver_health_ok": health_ok,
            "conditionally_supported": (
                health_ok and reference_adequate and all(passed_by_metric.values())
            ),
        }

    _bathy, _source, _strength_array, _strength, arrays = _load_canary_arrays(record)
    maximum_speed = math.sqrt(9.81 * float(np.max(arrays["initial_depth"])))
    return_time_bounds = {
        str(grid): (
            2.0 * ((grid - 64) // 2) * (1.0 / 64.0) / maximum_speed
        )
        for grid in ordered_grids
    }
    all_supported = all(
        bool(value["conditionally_supported"]) for value in assessments.values()
    )
    return {
        "artifact_kind": "buffered-central-crop-policy-validation",
        "status": "diagnostic_unfrozen_non_decisional",
        "qualified_id": str(record["qualified_id"]),
        "input_fingerprint": str(record["input_fingerprint"]),
        "policy": {
            "core_grid": 64,
            "candidate_grid": 96,
            "source_taper_cells": 8,
            "bathymetry_extension": "constant edge continuation",
            "fixed_outer_sponge_width_cells": sponge_width,
            "sponge_min_factor": 0.8,
            "sponge_profile": "cosine",
            "swe_boundary": "radiation",
            "boussinesq_boundary": "open",
        },
        "grids": ordered_grids,
        "largest_grid_round_trip_time_bound": return_time_bounds[str(largest_grid)],
        "production_horizon": 0.175,
        "largest_grid_return_safe_without_extra_factor": (
            return_time_bounds[str(largest_grid)] > 0.175
        ),
        "round_trip_time_bounds": return_time_bounds,
        "threshold_source": "frozen finite-horizon-v5 pre-outcome thresholds",
        "control_fraction": control_fraction,
        "all_solvers_conditionally_supported": all_supported,
        "assessments": assessments,
        "runs": sorted(run_rows, key=lambda row: (row["solver"], row["total_grid"])),
        "comparisons": sorted(
            comparisons, key=lambda row: (row["solver"], row["candidate_grid"])
        ),
        "duration_s": float(time.monotonic() - started),
        "workers": workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the provisional 96x96 buffered central-crop policy."
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--case-id", default="train:scenario_000001")
    parser.add_argument("--v5-freeze", type=Path, default=DEFAULT_V5_FREEZE)
    parser.add_argument("--grids", nargs="+", type=int, default=[96, 128, 184])
    parser.add_argument("--sponge-width", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    record = load_inventory_records(args.inventory, [args.case_id])[0]
    thresholds = _load_frozen_thresholds(args.v5_freeze)
    payload = run_validation(
        record,
        grids=args.grids,
        sponge_width=args.sponge_width,
        thresholds=thresholds,
        workers=args.workers,
    )
    written = write_result(args.output, payload)
    print(
        f"completed in {payload['duration_s']:.2f}s; "
        f"supported={payload['all_solvers_conditionally_supported']} "
        f"result={written['result']} sha256={written['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
