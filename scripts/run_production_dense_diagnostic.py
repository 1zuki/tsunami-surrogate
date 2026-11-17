#!/usr/bin/env python
"""Replay selected test cases with the exact production solver contract.

The run advances every natural step to the requested horizon, reconstructs the
50 requested outputs by adjacent-state interpolation, compares them with the
stored production publications, and records production boundary fluxes for the
two shallow-water references.  The data roots are explicit CLI arguments so a
replay can be moved between machines without editing the script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

from tsunami_surrogate.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    _block_mean_downsample_spatial,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
)

DEFAULT_RAW_ROOT = Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test/raw")
DEFAULT_CACHE_ROOT = Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test")
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
SOLVER_DIRS = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
}
REQUESTED_TIMES = np.asarray([8.4 * index for index in range(1, 51)], dtype=np.float64)
BUFFER = BufferedDomainConfig(
    enabled=True,
    buffer_cells=32,
    source_taper_cells=16,
    bathymetry_extension="edge",
    output_crop="central",
)


def _solver_from_name(name: str, cfg: Mapping[str, Any]) -> Any:
    if name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(dict(cfg))
    if name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(dict(cfg))
    if name == "boussinesq":
        return _make_boussinesq_solver_from_cfg(dict(cfg))
    raise ValueError(f"Unsupported solver {name!r}")


def _resolved_solver_cfg(dataset_cfg: Mapping[str, Any], solver_name: str) -> dict[str, Any]:
    resolved = dict(dataset_cfg["solver"])
    resolved.update(dict(dataset_cfg["solver_profiles"][solver_name]))
    return resolved


def _mass_flux_arrays(solver: Any, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if state is None:
        h, hu, hv = solver.h, solver.hu, solver.hv
    else:
        h, hu, hv = state
    b = solver.b
    if solver.__class__.__name__ == "MUSCLHRShallowWaterSolver":
        rec = solver._reconstructed_faces(h, hu, hv, b)
        x_states = solver._reconstructed_x_interface_states(rec, h, hu, hv, b)
        y_states = solver._reconstructed_y_interface_states(rec, h, hu, hv, b)
    else:
        x_states = solver._x_interface_states(h, hu, hv, b)
        y_states = solver._y_interface_states(h, hu, hv, b)
    x_flux, _, _ = solver._hydrostatic_interface_fluxes(*x_states, axis="x")
    y_flux, _, _ = solver._hydrostatic_interface_fluxes(*y_states, axis="y")
    return np.asarray(x_flux[0], dtype=np.float64), np.asarray(y_flux[0], dtype=np.float64)


def _flux_metrics(x_flux: np.ndarray, y_flux: np.ndarray, *, dx: float, dy: float) -> dict[str, float]:
    left = x_flux[0, :]
    right = x_flux[-1, :]
    bottom = y_flux[:, 0]
    top = y_flux[:, -1]
    outward_rate = dy * (float(np.sum(right)) - float(np.sum(left)))
    outward_rate += dx * (float(np.sum(top)) - float(np.sum(bottom)))
    return {
        "outward_rate": float(outward_rate),
        "max_abs_boundary_mass_flux": float(
            max(
                np.max(np.abs(left)),
                np.max(np.abs(right)),
                np.max(np.abs(bottom)),
                np.max(np.abs(top)),
            )
        ),
    }


def _array_rel_l2(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    denominator = max(float(np.linalg.norm(np.asarray(right, dtype=np.float64).ravel())), 1.0e-30)
    return float(np.linalg.norm(difference.ravel()) / denominator)


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_case_inputs(
    index: int,
    solver_name: str,
    *,
    raw_root: Path,
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict[str, Any]]:
    bathy_path = cache_root / "bathymetry" / f"sample_{index:06d}.npz"
    source_path = cache_root / "sources" / f"sample_{index:06d}.npz"
    raw_path = raw_root / SOLVER_DIRS[solver_name] / "samples" / f"sample_{index:06d}"
    with np.load(bathy_path, allow_pickle=False) as bathy_payload:
        bathymetry = np.asarray(bathy_payload["solver_bathymetry"], dtype=np.float32)
    with np.load(source_path, allow_pickle=False) as source_payload:
        source_field = np.asarray(source_payload["solver_source_field"], dtype=np.float32)
        source_strength = float(np.asarray(source_payload["source_strength"]).reshape(-1)[0])
    with np.load(raw_path / "sample.npz", allow_pickle=False) as raw_payload:
        expected_eta = np.asarray(raw_payload["trajectory_eta"], dtype=np.float32)
    provenance = {}
    provenance_path = raw_path / "provenance.npz"
    if provenance_path.is_file():
        with np.load(provenance_path, allow_pickle=False) as payload:
            provenance = {key: np.asarray(payload[key]) for key in payload.files}
    return bathymetry, source_field, source_strength, expected_eta, provenance


def _interpolate_state(left: np.ndarray, right: np.ndarray, left_t: float, right_t: float, query: float) -> np.ndarray:
    if query < left_t - 1.0e-10 or query > right_t + 1.0e-10 or right_t <= left_t:
        raise RuntimeError(f"Invalid requested-time bracket: {left_t}, {right_t}, {query}")
    weight = (query - left_t) / (right_t - left_t)
    weight = min(1.0, max(0.0, float(weight)))
    return np.asarray(left, dtype=np.float64) * (1.0 - weight) + np.asarray(right, dtype=np.float64) * weight


def _run_one(
    solver_name: str,
    scenario_id: str,
    index: int,
    solver_cfg: Mapping[str, Any],
    raw_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    bathymetry, source_field, source_strength, expected_eta, archived_provenance = _load_case_inputs(
        index,
        solver_name,
        raw_root=raw_root,
        cache_root=cache_root,
    )
    prepared = _prepare_buffered_domain(
        bathymetry,
        source_field,
        source_strength,
        0.0,
        BUFFER,
        source_already_tapered=True,
    )
    solver = _solver_from_name(solver_name, solver_cfg)
    solver.set_bathymetry(prepared["solver_bathymetry"])
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        solver.set_initial_condition(
            prepared["solver_h0"],
            hu0=np.zeros_like(prepared["solver_h0"]),
            hv0=np.zeros_like(prepared["solver_h0"]),
        )
    else:
        solver.set_initial_condition(
            prepared["solver_eta0"],
            eta_t0=np.zeros_like(prepared["solver_eta0"]),
        )

    dx = float(solver_cfg["dx"])
    dy = float(solver_cfg["dy"])
    cell_area = dx * dy
    current_time = 0.0
    left_state = np.asarray(solver.get_state(), dtype=np.float64).copy()
    initial_mass = (
        float(np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area)
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}
        else None
    )
    requested_outputs: list[np.ndarray] = []
    left_times: list[float] = []
    right_times: list[float] = []
    weights: list[float] = []
    bracket_widths: list[float] = []
    natural_dt: list[float] = []
    max_abs_flux = 0.0
    max_abs_outward_rate = 0.0
    cumulative_boundary_mass_change = 0.0
    cumulative_residual = 0.0
    max_abs_residual = 0.0
    max_cfl = 0.0
    min_depth = float(np.min(solver.h)) if hasattr(solver, "h") else float("nan")
    max_speed = 0.0
    cg_failed_steps = 0
    requested_index = 0
    step = 0

    while requested_index < REQUESTED_TIMES.size:
        left_time = current_time
        dt = float(solver.suggest_dt(target_cfl=float(solver_cfg["cfl"])))
        if not math.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"Invalid dt for {solver_name} {scenario_id}: {dt}")
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
            mass_before = float(np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area)
            if solver_name == "swe_muscl_hr":
                flux0 = _flux_metrics(*_mass_flux_arrays(solver, left_state), dx=dx, dy=dy)
                stage_state = solver._euler_step_from_state(left_state, dt)
                flux1 = _flux_metrics(*_mass_flux_arrays(solver, stage_state), dx=dx, dy=dy)
                outward_rate = 0.5 * (flux0["outward_rate"] + flux1["outward_rate"])
                max_abs_flux = max(max_abs_flux, flux0["max_abs_boundary_mass_flux"], flux1["max_abs_boundary_mass_flux"])
                max_abs_outward_rate = max(max_abs_outward_rate, abs(flux0["outward_rate"]), abs(flux1["outward_rate"]))
            else:
                flux = _flux_metrics(*_mass_flux_arrays(solver, left_state), dx=dx, dy=dy)
                outward_rate = flux["outward_rate"]
                max_abs_flux = max(max_abs_flux, flux["max_abs_boundary_mass_flux"])
                max_abs_outward_rate = max(max_abs_outward_rate, abs(outward_rate))
        solver.step(dt=dt, auto_dt=False)
        current_time = left_time + dt
        step += 1
        natural_dt.append(dt)
        right_state = np.asarray(solver.get_state(), dtype=np.float64).copy()
        if not np.isfinite(right_state).all():
            raise RuntimeError(f"Non-finite state for {solver_name} {scenario_id} at step {step}")
        max_cfl = max(max_cfl, float(solver.compute_cfl(dt=dt)))
        if hasattr(solver, "h"):
            min_depth = min(min_depth, float(np.min(solver.h)))
            u, v = solver.compute_velocity()
            max_speed = max(max_speed, float(np.max(np.hypot(u, v))))
        if hasattr(solver, "last_step_cg_failed_count"):
            cg_failed_steps += int(getattr(solver, "last_step_cg_failed_count", 0))

        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
            mass_after = float(np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area)
            flux_mass_change = -dt * outward_rate
            observed_mass_change = mass_after - mass_before
            residual = observed_mass_change - flux_mass_change
            cumulative_boundary_mass_change += flux_mass_change
            cumulative_residual += residual
            max_abs_residual = max(max_abs_residual, abs(residual))

        while requested_index < REQUESTED_TIMES.size and REQUESTED_TIMES[requested_index] <= current_time + 1.0e-10:
            query = float(REQUESTED_TIMES[requested_index])
            state = _interpolate_state(left_state, right_state, left_time, current_time, query)
            if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
                eta = state[0] + prepared["solver_bathymetry"]
            else:
                eta = state[0]
            crop = prepared["crop"]
            published = _block_mean_downsample_spatial(
                eta[crop[0], crop[1]][None, ...], (64, 64)
            )[0].astype(np.float32)
            requested_outputs.append(published)
            weight = (query - left_time) / (current_time - left_time)
            left_times.append(left_time)
            right_times.append(current_time)
            weights.append(float(weight))
            bracket_widths.append(current_time - left_time)
            requested_index += 1
        left_state = right_state

    published_eta = np.stack(requested_outputs, axis=0)
    difference = np.asarray(published_eta, dtype=np.float64) - np.asarray(expected_eta, dtype=np.float64)
    archived_dt = np.asarray(archived_provenance.get("natural_dt_history", []), dtype=np.float64)
    generated_dt = np.asarray(natural_dt, dtype=np.float64)
    dt_diff = None
    if archived_dt.size:
        if archived_dt.shape == generated_dt.shape:
            dt_diff = float(np.max(np.abs(archived_dt - generated_dt)))
        else:
            dt_diff = float("inf")
    final_mass = (
        float(np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area)
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}
        else None
    )
    mass_relative_change = (
        float((final_mass - initial_mass) / max(abs(initial_mass), 1.0e-30))
        if initial_mass is not None and final_mass is not None
        else None
    )
    cumulative_boundary_mass_change_relative = (
        float(cumulative_boundary_mass_change / max(abs(initial_mass), 1.0e-30))
        if initial_mass is not None
        else None
    )
    cumulative_residual_relative = (
        float(cumulative_residual / max(abs(initial_mass), 1.0e-30))
        if initial_mass is not None
        else None
    )
    return {
        "scenario_id": scenario_id,
        "sample_index": int(index),
        "solver": solver_name,
        "requested_count": int(published_eta.shape[0]),
        "requested_horizon": float(current_time),
        "natural_step_count": int(step),
        "natural_dt_min": float(np.min(generated_dt)),
        "natural_dt_max": float(np.max(generated_dt)),
        "archived_natural_step_count": int(archived_dt.size),
        "natural_dt_max_abs_difference": dt_diff,
        "replay_max_abs": float(np.max(np.abs(difference))),
        "replay_relative_l2": _array_rel_l2(published_eta, expected_eta),
        "replay_rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_cfl_seen": float(max_cfl),
        "target_cfl": float(solver_cfg["cfl"]),
        "min_depth_seen": float(min_depth),
        "max_speed_seen": float(max_speed),
        "cg_failed_steps": int(cg_failed_steps),
        "finite_final_state": bool(np.isfinite(solver.get_state()).all()),
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "mass_relative_change": mass_relative_change,
        "max_interpolation_bracket": float(max(bracket_widths)),
        "median_interpolation_bracket": float(np.median(bracket_widths)),
        "max_interpolation_weight": float(max(weights)),
        "min_interpolation_weight": float(min(weights)),
        "max_abs_boundary_mass_flux": float(max_abs_flux) if solver_name != "boussinesq" else None,
        "max_abs_outward_mass_rate": float(max_abs_outward_rate) if solver_name != "boussinesq" else None,
        "cumulative_boundary_mass_change": float(cumulative_boundary_mass_change) if solver_name != "boussinesq" else None,
        "cumulative_boundary_mass_change_relative": cumulative_boundary_mass_change_relative,
        "cumulative_sponge_projection_residual": float(cumulative_residual) if solver_name != "boussinesq" else None,
        "cumulative_sponge_projection_residual_relative": cumulative_residual_relative,
        "max_abs_sponge_projection_residual_per_step": float(max_abs_residual) if solver_name != "boussinesq" else None,
        "output_hash": _array_hash(published_eta),
        "elapsed_s": float(time.monotonic() - started),
    }


def _load_selection(path: Path) -> list[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["dense_validation"]["ordered_scenarios"]
    return [(str(row["scenario_id"]), int(str(row["scenario_id"]).rsplit("_", 1)[1])) for row in rows]


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)
    return {
        "count": int(values.size),
        "median": float(np.median(values)) if values.size else float("nan"),
        "p95": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        "max": float(np.max(values)) if values.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/dataset_test.yaml")
    parser.add_argument("--selection", default="configs/eval/common_time_validation_scenarios.json")
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/Windows/Dataset/current_production_dense_validation_20261117/production_replay"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    dataset_cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    raw_root = args.raw_root.resolve()
    cache_root = args.cache_root.resolve()
    selection_path = ROOT / args.selection if not Path(args.selection).is_absolute() else Path(args.selection)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = _load_selection(selection_path)
    if args.max_scenarios is not None:
        selected = selected[: max(0, int(args.max_scenarios))]

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "production_dense_rows.jsonl"
    rows: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    completed_keys = {(str(row["solver"]), int(row["sample_index"])) for row in rows}

    tasks = []
    for scenario_id, index in selected:
        for solver_name in SOLVERS:
            if (solver_name, index) not in completed_keys:
                tasks.append(
                    (
                        solver_name,
                        scenario_id,
                        index,
                        _resolved_solver_cfg(dataset_cfg, solver_name),
                        raw_root,
                        cache_root,
                    )
                )
    failures: list[str] = []
    print(
        f"tasks={len(tasks)} workers={args.workers} resumed={len(rows)} output={output_root}",
        flush=True,
    )
    checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_run_one, *task) for task in tasks]
        for position, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - record each worker failure and continue the audit
                failures.append(repr(exc))
                print(f"[{position}/{len(tasks)}] FAILED {exc!r}", flush=True)
                continue
            rows.append(row)
            checkpoint_handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
            checkpoint_handle.flush()
            print(
                f"[{position}/{len(tasks)}] {row['solver']} {row['scenario_id']} "
                f"steps={row['natural_step_count']} replay_rel={row['replay_relative_l2']:.3e}",
                flush=True,
            )
    checkpoint_handle.close()

    if failures:
        (output_root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} production replay failures")
    rows.sort(key=lambda row: (row["solver"], row["sample_index"]))
    by_solver = {}
    for solver_name in SOLVERS:
        subset = [row for row in rows if row["solver"] == solver_name]
        by_solver[solver_name] = {
            "scenario_count": len(subset),
            "finite_final_state_count": int(sum(bool(row["finite_final_state"]) for row in subset)),
            "replay_relative_l2": _summary(subset, "replay_relative_l2"),
            "replay_max_abs": _summary(subset, "replay_max_abs"),
            "natural_step_count": _summary(subset, "natural_step_count"),
            "natural_dt_max_abs_difference": _summary(subset, "natural_dt_max_abs_difference"),
            "max_interpolation_bracket": _summary(subset, "max_interpolation_bracket"),
            "max_cfl_seen": _summary(subset, "max_cfl_seen"),
            "cg_failed_steps": int(sum(int(row["cg_failed_steps"]) for row in subset)),
        }
        if solver_name != "boussinesq":
            by_solver[solver_name].update(
                {
                    "max_abs_boundary_mass_flux": _summary(subset, "max_abs_boundary_mass_flux"),
                    "max_abs_outward_mass_rate": _summary(subset, "max_abs_outward_mass_rate"),
                    "abs_cumulative_sponge_projection_residual": _summary(
                        [{**row, "abs_cumulative_sponge_projection_residual": abs(float(row["cumulative_sponge_projection_residual"]))} for row in subset],
                        "abs_cumulative_sponge_projection_residual",
                    ),
                    "abs_cumulative_sponge_projection_residual_relative": _summary(
                        [
                            {
                                **row,
                                "abs_cumulative_sponge_projection_residual_relative": abs(
                                    float(row["cumulative_sponge_projection_residual_relative"])
                                ),
                            }
                            for row in subset
                        ],
                        "abs_cumulative_sponge_projection_residual_relative",
                    ),
                }
            )

    manifest = {
        "schema_id": "tsunami-surrogate.current-production-dense-diagnostic.v1",
        "diagnostic": "production_natural_step_replay_and_boundary_flux",
        "config_path": str((ROOT / args.config).resolve()),
        "selection_path": str(selection_path.resolve()),
        "config_sha256": _sha256_file((ROOT / args.config).resolve()),
        "selection_audit_hash": str(selection_payload.get("audit_hash", "")),
        "selection_list_hash": str(selection_payload["dense_validation"]["list_hash"]),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "selection_count": len(selected),
        "solver_count": len(SOLVERS),
        "expected_task_count": int(len(selected) * len(SOLVERS)),
        "task_count": len(tasks),
        "resumed_row_count": int(len(rows) - len(tasks)),
        "rows_written": len(rows),
        "production_grid": {"solver_shape": [192, 192], "publication_shape": [64, 64], "buffer_cells": 32},
        "data_roots": {"raw_root": str(raw_root), "cache_root": str(cache_root)},
        "requested_times": REQUESTED_TIMES.tolist(),
        "by_solver": by_solver,
        "rows": rows,
    }
    (output_root / "production_dense_diagnostic.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else value) + "\n", encoding="utf-8")
    with (output_root / "production_dense_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output_root / 'production_dense_diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
