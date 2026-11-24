#!/usr/bin/env python
"""Replay selected test cases with the exact production solver contract.

The run advances through the natural step that brackets the requested horizon,
reconstructs the requested outputs by adjacent-state interpolation, compares them with the
stored production publications, and records production boundary fluxes for the
two shallow-water references.  The data roots are explicit CLI arguments so a
replay can be moved between machines without editing the script.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsunami_surrogate.data_gen.common_time_v2 import (
    PUBLICATION_SCHEMA_ID,
    stable_hash_payload,
)
from tsunami_surrogate.data_gen.simulate_dataset import (
    DatasetConfig,
    TsunamiDatasetBuilder,
    _block_mean_downsample_spatial,
    _extract_requested_states_from_bracket,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
    _resolved_solver_cfg_for_fde,
)
from tsunami_surrogate.evaluation.alignment import stable_hash_scenario_ids
from tsunami_surrogate.utils.hashing import sha256_file

DEFAULT_RAW_ROOT = Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test/raw")
DEFAULT_CACHE_ROOT = Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test")
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
SOLVER_DIRS = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
}
SCHEMA_ID = "tsunami-surrogate.current-production-dense-diagnostic.v2"
CHECKPOINT_NAME = "production_dense_rows.jsonl"
EXECUTION_MANIFEST_NAME = "production_dense_execution.json"
REPORT_NAME = "production_dense_diagnostic.json"
CSV_NAME = "production_dense_rows.csv"
SOURCES = (
    Path(__file__).resolve().relative_to(ROOT),
    Path("tsunami_surrogate/__init__.py"),
    Path("src/data_gen/common_time_v2.py"),
    Path("src/data_gen/generate_bathymetry.py"),
    Path("src/data_gen/generate_sources.py"),
    Path("src/data_gen/operational_timing.py"),
    Path("src/data_gen/simulate_dataset.py"),
    Path("src/evaluation/alignment.py"),
    Path("src/solver/boundary_conditions.py"),
    Path("src/solver/boussinesq.py"),
    Path("src/solver/hydrostatic_swe.py"),
    Path("src/solver/muscl_hr_swe.py"),
    Path("src/solver/operator_time.py"),
    Path("src/utils/__init__.py"),
    Path("src/utils/hashing.py"),
)


def _solver_from_name(name: str, cfg: Mapping[str, Any]) -> Any:
    if name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(dict(cfg))
    if name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(dict(cfg))
    if name == "boussinesq":
        return _make_boussinesq_solver_from_cfg(dict(cfg))
    raise ValueError(f"Unsupported solver {name!r}")


def _resolve_settings(
    cfg: dict[str, Any],
) -> tuple[DatasetConfig, dict[str, dict[str, Any]]]:
    dataset = TsunamiDatasetBuilder._parse_dataset_section(cfg)
    requested = dataset.requested_output
    paired = dataset.paired_inputs
    if (
        requested is None
        or requested.execution_scope != "production"
        or requested.split != "test"
    ):
        raise ValueError(
            "The replay requires an accepted production test requested-output contract"
        )
    if set(dataset.enabled_fdes) != set(SOLVERS):
        raise ValueError("The production replay requires all three reference solvers")
    if (
        not dataset.buffered_domain.enabled
        or paired.solver_input != "solver"
        or paired.solver_shape is None
    ):
        raise ValueError("The replay requires buffered paired solver-grid inputs")
    solver_shape = tuple(
        size + 2 * dataset.buffered_domain.buffer_cells for size in paired.solver_shape
    )
    base = TsunamiDatasetBuilder._parse_solver_section(cfg)
    solver_configs = {}
    for name in SOLVERS:
        settings = _resolved_solver_cfg_for_fde(base, dataset.solver_profiles, name)
        if (int(settings["nx"]), int(settings["ny"])) != solver_shape:
            raise ValueError(
                f"{name} grid does not match the paired solver shape plus the configured buffer"
            )
        target_cfl = float(settings.get("cfl", dataset.target_cfl))
        if not math.isfinite(target_cfl) or target_cfl <= 0.0:
            raise ValueError(f"Invalid target CFL for {name}: {target_cfl}")
        solver_configs[name] = {**settings, "cfl": target_cfl}
    if not math.isfinite(dataset.sea_level_offset):
        raise ValueError("dataset.sea_level_offset must be finite")
    return dataset, solver_configs


def _mass_flux_arrays(
    solver: Any, state: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
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
    return np.asarray(x_flux[0], dtype=np.float64), np.asarray(
        y_flux[0], dtype=np.float64
    )


def _flux_metrics(
    x_flux: np.ndarray, y_flux: np.ndarray, *, dx: float, dy: float
) -> dict[str, float]:
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
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    denominator = max(
        float(np.linalg.norm(np.asarray(right, dtype=np.float64).ravel())), 1.0e-30
    )
    return float(np.linalg.norm(difference.ravel()) / denominator)


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _case_artifact_paths(
    index: int,
    solver_name: str,
    *,
    raw_root: Path,
    cache_root: Path,
) -> dict[str, Path]:
    raw_path = raw_root / SOLVER_DIRS[solver_name] / "samples" / f"sample_{index:06d}"
    return {
        "cached_bathymetry": cache_root / "bathymetry" / f"sample_{index:06d}.npz",
        "cached_source": cache_root / "sources" / f"sample_{index:06d}.npz",
        **{
            name: raw_path / name
            for name in (
                "sample.npz",
                "provenance.npz",
                "meta.json",
                "publication.json",
            )
        },
    }


def _production_grid(dataset: DatasetConfig) -> dict[str, Any]:
    paired = dataset.paired_inputs
    buffer = dataset.buffered_domain.buffer_cells
    return {
        "solver_shape": [size + 2 * buffer for size in paired.solver_shape],
        "publication_shape": list(paired.target_shape),
        **dataset.buffered_domain.semantics(),
    }


def _load_case_inputs(
    scenario_id: str,
    index: int,
    solver_name: str,
    dataset: DatasetConfig,
    artifact_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, float, str, np.ndarray, np.ndarray, dict[str, Any]]:
    # Validate the same bytes the worker uses, even if a mounted input changes
    # after the parent created the execution manifest.
    contents = {key: path.read_bytes() for key, path in artifact_paths.items()}
    for key, value in contents.items():
        if hashlib.sha256(value).hexdigest() != artifact_hashes[key]:
            raise RuntimeError(
                f"Input changed after execution fingerprinting: {artifact_paths[key]}"
            )

    requested = dataset.requested_output
    identity = {
        "scenario_id": scenario_id,
        "sample_index": index,
        "solver_name": solver_name,
        "split": requested.split,
        "contract_hash": requested.contract_hash,
    }
    publication = _decode_json(contents["publication.json"])
    # Some archived health fields use NaN for unavailable measurements.
    meta = json.loads(contents["meta.json"])
    if publication.get("schema_id") != PUBLICATION_SCHEMA_ID:
        raise RuntimeError(f"{solver_name} {scenario_id}: invalid publication schema")
    if any(
        item.get(key) != value
        for item in (publication, meta)
        for key, value in identity.items()
    ):
        raise RuntimeError(
            f"{solver_name} {scenario_id}: archived identity or output contract mismatch"
        )
    if meta["computational_domain"] != _production_grid(dataset):
        raise RuntimeError(
            f"{solver_name} {scenario_id}: archived computational domain mismatch"
        )
    file_records = {item["name"]: item for item in publication["files"]}
    if len(file_records) != len(publication["files"]):
        raise RuntimeError(
            f"{solver_name} {scenario_id}: duplicate archived publication payloads"
        )
    for name in ("sample.npz", "provenance.npz", "meta.json"):
        archived = file_records.get(name, {})
        if archived.get("sha256") != artifact_hashes[name] or archived.get(
            "size_bytes"
        ) != len(contents[name]):
            raise RuntimeError(f"{solver_name} {scenario_id}: invalid archived {name}")

    with np.load(
        io.BytesIO(contents["cached_bathymetry"]), allow_pickle=False
    ) as bathy_payload:
        bathymetry = np.asarray(bathy_payload["solver_bathymetry"], dtype=np.float32)
        bathy_type = str(bathy_payload["bathymetry_type"].item())
    with np.load(
        io.BytesIO(contents["cached_source"]), allow_pickle=False
    ) as source_payload:
        source_field = np.asarray(
            source_payload["solver_source_field"], dtype=np.float32
        )
        source_strength = float(source_payload["source_strength"].item())
        source_type = str(source_payload["source_type"].item())
    with np.load(io.BytesIO(contents["sample.npz"]), allow_pickle=False) as raw_payload:
        expected_eta = np.asarray(raw_payload["trajectory_eta"], dtype=np.float32)
        archived_times = raw_payload["timestamps"]
        archived_strength = float(raw_payload["source_strength"].item())
    with np.load(
        io.BytesIO(contents["provenance.npz"]), allow_pickle=False
    ) as provenance:
        archived_provenance_times = provenance["requested_timestamps"]
        archived_dt = np.asarray(provenance["natural_dt_history"], dtype=np.float64)
        archived_count = int(provenance["total_natural_steps"].item())
        archived_endpoint = float(provenance["final_natural_timestamp"].item())
    paired = dataset.paired_inputs
    lineage = publication["input_lineage"]
    required_lineage = {
        "lineage_id": paired.lineage_id,
        "master_shape": list(paired.master_shape),
        "solver_shape": list(paired.solver_shape),
        "target_shape": list(paired.target_shape),
        "downsample_method": paired.downsample_method,
        "solver_input": paired.solver_input,
        "source_taper_stage": paired.source_taper_stage,
        "rough_zero_mean_rms_after_taper": paired.rough_zero_mean_rms_after_taper,
        "solver_bathymetry_sha256": _array_hash(bathymetry),
        "solver_source_sha256": _array_hash(source_field),
    }
    if lineage != meta["input_lineage"] or any(
        lineage.get(key) != value for key, value in required_lineage.items()
    ):
        raise RuntimeError(
            f"{solver_name} {scenario_id}: cached solver input lineage mismatch"
        )
    if (
        bathymetry.shape != paired.solver_shape
        or source_field.shape != paired.solver_shape
        or not np.isfinite(bathymetry).all()
        or not np.isfinite(source_field).all()
        or not math.isfinite(source_strength)
        or source_strength != archived_strength
        or source_strength != meta["source_strength"]
        or source_type != meta["source_type"]
        or bathy_type != meta["bathymetry_type"]
    ):
        raise RuntimeError(f"{solver_name} {scenario_id}: invalid cached solver inputs")
    times = requested.requested_times
    if (
        not np.array_equal(archived_times, times)
        or not np.array_equal(archived_provenance_times, times)
        or expected_eta.shape != (times.size, *paired.target_shape)
        or not np.isfinite(expected_eta).all()
        or archived_count < 1
        or archived_dt.shape != (archived_count,)
        or not np.isfinite(archived_dt).all()
        or np.any(archived_dt <= 0.0)
        or not math.isclose(
            archived_endpoint, float(archived_dt.sum()), rel_tol=1.0e-12
        )
        or archived_endpoint < times[-1]
    ):
        raise RuntimeError(
            f"{solver_name} {scenario_id}: invalid archived trajectory or natural-step provenance"
        )
    if any(
        meta.get(key) != publication.get(key) or not publication.get(key)
        for key in ("resolved_config_hash", "code_state_hash")
    ):
        raise RuntimeError(
            f"{solver_name} {scenario_id}: inconsistent archived execution provenance"
        )
    return (
        bathymetry,
        source_field,
        source_strength,
        source_type,
        expected_eta,
        archived_dt,
        {
            "archived_final_natural_timestamp": archived_endpoint,
            "archived_resolved_config_hash": publication["resolved_config_hash"],
            "archived_code_state_hash": publication["code_state_hash"],
        },
    )


def _run_one(
    solver_name: str,
    scenario_id: str,
    index: int,
    solver_cfg: Mapping[str, Any],
    dataset: DatasetConfig,
    artifact_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    requested = dataset.requested_output
    requested_times = requested.requested_times
    horizon = float(requested_times[-1])
    (
        bathymetry,
        source_field,
        source_strength,
        source_type,
        expected_eta,
        archived_dt,
        archived_provenance,
    ) = _load_case_inputs(
        scenario_id, index, solver_name, dataset, artifact_paths, artifact_hashes
    )
    prepared = _prepare_buffered_domain(
        bathymetry,
        source_field,
        source_strength,
        dataset.sea_level_offset,
        dataset.buffered_domain,
        source_type=source_type,
        source_already_tapered=dataset.paired_inputs.source_taper_stage == "master",
    )
    solver = _solver_from_name(solver_name, solver_cfg)
    solver.set_bathymetry(prepared["solver_bathymetry"])
    is_swe = solver_name != "boussinesq"
    if is_swe:
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
    if not np.isfinite(left_state).all():
        raise RuntimeError(f"Non-finite initial state for {solver_name} {scenario_id}")
    initial_mass = (
        float(np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area)
        if is_swe
        else None
    )
    requested_outputs: list[np.ndarray] = []
    weights: list[float] = []
    bracket_widths: list[float] = []
    natural_dt: list[float] = []
    max_abs_flux = 0.0
    max_abs_outward_rate = 0.0
    cumulative_boundary_mass_change = 0.0
    cumulative_residual = 0.0
    requested_mass_change = 0.0
    requested_boundary_mass_change = 0.0
    requested_residual = 0.0
    max_abs_residual = 0.0
    max_cfl = 0.0
    min_depth = float(np.min(solver.h)) if is_swe else None
    max_speed = 0.0 if is_swe else None
    cg_failed_steps = 0
    requested_index = 0
    step = 0

    while requested_index < requested_times.size:
        if step >= requested.max_natural_steps:
            raise RuntimeError(
                f"Natural-step cap reached for {solver_name} {scenario_id} at {current_time}"
            )
        left_time = current_time
        if dataset.auto_dt:
            solver.dt = solver.suggest_dt(target_cfl=solver_cfg["cfl"])
        dt = float(solver.dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"Invalid dt for {solver_name} {scenario_id}: {dt}")
        current_time = left_time + dt
        if not math.isfinite(current_time) or current_time <= left_time:
            raise RuntimeError(
                f"Natural time failed to advance for {solver_name} {scenario_id} at {left_time}"
            )
        if is_swe:
            mass_before = float(
                np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area
            )
            if solver_name == "swe_muscl_hr":
                flux0 = _flux_metrics(
                    *_mass_flux_arrays(solver, left_state), dx=dx, dy=dy
                )
                saved_diagnostics = solver.operator_diagnostics.copy()
                try:
                    stage_state = solver._euler_step_from_state(left_state, dt)
                finally:
                    solver.operator_diagnostics.update(saved_diagnostics)
                flux1 = _flux_metrics(
                    *_mass_flux_arrays(solver, stage_state), dx=dx, dy=dy
                )
                outward_rate = 0.5 * (flux0["outward_rate"] + flux1["outward_rate"])
                max_abs_flux = max(
                    max_abs_flux,
                    flux0["max_abs_boundary_mass_flux"],
                    flux1["max_abs_boundary_mass_flux"],
                )
                max_abs_outward_rate = max(
                    max_abs_outward_rate,
                    abs(flux0["outward_rate"]),
                    abs(flux1["outward_rate"]),
                )
            else:
                flux = _flux_metrics(
                    *_mass_flux_arrays(solver, left_state), dx=dx, dy=dy
                )
                outward_rate = flux["outward_rate"]
                max_abs_flux = max(max_abs_flux, flux["max_abs_boundary_mass_flux"])
                max_abs_outward_rate = max(max_abs_outward_rate, abs(outward_rate))
        solver.step(dt=dt, auto_dt=False)
        step += 1
        natural_dt.append(dt)
        right_state = np.asarray(solver.get_state(), dtype=np.float64).copy()
        if not np.isfinite(right_state).all():
            raise RuntimeError(
                f"Non-finite state for {solver_name} {scenario_id} at step {step}"
            )
        step_cfl = float(solver.compute_cfl(dt=dt))
        if not math.isfinite(step_cfl):
            raise RuntimeError(
                f"Non-finite CFL for {solver_name} {scenario_id} at step {step}"
            )
        max_cfl = max(max_cfl, step_cfl)
        if is_swe:
            min_depth = min(min_depth, float(np.min(solver.h)))
            u, v = solver.compute_velocity()
            max_speed = max(max_speed, float(np.max(np.hypot(u, v))))
        if hasattr(solver, "last_step_cg_failed_count"):
            cg_failed_steps += int(getattr(solver, "last_step_cg_failed_count", 0))

        if is_swe:
            mass_after = float(
                np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area
            )
            flux_mass_change = -dt * outward_rate
            observed_mass_change = mass_after - mass_before
            residual = observed_mass_change - flux_mass_change
            cumulative_boundary_mass_change += flux_mass_change
            cumulative_residual += residual
            max_abs_residual = max(max_abs_residual, abs(residual))
            # Estimate the balance at the requested horizon by interpolating
            # the cumulative budgets across the final natural step.
            fraction = (
                1.0
                if current_time <= horizon
                else (horizon - left_time) / (current_time - left_time)
            )
            requested_mass_change += fraction * observed_mass_change
            requested_boundary_mass_change += fraction * flux_mass_change
            requested_residual += fraction * residual

        bracket_end = int(np.searchsorted(requested_times, current_time, side="right"))
        if bracket_end > requested_index:
            states, provenance = _extract_requested_states_from_bracket(
                left_state=left_state,
                right_state=right_state,
                left_time=left_time,
                right_time=current_time,
                requested_times=requested_times[requested_index:bracket_end],
                right_natural_step_index=step,
            )
            eta = states[:, 0]
            if is_swe:
                eta = eta + prepared["solver_bathymetry"][None, ...]
            crop = prepared["crop"]
            published = _block_mean_downsample_spatial(
                eta[:, crop[0], crop[1]], dataset.paired_inputs.target_shape
            ).astype(np.float32)
            requested_outputs.append(published)
            weights.extend(provenance["interpolation_weights"])
            bracket_widths.extend(provenance["bracket_widths"])
            requested_index = bracket_end
        left_state = right_state

    published_eta = np.concatenate(requested_outputs, axis=0)
    difference = np.asarray(published_eta, dtype=np.float64) - np.asarray(
        expected_eta, dtype=np.float64
    )
    generated_dt = np.asarray(natural_dt, dtype=np.float64)
    matching_step_count = archived_dt.shape == generated_dt.shape
    dt_diff = (
        float(np.max(np.abs(archived_dt - generated_dt)))
        if matching_step_count
        else None
    )
    final_mass = (
        float(np.sum(np.asarray(solver.h, dtype=np.float64)) * cell_area)
        if is_swe
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
        "requested_horizon": horizon,
        "final_natural_timestamp": current_time,
        "natural_overshoot": current_time - horizon,
        "natural_step_count": int(step),
        "natural_dt_min": float(np.min(generated_dt)),
        "natural_dt_max": float(np.max(generated_dt)),
        "archived_natural_step_count": int(archived_dt.size),
        "natural_step_count_matches_archive": matching_step_count,
        "natural_dt_max_abs_difference": dt_diff,
        "replay_max_abs": float(np.max(np.abs(difference))),
        "replay_relative_l2": _array_rel_l2(published_eta, expected_eta),
        "replay_rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_cfl_seen": float(max_cfl),
        "target_cfl": float(solver_cfg["cfl"]),
        "min_depth_seen": min_depth,
        "max_speed_seen": max_speed,
        "cg_failed_steps": int(cg_failed_steps),
        "finite_final_state": bool(np.isfinite(solver.get_state()).all()),
        "initial_mass": initial_mass,
        "final_mass": final_mass,
        "mass_relative_change": mass_relative_change,
        "max_interpolation_bracket": float(max(bracket_widths)),
        "median_interpolation_bracket": float(np.median(bracket_widths)),
        "max_interpolation_weight": float(max(weights)),
        "min_interpolation_weight": float(min(weights)),
        "max_abs_boundary_mass_flux": float(max_abs_flux)
        if solver_name != "boussinesq"
        else None,
        "max_abs_outward_mass_rate": float(max_abs_outward_rate)
        if solver_name != "boussinesq"
        else None,
        "cumulative_boundary_mass_change": float(cumulative_boundary_mass_change)
        if solver_name != "boussinesq"
        else None,
        "cumulative_boundary_mass_change_relative": cumulative_boundary_mass_change_relative,
        "cumulative_sponge_projection_residual": float(cumulative_residual)
        if solver_name != "boussinesq"
        else None,
        "cumulative_sponge_projection_residual_relative": cumulative_residual_relative,
        "max_abs_sponge_projection_residual_per_step": float(max_abs_residual)
        if solver_name != "boussinesq"
        else None,
        "requested_horizon_mass_change_estimate": requested_mass_change
        if is_swe
        else None,
        "requested_horizon_boundary_mass_change_estimate": requested_boundary_mass_change
        if is_swe
        else None,
        "requested_horizon_sponge_projection_residual_estimate": requested_residual
        if is_swe
        else None,
        "requested_horizon_sponge_projection_residual_relative_estimate": (
            requested_residual / max(abs(initial_mass), 1.0e-30) if is_swe else None
        ),
        **archived_provenance,
        "output_hash": _array_hash(published_eta),
        "elapsed_s": float(time.monotonic() - started),
    }


def _load_selection(
    payload: Mapping[str, Any], limit: int | None
) -> list[tuple[str, int]]:
    selection = payload["dense_validation"]
    scenario_ids = [row["scenario_id"] for row in selection["ordered_scenarios"]]
    if not scenario_ids or any(
        not re.fullmatch(r"scenario_[0-9]{6}", case) for case in scenario_ids
    ):
        raise ValueError("Selection must contain valid scenario IDs")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Selection contains duplicate scenario IDs")
    if selection["count"] != len(scenario_ids) or selection[
        "list_hash"
    ] != stable_hash_scenario_ids(scenario_ids):
        raise ValueError("Selection count or ordered scenario hash mismatch")
    return [
        (scenario_id, int(scenario_id.rsplit("_", 1)[1]))
        for scenario_id in scenario_ids[:limit]
    ]


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64
    )
    return {
        "count": int(values.size),
        "median": float(np.median(values)) if values.size else None,
        "p95": float(np.quantile(values, 0.95)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
    }


def _decode_json(data: str | bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant: {value}")

    return json.loads(data, parse_constant=reject_constant)


def _write_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _fingerprint(kind: str, payload: Any) -> str:
    return stable_hash_payload(artifact_kind=kind, payload=payload, schema_id=SCHEMA_ID)


def _source_hashes() -> dict[str, str]:
    return {path.as_posix(): sha256_file(ROOT / path) for path in SOURCES}


def _prepare_execution(
    args: argparse.Namespace, dataset: DatasetConfig, solver_configs: Mapping[str, Any]
) -> tuple:
    requested = dataset.requested_output
    raw_root = args.raw_root.resolve()
    cache_root = args.cache_root.resolve()
    selection_path = (ROOT / args.selection).resolve()
    selection_bytes = selection_path.read_bytes()
    selection_payload = _decode_json(selection_bytes)
    selected = _load_selection(selection_payload, args.max_scenarios)
    if any(index > dataset.num_samples for _, index in selected):
        raise ValueError("Selected sample index exceeds the configured dataset")
    expected = {
        (solver, index): scenario_id
        for scenario_id, index in selected
        for solver in SOLVERS
    }
    artifact_paths = {
        key: _case_artifact_paths(
            key[1], key[0], raw_root=raw_root, cache_root=cache_root
        )
        for key in expected
    }
    file_hashes = {
        path: sha256_file(path)
        for path in dict.fromkeys(
            path for paths in artifact_paths.values() for path in paths.values()
        )
    }
    artifact_hashes = {
        key: {name: file_hashes[path] for name, path in paths.items()}
        for key, paths in artifact_paths.items()
    }
    execution_contract = {
        "config_path": str((ROOT / args.config).resolve()),
        "config_sha256": args.config_sha256,
        "selection_path": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "selection_audit_hash": selection_payload.get("audit_hash"),
        "selection_list_hash": selection_payload["dense_validation"]["list_hash"],
        "selected_scenarios": [
            {"scenario_id": case, "sample_index": index} for case, index in selected
        ],
        "solver_configs": solver_configs,
        "auto_dt": dataset.auto_dt,
        "sea_level_offset": dataset.sea_level_offset,
        "requested_output_contract": requested.contract,
        "requested_output_contract_hash": requested.contract_hash,
        "max_natural_steps": requested.max_natural_steps,
        "production_grid": _production_grid(dataset),
        "paired_inputs": dataset.paired_inputs.semantics(),
        "data_roots": {"raw_root": str(raw_root), "cache_root": str(cache_root)},
        "source_sha256": _source_hashes(),
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "dependencies": {
                name: version(name) for name in ("numpy", "scipy", "PyYAML")
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                )
            },
        },
        "selected_input_sha256": {
            case: {solver: artifact_hashes[solver, index] for solver in SOLVERS}
            for case, index in selected
        },
    }
    execution_manifest = {
        "schema_id": SCHEMA_ID,
        "execution_fingerprint": _fingerprint("execution-contract", execution_contract),
        "execution_contract": execution_contract,
    }
    return execution_manifest, expected, artifact_paths, artifact_hashes


def _validate_row(
    row: Mapping[str, Any],
    execution: Mapping[str, Any],
    expected: Mapping[tuple[str, int], str],
    completed_keys: set[tuple[str, int]],
) -> tuple[str, int]:
    if (
        row.get("schema_id") != SCHEMA_ID
        or row.get("execution_fingerprint") != execution["execution_fingerprint"]
    ):
        raise RuntimeError("Checkpoint row execution fingerprint mismatch")
    if (
        not isinstance(row.get("solver"), str)
        or type(row.get("sample_index")) is not int
    ):
        raise RuntimeError("Checkpoint row has invalid solver/sample identity")
    key = (row["solver"], row["sample_index"])
    if key not in expected or expected[key] != row.get("scenario_id"):
        raise RuntimeError(
            f"Checkpoint row outside the selected solver/scenario set: {key}"
        )
    if key in completed_keys:
        raise RuntimeError(f"Duplicate checkpoint row: {key}")
    row_content = {name: value for name, value in row.items() if name != "row_sha256"}
    if row.get("row_sha256") != _fingerprint("checkpoint-row", row_content):
        raise RuntimeError(f"Checkpoint row checksum mismatch: {key}")
    return key


def _load_checkpoint(
    output_root: Path,
    execution: Mapping[str, Any],
    expected: Mapping[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    manifest_path = output_root / EXECUTION_MANIFEST_NAME
    checkpoint_path = output_root / CHECKPOINT_NAME
    if manifest_path.is_file():
        archived_execution = _decode_json(manifest_path.read_bytes())
        if archived_execution != execution:
            raise RuntimeError(
                "Replay execution manifest mismatch. Use a new output root for different inputs or code."
            )
    else:
        if any(
            (output_root / name).exists()
            for name in (CHECKPOINT_NAME, REPORT_NAME, CSV_NAME, "failures.json")
        ):
            raise RuntimeError(
                "Replay artifacts lack an execution manifest. Use a new output root."
            )
        _write_json(manifest_path, execution)
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, int]] = set()
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or not line.endswith("\n"):
                    raise RuntimeError(
                        "Malformed checkpoint row or incomplete final line"
                    )
                row = _decode_json(line)
                completed_keys.add(
                    _validate_row(row, execution, expected, completed_keys)
                )
                rows.append(row)
    return rows, completed_keys


def _run_replay(args: argparse.Namespace, output_root: Path) -> None:
    config_bytes = (ROOT / args.config).read_bytes()
    args.config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    dataset, solver_configs = _resolve_settings(yaml.safe_load(config_bytes))
    execution, expected, artifact_paths, artifact_hashes = _prepare_execution(
        args, dataset, solver_configs
    )
    rows, completed_keys = _load_checkpoint(output_root, execution, expected)
    resumed_count = len(rows)
    tasks = [key for key in expected if key not in completed_keys]
    failures: list[str] = []
    print(
        f"tasks={len(tasks)} workers={args.workers} resumed={len(rows)} output={output_root}",
        flush=True,
    )
    if tasks:
        with (
            (output_root / CHECKPOINT_NAME).open(
                "a", encoding="utf-8"
            ) as checkpoint_handle,
            ProcessPoolExecutor(max_workers=args.workers) as pool,
        ):
            futures = {
                pool.submit(
                    _run_one,
                    solver,
                    expected[key],
                    index,
                    solver_configs[solver],
                    dataset,
                    artifact_paths[key],
                    artifact_hashes[key],
                ): key
                for key in tasks
                for solver, index in (key,)
            }
            for position, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    row = future.result()
                    row.update(
                        schema_id=SCHEMA_ID,
                        execution_fingerprint=execution["execution_fingerprint"],
                    )
                    row["row_sha256"] = _fingerprint("checkpoint-row", row)
                    if _validate_row(row, execution, expected, completed_keys) != key:
                        raise RuntimeError(f"Unexpected worker result for {key}")
                    encoded = json.dumps(row, sort_keys=True, allow_nan=False)
                except Exception as exc:  # noqa: BLE001 - record each worker failure and continue the audit
                    failures.append(f"{key}: {exc!r}")
                    print(
                        f"[{position}/{len(tasks)}] FAILED {key}: {exc!r}", flush=True
                    )
                    continue
                checkpoint_handle.write(encoded + "\n")
                checkpoint_handle.flush()
                os.fsync(checkpoint_handle.fileno())
                completed_keys.add(key)
                rows.append(row)
                print(
                    f"[{position}/{len(tasks)}] {row['solver']} {row['scenario_id']} "
                    f"steps={row['natural_step_count']} replay_rel={row['replay_relative_l2']:.3e}",
                    flush=True,
                )

    if failures:
        _write_json(output_root / "failures.json", failures)
        raise SystemExit(f"{len(failures)} production replay failures")
    if completed_keys != set(expected) or len(rows) != len(expected):
        raise RuntimeError(
            "Production replay does not have exact solver/scenario coverage"
        )
    rows.sort(key=lambda row: (row["solver"], row["sample_index"]))
    by_solver = {}
    for solver_name in SOLVERS:
        subset = [row for row in rows if row["solver"] == solver_name]
        by_solver[solver_name] = {
            "scenario_count": len(subset),
            "finite_final_state_count": int(
                sum(bool(row["finite_final_state"]) for row in subset)
            ),
            "replay_relative_l2": _summary(subset, "replay_relative_l2"),
            "replay_max_abs": _summary(subset, "replay_max_abs"),
            "natural_step_count": _summary(subset, "natural_step_count"),
            "natural_step_count_matches_archive_count": sum(
                row["natural_step_count_matches_archive"] for row in subset
            ),
            "natural_dt_max_abs_difference": _summary(
                subset, "natural_dt_max_abs_difference"
            ),
            "natural_overshoot": _summary(subset, "natural_overshoot"),
            "max_interpolation_bracket": _summary(subset, "max_interpolation_bracket"),
            "max_cfl_seen": _summary(subset, "max_cfl_seen"),
            "cg_failed_steps": int(sum(int(row["cg_failed_steps"]) for row in subset)),
        }
        if solver_name != "boussinesq":
            by_solver[solver_name].update(
                {
                    "max_abs_boundary_mass_flux": _summary(
                        subset, "max_abs_boundary_mass_flux"
                    ),
                    "max_abs_outward_mass_rate": _summary(
                        subset, "max_abs_outward_mass_rate"
                    ),
                    "abs_cumulative_sponge_projection_residual": _summary(
                        [
                            {
                                **row,
                                "abs_cumulative_sponge_projection_residual": abs(
                                    float(row["cumulative_sponge_projection_residual"])
                                ),
                            }
                            for row in subset
                        ],
                        "abs_cumulative_sponge_projection_residual",
                    ),
                    "abs_cumulative_sponge_projection_residual_relative": _summary(
                        [
                            {
                                **row,
                                "abs_cumulative_sponge_projection_residual_relative": abs(
                                    float(
                                        row[
                                            "cumulative_sponge_projection_residual_relative"
                                        ]
                                    )
                                ),
                            }
                            for row in subset
                        ],
                        "abs_cumulative_sponge_projection_residual_relative",
                    ),
                    "abs_requested_horizon_sponge_projection_residual_relative_estimate": _summary(
                        [
                            {
                                "abs_residual": abs(
                                    row[
                                        "requested_horizon_sponge_projection_residual_relative_estimate"
                                    ]
                                )
                            }
                            for row in subset
                        ],
                        "abs_residual",
                    ),
                }
            )

    contract = execution["execution_contract"]
    if _source_hashes() != contract["source_sha256"]:
        raise RuntimeError("Replay source files changed during the execution")
    manifest = {
        **execution,
        "diagnostic": "production_natural_step_replay_and_boundary_flux",
        "config_path": contract["config_path"],
        "selection_path": contract["selection_path"],
        "config_sha256": contract["config_sha256"],
        "selection_audit_hash": contract["selection_audit_hash"],
        "selection_list_hash": contract["selection_list_hash"],
        "script_sha256": contract["source_sha256"][SOURCES[0].as_posix()],
        "selection_count": len(contract["selected_scenarios"]),
        "solver_count": len(SOLVERS),
        "expected_task_count": len(expected),
        "task_count": len(tasks),
        "resumed_row_count": resumed_count,
        "rows_written": len(rows),
        "production_grid": contract["production_grid"],
        "data_roots": contract["data_roots"],
        "requested_times": dataset.requested_output.requested_times.tolist(),
        "boundary_budget": {
            "natural_step_metrics": "evaluated through final_natural_timestamp",
            "requested_horizon_estimate": "linearly interpolates cumulative mass change, boundary transport, and residual "
            "across the final natural step; the solver takes the full natural step",
        },
        "by_solver": by_solver,
        "rows": rows,
    }
    temporary_csv = output_root / f"{CSV_NAME}.tmp"
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(output_root / CSV_NAME)
    _write_json(output_root / REPORT_NAME, manifest)
    (output_root / "failures.json").unlink(missing_ok=True)
    print(f"saved={output_root / REPORT_NAME}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/dataset_test.yaml")
    parser.add_argument(
        "--selection", default="configs/eval/common_time_validation_scenarios.json"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/Windows/Dataset/current_production_dense_validation_20261117/production_replay"
        ),
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.max_scenarios is not None and args.max_scenarios < 1):
        parser.error("--workers and --max-scenarios must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another replay is writing to {output_root}") from exc
        _run_replay(args, output_root)
    finally:
        os.close(directory_fd)


if __name__ == "__main__":
    main()
