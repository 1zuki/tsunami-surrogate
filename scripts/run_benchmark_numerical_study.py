#!/usr/bin/env python
"""Measure full-horizon numerical sensitivities on frozen test inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for name in THREAD_ENV:
    os.environ[name] = "1"

import numpy as np
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsunami_surrogate.data_gen.common_time import hash_array, parse_requested_output_config
from tsunami_surrogate.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    QualityPolicy,
    _block_mean_downsample_spatial,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
    _quality_violations_for_health,
    _resolved_solver_cfg_for_fde,
    _simulate_requested_times_local,
)
from tsunami_surrogate.utils.hashing import sha256_file, stable_json_sha256

SCHEMA = "tsunami-surrogate.benchmark-numerical-study.v1"
DEFAULT_CONFIG = ROOT / "configs/eval/benchmark_numerical_study.yaml"
FACTORIES = {
    "swe_hydrostatic": _make_hydrostatic_solver_from_cfg,
    "swe_muscl_hr": _make_muscl_solver_from_cfg,
}
VARIANTS = (
    "baseline", "coarse", "fine", "common_cfl", "half_common_cfl",
    "no_sponge", "extended_no_sponge", "open_no_sponge", "lake_rest", "moving_equilibrium",
)
BOUNDARY_VARIANTS = ("no_sponge", "extended_no_sponge", "open_no_sponge")
HEALTH_COUNTERS = (
    "operator_nan_to_num_replacement_count",
    "operator_positivity_projection_count",
    "operator_dry_projection_count",
    "operator_muscl_cell_velocity_clip_count",
    "operator_muscl_face_velocity_clip_count",
)


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return data


def _write_atomic(path: Path, writer) -> None:
    """Publish the whole file atomically and fail if its destination exists."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def _write_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _write_atomic(path, lambda stream: stream.write(encoded))


def _file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _check_file(record: dict) -> Path:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Input or code changed after study planning: {path}")
    return path


def _load_roster(root: Path, study: dict, dataset: dict) -> tuple[list[dict], list[dict]]:
    manifest_path = root / "synthetic/scenario_manifest.jsonl"
    inventory_path = root / "synthetic/native_input_inventory.jsonl"
    records = [_file_record(manifest_path), _file_record(inventory_path)]
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    indices = [row["sample_index"] for row in rows]
    if len(rows) != dataset["dataset"]["num_samples"] or len(set(indices)) != len(rows):
        raise ValueError("Unexpected test manifest size or duplicate indices")
    first = {}
    for row in sorted(rows, key=lambda row: row["sample_index"]):
        first.setdefault((row["bathymetry_type"], row["source_type"]), row)

    pairs = {(row["bathymetry"], row["source"]) for row in study["cases"]}
    manifest_beds = {bed for bed, _ in first}
    manifest_sources = {source for _, source in first}
    if (len(study["cases"]) != 6 or len(pairs) != 6
        or {bed for bed, _ in pairs} != manifest_beds
        or {source for _, source in pairs} != manifest_sources):
        raise ValueError("Frozen pairs must cover the test-manifest bathymetry and source families")

    cases = []
    for case in study["cases"]:
        row = first[case["bathymetry"], case["source"]]
        if row["sample_index"] != case["index"]:
            raise ValueError(f"Frozen case is no longer the first matching manifest row: {case}")
        lineage = row["input_lineage"]
        if (lineage["inventory_sha256"] != records[1]["sha256"]
            or lineage["lineage_id"] != dataset["paired_inputs"]["lineage_id"]
            or lineage["solver_shape"] != [study["baseline_core"]] * 2):
            raise ValueError(f"Input lineage mismatch in {row['scenario_id']}")
        sample = f"sample_{case['index']:06d}"
        files = {kind: _file_record(root / kind / f"{sample}.npz") for kind in ("bathymetry", "sources")}
        publications = {}
        for solver_name, directory in (("swe_hydrostatic", "hydrostatic"), ("swe_muscl_hr", "muscl_hr")):
            folder = root / "raw" / directory / "samples" / sample
            publications[solver_name] = {
                "sample": _file_record(folder / "sample.npz"),
                "provenance": _file_record(folder / "provenance.npz"),
            }
        cases.append({**case, "row": row, "inputs": files, "publications": publications})
    return cases, records


def _load_arrays(case: dict, core: int, *, lake_rest: bool) -> tuple[np.ndarray, np.ndarray, float]:
    row = case["row"]
    with np.load(_check_file(case["inputs"]["bathymetry"]), allow_pickle=False) as payload:
        bathymetry = payload["solver_bathymetry"]
        bathymetry_type = str(payload["bathymetry_type"].item())
    with np.load(_check_file(case["inputs"]["sources"]), allow_pickle=False) as payload:
        source = payload["solver_source_field"]
        source_type = str(payload["source_type"].item())
        strength = float(payload["source_strength"].item())
    for kind, values in (("bathymetry", bathymetry), ("source", source)):
        if values.shape != (core, core) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen {kind} array for {row['scenario_id']}")
        if hash_array(values)["sha256"] != row["input_lineage"][f"solver_{kind}_sha256"]:
            raise ValueError(f"Frozen {kind} hash disagrees with manifest for {row['scenario_id']}")
    if (bathymetry_type != row["bathymetry_type"] or source_type != row["source_type"]
        or strength != float(row["source_strength"])):
        raise ValueError(f"Frozen input metadata disagrees with manifest for {row['scenario_id']}")
    return bathymetry, source, 0.0 if lake_rest else strength


def _resample(values: np.ndarray, old_core: int, new_core: int) -> np.ndarray:
    if new_core >= old_core and new_core % old_core == 0:
        factor = new_core // old_core
        return values if factor == 1 else values.repeat(factor, axis=0).repeat(factor, axis=1)
    if new_core > old_core or old_core % new_core:
        raise ValueError("The computational grids must be integer nested")
    factor = old_core // new_core
    return _block_mean_downsample_spatial(values, tuple(size // factor for size in values.shape))


def _steady_depth(bed: np.ndarray, discharge: float, head: float, gravity: float) -> np.ndarray:
    """Solve the positive subcritical branch of the steady SWE energy equation."""
    if gravity <= 0 or discharge == 0 or not np.isfinite([gravity, discharge, head]).all():
        raise ValueError("Expected a finite nonzero discharge, head, and positive gravity")
    critical_depth = (discharge**2 / gravity) ** (1 / 3)
    lower = np.full_like(bed, critical_depth, dtype=np.float64)
    upper = head - bed
    minimum_head = 1.5 * critical_depth + bed
    if not np.all(head > minimum_head):
        raise ValueError("Requested steady-flow parameters have no strict subcritical branch")
    for _ in range(60):
        middle = (lower + upper) / 2.0
        residual = middle + bed + discharge**2 / (2.0 * gravity * middle**2) - head
        lower = np.where(residual < 0, middle, lower)
        upper = np.where(residual >= 0, middle, upper)
    return (lower + upper) / 2.0


def _metrics(left: np.ndarray, reference: np.ndarray) -> dict:
    if left.shape != reference.shape or left.ndim != 3:
        raise ValueError("Expected same-shape trajectories with [time, x, y] axes")
    difference = left - reference
    frame_ss = np.sum(difference**2, axis=(1, 2))
    ref_ss = np.sum(reference**2, axis=(1, 2))
    return {
        "trajectory_relative_l2": float(np.sqrt(sum(frame_ss) / sum(ref_ss))) if sum(ref_ss) else None,
        "trajectory_rmse": float(np.sqrt(np.mean(difference**2))),
        "per_time_rmse": np.sqrt(frame_ss / (left.shape[1] * left.shape[2])).tolist(),
        "per_time_relative_l2": [
            float(np.sqrt(numerator / denominator)) if denominator else None
            for numerator, denominator in zip(frame_ss, ref_ss)
        ],
        "max_abs_error": float(np.max(np.abs(difference))),
    }


def _publication(case: dict, solver_name: str, contract_hash: str, times: np.ndarray, core: int):
    files = case["publications"][solver_name]
    with np.load(_check_file(files["sample"]), allow_pickle=False) as payload:
        checks = {
            "scenario_id": case["row"]["scenario_id"], "solver_name": solver_name,
            "split": "test", "contract_hash": contract_hash,
        }
        for key, expected in checks.items():
            if str(payload[key].item()) != expected:
                raise ValueError(f"Stored publication {key} mismatch for {case['row']['scenario_id']}")
        expected = payload["trajectory_eta"].astype(np.float64)
        if not np.array_equal(payload["timestamps"], times):
            raise ValueError("Stored publication has different requested timestamps")
    if expected.shape != (len(times), core, core) or not np.isfinite(expected).all():
        raise ValueError("Stored publication has invalid shape or values")
    with np.load(_check_file(files["provenance"]), allow_pickle=False) as provenance:
        if not np.array_equal(provenance["requested_timestamps"], times):
            raise ValueError("Stored provenance has different requested timestamps")
        dt = provenance["natural_dt_history"]
    return expected, dt


def _state_budget(
    states: np.ndarray, initial: np.ndarray, bed: np.ndarray, solver_cfg: dict,
) -> dict:
    """Integrate mass per density and mechanical energy per density over the full domain."""
    cell_area = solver_cfg["dx"] * solver_cfg["dy"]
    gravity = solver_cfg["g"]
    dry_tolerance = solver_cfg["dry_tolerance"]

    def totals(state: np.ndarray) -> tuple[float, float]:
        h, hu, hv = state
        kinetic = np.divide(hu**2 + hv**2, 2 * h, out=np.zeros_like(h), where=h > dry_tolerance)
        potential = gravity * h * (0.5 * h + bed)
        return float(cell_area * np.sum(h)), float(cell_area * np.sum(kinetic + potential))

    initial_volume, initial_energy = totals(initial)
    requested_totals = np.array([totals(state) for state in states])
    volume_change = requested_totals[:, 0] - initial_volume
    energy_change = requested_totals[:, 1] - initial_energy
    return {
        "initial_water_volume": initial_volume,
        "initial_mechanical_energy_per_density": initial_energy,
        "water_volume_change_by_time": volume_change.tolist(),
        "mechanical_energy_per_density_change_by_time": energy_change.tolist(),
        "water_volume_max_abs_change": float(np.max(np.abs(volume_change))),
        "mechanical_energy_per_density_max_abs_change": float(np.max(np.abs(energy_change))),
    }


def _solve(task: dict) -> tuple[np.ndarray, dict]:
    started = time.monotonic()
    name = task["solver"]
    variant = task["variant"]
    solver_cfg = task["solver_config"]
    dataset = task["dataset"]
    core = task["core"]
    times = np.asarray(task["times"], dtype=np.float64)
    moving = variant == "moving_equilibrium"
    lake_rest = variant == "lake_rest"
    case = task["case"]

    if moving:
        p = task["moving_equilibrium"]
        x = (np.arange(core) + 0.5) / core
        bed_line = p["mean_bed"] + p["bed_amplitude"] * np.cos(2 * np.pi * x)
        bed = np.repeat(bed_line[:, None], p["transverse_cells"], axis=1)
        h0 = _steady_depth(bed, p["discharge"], p["bernoulli_head"], solver_cfg["g"])
        hu0 = np.full_like(h0, p["discharge"])
        energy = h0 + bed + hu0**2 / (2 * solver_cfg["g"] * h0**2)
        equilibrium = {
            "initial_max_bernoulli_head_residual": float(np.max(np.abs(energy - p["bernoulli_head"]))),
            "initial_max_froude": float(np.max(np.abs(hu0) / (h0 * np.sqrt(solver_cfg["g"] * h0)))),
        }
        crop = (slice(None), slice(None))
    else:
        baseline_core = task["baseline_core"]
        domain = BufferedDomainConfig(**dataset["computational_domain"])
        bathymetry, source, strength = _load_arrays(case, baseline_core, lake_rest=lake_rest)
        prepared = _prepare_buffered_domain(
            bathymetry, source, strength, dataset["dataset"]["sea_level_offset"],
            domain, source_already_tapered=True,
        )
        bed = _resample(prepared["solver_bathymetry"], baseline_core, core)
        h0 = _resample(prepared["solver_h0"], baseline_core, core)
        width = task["buffer_cells"]
        extra = width - domain.buffer_cells * core // baseline_core
        if extra:
            bed = np.pad(bed, extra, mode="edge")
            h0 = np.pad(h0, extra, mode="edge")
        crop = (slice(width, width + core), slice(width, width + core))
        hu0 = np.zeros_like(h0)
        equilibrium = {}

    solver = FACTORIES[name](solver_cfg)
    solver.set_bathymetry(bed)
    solver.set_initial_condition(h0, hu0=hu0, hv0=np.zeros_like(h0))
    initial_state = solver.get_state()
    states, actual_times, dt, diagnostics = _simulate_requested_times_local(
        solver, auto_dt=True, target_cfl=solver_cfg["cfl"], requested_times=times,
        max_natural_steps=task["max_natural_steps"], collect_natural_step_health=True,
        requested_state_dtype=np.float64,
    )
    if not np.array_equal(actual_times, times):
        raise RuntimeError("Requested rollout returned a different time grid")
    eta_full = states[:, 0] + bed[None, ...]
    health = {
        "fde_name": name,
        "nan_count": 0,
        "inf_count": 0,
        "natural_step_count": len(dt),
        "natural_dt_sha256": hash_array(dt)["sha256"],
        "final_natural_timestamp": float(diagnostics["final_natural_timestamp"].item()),
        "target_cfl": solver_cfg["cfl"],
        "max_pre_step_cfl": float(np.max(diagnostics["pre_step_cfl"])),
        "max_post_step_cfl": float(np.max(diagnostics["post_step_cfl"])),
        "min_h": float(np.min(diagnostics["swe_min_depth"])),
        "max_abs_velocity": float(np.max(diagnostics["swe_max_speed"])),
        "max_abs_eta": float(np.max(np.abs(eta_full))),
        "dt_min": float(np.min(dt)),
        "dt_max": float(np.max(dt)),
        "bracket_width_max": float(np.max(diagnostics["bracket_widths"])),
        "exact_requested_knots": int(np.count_nonzero(diagnostics["exact_knot"])),
    }
    for key in HEALTH_COUNTERS:
        if key in diagnostics:
            health[key] = int(np.max(diagnostics[key]))
    violations = _quality_violations_for_health(health, QualityPolicy(**dataset["quality"]))
    if violations:
        raise RuntimeError(f"{task['id']} failed rollout quality: {violations}")

    if moving or lake_rest:
        fields = {
            "elevation": (eta_full, h0 + bed),
            "x_discharge": (states[:, 1], hu0),
            "y_discharge": (states[:, 2], np.zeros_like(h0)),
        }
        for field, (values, initial) in fields.items():
            per_time = np.max(np.abs(values - initial), axis=(1, 2))
            equilibrium[f"{field}_max_abs_drift"] = float(np.max(per_time))
            equilibrium[f"{field}_max_abs_drift_by_time"] = per_time.tolist()
        equilibrium["natural_step_max_speed"] = health["max_abs_velocity"]

    eta = eta_full[:, crop[0], crop[1]]
    if not moving:
        eta = _block_mean_downsample_spatial(eta, (task["publication_core"],) * 2)
    result = {"health": health, "state_budget": _state_budget(states, initial_state, bed, solver_cfg)}
    if moving or lake_rest:
        result["equilibrium"] = equilibrium
    if variant == "baseline":
        expected, archived_dt = _publication(
            case, name, task["contract_hash"], times, task["publication_core"],
        )
        result["against_archived_publication"] = _metrics(eta, expected)
        result["archived_natural_steps"] = len(archived_dt)
        result["archived_natural_history_exact"] = bool(np.array_equal(dt, archived_dt))
    result["elapsed_s"] = time.monotonic() - started
    return eta, result


def _make_task(variant: str, name: str, case: dict | None, shared: dict, core: int) -> dict:
    study = shared["study"]
    dataset = shared["dataset"]
    cfg = _resolved_solver_cfg_for_fde(dataset["solver"], dataset["solver_profiles"], name)
    ratio = core / study["baseline_core"]
    width = int(dataset["computational_domain"]["buffer_cells"] * ratio)
    if variant == "extended_no_sponge":
        width = study["extended_buffer_cells"]
    if variant == "moving_equilibrium":
        dx = study["moving_equilibrium"]["length"] / core
        cfg.update(nx=core, ny=study["moving_equilibrium"]["transverse_cells"],
                   dx=dx, dy=dx, boundary="periodic", use_sponge=False)
    else:
        cfg.update(nx=core + 2 * width, ny=core + 2 * width, dx=cfg["dx"] / ratio,
                   dy=cfg["dy"] / ratio, sponge_width=int(cfg["sponge_width"] * ratio))
    if variant in BOUNDARY_VARIANTS:
        cfg["use_sponge"] = False
    if variant == "open_no_sponge":
        cfg["boundary"] = "open"
    if variant in ("common_cfl", "half_common_cfl"):
        cfg["cfl"] = study["common_cfl"] / (2 if variant == "half_common_cfl" else 1)
    case_id = case["row"]["scenario_id"] if case else f"periodic_grid_{core}"
    return {
        "id": f"{case_id}__{name}__{variant}",
        "variant": variant, "solver": name, "case": case, "dataset": dataset, "core": core,
        "times": shared["times"], "contract_hash": shared["contract_hash"],
        "max_natural_steps": shared["max_natural_steps"], "buffer_cells": width,
        "baseline_core": study["baseline_core"], "publication_core": study["publication_core"],
        "moving_equilibrium": study["moving_equilibrium"], "solver_config": cfg,
    }


def _compare(entries: dict, case_id: str, name: str, left_variant: str, reference_variant: str,
             category: str, *, reference_solver: str | None = None) -> dict | None:
    reference_solver = reference_solver or name
    left = f"{case_id}__{name}__{left_variant}"
    reference = f"{case_id}__{reference_solver}__{reference_variant}"
    if left in entries and reference in entries:
        return {
            "case_id": case_id, "category": category, "left": left, "reference": reference,
            **_metrics(entries[left][0], entries[reference][0]),
        }
    return None


def _summary(plan: dict, entries: dict, fingerprint: str) -> dict:
    comparisons = []
    for case in plan["roster"]:
        case_id = case["row"]["scenario_id"]
        for name in FACTORIES:
            for left, reference, category in (
                ("coarse", "baseline", "spatial_self_difference"),
                ("baseline", "fine", "spatial_self_difference"),
                ("coarse", "fine", "spatial_self_difference"),
                ("baseline", "common_cfl", "production_to_common_cfl"),
                ("common_cfl", "half_common_cfl", "half_common_cfl"),
                ("baseline", "no_sponge", "sponge_on_off"),
                ("no_sponge", "extended_no_sponge", "buffer_extent_and_extended_exterior"),
                ("no_sponge", "open_no_sponge", "radiation_to_open"),
            ):
                row = _compare(entries, case_id, name, left, reference, category)
                if row is not None:
                    comparisons.append(row)
        for variant in VARIANTS:
            if variant in {"lake_rest", "moving_equilibrium"}:
                continue
            hydro, muscl = FACTORIES
            row = _compare(entries, case_id, hydro, variant, variant, "between_swe_solvers",
                           reference_solver=muscl)
            if row is not None:
                comparisons.append(row)
    return {
        "schema_id": SCHEMA,
        "plan_sha256": fingerprint,
        "complete_full_planned_study": plan["complete_full_scope"],
        "rollout_count": len(entries),
        "full_scope_rollout_count": plan["full_scope_rollout_count"],
        "requested_timestamps": plan["requested_timestamps"],
        "rollouts": {name: entries[name][1] for name in sorted(entries)},
        "comparisons": comparisons,
    }


def _read_checkpoint(path: Path, task: dict, plan_hash: str) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        eta = payload["eta"]
        record = json.loads(str(payload["record"].item()))
        if str(payload["record_sha256"].item()) != stable_json_sha256(record):
            raise ValueError(f"Checkpoint metadata checksum mismatch: {path}")
    if (record["id"] != task["id"] or record["plan_sha256"] != plan_hash
        or record["task_sha256"] != stable_json_sha256(task) or record["eta"] != hash_array(eta)
        or not np.isfinite(eta).all()):
        raise ValueError(f"Checkpoint does not match this exact study: {path}")
    spatial_shape = (task["core"], task["moving_equilibrium"]["transverse_cells"])
    if task["variant"] != "moving_equilibrium":
        spatial_shape = (task["publication_core"],) * 2
    if eta.shape != (len(task["times"]), *spatial_shape) or eta.dtype != np.float64:
        raise ValueError(f"Checkpoint trajectory shape or dtype mismatch: {path}")
    return eta, record["result"]


def _write_checkpoint(
    task: dict, directory: Path, plan_hash: str, eta: np.ndarray, result: dict,
) -> None:
    record = {
        "id": task["id"], "plan_sha256": plan_hash, "task_sha256": stable_json_sha256(task),
        "eta": hash_array(eta), "result": result,
    }
    _write_atomic(directory / f"{task['id']}.npz", lambda stream: np.savez_compressed(
        stream, eta=eta, record=json.dumps(record, allow_nan=False, sort_keys=True),
        record_sha256=stable_json_sha256(record),
    ))


def _plan(config_path: Path, dataset_root: Path | None, case_limit: int | None,
          variants: tuple[str, ...]) -> tuple[dict, list[dict]]:
    study = _read_yaml(config_path)
    dataset_path = ROOT / study["dataset_config"]
    dataset = _read_yaml(dataset_path)
    root = (dataset_root or Path(study["dataset_root"])).resolve()
    requested = parse_requested_output_config(dataset["requested_output"])
    if requested is None or requested.split != "test" or not dataset["dataset"]["auto_dt"]:
        raise ValueError("The study requires the accepted, adaptive requested-output test contract")
    if requested.status != "accepted" or requested.execution_scope != "production":
        raise ValueError("Expected an accepted production requested-time contract")
    base = study["baseline_core"]
    buffer = dataset["computational_domain"]["buffer_cells"]
    if (not dataset["computational_domain"]["enabled"] or base != study["coarse_core"] * 2
        or study["fine_core"] != 2 * base or study["publication_core"] != study["coarse_core"]
        or dataset["solver"]["nx"] != base + 2 * buffer or dataset["solver"]["ny"] != base + 2 * buffer
        or buffer % 2 or dataset["solver"]["sponge_width"] % 2
        or study["extended_buffer_cells"] <= buffer or not 0 < study["common_cfl"] <= 0.45):
        raise ValueError("Inconsistent computational, nested-grid, or CFL controls")
    case_count = len(study["cases"]) if case_limit is None else case_limit
    if not 1 <= case_count <= len(study["cases"]):
        raise ValueError("case limit must select at least one case in the frozen roster")
    moving_grids = study["moving_equilibrium"]["grids"]
    if len(set(moving_grids)) != len(moving_grids) or not all(8 <= grid <= 512 for grid in moving_grids):
        raise ValueError("Moving-equilibrium grids must be distinct and between 8 and 512")

    roster, manifests = _load_roster(root, study, dataset)
    inputs = [_file_record(config_path), _file_record(dataset_path), *manifests]
    code_paths = [
        Path(__file__).resolve(), ROOT / "src/data_gen/simulate_dataset.py",
        ROOT / "src/data_gen/common_time.py", ROOT / "src/utils/hashing.py",
    ]
    code_paths.extend(sorted((ROOT / "src/solver").glob("*.py")))
    inputs.extend(_file_record(path) for path in code_paths)
    for case in roster:
        inputs.extend(case["inputs"].values())
        for publication in case["publications"].values():
            inputs.extend(publication.values())
    shared = {
        "study": study, "dataset": dataset, "times": requested.requested_times.tolist(),
        "contract_hash": requested.contract_hash, "max_natural_steps": requested.max_natural_steps,
    }

    tasks = []
    seen_beds = set()
    selected_ids = {case["row"]["scenario_id"] for case in roster[:case_count]}
    for case in roster:
        case_variants = ["baseline", "coarse", "fine", "common_cfl", "half_common_cfl"]
        if case.get("boundary_controls"):
            case_variants.extend(BOUNDARY_VARIANTS)
        if case["bathymetry"] not in seen_beds:
            case_variants.append("lake_rest")
            seen_beds.add(case["bathymetry"])
        for name in FACTORIES:
            for variant in case_variants:
                core = study[f"{variant}_core"] if variant in {"coarse", "fine"} else base
                tasks.append(_make_task(variant, name, case, shared, core))
    for core in moving_grids:
        for name in FACTORIES:
            tasks.append(_make_task("moving_equilibrium", name, None, shared, core))
    selected = [
        task for task in tasks if task["variant"] in variants
        and (task["case"] is None or task["case"]["row"]["scenario_id"] in selected_ids)
    ]
    if not selected:
        raise ValueError("The requested case and variant filters select no rollouts")
    plan = {
        "schema_id": SCHEMA,
        "roster": roster,
        "requested_timestamps": shared["times"],
        "requested_contract_hash": requested.contract_hash,
        "code_and_input_files": inputs,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "architecture": platform.machine(),
                    "threads": {name: os.environ[name] for name in THREAD_ENV}},
        "complete_full_scope": len(selected) == len(tasks),
        "full_scope_rollout_count": len(tasks),
        "tasks": [
            {key: value for key, value in task.items() if key not in {"dataset", "case", "times"}}
            for task in selected
        ],
        "interpretation": {
            "scope": "Two shallow-water references on a fixed, outcome-blind six-case subset.",
            "space": (
                "All grids share the physical core and buffer extents. The production 128-cell "
                "bathymetry and initial conserved depth are treated as fixed cell averages "
                "after production initialization. Coarse values are block means and fine "
                "values repeat each frozen cell. Eta is area-averaged to the common 64-cell "
                "core. Fixed-CFL spatial sensitivities also change dt and do not estimate "
                "a formal convergence order or resolve smaller physical input scales."
            ),
            "time": (
                "Adaptive natural steps are bracketed and linearly interpolated to the "
                "production requested times. Common CFL does not impose common exact steps. "
                "Halving target CFL changes both integration and requested-time interpolation. "
                "New requested-time states and publication-core averages retain float64; "
                "the stored publication comparison includes archival float32 rounding."
            ),
            "boundary": (
                "Sponge removal preserves the radiation boundary. The extended no-sponge "
                "variant moves that boundary out using edge-replicated bathymetry and the "
                "same unchanged inner state. Its contrast includes the extended exterior. "
                "The open/radiation pair retains the original no-sponge domain."
            ),
            "equilibria": (
                "Lake-rest drift is measured on the whole buffered computational domain. "
                "The periodic moving study uses the subcritical steady Bernoulli branch "
                "for constant discharge and a smooth bed, sampled at cell centers. Drift "
                "is measured relative to that sampled reference at the requested times."
            ),
            "metrics": (
                "Relative L2 divides by the named reference. Frame errors with zero "
                "reference norm are reported as null. RMSE remains defined. "
                "The inter-code difference does not identify physical truth."
            ),
            "state_budget": (
                "Water volume and mechanical energy per density are integrated over the "
                "whole computational domain using the sampled wet state. The change "
                "combines boundary transport, sponge action, and numerical effects; "
                "no closed-domain conservation criterion is applied to those rollouts."
            ),
            "incomplete_scope": "Any filtered run is evidence only for the listed rollouts.",
        },
    }
    return plan, selected


def _execute(output: Path, plan: dict, tasks: list[dict], *, resume: bool, workers: int) -> dict:
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    directory = output / "checkpoints"
    final_path = output / "benchmark_numerical_study.json"
    plan_hash = stable_json_sha256(plan)
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            if old != plan:
                raise ValueError("Refusing resume: the frozen plan, inputs, code, or scope changed")
            directory.mkdir(exist_ok=True)
        else:
            _write_json(plan_path, plan)
            directory.mkdir()
        expected_names = {f"{task['id']}.npz" for task in tasks}
        unexpected = {path.name for path in directory.glob("*.npz")} - expected_names
        if unexpected:
            raise ValueError(f"Unexpected checkpoint files: {sorted(unexpected)}")
        missing = []
        entries = {}
        for task in tasks:
            path = directory / f"{task['id']}.npz"
            if path.exists():
                entries[task["id"]] = _read_checkpoint(path, task, plan_hash)
            else:
                missing.append(task)
        if final_path.exists() and missing:
            raise RuntimeError("A finalized study is missing checkpoints")
        if workers == 1:
            for task in missing:
                _write_checkpoint(task, directory, plan_hash, *_solve(task))
                print(f"Completed {task['id']}", flush=True)
        elif missing:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                pending = {pool.submit(_solve, task): task for task in missing}
                for future in as_completed(pending):
                    task = pending.pop(future)
                    _write_checkpoint(task, directory, plan_hash, *future.result())
                    print(f"Completed {task['id']}", flush=True)
        for task in missing:
            path = directory / f"{task['id']}.npz"
            entries[task["id"]] = _read_checkpoint(path, task, plan_hash)
        for file_record in plan["code_and_input_files"]:
            _check_file(file_record)
        summary = _summary(plan, entries, plan_hash)
        if final_path.exists():
            if json.loads(final_path.read_text(encoding="utf-8")) != summary:
                raise ValueError("Final summary disagrees with verified checkpoints")
        else:
            _write_json(final_path, summary)
        return summary
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Write the frozen plan without advancing solvers")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if args.case_limit is not None and args.case_limit <= 0:
        parser.error("--case-limit must be positive")
    if args.plan_only and args.resume:
        parser.error("--plan-only and --resume are mutually exclusive")
    plan, tasks = _plan(args.config, args.dataset_root, args.case_limit, tuple(args.variants))
    if args.plan_only:
        args.output.mkdir(parents=True, exist_ok=False)
        _write_json(args.output / "plan.json", plan)
        print(f"Froze {len(tasks)}/{plan['full_scope_rollout_count']} rollouts; resume with the same filters.")
        return
    summary = _execute(args.output, plan, tasks, resume=args.resume, workers=args.workers)
    print(f"Verified {summary['rollout_count']}/{summary['full_scope_rollout_count']} rollouts.")


if __name__ == "__main__":
    main()
