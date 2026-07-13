from __future__ import annotations

import csv
import io
import json
import math
import multiprocessing
import os
import platform
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    authoritative_input_fingerprint,
    candidate_requested_times,
    code_state,
    hash_array,
    sha256_file,
    stable_hash_payload,
)
from src.data_gen.simulate_dataset import _simulate_one_local
from src.solver.boussinesq import BoussinesqSolver
from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver
from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver


LEVEL_A_SCHEMA_ID = "tsunami-surrogate.common-time-v2.level-a.v1"
DECISIONS = {
    "pass_to_H1",
    "blocked_operator_semantics",
    "blocked_boussinesq_health",
    "blocked_boundary_behavior",
    "blocked_convergence",
    "implementation_failure",
}
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
TASK_SCHEMA_ID = "tsunami-surrogate.common-time-v2.level-a-task.v1"
TASK_ARTIFACT_SCHEMA_ID = "tsunami-surrogate.common-time-v2.level-a-task-artifact.v1"
TASK_RESULT_SCHEMA_ID = "tsunami-surrogate.common-time-v2.level-a-task-result.v1"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_VOLATILE_SCIENTIFIC_KEYS = {
    "elapsed_s",
    "runtime_s",
    "operational_provenance",
    "worker_metadata",
}
_DROP_SCIENTIFIC_VALUE = object()
DERIVED_REPLAY_REL_TOL = 2.0e-15
DERIVED_REPLAY_ABS_TOL = 1.0e-18
_DERIVED_ROW_COMPONENTS = {
    "boundary_sponge",
    "canary_bootstrap_descriptive",
    "operator_factor_identity",
    "operator_sensitivity_summary",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        return ""
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _json_safe(row.get(key)) for key in fields})
    return handle.getvalue()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_csv_text(rows), encoding="utf-8", newline="")


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_checksums(
    root: Path, *, allow_unlisted_prefixes: Sequence[str] = ()
) -> None:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise RuntimeError(f"Missing checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in listed:
            raise RuntimeError(f"Duplicate Level A checksum entry: {relative}")
        listed.add(relative)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Level A checksum mismatch: {relative}")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(prefix) for prefix in allow_unlisted_prefixes):
            continue
        actual.add(relative)
    covered_listed = {
        relative
        for relative in listed
        if not any(relative.startswith(prefix) for prefix in allow_unlisted_prefixes)
    }
    if covered_listed != actual:
        missing = sorted(actual - covered_listed)
        extra = sorted(covered_listed - actual)
        raise RuntimeError(
            f"Level A checksum coverage mismatch: missing={missing}, extra={extra}"
        )


def _validate_source_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != LEVEL_A_SCHEMA_ID:
        raise ValueError("Level A schema mismatch")
    grid = config.get("candidate_grid", {})
    expected = candidate_requested_times()
    derived = float(grid["step"]) * np.arange(1, int(grid["count"]) + 1)
    derived[-1] = float(grid["horizon"])
    if not np.array_equal(derived.astype(np.float64), expected):
        raise ValueError("Level A requested grid does not match common-time-v2")
    if list(config.get("decision_precedence", [])) != [
        "implementation_failure",
        "blocked_boussinesq_health",
        "blocked_boundary_behavior",
        "blocked_convergence",
        "blocked_operator_semantics",
        "pass_to_H1",
    ]:
        raise ValueError("Level A decision precedence is not frozen as required")
    boundary = config.get("boundary_packet", {})
    solver_specs = boundary.get("solvers", {})
    if set(solver_specs) != set(SOLVERS):
        raise ValueError("Level A boundary_packet must define every solver")
    for solver in SOLVERS:
        _validate_boundary_packet_spec(
            solver_specs[solver],
            nx=int(boundary.get("grid", 128)),
            sponge_width=max(1, int(boundary.get("grid", 128)) // 8),
        )
    conservation = config.get("thresholds", {}).get("conservation", {})
    required_conservation_contract = {
        "measurement_grid": "internal_natural_states",
        "measurement_dtype": "float64",
        "precision_floor_method": "float64_gamma_n_l1",
        "threshold_status": "retained_preexisting_normalized_limit",
    }
    for key, expected in required_conservation_contract.items():
        if conservation.get(key) != expected:
            raise ValueError(
                f"Level A conservation {key} must be frozen as {expected!r}"
            )
    execution = config.get("execution", {})
    expected_execution = {
        "requested_workers": 2,
        "requested_max_in_flight": 2,
        "process_start_method": "spawn",
        "thread_environment": {key: "1" for key in THREAD_ENV_KEYS},
    }
    if execution != expected_execution:
        raise ValueError("Level A execution policy is not frozen as required")


def _select_canaries(
    inventory: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    train = sorted(
        (dict(row) for row in inventory if row.get("split") == "train"),
        key=lambda row: (str(row["qualified_id"]), str(row["input_fingerprint"])),
    )
    source_families = sorted({str(row["source_type"]) for row in train})
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_bathymetry: set[str] = set()
    for source in source_families:
        candidates = [row for row in train if str(row["source_type"]) == source]
        candidates.sort(
            key=lambda row: (
                str(row["bathymetry_type"]) in used_bathymetry,
                str(row["qualified_id"]),
            )
        )
        if candidates and len(selected) < count:
            row = candidates[0]
            selected.append(row)
            used_ids.add(str(row["qualified_id"]))
            used_bathymetry.add(str(row["bathymetry_type"]))
    if len(selected) < count:
        remaining = [row for row in train if str(row["qualified_id"]) not in used_ids]
        remaining.sort(
            key=lambda row: (
                str(row["bathymetry_type"]) in used_bathymetry,
                str(row["qualified_id"]),
            )
        )
        selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise RuntimeError(f"Could not select {count} training canaries")
    keep = {
        "split",
        "qualified_id",
        "scenario_id",
        "sample_index",
        "bathymetry_type",
        "source_type",
        "source_strength",
        "input_fingerprint",
        "bathymetry_cache_path",
        "source_cache_path",
        "raw_sample_paths",
        "array_hashes",
    }
    return [{key: row[key] for key in keep if key in row} for row in selected]


def preregister_level_a(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Level A YAML must contain a mapping")
    _validate_source_contract(config)

    h0_dir = (
        repo_root
        / "artifacts/common_time_v2/h0/830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b"
    )
    h0_decision = _read_json(h0_dir / "h0_decision.json")
    if not h0_decision.get("audit_passed"):
        raise RuntimeError("H0 must pass before Level A preregistration")
    inventory_path = h0_dir / "h0_input_inventory.jsonl"
    inventory = _read_jsonl(inventory_path)
    canaries = _select_canaries(inventory, int(config["canaries"]["count"]))
    source_code = code_state(repo_root)
    execution_environment = _execution_environment_snapshot()
    expected_threads = config["execution"]["thread_environment"]
    if execution_environment["thread_environment"] != expected_threads:
        raise RuntimeError(
            "Level A preregistration environment does not match the frozen "
            "numerical-library thread policy"
        )
    blueprint_tasks = _build_level_a_task_plan(
        config,
        canaries,
        contract_hash="pending-contract-hash",
        code_state_hash=str(source_code["code_state_hash"]),
    )
    task_blueprint = [
        {
            "ordinal": task["ordinal"],
            "task_id": task["task_id"],
            "kind": task["kind"],
            "spec": task["spec"],
        }
        for task in blueprint_tasks
    ]
    payload = {
        "schema_id": LEVEL_A_SCHEMA_ID,
        "artifact_kind": "common-time-v2-level-a-preregistered-contract",
        "source_config": _json_safe(config),
        "source_config_sha256": sha256_file(config_path),
        "candidate_times": candidate_requested_times().tolist(),
        "h0_decision_sha256": sha256_file(h0_dir / "h0_decision.json"),
        "h0_inventory_sha256": sha256_file(inventory_path),
        "canaries": canaries,
        "code_state": source_code,
        "task_blueprint": task_blueprint,
        "worker_policy": _json_safe(config["execution"]),
        "execution_environment": execution_environment,
        "thresholds_frozen_before_execution": True,
    }
    contract_hash = stable_hash_payload(
        artifact_kind="common-time-v2-level-a-contract",
        payload=payload,
        schema_id=LEVEL_A_SCHEMA_ID,
    )
    payload["contract_hash"] = contract_hash
    task_plan = _build_level_a_task_plan(
        config,
        canaries,
        contract_hash=contract_hash,
        code_state_hash=str(source_code["code_state_hash"]),
    )
    base = output_root or (repo_root / "artifacts/common_time_v2/level_a")
    final = base / contract_hash
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite Level A root: {final}")
    staging = base / f".{contract_hash}.preregister-staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    _write_json(staging / "preregistered_contract.json", payload)
    _write_json(staging / "resolved_configs.json", config)
    _write_json(staging / "canary_selection.json", canaries)
    _write_json(staging / "task_plan.json", task_plan)
    report = f"""# Common-time-v2 Level A preregistration

- Contract hash: `{contract_hash}`
- Candidate grid: 50 positive outputs through `0.1750`
- Code-state hash: `{source_code["code_state_hash"]}`
- H0 inventory records: {len(inventory)}
- Training canaries: {len(canaries)}
- Frozen task count: {len(task_plan)}
- Worker policy: {config["execution"]["requested_workers"]} workers, {config["execution"]["requested_max_in_flight"]} maximum in-flight tasks, `{config["execution"]["process_start_method"]}` start method
- Thresholds frozen before execution: yes

Stage C thresholds are historical context only and were not inherited. `depth_scale=1.0` is the sole v2 production interpretation in this contract. The current `open` implementation is labelled zero-gradient edge padding, not radiative. A Level A pass permits only progression to H1; it does not accept the production contract.
"""
    (staging / "PREREGISTRATION.md").write_text(report, encoding="utf-8")
    _write_checksums(staging)
    base.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    return final


def _solver(
    name: str,
    *,
    nx: int,
    ny: int,
    cfl: float,
    boundary: str,
    use_sponge: bool,
    sponge_mode: str = "legacy_per_step",
    filter_mode: str = "disabled",
    filter_strength: float = 0.0,
    sponge_axes: str = "xy",
) -> Any:
    common = dict(
        nx=nx,
        ny=ny,
        dx=1.0 / nx,
        dy=1.0 / ny,
        dt=1.0e-4,
        g=9.81,
        cfl=cfl,
        boundary=boundary,
        use_sponge=use_sponge,
        sponge_width=max(1, nx // 8),
        sponge_min_factor=0.9,
        sponge_time_mode=sponge_mode,
        sponge_reference_dt=0.0035
        if sponge_mode == "elapsed_time_consistent"
        else None,
        sponge_axes=sponge_axes,
    )
    if name == "swe_hydrostatic":
        return HydrostaticShallowWaterSolver(
            **common, dry_tolerance=1e-6, max_velocity=30.0
        )
    if name == "swe_muscl_hr":
        return MUSCLHRShallowWaterSolver(
            **common, dry_tolerance=1e-6, max_velocity=30.0
        )
    return BoussinesqSolver(
        **common,
        alpha=1.0 / 3.0,
        min_depth=1e-4,
        depth_scale=1.0,
        mode="linear_constant_depth",
        filter_strength=filter_strength,
        filter_time_mode=filter_mode,
        filter_reference_dt=(
            0.0035 if filter_mode == "elapsed_time_consistent" else None
        ),
        cg_failure_mode="strict_v2",
        linear_solver_tol=1e-10,
        linear_solver_max_iter=500,
        check_finite=True,
    )


def _mode_exact(
    name: str,
    *,
    nx: int,
    ny: int,
    mode: int,
    amplitude: float,
    times: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    x = np.arange(nx, dtype=float)[:, None] / nx
    k = 2.0 * math.pi * mode
    if name == "boussinesq":
        kd = 2.0 * math.sin(0.5 * k / nx) * nx
        omega = math.sqrt(9.81 * kd * kd / (1.0 + kd * kd / 3.0))
        group = math.sqrt(9.81) * math.cos(0.5 * k / nx) / (1.0 + kd * kd / 3.0) ** 1.5
    else:
        omega = math.sqrt(9.81) * k
        group = math.sqrt(9.81)
    exact = np.stack(
        [
            amplitude * np.cos(k * x - omega * float(t)) * np.ones((1, ny), dtype=float)
            for t in times
        ],
        axis=0,
    )
    return exact, omega, group


def _run_mode(
    name: str, *, nx: int, ny: int, mode: int, cfl: float, amplitude: float
) -> dict[str, Any]:
    times = candidate_requested_times()
    exact, omega, group = _mode_exact(
        name, nx=nx, ny=ny, mode=mode, amplitude=amplitude, times=times
    )
    solver = _solver(name, nx=nx, ny=ny, cfl=cfl, boundary="periodic", use_sponge=False)
    bathymetry = -np.ones((nx, ny), dtype=float)
    x = np.arange(nx, dtype=float)[:, None] / nx
    k = 2.0 * math.pi * mode
    eta0 = amplitude * np.cos(k * x) * np.ones((1, ny), dtype=float)
    if name == "boussinesq":
        eta_t0 = amplitude * omega * np.sin(k * x) * np.ones((1, ny))
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(eta0, eta_t0=eta_t0)
    else:
        h0 = 1.0 + eta0
        hu0 = math.sqrt(9.81) * eta0
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(h0, hu0=hu0, hv0=np.zeros_like(h0))
    states, emitted, dt_history, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=cfl,
        include_initial_state=False,
        requested_times=times,
        max_natural_steps=10000,
        collect_natural_step_health=True,
    )
    eta = states[:, 0] if name == "boussinesq" else states[:, 0] + bathymetry
    basis = np.exp(-1j * k * np.arange(nx) / nx)[:, None]
    coeff = np.asarray([np.mean(frame * basis) for frame in eta])
    phase = np.unwrap(np.angle(coeff))
    slope = float(np.polyfit(emitted, phase, 1)[0])
    measured_omega = -slope
    amplitude_ratio = float(abs(coeff[-1]) / max(abs(coeff[0]), 1e-30))
    diff = eta - exact
    field_l2 = _relative_l2(eta, exact)
    widths = np.asarray(diagnostics["bracket_widths"], dtype=float)
    left_times = np.asarray(diagnostics["left_natural_timestamps"], dtype=float)
    right_times = np.asarray(diagnostics["right_natural_timestamps"], dtype=float)
    weights = np.asarray(diagnostics["interpolation_weights"], dtype=float)
    left_exact, _, _ = _mode_exact(
        name, nx=nx, ny=ny, mode=mode, amplitude=amplitude, times=left_times
    )
    right_exact, _, _ = _mode_exact(
        name, nx=nx, ny=ny, mode=mode, amplitude=amplitude, times=right_times
    )
    interpolated_exact = (
        left_exact * (1.0 - weights[:, None, None])
        + right_exact * weights[:, None, None]
    )
    interpolation_error = float(np.max(np.abs(interpolated_exact - exact)))
    interpolation_bound = float(amplitude * omega * omega * np.max(widths) ** 2 / 8.0)
    operator = solver.get_operator_diagnostics()
    return {
        "component": "analytical_mode",
        "solver": name,
        "grid": nx,
        "mode": mode,
        "wavenumber": k,
        "cfl": cfl,
        "measured_omega": measured_omega,
        "expected_omega": omega,
        "phase_speed_relative_error": abs(measured_omega - omega) / omega,
        "group_speed_expected": group,
        "amplitude_drift": abs(amplitude_ratio - 1.0),
        "field_relative_l2": field_l2,
        "interpolation_actual_max_abs_error": interpolation_error,
        "interpolation_absolute_bound": interpolation_bound,
        "interpolation_bound_floating_tolerance": float(
            32.0 * np.finfo(np.float64).eps * max(amplitude, 1.0)
        ),
        "output_count": int(emitted.size),
        "requested_times_exact": bool(
            np.array_equal(emitted, candidate_requested_times())
        ),
        "natural_steps": int(dt_history.size),
        "max_bracket_width": float(np.max(widths)),
        "finite": bool(np.isfinite(states).all()),
        "cg_failure_count": int(
            np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
        ),
        "nan_to_num_replacement_count": int(
            operator.get("nan_to_num_replacement_count", 0)
        ),
        "positivity_projection_count": int(
            operator.get("positivity_projection_count", 0)
        ),
        "dry_projection_count": int(operator.get("dry_projection_count", 0)),
        "operator": operator,
        "_trajectory_eta": eta,
    }


def _observed_order(coarse: float, fine: float) -> float | None:
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return float(math.log(coarse / fine, 2.0))


def _trajectory_rms_difference(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Trajectory shapes differ: {aa.shape} != {bb.shape}")
    return float(_stable_l2_norm(aa - bb) / math.sqrt(aa.size))


def _temporal_refinement_gate(
    trajectories: Sequence[np.ndarray], *, minimum_order: float, floor: float
) -> dict[str, Any]:
    if len(trajectories) != 3:
        raise ValueError(
            "Temporal refinement requires production, half, and quarter trajectories"
        )
    production_to_half = _trajectory_rms_difference(trajectories[0], trajectories[1])
    half_to_quarter = _trajectory_rms_difference(trajectories[1], trajectories[2])
    below_floor = production_to_half <= floor and half_to_quarter <= floor
    order = (
        None
        if below_floor or half_to_quarter <= 0.0
        else _observed_order(production_to_half, half_to_quarter)
    )
    passed = below_floor or (
        production_to_half > half_to_quarter
        and order is not None
        and order >= minimum_order
    )
    return {
        "production_to_half_trajectory_rms": production_to_half,
        "half_to_quarter_trajectory_rms": half_to_quarter,
        "roundoff_floor": floor,
        "both_below_floor": below_floor,
        "observed_order": order,
        "passed": passed,
    }


def _group_speed_gate(
    rows: Sequence[Mapping[str, Any]], *, relative_error_limit: float
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["wavenumber"]))
    if len(ordered) < 2:
        raise ValueError("Group-speed measurement requires at least two Fourier modes")
    wavenumbers = np.asarray([row["wavenumber"] for row in ordered], dtype=float)
    measured = np.asarray([row["measured_omega"] for row in ordered], dtype=float)
    expected = np.asarray([row["expected_omega"] for row in ordered], dtype=float)
    measured_group = float(np.polyfit(wavenumbers, measured, 1)[0])
    expected_group = float(np.polyfit(wavenumbers, expected, 1)[0])
    relative_error = abs(measured_group - expected_group) / max(
        abs(expected_group), 1e-30
    )
    return {
        "measured_group_speed": measured_group,
        "expected_group_speed": expected_group,
        "group_speed_relative_error": relative_error,
        "passed": relative_error <= relative_error_limit,
    }


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value for key, value in row.items() if not str(key).startswith("_")
    }


def _load_canary_arrays(
    row: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
    qualified_id = str(row.get("qualified_id", "unknown"))
    with np.load(
        Path(str(row["bathymetry_cache_path"])), allow_pickle=False
    ) as payload:
        missing = {"bathymetry", "bathymetry_type"} - set(payload.files)
        if missing:
            raise RuntimeError(
                f"Canary {qualified_id} bathymetry cache missing {sorted(missing)}"
            )
        bathymetry = np.asarray(payload["bathymetry"])
        bathymetry_type = str(np.asarray(payload["bathymetry_type"]).reshape(-1)[0])
    with np.load(Path(str(row["source_cache_path"])), allow_pickle=False) as payload:
        missing = {"source_field", "source_strength", "source_type"} - set(
            payload.files
        )
        if missing:
            raise RuntimeError(
                f"Canary {qualified_id} source cache missing {sorted(missing)}"
            )
        source = np.asarray(payload["source_field"])
        strength_array = np.asarray(payload["source_strength"])
        source_type = str(np.asarray(payload["source_type"]).reshape(-1)[0])
    if bathymetry_type != str(row["bathymetry_type"]):
        raise RuntimeError(f"Canary {qualified_id} bathymetry_type mismatch")
    if source_type != str(row["source_type"]):
        raise RuntimeError(f"Canary {qualified_id} source_type mismatch")
    strength = float(strength_array.reshape(-1)[0])
    if not np.isfinite(strength) or np.float32(strength) != np.float32(
        row["source_strength"]
    ):
        raise RuntimeError(f"Canary {qualified_id} source_strength mismatch")
    if not np.isfinite(bathymetry).all() or not np.isfinite(source).all():
        raise RuntimeError(
            f"Canary {qualified_id} contains nonfinite authoritative inputs"
        )

    expected_hashes = row.get("array_hashes")
    if not isinstance(expected_hashes, Mapping):
        raise RuntimeError(f"Canary {qualified_id} array_hashes are missing")
    primary = {"bathymetry": bathymetry, "source_field": source}
    for name, values in primary.items():
        if hash_array(values) != expected_hashes.get(name):
            raise RuntimeError(f"Canary {qualified_id} {name} hash mismatch")

    rest = np.maximum(-bathymetry, 0.0).astype(bathymetry.dtype, copy=False)
    eta0 = np.asarray(
        strength * source, dtype=np.dtype(expected_hashes["eta0"]["dtype"])
    )
    h0 = np.maximum(rest + eta0, 0.0).astype(
        np.dtype(expected_hashes["initial_depth"]["dtype"]), copy=False
    )
    free = (h0 + bathymetry).astype(
        np.dtype(expected_hashes["free_surface0"]["dtype"]), copy=False
    )
    arrays = {
        "bathymetry": bathymetry,
        "source_field": source,
        "rest_depth": rest,
        "eta0": eta0,
        "initial_depth": h0,
        "free_surface0": free,
    }
    for name, values in arrays.items():
        if hash_array(values) != expected_hashes.get(name):
            raise RuntimeError(f"Canary {qualified_id} derived {name} hash mismatch")
    fingerprint = authoritative_input_fingerprint(
        split=str(row["split"]),
        sample_index=int(row["sample_index"]),
        scenario_id=str(row["scenario_id"]),
        bathymetry_type=bathymetry_type,
        source_type=source_type,
        source_strength=strength_array,
        arrays=arrays,
    )
    if fingerprint != row["input_fingerprint"]:
        raise RuntimeError(f"Canary fingerprint mismatch: {qualified_id}")
    return bathymetry, source, strength_array, strength, arrays


def _preflight_canaries(canaries: Sequence[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    for row in canaries:
        missing = False
        for key in ("bathymetry_cache_path", "source_cache_path"):
            path = Path(str(row.get(key, "")))
            if not path.is_file():
                issues.append(f"missing {key} for {row.get('qualified_id')}: {path}")
                missing = True
        if missing:
            continue
        try:
            _load_canary_arrays(row)
        except Exception as exc:
            issues.append(
                f"invalid authoritative inputs for {row.get('qualified_id')}: "
                f"{type(exc).__name__}: {exc}"
            )
    return issues


def _scientific_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        if (
            value.get("component") == "canary_bootstrap_descriptive"
            and value.get("metric") == "runtime_s"
        ):
            return _DROP_SCIENTIFIC_VALUE
        normalized_mapping: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if str(key) in _VOLATILE_SCIENTIFIC_KEYS:
                continue
            normalized = _scientific_normalize(item)
            if normalized is not _DROP_SCIENTIFIC_VALUE:
                normalized_mapping[str(key)] = normalized
        return normalized_mapping
    if isinstance(value, (list, tuple)):
        normalized = [_scientific_normalize(item) for item in value]
        return [item for item in normalized if item is not _DROP_SCIENTIFIC_VALUE]
    if isinstance(value, np.ndarray):
        return {"array_hash": hash_array(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _scientific_digest(value: Any) -> str:
    return stable_hash_payload(
        artifact_kind="common-time-v2-level-a-scientific-result",
        payload=_scientific_normalize(value),
        schema_id=LEVEL_A_SCHEMA_ID,
    )


def _derived_replay_equal(stored: Any, recomputed: Any) -> bool:
    first = _scientific_normalize(stored)
    second = _scientific_normalize(recomputed)

    def equal(left: Any, right: Any) -> bool:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(
                equal(left[key], right[key]) for key in left
            )
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right)
            )
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        if isinstance(left, (float, np.floating)) and isinstance(
            right, (float, np.floating)
        ):
            return math.isclose(
                float(left),
                float(right),
                rel_tol=DERIVED_REPLAY_REL_TOL,
                abs_tol=DERIVED_REPLAY_ABS_TOL,
            )
        return left == right

    return equal(first, second)


def _recomputed_rows_equal(
    stored: Sequence[Mapping[str, Any]],
    recomputed: Sequence[Mapping[str, Any]],
) -> bool:
    if len(stored) != len(recomputed):
        return False
    for first, second in zip(stored, recomputed):
        if first.get("component") != second.get("component"):
            return False
        if first.get("component") in _DERIVED_ROW_COMPONENTS:
            if not _derived_replay_equal(first, second):
                return False
        elif _scientific_normalize(first) != _scientific_normalize(second):
            return False
    return True


def _thread_settings() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def _execution_environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": _thread_settings(),
    }


def _operational_provenance(
    *,
    requested_workers: int,
    effective_workers: int,
    requested_max_in_flight: int | None = None,
    effective_max_in_flight: int = 0,
    peak_in_flight_futures: int = 0,
    process_start_method: str | None = None,
) -> dict[str, Any]:
    return {
        "requested_workers": int(requested_workers),
        "effective_workers": int(effective_workers),
        "requested_max_in_flight": (
            None if requested_max_in_flight is None else int(requested_max_in_flight)
        ),
        "effective_max_in_flight": int(effective_max_in_flight),
        "peak_in_flight_futures": int(peak_in_flight_futures),
        "process_start_method": (
            process_start_method
            if process_start_method is not None
            else "spawn"
            if effective_workers > 1
            else "serial"
        ),
        **_execution_environment_snapshot(),
    }


def _make_level_a_task(
    *,
    ordinal: int,
    task_id: str,
    kind: str,
    spec: Mapping[str, Any],
    contract_hash: str,
    code_state_hash: str,
) -> dict[str, Any]:
    identity = {
        "schema_id": TASK_SCHEMA_ID,
        "contract_hash": str(contract_hash),
        "code_state_hash": str(code_state_hash),
        "ordinal": int(ordinal),
        "task_id": str(task_id),
        "kind": str(kind),
        "spec": _json_safe(spec),
    }
    identity["task_spec_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-level-a-task",
        payload=identity,
        schema_id=TASK_SCHEMA_ID,
    )
    return identity


def _build_level_a_task_plan(
    config: Mapping[str, Any],
    canaries: Sequence[Mapping[str, Any]],
    *,
    contract_hash: str,
    code_state_hash: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    def add(task_id: str, kind: str, spec: Mapping[str, Any]) -> None:
        tasks.append(
            _make_level_a_task(
                ordinal=len(tasks),
                task_id=task_id,
                kind=kind,
                spec=spec,
                contract_hash=contract_hash,
                code_state_hash=code_state_hash,
            )
        )

    amplitude = float(config["analytical"]["amplitude"])
    ny = int(config["analytical"]["transverse_cells"])
    for solver in SOLVERS:
        spatial_cfl = float(config["analytical"]["spatial_refinement_cfl"][solver])
        for index, grid in enumerate(config["analytical"]["grids"]):
            add(
                f"analytical/{solver}/spatial/{index}",
                "analytical",
                {
                    "role": "spatial",
                    "solver": solver,
                    "grid": int(grid),
                    "ny": ny,
                    "mode": 1,
                    "cfl": spatial_cfl,
                    "amplitude": amplitude,
                },
            )
        for index, cfl in enumerate(config["production"]["cfl"][solver]):
            add(
                f"analytical/{solver}/temporal/{index}",
                "analytical",
                {
                    "role": "temporal",
                    "solver": solver,
                    "grid": 128,
                    "ny": ny,
                    "mode": 1,
                    "cfl": float(cfl),
                    "amplitude": amplitude,
                },
            )
        production_cfl = float(config["production"]["cfl"][solver][0])
        for index, mode in enumerate(config["analytical"]["modes"]):
            add(
                f"analytical/{solver}/modal/{index}",
                "analytical",
                {
                    "role": "modal",
                    "solver": solver,
                    "grid": 128,
                    "ny": ny,
                    "mode": int(mode),
                    "cfl": production_cfl,
                    "amplitude": amplitude,
                },
            )

    for solver in SOLVERS:
        production, half = [
            float(value) for value in config["production"]["cfl"][solver][:2]
        ]
        variants = [
            (
                "legacy_per_step",
                "legacy_per_step" if solver == "boussinesq" else "disabled",
                0.01 if solver == "boussinesq" else 0.0,
            ),
            ("elapsed_no_filter", "disabled", 0.0),
        ]
        if solver == "boussinesq":
            variants.append(("elapsed_filter", "elapsed_time_consistent", 0.01))
        for variant, filter_mode, filter_strength in variants:
            for cfl_index, cfl in enumerate((production, half)):
                add(
                    f"operator/{solver}/{variant}/{cfl_index}",
                    "operator",
                    {
                        "solver": solver,
                        "variant": variant,
                        "filter_mode": filter_mode,
                        "filter_strength": filter_strength,
                        "cfl": cfl,
                        "nx": 64,
                        "ny": 4,
                        "sponge_axes": "x",
                    },
                )

    boundary_config = config["boundary_packet"]
    boundary_grid = int(boundary_config["grid"])
    boundary_ny = int(boundary_config["transverse_cells"])
    for solver in SOLVERS:
        boundary_spec = _json_safe(boundary_config["solvers"][solver])
        for variant, use_sponge in (("baseline", False), ("damped", True)):
            add(
                f"boundary/{solver}/{variant}",
                "boundary",
                {
                    "solver": solver,
                    "variant": variant,
                    "use_sponge": use_sponge,
                    "cfl": float(config["production"]["cfl"][solver][0]),
                    "nx": boundary_grid,
                    "ny": boundary_ny,
                    "sponge_axes": "x",
                    "packet": boundary_spec,
                },
            )

    for boundary in ("periodic", "reflective"):
        for solver in SOLVERS:
            add(
                f"conservation/{boundary}/{solver}",
                "conservation",
                {
                    "solver": solver,
                    "boundary": boundary,
                    "cfl": float(config["production"]["cfl"][solver][0]),
                    "nx": 64,
                    "ny": 4,
                    "precision_floor_safety_factor": float(
                        config["thresholds"]["conservation"][
                            "precision_floor_safety_factor"
                        ]
                    ),
                },
            )

    for canary in canaries:
        for solver in SOLVERS:
            add(
                f"canary/{canary['qualified_id']}/{solver}",
                "canary",
                {
                    "solver": solver,
                    "cfl": float(config["production"]["cfl"][solver][0]),
                    "canary": _json_safe(canary),
                },
            )

    counts: dict[str, int] = {}
    for task in tasks:
        counts[str(task["kind"])] = counts.get(str(task["kind"]), 0) + 1
    expected = {
        "analytical": 27,
        "operator": 14,
        "boundary": 6,
        "conservation": 6,
        "canary": 18,
    }
    if counts != expected:
        raise RuntimeError(f"Level A task plan mismatch: {counts} != {expected}")
    return tasks


def _validate_task_plan(tasks: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(task["task_id"]) for task in tasks]
    hashes = [str(task["task_spec_hash"]) for task in tasks]
    ordinals = [int(task["ordinal"]) for task in tasks]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate Level A task_id")
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Duplicate Level A task_spec_hash")
    if ordinals != list(range(len(tasks))):
        raise RuntimeError("Level A task ordinals are not contiguous and ordered")
    names = [_task_directory_name(task) for task in tasks]
    if len(set(names)) != len(names):
        raise RuntimeError("Colliding Level A task artifact directory names")
    for task in tasks:
        expected = _make_level_a_task(
            ordinal=int(task["ordinal"]),
            task_id=str(task["task_id"]),
            kind=str(task["kind"]),
            spec=dict(task["spec"]),
            contract_hash=str(task["contract_hash"]),
            code_state_hash=str(task["code_state_hash"]),
        )
        if _json_safe(task) != expected:
            raise RuntimeError(f"Level A task identity mismatch: {task['task_id']}")


def _task_directory_name(task: Mapping[str, Any]) -> str:
    return f"{int(task['ordinal']):03d}-{str(task['task_spec_hash'])[:16]}"


def _compute_level_a_task(
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray | None]:
    spec = task["spec"]
    kind = str(task["kind"])
    if kind == "fixture":
        delay = float(spec.get("delay_s", 0.0))
        if delay > 0.0:
            time.sleep(delay)
        if bool(spec.get("fail", False)):
            raise RuntimeError(str(spec.get("message", "fixture failure")))
        value = float(spec["value"])
        return (
            {
                "component": "fixture",
                "name": str(spec["name"]),
                "value": value,
                "runtime_s": delay,
            },
            np.asarray([value, value * value], dtype=np.float64),
        )
    if kind == "analytical":
        row = _run_mode(
            str(spec["solver"]),
            nx=int(spec["grid"]),
            ny=int(spec["ny"]),
            mode=int(spec["mode"]),
            cfl=float(spec["cfl"]),
            amplitude=float(spec["amplitude"]),
        )
        trajectory = np.asarray(row.pop("_trajectory_eta"))
        row["analytical_role"] = str(spec["role"])
        return row, trajectory
    if kind == "operator":
        nx, ny = int(spec["nx"]), int(spec["ny"])
        eta0 = _packet(nx, ny)
        variant = str(spec["variant"])
        sponge_mode = (
            "legacy_per_step"
            if variant == "legacy_per_step"
            else "elapsed_time_consistent"
        )
        eta, dt, diagnostics, solver = _trajectory_eta(
            str(spec["solver"]),
            nx=nx,
            ny=ny,
            cfl=float(spec["cfl"]),
            boundary="open",
            use_sponge=True,
            sponge_mode=sponge_mode,
            filter_mode=str(spec["filter_mode"]),
            filter_strength=float(spec["filter_strength"]),
            eta0=eta0,
            sponge_axes=str(spec["sponge_axes"]),
        )
        row = {
            "component": "operator_sensitivity",
            "solver": str(spec["solver"]),
            "variant": variant,
            "cfl": float(spec["cfl"]),
            "natural_steps": int(dt.size),
            "amplitude_ratio": float(
                np.max(np.abs(eta)) / max(np.max(np.abs(eta0)), 1e-30)
            ),
            "high_frequency_fraction": _high_frequency_fraction(eta),
            "cg_failure_count": int(
                np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
            ),
            "finite": bool(np.isfinite(eta).all()),
            "sponge_axes": str(spec["sponge_axes"]),
            "whole_domain_sponge": bool(np.all(solver.sponge_mask < 1.0)),
            "operator": solver.get_operator_diagnostics(),
        }
        return row, eta
    if kind == "boundary":
        packet_spec = spec["packet"]
        initial = _boundary_initial_conditions(
            str(spec["solver"]),
            nx=int(spec["nx"]),
            ny=int(spec["ny"]),
            spec=packet_spec,
        )
        eta0 = np.asarray(initial["eta0"])
        use_sponge = bool(spec["use_sponge"])
        eta, _, diagnostics, solver = _trajectory_eta(
            str(spec["solver"]),
            nx=int(spec["nx"]),
            ny=int(spec["ny"]),
            cfl=float(spec["cfl"]),
            boundary="open",
            use_sponge=use_sponge,
            sponge_mode="elapsed_time_consistent" if use_sponge else "legacy_per_step",
            eta0=eta0,
            h0=(
                None
                if initial.get("h0") is None
                else np.asarray(initial["h0"])
            ),
            hu0=(
                None
                if initial.get("hu0") is None
                else np.asarray(initial["hu0"])
            ),
            eta_t0=(
                None
                if initial.get("eta_t0") is None
                else np.asarray(initial["eta_t0"])
            ),
            sponge_axes=str(spec["sponge_axes"]),
        )
        return {
            "component": "boundary_trajectory",
            "solver": str(spec["solver"]),
            "variant": str(spec["variant"]),
            "characteristic_speed": float(initial["characteristic_speed"]),
            "sponge_axes": str(spec["sponge_axes"]),
            "whole_domain_sponge": bool(np.all(solver.sponge_mask < 1.0)),
            "finite": bool(np.isfinite(eta).all()),
            "cg_failure_count": int(
                np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
            ),
            "operator": solver.get_operator_diagnostics(),
        }, eta
    if kind == "conservation":
        return _run_float64_conservation(
            str(spec["solver"]),
            nx=int(spec["nx"]),
            ny=int(spec["ny"]),
            cfl=float(spec["cfl"]),
            boundary=str(spec["boundary"]),
            safety_factor=float(spec["precision_floor_safety_factor"]),
        ), None
    if kind == "canary":
        canary = spec["canary"]
        bathymetry, _source, _strength_array, _strength, arrays = _load_canary_arrays(
            canary
        )
        eta0 = arrays["eta0"]
        rest = arrays["rest_depth"]
        h0 = arrays["initial_depth"]
        solver_name = str(spec["solver"])
        started = time.monotonic()
        eta, dt, diagnostics, solver = _trajectory_eta(
            solver_name,
            nx=bathymetry.shape[0],
            ny=bathymetry.shape[1],
            cfl=float(spec["cfl"]),
            boundary="open",
            use_sponge=True,
            sponge_mode="elapsed_time_consistent",
            filter_mode="disabled",
            bathymetry=bathymetry,
            eta0=eta0,
            h0=h0,
        )
        effective_depth = (
            np.maximum(-bathymetry, 1e-4)
            if solver_name == "boussinesq"
            else np.maximum(rest, 1e-8)
        )
        return {
            "component": "production_amplitude_canary",
            "qualified_id": str(canary["qualified_id"]),
            "solver": solver_name,
            "runtime_s": time.monotonic() - started,
            "natural_steps": int(dt.size),
            "amplitude_growth": float(np.max(np.abs(eta)))
            / max(float(np.max(np.abs(eta0))), 1e-30),
            "max_eta_over_depth": float(
                np.max(np.abs(eta) / effective_depth[None, ...])
            ),
            "cg_failure_count": int(
                np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
            ),
            "finite": bool(np.isfinite(eta).all()),
            "max_bracket_width": float(np.max(diagnostics["bracket_widths"])),
            "operator": solver.get_operator_diagnostics(),
        }, None
    raise ValueError(f"Unknown Level A task kind: {kind}")


def _write_task_artifact(
    task: Mapping[str, Any],
    tasks_root: Path,
    row: Mapping[str, Any],
    trajectory: np.ndarray | None,
) -> Path:
    final = tasks_root / _task_directory_name(task)
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite Level A task artifact: {final}")
    staging = tasks_root / f".{_task_directory_name(task)}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        array_hashes: dict[str, Any] = {}
        if trajectory is not None:
            values = np.asarray(trajectory)
            np.save(staging / "trajectory.npy", values, allow_pickle=False)
            array_hashes["trajectory"] = hash_array(values)
        scientific_hash = _scientific_digest({"row": row, "array_hashes": array_hashes})
        result = {
            "schema_id": TASK_RESULT_SCHEMA_ID,
            "task_id": task["task_id"],
            "task_spec_hash": task["task_spec_hash"],
            "row": _json_safe(row),
            "array_hashes": array_hashes,
            "scientific_hash": scientific_hash,
            "worker_metadata": {
                "pid": os.getpid(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "thread_environment": _thread_settings(),
            },
        }
        expected_files = ["SHA256SUMS.txt", "manifest.json", "result.json"]
        if trajectory is not None:
            expected_files.append("trajectory.npy")
        manifest = {
            "schema_id": TASK_ARTIFACT_SCHEMA_ID,
            "status": "complete",
            "task": _json_safe(task),
            "expected_files": expected_files,
            "scientific_hash": scientific_hash,
        }
        _write_json(staging / "result.json", result)
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging)
        os.replace(staging, final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _run_level_a_task_worker(task: Mapping[str, Any], tasks_root: str) -> str:
    row, trajectory = _compute_level_a_task(task)
    _write_task_artifact(task, Path(tasks_root), row, trajectory)
    return str(task["task_id"])


def _validate_task_result_semantics(
    task: Mapping[str, Any], row: Mapping[str, Any], trajectory: np.ndarray | None
) -> None:
    kind = str(task["kind"])
    spec = task["spec"]
    if kind == "fixture":
        expected = {"component": "fixture", "name": str(spec["name"])}
    elif kind == "analytical":
        expected = {
            "component": "analytical_mode",
            "solver": str(spec["solver"]),
            "analytical_role": str(spec["role"]),
            "grid": int(spec["grid"]),
            "mode": int(spec["mode"]),
            "cfl": float(spec["cfl"]),
        }
    elif kind == "operator":
        expected = {
            "component": "operator_sensitivity",
            "solver": str(spec["solver"]),
            "variant": str(spec["variant"]),
            "cfl": float(spec["cfl"]),
            "sponge_axes": str(spec["sponge_axes"]),
        }
    elif kind == "boundary":
        expected = {
            "component": "boundary_trajectory",
            "solver": str(spec["solver"]),
            "variant": str(spec["variant"]),
            "sponge_axes": str(spec["sponge_axes"]),
        }
    elif kind == "conservation":
        expected = {
            "component": "conservation_health",
            "solver": str(spec["solver"]),
            "boundary": str(spec["boundary"]),
            "measurement_dtype": "float64",
            "measurement_grid": "internal_natural_states",
        }
    elif kind == "canary":
        expected = {
            "component": "production_amplitude_canary",
            "solver": str(spec["solver"]),
            "qualified_id": str(spec["canary"]["qualified_id"]),
        }
    else:
        raise RuntimeError(f"Unknown Level A task kind in artifact: {kind}")
    for key, expected_value in expected.items():
        actual = row.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (float, int)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=0.0
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise RuntimeError(
                f"Level A task result/spec mismatch for {task['task_id']}: {key}"
            )
    requires_trajectory = kind in {"fixture", "analytical", "operator", "boundary"}
    if requires_trajectory != (trajectory is not None):
        raise RuntimeError(
            f"Level A task trajectory presence mismatch: {task['task_id']}"
        )


def _load_task_artifact(task: Mapping[str, Any], tasks_root: Path) -> dict[str, Any]:
    root = tasks_root / _task_directory_name(task)
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"Incomplete Level A task artifact: {task['task_id']}")
    result = _read_json(result_path)
    expected_files = {"manifest.json", "result.json", "SHA256SUMS.txt"}
    if result.get("array_hashes", {}).get("trajectory") is not None:
        expected_files.add("trajectory.npy")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_dirs = [path.name for path in root.iterdir() if path.is_dir()]
    if actual_files != expected_files or actual_dirs:
        raise RuntimeError(
            f"Unexpected files in Level A task artifact: {task['task_id']}"
        )
    validate_checksums(root)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_id") != TASK_ARTIFACT_SCHEMA_ID
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError(f"Invalid Level A task manifest: {task['task_id']}")
    if manifest.get("expected_files") != sorted(expected_files):
        raise RuntimeError(
            f"Level A task expected-file list mismatch: {task['task_id']}"
        )
    if manifest.get("task") != _json_safe(task):
        raise RuntimeError(
            f"Level A task semantic identity mismatch: {task['task_id']}"
        )
    if result.get("schema_id") != TASK_RESULT_SCHEMA_ID:
        raise RuntimeError(f"Invalid Level A task result schema: {task['task_id']}")
    if (
        result.get("task_id") != task["task_id"]
        or result.get("task_spec_hash") != task["task_spec_hash"]
    ):
        raise RuntimeError(f"Level A task result identity mismatch: {task['task_id']}")
    trajectory = None
    array_hashes = result.get("array_hashes", {})
    if "trajectory" in array_hashes:
        trajectory = np.load(root / "trajectory.npy", allow_pickle=False)
        if hash_array(trajectory) != array_hashes["trajectory"]:
            raise RuntimeError(
                f"Level A task trajectory semantic hash mismatch: {task['task_id']}"
            )
    scientific_hash = _scientific_digest(
        {"row": result.get("row"), "array_hashes": array_hashes}
    )
    if scientific_hash != result.get(
        "scientific_hash"
    ) or scientific_hash != manifest.get("scientific_hash"):
        raise RuntimeError(f"Level A task scientific hash mismatch: {task['task_id']}")
    _validate_task_result_semantics(task, result["row"], trajectory)
    return {
        "task": dict(task),
        "row": result["row"],
        "trajectory": trajectory,
        "result": result,
    }


def _recover_incomplete_task_staging(
    tasks: Sequence[Mapping[str, Any]], tasks_root: Path
) -> None:
    prefixes = {f".{_task_directory_name(task)}.staging-" for task in tasks}
    for path in list(tasks_root.iterdir()):
        matching = next(
            (
                prefix
                for prefix in prefixes
                if path.name.startswith(prefix) and path.name[len(prefix) :].isdigit()
            ),
            None,
        )
        if matching is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"Unsafe Level A task staging path: {path.name}")
        shutil.rmtree(path)


def _scan_task_artifacts(
    tasks: Sequence[Mapping[str, Any]], tasks_root: Path
) -> tuple[dict[str, dict[str, Any]], list[Mapping[str, Any]]]:
    expected = {_task_directory_name(task): task for task in tasks}
    if tasks_root.exists():
        extras = sorted(
            path.name for path in tasks_root.iterdir() if path.name not in expected
        )
        if extras:
            raise RuntimeError(
                f"Unexpected or partial Level A task artifacts: {extras}"
            )
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[Mapping[str, Any]] = []
    for name, task in expected.items():
        path = tasks_root / name
        if not path.exists():
            missing.append(task)
            continue
        if not path.is_dir():
            raise RuntimeError(f"Level A task artifact is not a directory: {name}")
        payload = _load_task_artifact(task, tasks_root)
        task_id = str(task["task_id"])
        if task_id in loaded:
            raise RuntimeError(f"Duplicate Level A task artifact: {task_id}")
        loaded[task_id] = payload
    return loaded, missing


def _require_single_thread_backends() -> None:
    invalid = {
        key: value for key in THREAD_ENV_KEYS if (value := os.environ.get(key)) != "1"
    }
    if invalid:
        details = ", ".join(
            f"{key}={value!r}" for key, value in sorted(invalid.items())
        )
        raise RuntimeError(
            "Level A multiprocessing requires single-thread numerical backends: "
            + details
        )


def _execute_level_a_task_plan(
    tasks: Sequence[Mapping[str, Any]],
    *,
    tasks_root: Path,
    workers: int = 1,
    max_in_flight: int | None = None,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_in_flight is not None and max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    _validate_task_plan(tasks)
    tasks_root.mkdir(parents=True, exist_ok=True)
    if resume:
        _recover_incomplete_task_staging(tasks, tasks_root)
    loaded, missing = _scan_task_artifacts(tasks, tasks_root)
    if loaded and not resume:
        raise FileExistsError(f"Level A task artifacts already exist: {tasks_root}")
    effective = min(int(workers), max(1, len(missing))) if missing else 0
    if (
        missing
        and effective > 1
        and max_in_flight is not None
        and max_in_flight < effective
    ):
        raise ValueError(
            "max_in_flight must be at least the effective worker count "
            f"({effective})"
        )
    effective_max_in_flight = (
        min(
            len(missing),
            int(max_in_flight) if max_in_flight is not None else 2 * effective,
        )
        if missing and workers > 1
        else 0
    )
    peak_in_flight_futures = 0
    if missing and workers > 1:
        _require_single_thread_backends()
    if missing and workers == 1:
        for task in missing:
            try:
                returned = _run_level_a_task_worker(task, str(tasks_root))
            except Exception as exc:
                raise RuntimeError(f"Level A task failed: {task['task_id']}") from exc
            if returned != task["task_id"]:
                raise RuntimeError(
                    "Level A worker returned wrong task identity for "
                    f"{task['task_id']}: {returned}"
                )
    elif missing:
        context = multiprocessing.get_context("spawn")
        task_iter = iter(missing)
        in_flight: dict[Any, Mapping[str, Any]] = {}
        pool = ProcessPoolExecutor(max_workers=effective, mp_context=context)

        def submit_until_full() -> None:
            nonlocal peak_in_flight_futures
            while len(in_flight) < effective_max_in_flight:
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                future = pool.submit(
                    _run_level_a_task_worker, task, str(tasks_root)
                )
                in_flight[future] = task
            peak_in_flight_futures = max(
                peak_in_flight_futures, len(in_flight)
            )

        try:
            submit_until_full()
            while in_flight:
                completed, _ = wait(
                    tuple(in_flight), return_when=FIRST_COMPLETED
                )
                completed_tasks = sorted(
                    ((in_flight.pop(future), future) for future in completed),
                    key=lambda pair: int(pair[0]["ordinal"]),
                )
                for task, future in completed_tasks:
                    try:
                        returned = future.result()
                    except Exception as exc:
                        for pending in in_flight:
                            pending.cancel()
                        raise RuntimeError(
                            f"Level A task failed: {task['task_id']}"
                        ) from exc
                    if returned != task["task_id"]:
                        for pending in in_flight:
                            pending.cancel()
                        raise RuntimeError(
                            "Level A worker returned wrong task identity for "
                            f"{task['task_id']}: {returned}"
                        )
                submit_until_full()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    loaded, remaining = _scan_task_artifacts(tasks, tasks_root)
    if remaining:
        raise RuntimeError(
            "Missing Level A task artifacts after execution: "
            + ", ".join(str(task["task_id"]) for task in remaining)
        )
    ordered = [loaded[str(task["task_id"])] for task in tasks]
    worker_history = []
    for payload in ordered:
        metadata = payload["result"].get("worker_metadata", {})
        entry = {
            "pid": metadata.get("pid"),
            "python_version": metadata.get("python_version"),
            "numpy_version": metadata.get("numpy_version"),
            "thread_environment": metadata.get("thread_environment"),
        }
        if entry not in worker_history:
            worker_history.append(entry)
    provenance = _operational_provenance(
        requested_workers=int(workers),
        effective_workers=effective,
        requested_max_in_flight=max_in_flight,
        effective_max_in_flight=effective_max_in_flight,
        peak_in_flight_futures=peak_in_flight_futures,
        process_start_method=("spawn" if missing and workers > 1 else "serial"),
    )
    provenance["task_worker_history"] = worker_history
    return ordered, provenance


def _trajectory_eta(
    name: str,
    *,
    nx: int,
    ny: int,
    cfl: float,
    boundary: str,
    use_sponge: bool,
    sponge_mode: str,
    filter_mode: str = "disabled",
    filter_strength: float = 0.0,
    bathymetry: np.ndarray | None = None,
    eta0: np.ndarray | None = None,
    h0: np.ndarray | None = None,
    hu0: np.ndarray | None = None,
    hv0: np.ndarray | None = None,
    eta_t0: np.ndarray | None = None,
    sponge_axes: str = "xy",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], Any]:
    solver = _solver(
        name,
        nx=nx,
        ny=ny,
        cfl=cfl,
        boundary=boundary,
        use_sponge=use_sponge,
        sponge_mode=sponge_mode,
        filter_mode=filter_mode,
        filter_strength=filter_strength,
        sponge_axes=sponge_axes,
    )
    bathy = (
        -np.ones((nx, ny), dtype=float)
        if bathymetry is None
        else np.asarray(bathymetry, dtype=float)
    )
    initial_eta = (
        np.zeros((nx, ny), dtype=float)
        if eta0 is None
        else np.asarray(eta0, dtype=float)
    )
    solver.set_bathymetry(bathy)
    if name == "boussinesq":
        solver.set_initial_condition(
            initial_eta,
            eta_t0=(
                np.zeros_like(initial_eta)
                if eta_t0 is None
                else np.asarray(eta_t0, dtype=float)
            ),
        )
    else:
        initial_h = (
            np.maximum(-bathy + initial_eta, 0.0)
            if h0 is None
            else np.asarray(h0, dtype=float)
        )
        solver.set_initial_condition(
            initial_h,
            hu0=(
                np.zeros_like(initial_h)
                if hu0 is None
                else np.asarray(hu0, dtype=float)
            ),
            hv0=(
                np.zeros_like(initial_h)
                if hv0 is None
                else np.asarray(hv0, dtype=float)
            ),
        )
    states, _, dt_history, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=cfl,
        include_initial_state=False,
        requested_times=candidate_requested_times(),
        max_natural_steps=20000,
        collect_natural_step_health=True,
    )
    eta = states[:, 0] if name == "boussinesq" else states[:, 0] + bathy
    return eta, dt_history, diagnostics, solver


def _stable_sum(values: np.ndarray) -> float:
    return float(math.fsum(np.asarray(values, dtype=np.float64).ravel()))


def _stable_l2_norm(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return float(math.sqrt(math.fsum(float(value) * float(value) for value in flat)))


def _stable_mean(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("stable mean requires at least one value")
    return _stable_sum(flat) / flat.size


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return float(_stable_l2_norm(aa - bb) / max(_stable_l2_norm(bb), 1e-30))


def _packet(
    nx: int,
    ny: int,
    *,
    center: float = 0.25,
    sigma: float = 0.04,
    zero_mean: bool = True,
) -> np.ndarray:
    x = np.arange(nx, dtype=float)[:, None] / nx
    eta = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    if zero_mean:
        eta -= float(np.mean(eta))
    return 1.0e-5 * eta * np.ones((1, ny), dtype=float)


def _window_mask(nx: int, window: Sequence[float]) -> np.ndarray:
    if len(window) != 2:
        raise ValueError("boundary window must contain exactly two values")
    lower, upper = (float(value) for value in window)
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("boundary window must satisfy 0 <= lower < upper <= 1")
    x = np.arange(nx, dtype=np.float64) / nx
    mask = (x >= lower) & (x <= upper)
    if not np.any(mask):
        raise ValueError("boundary window selects no grid cells")
    return mask


def _x_sponge_region(nx: int, width: int) -> np.ndarray:
    effective = min(max(0, int(width)), max(1, nx // 2))
    mask = np.zeros(nx, dtype=bool)
    if effective:
        mask[:effective] = True
        mask[-effective:] = True
    return mask


def _validate_boundary_packet_spec(
    spec: Mapping[str, Any], *, nx: int, sponge_width: int
) -> None:
    center = float(spec["center_x"])
    sigma = float(spec["sigma"])
    if not (0.0 < center < 1.0) or sigma <= 0.0:
        raise ValueError("boundary packet center and sigma must be positive and in-domain")
    incident = _window_mask(nx, spec["incident_window"])
    reflected = _window_mask(nx, spec["reflected_window"])
    interior = _window_mask(nx, spec["interior_window"])
    if np.any(incident & reflected):
        raise ValueError("incident and reflected windows must not overlap")
    if np.any(interior & _x_sponge_region(nx, sponge_width)):
        raise ValueError("boundary interior window overlaps the x-only sponge")
    packet = _packet(nx, 1, center=center, sigma=sigma, zero_mean=False)[:, 0]
    initial_amp = max(float(np.max(np.abs(packet))), 1e-30)
    initial_reflected_ratio = float(np.max(np.abs(packet[reflected]))) / initial_amp
    if initial_reflected_ratio > float(spec["maximum_initial_reflected_ratio"]):
        raise ValueError("initial packet is not outside the reflected window")
    incident_energy_fraction = float(np.sum(packet[incident] ** 2)) / max(
        float(np.sum(packet**2)), 1e-30
    )
    if incident_energy_fraction < float(spec["minimum_initial_incident_energy_fraction"]):
        raise ValueError("incident window does not contain enough initial packet energy")
    arrival = float(spec["expected_boundary_arrival_time"])
    evaluation = float(spec["evaluation_time"])
    times = candidate_requested_times()
    if arrival <= 0.0 or evaluation <= arrival:
        raise ValueError("boundary evaluation must occur after expected arrival")
    if not np.any(np.isclose(times, evaluation, rtol=0.0, atol=1e-15)):
        raise ValueError("boundary evaluation_time must be a requested timestamp")


def _boundary_initial_conditions(
    solver_name: str, *, nx: int, ny: int, spec: Mapping[str, Any]
) -> dict[str, np.ndarray | float]:
    eta0 = _packet(
        nx,
        ny,
        center=float(spec["center_x"]),
        sigma=float(spec["sigma"]),
        zero_mean=False,
    )
    direction = str(spec.get("direction", "left"))
    if direction not in ("left", "right"):
        raise ValueError("boundary packet direction must be left or right")
    sign = -1.0 if direction == "left" else 1.0
    if solver_name == "boussinesq":
        dominant_wavenumber = 1.0 / float(spec["sigma"])
        phase_speed = math.sqrt(9.81) / math.sqrt(
            1.0 + dominant_wavenumber * dominant_wavenumber / 3.0
        )
        derivative = np.gradient(eta0, 1.0 / nx, axis=0, edge_order=2)
        eta_t0 = -sign * phase_speed * derivative
        return {
            "eta0": eta0,
            "eta_t0": eta_t0,
            "characteristic_speed": phase_speed,
        }
    phase_speed = math.sqrt(9.81)
    return {
        "eta0": eta0,
        "h0": 1.0 + eta0,
        "hu0": sign * phase_speed * eta0,
        "characteristic_speed": phase_speed,
    }


def _operator_discrepancy_metrics(
    production: np.ndarray,
    half_cfl: np.ndarray,
    *,
    sponge_region: np.ndarray,
    interior_region: np.ndarray,
) -> dict[str, float]:
    first = np.asarray(production)
    second = np.asarray(half_cfl)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("operator trajectories must have equal [time, x, y] shape")
    if sponge_region.shape != (first.shape[1],) or interior_region.shape != (
        first.shape[1],
    ):
        raise ValueError("operator region masks do not match the x dimension")
    if not np.any(sponge_region) or not np.any(interior_region):
        raise ValueError("operator regions must both be non-empty")
    return {
        "trajectory_relative_l2": _relative_l2(first, second),
        "final_time_relative_l2": _relative_l2(first[-1], second[-1]),
        "sponge_trajectory_relative_l2": _relative_l2(
            first[:, sponge_region, :], second[:, sponge_region, :]
        ),
        "sponge_final_time_relative_l2": _relative_l2(
            first[-1, sponge_region, :], second[-1, sponge_region, :]
        ),
        "interior_trajectory_relative_l2": _relative_l2(
            first[:, interior_region, :], second[:, interior_region, :]
        ),
        "interior_final_time_relative_l2": _relative_l2(
            first[-1, interior_region, :], second[-1, interior_region, :]
        ),
    }


def _boundary_metrics(
    *,
    baseline: np.ndarray,
    damped: np.ndarray,
    initial_packet: np.ndarray,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline.shape != damped.shape or baseline.ndim != 3:
        raise ValueError("boundary trajectories must have equal [time, x, y] shape")
    nx = int(baseline.shape[1])
    incident = _window_mask(nx, spec["incident_window"])
    reflected = _window_mask(nx, spec["reflected_window"])
    interior = _window_mask(nx, spec["interior_window"])
    times = candidate_requested_times()
    evaluation_matches = np.flatnonzero(
        np.isclose(times, float(spec["evaluation_time"]), rtol=0.0, atol=1e-15)
    )
    if evaluation_matches.size != 1:
        raise ValueError("boundary evaluation time is not unique")
    evaluation_index = int(evaluation_matches[0])
    arrival_index = int(
        np.searchsorted(
            times, float(spec["expected_boundary_arrival_time"]), side="left"
        )
    )
    if arrival_index > evaluation_index:
        raise ValueError("boundary evaluation precedes expected arrival")
    incident_amplitude = max(
        float(np.max(np.abs(initial_packet[incident]))), 1e-30
    )
    incident_energy = max(
        float(np.sum(initial_packet[incident] ** 2)), 1e-30
    )
    baseline_post_arrival = baseline[arrival_index : evaluation_index + 1, reflected]
    arrival_energy_ratio = float(np.max(np.sum(baseline_post_arrival**2, axis=(1, 2)))) / (
        incident_energy
    )
    reflected_frame = damped[evaluation_index, reflected]
    return {
        "evaluation_time": float(times[evaluation_index]),
        "expected_boundary_arrival_time": float(
            spec["expected_boundary_arrival_time"]
        ),
        "arrival_energy_ratio": arrival_energy_ratio,
        "arrival_observed": arrival_energy_ratio
        >= float(spec["minimum_arrival_energy_ratio"]),
        "incident_amplitude": incident_amplitude,
        "incident_energy": incident_energy,
        "reflected_amplitude_ratio": float(np.max(np.abs(reflected_frame)))
        / incident_amplitude,
        "reflected_energy_ratio": float(np.sum(reflected_frame**2))
        / incident_energy,
        "interior_relative_l2": _relative_l2(
            damped[evaluation_index, interior], baseline[evaluation_index, interior]
        ),
    }


def _summation_roundoff_floor(
    values: np.ndarray, *, safety_factor: float
) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    n = max(1, int(flat.size))
    eps = np.finfo(np.float64).eps
    gamma = (n * eps) / max(1.0 - n * eps, eps)
    return float(safety_factor * gamma * math.fsum(abs(float(v)) for v in flat))


def _invariant_metrics(
    values: Sequence[float],
    *,
    normalization_scale: float,
    roundoff_floor_absolute: float,
) -> dict[str, float]:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 2 or not np.isfinite(observed).all():
        raise ValueError("invariant history must contain at least two finite values")
    initial = float(observed[0])
    final = float(observed[-1])
    absolute_drift = abs(final - initial)
    max_absolute_drift = float(np.max(np.abs(observed - initial)))
    scale = max(abs(float(normalization_scale)), 1e-30)
    return {
        "initial_value": initial,
        "final_value": final,
        "absolute_drift": absolute_drift,
        "max_absolute_drift": max_absolute_drift,
        "normalization_scale": scale,
        "normalized_drift": max_absolute_drift / scale,
        "roundoff_floor_absolute": float(roundoff_floor_absolute),
        "roundoff_floor_normalized": float(roundoff_floor_absolute) / scale,
    }


def _run_float64_conservation(
    solver_name: str,
    *,
    nx: int,
    ny: int,
    cfl: float,
    boundary: str,
    safety_factor: float,
) -> dict[str, Any]:
    eta0 = _packet(nx, ny, center=0.5, sigma=0.06)
    solver = _solver(
        solver_name,
        nx=nx,
        ny=ny,
        cfl=cfl,
        boundary=boundary,
        use_sponge=False,
    )
    bathymetry = -np.ones((nx, ny), dtype=np.float64)
    solver.set_bathymetry(bathymetry)
    if solver_name == "boussinesq":
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
        invariant_name = "free_surface_integral"
        invariant_array = np.asarray(solver.eta, dtype=np.float64)
        normalization_scale = math.fsum(
            abs(float(value)) for value in invariant_array.ravel()
        )

        def invariant() -> float:
            return _stable_sum(solver.eta)

    else:
        solver.set_initial_condition(
            1.0 + eta0,
            hu0=np.zeros_like(eta0),
            hv0=np.zeros_like(eta0),
        )
        invariant_name = "total_water_depth"
        invariant_array = np.asarray(solver.h, dtype=np.float64)
        normalization_scale = abs(_stable_sum(invariant_array))

        def invariant() -> float:
            return _stable_sum(solver.h)

    if invariant_array.dtype != np.float64:
        raise RuntimeError("Level A conservation must observe float64 solver state")
    roundoff_floor = _summation_roundoff_floor(
        invariant_array, safety_factor=safety_factor
    )
    history = [invariant()]
    horizon = float(candidate_requested_times()[-1])
    current_time = 0.0
    natural_steps = 0
    cg_failures = 0
    while current_time < horizon:
        proposed = float(solver.suggest_dt(target_cfl=cfl))
        dt = min(proposed, horizon - current_time)
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("invalid natural conservation timestep")
        solver.dt = dt
        solver.step(dt=dt, auto_dt=False)
        current_time += dt
        natural_steps += 1
        if natural_steps > 20000:
            raise RuntimeError("natural conservation rollout exceeded step cap")
        cg_failures += int(getattr(solver, "last_step_cg_failed_count", 0))
        state = np.asarray(solver.get_state())
        if state.dtype != np.float64 or not np.isfinite(state).all():
            raise RuntimeError("conservation state lost float64 finite semantics")
        history.append(invariant())
    metrics = _invariant_metrics(
        history,
        normalization_scale=normalization_scale,
        roundoff_floor_absolute=roundoff_floor,
    )
    return {
        "component": "conservation_health",
        "solver": solver_name,
        "boundary": boundary,
        "invariant_name": invariant_name,
        **metrics,
        "measurement_dtype": "float64",
        "measurement_grid": "internal_natural_states",
        "precision_floor_method": "float64_gamma_n_l1",
        "precision_floor_safety_factor": float(safety_factor),
        "natural_steps": natural_steps,
        "final_time": current_time,
        "cg_failure_count": cg_failures,
        "finite": True,
        "operator": solver.get_operator_diagnostics(),
    }


def _high_frequency_fraction(trajectory: np.ndarray) -> float:
    eta = np.asarray(trajectory[-1], dtype=float)
    spectrum = np.abs(np.fft.rfft(eta, axis=0)) ** 2
    total = float(np.sum(spectrum))
    if total <= 1e-30:
        return 0.0
    cutoff = max(1, spectrum.shape[0] * 3 // 4)
    return float(np.sum(spectrum[cutoff:]) / total)


def _run_operator_component(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    nx, ny = 64, 4
    eta0 = _packet(nx, ny)
    threshold = float(
        config["thresholds"]["operator"]["elapsed_candidate_cfl_relative_l2"]
    )
    for name in SOLVERS:
        production, half = [float(v) for v in config["production"]["cfl"][name][:2]]
        modes = [
            (
                "legacy_per_step",
                "legacy_per_step" if name == "boussinesq" else "disabled",
                0.01 if name == "boussinesq" else 0.0,
            ),
            ("elapsed_no_filter", "disabled", 0.0),
        ]
        if name == "boussinesq":
            modes.append(("elapsed_filter", "elapsed_time_consistent", 0.01))
        by_variant: dict[str, dict[float, np.ndarray]] = {}
        for variant, filter_mode, strength in modes:
            sponge_mode = (
                "legacy_per_step"
                if variant == "legacy_per_step"
                else "elapsed_time_consistent"
            )
            for cfl in (production, half):
                eta, dt, diagnostics, solver = _trajectory_eta(
                    name,
                    nx=nx,
                    ny=ny,
                    cfl=cfl,
                    boundary="open",
                    use_sponge=True,
                    sponge_mode=sponge_mode,
                    filter_mode=filter_mode,
                    filter_strength=strength,
                    eta0=eta0,
                )
                by_variant.setdefault(variant, {})[cfl] = eta
                rows.append(
                    {
                        "component": "operator_sensitivity",
                        "solver": name,
                        "variant": variant,
                        "cfl": cfl,
                        "natural_steps": int(dt.size),
                        "amplitude_ratio": float(
                            np.max(np.abs(eta)) / max(np.max(np.abs(eta0)), 1e-30)
                        ),
                        "high_frequency_fraction": _high_frequency_fraction(eta),
                        "cg_failure_count": int(
                            np.sum(
                                np.asarray(
                                    diagnostics.get("cg_failed_count", []), dtype=int
                                )
                            )
                        ),
                        "operator": solver.get_operator_diagnostics(),
                    }
                )
        legacy_diff = _relative_l2(
            by_variant["legacy_per_step"][production],
            by_variant["legacy_per_step"][half],
        )
        elapsed_diff = _relative_l2(
            by_variant["elapsed_no_filter"][production],
            by_variant["elapsed_no_filter"][half],
        )
        rows.append(
            {
                "component": "operator_sensitivity_summary",
                "solver": name,
                "legacy_cfl_relative_l2": legacy_diff,
                "elapsed_no_filter_cfl_relative_l2": elapsed_diff,
            }
        )
        gates.append(
            {
                "gate": f"elapsed_operator_consistency_{name}",
                "category": "blocked_operator_semantics",
                "passed": elapsed_diff <= threshold and elapsed_diff <= legacy_diff,
                "legacy_relative_l2": legacy_diff,
                "elapsed_relative_l2": elapsed_diff,
            }
        )
        if name == "boussinesq":
            primary = next(
                row
                for row in rows
                if row.get("component") == "operator_sensitivity"
                and row.get("solver") == name
                and row.get("variant") == "elapsed_no_filter"
                and math.isclose(float(row["cfl"]), production)
            )
            fallback = next(
                row
                for row in rows
                if row.get("component") == "operator_sensitivity"
                and row.get("solver") == name
                and row.get("variant") == "elapsed_filter"
                and math.isclose(float(row["cfl"]), production)
            )
            initial_high_frequency_fraction = _high_frequency_fraction(eta0[None, ...])
            high_frequency_limit = max(
                2.0 * initial_high_frequency_fraction,
                float(
                    config["thresholds"]["boussinesq_no_filter"][
                        "high_frequency_fraction_absolute"
                    ]
                ),
            )
            no_filter_ok = (
                primary["cg_failure_count"] == 0
                and primary["amplitude_ratio"]
                <= float(
                    config["thresholds"]["boussinesq_no_filter"]["amplitude_growth"]
                )
                and primary["high_frequency_fraction"] <= high_frequency_limit
                and elapsed_diff <= threshold
            )
            filter_diff = _relative_l2(
                by_variant["elapsed_filter"][production],
                by_variant["elapsed_filter"][half],
            )
            filter_ok = (
                fallback["cg_failure_count"] == 0
                and fallback["amplitude_ratio"]
                <= float(
                    config["thresholds"]["boussinesq_no_filter"]["amplitude_growth"]
                )
                and fallback["high_frequency_fraction"] <= high_frequency_limit
                and filter_diff <= threshold
            )
            gates.extend(
                [
                    {
                        "gate": "boussinesq_no_filter_health",
                        "category": "informational",
                        "passed": no_filter_ok,
                        "high_frequency_limit": high_frequency_limit,
                    },
                    {
                        "gate": "boussinesq_elapsed_filter_fallback",
                        "category": "informational",
                        "passed": filter_ok,
                        "cfl_relative_l2": filter_diff,
                        "high_frequency_limit": high_frequency_limit,
                    },
                    {
                        "gate": "boussinesq_filter_acceptance",
                        "category": "blocked_boussinesq_health",
                        "passed": no_filter_ok or filter_ok,
                        "recommended_filter": (
                            "disabled"
                            if no_filter_ok
                            else "elapsed_time_consistent"
                            if filter_ok
                            else "none"
                        ),
                    },
                ]
            )
    reference_dt = float(config["operators"]["sponge"]["reference_dt"])
    m_ref = float(config["operators"]["sponge"]["reference_min_factor"])
    product = 1.0
    elapsed = 0.0
    for dt in (0.0011, 0.0007, 0.0017):
        product *= m_ref ** (dt / reference_dt)
        elapsed += dt
    expected = m_ref ** (elapsed / reference_dt)
    rel = abs(product - expected) / expected
    rows.append(
        {
            "component": "operator_factor_identity",
            "observed": product,
            "expected": expected,
            "relative_error": rel,
        }
    )
    gates.append(
        {
            "gate": "elapsed_sponge_factor_identity",
            "category": "blocked_operator_semantics",
            "passed": rel
            <= float(
                config["thresholds"]["operator"]["accumulated_factor_relative_error"]
            ),
        }
    )
    return rows, gates


def _run_boundary_component(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    nx, ny = 128, 4
    spec = config["boundary_packet"]
    eta0 = _packet(nx, ny, center=float(spec["center_x"]), sigma=float(spec["sigma"]))
    x = np.arange(nx, dtype=float) / nx
    reflected = (x >= float(spec["reflected_window"][0])) & (
        x <= float(spec["reflected_window"][1])
    )
    interior = (x >= float(spec["interior_window"][0])) & (
        x <= float(spec["interior_window"][1])
    )
    thresholds = config["thresholds"]["boundary"]
    for name in SOLVERS:
        cfl = float(config["production"]["cfl"][name][0])
        baseline, _, _, _ = _trajectory_eta(
            name,
            nx=nx,
            ny=ny,
            cfl=cfl,
            boundary="open",
            use_sponge=False,
            sponge_mode="legacy_per_step",
            eta0=eta0,
        )
        damped, _, _, _ = _trajectory_eta(
            name,
            nx=nx,
            ny=ny,
            cfl=cfl,
            boundary="open",
            use_sponge=True,
            sponge_mode="elapsed_time_consistent",
            eta0=eta0,
        )
        initial_amp = max(float(np.max(np.abs(eta0))), 1e-30)
        reflected_amp = float(np.max(np.abs(damped[-1, reflected]))) / initial_amp
        reflected_energy = float(np.sum(damped[-1, reflected] ** 2)) / max(
            float(np.sum(eta0**2)), 1e-30
        )
        interior_l2 = _relative_l2(damped[-1, interior], baseline[-1, interior])
        row = {
            "component": "boundary_sponge",
            "solver": name,
            "boundary_implementation": "zero_gradient_edge_padding",
            "reflected_amplitude_ratio": reflected_amp,
            "reflected_energy_ratio": reflected_energy,
            "interior_relative_l2": interior_l2,
        }
        rows.append(row)
        gates.append(
            {
                "gate": f"boundary_sponge_{name}",
                "category": "blocked_boundary_behavior",
                "passed": reflected_amp
                <= float(thresholds["reflected_amplitude_ratio"])
                and reflected_energy <= float(thresholds["reflected_energy_ratio"])
                and interior_l2 <= float(thresholds["interior_relative_l2"]),
            }
        )
    return rows, gates


def _run_conservation_component(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    nx, ny = 64, 4
    eta0 = _packet(nx, ny, center=0.5, sigma=0.06)
    thresholds = config["thresholds"]["conservation"]
    for boundary in ("periodic", "reflective"):
        for name in SOLVERS:
            cfl = float(config["production"]["cfl"][name][0])
            eta, _, diagnostics, _ = _trajectory_eta(
                name,
                nx=nx,
                ny=ny,
                cfl=cfl,
                boundary=boundary,
                use_sponge=False,
                sponge_mode="legacy_per_step",
                eta0=eta0,
            )
            mean0 = float(np.sum(eta0))
            mean_drift = float(np.max(np.abs(np.sum(eta, axis=(1, 2)) - mean0))) / max(
                float(np.sum(np.abs(eta0))), 1e-30
            )
            if name == "boussinesq":
                mass_drift = None
            else:
                mass0 = float(np.sum(1.0 + eta0))
                mass_drift = float(
                    np.max(np.abs(np.sum(1.0 + eta, axis=(1, 2)) - mass0))
                    / max(abs(mass0), 1e-30)
                )
            cg_failures = int(
                np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
            )
            row = {
                "component": "conservation_health",
                "solver": name,
                "boundary": boundary,
                "mass_relative_drift": mass_drift,
                "mean_integral_relative_drift": mean_drift,
                "cg_failure_count": cg_failures,
                "finite": bool(np.isfinite(eta).all()),
            }
            rows.append(row)
            gates.append(
                {
                    "gate": f"conservation_{boundary}_{name}",
                    "category": "blocked_convergence",
                    "passed": row["finite"]
                    and cg_failures == 0
                    and mean_drift <= float(thresholds["mean_integral_relative_drift"])
                    and (
                        mass_drift is None
                        or mass_drift <= float(thresholds["mass_relative_drift"])
                    ),
                }
            )
    return rows, gates


def _run_canary_component(
    config: Mapping[str, Any], canaries: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    thresholds = config["thresholds"]["canary"]
    for canary in canaries:
        bathymetry, source, _strength_array, strength, arrays = _load_canary_arrays(
            canary
        )
        eta0 = arrays["eta0"]
        rest = arrays["rest_depth"]
        h0 = arrays["initial_depth"]
        for name in SOLVERS:
            cfl = float(config["production"]["cfl"][name][0])
            started = time.monotonic()
            eta, dt, diagnostics, solver = _trajectory_eta(
                name,
                nx=bathymetry.shape[0],
                ny=bathymetry.shape[1],
                cfl=cfl,
                boundary="open",
                use_sponge=True,
                sponge_mode="elapsed_time_consistent",
                filter_mode="disabled",
                bathymetry=bathymetry,
                eta0=eta0,
                h0=h0,
            )
            effective_depth = (
                np.maximum(-bathymetry, 1e-4)
                if name == "boussinesq"
                else np.maximum(rest, 1e-8)
            )
            amplitude = float(np.max(np.abs(eta))) / max(
                float(np.max(np.abs(eta0))), 1e-30
            )
            eta_over_depth = float(np.max(np.abs(eta) / effective_depth[None, ...]))
            cg_failures = int(
                np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
            )
            row = {
                "component": "production_amplitude_canary",
                "qualified_id": canary["qualified_id"],
                "solver": name,
                "runtime_s": time.monotonic() - started,
                "natural_steps": int(dt.size),
                "amplitude_growth": amplitude,
                "max_eta_over_depth": eta_over_depth,
                "cg_failure_count": cg_failures,
                "finite": bool(np.isfinite(eta).all()),
                "max_bracket_width": float(np.max(diagnostics["bracket_widths"])),
                "operator": solver.get_operator_diagnostics(),
            }
            rows.append(row)
            gates.append(
                {
                    "gate": f"canary_{canary['qualified_id']}_{name}",
                    "category": (
                        "blocked_boussinesq_health"
                        if name == "boussinesq"
                        else "blocked_convergence"
                    ),
                    "passed": row["finite"]
                    and cg_failures == 0
                    and amplitude <= float(thresholds["amplitude_growth"])
                    and eta_over_depth <= float(thresholds["eta_over_depth"]),
                }
            )
    return rows, gates


def _bootstrap_canary_aggregates(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int
) -> list[dict[str, Any]]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    by_solver: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("component") == "production_amplitude_canary":
            by_solver.setdefault(str(row["solver"]), []).append(row)
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for solver, solver_rows in sorted(by_solver.items()):
        ordered = sorted(solver_rows, key=lambda row: str(row["qualified_id"]))
        if not ordered:
            continue
        for metric in ("amplitude_growth", "max_eta_over_depth", "runtime_s"):
            values = np.asarray([row[metric] for row in ordered], dtype=np.float64)
            draws = rng.integers(0, values.size, size=(resamples, values.size))
            means = np.asarray(
                [_stable_mean(values[indices]) for indices in draws],
                dtype=np.float64,
            )
            summaries.append(
                {
                    "component": "canary_bootstrap_descriptive",
                    "solver": solver,
                    "metric": metric,
                    "scenario_count": int(values.size),
                    "resamples": int(resamples),
                    "seed": int(seed),
                    "mean": _stable_mean(values),
                    "mean_ci95_low": float(np.quantile(means, 0.025)),
                    "mean_ci95_high": float(np.quantile(means, 0.975)),
                    "decision_role": "descriptive_only",
                }
            )
    return summaries


def _universal_health_gate(
    rows: Sequence[Mapping[str, Any]], *, universal: Mapping[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    expected_count = int(universal["exact_output_count"])
    expected_cg_failures = int(universal["cg_failure_count"])
    expected_replacements = int(universal["nan_to_num_replacement_count"])
    for index, row in enumerate(rows):
        component = str(row.get("component", "unknown"))
        label = f"{component}[{index}]"
        if component == "analytical_mode":
            if int(row.get("output_count", -1)) != expected_count:
                failures.append(f"{label}: output_count")
            if not bool(row.get("requested_times_exact", False)):
                failures.append(f"{label}: requested_times")
        if (
            "finite" in row
            and bool(universal["require_finite"])
            and not bool(row["finite"])
        ):
            failures.append(f"{label}: nonfinite")
        if int(row.get("cg_failure_count", 0)) != expected_cg_failures:
            failures.append(f"{label}: cg_failure_count")
        operator = row.get("operator")
        if isinstance(operator, Mapping):
            if "nan_to_num_replacement_count" not in operator:
                failures.append(f"{label}: missing_nan_to_num_replacement_count")
            elif int(operator["nan_to_num_replacement_count"]) != expected_replacements:
                failures.append(f"{label}: nan_to_num_replacement_count")
        elif component in {
            "analytical_mode",
            "operator_sensitivity",
            "boundary_sponge",
            "conservation_health",
            "production_amplitude_canary",
        }:
            failures.append(f"{label}: missing_operator_diagnostics")
    return {
        "gate": "universal_health_and_output_contract",
        "category": "implementation_failure",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _decision_from_gates(gates: Sequence[Mapping[str, Any]]) -> str:
    failed = {str(row["category"]) for row in gates if not bool(row["passed"])}
    for decision in (
        "implementation_failure",
        "blocked_boussinesq_health",
        "blocked_boundary_behavior",
        "blocked_convergence",
        "blocked_operator_semantics",
    ):
        if decision in failed:
            return decision
    return "pass_to_H1"


def _aggregate_level_a_tasks(
    task_results: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    operational_provenance: Mapping[str, Any],
    elapsed_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    analytical_payloads = [
        payload for payload in task_results if payload["task"]["kind"] == "analytical"
    ]
    for payload in analytical_payloads:
        row = dict(payload["row"])
        row["_trajectory_eta"] = payload["trajectory"]
        rows.append(row)

    thresholds = config["thresholds"]
    for name in SOLVERS:
        spatial = [
            row
            for row in rows
            if row["solver"] == name and row["analytical_role"] == "spatial"
        ]
        temporal = [
            row
            for row in rows
            if row["solver"] == name and row["analytical_role"] == "temporal"
        ]
        modal = [
            row
            for row in rows
            if row["solver"] == name and row["analytical_role"] == "modal"
        ]
        if (
            len(spatial) != 3
            or len(temporal) != 3
            or len(modal) != len(config["analytical"]["modes"])
        ):
            raise RuntimeError(f"Incomplete analytical task roles for {name}")
        spatial.sort(key=lambda row: int(row["grid"]))
        temporal.sort(key=lambda row: float(row["cfl"]), reverse=True)
        modal.sort(key=lambda row: int(row["mode"]))
        finest = spatial[-1]
        analytical = thresholds["analytical"][name]
        passed = (
            finest["finite"]
            and finest["output_count"]
            == int(thresholds["universal"]["exact_output_count"])
            and finest["requested_times_exact"]
            and finest["cg_failure_count"] == 0
            and finest["nan_to_num_replacement_count"]
            == int(thresholds["universal"]["nan_to_num_replacement_count"])
            and finest["positivity_projection_count"] == 0
            and finest["dry_projection_count"] == 0
            and finest["phase_speed_relative_error"]
            <= analytical["phase_speed_relative_error"]
            and finest["amplitude_drift"] <= analytical["amplitude_drift"]
            and finest["field_relative_l2"] <= analytical["field_relative_l2"]
        )
        gates.append(
            {
                "gate": f"analytical_{name}",
                "category": (
                    "blocked_boussinesq_health"
                    if name == "boussinesq"
                    else "blocked_convergence"
                ),
                "passed": passed,
            }
        )
        errors = [float(row["field_relative_l2"]) for row in spatial]
        floor = float(thresholds["temporal_roundoff_floor"])
        spatial_below_floor = all(value <= floor for value in errors)
        order = _observed_order(errors[1], errors[2])
        gates.append(
            {
                "gate": f"spatial_refinement_{name}",
                "category": "blocked_convergence",
                "passed": spatial_below_floor
                or (
                    errors[0] > errors[1] > errors[2]
                    and order is not None
                    and order >= float(thresholds["spatial_order_minimum"][name])
                ),
                "errors": errors,
                "all_below_roundoff_floor": spatial_below_floor,
                "observed_order": order,
            }
        )
        temporal_result = _temporal_refinement_gate(
            [row["_trajectory_eta"] for row in temporal],
            minimum_order=float(thresholds["temporal_order_minimum"][name]),
            floor=floor,
        )
        gates.append(
            {
                "gate": f"temporal_refinement_{name}",
                "category": "blocked_convergence",
                **temporal_result,
            }
        )

    interpolation_failures = [
        row
        for row in rows
        if float(row["interpolation_actual_max_abs_error"])
        > float(row["interpolation_absolute_bound"])
        + float(row["interpolation_bound_floating_tolerance"])
    ]
    gates.append(
        {
            "gate": "analytical_interpolation_bound",
            "category": "blocked_convergence",
            "passed": not interpolation_failures,
            "failure_count": len(interpolation_failures),
        }
    )
    boussinesq_modal = [
        row
        for row in rows
        if row["solver"] == "boussinesq" and row["analytical_role"] == "modal"
    ]
    group_result = _group_speed_gate(
        boussinesq_modal,
        relative_error_limit=float(
            thresholds["analytical"]["boussinesq"]["group_speed_relative_error"]
        ),
    )
    gates.append(
        {
            "gate": "analytical_boussinesq_group_speed",
            "category": "blocked_boussinesq_health",
            **group_result,
        }
    )

    operator_payloads = [
        payload for payload in task_results if payload["task"]["kind"] == "operator"
    ]
    operator_threshold = float(
        thresholds["operator"]["elapsed_candidate_cfl_relative_l2"]
    )
    operator_nx = 64
    operator_sponge = _x_sponge_region(operator_nx, max(1, operator_nx // 8))
    operator_interior = ~operator_sponge
    for name in SOLVERS:
        selected = [
            payload
            for payload in operator_payloads
            if payload["task"]["spec"]["solver"] == name
        ]
        by_variant: dict[str, dict[float, np.ndarray]] = {}
        solver_rows: list[dict[str, Any]] = []
        for payload in selected:
            row = dict(payload["row"])
            solver_rows.append(row)
            by_variant.setdefault(str(row["variant"]), {})[float(row["cfl"])] = (
                np.asarray(payload["trajectory"])
            )
        rows.extend(solver_rows)
        production, half = [
            float(value) for value in config["production"]["cfl"][name][:2]
        ]
        legacy_metrics = _operator_discrepancy_metrics(
            by_variant["legacy_per_step"][production],
            by_variant["legacy_per_step"][half],
            sponge_region=operator_sponge,
            interior_region=operator_interior,
        )
        elapsed_metrics = _operator_discrepancy_metrics(
            by_variant["elapsed_no_filter"][production],
            by_variant["elapsed_no_filter"][half],
            sponge_region=operator_sponge,
            interior_region=operator_interior,
        )
        legacy_diff = legacy_metrics["trajectory_relative_l2"]
        elapsed_diff = elapsed_metrics["trajectory_relative_l2"]
        whole_domain_sponge = any(
            bool(row.get("whole_domain_sponge", True)) for row in solver_rows
        )
        rows.append(
            {
                "component": "operator_sensitivity_summary",
                "solver": name,
                "legacy_cfl_relative_l2": legacy_diff,
                "elapsed_no_filter_cfl_relative_l2": elapsed_diff,
                "legacy_metrics": legacy_metrics,
                "elapsed_no_filter_metrics": elapsed_metrics,
                "sponge_axes": "x",
                "whole_domain_sponge": whole_domain_sponge,
            }
        )
        gates.append(
            {
                "gate": f"elapsed_operator_consistency_{name}",
                "category": "blocked_operator_semantics",
                "passed": not whole_domain_sponge
                and elapsed_diff <= operator_threshold
                and elapsed_diff <= legacy_diff,
                "legacy_relative_l2": legacy_diff,
                "elapsed_relative_l2": elapsed_diff,
                "elapsed_final_time_relative_l2": elapsed_metrics[
                    "final_time_relative_l2"
                ],
                "elapsed_sponge_relative_l2": elapsed_metrics[
                    "sponge_trajectory_relative_l2"
                ],
                "elapsed_interior_relative_l2": elapsed_metrics[
                    "interior_trajectory_relative_l2"
                ],
                "whole_domain_sponge": whole_domain_sponge,
            }
        )
        if name == "boussinesq":
            primary = next(
                row
                for row in solver_rows
                if row["variant"] == "elapsed_no_filter"
                and math.isclose(float(row["cfl"]), production)
            )
            fallback = next(
                row
                for row in solver_rows
                if row["variant"] == "elapsed_filter"
                and math.isclose(float(row["cfl"]), production)
            )
            initial_fraction = _high_frequency_fraction(_packet(64, 4)[None, ...])
            high_frequency_limit = max(
                2.0 * initial_fraction,
                float(
                    thresholds["boussinesq_no_filter"][
                        "high_frequency_fraction_absolute"
                    ]
                ),
            )
            no_filter_ok = (
                primary["cg_failure_count"] == 0
                and primary["amplitude_ratio"]
                <= float(thresholds["boussinesq_no_filter"]["amplitude_growth"])
                and primary["high_frequency_fraction"] <= high_frequency_limit
                and elapsed_diff <= operator_threshold
            )
            filter_diff = _relative_l2(
                by_variant["elapsed_filter"][production],
                by_variant["elapsed_filter"][half],
            )
            filter_ok = (
                fallback["cg_failure_count"] == 0
                and fallback["amplitude_ratio"]
                <= float(thresholds["boussinesq_no_filter"]["amplitude_growth"])
                and fallback["high_frequency_fraction"] <= high_frequency_limit
                and filter_diff <= operator_threshold
            )
            gates.extend(
                [
                    {
                        "gate": "boussinesq_no_filter_health",
                        "category": "informational",
                        "passed": no_filter_ok,
                        "high_frequency_limit": high_frequency_limit,
                    },
                    {
                        "gate": "boussinesq_elapsed_filter_fallback",
                        "category": "informational",
                        "passed": filter_ok,
                        "cfl_relative_l2": filter_diff,
                        "high_frequency_limit": high_frequency_limit,
                    },
                    {
                        "gate": "boussinesq_filter_acceptance",
                        "category": "blocked_boussinesq_health",
                        "passed": no_filter_ok or filter_ok,
                        "recommended_filter": (
                            "disabled"
                            if no_filter_ok
                            else "elapsed_time_consistent"
                            if filter_ok
                            else "none"
                        ),
                    },
                ]
            )

    reference_dt = float(config["operators"]["sponge"]["reference_dt"])
    reference_factor = float(config["operators"]["sponge"]["reference_min_factor"])
    product = 1.0
    elapsed = 0.0
    for dt in (0.0011, 0.0007, 0.0017):
        product *= reference_factor ** (dt / reference_dt)
        elapsed += dt
    expected_factor = reference_factor ** (elapsed / reference_dt)
    factor_error = abs(product - expected_factor) / expected_factor
    rows.append(
        {
            "component": "operator_factor_identity",
            "observed": product,
            "expected": expected_factor,
            "relative_error": factor_error,
        }
    )
    gates.append(
        {
            "gate": "elapsed_sponge_factor_identity",
            "category": "blocked_operator_semantics",
            "passed": factor_error
            <= float(thresholds["operator"]["accumulated_factor_relative_error"]),
        }
    )

    boundary_payloads = [
        payload for payload in task_results if payload["task"]["kind"] == "boundary"
    ]
    boundary_config = config["boundary_packet"]
    boundary_nx = int(boundary_config["grid"])
    boundary_ny = int(boundary_config["transverse_cells"])
    for name in SOLVERS:
        packet_spec = boundary_config["solvers"][name]
        _validate_boundary_packet_spec(
            packet_spec,
            nx=boundary_nx,
            sponge_width=max(1, boundary_nx // 8),
        )
        initial_packet = np.asarray(
            _boundary_initial_conditions(
                name, nx=boundary_nx, ny=boundary_ny, spec=packet_spec
            )["eta0"]
        )
        selected = [
            payload
            for payload in boundary_payloads
            if payload["task"]["spec"]["solver"] == name
        ]
        trajectories = {
            str(payload["task"]["spec"]["variant"]): np.asarray(payload["trajectory"])
            for payload in selected
        }
        if set(trajectories) != {"baseline", "damped"}:
            raise RuntimeError(f"Incomplete boundary tasks for {name}")
        baseline = trajectories["baseline"]
        damped = trajectories["damped"]
        metrics = _boundary_metrics(
            baseline=baseline,
            damped=damped,
            initial_packet=initial_packet,
            spec=packet_spec,
        )
        whole_domain_sponge = any(
            bool(payload["row"].get("whole_domain_sponge", True))
            for payload in selected
            if payload["task"]["spec"]["use_sponge"]
        )
        boundary_row = {
            "component": "boundary_sponge",
            "solver": name,
            "boundary_implementation": "zero_gradient_edge_padding",
            **metrics,
            "incident_window": _json_safe(packet_spec["incident_window"]),
            "reflected_window": _json_safe(packet_spec["reflected_window"]),
            "interior_window": _json_safe(packet_spec["interior_window"]),
            "sponge_axes": "x",
            "whole_domain_sponge": whole_domain_sponge,
            "finite": all(bool(payload["row"]["finite"]) for payload in selected),
            "cg_failure_count": sum(
                int(payload["row"]["cg_failure_count"]) for payload in selected
            ),
            "operator": {
                **(
                    {
                        "nan_to_num_replacement_count": sum(
                            int(
                                payload["row"]["operator"][
                                    "nan_to_num_replacement_count"
                                ]
                            )
                            for payload in selected
                        )
                    }
                    if all(
                        "nan_to_num_replacement_count" in payload["row"]["operator"]
                        for payload in selected
                    )
                    else {}
                ),
                "task_diagnostics": [
                    {
                        "variant": str(payload["row"]["variant"]),
                        "diagnostics": payload["row"]["operator"],
                    }
                    for payload in selected
                ],
            },
        }
        rows.append(boundary_row)
        boundary_thresholds = thresholds["boundary"]
        gates.append(
            {
                "gate": f"boundary_sponge_{name}",
                "category": "blocked_boundary_behavior",
                "passed": boundary_row["finite"]
                and boundary_row["cg_failure_count"] == 0
                and not whole_domain_sponge
                and bool(metrics["arrival_observed"])
                and metrics["reflected_amplitude_ratio"]
                <= float(boundary_thresholds["reflected_amplitude_ratio"])
                and metrics["reflected_energy_ratio"]
                <= float(boundary_thresholds["reflected_energy_ratio"])
                and metrics["interior_relative_l2"]
                <= float(boundary_thresholds["interior_relative_l2"]),
                "arrival_observed": bool(metrics["arrival_observed"]),
                "arrival_energy_ratio": float(metrics["arrival_energy_ratio"]),
                "whole_domain_sponge": whole_domain_sponge,
            }
        )

    conservation_payloads = [
        payload for payload in task_results if payload["task"]["kind"] == "conservation"
    ]
    conservation_thresholds = thresholds["conservation"]
    for payload in conservation_payloads:
        row = dict(payload["row"])
        rows.append(row)
        allowed_drift = max(
            float(conservation_thresholds["normalized_drift"]),
            float(row["roundoff_floor_normalized"]),
        )
        gates.append(
            {
                "gate": f"conservation_{row['boundary']}_{row['solver']}",
                "category": "blocked_convergence",
                "passed": row["finite"]
                and row["cg_failure_count"] == 0
                and row["measurement_dtype"] == "float64"
                and row["measurement_grid"] == "internal_natural_states"
                and row["normalized_drift"] <= allowed_drift,
                "normalized_drift": row["normalized_drift"],
                "allowed_normalized_drift": allowed_drift,
                "roundoff_floor_normalized": row["roundoff_floor_normalized"],
            }
        )

    canary_rows = [
        dict(payload["row"])
        for payload in task_results
        if payload["task"]["kind"] == "canary"
    ]
    rows.extend(canary_rows)
    canary_thresholds = thresholds["canary"]
    for row in canary_rows:
        gates.append(
            {
                "gate": f"canary_{row['qualified_id']}_{row['solver']}",
                "category": (
                    "blocked_boussinesq_health"
                    if row["solver"] == "boussinesq"
                    else "blocked_convergence"
                ),
                "passed": row["finite"]
                and row["cg_failure_count"] == 0
                and row["amplitude_growth"]
                <= float(canary_thresholds["amplitude_growth"])
                and row["max_eta_over_depth"]
                <= float(canary_thresholds["eta_over_depth"]),
            }
        )
    rows.extend(
        _bootstrap_canary_aggregates(
            canary_rows,
            seed=int(config["seeds"]["bootstrap"]),
            resamples=int(config["seeds"]["bootstrap_resamples"]),
        )
    )
    gates.append(_universal_health_gate(rows, universal=thresholds["universal"]))

    decision_name = _decision_from_gates(gates)
    filter_gate = next(
        row for row in gates if row["gate"] == "boussinesq_filter_acceptance"
    )
    recommended_filter = (
        str(filter_gate["recommended_filter"])
        if bool(filter_gate["passed"])
        else "undetermined"
    )
    elapsed_operator_gates = [
        row
        for row in gates
        if str(row["gate"]).startswith("elapsed_operator_consistency_")
        or row["gate"] == "elapsed_sponge_factor_identity"
    ]
    if len(elapsed_operator_gates) != len(SOLVERS) + 1:
        raise RuntimeError("Incomplete elapsed-operator gate set")
    decision = {
        "schema_id": LEVEL_A_SCHEMA_ID,
        "contract_hash": contract["contract_hash"],
        "decision": decision_name,
        "level_a_passed": decision_name == "pass_to_H1",
        "three_reference_contract_accepted": False,
        "h1_executed": False,
        "failed_gates": [row for row in gates if not row["passed"]],
        "elapsed_s": float(elapsed_s),
        "recommended_sponge": (
            "elapsed_time_consistent"
            if all(bool(row["passed"]) for row in elapsed_operator_gates)
            else "undetermined"
        ),
        "recommended_boussinesq_filter": recommended_filter,
        "boussinesq_depth_scale": 1.0,
        "boundary_interpretation": "zero_gradient_edge_padding_not_radiative",
        "operational_provenance": _json_safe(operational_provenance),
    }
    public_rows = [_public_row(row) for row in rows]
    decision["scientific_digest"] = _scientific_digest(
        {"rows": public_rows, "gates": gates, "decision": decision}
    )
    return public_rows, gates, decision


def _write_failure_execution(
    execution: Path,
    *,
    contract: Mapping[str, Any],
    issues: Sequence[str],
    elapsed: float,
    gate: str,
    report_reason: str,
) -> None:
    gates = [
        {
            "gate": gate,
            "category": "implementation_failure",
            "passed": False,
            "details": list(issues),
        }
    ]
    decision = {
        "schema_id": LEVEL_A_SCHEMA_ID,
        "artifact_kind": "common-time-v2-level-a-decision",
        "contract_hash": contract["contract_hash"],
        "decision": "implementation_failure",
        "level_a_passed": False,
        "three_reference_contract_accepted": False,
        "h1_executed": False,
        "failed_gates": gates,
        "elapsed_s": elapsed,
        "recommended_sponge": "undetermined",
        "recommended_boussinesq_filter": "undetermined",
    }
    _write_json(execution / "decision.json", decision)
    _write_json(execution / "summary.json", {"decision": decision, "rows": 0})
    _write_jsonl(execution / "detailed_rows.jsonl", [])
    _write_csv(execution / "detailed_rows.csv", [])
    (execution / "REPORT.md").write_text(
        "# Common-time-v2 Level A report\n\n"
        "Decision: `implementation_failure`.\n\n"
        f"{report_reason} "
        "No Level A numerical outcomes were inspected and no thresholds or settings were changed.\n\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n",
        encoding="utf-8",
    )


def _report_text(decision_name: str) -> str:
    return f"# Common-time-v2 Level A report\n\nDecision: `{decision_name}`.\n"


def _validate_failure_execution(
    execution: Path, *, expected_contract_hash: str
) -> None:
    expected_files = {
        "decision.json",
        "detailed_rows.csv",
        "detailed_rows.jsonl",
        "REPORT.md",
        "SHA256SUMS.txt",
        "summary.json",
    }
    actual_files = {path.name for path in execution.iterdir() if path.is_file()}
    if actual_files != expected_files or any(
        path.is_dir() for path in execution.iterdir()
    ):
        raise RuntimeError("Finalized Level A failure has an unexpected file set")
    validate_checksums(execution)
    decision = _read_json(execution / "decision.json")
    summary = _read_json(execution / "summary.json")
    if decision.get("decision") != "implementation_failure":
        raise RuntimeError("Finalized Level A failure has the wrong decision")
    if (
        decision.get("level_a_passed") is not False
        or decision.get("h1_executed") is not False
    ):
        raise RuntimeError(
            "Finalized Level A failure has contradictory progression flags"
        )
    failed_gates = decision.get("failed_gates")
    if not isinstance(failed_gates, list) or not failed_gates:
        raise RuntimeError("Finalized Level A failure has no failed gate evidence")
    if any(
        gate.get("category") != "implementation_failure"
        or gate.get("passed") is not False
        for gate in failed_gates
        if isinstance(gate, Mapping)
    ) or any(not isinstance(gate, Mapping) for gate in failed_gates):
        raise RuntimeError("Finalized Level A failure has contradictory gate evidence")
    if decision.get("contract_hash") != expected_contract_hash:
        raise RuntimeError("Finalized Level A failure contract mismatch")
    if summary.get("decision") != decision or int(summary.get("rows", -1)) != 0:
        raise RuntimeError("Finalized Level A failure summary mismatch")
    if _read_jsonl(execution / "detailed_rows.jsonl"):
        raise RuntimeError("Finalized Level A failure unexpectedly contains rows")
    if (execution / "detailed_rows.csv").read_text(encoding="utf-8") != "":
        raise RuntimeError("Finalized Level A failure unexpectedly contains CSV rows")
    report = (execution / "REPORT.md").read_text(encoding="utf-8")
    if "Decision: `implementation_failure`." not in report:
        raise RuntimeError("Finalized Level A failure report contradicts decision")
    for gate in failed_gates:
        for detail in gate.get("details", []):
            if str(detail) not in report:
                raise RuntimeError(
                    "Finalized Level A failure report omits gate evidence"
                )


def _validate_completed_execution(
    execution: Path,
    tasks: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    expected_files = {
        "detailed_rows.jsonl",
        "detailed_rows.csv",
        "aggregate_summary.json",
        "decision.json",
        "summary.json",
        "REPORT.md",
        "operational_provenance.json",
        "SHA256SUMS.txt",
    }
    actual_files = {path.name for path in execution.iterdir() if path.is_file()}
    actual_dirs = {path.name for path in execution.iterdir() if path.is_dir()}
    if actual_files != expected_files or actual_dirs != {"tasks"}:
        raise RuntimeError("Completed Level A execution has an unexpected file set")
    validate_checksums(execution)
    loaded, missing = _scan_task_artifacts(tasks, execution / "tasks")
    if missing or len(loaded) != len(tasks):
        raise RuntimeError("Completed Level A execution has an incomplete task set")
    operational = _read_json(execution / "operational_provenance.json")
    ordered = [loaded[str(task["task_id"])] for task in tasks]
    derived_rows, derived_gates, derived_decision = _aggregate_level_a_tasks(
        ordered,
        config=config,
        contract=contract,
        operational_provenance=operational,
        elapsed_s=0.0,
    )
    decision = _read_json(execution / "decision.json")
    summary = _read_json(execution / "summary.json")
    aggregate = _read_json(execution / "aggregate_summary.json")
    public_rows = _read_jsonl(execution / "detailed_rows.jsonl")
    if summary.get("decision") != decision:
        raise RuntimeError("Completed Level A decision and summary disagree")
    if decision.get("operational_provenance") != operational:
        raise RuntimeError("Completed Level A operational provenance mismatch")
    if decision.get("contract_hash") != contract["contract_hash"]:
        raise RuntimeError("Completed Level A execution contract mismatch")
    if int(summary.get("rows", -1)) != len(public_rows):
        raise RuntimeError("Completed Level A row count mismatch")
    if not _recomputed_rows_equal(public_rows, derived_rows):
        raise RuntimeError("Completed Level A rows contradict task evidence")
    if not _derived_replay_equal(aggregate.get("gates"), derived_gates):
        raise RuntimeError("Completed Level A gates contradict task evidence")
    if not _derived_replay_equal(decision, derived_decision):
        raise RuntimeError("Completed Level A decision contradicts task evidence")
    digest_decision = dict(decision)
    recorded_digest = digest_decision.pop("scientific_digest", None)
    expected_digest = _scientific_digest(
        {
            "rows": public_rows,
            "gates": aggregate.get("gates"),
            "decision": digest_decision,
        }
    )
    if recorded_digest != expected_digest:
        raise RuntimeError("Completed Level A scientific digest mismatch")
    if (execution / "detailed_rows.csv").read_text(encoding="utf-8") != _csv_text(
        public_rows
    ):
        raise RuntimeError("Completed Level A CSV contradicts JSON rows")
    if (execution / "REPORT.md").read_text(encoding="utf-8") != _report_text(
        str(decision["decision"])
    ):
        raise RuntimeError("Completed Level A report contradicts decision")


def _recover_execution_staging(staging: Path) -> None:
    recoverable_files = {
        "detailed_rows.jsonl",
        "detailed_rows.csv",
        "aggregate_summary.json",
        "decision.json",
        "summary.json",
        "operational_provenance.json",
        "REPORT.md",
        "SHA256SUMS.txt",
    }
    for path in list(staging.iterdir()):
        if path.name == "tasks":
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("Unsafe Level A tasks staging path")
            continue
        if path.name not in recoverable_files:
            raise RuntimeError(f"Unexpected Level A staging contents: {path.name}")
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unsafe Level A staging output: {path.name}")
        path.unlink()


def execute_level_a(
    *,
    repo_root: Path,
    contract_root: Path,
    workers: int = 1,
    max_in_flight: int | None = None,
    resume: bool = False,
) -> Path:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_in_flight is not None and max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    started = time.monotonic()
    repo_root = repo_root.resolve()
    contract_root = contract_root.resolve()
    execution = contract_root / "execution"
    allowed_unlisted = [".execution-staging/"]
    if resume and execution.exists():
        allowed_unlisted.append("execution/")
    validate_checksums(contract_root, allow_unlisted_prefixes=tuple(allowed_unlisted))
    contract = _read_json(contract_root / "preregistered_contract.json")
    if contract.get("schema_id") != LEVEL_A_SCHEMA_ID:
        raise RuntimeError("Level A contract schema mismatch")
    contract_identity = dict(contract)
    recorded_contract_hash = contract_identity.pop("contract_hash", None)
    expected_contract_hash = stable_hash_payload(
        artifact_kind="common-time-v2-level-a-contract",
        payload=contract_identity,
        schema_id=LEVEL_A_SCHEMA_ID,
    )
    if (
        recorded_contract_hash != expected_contract_hash
        or contract_root.name != expected_contract_hash
    ):
        raise RuntimeError("Level A content-addressed contract hash mismatch")
    worker_policy = contract.get("worker_policy", {})
    if workers != int(worker_policy.get("requested_workers", -1)):
        raise RuntimeError("Level A worker count differs from the frozen policy")
    if max_in_flight != worker_policy.get("requested_max_in_flight"):
        raise RuntimeError("Level A in-flight limit differs from the frozen policy")
    if worker_policy.get("process_start_method") != "spawn":
        raise RuntimeError("Level A process start method is not frozen as spawn")
    if _execution_environment_snapshot() != contract.get("execution_environment"):
        raise RuntimeError("Level A execution environment differs from preregistration")
    current_code = code_state(repo_root)
    config = contract["source_config"]
    tasks = _build_level_a_task_plan(
        config,
        contract["canaries"],
        contract_hash=str(contract["contract_hash"]),
        code_state_hash=str(contract["code_state"]["code_state_hash"]),
    )
    task_blueprint = [
        {
            "ordinal": task["ordinal"],
            "task_id": task["task_id"],
            "kind": task["kind"],
            "spec": task["spec"],
        }
        for task in tasks
    ]
    if task_blueprint != contract.get("task_blueprint"):
        raise RuntimeError("Level A task blueprint differs from preregistration")
    if tasks != _read_json(contract_root / "task_plan.json"):
        raise RuntimeError("Level A realized task plan differs from preregistration")
    execution = contract_root / "execution"
    if execution.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite Level A execution: {execution}"
            )
        existing_decision = _read_json(execution / "decision.json")
        if existing_decision.get("decision") == "implementation_failure":
            _validate_failure_execution(
                execution, expected_contract_hash=str(contract["contract_hash"])
            )
        else:
            _validate_completed_execution(
                execution,
                tasks,
                config=config,
                contract=contract,
            )
        return execution

    staging = contract_root / ".execution-staging"
    if staging.exists() and not resume:
        raise FileExistsError(staging)
    if not staging.exists():
        staging.mkdir(parents=False, exist_ok=False)
    else:
        _recover_execution_staging(staging)
    tasks_root = staging / "tasks"
    tasks_root.mkdir(parents=False, exist_ok=True)

    if current_code["code_state_hash"] != contract["code_state"]["code_state_hash"]:
        if any(tasks_root.iterdir()):
            raise RuntimeError("Cannot finalize code-state failure over task artifacts")
        _write_failure_execution(
            staging,
            contract=contract,
            issues=["code_state_hash changed after preregistration"],
            elapsed=time.monotonic() - started,
            gate="code_state_preflight",
            report_reason="The code state changed after preregistration.",
        )
        shutil.rmtree(tasks_root)
    else:
        issues = _preflight_canaries(contract["canaries"])
        if issues:
            if any(tasks_root.iterdir()):
                raise RuntimeError(
                    "Cannot finalize canary preflight failure over task artifacts"
                )
            _write_failure_execution(
                staging,
                contract=contract,
                issues=issues,
                elapsed=time.monotonic() - started,
                gate="authoritative_canary_preflight",
                report_reason=(
                    "The frozen authoritative training canaries were unavailable "
                    "or invalid at their H0-recorded paths."
                ),
            )
            shutil.rmtree(tasks_root)
        else:
            task_results, operational = _execute_level_a_task_plan(
                tasks,
                tasks_root=tasks_root,
                workers=workers,
                max_in_flight=max_in_flight,
                resume=resume,
            )
            public_rows, gates, decision = _aggregate_level_a_tasks(
                task_results,
                config=config,
                contract=contract,
                operational_provenance=operational,
                elapsed_s=time.monotonic() - started,
            )
            _write_jsonl(staging / "detailed_rows.jsonl", public_rows)
            _write_csv(staging / "detailed_rows.csv", public_rows)
            _write_json(staging / "aggregate_summary.json", {"gates": gates})
            _write_json(staging / "decision.json", decision)
            _write_json(
                staging / "summary.json",
                {"decision": decision, "rows": len(public_rows)},
            )
            _write_json(staging / "operational_provenance.json", operational)
            (staging / "REPORT.md").write_text(
                _report_text(str(decision["decision"])),
                encoding="utf-8",
            )

    _write_checksums(staging)
    os.replace(staging, execution)
    _write_checksums(contract_root)
    return execution
