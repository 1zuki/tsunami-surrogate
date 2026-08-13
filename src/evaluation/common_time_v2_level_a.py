from __future__ import annotations

import csv
from dataclasses import replace
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
from typing import Any, Callable, Mapping, Sequence

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
from src.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    _prepare_buffered_domain,
    _simulate_one_local,
)
from src.solver.boussinesq import BoussinesqSolver
from src.evaluation.boussinesq_boundary import (
    SpectralPacketSpec,
    build_reference_packet,
    cosine_taper,
    directional_states as boussinesq_directional_states,
    discrete_energy as boussinesq_discrete_energy,
    energy_density as boussinesq_energy_density,
    evolve_reference as evolve_boussinesq_reference,
    packet_timing as boussinesq_packet_timing,
)
from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver
from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver
from src.solver.operator_time import build_sponge_mask


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
    "boussinesq_h0_boundary_exposure",
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
        values: dict[str, Any] = {}
        for key in fields:
            value = _json_safe(row.get(key))
            values[key] = (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (Mapping, list))
                else value
            )
        writer.writerow(values)
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


def _resolved_boundary_packet_spec(
    boundary_config: Mapping[str, Any], solver_name: str
) -> dict[str, Any]:
    shared_keys = (
        "support_sigmas",
        "post_exit_observation_sigmas",
        "prearrival_sample_count",
        "post_exit_sample_count",
        "outgoing_tail_safety_factor",
    )
    resolved = {
        key: boundary_config[key]
        for key in shared_keys
        if key in boundary_config
    }
    resolved.update(dict(boundary_config["solvers"][solver_name]))
    resolved["solver"] = solver_name
    return _json_safe(resolved)


def _boussinesq_spectral_packet_spec(
    packet_spec: Mapping[str, Any], *, role: str
) -> SpectralPacketSpec:
    if packet_spec.get("protocol") != "spectral_large_domain_v1":
        raise ValueError("unsupported Boussinesq boundary protocol")
    if role not in ("reflection", "production_horizon"):
        raise ValueError("unsupported Boussinesq boundary role")
    values = packet_spec[role]
    return SpectralPacketSpec(
        length=float(values["length"]),
        dx=float(values["dx"]),
        ny=int(values["ny"]),
        dy=float(values["dy"]),
        center=float(values["center"]),
        carrier_wavenumber=float(values["carrier_wavenumber"]),
        spectral_width=float(values["spectral_width"]),
        amplitude=float(values["amplitude"]),
        direction=str(packet_spec["direction"]),
        depth=float(packet_spec["depth"]),
        gravity=float(packet_spec["gravity"]),
        alpha=float(packet_spec["alpha"]),
        spectral_energy_tail=float(values["spectral_energy_tail"]),
        spatial_energy_tail=float(values["spatial_energy_tail"]),
        reference_length=float(values["reference_length"]),
    )


def _boussinesq_spectral_packet_bundle(
    packet_spec: Mapping[str, Any], *, role: str
) -> tuple[SpectralPacketSpec, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    spec = _boussinesq_spectral_packet_spec(packet_spec, role=role)
    finite, reference, metadata = build_reference_packet(spec)
    timing = boussinesq_packet_timing(spec, metadata)
    if role == "production_horizon":
        support_width = float(metadata["spatial_support_right"]) - float(
            metadata["spatial_support_left"]
        )
        timing["reference_safe"] = bool(
            float(metadata["reference_distance"])
            > float(metadata["group_velocity_max"])
            * float(candidate_requested_times()[-1])
            + support_width
        )
    if not bool(timing["reference_safe"]):
        raise ValueError("Boussinesq large-domain reference is not return-safe")
    return spec, finite, reference, metadata, timing


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
        spec = _resolved_boundary_packet_spec(boundary, solver)
        if solver == "boussinesq":
            for role in ("reflection", "production_horizon"):
                spectral_spec, _finite, _reference, _metadata, timing = (
                    _boussinesq_spectral_packet_bundle(spec, role=role)
                )
                if role == "reflection" and float(
                    spec[role]["cg_absolute_residual_tolerance"]
                ) <= 0.0:
                    raise ValueError(
                        "Boussinesq reflection requires a positive frozen CG floor"
                    )
                if spectral_spec.nx <= 1 or not bool(timing["reference_safe"]):
                    raise ValueError("invalid Boussinesq spectral boundary protocol")
            continue
        _validate_boundary_packet_spec(
            spec,
            nx=int(boundary.get("grid", 128)),
            sponge_width=max(
                1,
                int(
                    round(
                        int(boundary.get("grid", 128))
                        * float(
                            config["operators"]["hydrostatic_gate"][
                                "candidate_sponge"
                            ]["width_fraction"]
                        )
                    )
                ),
            ),
        )
        _boundary_timing(solver, spec)
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
    production = config.get("production", {})
    expected_domain = {
        "buffer_cells": 16,
        "source_taper_cells": 8,
        "bathymetry_extension": "edge",
        "output_crop": "central",
    }
    expected_boundary = {
        "swe_hydrostatic": "radiation",
        "swe_muscl_hr": "radiation",
        "boussinesq": "open_zero_gradient_edge_padding",
    }
    expected_sponge = {
        "enabled": True,
        "axes": "xy",
        "width": 16,
        "min_factor": 0.8,
        "profile": "cosine",
        "time_mode": "elapsed_time_consistent",
        "reference_dt": 0.0035,
    }
    if (
        int(production.get("computational_grid", -1)) != 96
        or int(production.get("publication_grid", -1)) != 64
        or production.get("computational_domain") != expected_domain
        or production.get("boundary_implementation") != expected_boundary
        or production.get("sponge") != expected_sponge
    ):
        raise ValueError(
            "Level A production policy must freeze the reviewed 96-to-64 "
            "buffered computational contract"
        )
    if boundary.get("gate_candidate") != {
        "swe_hydrostatic": "radiation_cosine_sponge",
        "swe_muscl_hr": "radiation_cosine_sponge",
        "boussinesq": "zero_gradient_cosine_sponge",
    }:
        raise ValueError("Level A boundary gates do not match the production policy")
    hydro_gate = config.get("operators", {}).get("hydrostatic_gate", {})
    hydro_sponge = hydro_gate.get("candidate_sponge", {})
    if (
        hydro_gate.get("candidate_boundary") != "radiation"
        or hydro_sponge.get("axes") != "x"
        or hydro_sponge.get("profile") != "cosine"
        or not math.isclose(
            float(hydro_sponge.get("width_fraction", -1.0)),
            1.0 / 6.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or float(hydro_sponge.get("reference_min_factor", -1.0)) != 0.8
    ):
        raise ValueError(
            "Level A Hydrostatic production-sensitivity task does not match "
            "the buffered candidate"
        )
    execution = config.get("execution", {})
    expected_execution = {
        "requested_workers": 8,
        "requested_max_in_flight": 8,
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
    h0_root: Path | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Level A YAML must contain a mapping")
    _validate_source_contract(config)

    h0_dir = (
        h0_root.resolve()
        if h0_root is not None
        else repo_root
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
        "h0_root": str(h0_dir),
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
- Production computation/publication grids: {config["production"]["computational_grid"]} x {config["production"]["computational_grid"]} -> central {config["production"]["publication_grid"]} x {config["production"]["publication_grid"]} crop
- Production outer treatment: SWE radiation, Boussinesq open zero-gradient, external {config["production"]["sponge"]["width"]}-cell elapsed-time cosine sponge
- Thresholds frozen before execution: yes

Stage C thresholds are historical context only and were not inherited. `depth_scale=1.0` is the sole v2 production interpretation in this contract. Boussinesq `open` is explicitly labelled zero-gradient edge padding, not radiative. The buffered production treatment is a frozen candidate under test, not a pre-accepted boundary claim. A Level A pass permits only progression to H1; it does not accept the production contract.
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
    boundary: Any,
    use_sponge: bool,
    sponge_mode: str = "legacy_per_step",
    filter_mode: str = "disabled",
    filter_strength: float = 0.0,
    sponge_axes: str = "xy",
    sponge_width: int | None = None,
    sponge_min_factor: float = 0.9,
    sponge_profile: str = "quadratic",
    reconstruction_limiter: str = "minmod",
    dx: float | None = None,
    dy: float | None = None,
    cg_failure_mode: str = "strict_v2",
    linear_solver_abs_tol: float = 0.0,
) -> Any:
    common = dict(
        nx=nx,
        ny=ny,
        dx=1.0 / nx if dx is None else float(dx),
        dy=1.0 / ny if dy is None else float(dy),
        dt=1.0e-4,
        g=9.81,
        cfl=cfl,
        boundary=boundary,
        use_sponge=use_sponge,
        sponge_width=max(1, nx // 8) if sponge_width is None else sponge_width,
        sponge_min_factor=sponge_min_factor,
        sponge_time_mode=sponge_mode,
        sponge_reference_dt=0.0035
        if sponge_mode == "elapsed_time_consistent"
        else None,
        sponge_axes=sponge_axes,
        sponge_profile=sponge_profile,
    )
    if name == "swe_hydrostatic":
        return HydrostaticShallowWaterSolver(
            **common, dry_tolerance=1e-6, max_velocity=30.0
        )
    if name == "swe_muscl_hr":
        return MUSCLHRShallowWaterSolver(
            **common,
            dry_tolerance=1e-6,
            max_velocity=30.0,
            reconstruction_limiter=reconstruction_limiter,
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
        cg_failure_mode=cg_failure_mode,
        linear_solver_abs_tol=linear_solver_abs_tol,
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
    name: str,
    *,
    nx: int,
    ny: int,
    mode: int,
    cfl: float,
    amplitude: float,
    reconstruction_limiter: str = "minmod",
) -> dict[str, Any]:
    times = candidate_requested_times()
    exact, omega, group = _mode_exact(
        name, nx=nx, ny=ny, mode=mode, amplitude=amplitude, times=times
    )
    solver = _solver(
        name,
        nx=nx,
        ny=ny,
        cfl=cfl,
        boundary="periodic",
        use_sponge=False,
        reconstruction_limiter=reconstruction_limiter,
    )
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
        requested_state_dtype=np.float64,
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
        "reconstruction_limiter": (
            reconstruction_limiter if name == "swe_muscl_hr" else "not_applicable"
        ),
        "measurement_dtype": str(np.asarray(eta).dtype),
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
                    "reconstruction_limiter": "minmod",
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
                    "reconstruction_limiter": "minmod",
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
                    "reconstruction_limiter": "minmod",
                },
            )

    for solver in SOLVERS:
        production, half, quarter = [
            float(value) for value in config["production"]["cfl"][solver]
        ]
        if solver == "swe_hydrostatic":
            gate = config["operators"]["hydrostatic_gate"]
            clean_grid = int(gate["clean_grid"])
            reference_cfl = production / float(gate["reference_cfl_divisor"])
            for cfl_index, cfl in enumerate(
                (production, half, quarter, reference_cfl)
            ):
                add(
                    f"operator/{solver}/clean_temporal/{cfl_index}",
                    "operator",
                    {
                        "solver": solver,
                        "operator_role": "clean_temporal",
                        "variant": "no_sponge_periodic",
                        "filter_mode": "disabled",
                        "filter_strength": 0.0,
                        "cfl": cfl,
                        "nx": clean_grid,
                        "ny": 4,
                        "boundary": str(gate["clean_boundary"]),
                        "use_sponge": False,
                        "sponge_axes": "x",
                        "sponge_profile": "quadratic",
                    },
                )
            for grid in gate["spatial_grids"]:
                if int(grid) == clean_grid:
                    continue
                add(
                    f"operator/{solver}/spatial/{int(grid)}",
                    "operator",
                    {
                        "solver": solver,
                        "operator_role": "spatial_reference",
                        "variant": "no_sponge_periodic",
                        "filter_mode": "disabled",
                        "filter_strength": 0.0,
                        "cfl": reference_cfl,
                        "nx": int(grid),
                        "ny": 4,
                        "boundary": str(gate["clean_boundary"]),
                        "use_sponge": False,
                        "sponge_axes": "x",
                        "sponge_profile": "quadratic",
                    },
                )
            candidate = gate["candidate_sponge"]
            for cfl_index, cfl in enumerate((production, half, quarter)):
                add(
                    f"operator/{solver}/production_pipeline/{cfl_index}",
                    "operator",
                    {
                        "solver": solver,
                        "operator_role": "production_pipeline",
                        "variant": "proposed_radiation_cosine_buffered",
                        "filter_mode": "disabled",
                        "filter_strength": 0.0,
                        "cfl": cfl,
                        "nx": clean_grid,
                        "ny": 4,
                        "boundary": str(gate["candidate_boundary"]),
                        "use_sponge": bool(candidate["enabled"]),
                        "sponge_axes": str(candidate["axes"]),
                        "sponge_width": max(
                            1,
                            int(
                                round(
                                    clean_grid * float(candidate["width_fraction"])
                                )
                            ),
                        ),
                        "sponge_min_factor": float(
                            candidate["reference_min_factor"]
                        ),
                        "sponge_profile": str(candidate["profile"]),
                    },
                )
            continue
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
                        "operator_role": "legacy_operator_comparison",
                        "variant": variant,
                        "filter_mode": filter_mode,
                        "filter_strength": filter_strength,
                        "cfl": cfl,
                        "nx": 64,
                        "ny": 4,
                        "boundary": "open",
                        "use_sponge": True,
                        "sponge_axes": "x",
                        "sponge_profile": "quadratic",
                    },
                )

    boundary_config = config["boundary_packet"]
    boundary_grid = int(boundary_config["grid"])
    boundary_ny = int(boundary_config["transverse_cells"])
    for solver in SOLVERS:
        boundary_spec = _resolved_boundary_packet_spec(boundary_config, solver)
        if solver == "boussinesq":
            roles = ("reflection", "production_horizon")
        else:
            roles = ("reflection",)
        for variant in sorted(boundary_config["candidates"]):
            candidate = boundary_config["candidates"][variant]
            if bool(candidate.get("swe_only", False)) and solver == "boussinesq":
                continue
            if bool(candidate.get("boussinesq_only", False)) and solver != "boussinesq":
                continue
            for role in roles:
                if solver == "boussinesq":
                    spectral_spec, _finite, _reference, metadata, timing = (
                        _boussinesq_spectral_packet_bundle(
                            boundary_spec, role=role
                        )
                    )
                    nx = spectral_spec.nx
                    ny = spectral_spec.ny
                    dx = spectral_spec.dx
                    dy = spectral_spec.dy
                    requested_times = (
                        candidate_requested_times().tolist()
                        if role == "production_horizon"
                        else timing["requested_times"]
                    )
                else:
                    timing = _boundary_timing(solver, boundary_spec)
                    metadata = {}
                    nx = boundary_grid
                    ny = boundary_ny
                    dx = 1.0 / nx
                    dy = 1.0 / ny
                    requested_times = timing["requested_times"]
                sponge_name = str(candidate["sponge"])
                use_sponge = sponge_name != "disabled"
                if sponge_name == "provisional_cosine":
                    sponge_profile = "cosine"
                    sponge_width = max(1, int(round(3 * nx / 16)))
                    sponge_min_factor = 0.8
                else:
                    sponge_profile = "quadratic"
                    sponge_width = max(1, nx // 8)
                    sponge_min_factor = float(
                        config["operators"]["sponge"]["reference_min_factor"]
                    )
                sponge_axes = "x"
                if (
                    solver == "boussinesq"
                    and role == "production_horizon"
                    and sponge_name != "provisional_cosine"
                ):
                    production_spec = boundary_spec["production_horizon"]
                    sponge_axes = str(production_spec["current_sponge_axes"])
                    sponge_width = int(production_spec["current_sponge_width"])
                task_role = role if solver == "boussinesq" else "swe_reflection"
                add(
                    f"boundary/{solver}/{task_role}/{variant}",
                    "boundary",
                    {
                        "solver": solver,
                        "boundary_role": task_role,
                        "variant": variant,
                        "use_sponge": use_sponge,
                        "boundary": str(candidate["boundary"]),
                        "cfl": float(config["production"]["cfl"][solver][0]),
                        "nx": nx,
                        "ny": ny,
                        "dx": dx,
                        "dy": dy,
                        "sponge_axes": sponge_axes,
                        "sponge_width": sponge_width,
                        "sponge_min_factor": sponge_min_factor,
                        "sponge_profile": sponge_profile,
                        "requested_times": requested_times,
                        "timing": timing,
                        "packet_metadata": metadata,
                        "packet": boundary_spec,
                        "cg_failure_mode": (
                            "legacy_posthoc"
                            if solver == "boussinesq" and role == "reflection"
                            else "strict_v2"
                        ),
                        "cg_absolute_residual_tolerance": (
                            float(boundary_spec[role]["cg_absolute_residual_tolerance"])
                            if solver == "boussinesq"
                            else 0.0
                        ),
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
            production = config["production"]
            domain = production["computational_domain"]
            sponge = production["sponge"]
            boundary_name = str(production["boundary_implementation"][solver])
            canary_spec: dict[str, Any] = {
                "solver": solver,
                "cfl": float(config["production"]["cfl"][solver][0]),
                "canary": _json_safe(canary),
                "computational_grid": int(production["computational_grid"]),
                "publication_grid": int(production["publication_grid"]),
                "buffer_cells": int(domain["buffer_cells"]),
                "source_taper_cells": int(domain["source_taper_cells"]),
                "bathymetry_extension": str(domain["bathymetry_extension"]),
                "output_crop": str(domain["output_crop"]),
                "boundary": boundary_name,
                "use_sponge": bool(sponge["enabled"]),
                "sponge_axes": str(sponge["axes"]),
                "sponge_width": int(sponge["width"]),
                "sponge_min_factor": float(sponge["min_factor"]),
                "sponge_profile": str(sponge["profile"]),
                "sponge_time_mode": str(sponge["time_mode"]),
                "sponge_reference_dt": float(sponge["reference_dt"]),
            }
            if solver == "boussinesq":
                production_boussinesq = config["production"]["boussinesq"]
                canary_spec.update(
                    {
                        "sponge_axes": str(production_boussinesq["sponge_axes"]),
                        "sponge_width": int(production_boussinesq["sponge_width"]),
                        "sponge_min_factor": float(
                            production_boussinesq["sponge_min_factor"]
                        ),
                        "boundary_candidate_exposure": {
                            "zero_gradient_no_sponge": {
                                "use_sponge": False,
                                "axes": str(sponge["axes"]),
                                "width": int(sponge["width"]),
                                "min_factor": float(sponge["min_factor"]),
                                "profile": "quadratic",
                            },
                            "current_elapsed_sponge": {
                                "use_sponge": True,
                                "axes": str(production_boussinesq["sponge_axes"]),
                                "width": int(production_boussinesq["sponge_width"]),
                                "min_factor": float(
                                    production_boussinesq["sponge_min_factor"]
                                ),
                                "profile": "quadratic",
                            },
                            "zero_gradient_cosine_sponge": {
                                "use_sponge": True,
                                "axes": str(sponge["axes"]),
                                "width": int(sponge["width"]),
                                "min_factor": float(sponge["min_factor"]),
                                "profile": str(sponge["profile"]),
                            },
                        },
                    }
                )
            add(
                f"canary/{canary['qualified_id']}/{solver}",
                "canary",
                canary_spec,
            )

    counts: dict[str, int] = {}
    for task in tasks:
        counts[str(task["kind"])] = counts.get(str(task["kind"]), 0) + 1
    expected = {
        "analytical": 27,
        "operator": 19,
        "boundary": 16,
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
            reconstruction_limiter=str(
                spec.get("reconstruction_limiter", "minmod")
            ),
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
        boundary: Any = str(spec.get("boundary", "open"))
        if str(spec.get("operator_role")) == "production_pipeline":
            boundary = (boundary, "open")
        started = time.monotonic()
        eta, dt, diagnostics, solver = _trajectory_eta(
            str(spec["solver"]),
            nx=nx,
            ny=ny,
            cfl=float(spec["cfl"]),
            boundary=boundary,
            use_sponge=bool(spec.get("use_sponge", True)),
            sponge_mode=sponge_mode,
            filter_mode=str(spec["filter_mode"]),
            filter_strength=float(spec["filter_strength"]),
            eta0=eta0,
            sponge_axes=str(spec["sponge_axes"]),
            sponge_width=(
                None
                if spec.get("sponge_width") is None
                else int(spec["sponge_width"])
            ),
            sponge_min_factor=float(spec.get("sponge_min_factor", 0.9)),
            sponge_profile=str(spec.get("sponge_profile", "quadratic")),
        )
        row = {
            "component": "operator_sensitivity",
            "solver": str(spec["solver"]),
            "operator_role": str(
                spec.get("operator_role", "legacy_operator_comparison")
            ),
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
            "measurement_dtype": str(np.asarray(eta).dtype),
            "boundary": _json_safe(boundary),
            "use_sponge": bool(spec.get("use_sponge", True)),
            "sponge_axes": str(spec["sponge_axes"]),
            "sponge_width": int(getattr(solver, "sponge_width", 0)),
            "sponge_min_factor": float(
                getattr(solver, "sponge_min_factor", 1.0)
            ),
            "sponge_profile": str(
                getattr(solver, "sponge_profile", "quadratic")
            ),
            "whole_domain_sponge": bool(np.all(solver.sponge_mask < 1.0)),
            "runtime_s": time.monotonic() - started,
            "operator": solver.get_operator_diagnostics(),
        }
        return row, eta
    if kind == "boundary":
        packet_spec = spec["packet"]
        solver_name = str(spec["solver"])
        if solver_name == "boussinesq":
            spectral_spec, finite_state, _reference, metadata, _timing = (
                _boussinesq_spectral_packet_bundle(
                    packet_spec,
                    role=str(spec["boundary_role"]),
                )
            )
            initial: dict[str, Any] = {
                "eta0": finite_state[0],
                "eta_t0": finite_state[1],
                "characteristic_speed": metadata["group_velocity_max"],
            }
            depth = spectral_spec.depth
        else:
            initial = _boundary_initial_conditions(
                solver_name,
                nx=int(spec["nx"]),
                ny=int(spec["ny"]),
                spec=packet_spec,
            )
            depth = float(packet_spec.get("depth", 1.0))
        eta0 = np.asarray(initial["eta0"])
        use_sponge = bool(spec["use_sponge"])
        bathymetry = -depth * np.ones(
            (int(spec["nx"]), int(spec["ny"])), dtype=np.float64
        )
        started = time.monotonic()
        states, dt, diagnostics, solver = _trajectory_eta(
            solver_name,
            nx=int(spec["nx"]),
            ny=int(spec["ny"]),
            cfl=float(spec["cfl"]),
            boundary=(str(spec["boundary"]), "open"),
            use_sponge=use_sponge,
            sponge_mode="elapsed_time_consistent" if use_sponge else "legacy_per_step",
            eta0=eta0,
            bathymetry=bathymetry,
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
            sponge_width=int(spec["sponge_width"]),
            sponge_min_factor=float(spec["sponge_min_factor"]),
            sponge_profile=str(spec["sponge_profile"]),
            requested_times=np.asarray(spec["requested_times"], dtype=np.float64),
            return_full_state=True,
            dx=float(spec["dx"]),
            dy=float(spec["dy"]),
            cg_failure_mode=str(spec.get("cg_failure_mode", "strict_v2")),
            linear_solver_abs_tol=float(
                spec.get("cg_absolute_residual_tolerance", 0.0)
            ),
        )
        return {
            "component": "boundary_trajectory",
            "solver": str(spec["solver"]),
            "boundary_role": str(spec["boundary_role"]),
            "variant": str(spec["variant"]),
            "characteristic_speed": float(initial["characteristic_speed"]),
            "boundary": str(spec["boundary"]),
            "sponge_axes": str(spec["sponge_axes"]),
            "sponge_width": int(spec["sponge_width"]),
            "sponge_min_factor": float(spec["sponge_min_factor"]),
            "sponge_profile": str(spec["sponge_profile"]),
            "whole_domain_sponge": bool(np.all(solver.sponge_mask < 1.0)),
            "finite": bool(np.isfinite(states).all()),
            "measurement_dtype": str(np.asarray(states).dtype),
            "natural_steps": int(dt.size),
            "runtime_s": time.monotonic() - started,
            "max_post_step_cfl": float(
                np.max(np.asarray(diagnostics["post_step_cfl"], dtype=np.float64))
            ),
            "cg_failure_count": int(
                np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=int))
            ),
            "cg_failure_mode": str(getattr(solver, "cg_failure_mode", "not_applicable")),
            "operator": solver.get_operator_diagnostics(),
        }, states
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
        bathymetry, source, _strength_array, strength, arrays = _load_canary_arrays(
            canary
        )
        solver_name = str(spec["solver"])
        from src.evaluation.buffered_crop_benchmark import run_buffered_case

        buffered_row, eta = run_buffered_case(
            canary,
            solver_name=solver_name,
            total_grid=int(spec["computational_grid"]),
            core_grid=int(spec["publication_grid"]),
            source_taper_cells=int(spec["source_taper_cells"]),
            sponge_min_factor=float(spec["sponge_min_factor"]),
            sponge_width_cells=int(spec["sponge_width"]),
        )
        if (
            int(buffered_row["buffer_cells"]) != int(spec["buffer_cells"])
            or buffered_row["outer_boundary"]
            != ("open" if solver_name == "boussinesq" else "radiation")
            or float(buffered_row["sponge_core_min"]) != 1.0
            or float(buffered_row["source_edge_max_abs"]) != 0.0
        ):
            raise RuntimeError("buffered canary execution changed the frozen policy")
        prepared = _prepare_buffered_domain(
            bathymetry,
            source,
            strength,
            0.0,
            BufferedDomainConfig(
                enabled=True,
                buffer_cells=int(spec["buffer_cells"]),
                source_taper_cells=int(spec["source_taper_cells"]),
                bathymetry_extension=str(spec["bathymetry_extension"]),
                output_crop=str(spec["output_crop"]),
            ),
        )
        health = buffered_row["health"]
        effective_depth = (
            np.maximum(-bathymetry, 1e-4)
            if solver_name == "boussinesq"
            else np.maximum(arrays["rest_depth"], 1e-8)
        )
        canary_row = {
            "component": "production_amplitude_canary",
            "qualified_id": str(canary["qualified_id"]),
            "solver": solver_name,
            "runtime_s": float(health["runtime_s"]),
            "natural_steps": int(health["natural_steps"]),
            "amplitude_growth": float(np.max(np.abs(eta)))
            / max(float(buffered_row["initial_core_amplitude"]), 1e-30),
            "max_eta_over_depth": float(
                np.max(np.abs(eta) / effective_depth[None, ...])
            ),
            "cg_failure_count": int(health["cg_failure_count"]),
            "finite": bool(health["finite"] and np.isfinite(eta).all()),
            "measurement_dtype": str(np.asarray(eta).dtype),
            "output_count": int(np.asarray(eta).shape[0]),
            "requested_times_exact": bool(health["requested_times_exact"]),
            "max_post_step_cfl": float(health["max_post_step_cfl"]),
            "computational_grid": int(buffered_row["total_grid"]),
            "publication_grid": int(buffered_row["core_grid"]),
            "publication_shape": list(map(int, np.asarray(eta).shape)),
            "buffer_cells": int(buffered_row["buffer_cells"]),
            "source_taper_cells": int(buffered_row["source_taper_cells"]),
            "source_edge_max_abs": float(buffered_row["source_edge_max_abs"]),
            "sponge_core_min": float(buffered_row["sponge_core_min"]),
            "outer_boundary": str(buffered_row["outer_boundary"]),
            "operator": health["operator"],
        }
        if solver_name == "boussinesq":
            horizon = float(candidate_requested_times()[-1])
            solver_eta0 = np.asarray(prepared["solver_eta0"], dtype=np.float64)
            solver_depth = np.maximum(
                -np.asarray(prepared["solver_bathymetry"], dtype=np.float64),
                1.0e-4,
            )
            production_mask = build_sponge_mask(
                nx=int(spec["computational_grid"]),
                ny=int(spec["computational_grid"]),
                width=int(spec["sponge_width"]),
                min_factor=float(spec["sponge_min_factor"]),
                axes=str(spec["sponge_axes"]),
                profile=str(spec["sponge_profile"]),
            )
            canary_row.update(
                _boussinesq_h0_exposure_metrics(
                    solver_eta0,
                    solver_depth,
                    production_mask,
                    horizon=horizon,
                )
            )
            candidate_exposure: dict[str, Any] = {}
            for variant, candidate in spec["boundary_candidate_exposure"].items():
                if bool(candidate["use_sponge"]):
                    candidate_mask = build_sponge_mask(
                        nx=solver_eta0.shape[0],
                        ny=solver_eta0.shape[1],
                        width=int(candidate["width"]),
                        min_factor=float(candidate["min_factor"]),
                        axes=str(candidate["axes"]),
                        profile=str(candidate["profile"]),
                    )
                else:
                    candidate_mask = np.ones_like(solver_eta0, dtype=np.float64)
                candidate_exposure[str(variant)] = _boussinesq_h0_exposure_metrics(
                    solver_eta0,
                    solver_depth,
                    candidate_mask,
                    horizon=horizon,
                )
            canary_row["boundary_candidate_exposure"] = candidate_exposure
        return canary_row, None
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
            "measurement_dtype": "float64",
        }
    elif kind == "operator":
        expected = {
            "component": "operator_sensitivity",
            "solver": str(spec["solver"]),
            "operator_role": str(spec["operator_role"]),
            "variant": str(spec["variant"]),
            "cfl": float(spec["cfl"]),
            "sponge_axes": str(spec["sponge_axes"]),
            "measurement_dtype": "float64",
        }
    elif kind == "boundary":
        expected = {
            "component": "boundary_trajectory",
            "solver": str(spec["solver"]),
            "variant": str(spec["variant"]),
            "boundary": str(spec["boundary"]),
            "sponge_axes": str(spec["sponge_axes"]),
            "sponge_profile": str(spec["sponge_profile"]),
            "measurement_dtype": "float64",
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
            "measurement_dtype": "float64",
            "computational_grid": int(spec["computational_grid"]),
            "publication_grid": int(spec["publication_grid"]),
            "buffer_cells": int(spec["buffer_cells"]),
            "source_taper_cells": int(spec["source_taper_cells"]),
            "source_edge_max_abs": 0.0,
            "sponge_core_min": 1.0,
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
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
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
    progress_started = time.monotonic()
    completed_count = len(loaded)

    def emit_progress(event: str, task: Mapping[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {
            "event": event,
            "completed": int(completed_count),
            "total": int(len(tasks)),
            "pending": int(len(tasks) - completed_count),
            "workers": int(workers),
            "elapsed_s": float(time.monotonic() - progress_started),
        }
        if task is not None:
            payload.update(
                {
                    "task_id": str(task["task_id"]),
                    "kind": str(task["kind"]),
                    "ordinal": int(task["ordinal"]),
                }
            )
        progress_callback(payload)

    emit_progress("start")
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
            completed_count += 1
            emit_progress("task_completed", task)
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
                    completed_count += 1
                    emit_progress("task_completed", task)
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
    emit_progress("complete")
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
    sponge_width: int | None = None,
    sponge_min_factor: float = 0.9,
    sponge_profile: str = "quadratic",
    reconstruction_limiter: str = "minmod",
    requested_times: np.ndarray | None = None,
    return_full_state: bool = False,
    dx: float | None = None,
    dy: float | None = None,
    cg_failure_mode: str = "strict_v2",
    linear_solver_abs_tol: float = 0.0,
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
        sponge_width=sponge_width,
        sponge_min_factor=sponge_min_factor,
        sponge_profile=sponge_profile,
        reconstruction_limiter=reconstruction_limiter,
        dx=dx,
        dy=dy,
        cg_failure_mode=cg_failure_mode,
        linear_solver_abs_tol=linear_solver_abs_tol,
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
    times = (
        candidate_requested_times()
        if requested_times is None
        else np.asarray(requested_times, dtype=np.float64)
    )
    states, _, dt_history, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=cfl,
        include_initial_state=False,
        requested_times=times,
        max_natural_steps=20000,
        collect_natural_step_health=True,
        requested_state_dtype=np.float64,
    )
    eta = states[:, 0] if name == "boussinesq" else states[:, 0] + bathy
    trajectory = states if return_full_state else eta
    return trajectory, dt_history, diagnostics, solver


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


def _boussinesq_h0_exposure_metrics(
    eta0: np.ndarray,
    effective_depth: np.ndarray,
    sponge_mask: np.ndarray,
    *,
    horizon: float,
    energy_tail: float = 1.0e-6,
) -> dict[str, Any]:
    values = np.asarray(eta0, dtype=np.float64)
    depth = np.asarray(effective_depth, dtype=np.float64)
    damping = np.asarray(sponge_mask, dtype=np.float64)
    if values.shape != depth.shape or values.shape != damping.shape:
        raise ValueError("Boussinesq H0 exposure arrays must share one shape")
    if values.ndim != 2 or horizon <= 0.0 or not (0.0 < energy_tail < 1.0):
        raise ValueError("invalid Boussinesq H0 exposure specification")
    weights = values**2
    total = max(float(math.fsum(weights.ravel())), 1.0e-300)
    flat_order = np.argsort(weights.ravel())[::-1]
    cumulative = np.cumsum(weights.ravel()[flat_order])
    support_count = min(
        int(np.searchsorted(cumulative, (1.0 - energy_tail) * total)) + 1,
        weights.size,
    )
    significant = np.zeros(weights.size, dtype=bool)
    significant[flat_order[:support_count]] = True
    significant = significant.reshape(weights.shape)
    x = (np.arange(weights.shape[0], dtype=np.float64) + 0.5) / weights.shape[0]
    y = (np.arange(weights.shape[1], dtype=np.float64) + 0.5) / weights.shape[1]
    x_distance = np.minimum(x, 1.0 - x)[:, None]
    y_distance = np.minimum(y, 1.0 - y)[None, :]
    distance = np.minimum(x_distance, y_distance)
    support_distance = float(np.min(distance[significant]))
    reach = float(math.sqrt(9.81 * float(np.max(depth))) * horizon)
    sponge = damping < 1.0
    return {
        "initial_sponge_energy_fraction": float(
            math.fsum(weights[sponge].ravel()) / total
        ),
        "significant_source_distance_to_boundary": support_distance,
        "conservative_long_wave_reach": reach,
        "production_horizon": float(horizon),
        "significant_source_overlaps_sponge": bool(np.any(significant & sponge)),
        "conservative_boundary_reachable": bool(support_distance <= reach),
        "exposure_role": "diagnostic_not_used_to_tune_thresholds",
    }


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
    direction = str(spec.get("direction", "left"))
    if direction not in ("left", "right"):
        raise ValueError("boundary packet direction must be left or right")
    support_sigmas = float(spec["support_sigmas"])
    post_exit_sigmas = float(spec["post_exit_observation_sigmas"])
    if support_sigmas <= 0.0 or post_exit_sigmas <= 0.0:
        raise ValueError("boundary packet support and observation widths must be positive")
    support_lower = center - support_sigmas * sigma
    support_upper = center + support_sigmas * sigma
    if support_lower <= 0.0 or support_upper >= 1.0:
        raise ValueError("boundary packet analytical support must begin inside the domain")
    incident = _window_mask(nx, spec["incident_window"])
    reflected = _window_mask(nx, spec["reflected_window"])
    interior = _window_mask(nx, spec["interior_window"])
    if np.any(interior & _x_sponge_region(nx, sponge_width)):
        raise ValueError("boundary interior window overlaps the x-only sponge")
    packet = _packet(nx, 1, center=center, sigma=sigma, zero_mean=False)[:, 0]
    incident_energy_fraction = float(np.sum(packet[incident] ** 2)) / max(
        float(np.sum(packet**2)), 1e-30
    )
    if incident_energy_fraction < float(spec["minimum_initial_incident_energy_fraction"]):
        raise ValueError("incident window does not contain enough initial packet energy")
    captured_distance = (2.0 * support_sigmas + post_exit_sigmas) * sigma
    x = np.arange(nx, dtype=np.float64) / nx
    if direction == "left":
        capture_limit = float(np.max(x[reflected]))
    else:
        capture_limit = 1.0 - float(np.min(x[reflected]))
    if capture_limit < captured_distance:
        raise ValueError("reflected window cannot contain the separated packet")


def _boundary_characteristic_speed(
    solver_name: str, spec: Mapping[str, Any]
) -> float:
    depth = float(spec.get("depth", 1.0))
    if depth <= 0.0:
        raise ValueError("boundary packet depth must be positive")
    if solver_name == "boussinesq":
        dominant_wavenumber = 1.0 / float(spec["sigma"])
        return math.sqrt(9.81 * depth) / math.sqrt(
            1.0 + (depth * dominant_wavenumber) ** 2 / 3.0
        )
    return math.sqrt(9.81 * depth)


def _boussinesq_directional_components(
    states: np.ndarray, *, depth: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] < 2:
        raise ValueError("Boussinesq directional decomposition requires [time,2,x,y]")
    nx = int(values.shape[2])
    dx = 1.0 / nx
    wavenumber = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    discrete_wavenumber = 2.0 * np.sin(0.5 * wavenumber * dx) / dx
    signed_omega = np.sign(discrete_wavenumber) * np.sqrt(
        9.81 * depth * discrete_wavenumber**2
        / (1.0 + depth * depth * discrete_wavenumber**2 / 3.0)
    )
    eta_hat = np.fft.fft(values[:, 0], axis=1)
    eta_t_hat = np.fft.fft(values[:, 1], axis=1)
    inverse_omega = np.zeros_like(signed_omega)
    nonzero = signed_omega != 0.0
    inverse_omega[nonzero] = 1.0 / signed_omega[nonzero]
    scaled_rate = 1j * eta_t_hat * inverse_omega[None, :, None]
    rightgoing_hat = 0.5 * (eta_hat + scaled_rate)
    leftgoing_hat = 0.5 * (eta_hat - scaled_rate)
    rightgoing_hat[:, ~nonzero, :] = 0.0
    leftgoing_hat[:, ~nonzero, :] = 0.0
    return (
        np.fft.ifft(rightgoing_hat, axis=1).real,
        np.fft.ifft(leftgoing_hat, axis=1).real,
    )


def _boussinesq_directional_rate(
    eta: np.ndarray, *, depth: float, direction: str
) -> np.ndarray:
    values = np.asarray(eta, dtype=np.float64)
    states = np.stack([values, np.zeros_like(values)], axis=0)[None, ...]
    nx = int(values.shape[0])
    dx = 1.0 / nx
    wavenumber = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    discrete_wavenumber = 2.0 * np.sin(0.5 * wavenumber * dx) / dx
    signed_omega = np.sign(discrete_wavenumber) * np.sqrt(
        9.81 * depth * discrete_wavenumber**2
        / (1.0 + depth * depth * discrete_wavenumber**2 / 3.0)
    )
    eta_hat = np.fft.fft(states[0, 0], axis=0)
    sign = 1.0 if direction == "left" else -1.0
    rate_hat = sign * 1j * signed_omega[:, None] * eta_hat
    return np.fft.ifft(rate_hat, axis=0).real


def _boundary_timing(
    solver_name: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    center = float(spec["center_x"])
    sigma = float(spec["sigma"])
    support = float(spec["support_sigmas"])
    post_exit = float(spec["post_exit_observation_sigmas"])
    speed = _boundary_characteristic_speed(solver_name, spec)
    front_speed = (
        math.sqrt(9.81 * float(spec.get("depth", 1.0)))
        if solver_name == "boussinesq"
        else speed
    )
    lower = center - support * sigma
    upper = center + support * sigma
    if str(spec["direction"]) == "left":
        leading_distance = lower
        trailing_distance = upper
    else:
        leading_distance = 1.0 - upper
        trailing_distance = 1.0 - lower
    arrival = leading_distance / front_speed
    center_arrival = (
        center / speed
        if str(spec["direction"]) == "left"
        else (1.0 - center) / speed
    )
    exit_time = trailing_distance / speed
    observation_end = exit_time + post_exit * sigma / speed
    pre_count = int(spec["prearrival_sample_count"])
    post_count = int(spec["post_exit_sample_count"])
    if pre_count < 2 or post_count < 2:
        raise ValueError("boundary timing requires at least two pre/post samples")
    prearrival = np.linspace(0.25 * arrival, 0.9 * arrival, pre_count)
    postexit = np.linspace(exit_time, observation_end, post_count)
    requested = np.unique(np.concatenate([prearrival, postexit])).astype(np.float64)
    if (
        requested.size != pre_count + post_count
        or np.any(requested <= 0.0)
        or np.any(np.diff(requested) <= 0.0)
        or float(prearrival[-1]) >= arrival
        or float(postexit[0]) < exit_time
    ):
        raise ValueError("boundary packet timing is not temporally separated")
    analytical_tail = 0.5 * math.erfc(support)
    return {
        "characteristic_speed": speed,
        "front_speed_bound": front_speed,
        "leading_edge_arrival_time": arrival,
        "center_arrival_time": center_arrival,
        "trailing_edge_exit_time": exit_time,
        "observation_end_time": observation_end,
        "prearrival_times": prearrival.tolist(),
        "postexit_times": postexit.tolist(),
        "requested_times": requested.tolist(),
        "analytical_outgoing_energy_tail_fraction": analytical_tail,
        "maximum_outgoing_energy_fraction_after_exit": analytical_tail
        * float(spec["outgoing_tail_safety_factor"]),
    }


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
    phase_speed = _boundary_characteristic_speed(solver_name, spec)
    depth = float(spec.get("depth", 1.0))
    if solver_name == "boussinesq":
        eta_t0 = _boussinesq_directional_rate(
            eta0, depth=depth, direction=direction
        )
        return {
            "eta0": eta0,
            "eta_t0": eta_t0,
            "characteristic_speed": phase_speed,
        }
    return {
        "eta0": eta0,
        "h0": depth + eta0,
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
    def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        difference = np.asarray(a, dtype=np.float64) - np.asarray(
            b, dtype=np.float64
        )
        return {
            "absolute_rms": _stable_l2_norm(difference)
            / math.sqrt(difference.size),
            "relative_l2": _stable_l2_norm(difference)
            / max(_stable_l2_norm(np.asarray(b, dtype=np.float64)), 1.0e-30),
        }

    full = metrics(first, second)
    final = metrics(first[-1], second[-1])
    sponge = metrics(first[:, sponge_region, :], second[:, sponge_region, :])
    sponge_final = metrics(
        first[-1, sponge_region, :], second[-1, sponge_region, :]
    )
    interior = metrics(
        first[:, interior_region, :], second[:, interior_region, :]
    )
    interior_final = metrics(
        first[-1, interior_region, :], second[-1, interior_region, :]
    )
    return {
        "trajectory_absolute_rms": full["absolute_rms"],
        "trajectory_relative_l2": _relative_l2(first, second),
        "final_time_absolute_rms": final["absolute_rms"],
        "final_time_relative_l2": _relative_l2(first[-1], second[-1]),
        "sponge_trajectory_absolute_rms": sponge["absolute_rms"],
        "sponge_trajectory_relative_l2": _relative_l2(
            first[:, sponge_region, :], second[:, sponge_region, :]
        ),
        "sponge_final_time_absolute_rms": sponge_final["absolute_rms"],
        "sponge_final_time_relative_l2": _relative_l2(
            first[-1, sponge_region, :], second[-1, sponge_region, :]
        ),
        "interior_trajectory_absolute_rms": interior["absolute_rms"],
        "interior_trajectory_relative_l2": _relative_l2(
            first[:, interior_region, :], second[:, interior_region, :]
        ),
        "interior_final_time_absolute_rms": interior_final["absolute_rms"],
        "interior_final_time_relative_l2": _relative_l2(
            first[-1, interior_region, :], second[-1, interior_region, :]
        ),
    }


def _evaluation_precision_floor(
    trajectories: Sequence[np.ndarray], *, safety_factor: float
) -> float:
    scales = [
        _stable_l2_norm(np.asarray(values, dtype=np.float64))
        / math.sqrt(np.asarray(values).size)
        for values in trajectories
    ]
    return float(
        safety_factor * np.finfo(np.float64).eps * max([*scales, 1.0e-30])
    )


def _restrict_x_aligned(values: np.ndarray, target_nx: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("spatial restriction requires [time, x, y]")
    source_nx = int(array.shape[1])
    if target_nx <= 0 or source_nx % int(target_nx) != 0:
        raise ValueError("fine x grid must be an integer multiple of target_nx")
    ratio = source_nx // int(target_nx)
    return np.asarray(array[:, ::ratio, :], dtype=np.float64)


def _hydro_clean_temporal_metrics(
    trajectories: Sequence[np.ndarray],
    *,
    minimum_order: float,
    precision_floor_safety_factor: float,
) -> dict[str, Any]:
    if len(trajectories) != 4:
        raise ValueError(
            "Hydro clean temporal comparison requires production through eighth CFL"
        )
    differences = [
        _trajectory_rms_difference(trajectories[index], trajectories[index + 1])
        for index in range(3)
    ]
    orders = [
        _observed_order(differences[index], differences[index + 1])
        for index in range(2)
    ]
    floor = _evaluation_precision_floor(
        trajectories, safety_factor=precision_floor_safety_factor
    )
    below_floor = all(value <= floor for value in differences)
    monotone = differences[0] > differences[1] > differences[2]
    passed = below_floor or (
        monotone
        and all(order is not None and order >= minimum_order for order in orders)
    )
    reference = np.asarray(trajectories[-1], dtype=np.float64)
    reference_errors = []
    for trajectory in trajectories[:-1]:
        difference = np.asarray(trajectory, dtype=np.float64) - reference
        reference_errors.append(
            {
                "absolute_rms": _stable_l2_norm(difference)
                / math.sqrt(difference.size),
                "relative_l2": _stable_l2_norm(difference)
                / max(_stable_l2_norm(reference), 1.0e-30),
            }
        )
    return {
        "pairwise_absolute_rms": differences,
        "pairwise_orders": orders,
        "reference_errors": reference_errors,
        "precision_floor_absolute_rms": floor,
        "precision_floor_method": "float64_eps_scaled_trajectory_rms",
        "all_below_precision_floor": below_floor,
        "monotone_refinement": monotone,
        "minimum_order": minimum_order,
        "passed": passed,
    }


def _hydro_spatial_control_metrics(
    trajectories_by_grid: Mapping[int, np.ndarray],
    *,
    minimum_order: float,
    precision_floor_safety_factor: float,
) -> dict[str, Any]:
    grids = sorted(int(grid) for grid in trajectories_by_grid)
    if len(grids) != 3 or grids[1] != 2 * grids[0] or grids[2] != 2 * grids[1]:
        raise ValueError("Hydro spatial control requires three factor-two grids")
    coarse, middle, fine = (np.asarray(trajectories_by_grid[grid]) for grid in grids)
    middle_on_coarse = _restrict_x_aligned(middle, grids[0])
    fine_on_middle = _restrict_x_aligned(fine, grids[1])
    errors = [
        _trajectory_rms_difference(coarse, middle_on_coarse),
        _trajectory_rms_difference(middle, fine_on_middle),
    ]
    order = _observed_order(errors[0], errors[1])
    floor = _evaluation_precision_floor(
        [coarse, middle, fine], safety_factor=precision_floor_safety_factor
    )
    below_floor = all(value <= floor for value in errors)
    passed = below_floor or (
        errors[0] > errors[1]
        and order is not None
        and order >= minimum_order
    )
    return {
        "grids": grids,
        "restriction_method": "aligned_x_i_over_n_subsampling",
        "pairwise_absolute_rms": errors,
        "observed_order": order,
        "precision_floor_absolute_rms": floor,
        "all_below_precision_floor": below_floor,
        "monotone_refinement": errors[0] > errors[1],
        "minimum_order": minimum_order,
        "passed": passed,
    }


def _boundary_metrics(
    *,
    solver_name: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
    initial_conditions: Mapping[str, Any],
    bathymetry: np.ndarray,
    spec: Mapping[str, Any],
    timing: Mapping[str, Any],
    timestamps: np.ndarray,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.ndim != 4:
        raise ValueError(
            "boundary trajectories must have equal [time, state, x, y] shape"
        )
    nx = int(baseline.shape[2])
    incident = _window_mask(nx, spec["incident_window"])
    reflected = _window_mask(nx, spec["reflected_window"])
    interior = _window_mask(nx, spec["interior_window"])
    times = np.asarray(timestamps, dtype=np.float64)
    expected_times = np.asarray(timing["requested_times"], dtype=np.float64)
    if not np.array_equal(times, expected_times):
        raise ValueError("boundary trajectory timestamps do not match derived timing")
    arrival = float(timing["leading_edge_arrival_time"])
    exit_time = float(timing["trailing_edge_exit_time"])
    prearrival = times < arrival
    postexit = times >= exit_time
    temporally_separated = bool(
        np.any(prearrival)
        and np.any(postexit)
        and float(np.max(times[prearrival])) < arrival
        and float(np.min(times[postexit])) >= exit_time
    )
    if not temporally_separated:
        raise ValueError("boundary measurement is not temporally separated")

    initial_eta = np.asarray(initial_conditions["eta0"], dtype=np.float64)
    if solver_name == "boussinesq":
        baseline_eta = np.asarray(baseline[:, 0], dtype=np.float64)
        candidate_eta = np.asarray(candidate[:, 0], dtype=np.float64)
        depth = float(spec.get("depth", 1.0))
        rightgoing, leftgoing = _boussinesq_directional_components(
            candidate, depth=depth
        )
        initial_state = np.stack(
            [
                initial_eta,
                np.asarray(initial_conditions["eta_t0"], dtype=np.float64),
            ],
            axis=0,
        )[None, ...]
        initial_rightgoing, initial_leftgoing = _boussinesq_directional_components(
            initial_state, depth=depth
        )
        if str(spec["direction"]) == "left":
            initial_outgoing = initial_leftgoing[0]
            outgoing_component = leftgoing
            reflected_component = rightgoing
        else:
            initial_outgoing = initial_rightgoing[0]
            outgoing_component = rightgoing
            reflected_component = leftgoing
        initial_total_energy = max(
            _stable_sum(initial_outgoing[incident] ** 2), 1.0e-30
        )
        postexit_indices = np.flatnonzero(postexit)
        clearance_by_time = np.asarray(
            [
                _stable_sum(outgoing_component[index] ** 2)
                / initial_total_energy
                for index in postexit_indices
            ],
            dtype=np.float64,
        )
        qualified = np.flatnonzero(
            clearance_by_time
            <= float(timing["maximum_outgoing_energy_fraction_after_exit"])
        )
        exit_achieved = qualified.size > 0
        clearance_position = int(qualified[0]) if exit_achieved else -1
        clearance_index = int(postexit_indices[clearance_position])
        outgoing_clearance = float(clearance_by_time[clearance_position])
        numerical_exit_time = float(times[clearance_index])
        measurement_postexit = np.zeros(times.shape, dtype=bool)
        measurement_postexit[clearance_index:] = True
        decomposition = "discrete_boussinesq_spectral_directional_components"
    else:
        rest_depth = np.maximum(-np.asarray(bathymetry, dtype=np.float64), 1.0e-12)
        wave_speed = np.sqrt(9.81 * rest_depth)
        baseline_eta = np.asarray(baseline[:, 0], dtype=np.float64) - rest_depth
        candidate_eta = np.asarray(candidate[:, 0], dtype=np.float64) - rest_depth
        candidate_discharge = np.asarray(candidate[:, 1], dtype=np.float64)
        rightgoing = 0.5 * (candidate_eta + candidate_discharge / wave_speed)
        leftgoing = 0.5 * (candidate_eta - candidate_discharge / wave_speed)
        initial_discharge = np.asarray(initial_conditions["hu0"], dtype=np.float64)
        initial_rightgoing = 0.5 * (initial_eta + initial_discharge / wave_speed)
        initial_leftgoing = 0.5 * (initial_eta - initial_discharge / wave_speed)
        if str(spec["direction"]) == "left":
            initial_outgoing = initial_leftgoing
            outgoing_component = leftgoing
            reflected_component = rightgoing
        else:
            initial_outgoing = initial_rightgoing
            outgoing_component = rightgoing
            reflected_component = leftgoing
        initial_total_energy = max(
            _stable_sum(initial_outgoing[incident] ** 2), 1.0e-30
        )
        postexit_indices = np.flatnonzero(postexit)
        clearance_by_time = np.asarray(
            [
                _stable_sum(outgoing_component[index] ** 2)
                / initial_total_energy
                for index in postexit_indices
            ],
            dtype=np.float64,
        )
        qualified = np.flatnonzero(
            clearance_by_time
            <= float(timing["maximum_outgoing_energy_fraction_after_exit"])
        )
        exit_achieved = qualified.size > 0
        clearance_position = int(qualified[0]) if exit_achieved else -1
        clearance_index = int(postexit_indices[clearance_position])
        outgoing_clearance = float(clearance_by_time[clearance_position])
        numerical_exit_time = float(times[clearance_index])
        measurement_postexit = np.zeros(times.shape, dtype=bool)
        measurement_postexit[clearance_index:] = True
        decomposition = "linear_shallow_water_characteristics"

    incident_amplitude = max(
        float(np.max(np.abs(initial_outgoing[incident]))), 1e-30
    )
    incident_energy = max(
        _stable_sum(initial_outgoing[incident] ** 2), 1e-30
    )
    reflected_postexit = reflected_component[measurement_postexit][:, reflected, :]
    reflected_energy_by_time = np.asarray(
        [_stable_sum(frame**2) for frame in reflected_postexit], dtype=np.float64
    )
    pre_difference = (
        candidate_eta[prearrival][:, interior, :]
        - baseline_eta[prearrival][:, interior, :]
    )
    pre_reference = baseline_eta[prearrival][:, interior, :]
    post_interior = candidate_eta[measurement_postexit][:, interior, :]
    post_interior_energy = np.asarray(
        [_stable_sum(frame**2) for frame in post_interior], dtype=np.float64
    )
    return {
        "leading_edge_arrival_time": arrival,
        "center_arrival_time": float(timing["center_arrival_time"]),
        "trailing_edge_exit_time": exit_time,
        "observation_end_time": float(timing["observation_end_time"]),
        "measurement_temporally_separated": temporally_separated,
        "packet_exit_achieved": bool(exit_achieved),
        "reflection_metrics_valid": bool(exit_achieved),
        "numerical_packet_exit_time": numerical_exit_time,
        "outgoing_energy_fraction_after_exit": outgoing_clearance,
        "maximum_outgoing_energy_fraction_after_exit": float(
            timing["maximum_outgoing_energy_fraction_after_exit"]
        ),
        "decomposition": decomposition,
        "incident_amplitude": incident_amplitude,
        "incident_energy": incident_energy,
        "reflected_amplitude_ratio": float(np.max(np.abs(reflected_postexit)))
        / incident_amplitude,
        "reflected_energy_ratio": float(np.max(reflected_energy_by_time))
        / incident_energy,
        "interior_relative_l2": _stable_l2_norm(pre_difference)
        / max(_stable_l2_norm(pre_reference), 1.0e-30),
        "post_exit_interior_energy_ratio": float(np.max(post_interior_energy))
        / incident_energy,
    }


def _boussinesq_reference_crop(
    reference_state: np.ndarray,
    times: np.ndarray,
    *,
    spec: SpectralPacketSpec,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    evolved = evolve_boussinesq_reference(reference_state, times, spec=spec)
    start = int(metadata["crop_start"])
    stop = int(metadata["crop_stop"])
    return np.asarray(evolved[:, :, start:stop], dtype=np.float64)


def _boussinesq_reference_refinement_error(
    *,
    spec: SpectralPacketSpec,
    coarse_reference: np.ndarray,
    coarse_metadata: Mapping[str, Any],
    times: np.ndarray,
) -> float:
    coarse = _boussinesq_reference_crop(
        coarse_reference,
        times,
        spec=spec,
        metadata=coarse_metadata,
    )
    fine_spec = replace(spec, dx=0.5 * spec.dx)
    _fine_initial, fine_reference, fine_metadata = build_reference_packet(fine_spec)
    fine = _boussinesq_reference_crop(
        fine_reference,
        times,
        spec=fine_spec,
        metadata=fine_metadata,
    )[:, :, ::2]
    initial_energy = max(
        boussinesq_discrete_energy(
            coarse[0],
            dx=spec.dx,
            dy=spec.dy,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        ),
        1.0e-300,
    )
    ratios = []
    for difference in coarse - fine:
        ratios.append(
            boussinesq_discrete_energy(
                difference,
                dx=spec.dx,
                dy=spec.dy,
                depth=spec.depth,
                gravity=spec.gravity,
                alpha=spec.alpha,
            )
            / initial_energy
        )
    return float(max(ratios, default=0.0))


def _boussinesq_reference_boundary_metrics(
    *,
    candidate: np.ndarray,
    timestamps: np.ndarray,
    packet_spec: Mapping[str, Any],
    role: str,
    sponge_width: int,
    reference_refinement_error_ratio: float,
    uncertainty_fraction: float,
    reflected_energy_ceiling: float,
    production_error_ceiling: float,
    precision_floor_safety_factor: float,
) -> dict[str, Any]:
    spec, initial, reference_state, metadata, timing = (
        _boussinesq_spectral_packet_bundle(packet_spec, role=role)
    )
    times = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(candidate, dtype=np.float64)
    if values.shape != (times.size, 2, spec.nx, spec.ny):
        raise ValueError("Boussinesq boundary trajectory shape mismatch")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Boussinesq boundary timestamps must increase")
    reference = _boussinesq_reference_crop(
        reference_state,
        times,
        spec=spec,
        metadata=metadata,
    )
    difference = values - reference
    taper_cells = int(packet_spec[role]["taper_edge_cells"])
    taper = cosine_taper(spec.nx, taper_cells)[:, None]
    tapered_difference = difference * taper[None, None, :, :]
    right_error, left_error = boussinesq_directional_states(
        tapered_difference,
        dx=spec.dx,
        depth=spec.depth,
        gravity=spec.gravity,
        alpha=spec.alpha,
    )
    if spec.direction == "left":
        reflected_error = right_error
        outgoing_candidate = boussinesq_directional_states(
            values * taper[None, None, :, :],
            dx=spec.dx,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        )[1]
    else:
        reflected_error = left_error
        outgoing_candidate = boussinesq_directional_states(
            values * taper[None, None, :, :],
            dx=spec.dx,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        )[0]

    initial_energy = max(
        boussinesq_discrete_energy(
            initial,
            dx=spec.dx,
            dy=spec.dy,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        ),
        1.0e-300,
    )
    reference_right, reference_left = boussinesq_directional_states(
        reference_state[None],
        dx=spec.dx,
        depth=spec.depth,
        gravity=spec.gravity,
        alpha=spec.alpha,
    )
    initial_wrong = (
        reference_right[0] if spec.direction == "left" else reference_left[0]
    )
    reference_initial_energy = max(
        boussinesq_discrete_energy(
            reference_state,
            dx=spec.dx,
            dy=spec.dy,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        ),
        1.0e-300,
    )
    wrong_way_ratio = boussinesq_discrete_energy(
        initial_wrong,
        dx=spec.dx,
        dy=spec.dy,
        depth=spec.depth,
        gravity=spec.gravity,
        alpha=spec.alpha,
    ) / reference_initial_energy
    precision_floor = float(
        precision_floor_safety_factor * np.finfo(np.float64).eps
    )
    numerical_floor = max(
        precision_floor,
        4.0 * float(reference_refinement_error_ratio),
        4.0 * float(wrong_way_ratio),
    )

    common_sponge_width = max(sponge_width, int(round(3 * spec.nx / 16)))
    interior = np.zeros(spec.nx, dtype=bool)
    interior[common_sponge_width : spec.nx - common_sponge_width] = True
    boundary = ~interior
    interior_energy_errors: list[float] = []
    boundary_energy_errors: list[float] = []
    interior_absolute_rms: list[float] = []
    for frame in difference:
        density = boussinesq_energy_density(
            frame,
            dx=spec.dx,
            dy=spec.dy,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        )
        interior_energy_errors.append(
            float(math.fsum(density[interior].ravel()) * spec.dx * spec.dy)
            / initial_energy
        )
        boundary_energy_errors.append(
            float(math.fsum(density[boundary].ravel()) * spec.dx * spec.dy)
            / initial_energy
        )
        interior_absolute_rms.append(
            float(math.sqrt(np.mean(frame[0, interior] ** 2)))
        )

    if role == "reflection":
        measurement = times >= float(timing["trailing_edge_exit_time"])
        separated = (
            bool(timing["reference_safe"])
            and np.any(measurement)
            and float(times[-1]) >= float(timing["observation_end_time"])
        )
    else:
        measurement = times <= float(candidate_requested_times()[-1])
        separated = bool(timing["reference_safe"]) and np.any(measurement)
    if np.any(measurement):
        reflected_amplitude = float(
            np.max(np.abs(reflected_error[measurement, 0]))
            / max(float(np.max(np.abs(initial[0]))), 1.0e-300)
        )
        reflected_energy = max(
            boussinesq_discrete_energy(
                frame,
                dx=spec.dx,
                dy=spec.dy,
                depth=spec.depth,
                gravity=spec.gravity,
                alpha=spec.alpha,
            )
            / initial_energy
            for frame in reflected_error[measurement]
        )
        remaining_outgoing = max(
            boussinesq_discrete_energy(
                frame,
                dx=spec.dx,
                dy=spec.dy,
                depth=spec.depth,
                gravity=spec.gravity,
                alpha=spec.alpha,
            )
            / initial_energy
            for frame in outgoing_candidate[measurement]
        )
    else:
        reflected_amplitude = math.inf
        reflected_energy = math.inf
        remaining_outgoing = math.inf
    uncertainty_limit = uncertainty_fraction * min(
        reflected_energy_ceiling,
        production_error_ceiling,
    )
    return {
        "protocol": "spectral_large_domain_v1",
        "boundary_role": role,
        "measurement_temporally_separated": bool(separated),
        "reflection_metrics_valid": bool(separated),
        "spectral_exit_horizon_achieved": bool(
            separated if role == "reflection" else False
        ),
        "reference_safe": bool(timing["reference_safe"]),
        "leading_edge_arrival_time": float(timing["leading_edge_arrival_time"]),
        "trailing_edge_exit_time": float(timing["trailing_edge_exit_time"]),
        "observation_end_time": float(timing["observation_end_time"]),
        "significant_k_min": float(metadata["significant_k_min"]),
        "significant_k_max": float(metadata["significant_k_max"]),
        "group_velocity_min": float(metadata["group_velocity_min"]),
        "group_velocity_max": float(metadata["group_velocity_max"]),
        "reference_initial_mean_eta": float(np.mean(reference_state[0])),
        "finite_crop_initial_mean_eta": float(np.mean(initial[0])),
        "initial_wrong_way_energy_ratio": float(wrong_way_ratio),
        "reference_refinement_error_ratio": float(
            reference_refinement_error_ratio
        ),
        "reference_uncertainty_limit": float(uncertainty_limit),
        "reference_adequate": bool(
            reference_refinement_error_ratio <= uncertainty_limit
        ),
        "normalized_numerical_floor": float(numerical_floor),
        "reflected_amplitude_ratio": reflected_amplitude,
        "reflected_energy_ratio": float(reflected_energy),
        "interior_energy_error_ratio": float(max(interior_energy_errors)),
        "interior_absolute_rms": float(max(interior_absolute_rms)),
        "boundary_layer_energy_error_ratio": float(max(boundary_energy_errors)),
        "remaining_outgoing_energy_ratio": float(remaining_outgoing),
        "interior_start_index": int(common_sponge_width),
        "interior_stop_index": int(spec.nx - common_sponge_width),
        "measurement_dtype": str(values.dtype),
    }


def _aggregate_boussinesq_boundary_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    packet_spec: Mapping[str, Any],
    boundary_config: Mapping[str, Any],
    boundary_thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    expected_variants = {
        str(variant)
        for variant, candidate in boundary_config["candidates"].items()
        if not bool(candidate.get("swe_only", False))
    }
    selected_variant = str(boundary_config["gate_candidate"]["boussinesq"])
    for role in ("reflection", "production_horizon"):
        role_payloads = [
            payload
            for payload in payloads
            if payload["task"]["spec"]["solver"] == "boussinesq"
            and payload["task"]["spec"]["boundary_role"] == role
        ]
        payloads_by_variant = {
            str(payload["task"]["spec"]["variant"]): payload
            for payload in role_payloads
        }
        if set(payloads_by_variant) != expected_variants:
            raise RuntimeError(f"Incomplete Boussinesq {role} boundary tasks")
        timestamps = np.asarray(
            next(iter(payloads_by_variant.values()))["task"]["spec"][
                "requested_times"
            ],
            dtype=np.float64,
        )
        spec, _initial, reference, metadata, timing = (
            _boussinesq_spectral_packet_bundle(packet_spec, role=role)
        )
        expected_times = (
            candidate_requested_times()
            if role == "production_horizon"
            else np.asarray(timing["requested_times"], dtype=np.float64)
        )
        if not np.array_equal(timestamps, expected_times):
            raise RuntimeError("Boussinesq boundary task timing changed")
        reference_error = _boussinesq_reference_refinement_error(
            spec=spec,
            coarse_reference=reference,
            coarse_metadata=metadata,
            times=timestamps,
        )
        baseline_runtime = max(
            float(payloads_by_variant["zero_gradient_no_sponge"]["row"]["runtime_s"]),
            1.0e-30,
        )
        candidate_rows: dict[str, dict[str, Any]] = {}
        for variant in sorted(payloads_by_variant):
            payload = payloads_by_variant[variant]
            task_row = payload["row"]
            metrics = _boussinesq_reference_boundary_metrics(
                candidate=np.asarray(payload["trajectory"]),
                timestamps=timestamps,
                packet_spec=packet_spec,
                role=role,
                sponge_width=int(task_row["sponge_width"]),
                reference_refinement_error_ratio=reference_error,
                uncertainty_fraction=float(
                    boundary_thresholds[
                        "boussinesq_reference_uncertainty_fraction"
                    ]
                ),
                reflected_energy_ceiling=float(
                    boundary_thresholds["reflected_energy_ratio"]
                ),
                production_error_ceiling=float(
                    boundary_thresholds[
                        "boussinesq_production_interior_energy_ratio"
                    ]
                ),
                precision_floor_safety_factor=float(
                    boundary_thresholds[
                        "boussinesq_precision_floor_safety_factor"
                    ]
                ),
            )
            row = {
                "component": "boundary_sponge",
                "solver": "boussinesq",
                "variant": variant,
                "candidate_status": boundary_config["gate_candidate_status"],
                "boundary_implementation": task_row["boundary"],
                **metrics,
                "sponge_axes": task_row["sponge_axes"],
                "sponge_width": int(task_row["sponge_width"]),
                "sponge_min_factor": float(task_row["sponge_min_factor"]),
                "sponge_profile": task_row["sponge_profile"],
                "whole_domain_sponge": bool(task_row["whole_domain_sponge"]),
                "finite": bool(task_row["finite"]),
                "cg_failure_count": int(task_row["cg_failure_count"]),
                "cg_failure_mode": str(task_row["cg_failure_mode"]),
                "natural_steps": int(task_row["natural_steps"]),
                "runtime_s": float(task_row["runtime_s"]),
                "relative_runtime_cost": float(task_row["runtime_s"])
                / baseline_runtime,
                "max_post_step_cfl": float(task_row["max_post_step_cfl"]),
                "operator": task_row["operator"],
                "decision_role": (
                    "provisional_gate_candidate"
                    if variant == selected_variant
                    else "non_decisional_candidate_diagnostic"
                ),
            }
            candidate_rows[variant] = row
            rows.append(row)

        selected = candidate_rows[selected_variant]
        floor = float(selected["normalized_numerical_floor"])
        common_pass = (
            selected["finite"]
            and selected["cg_failure_count"] == 0
            and selected["measurement_dtype"] == "float64"
            and not selected["whole_domain_sponge"]
            and selected["reference_safe"]
            and selected["reference_adequate"]
        )
        if role == "reflection":
            passed = (
                common_pass
                and selected["measurement_temporally_separated"]
                and selected["reflection_metrics_valid"]
                and selected["reflected_amplitude_ratio"]
                <= float(boundary_thresholds["reflected_amplitude_ratio"]) + floor
                and selected["reflected_energy_ratio"]
                <= float(boundary_thresholds["reflected_energy_ratio"]) + floor
            )
            gate_name = "boundary_boussinesq_reflection_separation"
        else:
            passed = (
                common_pass
                and selected["interior_energy_error_ratio"]
                <= float(
                    boundary_thresholds[
                        "boussinesq_production_interior_energy_ratio"
                    ]
                )
                + floor
                and selected["reflected_energy_ratio"]
                <= float(boundary_thresholds["reflected_energy_ratio"]) + floor
            )
            gate_name = "boundary_boussinesq_production_contamination"
        gates.append(
            {
                "gate": gate_name,
                "category": "blocked_boundary_behavior",
                "passed": bool(passed),
                "candidate": selected_variant,
                "candidate_status": boundary_config["gate_candidate_status"],
                "reference_adequate": bool(selected["reference_adequate"]),
                "reference_safe": bool(selected["reference_safe"]),
                "normalized_numerical_floor": floor,
            }
        )
    return rows, gates


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
        config["thresholds"]["operator"][
            "non_hydro_elapsed_candidate_cfl_relative_l2"
        ]
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
        thresholds["operator"]["non_hydro_elapsed_candidate_cfl_relative_l2"]
    )
    operator_nx = 64
    for name in SOLVERS:
        selected = [
            payload
            for payload in operator_payloads
            if payload["task"]["spec"]["solver"] == name
        ]
        if name == "swe_hydrostatic":
            solver_rows = [dict(payload["row"]) for payload in selected]
            rows.extend(solver_rows)
            clean_payloads = [
                payload
                for payload in selected
                if payload["task"]["spec"]["operator_role"] == "clean_temporal"
            ]
            clean_payloads.sort(
                key=lambda payload: float(payload["task"]["spec"]["cfl"]),
                reverse=True,
            )
            operator_thresholds = thresholds["operator"]
            clean_metrics = _hydro_clean_temporal_metrics(
                [np.asarray(payload["trajectory"]) for payload in clean_payloads],
                minimum_order=float(
                    operator_thresholds["clean_temporal_order_minimum"]
                ),
                precision_floor_safety_factor=float(
                    operator_thresholds["precision_floor_safety_factor"]
                ),
            )
            reference_cfl = float(clean_payloads[-1]["task"]["spec"]["cfl"])
            spatial_by_grid = {
                int(payload["task"]["spec"]["nx"]): np.asarray(
                    payload["trajectory"]
                )
                for payload in selected
                if payload["task"]["spec"]["operator_role"]
                == "spatial_reference"
            }
            spatial_by_grid[operator_nx] = np.asarray(
                clean_payloads[-1]["trajectory"]
            )
            spatial_metrics = _hydro_spatial_control_metrics(
                spatial_by_grid,
                minimum_order=float(
                    operator_thresholds["spatial_order_diagnostic_minimum"]
                ),
                precision_floor_safety_factor=float(
                    operator_thresholds["precision_floor_safety_factor"]
                ),
            )
            reference_ratio = clean_metrics["pairwise_absolute_rms"][-1] / max(
                spatial_metrics["pairwise_absolute_rms"][-1], 1.0e-30
            )
            reference_adequate = reference_ratio <= float(
                operator_thresholds[
                    "reference_to_spatial_error_ratio_maximum"
                ]
            )
            clean_summary = {
                "component": "hydro_clean_temporal_spatial_summary",
                "solver": name,
                "measurement_dtype": "float64",
                "reference_cfl": reference_cfl,
                "temporal": clean_metrics,
                "spatial": spatial_metrics,
                "spatial_control_role": (
                    "measured_discretization_scale_and_asymptotic_diagnostic"
                ),
                "temporal_reference_to_spatial_error_ratio": reference_ratio,
                "reference_adequate": reference_adequate,
                "threshold_rationale": (
                    "first_order_monotone_refinement_with_float64_precision_floor_"
                    "and_temporal_reference_below_measured_spatial_error;_"
                    "localized_packet_spatial_order_is_reported_not_forced"
                ),
            }
            rows.append(clean_summary)
            gates.append(
                {
                    "gate": "hydro_clean_temporal_spatial_consistency",
                    "category": "blocked_operator_semantics",
                    "passed": bool(clean_metrics["passed"])
                    and reference_adequate,
                    "temporal_passed": bool(clean_metrics["passed"]),
                    "spatial_expected_order_met": bool(spatial_metrics["passed"]),
                    "spatial_classification": (
                        "asymptotic_first_order_or_better"
                        if bool(spatial_metrics["passed"])
                        else "localized_packet_not_asymptotic_on_control_grids"
                    ),
                    "reference_adequate": reference_adequate,
                    "temporal_orders": clean_metrics["pairwise_orders"],
                    "spatial_order": spatial_metrics["observed_order"],
                    "reference_to_spatial_error_ratio": reference_ratio,
                }
            )

            pipeline = [
                payload
                for payload in selected
                if payload["task"]["spec"]["operator_role"]
                == "production_pipeline"
            ]
            pipeline.sort(
                key=lambda payload: float(payload["task"]["spec"]["cfl"]),
                reverse=True,
            )
            width = int(pipeline[0]["row"]["sponge_width"])
            sponge_region = _x_sponge_region(operator_nx, width)
            interior_region = ~sponge_region
            boundary_region = _x_sponge_region(operator_nx, 2)
            nonboundary_region = ~boundary_region
            production_to_half = _operator_discrepancy_metrics(
                np.asarray(pipeline[0]["trajectory"]),
                np.asarray(pipeline[1]["trajectory"]),
                sponge_region=sponge_region,
                interior_region=interior_region,
            )
            half_to_quarter = _operator_discrepancy_metrics(
                np.asarray(pipeline[1]["trajectory"]),
                np.asarray(pipeline[2]["trajectory"]),
                sponge_region=sponge_region,
                interior_region=interior_region,
            )
            boundary_contribution = _operator_discrepancy_metrics(
                np.asarray(pipeline[0]["trajectory"]),
                np.asarray(pipeline[1]["trajectory"]),
                sponge_region=boundary_region,
                interior_region=nonboundary_region,
            )
            spatial_scale = float(spatial_metrics["pairwise_absolute_rms"][-1])
            pipeline_absolute = float(
                production_to_half["trajectory_absolute_rms"]
            )
            classification = (
                "within_measured_spatial_discretization"
                if pipeline_absolute <= spatial_scale
                else "production_pipeline_cfl_sensitive"
            )
            pipeline_summary = {
                "component": "hydro_production_pipeline_sensitivity",
                "solver": name,
                "candidate_status": config["operators"]["hydrostatic_gate"][
                    "candidate_pipeline_status"
                ],
                "candidate_boundary": pipeline[0]["row"]["boundary"],
                "candidate_sponge": {
                    "profile": pipeline[0]["row"]["sponge_profile"],
                    "width": width,
                    "minimum_reference_factor": pipeline[0]["row"][
                        "sponge_min_factor"
                    ],
                },
                "production_to_half": production_to_half,
                "half_to_quarter": half_to_quarter,
                "boundary_region_production_to_half": boundary_contribution,
                "spatial_control_absolute_rms": spatial_scale,
                "classification": classification,
                "decision_role": "informational_pipeline_sensitivity",
            }
            rows.append(pipeline_summary)
            pipeline_healthy = all(
                bool(payload["row"]["finite"])
                and payload["row"]["measurement_dtype"] == "float64"
                and not bool(payload["row"]["whole_domain_sponge"])
                for payload in pipeline
            )
            gates.append(
                {
                    "gate": "hydro_production_pipeline_sensitivity",
                    "category": "informational",
                    "passed": pipeline_healthy,
                    "classification": classification,
                    "decision_role": "health_only_not_temporal_order",
                }
            )
            continue

        operator_sponge = _x_sponge_region(
            operator_nx, max(1, operator_nx // 8)
        )
        operator_interior = ~operator_sponge
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
        packet_spec = _resolved_boundary_packet_spec(boundary_config, name)
        if name == "boussinesq":
            bouss_rows, bouss_gates = _aggregate_boussinesq_boundary_payloads(
                boundary_payloads,
                packet_spec=packet_spec,
                boundary_config=boundary_config,
                boundary_thresholds=thresholds["boundary"],
            )
            rows.extend(bouss_rows)
            gates.extend(bouss_gates)
            continue
        timing = _boundary_timing(name, packet_spec)
        provisional_width = max(1, int(round(3 * boundary_nx / 16)))
        _validate_boundary_packet_spec(
            packet_spec,
            nx=boundary_nx,
            sponge_width=provisional_width,
        )
        initial_conditions = _boundary_initial_conditions(
            name, nx=boundary_nx, ny=boundary_ny, spec=packet_spec
        )
        depth = float(packet_spec.get("depth", 1.0))
        bathymetry = -depth * np.ones(
            (boundary_nx, boundary_ny), dtype=np.float64
        )
        selected = [
            payload
            for payload in boundary_payloads
            if payload["task"]["spec"]["solver"] == name
        ]
        payloads_by_variant = {
            str(payload["task"]["spec"]["variant"]): payload
            for payload in selected
        }
        expected_variants = {
            variant
            for variant, candidate in boundary_config["candidates"].items()
            if not (
                bool(candidate.get("swe_only", False)) and name == "boussinesq"
            )
            and not (
                bool(candidate.get("boussinesq_only", False))
                and name != "boussinesq"
            )
        }
        if set(payloads_by_variant) != expected_variants:
            raise RuntimeError(f"Incomplete boundary tasks for {name}")
        baseline_payload = payloads_by_variant["zero_gradient_no_sponge"]
        baseline = np.asarray(baseline_payload["trajectory"])
        baseline_runtime = max(
            float(baseline_payload["row"]["runtime_s"]), 1.0e-30
        )
        candidate_rows: dict[str, dict[str, Any]] = {}
        for variant in sorted(payloads_by_variant):
            payload = payloads_by_variant[variant]
            metrics = _boundary_metrics(
                solver_name=name,
                baseline=baseline,
                candidate=np.asarray(payload["trajectory"]),
                initial_conditions=initial_conditions,
                bathymetry=bathymetry,
                spec=packet_spec,
                timing=timing,
                timestamps=np.asarray(spec_time := payload["task"]["spec"]["requested_times"], dtype=np.float64),
            )
            if list(spec_time) != list(timing["requested_times"]):
                raise RuntimeError("boundary task timing changed after preregistration")
            task_row = payload["row"]
            boundary_row = {
                "component": "boundary_sponge",
                "solver": name,
                "variant": variant,
                "candidate_status": boundary_config["gate_candidate_status"],
                "boundary_implementation": task_row["boundary"],
                **metrics,
                "incident_window": _json_safe(packet_spec["incident_window"]),
                "reflected_window": _json_safe(packet_spec["reflected_window"]),
                "interior_window": _json_safe(packet_spec["interior_window"]),
                "sponge_axes": task_row["sponge_axes"],
                "sponge_width": task_row["sponge_width"],
                "sponge_min_factor": task_row["sponge_min_factor"],
                "sponge_profile": task_row["sponge_profile"],
                "whole_domain_sponge": bool(task_row["whole_domain_sponge"]),
                "finite": bool(task_row["finite"]),
                "cg_failure_count": int(task_row["cg_failure_count"]),
                "natural_steps": int(task_row["natural_steps"]),
                "runtime_s": float(task_row["runtime_s"]),
                "relative_runtime_cost": float(task_row["runtime_s"])
                / baseline_runtime,
                "max_post_step_cfl": float(task_row["max_post_step_cfl"]),
                "measurement_dtype": task_row["measurement_dtype"],
                "operator": task_row["operator"],
                "decision_role": (
                    "provisional_gate_candidate"
                    if variant == boundary_config["gate_candidate"][name]
                    else "non_decisional_candidate_diagnostic"
                ),
            }
            candidate_rows[variant] = boundary_row
            rows.append(boundary_row)

        selected_variant = str(boundary_config["gate_candidate"][name])
        boundary_row = candidate_rows[selected_variant]
        boundary_thresholds = thresholds["boundary"]
        gates.append(
            {
                "gate": f"boundary_sponge_{name}",
                "category": "blocked_boundary_behavior",
                "passed": boundary_row["finite"]
                and boundary_row["cg_failure_count"] == 0
                and boundary_row["measurement_dtype"] == "float64"
                and not boundary_row["whole_domain_sponge"]
                and bool(boundary_row["measurement_temporally_separated"])
                and bool(boundary_row["packet_exit_achieved"])
                and boundary_row["reflected_amplitude_ratio"]
                <= float(boundary_thresholds["reflected_amplitude_ratio"])
                and boundary_row["reflected_energy_ratio"]
                <= float(boundary_thresholds["reflected_energy_ratio"])
                and boundary_row["interior_relative_l2"]
                <= float(boundary_thresholds["interior_relative_l2"]),
                "candidate": selected_variant,
                "candidate_status": boundary_config["gate_candidate_status"],
                "measurement_temporally_separated": bool(
                    boundary_row["measurement_temporally_separated"]
                ),
                "packet_exit_achieved": bool(boundary_row["packet_exit_achieved"]),
                "whole_domain_sponge": boundary_row["whole_domain_sponge"],
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
    boussinesq_exposure = [
        row for row in canary_rows if row["solver"] == "boussinesq"
    ]
    if boussinesq_exposure:
        exposure_variants = sorted(
            boussinesq_exposure[0]["boundary_candidate_exposure"]
        )
        candidate_exposure_summary = {
            variant: {
                "sponge_overlap_count": sum(
                    bool(row["boundary_candidate_exposure"][variant][
                        "significant_source_overlaps_sponge"
                    ])
                    for row in boussinesq_exposure
                ),
                "maximum_initial_sponge_energy_fraction": max(
                    float(row["boundary_candidate_exposure"][variant][
                        "initial_sponge_energy_fraction"
                    ])
                    for row in boussinesq_exposure
                ),
            }
            for variant in exposure_variants
        }
        exposure_row = {
            "component": "boussinesq_h0_boundary_exposure",
            "decision_role": "production_contamination_gate_input",
            "canary_count": len(boussinesq_exposure),
            "sponge_overlap_count": sum(
                bool(row["significant_source_overlaps_sponge"])
                for row in boussinesq_exposure
            ),
            "conservative_reachable_count": sum(
                bool(row["conservative_boundary_reachable"])
                for row in boussinesq_exposure
            ),
            "maximum_initial_sponge_energy_fraction": max(
                float(row["initial_sponge_energy_fraction"])
                for row in boussinesq_exposure
            ),
            "minimum_significant_source_distance_to_boundary": min(
                float(row["significant_source_distance_to_boundary"])
                for row in boussinesq_exposure
            ),
            "maximum_conservative_long_wave_reach": max(
                float(row["conservative_long_wave_reach"])
                for row in boussinesq_exposure
            ),
            "candidate_exposure_summary": candidate_exposure_summary,
        }
        rows.append(exposure_row)
        exposure_limit = float(
            thresholds["boundary"]["boussinesq_h0_initial_sponge_energy_ratio"]
        )
        production_gate = next(
            gate
            for gate in gates
            if gate["gate"] == "boundary_boussinesq_production_contamination"
        )
        selected_exposure_variant = str(
            boundary_config["gate_candidate"]["boussinesq"]
        )
        selected_initial_fraction = float(
            candidate_exposure_summary[selected_exposure_variant][
                "maximum_initial_sponge_energy_fraction"
            ]
        )
        production_gate["h0_initial_sponge_energy_ratio"] = float(
            selected_initial_fraction
        )
        production_gate["h0_initial_sponge_energy_ratio_limit"] = exposure_limit
        production_gate["h0_sponge_overlap_count"] = int(
            candidate_exposure_summary[selected_exposure_variant][
                "sponge_overlap_count"
            ]
        )
        production_gate["passed"] = bool(production_gate["passed"]) and (
            selected_initial_fraction <= exposure_limit
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
        or row["gate"] == "hydro_clean_temporal_spatial_consistency"
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
    if (execution / "detailed_rows.csv").read_bytes() != _csv_text(
        public_rows
    ).encode("utf-8"):
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
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
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
                progress_callback=progress_callback,
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
            _write_json(staging / "operational_provenance.json", operational) # i honestly dunno what this even is lmao
            (staging / "REPORT.md").write_text(
                _report_text(str(decision["decision"])),
                encoding="utf-8",
            )

    _write_checksums(staging)
    os.replace(staging, execution)
    _write_checksums(contract_root)
    return execution
