"""Immutable broad numerical validation for the corrected R6 benchmark.

The archived H0, Level A, H1, and H2 artifacts target an earlier 96-cell,
0.175-time candidate.  This module never imports or repoints those runners.
It freezes a separate R6 campaign against the 384 -> 128 -> 192 -> 64 data
contract and the 50 requested times from 8.4 through 420.0.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from scripts.run_production_contract_validation import run_validation
from src.data_gen.common_time_v2 import code_state, hash_array, stable_hash_payload
from src.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
    _simulate_one_local,
)
from src.evaluation.boussinesq_boundary import discrete_dispersion, directional_rate
from src.evaluation.common_time_v2_level_a import validate_checksums
from src.utils.hashing import sha256_file


SCHEMA_ID = "tsunami-surrogate.r6-broad-validation.v1"
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
SOLVER_DIRS = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class R6ValidationError(RuntimeError):
    """Raised when an R6 validation gate cannot be established."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise R6ValidationError(f"Missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise R6ValidationError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise R6ValidationError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise R6ValidationError(f"Missing JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R6ValidationError(f"Malformed JSONL at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise R6ValidationError(f"Expected object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise R6ValidationError("R6 artifacts may not serialize non-finite floats")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _archive_workspace(workspace: Path, archive: Path) -> None:
    subprocess.run(
        ["tar", "--zstd", "-cf", str(archive), "-C", str(workspace.parent), workspace.name],
        check=True,
    )


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _stage_record(
    *,
    stage_id: str,
    status: str,
    artifact_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    try:
        display_path = str(artifact_path.relative_to(repo_root))
    except ValueError:
        display_path = str(artifact_path)
    return {
        "id": stage_id,
        "status": status,
        "decision": status,
        "artifact_hash": _sha256(artifact_path),
        "artifact_path": display_path,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise R6ValidationError(f"Expected YAML mapping: {path}")
    return payload


def _requested_times(spec: Mapping[str, Any]) -> np.ndarray:
    start = float(spec["start"])
    step = float(spec["step"])
    count = int(spec["count"])
    horizon = float(spec["horizon"])
    times = start + step * np.arange(count, dtype=np.float64)
    times[-1] = horizon
    if (
        times.shape != (50,)
        or times[0] != 8.4
        or times[-1] != 420.0
        or not np.all(np.diff(times) > 0.0)
    ):
        raise R6ValidationError("R6 requested-time construction mismatch")
    return times


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != SCHEMA_ID:
        raise R6ValidationError("R6 broad validation schema mismatch")
    if config.get("artifact_kind") != "r6-broad-numerical-validation-candidate":
        raise R6ValidationError("R6 broad validation artifact kind mismatch")
    production = config.get("corrected_production")
    if not isinstance(production, Mapping):
        raise R6ValidationError("R6 production contract is missing")
    expected_shapes = {
        "master_shape": [384, 384],
        "solver_input_shape": [128, 128],
        "computational_shape": [192, 192],
        "publication_shape": [64, 64],
    }
    for key, expected in expected_shapes.items():
        if list(production.get(key, [])) != expected:
            raise R6ValidationError(f"R6 {key} mismatch")
    if int(production.get("buffer_cells", -1)) != 32:
        raise R6ValidationError("R6 buffer-cell contract mismatch")
    if float(production.get("cell_size", 0.0)) != 18.75:
        raise R6ValidationError("R6 cell-size contract mismatch")
    _requested_times(production["requested_times"])
    profiles = production.get("solver_profiles")
    if profiles != {
        "swe_hydrostatic": {"cfl": 0.1125, "boundary": "radiation"},
        "swe_muscl_hr": {"cfl": 0.225, "boundary": "radiation"},
        "boussinesq": {"cfl": 0.35, "boundary": "open"},
    }:
        raise R6ValidationError("R6 solver-profile contract mismatch")
    if config.get("execution", {}).get("workers") != 8:
        raise R6ValidationError("R6 validation worker policy must remain eight")
    expected_threads = {key: "1" for key in THREAD_ENV_KEYS}
    if config.get("execution", {}).get("thread_environment") != expected_threads:
        raise R6ValidationError("R6 validation thread policy mismatch")


def _configure_threads(config: Mapping[str, Any]) -> None:
    expected = config["execution"]["thread_environment"]
    invalid = {
        key: os.environ.get(key)
        for key, value in expected.items()
        if os.environ.get(key) not in (None, value)
    }
    if invalid:
        details = ", ".join(f"{key}={value}" for key, value in sorted(invalid.items()))
        raise R6ValidationError(
            f"R6 validation requires single-thread numerical backends; found {details}"
        )
    for key, value in expected.items():
        os.environ.setdefault(str(key), str(value))


def _contract(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    production = config["corrected_production"]
    contract_path = repo_root / str(production["contract_path"])
    contract = _load_yaml(contract_path)
    if contract["scientific_scope"]["contract_hash"] != production["contract_hash"]:
        raise R6ValidationError("R6 suite contract hash mismatch")
    scope = contract["scientific_scope"]
    if (
        scope["computational_domain"]["solver_shape"] != [192, 192]
        or scope["computational_domain"]["publication_shape"] != [64, 64]
        or int(scope["computational_domain"]["buffer_cells"]) != 32
    ):
        raise R6ValidationError("R6 suite geometry mismatch")
    if not np.array_equal(_requested_times(production["requested_times"]), _contract_times(contract)):
        raise R6ValidationError("R6 suite requested times mismatch")
    return contract


def _contract_times(contract: Mapping[str, Any]) -> np.ndarray:
    return _requested_times(contract["scientific_scope"]["requested_times"])


def _verify_completed_geoclaw(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["external_comparator"]
    root = repo_root / str(spec["summary"])
    summary = _read_json(root)
    if summary.get("schema_id") != "tsunami-surrogate.corrected-production-geoclaw-swe.v1":
        raise R6ValidationError("R6 GeoClaw summary schema mismatch")
    checks = {
        "status": spec["expected_status"],
        "bundle_hash": spec["expected_bundle_hash"],
        "case_count": int(spec["expected_case_count"]),
        "comparison_count": int(spec["expected_comparison_count"]),
        "computational_shape": [192, 192],
        "publication_shape": [64, 64],
        "requested_time_count": 50,
        "requested_time_horizon": 420.0,
    }
    for key, expected in checks.items():
        if summary.get(key) != expected:
            raise R6ValidationError(
                f"Completed R6 GeoClaw {key} mismatch: {summary.get(key)!r} != {expected!r}"
            )
    manifest = repo_root / str(spec["checksum_manifest"])
    validate_checksums(manifest.parent)
    return {
        "summary": str(root.relative_to(repo_root)),
        "summary_sha256": _sha256(root),
        "checksum_manifest": str(manifest.relative_to(repo_root)),
        "checksum_manifest_sha256": _sha256(manifest),
        "bundle_hash": summary["bundle_hash"],
        "case_count": summary["case_count"],
        "comparison_count": summary["comparison_count"],
        "external_revisions": dict(summary["external_revisions"]),
    }


def _reuse_production_validation(
    *, repo_root: Path, summary_path: Path, workspace: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = _read_json(summary_path)
    if (
        summary.get("evaluation_type") != "current_production_contract_validation"
        or summary.get("status") != "passed"
        or summary.get("contract_hash") != contract["scientific_scope"]["contract_hash"]
    ):
        raise R6ValidationError("Reused production-validation summary is not a passed R6 audit")
    source_root = summary_path.parent
    canary_path = source_root / str(summary.get("canaries", "canary_results.json"))
    if not canary_path.is_file():
        raise R6ValidationError(f"Reused production-validation canaries are missing: {canary_path}")
    destination = workspace / "h0_full_dataset_audit"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(summary_path, destination / "summary.json")
    shutil.copy2(canary_path, destination / "canary_results.json")
    return {
        "status": "passed",
        "evaluation_type": summary["evaluation_type"],
        "summary": "h0_full_dataset_audit/summary.json",
        "summary_sha256": _sha256(destination / "summary.json"),
        "source_summary": str(summary_path),
        "deep_payload_audit": bool(summary.get("deep_payload_audit", False)),
    }


def _generation_config(repo_root: Path, split: str) -> tuple[dict[str, Any], Path]:
    paths = {
        "train": repo_root / "configs/data/dataset.yaml",
        "eval": repo_root / "configs/data/dataset_eval.yaml",
        "test": repo_root / "configs/data/dataset_test.yaml",
    }
    path = paths[split]
    return _load_yaml(path), path


def _solver_cfg(cfg: Mapping[str, Any], solver_name: str, *, cfl: float | None = None) -> dict[str, Any]:
    resolved = dict(cfg["solver"])
    resolved.update(dict(cfg["solver_profiles"][solver_name]))
    if cfl is not None:
        resolved["cfl"] = float(cfl)
    return resolved


def _make_solver(name: str, cfg: Mapping[str, Any]) -> Any:
    if name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(dict(cfg))
    if name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(dict(cfg))
    if name == "boussinesq":
        return _make_boussinesq_solver_from_cfg(dict(cfg))
    raise R6ValidationError(f"Unknown R6 solver: {name}")


def _set_initial_state(
    solver: Any,
    name: str,
    bathymetry: np.ndarray,
    eta0: np.ndarray,
) -> None:
    solver.set_bathymetry(bathymetry)
    if name == "boussinesq":
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
    else:
        solver.set_initial_condition(
            np.maximum(-bathymetry + eta0, 0.0),
            hu0=np.zeros_like(eta0),
            hv0=np.zeros_like(eta0),
        )


def _trajectory_eta(states: np.ndarray, name: str, bathymetry: np.ndarray) -> np.ndarray:
    if name == "boussinesq":
        return np.asarray(states[:, 0], dtype=np.float64)
    return np.asarray(states[:, 0], dtype=np.float64) + bathymetry[None, ...]


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    reference = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(reference.ravel()), 1.0e-30))


def _load_record_arrays(
    repo_root: Path, record: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(repo_root / str(record["bathymetry_cache_path"]), allow_pickle=False) as payload:
        bathymetry = np.asarray(payload["solver_bathymetry"], dtype=np.float32)
    with np.load(repo_root / str(record["source_cache_path"]), allow_pickle=False) as payload:
        source = np.asarray(payload["solver_source_field"], dtype=np.float32)
        strength = float(np.asarray(payload["source_strength"]).reshape(-1)[0])
    if bathymetry.shape != (128, 128) or source.shape != (128, 128):
        raise R6ValidationError(f"R6 cached solver inputs are not 128x128: {record['qualified_id']}")
    return bathymetry, source, strength


def _buffered_arrays(
    repo_root: Path, record: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    bathymetry, source, strength = _load_record_arrays(repo_root, record)
    prepared = _prepare_buffered_domain(
        bathymetry,
        source,
        strength,
        sea_level_offset=0.0,
        config=BufferedDomainConfig(
            enabled=True,
            buffer_cells=32,
            source_taper_cells=16,
            bathymetry_extension="edge",
            output_crop="central",
        ),
        source_type=str(record["source_type"]),
        source_already_tapered=True,
    )
    if prepared["solver_bathymetry"].shape != (192, 192):
        raise R6ValidationError("R6 buffered computation is not 192x192")
    return prepared, bathymetry, source, strength


def _select_family_records(
    repo_root: Path,
    *,
    split: str,
    rank: int,
) -> list[dict[str, Any]]:
    manifest = repo_root / f"data/{split}/synthetic/scenario_manifest.jsonl"
    rows = _read_jsonl(manifest)
    selected: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cell = (str(row["bathymetry_type"]), str(row["source_type"]))
        by_cell.setdefault(cell, []).append(row)
    expected_cells = 5 * 6
    if len(by_cell) != expected_cells:
        raise R6ValidationError(f"Expected {expected_cells} R6 family cells, found {len(by_cell)}")
    for cell in sorted(by_cell):
        ordered = sorted(by_cell[cell], key=lambda item: int(item["sample_index"]))
        if len(ordered) < rank:
            raise R6ValidationError(f"R6 family cell has fewer than {rank} records: {cell}")
        row = dict(ordered[rank - 1])
        row["qualified_id"] = f"{split}:{row['scenario_id']}"
        selected.append(row)
    return selected


def _run_requested_rollout(
    *,
    name: str,
    cfg: Mapping[str, Any],
    prepared: Mapping[str, Any],
    times: np.ndarray,
    max_natural_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    solver = _make_solver(name, cfg)
    bathymetry = np.asarray(prepared["solver_bathymetry"], dtype=np.float64)
    eta0 = np.asarray(prepared["solver_eta0"], dtype=np.float64)
    _set_initial_state(solver, name, bathymetry, eta0)
    states, emitted, dt_history, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=float(cfg["cfl"]),
        include_initial_state=False,
        requested_times=times,
        max_natural_steps=max_natural_steps,
        collect_natural_step_health=True,
        requested_state_dtype=np.float64,
    )
    if not np.array_equal(np.asarray(emitted, dtype=np.float64), times):
        raise R6ValidationError(f"R6 requested times changed for {name}")
    if not np.isfinite(states).all():
        raise R6ValidationError(f"R6 requested rollout is non-finite for {name}")
    post_cfl = np.asarray(diagnostics.get("post_step_cfl", []), dtype=np.float64)
    finite = np.asarray(diagnostics.get("finite_state_flag", []), dtype=bool)
    if post_cfl.size == 0 or not np.isfinite(post_cfl).all() or not bool(finite.all()):
        raise R6ValidationError(f"R6 natural-step health is incomplete for {name}")
    if float(np.max(post_cfl)) > float(cfg["cfl"]) * 1.01:
        raise R6ValidationError(f"R6 CFL exceeds the configured envelope for {name}")
    cg_failed = np.asarray(diagnostics.get("cg_failed_count", []), dtype=np.int64)
    if name == "boussinesq" and (cg_failed.size == 0 or int(np.sum(cg_failed)) != 0):
        raise R6ValidationError("R6 Boussinesq CG health failed")
    eta = _trajectory_eta(states, name, bathymetry)
    crop = prepared["crop"]
    publication = eta[:, crop[0], crop[1]].reshape(50, 64, 2, 64, 2).mean(axis=(2, 4))
    details = {
        "trajectory_hash": hash_array(eta),
        "publication_hash": hash_array(publication),
        "natural_step_count": int(dt_history.size),
        "max_post_step_cfl": float(np.max(post_cfl)),
        "requested_time_count": int(emitted.size),
        "final_natural_timestamp": float(
            np.asarray(diagnostics["final_natural_timestamp"], dtype=np.float64)[0]
        ),
        "boussinesq_cg_failed_count": int(np.sum(cg_failed)),
        "operator": solver.get_operator_diagnostics(),
    }
    return publication, details


def _phase_speed_check(
    *,
    name: str,
    grid: int,
    physical_length: float,
    depth: float,
    amplitude: float,
    mode: int,
    cfl: float,
) -> dict[str, Any]:
    ny = 4
    dx = physical_length / float(grid)
    dy = physical_length / float(grid)
    x = np.arange(grid, dtype=np.float64)[:, None] * dx
    wavenumber = 2.0 * math.pi * float(mode) / physical_length
    bathymetry = -depth * np.ones((grid, ny), dtype=np.float64)
    eta0 = amplitude * np.cos(wavenumber * x) * np.ones((1, ny), dtype=np.float64)
    cfg: dict[str, Any] = {
        "nx": grid,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "dt": min(0.02, 0.2 * dx / math.sqrt(9.81 * depth)),
        "g": 9.81,
        "cfl": cfl,
        "boundary": "periodic",
        "use_sponge": False,
        "sponge_width": 0,
        "sponge_min_factor": 0.8,
        "sponge_axes": "xy",
        "sponge_profile": "cosine",
        "sponge_time_mode": "legacy_per_step",
        "dry_tolerance": 1.0e-6,
        "max_velocity": 30.0,
        "alpha": 1.0 / 3.0,
        "min_depth": 1.0e-4,
        "depth_scale": 1.0,
        "mode": "linear_constant_depth",
        "filter_strength": 0.0,
        "filter_time_mode": "disabled",
        "linear_solver_tol": 1.0e-10,
        "linear_solver_abs_tol": 0.0,
        "linear_solver_max_iter": 750,
        "linear_solver_preconditioner": "sparse_lu",
        "cg_failure_mode": "strict_v2",
        "check_finite": True,
    }
    solver = _make_solver(name, cfg)
    _set_initial_state(solver, name, bathymetry, eta0)
    if name == "boussinesq":
        expected_omega = float(
            discrete_dispersion(
                np.asarray([wavenumber]), dx=dx, depth=depth, alpha=1.0 / 3.0
            )[0][0]
        )
        solver.set_initial_condition(
            eta0,
            eta_t0=directional_rate(
                eta0,
                dx=dx,
                depth=depth,
                direction="left",
                alpha=1.0 / 3.0,
            ),
        )
    else:
        expected_omega = math.sqrt(9.81 * depth) * wavenumber
        speed = math.sqrt(9.81 * depth)
        solver.set_initial_condition(
            depth + eta0,
            hu0=speed * eta0,
            hv0=np.zeros_like(eta0),
        )
    period = 2.0 * math.pi / expected_omega
    duration = min(max(0.5 * period, 8.4), 84.0)
    times = np.linspace(duration / 10.0, duration, 10, dtype=np.float64)
    states, emitted, _dt, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=cfl,
        include_initial_state=False,
        requested_times=times,
        max_natural_steps=20_000,
        collect_natural_step_health=True,
        requested_state_dtype=np.float64,
    )
    eta = _trajectory_eta(states, name, bathymetry)
    basis = np.exp(-1j * wavenumber * x)
    coefficient = np.asarray([np.mean(frame * basis) for frame in eta])
    measured_omega = float(
        abs(np.polyfit(emitted, np.unwrap(np.angle(coefficient)), 1)[0])
    )
    amplitude_drift = abs(float(abs(coefficient[-1]) / max(abs(coefficient[0]), 1.0e-30)) - 1.0)
    cg_failed = np.asarray(diagnostics.get("cg_failed_count", []), dtype=np.int64)
    return {
        "solver": name,
        "grid": grid,
        "cell_size": dx,
        "mode": mode,
        "expected_omega": expected_omega,
        "measured_omega": measured_omega,
        "phase_speed_relative_error": abs(measured_omega - expected_omega) / expected_omega,
        "amplitude_drift": amplitude_drift,
        "cg_failed_count": int(np.sum(cg_failed)),
        "finite": bool(np.isfinite(states).all()),
    }


def _lake_at_rest_check(
    *, name: str, cfg: Mapping[str, Any], prepared: Mapping[str, Any], steps: int
) -> dict[str, Any]:
    solver_cfg = dict(cfg)
    solver_cfg.update({"boundary": "reflective", "use_sponge": False})
    solver = _make_solver(name, solver_cfg)
    bathymetry = np.asarray(prepared["solver_bathymetry"], dtype=np.float64)
    zero = np.zeros_like(bathymetry)
    _set_initial_state(solver, name, bathymetry, zero)
    initial = np.asarray(solver.compute_free_surface(), dtype=np.float64)
    max_drift = 0.0
    max_rate = 0.0
    for _ in range(steps):
        dt = float(solver.suggest_dt(target_cfl=float(solver_cfg["cfl"])))
        solver.step(dt=dt, auto_dt=False)
        max_drift = max(max_drift, float(np.max(np.abs(solver.compute_free_surface() - initial))))
        if name == "boussinesq":
            max_rate = max(max_rate, float(np.max(np.abs(solver.eta_t))))
        else:
            u, v = solver.compute_velocity()
            max_rate = max(max_rate, float(np.max(np.abs(u))), float(np.max(np.abs(v))))
    return {
        "solver": name,
        "max_eta_drift": max_drift,
        "max_velocity_or_eta_t": max_rate,
        "finite": bool(np.isfinite(solver.get_state()).all()),
    }


def _conservation_check(
    *, name: str, cfg: Mapping[str, Any], prepared: Mapping[str, Any], steps: int
) -> dict[str, Any]:
    if name == "boussinesq":
        return {"solver": name, "not_applicable": True}
    solver_cfg = dict(cfg)
    solver_cfg.update({"boundary": "reflective", "use_sponge": False})
    solver = _make_solver(name, solver_cfg)
    bathymetry = np.asarray(prepared["solver_bathymetry"], dtype=np.float64)
    eta0 = np.asarray(prepared["solver_eta0"], dtype=np.float64)
    _set_initial_state(solver, name, bathymetry, eta0)
    initial_mass = float(np.sum(solver.h, dtype=np.float64))
    for _ in range(steps):
        dt = float(solver.suggest_dt(target_cfl=float(solver_cfg["cfl"])))
        solver.step(dt=dt, auto_dt=False)
    final_mass = float(np.sum(solver.h, dtype=np.float64))
    return {
        "solver": name,
        "mass_relative_change": abs(final_mass - initial_mass) / max(abs(initial_mass), 1.0e-30),
        "finite": bool(np.isfinite(solver.get_state()).all()),
    }


def _run_level_a(
    *, repo_root: Path, config: Mapping[str, Any], workspace: Path, times: np.ndarray
) -> dict[str, Any]:
    spec = config["level_a"]
    production = config["corrected_production"]
    profiles = production["solver_profiles"]
    analytical_rows: list[dict[str, Any]] = []
    for name in SOLVERS:
        for grid in spec["analytical"]["grids"]:
            row = _phase_speed_check(
                name=name,
                grid=int(grid),
                physical_length=float(spec["analytical"]["physical_length"]),
                depth=float(spec["analytical"]["depth"]),
                amplitude=float(spec["analytical"]["amplitude"]),
                mode=int(spec["analytical"]["mode"]),
                cfl=float(profiles[name]["cfl"]),
            )
            if not row["finite"] or int(row["cg_failed_count"]) != 0:
                raise R6ValidationError(f"R6 analytical health failed: {name}/{grid}")
            if row["phase_speed_relative_error"] > float(spec["analytical"]["phase_speed_relative_error"][name]):
                raise R6ValidationError(f"R6 analytical phase-speed gate failed: {name}/{grid}")
            if row["amplitude_drift"] > float(spec["analytical"]["amplitude_drift"][name]):
                raise R6ValidationError(f"R6 analytical amplitude gate failed: {name}/{grid}")
            analytical_rows.append(row)
    for name in SOLVERS:
        rows = [row for row in analytical_rows if row["solver"] == name]
        coarse, fine = sorted(rows, key=lambda row: int(row["grid"]))
        if fine["phase_speed_relative_error"] > coarse["phase_speed_relative_error"] * float(spec["analytical"]["spatial_refinement_allowance"]):
            raise R6ValidationError(f"R6 analytical refinement gate failed: {name}")

    test_cfg, _ = _generation_config(repo_root, "test")
    record = _read_jsonl(repo_root / "data/test/synthetic/scenario_manifest.jsonl")[
        int(spec["invariants"]["test_sample_index"]) - 1
    ]
    record = dict(record)
    record["qualified_id"] = f"test:{record['scenario_id']}"
    prepared, _bathy, _source, _strength = _buffered_arrays(repo_root, record)
    invariants: list[dict[str, Any]] = []
    for name in SOLVERS:
        cfg = _solver_cfg(test_cfg, name)
        rest = _lake_at_rest_check(
            name=name, cfg=cfg, prepared=prepared, steps=int(spec["invariants"]["steps"])
        )
        if (
            not rest["finite"]
            or rest["max_eta_drift"] > float(spec["invariants"]["lake_at_rest_eta_drift"])
            or rest["max_velocity_or_eta_t"] > float(spec["invariants"]["lake_at_rest_velocity_or_eta_t"])
        ):
            raise R6ValidationError(f"R6 lake-at-rest gate failed: {name}")
        conservation = _conservation_check(
            name=name, cfg=cfg, prepared=prepared, steps=int(spec["invariants"]["steps"])
        )
        if (
            not conservation.get("not_applicable", False)
            and (
                not conservation["finite"]
                or conservation["mass_relative_change"] > float(spec["invariants"]["swe_mass_relative_change"])
            )
        ):
            raise R6ValidationError(f"R6 conservation gate failed: {name}")
        invariants.append({"lake_at_rest": rest, "conservation": conservation})
    result = {
        "stage": "level_a",
        "status": "passed",
        "requested_times": times.tolist(),
        "analytical": analytical_rows,
        "invariants": invariants,
    }
    _write_json(workspace / "level_a.json", result)
    return result


def _run_h1(
    *, repo_root: Path, config: Mapping[str, Any], workspace: Path, times: np.ndarray
) -> dict[str, Any]:
    spec = config["h1_integrated_health"]
    records = _select_family_records(repo_root, split=str(spec["split"]), rank=int(spec["count_per_family_cell"]))
    cfg, _ = _generation_config(repo_root, str(spec["split"]))
    rows: list[dict[str, Any]] = []
    replay_records = records[: int(spec["replay_case_count"])]
    for index, record in enumerate(records, 1):
        prepared, _bathy, _source, _strength = _buffered_arrays(repo_root, record)
        entry: dict[str, Any] = {
            "qualified_id": record["qualified_id"],
            "sample_index": int(record["sample_index"]),
            "bathymetry_type": record["bathymetry_type"],
            "source_type": record["source_type"],
            "solvers": [],
        }
        for name in SOLVERS:
            trajectory, details = _run_requested_rollout(
                name=name,
                cfg=_solver_cfg(cfg, name),
                prepared=prepared,
                times=times,
                max_natural_steps=int(spec["max_natural_steps"]),
            )
            entry["solvers"].append({"solver": name, **details})
            if record in replay_records:
                repeat, repeat_details = _run_requested_rollout(
                    name=name,
                    cfg=_solver_cfg(cfg, name),
                    prepared=prepared,
                    times=times,
                    max_natural_steps=int(spec["max_natural_steps"]),
                )
                max_abs = float(np.max(np.abs(trajectory - repeat)))
                rel_l2 = _relative_l2(trajectory, repeat)
                if max_abs > float(spec["replay_max_abs"]) or rel_l2 > float(spec["replay_relative_l2"]):
                    raise R6ValidationError(f"R6 deterministic replay failed: {record['qualified_id']}/{name}")
                entry["solvers"][-1]["replay"] = {
                    "max_abs": max_abs,
                    "relative_l2": rel_l2,
                    "trajectory_hash": repeat_details["publication_hash"],
                }
        rows.append(entry)
        print(f"[r6-validation:h1] {index}/{len(records)} {record['qualified_id']}", flush=True)
    result = {
        "stage": "h1",
        "status": "passed",
        "case_count": len(rows),
        "solver_run_count": len(rows) * len(SOLVERS) + len(replay_records) * len(SOLVERS),
        "rows": rows,
    }
    _write_json(workspace / "h1.json", result)
    return result


def _run_h2(
    *, repo_root: Path, config: Mapping[str, Any], workspace: Path, times: np.ndarray
) -> dict[str, Any]:
    spec = config["h2_sensitivity"]
    records = _select_family_records(
        repo_root,
        split=str(spec["split"]),
        rank=int(spec["selection_rank_per_family_cell"]),
    )
    cfg, _ = _generation_config(repo_root, str(spec["split"]))
    factor = float(spec["reference_cfl_factor"])
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        prepared, _bathy, _source, _strength = _buffered_arrays(repo_root, record)
        entry: dict[str, Any] = {
            "qualified_id": record["qualified_id"],
            "sample_index": int(record["sample_index"]),
            "bathymetry_type": record["bathymetry_type"],
            "source_type": record["source_type"],
            "solvers": [],
        }
        for name in SOLVERS:
            production_cfg = _solver_cfg(cfg, name)
            reference_cfg = _solver_cfg(cfg, name, cfl=float(production_cfg["cfl"]) * factor)
            production, production_details = _run_requested_rollout(
                name=name,
                cfg=production_cfg,
                prepared=prepared,
                times=times,
                max_natural_steps=int(spec["max_natural_steps"]),
            )
            reference, reference_details = _run_requested_rollout(
                name=name,
                cfg=reference_cfg,
                prepared=prepared,
                times=times,
                max_natural_steps=int(spec["max_natural_steps"]),
            )
            rel_l2 = _relative_l2(production, reference)
            if rel_l2 > float(spec["trajectory_relative_l2_max"][name]):
                raise R6ValidationError(f"R6 H2 sensitivity gate failed: {record['qualified_id']}/{name}")
            entry["solvers"].append(
                {
                    "solver": name,
                    "production_cfl": float(production_cfg["cfl"]),
                    "reference_cfl": float(reference_cfg["cfl"]),
                    "trajectory_relative_l2": rel_l2,
                    "production": production_details,
                    "reference": reference_details,
                }
            )
        rows.append(entry)
        print(f"[r6-validation:h2] {index}/{len(records)} {record['qualified_id']}", flush=True)
    result = {
        "stage": "h2",
        "status": "passed",
        "case_count": len(rows),
        "solver_pair_count": len(rows) * len(SOLVERS),
        "rows": rows,
    }
    _write_json(workspace / "h2.json", result)
    return result


def _run_boussinesq_dispersion(
    *, config: Mapping[str, Any], workspace: Path
) -> dict[str, Any]:
    spec = config["boussinesq_dispersion"]
    nx = int(spec["nx"])
    ny = int(spec["ny"])
    length = float(spec["physical_length"])
    dx = length / nx
    depth = float(spec["depth"])
    rows: list[dict[str, Any]] = []
    for mode in spec["modes"]:
        mode_int = int(mode)
        wavenumber = 2.0 * math.pi * mode_int / length
        expected_omega = float(
            discrete_dispersion(
                np.asarray([wavenumber]), dx=dx, depth=depth, alpha=1.0 / 3.0
            )[0][0]
        )
        x = np.arange(nx, dtype=np.float64)[:, None] * dx
        eta0 = float(spec["amplitude"]) * np.cos(wavenumber * x) * np.ones((1, ny))
        solver = _make_boussinesq_solver_from_cfg(
            {
                "nx": nx,
                "ny": ny,
                "dx": dx,
                "dy": dx,
                "dt": min(0.02, 0.2 * dx / math.sqrt(9.81 * depth)),
                "g": 9.81,
                "cfl": 0.35,
                "alpha": 1.0 / 3.0,
                "min_depth": 1.0e-4,
                "depth_scale": 1.0,
                "boundary": "periodic",
                "mode": "linear_constant_depth",
                "use_sponge": False,
                "sponge_width": 0,
                "sponge_min_factor": 0.8,
                "filter_strength": 0.0,
                "filter_time_mode": "disabled",
                "linear_solver_tol": float(spec["linear_solver_tol"]),
                "linear_solver_abs_tol": 0.0,
                "linear_solver_max_iter": int(spec["linear_solver_max_iter"]),
                "linear_solver_preconditioner": "sparse_lu",
                "cg_failure_mode": "strict_v2",
                "check_finite": True,
            }
        )
        solver.set_bathymetry(-depth * np.ones((nx, ny), dtype=np.float64))
        solver.set_initial_condition(
            eta0,
            eta_t0=directional_rate(
                eta0, dx=dx, depth=depth, direction="left", alpha=1.0 / 3.0
            ),
        )
        duration = min(max(0.5 * 2.0 * math.pi / expected_omega, 8.4), 84.0)
        sample_times = np.linspace(duration / 10.0, duration, 10, dtype=np.float64)
        states, emitted, _dt, diagnostics = _simulate_one_local(
            solver,
            n_steps=1,
            save_every=1,
            auto_dt=True,
            target_cfl=0.35,
            include_initial_state=False,
            requested_times=sample_times,
            max_natural_steps=20_000,
            collect_natural_step_health=True,
            requested_state_dtype=np.float64,
        )
        basis = np.exp(-1j * wavenumber * x)
        coefficients = np.asarray([np.mean(frame[0] * basis) for frame in states])
        measured_omega = float(
            abs(np.polyfit(emitted, np.unwrap(np.angle(coefficients)), 1)[0])
        )
        rel_error = abs(measured_omega - expected_omega) / expected_omega
        failed = np.asarray(diagnostics.get("cg_failed_count", []), dtype=np.int64)
        if rel_error > float(spec["phase_speed_relative_error_max"]) or int(np.sum(failed)) != 0:
            raise R6ValidationError(f"R6 Boussinesq dispersion gate failed: mode={mode_int}")
        rows.append(
            {
                "mode": mode_int,
                "expected_omega": expected_omega,
                "measured_omega": measured_omega,
                "phase_speed_relative_error": rel_error,
                "cg_failed_count": int(np.sum(failed)),
                "finite": bool(np.isfinite(states).all()),
            }
        )
    result = {"stage": "boussinesq_dispersion", "status": "passed", "rows": rows}
    _write_json(workspace / "boussinesq_dispersion.json", result)
    return result


def preflight_r6_broad_validation(
    *, repo_root: Path, config_path: Path, output_root: Path
) -> str:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = _load_yaml(config_path)
    _validate_config(config)
    _configure_threads(config)
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse R6 broad-validation output root: {output_root}")
    contract = _contract(repo_root, config)
    geoclaw = _verify_completed_geoclaw(repo_root, config)
    for split, expected in (("train", 10_000), ("eval", 1_000), ("test", 2_500)):
        rows = _read_jsonl(repo_root / f"data/{split}/synthetic/scenario_manifest.jsonl")
        if len(rows) != expected:
            raise R6ValidationError(f"R6 {split} manifest count mismatch: {len(rows)} != {expected}")
    return json.dumps(
        {
            "schema_id": SCHEMA_ID,
            "status": "ready",
            "contract_hash": contract["scientific_scope"]["contract_hash"],
            "geoclaw": geoclaw,
            "output_root": str(output_root),
        },
        indent=2,
        sort_keys=True,
    )


def execute_r6_broad_validation(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    resume: bool,
    production_validation_summary: Path | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = _load_yaml(config_path)
    _validate_config(config)
    _configure_threads(config)
    contract = _contract(repo_root, config)
    times = _contract_times(contract)
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"Refusing to reuse R6 broad-validation output root: {output_root}")
        summary_path = output_root / "summary.json"
        if summary_path.is_file():
            validate_checksums(output_root)
            return summary_path
        raise R6ValidationError("Partial R6 broad validation may only resume after a fresh implementation")

    output_root.mkdir(parents=True, exist_ok=False)
    workspace = output_root / ".workspace"
    workspace.mkdir()
    started = time.monotonic()
    try:
        _write_json(
            workspace / "frozen_contract.json",
            {
                "schema_id": SCHEMA_ID,
                "source_config": config,
                "source_config_sha256": _sha256(config_path),
                "suite_contract_sha256": _sha256(repo_root / str(config["corrected_production"]["contract_path"])),
                "code_state": code_state(repo_root),
                "requested_times": times.tolist(),
                "thresholds_frozen_before_execution": True,
            },
        )
        print("[r6-validation] H0 full frozen-dataset audit", flush=True)
        if production_validation_summary is None:
            h0_root = workspace / "h0_full_dataset_audit"
            h0_summary_path = run_validation(
                contract_path=repo_root / str(config["corrected_production"]["contract_path"]),
                output_root=h0_root,
                canary_count=int(config["h0_full_dataset_audit"]["representative_solver_canaries"]),
                deep_payload_audit=bool(config["h0_full_dataset_audit"]["deep_payload_audit"]),
            )
            h0_summary = _read_json(h0_summary_path)
        else:
            h0_summary = _reuse_production_validation(
                repo_root=repo_root,
                summary_path=production_validation_summary,
                workspace=workspace,
                contract=contract,
            )
        if h0_summary.get("status") != "passed":
            raise R6ValidationError("R6 H0 full dataset audit did not pass")

        print("[r6-validation] Level A current-grid checks", flush=True)
        level_a = _run_level_a(repo_root=repo_root, config=config, workspace=workspace, times=times)
        print("[r6-validation] H1 integrated health/repeatability", flush=True)
        h1 = _run_h1(repo_root=repo_root, config=config, workspace=workspace, times=times)
        print("[r6-validation] H2 paired CFL sensitivity", flush=True)
        h2 = _run_h2(repo_root=repo_root, config=config, workspace=workspace, times=times)
        print("[r6-validation] Boussinesq dispersion and CG health", flush=True)
        dispersion = _run_boussinesq_dispersion(config=config, workspace=workspace)
        print("[r6-validation] verifying completed R6 GeoClaw diagnostic", flush=True)
        geoclaw = _verify_completed_geoclaw(repo_root, config)

        archive = output_root / "r6_broad_validation.tar.zst"
        _archive_workspace(workspace, archive)
        summary = {
            "schema_id": SCHEMA_ID,
            "evaluation_type": "r6_broad_numerical_validation",
            "status": "passed",
            "interpretation": (
                "Fresh evidence for the corrected R6 data contract. It does not relabel, "
                "replace, or rely on the archived 96x96 / 0.175 H0, Level A, H1, or H2 studies."
            ),
            "corrected_production_contract_hash": config["corrected_production"]["contract_hash"],
            "production_lineage": {
                "master_shape": [384, 384],
                "solver_input_shape": [128, 128],
                "computational_shape": [192, 192],
                "publication_shape": [64, 64],
                "buffer_cells": 32,
                "requested_times": times.tolist(),
            },
            "code_state": code_state(repo_root),
            "external_revisions": geoclaw["external_revisions"],
            "stages": [
                _stage_record(
                    stage_id="h0_full_dataset_audit",
                    status="passed",
                    artifact_path=workspace / "h0_full_dataset_audit/summary.json",
                    repo_root=output_root,
                ),
                _stage_record(
                    stage_id="level_a_current_grid",
                    status=level_a["status"],
                    artifact_path=workspace / "level_a.json",
                    repo_root=output_root,
                ),
                _stage_record(
                    stage_id="r6_geoclaw",
                    status="completed",
                    artifact_path=repo_root / str(config["external_comparator"]["summary"]),
                    repo_root=output_root,
                ),
                _stage_record(
                    stage_id="h1_integrated_health",
                    status=h1["status"],
                    artifact_path=workspace / "h1.json",
                    repo_root=output_root,
                ),
                _stage_record(
                    stage_id="h2_sensitivity",
                    status=h2["status"],
                    artifact_path=workspace / "h2.json",
                    repo_root=output_root,
                ),
                _stage_record(
                    stage_id="boussinesq_dispersion",
                    status=dispersion["status"],
                    artifact_path=workspace / "boussinesq_dispersion.json",
                    repo_root=output_root,
                ),
            ],
            "archive_path": (Path(output_root.name) / archive.name).as_posix(),
            "archive_sha256": _sha256(archive),
            "archive_size_bytes": int(archive.stat().st_size),
            "duration_seconds": time.monotonic() - started,
        }
        _write_json(output_root / "summary.json", summary)
        _write_checksums(output_root)
        shutil.rmtree(workspace)
        return output_root / "summary.json"
    except Exception:
        # Preserve the frozen configuration and any partial observations for
        # diagnosis, but never synthesize a passing checksum manifest.
        raise
