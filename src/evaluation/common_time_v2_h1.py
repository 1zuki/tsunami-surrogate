from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    candidate_requested_times,
    code_state,
    hash_array,
    sha256_file,
    stable_hash_payload,
)
from src.evaluation.buffered_crop_benchmark import (
    SOLVERS,
    run_buffered_case_detailed,
)
from src.evaluation.common_time_v2_level_a import (
    _load_canary_arrays,
    validate_checksums as validate_level_a_checksums,
)


SCHEMA_ID = "tsunami-surrogate.common-time-v2.h1.v1"
CONFIG_SCHEMA_ID = "tsunami-surrogate.common-time-v2.h1-config.v1"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
FROZEN_FILENAMES = (
    "preregistered_contract.json",
    "selected_scenarios.json",
    "task_plan.json",
)
FROZEN_CHECKSUMS = "CONTRACT_SHA256SUMS.txt"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _validate_standard_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise RuntimeError(f"Missing checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {root.name}/{relative}")
        listed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if listed != actual:
        raise RuntimeError(f"Checksum inventory mismatch: {root}")


def _write_frozen_checksums(root: Path) -> None:
    rows = [f"{sha256_file(root / name)}  {name}" for name in FROZEN_FILENAMES]
    _write_text(root / FROZEN_CHECKSUMS, "\n".join(rows) + "\n")


def validate_frozen_checksums(root: Path) -> None:
    manifest = root / FROZEN_CHECKSUMS
    if not manifest.is_file():
        raise RuntimeError(f"Missing H1 frozen checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        if relative not in FROZEN_FILENAMES:
            raise RuntimeError(f"Unexpected H1 frozen checksum entry: {relative}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"H1 frozen checksum mismatch: {relative}")
        listed.add(relative)
    if listed != set(FROZEN_FILENAMES):
        raise RuntimeError("H1 frozen checksum inventory is incomplete")
    unexpected = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and path.name not in {*FROZEN_FILENAMES, FROZEN_CHECKSUMS}
    }
    if unexpected:
        raise RuntimeError(f"Unexpected top-level files in H1 artifact: {unexpected}")


def _write_execution_checksums(execution_root: Path) -> None:
    rows = []
    for path in sorted(execution_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(
                f"{sha256_file(path)}  {path.relative_to(execution_root).as_posix()}"
            )
    _write_text(execution_root / "SHA256SUMS.txt", "\n".join(rows) + "\n")


def validate_execution_checksums(execution_root: Path) -> None:
    _validate_standard_checksums(execution_root)


def validate_h1_checksums(root: Path) -> None:
    validate_frozen_checksums(root)
    execution_root = root / "execution"
    if execution_root.exists():
        if (execution_root / "result.json").is_file():
            validate_execution_checksums(execution_root)
        elif (execution_root / "SHA256SUMS.txt").exists():
            raise RuntimeError("Partial H1 execution must not have a final checksum manifest")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("H1 YAML must contain a mapping")
    _validate_config(payload)
    return payload


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != CONFIG_SCHEMA_ID or config.get("stage") != "H1":
        raise ValueError("H1 config schema/stage mismatch")
    if config.get("claim_scope") != "implementation_and_long_horizon_health_only":
        raise ValueError("H1 must remain an implementation/health smoke")

    expected_prerequisites = {
        "h0_contract_hash": (
            "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b"
        ),
        "level_a_contract_hash": (
            "be1af7dce1f48942e6d20a96bb06b1359655903847c7580954901e2dcfa3332b"
        ),
        "level_b_bundle_hash": (
            "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
        ),
        "require_h0_pass": True,
        "require_level_a_decision": "pass_to_H1",
        "require_level_b_decision": "pass_to_H1",
    }
    if config.get("prerequisites") != expected_prerequisites:
        raise ValueError("H1 prerequisite identities or decisions changed")

    selection = config["selection"]
    bathymetry = tuple(selection["bathymetry_families"])
    sources = tuple(selection["source_families"])
    if (
        selection["split"] != "train"
        or int(selection["expected_split_count"]) != 10_000
        or int(selection["count_per_cell"]) != 1
        or len(bathymetry) != 5
        or len(set(bathymetry)) != 5
        or len(sources) != 6
        or len(set(sources)) != 6
    ):
        raise ValueError("H1 selection must be one training case per 5x6 family cell")
    expected_selection = {
        "split": "train",
        "expected_split_count": 10_000,
        "count_per_cell": 1,
        "selection_seed": "common-time-v2-h1-balanced-selection-v1",
        "replay_selection_seed": "common-time-v2-h1-replay-selection-v1",
        "bathymetry_families": [
            "canyon",
            "continental",
            "island",
            "seamounts",
            "trench",
        ],
        "source_families": [
            "dipole",
            "fault",
            "gaussian",
            "multi-gauss",
            "okada-like",
            "rough",
        ],
    }
    if dict(selection) != expected_selection:
        raise ValueError("H1 balanced selection policy changed")

    candidate = config["candidate"]
    expected_candidate = {
        "solvers": list(SOLVERS),
        "computational_grid": 96,
        "publication_grid": 64,
        "buffer_cells_per_side": 16,
        "source_taper_cells": 8,
        "bathymetry_extension": "edge",
        "output_crop": "central",
        "dx": 1.0 / 64.0,
        "dy": 1.0 / 64.0,
        "requested_time_start": 0.0035,
        "requested_time_step": 0.0035,
        "requested_time_count": 50,
        "horizon": 0.175,
        "requested_state_dtype": "float64",
        "max_natural_steps": 20_000,
        "sponge_width_cells": 16,
        "sponge_min_factor": 0.8,
        "sponge_axes": "xy",
        "sponge_profile": "cosine",
        "sponge_time_mode": "elapsed_time_consistent",
        "sponge_reference_dt": 0.0035,
        "swe_cfl": 0.45,
        "swe_boundary": "radiation",
        "dry_tolerance": 1.0e-6,
        "max_velocity": 30.0,
        "muscl_limiter": "minmod",
        "boussinesq_cfl": 0.35,
        "boussinesq_boundary": "open",
        "boussinesq_boundary_interpretation": "zero_gradient_edge_padding",
        "boussinesq_depth_scale": 1.0,
        "boussinesq_mode": "linear_variable_depth",
        "boussinesq_filter_strength": 0.0,
        "boussinesq_filter_time_mode": "disabled",
        "boussinesq_linear_solver_tol": 1.0e-10,
        "boussinesq_linear_solver_abs_tol": 0.0,
        "boussinesq_linear_solver_max_iter": 750,
        "boussinesq_cg_failure_mode": "strict_v2",
    }
    if dict(candidate) != expected_candidate:
        raise ValueError("H1 candidate settings differ from the reviewed 96-to-64 policy")
    configured_times = (
        np.arange(1, 51, dtype=np.float64) * np.float64(0.0035)
    )
    configured_times[-1] = np.float64(0.175)
    if not np.array_equal(candidate_requested_times(), configured_times):
        raise ValueError("H1 requested-time vector changed")

    expected_health_gates = {
        "require_exact_requested_times": True,
        "require_adjacent_bracket_provenance": True,
        "require_no_missing_duplicate_or_extrapolated_outputs": True,
        "require_complete_float64_natural_health": True,
        "reject_nonfinite": True,
        "min_h_tolerance": -1.0e-6,
        "max_abs_eta_limit": 5.0,
        "max_velocity_limit": 30.0,
        "max_eta_over_depth": 1.0,
        "require_cg_converged": True,
        "require_two_cg_solves_per_natural_step": True,
        "deterministic_replay_comparison": "exact_scientific_digest",
    }
    if config.get("health_gates") != expected_health_gates:
        raise ValueError("H1 health gates changed")

    replay = config["replay"]
    if replay != {
        "selected_case_count": 1,
        "runs_per_solver": 1,
        "comparison": "exact_scientific_digest",
    }:
        raise ValueError("H1 deterministic replay policy changed")

    execution = config["execution"]
    if execution != {
        "requested_workers": 8,
        "requested_max_in_flight": 8,
        "process_start_method": "spawn",
        "numerical_library_threads": 1,
        "resume_policy": "validate_then_skip",
        "corruption_policy": "fail_closed",
    }:
        raise ValueError("H1 execution policy changed")
    if config["decisions"] != {
        "pass": "pass_to_H2",
        "health_failure": "blocked_h1_implementation_health",
        "implementation_failure": "implementation_failure",
    }:
        raise ValueError("H1 decision vocabulary changed")


def _environment_snapshot() -> dict[str, Any]:
    packages = sorted(
        {
            (
                str(distribution.metadata.get("Name", "")).lower(),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    package_rows = [{"name": name, "version": version} for name, version in packages]
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "package_inventory": package_rows,
        "package_inventory_hash": stable_hash_payload(
            artifact_kind="h1-python-package-inventory",
            payload=package_rows,
            schema_id=SCHEMA_ID,
        ),
    }


def _verify_prerequisites(
    *,
    config: Mapping[str, Any],
    h0_root: Path,
    level_a_root: Path,
    level_b_bundle_root: Path,
    level_b_evaluation_root: Path,
) -> dict[str, Any]:
    _validate_standard_checksums(h0_root)
    validate_level_a_checksums(level_a_root)
    _validate_standard_checksums(level_b_bundle_root)
    _validate_standard_checksums(level_b_evaluation_root)

    prerequisites = config["prerequisites"]
    h0_summary = _read_json(h0_root / "h0_summary.json")
    h0_decision = _read_json(h0_root / "h0_decision.json")
    if (
        h0_root.name != prerequisites["h0_contract_hash"]
        or h0_summary.get("inventory_count") != 13_500
        or h0_decision.get("audit_passed") is not True
    ):
        raise RuntimeError("H0 prerequisite identity or decision mismatch")

    level_a_contract = _read_json(level_a_root / "preregistered_contract.json")
    level_a_decision = _read_json(level_a_root / "execution" / "decision.json")
    if (
        level_a_root.name != prerequisites["level_a_contract_hash"]
        or level_a_contract.get("contract_hash") != level_a_root.name
        or level_a_decision.get("contract_hash") != level_a_root.name
        or level_a_decision.get("decision")
        != prerequisites["require_level_a_decision"]
        or level_a_decision.get("level_a_passed") is not True
    ):
        raise RuntimeError("Level A prerequisite identity or decision mismatch")

    level_b_contract = _read_json(level_b_bundle_root / "frozen_contract.json")
    level_b_decision = _read_json(level_b_evaluation_root / "decision.json")
    if (
        level_b_bundle_root.name != prerequisites["level_b_bundle_hash"]
        or level_b_evaluation_root.name != prerequisites["level_b_bundle_hash"]
        or level_b_contract.get("bundle_hash") != level_b_bundle_root.name
        or level_b_decision.get("bundle_hash") != level_b_bundle_root.name
        or level_b_decision.get("decision")
        != prerequisites["require_level_b_decision"]
        or level_b_decision.get("minimum_level_b_passed") is not True
    ):
        raise RuntimeError("Minimum Level B prerequisite identity or decision mismatch")

    inventory_path = h0_root / "h0_input_inventory.jsonl"
    return {
        "h0": {
            "root": str(h0_root),
            "contract_hash": h0_root.name,
            "decision": "pass",
            "inventory_path": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
            "inventory_count": int(h0_summary["inventory_count"]),
        },
        "level_a": {
            "root": str(level_a_root),
            "contract_hash": level_a_root.name,
            "decision": str(level_a_decision["decision"]),
            "scientific_digest": str(level_a_decision["scientific_digest"]),
            "checksums_sha256": sha256_file(level_a_root / "SHA256SUMS.txt"),
        },
        "level_b": {
            "bundle_root": str(level_b_bundle_root),
            "evaluation_root": str(level_b_evaluation_root),
            "bundle_hash": level_b_bundle_root.name,
            "decision": str(level_b_decision["decision"]),
            "checksums_sha256": sha256_file(
                level_b_evaluation_root / "SHA256SUMS.txt"
            ),
        },
    }


def _selection_rank(
    record: Mapping[str, Any],
    *,
    inventory_sha256: str,
    seed: str,
    purpose: str,
) -> str:
    return stable_hash_payload(
        artifact_kind="common-time-v2-h1-selection-rank",
        payload={
            "purpose": purpose,
            "seed": seed,
            "inventory_sha256": inventory_sha256,
            "qualified_id": record["qualified_id"],
            "input_fingerprint": record["input_fingerprint"],
            "bathymetry_type": record["bathymetry_type"],
            "source_type": record["source_type"],
        },
        schema_id=SCHEMA_ID,
    )


def select_h1_scenarios(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_config: Mapping[str, Any],
    inventory_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = str(selection_config["split"])
    expected_count = int(selection_config["expected_split_count"])
    bathymetry_families = tuple(map(str, selection_config["bathymetry_families"]))
    source_families = tuple(map(str, selection_config["source_families"]))
    seed = str(selection_config["selection_seed"])

    training = [dict(row) for row in rows if row.get("split") == split]
    if len(training) != expected_count:
        raise RuntimeError(
            f"H1 expected {expected_count} {split} inputs, found {len(training)}"
        )
    qualified_ids = [str(row.get("qualified_id")) for row in training]
    if len(set(qualified_ids)) != len(qualified_ids):
        raise RuntimeError("H1 authoritative training inventory has duplicate identities")

    expected_cells = {
        (bathymetry, source)
        for bathymetry in bathymetry_families
        for source in source_families
    }
    cell_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        cell: [] for cell in expected_cells
    }
    unexpected_cells: set[tuple[str, str]] = set()
    for row in training:
        cell = (str(row["bathymetry_type"]), str(row["source_type"]))
        if cell not in cell_rows:
            unexpected_cells.add(cell)
            continue
        cell_rows[cell].append(row)
    if unexpected_cells:
        raise RuntimeError(f"Unexpected H1 family cells: {sorted(unexpected_cells)}")
    empty = [cell for cell, candidates in cell_rows.items() if not candidates]
    if empty:
        raise RuntimeError(f"Empty H1 family cells: {sorted(empty)}")

    selected: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(sorted(expected_cells)):
        ranked = sorted(
            (
                _selection_rank(
                    record,
                    inventory_sha256=inventory_sha256,
                    seed=seed,
                    purpose="balanced_cell_selection",
                ),
                str(record["qualified_id"]),
                record,
            )
            for record in cell_rows[cell]
        )
        rank, _qualified_id, record = ranked[0]
        selected.append(
            {
                "selection_ordinal": ordinal,
                "bathymetry_type": cell[0],
                "source_type": cell[1],
                "cell_candidate_count": len(ranked),
                "selection_rank": rank,
                "record": record,
            }
        )

    replay_seed = str(selection_config["replay_selection_seed"])
    replay_ranked = sorted(
        (
            _selection_rank(
                entry["record"],
                inventory_sha256=inventory_sha256,
                seed=replay_seed,
                purpose="deterministic_replay_case",
            ),
            int(entry["selection_ordinal"]),
        )
        for entry in selected
    )
    replay_selection_ordinal = replay_ranked[0][1]
    summary = {
        "split": split,
        "authoritative_split_count": len(training),
        "family_cell_count": len(expected_cells),
        "selected_count": len(selected),
        "selected_qualified_ids": [
            entry["record"]["qualified_id"] for entry in selected
        ],
        "selected_input_fingerprints": [
            entry["record"]["input_fingerprint"] for entry in selected
        ],
        "replay_selection_ordinal": replay_selection_ordinal,
        "replay_qualified_id": selected[replay_selection_ordinal]["record"][
            "qualified_id"
        ],
        "validation_inputs_inspected_scientifically": False,
        "test_inputs_inspected_scientifically": False,
        "eventual_training_overlap": "all H1 cases may remain in training",
    }
    return selected, summary


def _candidate_config_hash(candidate: Mapping[str, Any]) -> str:
    return stable_hash_payload(
        artifact_kind="common-time-v2-h1-candidate-config",
        payload=dict(candidate),
        schema_id=SCHEMA_ID,
    )


def _task_identity(
    *,
    ordinal: int,
    run_kind: str,
    solver: str,
    selection: Mapping[str, Any],
    candidate_config_hash: str,
    reference_primary_task_id: str | None = None,
) -> dict[str, Any]:
    record = selection["record"]
    identity = {
        "ordinal": int(ordinal),
        "run_kind": str(run_kind),
        "solver": str(solver),
        "selection_ordinal": int(selection["selection_ordinal"]),
        "qualified_id": str(record["qualified_id"]),
        "input_fingerprint": str(record["input_fingerprint"]),
        "candidate_config_hash": str(candidate_config_hash),
        "reference_primary_task_id": reference_primary_task_id,
    }
    identity["task_id"] = stable_hash_payload(
        artifact_kind="common-time-v2-h1-task",
        payload=identity,
        schema_id=SCHEMA_ID,
    )[:24]
    return identity


def _build_tasks(
    selected: Sequence[Mapping[str, Any]],
    *,
    solvers: Sequence[str],
    replay_selection_ordinal: int,
    candidate_config_hash: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    primary_by_selection_solver: dict[tuple[int, str], str] = {}
    ordinal = 0
    for selection in selected:
        for solver in solvers:
            task = _task_identity(
                ordinal=ordinal,
                run_kind="primary",
                solver=solver,
                selection=selection,
                candidate_config_hash=candidate_config_hash,
            )
            tasks.append(task)
            primary_by_selection_solver[
                (int(selection["selection_ordinal"]), str(solver))
            ] = str(task["task_id"])
            ordinal += 1

    replay_selection = selected[replay_selection_ordinal]
    for solver in solvers:
        reference = primary_by_selection_solver[(replay_selection_ordinal, str(solver))]
        tasks.append(
            _task_identity(
                ordinal=ordinal,
                run_kind="replay",
                solver=solver,
                selection=replay_selection,
                candidate_config_hash=candidate_config_hash,
                reference_primary_task_id=reference,
            )
        )
        ordinal += 1
    return tasks


def _contract_hash(contract: Mapping[str, Any]) -> str:
    identity = dict(contract)
    identity.pop("contract_hash", None)
    return stable_hash_payload(
        artifact_kind="common-time-v2-h1-preregistered-contract",
        payload=identity,
        schema_id=SCHEMA_ID,
    )


def _validate_contract_identity(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("H1 contract schema mismatch")
    if contract.get("contract_hash") != _contract_hash(contract):
        raise RuntimeError("H1 contract content hash mismatch")
    selected = contract["selected_scenarios"]
    tasks = contract["tasks"]
    if len(selected) != 30 or len(tasks) != 93:
        raise RuntimeError("H1 contract must freeze 30 scenarios and 93 tasks")
    if sum(task["run_kind"] == "primary" for task in tasks) != 90:
        raise RuntimeError("H1 contract primary task count changed")
    if sum(task["run_kind"] == "replay" for task in tasks) != 3:
        raise RuntimeError("H1 contract replay task count changed")
    if [int(task["ordinal"]) for task in tasks] != list(range(len(tasks))):
        raise RuntimeError("H1 task ordering is not contiguous and frozen")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("H1 task identities are not unique")


def freeze_h1_contract(
    *,
    repo_root: Path,
    config_path: Path,
    h0_root: Path,
    level_a_root: Path,
    level_b_bundle_root: Path,
    level_b_evaluation_root: Path,
    output_base: Path,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    h0_root = h0_root.resolve()
    level_a_root = level_a_root.resolve()
    level_b_bundle_root = level_b_bundle_root.resolve()
    level_b_evaluation_root = level_b_evaluation_root.resolve()
    output_base = output_base.resolve()
    config = _load_config(config_path)
    prerequisite_evidence = _verify_prerequisites(
        config=config,
        h0_root=h0_root,
        level_a_root=level_a_root,
        level_b_bundle_root=level_b_bundle_root,
        level_b_evaluation_root=level_b_evaluation_root,
    )
    inventory_path = Path(prerequisite_evidence["h0"]["inventory_path"])
    inventory_sha256 = str(prerequisite_evidence["h0"]["inventory_sha256"])
    selected, selection_summary = select_h1_scenarios(
        _read_jsonl(inventory_path),
        selection_config=config["selection"],
        inventory_sha256=inventory_sha256,
    )

    for selection in selected:
        _load_canary_arrays(selection["record"])

    candidate_hash = _candidate_config_hash(config["candidate"])
    tasks = _build_tasks(
        selected,
        solvers=config["candidate"]["solvers"],
        replay_selection_ordinal=int(selection_summary["replay_selection_ordinal"]),
        candidate_config_hash=candidate_hash,
    )
    contract: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h1-preregistered-contract",
        "stage": "H1",
        "status": "frozen_unexecuted",
        "claim_scope": config["claim_scope"],
        "scientific_outcome_viewed_before_freeze": False,
        "prerequisites": prerequisite_evidence,
        "code_state": code_state(repo_root),
        "code_state_policy": {
            "freeze_current_h1_code": True,
            "allowed_post_level_b_scopes": [
                "h1_contract_selection_runner_and_tests",
                "evaluation_only_buffered_diagnostics_access",
            ],
            "forbidden_post_level_b_scopes": [
                "solver_numerics",
                "dataset_publication_semantics",
                "candidate_grid_boundary_sponge_or_filter_settings",
                "requested_time_or_extraction_semantics",
                "level_a_or_level_b_artifacts",
            ],
        },
        "environment": _environment_snapshot(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config,
        "candidate_config_hash": candidate_hash,
        "selection_summary": selection_summary,
        "selected_scenarios": selected,
        "tasks": tasks,
        "worker_policy": dict(config["execution"]),
        "decision_policy": dict(config["decisions"]),
        "immutability": {
            "post_result_changes_forbidden": True,
            "solver_or_metric_change_requires_new_h1_contract": True,
            "validation_and_test_scientific_outcomes_excluded": True,
            "h1_is_not_scientific_acceptance": True,
        },
    }
    contract["contract_hash"] = _contract_hash(contract)
    _validate_contract_identity(contract)
    output_root = output_base / contract["contract_hash"]

    selected_payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h1-selected-scenarios",
        "contract_hash": contract["contract_hash"],
        "selection_summary": selection_summary,
        "selected_scenarios": selected,
    }
    task_payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h1-task-plan",
        "contract_hash": contract["contract_hash"],
        "candidate_config_hash": candidate_hash,
        "tasks": tasks,
    }
    if output_root.exists():
        contract_path = output_root / "preregistered_contract.json"
        if not contract_path.is_file() or _read_json(contract_path) != contract:
            raise FileExistsError(
                f"Refusing to replace a different H1 artifact: {output_root}"
            )
        validate_frozen_checksums(output_root)
        return output_root
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "preregistered_contract.json", contract)
    _write_json(output_root / "selected_scenarios.json", selected_payload)
    _write_json(output_root / "task_plan.json", task_payload)
    _write_frozen_checksums(output_root)
    validate_frozen_checksums(output_root)
    return output_root


def _json_array(values: Any) -> list[Any]:
    return np.asarray(values).tolist()


def _all_finite(values: Any) -> bool:
    array = np.asarray(values)
    return bool(np.isfinite(array).all())


def _required_diagnostic_keys(solver: str) -> tuple[str, ...]:
    common = (
        "requested_timestamps",
        "left_natural_timestamps",
        "right_natural_timestamps",
        "interpolation_weights",
        "bracket_widths",
        "exact_knot",
        "natural_step_indices",
        "total_natural_steps",
        "natural_dt_history",
        "final_natural_timestamp",
        "natural_health_step_indices",
        "left_natural_step_times",
        "right_natural_step_times",
        "proposed_dt",
        "pre_step_cfl",
        "post_step_cfl",
        "finite_state_flag",
    )
    if solver in {"swe_hydrostatic", "swe_muscl_hr"}:
        return common + ("swe_min_depth", "swe_max_speed", "swe_dry_cell_count")
    return common + (
        "cg_step_converged",
        "cg_failed_count",
        "cg_max_iterations",
        "cg_max_residual_ratio",
        "cg_solve0_converged",
        "cg_solve0_iterations",
        "cg_solve0_initial_residual",
        "cg_solve0_final_residual",
        "cg_solve0_residual_ratio",
        "cg_solve1_converged",
        "cg_solve1_iterations",
        "cg_solve1_initial_residual",
        "cg_solve1_final_residual",
        "cg_solve1_residual_ratio",
        "filter_enabled",
        "filter_application_count",
    )


def _summarize_diagnostics(
    *,
    solver: str,
    details: Mapping[str, Any],
    expected_times: np.ndarray,
    target_cfl: float,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any], dict[str, Any]]:
    diagnostics = details["diagnostics"]
    missing_keys = [
        key for key in _required_diagnostic_keys(solver) if key not in diagnostics
    ]
    if missing_keys:
        raise RuntimeError(f"Missing H1 diagnostics for {solver}: {missing_keys}")

    requested = np.asarray(diagnostics["requested_timestamps"], dtype=np.float64)
    left = np.asarray(diagnostics["left_natural_timestamps"], dtype=np.float64)
    right = np.asarray(diagnostics["right_natural_timestamps"], dtype=np.float64)
    weights = np.asarray(diagnostics["interpolation_weights"], dtype=np.float64)
    widths = np.asarray(diagnostics["bracket_widths"], dtype=np.float64)
    exact = np.asarray(diagnostics["exact_knot"], dtype=np.bool_)
    step_indices = np.asarray(diagnostics["natural_step_indices"], dtype=np.int64)
    requested_arrays = (requested, left, right, weights, widths, exact, step_indices)
    requested_lengths = {int(values.size) for values in requested_arrays}
    tolerance = 16.0 * np.finfo(np.float64).eps

    exact_semantics = bool(
        np.all(widths[exact] == 0.0)
        and np.all(left[exact] == requested[exact])
        and np.all(right[exact] == requested[exact])
        and np.all(weights[exact] == 0.0)
    )
    nonexact = ~exact
    nonexact_semantics = bool(
        np.all(right[nonexact] > left[nonexact])
        and np.all(widths[nonexact] > 0.0)
        and np.all(weights[nonexact] > 0.0)
        and np.all(weights[nonexact] < 1.0)
        and np.allclose(
            widths[nonexact],
            right[nonexact] - left[nonexact],
            rtol=0.0,
            atol=tolerance,
        )
        and np.allclose(
            requested[nonexact],
            left[nonexact] + weights[nonexact] * widths[nonexact],
            rtol=0.0,
            atol=tolerance,
        )
    )

    total_steps_array = np.asarray(
        diagnostics["total_natural_steps"], dtype=np.int64
    ).reshape(-1)
    final_time_array = np.asarray(
        diagnostics["final_natural_timestamp"], dtype=np.float64
    ).reshape(-1)
    total_steps = int(total_steps_array[0]) if total_steps_array.size == 1 else -1
    dt_history = np.asarray(details["natural_dt_history"], dtype=np.float64)
    recorded_dt_history = np.asarray(
        diagnostics["natural_dt_history"], dtype=np.float64
    )
    health_steps = np.asarray(
        diagnostics["natural_health_step_indices"], dtype=np.int64
    )
    left_step_times = np.asarray(
        diagnostics["left_natural_step_times"], dtype=np.float64
    )
    right_step_times = np.asarray(
        diagnostics["right_natural_step_times"], dtype=np.float64
    )
    proposed_dt = np.asarray(diagnostics["proposed_dt"], dtype=np.float64)
    pre_cfl = np.asarray(diagnostics["pre_step_cfl"], dtype=np.float64)
    post_cfl = np.asarray(diagnostics["post_step_cfl"], dtype=np.float64)
    finite_flags = np.asarray(diagnostics["finite_state_flag"], dtype=np.bool_)
    natural_arrays = (
        dt_history,
        recorded_dt_history,
        health_steps,
        left_step_times,
        right_step_times,
        proposed_dt,
        pre_cfl,
        post_cfl,
        finite_flags,
    )
    natural_lengths = {int(values.size) for values in natural_arrays}

    checks = {
        "requested_output_count": requested_lengths == {expected_times.size},
        "requested_times_exact": bool(np.array_equal(requested, expected_times)),
        "requested_times_unique_strict": bool(
            requested.size == np.unique(requested).size
            and np.all(np.diff(requested) > 0.0)
        ),
        "brackets_finite": all(_all_finite(values) for values in requested_arrays[:-1]),
        "brackets_contain_requests": bool(
            np.all(left <= requested) and np.all(requested <= right)
        ),
        "exact_knot_semantics": exact_semantics,
        "nonexact_interpolation_semantics": nonexact_semantics,
        "natural_step_indices_valid": bool(
            total_steps > 0
            and np.all(step_indices >= 1)
            and np.all(step_indices <= total_steps)
            and np.all(np.diff(step_indices) >= 0)
        ),
        "natural_history_complete": natural_lengths == {total_steps},
        "natural_dt_history_exact": bool(
            np.array_equal(dt_history, recorded_dt_history)
            and np.array_equal(dt_history, proposed_dt)
        ),
        "natural_step_indices_complete": bool(
            np.array_equal(health_steps, np.arange(1, total_steps + 1))
        ),
        "natural_step_times_contiguous": bool(
            total_steps > 0
            and left_step_times[0] == 0.0
            and np.array_equal(left_step_times[1:], right_step_times[:-1])
            and np.allclose(
                right_step_times - left_step_times,
                dt_history,
                rtol=0.0,
                atol=tolerance,
            )
            and final_time_array.size == 1
            and final_time_array[0] == right_step_times[-1]
        ),
        "natural_health_finite": all(
            _all_finite(values)
            for values in (
                dt_history,
                left_step_times,
                right_step_times,
                pre_cfl,
                post_cfl,
            )
        ),
        "natural_dt_positive": bool(np.all(dt_history > 0.0)),
        "cfl_history_nonnegative": bool(
            np.all(pre_cfl >= 0.0) and np.all(post_cfl >= 0.0)
        ),
        "pre_step_cfl_respects_target": bool(
            np.all(pre_cfl <= float(target_cfl) * (1.0 + 1.0e-10))
        ),
        "all_natural_states_finite": bool(np.all(finite_flags)),
        "final_natural_time_covers_horizon": bool(
            final_time_array.size == 1
            and final_time_array[0] >= expected_times[-1]
            and left_step_times[-1] < expected_times[-1]
        ),
    }
    if checks["natural_step_indices_valid"]:
        bracket_indices = step_indices - 1
        checks["adjacent_brackets_match_natural_history"] = bool(
            np.all(
                left[nonexact]
                == left_step_times[bracket_indices[nonexact]]
            )
            and np.all(
                right[nonexact]
                == right_step_times[bracket_indices[nonexact]]
            )
            and np.all(
                requested[exact]
                == right_step_times[bracket_indices[exact]]
            )
        )
    else:
        checks["adjacent_brackets_match_natural_history"] = False

    provenance_keys = (
        "requested_timestamps",
        "left_natural_timestamps",
        "right_natural_timestamps",
        "interpolation_weights",
        "bracket_widths",
        "exact_knot",
        "natural_step_indices",
    )
    natural_keys = [
        "natural_dt_history",
        "natural_health_step_indices",
        "left_natural_step_times",
        "right_natural_step_times",
        "proposed_dt",
        "pre_step_cfl",
        "post_step_cfl",
        "finite_state_flag",
    ]
    if solver in {"swe_hydrostatic", "swe_muscl_hr"}:
        natural_keys.extend(("swe_min_depth", "swe_max_speed", "swe_dry_cell_count"))
    else:
        natural_keys.extend(
            key
            for key in _required_diagnostic_keys(solver)
            if key.startswith("cg_") or key.startswith("filter_")
        )
    provenance = {key: _json_array(diagnostics[key]) for key in provenance_keys}
    natural_health = {
        key: _json_array(
            details["natural_dt_history"]
            if key == "natural_dt_history"
            else diagnostics[key]
        )
        for key in natural_keys
    }
    all_hashes = {
        key: hash_array(values)
        for key, values in sorted(diagnostics.items())
    }
    summary = {
        "total_natural_steps": total_steps,
        "final_natural_timestamp": float(final_time_array[0]),
        "exact_knot_count": int(np.count_nonzero(exact)),
        "max_bracket_width": float(np.max(widths)),
        "max_pre_step_cfl": float(np.max(pre_cfl)),
        "max_post_step_cfl": float(np.max(post_cfl)),
    }
    return checks, provenance, natural_health, {
        "diagnostic_array_hashes": all_hashes,
        "summary": summary,
    }


def _solver_health_checks(
    *,
    solver: str,
    trajectory: np.ndarray,
    diagnostics: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    bathymetry: np.ndarray,
    health_gates: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    raw_eta = np.asarray(trajectory)
    eta = np.asarray(raw_eta, dtype=np.float64)
    if solver == "boussinesq":
        effective_depth = np.maximum(
            -np.asarray(bathymetry, dtype=np.float64), 1.0e-4
        )
    else:
        effective_depth = np.maximum(
            np.asarray(arrays["rest_depth"], dtype=np.float64), 1.0e-8
        )
    max_abs_eta = float(np.max(np.abs(eta)))
    max_eta_over_depth = float(np.max(np.abs(eta) / effective_depth[None, ...]))
    metrics: dict[str, Any] = {
        "max_abs_eta": max_abs_eta,
        "max_eta_over_depth": max_eta_over_depth,
    }
    checks = {
        "trajectory_shape": list(eta.shape)
        == [
            int(candidate["requested_time_count"]),
            int(candidate["publication_grid"]),
            int(candidate["publication_grid"]),
        ],
        "trajectory_float64": raw_eta.dtype == np.dtype(np.float64),
        "trajectory_finite": bool(np.isfinite(eta).all()),
        "max_abs_eta_within_limit": max_abs_eta
        <= float(health_gates["max_abs_eta_limit"]),
        "max_eta_over_depth_within_limit": max_eta_over_depth
        <= float(health_gates["max_eta_over_depth"]),
    }
    total_steps = int(np.asarray(diagnostics["total_natural_steps"]).reshape(-1)[0])
    if solver in {"swe_hydrostatic", "swe_muscl_hr"}:
        min_depth = np.asarray(diagnostics["swe_min_depth"], dtype=np.float64)
        max_speed = np.asarray(diagnostics["swe_max_speed"], dtype=np.float64)
        dry_cells = np.asarray(diagnostics["swe_dry_cell_count"], dtype=np.int64)
        metrics.update(
            {
                "minimum_depth": float(np.min(min_depth)),
                "maximum_speed": float(np.max(max_speed)),
                "maximum_dry_cell_count": int(np.max(dry_cells)),
            }
        )
        checks.update(
            {
                "swe_health_history_complete": {
                    min_depth.size, max_speed.size, dry_cells.size
                }
                == {total_steps},
                "swe_health_finite": bool(
                    np.isfinite(min_depth).all() and np.isfinite(max_speed).all()
                ),
                "swe_minimum_depth_within_tolerance": bool(
                    np.min(min_depth) >= float(health_gates["min_h_tolerance"])
                ),
                "swe_velocity_within_limit": bool(
                    np.max(max_speed)
                    <= float(health_gates["max_velocity_limit"])
                    * (1.0 + 1.0e-12)
                ),
                "swe_dry_cell_counts_valid": bool(
                    np.all(dry_cells >= 0)
                    and np.all(
                        dry_cells
                        <= int(candidate["computational_grid"]) ** 2
                    )
                ),
            }
        )
    else:
        cg_failed = np.asarray(diagnostics["cg_failed_count"], dtype=np.int64)
        step_converged = np.asarray(
            diagnostics["cg_step_converged"], dtype=np.bool_
        )
        solve0_converged = np.asarray(
            diagnostics["cg_solve0_converged"], dtype=np.bool_
        )
        solve1_converged = np.asarray(
            diagnostics["cg_solve1_converged"], dtype=np.bool_
        )
        solve0_iterations = np.asarray(
            diagnostics["cg_solve0_iterations"], dtype=np.int64
        )
        solve1_iterations = np.asarray(
            diagnostics["cg_solve1_iterations"], dtype=np.int64
        )
        residual_keys = (
            "cg_max_residual_ratio",
            "cg_solve0_initial_residual",
            "cg_solve0_final_residual",
            "cg_solve0_residual_ratio",
            "cg_solve1_initial_residual",
            "cg_solve1_final_residual",
            "cg_solve1_residual_ratio",
        )
        residual_arrays = [
            np.asarray(diagnostics[key], dtype=np.float64) for key in residual_keys
        ]
        metrics.update(
            {
                "cg_failure_count": int(np.sum(cg_failed)),
                "cg_iterations_max": int(
                    max(np.max(solve0_iterations), np.max(solve1_iterations))
                ),
                "cg_residual_ratio_max": float(
                    max(
                        np.max(np.asarray(diagnostics["cg_solve0_residual_ratio"])),
                        np.max(np.asarray(diagnostics["cg_solve1_residual_ratio"])),
                    )
                ),
            }
        )
        checks.update(
            {
                "boussinesq_cg_history_complete": all(
                    values.size == total_steps
                    for values in (
                        cg_failed,
                        step_converged,
                        solve0_converged,
                        solve1_converged,
                        solve0_iterations,
                        solve1_iterations,
                        *residual_arrays,
                    )
                ),
                "boussinesq_cg_residuals_finite_nonnegative": all(
                    np.isfinite(values).all() and np.all(values >= 0.0)
                    for values in residual_arrays
                ),
                "boussinesq_cg_all_converged": bool(
                    np.all(cg_failed == 0)
                    and np.all(step_converged)
                    and np.all(solve0_converged)
                    and np.all(solve1_converged)
                ),
                "boussinesq_cg_iterations_valid": bool(
                    np.all(solve0_iterations >= 0)
                    and np.all(solve1_iterations >= 0)
                    and np.all(
                        solve0_iterations
                        <= int(candidate["boussinesq_linear_solver_max_iter"])
                    )
                    and np.all(
                        solve1_iterations
                        <= int(candidate["boussinesq_linear_solver_max_iter"])
                    )
                ),
                "boussinesq_filter_disabled": bool(
                    not np.any(np.asarray(diagnostics["filter_enabled"], dtype=np.bool_))
                    and np.all(
                        np.asarray(
                            diagnostics["filter_application_count"], dtype=np.int64
                        )
                        == 0
                    )
                ),
            }
        )
    return checks, metrics


def _run_task(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    health_gates: Mapping[str, Any],
    contract_hash: str,
) -> dict[str, Any]:
    started = time.monotonic()
    row, trajectory, details = run_buffered_case_detailed(
        record,
        solver_name=str(task["solver"]),
        total_grid=int(candidate["computational_grid"]),
        core_grid=int(candidate["publication_grid"]),
        source_taper_cells=int(candidate["source_taper_cells"]),
        sponge_min_factor=float(candidate["sponge_min_factor"]),
        sponge_width_cells=int(candidate["sponge_width_cells"]),
    )
    bathymetry, _source, _strength_array, _strength, arrays = _load_canary_arrays(
        record
    )
    expected_times = candidate_requested_times()
    target_cfl = (
        float(candidate["boussinesq_cfl"])
        if task["solver"] == "boussinesq"
        else float(candidate["swe_cfl"])
    )
    provenance_checks, provenance, natural_health, diagnostic_evidence = (
        _summarize_diagnostics(
            solver=str(task["solver"]),
            details=details,
            expected_times=expected_times,
            target_cfl=target_cfl,
        )
    )
    solver_checks, health_metrics = _solver_health_checks(
        solver=str(task["solver"]),
        trajectory=trajectory,
        diagnostics=details["diagnostics"],
        arrays=arrays,
        bathymetry=bathymetry,
        health_gates=health_gates,
        candidate=candidate,
    )
    expected_boundary = "open" if task["solver"] == "boussinesq" else "radiation"
    identity_checks = {
        "qualified_id_matches": row["qualified_id"] == task["qualified_id"],
        "input_fingerprint_matches": row["input_fingerprint"]
        == task["input_fingerprint"],
        "solver_matches": row["solver"] == task["solver"],
        "candidate_grid_matches": (
            int(row["total_grid"]) == int(candidate["computational_grid"])
            and int(row["core_grid"]) == int(candidate["publication_grid"])
            and int(row["buffer_cells"]) == int(candidate["buffer_cells_per_side"])
        ),
        "candidate_source_taper_matches": int(row["source_taper_cells"])
        == int(candidate["source_taper_cells"]),
        "candidate_sponge_matches": (
            int(row["sponge_width_cells"])
            == int(candidate["sponge_width_cells"])
            and float(row["sponge_core_min"]) == 1.0
        ),
        "candidate_boundary_matches": row["outer_boundary"] == expected_boundary,
        "source_edge_is_exact_zero": float(row["source_edge_max_abs"]) == 0.0,
        "full_requested_state_float64": details["full_requested_state_hash"]["dtype"]
        == np.dtype(np.float64).str,
        "row_reports_float64": row["health"]["measurement_dtype"] == "float64",
        "row_reports_finite": bool(row["health"]["finite"]),
    }
    checks = {**identity_checks, **provenance_checks, **solver_checks}
    array_hashes = {
        "cropped_eta_trajectory": hash_array(trajectory),
        "full_requested_state": details["full_requested_state_hash"],
        "diagnostics": diagnostic_evidence["diagnostic_array_hashes"],
    }
    scientific_digest = stable_hash_payload(
        artifact_kind="common-time-v2-h1-scientific-task-digest",
        payload={
            "qualified_id": task["qualified_id"],
            "input_fingerprint": task["input_fingerprint"],
            "solver": task["solver"],
            "candidate_config_hash": task["candidate_config_hash"],
            "array_hashes": array_hashes,
            "health_metrics": health_metrics,
            "checks": checks,
        },
        schema_id=SCHEMA_ID,
    )
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h1-task-result",
        "contract_hash": str(contract_hash),
        "task_id": str(task["task_id"]),
        "ordinal": int(task["ordinal"]),
        "run_kind": str(task["run_kind"]),
        "reference_primary_task_id": task["reference_primary_task_id"],
        "qualified_id": str(task["qualified_id"]),
        "input_fingerprint": str(task["input_fingerprint"]),
        "solver": str(task["solver"]),
        "candidate_config_hash": str(task["candidate_config_hash"]),
        "passed": bool(all(checks.values())),
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "checks": checks,
        "health_metrics": health_metrics,
        "diagnostic_summary": diagnostic_evidence["summary"],
        "requested_output_provenance": provenance,
        "natural_step_health": natural_health,
        "array_hashes": array_hashes,
        "scientific_digest": scientific_digest,
        "runtime_s": float(time.monotonic() - started),
        "worker": {
            "pid": os.getpid(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "thread_environment": {
                key: os.environ.get(key) for key in THREAD_ENV_KEYS
            },
        },
    }
    identity = dict(payload)
    payload["result_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-h1-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    return payload


def _task_path(root: Path, task: Mapping[str, Any]) -> Path:
    return (
        root
        / "execution"
        / "tasks"
        / f"{int(task['ordinal']):04d}-{task['task_id']}.json"
    )


def _validate_task_result(
    payload: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    contract_hash: str,
) -> None:
    for key in (
        "task_id",
        "ordinal",
        "run_kind",
        "qualified_id",
        "input_fingerprint",
        "solver",
        "candidate_config_hash",
        "reference_primary_task_id",
    ):
        if payload.get(key) != task.get(key):
            raise RuntimeError(f"H1 task result identity mismatch for {key}")
    if payload.get("contract_hash") != contract_hash:
        raise RuntimeError("H1 task result contract hash mismatch")
    identity = dict(payload)
    recorded_hash = identity.pop("result_hash", None)
    expected_hash = stable_hash_payload(
        artifact_kind="common-time-v2-h1-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    if recorded_hash != expected_hash:
        raise RuntimeError(f"H1 task result hash mismatch: {task['task_id']}")
    if payload.get("passed") != (not payload.get("failed_checks")):
        raise RuntimeError(f"H1 task pass/failure inconsistency: {task['task_id']}")


def _load_completed(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    tasks = contract["tasks"]
    expected_paths = {_task_path(root, task): task for task in tasks}
    tasks_root = root / "execution" / "tasks"
    if tasks_root.exists():
        unexpected = {
            path
            for path in tasks_root.iterdir()
            if path.is_file() and path not in expected_paths
        }
        if unexpected:
            raise RuntimeError(
                f"Unexpected H1 task result files: {sorted(map(str, unexpected))}"
            )
    completed: dict[str, dict[str, Any]] = {}
    for path, task in expected_paths.items():
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_task_result(
            payload, task=task, contract_hash=str(contract["contract_hash"])
        )
        completed[str(task["task_id"])] = payload
    return completed


def _verify_execution_environment(
    contract: Mapping[str, Any], *, repo_root: Path
) -> None:
    current_code = code_state(repo_root)
    if current_code["code_state_hash"] != contract["code_state"]["code_state_hash"]:
        raise RuntimeError("Code state changed after H1 freeze; create a new H1 contract")
    current_environment = _environment_snapshot()
    frozen_environment = contract["environment"]
    for key in (
        "python_version",
        "python_executable",
        "numpy_version",
        "platform",
        "machine",
        "package_inventory_hash",
    ):
        if current_environment[key] != frozen_environment[key]:
            raise RuntimeError(f"H1 execution environment changed: {key}")


def _execution_summary(
    ordered: Sequence[Mapping[str, Any]],
    replay_mismatches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary = [row for row in ordered if row["run_kind"] == "primary"]
    by_solver: dict[str, Any] = {}
    for solver in SOLVERS:
        rows = [row for row in primary if row["solver"] == solver]
        by_solver[solver] = {
            "task_count": len(rows),
            "passed_count": sum(bool(row["passed"]) for row in rows),
            "failed_count": sum(not bool(row["passed"]) for row in rows),
            "runtime_s_sum": float(sum(float(row["runtime_s"]) for row in rows)),
            "natural_steps_min": min(
                int(row["diagnostic_summary"]["total_natural_steps"]) for row in rows
            ),
            "natural_steps_max": max(
                int(row["diagnostic_summary"]["total_natural_steps"]) for row in rows
            ),
            "max_abs_eta": max(float(row["health_metrics"]["max_abs_eta"]) for row in rows),
            "max_eta_over_depth": max(
                float(row["health_metrics"]["max_eta_over_depth"]) for row in rows
            ),
            "max_post_step_cfl": max(
                float(row["diagnostic_summary"]["max_post_step_cfl"]) for row in rows
            ),
        }
        if solver in {"swe_hydrostatic", "swe_muscl_hr"}:
            by_solver[solver].update(
                {
                    "minimum_depth": min(
                        float(row["health_metrics"]["minimum_depth"]) for row in rows
                    ),
                    "maximum_speed": max(
                        float(row["health_metrics"]["maximum_speed"]) for row in rows
                    ),
                    "maximum_dry_cell_count": max(
                        int(row["health_metrics"]["maximum_dry_cell_count"])
                        for row in rows
                    ),
                }
            )
        else:
            by_solver[solver].update(
                {
                    "cg_failure_count": sum(
                        int(row["health_metrics"]["cg_failure_count"]) for row in rows
                    ),
                    "cg_iterations_max": max(
                        int(row["health_metrics"]["cg_iterations_max"]) for row in rows
                    ),
                    "cg_residual_ratio_max": max(
                        float(row["health_metrics"]["cg_residual_ratio_max"])
                        for row in rows
                    ),
                }
            )
    return {
        "by_solver": by_solver,
        "replay_count": sum(row["run_kind"] == "replay" for row in ordered),
        "replay_mismatch_count": len(replay_mismatches),
    }


def _find_replay_mismatches(
    ordered: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_task_id = {str(payload["task_id"]): payload for payload in ordered}
    mismatches: list[dict[str, Any]] = []
    for payload in ordered:
        if payload["run_kind"] != "replay":
            continue
        reference_id = str(payload["reference_primary_task_id"])
        if reference_id not in by_task_id:
            raise RuntimeError(f"H1 replay reference is missing: {reference_id}")
        reference = by_task_id[reference_id]
        if (
            reference["run_kind"] != "primary"
            or reference["qualified_id"] != payload["qualified_id"]
            or reference["solver"] != payload["solver"]
        ):
            raise RuntimeError(f"H1 replay reference identity mismatch: {reference_id}")
        if payload["scientific_digest"] != reference["scientific_digest"]:
            mismatches.append(
                {
                    "replay_task_id": payload["task_id"],
                    "primary_task_id": reference["task_id"],
                    "qualified_id": payload["qualified_id"],
                    "solver": payload["solver"],
                    "primary_scientific_digest": reference["scientific_digest"],
                    "replay_scientific_digest": payload["scientific_digest"],
                }
            )
    return mismatches


def _report_text(result: Mapping[str, Any]) -> str:
    lines = [
        "# Common-time-v2 H1 result",
        "",
        f"- Contract: `{result['contract_hash']}`",
        f"- Decision: `{result['decision']}`",
        f"- Tasks: {result['task_count']} ({result['primary_task_count']} primary, "
        f"{result['replay_task_count']} replay)",
        f"- Failed health tasks: {len(result['failed_task_ids'])}",
        f"- Replay mismatches: {len(result['replay_mismatches'])}",
        f"- Wall duration: {float(result['wall_duration_s']):.1f} s",
        "",
        "H1 is an implementation and long-horizon health smoke. It is not "
        "scientific acceptance and does not inspect validation or test outcomes.",
        "",
    ]
    for solver, summary in result["summary"]["by_solver"].items():
        lines.append(
            f"- `{solver}`: {summary['passed_count']}/{summary['task_count']} "
            f"primary tasks passed; natural steps "
            f"{summary['natural_steps_min']}–{summary['natural_steps_max']}."
        )
    return "\n".join(lines) + "\n"


def execute_h1_contract(
    *,
    repo_root: Path,
    contract_root: Path,
    workers: int,
    max_in_flight: int | None,
    resume: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    contract_root = contract_root.resolve()
    validate_frozen_checksums(contract_root)
    contract = _read_json(contract_root / "preregistered_contract.json")
    _validate_contract_identity(contract)
    if Path(contract["config_path"]).resolve().parents[2] != repo_root:
        raise RuntimeError("H1 contract repository root mismatch")
    _verify_execution_environment(contract, repo_root=repo_root)
    _verify_prerequisites(
        config=contract["resolved_config"],
        h0_root=Path(contract["prerequisites"]["h0"]["root"]),
        level_a_root=Path(contract["prerequisites"]["level_a"]["root"]),
        level_b_bundle_root=Path(contract["prerequisites"]["level_b"]["bundle_root"]),
        level_b_evaluation_root=Path(
            contract["prerequisites"]["level_b"]["evaluation_root"]
        ),
    )

    worker_policy = contract["worker_policy"]
    frozen_workers = int(worker_policy["requested_workers"])
    frozen_in_flight = int(worker_policy["requested_max_in_flight"])
    effective_in_flight = frozen_in_flight if max_in_flight is None else max_in_flight
    if workers != frozen_workers or effective_in_flight != frozen_in_flight:
        raise RuntimeError(
            f"H1 requires frozen workers/max-in-flight "
            f"{frozen_workers}/{frozen_in_flight}"
        )
    if workers <= 0 or effective_in_flight < workers:
        raise ValueError("H1 workers/max-in-flight must be positive and bounded")

    execution_root = contract_root / "execution"
    result_path = execution_root / "result.json"
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"H1 result already finalized: {result_path}")
        validate_execution_checksums(execution_root)
        return result_path

    tasks = contract["tasks"]
    selected_by_ordinal = {
        int(entry["selection_ordinal"]): entry["record"]
        for entry in contract["selected_scenarios"]
    }
    completed = _load_completed(contract_root, contract)
    if completed and not resume:
        raise FileExistsError("Partial H1 results exist; rerun with --resume")
    pending = [task for task in tasks if str(task["task_id"]) not in completed]
    (execution_root / "tasks").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if progress is not None:
        progress(
            {
                "event": "start",
                "completed": len(completed),
                "total": len(tasks),
                "pending": len(pending),
                "workers": workers,
                "max_in_flight": effective_in_flight,
            }
        )

    context = multiprocessing.get_context(str(worker_policy["process_start_method"]))
    pending_iter = iter(pending)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        active: dict[Any, Mapping[str, Any]] = {}

        def submit_until_full() -> None:
            while len(active) < effective_in_flight:
                try:
                    task = next(pending_iter)
                except StopIteration:
                    break
                record = selected_by_ordinal[int(task["selection_ordinal"])]
                future = executor.submit(
                    _run_task,
                    task,
                    record,
                    contract["resolved_config"]["candidate"],
                    contract["resolved_config"]["health_gates"],
                    contract["contract_hash"],
                )
                active[future] = task

        submit_until_full()
        while active:
            done, _not_done = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                payload = future.result()
                _validate_task_result(
                    payload,
                    task=task,
                    contract_hash=str(contract["contract_hash"]),
                )
                _write_json(_task_path(contract_root, task), payload)
                completed[str(task["task_id"])] = payload
                if progress is not None:
                    progress(
                        {
                            "event": "task_complete",
                            "completed": len(completed),
                            "total": len(tasks),
                            "run_kind": task["run_kind"],
                            "qualified_id": task["qualified_id"],
                            "solver": task["solver"],
                            "runtime_s": payload["runtime_s"],
                            "passed": payload["passed"],
                            "elapsed_s": time.monotonic() - started,
                        }
                    )
            submit_until_full()

    ordered = [completed[str(task["task_id"])] for task in tasks]
    failed_task_ids = [
        str(payload["task_id"]) for payload in ordered if not payload["passed"]
    ]
    replay_mismatches = _find_replay_mismatches(ordered)
    decision = (
        contract["decision_policy"]["pass"]
        if not failed_task_ids and not replay_mismatches
        else contract["decision_policy"]["health_failure"]
    )
    result: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h1-result",
        "contract_hash": contract["contract_hash"],
        "decision": decision,
        "h1_passed": decision == contract["decision_policy"]["pass"],
        "claim_scope": contract["claim_scope"],
        "task_count": len(ordered),
        "primary_task_count": sum(row["run_kind"] == "primary" for row in ordered),
        "replay_task_count": sum(row["run_kind"] == "replay" for row in ordered),
        "failed_task_ids": failed_task_ids,
        "replay_mismatches": replay_mismatches,
        "wall_duration_s": float(time.monotonic() - started),
        "sum_task_runtime_s": float(sum(float(row["runtime_s"]) for row in ordered)),
        "effective_workers": workers,
        "effective_max_in_flight": effective_in_flight,
        "summary": _execution_summary(ordered, replay_mismatches),
        "task_rows": [
            {
                "ordinal": row["ordinal"],
                "task_id": row["task_id"],
                "run_kind": row["run_kind"],
                "qualified_id": row["qualified_id"],
                "solver": row["solver"],
                "passed": row["passed"],
                "failed_checks": row["failed_checks"],
                "scientific_digest": row["scientific_digest"],
                "runtime_s": row["runtime_s"],
                "health_metrics": row["health_metrics"],
                "diagnostic_summary": row["diagnostic_summary"],
            }
            for row in ordered
        ],
        "operational_provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "process_start_method": str(worker_policy["process_start_method"]),
            "thread_environment": {
                key: os.environ.get(key) for key in THREAD_ENV_KEYS
            },
            "workers": sorted(
                {
                    (
                        int(row["worker"]["pid"]),
                        str(row["worker"]["python_version"]),
                        str(row["worker"]["numpy_version"]),
                        json.dumps(
                            row["worker"]["thread_environment"], sort_keys=True
                        ),
                    )
                    for row in ordered
                }
            ),
        },
        "validation_and_test_scientific_outcomes_inspected": False,
        "h2_executed": False,
        "mass_generation_authorized": False,
    }
    _write_json(result_path, result)
    _write_json(
        execution_root / "decision.json",
        {
            "schema_id": SCHEMA_ID,
            "contract_hash": contract["contract_hash"],
            "decision": decision,
            "h1_passed": result["h1_passed"],
            "failed_task_ids": failed_task_ids,
            "replay_mismatch_count": len(replay_mismatches),
            "h2_executed": False,
            "mass_generation_authorized": False,
        },
    )
    _write_text(execution_root / "REPORT.md", _report_text(result))
    _write_execution_checksums(execution_root)
    validate_execution_checksums(execution_root)
    if progress is not None:
        progress(
            {
                "event": "finalized",
                "completed": len(ordered),
                "total": len(tasks),
                "decision": decision,
                "passed": result["h1_passed"],
                "duration_s": result["wall_duration_s"],
            }
        )
    return result_path


def h1_status(contract_root: Path) -> dict[str, Any]:
    contract_root = contract_root.resolve()
    validate_frozen_checksums(contract_root)
    contract = _read_json(contract_root / "preregistered_contract.json")
    _validate_contract_identity(contract)
    completed = _load_completed(contract_root, contract)
    result_path = contract_root / "execution" / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    return {
        "contract_hash": contract["contract_hash"],
        "selected_scenario_count": len(contract["selected_scenarios"]),
        "completed": len(completed),
        "total": len(contract["tasks"]),
        "pending": len(contract["tasks"]) - len(completed),
        "failed_completed_tasks": sorted(
            task_id
            for task_id, payload in completed.items()
            if not payload["passed"]
        ),
        "finalized": result is not None,
        "decision": None if result is None else result["decision"],
    }
