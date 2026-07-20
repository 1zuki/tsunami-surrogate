from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    candidate_requested_times,
    code_state,
    stable_hash_payload,
    sha256_file,
)
from src.evaluation.common_time_v2_h1 import THREAD_ENV_KEYS
from src.evaluation.common_time_v2_h2 import (
    _distribution,
    _environment_snapshot,
    _read_json,
    _run_variant,
    _write_json,
    _write_text,
    paired_cfl_metrics,
    validate_h2_checksums,
)
from src.evaluation.common_time_v2_level_a import _load_canary_arrays


SCHEMA_ID = "tsunami-surrogate.common-time-v2.h2-swe-cfl-diagnostic.v1"
CONFIG_SCHEMA_ID = (
    "tsunami-surrogate.common-time-v2.h2-swe-cfl-diagnostic-config.v1"
)
FROZEN_FILENAMES = ("diagnostic_contract.json", "task_plan.json")
FROZEN_CHECKSUMS = "CONTRACT_SHA256SUMS.txt"
COMPARISON_NAMES = (
    "production_to_half",
    "half_to_quarter",
    "production_to_quarter",
)
VARIANT_NAMES = ("production", "half", "quarter")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a YAML object: {path}")
    if payload.get("schema_id") != CONFIG_SCHEMA_ID:
        raise RuntimeError("SWE CFL diagnostic configuration schema mismatch")
    if payload.get("decision_role") != "non_decisional_post_h2_diagnosis":
        raise RuntimeError("SWE CFL diagnostic must remain non-decisional")
    if payload.get("selection_is_outcome_targeted") is not True:
        raise RuntimeError("Outcome-targeted case selection must be disclosed")
    if payload.get("validation_or_test_outcomes_inspected") is not False:
        raise RuntimeError("Validation/test scientific outcomes must remain excluded")
    if payload.get("solvers") != ["swe_hydrostatic", "swe_muscl_hr"]:
        raise RuntimeError("SWE CFL diagnostic solver list changed")
    if payload["analysis"].get("comparisons") != list(COMPARISON_NAMES):
        raise RuntimeError("SWE CFL diagnostic comparison list changed")
    if len(payload.get("cases", [])) != 8:
        raise RuntimeError("SWE CFL diagnostic must freeze exactly eight cases")
    qualified_ids = [str(case["qualified_id"]) for case in payload["cases"]]
    fingerprints = [str(case["input_fingerprint"]) for case in payload["cases"]]
    if len(set(qualified_ids)) != len(qualified_ids):
        raise RuntimeError("SWE CFL diagnostic case IDs must be unique")
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("SWE CFL diagnostic fingerprints must be unique")
    for solver in payload["solvers"]:
        ladder = [float(value) for value in payload["cfl_ladder"][solver]]
        if len(ladder) != 3 or not (
            math.isclose(ladder[1], 0.5 * ladder[0])
            and math.isclose(ladder[2], 0.5 * ladder[1])
        ):
            raise RuntimeError(f"Invalid production/half/quarter ladder for {solver}")
    execution = payload["execution"]
    if int(execution["workers"]) <= 0:
        raise RuntimeError("SWE CFL diagnostic workers must be positive")
    if int(execution["max_in_flight"]) < int(execution["workers"]):
        raise RuntimeError("SWE CFL diagnostic max-in-flight must cover workers")
    return payload


def _contract_hash(contract: Mapping[str, Any]) -> str:
    identity = dict(contract)
    identity.pop("study_hash", None)
    return stable_hash_payload(
        artifact_kind="common-time-v2-h2-swe-cfl-diagnostic-contract",
        payload=identity,
        schema_id=SCHEMA_ID,
    )


def _task_identity(
    *,
    ordinal: int,
    solver: str,
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    source_h2_task: Mapping[str, Any],
    cfl_ladder: Sequence[float],
    candidate_config_hash: str,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "ordinal": int(ordinal),
        "solver": str(solver),
        "qualified_id": str(case["qualified_id"]),
        "input_fingerprint": str(case["input_fingerprint"]),
        "bathymetry_type": str(case["bathymetry_type"]),
        "source_type": str(case["source_type"]),
        "selection_role": str(case["selection_role"]),
        "record": dict(record),
        "cfl_ladder": [float(value) for value in cfl_ladder],
        "candidate_config_hash": str(candidate_config_hash),
        "source_h2_task_id": str(source_h2_task["task_id"]),
        "source_h2_scientific_digest": str(source_h2_task["scientific_digest"]),
        "source_h2_production_to_half_metrics": dict(source_h2_task["metrics"]),
    }
    identity = {
        key: value
        for key, value in task.items()
        if key
        not in {
            "record",
            "source_h2_production_to_half_metrics",
        }
    }
    task["task_id"] = stable_hash_payload(
        artifact_kind="common-time-v2-h2-swe-cfl-diagnostic-task",
        payload=identity,
        schema_id=SCHEMA_ID,
    )[:24]
    return task


def _build_tasks(
    config: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_by_id = {
        str(entry["record"]["qualified_id"]): entry["record"]
        for entry in source_contract["selected_scenarios"]
    }
    source_rows = {
        (str(row["qualified_id"]), str(row["solver"])): row
        for row in source_result["task_rows"]
        if row["run_kind"] == "primary"
    }
    tasks: list[dict[str, Any]] = []
    ordinal = 0
    for case in config["cases"]:
        qualified_id = str(case["qualified_id"])
        if qualified_id not in selected_by_id:
            raise RuntimeError(
                f"Diagnostic case is absent from the source H2 selection: {qualified_id}"
            )
        record = selected_by_id[qualified_id]
        for field in (
            "input_fingerprint",
            "bathymetry_type",
            "source_type",
        ):
            if str(record[field]) != str(case[field]):
                raise RuntimeError(
                    f"Diagnostic case {qualified_id} disagrees with source H2: {field}"
                )
        for solver in config["solvers"]:
            source_key = (qualified_id, str(solver))
            if source_key not in source_rows:
                raise RuntimeError(f"Missing source H2 task row: {source_key}")
            tasks.append(
                _task_identity(
                    ordinal=ordinal,
                    solver=str(solver),
                    case=case,
                    record=record,
                    source_h2_task=source_rows[source_key],
                    cfl_ladder=config["cfl_ladder"][solver],
                    candidate_config_hash=str(
                        source_contract["candidate_config_hash"]
                    ),
                )
            )
            ordinal += 1
    return tasks


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("SWE CFL diagnostic contract schema mismatch")
    if contract.get("study_hash") != _contract_hash(contract):
        raise RuntimeError("SWE CFL diagnostic content hash mismatch")
    tasks = contract.get("tasks", [])
    if len(tasks) != 16:
        raise RuntimeError("SWE CFL diagnostic must contain exactly 16 tasks")
    if [int(task["ordinal"]) for task in tasks] != list(range(16)):
        raise RuntimeError("SWE CFL diagnostic task ordering changed")
    if len({str(task["task_id"]) for task in tasks}) != 16:
        raise RuntimeError("SWE CFL diagnostic task IDs are not unique")
    if set(str(task["solver"]) for task in tasks) != {
        "swe_hydrostatic",
        "swe_muscl_hr",
    }:
        raise RuntimeError("SWE CFL diagnostic task solver coverage changed")
    if contract.get("decision_role") != "non_decisional_post_h2_diagnosis":
        raise RuntimeError("SWE CFL diagnostic contract cannot be decisional")


def _write_frozen_checksums(root: Path) -> None:
    rows = [f"{sha256_file(root / name)}  {name}" for name in FROZEN_FILENAMES]
    _write_text(root / FROZEN_CHECKSUMS, "\n".join(rows) + "\n")


def validate_frozen_checksums(root: Path) -> None:
    manifest = root / FROZEN_CHECKSUMS
    if not manifest.is_file():
        raise RuntimeError(f"Missing diagnostic checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        if relative not in FROZEN_FILENAMES:
            raise RuntimeError(f"Unexpected diagnostic checksum entry: {relative}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Diagnostic frozen checksum mismatch: {relative}")
        listed.add(relative)
    if listed != set(FROZEN_FILENAMES):
        raise RuntimeError("Diagnostic frozen checksum inventory is incomplete")


def _write_execution_checksums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    _write_text(root / "SHA256SUMS.txt", "\n".join(rows) + "\n")


def validate_execution_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise RuntimeError(f"Missing diagnostic execution manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Diagnostic execution checksum mismatch: {relative}")
        listed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if listed != actual:
        raise RuntimeError("Diagnostic execution checksum inventory mismatch")


def validate_diagnostic_checksums(root: Path) -> None:
    validate_frozen_checksums(root)
    execution_root = root / "execution"
    if (execution_root / "result.json").is_file():
        validate_execution_checksums(execution_root)
    elif (execution_root / "SHA256SUMS.txt").exists():
        raise RuntimeError("Partial diagnostic execution has a final manifest")


def freeze_diagnostic(
    *,
    repo_root: Path,
    config_path: Path,
    source_h2_root: Path,
    output_base: Path,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    source_h2_root = source_h2_root.resolve()
    output_base = output_base.resolve()
    config = _load_config(config_path)
    validate_h2_checksums(source_h2_root)
    source_contract = _read_json(source_h2_root / "preregistered_contract.json")
    source_result = _read_json(source_h2_root / "execution" / "result.json")
    source_hash = str(config["source_h2_contract_hash"])
    if (
        source_contract.get("contract_hash") != source_hash
        or source_h2_root.name != source_hash
    ):
        raise RuntimeError("Configured source H2 contract hash mismatch")
    if source_result.get("contract_hash") != source_hash:
        raise RuntimeError("Source H2 result/contract mismatch")
    if source_result.get("decision") != config["source_h2_decision"]:
        raise RuntimeError("Source H2 decision changed")

    tasks = _build_tasks(config, source_contract, source_result)
    contract: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-swe-cfl-diagnostic-contract",
        "study": str(config["study"]),
        "status": "frozen_unexecuted",
        "decision_role": str(config["decision_role"]),
        "selection_is_outcome_targeted": True,
        "validation_or_test_outcomes_inspected": False,
        "source_h2": {
            "root": str(source_h2_root),
            "contract_hash": source_hash,
            "decision": str(source_result["decision"]),
            "contract_sha256": sha256_file(
                source_h2_root / "preregistered_contract.json"
            ),
            "result_sha256": sha256_file(
                source_h2_root / "execution" / "result.json"
            ),
            "checksums_validated_before_freeze": True,
        },
        "code_state": code_state(repo_root),
        "environment": _environment_snapshot(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config,
        "candidate": dict(source_contract["resolved_config"]["candidate"]),
        "health_gates": dict(source_contract["resolved_config"]["health_gates"]),
        "metrics_config": dict(source_contract["resolved_config"]["metrics"]),
        "source_h2_thresholds": {
            solver: dict(source_contract["resolved_config"]["thresholds"][solver])
            for solver in config["solvers"]
        },
        "candidate_config_hash": str(source_contract["candidate_config_hash"]),
        "worker_policy": dict(config["execution"]),
        "tasks": tasks,
        "interpretation": {
            "purpose": (
                "diagnose whether the failed production-to-half SWE sensitivity "
                "contracts under half-to-quarter CFL refinement"
            ),
            "not_a_replacement_h2": True,
            "cannot_authorize_mass_generation": True,
            "outcome_targeted_sample_cannot_establish_population_pass": True,
            "thresholds_are_replayed_for_screening_only": True,
        },
    }
    contract["study_hash"] = _contract_hash(contract)
    _validate_contract(contract)
    output_root = output_base / str(contract["study_hash"])
    task_plan = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-swe-cfl-diagnostic-task-plan",
        "study_hash": contract["study_hash"],
        "task_count": len(tasks),
        "solver_execution_count": len(VARIANT_NAMES) * len(tasks),
        "tasks": tasks,
    }
    if output_root.exists():
        path = output_root / "diagnostic_contract.json"
        if not path.is_file() or _read_json(path) != contract:
            raise FileExistsError(
                f"Refusing to replace a different diagnostic: {output_root}"
            )
        validate_frozen_checksums(output_root)
        return output_root
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "diagnostic_contract.json", contract)
    _write_json(output_root / "task_plan.json", task_plan)
    _write_frozen_checksums(output_root)
    validate_frozen_checksums(output_root)
    return output_root


def observed_order(
    production_to_half_absolute_rms: float,
    half_to_quarter_absolute_rms: float,
) -> float | None:
    coarse = float(production_to_half_absolute_rms)
    fine = float(half_to_quarter_absolute_rms)
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return math.log2(coarse / fine)


def _numeric_mismatches(
    current: Any,
    frozen: Any,
    *,
    rtol: float,
    atol: float,
    path: str = "metrics",
) -> list[str]:
    if isinstance(current, bool) or isinstance(frozen, bool):
        return [] if current is frozen else [path]
    if current is None or frozen is None:
        return [] if current is frozen else [path]
    if isinstance(current, Mapping) and isinstance(frozen, Mapping):
        if set(current) != set(frozen):
            return [f"{path}.keys"]
        mismatches: list[str] = []
        for key in sorted(current):
            mismatches.extend(
                _numeric_mismatches(
                    current[key],
                    frozen[key],
                    rtol=rtol,
                    atol=atol,
                    path=f"{path}.{key}",
                )
            )
        return mismatches
    if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        if not isinstance(frozen, Sequence) or isinstance(frozen, (str, bytes)):
            return [path]
        if len(current) != len(frozen):
            return [f"{path}.length"]
        mismatches = []
        for index, (left, right) in enumerate(zip(current, frozen)):
            mismatches.extend(
                _numeric_mismatches(
                    left,
                    right,
                    rtol=rtol,
                    atol=atol,
                    path=f"{path}[{index}]",
                )
            )
        return mismatches
    if isinstance(current, (int, float)) and isinstance(frozen, (int, float)):
        return (
            []
            if math.isclose(float(current), float(frozen), rel_tol=rtol, abs_tol=atol)
            else [path]
        )
    return [] if current == frozen else [path]


def _metrics(
    left: np.ndarray,
    right: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return paired_cfl_metrics(
        left,
        right,
        relative_floor_absolute_rms=float(
            config["relative_floor_absolute_rms"]
        ),
        phase_activity_floor_absolute_rms=float(
            config["phase_activity_floor_absolute_rms"]
        ),
        boundary_band_cells=int(config["boundary_band_cells"]),
    )


def _finite_metric_scalars(metrics: Mapping[str, Any]) -> bool:
    for key, value in metrics.items():
        if value is None or isinstance(value, (bool, str, list, dict)):
            continue
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            return False
    return True


def _run_task(
    task: Mapping[str, Any],
    candidate: Mapping[str, Any],
    health_gates: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
    analysis: Mapping[str, Any],
    study_hash: str,
) -> dict[str, Any]:
    started = time.monotonic()
    record = task["record"]
    bathymetry, _source, _strength_array, _strength, arrays = _load_canary_arrays(
        record
    )
    trajectories: dict[str, np.ndarray] = {}
    variants: dict[str, dict[str, Any]] = {}
    for name, target_cfl in zip(VARIANT_NAMES, task["cfl_ladder"]):
        trajectory, evidence = _run_variant(
            record=record,
            solver=str(task["solver"]),
            candidate=candidate,
            health_gates=health_gates,
            target_cfl=float(target_cfl),
            bathymetry=bathymetry,
            arrays=arrays,
        )
        trajectories[name] = trajectory
        variants[name] = evidence

    comparisons = {
        "production_to_half": _metrics(
            trajectories["production"], trajectories["half"], metrics_config
        ),
        "half_to_quarter": _metrics(
            trajectories["half"], trajectories["quarter"], metrics_config
        ),
        "production_to_quarter": _metrics(
            trajectories["production"], trajectories["quarter"], metrics_config
        ),
    }
    coarse_absolute = float(
        comparisons["production_to_half"]["trajectory_absolute_rms"]
    )
    fine_absolute = float(
        comparisons["half_to_quarter"]["trajectory_absolute_rms"]
    )
    contraction_ratio = (
        fine_absolute / coarse_absolute if coarse_absolute > 0.0 else None
    )
    order = observed_order(coarse_absolute, fine_absolute)
    per_time_order = [
        observed_order(coarse, fine)
        for coarse, fine in zip(
            comparisons["production_to_half"]["per_time_absolute_rms"],
            comparisons["half_to_quarter"]["per_time_absolute_rms"],
        )
    ]
    rtol = float(analysis["h2_metric_replay_relative_tolerance"])
    atol = float(analysis["h2_metric_replay_absolute_tolerance"])
    replay_mismatches = _numeric_mismatches(
        comparisons["production_to_half"],
        task["source_h2_production_to_half_metrics"],
        rtol=rtol,
        atol=atol,
    )
    timestamps = [
        variants[name]["requested_output_provenance"]["requested_timestamps"]
        for name in VARIANT_NAMES
    ]
    shapes = [trajectories[name].shape for name in VARIANT_NAMES]
    checks = {
        "all_variants_healthy": all(
            bool(variants[name]["passed"]) for name in VARIANT_NAMES
        ),
        "requested_times_exactly_equal": (
            timestamps[0]
            == timestamps[1]
            == timestamps[2]
            == candidate_requested_times().tolist()
        ),
        "all_requested_states_float64": all(
            trajectories[name].dtype == np.dtype(np.float64)
            for name in VARIANT_NAMES
        ),
        "all_shapes_equal": shapes[0] == shapes[1] == shapes[2],
        "all_metrics_finite": all(
            _finite_metric_scalars(metrics) for metrics in comparisons.values()
        ),
        "source_h2_production_to_half_replayed": not replay_mismatches,
    }
    digest = stable_hash_payload(
        artifact_kind="common-time-v2-h2-swe-cfl-diagnostic-scientific-digest",
        payload={
            "task_id": task["task_id"],
            "solver": task["solver"],
            "qualified_id": task["qualified_id"],
            "cfl_ladder": task["cfl_ladder"],
            "variant_array_hashes": {
                name: variants[name]["array_hashes"] for name in VARIANT_NAMES
            },
            "variant_health": {
                name: {
                    "checks": variants[name]["checks"],
                    "health_metrics": variants[name]["health_metrics"],
                }
                for name in VARIANT_NAMES
            },
            "comparisons": comparisons,
            "contraction_ratio": contraction_ratio,
            "observed_order": order,
            "per_time_observed_order": per_time_order,
        },
        schema_id=SCHEMA_ID,
    )
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-swe-cfl-diagnostic-task-result",
        "study_hash": str(study_hash),
        "task_id": str(task["task_id"]),
        "ordinal": int(task["ordinal"]),
        "solver": str(task["solver"]),
        "qualified_id": str(task["qualified_id"]),
        "input_fingerprint": str(task["input_fingerprint"]),
        "bathymetry_type": str(task["bathymetry_type"]),
        "source_type": str(task["source_type"]),
        "selection_role": str(task["selection_role"]),
        "cfl_ladder": [float(value) for value in task["cfl_ladder"]],
        "source_h2_task_id": str(task["source_h2_task_id"]),
        "passed_health_and_replay": bool(all(checks.values())),
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "checks": checks,
        "source_h2_metric_replay_mismatches": replay_mismatches,
        "comparisons": comparisons,
        "contraction_ratio_half_to_quarter_over_production_to_half": (
            contraction_ratio
        ),
        "observed_order": order,
        "per_time_observed_order": per_time_order,
        "variants": variants,
        "scientific_digest": digest,
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
        artifact_kind="common-time-v2-h2-swe-cfl-diagnostic-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    return payload


def _task_path(root: Path, task: Mapping[str, Any]) -> Path:
    return (
        root
        / "execution"
        / "tasks"
        / f"{int(task['ordinal']):03d}-{task['task_id']}.json"
    )


def _validate_task_result(
    payload: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    study_hash: str,
) -> None:
    for key in (
        "task_id",
        "ordinal",
        "solver",
        "qualified_id",
        "input_fingerprint",
        "bathymetry_type",
        "source_type",
        "selection_role",
        "cfl_ladder",
        "source_h2_task_id",
    ):
        if payload.get(key) != task.get(key):
            raise RuntimeError(f"Diagnostic task identity mismatch for {key}")
    if payload.get("study_hash") != study_hash:
        raise RuntimeError("Diagnostic task study hash mismatch")
    identity = dict(payload)
    recorded = identity.pop("result_hash", None)
    expected = stable_hash_payload(
        artifact_kind="common-time-v2-h2-swe-cfl-diagnostic-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    if recorded != expected:
        raise RuntimeError(f"Diagnostic task hash mismatch: {task['task_id']}")
    if payload.get("passed_health_and_replay") != (
        not payload.get("failed_checks")
    ):
        raise RuntimeError(f"Diagnostic task health inconsistency: {task['task_id']}")


def _load_completed(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    expected = {_task_path(root, task): task for task in contract["tasks"]}
    tasks_root = root / "execution" / "tasks"
    if tasks_root.exists():
        unexpected = {
            path for path in tasks_root.iterdir() if path.is_file() and path not in expected
        }
        if unexpected:
            raise RuntimeError(
                f"Unexpected diagnostic task files: {sorted(map(str, unexpected))}"
            )
    completed: dict[str, dict[str, Any]] = {}
    for path, task in expected.items():
        if not path.is_file():
            continue
        payload = _read_json(path)
        _validate_task_result(
            payload, task=task, study_hash=str(contract["study_hash"])
        )
        completed[str(task["task_id"])] = payload
    return completed


def _screening_gates(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    trajectory = _distribution(
        [
            float(row["comparisons"]["half_to_quarter"]["trajectory_relative_l2"])
            for row in rows
        ]
    )
    amplitude = _distribution(
        [
            float(
                row["comparisons"]["half_to_quarter"][
                    "peak_amplitude_relative_error"
                ]
            )
            for row in rows
        ]
    )
    phase_values = [
        float(value)
        for row in rows
        if (
            value := row["comparisons"]["half_to_quarter"][
                "phase_correlation_loss"
            ]
        )
        is not None
    ]
    phase = _distribution(phase_values) if phase_values else {"count": 0}
    per_time_values = [
        [
            float(row["comparisons"]["half_to_quarter"]["per_time_normalized_rmse"][i])
            for row in rows
        ]
        for i in range(len(candidate_requested_times()))
    ]
    worst_p95 = max(_distribution(values)["p95"] for values in per_time_values)
    worst_max = max(max(values) for values in per_time_values)

    def gate(name: str, observed: float, threshold_key: str) -> dict[str, Any]:
        limit = float(thresholds[threshold_key])
        return {
            "gate": name,
            "observed": float(observed),
            "threshold": limit,
            "passed": float(observed) <= limit,
            "role": "outcome_targeted_screening_only",
        }

    gates = [
        gate(
            "trajectory_relative_l2_median",
            trajectory["median"],
            "trajectory_relative_l2_median",
        ),
        gate(
            "trajectory_relative_l2_p95",
            trajectory["p95"],
            "trajectory_relative_l2_p95",
        ),
        gate(
            "trajectory_relative_l2_max",
            trajectory["max"],
            "trajectory_relative_l2_max",
        ),
        gate(
            "per_time_normalized_rmse_p95",
            worst_p95,
            "per_time_normalized_rmse_p95",
        ),
        gate(
            "per_time_normalized_rmse_max",
            worst_max,
            "per_time_normalized_rmse_max",
        ),
        gate(
            "peak_amplitude_relative_error_median",
            amplitude["median"],
            "peak_amplitude_relative_error_median",
        ),
        gate(
            "peak_amplitude_relative_error_p95",
            amplitude["p95"],
            "peak_amplitude_relative_error_p95",
        ),
        gate(
            "peak_amplitude_relative_error_max",
            amplitude["max"],
            "peak_amplitude_relative_error_max",
        ),
    ]
    if phase_values:
        gates.extend(
            [
                gate(
                    "phase_correlation_loss_median",
                    phase["median"],
                    "phase_correlation_loss_median",
                ),
                gate(
                    "phase_correlation_loss_p95",
                    phase["p95"],
                    "phase_correlation_loss_p95",
                ),
                gate(
                    "phase_correlation_loss_max",
                    phase["max"],
                    "phase_correlation_loss_max",
                ),
            ]
        )
    return gates


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    by_solver: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for solver in contract["resolved_config"]["solvers"]:
        solver_rows = [row for row in rows if row["solver"] == solver]
        comparison_summary: dict[str, Any] = {}
        for comparison in COMPARISON_NAMES:
            comparison_summary[comparison] = {
                "trajectory_absolute_rms": _distribution(
                    [
                        float(row["comparisons"][comparison]["trajectory_absolute_rms"])
                        for row in solver_rows
                    ]
                ),
                "trajectory_relative_l2": _distribution(
                    [
                        float(row["comparisons"][comparison]["trajectory_relative_l2"])
                        for row in solver_rows
                    ]
                ),
                "peak_amplitude_relative_error": _distribution(
                    [
                        float(
                            row["comparisons"][comparison][
                                "peak_amplitude_relative_error"
                            ]
                        )
                        for row in solver_rows
                    ]
                ),
            }
        orders = [
            float(row["observed_order"])
            for row in solver_rows
            if row["observed_order"] is not None
        ]
        contractions = [
            float(
                row[
                    "contraction_ratio_half_to_quarter_over_production_to_half"
                ]
            )
            for row in solver_rows
            if row[
                "contraction_ratio_half_to_quarter_over_production_to_half"
            ]
            is not None
        ]
        gates = _screening_gates(
            solver_rows, contract["source_h2_thresholds"][solver]
        )
        for gate in gates:
            gate["solver"] = solver
        all_gates.extend(gates)
        by_solver[solver] = {
            "task_count": len(solver_rows),
            "comparisons": comparison_summary,
            "contraction_ratio": _distribution(contractions),
            "observed_order": _distribution(orders),
            "half_to_quarter_smaller_count": sum(
                value < 1.0 for value in contractions
            ),
            "screening_gates": gates,
            "screening_passed": all(gate["passed"] for gate in gates),
        }
    return {
        "by_solver": by_solver,
        "screening_gate_count": len(all_gates),
        "failed_screening_gates": [
            gate for gate in all_gates if not gate["passed"]
        ],
        "all_tasks_healthy_and_replayed": all(
            bool(row["passed_health_and_replay"]) for row in rows
        ),
        "interpretation": (
            "Outcome-targeted diagnostic evidence only; passing screening gates "
            "does not constitute an H2 population result."
        ),
    }


def _report_text(result: Mapping[str, Any]) -> str:
    lines = [
        "# H2 SWE CFL Refinement Diagnostic",
        "",
        f"- Study hash: `{result['study_hash']}`",
        f"- Source H2: `{result['source_h2_contract_hash']}`",
        "- Decision role: non-decisional post-H2 diagnosis",
        f"- Tasks: {result['task_count']} (three trajectories per task)",
        f"- All task health/replay checks: {result['summary']['all_tasks_healthy_and_replayed']}",
        "",
        "The eight cases were selected after the source H2 outcomes were viewed. "
        "Therefore this study can diagnose contraction and screen a candidate CFL, "
        "but it cannot declare H2 passed.",
        "",
        "## Solver summary",
        "",
    ]
    for solver, summary in result["summary"]["by_solver"].items():
        coarse = summary["comparisons"]["production_to_half"][
            "trajectory_relative_l2"
        ]
        fine = summary["comparisons"]["half_to_quarter"][
            "trajectory_relative_l2"
        ]
        order = summary["observed_order"]
        lines.extend(
            [
                f"### {solver}",
                "",
                (
                    f"- Production-to-half relative L2: median {coarse['median']:.6g}, "
                    f"p95 {coarse['p95']:.6g}, max {coarse['max']:.6g}."
                ),
                (
                    f"- Half-to-quarter relative L2: median {fine['median']:.6g}, "
                    f"p95 {fine['p95']:.6g}, max {fine['max']:.6g}."
                ),
                (
                    f"- Observed order from absolute RMS: median {order['median']:.4g}, "
                    f"min {order['min']:.4g}, max {order['max']:.4g}."
                ),
                (
                    "- Frozen-threshold screening on this targeted subset: "
                    f"{'pass' if summary['screening_passed'] else 'FAIL'}."
                ),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def execute_diagnostic(
    *,
    repo_root: Path,
    diagnostic_root: Path,
    workers: int,
    max_in_flight: int | None,
    resume: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    diagnostic_root = diagnostic_root.resolve()
    validate_frozen_checksums(diagnostic_root)
    contract = _read_json(diagnostic_root / "diagnostic_contract.json")
    _validate_contract(contract)
    if code_state(repo_root) != contract["code_state"]:
        raise RuntimeError("Code state differs from the frozen diagnostic contract")
    source_root = Path(contract["source_h2"]["root"])
    validate_h2_checksums(source_root)
    if sha256_file(source_root / "execution" / "result.json") != contract[
        "source_h2"
    ]["result_sha256"]:
        raise RuntimeError("Source H2 result changed after diagnostic freeze")

    policy = contract["worker_policy"]
    frozen_workers = int(policy["workers"])
    frozen_in_flight = int(policy["max_in_flight"])
    effective_in_flight = frozen_in_flight if max_in_flight is None else max_in_flight
    if workers != frozen_workers or effective_in_flight != frozen_in_flight:
        raise RuntimeError(
            "Diagnostic requires frozen workers/max-in-flight "
            f"{frozen_workers}/{frozen_in_flight}"
        )
    if workers <= 0 or effective_in_flight < workers:
        raise ValueError("Diagnostic workers/max-in-flight must be positive")

    execution_root = diagnostic_root / "execution"
    result_path = execution_root / "result.json"
    manifest = execution_root / "SHA256SUMS.txt"
    if manifest.exists():
        if not result_path.is_file():
            raise RuntimeError("Final checksum manifest exists without result.json")
        if not resume:
            raise FileExistsError(f"Diagnostic already finalized: {result_path}")
        validate_execution_checksums(execution_root)
        return result_path
    if result_path.exists() and not resume:
        raise FileExistsError("Incomplete finalization exists; rerun with --resume")

    tasks = contract["tasks"]
    completed = _load_completed(diagnostic_root, contract)
    if completed and not resume:
        raise FileExistsError("Partial diagnostic exists; rerun with --resume")
    pending = [task for task in tasks if str(task["task_id"]) not in completed]
    (execution_root / "tasks").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_completed = len(completed)
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

    context = multiprocessing.get_context(str(policy["process_start_method"]))
    pending_iter = iter(pending)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        active: dict[Any, Mapping[str, Any]] = {}

        def submit_until_full() -> None:
            while len(active) < effective_in_flight:
                try:
                    task = next(pending_iter)
                except StopIteration:
                    break
                future = executor.submit(
                    _run_task,
                    task,
                    contract["candidate"],
                    contract["health_gates"],
                    contract["metrics_config"],
                    contract["resolved_config"]["analysis"],
                    contract["study_hash"],
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
                    study_hash=str(contract["study_hash"]),
                )
                _write_json(_task_path(diagnostic_root, task), payload)
                completed[str(task["task_id"])] = payload
                if progress is not None:
                    elapsed = time.monotonic() - started
                    completed_this_run = len(completed) - initial_completed
                    remaining = len(tasks) - len(completed)
                    eta_s = (
                        elapsed * remaining / completed_this_run
                        if completed_this_run
                        else None
                    )
                    progress(
                        {
                            "event": "task_complete",
                            "completed": len(completed),
                            "total": len(tasks),
                            "qualified_id": task["qualified_id"],
                            "solver": task["solver"],
                            "runtime_s": payload["runtime_s"],
                            "passed": payload["passed_health_and_replay"],
                            "observed_order": payload["observed_order"],
                            "contraction_ratio": payload[
                                "contraction_ratio_half_to_quarter_over_production_to_half"
                            ],
                            "elapsed_s": elapsed,
                            "eta_s": eta_s,
                        }
                    )
            submit_until_full()

    ordered = [completed[str(task["task_id"])] for task in tasks]
    summary = _aggregate(ordered, contract)
    result = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-swe-cfl-diagnostic-result",
        "study_hash": contract["study_hash"],
        "source_h2_contract_hash": contract["source_h2"]["contract_hash"],
        "decision_role": contract["decision_role"],
        "selection_is_outcome_targeted": True,
        "task_count": len(ordered),
        "solver_execution_count": len(VARIANT_NAMES) * len(ordered),
        "wall_duration_this_invocation_s": float(time.monotonic() - started),
        "sum_task_runtime_s": float(
            math.fsum(float(row["runtime_s"]) for row in ordered)
        ),
        "summary": summary,
        "task_rows": ordered,
        "operational_provenance": {
            "workers": workers,
            "max_in_flight": effective_in_flight,
            "process_start_method": str(policy["process_start_method"]),
            "thread_environment": {
                key: os.environ.get(key) for key in THREAD_ENV_KEYS
            },
            "worker_pids": sorted(
                {int(row["worker"]["pid"]) for row in ordered}
            ),
        },
        "scientific_conclusion_deferred_to_review": True,
        "fresh_h2_pass_not_claimed": True,
        "mass_generation_authorized": False,
    }
    _write_json(result_path, result)
    _write_text(execution_root / "REPORT.md", _report_text(result))
    _write_execution_checksums(execution_root)
    validate_execution_checksums(execution_root)
    if progress is not None:
        progress(
            {
                "event": "finalized",
                "completed": len(ordered),
                "total": len(tasks),
                "duration_s": result["wall_duration_this_invocation_s"],
                "healthy": summary["all_tasks_healthy_and_replayed"],
                "failed_screening_gate_count": len(
                    summary["failed_screening_gates"]
                ),
            }
        )
    return result_path


def diagnostic_status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validate_frozen_checksums(root)
    contract = _read_json(root / "diagnostic_contract.json")
    _validate_contract(contract)
    completed = _load_completed(root, contract)
    result_path = root / "execution" / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    return {
        "study_hash": contract["study_hash"],
        "completed": len(completed),
        "total": len(contract["tasks"]),
        "pending": len(contract["tasks"]) - len(completed),
        "failed_completed_tasks": sorted(
            task_id
            for task_id, payload in completed.items()
            if not payload["passed_health_and_replay"]
        ),
        "finalized": result is not None,
        "all_tasks_healthy_and_replayed": (
            None
            if result is None
            else result["summary"]["all_tasks_healthy_and_replayed"]
        ),
    }
