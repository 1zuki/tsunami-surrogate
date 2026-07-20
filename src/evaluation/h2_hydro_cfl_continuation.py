from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
)
from src.evaluation.common_time_v2_level_a import _load_canary_arrays
from src.evaluation.h2_swe_cfl_diagnostic import (
    observed_order,
    validate_diagnostic_checksums,
)


SCHEMA_ID = "tsunami-surrogate.common-time-v2.h2-hydro-cfl-continuation.v1"
CONFIG_SCHEMA_ID = (
    "tsunami-surrogate.common-time-v2.h2-hydro-cfl-continuation-config.v1"
)
FROZEN_FILENAMES = ("continuation_contract.json", "task_plan.json")
FROZEN_CHECKSUMS = "CONTRACT_SHA256SUMS.txt"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_id") != CONFIG_SCHEMA_ID:
        raise RuntimeError("Hydro CFL continuation configuration schema mismatch")
    if payload.get("decision_role") != "non_decisional_post_h2_diagnosis":
        raise RuntimeError("Hydro CFL continuation must remain non-decisional")
    if payload.get("selection_is_outcome_targeted") is not True:
        raise RuntimeError("Outcome-targeted selection must be disclosed")
    if payload.get("validation_or_test_outcomes_inspected") is not False:
        raise RuntimeError("Validation/test scientific outcomes must remain excluded")
    if payload.get("solver") != "swe_hydrostatic":
        raise RuntimeError("Hydro CFL continuation solver changed")
    pair = payload["cfl_pair"]
    production = float(pair["candidate_production"])
    reference = float(pair["candidate_reference"])
    if not math.isclose(reference, 0.5 * production):
        raise RuntimeError("Hydro continuation reference CFL must be half production")
    if len(payload.get("cases", [])) != 3:
        raise RuntimeError("Hydro continuation must freeze exactly three cases")
    ids = [str(case["qualified_id"]) for case in payload["cases"]]
    if len(set(ids)) != 3:
        raise RuntimeError("Hydro continuation case IDs must be unique")
    execution = payload["execution"]
    if int(execution["workers"]) != 3 or int(execution["max_in_flight"]) != 3:
        raise RuntimeError("Hydro continuation freezes three parallel workers")
    return payload


def _study_hash(contract: Mapping[str, Any]) -> str:
    identity = dict(contract)
    identity.pop("study_hash", None)
    return stable_hash_payload(
        artifact_kind="common-time-v2-h2-hydro-cfl-continuation-contract",
        payload=identity,
        schema_id=SCHEMA_ID,
    )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("Hydro CFL continuation contract schema mismatch")
    if contract.get("study_hash") != _study_hash(contract):
        raise RuntimeError("Hydro CFL continuation content hash mismatch")
    tasks = contract.get("tasks", [])
    if len(tasks) != 3:
        raise RuntimeError("Hydro CFL continuation must contain three tasks")
    if [int(task["ordinal"]) for task in tasks] != [0, 1, 2]:
        raise RuntimeError("Hydro CFL continuation task ordering changed")
    if len({str(task["task_id"]) for task in tasks}) != 3:
        raise RuntimeError("Hydro CFL continuation task IDs are not unique")


def _write_frozen_checksums(root: Path) -> None:
    rows = [f"{sha256_file(root / name)}  {name}" for name in FROZEN_FILENAMES]
    _write_text(root / FROZEN_CHECKSUMS, "\n".join(rows) + "\n")


def validate_frozen_checksums(root: Path) -> None:
    manifest = root / FROZEN_CHECKSUMS
    if not manifest.is_file():
        raise RuntimeError(f"Missing continuation checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        if relative not in FROZEN_FILENAMES:
            raise RuntimeError(f"Unexpected continuation checksum entry: {relative}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Continuation frozen checksum mismatch: {relative}")
        listed.add(relative)
    if listed != set(FROZEN_FILENAMES):
        raise RuntimeError("Continuation frozen checksum inventory is incomplete")


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
        raise RuntimeError(f"Missing continuation execution manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Continuation execution checksum mismatch: {relative}")
        listed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if listed != actual:
        raise RuntimeError("Continuation execution checksum inventory mismatch")


def validate_continuation_checksums(root: Path) -> None:
    validate_frozen_checksums(root)
    execution_root = root / "execution"
    if (execution_root / "result.json").is_file():
        validate_execution_checksums(execution_root)
    elif (execution_root / "SHA256SUMS.txt").exists():
        raise RuntimeError("Partial continuation has a final checksum manifest")


def freeze_continuation(
    *,
    repo_root: Path,
    config_path: Path,
    source_diagnostic_root: Path,
    output_base: Path,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    source_diagnostic_root = source_diagnostic_root.resolve()
    output_base = output_base.resolve()
    config = _load_config(config_path)
    validate_diagnostic_checksums(source_diagnostic_root)
    source_contract = _read_json(
        source_diagnostic_root / "diagnostic_contract.json"
    )
    source_result = _read_json(source_diagnostic_root / "execution" / "result.json")
    source_hash = str(config["source_diagnostic_hash"])
    if (
        source_contract.get("study_hash") != source_hash
        or source_result.get("study_hash") != source_hash
        or source_diagnostic_root.name != source_hash
    ):
        raise RuntimeError("Hydro continuation source diagnostic hash mismatch")
    source_tasks = {
        (str(task["qualified_id"]), str(task["solver"])): task
        for task in source_contract["tasks"]
    }
    source_rows = {
        (str(row["qualified_id"]), str(row["solver"])): row
        for row in source_result["task_rows"]
    }
    tasks: list[dict[str, Any]] = []
    for ordinal, case in enumerate(config["cases"]):
        key = (str(case["qualified_id"]), "swe_hydrostatic")
        if key not in source_tasks or key not in source_rows:
            raise RuntimeError(f"Missing source diagnostic Hydro task: {key[0]}")
        source_task = source_tasks[key]
        source_row = source_rows[key]
        if source_task["input_fingerprint"] != case["input_fingerprint"]:
            raise RuntimeError(f"Hydro continuation fingerprint mismatch: {key[0]}")
        task: dict[str, Any] = {
            "ordinal": ordinal,
            "qualified_id": key[0],
            "input_fingerprint": str(case["input_fingerprint"]),
            "selection_role": str(case["selection_role"]),
            "record": dict(source_task["record"]),
            "candidate_production_cfl": float(
                config["cfl_pair"]["candidate_production"]
            ),
            "candidate_reference_cfl": float(
                config["cfl_pair"]["candidate_reference"]
            ),
            "source_diagnostic_task_id": str(source_row["task_id"]),
            "source_candidate_production_array_hash": dict(
                source_row["variants"]["quarter"]["array_hashes"][
                    "cropped_eta_trajectory"
                ]
            ),
            "prior_half_to_quarter_metrics": dict(
                source_row["comparisons"]["half_to_quarter"]
            ),
        }
        identity = {
            key: value
            for key, value in task.items()
            if key not in {"record", "prior_half_to_quarter_metrics"}
        }
        task["task_id"] = stable_hash_payload(
            artifact_kind="common-time-v2-h2-hydro-cfl-continuation-task",
            payload=identity,
            schema_id=SCHEMA_ID,
        )[:24]
        tasks.append(task)

    contract: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-hydro-cfl-continuation-contract",
        "study": str(config["study"]),
        "status": "frozen_unexecuted",
        "decision_role": str(config["decision_role"]),
        "selection_is_outcome_targeted": True,
        "validation_or_test_outcomes_inspected": False,
        "source_diagnostic": {
            "root": str(source_diagnostic_root),
            "study_hash": source_hash,
            "contract_sha256": sha256_file(
                source_diagnostic_root / "diagnostic_contract.json"
            ),
            "result_sha256": sha256_file(
                source_diagnostic_root / "execution" / "result.json"
            ),
        },
        "code_state": code_state(repo_root),
        "environment": _environment_snapshot(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config,
        "candidate": dict(source_contract["candidate"]),
        "health_gates": dict(source_contract["health_gates"]),
        "metrics_config": dict(source_contract["metrics_config"]),
        "source_h2_thresholds": dict(
            source_contract["source_h2_thresholds"]["swe_hydrostatic"]
        ),
        "worker_policy": dict(config["execution"]),
        "tasks": tasks,
        "interpretation": {
            "purpose": (
                "test whether Hydro CFL 0.1125 has acceptable sensitivity "
                "against CFL 0.05625 on the three worst first-time cases"
            ),
            "outcome_targeted_sample_cannot_establish_population_pass": True,
            "not_a_replacement_h2": True,
            "cannot_authorize_mass_generation": True,
        },
    }
    contract["study_hash"] = _study_hash(contract)
    _validate_contract(contract)
    root = output_base / str(contract["study_hash"])
    task_plan = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-hydro-cfl-continuation-task-plan",
        "study_hash": contract["study_hash"],
        "task_count": 3,
        "solver_execution_count": 6,
        "tasks": tasks,
    }
    if root.exists():
        path = root / "continuation_contract.json"
        if not path.is_file() or _read_json(path) != contract:
            raise FileExistsError(f"Refusing to replace a different study: {root}")
        validate_frozen_checksums(root)
        return root
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "continuation_contract.json", contract)
    _write_json(root / "task_plan.json", task_plan)
    _write_frozen_checksums(root)
    validate_frozen_checksums(root)
    return root


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


def _run_task(
    task: Mapping[str, Any],
    candidate: Mapping[str, Any],
    health_gates: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
    study_hash: str,
) -> dict[str, Any]:
    started = time.monotonic()
    record = task["record"]
    bathymetry, _source, _strength_array, _strength, arrays = _load_canary_arrays(
        record
    )
    production, production_evidence = _run_variant(
        record=record,
        solver="swe_hydrostatic",
        candidate=candidate,
        health_gates=health_gates,
        target_cfl=float(task["candidate_production_cfl"]),
        bathymetry=bathymetry,
        arrays=arrays,
    )
    reference, reference_evidence = _run_variant(
        record=record,
        solver="swe_hydrostatic",
        candidate=candidate,
        health_gates=health_gates,
        target_cfl=float(task["candidate_reference_cfl"]),
        bathymetry=bathymetry,
        arrays=arrays,
    )
    metrics = _metrics(production, reference, metrics_config)
    previous_absolute = float(
        task["prior_half_to_quarter_metrics"]["trajectory_absolute_rms"]
    )
    current_absolute = float(metrics["trajectory_absolute_rms"])
    ratio = current_absolute / previous_absolute if previous_absolute > 0.0 else None
    order = observed_order(previous_absolute, current_absolute)
    checks = {
        "candidate_production_healthy": bool(production_evidence["passed"]),
        "candidate_reference_healthy": bool(reference_evidence["passed"]),
        "requested_times_exactly_equal": (
            production_evidence["requested_output_provenance"][
                "requested_timestamps"
            ]
            == reference_evidence["requested_output_provenance"][
                "requested_timestamps"
            ]
            == candidate_requested_times().tolist()
        ),
        "states_are_float64": (
            production.dtype == np.dtype(np.float64)
            and reference.dtype == np.dtype(np.float64)
        ),
        "shapes_equal": production.shape == reference.shape,
        "candidate_production_array_replays_source_quarter": (
            production_evidence["array_hashes"]["cropped_eta_trajectory"]
            == task["source_candidate_production_array_hash"]
        ),
        "metrics_finite": all(
            math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and value is not None
        ),
    }
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-hydro-cfl-continuation-task-result",
        "study_hash": study_hash,
        "task_id": str(task["task_id"]),
        "ordinal": int(task["ordinal"]),
        "qualified_id": str(task["qualified_id"]),
        "input_fingerprint": str(task["input_fingerprint"]),
        "selection_role": str(task["selection_role"]),
        "candidate_production_cfl": float(task["candidate_production_cfl"]),
        "candidate_reference_cfl": float(task["candidate_reference_cfl"]),
        "source_diagnostic_task_id": str(task["source_diagnostic_task_id"]),
        "passed_health_and_replay": bool(all(checks.values())),
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "checks": checks,
        "prior_half_to_quarter_metrics": dict(
            task["prior_half_to_quarter_metrics"]
        ),
        "quarter_to_eighth_metrics": metrics,
        "contraction_ratio": ratio,
        "observed_order": order,
        "production": production_evidence,
        "reference": reference_evidence,
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
        artifact_kind="common-time-v2-h2-hydro-cfl-continuation-task-result",
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
        "qualified_id",
        "input_fingerprint",
        "selection_role",
        "candidate_production_cfl",
        "candidate_reference_cfl",
        "source_diagnostic_task_id",
    ):
        if payload.get(key) != task.get(key):
            raise RuntimeError(f"Continuation task identity mismatch for {key}")
    if payload.get("study_hash") != study_hash:
        raise RuntimeError("Continuation task study hash mismatch")
    identity = dict(payload)
    recorded = identity.pop("result_hash", None)
    expected = stable_hash_payload(
        artifact_kind="common-time-v2-h2-hydro-cfl-continuation-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    if recorded != expected:
        raise RuntimeError(f"Continuation task hash mismatch: {task['task_id']}")
    if payload.get("passed_health_and_replay") != (
        not payload.get("failed_checks")
    ):
        raise RuntimeError(f"Continuation task health inconsistency: {task['task_id']}")


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
                f"Unexpected continuation task files: {sorted(map(str, unexpected))}"
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


def _screening(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    trajectory = _distribution(
        [
            float(row["quarter_to_eighth_metrics"]["trajectory_relative_l2"])
            for row in rows
        ]
    )
    per_time = [
        float(value)
        for row in rows
        for value in row["quarter_to_eighth_metrics"]["per_time_normalized_rmse"]
    ]
    values = {
        "trajectory_relative_l2_max": trajectory["max"],
        "per_time_normalized_rmse_max": max(per_time),
    }
    return [
        {
            "gate": name,
            "observed": observed,
            "threshold": float(thresholds[name]),
            "passed": observed <= float(thresholds[name]),
            "role": "outcome_targeted_screening_only",
        }
        for name, observed in values.items()
    ]


def execute_continuation(
    *,
    repo_root: Path,
    continuation_root: Path,
    workers: int,
    max_in_flight: int | None,
    resume: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    continuation_root = continuation_root.resolve()
    validate_frozen_checksums(continuation_root)
    contract = _read_json(continuation_root / "continuation_contract.json")
    _validate_contract(contract)
    if code_state(repo_root) != contract["code_state"]:
        raise RuntimeError("Code state differs from frozen continuation contract")
    source_root = Path(contract["source_diagnostic"]["root"])
    validate_diagnostic_checksums(source_root)
    if sha256_file(source_root / "execution" / "result.json") != contract[
        "source_diagnostic"
    ]["result_sha256"]:
        raise RuntimeError("Source diagnostic result changed")

    policy = contract["worker_policy"]
    frozen_workers = int(policy["workers"])
    frozen_in_flight = int(policy["max_in_flight"])
    effective_in_flight = frozen_in_flight if max_in_flight is None else max_in_flight
    if workers != frozen_workers or effective_in_flight != frozen_in_flight:
        raise RuntimeError(
            "Continuation requires frozen workers/max-in-flight "
            f"{frozen_workers}/{frozen_in_flight}"
        )
    execution_root = continuation_root / "execution"
    result_path = execution_root / "result.json"
    manifest = execution_root / "SHA256SUMS.txt"
    if manifest.exists():
        if not result_path.is_file():
            raise RuntimeError("Final manifest exists without result.json")
        if not resume:
            raise FileExistsError(f"Continuation already finalized: {result_path}")
        validate_execution_checksums(execution_root)
        return result_path
    tasks = contract["tasks"]
    completed = _load_completed(continuation_root, contract)
    if completed and not resume:
        raise FileExistsError("Partial continuation exists; rerun with --resume")
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
                _write_json(_task_path(continuation_root, task), payload)
                completed[str(task["task_id"])] = payload
                if progress is not None:
                    elapsed = time.monotonic() - started
                    completed_this_run = len(completed) - initial_completed
                    remaining = len(tasks) - len(completed)
                    progress(
                        {
                            "event": "task_complete",
                            "completed": len(completed),
                            "total": len(tasks),
                            "qualified_id": task["qualified_id"],
                            "runtime_s": payload["runtime_s"],
                            "passed": payload["passed_health_and_replay"],
                            "observed_order": payload["observed_order"],
                            "contraction_ratio": payload["contraction_ratio"],
                            "per_time_max": max(
                                payload["quarter_to_eighth_metrics"][
                                    "per_time_normalized_rmse"
                                ]
                            ),
                            "elapsed_s": elapsed,
                            "eta_s": (
                                elapsed * remaining / completed_this_run
                                if completed_this_run
                                else None
                            ),
                        }
                    )
            submit_until_full()

    ordered = [completed[str(task["task_id"])] for task in tasks]
    gates = _screening(ordered, contract["source_h2_thresholds"])
    result = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-hydro-cfl-continuation-result",
        "study_hash": contract["study_hash"],
        "source_diagnostic_hash": contract["source_diagnostic"]["study_hash"],
        "decision_role": contract["decision_role"],
        "selection_is_outcome_targeted": True,
        "task_count": 3,
        "solver_execution_count": 6,
        "all_tasks_healthy_and_replayed": all(
            bool(row["passed_health_and_replay"]) for row in ordered
        ),
        "observed_order": _distribution(
            [float(row["observed_order"]) for row in ordered]
        ),
        "contraction_ratio": _distribution(
            [float(row["contraction_ratio"]) for row in ordered]
        ),
        "screening_gates": gates,
        "screening_passed": all(gate["passed"] for gate in gates),
        "task_rows": ordered,
        "wall_duration_this_invocation_s": float(time.monotonic() - started),
        "sum_task_runtime_s": float(
            math.fsum(float(row["runtime_s"]) for row in ordered)
        ),
        "fresh_h2_pass_not_claimed": True,
        "mass_generation_authorized": False,
    }
    _write_json(result_path, result)
    _write_execution_checksums(execution_root)
    validate_execution_checksums(execution_root)
    if progress is not None:
        progress(
            {
                "event": "finalized",
                "duration_s": result["wall_duration_this_invocation_s"],
                "healthy": result["all_tasks_healthy_and_replayed"],
                "screening_passed": result["screening_passed"],
            }
        )
    return result_path


def continuation_status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validate_frozen_checksums(root)
    contract = _read_json(root / "continuation_contract.json")
    _validate_contract(contract)
    completed = _load_completed(root, contract)
    result_path = root / "execution" / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    return {
        "study_hash": contract["study_hash"],
        "completed": len(completed),
        "total": 3,
        "pending": 3 - len(completed),
        "failed_completed_tasks": sorted(
            task_id
            for task_id, payload in completed.items()
            if not payload["passed_health_and_replay"]
        ),
        "finalized": result is not None,
        "screening_passed": (
            None if result is None else bool(result["screening_passed"])
        ),
    }
