from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data_gen.common_time_v2 import (
    code_state,
    sha256_file,
    stable_hash_payload,
)
from src.evaluation.buffered_crop_benchmark import SOLVERS, run_buffered_case
from src.evaluation.common_time_v2_level_a import _select_canaries


SCHEMA_ID = "tsunami-surrogate.buffered-production-health.v1"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


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


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise RuntimeError(f"Missing checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        listed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if listed != actual:
        raise RuntimeError("Health artifact checksum inventory mismatch")


def _task_path(root: Path, task: Mapping[str, Any]) -> Path:
    return root / "tasks" / f"{int(task['ordinal']):04d}-{task['task_id']}.json"


def _task_identity(record: Mapping[str, Any], solver: str, ordinal: int) -> dict[str, Any]:
    identity = {
        "ordinal": int(ordinal),
        "qualified_id": str(record["qualified_id"]),
        "input_fingerprint": str(record["input_fingerprint"]),
        "solver": str(solver),
        "total_grid": 96,
        "core_grid": 64,
        "buffer_cells": 16,
        "source_taper_cells": 8,
        "fixed_outer_sponge_width_cells": 16,
        "sponge_min_factor": 0.8,
        "sponge_profile": "cosine",
        "swe_boundary": "radiation",
        "boussinesq_boundary": "open",
    }
    identity["task_id"] = stable_hash_payload(
        artifact_kind="buffered-production-health-task",
        payload=identity,
        schema_id=SCHEMA_ID,
    )[:20]
    return identity


def audit_health_contract(
    *,
    repo_root: Path,
    inventory_path: Path,
    output_root: Path,
    canary_count: int = 6,
) -> Path:
    repo_root = repo_root.resolve()
    inventory_path = inventory_path.resolve()
    output_root = output_root.resolve()
    if canary_count <= 0:
        raise ValueError("canary_count must be positive")
    canaries = _select_canaries(_read_jsonl(inventory_path), canary_count)
    tasks: list[dict[str, Any]] = []
    ordinal = 0
    for record in canaries:
        for solver in SOLVERS:
            tasks.append({
                **_task_identity(record, solver, ordinal),
                "record": record,
            })
            ordinal += 1
    payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "buffered-production-health-contract",
        "status": "diagnostic_unfrozen_health_only",
        "code_state": code_state(repo_root),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "canaries": canaries,
        "tasks": tasks,
        "policy": {
            "computational_grid": [96, 96],
            "publication_crop": [64, 64],
            "buffer_cells_per_side": 16,
            "source_taper_cells": 8,
            "bathymetry_extension": "constant edge continuation",
            "swe_boundary": "radiation",
            "boussinesq_boundary": "open",
            "sponge_width_cells": 16,
            "sponge_min_factor": 0.8,
            "sponge_profile": "cosine",
            "requested_state_dtype": "float64",
            "requested_time_count": 50,
            "horizon": 0.175,
        },
        "claim_scope": (
            "solver health, requested-output integrity, and crop plumbing only; "
            "not boundary independence or external numerical validation"
        ),
    }
    contract_identity = dict(payload)
    payload["contract_hash"] = stable_hash_payload(
        artifact_kind="buffered-production-health-contract",
        payload=contract_identity,
        schema_id=SCHEMA_ID,
    )
    if output_root.exists():
        contract_path = output_root / "contract.json"
        if contract_path.is_file():
            existing = _read_json(contract_path)
            if existing != payload:
                raise FileExistsError(
                    f"Refusing to replace a different health contract: {output_root}"
                )
            return output_root
        if any(output_root.iterdir()):
            raise FileExistsError(
                f"Refusing to initialize a non-empty health root: {output_root}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "tasks").mkdir(exist_ok=False)
    _write_json(output_root / "contract.json", payload)
    _write_checksums(output_root)
    return output_root


def _run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    row, trajectory = run_buffered_case(
        task["record"],
        solver_name=str(task["solver"]),
        total_grid=96,
        source_taper_cells=8,
        sponge_min_factor=0.8,
        sponge_width_cells=16,
    )
    payload = {
        "schema_id": SCHEMA_ID,
        "task_id": str(task["task_id"]),
        "ordinal": int(task["ordinal"]),
        "qualified_id": str(task["qualified_id"]),
        "input_fingerprint": str(task["input_fingerprint"]),
        "solver": str(task["solver"]),
        "trajectory_shape": list(map(int, trajectory.shape)),
        "trajectory_finite": bool(np.isfinite(trajectory).all()),
        "row": row,
    }
    payload["result_hash"] = stable_hash_payload(
        artifact_kind="buffered-production-health-task-result",
        payload=payload,
        schema_id=SCHEMA_ID,
    )
    return payload


def _load_completed(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = _task_path(root, task)
        if not path.exists():
            continue
        payload = _read_json(path)
        task_id = str(task["task_id"])
        if payload.get("task_id") != task_id:
            raise RuntimeError(f"Health task identity mismatch: {path}")
        identity = dict(payload)
        recorded_hash = identity.pop("result_hash", None)
        expected_hash = stable_hash_payload(
            artifact_kind="buffered-production-health-task-result",
            payload=identity,
            schema_id=SCHEMA_ID,
        )
        if recorded_hash != expected_hash:
            raise RuntimeError(f"Health task result hash mismatch: {path}")
        completed[task_id] = payload
    return completed


def _row_passes(payload: Mapping[str, Any]) -> bool:
    row = payload["row"]
    health = row["health"]
    return bool(
        payload["trajectory_finite"]
        and payload["trajectory_shape"] == [50, 64, 64]
        and health["finite"]
        and health["requested_times_exact"]
        and health["measurement_dtype"] == "float64"
        and int(health["cg_failure_count"]) == 0
        and float(row["source_edge_max_abs"]) == 0.0
        and float(row["sponge_core_min"]) == 1.0
    )


def health_status(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    contract = _read_json(output_root / "contract.json")
    tasks = contract["tasks"]
    completed = _load_completed(output_root, tasks)
    failed = [
        task_id for task_id, payload in completed.items() if not _row_passes(payload)
    ]
    return {
        "contract_hash": contract["contract_hash"],
        "completed": len(completed),
        "total": len(tasks),
        "pending": len(tasks) - len(completed),
        "failed_health_tasks": failed,
        "finalized": (output_root / "result.json").is_file(),
    }


def execute_health_contract(
    *,
    repo_root: Path,
    output_root: Path,
    workers: int,
    resume: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    contract = _read_json(output_root / "contract.json")
    if contract.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("Buffered production health schema mismatch")
    if (
        code_state(repo_root)["code_state_hash"]
        != contract["code_state"]["code_state_hash"]
    ):
        raise RuntimeError("Code state changed after health audit; create a fresh audit root")
    result_path = output_root / "result.json"
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Health result already finalized: {result_path}")
        validate_checksums(output_root)
        return result_path
    tasks = contract["tasks"]
    completed = _load_completed(output_root, tasks)
    if completed and not resume:
        raise FileExistsError("Partial health tasks exist; rerun with --resume")
    pending = [task for task in tasks if str(task["task_id"]) not in completed]
    started = time.monotonic()
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
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, max(1, len(pending))), mp_context=context
    ) as executor:
        futures = {executor.submit(_run_task, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            payload = future.result()
            _write_json(_task_path(output_root, task), payload)
            completed[str(task["task_id"])] = payload
            if progress is not None:
                progress({
                    "event": "task_complete",
                    "completed": len(completed),
                    "total": len(tasks),
                    "qualified_id": task["qualified_id"],
                    "solver": task["solver"],
                    "runtime_s": payload["row"]["health"]["runtime_s"],
                    "health_passed": _row_passes(payload),
                    "elapsed_s": time.monotonic() - started,
                })
    ordered = [completed[str(task["task_id"])] for task in tasks]
    failed = [payload["task_id"] for payload in ordered if not _row_passes(payload)]
    result = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "buffered-production-health-result",
        "status": "pass" if not failed else "fail",
        "contract_hash": contract["contract_hash"],
        "task_count": len(tasks),
        "failed_health_tasks": failed,
        "all_health_checks_passed": not failed,
        "duration_s": time.monotonic() - started,
        "workers": workers,
        "claim_scope": contract["claim_scope"],
        "rows": ordered,
    }
    _write_json(result_path, result)
    _write_checksums(output_root)
    if progress is not None:
        progress(
            {
                "event": "finalized",
                "completed": len(tasks),
                "total": len(tasks),
                "passed": not failed,
                "duration_s": result["duration_s"],
            }
        )
    return result_path
