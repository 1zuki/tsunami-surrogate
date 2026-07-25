from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import sys
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

from src.data_gen.common_time_v2 import sha256_file


OPERATIONAL_TIMING_SCHEMA_ID = (
    "tsunami-surrogate.common-time-v2.generation-operational-timing.v1"
)
OPERATIONAL_TIMING_SUMMARY_SCHEMA_ID = (
    "tsunami-surrogate.common-time-v2.generation-operational-summary.v1"
)
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.staging")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(staging, path)


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) * 1024
    return None


def machine_snapshot(
    output_dir: Path, operational_config: Mapping[str, Any]
) -> dict[str, Any]:
    storage = os.statvfs(output_dir)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": importlib.metadata.version("numpy"),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "memory_total_bytes": _memory_total_bytes(),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "cloud_provider": operational_config.get("cloud_provider"),
        "cloud_zone": operational_config.get("cloud_zone"),
        "machine_type": operational_config.get("machine_type"),
        "storage": {
            "path": str(output_dir.resolve()),
            "class": operational_config.get("storage_class"),
            "total_bytes": int(storage.f_frsize * storage.f_blocks),
            "free_bytes_at_start": int(storage.f_frsize * storage.f_bavail),
        },
    }


class GenerationTimingRecorder:
    """Checkpoint nondeterministic generation timing outside scientific artifacts."""

    def __init__(
        self,
        *,
        output_dir: Path,
        split: str,
        contract_hash: str,
        code_state_hash: str,
        config_path: Path,
        config_sha256: str,
        solver_names: Sequence[str],
        requested_workers: int,
        requested_max_in_flight: int | None,
        operational_config: Mapping[str, Any],
        generation_contract_hash: str | None = None,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.split = str(split)
        self.started_monotonic = time.monotonic()
        self.invocation_id = uuid4().hex
        self.path: Path | None = None
        self._phase_started: dict[str, float] = {}
        self.payload: dict[str, Any] = {
            "schema_id": OPERATIONAL_TIMING_SCHEMA_ID,
            "artifact_kind": "common-time-v2-generation-operational-timing",
            "invocation_id": self.invocation_id,
            "status": "initializing",
            "complete": False,
            "exit_status": "running",
            "started_utc": _utc_now(),
            "ended_utc": None,
            "active_wall_s": 0.0,
            "resolved_command": " ".join(shlex.quote(str(value)) for value in sys.argv),
            "config_path": str(config_path.resolve()),
            "resolved_config_sha256": str(config_sha256),
            "split": self.split,
            "start_index": None,
            "stop_index": None,
            "contract_hash": str(contract_hash),
            "generation_contract_hash": generation_contract_hash,
            "code_state_hash": str(code_state_hash),
            "solver_names": sorted(str(value) for value in solver_names),
            "worker_policy": {
                "requested_workers": int(requested_workers),
                "requested_max_in_flight": requested_max_in_flight,
                "process_start_method": (
                    "spawn" if int(requested_workers) > 1 else "serial"
                ),
            },
            "machine": machine_snapshot(self.output_dir, operational_config),
            "hourly_cost_usd": operational_config.get("hourly_cost_usd"),
            "estimated_cost_usd": None,
            "resume": False,
            "allow_override": False,
            "phase_timings_s": {},
            "counts": {
                "planned_scenarios": 0,
                "completed_scenarios": 0,
                "generated_scenarios": 0,
                "reused_scenarios": 0,
                "failed_scenarios": 0,
                "planned_solver_rollouts": 0,
                "generated_solver_rollouts": 0,
                "reused_solver_rollouts": 0,
                "accepted_solver_rollouts": 0,
                "rejected_solver_rollouts": 0,
            },
            "per_solver": {
                name: {
                    "generated": 0,
                    "reused": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "solve_s": 0.0,
                    "serialization_s": 0.0,
                    "validation_s": 0.0,
                    "worker_s": 0.0,
                    "natural_steps": 0,
                }
                for name in sorted(str(value) for value in solver_names)
            },
            "completed_samples": [],
            "failures": [],
            "shard_manifest": None,
            "shard_manifest_sha256": None,
            "throughput": {},
            "last_checkpoint_utc": None,
        }

    def begin_range(
        self,
        *,
        start_index: int,
        stop_index: int,
        planned_scenarios: int,
        resume: bool,
        allow_override: bool,
    ) -> Path:
        if self.path is not None:
            raise RuntimeError("Operational timing range already initialized")
        self.path = (
            self.output_dir
            / "operational_runs"
            / (
                f"{self.split}_{int(start_index):06d}_{int(stop_index):06d}_"
                f"{self.invocation_id}.json"
            )
        )
        counts = self.payload["counts"]
        counts["planned_scenarios"] = int(planned_scenarios)
        counts["planned_solver_rollouts"] = int(planned_scenarios) * len(
            self.payload["solver_names"]
        )
        requested_workers = int(self.payload["worker_policy"]["requested_workers"])
        effective_workers = min(requested_workers, max(1, int(planned_scenarios)))
        requested_window = self.payload["worker_policy"]["requested_max_in_flight"]
        effective_window = min(
            int(planned_scenarios),
            max(
                effective_workers,
                int(requested_window)
                if requested_window is not None
                else 2 * effective_workers,
            ),
        )
        self.payload["worker_policy"].update(
            {
                "effective_workers": effective_workers,
                "effective_max_in_flight": effective_window,
            }
        )
        self.payload.update(
            {
                "status": "running",
                "start_index": int(start_index),
                "stop_index": int(stop_index),
                "resume": bool(resume),
                "allow_override": bool(allow_override),
            }
        )
        self.checkpoint()
        return self.path

    def start_phase(self, name: str) -> None:
        if name in self._phase_started:
            raise RuntimeError(f"Operational phase already running: {name}")
        self._phase_started[name] = time.monotonic()

    def end_phase(self, name: str) -> float:
        started = self._phase_started.pop(name, None)
        if started is None:
            raise RuntimeError(f"Operational phase was not started: {name}")
        elapsed = max(0.0, time.monotonic() - started)
        timings = self.payload["phase_timings_s"]
        timings[name] = float(timings.get(name, 0.0) + elapsed)
        self.checkpoint()
        return elapsed

    def record_sample(self, record: Mapping[str, Any]) -> None:
        operational = record.get("_operational")
        if not isinstance(operational, Mapping):
            raise RuntimeError("Generation worker omitted operational timing")
        sample = json.loads(json.dumps(operational))
        solvers = sample.get("solvers", [])
        if not isinstance(solvers, list):
            raise RuntimeError("Generation worker solver timing must be a list")
        generated = 0
        reused = 0
        for solver_row in solvers:
            solver = str(solver_row["solver"])
            if solver not in self.payload["per_solver"]:
                raise RuntimeError(f"Unexpected operational solver timing: {solver}")
            destination = self.payload["per_solver"][solver]
            status = str(solver_row["status"])
            if status == "generated":
                generated += 1
                destination["generated"] += 1
                self.payload["counts"]["generated_solver_rollouts"] += 1
            elif status == "reused":
                reused += 1
                destination["reused"] += 1
                self.payload["counts"]["reused_solver_rollouts"] += 1
            else:
                raise RuntimeError(f"Unknown operational solver status: {status}")
            destination["accepted"] += 1
            self.payload["counts"]["accepted_solver_rollouts"] += 1
            for key in (
                "solve_s",
                "serialization_s",
                "validation_s",
                "worker_s",
            ):
                destination[key] += float(solver_row.get(key, 0.0))
            destination["natural_steps"] += int(solver_row.get("natural_steps", 0))
        counts = self.payload["counts"]
        counts["completed_scenarios"] += 1
        if generated:
            counts["generated_scenarios"] += 1
        elif reused:
            counts["reused_scenarios"] += 1
        self.payload["completed_samples"].append(sample)
        self.checkpoint()

    def record_failure(self, sample_index: int, error: BaseException) -> None:
        self.payload["counts"]["failed_scenarios"] += 1
        self.payload["failures"].append(
            {
                "sample_index": int(sample_index),
                "error_type": type(error).__name__,
                "message": str(error)[:1000],
                "recorded_utc": _utc_now(),
            }
        )
        self.checkpoint()

    def set_shard_manifest(self, path: Path) -> None:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.output_dir).as_posix()
        except ValueError as exc:
            raise RuntimeError("Operational shard must live below output_dir") from exc
        self.payload["shard_manifest"] = relative
        self.payload["shard_manifest_sha256"] = sha256_file(resolved)
        self.checkpoint()

    def progress(self, *, completed: int, total: int) -> dict[str, float | None]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        rate = float(completed / elapsed) if completed > 0 and elapsed > 0.0 else None
        eta = (
            float((total - completed) / rate)
            if rate is not None and completed < total
            else 0.0
            if completed >= total
            else None
        )
        return {"elapsed_s": elapsed, "rate_per_s": rate, "eta_s": eta}

    def checkpoint(self) -> None:
        if self.path is None:
            return
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        self.payload["active_wall_s"] = float(elapsed)
        self.payload["last_checkpoint_utc"] = _utc_now()
        counts = self.payload["counts"]
        accepted = int(counts["accepted_solver_rollouts"])
        self.payload["throughput"] = {
            "accepted_solver_rollouts_per_s": (
                float(accepted / elapsed) if elapsed > 0.0 else 0.0
            ),
            "accepted_solver_rollouts_per_hour": (
                float(accepted * 3600.0 / elapsed) if elapsed > 0.0 else 0.0
            ),
        }
        _atomic_write_json(self.path, self.payload)

    def finalize(
        self, *, status: str, error: BaseException | None = None
    ) -> Path | None:
        if self.path is None:
            return None
        if status not in {"complete", "failed"}:
            raise ValueError(
                "Operational timing final status must be complete or failed"
            )
        if error is not None and not self.payload["failures"]:
            self.payload["failures"].append(
                {
                    "sample_index": None,
                    "error_type": type(error).__name__,
                    "message": str(error)[:1000],
                    "recorded_utc": _utc_now(),
                }
            )
        self.payload["status"] = status
        self.payload["complete"] = status == "complete"
        self.payload["exit_status"] = "success" if status == "complete" else "failure"
        self.payload["ended_utc"] = _utc_now()
        self.checkpoint()
        rate = self.payload.get("hourly_cost_usd")
        if rate is not None:
            self.payload["estimated_cost_usd"] = (
                float(rate) * float(self.payload["active_wall_s"]) / 3600.0
            )
            self.checkpoint()
        return self.path


def validate_generation_timing(
    path: str | Path, *, verify_shard: bool = True
) -> dict[str, Any]:
    timing_path = Path(path)
    with timing_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_id") != OPERATIONAL_TIMING_SCHEMA_ID:
        raise RuntimeError("Generation timing schema mismatch")
    if payload.get("status") not in {"running", "complete", "failed"}:
        raise RuntimeError("Generation timing status mismatch")
    if not math.isfinite(float(payload.get("active_wall_s", float("nan")))):
        raise RuntimeError("Generation timing active wall time is invalid")
    if float(payload["active_wall_s"]) < 0.0:
        raise RuntimeError("Generation timing active wall time is negative")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("Generation timing counts are missing")
    for key, value in counts.items():
        if int(value) < 0:
            raise RuntimeError(f"Generation timing count is negative: {key}")
    if int(counts["completed_scenarios"]) != len(payload.get("completed_samples", [])):
        raise RuntimeError("Generation timing completed sample count mismatch")
    if payload.get("status") == "complete":
        if not bool(payload.get("complete")) or payload.get("exit_status") != "success":
            raise RuntimeError("Completed generation timing has contradictory status")
        if verify_shard:
            relative = payload.get("shard_manifest")
            if not relative:
                raise RuntimeError(
                    "Completed generation timing is missing shard manifest"
                )
            output_dir = timing_path.resolve().parents[1]
            shard = output_dir / str(relative)
            if not shard.is_file():
                raise RuntimeError("Generation timing shard manifest is missing")
            if sha256_file(shard) != payload.get("shard_manifest_sha256"):
                raise RuntimeError("Generation timing shard manifest hash mismatch")
    return payload


def summarize_generation_timings(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    paths = sorted((root / "operational_runs").glob("*.json"))
    rows = [validate_generation_timing(path) for path in paths]
    complete_rows = [row for row in rows if row["status"] == "complete"]
    incomplete_rows = [row for row in rows if row["status"] != "complete"]
    accepted_publications: dict[str, str] = {}
    accepted_shards: set[str] = set()
    for row in complete_rows:
        relative = str(row["shard_manifest"])
        accepted_shards.add(relative)
        with (root / relative).open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        for publication in shard.get("publications", []):
            identity = str(publication["qualified_id"])
            digest = str(publication["publication_hash"])
            previous = accepted_publications.setdefault(identity, digest)
            if previous != digest:
                raise RuntimeError(
                    f"Conflicting accepted publication hashes for {identity}"
                )
    accepted_scenarios = {
        identity.rsplit(":", 1)[0] for identity in accepted_publications
    }
    successful_wall_s = sum(float(row["active_wall_s"]) for row in complete_rows)
    incomplete_wall_s = sum(float(row["active_wall_s"]) for row in incomplete_rows)
    aggregate: dict[str, Any] = {
        "schema_id": OPERATIONAL_TIMING_SUMMARY_SCHEMA_ID,
        "artifact_kind": "common-time-v2-generation-operational-summary",
        "generated_utc": _utc_now(),
        "output_dir": str(root),
        "invocation_count": len(rows),
        "complete_invocations": sum(row["status"] == "complete" for row in rows),
        "failed_invocations": sum(row["status"] == "failed" for row in rows),
        "running_or_interrupted_invocations": sum(
            row["status"] == "running" for row in rows
        ),
        "successful_wall_s": successful_wall_s,
        "successful_wall_hours": successful_wall_s / 3600.0,
        "failed_or_interrupted_wall_s": incomplete_wall_s,
        "failed_or_interrupted_wall_hours": incomplete_wall_s / 3600.0,
        "accepted_artifacts": {
            "unique_shards": len(accepted_shards),
            "unique_scenarios": len(accepted_scenarios),
            "unique_solver_rollouts": len(accepted_publications),
        },
        "contract_hashes": sorted({str(row["contract_hash"]) for row in rows}),
        "generation_contract_hashes": sorted(
            {
                str(row["generation_contract_hash"])
                for row in rows
                if row.get("generation_contract_hash") is not None
            }
        ),
        "code_state_hashes": sorted({str(row["code_state_hash"]) for row in rows}),
        "counts": {},
        "per_solver": {},
        "failed_or_interrupted_per_solver": {},
        "invocation_files": [path.relative_to(root).as_posix() for path in paths],
    }
    count_keys = sorted({key for row in rows for key in row.get("counts", {})})
    aggregate["counts"] = {
        key: sum(int(row.get("counts", {}).get(key, 0)) for row in rows)
        for key in count_keys
    }
    solver_names = sorted({name for row in rows for name in row.get("per_solver", {})})
    for solver in solver_names:
        entries = [row.get("per_solver", {}).get(solver, {}) for row in complete_rows]
        incomplete_entries = [
            row.get("per_solver", {}).get(solver, {}) for row in incomplete_rows
        ]
        aggregate["per_solver"][solver] = {
            key: sum(int(entry.get(key, 0)) for entry in entries)
            for key in (
                "generated",
                "reused",
                "accepted",
                "rejected",
                "natural_steps",
            )
        }
        aggregate["per_solver"][solver].update(
            {
                key: sum(float(entry.get(key, 0.0)) for entry in entries)
                for key in (
                    "solve_s",
                    "serialization_s",
                    "validation_s",
                    "worker_s",
                )
            }
        )
        aggregate["failed_or_interrupted_per_solver"][solver] = {
            key: sum(float(entry.get(key, 0.0)) for entry in incomplete_entries)
            for key in ("solve_s", "serialization_s", "validation_s", "worker_s")
        }
    aggregate_solver_worker_s = sum(
        float(row["worker_s"]) for row in aggregate["per_solver"].values()
    )
    aggregate["aggregate_solver_worker_s"] = aggregate_solver_worker_s
    aggregate["aggregate_solver_worker_hours"] = aggregate_solver_worker_s / 3600.0
    aggregate["successful_estimated_cost_usd"] = sum(
        float(row.get("estimated_cost_usd") or 0.0) for row in complete_rows
    )
    aggregate["failed_or_interrupted_estimated_cost_usd"] = sum(
        float(row.get("estimated_cost_usd") or 0.0) for row in incomplete_rows
    )
    return aggregate


def write_generation_timing_summary(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    path = root / "operational_timing_summary.json"
    _atomic_write_json(path, summarize_generation_timings(root))
    return path
