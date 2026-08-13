from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.hashing import sha256_file


CONTRACT_SCHEMA_ID = "tsunami-surrogate.common-time-v2.contract.v1"
ETA_SAMPLE_SCHEMA_ID = "tsunami-surrogate.common-time-v2.eta-sample.v1"
PUBLICATION_SCHEMA_ID = "tsunami-surrogate.common-time-v2.publication.v1"
OPERATIONAL_SHARD_SCHEMA_ID = "tsunami-surrogate.common-time-v2.operational-shard.v1"
PROVISIONAL_STATUS = "provisional"
ACCEPTED_STATUS = "accepted"
CANDIDATE_START = 0.0035
CANDIDATE_STEP = 0.0035
CANDIDATE_COUNT = 50
CANDIDATE_HORIZON = 0.175


@dataclass(frozen=True)
class RequestedOutputConfig:
    schema_id: str
    status: str
    execution_scope: str
    split: str
    requested_times: np.ndarray
    max_natural_steps: int
    collect_natural_step_health: bool
    eta_primary: bool
    debug_full_states: bool
    acknowledged_provisional: bool
    contract: dict[str, Any]
    contract_hash: str

    @property
    def enabled(self) -> bool:
        return True


_ALLOWED_REQUESTED_KEYS = {
    "enabled",
    "schema_id",
    "status",
    "execution_scope",
    "split",
    "start",
    "step",
    "count",
    "horizon",
    "times",
    "max_natural_steps",
    "collect_natural_step_health",
    "eta_primary",
    "debug_full_states",
    "acknowledge_provisional",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Cannot hash a non-finite float")
        return value
    return value


def stable_hash_payload(*, artifact_kind: str, payload: Any, schema_id: str) -> str:
    envelope = {
        "schema_id": str(schema_id),
        "artifact_kind": str(artifact_kind),
        "payload": _json_safe(payload),
    }
    encoded = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_array(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": [int(v) for v in array.shape],
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def split_qualified_identity(split: str, scenario_id: str) -> dict[str, str]:
    split_text = str(split).strip().lower()
    if split_text == "validation":
        split_text = "eval"
    if split_text not in {"train", "eval", "test"}:
        raise ValueError("split must be one of: train, eval, test")
    scenario_text = str(scenario_id).strip()
    if not scenario_text:
        raise ValueError("scenario_id must be non-empty")
    return {
        "split": split_text,
        "scenario_id": scenario_text,
        "qualified_id": f"{split_text}:{scenario_text}",
    }


def candidate_requested_times() -> np.ndarray:
    values = CANDIDATE_STEP * np.arange(1, CANDIDATE_COUNT + 1, dtype=np.float64)
    values[-1] = np.float64(CANDIDATE_HORIZON)
    return values


def validate_candidate_times(values: Any) -> np.ndarray:
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or times.size != CANDIDATE_COUNT:
        raise ValueError(
            f"requested times must be a 1-D vector of length {CANDIDATE_COUNT}"
        )
    if not np.isfinite(times).all() or np.any(times <= 0.0):
        raise ValueError("requested times must be finite and strictly positive")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("requested times must be strictly increasing")
    expected = candidate_requested_times()
    if not np.array_equal(times, expected):
        raise ValueError(
            "requested times do not exactly match the provisional 50-time grid"
        )
    return times.copy()


def build_candidate_contract(*, status: str = PROVISIONAL_STATUS) -> dict[str, Any]:
    normalized_status = str(status).strip().lower()
    if normalized_status not in {PROVISIONAL_STATUS, ACCEPTED_STATUS}:
        raise ValueError("common-time-v2 contract status must be provisional or accepted")
    return {
        "schema_id": CONTRACT_SCHEMA_ID,
        "status": normalized_status,
        "field": "eta",
        "time_semantics": "elapsed-benchmark-time-units",
        "requested_times": candidate_requested_times().tolist(),
        "initial_state_in_target": False,
        "extraction": "adjacent-natural-step-linear",
        "exact_knot": "copy-natural-state-with-zero-width-provenance",
        "multiple_requests_per_bracket": True,
        "extrapolation": "forbidden",
        "interpolation_feedback": "forbidden",
        "timestamps_dtype": "float64",
        "trajectory_eta_dtype": "float32",
        "eta_primary": True,
        "sample_schema_id": ETA_SAMPLE_SCHEMA_ID,
        "required_provenance": [
            "requested_timestamps",
            "left_natural_timestamps",
            "right_natural_timestamps",
            "interpolation_weights",
            "bracket_widths",
            "exact_knot",
            "natural_step_indices",
            "natural_dt_history",
            "total_natural_steps",
        ],
    }


def contract_hash(
    contract: Mapping[str, Any] | None = None, *, status: str = PROVISIONAL_STATUS
) -> str:
    payload = (
        build_candidate_contract(status=status)
        if contract is None
        else dict(contract)
    )
    return stable_hash_payload(
        artifact_kind="semantic-output-contract",
        payload=payload,
        schema_id=CONTRACT_SCHEMA_ID,
    )


def parse_requested_output_config(raw: Any) -> RequestedOutputConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("requested_output section must be a mapping")
    unknown = sorted(set(str(k) for k in raw) - _ALLOWED_REQUESTED_KEYS)
    if unknown:
        raise ValueError(f"Unknown requested_output keys: {unknown}")
    if not bool(raw.get("enabled", False)):
        return None

    schema_id = str(raw.get("schema_id", CONTRACT_SCHEMA_ID))
    if schema_id != CONTRACT_SCHEMA_ID:
        raise ValueError(f"requested_output.schema_id must be {CONTRACT_SCHEMA_ID!r}")
    status = str(raw.get("status", PROVISIONAL_STATUS)).strip().lower()
    if status not in {PROVISIONAL_STATUS, ACCEPTED_STATUS}:
        raise ValueError("requested_output.status must be provisional or accepted")
    execution_scope = str(raw.get("execution_scope", "preparation-only")).strip()
    if status == PROVISIONAL_STATUS and execution_scope != "preparation-only":
        raise ValueError(
            "provisional requested_output.execution_scope must be preparation-only"
        )
    if status == ACCEPTED_STATUS and execution_scope != "production":
        raise ValueError("accepted requested_output.execution_scope must be production")
    split = split_qualified_identity(str(raw.get("split", "train")), "placeholder")[
        "split"
    ]

    start = float(raw.get("start", CANDIDATE_START))
    step = float(raw.get("step", CANDIDATE_STEP))
    count = int(raw.get("count", CANDIDATE_COUNT))
    horizon = float(raw.get("horizon", CANDIDATE_HORIZON))
    if (start, step, count, horizon) != (
        CANDIDATE_START,
        CANDIDATE_STEP,
        CANDIDATE_COUNT,
        CANDIDATE_HORIZON,
    ):
        raise ValueError(
            "requested_output start/step/count/horizon must match the provisional contract"
        )
    derived = candidate_requested_times()
    times = validate_candidate_times(raw.get("times", derived))

    max_steps = int(raw.get("max_natural_steps", 1000))
    if max_steps <= 0:
        raise ValueError("requested_output.max_natural_steps must be positive")
    collect_health = bool(raw.get("collect_natural_step_health", True))
    if not collect_health:
        raise ValueError("common-time-v2 requires natural-step health collection")
    eta_primary = bool(raw.get("eta_primary", True))
    if not eta_primary:
        raise ValueError("common-time-v2 requires eta_primary=true")

    contract = build_candidate_contract(status=status)
    return RequestedOutputConfig(
        schema_id=schema_id,
        status=status,
        execution_scope=execution_scope,
        split=split,
        requested_times=times,
        max_natural_steps=max_steps,
        collect_natural_step_health=collect_health,
        eta_primary=eta_primary,
        debug_full_states=bool(raw.get("debug_full_states", False)),
        acknowledged_provisional=bool(raw.get("acknowledge_provisional", False)),
        contract=contract,
        contract_hash=contract_hash(contract),
    )


def resolved_config_hash(
    *,
    solver_name: str,
    solver_config: Mapping[str, Any],
    dataset_semantics: Mapping[str, Any],
) -> str:
    return stable_hash_payload(
        artifact_kind="resolved-generation-config",
        payload={
            "solver_name": str(solver_name),
            "solver": dict(solver_config),
            "dataset_semantics": dict(dataset_semantics),
        },
        schema_id=CONTRACT_SCHEMA_ID,
    )


def code_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )

    commit_result = _git("rev-parse", "HEAD")
    commit = (
        commit_result.stdout.strip() if commit_result.returncode == 0 else "unknown"
    )
    status_result = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "scripts",
        "configs",
        "pyproject.toml",
        "requirements.txt",
        "requirements.lock",
    )
    status = status_result.stdout if status_result.returncode == 0 else "unknown"

    included_roots = ("src", "scripts", "configs")
    included_names = {"pyproject.toml", "requirements.txt", "requirements.lock"}
    included_suffixes = {".py", ".yaml", ".yml", ".toml", ".lock"}
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if not (rel.parts[0] in included_roots or rel.as_posix() in included_names):
            continue
        if (
            path.suffix not in included_suffixes
            and rel.as_posix() not in included_names
        ):
            continue
        if "__pycache__" in rel.parts:
            continue
        files.append(
            {
                "path": rel.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    source_inventory_hash = stable_hash_payload(
        artifact_kind="source-file-inventory",
        payload=files,
        schema_id=CONTRACT_SCHEMA_ID,
    )
    payload = {
        "git_commit": commit,
        "dirty": bool(status.strip()),
        "source_inventory_hash": source_inventory_hash,
        "source_file_count": len(files),
    }
    payload["code_state_hash"] = stable_hash_payload(
        artifact_kind="code-state", payload=payload, schema_id=CONTRACT_SCHEMA_ID
    )
    return payload


def authoritative_input_fingerprint(
    *,
    split: str,
    sample_index: int,
    scenario_id: str,
    bathymetry_type: str,
    source_type: str,
    source_strength: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> str:
    identity = split_qualified_identity(split, scenario_id)
    return stable_hash_payload(
        artifact_kind="authoritative-static-input-fingerprint",
        schema_id=CONTRACT_SCHEMA_ID,
        payload={
            **identity,
            "sample_index": int(sample_index),
            "bathymetry_type": str(bathymetry_type),
            "source_type": str(source_type),
            "source_strength": hash_array(np.asarray(source_strength)),
            "arrays": {
                str(name): hash_array(values) for name, values in sorted(arrays.items())
            },
        },
    )


def exact_times_equal(values: Any, expected: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    target = np.asarray(expected, dtype=np.float64)
    return array.shape == target.shape and bool(np.array_equal(array, target))


def validate_publication(
    sample_dir: str | Path,
    *,
    expected_identity: Mapping[str, str] | None = None,
    expected_contract_hash: str | None = None,
    expected_config_hash: str | None = None,
    expected_code_state_hash: str | None = None,
    expected_input_fingerprint: str | None = None,
    expected_times: Sequence[float] | None = None,
    expected_solver_name: str | None = None,
    expected_sample_index: int | None = None,
    expected_authoritative_input_fingerprint: str | None = None,
    expected_authoritative_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    directory = Path(sample_dir)
    publication_path = directory / "publication.json"
    if not publication_path.is_file():
        raise RuntimeError(
            f"Missing requested-output publication marker: {publication_path}"
        )
    with publication_path.open("r", encoding="utf-8") as handle:
        publication = json.load(handle)
    if publication.get("schema_id") != PUBLICATION_SCHEMA_ID:
        raise RuntimeError("Requested-output publication schema mismatch")
    listed_files = publication.get("files")
    if not isinstance(listed_files, list):
        raise RuntimeError("Requested-output publication files must be a list")
    file_names = [str(item.get("name", "")) for item in listed_files]
    required_names = {"sample.npz", "provenance.npz", "meta.json"}
    if not required_names.issubset(file_names):
        raise RuntimeError("Requested-output publication is missing mandatory payloads")
    if len(file_names) != len(set(file_names)):
        raise RuntimeError("Requested-output publication contains duplicate payloads")
    for name in file_names:
        candidate = Path(name)
        if candidate.is_absolute() or candidate.name != name or ".." in candidate.parts:
            raise RuntimeError(
                "Requested-output publication contains an unsafe payload path"
            )
    for item in listed_files:
        path = directory / str(item["name"])
        if not path.is_file():
            raise RuntimeError(f"Requested-output payload is missing: {path}")
        if int(path.stat().st_size) != int(item["size_bytes"]):
            raise RuntimeError(f"Requested-output payload size mismatch: {path}")
        if sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"Requested-output payload hash mismatch: {path}")

    checks = {
        "contract_hash": expected_contract_hash,
        "resolved_config_hash": expected_config_hash,
        "code_state_hash": expected_code_state_hash,
        "input_fingerprint": expected_input_fingerprint,
        "authoritative_input_fingerprint": (
            expected_authoritative_input_fingerprint
        ),
        "authoritative_inventory_sha256": expected_authoritative_inventory_sha256,
    }
    for key, expected in checks.items():
        if expected is not None and publication.get(key) != expected:
            raise RuntimeError(f"Requested-output publication {key} mismatch")
    if expected_identity is not None:
        for key in ("split", "scenario_id"):
            if publication.get(key) != expected_identity.get(key):
                raise RuntimeError(f"Requested-output publication {key} mismatch")
    if (
        expected_solver_name is not None
        and publication.get("solver_name") != expected_solver_name
    ):
        raise RuntimeError("Requested-output publication solver_name mismatch")
    if expected_sample_index is not None and int(
        publication.get("sample_index", -1)
    ) != int(expected_sample_index):
        raise RuntimeError("Requested-output publication sample_index mismatch")
    if expected_times is not None:
        sample_path = directory / "sample.npz"
        with np.load(sample_path, allow_pickle=False) as payload:
            if "timestamps" not in payload or not exact_times_equal(
                payload["timestamps"], expected_times
            ):
                raise RuntimeError("Requested-output publication timestamp mismatch")
    return publication


def validate_operational_shard(
    manifest_path: str | Path,
    *,
    expected_contract_hash: str,
    expected_publication_hashes: Mapping[str, str],
    expected_split: str | None = None,
    expected_start_index: int | None = None,
    expected_stop_index: int | None = None,
    expected_solver_names: Sequence[str] | None = None,
    expected_config_hashes: Mapping[str, str] | None = None,
    expected_code_state_hash: str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_id") != OPERATIONAL_SHARD_SCHEMA_ID:
        raise RuntimeError("Operational shard schema mismatch")
    if manifest.get("contract_hash") != expected_contract_hash:
        raise RuntimeError("Operational shard contract hash mismatch")
    optional_checks = {
        "split": expected_split,
        "start_index": expected_start_index,
        "stop_index": expected_stop_index,
        "code_state_hash": expected_code_state_hash,
    }
    for key, expected in optional_checks.items():
        if expected is not None and manifest.get(key) != expected:
            raise RuntimeError(f"Operational shard {key} mismatch")
    if expected_solver_names is not None and manifest.get("solver_names") != sorted(
        str(value) for value in expected_solver_names
    ):
        raise RuntimeError("Operational shard solver_names mismatch")
    if expected_config_hashes is not None and manifest.get(
        "resolved_config_hashes"
    ) != {
        str(key): str(value) for key, value in sorted(expected_config_hashes.items())
    }:
        raise RuntimeError("Operational shard resolved_config_hashes mismatch")
    observed = {
        str(item["qualified_id"]): str(item["publication_hash"])
        for item in manifest.get("publications", [])
    }
    if observed != {str(k): str(v) for k, v in expected_publication_hashes.items()}:
        raise RuntimeError("Operational shard publication set mismatch")
    if not bool(manifest.get("complete", False)):
        raise RuntimeError("Operational shard is incomplete")
    return manifest


def write_operational_shard_manifest(
    path: str | Path,
    *,
    split: str,
    start_index: int,
    stop_index: int,
    contract_hash_value: str,
    publication_hashes: Mapping[str, str],
    complete: bool,
    solver_names: Sequence[str] = (),
    resolved_config_hashes: Mapping[str, str] | None = None,
    code_state_hash: str | None = None,
) -> dict[str, Any]:
    if start_index < 1 or stop_index < start_index:
        raise ValueError("Invalid operational shard range")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite operational shard: {output}")
    manifest = {
        "schema_id": OPERATIONAL_SHARD_SCHEMA_ID,
        "artifact_kind": "requested-output-operational-shard",
        "split": split_qualified_identity(split, "placeholder")["split"],
        "start_index": int(start_index),
        "stop_index": int(stop_index),
        "contract_hash": str(contract_hash_value),
        "solver_names": sorted(str(value) for value in solver_names),
        "resolved_config_hashes": {
            str(key): str(value)
            for key, value in sorted((resolved_config_hashes or {}).items())
        },
        "code_state_hash": code_state_hash,
        "complete": bool(complete),
        "publications": [
            {"qualified_id": str(key), "publication_hash": str(value)}
            for key, value in sorted(publication_hashes.items())
        ],
    }
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(staging, output)
    return manifest


def atomic_replace_directory(staging: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite finalized publication: {final}")
    os.replace(staging, final)
