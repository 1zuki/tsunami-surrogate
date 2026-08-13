#!/usr/bin/env python
"""Fail-closed preflight for the final common-time-v2 evaluation suite."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.common_time_v2 import (  # noqa: E402
    PUBLICATION_SCHEMA_ID,
    candidate_requested_times,
    stable_hash_payload,
)
from src.models.signature import model_config_signature  # noqa: E402
from src.training.checkpointing import capture_data_provenance  # noqa: E402
from src.training.checkpointing import training_contract_signature  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.hashing import sha256_file  # noqa: E402


SUITE_SCHEMA_ID = "tsunami-surrogate.final-v2-evaluation-suite.v1"
REPORT_SCHEMA_ID = "tsunami-surrogate.final-v2-evaluation-preflight.v1"


class PreflightError(RuntimeError):
    """Raised when an evaluation prerequisite is absent or incompatible."""


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _canonical_dataset_path(value: str | Path) -> Path:
    path = _repo_path(value)
    if (
        path.name == "eval_dataset.npz"
        and not path.is_file()
        and (path.parent / "shards_manifest.json").is_file()
    ):
        path = path.parent
    return path.resolve()


def _evaluation_code_state() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        listed = subprocess.check_output(
            [
                "git",
                "ls-files",
                "-co",
                "--exclude-standard",
                "-z",
                "--",
                "configs",
                "scripts",
                "src",
            ],
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError("Could not capture evaluation code state") from exc
    digest = hashlib.sha256()
    files = sorted(Path(raw.decode("utf-8")) for raw in listed.split(b"\0") if raw)
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "git_commit": head,
        "evaluation_tree_sha256": digest.hexdigest(),
        "files_hashed": len(files),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreflightError(f"Missing required JSON file: {_relative(path)}") from exc
    except json.JSONDecodeError as exc:
        raise PreflightError(f"Malformed JSON file: {_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"Expected JSON object: {_relative(path)}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise PreflightError(f"Missing required JSONL file: {_relative(path)}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(
                    f"Malformed JSONL row: {_relative(path)}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PreflightError(
                    f"Expected JSON object row: {_relative(path)}:{line_number}"
                )
            yield row


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    os.replace(staging, path)


def load_suite_contract(path: str | Path) -> dict[str, Any]:
    contract_path = _repo_path(path)
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise PreflightError(
            f"Missing evaluation contract: {_relative(contract_path)}"
        ) from exc
    if not isinstance(payload, dict):
        raise PreflightError("Evaluation contract must be a YAML mapping")
    if payload.get("schema_id") != SUITE_SCHEMA_ID:
        raise PreflightError(f"Evaluation contract schema must be {SUITE_SCHEMA_ID!r}")
    return payload


def _expected_times(contract: Mapping[str, Any]) -> np.ndarray:
    requested = contract.get("scientific_scope", {}).get("requested_times", {})
    expected = candidate_requested_times()
    observed = (
        float(requested.get("start", -1.0)),
        float(requested.get("step", -1.0)),
        int(requested.get("count", -1)),
        float(requested.get("horizon", -1.0)),
    )
    required = (0.0035, 0.0035, 50, 0.175)
    if observed != required:
        raise PreflightError(
            f"Evaluation contract requested-time tuple is {observed}, expected {required}"
        )
    return expected


def _validate_shard_manifest(
    split_dir: Path,
    *,
    expected_count: int,
    expected_inputs_shape: tuple[int, int, int],
    expected_targets_shape: tuple[int, int, int],
) -> dict[str, Any]:
    manifest = _read_json(split_dir / "shards_manifest.json")
    if int(manifest.get("num_samples", -1)) != int(expected_count):
        raise PreflightError(
            f"Sample count mismatch in {_relative(split_dir)}: "
            f"{manifest.get('num_samples')} != {expected_count}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PreflightError(f"No shards declared in {_relative(split_dir)}")
    counted = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise PreflightError(f"Invalid shard row in {_relative(split_dir)}")
        shard_path = split_dir / str(shard.get("file", ""))
        if not shard_path.is_file():
            raise PreflightError(f"Missing dataset shard: {_relative(shard_path)}")
        n = int(shard.get("num_samples", -1))
        inputs_shape = tuple(int(v) for v in shard.get("inputs_shape", []))
        targets_shape = tuple(int(v) for v in shard.get("targets_shape", []))
        if inputs_shape != (n, *expected_inputs_shape):
            raise PreflightError(
                f"Input shape mismatch in {_relative(shard_path)}: {inputs_shape}"
            )
        if targets_shape != (n, *expected_targets_shape):
            raise PreflightError(
                f"Target shape mismatch in {_relative(shard_path)}: {targets_shape}"
            )
        counted += n
    if counted != expected_count:
        raise PreflightError(
            f"Shard row count mismatch in {_relative(split_dir)}: "
            f"{counted} != {expected_count}"
        )
    return manifest


def _validate_flat_manifest(
    split_dir: Path,
    *,
    expected_count: int,
    expected_inputs_shape: tuple[int, int, int],
    expected_targets_shape: tuple[int, int, int],
) -> dict[str, Any]:
    manifest = _read_json(split_dir / "eval_manifest.json")
    if manifest.get("schema_id") != "tsunami-surrogate.processed-dataset.v2":
        raise PreflightError(
            f"Flat processed manifest is not v2-bound: {_relative(split_dir)}"
        )
    if bool(manifest.get("sharded", False)):
        raise PreflightError(
            f"Expected a flat processed dataset: {_relative(split_dir)}"
        )
    inputs_shape = tuple(int(v) for v in manifest.get("inputs_shape", []))
    targets_shape = tuple(int(v) for v in manifest.get("targets_shape", []))
    if inputs_shape != (expected_count, *expected_inputs_shape):
        raise PreflightError(
            f"Flat input shape mismatch in {_relative(split_dir)}: {inputs_shape}"
        )
    if targets_shape != (expected_count, *expected_targets_shape):
        raise PreflightError(
            f"Flat target shape mismatch in {_relative(split_dir)}: {targets_shape}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise PreflightError(
            f"Flat processed artifact hashes are missing: {_relative(split_dir)}"
        )
    for name, expected_hash in artifacts.items():
        path = split_dir / str(name)
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise PreflightError(
                f"Flat processed artifact hash mismatch: {_relative(path)}"
            )
    return manifest


def _validate_raw_timestamp_payload(
    sample_dir: Path,
    *,
    row: Mapping[str, Any],
    expected_times: np.ndarray,
    expected_shape: tuple[int, int, int],
    expected_contract_hash: str,
    expected_schema_id: str,
    expected_publication_split: str,
    expected_publication_hash: str | None,
    deep_payload_audit: bool,
) -> None:
    publication = _read_json(sample_dir / "publication.json")
    if publication.get("schema_id") != PUBLICATION_SCHEMA_ID:
        raise PreflightError(f"Publication schema mismatch: {_relative(sample_dir)}")
    checks = {
        "contract_hash": expected_contract_hash,
        "scenario_id": str(row.get("scenario_id")),
        "sample_index": int(row.get("sample_index", -1)),
        "solver_name": str(row.get("solver_name")),
        "split": expected_publication_split,
    }
    for key, expected in checks.items():
        if publication.get(key) != expected:
            raise PreflightError(
                f"Publication {key} mismatch in {_relative(sample_dir)}: "
                f"{publication.get(key)!r} != {expected!r}"
            )
    if expected_publication_hash is not None:
        observed_hash = stable_hash_payload(
            artifact_kind="requested-output-publication-record",
            payload=publication,
            schema_id=PUBLICATION_SCHEMA_ID,
        )
        if observed_hash != expected_publication_hash:
            raise PreflightError(
                f"Operational publication hash mismatch: {_relative(sample_dir)}"
            )
    listed_files = publication.get("files")
    if not isinstance(listed_files, list):
        raise PreflightError(
            f"Publication payload inventory is missing: {_relative(sample_dir)}"
        )
    required_names = {"sample.npz", "provenance.npz", "meta.json"}
    observed_names = {str(item.get("name", "")) for item in listed_files}
    if not required_names.issubset(observed_names):
        raise PreflightError(
            f"Publication payload inventory is incomplete: {_relative(sample_dir)}"
        )
    for item in listed_files:
        payload_path = sample_dir / str(item.get("name", ""))
        if not payload_path.is_file() or int(payload_path.stat().st_size) != int(
            item.get("size_bytes", -1)
        ):
            raise PreflightError(
                f"Publication payload is missing or size-mismatched: "
                f"{_relative(payload_path)}"
            )
    if not deep_payload_audit:
        return
    for item in listed_files:
        payload_path = sample_dir / str(item["name"])
        if sha256_file(payload_path) != str(item.get("sha256", "")):
            raise PreflightError(
                f"Publication payload hash mismatch: {_relative(payload_path)}"
            )
    sample_path = sample_dir / "sample.npz"
    if not sample_path.is_file():
        raise PreflightError(f"Missing raw sample payload: {_relative(sample_path)}")
    try:
        with np.load(sample_path, allow_pickle=False) as payload:
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
            if "trajectory_eta" not in payload:
                raise KeyError("trajectory_eta")
            trajectory_shape = tuple(
                int(value) for value in np.asarray(payload["trajectory_eta"]).shape
            )
            schema_id = str(np.asarray(payload["schema_id"]).reshape(-1)[0])
            contract_hash = str(np.asarray(payload["contract_hash"]).reshape(-1)[0])
    except (KeyError, OSError, ValueError) as exc:
        raise PreflightError(
            f"Unreadable common-time-v2 payload: {_relative(sample_path)}"
        ) from exc
    if not np.array_equal(timestamps, expected_times):
        raise PreflightError(f"Requested timestamps mismatch: {_relative(sample_path)}")
    if trajectory_shape != expected_shape:
        raise PreflightError(
            f"Raw trajectory shape mismatch in {_relative(sample_path)}: "
            f"{trajectory_shape} != {expected_shape}"
        )
    if schema_id != expected_schema_id or contract_hash != expected_contract_hash:
        raise PreflightError(
            f"Payload schema/contract mismatch: {_relative(sample_path)}"
        )


def _run_payload_audits(
    jobs: list[dict[str, Any]],
    *,
    deep_payload_audit: bool,
    workers: int,
    label: str,
    progress_callback: Callable[[str], None] | None,
) -> None:
    if not jobs:
        return
    if workers <= 0:
        raise PreflightError("Payload-audit worker count must be positive")
    total = len(jobs)
    active_workers = min(workers, total)
    if progress_callback is not None:
        progress_callback(
            f"[payload-audit] start {label} samples={total} workers={active_workers}"
        )

    def audit(job: Mapping[str, Any]) -> None:
        _validate_raw_timestamp_payload(
            **job,
            deep_payload_audit=deep_payload_audit,
        )

    progress_interval = max(100, min(1000, total // 10 or 1))
    with ThreadPoolExecutor(max_workers=active_workers) as executor:
        for completed, _ in enumerate(executor.map(audit, jobs), start=1):
            if (
                progress_callback is not None
                and (completed % progress_interval == 0 or completed == total)
            ):
                progress_callback(
                    f"[payload-audit] progress {label} {completed}/{total}"
                )
    if progress_callback is not None:
        progress_callback(f"[payload-audit] complete {label} samples={total}")


def _validate_processed_split(
    split_dir: Path,
    *,
    expected_count: int,
    expected_solver: str,
    expected_publication_split: str,
    expected_contract_hash: str,
    expected_schema_id: str,
    expected_times: np.ndarray,
    publication_shape: tuple[int, int],
    solver_shape: tuple[int, int] | None,
    lineage_schema_id: str | None = None,
    expected_publication_hashes: Mapping[str, str] | None = None,
    deep_payload_audit: bool = False,
    payload_audit_workers: int = 8,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    if (split_dir / "shards_manifest.json").is_file():
        _validate_shard_manifest(
            split_dir,
            expected_count=expected_count,
            expected_inputs_shape=(3, *publication_shape),
            expected_targets_shape=(50, *publication_shape),
        )
    else:
        _validate_flat_manifest(
            split_dir,
            expected_count=expected_count,
            expected_inputs_shape=(3, *publication_shape),
            expected_targets_shape=(50, *publication_shape),
        )
    rows: dict[str, dict[str, Any]] = {}
    payload_audit_jobs: list[dict[str, Any]] = []
    if progress_callback is not None:
        progress_callback(f"[dataset-preflight] metadata {_relative(split_dir)}")
    for row in _iter_jsonl(split_dir / "meta.jsonl"):
        scenario_id = str(row.get("scenario_id", "")).strip()
        if not scenario_id or scenario_id in rows:
            raise PreflightError(
                f"Missing/duplicate scenario ID in {_relative(split_dir)}: {scenario_id!r}"
            )
        required = {
            "schema_id": expected_schema_id,
            "contract_hash": expected_contract_hash,
            "solver_name": expected_solver,
            "split": expected_publication_split,
            "quality_status": "ok",
            "requested_output_count": 50,
            "covered_requested_output_count": 50,
        }
        for key, expected in required.items():
            if row.get(key) != expected:
                raise PreflightError(
                    f"{key} mismatch for {scenario_id} in {_relative(split_dir)}: "
                    f"{row.get(key)!r} != {expected!r}"
                )
        if tuple(row.get("trajectory_eta_shape", [])) != (
            50,
            *publication_shape,
        ):
            raise PreflightError(
                f"Processed trajectory shape mismatch for {scenario_id}"
            )
        domain = row.get("computational_domain")
        if not isinstance(domain, Mapping):
            raise PreflightError(
                f"Missing computational-domain provenance for {scenario_id}"
            )
        if tuple(domain.get("publication_shape", [])) != publication_shape:
            raise PreflightError(f"Publication-domain mismatch for {scenario_id}")
        if (
            solver_shape is not None
            and tuple(domain.get("solver_shape", [])) != solver_shape
        ):
            raise PreflightError(f"Solver-domain mismatch for {scenario_id}")
        if lineage_schema_id is not None:
            lineage = row.get("input_lineage")
            if (
                not isinstance(lineage, Mapping)
                or lineage.get("schema_id") != lineage_schema_id
            ):
                raise PreflightError(
                    f"Input-lineage mismatch for {scenario_id} in {_relative(split_dir)}"
                )
        sample_dir = _repo_path(str(row.get("sample_dir", "")))
        publication_key = (
            f"{expected_publication_split}:{scenario_id}:{expected_solver}"
        )
        payload_audit_jobs.append(
            {
                "sample_dir": sample_dir,
                "row": row,
                "expected_times": expected_times,
                "expected_shape": (50, *publication_shape),
                "expected_contract_hash": expected_contract_hash,
                "expected_schema_id": expected_schema_id,
                "expected_publication_split": expected_publication_split,
                "expected_publication_hash": (
                    None
                    if expected_publication_hashes is None
                    else expected_publication_hashes.get(publication_key)
                ),
            }
        )
        if (
            expected_publication_hashes is not None
            and publication_key not in expected_publication_hashes
        ):
            raise PreflightError(
                f"Processed row is absent from the frozen operational shard: "
                f"{publication_key}"
            )
        rows[scenario_id] = row
    if len(rows) != expected_count:
        raise PreflightError(
            f"Metadata row count mismatch in {_relative(split_dir)}: "
            f"{len(rows)} != {expected_count}"
        )
    _run_payload_audits(
        payload_audit_jobs,
        deep_payload_audit=deep_payload_audit,
        workers=payload_audit_workers,
        label=_relative(split_dir),
        progress_callback=progress_callback,
    )
    return rows


def _checkpoint_load(path: Path) -> dict[str, Any]:
    try:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise PreflightError(f"Unreadable checkpoint: {_relative(path)}") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"Checkpoint is not a mapping: {_relative(path)}")
    return payload


def _artifact_hashes(entry: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(entry, Mapping):
        return {}
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return {}
    return {str(key): str(value) for key, value in artifacts.items()}


def _training_entry(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    provenance = payload.get("data_provenance")
    if not isinstance(provenance, Mapping):
        return None
    datasets = provenance.get("datasets")
    if not isinstance(datasets, Mapping):
        return None
    for key in ("train_path", "path"):
        value = datasets.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def validate_completed_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    best_path = _repo_path(checkpoint_path)
    if not best_path.is_file():
        raise PreflightError(f"Missing required checkpoint: {_relative(best_path)}")
    run_dir = best_path.parent
    last_path = run_dir / "checkpoints" / "last.pt"
    history_path = run_dir / "history.json"
    if not last_path.is_file() or not history_path.is_file():
        raise PreflightError(
            f"Checkpoint run is missing last.pt/history.json: {_relative(run_dir)}"
        )

    payload = _checkpoint_load(last_path)
    best_payload = _checkpoint_load(best_path)
    cfg = payload.get("config")
    if not isinstance(cfg, Mapping):
        raise PreflightError(f"Checkpoint config is missing: {_relative(last_path)}")
    seed = int(cfg.get("seed", -1))
    if expected_seed is not None and seed != expected_seed:
        raise PreflightError(
            f"Checkpoint seed mismatch in {_relative(last_path)}: "
            f"{seed} != {expected_seed}"
        )

    train_cfg = cfg.get("train")
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}
    early_cfg = train_cfg.get("early_stopping")
    if not isinstance(early_cfg, Mapping):
        early_cfg = {}
    configured_epochs = int(train_cfg.get("epochs", -1))
    patience = int(early_cfg.get("patience", -1))
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, Mapping):
        trainer_state = {}
    last_epoch = int(trainer_state.get("epoch", payload.get("epoch", -1)))
    best_epoch = int(best_payload.get("epoch", -1))
    early_count = int(trainer_state.get("early_count", -1))
    horizon_complete = configured_epochs > 0 and last_epoch >= configured_epochs
    early_complete = patience > 0 and early_count >= patience
    if not (horizon_complete or early_complete):
        raise PreflightError(
            f"Training is incomplete for {_relative(run_dir)}: "
            f"last_epoch={last_epoch}, configured_epochs={configured_epochs}, "
            f"early_count={early_count}, patience={patience}"
        )
    if best_epoch < 1 or best_epoch > last_epoch:
        raise PreflightError(
            f"Best-checkpoint epoch is invalid for {_relative(run_dir)}: "
            f"best_epoch={best_epoch}, last_epoch={last_epoch}"
        )
    expected_signature = model_config_signature(cfg)
    last_signature = payload.get("model_signature")
    best_signature = best_payload.get("model_signature")
    if last_signature != expected_signature or best_signature != expected_signature:
        raise PreflightError(
            f"Best/last model signature mismatch for {_relative(run_dir)}"
        )
    best_cfg = best_payload.get("config")
    if not isinstance(best_cfg, Mapping):
        raise PreflightError(
            f"Best checkpoint config is missing: {_relative(best_path)}"
        )
    if int(best_cfg.get("seed", -1)) != seed:
        raise PreflightError(f"Best/last seed mismatch for {_relative(run_dir)}")
    if best_payload.get("data_provenance") != payload.get("data_provenance"):
        raise PreflightError(
            f"Best/last data provenance mismatch for {_relative(run_dir)}"
        )
    best_contract = best_payload.get("training_contract")
    if best_contract is None:
        best_contract = training_contract_signature(best_cfg)
    last_contract = payload.get("training_contract")
    if last_contract is None:
        last_contract = training_contract_signature(cfg)
    if best_contract != last_contract:
        raise PreflightError(
            f"Best/last training contract mismatch for {_relative(run_dir)}"
        )
    for checkpoint_label, checkpoint_payload in (
        ("best", best_payload),
        ("last", payload),
    ):
        metrics = checkpoint_payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PreflightError(
                f"{checkpoint_label} checkpoint metrics are missing: "
                f"{_relative(run_dir)}"
            )
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise PreflightError(
                    f"Non-finite {checkpoint_label} metric {name} in "
                    f"{_relative(run_dir)}"
                )

    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            f"Unreadable training history: {_relative(history_path)}"
        ) from exc
    if not isinstance(history, list) or not history:
        raise PreflightError(f"Empty training history: {_relative(history_path)}")
    observed_epochs = [
        int(row.get("epoch", -1)) for row in history if isinstance(row, Mapping)
    ]
    if observed_epochs != list(range(1, last_epoch + 1)):
        raise PreflightError(
            f"Training history is not contiguous through epoch {last_epoch}: "
            f"{_relative(history_path)}"
        )

    checkpoint_entry = _training_entry(payload)
    expected_hashes = _artifact_hashes(checkpoint_entry)
    runtime_provenance = capture_data_provenance(cfg)
    runtime_entry = _training_entry({"data_provenance": runtime_provenance})
    observed_hashes = _artifact_hashes(runtime_entry)
    if not expected_hashes or expected_hashes != observed_hashes:
        raise PreflightError(
            f"Checkpoint-bound training-data hashes do not match current data for "
            f"{_relative(run_dir)}"
        )

    summary = {
        "checkpoint": _relative(best_path),
        "last_checkpoint": _relative(last_path),
        "seed": seed,
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "configured_epochs": configured_epochs,
        "early_count": early_count,
        "patience": patience,
        "completion": "horizon" if horizon_complete else "early_stopping",
        "training_data_identity_strength": (
            None
            if checkpoint_entry is None
            else checkpoint_entry.get("identity_strength")
        ),
        "training_data_artifacts": expected_hashes,
        "best_metrics": {
            str(name): float(value)
            for name, value in best_payload["metrics"].items()
            if not isinstance(value, bool) and isinstance(value, (int, float))
        },
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
    }
    del payload, best_payload
    gc.collect()
    return summary


def _validate_checksum_file(path: Path) -> int:
    if not path.is_file():
        raise PreflightError(f"Missing checksum file: {_relative(path)}")
    checked = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        digest, separator, relative_name = line.partition("  ")
        if not separator or len(digest) != 64:
            raise PreflightError(
                f"Malformed checksum row: {_relative(path)}:{line_number}"
            )
        target = path.parent / relative_name
        if not target.is_file():
            raise PreflightError(f"Checksum target is missing: {_relative(target)}")
        observed = hashlib.sha256(target.read_bytes()).hexdigest()
        if observed != digest:
            raise PreflightError(f"Checksum mismatch: {_relative(target)}")
        checked += 1
    if checked == 0:
        raise PreflightError(f"Checksum file is empty: {_relative(path)}")
    return checked


def _validate_numerical_artifacts(
    specs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        root = _repo_path(str(spec["root"]))
        checked = sum(
            _validate_checksum_file(root / str(relative))
            for relative in spec.get("checksum_files", [])
        )
        decision_path = root / str(spec["decision_path"])
        decision = _read_json(decision_path)
        key = str(spec["decision_key"])
        expected = spec["expected_decision"]
        if decision.get(key) != expected:
            raise PreflightError(
                f"Numerical decision mismatch in {_relative(decision_path)}: "
                f"{decision.get(key)!r} != {expected!r}"
            )
        summaries.append(
            {
                "id": str(spec["id"]),
                "root": _relative(root),
                "checksums_verified": checked,
                "decision": expected,
            }
        )
    return summaries


def _validate_frozen_file(spec: Mapping[str, Any], *, label: str) -> Path:
    path = _repo_path(str(spec["path"]))
    if not path.is_file():
        raise PreflightError(f"Missing frozen {label}: {_relative(path)}")
    expected = str(spec["sha256"])
    observed = sha256_file(path)
    if observed != expected:
        raise PreflightError(f"Frozen {label} hash mismatch: {_relative(path)}")
    return path


def _validate_generation_artifacts(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    expected_contract_hash = str(contract["scientific_scope"]["contract_hash"])
    expected_code_state = (
        "1199fad6503750b6892cd451ca7130c885818d49e14165704cce5590a478ac93"
    )
    expected_solvers = ["boussinesq", "swe_hydrostatic", "swe_muscl_hr"]
    summaries: list[dict[str, Any]] = []
    publication_indexes: dict[str, dict[str, str]] = {}
    frozen = contract["main_datasets"].get("frozen_generation_artifacts", {})
    split_names = {"train": "train", "val": "eval", "test": "test"}
    for processed_split, publication_split in split_names.items():
        split_spec = frozen.get(processed_split)
        if not isinstance(split_spec, Mapping):
            raise PreflightError(
                f"Missing frozen generation artifacts for {processed_split}"
            )
        _validate_frozen_file(
            split_spec["config_snapshot"],
            label=f"{processed_split} configuration snapshot",
        )
        _validate_frozen_file(
            split_spec["scenario_manifest"],
            label=f"{processed_split} scenario manifest",
        )
        for solver, manifest_spec in split_spec["solver_manifests"].items():
            _validate_frozen_file(
                manifest_spec,
                label=f"{processed_split} {solver} manifest",
            )
        shard_path = _validate_frozen_file(
            split_spec["operational_shard"],
            label=f"{processed_split} operational shard",
        )
        shard = _read_json(shard_path)
        required = {
            "schema_id": "tsunami-surrogate.common-time-v2.operational-shard.v1",
            "split": publication_split,
            "contract_hash": expected_contract_hash,
            "code_state_hash": expected_code_state,
            "complete": True,
            "solver_names": expected_solvers,
        }
        for key, expected in required.items():
            if shard.get(key) != expected:
                raise PreflightError(
                    f"Operational-shard {key} mismatch in {_relative(shard_path)}"
                )
        publication_count = len(shard.get("publications", []))
        expected_publications = int(split_spec["operational_shard"]["publications"])
        if publication_count != expected_publications:
            raise PreflightError(
                f"Operational-shard publication count mismatch in "
                f"{_relative(shard_path)}: "
                f"{publication_count} != {expected_publications}"
            )
        publication_index = {
            str(item["qualified_id"]): str(item["publication_hash"])
            for item in shard.get("publications", [])
            if isinstance(item, Mapping)
        }
        if len(publication_index) != publication_count:
            raise PreflightError(
                f"Operational-shard publication identities are duplicated or "
                f"malformed: {_relative(shard_path)}"
            )
        publication_indexes[processed_split] = publication_index
        summaries.append(
            {
                "split": processed_split,
                "publication_split": publication_split,
                "operational_shard": _relative(shard_path),
                "publications": publication_count,
            }
        )
    return summaries, publication_indexes


def _validate_required_file(path_value: str | Path) -> Path:
    path = _repo_path(path_value)
    if not path.is_file():
        raise PreflightError(f"Missing required file: {_relative(path)}")
    return path


def _resolved_config_dataset_paths(config_path: str | Path) -> list[str]:
    path = _validate_required_file(config_path)
    cfg = load_config(path)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    if not isinstance(eval_cfg, Mapping):
        eval_cfg = {}
    paths: list[str] = []
    real_resolution = eval_cfg.get("real_resolution")
    if isinstance(real_resolution, Mapping):
        for row in real_resolution.get("suites", []):
            if isinstance(row, Mapping) and row.get("path"):
                paths.append(str(row["path"]))
    for row in eval_cfg.get("window_suites", []):
        if isinstance(row, Mapping) and row.get("path"):
            paths.append(str(row["path"]))
    if not paths and eval_cfg.get("dataset_path"):
        paths.append(str(eval_cfg["dataset_path"]))
    data_cfg = cfg.get("data")
    if not paths and isinstance(data_cfg, Mapping) and data_cfg.get("test_path"):
        paths.append(str(data_cfg["test_path"]))
    dataset_cfg = cfg.get("dataset")
    if not paths and isinstance(dataset_cfg, Mapping) and dataset_cfg.get("path"):
        paths.append(str(dataset_cfg["path"]))
    if not paths:
        raise PreflightError(
            f"Evaluation config has no resolvable test dataset: {_relative(path)}"
        )
    return [_relative(_canonical_dataset_path(value)) for value in paths]


def _expected_config_datasets(
    contract: Mapping[str, Any],
) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}

    def bind(config: str, paths: Iterable[str | Path]) -> None:
        normalized = sorted(
            {_relative(_canonical_dataset_path(path)) for path in paths}
        )
        previous = expected.get(config)
        if previous is not None and previous != normalized:
            raise PreflightError(
                f"Evaluation config has conflicting dataset declarations: {config}"
            )
        expected[config] = normalized

    main = contract["main_datasets"]
    main_test = {
        reference: Path(str(main[reference]["processed_root"])) / "test"
        for reference in ("hydrostatic", "muscl_hr", "boussinesq")
    }
    for group in ("direct_models", "window_models"):
        for row in contract.get(group, []):
            bind(str(row["config"]), [main_test[str(row["reference"])]])
    for row in contract.get("sample_scaling", []):
        bind(str(row["config"]), [main_test["hydrostatic"]])
    for row in contract.get("native_muscl", []):
        bind(
            str(row["config"]),
            [Path(str(row["processed_root"])) / "test"],
        )
    for row in contract.get("strict_holdouts", []):
        root = Path(str(row["manifest"])).parent
        bind(str(row["config_id"]), [root / "test_id"])
        bind(str(row["config_heldout"]), [root / "test_heldout"])
        bind(str(row["config_full"]), [root / "test_heldout"])
    real = contract.get("real_bathymetry", {})
    real_paths = [
        Path(str(real["processed_root"])) / str(suite) / "hydrostatic" / "test"
        for suite in real.get("suites", {})
    ]
    for group in ("direct", "window"):
        for row in real.get(group, []):
            bind(str(row["config"]), real_paths)
    for row in contract.get("ensemble", {}).get("configs", []):
        bind(str(row["config"]), [main_test["hydrostatic"]])
    paper = contract.get("paper_evidence", {})
    proxy = paper.get("proxy_resolution")
    if isinstance(proxy, Mapping):
        bind(str(proxy["config"]), [str(proxy["dataset"])])
    return expected


def _validate_evaluation_config_bindings(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for config, expected_paths in sorted(_expected_config_datasets(contract).items()):
        path = _validate_required_file(config)
        observed_paths = sorted(_resolved_config_dataset_paths(path))
        if observed_paths != expected_paths:
            raise PreflightError(
                f"Evaluation dataset mismatch for {config}: "
                f"{observed_paths} != {expected_paths}"
            )
        summaries.append(
            {
                "config": _relative(path),
                "config_sha256": sha256_file(path),
                "dataset_paths": observed_paths,
            }
        )
    return summaries


def _checkpoint_specs(contract: Mapping[str, Any]) -> list[tuple[str, str, int | None]]:
    specs: list[tuple[str, str, int | None]] = []
    for group in ("direct_models", "window_models", "sample_scaling", "native_muscl"):
        for row in contract.get(group, []):
            _validate_required_file(str(row["config"]))
            expected_seed = 42 if group == "sample_scaling" else 18
            specs.append(
                (f"{group}:{row['id']}", str(row["checkpoint"]), expected_seed)
            )
    strict = contract.get("strict_holdouts", [])
    for row in strict:
        for key in ("config_id", "config_heldout", "config_full", "manifest"):
            _validate_required_file(str(row[key]))
        specs.append((f"strict_holdout:{row['id']}", str(row["checkpoint"]), 18))
    full_checkpoint = contract.get("strict_holdout_full_checkpoint")
    if full_checkpoint:
        specs.append(("strict_holdout:full_model", str(full_checkpoint), 18))
    return specs


def _roster_sha256(scenario_ids: Iterable[str]) -> str:
    payload = json.dumps(
        sorted(str(value) for value in scenario_ids),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sample_scaling_rosters(
    specs: Iterable[Mapping[str, Any]],
    train_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered_ids = list(train_rows)
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        count = int(spec["train_samples"])
        rng = np.random.default_rng(42)
        selected = [
            ordered_ids[int(i)] for i in rng.permutation(len(ordered_ids))[:count]
        ]
        observed = _roster_sha256(selected)
        expected = str(spec["roster_sha256"])
        if observed != expected:
            raise PreflightError(
                f"Sample-scaling roster mismatch for {spec['id']}: "
                f"{observed} != {expected}"
            )
        summaries.append(
            {
                "id": str(spec["id"]),
                "train_samples": count,
                "roster_sha256": observed,
            }
        )
    return summaries


def _validate_strict_holdout_ancestry(
    specs: Iterable[Mapping[str, Any]],
    hydrostatic_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        manifest_path = _repo_path(str(spec["manifest"]))
        manifest = _read_json(manifest_path)
        if manifest.get("source_root") != "data/processed/hydrostatic":
            raise PreflightError(
                f"Strict-holdout source root mismatch: {_relative(manifest_path)}"
            )
        sanity = manifest.get("sanity_checks")
        if not isinstance(sanity, Mapping) or not bool(sanity.get("passed", False)):
            raise PreflightError(
                f"Strict-holdout sanity checks did not pass: {_relative(manifest_path)}"
            )
        holdout_key = str(manifest.get("holdout_key"))
        holdout_value = str(manifest.get("holdout_value"))
        root = manifest_path.parent
        split_rows = {
            str(row.get("split")): row
            for row in manifest.get("splits", [])
            if isinstance(row, Mapping)
        }
        checked = 0
        for split_name in ("train", "val", "test_id", "test_heldout"):
            manifest_row = split_rows.get(split_name)
            if not isinstance(manifest_row, Mapping):
                raise PreflightError(
                    f"Strict-holdout manifest is missing split {split_name}: "
                    f"{_relative(manifest_path)}"
                )
            source_split = "test" if split_name.startswith("test_") else split_name
            source_rows = hydrostatic_index[source_split]
            seen: set[str] = set()
            for row in _iter_jsonl(root / split_name / "meta.jsonl"):
                scenario_id = str(row.get("scenario_id", ""))
                if not scenario_id or scenario_id in seen:
                    raise PreflightError(
                        f"Duplicate strict-holdout identity {scenario_id!r}"
                    )
                seen.add(scenario_id)
                source = source_rows.get(scenario_id)
                if source is None:
                    raise PreflightError(
                        f"Strict-holdout sample has no v2 ancestor: "
                        f"{split_name}:{scenario_id}"
                    )
                for key in ("source_type", "bathymetry_type", "solver_name"):
                    if row.get(key) != source.get(key):
                        raise PreflightError(
                            f"Strict-holdout ancestry mismatch for "
                            f"{split_name}:{scenario_id}:{key}"
                        )
                is_heldout = str(source.get(holdout_key)) == holdout_value
                expected_heldout = split_name == "test_heldout"
                if is_heldout != expected_heldout:
                    raise PreflightError(
                        f"Strict-holdout family predicate mismatch for "
                        f"{split_name}:{scenario_id}"
                    )
                checked += 1
            expected_count = int(manifest_row.get("num_samples", -1))
            if len(seen) != expected_count:
                raise PreflightError(
                    f"Strict-holdout count mismatch for {split_name}: "
                    f"{len(seen)} != {expected_count}"
                )
        summaries.append(
            {
                "id": str(spec["id"]),
                "manifest": _relative(manifest_path),
                "ancestry_rows_verified": checked,
            }
        )
    return summaries


def _validate_ensemble(
    contract: Mapping[str, Any],
    *,
    include_ensemble: bool,
) -> dict[str, Any]:
    ensemble = contract.get("ensemble", {})
    seeds = [int(value) for value in ensemble.get("required_members", [])]
    template = str(ensemble.get("checkpoint_template", ""))
    members: list[dict[str, Any]] = []
    failures: list[str] = []
    for seed in seeds:
        path = template.format(seed=seed)
        try:
            summary = validate_completed_checkpoint(path, expected_seed=seed)
            members.append({"seed": seed, "status": "complete", **summary})
        except PreflightError as exc:
            members.append({"seed": seed, "status": "incomplete_or_missing"})
            failures.append(f"seed {seed}: {exc}")
    status = "ready" if not failures else "deferred"
    if include_ensemble and failures:
        raise PreflightError(
            "Ensemble was requested but is not complete:\n  - "
            + "\n  - ".join(failures)
        )
    return {
        "requested": bool(include_ensemble),
        "status": status,
        "required_members": seeds,
        "members": members,
    }


def _matches_slice(row: Mapping[str, Any], key: str, value: Any) -> bool:
    item_key = (
        key.removesuffix("_in")
        .removesuffix("_not")
        .removesuffix("_min")
        .removesuffix("_max")
    )
    observed = row.get(item_key)
    if key.endswith("_in"):
        return str(observed) in {
            str(item) for item in (value if isinstance(value, list) else [value])
        }
    if key.endswith("_not_in"):
        return str(observed) not in {
            str(item) for item in (value if isinstance(value, list) else [value])
        }
    if key.endswith("_min"):
        return float(observed) >= float(value)
    if key.endswith("_max"):
        return float(observed) <= float(value)
    raise PreflightError(f"Unsupported paper-evidence slice filter: {key}")


def _validate_paper_evidence(
    contract: Mapping[str, Any],
    *,
    main_indices: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    publication_shape: tuple[int, int],
    requested: bool,
) -> dict[str, Any]:
    paper = contract.get("paper_evidence")
    if not isinstance(paper, Mapping):
        raise PreflightError("Evaluation contract is missing paper_evidence")

    bootstrap = paper.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise PreflightError("Paper-evidence bootstrap contract is missing")
    bootstrap_summary = {
        "seed": int(bootstrap.get("seed", -1)),
        "resamples": int(bootstrap.get("resamples", -1)),
        "confidence_level": float(bootstrap.get("confidence_level", -1.0)),
    }
    if bootstrap_summary["seed"] < 0:
        raise PreflightError("Paper-evidence bootstrap seed must be nonnegative")
    if bootstrap_summary["resamples"] < 1000:
        raise PreflightError(
            "Paper-evidence bootstrap requires at least 1000 resamples"
        )
    if bootstrap_summary["confidence_level"] != 0.95:
        raise PreflightError(
            "Paper-evidence confidence level must remain frozen at 0.95"
        )

    slices = paper.get("metadata_slices")
    if not isinstance(slices, list) or not slices:
        raise PreflightError("Paper-evidence metadata slices are missing")
    slice_ids = [str(row.get("id", "")) for row in slices if isinstance(row, Mapping)]
    if len(slice_ids) != len(slices) or len(slice_ids) != len(set(slice_ids)):
        raise PreflightError("Paper-evidence slice IDs are missing or duplicated")

    slice_counts: dict[str, dict[str, int]] = {}
    metadata_keys = ("source_type", "bathymetry_type", "source_strength")
    baseline_rows = main_indices["hydrostatic"]["test"]
    for reference in ("muscl_hr", "boussinesq"):
        comparison_rows = main_indices[reference]["test"]
        for scenario_id, baseline in baseline_rows.items():
            comparison = comparison_rows[scenario_id]
            for key in metadata_keys:
                if comparison.get(key) != baseline.get(key):
                    raise PreflightError(
                        "Cross-reference slice metadata mismatch for "
                        f"{scenario_id}:{key}"
                    )
    for reference in ("hydrostatic", "muscl_hr", "boussinesq"):
        rows = main_indices[reference]["test"]
        counts: dict[str, int] = {}
        for spec in slices:
            if not isinstance(spec, Mapping):
                raise PreflightError("Invalid paper-evidence slice declaration")
            label = str(spec["id"])
            key = str(spec["filter"])
            count = sum(
                1
                for row in rows.values()
                if _matches_slice(row, key, spec.get("value"))
            )
            if count <= 0:
                raise PreflightError(
                    f"Paper-evidence slice {label!r} is empty for {reference}"
                )
            counts[label] = count
        slice_counts[reference] = counts
    if len({tuple(sorted(counts.items())) for counts in slice_counts.values()}) != 1:
        raise PreflightError("Paper-evidence slice counts differ across references")

    group_counts = {
        group_key: {
            label: sum(
                1
                for row in baseline_rows.values()
                if str(row.get(group_key, "unknown")) == label
            )
            for label in sorted(
                {str(row.get(group_key, "unknown")) for row in baseline_rows.values()}
            )
        }
        for group_key in ("source_type", "bathymetry_type")
    }

    main = contract["main_datasets"]
    main_test_paths = {
        reference: _relative(
            _canonical_dataset_path(
                Path(str(main[reference]["processed_root"])) / "test"
            )
        )
        for reference in ("hydrostatic", "muscl_hr", "boussinesq")
    }
    direct_by_reference = {
        str(row["reference"]): row
        for row in contract.get("direct_models", [])
        if str(row.get("id", "")) in {"fno", "fno_muscl_hr", "fno_boussinesq"}
    }

    def validate_models(
        rows: Any,
        *,
        label: str,
        expected_references: set[str] | None = None,
    ) -> list[dict[str, str]]:
        if not isinstance(rows, list) or not rows:
            raise PreflightError(f"Paper-evidence {label} declarations are missing")
        summaries: list[dict[str, str]] = []
        observed_references: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise PreflightError(f"Invalid paper-evidence {label} row")
            reference = str(row.get("reference", "hydrostatic"))
            observed_references.add(reference)
            config = _relative(_validate_required_file(str(row["config"])))
            checkpoint = _relative(_validate_required_file(str(row["checkpoint"])))
            dataset = _relative(_canonical_dataset_path(str(row["dataset"])))
            if dataset != main_test_paths[reference]:
                raise PreflightError(
                    f"Paper-evidence {label} dataset mismatch for {reference}: "
                    f"{dataset} != {main_test_paths[reference]}"
                )
            summaries.append(
                {
                    "id": str(row.get("id", reference)),
                    "reference": reference,
                    "config": config,
                    "checkpoint": checkpoint,
                    "dataset": dataset,
                }
            )
        if (
            expected_references is not None
            and observed_references != expected_references
        ):
            raise PreflightError(
                f"Paper-evidence {label} references are "
                f"{sorted(observed_references)}, expected "
                f"{sorted(expected_references)}"
            )
        return summaries

    direct_models = validate_models(
        paper.get("direct_slice_models"),
        label="direct-slice",
        expected_references={"hydrostatic", "muscl_hr", "boussinesq"},
    )
    for summary in direct_models:
        core = direct_by_reference.get(summary["reference"])
        if not isinstance(core, Mapping) or (
            str(core["config"]) != summary["config"]
            or str(core["checkpoint"]) != summary["checkpoint"]
        ):
            raise PreflightError(
                "Paper-evidence direct model is not the frozen core FNO for "
                f"{summary['reference']}"
            )

    window_models = validate_models(
        paper.get("window_slice_models"),
        label="window-slice",
        expected_references={"hydrostatic"},
    )
    reference_models = validate_models(
        paper.get("reference_analysis", {}).get("models"),
        label="reference-analysis",
        expected_references={"hydrostatic", "muscl_hr", "boussinesq"},
    )
    wave_models = validate_models(
        paper.get("wave_metrics", {}).get("models"),
        label="wave-metrics",
        expected_references={"hydrostatic", "muscl_hr", "boussinesq"},
    )

    proxy = paper.get("proxy_resolution")
    if not isinstance(proxy, Mapping):
        raise PreflightError("Paper-evidence proxy-resolution contract is missing")
    proxy_grids = [int(value) for value in proxy.get("grids", [])]
    if proxy_grids != [32, 64, 128]:
        raise PreflightError(
            "Proxy-resolution grids must remain frozen at [32, 64, 128]"
        )
    proxy_summary = {
        "config": _relative(_validate_required_file(str(proxy["config"]))),
        "checkpoint": _relative(_validate_required_file(str(proxy["checkpoint"]))),
        "dataset": _relative(_canonical_dataset_path(str(proxy["dataset"]))),
        "grids": proxy_grids,
    }
    if proxy_summary["dataset"] != main_test_paths["hydrostatic"]:
        raise PreflightError("Proxy-resolution dataset is not hydrostatic v2 test")

    native = paper.get("native_transfer")
    if not isinstance(native, Mapping):
        raise PreflightError("Paper-evidence native-transfer contract is missing")
    native_grids = [int(value) for value in native.get("grids", [])]
    native_configs = [str(value) for value in native.get("configs", [])]
    native_checkpoints = [str(value) for value in native.get("checkpoints", [])]
    native_datasets = [
        _relative(_canonical_dataset_path(str(value)))
        for value in native.get("datasets", [])
    ]
    expected_native = list(contract.get("native_muscl", []))
    if (
        native_grids != [int(row["grid"]) for row in expected_native]
        or native_configs != [str(row["config"]) for row in expected_native]
        or native_checkpoints != [str(row["checkpoint"]) for row in expected_native]
        or native_datasets
        != [
            _relative(
                _canonical_dataset_path(Path(str(row["processed_root"])) / "test")
            )
            for row in expected_native
        ]
    ):
        raise PreflightError(
            "Paper-evidence native transfer does not match the frozen MUSCL roster"
        )

    wave = paper.get("wave_metrics")
    gauges = (
        [
            [int(value) for value in gauge]
            for gauge in wave.get("gauges", [])
            if isinstance(gauge, list)
        ]
        if isinstance(wave, Mapping)
        else []
    )
    expected_gauges = [[row, col] for row in (16, 32, 48) for col in (16, 32, 48)]
    if gauges != expected_gauges:
        raise PreflightError("Virtual gauges must remain the frozen 3x3 interior grid")
    if any(
        row < 0 or col < 0 or row >= publication_shape[0] or col >= publication_shape[1]
        for row, col in gauges
    ):
        raise PreflightError("A frozen virtual gauge lies outside the publication grid")
    if (
        float(wave.get("arrival_threshold_fraction", -1.0)) != 0.10
        or float(wave.get("peak_plateau_fraction", -1.0)) != 0.99
    ):
        raise PreflightError("Wave-metric thresholds differ from the frozen contract")

    ensemble = paper.get("ensemble")
    if not isinstance(ensemble, Mapping):
        raise PreflightError("Paper-evidence ensemble contract is missing")
    ensemble_summary = {
        "config": _relative(_validate_required_file(str(ensemble["config"]))),
        "val_dataset": _relative(_canonical_dataset_path(str(ensemble["val_dataset"]))),
        "test_dataset": _relative(
            _canonical_dataset_path(str(ensemble["test_dataset"]))
        ),
    }
    expected_val = _relative(
        _canonical_dataset_path(
            Path(str(main["hydrostatic"]["processed_root"])) / "val"
        )
    )
    if (
        ensemble_summary["val_dataset"] != expected_val
        or ensemble_summary["test_dataset"] != main_test_paths["hydrostatic"]
    ):
        raise PreflightError(
            "Paper-evidence ensemble datasets are not the frozen hydrostatic v2 splits"
        )

    input_stats: dict[str, Any] = {}
    input_stats_hashes: dict[str, str] = {}
    for reference in ("hydrostatic", "muscl_hr", "boussinesq"):
        stats_path = _repo_path(
            Path(str(main[reference]["processed_root"])) / "normalization_stats.json"
        )
        stats = _read_json(stats_path)
        input_stats[reference] = stats.get("inputs")
        input_stats_hashes[reference] = sha256_file(stats_path)
    if any(
        input_stats[reference] != input_stats["hydrostatic"]
        for reference in ("muscl_hr", "boussinesq")
    ):
        raise PreflightError(
            "Cross-reference paper analysis requires identical input normalization"
        )

    required_exclusions = {
        "hydrostatic_native_resolution_legacy_paths",
        "shared_from_64_resolution_studies",
        "destructive_ood_rebuilds",
        "legacy_stage_c_reconstruction",
        "unsupported_multi_seed_architecture_matrix",
        "stale_paper_figure_copying",
    }
    exclusions = {str(value) for value in contract.get("excluded", [])}
    missing_exclusions = sorted(required_exclusions - exclusions)
    if missing_exclusions:
        raise PreflightError(
            "Paper-evidence exclusions are incomplete: " + ", ".join(missing_exclusions)
        )

    return {
        "requested": bool(requested),
        "status": "ready" if requested else "available",
        "bootstrap": bootstrap_summary,
        "slice_ids": slice_ids,
        "slice_counts": slice_counts,
        "group_counts": group_counts,
        "direct_models": direct_models,
        "window_models": window_models,
        "reference_models": reference_models,
        "wave_models": wave_models,
        "proxy_resolution": proxy_summary,
        "native_transfer": {
            "grids": native_grids,
            "configs": native_configs,
            "checkpoints": native_checkpoints,
            "datasets": native_datasets,
        },
        "gauges": gauges,
        "arrival_threshold_fraction": float(wave["arrival_threshold_fraction"]),
        "peak_plateau_fraction": float(wave["peak_plateau_fraction"]),
        "ensemble": ensemble_summary,
        "input_normalization_sha256s": input_stats_hashes,
        "excluded": sorted(exclusions),
        "limitations": [
            "Metadata-selected slices are difficult subgroups of the current v2 test roster, not independently generated unseen-family datasets.",
            "The seven ensemble members measure hydrostatic FNO seed stability and uncertainty only; they do not replace a replicated multi-architecture matrix.",
            "Proxy resolution resizes processed fields; native MUSCL-HR transfer is reported separately.",
        ],
    }


def run_preflight(
    contract: Mapping[str, Any],
    *,
    output_root: str | Path | None,
    include_ensemble: bool,
    require_real_bathymetry: bool,
    include_paper_evidence: bool = False,
    deep_payload_audit: bool = False,
    payload_audit_workers: int = 8,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if payload_audit_workers <= 0:
        raise PreflightError("Payload-audit worker count must be positive")
    scientific = contract["scientific_scope"]
    contract_hash = str(scientific["contract_hash"])
    sample_schema_id = str(scientific["sample_schema_id"])
    times = _expected_times(contract)
    domain = scientific["computational_domain"]
    solver_shape = tuple(int(v) for v in domain["solver_shape"])
    publication_shape = tuple(int(v) for v in domain["publication_shape"])

    if output_root is not None:
        run_root = _repo_path(output_root).resolve()
        evaluation_root = (ROOT / "evaluation_runs").resolve()
        if run_root == evaluation_root or evaluation_root not in run_root.parents:
            raise PreflightError(
                "Evaluation output root must be a new child of evaluation_runs/"
            )
        if run_root.exists():
            raise PreflightError(
                f"Evaluation output root already exists: {_relative(run_root)}"
            )

    generation, publication_indexes = _validate_generation_artifacts(contract)
    main_cfg = contract["main_datasets"]
    split_cfg = main_cfg["splits"]
    main_indices: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    dataset_summaries: list[dict[str, Any]] = []
    for reference in ("hydrostatic", "muscl_hr", "boussinesq"):
        reference_cfg = main_cfg[reference]
        solver_name = str(reference_cfg["solver_name"])
        processed_root = _repo_path(str(reference_cfg["processed_root"]))
        main_indices[reference] = {}
        for split_name in ("train", "val", "test"):
            spec = split_cfg[split_name]
            rows = _validate_processed_split(
                processed_root / split_name,
                expected_count=int(spec["count"]),
                expected_solver=solver_name,
                expected_publication_split=str(spec["publication_split"]),
                expected_contract_hash=contract_hash,
                expected_schema_id=sample_schema_id,
                expected_times=times,
                publication_shape=publication_shape,
                solver_shape=solver_shape,
                expected_publication_hashes=publication_indexes[split_name],
                deep_payload_audit=deep_payload_audit,
                payload_audit_workers=payload_audit_workers,
                progress_callback=progress_callback,
            )
            main_indices[reference][split_name] = rows
            dataset_summaries.append(
                {
                    "reference": reference,
                    "split": split_name,
                    "processed_root": _relative(processed_root / split_name),
                    "samples_verified": len(rows),
                }
            )
    for split_name in ("train", "val", "test"):
        expected_roster = set(main_indices["hydrostatic"][split_name])
        for reference in ("muscl_hr", "boussinesq"):
            observed_roster = set(main_indices[reference][split_name])
            if observed_roster != expected_roster:
                raise PreflightError(
                    f"Cross-reference scenario roster mismatch for "
                    f"{reference}:{split_name}"
                )

    checkpoint_summaries: list[dict[str, Any]] = []
    for label, path, seed in _checkpoint_specs(contract):
        checkpoint_summaries.append(
            {"id": label, **validate_completed_checkpoint(path, expected_seed=seed)}
        )

    sample_scaling_summaries = _validate_sample_scaling_rosters(
        contract.get("sample_scaling", []),
        main_indices["hydrostatic"]["train"],
    )

    native_summaries: list[dict[str, Any]] = []
    native_rosters: dict[str, dict[str, set[str]]] = {}
    for spec in contract.get("native_muscl", []):
        grid = int(spec["grid"])
        root = _repo_path(str(spec["processed_root"]))
        native_rosters[str(spec["id"])] = {}
        for split_name, expected_count in spec["counts"].items():
            rows = _validate_processed_split(
                root / split_name,
                expected_count=int(expected_count),
                expected_solver="swe_muscl_hr",
                expected_publication_split="train",
                expected_contract_hash=contract_hash,
                expected_schema_id=sample_schema_id,
                expected_times=times,
                publication_shape=(grid, grid),
                solver_shape=None,
                lineage_schema_id="tsunami-surrogate.native-resolution-inputs.v1",
                deep_payload_audit=deep_payload_audit,
                payload_audit_workers=payload_audit_workers,
                progress_callback=progress_callback,
            )
            roster = set(rows)
            native_rosters[str(spec["id"])][split_name] = roster
            expected_roster_hash = str(contract["native_muscl_rosters"][split_name])
            observed_roster_hash = _roster_sha256(roster)
            if observed_roster_hash != expected_roster_hash:
                raise PreflightError(
                    f"Native MUSCL roster mismatch for {spec['id']}:{split_name}"
                )
            native_summaries.append(
                {
                    "id": str(spec["id"]),
                    "split": split_name,
                    "samples_verified": len(rows),
                    "roster_sha256": observed_roster_hash,
                }
            )
        split_sets = native_rosters[str(spec["id"])]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if split_sets[left].intersection(split_sets[right]):
                raise PreflightError(
                    f"Native MUSCL split leakage for {spec['id']}: {left}/{right}"
                )
    if native_rosters:
        baseline = next(iter(native_rosters.values()))
        for native_id, split_sets in native_rosters.items():
            if any(
                split_sets[name] != baseline[name] for name in ("train", "val", "test")
            ):
                raise PreflightError(
                    f"Native MUSCL rosters differ across resolutions: {native_id}"
                )

    holdout_summaries = _validate_strict_holdout_ancestry(
        contract.get("strict_holdouts", []),
        main_indices["hydrostatic"],
    )

    real_summaries: list[dict[str, Any]] = []
    real_cfg = contract.get("real_bathymetry", {})
    real_root = _repo_path(str(real_cfg.get("processed_root", "")))
    if require_real_bathymetry:
        if not real_root.exists():
            raise PreflightError(
                f"Missing v2 real-bathymetry processed root: {_relative(real_root)}"
            )
        for suite, expected_count in real_cfg.get("suites", {}).items():
            split_dir = real_root / str(suite) / "hydrostatic" / "test"
            rows = _validate_processed_split(
                split_dir,
                expected_count=int(expected_count),
                expected_solver="swe_hydrostatic",
                expected_publication_split="test",
                expected_contract_hash=contract_hash,
                expected_schema_id=sample_schema_id,
                expected_times=times,
                publication_shape=publication_shape,
                solver_shape=solver_shape,
                lineage_schema_id=str(real_cfg["lineage_schema_id"]),
                # The auxiliary is only 13 samples and has no separately frozen
                # operational-shard inventory, so always validate its payloads.
                deep_payload_audit=True,
                payload_audit_workers=payload_audit_workers,
                progress_callback=progress_callback,
            )
            real_summaries.append({"suite": str(suite), "samples_verified": len(rows)})

    numerical = _validate_numerical_artifacts(
        contract.get("accepted_numerical_artifacts", [])
    )
    evaluation_configs = _validate_evaluation_config_bindings(contract)
    paper_evidence = _validate_paper_evidence(
        contract,
        main_indices=main_indices,
        publication_shape=publication_shape,
        requested=include_paper_evidence,
    )
    ensemble = _validate_ensemble(
        contract,
        include_ensemble=include_ensemble or include_paper_evidence,
    )
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "suite_id": contract["suite_id"],
        "status": "passed",
        "output_root": None if output_root is None else str(output_root),
        "scientific_scope": {
            "contract_hash": contract_hash,
            "sample_schema_id": sample_schema_id,
            "requested_times": times.tolist(),
            "solver_shape": list(solver_shape),
            "publication_shape": list(publication_shape),
        },
        "datasets": dataset_summaries,
        "sample_scaling": sample_scaling_summaries,
        "native_muscl": native_summaries,
        "strict_holdouts": holdout_summaries,
        "real_bathymetry": real_summaries,
        "evaluation_configs": evaluation_configs,
        "checkpoints": checkpoint_summaries,
        "code_state": _evaluation_code_state(),
        "accepted_numerical_artifacts": numerical,
        "frozen_generation_artifacts": generation,
        "deep_payload_audit": bool(deep_payload_audit),
        "payload_audit_workers": int(payload_audit_workers),
        "ensemble": ensemble,
        "paper_evidence": paper_evidence,
        "limitations": [
            "Main and strict-holdout checkpoint provenance is manifest-bound rather than shard-content-bound because those processed containers predate processed-dataset.v2.",
            "Legacy Stage-C reconstruction, stale Hydrostatic native-resolution paths, shared-from-64 studies, destructive OOD rebuilding, and the unsupported multi-seed architecture matrix are excluded.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/eval/final_v2_suite.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--include-ensemble", action="store_true")
    parser.add_argument("--include-paper-evidence", action="store_true")
    parser.add_argument(
        "--deep-payload-audit",
        action="store_true",
        help=(
            "Re-hash and reopen every raw sample payload. The normal preflight "
            "validates frozen publication records and file sizes without the "
            "30+ GiB payload pass."
        ),
    )
    parser.add_argument(
        "--payload-audit-workers",
        type=int,
        default=8,
        help="Worker threads used to hash and reopen independent raw payloads.",
    )
    parser.add_argument(
        "--allow-missing-real-bathymetry",
        action="store_true",
        help="Permit preflight before the v2-compatible auxiliary has been built.",
    )
    args = parser.parse_args()

    contract = load_suite_contract(args.contract)
    report = run_preflight(
        contract,
        output_root=args.output_root,
        include_ensemble=bool(args.include_ensemble),
        include_paper_evidence=bool(args.include_paper_evidence),
        require_real_bathymetry=not bool(args.allow_missing_real_bathymetry),
        deep_payload_audit=bool(args.deep_payload_audit),
        payload_audit_workers=int(args.payload_audit_workers),
        progress_callback=lambda message: print(message, flush=True),
    )
    if args.report:
        _write_json_atomic(_repo_path(args.report), report)
    print(
        f"[eval-preflight] status=passed datasets={len(report['datasets'])} "
        f"checkpoints={len(report['checkpoints'])} "
        f"ensemble={report['ensemble']['status']} "
        f"paper_evidence={report['paper_evidence']['status']}"
    )


if __name__ == "__main__":
    main()
