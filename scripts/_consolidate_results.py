#!/usr/bin/env python
"""Validate and consolidate one isolated evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ConsolidationError(RuntimeError):
    """Raised when a required result is absent, stale, or malformed."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConsolidationError(f"Missing required result: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConsolidationError(f"Malformed result JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ConsolidationError(f"Expected result object: {path}")
    return payload


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ConsolidationError(f"Non-finite value at {label}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, label=f"{label}[{index}]")


def _matches_path(observed: Any, expected: str) -> bool:
    if observed is None:
        return False
    observed_path = Path(str(observed))
    expected_path = Path(expected)
    if observed_path == expected_path:
        return True
    try:
        return observed_path.resolve() == (ROOT / expected_path).resolve()
    except OSError:
        return False


def _matches_run_path(
    observed: Any,
    expected: str,
    *,
    run_root: Path,
) -> bool:
    if observed is None:
        return False
    observed_path = Path(str(observed))
    expected_path = Path(expected)
    candidates = [
        expected_path,
        run_root / expected_path,
        ROOT / expected_path,
    ]
    if any(observed_path == candidate for candidate in candidates):
        return True
    try:
        observed_resolved = observed_path.resolve()
        return any(
            observed_resolved == candidate.resolve()
            for candidate in candidates
        )
    except OSError:
        return False


def _canonical_dataset_path(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    if (
        path.name == "eval_dataset.npz"
        and not path.is_file()
        and (path.parent / "shards_manifest.json").is_file()
    ):
        path = path.parent
    return path.resolve()


def _nested_value(payload: Any, dotted_key: str) -> Any:
    current = payload
    for key in str(dotted_key).split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(dotted_key)
        current = current[key]
    return current


def _require_nested(payload: Mapping[str, Any], key: str, *, label: str) -> Any:
    try:
        return _nested_value(payload, key)
    except KeyError as exc:
        raise ConsolidationError(
            f"Missing required key {key!r} for {label}"
        ) from exc


def _row_collection(
    payload: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Any]]:
    rows_key = str(cell.get("rows_key", "rows"))
    raw_rows = payload.get(rows_key)
    collection_type = str(cell.get("row_collection_type", "list"))
    if collection_type == "mapping":
        if not isinstance(raw_rows, Mapping):
            raise ConsolidationError(
                f"Expected mapping rows at {rows_key!r} for cell "
                f"{cell.get('id')}"
            )
        rows: list[Mapping[str, Any]] = []
        identities: list[Any] = []
        for key, value in raw_rows.items():
            if not isinstance(value, Mapping):
                raise ConsolidationError(
                    f"Invalid mapping row {key!r} for cell {cell.get('id')}"
                )
            rows.append(value)
            identities.append(str(key))
        return rows, identities
    if not isinstance(raw_rows, list):
        raise ConsolidationError(
            f"Expected list rows at {rows_key!r} for cell {cell.get('id')}"
        )
    rows = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ConsolidationError(
                f"Invalid result row for cell {cell.get('id')}"
            )
        rows.append(row)
    identity_keys = cell.get("row_identity_keys")
    if isinstance(identity_keys, list):
        identities = [
            [row.get(str(key)) for key in identity_keys]
            for row in rows
        ]
    else:
        identity_key = str(cell.get("row_identity_key", "label"))
        identities = [row.get(identity_key) for row in rows]
    return rows, identities


def _collect_dataset_paths(payload: Mapping[str, Any]) -> list[str]:
    observed: list[str] = []
    for key in ("dataset_path", "val_dataset", "test_dataset"):
        if payload.get(key):
            observed.append(str(payload[key]))
    dataset_paths = payload.get("dataset_paths")
    if isinstance(dataset_paths, Mapping):
        observed.extend(
            str(value)
            for value in dataset_paths.values()
            if value
        )
    elif isinstance(dataset_paths, list):
        observed.extend(str(value) for value in dataset_paths if value)
    for rows_key in ("rows", "pairs", "directions", "members"):
        rows = payload.get(rows_key)
        if not isinstance(rows, list):
            continue
        observed.extend(
            str(row["dataset_path"])
            for row in rows
            if isinstance(row, Mapping) and row.get("dataset_path")
        )
    return observed


def _validate_npz(path: Path, cell: Mapping[str, Any]) -> None:
    required_keys = [str(key) for key in cell.get("npz_required_keys", [])]
    if not required_keys:
        return
    allowed_nonfinite = {
        str(key) for key in cell.get("npz_allow_nonfinite_keys", [])
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in required_keys if key not in payload]
            if missing:
                raise ConsolidationError(
                    f"Missing NPZ keys for cell {cell.get('id')}: {missing}"
                )
            shapes: set[tuple[int, ...]] = set()
            for key in required_keys:
                array = np.asarray(payload[key])
                if array.size == 0:
                    raise ConsolidationError(
                        f"Empty NPZ array {key!r} for cell {cell.get('id')}"
                    )
                shapes.add(tuple(int(value) for value in array.shape))
                if (
                    key not in allowed_nonfinite
                    and np.issubdtype(array.dtype, np.number)
                    and not bool(np.isfinite(array).all())
                ):
                    raise ConsolidationError(
                        f"Non-finite NPZ array {key!r} for cell "
                        f"{cell.get('id')}"
                    )
            if len(shapes) != 1:
                raise ConsolidationError(
                    f"Arrival-map NPZ shapes differ for cell {cell.get('id')}: "
                    f"{sorted(shapes)}"
                )
    except (OSError, ValueError) as exc:
        raise ConsolidationError(f"Unreadable NPZ artifact: {path}") from exc


def _changed_evaluation_paths(expected_commit: str) -> list[str]:
    try:
        changed = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                expected_commit,
                "--",
                "configs",
                "scripts",
                "src",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        untracked = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "configs",
                "scripts",
                "src",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConsolidationError(
            "Could not identify evaluation-tree changes"
        ) from exc
    return sorted({path for path in (*changed, *untracked) if path})


def _validate_live_bindings(
    manifest: Mapping[str, Any],
    *,
    allow_consolidator_only_repair: bool = False,
) -> dict[str, Any] | None:
    code_state = manifest.get("code_state")
    if not isinstance(code_state, Mapping):
        raise ConsolidationError("Run manifest is missing evaluation code state")
    expected_commit = str(code_state.get("git_commit", ""))
    expected_tree = str(code_state.get("evaluation_tree_sha256", ""))
    try:
        current_commit = subprocess.check_output(
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
        raise ConsolidationError("Could not verify evaluation code state") from exc
    digest = hashlib.sha256()
    for relative in sorted(
        Path(raw.decode("utf-8"))
        for raw in listed.split(b"\0")
        if raw
    ):
        path = ROOT / relative
        if not path.is_file():
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    current_tree = digest.hexdigest()
    repair: dict[str, Any] | None = None
    if (
        not expected_commit
        or not expected_tree
        or current_commit != expected_commit
        or current_tree != expected_tree
    ):
        if not allow_consolidator_only_repair:
            if not expected_commit or current_commit != expected_commit:
                raise ConsolidationError(
                    "Git commit changed after evaluation preflight"
                )
            raise ConsolidationError(
                "Evaluation code/config tree changed after preflight"
            )
        changed_paths = _changed_evaluation_paths(expected_commit)
        allowed_path = "scripts/_consolidate_results.py"
        if changed_paths != [allowed_path]:
            raise ConsolidationError(
                "Consolidator-only repair rejected changed evaluation paths: "
                f"{changed_paths}"
            )
        repair = {
            "mode": "consolidator_only",
            "preflight_git_commit": expected_commit,
            "preflight_evaluation_tree_sha256": expected_tree,
            "repair_git_commit": current_commit,
            "repair_evaluation_tree_sha256": current_tree,
            "changed_paths": changed_paths,
            "consolidator_sha256": _sha256(ROOT / allowed_path),
        }

    for raw_cell in manifest.get("cells", []):
        if not isinstance(raw_cell, Mapping):
            continue
        config = raw_cell.get("config")
        config_hash = raw_cell.get("config_sha256")
        if config is not None:
            path = ROOT / str(config)
            if not path.is_file() or _sha256(path) != str(config_hash):
                raise ConsolidationError(
                    f"Evaluation config changed after preflight: {config}"
                )
        configs = raw_cell.get("configs")
        config_hashes = raw_cell.get("config_sha256s")
        if isinstance(configs, list):
            if not isinstance(config_hashes, list) or len(configs) != len(
                config_hashes
            ):
                raise ConsolidationError(
                    f"Config hash set is incomplete for {raw_cell.get('id')}"
                )
            for config_path, expected_hash in zip(configs, config_hashes):
                path = ROOT / str(config_path)
                if not path.is_file() or _sha256(path) != str(expected_hash):
                    raise ConsolidationError(
                        f"Evaluation config changed after preflight: "
                        f"{config_path}"
                    )
        checkpoint = raw_cell.get("checkpoint")
        checkpoint_hash = raw_cell.get("checkpoint_sha256")
        if checkpoint is not None:
            path = ROOT / str(checkpoint)
            if not path.is_file() or _sha256(path) != str(checkpoint_hash):
                raise ConsolidationError(
                    f"Checkpoint changed after preflight: {checkpoint}"
                )
        checkpoints = raw_cell.get("checkpoints")
        checkpoint_hashes = raw_cell.get("checkpoint_sha256s")
        if isinstance(checkpoints, list):
            if not isinstance(checkpoint_hashes, list) or len(
                checkpoints
            ) != len(checkpoint_hashes):
                raise ConsolidationError(
                    f"Checkpoint hash set is incomplete for {raw_cell.get('id')}"
                )
            for checkpoint_path, expected_hash in zip(
                checkpoints, checkpoint_hashes
            ):
                path = ROOT / str(checkpoint_path)
                if not path.is_file() or _sha256(path) != str(expected_hash):
                    raise ConsolidationError(
                        "Ensemble checkpoint changed after preflight: "
                        f"{checkpoint_path}"
                    )
    return repair


def _expected_value_matches(observed: Any, expected: Any) -> bool:
    if isinstance(observed, bool) or isinstance(expected, bool):
        return observed == expected
    if isinstance(expected, float) and isinstance(observed, (int, float)):
        return math.isclose(
            float(observed),
            float(expected),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    return observed == expected


def _validate_cell(
    *,
    run_root: Path,
    cell: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    relative_path = Path(str(cell.get("path", "")))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ConsolidationError(
            f"Unsafe result path for cell {cell.get('id')}: {relative_path}"
        )
    path = run_root / relative_path
    resolved_path = path.resolve()
    if resolved_path == run_root or run_root not in resolved_path.parents:
        raise ConsolidationError(
            f"Result path escapes run root for cell {cell.get('id')}: "
            f"{relative_path}"
        )
    if bool(cell.get("file_only", False)):
        if not path.is_file():
            raise ConsolidationError(
                f"Missing required companion artifact: {path}"
            )
        if path.stat().st_size <= 0:
            raise ConsolidationError(
                f"Required companion artifact is empty: {path}"
            )
        if path.suffix == ".npz":
            _validate_npz(path, cell)
        return {"path": str(relative_path), "file_only": True}, path
    payload = _read_object(path)
    _assert_finite(payload, label=str(cell.get("id")))

    expected_type = cell.get("evaluation_type")
    if expected_type is not None and payload.get("evaluation_type") != expected_type:
        raise ConsolidationError(
            f"Evaluation type mismatch for {cell.get('id')}: "
            f"{payload.get('evaluation_type')!r} != {expected_type!r}"
        )
    for key in cell.get("required_keys", []):
        _require_nested(payload, str(key), label=f"cell {cell.get('id')}")
    for key in cell.get("require_empty_keys", []):
        if payload.get(key) not in ([], {}, None):
            raise ConsolidationError(
                f"Expected empty key {key!r} for cell {cell.get('id')}"
            )
    expected_values = cell.get("expected_values")
    if isinstance(expected_values, Mapping):
        for key, expected in expected_values.items():
            try:
                observed = _nested_value(payload, str(key))
            except KeyError:
                observed = None
            if not _expected_value_matches(observed, expected):
                raise ConsolidationError(
                    f"Value mismatch for {cell.get('id')}:{key}: "
                    f"{observed!r} != {expected!r}"
                )
    expected_config = cell.get("config")
    if expected_config is not None and not _matches_path(
        payload.get("config_path"), str(expected_config)
    ):
        raise ConsolidationError(
            f"Config identity mismatch for cell {cell.get('id')}"
        )
    expected_checkpoint = cell.get("checkpoint")
    if expected_checkpoint is not None:
        observed_checkpoint = payload.get(
            "checkpoint_path", payload.get("checkpoint")
        )
        if not _matches_path(observed_checkpoint, str(expected_checkpoint)):
            raise ConsolidationError(
                f"Checkpoint identity mismatch for cell {cell.get('id')}"
            )
    expected_configs = cell.get("configs")
    if isinstance(expected_configs, list):
        observed_configs = payload.get("configs")
        if not isinstance(observed_configs, list) or len(
            observed_configs
        ) != len(expected_configs):
            raise ConsolidationError(
                f"Config-set size mismatch for cell {cell.get('id')}"
            )
        if any(
            not _matches_path(observed, str(expected))
            for observed, expected in zip(observed_configs, expected_configs)
        ):
            raise ConsolidationError(
                f"Config-set identity mismatch for cell {cell.get('id')}"
            )
    expected_num_samples = cell.get("num_samples")
    if expected_num_samples is not None:
        observed_num_samples = payload.get(
            "num_samples", payload.get("num_samples_seen")
        )
        if observed_num_samples is None or int(float(observed_num_samples)) != int(
            expected_num_samples
        ):
            raise ConsolidationError(
                f"Sample count mismatch for cell {cell.get('id')}: "
                f"{observed_num_samples!r} != {expected_num_samples}"
            )
    expected_dataset_paths = cell.get("expected_dataset_paths")
    if isinstance(expected_dataset_paths, list):
        observed_paths = _collect_dataset_paths(payload)
        observed = sorted(
            {
                str(_canonical_dataset_path(path))
                for path in observed_paths
            }
        )
        required = sorted(
            {
                str(_canonical_dataset_path(path))
                for path in expected_dataset_paths
            }
        )
        if observed != required:
            raise ConsolidationError(
                f"Dataset identity mismatch for cell {cell.get('id')}: "
                f"{observed} != {required}"
            )
    if bool(cell.get("require_physical_metrics", False)):
        for key in ("mae_physical", "rmse_physical", "rel_l2_physical"):
            if key not in payload:
                raise ConsolidationError(
                    f"Missing required physical metric {key!r} for "
                    f"cell {cell.get('id')}"
                )
    if bool(cell.get("require_physical_target_units", False)):
        if payload.get("target_units") != "physical":
            raise ConsolidationError(
                f"Physical diagnostics are required for cell {cell.get('id')}"
            )
    if bool(cell.get("require_normalization_bridge", False)):
        bridge = payload.get("normalization_bridge")
        if not isinstance(bridge, Mapping) or not bool(
            bridge.get("enabled", False)
        ):
            raise ConsolidationError(
                f"Normalization-bridge provenance is required for "
                f"cell {cell.get('id')}"
            )
    if bool(cell.get("seeded_window_rollout", False)):
        if not bool(payload.get("seeded_with_true_first_frame", False)):
            raise ConsolidationError(
                f"Window task provenance is missing for cell {cell.get('id')}"
            )
    companion_path_fields = cell.get("companion_path_fields")
    if isinstance(companion_path_fields, Mapping):
        for key, expected in companion_path_fields.items():
            observed = _require_nested(
                payload,
                str(key),
                label=f"cell {cell.get('id')}",
            )
            if not _matches_run_path(
                observed,
                str(expected),
                run_root=run_root,
            ):
                raise ConsolidationError(
                    f"Companion path mismatch for cell {cell.get('id')}: "
                    f"{key}={observed!r}"
                )
    companion_sha256_fields = cell.get("companion_sha256_fields")
    if isinstance(companion_sha256_fields, Mapping):
        for key, expected_path in companion_sha256_fields.items():
            observed = _require_nested(
                payload,
                str(key),
                label=f"cell {cell.get('id')}",
            )
            path = run_root / str(expected_path)
            if not path.is_file() or str(observed) != _sha256(path):
                raise ConsolidationError(
                    f"Companion checksum mismatch for cell {cell.get('id')}: "
                    f"{key}"
                )

    expected_rows = cell.get("row_count")
    rows: list[Mapping[str, Any]] | None = None
    row_identities: list[Any] | None = None
    if expected_rows is not None:
        rows, row_identities = _row_collection(payload, cell)
        if len(rows) != int(expected_rows):
            raise ConsolidationError(
                f"Row count mismatch for cell {cell.get('id')}: "
                f"{len(rows)} != {expected_rows}"
            )
    expected_row_identities = cell.get("row_identities")
    if isinstance(expected_row_identities, list):
        if rows is None or row_identities is None:
            rows, row_identities = _row_collection(payload, cell)
        observed_identities = [
            json.dumps(value, sort_keys=True)
            for value in row_identities
        ]
        required_identities = [
            json.dumps(value, sort_keys=True)
            for value in expected_row_identities
        ]
        if (
            len(observed_identities) != len(set(observed_identities))
            or set(observed_identities) != set(required_identities)
        ):
            raise ConsolidationError(
                f"Row identities mismatch for cell {cell.get('id')}: "
                f"{observed_identities} != {required_identities}"
            )
    expected_row_counts = cell.get("row_sample_counts")
    if isinstance(expected_row_counts, Mapping):
        if rows is None:
            rows, row_identities = _row_collection(payload, cell)
        observed = {
            str(row.get("label")): int(float(row.get("num_samples", -1)))
            for row in rows
        }
        required = {
            str(key): int(value) for key, value in expected_row_counts.items()
        }
        if observed != required:
            raise ConsolidationError(
                f"Per-suite sample counts mismatch for cell {cell.get('id')}: "
                f"{observed} != {required}"
            )
    row_required_keys = cell.get("row_required_keys", [])
    if row_required_keys:
        if rows is None:
            rows, row_identities = _row_collection(payload, cell)
        for index, row in enumerate(rows):
            for key in row_required_keys:
                _require_nested(
                    row,
                    str(key),
                    label=f"cell {cell.get('id')} row {index}",
                )
    row_bindings = cell.get("row_bindings")
    if isinstance(row_bindings, list):
        if rows is None or row_identities is None:
            rows, row_identities = _row_collection(payload, cell)
        row_by_identity = {
            json.dumps(identity, sort_keys=True): row
            for identity, row in zip(row_identities, rows)
        }
        for binding in row_bindings:
            if not isinstance(binding, Mapping):
                raise ConsolidationError(
                    f"Invalid row binding for cell {cell.get('id')}"
                )
            identity = json.dumps(binding.get("identity"), sort_keys=True)
            row = row_by_identity.get(identity)
            if row is None:
                raise ConsolidationError(
                    f"Missing bound row {identity} for cell {cell.get('id')}"
                )
            for key in ("config_path", "checkpoint_path", "dataset_path"):
                expected = binding.get(key)
                if expected is None:
                    continue
                if not _matches_path(row.get(key), str(expected)):
                    raise ConsolidationError(
                        f"Row binding mismatch for cell {cell.get('id')}: "
                        f"{identity}:{key}"
                    )
    row_num_samples = cell.get("row_num_samples")
    if row_num_samples is not None:
        if rows is None:
            rows, row_identities = _row_collection(payload, cell)
        for index, row in enumerate(rows):
            try:
                observed_count = _nested_value(row, "num_samples")
            except KeyError:
                try:
                    observed_count = _nested_value(row, "metrics.num_samples")
                except KeyError as exc:
                    raise ConsolidationError(
                        f"Missing row sample count for cell {cell.get('id')} "
                        f"row {index}"
                    ) from exc
            if int(float(observed_count)) != int(row_num_samples):
                raise ConsolidationError(
                    f"Row sample count mismatch for cell {cell.get('id')} "
                    f"row {index}: {observed_count} != {row_num_samples}"
                )
    expected_dataset_totals = cell.get("dataset_totals")
    if isinstance(expected_dataset_totals, Mapping):
        observed_totals = payload.get("total_samples_by_dataset")
        required_totals = {
            str(key): int(value)
            for key, value in expected_dataset_totals.items()
        }
        if not isinstance(observed_totals, Mapping) or {
            str(key): int(value)
            for key, value in observed_totals.items()
        } != required_totals:
            raise ConsolidationError(
                f"Dataset totals mismatch for cell {cell.get('id')}"
            )
    expected_checkpoints = cell.get("checkpoints")
    if isinstance(expected_checkpoints, list):
        observed_checkpoints = payload.get("checkpoints")
        if not isinstance(observed_checkpoints, list) or len(
            observed_checkpoints
        ) != len(expected_checkpoints):
            raise ConsolidationError(
                f"Checkpoint-set size mismatch for cell {cell.get('id')}"
            )
        if any(
            not _matches_path(observed, str(expected))
            for observed, expected in zip(
                observed_checkpoints, expected_checkpoints
            )
        ):
            raise ConsolidationError(
                f"Checkpoint-set identity mismatch for cell {cell.get('id')}"
            )
    return payload, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(staging, path)


def consolidate(
    *,
    run_root: Path,
    manifest_path: Path,
    output_path: Path,
    completion_manifest_path: Path,
    allow_consolidator_only_repair: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    for label, path in (
        ("run manifest", manifest_path),
        ("consolidated output", output_path),
        ("completion manifest", completion_manifest_path),
    ):
        resolved = path.resolve()
        if resolved == run_root or run_root not in resolved.parents:
            raise ConsolidationError(f"{label} must be inside the run root")
    manifest = _read_object(manifest_path)
    if manifest.get("schema_id") != "tsunami-surrogate.evaluation-run-manifest.v1":
        raise ConsolidationError("Evaluation run manifest schema mismatch")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ConsolidationError("Evaluation run manifest declares no cells")
    seen_keys: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    merged: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            raise ConsolidationError("Invalid cell declaration")
        cell_id = str(raw_cell.get("id", "")).strip()
        group = str(raw_cell.get("group", "")).strip()
        relative_path = str(raw_cell.get("path", "")).strip()
        if not cell_id or not group or not relative_path:
            raise ConsolidationError("Cell id, group, and path are required")
        cell_key = (group, cell_id)
        if cell_key in seen_keys:
            raise ConsolidationError(
                f"Duplicate cell identity: {group}:{cell_id}"
            )
        if relative_path in seen_paths:
            raise ConsolidationError(f"Duplicate cell path: {relative_path}")
        seen_keys.add(cell_key)
        seen_paths.add(relative_path)

        payload, path = _validate_cell(run_root=run_root, cell=raw_cell)
        merged.setdefault(group, {})[cell_id] = payload
        artifacts.append(
            {
                "cell_id": cell_id,
                "group": group,
                "path": relative_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )

    preflight_path = run_root / str(
        manifest.get("preflight_report", "preflight_report.json")
    )
    if not preflight_path.is_file() or _sha256(preflight_path) != str(
        manifest.get("preflight_report_sha256", "")
    ):
        raise ConsolidationError(
            "Preflight report is missing or changed after manifest creation"
        )
    consolidation_repair = _validate_live_bindings(
        manifest,
        allow_consolidator_only_repair=allow_consolidator_only_repair,
    )

    allowed_files = {
        str(Path(str(cell["path"])))
        for cell in cells
        if isinstance(cell, Mapping)
    }
    allowed_files.update(
        {
            str(manifest_path.relative_to(run_root)),
            "preflight_report.json",
            str(output_path.relative_to(run_root)),
            str(completion_manifest_path.relative_to(run_root)),
        }
    )
    undeclared = sorted(
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if path.is_file()
        and str(path.relative_to(run_root)) not in allowed_files
    )
    if undeclared:
        raise ConsolidationError(
            "Undeclared evaluation outputs are present:\n  - "
            + "\n  - ".join(undeclared)
        )

    output = {
        "schema_id": "tsunami-surrogate.consolidated-evaluation.v1",
        "suite_id": manifest.get("suite_id"),
        "run_id": manifest.get("run_id"),
        "groups": merged,
    }
    _write_atomic(output_path, output)
    artifacts.append(
        {
            "cell_id": "all_results",
            "group": "consolidated",
            "path": str(output_path.relative_to(run_root)),
            "size_bytes": int(output_path.stat().st_size),
            "sha256": _sha256(output_path),
        }
    )
    artifacts.append(
        {
            "cell_id": "preflight_report",
            "group": "provenance",
            "path": str(preflight_path.relative_to(run_root)),
            "size_bytes": int(preflight_path.stat().st_size),
            "sha256": _sha256(preflight_path),
        }
    )
    completion = {
        "schema_id": "tsunami-surrogate.evaluation-run-completion.v1",
        "suite_id": manifest.get("suite_id"),
        "run_id": manifest.get("run_id"),
        "status": "validated",
        "run_manifest": str(manifest_path.relative_to(run_root)),
        "run_manifest_sha256": _sha256(manifest_path),
        "artifacts": artifacts,
    }
    if consolidation_repair is not None:
        completion["consolidation_repair"] = consolidation_repair
    _write_atomic(completion_manifest_path, completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--completion-manifest", default=None)
    parser.add_argument(
        "--allow-consolidator-only-repair",
        action="store_true",
        help=(
            "Permit reuse of completed staging outputs only when the sole "
            "evaluation-tree change since preflight is this consolidator."
        ),
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else run_root / "run_manifest.json"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_root / "all_results.json"
    )
    completion_path = (
        Path(args.completion_manifest).resolve()
        if args.completion_manifest
        else run_root / "completion_manifest.json"
    )
    completion = consolidate(
        run_root=run_root,
        manifest_path=manifest_path,
        output_path=output_path,
        completion_manifest_path=completion_path,
        allow_consolidator_only_repair=bool(
            args.allow_consolidator_only_repair
        ),
    )
    print(
        f"[consolidate] status={completion['status']} "
        f"artifacts={len(completion['artifacts'])} -> {output_path}"
    )


if __name__ == "__main__":
    main()
