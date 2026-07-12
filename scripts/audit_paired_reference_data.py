#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ShardedTsunamiDataset, TsunamiDataset
from src.evaluation.alignment import (
    DEFAULT_COMMON_TIME_GRID,
    DEFAULT_ENDPOINT_TOLERANCE,
    DEFAULT_ZERO_TIME_TOLERANCE,
    MODE_COMMON_TIME,
    SCHEMA_ID,
    stable_hash_payload,
    stable_hash_scenario_ids,
    validate_common_time_grid,
    validate_timestamps,
)
from src.evaluation.normalization_bridge import (
    load_input_order,
    load_standardization_spec,
)
from src.evaluation.target_scaling import resolve_dataset_npz
from src.utils.config import load_config
from src.utils.io import get_git_commit, save_json


SOLVER_ORDER = ("hydrostatic", "muscl_hr", "boussinesq")
DISPLAY_SOLVER_NAMES = {
    "hydrostatic": "Hydrostatic",
    "muscl_hr": "MUSCL-HR",
    "boussinesq": "Boussinesq",
}
COMMON_SAMPLE_FIELD_CANDIDATES = {
    "bathymetry": ("bathymetry",),
    "source_field": ("source_field", "source"),
    "eta0": ("eta0",),
    "rest_depth": ("rest_depth",),
    "initial_depth": ("initial_depth", "h0"),
    "free_surface0": ("free_surface0", "initial_surface"),
}
TIMESTAMP_FIELD_CANDIDATES = ("timestamps",)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _ensure_mapping(raw: Any, *, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT / path


def _normalize_override_mapping(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    if overrides is None:
        return {}
    return {str(key): str(value) for key, value in overrides.items()}


def _mapping_arg(values: Iterable[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values or ():
        if "=" not in str(raw):
            raise ValueError(f"Expected KEY=PATH mapping, got {raw!r}")
        key, value = str(raw).split("=", 1)
        key_text = key.strip()
        value_text = value.strip()
        if not key_text or not value_text:
            raise ValueError(f"Invalid KEY=PATH mapping {raw!r}")
        out[key_text] = value_text
    return out


def _repo_root_candidates(path_text: str) -> list[Path]:
    candidates = [ROOT]
    marker = "/data/"
    text = str(path_text)
    if marker in text:
        prefix = text.split(marker, 1)[0]
        if prefix:
            candidates.append(Path(prefix))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _candidate_sample_dirs(
    sample_dir_text: str,
    *,
    solver_key: str,
    sample_index: int,
    raw_root_override: Path | None,
) -> Iterable[Path]:
    sample_name = f"sample_{int(sample_index):06d}"
    if raw_root_override is not None:
        yield raw_root_override / sample_name

    raw_path = Path(str(sample_dir_text))
    yield raw_path

    text = str(sample_dir_text)
    marker = "/data/"
    if marker in text:
        rel = Path("data") / Path(text.split(marker, 1)[1])
        for repo_root in _repo_root_candidates(text):
            yield repo_root / rel

    yield ROOT / "data" / "test" / "raw" / solver_key / "samples" / sample_name


def resolve_sample_dir(
    sample_dir_text: str,
    *,
    solver_key: str,
    sample_index: int,
    raw_root_override: Path | None,
) -> Path:
    seen: set[str] = set()
    for candidate in _candidate_sample_dirs(
        sample_dir_text,
        solver_key=solver_key,
        sample_index=sample_index,
        raw_root_override=raw_root_override,
    ):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve sample directory for solver={solver_key!r} "
        f"sample_index={sample_index} from {sample_dir_text!r}"
    )


def _resolve_relative_path(
    path_text: str,
    *,
    repo_roots: Iterable[Path],
) -> Path:
    path = Path(str(path_text))
    if path.is_absolute() and path.exists():
        return path
    for repo_root in repo_roots:
        candidate = repo_root / path
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    raise FileNotFoundError(path_text)


def _load_npz_key_inventory(path: Path) -> list[str]:
    with np.load(path) as payload:
        return sorted(payload.files)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected object JSON in {path}")
    return dict(data)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _normalize_scalar(value.item())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite scalar value {value!r}")
        return float(value)
    return str(value)


def _compare_scalar(
    *,
    left: Any,
    right: Any,
    atol: float,
) -> bool:
    left_value = _normalize_scalar(left)
    right_value = _normalize_scalar(right)
    if isinstance(left_value, float) or isinstance(right_value, float):
        return bool(abs(float(left_value) - float(right_value)) <= float(atol))
    return left_value == right_value


def _load_required_array(
    npz_path: Path, field_name: str, candidates: Iterable[str]
) -> np.ndarray:
    with np.load(npz_path) as payload:
        keys = sorted(payload.files)
        for key in candidates:
            if key not in payload:
                continue
            values = np.asarray(payload[key], dtype=np.float32)
            if values.size == 0:
                raise ValueError(f"Field {field_name!r} in {npz_path} is empty")
            return values
    raise KeyError(
        f"Missing field {field_name!r} in {npz_path}. "
        f"Looked for keys {tuple(candidates)!r}, available keys are {keys!r}."
    )


def _load_required_scalar(
    npz_path: Path, field_name: str, candidates: Iterable[str]
) -> Any:
    with np.load(npz_path) as payload:
        keys = sorted(payload.files)
        for key in candidates:
            if key not in payload:
                continue
            values = np.asarray(payload[key]).reshape(-1)
            if values.size == 0:
                raise ValueError(f"Field {field_name!r} in {npz_path} is empty")
            return values[0]
    raise KeyError(
        f"Missing field {field_name!r} in {npz_path}. "
        f"Looked for keys {tuple(candidates)!r}, available keys are {keys!r}."
    )


def _load_timestamps(sample_dir: Path) -> np.ndarray:
    for filename in ("sample.npz", "rollout.npz"):
        npz_path = sample_dir / filename
        if not npz_path.is_file():
            continue
        with np.load(npz_path) as payload:
            for key in TIMESTAMP_FIELD_CANDIDATES:
                if key not in payload:
                    continue
                values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
                if values.size == 0:
                    raise ValueError(f"Timestamps in {npz_path} are empty")
                return values
    raise KeyError(
        f"Could not find timestamps in {sample_dir}/sample.npz or rollout.npz"
    )


def _load_sample_materials(sample_dir: Path) -> dict[str, Any]:
    sample_npz = sample_dir / "sample.npz"
    if not sample_npz.is_file():
        raise FileNotFoundError(sample_npz)
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    arrays: dict[str, np.ndarray] = {}
    for field_name, candidates in COMMON_SAMPLE_FIELD_CANDIDATES.items():
        arrays[field_name] = _load_required_array(sample_npz, field_name, candidates)
    arrays["timestamps"] = _load_timestamps(sample_dir)
    scenario_id = _load_required_scalar(sample_npz, "scenario_id", ("scenario_id",))
    solver_name = _load_required_scalar(sample_npz, "solver_name", ("solver_name",))

    return {
        "sample_npz": sample_npz,
        "meta": _load_json(meta_path),
        "scenario_id": str(scenario_id),
        "solver_name": str(solver_name),
        "arrays": arrays,
    }


def _array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    return stable_hash_payload(
        {
            "dtype": str(arr.dtype),
            "shape": list(map(int, arr.shape)),
            "bytes_sha256": hashlib.sha256(arr.view(np.uint8).tobytes()).hexdigest(),
        }
    )


def _append_issue(
    issues: list[dict[str, Any]],
    *,
    check: str,
    message: str,
    scenario_id: str | None = None,
    solver: str | None = None,
    field: str | None = None,
) -> None:
    issue = {
        "severity": "error",
        "check": str(check),
        "message": str(message),
    }
    if scenario_id is not None:
        issue["scenario_id"] = str(scenario_id)
    if solver is not None:
        issue["solver"] = str(solver)
    if field is not None:
        issue["field"] = str(field)
    issues.append(issue)


def _compare_arrays_exact(
    *,
    expected: np.ndarray,
    actual: np.ndarray,
    check: str,
    issues: list[dict[str, Any]],
    scenario_id: str,
    solver: str,
    field: str,
) -> str:
    expected_values = np.asarray(expected, dtype=np.float32)
    actual_values = np.asarray(actual, dtype=np.float32)
    if expected_values.shape != actual_values.shape:
        _append_issue(
            issues,
            check=check,
            message=(
                f"Shape mismatch for {field}: "
                f"expected {expected_values.shape}, got {actual_values.shape}"
            ),
            scenario_id=scenario_id,
            solver=solver,
            field=field,
        )
        return _array_hash(actual_values)
    if not np.isfinite(actual_values).all():
        _append_issue(
            issues,
            check=check,
            message=f"Non-finite values found in {field}",
            scenario_id=scenario_id,
            solver=solver,
            field=field,
        )
        return _array_hash(actual_values)
    if not np.array_equal(expected_values, actual_values):
        diff = np.asarray(actual_values, dtype=np.float64) - np.asarray(
            expected_values,
            dtype=np.float64,
        )
        _append_issue(
            issues,
            check=check,
            message=(
                f"Exact common-field mismatch for {field}; "
                f"max_abs_diff={float(np.max(np.abs(diff))):.8g}"
            ),
            scenario_id=scenario_id,
            solver=solver,
            field=field,
        )
    return _array_hash(actual_values)


def _derive_sea_level_offset(
    *,
    bathymetry: np.ndarray,
    rest_depth: np.ndarray,
    tolerance: float,
) -> tuple[float, dict[str, Any]]:
    wet_mask = np.asarray(rest_depth, dtype=np.float64) > 0.0
    candidates = (
        np.asarray(rest_depth, dtype=np.float64)[wet_mask]
        + np.asarray(
            bathymetry,
            dtype=np.float64,
        )[wet_mask]
    )
    if candidates.size == 0:
        raise ValueError("Could not derive sea-level offset because no wet cells exist")
    spread = float(np.max(candidates) - np.min(candidates))
    estimate = float(np.median(candidates))
    if spread > float(tolerance):
        raise ValueError(
            f"Derived sea-level offset is inconsistent across wet cells: spread={spread}"
        )
    return estimate, {
        "status": "derived_from_rest_depth_and_bathymetry",
        "wet_cell_count": int(candidates.size),
        "estimate": estimate,
        "candidate_min": float(np.min(candidates)),
        "candidate_max": float(np.max(candidates)),
        "candidate_spread": spread,
    }


def _scenario_record_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row.get("sample_index", -1)), str(row.get("scenario_id", ""))


def _load_scenario_manifest(
    config_path: Path | None,
    *,
    first_processed_rows: list[Mapping[str, Any]],
    explicit_path: Path | None,
) -> tuple[Path, list[dict[str, Any]]]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    if config_path is not None:
        candidates.append(config_path)
    candidates.append(ROOT / "data" / "test" / "synthetic" / "scenario_manifest.jsonl")
    if first_processed_rows:
        sample_dir = str(first_processed_rows[0].get("sample_dir", ""))
        for repo_root in _repo_root_candidates(sample_dir):
            candidates.append(
                repo_root / "data" / "test" / "synthetic" / "scenario_manifest.jsonl"
            )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate, _read_jsonl(candidate)
    raise FileNotFoundError("Could not resolve scenario_manifest.jsonl")


def _inspect_archive_structures(
    scenario_manifest_row: Mapping[str, Any],
    sample_dirs: Mapping[str, Path],
) -> dict[str, Any]:
    sample_dir = next(iter(sample_dirs.values()))
    repo_roots = _repo_root_candidates(str(sample_dir))
    source_cache_path = _resolve_relative_path(
        str(scenario_manifest_row["source_cache_path"]),
        repo_roots=repo_roots,
    )
    bathymetry_cache_path = _resolve_relative_path(
        str(scenario_manifest_row["bathymetry_cache_path"]),
        repo_roots=repo_roots,
    )

    inventory = {
        "source_cache_keys": _load_npz_key_inventory(source_cache_path),
        "bathymetry_cache_keys": _load_npz_key_inventory(bathymetry_cache_path),
    }
    for solver_key, sample_path in sample_dirs.items():
        inventory[solver_key] = {
            "sample_npz_keys": _load_npz_key_inventory(sample_path / "sample.npz"),
            "rollout_npz_keys": _load_npz_key_inventory(sample_path / "rollout.npz"),
            "meta_json_keys": sorted(_load_json(sample_path / "meta.json").keys()),
            "meta_solver_keys": sorted(
                _ensure_mapping(
                    _load_json(sample_path / "meta.json").get("solver"),
                    label="meta.solver",
                ).keys()
            ),
        }
    return inventory


def _normalize_input_order(raw_values: Iterable[Any], *, label: str) -> list[str]:
    order = [str(value).strip() for value in raw_values]
    if not order or any(not name for name in order):
        raise ValueError(f"{label} must contain non-empty channel names")
    if len(set(order)) != len(order):
        raise ValueError(f"{label} must not contain duplicate channel names: {order!r}")
    return order


def _read_input_order_manifest(manifest_path: Path) -> list[str] | None:
    if not manifest_path.is_file():
        return None
    payload = _load_json(manifest_path)
    raw_values = payload.get("input_order")
    if not isinstance(raw_values, list):
        return None
    return _normalize_input_order(raw_values, label=f"input_order in {manifest_path}")


def _discover_processed_input_order(
    processed_dataset_path: Path,
) -> tuple[list[str], dict[str, str]]:
    processed_path = Path(processed_dataset_path)
    manifest_candidates: list[tuple[str, Path]] = []
    if processed_path.is_dir():
        manifest_candidates.extend(
            [
                ("eval_manifest", processed_path / "eval_manifest.json"),
                ("shards_manifest", processed_path / "shards_manifest.json"),
            ]
        )
    else:
        manifest_candidates.extend(
            [
                ("eval_manifest", processed_path.parent / "eval_manifest.json"),
                ("shards_manifest", processed_path.parent / "shards_manifest.json"),
            ]
        )

    for source_name, manifest_path in manifest_candidates:
        order = _read_input_order_manifest(manifest_path)
        if order is not None:
            return order, {"source": source_name, "path": str(manifest_path)}

    order = load_input_order(processed_path)
    return order, {
        "source": "archive",
        "path": str(resolve_dataset_npz(processed_path)),
    }


def _resolve_reconstruction_channel(
    channel_name: str,
    raw_arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    name = str(channel_name)
    if name == "bathymetry":
        return np.asarray(raw_arrays["bathymetry"], dtype=np.float32)
    if name == "source":
        return np.asarray(raw_arrays["source_field"], dtype=np.float32)
    if name == "initial_depth":
        return np.asarray(raw_arrays["initial_depth"], dtype=np.float32)
    if name == "initial_surface":
        if "initial_surface" in raw_arrays:
            return np.asarray(raw_arrays["initial_surface"], dtype=np.float32)
        return np.asarray(raw_arrays["free_surface0"], dtype=np.float32)
    raise KeyError(f"Unsupported reconstructed input channel {name!r}")


def _reconstruct_processed_inputs(
    *,
    raw_arrays: Mapping[str, np.ndarray],
    input_order: Iterable[str],
    normalization_inputs: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for channel_name in input_order:
        values = _resolve_reconstruction_channel(channel_name, raw_arrays)
        if channel_name in normalization_inputs:
            offset, scale = normalization_inputs[channel_name]
            values = (values - float(offset)) / float(scale)
        channels.append(np.asarray(values, dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _iter_processed_input_records(
    processed_dataset_path: Path,
) -> Iterable[tuple[str, np.ndarray]]:
    processed_path = Path(processed_dataset_path)
    if processed_path.is_dir() and (processed_path / "shards_manifest.json").is_file():
        dataset = ShardedTsunamiDataset(processed_path, cache_size=1)
    else:
        dataset = TsunamiDataset(processed_path)

    for idx in range(len(dataset)):
        item = dataset[idx]
        scenario_id = str(item.get("scenario_id", "")).strip()
        if not scenario_id:
            raise ValueError(
                f"Processed dataset item {idx} in {processed_dataset_path} is missing scenario_id"
            )
        tensor = item["x"]
        values = (
            tensor.detach().cpu().numpy()
            if hasattr(tensor, "detach")
            else np.asarray(tensor)
        )
        yield scenario_id, np.asarray(values, dtype=np.float32)


def _run_reconstruction_control(
    *,
    expected_inputs_by_scenario: Mapping[str, np.ndarray],
    processed_dataset_path: Path,
    input_order: Iterable[str],
    comparison_atol: float,
) -> dict[str, Any]:
    normalized_input_order = list(input_order)
    checked_scenario_count = 0
    checked_channel_count = 0
    mismatch_count = 0
    mismatch_cell_count = 0
    processed_sample_count = 0
    max_abs_diff = 0.0
    max_abs_diff_by_channel = {str(name): 0.0 for name in normalized_input_order}
    mismatch_count_by_channel: Counter[str] = Counter()
    seen_scenarios: set[str] = set()
    duplicate_scenarios: list[str] = []
    unexpected_scenarios: list[str] = []
    examples: list[dict[str, Any]] = []

    for scenario_id, actual in _iter_processed_input_records(processed_dataset_path):
        processed_sample_count += 1
        if scenario_id in seen_scenarios:
            duplicate_scenarios.append(scenario_id)
            if len(examples) < 10:
                examples.append(
                    {
                        "kind": "duplicate_processed_scenario_id",
                        "scenario_id": scenario_id,
                    }
                )
            continue
        seen_scenarios.add(scenario_id)

        expected = expected_inputs_by_scenario.get(scenario_id)
        if expected is None:
            unexpected_scenarios.append(scenario_id)
            if len(examples) < 10:
                examples.append(
                    {
                        "kind": "unexpected_processed_scenario_id",
                        "scenario_id": scenario_id,
                    }
                )
            continue

        expected_values = np.asarray(expected, dtype=np.float32)
        actual_values = np.asarray(actual, dtype=np.float32)
        if actual_values.ndim != 3 or actual_values.shape != expected_values.shape:
            mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "kind": "shape_mismatch",
                        "scenario_id": scenario_id,
                        "expected_shape": list(map(int, expected_values.shape)),
                        "actual_shape": list(map(int, actual_values.shape)),
                    }
                )
            continue

        if int(actual_values.shape[0]) != len(normalized_input_order):
            mismatch_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "kind": "channel_count_mismatch",
                        "scenario_id": scenario_id,
                        "expected_channels": int(len(normalized_input_order)),
                        "actual_channels": int(actual_values.shape[0]),
                    }
                )
            continue

        diff = np.abs(
            np.asarray(actual_values, dtype=np.float64)
            - np.asarray(expected_values, dtype=np.float64)
        )
        channel_max = np.max(diff.reshape(diff.shape[0], -1), axis=1)
        checked_scenario_count += 1
        checked_channel_count += int(diff.shape[0])
        max_abs_diff = max(max_abs_diff, float(np.max(channel_max)))

        for channel_index, channel_name in enumerate(normalized_input_order):
            channel_text = str(channel_name)
            channel_max_abs = float(channel_max[channel_index])
            max_abs_diff_by_channel[channel_text] = max(
                float(max_abs_diff_by_channel[channel_text]),
                channel_max_abs,
            )
            if channel_max_abs > float(comparison_atol):
                mismatch_count += 1
                mismatch_count_by_channel[channel_text] += 1
                mismatch_cell_count += int(
                    np.sum(diff[channel_index] > float(comparison_atol))
                )
                if len(examples) < 10:
                    examples.append(
                        {
                            "kind": "value_mismatch",
                            "scenario_id": scenario_id,
                            "field": channel_text,
                            "max_abs_diff": channel_max_abs,
                        }
                    )

    missing_scenarios = sorted(set(expected_inputs_by_scenario) - seen_scenarios)
    for scenario_id in missing_scenarios[: max(0, 10 - len(examples))]:
        examples.append(
            {
                "kind": "missing_processed_scenario_id",
                "scenario_id": scenario_id,
            }
        )

    failure_reasons: list[str] = []
    if duplicate_scenarios:
        failure_reasons.append(
            f"duplicate_processed_scenario_ids={len(duplicate_scenarios)}"
        )
    if unexpected_scenarios:
        failure_reasons.append(
            f"unexpected_processed_scenario_ids={len(unexpected_scenarios)}"
        )
    if missing_scenarios:
        failure_reasons.append(
            f"missing_processed_scenario_ids={len(missing_scenarios)}"
        )
    if mismatch_count > 0:
        failure_reasons.append(
            f"channel_mismatches={mismatch_count} beyond atol={float(comparison_atol):.3g}"
        )

    status = "pass" if not failure_reasons else "fail"
    return {
        "status": status,
        "reason": "" if status == "pass" else ", ".join(failure_reasons),
        "blocking": status == "fail",
        "expected_scenario_count": int(len(expected_inputs_by_scenario)),
        "processed_dataset_sample_count": int(processed_sample_count),
        "checked_scenario_count": int(checked_scenario_count),
        "checked_channel_count": int(checked_channel_count),
        "mismatch_count": int(mismatch_count),
        "mismatch_cell_count": int(mismatch_cell_count),
        "duplicate_scenario_count": int(len(duplicate_scenarios)),
        "unexpected_scenario_count": int(len(unexpected_scenarios)),
        "missing_scenario_count": int(len(missing_scenarios)),
        "max_abs_diff": float(max_abs_diff),
        "max_abs_diff_by_channel": {
            str(name): float(value) for name, value in max_abs_diff_by_channel.items()
        },
        "mismatch_count_by_channel": {
            str(name): int(count)
            for name, count in sorted(mismatch_count_by_channel.items())
        },
        "examples": examples,
    }


def run_paired_reference_audit(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    processed_root_overrides: Mapping[str, str] | None = None,
    raw_root_overrides: Mapping[str, str] | None = None,
    scenario_manifest_override: str | None = None,
    output_dir_override: str | None = None,
    expected_scenario_count_override: int | None = None,
) -> dict[str, Any]:
    cfg = dict(config)
    if str(cfg.get("schema_id", SCHEMA_ID)) != SCHEMA_ID:
        raise ValueError(f"Config schema_id must be {SCHEMA_ID!r}")

    alignment_cfg = _ensure_mapping(cfg.get("alignment"), label="alignment")
    audit_cfg = _ensure_mapping(cfg.get("audit"), label="audit")

    mode = str(alignment_cfg.get("mode", ""))
    if mode != MODE_COMMON_TIME:
        raise ValueError(
            "Paired reference audit currently supports only mode='common-time'"
        )
    common_time_grid = validate_common_time_grid(
        _ensure_mapping(
            alignment_cfg.get("common_time_grid"), label="common_time_grid"
        ).get("values", DEFAULT_COMMON_TIME_GRID)
    )
    endpoint_tolerance = float(
        _ensure_mapping(
            alignment_cfg.get("common_time_grid"), label="common_time_grid"
        ).get("endpoint_tolerance", DEFAULT_ENDPOINT_TOLERANCE)
    )
    zero_tolerance = float(
        _ensure_mapping(audit_cfg.get("timestamp"), label="timestamp").get(
            "initial_zero_tolerance",
            DEFAULT_ZERO_TIME_TOLERANCE,
        )
    )
    scalar_tolerance = float(
        _ensure_mapping(audit_cfg.get("equality"), label="equality").get(
            "scalar_atol",
            1.0e-12,
        )
    )
    sea_level_tolerance = float(
        _ensure_mapping(audit_cfg.get("equality"), label="equality").get(
            "array_atol",
            1.0e-6,
        )
    )
    clipping_cfg = _ensure_mapping(audit_cfg.get("clipping"), label="clipping")
    block_true_pre_clipping = bool(clipping_cfg.get("block_true_pre_clipping", True))
    roundoff_tolerance = float(
        clipping_cfg.get("roundoff_residual_atol", sea_level_tolerance)
    )
    if not math.isfinite(roundoff_tolerance) or roundoff_tolerance < 0.0:
        raise ValueError(
            f"clipping.roundoff_residual_atol must be finite and non-negative, got {roundoff_tolerance}"
        )

    reconstruction_cfg = _ensure_mapping(
        audit_cfg.get("reconstruction_control"),
        label="reconstruction_control",
    )
    reconstruction_control: dict[str, Any] = {
        "enabled": bool(reconstruction_cfg.get("enabled", True)),
        "status": "not_checked",
        "reason": "",
        "blocking": False,
    }
    reconstruction_expected_inputs: dict[str, np.ndarray] = {}
    reconstruction_input_order: list[str] = []
    reconstruction_normalization_inputs: dict[str, tuple[float, float]] = {}
    reconstruction_stats_path: Path | None = None
    reconstruction_dataset_path: Path | None = None
    reconstruction_input_order_provenance: dict[str, str] | None = None
    reconstruction_atol = float(
        reconstruction_cfg.get("float32_atol", sea_level_tolerance)
    )
    if not math.isfinite(reconstruction_atol) or reconstruction_atol < 0.0:
        raise ValueError(
            f"reconstruction_control.float32_atol must be finite and non-negative, got {reconstruction_atol}"
        )
    if not reconstruction_control["enabled"]:
        reconstruction_control["reason"] = "Control disabled in configuration"
    else:
        reconstruction_dataset_path = _resolve_repo_path(
            reconstruction_cfg.get("processed_dataset_path")
            or reconstruction_cfg.get("processed_eval_dataset_path")
        )
        reconstruction_stats_path = _resolve_repo_path(
            reconstruction_cfg.get("normalization_stats_path")
        )
        reconstruction_control["float32_atol"] = float(reconstruction_atol)
        if (
            reconstruction_dataset_path is None
            or not reconstruction_dataset_path.exists()
        ):
            reconstruction_control["reason"] = (
                "Processed evaluation dataset is unavailable, so raw-channel "
                "reconstruction cannot be checked in this step"
            )
        elif (
            reconstruction_stats_path is None or not reconstruction_stats_path.exists()
        ):
            reconstruction_control["reason"] = (
                "Normalization stats are unavailable, so raw-channel "
                "reconstruction cannot be checked in this step"
            )
        else:
            try:
                reconstruction_input_order, reconstruction_input_order_provenance = (
                    _discover_processed_input_order(reconstruction_dataset_path)
                )
                normalization_spec = load_standardization_spec(
                    reconstruction_stats_path
                )
            except Exception as exc:
                reconstruction_control.update(
                    status="fail",
                    reason=repr(exc),
                    blocking=True,
                )
            else:
                reconstruction_normalization_inputs = dict(normalization_spec.inputs)
                reconstruction_control.update(
                    {
                        "status": "pending",
                        "processed_dataset_path": str(reconstruction_dataset_path),
                        "processed_dataset_kind": (
                            "sharded"
                            if reconstruction_dataset_path.is_dir()
                            and (
                                reconstruction_dataset_path / "shards_manifest.json"
                            ).is_file()
                            else "archive"
                        ),
                        "input_order": list(reconstruction_input_order),
                        "provenance": {
                            "input_order_source": str(
                                reconstruction_input_order_provenance["source"]
                            ),
                            "input_order_path": str(
                                reconstruction_input_order_provenance["path"]
                            ),
                            "normalization_stats_path": str(normalization_spec.path),
                            "normalized_input_channels": sorted(
                                normalization_spec.inputs
                            ),
                        },
                    }
                )

    processed_root_cfg = _ensure_mapping(
        audit_cfg.get("processed_test_roots"),
        label="audit.processed_test_roots",
    )
    processed_overrides = _normalize_override_mapping(processed_root_overrides)
    raw_root_cfg = _ensure_mapping(
        audit_cfg.get("raw_test_solver_roots"),
        label="audit.raw_test_solver_roots",
    )
    raw_overrides = _normalize_override_mapping(raw_root_overrides)
    expected_scenario_count = int(
        expected_scenario_count_override
        if expected_scenario_count_override is not None
        else audit_cfg.get("expected_scenario_count", 2500)
    )

    results_dir = (
        _resolve_repo_path(output_dir_override)
        if output_dir_override is not None
        else _resolve_repo_path(
            str(audit_cfg.get("results_dir", "results/common_time_validation/audit"))
        )
    )
    if results_dir is None:
        raise ValueError("Could not resolve results directory")

    processed_rows_by_solver: dict[str, list[dict[str, Any]]] = {}
    processed_root_paths: dict[str, str] = {}
    for solver_key in SOLVER_ORDER:
        root_text = str(
            processed_overrides.get(
                solver_key,
                processed_root_cfg.get(
                    solver_key,
                    f"data/processed/{solver_key}/test",
                ),
            )
        )
        processed_root = _resolve_repo_path(root_text)
        if processed_root is None:
            raise ValueError(f"Could not resolve processed root for {solver_key}")
        meta_path = processed_root / "meta.jsonl"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        processed_root_paths[solver_key] = str(processed_root)
        processed_rows_by_solver[solver_key] = _read_jsonl(meta_path)

    scenario_manifest_path, scenario_manifest_rows = _load_scenario_manifest(
        _resolve_repo_path(audit_cfg.get("scenario_manifest_path")),
        first_processed_rows=processed_rows_by_solver["hydrostatic"],
        explicit_path=_resolve_repo_path(scenario_manifest_override),
    )
    scenario_manifest_by_id = {
        str(row["scenario_id"]): row for row in scenario_manifest_rows
    }

    issues: list[dict[str, Any]] = []
    key_inventory: dict[str, Any] = {}
    if reconstruction_control["status"] == "fail":
        _append_issue(
            issues,
            check="reconstruction_control",
            message=(
                "Failed to initialize processed-input reconstruction control: "
                f"{reconstruction_control['reason']}"
            ),
            solver="hydrostatic",
        )

    ordered_pairs_by_solver: dict[str, list[tuple[int, str]]] = {
        solver_key: [_scenario_record_key(row) for row in rows]
        for solver_key, rows in processed_rows_by_solver.items()
    }
    ordered_scenario_ids_by_solver = {
        solver_key: [scenario_id for _, scenario_id in pairs]
        for solver_key, pairs in ordered_pairs_by_solver.items()
    }

    counts = {
        "expected_scenario_count": int(expected_scenario_count),
        "scenario_manifest_count": int(len(scenario_manifest_rows)),
        "processed_test_rows": {
            solver_key: int(len(rows))
            for solver_key, rows in processed_rows_by_solver.items()
        },
    }
    for solver_key, rows in processed_rows_by_solver.items():
        if len(rows) != expected_scenario_count:
            _append_issue(
                issues,
                check="processed_count",
                message=(
                    f"{DISPLAY_SOLVER_NAMES[solver_key]} has {len(rows)} processed rows, "
                    f"expected {expected_scenario_count}"
                ),
                solver=solver_key,
            )
        scenario_ids = ordered_scenario_ids_by_solver[solver_key]
        if len(set(scenario_ids)) != len(scenario_ids):
            duplicates = sorted(
                scenario_id
                for scenario_id, count in Counter(scenario_ids).items()
                if count > 1
            )
            _append_issue(
                issues,
                check="duplicate_scenario_ids",
                message=f"Duplicate scenario IDs found: {duplicates[:10]!r}",
                solver=solver_key,
            )

    reference_pairs = ordered_pairs_by_solver["hydrostatic"]
    reference_ids = ordered_scenario_ids_by_solver["hydrostatic"]
    for solver_key in SOLVER_ORDER[1:]:
        if ordered_pairs_by_solver[solver_key] != reference_pairs:
            _append_issue(
                issues,
                check="ordered_mapping_mismatch",
                message=(
                    f"Ordered sample_index/scenario_id mapping differs between "
                    f"Hydrostatic and {DISPLAY_SOLVER_NAMES[solver_key]}"
                ),
                solver=solver_key,
            )

    if len(scenario_manifest_rows) != expected_scenario_count:
        _append_issue(
            issues,
            check="scenario_manifest_count",
            message=(
                f"Scenario manifest has {len(scenario_manifest_rows)} rows, "
                f"expected {expected_scenario_count}"
            ),
        )

    family_cell_counts = Counter()
    clipping_family_counts = Counter()
    clipping_scenario_count = 0
    clipping_cell_count = 0
    roundoff_nonzero_cell_count = 0
    roundoff_exceeds_tolerance_cell_count = 0
    roundoff_residuals: list[float] = []
    timestamp_counts_by_solver: dict[str, Counter[int]] = {
        solver_key: Counter() for solver_key in SOLVER_ORDER
    }
    timestamp_coverage_failures = Counter()
    timestamp_monotonic_failures = Counter()
    timestamp_nonfinite_failures = Counter()
    timestamp_zero_failures = Counter()
    common_field_mismatch_counts = Counter()
    solver_specific_values: dict[str, defaultdict[str, Counter[str]]] = {
        solver_key: defaultdict(Counter) for solver_key in SOLVER_ORDER
    }
    boussinesq_effective_depth_bounds = {"min": float("inf"), "max": float("-inf")}
    eligible_scenarios: list[dict[str, Any]] = []
    scenario_common_hashes: list[str] = []
    sea_level_offset_estimates: list[float] = []
    sea_level_candidate_spreads: list[float] = []

    row_maps_by_solver = {
        solver_key: {str(row["scenario_id"]): row for row in rows}
        for solver_key, rows in processed_rows_by_solver.items()
    }

    for scenario_id in reference_ids:
        manifest_row = scenario_manifest_by_id.get(scenario_id)
        if manifest_row is None:
            _append_issue(
                issues,
                check="scenario_manifest_missing",
                message="Scenario is missing from scenario manifest",
                scenario_id=scenario_id,
            )
            continue

        sample_index = int(manifest_row["sample_index"])
        family_key = (
            str(manifest_row["bathymetry_type"]),
            str(manifest_row["source_type"]),
        )
        family_cell_counts[family_key] += 1

        sample_dirs: dict[str, Path] = {}
        sample_materials: dict[str, dict[str, Any]] = {}
        for solver_key in SOLVER_ORDER:
            row = row_maps_by_solver[solver_key].get(scenario_id)
            if row is None:
                _append_issue(
                    issues,
                    check="processed_missing_scenario",
                    message="Scenario is missing from processed metadata",
                    scenario_id=scenario_id,
                    solver=solver_key,
                )
                continue
            try:
                raw_root_override = _resolve_repo_path(
                    raw_overrides.get(solver_key, raw_root_cfg.get(solver_key))
                )
                sample_dir = resolve_sample_dir(
                    str(row["sample_dir"]),
                    solver_key=solver_key,
                    sample_index=int(row["sample_index"]),
                    raw_root_override=raw_root_override,
                )
                sample_dirs[solver_key] = sample_dir
                sample_materials[solver_key] = _load_sample_materials(sample_dir)
            except Exception as exc:
                _append_issue(
                    issues,
                    check="sample_resolution",
                    message=repr(exc),
                    scenario_id=scenario_id,
                    solver=solver_key,
                )

        if len(sample_materials) != len(SOLVER_ORDER):
            continue
        if not key_inventory:
            key_inventory = _inspect_archive_structures(manifest_row, sample_dirs)

        repo_roots = _repo_root_candidates(str(sample_dirs["hydrostatic"]))
        bathymetry_cache_path = _resolve_relative_path(
            str(manifest_row["bathymetry_cache_path"]),
            repo_roots=repo_roots,
        )
        source_cache_path = _resolve_relative_path(
            str(manifest_row["source_cache_path"]),
            repo_roots=repo_roots,
        )
        with np.load(bathymetry_cache_path) as payload:
            reference_bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
            reference_bathymetry_type = str(
                np.asarray(payload["bathymetry_type"]).reshape(-1)[0]
            )
        with np.load(source_cache_path) as payload:
            reference_source_field = np.asarray(
                payload["source_field"], dtype=np.float32
            )
            reference_source_type = str(
                np.asarray(payload["source_type"]).reshape(-1)[0]
            )
            reference_source_strength = float(
                np.asarray(payload["source_strength"]).reshape(-1)[0]
            )

        try:
            sea_level_offset, sea_level_details = _derive_sea_level_offset(
                bathymetry=sample_materials["hydrostatic"]["arrays"]["bathymetry"],
                rest_depth=sample_materials["hydrostatic"]["arrays"]["rest_depth"],
                tolerance=sea_level_tolerance,
            )
            sea_level_offset_estimates.append(float(sea_level_offset))
            sea_level_candidate_spreads.append(
                float(sea_level_details["candidate_spread"])
            )
        except Exception as exc:
            _append_issue(
                issues,
                check="sea_level_offset",
                message=repr(exc),
                scenario_id=scenario_id,
            )
            continue

        expected_eta0 = np.asarray(
            np.float32(reference_source_strength) * reference_source_field,
            dtype=np.float32,
        )
        expected_rest_depth = np.asarray(
            np.maximum(
                -np.asarray(reference_bathymetry, dtype=np.float64)
                + float(sea_level_offset),
                0.0,
            ),
            dtype=np.float32,
        )
        expected_initial_depth = np.asarray(
            np.maximum(
                np.asarray(expected_rest_depth, dtype=np.float64)
                + np.asarray(expected_eta0, dtype=np.float64),
                0.0,
            ),
            dtype=np.float32,
        )
        expected_free_surface0 = np.asarray(
            np.asarray(expected_initial_depth, dtype=np.float64)
            + np.asarray(reference_bathymetry, dtype=np.float64),
            dtype=np.float32,
        )

        true_clipping_mask = (
            np.asarray(expected_rest_depth, dtype=np.float64)
            + np.asarray(expected_eta0, dtype=np.float64)
        ) < 0.0
        if np.any(true_clipping_mask):
            clipping_scenario_count += 1
            clip_cells = int(np.sum(true_clipping_mask))
            clipping_cell_count += clip_cells
            clipping_family_counts[family_key] += clip_cells
            if block_true_pre_clipping:
                _append_issue(
                    issues,
                    check="true_pre_clipping",
                    message=(
                        "Initial-depth reconstruction requires clipping in "
                        f"{clip_cells} cells before serialization"
                    ),
                    scenario_id=scenario_id,
                    solver="hydrostatic",
                    field="initial_depth",
                )

        hydro_free_surface0 = np.asarray(
            sample_materials["hydrostatic"]["arrays"]["free_surface0"],
            dtype=np.float64,
        )
        hydro_eta0 = np.asarray(
            sample_materials["hydrostatic"]["arrays"]["eta0"],
            dtype=np.float64,
        )
        roundoff_residual = np.abs(hydro_free_surface0 - hydro_eta0)
        roundoff_mask = np.logical_and(
            np.logical_not(true_clipping_mask),
            roundoff_residual > 0.0,
        )
        roundoff_nonzero_cell_count += int(np.sum(roundoff_mask))
        if np.any(roundoff_mask):
            roundoff_residuals.extend(
                roundoff_residual[roundoff_mask].reshape(-1).tolist()
            )
        roundoff_exceed_mask = np.logical_and(
            roundoff_mask,
            roundoff_residual > float(roundoff_tolerance),
        )
        roundoff_exceeds_tolerance_cell_count += int(np.sum(roundoff_exceed_mask))
        if np.any(roundoff_exceed_mask):
            _append_issue(
                issues,
                check="free_surface0_minus_eta0_residual",
                message=(
                    "Non-clipping free_surface0-eta0 residual exceeds tolerance; "
                    f"max_abs_diff={float(np.max(roundoff_residual[roundoff_exceed_mask])):.8g}"
                ),
                scenario_id=scenario_id,
                solver="hydrostatic",
                field="free_surface0",
            )

        if (
            reconstruction_control["enabled"]
            and reconstruction_control["status"] == "pending"
        ):
            try:
                reconstruction_expected_inputs[scenario_id] = (
                    _reconstruct_processed_inputs(
                        raw_arrays=sample_materials["hydrostatic"]["arrays"],
                        input_order=reconstruction_input_order,
                        normalization_inputs=reconstruction_normalization_inputs,
                    )
                )
            except Exception as exc:
                reconstruction_control.update(
                    status="fail",
                    reason=(
                        "Failed to reconstruct raw inputs for processed comparison: "
                        f"{repr(exc)}"
                    ),
                    blocking=True,
                )
                _append_issue(
                    issues,
                    check="reconstruction_control",
                    message=reconstruction_control["reason"],
                    scenario_id=scenario_id,
                    solver="hydrostatic",
                )

        common_array_hashes: dict[str, str] = {}
        reference_common_scalars = {
            "sample_index": sample_index,
            "scenario_id": scenario_id,
            "bathymetry_type": str(manifest_row["bathymetry_type"]),
            "source_type": str(manifest_row["source_type"]),
            "source_strength": float(manifest_row["source_strength"]),
            "sea_level_offset": float(sea_level_offset),
        }

        for solver_key, materials in sample_materials.items():
            row = row_maps_by_solver[solver_key][scenario_id]
            meta = materials["meta"]
            meta_solver = _ensure_mapping(meta.get("solver"), label="meta.solver")
            arrays = materials["arrays"]

            if int(row["sample_index"]) != sample_index:
                _append_issue(
                    issues,
                    check="sample_index_mismatch",
                    message=(
                        f"Processed sample_index {row['sample_index']} does not match "
                        f"scenario manifest sample_index {sample_index}"
                    ),
                    scenario_id=scenario_id,
                    solver=solver_key,
                )

            if str(materials["scenario_id"]) != scenario_id:
                _append_issue(
                    issues,
                    check="scenario_id_mismatch",
                    message=(
                        f"sample.npz scenario_id {materials['scenario_id']!r} does not match "
                        f"expected {scenario_id!r}"
                    ),
                    scenario_id=scenario_id,
                    solver=solver_key,
                )

            for field_name, expected_array in (
                ("bathymetry", reference_bathymetry),
                ("source_field", reference_source_field),
                ("eta0", expected_eta0),
                ("rest_depth", expected_rest_depth),
                ("initial_depth", expected_initial_depth),
                ("free_surface0", expected_free_surface0),
            ):
                field_hash = _compare_arrays_exact(
                    expected=expected_array,
                    actual=arrays[field_name],
                    check="common_field_equality",
                    issues=issues,
                    scenario_id=scenario_id,
                    solver=solver_key,
                    field=field_name,
                )
                common_field_mismatch_counts[field_name] += int(
                    not np.array_equal(
                        np.asarray(expected_array, dtype=np.float32),
                        np.asarray(arrays[field_name], dtype=np.float32),
                    )
                )
                if solver_key == "hydrostatic":
                    common_array_hashes[field_name] = field_hash

            for scalar_name, actual_value, expected_value in (
                ("scenario_id", materials["scenario_id"], scenario_id),
                ("sample_index", row["sample_index"], sample_index),
                (
                    "bathymetry_type",
                    row.get("bathymetry_type"),
                    reference_bathymetry_type,
                ),
                ("source_type", row.get("source_type"), reference_source_type),
                (
                    "source_strength",
                    row.get("source_strength"),
                    reference_source_strength,
                ),
                ("meta.scenario_id", meta.get("scenario_id"), scenario_id),
                ("meta.sample_index", meta.get("sample_index"), sample_index),
                (
                    "meta.bathymetry_type",
                    meta.get("bathymetry_type"),
                    reference_bathymetry_type,
                ),
                ("meta.source_type", meta.get("source_type"), reference_source_type),
                (
                    "meta.source_strength",
                    meta.get("source_strength"),
                    reference_source_strength,
                ),
            ):
                if not _compare_scalar(
                    left=actual_value, right=expected_value, atol=scalar_tolerance
                ):
                    _append_issue(
                        issues,
                        check="common_scalar_equality",
                        message=(
                            f"Scalar mismatch for {scalar_name}: "
                            f"{actual_value!r} != {expected_value!r}"
                        ),
                        scenario_id=scenario_id,
                        solver=solver_key,
                        field=scalar_name,
                    )

            common_solver_scalars = {
                "dx": meta_solver.get("dx"),
                "dy": meta_solver.get("dy"),
                "g": meta_solver.get("g"),
                "boundary": meta_solver.get("boundary"),
                "use_sponge": meta_solver.get("use_sponge"),
                "sponge_width": meta_solver.get("sponge_width"),
                "sponge_min_factor": meta_solver.get("sponge_min_factor"),
                "sea_level_offset": sea_level_offset,
            }
            for field_name, actual_value in common_solver_scalars.items():
                if actual_value is None:
                    _append_issue(
                        issues,
                        check="missing_common_solver_setting",
                        message=f"Required common solver setting {field_name!r} is missing",
                        scenario_id=scenario_id,
                        solver=solver_key,
                        field=field_name,
                    )
            if solver_key == "hydrostatic":
                reference_common_scalars.update(common_solver_scalars)
            else:
                for field_name, actual_value in common_solver_scalars.items():
                    if not _compare_scalar(
                        left=actual_value,
                        right=reference_common_scalars[field_name],
                        atol=scalar_tolerance,
                    ):
                        _append_issue(
                            issues,
                            check="common_solver_setting_equality",
                            message=(
                                f"Cross-solver mismatch for {field_name}: "
                                f"{actual_value!r} != {reference_common_scalars[field_name]!r}"
                            ),
                            scenario_id=scenario_id,
                            solver=solver_key,
                            field=field_name,
                        )

            try:
                timestamps = validate_timestamps(
                    arrays["timestamps"],
                    zero_tolerance=zero_tolerance,
                )
            except Exception as exc:
                issue_text = str(exc)
                if "finite" in issue_text:
                    timestamp_nonfinite_failures[solver_key] += 1
                elif "strictly increasing" in issue_text:
                    timestamp_monotonic_failures[solver_key] += 1
                else:
                    timestamp_zero_failures[solver_key] += 1
                _append_issue(
                    issues,
                    check="timestamp_validation",
                    message=repr(exc),
                    scenario_id=scenario_id,
                    solver=solver_key,
                    field="timestamps",
                )
                continue

            timestamp_counts_by_solver[solver_key][int(timestamps.size)] += 1
            if "num_frames" in meta and int(meta["num_frames"]) != int(timestamps.size):
                _append_issue(
                    issues,
                    check="timestamp_count_mismatch",
                    message=(
                        f"meta.num_frames={meta['num_frames']} does not match "
                        f"timestamps length {timestamps.size}"
                    ),
                    scenario_id=scenario_id,
                    solver=solver_key,
                    field="timestamps",
                )
            if np.any(common_time_grid < timestamps[0] - endpoint_tolerance) or np.any(
                common_time_grid > timestamps[-1] + endpoint_tolerance
            ):
                timestamp_coverage_failures[solver_key] += 1
                _append_issue(
                    issues,
                    check="timestamp_coverage",
                    message=(
                        "Configured common-time grid is not fully supported by raw timestamps"
                    ),
                    scenario_id=scenario_id,
                    solver=solver_key,
                    field="timestamps",
                )

            for field_name in ("dry_tolerance", "max_velocity"):
                if field_name in meta_solver:
                    solver_specific_values[solver_key][field_name][
                        repr(_normalize_scalar(meta_solver[field_name]))
                    ] += 1

            if solver_key == "boussinesq":
                for field_name in (
                    "alpha",
                    "filter_strength",
                    "linear_solver_tol",
                    "linear_solver_max_iter",
                    "min_depth",
                    "depth_scale",
                    "mode",
                ):
                    if field_name in meta_solver:
                        solver_specific_values[solver_key][field_name][
                            repr(_normalize_scalar(meta_solver[field_name]))
                        ] += 1
                depth_scale = float(meta_solver.get("depth_scale", 1.0))
                min_depth = float(meta_solver.get("min_depth", 0.0))
                effective_depth = np.maximum(
                    (
                        -np.asarray(reference_bathymetry, dtype=np.float64)
                        + float(sea_level_offset)
                    )
                    * depth_scale,
                    min_depth,
                )
                boussinesq_effective_depth_bounds["min"] = min(
                    boussinesq_effective_depth_bounds["min"],
                    float(np.min(effective_depth)),
                )
                boussinesq_effective_depth_bounds["max"] = max(
                    boussinesq_effective_depth_bounds["max"],
                    float(np.max(effective_depth)),
                )

        scenario_common_hash = stable_hash_payload(
            {
                "schema_id": SCHEMA_ID,
                "artifact_kind": "paired-reference-scenario-fingerprint",
                "sample_index": sample_index,
                "scenario_id": scenario_id,
                "bathymetry_type": reference_common_scalars["bathymetry_type"],
                "source_type": reference_common_scalars["source_type"],
                "source_strength": reference_common_scalars["source_strength"],
                "dx": reference_common_scalars["dx"],
                "dy": reference_common_scalars["dy"],
                "g": reference_common_scalars["g"],
                "boundary": reference_common_scalars["boundary"],
                "use_sponge": reference_common_scalars["use_sponge"],
                "sponge_width": reference_common_scalars["sponge_width"],
                "sponge_min_factor": reference_common_scalars["sponge_min_factor"],
                "sea_level_offset": reference_common_scalars["sea_level_offset"],
                "array_hashes": common_array_hashes,
            }
        )
        scenario_common_hashes.append(scenario_common_hash)
        eligible_scenarios.append(
            {
                "sample_index": sample_index,
                "scenario_id": scenario_id,
                "bathymetry_type": reference_common_scalars["bathymetry_type"],
                "source_type": reference_common_scalars["source_type"],
                "source_strength": reference_common_scalars["source_strength"],
                "common_fingerprint_hash": scenario_common_hash,
            }
        )

    clipping_fraction = (
        float(
            clipping_cell_count
            / max(expected_scenario_count * int(reference_bathymetry.size), 1)
        )
        if eligible_scenarios
        else 0.0
    )
    roundoff_summary = {
        "atol": float(roundoff_tolerance),
        "nonzero_cell_count": int(roundoff_nonzero_cell_count),
        "exceeds_tolerance_cell_count": int(roundoff_exceeds_tolerance_cell_count),
        "max_abs_residual": float(
            max(roundoff_residuals) if roundoff_residuals else 0.0
        ),
        "p50_abs_residual": float(np.percentile(roundoff_residuals, 50))
        if roundoff_residuals
        else 0.0,
        "p95_abs_residual": float(np.percentile(roundoff_residuals, 95))
        if roundoff_residuals
        else 0.0,
    }
    sea_level_summary = {
        "status": "derived_from_rest_depth_and_bathymetry",
        "scenario_count": int(len(sea_level_offset_estimates)),
        "estimate_min": float(min(sea_level_offset_estimates))
        if sea_level_offset_estimates
        else None,
        "estimate_max": float(max(sea_level_offset_estimates))
        if sea_level_offset_estimates
        else None,
        "candidate_spread_max": float(max(sea_level_candidate_spreads))
        if sea_level_candidate_spreads
        else None,
    }
    if (
        reconstruction_control["enabled"]
        and reconstruction_control["status"] == "pending"
        and reconstruction_dataset_path is not None
    ):
        reconstruction_summary = _run_reconstruction_control(
            expected_inputs_by_scenario=reconstruction_expected_inputs,
            processed_dataset_path=reconstruction_dataset_path,
            input_order=reconstruction_input_order,
            comparison_atol=reconstruction_atol,
        )
        reconstruction_control.update(reconstruction_summary)
        if reconstruction_control["status"] == "fail":
            _append_issue(
                issues,
                check="reconstruction_control",
                message=(
                    "Processed-input reconstruction control failed: "
                    f"{reconstruction_control['reason']}"
                ),
                solver="hydrostatic",
            )

    family_cells = [
        {
            "bathymetry_type": bathymetry_type,
            "source_type": source_type,
            "eligible_count": int(count),
            "clipping_cell_count": int(
                clipping_family_counts.get((bathymetry_type, source_type), 0)
            ),
        }
        for (bathymetry_type, source_type), count in sorted(family_cell_counts.items())
    ]
    ordered_scenario_hash = stable_hash_scenario_ids(reference_ids)
    audit_hash = stable_hash_payload(
        {
            "schema_id": SCHEMA_ID,
            "artifact_kind": "paired-reference-audit",
            "mode": MODE_COMMON_TIME,
            "field": str(alignment_cfg.get("field", "")),
            "common_time_grid": common_time_grid.tolist(),
            "ordered_scenario_hash": ordered_scenario_hash,
            "scenario_common_hashes": scenario_common_hashes,
            "family_cells": family_cells,
        }
    )

    status = "pass" if not issues else "fail"
    summary = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "paired-reference-audit",
        "status": status,
        "audit_hash": audit_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "config_path": str(config_path) if config_path is not None else None,
            "git_commit": get_git_commit(),
        },
        "config": {
            "schema_id": cfg.get("schema_id", SCHEMA_ID),
            "alignment": alignment_cfg,
            "audit": {
                **audit_cfg,
                "processed_test_roots": processed_root_paths,
                "raw_test_solver_roots": {
                    solver_key: str(
                        _resolve_repo_path(
                            raw_overrides.get(solver_key, raw_root_cfg.get(solver_key))
                        )
                    )
                    if (
                        raw_overrides.get(solver_key, raw_root_cfg.get(solver_key))
                        is not None
                    )
                    else None
                    for solver_key in SOLVER_ORDER
                },
                "scenario_manifest_path": str(scenario_manifest_path),
                "results_dir": str(results_dir),
                "expected_scenario_count": expected_scenario_count,
            },
        },
        "alignment": {
            "mode": MODE_COMMON_TIME,
            "field": str(alignment_cfg.get("field", "")),
            "elevation_semantics": str(alignment_cfg.get("elevation_semantics", "")),
            "time_semantics": str(alignment_cfg.get("time_semantics", "")),
            "initial_frame_treatment": str(
                alignment_cfg.get("initial_frame_treatment", "")
            ),
            "aggregation": alignment_cfg.get("aggregation", {}),
            "common_time_grid": common_time_grid.tolist(),
            "common_time_horizon": float(common_time_grid[-1]),
            "endpoint_tolerance": float(endpoint_tolerance),
        },
        "counts": {
            **counts,
            "eligible_scenario_count": int(len(eligible_scenarios)),
            "issue_count": int(len(issues)),
        },
        "archive_key_inventory": key_inventory,
        "scenario_order": {
            "ordered_scenario_ids": list(reference_ids),
            "ordered_scenario_hash": ordered_scenario_hash,
        },
        "eligible_scenarios": eligible_scenarios,
        "family_cells": family_cells,
        "common_field_mismatches": {
            "count_by_field": {
                field: int(count)
                for field, count in common_field_mismatch_counts.items()
            }
        },
        "timestamp_audit": {
            "zero_tolerance": float(zero_tolerance),
            "coverage_endpoint_tolerance": float(endpoint_tolerance),
            "counts_by_solver": {
                solver_key: {
                    str(count): int(total)
                    for count, total in sorted(
                        timestamp_counts_by_solver[solver_key].items()
                    )
                }
                for solver_key in SOLVER_ORDER
            },
            "nonfinite_failures": {
                solver_key: int(timestamp_nonfinite_failures[solver_key])
                for solver_key in SOLVER_ORDER
            },
            "monotonic_failures": {
                solver_key: int(timestamp_monotonic_failures[solver_key])
                for solver_key in SOLVER_ORDER
            },
            "initial_zero_failures": {
                solver_key: int(timestamp_zero_failures[solver_key])
                for solver_key in SOLVER_ORDER
            },
            "coverage_failures": {
                solver_key: int(timestamp_coverage_failures[solver_key])
                for solver_key in SOLVER_ORDER
            },
        },
        "clipping_audit": {
            "policy": {
                "block_true_pre_clipping": bool(block_true_pre_clipping),
                "roundoff_residual_atol": float(roundoff_tolerance),
            },
            "true_pre_clipping_scenario_count": int(clipping_scenario_count),
            "true_pre_clipping_cell_count": int(clipping_cell_count),
            "true_pre_clipping_fraction": clipping_fraction,
            "family_cell_counts_with_true_clipping": [
                {
                    "bathymetry_type": bathymetry_type,
                    "source_type": source_type,
                    "clipping_cell_count": int(count),
                }
                for (bathymetry_type, source_type), count in sorted(
                    clipping_family_counts.items()
                )
            ],
            "free_surface0_minus_eta0_residual": roundoff_summary,
            "sea_level_offset_derivation": sea_level_summary,
        },
        "solver_specific_settings": {
            solver_key: {
                field_name: {
                    "unique_values": [
                        {"value": value, "count": int(count)}
                        for value, count in sorted(counter.items())
                    ]
                }
                for field_name, counter in sorted(fields.items())
            }
            for solver_key, fields in solver_specific_values.items()
        },
        "boussinesq_effective_depth_summary": {
            "min": None
            if not math.isfinite(boussinesq_effective_depth_bounds["min"])
            else float(boussinesq_effective_depth_bounds["min"]),
            "max": None
            if not math.isfinite(boussinesq_effective_depth_bounds["max"])
            else float(boussinesq_effective_depth_bounds["max"]),
        },
        "reconstruction_control": reconstruction_control,
        "issues": issues,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paired raw reference data for strict common-time solver comparison."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/eval/common_time_alignment.yaml",
        help="Alignment/audit YAML config.",
    )
    parser.add_argument(
        "--processed-test-root",
        action="append",
        default=[],
        help="Override processed test root as solver=PATH.",
    )
    parser.add_argument(
        "--raw-test-root",
        action="append",
        default=[],
        help="Override raw solver samples root as solver=PATH.",
    )
    parser.add_argument(
        "--scenario-manifest-path",
        default=None,
        help="Optional explicit scenario_manifest.jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--expected-scenario-count",
        type=int,
        default=None,
        help="Optional override for the expected scenario count.",
    )
    args = parser.parse_args()

    config_path = (
        ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    )
    config = load_config(config_path)
    summary = run_paired_reference_audit(
        config,
        config_path=config_path,
        processed_root_overrides=_mapping_arg(args.processed_test_root),
        raw_root_overrides=_mapping_arg(args.raw_test_root),
        scenario_manifest_override=args.scenario_manifest_path,
        output_dir_override=args.output_dir,
        expected_scenario_count_override=args.expected_scenario_count,
    )

    output_dir = Path(summary["config"]["audit"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paired_reference_audit.json"
    save_json(summary, output_path)

    print(f"[audit] wrote {output_path}")
    print(f"[audit] status={summary['status']} audit_hash={summary['audit_hash']}")

    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
