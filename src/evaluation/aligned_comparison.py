from __future__ import annotations

import json
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.data.dataset import ShardedTsunamiDataset, TsunamiDataset
from src.evaluation.alignment import (
    DEFAULT_ENDPOINT_TOLERANCE,
    DEFAULT_ZERO_TIME_TOLERANCE,
    MODE_COMMON_TIME,
    MODE_SAVED_INDEX_LEGACY,
    SCHEMA_ID,
    align_elevation_series,
    compute_error_metrics,
    generate_paired_bootstrap_indices,
    stable_hash_payload,
    stable_hash_scenario_ids,
    validate_alignment_metadata,
    validate_common_time_grid,
    validate_timestamps,
)
from src.evaluation.normalization_bridge import (
    load_input_order,
    load_standardization_spec,
    normalize_raw_inputs_for_model,
)
from src.evaluation.target_scaling import resolve_dataset_npz


ALIGNED_SOLVER_COMPARISON_ARTIFACT = "aligned-solver-comparison"
EMULATOR_SUPERIORITY_ARTIFACT = "aligned-emulator-superiority"
PAIRED_REFERENCE_AUDIT_ARTIFACT = "paired-reference-audit"
SCENARIO_SELECTION_ARTIFACT = "common-time-validation-scenarios"
DENSE_VALIDATION_ARTIFACT = "dense-reference-validation"
DENSE_VALIDATION_DECISION_ARTIFACT = "dense-reference-validation-decision"
SUPPORTED_SUITES = ("smoke", "dense_validation", "full")
DEFAULT_ALIGNMENT_FIELD = "trajectory_eta"
ALIGNMENT_ELEVATION_SEMANTICS = "sea_level_offset_relative_surface_elevation"
ALIGNMENT_TIME_SEMANTICS = "solver_benchmark_time"
COMMON_TIME_INITIAL_FRAME_TREATMENT = (
    "require_saved_zero_frame_but_exclude_zero_from_common_grid"
)
COMMON_TIME_AGGREGATION = {
    "global_metric": "equal_scenario_weight_field_rmse",
    "scenario_weighting": "equal_scenario_weight",
    "field_mask": "full_field",
}
UNITS = {
    "time": "solver_benchmark_time",
    "elevation": "benchmark_scale_eta",
}
UNSUPPORTED_DENSE_FALLBACK_MESSAGE = (
    "Dense-field fallback is not currently supported because the stored dense validation "
    "artifacts stream metrics only and do not persist the aligned dense fields needed to "
    "reconstruct numerator/denominator tensors."
)
RAW_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "trajectory_eta": ("trajectory_eta",),
    "timestamps": ("timestamps",),
    "bathymetry": ("bathymetry",),
    "source_field": ("source_field", "source"),
    "initial_depth": ("initial_depth", "h0"),
    "eta0": ("eta0",),
    "free_surface0": ("free_surface0", "initial_surface"),
    "scenario_id": ("scenario_id",),
}


@dataclass(frozen=True)
class ScenarioDescriptor:
    scenario_id: str
    bathymetry_type: str
    source_type: str
    source_strength: float


@dataclass(frozen=True)
class SuiteContract:
    suite_name: str
    suite_label: str
    ordered_scenarios: tuple[ScenarioDescriptor, ...]
    ordered_scenario_ids: tuple[str, ...]
    ordered_scenario_hash: str
    common_time_grid: np.ndarray
    endpoint_tolerance: float
    field: str
    elevation_semantics: str
    time_semantics: str
    initial_frame_treatment: str
    audit_hash: str
    audit_artifact_hash: str
    scenario_selection_hash: str | None
    dense_validation_decision_hash: str | None
    dense_validation_summary_hash: str | None


@dataclass
class ProcessedInputLookup:
    dataset: Any
    dataset_path: Path
    input_order: tuple[str, ...]
    index_by_scenario_id: dict[str, int]

    def get(self, scenario_id: str) -> np.ndarray:
        if scenario_id not in self.index_by_scenario_id:
            raise KeyError(
                f"Processed dataset {self.dataset_path} is missing scenario_id={scenario_id!r}"
            )
        item = self.dataset[self.index_by_scenario_id[scenario_id]]
        tensor = item["x"]
        values = (
            tensor.detach().cpu().numpy()
            if hasattr(tensor, "detach")
            else np.asarray(tensor)
        )
        return np.asarray(values, dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True))
            handle.write("\n")


def require_explicit_mode(mode: str | None) -> str:
    if mode is None or not str(mode).strip():
        raise ValueError(
            "Explicit mode is required. Choose mode='common-time' or "
            "mode='saved-index-legacy'."
        )
    mode_text = str(mode).strip()
    if mode_text not in {MODE_COMMON_TIME, MODE_SAVED_INDEX_LEGACY}:
        raise ValueError(
            f"Unsupported mode {mode_text!r}. Expected one of "
            f"{(MODE_COMMON_TIME, MODE_SAVED_INDEX_LEGACY)!r}."
        )
    return mode_text


def validate_dense_fallback_policy(policy: str | None) -> str:
    text = str(policy or "unsupported").strip().lower()
    if text in {"unsupported", "none", "disabled"}:
        return "unsupported"
    if text in {"dense-fields", "dense_field_artifacts", "dense"}:
        raise NotImplementedError(UNSUPPORTED_DENSE_FALLBACK_MESSAGE)
    raise ValueError(
        "Unsupported dense fallback policy. Expected one of "
        "{'unsupported', 'none', 'disabled', 'dense-fields'}."
    )


def _ensure_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _ensure_mapping(payload, label=str(path))


def _hash_artifact(payload: Mapping[str, Any]) -> str:
    return stable_hash_payload(payload)


def _normalize_suite_name(suite_name: str) -> str:
    normalized = str(suite_name).strip().lower()
    if normalized not in SUPPORTED_SUITES:
        raise ValueError(
            f"Unsupported suite {suite_name!r}. Expected one of {SUPPORTED_SUITES!r}."
        )
    return normalized


def _alignment_grid_and_tolerance(
    alignment_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    grid_cfg = _ensure_mapping(
        alignment_cfg.get("common_time_grid"), label="common_time_grid"
    )
    grid = validate_common_time_grid(grid_cfg.get("values"))
    endpoint_tolerance = float(
        grid_cfg.get("endpoint_tolerance", DEFAULT_ENDPOINT_TOLERANCE)
    )
    if not math.isfinite(endpoint_tolerance) or endpoint_tolerance < 0.0:
        raise ValueError(
            f"common_time_grid.endpoint_tolerance must be finite and non-negative, got {endpoint_tolerance}"
        )
    return grid, endpoint_tolerance


def _validate_audit_alignment(
    audit_artifact: Mapping[str, Any],
    *,
    common_time_grid: np.ndarray,
    field: str,
    elevation_semantics: str,
    time_semantics: str,
    initial_frame_treatment: str,
) -> None:
    if str(audit_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Audit artifact schema_id must be {SCHEMA_ID!r}")
    if str(audit_artifact.get("artifact_kind", "")) != PAIRED_REFERENCE_AUDIT_ARTIFACT:
        raise ValueError(
            f"Expected audit artifact_kind={PAIRED_REFERENCE_AUDIT_ARTIFACT!r}"
        )
    if str(audit_artifact.get("status", "")) != "pass":
        raise ValueError("Aligned comparison requires a passing paired-reference audit")

    alignment = _ensure_mapping(
        audit_artifact.get("alignment"), label="audit.alignment"
    )
    metadata = validate_alignment_metadata(
        {
            "schema_id": SCHEMA_ID,
            "mode": alignment.get("mode", MODE_COMMON_TIME),
            "ordered_scenario_ids": audit_artifact.get("scenario_order", {}).get(
                "ordered_scenario_ids", []
            ),
            "common_time_grid": alignment.get("common_time_grid"),
            "field": alignment.get("field"),
            "elevation_semantics": alignment.get("elevation_semantics"),
            "time_semantics": alignment.get("time_semantics"),
            "initial_frame_treatment": alignment.get("initial_frame_treatment"),
            "aggregation": alignment.get("aggregation", COMMON_TIME_AGGREGATION),
        }
    )
    if metadata["mode"] != MODE_COMMON_TIME:
        raise ValueError("Audit alignment mode must be 'common-time'")
    audit_grid = validate_common_time_grid(metadata["common_time_grid"])
    if not np.array_equal(audit_grid, common_time_grid):
        raise ValueError("Audit common_time_grid does not match the requested grid")
    if str(metadata["field"]) != str(field):
        raise ValueError("Audit field does not match requested field")
    if str(metadata["elevation_semantics"]) != str(elevation_semantics):
        raise ValueError("Audit elevation semantics do not match requested semantics")
    if str(metadata["time_semantics"]) != str(time_semantics):
        raise ValueError("Audit time semantics do not match requested semantics")
    if str(metadata["initial_frame_treatment"]) != str(initial_frame_treatment):
        raise ValueError(
            "Audit initial_frame_treatment does not match requested semantics"
        )


def _load_selection_artifact(
    scenario_selection_path: str | Path,
    *,
    audit_hash: str,
) -> dict[str, Any]:
    selection_artifact = _load_json_mapping(scenario_selection_path)
    if str(selection_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Scenario selection schema_id must be {SCHEMA_ID!r}")
    if str(selection_artifact.get("artifact_kind", "")) != SCENARIO_SELECTION_ARTIFACT:
        raise ValueError(
            f"Expected scenario selection artifact_kind={SCENARIO_SELECTION_ARTIFACT!r}"
        )
    if str(selection_artifact.get("audit_hash", "")) != str(audit_hash):
        raise ValueError(
            "Scenario selection audit_hash does not match the audit artifact"
        )
    return selection_artifact


def _scenario_descriptors_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[ScenarioDescriptor, ...]:
    descriptors = []
    for row in rows:
        entry = _ensure_mapping(row, label="ordered_scenarios[]")
        descriptors.append(
            ScenarioDescriptor(
                scenario_id=str(entry.get("scenario_id", "")).strip(),
                bathymetry_type=str(entry.get("bathymetry_type", "")).strip(),
                source_type=str(entry.get("source_type", "")).strip(),
                source_strength=float(entry.get("source_strength", 0.0)),
            )
        )
    if any(not item.scenario_id for item in descriptors):
        raise ValueError("ordered_scenarios contain empty scenario_id values")
    return tuple(descriptors)


def _validate_selection_suite(
    selection_artifact: Mapping[str, Any],
    *,
    suite_name: str,
) -> tuple[str, tuple[ScenarioDescriptor, ...], tuple[str, ...]]:
    suite_payload = _ensure_mapping(
        selection_artifact.get(suite_name), label=suite_name
    )
    ordered_rows = _scenario_descriptors_from_rows(
        suite_payload.get("ordered_scenarios", [])
    )
    ordered_ids = tuple(
        str(value) for value in suite_payload.get("ordered_scenario_ids", [])
    )
    if not ordered_rows or not ordered_ids:
        raise ValueError(f"Scenario selection suite {suite_name!r} is empty")
    if tuple(item.scenario_id for item in ordered_rows) != ordered_ids:
        raise ValueError(
            f"Scenario selection suite {suite_name!r} has mismatched ordered_scenarios and ordered_scenario_ids"
        )
    expected_hash = stable_hash_scenario_ids(list(ordered_ids))
    observed_hash = str(suite_payload.get("list_hash", "")).strip()
    if observed_hash and observed_hash != expected_hash:
        raise ValueError(
            f"Scenario selection suite {suite_name!r} list_hash does not match its ordered scenario ids"
        )
    return str(suite_payload.get("label", suite_name)), ordered_rows, ordered_ids


def _load_dense_validation_gate(
    dense_validation_decision_path: str | Path,
    *,
    audit_hash: str,
    dense_validation_list_hash: str,
    common_time_grid: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision_path = Path(dense_validation_decision_path)
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"Full common-time evaluation requires a passing dense validation decision at "
            f"{decision_path}. Run scripts/run_dense_reference_validation.py first."
        )
    decision_artifact = _load_json_mapping(decision_path)
    if str(decision_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Dense validation decision schema_id must be {SCHEMA_ID!r}")
    if (
        str(decision_artifact.get("artifact_kind", ""))
        != DENSE_VALIDATION_DECISION_ARTIFACT
    ):
        raise ValueError(
            f"Expected dense validation decision artifact_kind={DENSE_VALIDATION_DECISION_ARTIFACT!r}"
        )
    if str(decision_artifact.get("status", "")) != "pass":
        raise ValueError(
            f"Dense validation decision at {decision_path} did not pass. "
            "Full sparse common-time evaluation remains blocked."
        )
    decision_suite = _ensure_mapping(
        decision_artifact.get("suite"), label="decision.suite"
    )
    if str(decision_suite.get("name", "")) != "dense_validation":
        raise ValueError(
            "Dense validation decision must come from suite='dense_validation'"
        )

    summary_path = decision_path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Dense validation summary is missing next to {decision_path}; "
            "summary.json is required to validate audit and scenario-set hashes."
        )
    summary_artifact = _load_json_mapping(summary_path)
    if str(summary_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Dense validation summary schema_id must be {SCHEMA_ID!r}")
    if str(summary_artifact.get("artifact_kind", "")) != DENSE_VALIDATION_ARTIFACT:
        raise ValueError(
            f"Expected dense validation summary artifact_kind={DENSE_VALIDATION_ARTIFACT!r}"
        )
    if str(summary_artifact.get("status", "")) != "pass":
        raise ValueError("Dense validation summary did not pass")
    summary_inputs = _ensure_mapping(
        summary_artifact.get("inputs"), label="summary.inputs"
    )
    if str(summary_inputs.get("audit_hash", "")) != str(audit_hash):
        raise ValueError(
            "Dense validation summary audit_hash does not match the audit artifact"
        )
    summary_hash = str(
        _ensure_mapping(
            summary_artifact.get("scenario_order"), label="summary.scenario_order"
        ).get("ordered_scenario_hash", "")
    )
    if summary_hash != str(dense_validation_list_hash):
        raise ValueError(
            "Dense validation summary scenario set does not match the current dense-validation selection"
        )
    summary_alignment = _ensure_mapping(
        summary_artifact.get("alignment"), label="summary.alignment"
    )
    summary_grid = validate_common_time_grid(summary_alignment.get("common_time_grid"))
    if not np.array_equal(summary_grid, common_time_grid):
        raise ValueError("Dense validation summary common_time_grid does not match")
    return decision_artifact, summary_artifact


def resolve_suite_contract(
    *,
    alignment_cfg: Mapping[str, Any],
    audit_artifact_path: str | Path,
    scenario_selection_path: str | Path | None,
    suite_name: str,
    dense_validation_decision_path: str | Path | None = None,
    require_full_suite_dense_decision: bool = True,
    dense_fallback_policy: str | None = None,
) -> SuiteContract:
    suite = _normalize_suite_name(suite_name)
    validate_dense_fallback_policy(dense_fallback_policy)

    common_time_grid, endpoint_tolerance = _alignment_grid_and_tolerance(alignment_cfg)
    field = str(alignment_cfg.get("field", DEFAULT_ALIGNMENT_FIELD)).strip()
    elevation_semantics = str(
        alignment_cfg.get("elevation_semantics", ALIGNMENT_ELEVATION_SEMANTICS)
    ).strip()
    time_semantics = str(
        alignment_cfg.get("time_semantics", ALIGNMENT_TIME_SEMANTICS)
    ).strip()
    initial_frame_treatment = str(
        alignment_cfg.get(
            "initial_frame_treatment", COMMON_TIME_INITIAL_FRAME_TREATMENT
        )
    ).strip()

    audit_artifact = _load_json_mapping(audit_artifact_path)
    _validate_audit_alignment(
        audit_artifact,
        common_time_grid=common_time_grid,
        field=field,
        elevation_semantics=elevation_semantics,
        time_semantics=time_semantics,
        initial_frame_treatment=initial_frame_treatment,
    )
    audit_hash = str(audit_artifact.get("audit_hash", "")).strip()
    if not audit_hash:
        raise ValueError("Audit artifact is missing audit_hash")

    eligible_rows = {
        str(row["scenario_id"]): _ensure_mapping(row, label="eligible_scenarios[]")
        for row in audit_artifact.get("eligible_scenarios", [])
    }
    if not eligible_rows:
        raise ValueError("Audit artifact is missing eligible_scenarios")

    selection_hash = None
    decision_hash = None
    summary_hash = None
    suite_label = suite

    if suite == "full":
        if scenario_selection_path is None:
            raise ValueError(
                "suite='full' requires a scenario selection artifact so the dense-validation "
                "gate can be checked against the current dense-validation subset."
            )
        selection_artifact = _load_selection_artifact(
            scenario_selection_path,
            audit_hash=audit_hash,
        )
        selection_hash = _hash_artifact(selection_artifact)
        _, dense_rows, dense_ids = _validate_selection_suite(
            selection_artifact,
            suite_name="dense_validation",
        )
        dense_list_hash = stable_hash_scenario_ids(list(dense_ids))
        if require_full_suite_dense_decision:
            if dense_validation_decision_path is None:
                raise FileNotFoundError(
                    "suite='full' requires dense_validation_decision_path because sparse "
                    "common-time interpolation for the full audited set is blocked until "
                    "dense validation passes."
                )
            decision_artifact, summary_artifact = _load_dense_validation_gate(
                dense_validation_decision_path,
                audit_hash=audit_hash,
                dense_validation_list_hash=dense_list_hash,
                common_time_grid=common_time_grid,
            )
            decision_hash = _hash_artifact(decision_artifact)
            summary_hash = _hash_artifact(summary_artifact)

        ordered_ids = tuple(
            str(value)
            for value in _ensure_mapping(
                audit_artifact.get("scenario_order"),
                label="audit.scenario_order",
            ).get("ordered_scenario_ids", [])
        )
        if not ordered_ids:
            raise ValueError("Audit artifact scenario_order is empty")
        ordered_rows = []
        for scenario_id in ordered_ids:
            if scenario_id not in eligible_rows:
                raise ValueError(
                    f"Audit scenario_order contains scenario_id={scenario_id!r} that is not "
                    "present in eligible_scenarios"
                )
            row = eligible_rows[scenario_id]
            ordered_rows.append(
                ScenarioDescriptor(
                    scenario_id=scenario_id,
                    bathymetry_type=str(row.get("bathymetry_type", "")).strip(),
                    source_type=str(row.get("source_type", "")).strip(),
                    source_strength=float(row.get("source_strength", 0.0)),
                )
            )
        suite_label = "full_audited_common_time_suite"
    else:
        if scenario_selection_path is None:
            raise ValueError(f"suite={suite!r} requires scenario_selection_path")
        selection_artifact = _load_selection_artifact(
            scenario_selection_path,
            audit_hash=audit_hash,
        )
        selection_hash = _hash_artifact(selection_artifact)
        suite_label, ordered_rows, ordered_ids = _validate_selection_suite(
            selection_artifact,
            suite_name=suite,
        )

    ordered_scenario_hash = stable_hash_scenario_ids(list(ordered_ids))
    return SuiteContract(
        suite_name=suite,
        suite_label=suite_label,
        ordered_scenarios=tuple(ordered_rows),
        ordered_scenario_ids=tuple(ordered_ids),
        ordered_scenario_hash=ordered_scenario_hash,
        common_time_grid=np.asarray(common_time_grid, dtype=np.float64),
        endpoint_tolerance=float(endpoint_tolerance),
        field=field,
        elevation_semantics=elevation_semantics,
        time_semantics=time_semantics,
        initial_frame_treatment=initial_frame_treatment,
        audit_hash=audit_hash,
        audit_artifact_hash=_hash_artifact(audit_artifact),
        scenario_selection_hash=selection_hash,
        dense_validation_decision_hash=decision_hash,
        dense_validation_summary_hash=summary_hash,
    )


def _load_raw_scalar(sample_dir: Path, field_name: str) -> str:
    candidates = RAW_FIELD_CANDIDATES[field_name]
    for filename in ("sample.npz", "rollout.npz"):
        npz_path = sample_dir / filename
        if not npz_path.is_file():
            continue
        with np.load(npz_path, allow_pickle=True) as payload:
            for key in candidates:
                if key not in payload:
                    continue
                values = np.asarray(payload[key]).reshape(-1)
                if values.size == 0:
                    raise ValueError(f"Field {field_name!r} in {npz_path} is empty")
                return str(values[0])
    raise KeyError(f"Missing field {field_name!r} in {sample_dir}")


def _load_raw_array(sample_dir: Path, field_name: str) -> np.ndarray:
    candidates = RAW_FIELD_CANDIDATES[field_name]
    for filename in ("sample.npz", "rollout.npz"):
        npz_path = sample_dir / filename
        if not npz_path.is_file():
            continue
        with np.load(npz_path, allow_pickle=True) as payload:
            for key in candidates:
                if key not in payload:
                    continue
                return np.asarray(payload[key], dtype=np.float32)
    raise KeyError(f"Missing field {field_name!r} in {sample_dir}")


def build_raw_scenario_index(samples_root: str | Path) -> dict[str, Path]:
    root = Path(samples_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    pattern = re.compile(r"^sample_\d{6}$")
    index: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or pattern.match(child.name) is None:
            continue
        scenario_id = _load_raw_scalar(child, "scenario_id")
        if not scenario_id:
            raise ValueError(f"Empty scenario_id in {child}")
        if scenario_id in index:
            raise ValueError(
                f"Duplicate scenario_id={scenario_id!r} found in raw root {root}"
            )
        index[scenario_id] = child
    if not index:
        raise ValueError(f"No raw sample directories found under {root}")
    return index


def load_raw_reference_sample(sample_dir: str | Path) -> dict[str, Any]:
    root = Path(sample_dir)
    trajectory_eta = _load_raw_array(root, "trajectory_eta")
    timestamps = np.asarray(
        _load_raw_array(root, "timestamps"), dtype=np.float64
    ).reshape(-1)
    return {
        "sample_dir": str(root),
        "scenario_id": _load_raw_scalar(root, "scenario_id"),
        "trajectory_eta": np.asarray(trajectory_eta, dtype=np.float64),
        "timestamps": timestamps,
        "bathymetry": _load_raw_array(root, "bathymetry"),
        "source_field": _load_raw_array(root, "source_field"),
        "initial_depth": _load_raw_array(root, "initial_depth"),
        "eta0": _load_raw_array(root, "eta0"),
        "free_surface0": _load_raw_array(root, "free_surface0"),
    }


def iter_paired_raw_reference_samples(
    *,
    contract: SuiteContract,
    left_root: str | Path,
    right_root: str | Path,
) -> Iterator[dict[str, Any]]:
    left_index = build_raw_scenario_index(left_root)
    right_index = build_raw_scenario_index(right_root)
    for descriptor in contract.ordered_scenarios:
        if descriptor.scenario_id not in left_index:
            raise KeyError(
                f"Left raw root is missing scenario_id={descriptor.scenario_id!r}"
            )
        if descriptor.scenario_id not in right_index:
            raise KeyError(
                f"Right raw root is missing scenario_id={descriptor.scenario_id!r}"
            )
        left_sample = load_raw_reference_sample(left_index[descriptor.scenario_id])
        right_sample = load_raw_reference_sample(right_index[descriptor.scenario_id])
        if str(left_sample["scenario_id"]) != str(right_sample["scenario_id"]):
            raise ValueError(
                f"Scenario pairing mismatch for scenario_id={descriptor.scenario_id!r}"
            )
        yield {
            "scenario_id": descriptor.scenario_id,
            "bathymetry_type": descriptor.bathymetry_type,
            "source_type": descriptor.source_type,
            "source_strength": descriptor.source_strength,
            "left": left_sample,
            "right": right_sample,
        }


def build_processed_input_lookup(
    processed_dataset_path: str | Path,
) -> ProcessedInputLookup:
    resolved_path = resolve_dataset_npz(processed_dataset_path)
    dataset_root = (
        resolved_path.parent if resolved_path.name.endswith(".npz") else resolved_path
    )
    if dataset_root.is_dir() and (dataset_root / "shards_manifest.json").is_file():
        dataset = ShardedTsunamiDataset(dataset_root, cache_size=1)
    else:
        dataset = TsunamiDataset(processed_dataset_path)

    index_by_scenario_id: dict[str, int] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        scenario_id = str(item.get("scenario_id", "")).strip()
        if not scenario_id:
            raise ValueError(
                f"Processed dataset {processed_dataset_path} has an empty scenario_id at index {idx}"
            )
        if scenario_id in index_by_scenario_id:
            raise ValueError(
                f"Processed dataset {processed_dataset_path} contains duplicate scenario_id={scenario_id!r}"
            )
        index_by_scenario_id[scenario_id] = idx

    return ProcessedInputLookup(
        dataset=dataset,
        dataset_path=resolved_path,
        input_order=tuple(load_input_order(processed_dataset_path)),
        index_by_scenario_id=index_by_scenario_id,
    )


def load_model_input_order(
    *,
    processed_test_path: str | Path,
    checkpoint_train_path: str | Path | None = None,
) -> tuple[str, ...]:
    primary_order = tuple(load_input_order(processed_test_path))
    if checkpoint_train_path is None:
        return primary_order
    train_order = tuple(load_input_order(checkpoint_train_path))
    if train_order != primary_order:
        raise ValueError(
            "Model input_order mismatch between the processed test dataset and the checkpoint "
            f"training dataset: {primary_order!r} != {train_order!r}"
        )
    return primary_order


def reconstruct_model_a_input(
    *,
    raw_sample: Mapping[str, Any],
    input_order: Sequence[str],
    normalization_stats_path: str | Path,
) -> np.ndarray:
    stats = load_standardization_spec(normalization_stats_path)
    raw_inputs = {
        "bathymetry": raw_sample["bathymetry"],
        "source_field": raw_sample["source_field"],
        "source": raw_sample["source_field"],
        "initial_depth": raw_sample["initial_depth"],
        "eta0": raw_sample["eta0"],
        "free_surface0": raw_sample["free_surface0"],
        "initial_surface": raw_sample["free_surface0"],
    }
    return normalize_raw_inputs_for_model(
        raw_inputs,
        input_order=input_order,
        model_stats=stats,
    )


def verify_reconstructed_input_match(
    *,
    scenario_id: str,
    reconstructed_input: np.ndarray,
    lookup: ProcessedInputLookup,
    atol: float,
) -> dict[str, Any]:
    actual = lookup.get(scenario_id)
    expected = np.asarray(reconstructed_input, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(
            f"Processed input reconstruction shape mismatch for scenario_id={scenario_id!r}: "
            f"{actual.shape} != {expected.shape}"
        )
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
    if not np.allclose(actual, expected, atol=float(atol), rtol=0.0):
        raise ValueError(
            f"Processed input reconstruction mismatch for scenario_id={scenario_id!r}: "
            f"max_abs_diff={max_abs_diff:.8g} exceeds atol={float(atol):.8g}"
        )
    return {
        "scenario_id": scenario_id,
        "max_abs_diff": max_abs_diff,
    }


def prediction_positive_timestamps(
    reference_timestamps: Sequence[float] | np.ndarray,
    *,
    expected_output_channels: int,
    zero_tolerance: float = DEFAULT_ZERO_TIME_TOLERANCE,
) -> np.ndarray:
    timestamps = validate_timestamps(
        reference_timestamps, zero_tolerance=zero_tolerance
    )
    positive = np.asarray(
        timestamps[timestamps > float(zero_tolerance)],
        dtype=np.float64,
    )
    if int(positive.shape[0]) != int(expected_output_channels):
        raise ValueError(
            "Prediction channels must match the positive reference-A timestamps exactly: "
            f"{positive.shape[0]} != {expected_output_channels}"
        )
    return positive


def align_positive_time_series(
    elevation: Any,
    timestamps: Sequence[float] | np.ndarray,
    *,
    common_time_grid: Sequence[float] | np.ndarray,
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE,
) -> np.ndarray:
    values = np.asarray(elevation, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            f"Prediction elevation must have shape [T,H,W], got {values.shape}"
        )
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.shape[0] != ts.shape[0]:
        raise ValueError(
            "Prediction timestamps must match prediction channels exactly: "
            f"{ts.shape[0]} != {values.shape[0]}"
        )
    if ts.size == 0:
        raise ValueError("Prediction timestamps must be non-empty")
    if not np.isfinite(ts).all():
        raise ValueError("Prediction timestamps must be finite")
    if np.any(ts <= 0.0):
        raise ValueError("Prediction timestamps must be strictly positive")
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError("Prediction timestamps must be strictly increasing")

    grid = validate_common_time_grid(common_time_grid)
    lower = float(ts[0])
    upper = float(ts[-1])
    if np.any(grid < lower - float(endpoint_tolerance)) or np.any(
        grid > upper + float(endpoint_tolerance)
    ):
        raise ValueError(
            "Common-time grid extends beyond prediction timestamp support without extrapolation"
        )
    queries = np.asarray(grid, dtype=np.float64).copy()
    queries[np.abs(queries - lower) <= float(endpoint_tolerance)] = lower
    queries[np.abs(queries - upper) <= float(endpoint_tolerance)] = upper

    if ts.shape[0] == 1:
        if not np.allclose(queries, ts[0], atol=float(endpoint_tolerance), rtol=0.0):
            raise ValueError(
                "A single prediction timestamp cannot support interpolation"
            )
        return np.repeat(values, queries.shape[0], axis=0)

    right = np.searchsorted(ts, queries, side="left")
    right = np.clip(right, 1, ts.shape[0] - 1)
    left = right - 1
    left_times = ts[left]
    right_times = ts[right]
    denom = right_times - left_times
    if np.any(denom <= 0.0):
        raise ValueError("Prediction timestamps must be strictly increasing")
    weights = ((queries - left_times) / denom).astype(np.float64)
    return np.asarray(
        values[left] * (1.0 - weights[:, None, None])
        + values[right] * weights[:, None, None],
        dtype=np.float64,
    )


def verify_common_raw_identity(
    *,
    scenario_id: str,
    left_sample: Mapping[str, Any],
    right_sample: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for field_name in ("bathymetry", "source_field", "initial_depth"):
        left = np.asarray(left_sample[field_name], dtype=np.float32)
        right = np.asarray(right_sample[field_name], dtype=np.float32)
        if left.shape != right.shape:
            raise ValueError(
                f"Common raw field {field_name!r} shape mismatch for scenario_id={scenario_id!r}: "
                f"{left.shape} != {right.shape}"
            )
        if not np.array_equal(left, right):
            diff = np.asarray(left, dtype=np.float64) - np.asarray(
                right, dtype=np.float64
            )
            raise ValueError(
                f"Common raw field {field_name!r} mismatch for scenario_id={scenario_id!r}: "
                f"max_abs_diff={float(np.max(np.abs(diff))):.8g}"
            )
        hashes[field_name] = stable_hash_payload(
            {
                "dtype": str(left.dtype),
                "shape": list(map(int, left.shape)),
                "bytes_sha256": hashlib.sha256(
                    np.ascontiguousarray(left).view(np.uint8).tobytes()
                ).hexdigest(),
            }
        )
    return hashes


def _bootstrap_bounds(
    resampled: np.ndarray,
    *,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    lower_q = (1.0 - float(confidence_level)) * 50.0
    upper_q = 100.0 - lower_q
    return (
        np.percentile(resampled, lower_q, axis=0),
        np.percentile(resampled, upper_q, axis=0),
    )


def _summarize_scalar_bootstrap(
    *,
    point_estimate: float,
    resampled: np.ndarray,
    confidence_level: float,
) -> dict[str, Any]:
    finite = np.asarray(resampled, dtype=np.float64)
    finite_mask = np.isfinite(finite)
    lower = None
    upper = None
    if np.any(finite_mask):
        lower, upper = _bootstrap_bounds(
            finite[finite_mask],
            confidence_level=confidence_level,
        )
        lower = float(np.asarray(lower).reshape(-1)[0])
        upper = float(np.asarray(upper).reshape(-1)[0])
    return {
        "point_estimate": float(point_estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "finite_replicate_count": int(np.count_nonzero(finite_mask)),
        "nonfinite_replicate_count": int(
            finite_mask.size - np.count_nonzero(finite_mask)
        ),
    }


def _resampled_mean(values: np.ndarray, bootstrap_indices: np.ndarray) -> np.ndarray:
    sampled = np.asarray(values, dtype=np.float64)[bootstrap_indices]
    return np.mean(sampled, axis=1, dtype=np.float64)


def _resampled_rmse(
    mse_values: np.ndarray, bootstrap_indices: np.ndarray
) -> np.ndarray:
    sampled = np.asarray(mse_values, dtype=np.float64)[bootstrap_indices]
    return np.sqrt(np.mean(sampled, axis=1, dtype=np.float64))


def _per_time_summary(
    *,
    common_time_grid: np.ndarray,
    scenario_time_mse: np.ndarray,
    scenario_time_mae: np.ndarray,
    scenario_time_max: np.ndarray,
    bootstrap_indices: np.ndarray,
    confidence_level: float,
) -> list[dict[str, Any]]:
    point_rmse = np.sqrt(np.mean(scenario_time_mse, axis=0, dtype=np.float64))
    rmse_resampled = np.sqrt(
        np.mean(scenario_time_mse[bootstrap_indices], axis=1, dtype=np.float64)
    )
    rmse_lower, rmse_upper = _bootstrap_bounds(
        rmse_resampled,
        confidence_level=confidence_level,
    )
    point_mae = np.mean(scenario_time_mae, axis=0, dtype=np.float64)
    mae_resampled = np.mean(
        scenario_time_mae[bootstrap_indices], axis=1, dtype=np.float64
    )
    mae_lower, mae_upper = _bootstrap_bounds(
        mae_resampled,
        confidence_level=confidence_level,
    )
    point_max = np.mean(scenario_time_max, axis=0, dtype=np.float64)
    max_resampled = np.mean(
        scenario_time_max[bootstrap_indices], axis=1, dtype=np.float64
    )
    max_lower, max_upper = _bootstrap_bounds(
        max_resampled,
        confidence_level=confidence_level,
    )

    rows: list[dict[str, Any]] = []
    for index, time_value in enumerate(common_time_grid):
        rows.append(
            {
                "time": float(time_value),
                "field_rmse": float(point_rmse[index]),
                "field_rmse_ci_lower": float(rmse_lower[index]),
                "field_rmse_ci_upper": float(rmse_upper[index]),
                "scenario_mae_mean": float(point_mae[index]),
                "scenario_mae_mean_ci_lower": float(mae_lower[index]),
                "scenario_mae_mean_ci_upper": float(mae_upper[index]),
                "scenario_max_error_mean": float(point_max[index]),
                "scenario_max_error_mean_ci_lower": float(max_lower[index]),
                "scenario_max_error_mean_ci_upper": float(max_upper[index]),
            }
        )
    return rows


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[Any, list[Mapping[str, Any]]]]:
    groups: dict[str, dict[Any, list[Mapping[str, Any]]]] = {
        "bathymetry": defaultdict(list),
        "source": defaultdict(list),
        "joint": defaultdict(list),
    }
    for row in rows:
        bathymetry_type = str(row["bathymetry_type"])
        source_type = str(row["source_type"])
        groups["bathymetry"][bathymetry_type].append(row)
        groups["source"][source_type].append(row)
        groups["joint"][(bathymetry_type, source_type)].append(row)
    return groups


def _summarize_group_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str],
    mse_key: str,
) -> dict[str, list[dict[str, Any]]]:
    groups = _group_rows(rows)

    def _make_rows(group_name: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key in sorted(groups[group_name]):
            bucket = groups[group_name][key]
            summary: dict[str, Any] = {
                "scenario_count": int(len(bucket)),
                "scenario_set_hash": stable_hash_scenario_ids(
                    [str(item["scenario_id"]) for item in bucket]
                ),
                "global_field_rmse": float(
                    math.sqrt(
                        np.mean(
                            np.asarray(
                                [float(item[mse_key]) for item in bucket],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
            }
            if group_name == "joint":
                bathymetry_type, source_type = key
                summary["bathymetry_type"] = bathymetry_type
                summary["source_type"] = source_type
            elif group_name == "bathymetry":
                summary["bathymetry_type"] = str(key)
            else:
                summary["source_type"] = str(key)

            for metric_key in metric_keys:
                values = np.asarray(
                    [float(item[metric_key]) for item in bucket], dtype=np.float64
                )
                summary[f"{metric_key}_mean"] = float(np.mean(values))
                summary[f"{metric_key}_median"] = float(np.median(values))
            output.append(summary)
        return output

    return {
        "by_bathymetry": _make_rows("bathymetry"),
        "by_source": _make_rows("source"),
        "by_joint_family": _make_rows("joint"),
    }


def compare_solver_scenarios(
    *,
    contract: SuiteContract,
    solver_a_name: str,
    solver_b_name: str,
    paired_scenarios: Iterable[Mapping[str, Any]],
    mode: str,
    bootstrap_seed: int,
    num_resamples: int,
    confidence_level: float,
    initial_frame_policy: str | None = None,
    git_commit: str = "unknown",
    script_path: str | None = None,
) -> dict[str, Any]:
    mode_text = require_explicit_mode(mode)
    if mode_text == MODE_SAVED_INDEX_LEGACY and initial_frame_policy not in {
        "include",
        "exclude",
    }:
        raise ValueError(
            "saved-index-legacy mode requires explicit initial_frame_policy='include' or 'exclude'"
        )

    scenario_rows: list[dict[str, Any]] = []
    scenario_time_mse: list[np.ndarray] = []
    scenario_time_mae: list[np.ndarray] = []
    scenario_time_max: list[np.ndarray] = []
    observed_ids: list[str] = []

    rel_ab_key = f"relative_l2_{solver_a_name}_to_{solver_b_name}"
    rel_ba_key = f"relative_l2_{solver_b_name}_to_{solver_a_name}"
    frame_indices: np.ndarray | None = None
    legacy_initial_frame_treatment = None
    expected_legacy_total_frames: int | None = None

    if mode_text == MODE_SAVED_INDEX_LEGACY:
        legacy_initial_frame_treatment = (
            "include_saved_zero_frame"
            if initial_frame_policy == "include"
            else "exclude_saved_zero_frame"
        )

    for paired in paired_scenarios:
        scenario_id = str(paired["scenario_id"])
        observed_ids.append(scenario_id)
        left = _ensure_mapping(paired.get("left"), label="paired.left")
        right = _ensure_mapping(paired.get("right"), label="paired.right")
        left_eta = np.asarray(left["trajectory_eta"], dtype=np.float64)
        right_eta = np.asarray(right["trajectory_eta"], dtype=np.float64)
        left_ts = np.asarray(left["timestamps"], dtype=np.float64)
        right_ts = np.asarray(right["timestamps"], dtype=np.float64)

        if left_eta.ndim != 3 or right_eta.ndim != 3:
            raise ValueError(
                f"trajectory_eta must have shape [T,H,W] for scenario_id={scenario_id!r}"
            )
        if tuple(left_eta.shape[1:]) != tuple(right_eta.shape[1:]):
            raise ValueError(
                f"Spatial shape mismatch for scenario_id={scenario_id!r}: "
                f"{left_eta.shape[1:]} != {right_eta.shape[1:]}"
            )

        if mode_text == MODE_COMMON_TIME:
            aligned_left = align_elevation_series(
                left_eta,
                left_ts,
                mode=MODE_COMMON_TIME,
                common_time_grid=contract.common_time_grid,
                endpoint_tolerance=contract.endpoint_tolerance,
            )
            aligned_right = align_elevation_series(
                right_eta,
                right_ts,
                mode=MODE_COMMON_TIME,
                common_time_grid=contract.common_time_grid,
                endpoint_tolerance=contract.endpoint_tolerance,
            )
            per_time_length = int(contract.common_time_grid.shape[0])
            initial_frame_treatment = contract.initial_frame_treatment
        else:
            validate_timestamps(left_ts)
            validate_timestamps(right_ts)
            if left_eta.shape[0] != right_eta.shape[0]:
                raise ValueError(
                    f"saved-index-legacy mode requires equal frame counts for scenario_id={scenario_id!r}: "
                    f"{left_eta.shape[0]} != {right_eta.shape[0]}"
                )
            if left_ts.shape[0] != right_ts.shape[0]:
                raise ValueError(
                    f"saved-index-legacy mode requires equal timestamp counts for scenario_id={scenario_id!r}: "
                    f"{left_ts.shape[0]} != {right_ts.shape[0]}"
                )
            if expected_legacy_total_frames is None:
                expected_legacy_total_frames = int(left_eta.shape[0])
            elif int(left_eta.shape[0]) != int(expected_legacy_total_frames):
                raise ValueError(
                    "saved-index-legacy mode requires the same saved frame count for every "
                    f"scenario in the comparison suite: {left_eta.shape[0]} != "
                    f"{expected_legacy_total_frames} for scenario_id={scenario_id!r}"
                )
            if frame_indices is None:
                frame_indices = np.arange(left_eta.shape[0], dtype=np.int64)
                if initial_frame_policy == "exclude":
                    frame_indices = frame_indices[1:]
                if frame_indices.size == 0:
                    raise ValueError("saved-index-legacy frame selection is empty")
            if np.max(frame_indices) >= left_eta.shape[0]:
                raise ValueError("saved-index-legacy frame_indices are out of range")
            aligned_left = align_elevation_series(
                left_eta,
                left_ts,
                mode=MODE_SAVED_INDEX_LEGACY,
                frame_indices=frame_indices,
            )
            aligned_right = align_elevation_series(
                right_eta,
                right_ts,
                mode=MODE_SAVED_INDEX_LEGACY,
                frame_indices=frame_indices,
            )
            per_time_length = int(frame_indices.shape[0])
            initial_frame_treatment = str(legacy_initial_frame_treatment)

        if aligned_left.shape != aligned_right.shape:
            raise ValueError(
                f"Aligned shapes differ for scenario_id={scenario_id!r}: "
                f"{aligned_left.shape} != {aligned_right.shape}"
            )
        if aligned_left.shape[0] != per_time_length:
            raise ValueError(
                f"Aligned frame count mismatch for scenario_id={scenario_id!r}: "
                f"{aligned_left.shape[0]} != {per_time_length}"
            )

        metrics_ab = compute_error_metrics(aligned_left, aligned_right)
        metrics_ba = compute_error_metrics(aligned_right, aligned_left)
        diff = np.asarray(aligned_left - aligned_right, dtype=np.float64)
        abs_diff = np.abs(diff)

        row = {
            "scenario_id": scenario_id,
            "bathymetry_type": str(paired["bathymetry_type"]),
            "source_type": str(paired["source_type"]),
            "source_strength": float(paired["source_strength"]),
            "mae": float(metrics_ab["mae"]),
            "rmse": float(metrics_ab["rmse"]),
            "max_error": float(metrics_ab["max_error"]),
            "mse": float(metrics_ab["mse"]),
            rel_ab_key: float(metrics_ab["relative_l2"]),
            rel_ba_key: float(metrics_ba["relative_l2"]),
        }
        scenario_rows.append(row)
        scenario_time_mse.append(np.mean(diff * diff, axis=(1, 2), dtype=np.float64))
        scenario_time_mae.append(np.mean(abs_diff, axis=(1, 2), dtype=np.float64))
        scenario_time_max.append(np.max(abs_diff, axis=(1, 2)))

    if tuple(observed_ids) != contract.ordered_scenario_ids:
        raise ValueError(
            "Paired scenarios were not streamed in the contract's exact ordered scenario ids"
        )

    bootstrap_indices = generate_paired_bootstrap_indices(
        num_scenarios=len(scenario_rows),
        num_resamples=int(num_resamples),
        seed=int(bootstrap_seed),
    )
    scenario_mse = np.asarray(
        [float(row["mse"]) for row in scenario_rows], dtype=np.float64
    )
    scenario_mae = np.asarray(
        [float(row["mae"]) for row in scenario_rows], dtype=np.float64
    )
    scenario_rmse = np.asarray(
        [float(row["rmse"]) for row in scenario_rows], dtype=np.float64
    )
    scenario_max = np.asarray(
        [float(row["max_error"]) for row in scenario_rows], dtype=np.float64
    )
    scenario_rel_ab = np.asarray(
        [float(row[rel_ab_key]) for row in scenario_rows], dtype=np.float64
    )
    scenario_rel_ba = np.asarray(
        [float(row[rel_ba_key]) for row in scenario_rows], dtype=np.float64
    )

    global_field_rmse = float(math.sqrt(np.mean(scenario_mse)))
    bootstrap_summary = {
        "seed": int(bootstrap_seed),
        "num_resamples": int(num_resamples),
        "confidence_level": float(confidence_level),
        "metrics": {
            "global_field_rmse": _summarize_scalar_bootstrap(
                point_estimate=global_field_rmse,
                resampled=_resampled_rmse(scenario_mse, bootstrap_indices),
                confidence_level=confidence_level,
            ),
            "scenario_mae_mean": _summarize_scalar_bootstrap(
                point_estimate=float(np.mean(scenario_mae)),
                resampled=_resampled_mean(scenario_mae, bootstrap_indices),
                confidence_level=confidence_level,
            ),
            "scenario_rmse_mean": _summarize_scalar_bootstrap(
                point_estimate=float(np.mean(scenario_rmse)),
                resampled=_resampled_mean(scenario_rmse, bootstrap_indices),
                confidence_level=confidence_level,
            ),
            "scenario_max_error_mean": _summarize_scalar_bootstrap(
                point_estimate=float(np.mean(scenario_max)),
                resampled=_resampled_mean(scenario_max, bootstrap_indices),
                confidence_level=confidence_level,
            ),
            rel_ab_key: _summarize_scalar_bootstrap(
                point_estimate=float(np.mean(scenario_rel_ab)),
                resampled=_resampled_mean(scenario_rel_ab, bootstrap_indices),
                confidence_level=confidence_level,
            ),
            rel_ba_key: _summarize_scalar_bootstrap(
                point_estimate=float(np.mean(scenario_rel_ba)),
                resampled=_resampled_mean(scenario_rel_ba, bootstrap_indices),
                confidence_level=confidence_level,
            ),
        },
    }

    time_mse = np.stack(scenario_time_mse, axis=0)
    time_mae = np.stack(scenario_time_mae, axis=0)
    time_max = np.stack(scenario_time_max, axis=0)
    common_time_grid = (
        contract.common_time_grid
        if mode_text == MODE_COMMON_TIME
        else np.asarray(frame_indices, dtype=np.float64)
    )
    per_time_metrics = _per_time_summary(
        common_time_grid=common_time_grid,
        scenario_time_mse=time_mse,
        scenario_time_mae=time_mae,
        scenario_time_max=time_max,
        bootstrap_indices=bootstrap_indices,
        confidence_level=confidence_level,
    )

    summary = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": ALIGNED_SOLVER_COMPARISON_ARTIFACT,
        "output_mode": mode_text,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": script_path,
            "git_commit": str(git_commit),
        },
        "comparison": {
            "solver_a": str(solver_a_name),
            "solver_b": str(solver_b_name),
        },
        "inputs": {
            "audit_hash": contract.audit_hash,
            "audit_artifact_hash": contract.audit_artifact_hash,
            "scenario_selection_hash": contract.scenario_selection_hash,
            "dense_validation_decision_hash": contract.dense_validation_decision_hash,
            "dense_validation_summary_hash": contract.dense_validation_summary_hash,
        },
        "scenario_order": {
            "suite": contract.suite_name,
            "suite_label": contract.suite_label,
            "ordered_scenario_ids": list(contract.ordered_scenario_ids),
            "ordered_scenario_hash": contract.ordered_scenario_hash,
        },
        "alignment": {
            "schema_id": SCHEMA_ID,
            "mode": mode_text,
            "field": contract.field,
            "elevation_semantics": contract.elevation_semantics,
            "time_semantics": contract.time_semantics,
            "initial_frame_treatment": str(initial_frame_treatment),
            "aggregation": COMMON_TIME_AGGREGATION,
            "common_time_grid": (
                contract.common_time_grid.tolist()
                if mode_text == MODE_COMMON_TIME
                else None
            ),
            "common_time_horizon": (
                float(contract.common_time_grid[-1])
                if mode_text == MODE_COMMON_TIME
                else None
            ),
            "frame_indices": (
                None
                if frame_indices is None
                else [int(value) for value in frame_indices.tolist()]
            ),
            "per_time_axis": (
                "common_time" if mode_text == MODE_COMMON_TIME else "saved_frame_index"
            ),
            "common_time_evaluator_compatible": mode_text == MODE_COMMON_TIME,
        },
        "units": dict(UNITS),
        "counts": {
            "scenario_count": int(len(scenario_rows)),
        },
        "aggregate_metrics": {
            "global_field_rmse": global_field_rmse,
            "scenario_mae_mean": float(np.mean(scenario_mae)),
            "scenario_rmse_mean": float(np.mean(scenario_rmse)),
            "scenario_max_error_mean": float(np.mean(scenario_max)),
            rel_ab_key: float(np.mean(scenario_rel_ab)),
            rel_ba_key: float(np.mean(scenario_rel_ba)),
        },
        "bootstrap": bootstrap_summary,
        "family_summaries": _summarize_group_metrics(
            scenario_rows,
            metric_keys=("mae", "rmse", "max_error", rel_ab_key, rel_ba_key),
            mse_key="mse",
        ),
        "per_time_metrics": per_time_metrics,
        "scenario_metrics": scenario_rows,
    }
    return summary


def _safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return 0.0 if float(numerator) == 0.0 else float("inf")
    return float(numerator) / float(denominator)


def classify_benchmark_specific_superiority(
    *,
    rho_point: float,
    rho_ci_upper: float | None,
    denominator_point: float,
) -> str:
    if float(denominator_point) == 0.0:
        return "invalid_zero_denominator"
    if rho_ci_upper is None or not math.isfinite(float(rho_ci_upper)):
        return "invalid_nonfinite_bootstrap"
    if float(rho_ci_upper) < 1.0:
        return "supported_benchmark_specific_superiority"
    if float(rho_point) < 1.0:
        return "inconclusive_benchmark_specific_superiority"
    return "negative_result"


def build_emulator_superiority_metric_row(
    *,
    scenario_id: str,
    bathymetry_type: str,
    source_type: str,
    source_strength: float,
    pred_aligned: Any,
    ref_a_aligned: Any,
    ref_b_aligned: Any,
) -> dict[str, Any]:
    pred_values = np.asarray(pred_aligned, dtype=np.float64)
    ref_a_values = np.asarray(ref_a_aligned, dtype=np.float64)
    ref_b_values = np.asarray(ref_b_aligned, dtype=np.float64)
    if (
        pred_values.shape != ref_a_values.shape
        or pred_values.shape != ref_b_values.shape
    ):
        raise ValueError(
            f"Aligned tensors must share shape for scenario_id={scenario_id!r}: "
            f"{pred_values.shape}, {ref_a_values.shape}, {ref_b_values.shape}"
        )
    numerator = compute_error_metrics(pred_values, ref_b_values)
    denominator = compute_error_metrics(ref_a_values, ref_b_values)
    control = compute_error_metrics(pred_values, ref_a_values)
    return {
        "scenario_id": str(scenario_id),
        "bathymetry_type": str(bathymetry_type),
        "source_type": str(source_type),
        "source_strength": float(source_strength),
        "numerator_mse": float(numerator["mse"]),
        "numerator_rmse": float(numerator["rmse"]),
        "numerator_mae": float(numerator["mae"]),
        "numerator_max_error": float(numerator["max_error"]),
        "denominator_mse": float(denominator["mse"]),
        "denominator_rmse": float(denominator["rmse"]),
        "denominator_mae": float(denominator["mae"]),
        "denominator_max_error": float(denominator["max_error"]),
        "same_reference_control_mse": float(control["mse"]),
        "same_reference_control_rmse": float(control["rmse"]),
        "same_reference_control_mae": float(control["mae"]),
        "same_reference_control_max_error": float(control["max_error"]),
        "rho": _safe_ratio(float(numerator["rmse"]), float(denominator["rmse"])),
        "same_reference_control_ratio": _safe_ratio(
            float(control["rmse"]),
            float(denominator["rmse"]),
        ),
    }


def evaluate_emulator_superiority_metric_rows(
    *,
    contract: SuiteContract,
    direction_name: str,
    model_solver_name: str,
    benchmark_solver_name: str,
    scenario_metric_rows: Iterable[Mapping[str, Any]],
    bootstrap_seed: int,
    num_resamples: int,
    confidence_level: float,
    git_commit: str = "unknown",
    script_path: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    required_scalar_keys = (
        "numerator_mse",
        "numerator_rmse",
        "numerator_mae",
        "numerator_max_error",
        "denominator_mse",
        "denominator_rmse",
        "denominator_mae",
        "denominator_max_error",
        "same_reference_control_mse",
        "same_reference_control_rmse",
        "same_reference_control_mae",
        "same_reference_control_max_error",
    )
    for scenario in scenario_metric_rows:
        row = dict(scenario)
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id:
            raise ValueError(
                "scenario_metric_rows require non-empty scenario_id values"
            )
        observed_ids.append(scenario_id)
        normalized_row = {
            "scenario_id": scenario_id,
            "bathymetry_type": str(row["bathymetry_type"]),
            "source_type": str(row["source_type"]),
            "source_strength": float(row["source_strength"]),
        }
        for key in required_scalar_keys:
            normalized_row[key] = float(row[key])
        normalized_row["rho"] = float(
            row.get(
                "rho",
                _safe_ratio(
                    normalized_row["numerator_rmse"],
                    normalized_row["denominator_rmse"],
                ),
            )
        )
        normalized_row["same_reference_control_ratio"] = float(
            row.get(
                "same_reference_control_ratio",
                _safe_ratio(
                    normalized_row["same_reference_control_rmse"],
                    normalized_row["denominator_rmse"],
                ),
            )
        )
        rows.append(normalized_row)

    if tuple(observed_ids) != contract.ordered_scenario_ids:
        raise ValueError(
            "Emulator-superiority scenarios were not evaluated in the contract's exact ordered scenario ids"
        )

    bootstrap_indices = generate_paired_bootstrap_indices(
        num_scenarios=len(rows),
        num_resamples=int(num_resamples),
        seed=int(bootstrap_seed),
    )
    numerator_mse = np.asarray(
        [float(row["numerator_mse"]) for row in rows], dtype=np.float64
    )
    denominator_mse = np.asarray(
        [float(row["denominator_mse"]) for row in rows], dtype=np.float64
    )
    control_mse = np.asarray(
        [float(row["same_reference_control_mse"]) for row in rows],
        dtype=np.float64,
    )

    numerator_point = float(math.sqrt(np.mean(numerator_mse)))
    denominator_point = float(math.sqrt(np.mean(denominator_mse)))
    control_point = float(math.sqrt(np.mean(control_mse)))
    rho_point = _safe_ratio(numerator_point, denominator_point)
    control_ratio_point = _safe_ratio(control_point, denominator_point)

    numerator_resampled = _resampled_rmse(numerator_mse, bootstrap_indices)
    denominator_resampled = _resampled_rmse(denominator_mse, bootstrap_indices)
    control_resampled = _resampled_rmse(control_mse, bootstrap_indices)
    rho_resampled = np.asarray(
        [
            _safe_ratio(float(num_value), float(den_value))
            for num_value, den_value in zip(numerator_resampled, denominator_resampled)
        ],
        dtype=np.float64,
    )
    control_ratio_resampled = np.asarray(
        [
            _safe_ratio(float(ctrl_value), float(den_value))
            for ctrl_value, den_value in zip(control_resampled, denominator_resampled)
        ],
        dtype=np.float64,
    )

    rho_summary = _summarize_scalar_bootstrap(
        point_estimate=rho_point,
        resampled=rho_resampled,
        confidence_level=confidence_level,
    )
    classification = classify_benchmark_specific_superiority(
        rho_point=float(rho_point),
        rho_ci_upper=rho_summary["ci_upper"],
        denominator_point=float(denominator_point),
    )

    summary = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": EMULATOR_SUPERIORITY_ARTIFACT,
        "output_mode": MODE_COMMON_TIME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": script_path,
            "git_commit": str(git_commit),
        },
        "direction": {
            "name": str(direction_name),
            "model_solver": str(model_solver_name),
            "benchmark_solver": str(benchmark_solver_name),
        },
        "inputs": {
            "audit_hash": contract.audit_hash,
            "audit_artifact_hash": contract.audit_artifact_hash,
            "scenario_selection_hash": contract.scenario_selection_hash,
            "dense_validation_decision_hash": contract.dense_validation_decision_hash,
            "dense_validation_summary_hash": contract.dense_validation_summary_hash,
        },
        "scenario_order": {
            "suite": contract.suite_name,
            "suite_label": contract.suite_label,
            "ordered_scenario_ids": list(contract.ordered_scenario_ids),
            "ordered_scenario_hash": contract.ordered_scenario_hash,
        },
        "alignment": {
            "schema_id": SCHEMA_ID,
            "mode": MODE_COMMON_TIME,
            "field": contract.field,
            "elevation_semantics": contract.elevation_semantics,
            "time_semantics": contract.time_semantics,
            "initial_frame_treatment": contract.initial_frame_treatment,
            "aggregation": COMMON_TIME_AGGREGATION,
            "common_time_grid": contract.common_time_grid.tolist(),
            "common_time_horizon": float(contract.common_time_grid[-1]),
        },
        "units": dict(UNITS),
        "aggregation": {
            "primary_global_metric": "equal_scenario_weight_field_rmse_from_scenario_mse",
            "field_mask": "full_field",
            "scenario_weighting": "equal_scenario_weight",
        },
        "counts": {
            "scenario_count": int(len(rows)),
        },
        "metrics": {
            "numerator_global_field_rmse": numerator_point,
            "denominator_global_field_rmse": denominator_point,
            "same_reference_control_global_field_rmse": control_point,
            "rho": rho_point,
            "same_reference_control_ratio": control_ratio_point,
        },
        "bootstrap": {
            "seed": int(bootstrap_seed),
            "num_resamples": int(num_resamples),
            "confidence_level": float(confidence_level),
            "metrics": {
                "numerator_global_field_rmse": _summarize_scalar_bootstrap(
                    point_estimate=numerator_point,
                    resampled=numerator_resampled,
                    confidence_level=confidence_level,
                ),
                "denominator_global_field_rmse": _summarize_scalar_bootstrap(
                    point_estimate=denominator_point,
                    resampled=denominator_resampled,
                    confidence_level=confidence_level,
                ),
                "same_reference_control_global_field_rmse": _summarize_scalar_bootstrap(
                    point_estimate=control_point,
                    resampled=control_resampled,
                    confidence_level=confidence_level,
                ),
                "rho": rho_summary,
                "same_reference_control_ratio": _summarize_scalar_bootstrap(
                    point_estimate=control_ratio_point,
                    resampled=control_ratio_resampled,
                    confidence_level=confidence_level,
                ),
            },
            "rho_zero_denominator_replicate_count": int(
                np.count_nonzero(denominator_resampled == 0.0)
            ),
        },
        "benchmark_specific_superiority": {
            "classification": classification,
            "wording": (
                "rho compares the emulator's common-time field error against the benchmark "
                "reference B to the reference-A vs benchmark-B common-time disagreement."
            ),
        },
        "family_summaries": {
            "by_bathymetry": [],
            "by_source": [],
            "by_joint_family": [],
        },
        "scenario_metrics": rows,
    }

    family_rows = _group_rows(rows)
    for kind, key_name in (
        ("bathymetry", "by_bathymetry"),
        ("source", "by_source"),
        ("joint", "by_joint_family"),
    ):
        output_rows: list[dict[str, Any]] = []
        for key in sorted(family_rows[kind]):
            bucket = family_rows[kind][key]
            numerator_family = float(
                math.sqrt(
                    np.mean(
                        np.asarray(
                            [row["numerator_mse"] for row in bucket], dtype=np.float64
                        )
                    )
                )
            )
            denominator_family = float(
                math.sqrt(
                    np.mean(
                        np.asarray(
                            [row["denominator_mse"] for row in bucket], dtype=np.float64
                        )
                    )
                )
            )
            control_family = float(
                math.sqrt(
                    np.mean(
                        np.asarray(
                            [row["same_reference_control_mse"] for row in bucket],
                            dtype=np.float64,
                        )
                    )
                )
            )
            family_summary: dict[str, Any] = {
                "scenario_count": int(len(bucket)),
                "scenario_set_hash": stable_hash_scenario_ids(
                    [str(item["scenario_id"]) for item in bucket]
                ),
                "numerator_global_field_rmse": numerator_family,
                "denominator_global_field_rmse": denominator_family,
                "same_reference_control_global_field_rmse": control_family,
                "rho": _safe_ratio(numerator_family, denominator_family),
                "same_reference_control_ratio": _safe_ratio(
                    control_family, denominator_family
                ),
            }
            if kind == "joint":
                bathymetry_type, source_type = key
                family_summary["bathymetry_type"] = bathymetry_type
                family_summary["source_type"] = source_type
            elif kind == "bathymetry":
                family_summary["bathymetry_type"] = str(key)
            else:
                family_summary["source_type"] = str(key)
            output_rows.append(family_summary)
        summary["family_summaries"][key_name] = output_rows

    return summary


def evaluate_emulator_superiority_scenarios(
    *,
    contract: SuiteContract,
    direction_name: str,
    model_solver_name: str,
    benchmark_solver_name: str,
    scenario_rows: Iterable[Mapping[str, Any]],
    bootstrap_seed: int,
    num_resamples: int,
    confidence_level: float,
    git_commit: str = "unknown",
    script_path: str | None = None,
) -> dict[str, Any]:
    metric_rows = (
        build_emulator_superiority_metric_row(
            scenario_id=str(scenario["scenario_id"]),
            bathymetry_type=str(scenario["bathymetry_type"]),
            source_type=str(scenario["source_type"]),
            source_strength=float(scenario["source_strength"]),
            pred_aligned=scenario["pred_aligned"],
            ref_a_aligned=scenario["ref_a_aligned"],
            ref_b_aligned=scenario["ref_b_aligned"],
        )
        for scenario in scenario_rows
    )
    return evaluate_emulator_superiority_metric_rows(
        contract=contract,
        direction_name=direction_name,
        model_solver_name=model_solver_name,
        benchmark_solver_name=benchmark_solver_name,
        scenario_metric_rows=metric_rows,
        bootstrap_seed=bootstrap_seed,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        git_commit=git_commit,
        script_path=script_path,
    )


def validate_common_time_solver_comparison_artifact(
    artifact: Mapping[str, Any],
    *,
    contract: SuiteContract,
) -> None:
    if str(artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Solver comparison schema_id must be {SCHEMA_ID!r}")
    if str(artifact.get("artifact_kind", "")) != ALIGNED_SOLVER_COMPARISON_ARTIFACT:
        raise ValueError(
            f"Expected artifact_kind={ALIGNED_SOLVER_COMPARISON_ARTIFACT!r}"
        )
    if str(artifact.get("output_mode", "")) != MODE_COMMON_TIME:
        raise ValueError(
            "Legacy saved-index solver comparison artifacts cannot be consumed by "
            "common-time emulator-superiority evaluation."
        )
    scenario_order = _ensure_mapping(
        artifact.get("scenario_order"), label="artifact.scenario_order"
    )
    if (
        tuple(str(value) for value in scenario_order.get("ordered_scenario_ids", []))
        != contract.ordered_scenario_ids
    ):
        raise ValueError(
            "Solver comparison artifact scenario ids do not match the current suite"
        )
    if (
        str(scenario_order.get("ordered_scenario_hash", ""))
        != contract.ordered_scenario_hash
    ):
        raise ValueError(
            "Solver comparison artifact scenario hash does not match the current suite"
        )
    inputs = _ensure_mapping(artifact.get("inputs"), label="artifact.inputs")
    if str(inputs.get("audit_hash", "")) != contract.audit_hash:
        raise ValueError(
            "Solver comparison artifact audit_hash does not match the current audit"
        )
    alignment = _ensure_mapping(artifact.get("alignment"), label="artifact.alignment")
    artifact_grid = validate_common_time_grid(alignment.get("common_time_grid"))
    if not np.array_equal(artifact_grid, contract.common_time_grid):
        raise ValueError(
            "Solver comparison artifact common_time_grid does not match the current suite"
        )
