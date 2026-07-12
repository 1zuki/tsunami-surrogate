from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_ID = "tsunami-surrogate.alignment.v1"
MODE_COMMON_TIME = "common-time"
MODE_SAVED_INDEX_LEGACY = "saved-index-legacy"
VALID_ALIGNMENT_MODES = (MODE_COMMON_TIME, MODE_SAVED_INDEX_LEGACY)
DEFAULT_COMMON_TIME_GRID = np.asarray(
    [0.004 * float(index) for index in range(1, 21)],
    dtype=np.float64,
)
DEFAULT_ENDPOINT_TOLERANCE = 1.0e-6
DEFAULT_ZERO_TIME_TOLERANCE = 1.0e-7
DEFAULT_RELATIVE_L2_EPS = 1.0e-12
DEFAULT_COMPATIBILITY_KEYS = (
    "schema_id",
    "mode",
    "ordered_scenario_ids",
    "common_time_grid",
    "common_time_horizon",
    "field",
    "elevation_semantics",
    "time_semantics",
    "initial_frame_treatment",
    "aggregation",
)


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


def stable_hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_hash_scenario_ids(ordered_scenario_ids: Sequence[str]) -> str:
    return stable_hash_payload(
        {
            "schema_id": SCHEMA_ID,
            "artifact_kind": "ordered-scenario-id-list",
            "ordered_scenario_ids": [str(value) for value in ordered_scenario_ids],
        }
    )


def build_common_time_grid(
    start: float,
    stop: float,
    step: float,
) -> np.ndarray:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step):
        raise ValueError("Common-time grid bounds must be finite")
    if start <= 0.0 or stop <= 0.0 or step <= 0.0:
        raise ValueError("Common-time grid requires strictly positive start/stop/step")
    if stop < start:
        raise ValueError("Common-time grid stop must be >= start")

    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Common-time grid is empty")
    if not np.isclose(values[-1], stop, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            "Common-time grid stop is not an integer number of steps from start"
        )
    values[-1] = stop
    return validate_common_time_grid(values)


def validate_alignment_mode(mode: str) -> str:
    mode_text = str(mode).strip()
    if mode_text not in VALID_ALIGNMENT_MODES:
        raise ValueError(
            f"Unsupported alignment mode {mode_text!r}. "
            f"Expected one of {VALID_ALIGNMENT_MODES}."
        )
    return mode_text


def validate_common_time_grid(common_time_grid: Any) -> np.ndarray:
    grid = np.asarray(common_time_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("Common-time grid must be a non-empty 1-D array")
    grid = grid.reshape(-1)
    if grid.size == 0:
        raise ValueError("Common-time grid must be a non-empty 1-D array")
    if not np.isfinite(grid).all():
        raise ValueError("Common-time grid must be finite")
    if np.any(grid <= 0.0):
        raise ValueError("Common-time grid must be strictly positive")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("Common-time grid must be strictly increasing")
    return grid


def validate_timestamps(
    timestamps: Any,
    *,
    zero_tolerance: float = DEFAULT_ZERO_TIME_TOLERANCE,
) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Timestamps must be a non-empty 1-D array")
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("Timestamps must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("Timestamps must be finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Timestamps must be strictly increasing")
    if abs(float(values[0])) > float(zero_tolerance):
        raise ValueError(
            f"Timestamps must start at zero within tolerance {zero_tolerance}"
        )
    return values


def validate_elevation_series(
    elevation: Any,
    timestamps: Any,
    *,
    zero_tolerance: float = DEFAULT_ZERO_TIME_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(elevation, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Elevation must have shape [T,H,W], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Elevation values must be finite")

    ts = validate_timestamps(timestamps, zero_tolerance=zero_tolerance)
    if int(values.shape[0]) != int(ts.shape[0]):
        raise ValueError(
            "Elevation time dimension and timestamp length must match: "
            f"{values.shape[0]} != {ts.shape[0]}"
        )
    return values, ts


def _snap_common_time_queries(
    common_time_grid: np.ndarray,
    timestamps: np.ndarray,
    *,
    endpoint_tolerance: float,
) -> np.ndarray:
    queries = np.asarray(common_time_grid, dtype=np.float64).copy()
    lower = float(timestamps[0])
    upper = float(timestamps[-1])
    if np.any(queries < lower - endpoint_tolerance) or np.any(
        queries > upper + endpoint_tolerance
    ):
        raise ValueError(
            "Common-time grid extends beyond timestamp support without extrapolation"
        )

    queries[np.abs(queries - lower) <= endpoint_tolerance] = lower
    queries[np.abs(queries - upper) <= endpoint_tolerance] = upper
    return queries


def align_elevation_series(
    elevation: Any,
    timestamps: Any,
    *,
    mode: str,
    common_time_grid: Sequence[float] | np.ndarray | None = None,
    frame_indices: Sequence[int] | np.ndarray | None = None,
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE,
    zero_tolerance: float = DEFAULT_ZERO_TIME_TOLERANCE,
) -> np.ndarray:
    mode_text = validate_alignment_mode(mode)
    values, ts = validate_elevation_series(
        elevation,
        timestamps,
        zero_tolerance=zero_tolerance,
    )

    if mode_text == MODE_COMMON_TIME:
        if common_time_grid is None:
            raise ValueError(
                "Explicit common_time_grid is required for mode='common-time'"
            )
        grid = validate_common_time_grid(common_time_grid)
        queries = _snap_common_time_queries(
            grid,
            ts,
            endpoint_tolerance=float(endpoint_tolerance),
        )

        right = np.searchsorted(ts, queries, side="left")
        right = np.clip(right, 1, ts.shape[0] - 1)
        left = right - 1
        left_times = ts[left]
        right_times = ts[right]
        denom = right_times - left_times
        if np.any(denom <= 0.0):
            raise ValueError("Timestamps must be strictly increasing for interpolation")

        weights = ((queries - left_times) / denom).astype(np.float64)
        aligned = (
            values[left] * (1.0 - weights[:, None, None])
            + values[right] * weights[:, None, None]
        )
        return np.asarray(aligned, dtype=np.float64)

    if frame_indices is None:
        raise ValueError(
            "Explicit frame_indices are required for mode='saved-index-legacy'"
        )
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("frame_indices must be a non-empty 1-D array")
    indices = indices.reshape(-1)
    if indices.size == 0:
        raise ValueError("frame_indices must be a non-empty 1-D array")
    if np.any(indices < 0) or np.any(indices >= values.shape[0]):
        raise ValueError("frame_indices contain out-of-range values")
    return np.asarray(values[indices], dtype=np.float64)


def compute_error_metrics(
    candidate: Any,
    reference: Any,
    *,
    relative_l2_eps: float = DEFAULT_RELATIVE_L2_EPS,
) -> dict[str, float]:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if candidate_values.shape != reference_values.shape:
        raise ValueError(
            "Candidate and reference shapes must match for metric computation: "
            f"{candidate_values.shape} != {reference_values.shape}"
        )
    if (
        not np.isfinite(candidate_values).all()
        or not np.isfinite(reference_values).all()
    ):
        raise ValueError("Metrics require finite candidate and reference values")

    diff = candidate_values - reference_values
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))
    rel_l2 = float(
        np.linalg.norm(diff.reshape(-1), ord=2)
        / max(np.linalg.norm(reference_values.reshape(-1), ord=2), relative_l2_eps)
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "relative_l2": rel_l2,
        "max_error": max_abs,
        "mse": mse,
    }


def compute_equal_scenario_global_rmse(
    scenario_metrics: Sequence[Mapping[str, Any]],
) -> float:
    if not scenario_metrics:
        raise ValueError("At least one scenario metric row is required")
    mse_values = []
    for row in scenario_metrics:
        if "mse" in row:
            mse = float(row["mse"])
        elif "rmse" in row:
            rmse = float(row["rmse"])
            mse = rmse * rmse
        else:
            raise KeyError("Scenario metric rows require 'mse' or 'rmse'")
        if not math.isfinite(mse) or mse < 0.0:
            raise ValueError(f"Invalid scenario mean-squared error {mse}")
        mse_values.append(mse)
    return float(np.sqrt(np.mean(np.asarray(mse_values, dtype=np.float64))))


def generate_paired_bootstrap_indices(
    *,
    num_scenarios: int,
    num_resamples: int,
    seed: int,
) -> np.ndarray:
    if int(num_scenarios) <= 0:
        raise ValueError("num_scenarios must be positive")
    if int(num_resamples) <= 0:
        raise ValueError("num_resamples must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0,
        int(num_scenarios),
        size=(int(num_resamples), int(num_scenarios)),
        endpoint=False,
        dtype=np.int64,
    )


def summarize_paired_bootstrap(
    values_by_name: Mapping[str, Sequence[float] | np.ndarray],
    *,
    bootstrap_indices: np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if not values_by_name:
        raise ValueError("values_by_name must not be empty")

    indices = np.asarray(bootstrap_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.size == 0:
        raise ValueError("bootstrap_indices must have shape [R,N]")

    num_scenarios = int(indices.shape[1])
    lower_q = (1.0 - float(confidence_level)) * 50.0
    upper_q = 100.0 - lower_q

    summaries: dict[str, Any] = {
        "num_scenarios": num_scenarios,
        "num_resamples": int(indices.shape[0]),
        "confidence_level": float(confidence_level),
        "metrics": {},
    }
    for name, raw_values in values_by_name.items():
        values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        if int(values.shape[0]) != num_scenarios:
            raise ValueError(
                f"Bootstrap values for {name!r} have length {values.shape[0]}, "
                f"expected {num_scenarios}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Bootstrap values for {name!r} must be finite")

        sampled = values[indices]
        resampled_means = np.mean(sampled, axis=1, dtype=np.float64)
        summaries["metrics"][str(name)] = {
            "point_estimate": float(np.mean(values, dtype=np.float64)),
            "bootstrap_mean": float(np.mean(resampled_means, dtype=np.float64)),
            "bootstrap_std": float(np.std(resampled_means, dtype=np.float64)),
            "ci_lower": float(np.percentile(resampled_means, lower_q)),
            "ci_upper": float(np.percentile(resampled_means, upper_q)),
        }
    return summaries


def summarize_metrics_by_family(
    scenario_metrics: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = ("mae", "rmse", "relative_l2", "max_error"),
    bathymetry_key: str = "bathymetry_type",
    source_key: str = "source_type",
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, defaultdict[Any, list[Mapping[str, Any]]]] = {
        "bathymetry": defaultdict(list),
        "source": defaultdict(list),
        "joint": defaultdict(list),
    }

    for row in scenario_metrics:
        bathymetry_type = str(row[bathymetry_key])
        source_type = str(row[source_key])
        buckets["bathymetry"][bathymetry_type].append(row)
        buckets["source"][source_type].append(row)
        buckets["joint"][(bathymetry_type, source_type)].append(row)

    def _summaries(kind: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(buckets[kind]):
            group_rows = buckets[kind][key]
            summary: dict[str, Any] = {
                "scenario_count": int(len(group_rows)),
                "scenario_set_hash": stable_hash_scenario_ids(
                    [str(item["scenario_id"]) for item in group_rows]
                ),
            }
            if kind == "joint":
                bathymetry_type, source_type = key
                summary["bathymetry_type"] = bathymetry_type
                summary["source_type"] = source_type
            elif kind == "bathymetry":
                summary["bathymetry_type"] = str(key)
            else:
                summary["source_type"] = str(key)

            for metric_name in metric_names:
                values = np.asarray(
                    [float(item[metric_name]) for item in group_rows],
                    dtype=np.float64,
                )
                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_median"] = float(np.median(values))
            rows.append(summary)
        return rows

    return {
        "by_bathymetry": _summaries("bathymetry"),
        "by_source": _summaries("source"),
        "by_joint_family": _summaries("joint"),
    }


def validate_alignment_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    schema_id = str(payload.get("schema_id", ""))
    if schema_id != SCHEMA_ID:
        raise ValueError(
            f"Alignment metadata schema_id must be {SCHEMA_ID!r}, got {schema_id!r}"
        )
    mode = validate_alignment_mode(str(payload.get("mode", "")))

    ordered_scenario_ids = payload.get("ordered_scenario_ids")
    if not isinstance(ordered_scenario_ids, list) or not ordered_scenario_ids:
        raise ValueError("Alignment metadata requires non-empty ordered_scenario_ids")
    if any(not str(value).strip() for value in ordered_scenario_ids):
        raise ValueError("ordered_scenario_ids must not contain empty values")

    if mode == MODE_COMMON_TIME:
        if "common_time_grid" not in payload:
            raise ValueError("Alignment metadata requires common_time_grid")
        grid = validate_common_time_grid(payload["common_time_grid"])
        payload["common_time_grid"] = grid.tolist()
        payload["common_time_horizon"] = float(grid[-1])
    else:
        if "frame_indices" not in payload:
            raise ValueError("Saved-index legacy metadata requires frame_indices")

    for key in (
        "field",
        "elevation_semantics",
        "time_semantics",
        "initial_frame_treatment",
    ):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"Alignment metadata requires non-empty {key!r}")

    aggregation = payload.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("Alignment metadata requires aggregation mapping")
    payload["mode"] = mode
    payload["ordered_scenario_ids"] = [str(value) for value in ordered_scenario_ids]
    return payload


def validate_alignment_compatibility(
    reference_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    *,
    compatibility_keys: Sequence[str] = DEFAULT_COMPATIBILITY_KEYS,
) -> None:
    left = validate_alignment_metadata(reference_metadata)
    right = validate_alignment_metadata(candidate_metadata)

    for key in compatibility_keys:
        if left.get(key) != right.get(key):
            raise ValueError(
                f"Alignment compatibility check failed for {key!r}: "
                f"{left.get(key)!r} != {right.get(key)!r}"
            )
