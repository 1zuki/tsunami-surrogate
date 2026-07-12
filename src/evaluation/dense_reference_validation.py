from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.data_gen.simulate_dataset import (
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _simulate_one_local,
)
from src.evaluation.alignment import (
    DEFAULT_ENDPOINT_TOLERANCE,
    MODE_COMMON_TIME,
    SCHEMA_ID,
    align_elevation_series,
    compute_error_metrics,
    generate_paired_bootstrap_indices,
    stable_hash_payload,
    stable_hash_scenario_ids,
    summarize_paired_bootstrap,
    validate_common_time_grid,
    validate_elevation_series,
    validate_timestamps,
)
from src.utils.config import load_config
from src.utils.io import get_git_commit, save_json


ROOT = Path(__file__).resolve().parents[2]
SOLVER_ORDER = ("hydrostatic", "muscl_hr", "boussinesq")
DISPLAY_SOLVER_NAMES = {
    "hydrostatic": "Hydrostatic",
    "muscl_hr": "MUSCL-HR",
    "boussinesq": "Boussinesq",
}
FDE_NAME_BY_SOLVER_KEY = {
    "hydrostatic": "swe_hydrostatic",
    "muscl_hr": "swe_muscl_hr",
    "boussinesq": "boussinesq",
}
COMMON_INPUT_FIELDS = (
    "bathymetry",
    "source_field",
    "eta0",
    "initial_depth",
    "rest_depth",
    "free_surface0",
)
SUITE_NAMES = ("smoke", "dense_validation")
REPLAY_CONTROL_FIELDS = (
    "n_steps",
    "save_every",
    "auto_dt",
    "target_cfl",
    "include_initial_state",
)
ALIGNMENT_ELEVATION_SEMANTICS = "sea_level_offset_relative_surface_elevation"
ALIGNMENT_TIME_SEMANTICS = "solver_benchmark_time"
ALIGNMENT_INITIAL_FRAME_TREATMENT = "exclude_zero_from_common_grid"
ALIGNMENT_AGGREGATION = {
    "global_metric": "equal_scenario_weight_field_rmse",
}


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    label: str
    purpose: str
    manuscript_claims_allowed: bool
    ordered_scenarios: tuple[dict[str, Any], ...]
    ordered_scenario_ids: tuple[str, ...]
    list_hash: str


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


def mapping_arg(values: Iterable[str] | None) -> dict[str, str]:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected object JSON in {path}")
    return dict(payload)


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
        f"Could not resolve sample directory for sample_index={sample_index} from {sample_dir_text!r}"
    )


def _candidate_existing_paths(
    path_text: str,
    *,
    repo_roots: Iterable[Path],
) -> Iterable[Path]:
    raw_path = Path(str(path_text))
    yield raw_path

    for repo_root in repo_roots:
        yield repo_root / raw_path

    text = str(path_text)
    for marker in ("/configs/", "/data/", "/results/"):
        if marker not in text:
            continue
        rel = Path(marker.strip("/")) / Path(text.split(marker, 1)[1])
        for repo_root in repo_roots:
            yield repo_root / rel
        yield ROOT / rel


def _resolve_existing_path(
    path_text: str,
    *,
    repo_roots: Iterable[Path],
    label: str,
) -> Path:
    seen: set[str] = set()
    for candidate in _candidate_existing_paths(
        path_text,
        repo_roots=repo_roots,
    ):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {label}: {path_text}")


def _array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    return stable_hash_payload(
        {
            "dtype": str(arr.dtype),
            "shape": list(map(int, arr.shape)),
            "bytes_sha256": hashlib.sha256(arr.view(np.uint8).tobytes()).hexdigest(),
        }
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    num = float(numerator)
    den = float(denominator)
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def aggregate_rmse(mse_values: Sequence[float]) -> float:
    if not mse_values:
        return float("inf")
    arr = np.asarray(mse_values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr)))


def field_rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def resolve_validation_suite(
    selection_artifact: Mapping[str, Any],
    suite_name: str,
) -> SuiteSpec:
    normalized = str(suite_name).strip().lower()
    if normalized not in SUITE_NAMES:
        raise ValueError(
            f"Unsupported suite {suite_name!r}. Expected one of {SUITE_NAMES}."
        )
    suite_payload = _ensure_mapping(
        selection_artifact.get(normalized),
        label=normalized,
    )
    ordered_scenarios = tuple(
        dict(_ensure_mapping(item, label=f"{normalized}.ordered_scenarios[]"))
        for item in suite_payload.get("ordered_scenarios", [])
    )
    ordered_scenario_ids = tuple(
        str(value) for value in suite_payload.get("ordered_scenario_ids", [])
    )
    if len(ordered_scenarios) != len(ordered_scenario_ids):
        raise ValueError(
            f"{normalized} ordered_scenarios length does not match ordered_scenario_ids"
        )
    if (
        tuple(str(item.get("scenario_id", "")) for item in ordered_scenarios)
        != ordered_scenario_ids
    ):
        raise ValueError(
            f"{normalized} ordered_scenario_ids do not match ordered_scenarios"
        )

    default_label = (
        "implementation_only_smoke"
        if normalized == "smoke"
        else "dense_reference_validation"
    )
    purpose = (
        "implementation_only_smoke"
        if normalized == "smoke"
        else "dense_reference_validation"
    )
    return SuiteSpec(
        name=normalized,
        label=str(suite_payload.get("label", default_label)),
        purpose=purpose,
        manuscript_claims_allowed=normalized == "dense_validation",
        ordered_scenarios=ordered_scenarios,
        ordered_scenario_ids=ordered_scenario_ids,
        list_hash=str(
            suite_payload.get(
                "list_hash",
                stable_hash_scenario_ids(list(ordered_scenario_ids)),
            )
        ),
    )


def _load_legacy_sample(sample_dir: Path) -> dict[str, Any]:
    sample_npz = sample_dir / "sample.npz"
    meta_path = sample_dir / "meta.json"
    if not sample_npz.is_file():
        raise FileNotFoundError(sample_npz)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    arrays: dict[str, np.ndarray] = {}
    with np.load(sample_npz) as payload:
        for field in COMMON_INPUT_FIELDS:
            arrays[field] = np.asarray(payload[field], dtype=np.float32)
        arrays["trajectory"] = np.asarray(payload["trajectory"], dtype=np.float32)
        arrays["trajectory_eta"] = np.asarray(
            payload["trajectory_eta"], dtype=np.float32
        )
        arrays["timestamps"] = np.asarray(
            payload["timestamps"], dtype=np.float32
        ).reshape(-1)
        arrays["dt_history"] = np.asarray(
            payload["dt_history"], dtype=np.float32
        ).reshape(-1)
        scenario_id = str(np.asarray(payload["scenario_id"]).reshape(-1)[0])
        solver_name = str(np.asarray(payload["solver_name"]).reshape(-1)[0])

    return {
        "sample_dir": str(sample_dir),
        "sample_npz": sample_npz,
        "meta": _load_json(meta_path),
        "scenario_id": scenario_id,
        "solver_name": solver_name,
        "arrays": arrays,
    }


def _normalize_replay_control(
    raw_control: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    control_cfg = _ensure_mapping(raw_control, label=label)
    missing = [field for field in REPLAY_CONTROL_FIELDS if field not in control_cfg]
    if missing:
        raise KeyError(f"{label} is missing required fields {missing}")
    return {
        "n_steps": int(control_cfg["n_steps"]),
        "save_every": int(control_cfg["save_every"]),
        "auto_dt": bool(control_cfg["auto_dt"]),
        "target_cfl": float(control_cfg["target_cfl"]),
        "include_initial_state": bool(control_cfg["include_initial_state"]),
    }


def _load_dataset_rollout_control(
    meta: Mapping[str, Any],
    *,
    repo_roots: Iterable[Path],
    explicit_replay_control: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config_path_text = str(meta.get("dataset_config_path", "")).strip()
    normalized_explicit = (
        None
        if explicit_replay_control is None
        else _normalize_replay_control(
            explicit_replay_control,
            label="dense_reference_validation.replay_control",
        )
    )
    if normalized_explicit is None:
        raise KeyError(
            "dense_reference_validation.replay_control is required for immutable dense replay provenance"
        )

    control_hash = stable_hash_payload(normalized_explicit)
    base_record = {
        **normalized_explicit,
        "replay_control": dict(normalized_explicit),
        "replay_control_hash": control_hash,
        "hash": control_hash,
        "original_dataset_config_path": config_path_text or None,
    }

    if not config_path_text:
        return {
            **base_record,
            "status": "historical_config_missing_explicit_replay_control",
            "source": "explicit_replay_control_fallback",
            "resolved_dataset_config_path": None,
            "historical_dataset_config_available": False,
            "historical_dataset_config_error": "dataset_config_path missing from legacy sample metadata",
            "historical_recovered_replay_control": None,
            "historical_recovered_replay_control_hash": None,
        }

    try:
        config_path = _resolve_existing_path(
            config_path_text,
            repo_roots=repo_roots,
            label="dataset_config_path",
        )
    except FileNotFoundError as exc:
        return {
            **base_record,
            "status": "historical_config_missing_explicit_replay_control",
            "source": "explicit_replay_control_fallback",
            "resolved_dataset_config_path": None,
            "historical_dataset_config_available": False,
            "historical_dataset_config_error": str(exc),
            "historical_recovered_replay_control": None,
            "historical_recovered_replay_control_hash": None,
        }

    cfg = load_config(config_path)
    dataset_cfg = _ensure_mapping(cfg.get("dataset"), label="dataset")
    recovered_control = _normalize_replay_control(
        dataset_cfg,
        label=f"dataset config {config_path}",
    )
    if recovered_control != normalized_explicit:
        raise ValueError(
            "Historical dataset replay control mismatch: "
            f"resolved_dataset_config_path={config_path} "
            f"explicit_replay_control={normalized_explicit!r} "
            f"historical_recovered_replay_control={recovered_control!r}"
        )
    recovered_hash = stable_hash_payload(recovered_control)
    return {
        **base_record,
        "status": "historical_config_recovered_matches_explicit_replay_control",
        "source": "historical_dataset_config_verified_by_explicit_replay_control",
        "resolved_dataset_config_path": str(config_path),
        "historical_dataset_config_available": True,
        "historical_dataset_config_error": None,
        "historical_recovered_replay_control": recovered_control,
        "historical_recovered_replay_control_hash": recovered_hash,
    }


def build_solver_from_legacy_sample(
    solver_key: str,
    *,
    stored_solver_cfg: Mapping[str, Any],
    sample_arrays: Mapping[str, np.ndarray],
) -> Any:
    solver_cfg = dict(stored_solver_cfg)
    bathymetry = np.asarray(sample_arrays["bathymetry"], dtype=np.float32)
    initial_depth = np.asarray(sample_arrays["initial_depth"], dtype=np.float32)
    eta0 = np.asarray(sample_arrays["eta0"], dtype=np.float32)

    if solver_key == "hydrostatic":
        solver = _make_hydrostatic_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(
            initial_depth,
            hu0=np.zeros_like(initial_depth),
            hv0=np.zeros_like(initial_depth),
        )
        return solver

    if solver_key == "muscl_hr":
        solver = _make_muscl_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(
            initial_depth,
            hu0=np.zeros_like(initial_depth),
            hv0=np.zeros_like(initial_depth),
        )
        return solver

    if solver_key == "boussinesq":
        solver = _make_boussinesq_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
        return solver

    raise ValueError(f"Unsupported solver_key {solver_key!r}")


def extract_effective_solver_config(
    solver_key: str,
    solver: Any,
) -> dict[str, Any]:
    boundary_x = str(getattr(solver, "boundary_x"))
    boundary_y = str(getattr(solver, "boundary_y"))
    boundary: str | list[str]
    if boundary_x == boundary_y:
        boundary = boundary_x
    else:
        boundary = [boundary_x, boundary_y]

    effective = {
        "nx": int(solver.nx),
        "ny": int(solver.ny),
        "dx": float(solver.dx),
        "dy": float(solver.dy),
        "dt": float(solver.dt),
        "g": float(solver.g),
        "cfl": float(solver.cfl),
        "boundary": boundary,
        "use_sponge": bool(getattr(solver, "use_sponge")),
        "sponge_width": int(getattr(solver, "sponge_width")),
        "sponge_min_factor": float(getattr(solver, "sponge_min_factor")),
    }

    if solver_key in {"hydrostatic", "muscl_hr"}:
        effective.update(
            {
                "dry_tolerance": float(solver.dry_tolerance),
                "max_velocity": float(solver.max_velocity),
            }
        )
        return effective

    if solver_key == "boussinesq":
        effective.update(
            {
                "alpha": float(solver.alpha),
                "min_depth": float(solver.min_depth),
                "sea_level_offset": float(solver.sea_level_offset),
                "depth_scale": float(solver.depth_scale),
                "mode": str(solver.mode),
                "filter_strength": float(solver.filter_strength),
                "linear_solver_tol": float(solver.linear_solver_tol),
                "linear_solver_max_iter": int(solver.linear_solver_max_iter),
                "check_finite": bool(solver.check_finite),
            }
        )
        return effective

    raise ValueError(f"Unsupported solver_key {solver_key!r}")


def summarize_dense_rollout_diagnostics(
    diagnostics: Mapping[str, np.ndarray],
    *,
    dense_timestamps: Any | None = None,
    dense_dt_history: Any | None = None,
) -> dict[str, Any]:
    required = (
        "proposed_dt",
        "pre_step_cfl",
        "post_step_cfl",
        "elapsed_benchmark_time",
        "finite_state_flag",
    )
    missing = [key for key in required if key not in diagnostics]
    if missing:
        raise KeyError(f"Dense rollout diagnostics are missing required keys {missing}")

    proposed_dt = np.asarray(diagnostics["proposed_dt"], dtype=np.float64)
    pre_cfl = np.asarray(diagnostics["pre_step_cfl"], dtype=np.float64)
    post_cfl = np.asarray(diagnostics["post_step_cfl"], dtype=np.float64)
    elapsed = np.asarray(diagnostics["elapsed_benchmark_time"], dtype=np.float64)
    finite_state = np.asarray(diagnostics["finite_state_flag"], dtype=np.bool_)
    benchmark_time_matches_dense_timestamps = True
    benchmark_time_matches_cumulative_dt = True
    benchmark_time_to_dense_timestamps_max_abs_diff = 0.0
    benchmark_time_to_cumulative_dt_max_abs_diff = 0.0

    if dense_timestamps is not None:
        ts = validate_timestamps(dense_timestamps)
        expected = np.asarray(ts[1:], dtype=np.float64)
        if expected.shape != elapsed.shape:
            raise ValueError(
                "Dense timestamps must contain exactly one more frame than "
                f"elapsed_benchmark_time rows: {expected.shape} != {elapsed.shape}"
            )
        benchmark_time_to_dense_timestamps_max_abs_diff = (
            float(np.max(np.abs(elapsed - expected))) if elapsed.size else 0.0
        )
        benchmark_time_matches_dense_timestamps = bool(
            np.array_equal(
                np.asarray(elapsed, dtype=np.float32),
                np.asarray(expected, dtype=np.float32),
            )
        )

    if dense_dt_history is not None:
        dt_history = np.asarray(dense_dt_history, dtype=np.float32).reshape(-1)
        expected = np.cumsum(dt_history[1:], dtype=np.float64)
        if expected.shape != elapsed.shape:
            raise ValueError(
                "Dense dt_history must contain exactly one more frame than "
                f"elapsed_benchmark_time rows: {expected.shape} != {elapsed.shape}"
            )
        benchmark_time_to_cumulative_dt_max_abs_diff = (
            float(np.max(np.abs(elapsed - expected))) if elapsed.size else 0.0
        )
        benchmark_time_matches_cumulative_dt = bool(
            np.allclose(elapsed, expected, atol=1.0e-7, rtol=0.0)
        )

    summary = {
        "diagnostic_row_count": int(proposed_dt.shape[0]),
        "dt_min": float(np.min(proposed_dt)) if proposed_dt.size else 0.0,
        "dt_max": float(np.max(proposed_dt)) if proposed_dt.size else 0.0,
        "pre_step_cfl_max": float(np.max(pre_cfl)) if pre_cfl.size else 0.0,
        "post_step_cfl_max": float(np.max(post_cfl)) if post_cfl.size else 0.0,
        "benchmark_time_final": float(elapsed[-1]) if elapsed.size else 0.0,
        "benchmark_time_matches_dense_timestamps": benchmark_time_matches_dense_timestamps,
        "benchmark_time_matches_cumulative_dt": benchmark_time_matches_cumulative_dt,
        "benchmark_time_to_dense_timestamps_max_abs_diff": (
            benchmark_time_to_dense_timestamps_max_abs_diff
        ),
        "benchmark_time_to_cumulative_dt_max_abs_diff": (
            benchmark_time_to_cumulative_dt_max_abs_diff
        ),
        "finite_state_failure_count": int(np.count_nonzero(~finite_state)),
    }

    if "swe_min_depth" in diagnostics:
        swe_min_depth = np.asarray(diagnostics["swe_min_depth"], dtype=np.float64)
        swe_max_speed = np.asarray(diagnostics["swe_max_speed"], dtype=np.float64)
        swe_dry_cell_count = np.asarray(
            diagnostics["swe_dry_cell_count"], dtype=np.int32
        )
        summary.update(
            {
                "swe_min_depth_min": float(np.min(swe_min_depth))
                if swe_min_depth.size
                else 0.0,
                "swe_max_speed_max": float(np.max(swe_max_speed))
                if swe_max_speed.size
                else 0.0,
                "swe_dry_cell_count_max": int(np.max(swe_dry_cell_count))
                if swe_dry_cell_count.size
                else 0,
            }
        )

    if "cg_failed_count" in diagnostics:
        cg_failed = np.asarray(diagnostics["cg_failed_count"], dtype=np.int32)
        cg_max_iterations = np.asarray(diagnostics["cg_max_iterations"], dtype=np.int32)
        cg_max_residual_ratio = np.asarray(
            diagnostics["cg_max_residual_ratio"], dtype=np.float64
        )
        summary.update(
            {
                "cg_failed_step_count": int(np.count_nonzero(cg_failed > 0)),
                "cg_failed_solve_count": int(np.sum(cg_failed)),
                "cg_max_iterations": int(np.max(cg_max_iterations))
                if cg_max_iterations.size
                else 0,
                "cg_max_residual_ratio": float(np.max(cg_max_residual_ratio))
                if cg_max_residual_ratio.size
                else 0.0,
            }
        )

    return summary


def compute_legacy_knot_reproduction_metrics(
    *,
    dense_timestamps: Any,
    dense_trajectory_eta: Any,
    legacy_timestamps: Any,
    legacy_trajectory_eta: Any,
    expected_natural_steps: int,
    legacy_knot_stride: int,
    timestamp_abs_tolerance: float,
    eta_max_abs_tolerance: float,
    relative_rmse_tolerance: float,
) -> dict[str, Any]:
    dense_ts = validate_timestamps(dense_timestamps)
    legacy_ts = validate_timestamps(legacy_timestamps)
    dense_eta, dense_ts = validate_elevation_series(dense_trajectory_eta, dense_ts)
    legacy_eta, legacy_ts = validate_elevation_series(legacy_trajectory_eta, legacy_ts)

    if int(expected_natural_steps) <= 0:
        raise ValueError("expected_natural_steps must be positive")
    if int(legacy_knot_stride) <= 0:
        raise ValueError("legacy_knot_stride must be positive")

    indices = np.arange(
        0,
        int(expected_natural_steps) + 1,
        int(legacy_knot_stride),
        dtype=np.int64,
    )
    expected_dense_frame_count = int(expected_natural_steps) + 1
    expected_legacy_frame_count = int(indices.shape[0])
    frame_count_match = (
        int(dense_eta.shape[0]) == expected_dense_frame_count
        and int(legacy_eta.shape[0]) == expected_legacy_frame_count
    )
    indices_in_range = bool(
        indices.size > 0 and int(indices[-1]) < int(dense_eta.shape[0])
    )
    selected_dense_ts = (
        dense_ts[indices] if indices_in_range else np.asarray([], dtype=np.float64)
    )
    selected_dense_eta = (
        dense_eta[indices] if indices_in_range else np.asarray([], dtype=np.float64)
    )

    timestamp_diff = (
        np.asarray(selected_dense_ts, dtype=np.float64)
        - np.asarray(legacy_ts, dtype=np.float64)
        if frame_count_match and indices_in_range
        else np.asarray([], dtype=np.float64)
    )
    eta_diff = (
        np.asarray(selected_dense_eta, dtype=np.float64)
        - np.asarray(legacy_eta, dtype=np.float64)
        if frame_count_match and indices_in_range
        else np.asarray([], dtype=np.float64)
    )
    max_abs_timestamp_diff = (
        float(np.max(np.abs(timestamp_diff))) if timestamp_diff.size else float("inf")
    )
    max_abs_eta_diff = (
        float(np.max(np.abs(eta_diff))) if eta_diff.size else float("inf")
    )
    rmse = (
        float(np.sqrt(np.mean(eta_diff * eta_diff))) if eta_diff.size else float("inf")
    )
    legacy_rms = field_rms(legacy_eta)
    relative_rmse = safe_ratio(rmse, legacy_rms)

    dense_timestamp_hash = _array_hash(np.asarray(selected_dense_ts, dtype=np.float32))
    legacy_timestamp_hash = _array_hash(np.asarray(legacy_ts, dtype=np.float32))
    dense_eta_hash = _array_hash(np.asarray(selected_dense_eta, dtype=np.float32))
    legacy_eta_hash = _array_hash(np.asarray(legacy_eta, dtype=np.float32))

    issues: list[str] = []
    if not frame_count_match:
        issues.append(
            "frame_count_mismatch"
            f"(dense={int(dense_eta.shape[0])}, legacy={int(legacy_eta.shape[0])}, "
            f"expected_dense={expected_dense_frame_count}, expected_legacy={expected_legacy_frame_count})"
        )
    if not indices_in_range:
        issues.append("legacy_knot_indices_out_of_range")
    if max_abs_timestamp_diff > float(timestamp_abs_tolerance):
        issues.append(
            f"timestamp_max_abs_diff({max_abs_timestamp_diff:.6g}) > {float(timestamp_abs_tolerance):.6g}"
        )
    if max_abs_eta_diff > float(eta_max_abs_tolerance):
        issues.append(
            f"eta_max_abs_diff({max_abs_eta_diff:.6g}) > {float(eta_max_abs_tolerance):.6g}"
        )
    if relative_rmse > float(relative_rmse_tolerance):
        issues.append(
            f"relative_rmse({relative_rmse:.6g}) > {float(relative_rmse_tolerance):.6g}"
        )

    passed = not issues
    return {
        "pass": passed,
        "issues": issues,
        "expected_natural_steps": int(expected_natural_steps),
        "legacy_knot_stride": int(legacy_knot_stride),
        "legacy_knot_indices": indices.tolist(),
        "dense_frame_count": int(dense_eta.shape[0]),
        "legacy_frame_count": int(legacy_eta.shape[0]),
        "frame_count_match": frame_count_match,
        "timestamp_max_abs_diff": max_abs_timestamp_diff,
        "eta_max_abs_diff": max_abs_eta_diff,
        "rmse": rmse,
        "legacy_trajectory_rms": legacy_rms,
        "relative_rmse": relative_rmse,
        "timestamp_float32_hash_match": dense_timestamp_hash == legacy_timestamp_hash,
        "trajectory_eta_float32_hash_match": dense_eta_hash == legacy_eta_hash,
        "dense_timestamp_float32_hash": dense_timestamp_hash,
        "legacy_timestamp_float32_hash": legacy_timestamp_hash,
        "dense_trajectory_eta_float32_hash": dense_eta_hash,
        "legacy_trajectory_eta_float32_hash": legacy_eta_hash,
    }


def _interpolate_to_queries(
    *,
    elevation: np.ndarray,
    timestamps: np.ndarray,
    queries: np.ndarray,
    endpoint_tolerance: float,
) -> np.ndarray:
    values, ts = validate_elevation_series(elevation, timestamps)
    q = np.asarray(queries, dtype=np.float64).reshape(-1)
    if q.size == 0:
        return np.empty((0, values.shape[1], values.shape[2]), dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError("Interpolation query times must be finite")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("Interpolation query times must be strictly increasing")

    lower = float(ts[0])
    upper = float(ts[-1])
    if np.any(q < lower - float(endpoint_tolerance)) or np.any(
        q > upper + float(endpoint_tolerance)
    ):
        raise ValueError("Interpolation queries would require extrapolation")

    snapped = q.copy()
    snapped[np.abs(snapped - lower) <= float(endpoint_tolerance)] = lower
    snapped[np.abs(snapped - upper) <= float(endpoint_tolerance)] = upper

    right = np.searchsorted(ts, snapped, side="left")
    right = np.clip(right, 1, ts.shape[0] - 1)
    left = right - 1
    left_times = ts[left]
    right_times = ts[right]
    denom = right_times - left_times
    if np.any(denom <= 0.0):
        raise ValueError("Interpolation requires strictly increasing timestamps")

    weights = ((snapped - left_times) / denom).astype(np.float64)
    return np.asarray(
        values[left] * (1.0 - weights[:, None, None])
        + values[right] * weights[:, None, None],
        dtype=np.float64,
    )


def compute_sparse_interpolation_metrics(
    *,
    sparse_trajectory_eta: Any,
    sparse_timestamps: Any,
    dense_trajectory_eta: Any,
    dense_timestamps: Any,
    common_time_grid: Sequence[float] | np.ndarray,
    interpolation_horizon: float,
    endpoint_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    sparse_ts = validate_timestamps(sparse_timestamps)
    dense_ts = validate_timestamps(dense_timestamps)
    sparse_eta, sparse_ts = validate_elevation_series(sparse_trajectory_eta, sparse_ts)
    dense_eta, dense_ts = validate_elevation_series(dense_trajectory_eta, dense_ts)
    horizon = float(interpolation_horizon)
    if horizon <= 0.0:
        raise ValueError("interpolation_horizon must be positive")

    dense_mask = (dense_ts > 0.0) & (dense_ts <= horizon + float(endpoint_tolerance))
    dense_queries = np.asarray(dense_ts[dense_mask], dtype=np.float64)
    if dense_queries.size == 0:
        raise ValueError(
            "Dense trajectory has no natural steps within the interpolation horizon"
        )

    interpolated_dense_support = _interpolate_to_queries(
        elevation=sparse_eta,
        timestamps=sparse_ts,
        queries=dense_queries,
        endpoint_tolerance=endpoint_tolerance,
    )
    dense_support_reference = np.asarray(dense_eta[dense_mask], dtype=np.float64)
    dense_support_metrics = compute_error_metrics(
        interpolated_dense_support,
        dense_support_reference,
    )
    dense_support_rms = field_rms(dense_support_reference)
    dense_support_metrics.update(
        {
            "query_count": int(dense_queries.shape[0]),
            "trajectory_rms": dense_support_rms,
            "rmse_over_trajectory_rms": safe_ratio(
                float(dense_support_metrics["rmse"]),
                dense_support_rms,
            ),
        }
    )

    grid = validate_common_time_grid(common_time_grid)
    legacy_on_grid = align_elevation_series(
        sparse_eta,
        sparse_ts,
        mode=MODE_COMMON_TIME,
        common_time_grid=grid,
        endpoint_tolerance=float(endpoint_tolerance),
    )
    dense_on_grid = align_elevation_series(
        dense_eta,
        dense_ts,
        mode=MODE_COMMON_TIME,
        common_time_grid=grid,
        endpoint_tolerance=float(endpoint_tolerance),
    )
    common_grid_metrics = compute_error_metrics(
        legacy_on_grid,
        dense_on_grid,
    )
    common_grid_rms = field_rms(dense_on_grid)
    common_grid_metrics.update(
        {
            "query_count": int(grid.shape[0]),
            "trajectory_rms": common_grid_rms,
            "rmse_over_trajectory_rms": safe_ratio(
                float(common_grid_metrics["rmse"]),
                common_grid_rms,
            ),
        }
    )

    return (
        {
            "zero_extrapolation": True,
            "dense_support": dense_support_metrics,
            "common_grid": common_grid_metrics,
        },
        np.asarray(dense_on_grid, dtype=np.float64),
    )


def compute_pairwise_dense_solver_gaps(
    dense_common_grid_by_solver: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    pair_metrics: dict[str, Any] = {}
    rmse_values: list[float] = []
    mse_values: list[float] = []
    for left_index, left_solver in enumerate(SOLVER_ORDER):
        if left_solver not in dense_common_grid_by_solver:
            continue
        for right_solver in SOLVER_ORDER[left_index + 1 :]:
            if right_solver not in dense_common_grid_by_solver:
                continue
            pair_key = f"{left_solver}__vs__{right_solver}"
            metrics = compute_error_metrics(
                dense_common_grid_by_solver[left_solver],
                dense_common_grid_by_solver[right_solver],
            )
            pair_metrics[pair_key] = metrics
            rmse_values.append(float(metrics["rmse"]))
            mse_values.append(float(metrics["mse"]))

    smallest_pair_key = None
    smallest_pair_rmse = float("inf")
    for pair_key, metrics in pair_metrics.items():
        pair_rmse = float(metrics["rmse"])
        if pair_rmse < smallest_pair_rmse:
            smallest_pair_rmse = pair_rmse
            smallest_pair_key = pair_key

    return {
        "pairs": pair_metrics,
        "aggregate_mse_values": mse_values,
        "smallest_pair_key": smallest_pair_key,
        "smallest_pair_rmse": smallest_pair_rmse,
    }


def evaluate_dense_reference_criteria(
    *,
    suite_spec: SuiteSpec,
    zero_extrapolation_failures: int,
    reproduction_failure_count: int,
    aggregate_interp_to_gap_ratio: float,
    scenario_ratio_median: float,
    scenario_ratio_p95: float,
    family_summaries: Sequence[Mapping[str, Any]],
    aggregate_interp_to_rms_ratio: float,
    scenario_rms_ratio_median: float,
    scenario_rms_ratio_p95: float,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    family_median_limit = float(thresholds["family_cell_median_max"])
    family_fraction_limit = float(thresholds["family_cell_fraction_above_p95_max"])
    max_family_median = max(
        (float(row["scenario_ratio_median"]) for row in family_summaries),
        default=0.0,
    )
    max_family_fraction = max(
        (
            float(row["scenario_ratio_fraction_above_p95_threshold"])
            for row in family_summaries
        ),
        default=0.0,
    )

    criteria = {
        "zero_extrapolation": {
            "observed": int(zero_extrapolation_failures),
            "required": 0,
            "pass": int(zero_extrapolation_failures) == 0,
        },
        "all_legacy_reproduction_pass": {
            "observed": int(reproduction_failure_count),
            "required": 0,
            "pass": int(reproduction_failure_count) == 0,
        },
        "aggregate_interp_rmse_over_smallest_aggregate_solver_gap": {
            "observed": float(aggregate_interp_to_gap_ratio),
            "threshold_max": float(thresholds["aggregate_ratio_max"]),
            "pass": float(aggregate_interp_to_gap_ratio)
            <= float(thresholds["aggregate_ratio_max"]),
        },
        "scenario_ratio_distribution": {
            "observed_median": float(scenario_ratio_median),
            "observed_p95": float(scenario_ratio_p95),
            "threshold_median_max": float(thresholds["scenario_ratio_median_max"]),
            "threshold_p95_max": float(thresholds["scenario_ratio_p95_max"]),
            "pass": (
                float(scenario_ratio_median)
                <= float(thresholds["scenario_ratio_median_max"])
                and float(scenario_ratio_p95)
                <= float(thresholds["scenario_ratio_p95_max"])
            ),
        },
        "family_cell_ratio_guardrail": {
            "observed_max_median": float(max_family_median),
            "observed_max_fraction_above_p95_threshold": float(max_family_fraction),
            "threshold_median_max": family_median_limit,
            "threshold_fraction_above_p95_threshold_max": family_fraction_limit,
            "pass": (
                float(max_family_median) <= family_median_limit
                and float(max_family_fraction) <= family_fraction_limit
            ),
        },
        "interp_rmse_over_trajectory_rms": {
            "observed_aggregate": float(aggregate_interp_to_rms_ratio),
            "observed_median": float(scenario_rms_ratio_median),
            "observed_p95": float(scenario_rms_ratio_p95),
            "threshold_aggregate_max": float(thresholds["interp_to_rms_aggregate_max"]),
            "threshold_median_max": float(thresholds["interp_to_rms_median_max"]),
            "threshold_p95_max": float(thresholds["interp_to_rms_p95_max"]),
            "pass": (
                float(aggregate_interp_to_rms_ratio)
                <= float(thresholds["interp_to_rms_aggregate_max"])
                and float(scenario_rms_ratio_median)
                <= float(thresholds["interp_to_rms_median_max"])
                and float(scenario_rms_ratio_p95)
                <= float(thresholds["interp_to_rms_p95_max"])
            ),
        },
    }

    implementation_criteria = {"zero_extrapolation", "all_legacy_reproduction_pass"}
    for name, criterion in criteria.items():
        criterion["gating"] = (
            name in implementation_criteria or suite_spec.manuscript_claims_allowed
        )

    gating_criteria = [item for item in criteria.values() if bool(item["gating"])]
    return {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "dense-reference-validation-decision",
        "suite": {
            "name": suite_spec.name,
            "label": suite_spec.label,
            "purpose": suite_spec.purpose,
            "manuscript_claims_allowed": suite_spec.manuscript_claims_allowed,
        },
        "decision_scope": (
            "scientific_interpolation"
            if suite_spec.manuscript_claims_allowed
            else "implementation_only"
        ),
        "status": "pass"
        if all(bool(item["pass"]) for item in gating_criteria)
        else "fail",
        "criteria": criteria,
    }


def _processed_root_from_sources(
    solver_key: str,
    *,
    audit_artifact: Mapping[str, Any],
    dense_cfg: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> Path:
    config_audit = _ensure_mapping(
        audit_artifact.get("config"), label="audit_artifact.config"
    )
    audit_cfg = _ensure_mapping(
        config_audit.get("audit"), label="audit_artifact.config.audit"
    )
    dense_processed_roots = _ensure_mapping(
        dense_cfg.get("processed_test_roots"),
        label="dense_reference_validation.processed_test_roots",
    )
    path_text = overrides.get(
        solver_key,
        dense_processed_roots.get(
            solver_key,
            _ensure_mapping(
                audit_cfg.get("processed_test_roots"),
                label="audit.processed_test_roots",
            ).get(solver_key),
        ),
    )
    if not path_text:
        raise KeyError(f"Missing processed test root for {solver_key}")
    path = _resolve_repo_path(str(path_text))
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Processed test root not found for {solver_key}: {path_text}"
        )
    return path


def _raw_root_override_from_sources(
    solver_key: str,
    *,
    audit_artifact: Mapping[str, Any],
    dense_cfg: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> Path | None:
    config_audit = _ensure_mapping(
        audit_artifact.get("config"), label="audit_artifact.config"
    )
    audit_cfg = _ensure_mapping(
        config_audit.get("audit"), label="audit_artifact.config.audit"
    )
    dense_raw_roots = _ensure_mapping(
        dense_cfg.get("raw_test_solver_roots"),
        label="dense_reference_validation.raw_test_solver_roots",
    )
    path_text = overrides.get(
        solver_key,
        dense_raw_roots.get(
            solver_key,
            _ensure_mapping(
                audit_cfg.get("raw_test_solver_roots"),
                label="audit.raw_test_solver_roots",
            ).get(solver_key),
        ),
    )
    return _resolve_repo_path(str(path_text)) if path_text else None


def _common_grid_from_config(
    alignment_cfg: Mapping[str, Any],
    dense_cfg: Mapping[str, Any],
) -> np.ndarray:
    dense_grid_cfg = _ensure_mapping(
        dense_cfg.get("common_time_grid"),
        label="dense_reference_validation.common_time_grid",
    )
    if "values" in dense_grid_cfg:
        return validate_common_time_grid(dense_grid_cfg["values"])
    alignment_grid_cfg = _ensure_mapping(
        alignment_cfg.get("common_time_grid"),
        label="alignment.common_time_grid",
    )
    return validate_common_time_grid(alignment_grid_cfg.get("values"))


def _scenario_list_from_selection(
    suite_spec: SuiteSpec,
) -> dict[str, dict[str, Any]]:
    return {str(row["scenario_id"]): dict(row) for row in suite_spec.ordered_scenarios}


def run_dense_reference_validation(
    config: Mapping[str, Any],
    *,
    suite_name: str,
    config_path: Path | None = None,
    audit_artifact_path: str | None = None,
    scenario_selection_path: str | None = None,
    processed_root_overrides: Mapping[str, str] | None = None,
    raw_root_overrides: Mapping[str, str] | None = None,
    output_root_override: str | None = None,
    progress_callback: Callable[[int, int, str | None], None] | None = None,
) -> dict[str, Any]:
    dense_cfg = _ensure_mapping(
        config.get("dense_reference_validation"),
        label="dense_reference_validation",
    )
    alignment_cfg = _ensure_mapping(config.get("alignment"), label="alignment")

    audit_path = (
        _resolve_repo_path(audit_artifact_path)
        if audit_artifact_path is not None
        else _resolve_repo_path(
            str(
                dense_cfg.get(
                    "audit_artifact",
                    "results/common_time_validation/audit/paired_reference_audit.json",
                )
            )
        )
    )
    if audit_path is None:
        raise ValueError("Could not resolve audit_artifact path")
    selection_path = (
        _resolve_repo_path(scenario_selection_path)
        if scenario_selection_path is not None
        else _resolve_repo_path(
            str(
                dense_cfg.get(
                    "scenario_selection_path",
                    "configs/eval/common_time_validation_scenarios.json",
                )
            )
        )
    )
    if selection_path is None:
        raise ValueError("Could not resolve scenario_selection_path")
    audit_artifact = _load_json(audit_path)
    selection_artifact = _load_json(selection_path)

    if str(audit_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Audit artifact schema_id must be {SCHEMA_ID!r}")
    if str(audit_artifact.get("artifact_kind", "")) != "paired-reference-audit":
        raise ValueError("Expected a paired-reference-audit artifact")
    if str(audit_artifact.get("status", "")) != "pass":
        raise ValueError("Dense reference validation requires a passing audit artifact")
    if str(selection_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Scenario selection schema_id must be {SCHEMA_ID!r}")
    if (
        str(selection_artifact.get("artifact_kind", ""))
        != "common-time-validation-scenarios"
    ):
        raise ValueError("Expected a common-time-validation-scenarios artifact")
    selection_audit_hash = str(selection_artifact.get("audit_hash", "")).strip()
    if not selection_audit_hash:
        raise ValueError("Scenario selection artifact is missing audit_hash")
    if selection_audit_hash != str(audit_artifact["audit_hash"]):
        raise ValueError(
            "Scenario selection audit_hash does not match the paired-reference audit artifact"
        )

    suite_spec = resolve_validation_suite(selection_artifact, suite_name)
    common_grid = _common_grid_from_config(alignment_cfg, dense_cfg)
    endpoint_tolerance = float(
        _ensure_mapping(
            dense_cfg.get("common_time_grid"),
            label="dense_reference_validation.common_time_grid",
        ).get(
            "endpoint_tolerance",
            _ensure_mapping(
                alignment_cfg.get("common_time_grid"),
                label="alignment.common_time_grid",
            ).get("endpoint_tolerance", DEFAULT_ENDPOINT_TOLERANCE),
        )
    )
    expected_natural_steps = int(dense_cfg.get("expected_natural_steps", 250))
    legacy_knot_stride = int(dense_cfg.get("legacy_knot_stride", 5))
    tolerances = _ensure_mapping(
        dense_cfg.get("tolerances"), label="dense_reference_validation.tolerances"
    )
    timestamp_tolerance = float(tolerances.get("timestamp_abs", 5.0e-8))
    eta_tolerance = float(tolerances.get("eta_max_abs", 1.0e-6))
    relative_rmse_tolerance = float(
        tolerances.get("reproduction_relative_rmse", 1.0e-6)
    )
    interpolation_horizon = float(
        tolerances.get("interpolation_horizon", float(common_grid[-1]))
    )
    criteria_cfg = _ensure_mapping(
        dense_cfg.get("criteria"),
        label="dense_reference_validation.criteria",
    )
    thresholds = {
        "aggregate_ratio_max": float(criteria_cfg.get("aggregate_ratio_max", 0.10)),
        "scenario_ratio_median_max": float(
            criteria_cfg.get("scenario_ratio_median_max", 0.10)
        ),
        "scenario_ratio_p95_max": float(
            criteria_cfg.get("scenario_ratio_p95_max", 0.25)
        ),
        "family_cell_median_max": float(
            criteria_cfg.get("family_cell_median_max", 0.10)
        ),
        "family_cell_fraction_above_p95_max": float(
            criteria_cfg.get("family_cell_fraction_above_p95_max", 0.5)
        ),
        "interp_to_rms_aggregate_max": float(
            criteria_cfg.get("interp_to_rms_aggregate_max", 0.01)
        ),
        "interp_to_rms_median_max": float(
            criteria_cfg.get("interp_to_rms_median_max", 0.01)
        ),
        "interp_to_rms_p95_max": float(
            criteria_cfg.get("interp_to_rms_p95_max", 0.025)
        ),
    }
    bootstrap_cfg = _ensure_mapping(
        dense_cfg.get("bootstrap"),
        label="dense_reference_validation.bootstrap",
    )
    bootstrap_seed = int(bootstrap_cfg.get("seed", 20260712))
    bootstrap_resamples = int(bootstrap_cfg.get("num_resamples", 10000))
    bootstrap_confidence = float(bootstrap_cfg.get("confidence_level", 0.95))
    explicit_replay_control = dense_cfg.get("replay_control")

    output_root = (
        _resolve_repo_path(output_root_override)
        if output_root_override is not None
        else _resolve_repo_path(
            str(
                dense_cfg.get(
                    "results_root",
                    "results/common_time_validation/dense_reference_validation",
                )
            )
        )
    )
    if output_root is None:
        raise ValueError("Could not resolve dense reference validation output root")
    suite_output_dir = output_root / suite_spec.name
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    scenario_metrics_path = suite_output_dir / "scenario_metrics.jsonl"
    if scenario_metrics_path.exists():
        scenario_metrics_path.unlink()

    processed_overrides = _normalize_override_mapping(processed_root_overrides)
    raw_overrides = _normalize_override_mapping(raw_root_overrides)

    processed_rows_by_solver: dict[str, dict[str, dict[str, Any]]] = {}
    processed_root_paths: dict[str, str] = {}
    raw_root_paths: dict[str, str | None] = {}
    for solver_key in SOLVER_ORDER:
        processed_root = _processed_root_from_sources(
            solver_key,
            audit_artifact=audit_artifact,
            dense_cfg=dense_cfg,
            overrides=processed_overrides,
        )
        processed_root_paths[solver_key] = str(processed_root)
        rows = _read_jsonl(processed_root / "meta.jsonl")
        processed_rows_by_solver[solver_key] = {
            str(row["scenario_id"]): row for row in rows
        }
        raw_root = _raw_root_override_from_sources(
            solver_key,
            audit_artifact=audit_artifact,
            dense_cfg=dense_cfg,
            overrides=raw_overrides,
        )
        raw_root_paths[solver_key] = None if raw_root is None else str(raw_root)

    selected_scenarios = _scenario_list_from_selection(suite_spec)
    total_scenarios = len(suite_spec.ordered_scenario_ids)
    scenario_records: list[dict[str, Any]] = []
    solver_health_accumulator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    interp_mse_values: list[float] = []
    interp_trajectory_rms_values: list[float] = []
    pairwise_mse_values: dict[str, list[float]] = defaultdict(list)
    scenario_ratios: list[float] = []
    scenario_rms_ratios: list[float] = []
    zero_extrapolation_failures = 0
    reproduction_failure_count = 0

    with scenario_metrics_path.open("w", encoding="utf-8") as scenario_handle:
        for scenario_id in suite_spec.ordered_scenario_ids:
            selection_row = dict(selected_scenarios[scenario_id])
            legacy_samples: dict[str, dict[str, Any]] = {}
            sample_indices: list[int] = []
            for solver_key in SOLVER_ORDER:
                row = processed_rows_by_solver[solver_key].get(scenario_id)
                if row is None:
                    raise KeyError(
                        f"Scenario {scenario_id!r} missing from processed metadata for solver {solver_key!r}"
                    )
                sample_index = int(row["sample_index"])
                sample_indices.append(sample_index)
                sample_dir = resolve_sample_dir(
                    str(row["sample_dir"]),
                    solver_key=solver_key,
                    sample_index=sample_index,
                    raw_root_override=(
                        Path(raw_root_paths[solver_key])
                        if raw_root_paths[solver_key] is not None
                        else None
                    ),
                )
                legacy_samples[solver_key] = _load_legacy_sample(sample_dir)

            scenario_record: dict[str, Any] = {
                "scenario_id": scenario_id,
                "sample_index": int(sample_indices[0]),
                "bathymetry_type": str(selection_row.get("bathymetry_type", "")),
                "source_type": str(selection_row.get("source_type", "")),
                "source_strength": float(
                    selection_row.get("source_strength", float("nan"))
                ),
                "suite": suite_spec.name,
                "solver_results": {},
            }

            input_hashes_by_field: dict[str, dict[str, str]] = {}
            shared_input_hashes: dict[str, str] = {}
            shared_input_hash_mismatches: list[str] = []
            dataset_control_hashes: dict[str, str] = {}
            dense_rollouts: dict[str, dict[str, Any]] = {}
            solver_reproduction_failures = 0
            all_reproduction_pass = True

            for field in COMMON_INPUT_FIELDS:
                per_solver_hashes = {
                    solver_key: _array_hash(legacy_samples[solver_key]["arrays"][field])
                    for solver_key in SOLVER_ORDER
                }
                input_hashes_by_field[field] = per_solver_hashes
                unique_hashes = set(per_solver_hashes.values())
                shared_input_hashes[field] = per_solver_hashes[SOLVER_ORDER[0]]
                if len(unique_hashes) != 1:
                    shared_input_hash_mismatches.append(field)

            for solver_key in SOLVER_ORDER:
                sample = legacy_samples[solver_key]
                meta = _ensure_mapping(sample["meta"], label=f"{solver_key}.meta")
                solver_meta = _ensure_mapping(
                    meta.get("solver"), label=f"{solver_key}.meta.solver"
                )
                repo_roots = _repo_root_candidates(str(sample["sample_dir"]))
                dataset_control = _load_dataset_rollout_control(
                    meta,
                    repo_roots=repo_roots,
                    explicit_replay_control=explicit_replay_control,
                )
                dataset_control_hashes[solver_key] = str(dataset_control["hash"])
                dataset_control_matches_expected = (
                    int(dataset_control["n_steps"]) == expected_natural_steps
                    and int(dataset_control["save_every"]) == legacy_knot_stride
                    and bool(dataset_control["include_initial_state"]) is True
                )

                solver = build_solver_from_legacy_sample(
                    solver_key,
                    stored_solver_cfg=solver_meta,
                    sample_arrays=sample["arrays"],
                )
                effective_solver_cfg = extract_effective_solver_config(
                    solver_key, solver
                )
                stored_solver_hash = stable_hash_payload(solver_meta)
                effective_solver_hash = stable_hash_payload(effective_solver_cfg)
                (
                    dense_trajectory,
                    dense_timestamps,
                    dense_dt_history,
                    dense_diagnostics,
                ) = _simulate_one_local(
                    solver=solver,
                    n_steps=int(dataset_control["n_steps"]),
                    save_every=int(dataset_control["save_every"]),
                    auto_dt=bool(dataset_control["auto_dt"]),
                    target_cfl=float(dataset_control["target_cfl"]),
                    include_initial_state=bool(
                        dataset_control["include_initial_state"]
                    ),
                    record_every_step=True,
                    dense_diagnostics=True,
                )
                dense_trajectory_eta = (
                    dense_trajectory[:, 0]
                    if solver_key == "boussinesq"
                    else dense_trajectory[:, 0]
                    + np.asarray(sample["arrays"]["bathymetry"], dtype=np.float32)[
                        None, ...
                    ]
                )
                health_summary = summarize_dense_rollout_diagnostics(
                    dense_diagnostics,
                    dense_timestamps=dense_timestamps,
                    dense_dt_history=dense_dt_history,
                )
                if not bool(health_summary["benchmark_time_matches_dense_timestamps"]):
                    raise RuntimeError(
                        "elapsed_benchmark_time does not match dense timestamps "
                        f"for scenario={scenario_id} solver={solver_key}"
                    )
                if not bool(health_summary["benchmark_time_matches_cumulative_dt"]):
                    raise RuntimeError(
                        "elapsed_benchmark_time does not match cumulative dt_history "
                        f"for scenario={scenario_id} solver={solver_key}"
                    )
                solver_health_accumulator[solver_key].append(health_summary)

                reproduction = compute_legacy_knot_reproduction_metrics(
                    dense_timestamps=dense_timestamps,
                    dense_trajectory_eta=dense_trajectory_eta,
                    legacy_timestamps=sample["arrays"]["timestamps"],
                    legacy_trajectory_eta=sample["arrays"]["trajectory_eta"],
                    expected_natural_steps=expected_natural_steps,
                    legacy_knot_stride=legacy_knot_stride,
                    timestamp_abs_tolerance=timestamp_tolerance,
                    eta_max_abs_tolerance=eta_tolerance,
                    relative_rmse_tolerance=relative_rmse_tolerance,
                )
                if not dataset_control_matches_expected:
                    reproduction["pass"] = False
                    reproduction["issues"] = list(reproduction["issues"]) + [
                        "dataset_control_mismatch"
                        f"(n_steps={dataset_control['n_steps']}, save_every={dataset_control['save_every']}, "
                        f"include_initial_state={dataset_control['include_initial_state']})"
                    ]

                dense_rollouts[solver_key] = {
                    "trajectory_eta": dense_trajectory_eta,
                    "timestamps": dense_timestamps,
                }
                scenario_record["solver_results"][solver_key] = {
                    "display_name": DISPLAY_SOLVER_NAMES[solver_key],
                    "legacy_sample_dir": str(sample["sample_dir"]),
                    "stored_solver_config": solver_meta,
                    "stored_solver_config_hash": stored_solver_hash,
                    "effective_constructor_config": effective_solver_cfg,
                    "effective_constructor_config_hash": effective_solver_hash,
                    "input_hashes": {
                        field: input_hashes_by_field[field][solver_key]
                        for field in COMMON_INPUT_FIELDS
                    },
                    "dataset_rollout_control": dataset_control,
                    "dense_rollout": {
                        "frame_count": int(dense_trajectory.shape[0]),
                        "dt_history_count": int(np.asarray(dense_dt_history).shape[0]),
                        "diagnostic_row_count": int(
                            np.asarray(dense_diagnostics["proposed_dt"]).shape[0]
                        ),
                    },
                    "health_summary": health_summary,
                    "reproduction": reproduction,
                    "interpolation": None,
                }
                if not reproduction["pass"]:
                    all_reproduction_pass = False
                    solver_reproduction_failures += 1

            dataset_control_mismatch = len(set(dataset_control_hashes.values())) != 1
            if dataset_control_mismatch:
                all_reproduction_pass = False

            scenario_record["shared_input_hashes"] = shared_input_hashes
            scenario_record["shared_input_hashes_by_solver"] = input_hashes_by_field
            scenario_record[
                "shared_input_hash_match"
            ] = not shared_input_hash_mismatches
            scenario_record["shared_input_hash_mismatches"] = (
                shared_input_hash_mismatches
            )
            scenario_record["dataset_control_hashes"] = dataset_control_hashes
            scenario_record["dataset_control_hash_match"] = not dataset_control_mismatch
            scenario_record["all_legacy_reproduction_pass"] = (
                all_reproduction_pass
                and not shared_input_hash_mismatches
                and not dataset_control_mismatch
            )
            if not scenario_record["all_legacy_reproduction_pass"]:
                reproduction_failure_count += max(1, solver_reproduction_failures)

            if (
                scenario_record["all_legacy_reproduction_pass"]
                and not shared_input_hash_mismatches
                and not dataset_control_mismatch
            ):
                dense_common_grid_by_solver: dict[str, np.ndarray] = {}
                zero_extrapolation = True
                scenario_interp_mse_values: list[float] = []
                scenario_interp_rms_values: list[float] = []
                for solver_key in SOLVER_ORDER:
                    sample = legacy_samples[solver_key]
                    try:
                        interpolation, dense_common_grid = (
                            compute_sparse_interpolation_metrics(
                                sparse_trajectory_eta=sample["arrays"][
                                    "trajectory_eta"
                                ],
                                sparse_timestamps=sample["arrays"]["timestamps"],
                                dense_trajectory_eta=dense_rollouts[solver_key][
                                    "trajectory_eta"
                                ],
                                dense_timestamps=dense_rollouts[solver_key][
                                    "timestamps"
                                ],
                                common_time_grid=common_grid,
                                interpolation_horizon=interpolation_horizon,
                                endpoint_tolerance=endpoint_tolerance,
                            )
                        )
                    except ValueError as exc:
                        zero_extrapolation = False
                        zero_extrapolation_failures += 1
                        scenario_record["solver_results"][solver_key][
                            "interpolation"
                        ] = {
                            "zero_extrapolation": False,
                            "error": repr(exc),
                        }
                        continue

                    dense_common_grid_by_solver[solver_key] = dense_common_grid
                    scenario_interp_mse_values.append(
                        float(interpolation["common_grid"]["mse"])
                    )
                    scenario_interp_rms_values.append(
                        float(interpolation["common_grid"]["trajectory_rms"]) ** 2
                    )
                    scenario_record["solver_results"][solver_key]["interpolation"] = (
                        interpolation
                    )

                all_solver_interpolations_complete = zero_extrapolation and len(
                    dense_common_grid_by_solver
                ) == len(SOLVER_ORDER)
                if all_solver_interpolations_complete:
                    pairwise_gaps = compute_pairwise_dense_solver_gaps(
                        dense_common_grid_by_solver
                    )
                    for pair_key, metrics in pairwise_gaps["pairs"].items():
                        pairwise_mse_values[pair_key].append(float(metrics["mse"]))
                    interp_mse_values.extend(scenario_interp_mse_values)
                    interp_trajectory_rms_values.extend(scenario_interp_rms_values)

                    scenario_interp_rmse = aggregate_rmse(scenario_interp_mse_values)
                    scenario_trajectory_rms = aggregate_rmse(scenario_interp_rms_values)
                    scenario_ratio = safe_ratio(
                        scenario_interp_rmse,
                        float(pairwise_gaps["smallest_pair_rmse"]),
                    )
                    scenario_rms_ratio = safe_ratio(
                        scenario_interp_rmse,
                        scenario_trajectory_rms,
                    )
                    scenario_ratios.append(float(scenario_ratio))
                    scenario_rms_ratios.append(float(scenario_rms_ratio))
                    scenario_record["eligible_for_interpolation"] = True
                    scenario_record["zero_extrapolation"] = True
                    scenario_record["pairwise_dense_solver_gaps"] = pairwise_gaps[
                        "pairs"
                    ]
                    scenario_record["scenario_interp_rmse"] = float(
                        scenario_interp_rmse
                    )
                    scenario_record["scenario_trajectory_rms"] = float(
                        scenario_trajectory_rms
                    )
                    scenario_record["scenario_interp_rmse_over_smallest_gap"] = float(
                        scenario_ratio
                    )
                    scenario_record["scenario_interp_rmse_over_trajectory_rms"] = float(
                        scenario_rms_ratio
                    )
                else:
                    scenario_record["eligible_for_interpolation"] = False
                    scenario_record["zero_extrapolation"] = zero_extrapolation
                    scenario_record["pairwise_dense_solver_gaps"] = {}
                    scenario_record["scenario_interp_rmse"] = None
                    scenario_record["scenario_trajectory_rms"] = None
                    scenario_record["scenario_interp_rmse_over_smallest_gap"] = None
                    scenario_record["scenario_interp_rmse_over_trajectory_rms"] = None
                    scenario_record["interpolation_gate_reason"] = (
                        "incomplete_solver_interpolation"
                    )
            else:
                scenario_record["eligible_for_interpolation"] = False
                scenario_record["zero_extrapolation"] = False
                scenario_record["pairwise_dense_solver_gaps"] = {}
                scenario_record["scenario_interp_rmse"] = None
                scenario_record["scenario_trajectory_rms"] = None
                scenario_record["scenario_interp_rmse_over_smallest_gap"] = None
                scenario_record["scenario_interp_rmse_over_trajectory_rms"] = None

            scenario_records.append(scenario_record)
            scenario_handle.write(json.dumps(scenario_record) + "\n")
            if progress_callback is not None:
                progress_callback(len(scenario_records), total_scenarios, scenario_id)

    aggregate_interp_rmse = aggregate_rmse(interp_mse_values)
    aggregate_trajectory_rms = aggregate_rmse(interp_trajectory_rms_values)
    aggregate_gap_by_pair = {
        pair_key: aggregate_rmse(mse_values)
        for pair_key, mse_values in sorted(pairwise_mse_values.items())
    }
    smallest_aggregate_gap = (
        min(aggregate_gap_by_pair.values()) if aggregate_gap_by_pair else 0.0
    )
    aggregate_interp_to_gap_ratio = safe_ratio(
        aggregate_interp_rmse, smallest_aggregate_gap
    )
    aggregate_interp_to_rms_ratio = (
        safe_ratio(aggregate_interp_rmse, aggregate_trajectory_rms)
        if interp_mse_values
        else float("inf")
    )
    scenario_ratio_median = (
        float(np.median(scenario_ratios)) if scenario_ratios else float("inf")
    )
    scenario_ratio_p95 = (
        float(np.percentile(scenario_ratios, 95)) if scenario_ratios else float("inf")
    )
    scenario_rms_ratio_median = (
        float(np.median(scenario_rms_ratios)) if scenario_rms_ratios else float("inf")
    )
    scenario_rms_ratio_p95 = (
        float(np.percentile(scenario_rms_ratios, 95))
        if scenario_rms_ratios
        else float("inf")
    )

    family_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    family_rms_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scenario_records:
        if not row["eligible_for_interpolation"]:
            continue
        key = (str(row["bathymetry_type"]), str(row["source_type"]))
        family_values[key].append(float(row["scenario_interp_rmse_over_smallest_gap"]))
        family_rms_values[key].append(
            float(row["scenario_interp_rmse_over_trajectory_rms"])
        )

    family_summaries = []
    for bathymetry_type, source_type in sorted(family_values):
        ratios = np.asarray(
            family_values[(bathymetry_type, source_type)], dtype=np.float64
        )
        rms_ratios = np.asarray(
            family_rms_values[(bathymetry_type, source_type)], dtype=np.float64
        )
        family_summaries.append(
            {
                "bathymetry_type": bathymetry_type,
                "source_type": source_type,
                "scenario_count": int(ratios.shape[0]),
                "scenario_ratio_median": float(np.median(ratios)),
                "scenario_ratio_p95": float(np.percentile(ratios, 95)),
                "scenario_ratio_fraction_above_p95_threshold": float(
                    np.mean(ratios > float(thresholds["scenario_ratio_p95_max"]))
                ),
                "interp_rmse_over_trajectory_rms_median": float(np.median(rms_ratios)),
                "interp_rmse_over_trajectory_rms_p95": float(
                    np.percentile(rms_ratios, 95)
                ),
            }
        )

    bootstrap_summary: dict[str, Any]
    if scenario_ratios:
        bootstrap_indices = generate_paired_bootstrap_indices(
            num_scenarios=len(scenario_ratios),
            num_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        bootstrap_summary = summarize_paired_bootstrap(
            {
                "scenario_interp_rmse_over_smallest_gap": np.asarray(
                    scenario_ratios, dtype=np.float64
                ),
                "scenario_interp_rmse_over_trajectory_rms": np.asarray(
                    scenario_rms_ratios,
                    dtype=np.float64,
                ),
            },
            bootstrap_indices=bootstrap_indices,
            confidence_level=bootstrap_confidence,
        )
    else:
        bootstrap_summary = {
            "status": "not_computed",
            "reason": "No scenarios satisfied legacy reproduction and shared-input gating",
        }

    solver_health_summary = {}
    for solver_key, rows in solver_health_accumulator.items():
        solver_health_summary[solver_key] = {
            "scenario_count": int(len(rows)),
            "diagnostic_row_count": int(
                sum(int(row["diagnostic_row_count"]) for row in rows)
            ),
            "finite_state_failure_count": int(
                sum(int(row["finite_state_failure_count"]) for row in rows)
            ),
            "dt_min": float(min(float(row["dt_min"]) for row in rows)),
            "dt_max": float(max(float(row["dt_max"]) for row in rows)),
            "pre_step_cfl_max": float(
                max(float(row["pre_step_cfl_max"]) for row in rows)
            ),
            "post_step_cfl_max": float(
                max(float(row["post_step_cfl_max"]) for row in rows)
            ),
            "benchmark_time_final_max": float(
                max(float(row["benchmark_time_final"]) for row in rows)
            ),
            "benchmark_time_dense_timestamp_mismatch_count": int(
                sum(
                    int(not bool(row["benchmark_time_matches_dense_timestamps"]))
                    for row in rows
                )
            ),
            "benchmark_time_cumulative_dt_mismatch_count": int(
                sum(
                    int(not bool(row["benchmark_time_matches_cumulative_dt"]))
                    for row in rows
                )
            ),
            "benchmark_time_to_dense_timestamps_max_abs_diff": float(
                max(
                    float(row["benchmark_time_to_dense_timestamps_max_abs_diff"])
                    for row in rows
                )
            ),
            "benchmark_time_to_cumulative_dt_max_abs_diff": float(
                max(
                    float(row["benchmark_time_to_cumulative_dt_max_abs_diff"])
                    for row in rows
                )
            ),
        }
        if any("swe_min_depth_min" in row for row in rows):
            solver_health_summary[solver_key].update(
                {
                    "swe_min_depth_min": float(
                        min(
                            float(row["swe_min_depth_min"])
                            for row in rows
                            if "swe_min_depth_min" in row
                        )
                    ),
                    "swe_max_speed_max": float(
                        max(
                            float(row["swe_max_speed_max"])
                            for row in rows
                            if "swe_max_speed_max" in row
                        )
                    ),
                    "swe_dry_cell_count_max": int(
                        max(
                            int(row["swe_dry_cell_count_max"])
                            for row in rows
                            if "swe_dry_cell_count_max" in row
                        )
                    ),
                }
            )
        if any("cg_failed_solve_count" in row for row in rows):
            solver_health_summary[solver_key].update(
                {
                    "cg_failed_step_count": int(
                        sum(
                            int(row["cg_failed_step_count"])
                            for row in rows
                            if "cg_failed_step_count" in row
                        )
                    ),
                    "cg_failed_solve_count": int(
                        sum(
                            int(row["cg_failed_solve_count"])
                            for row in rows
                            if "cg_failed_solve_count" in row
                        )
                    ),
                    "cg_max_iterations": int(
                        max(
                            int(row["cg_max_iterations"])
                            for row in rows
                            if "cg_max_iterations" in row
                        )
                    ),
                    "cg_max_residual_ratio": float(
                        max(
                            float(row["cg_max_residual_ratio"])
                            for row in rows
                            if "cg_max_residual_ratio" in row
                        )
                    ),
                }
            )

    decision_artifact = evaluate_dense_reference_criteria(
        suite_spec=suite_spec,
        zero_extrapolation_failures=zero_extrapolation_failures,
        reproduction_failure_count=reproduction_failure_count,
        aggregate_interp_to_gap_ratio=aggregate_interp_to_gap_ratio,
        scenario_ratio_median=scenario_ratio_median,
        scenario_ratio_p95=scenario_ratio_p95,
        family_summaries=family_summaries,
        aggregate_interp_to_rms_ratio=aggregate_interp_to_rms_ratio,
        scenario_rms_ratio_median=scenario_rms_ratio_median,
        scenario_rms_ratio_p95=scenario_rms_ratio_p95,
        thresholds=thresholds,
    )
    decision_path = suite_output_dir / "decision.json"
    save_json(decision_artifact, decision_path)

    summary = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "dense-reference-validation",
        "status": str(decision_artifact["status"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": {
            "name": suite_spec.name,
            "label": suite_spec.label,
            "purpose": suite_spec.purpose,
            "manuscript_claims_allowed": suite_spec.manuscript_claims_allowed,
            "count": int(len(suite_spec.ordered_scenario_ids)),
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "config_path": str(config_path) if config_path is not None else None,
            "git_commit": get_git_commit(),
        },
        "inputs": {
            "audit_artifact_path": str(audit_path),
            "scenario_selection_path": str(selection_path),
            "audit_hash": str(audit_artifact["audit_hash"]),
            "selection_audit_hash": selection_audit_hash,
            "selection_audit_hash_match": True,
            "scenario_list_hash": suite_spec.list_hash,
            "processed_test_roots": processed_root_paths,
            "raw_test_solver_roots": raw_root_paths,
        },
        "alignment": {
            "mode": MODE_COMMON_TIME,
            "common_time_grid": common_grid.tolist(),
            "common_time_horizon": float(common_grid[-1]),
            "endpoint_tolerance": float(endpoint_tolerance),
            "field": "trajectory_eta",
            "elevation_semantics": ALIGNMENT_ELEVATION_SEMANTICS,
            "time_semantics": ALIGNMENT_TIME_SEMANTICS,
            "initial_frame_treatment": ALIGNMENT_INITIAL_FRAME_TREATMENT,
            "aggregation": ALIGNMENT_AGGREGATION,
        },
        "legacy_rollout_contract": {
            "expected_natural_steps": int(expected_natural_steps),
            "expected_dense_frame_count": int(expected_natural_steps + 1),
            "legacy_knot_stride": int(legacy_knot_stride),
            "expected_legacy_frame_count": int(
                expected_natural_steps // legacy_knot_stride + 1
            ),
            "interpolation_horizon": float(interpolation_horizon),
        },
        "tolerances": {
            "timestamp_abs": float(timestamp_tolerance),
            "eta_max_abs": float(eta_tolerance),
            "reproduction_relative_rmse": float(relative_rmse_tolerance),
        },
        "criteria_thresholds": thresholds,
        "scenario_order": {
            "ordered_scenario_ids": list(suite_spec.ordered_scenario_ids),
            "ordered_scenario_hash": stable_hash_scenario_ids(
                list(suite_spec.ordered_scenario_ids)
            ),
        },
        "counts": {
            "scenario_count": int(len(suite_spec.ordered_scenario_ids)),
            "eligible_for_interpolation_count": int(
                sum(
                    int(bool(row["eligible_for_interpolation"]))
                    for row in scenario_records
                )
            ),
            "reproduction_failure_count": int(reproduction_failure_count),
            "zero_extrapolation_failure_count": int(zero_extrapolation_failures),
        },
        "pairwise_dense_solver_gap_aggregate": {
            "rmse_by_pair": aggregate_gap_by_pair,
            "smallest_pair_rmse": float(smallest_aggregate_gap),
        },
        "aggregate_metrics": {
            "interp_rmse": float(aggregate_interp_rmse),
            "trajectory_rms": float(aggregate_trajectory_rms),
            "interp_rmse_over_smallest_gap": float(aggregate_interp_to_gap_ratio),
            "interp_rmse_over_trajectory_rms": float(aggregate_interp_to_rms_ratio),
            "scenario_ratio_median": float(scenario_ratio_median),
            "scenario_ratio_p95": float(scenario_ratio_p95),
            "scenario_rms_ratio_median": float(scenario_rms_ratio_median),
            "scenario_rms_ratio_p95": float(scenario_rms_ratio_p95),
        },
        "family_summaries": family_summaries,
        "bootstrap_summaries": bootstrap_summary,
        "solver_health_summaries": solver_health_summary,
        "artifacts_written": {
            "suite_output_dir": str(suite_output_dir),
            "scenario_metrics_jsonl": str(scenario_metrics_path),
            "decision_json": str(decision_path),
            "dense_rollout_arrays_written": False,
        },
    }
    summary_path = suite_output_dir / "summary.json"
    save_json(summary, summary_path)
    summary["artifacts_written"]["summary_json"] = str(summary_path)
    save_json(summary, summary_path)
    return summary
