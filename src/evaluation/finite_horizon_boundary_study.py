from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import shutil
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    code_state,
    hash_array,
    sha256_file,
    stable_hash_payload,
)
from src.data_gen.simulate_dataset import _simulate_one_local
from src.evaluation.common_time_v2_level_a import (
    _load_canary_arrays,
    _select_canaries,
    _solver,
)
from src.solver.operator_time import build_sponge_mask


SCHEMA_ID = "tsunami-surrogate.common-time-v2.finite-horizon-boundary-study.v5"
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
EDGES = ("left", "right", "bottom", "top")
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
METRIC_NAMES = (
    "absolute_rms",
    "relative_l2",
    "interior_absolute_rms",
    "interior_relative_l2",
    "boundary_absolute_rms",
    "boundary_relative_l2",
    "amplitude_absolute_error",
    "amplitude_relative_error",
    "phase_correlation_loss",
)
REFERENCE_SCALE_NAMES = (
    "reference_rms",
    "interior_reference_rms",
    "boundary_reference_rms",
    "reference_amplitude",
)
DENOMINATOR_FLOOR_FLAGS = (
    "relative_l2_denominator_floor_used",
    "interior_denominator_floor_used",
    "boundary_denominator_floor_used",
    "amplitude_denominator_floor_used",
)
PADDING_CONTROL_THRESHOLDS = {
    "relative_l2": "shared_crop_relative_l2",
    "interior_relative_l2": "interior_relative_l2",
    "amplitude_relative_error": "amplitude_relative_error",
    "phase_correlation_loss": "phase_correlation_loss",
}
PADDING_CONVERGENCE_METRICS = {
    "relative_l2": ("absolute_rms", "reference_rms"),
    "interior_relative_l2": (
        "interior_absolute_rms",
        "interior_reference_rms",
    ),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_id") != SCHEMA_ID:
        raise ValueError("invalid finite-horizon boundary-study configuration")
    production = config["production"]
    horizon = float(production["horizon"])
    count = int(production["requested_time_count"])
    start = float(production["requested_time_start"])
    step = float(production["requested_time_step"])
    if count <= 0 or not math.isclose(start + step * (count - 1), horizon):
        raise ValueError("requested-time grid does not end at the production horizon")
    if config.get("status") != "diagnostic_unfrozen_non_decisional":
        raise ValueError("the boundary study must remain explicitly non-decisional")
    execution = config.get("execution", {})
    workers = execution.get("workers")
    maximum = execution.get("max_in_flight")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("execution.workers must be a positive integer")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("execution.max_in_flight must be a positive integer")
    if maximum < workers:
        raise ValueError("execution.max_in_flight must be at least execution.workers")
    progress_every = execution.get("progress_every")
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every < 1
    ):
        raise ValueError("execution.progress_every must be a positive integer")
    if execution.get("process_start_method") != "spawn":
        raise ValueError("finite-horizon execution requires spawn")
    expected_threads = execution.get("thread_environment")
    if expected_threads != {key: "1" for key in THREAD_KEYS}:
        raise ValueError("finite-horizon execution requires the exact single-thread policy")
    reference = config.get("large_domain_reference", {})
    offsets = reference.get("boussinesq_padding_control_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) < 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in offsets
        )
        or offsets[0] != 0
        or any(right <= left for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError(
            "Boussinesq padding-control offsets must contain at least three "
            "strictly increasing nonnegative integers starting at zero"
        )
    if reference.get("perturbation_extension") != "edge_cosine_taper":
        raise ValueError(
            "large-domain perturbation extension must be edge_cosine_taper"
        )
    taper_cells = reference.get("perturbation_taper_cells")
    if (
        isinstance(taper_cells, bool)
        or not isinstance(taper_cells, int)
        or taper_cells < 2
    ):
        raise ValueError("perturbation taper must contain at least two cells")
    if reference.get("bathymetry_extension") != "edge":
        raise ValueError("large-domain bathymetry extension must be edge")
    if reference.get("extension_status") != (
        "explicit_edge_cosine_taper_source_and_edge_bathymetry_assumption"
    ):
        raise ValueError("large-domain extension status must be explicit")
    if reference.get("production_policy_eligible") is not False:
        raise ValueError(
            "assumed large-domain extension cannot be production-policy eligible"
        )
    safety_factor = reference.get("boundary_influence_safety_factor")
    if not isinstance(safety_factor, (int, float)) or isinstance(
        safety_factor, bool
    ) or float(safety_factor) <= 1.0:
        raise ValueError("boundary-influence safety factor must exceed one")
    stencil_cells = reference.get("stencil_safety_cells")
    if (
        isinstance(stencil_cells, bool)
        or not isinstance(stencil_cells, int)
        or stencil_cells < 1
    ):
        raise ValueError("reference stencil safety cells must be positive")
    reference_cg_base = config.get("large_domain_reference", {}).get(
        "boussinesq_reference_cg_base_max_iterations"
    )
    if (
        isinstance(reference_cg_base, bool)
        or not isinstance(reference_cg_base, int)
        or reference_cg_base <= 0
    ):
        raise ValueError(
            "Boussinesq reference CG base max iterations must be positive"
        )
    fraction = config.get("proposed_future_thresholds", {}).get(
        "reference_padding_control_fraction"
    )
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not (
        0.0 < float(fraction) < 1.0
    ):
        raise ValueError("reference padding-control fraction must lie in (0, 1)")
    roundoff_safety = config.get("proposed_future_thresholds", {}).get(
        "float64_roundoff_safety_factor"
    )
    if (
        not isinstance(roundoff_safety, (int, float))
        or isinstance(roundoff_safety, bool)
        or not math.isfinite(float(roundoff_safety))
        or float(roundoff_safety) < 1.0
    ):
        raise ValueError("float64 roundoff safety factor must be at least one")
    candidates = set(config.get("static_audit", {}).get("candidate_sponges", {}))
    if not candidates:
        raise ValueError("at least one finite-domain boundary candidate is required")
    policies = config.get("candidate_policies")
    if not isinstance(policies, Mapping) or not policies:
        raise ValueError("candidate_policies must be a non-empty mapping")
    for policy_name, policy in policies.items():
        if policy_name == "larger_domain_central_crop":
            if policy != "padded_reference":
                raise ValueError("larger-domain policy must name padded_reference")
            continue
        if not isinstance(policy, Mapping) or set(policy) != set(SOLVERS):
            raise ValueError(f"policy {policy_name} must map every solver")
        if not set(str(value) for value in policy.values()) <= candidates:
            raise ValueError(f"policy {policy_name} names an unknown candidate")
    return config


def requested_times(config: Mapping[str, Any]) -> np.ndarray:
    production = config["production"]
    start = float(production["requested_time_start"])
    step = float(production["requested_time_step"])
    count = int(production["requested_time_count"])
    return start + step * np.arange(count, dtype=np.float64)


def significant_source_mask(
    eta0: np.ndarray, *, energy_tail: float
) -> tuple[np.ndarray, float]:
    values = np.asarray(eta0, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("eta0 must be one finite two-dimensional field")
    if not 0.0 < energy_tail < 1.0:
        raise ValueError("energy_tail must lie in (0, 1)")
    energy = values.ravel() ** 2
    total = float(math.fsum(float(item) for item in energy))
    if total <= 0.0:
        raise ValueError("source energy must be positive")
    order = np.argsort(-energy, kind="stable")
    cumulative = np.cumsum(energy[order], dtype=np.float64)
    count = min(
        int(np.searchsorted(cumulative, (1.0 - energy_tail) * total)) + 1,
        energy.size,
    )
    mask = np.zeros(energy.size, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(values.shape), total


def _edge_distance_fields(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    nx, ny = shape
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    return {
        "left": np.broadcast_to(x[:, None], shape),
        "right": np.broadcast_to((1.0 - x)[:, None], shape),
        "bottom": np.broadcast_to(y[None, :], shape),
        "top": np.broadcast_to((1.0 - y)[None, :], shape),
    }


def _candidate_mask(
    shape: tuple[int, int],
    candidate: Mapping[str, Any],
    *,
    global_source_union: np.ndarray | None = None,
) -> np.ndarray:
    if not bool(candidate["enabled"]):
        return np.ones(shape, dtype=np.float64)
    mask = build_sponge_mask(
        nx=shape[0],
        ny=shape[1],
        width=int(candidate["width"]),
        min_factor=float(candidate["minimum_factor"]),
        axes=str(candidate["axes"]),
        profile=str(candidate["profile"]),
    )
    if bool(candidate.get("exclude_global_source_union", False)):
        if global_source_union is None or global_source_union.shape != shape:
            raise ValueError("global source union is required for exclusion sponge")
        mask = mask.copy()
        mask[global_source_union] = 1.0
    return mask


def perturbation_edge_diagnostics(
    eta0: np.ndarray, *, absolute_floor: float
) -> dict[str, Any]:
    """Describe whether a zero exterior would alter the initial seam."""
    values = np.asarray(eta0, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("eta0 must contain a two-dimensional cell field")
    boundary = np.concatenate(
        (
            values[0, :],
            values[-1, :],
            values[1:-1, 0],
            values[1:-1, -1],
        )
    )
    peak = float(np.max(np.abs(values)))
    edge_max = float(np.max(np.abs(boundary)))
    edge_rms = float(np.sqrt(np.mean(boundary * boundary)))
    return {
        "field_peak_abs": peak,
        "edge_max_abs": edge_max,
        "edge_rms": edge_rms,
        "edge_to_peak_ratio": edge_max / max(peak, absolute_floor),
        "zero_extension_seam_jump_max": edge_max,
        "zero_extension_compatible_at_absolute_floor": edge_max <= absolute_floor,
        "selected_extension": "edge_cosine_taper",
        "selected_extension_seam_jump_max": 0.0,
    }


def audit_source_geometry(
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    global_source_union: np.ndarray | None = None,
) -> dict[str, Any]:
    eta0 = np.asarray(arrays["eta0"], dtype=np.float64)
    h0 = np.asarray(arrays["initial_depth"], dtype=np.float64)
    audit = config["static_audit"]
    tail = float(audit["source_energy_tail"])
    support, total_energy = significant_source_mask(eta0, energy_tail=tail)
    weights = eta0**2
    distances = _edge_distance_fields(eta0.shape)
    c_bound = math.sqrt(float(config["production"]["gravity"]) * float(np.max(h0)))
    band_distance = float(audit["scientific_interior_band_cells"]) / eta0.shape[0]
    edge_metrics: dict[str, Any] = {}
    for edge in EDGES:
        field = distances[edge]
        minimum = float(np.min(field[support]))
        edge_metrics[edge] = {
            "minimum_support_distance": minimum,
            "energy_weighted_distance": float(
                math.fsum(float(item) for item in (weights * field).ravel())
                / total_energy
            ),
            "earliest_boundary_arrival": minimum / c_bound,
            "earliest_reflected_return_to_source": 2.0 * minimum / c_bound,
            "earliest_reflected_entry_to_interior": (minimum + band_distance) / c_bound,
        }
    candidate_exposure: dict[str, Any] = {}
    for name, candidate in audit["candidate_sponges"].items():
        mask = _candidate_mask(
            eta0.shape,
            candidate,
            global_source_union=global_source_union,
        )
        sponge = mask < 1.0
        candidate_exposure[str(name)] = {
            "support_overlap": bool(np.any(support & sponge)),
            "source_energy_fraction_in_sponge": float(
                math.fsum(float(item) for item in weights[sponge]) / total_energy
            ),
            "damped_cell_fraction": float(np.mean(sponge)),
        }
    earliest_arrival = min(
        item["earliest_boundary_arrival"] for item in edge_metrics.values()
    )
    earliest_source_return = min(
        item["earliest_reflected_return_to_source"] for item in edge_metrics.values()
    )
    earliest_interior_return = min(
        item["earliest_reflected_entry_to_interior"] for item in edge_metrics.values()
    )
    extension_diagnostics = perturbation_edge_diagnostics(
        eta0,
        absolute_floor=float(
            config["proposed_future_thresholds"]["absolute_rms_floor"]
        ),
    )
    authoritative_split = str(row["split"])
    report_split = "val" if authoritative_split == "eval" else authoritative_split
    return {
        "split": report_split,
        "authoritative_split": authoritative_split,
        "qualified_id": str(row["qualified_id"]),
        "scenario_id": str(row["scenario_id"]),
        "sample_index": int(row["sample_index"]),
        "bathymetry_type": str(row["bathymetry_type"]),
        "source_type": str(row["source_type"]),
        "input_fingerprint": str(row["input_fingerprint"]),
        "source_energy": total_energy,
        "significant_support_cell_count": int(np.count_nonzero(support)),
        "significant_support_cell_fraction": float(np.mean(support)),
        "conservative_wave_speed_bound": c_bound,
        "earliest_boundary_arrival": earliest_arrival,
        "earliest_reflected_return_to_source": earliest_source_return,
        "earliest_reflected_entry_to_interior": earliest_interior_return,
        "boundary_reachable_by_horizon": bool(
            earliest_arrival <= float(config["production"]["horizon"])
        ),
        "reflected_source_return_possible_by_horizon": bool(
            earliest_source_return <= float(config["production"]["horizon"])
        ),
        "reflected_interior_entry_possible_by_horizon": bool(
            earliest_interior_return <= float(config["production"]["horizon"])
        ),
        "edge_metrics": edge_metrics,
        "candidate_exposure": candidate_exposure,
        "reference_extension_exposure": extension_diagnostics,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def summarize_static_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        keys = (
            (str(row["split"]), "all", "all"),
            (str(row["split"]), str(row["bathymetry_type"]), str(row["source_type"])),
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    summaries = []
    for (split, bathy, source), group in sorted(groups.items()):
        candidates = sorted(group[0]["candidate_exposure"])
        summaries.append(
            {
                "split": split,
                "bathymetry_type": bathy,
                "source_type": source,
                "count": len(group),
                "conservative_wave_speed_bound": _quantiles(
                    [float(row["conservative_wave_speed_bound"]) for row in group]
                ),
                "earliest_boundary_arrival": _quantiles(
                    [float(row["earliest_boundary_arrival"]) for row in group]
                ),
                "earliest_reflected_return_to_source": _quantiles(
                    [float(row["earliest_reflected_return_to_source"]) for row in group]
                ),
                "earliest_reflected_entry_to_interior": _quantiles(
                    [
                        float(row["earliest_reflected_entry_to_interior"])
                        for row in group
                    ]
                ),
                "boundary_reachable_count": sum(
                    bool(row["boundary_reachable_by_horizon"]) for row in group
                ),
                "reflected_source_return_possible_count": sum(
                    bool(row["reflected_source_return_possible_by_horizon"])
                    for row in group
                ),
                "reflected_interior_entry_possible_count": sum(
                    bool(row["reflected_interior_entry_possible_by_horizon"])
                    for row in group
                ),
                "reference_extension_exposure": {
                    "edge_max_abs": _quantiles(
                        [
                            float(
                                row["reference_extension_exposure"]["edge_max_abs"]
                            )
                            for row in group
                        ]
                    ),
                    "edge_rms": _quantiles(
                        [
                            float(row["reference_extension_exposure"]["edge_rms"])
                            for row in group
                        ]
                    ),
                    "edge_to_peak_ratio": _quantiles(
                        [
                            float(
                                row["reference_extension_exposure"][
                                    "edge_to_peak_ratio"
                                ]
                            )
                            for row in group
                        ]
                    ),
                    "zero_extension_compatible_count": sum(
                        bool(
                            row["reference_extension_exposure"][
                                "zero_extension_compatible_at_absolute_floor"
                            ]
                        )
                        for row in group
                    ),
                },
                "candidate_exposure": {
                    name: {
                        "support_overlap_count": sum(
                            bool(row["candidate_exposure"][name]["support_overlap"])
                            for row in group
                        ),
                        "source_energy_fraction": _quantiles(
                            [
                                float(
                                    row["candidate_exposure"][name][
                                        "source_energy_fraction_in_sponge"
                                    ]
                                )
                                for row in group
                            ]
                        ),
                    }
                    for name in candidates
                },
            }
        )
    return {"groups": summaries}


def _selection_row(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    keep = {
        "split",
        "qualified_id",
        "scenario_id",
        "sample_index",
        "bathymetry_type",
        "source_type",
        "source_strength",
        "input_fingerprint",
        "bathymetry_cache_path",
        "source_cache_path",
        "raw_sample_paths",
        "array_hashes",
    }
    return {"selection_role": role, **{key: row[key] for key in keep if key in row}}


def select_diagnostic_cases(
    inventory: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canaries = _select_canaries(inventory, 6)
    by_id = {str(row["qualified_id"]): row for row in inventory}
    train_audit = [row for row in audit_rows if row["split"] == "train"]
    source_families = sorted({str(row["source_type"]) for row in train_audit})
    per_family = int(
        config["static_audit"]["worst_risk_training_cases_per_source_family"]
    )
    worst: list[Mapping[str, Any]] = []
    for source in source_families:
        candidates = [row for row in train_audit if row["source_type"] == source]
        candidates.sort(
            key=lambda row: (
                float(row["earliest_reflected_entry_to_interior"]),
                float(row["earliest_boundary_arrival"]),
                str(row["qualified_id"]),
            )
        )
        worst.extend(candidates[:per_family])
    selected: list[dict[str, Any]] = []
    roles_by_id: dict[str, list[str]] = {}
    for row in canaries:
        roles_by_id.setdefault(str(row["qualified_id"]), []).append("level_a_canary")
    for row in worst:
        roles_by_id.setdefault(str(row["qualified_id"]), []).append(
            "static_worst_risk_by_source_family"
        )
    for qualified_id in sorted(roles_by_id):
        selected_row = _selection_row(
            by_id[qualified_id], "+".join(roles_by_id[qualified_id])
        )
        audit = next(row for row in audit_rows if row["qualified_id"] == qualified_id)
        selected_row["static_risk"] = {
            "earliest_boundary_arrival": audit["earliest_boundary_arrival"],
            "earliest_reflected_return_to_source": audit[
                "earliest_reflected_return_to_source"
            ],
            "earliest_reflected_entry_to_interior": audit[
                "earliest_reflected_entry_to_interior"
            ],
            "source_edge_max_abs": audit["reference_extension_exposure"][
                "edge_max_abs"
            ],
            "source_edge_to_peak_ratio": audit["reference_extension_exposure"][
                "edge_to_peak_ratio"
            ],
            "zero_extension_compatible_at_absolute_floor": audit[
                "reference_extension_exposure"
            ]["zero_extension_compatible_at_absolute_floor"],
        }
        selected.append(selected_row)
    return selected


def _semantic_hash(kind: str, payload: Any) -> str:
    return stable_hash_payload(artifact_kind=kind, schema_id=SCHEMA_ID, payload=payload)


def build_task_plan(
    selection: Sequence[Mapping[str, Any]], *, study_hash: str,
    config_sha256: str, selection_sha256: str, code_state_hash: str,
    source_union_hash: str,
) -> list[dict[str, Any]]:
    """Build the complete, deterministic solver-case execution plan."""
    plan: list[dict[str, Any]] = []
    for case in selection:
        for solver in SOLVERS:
            spec = {
                "study_hash": study_hash,
                "config_sha256": config_sha256,
                "selection_sha256": selection_sha256,
                "code_state_hash": code_state_hash,
                "qualified_id": str(case["qualified_id"]),
                "input_fingerprint": str(case["input_fingerprint"]),
                "source_union_hash": source_union_hash,
                "solver": solver,
                "case": {
                    "sample_index": int(case["sample_index"]),
                    "bathymetry_type": str(case["bathymetry_type"]),
                    "source_type": str(case["source_type"]),
                    "selection_role": str(case["selection_role"]),
                },
            }
            spec_hash = _semantic_hash("finite-horizon-task-spec", spec)
            plan.append({
                "ordinal": len(plan),
                "task_id": f"finite-horizon/{case['qualified_id']}/{solver}",
                "spec_hash": spec_hash,
                **spec,
            })
    _validate_task_plan(plan)
    return plan


def _validate_task_plan(plan: Sequence[Mapping[str, Any]]) -> None:
    if len(plan) != len({str(task["task_id"]) for task in plan}):
        raise RuntimeError("duplicate finite-horizon task ID")
    if len(plan) != len({str(task["spec_hash"]) for task in plan}):
        raise RuntimeError("duplicate finite-horizon task spec hash")
    if [int(task["ordinal"]) for task in plan] != list(range(len(plan))):
        raise RuntimeError("finite-horizon task ordinals are not contiguous")
    directories = [_task_directory_name(task) for task in plan]
    if len(directories) != len(set(directories)):
        raise RuntimeError("finite-horizon task directory collision")
    spec_keys = {
        "study_hash",
        "config_sha256",
        "selection_sha256",
        "code_state_hash",
        "qualified_id",
        "input_fingerprint",
        "source_union_hash",
        "solver",
        "case",
    }
    for task in plan:
        spec = {key: task[key] for key in spec_keys}
        if task["spec_hash"] != _semantic_hash("finite-horizon-task-spec", spec):
            raise RuntimeError("finite-horizon task spec hash mismatch")
        expected_id = f"finite-horizon/{task['qualified_id']}/{task['solver']}"
        if task["task_id"] != expected_id or task["solver"] not in SOLVERS:
            raise RuntimeError("finite-horizon task identity mismatch")


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _source_snapshot_entries(repo_root: Path) -> list[dict[str, Any]]:
    root = repo_root.resolve()
    included_roots = {"src", "scripts", "configs"}
    included_names = {"pyproject.toml", "requirements.txt", "requirements.lock"}
    included_suffixes = {".py", ".yaml", ".yml", ".toml", ".lock"}
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not (
            relative.parts[0] in included_roots
            or relative.as_posix() in included_names
        ):
            continue
        if (
            path.suffix not in included_suffixes
            and relative.as_posix() not in included_names
        ):
            continue
        if "__pycache__" in relative.parts:
            continue
        content = path.read_bytes()
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "content": content,
            }
        )
    return entries


def _write_source_snapshot(
    repo_root: Path, output_dir: Path, frozen_code_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Store exact evaluation sources with deterministic ZIP metadata."""
    entries = _source_snapshot_entries(repo_root)
    manifest = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "finite-horizon-source-snapshot",
        "code_state_hash": str(frozen_code_state["code_state_hash"]),
        "source_file_count": len(entries),
        "files": [
            {key: entry[key] for key in ("path", "size_bytes", "sha256")}
            for entry in entries
        ],
    }
    archive_path = output_dir / "source_snapshot.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for entry in entries:
            info = zipfile.ZipInfo(str(entry["path"]), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entry["content"])
    _write_json(output_dir / "source_snapshot_manifest.json", manifest)
    return manifest


def _validate_source_snapshot(
    output_dir: Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = output_dir / "source_snapshot_manifest.json"
    archive_path = output_dir / "source_snapshot.zip"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_id") != SCHEMA_ID
        or manifest.get("artifact_kind") != "finite-horizon-source-snapshot"
        or manifest.get("code_state_hash") != freeze.get("code_state_hash")
    ):
        raise RuntimeError("source snapshot manifest binding mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("source_file_count") != len(files):
        raise RuntimeError("source snapshot manifest coverage mismatch")
    expected = [str(item["path"]) for item in files]
    if expected != sorted(expected) or len(set(expected)) != len(expected):
        raise RuntimeError("source snapshot paths are not unique and sorted")
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != expected:
            raise RuntimeError("source snapshot ZIP coverage mismatch")
        for item in files:
            content = archive.read(str(item["path"]))
            if (
                len(content) != int(item["size_bytes"])
                or hashlib.sha256(content).hexdigest() != item["sha256"]
            ):
                raise RuntimeError("source snapshot content checksum mismatch")
    return manifest


def _freeze_self_hash(freeze: Mapping[str, Any]) -> str:
    return _semantic_hash(
        "finite-horizon-boundary-study-static-freeze",
        {key: value for key, value in freeze.items() if key != "study_hash"},
    )


def run_static_audit(
    *, repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite boundary study: {output_dir}")
    inventory_path = repo_root / str(config["authoritative_inventory"])
    inventory = _read_jsonl(inventory_path)
    if len(inventory) != 13_500:
        raise RuntimeError(
            f"expected 13,500 authoritative inputs, found {len(inventory)}"
        )
    shape = tuple(inventory[0]["array_hashes"]["eta0"]["shape"])
    union = np.zeros(shape, dtype=bool)
    loaded: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for row in inventory:
        _bathy, _source, _strength_array, _strength, arrays = _load_canary_arrays(row)
        support, _energy = significant_source_mask(
            arrays["eta0"],
            energy_tail=float(config["static_audit"]["source_energy_tail"]),
        )
        union |= support
        loaded.append((row, arrays))
    rows = [
        audit_source_geometry(row, arrays, config, global_source_union=union)
        for row, arrays in loaded
    ]
    selection = select_diagnostic_cases(inventory, rows, config)
    state = code_state(repo_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.audit-staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"static-audit staging already exists: {staging}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "static_audit_rows.jsonl", rows)
        summary = summarize_static_audit(rows)
        summary["authoritative_count"] = len(rows)
        summary["split_counts"] = {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        }
        summary["global_significant_source_union_cell_fraction"] = float(
            np.mean(union)
        )
        summary["fixed_source_exclusion_remaining_damped_cell_fraction"] = float(
            np.mean(
                _candidate_mask(
                    shape,
                    config["static_audit"]["candidate_sponges"][
                        "fixed_source_exclusion_sponge"
                    ],
                    global_source_union=union,
                )
                < 1.0
            )
        )
        _write_json(staging / "static_audit_summary.json", summary)
        _write_json(staging / "diagnostic_selection.json", selection)
        np.save(staging / "global_source_union.npy", union, allow_pickle=False)
        config_hash = sha256_file(config_path)
        inventory_hash = sha256_file(inventory_path)
        selection_hash = sha256_file(staging / "diagnostic_selection.json")
        union_raw = hash_array(union)
        union_semantic = _semantic_hash("global-significant-source-union", union_raw)
        binding_hash = _semantic_hash(
            "finite-horizon-static-bindings",
            {
                "config_sha256": config_hash,
                "inventory_sha256": inventory_hash,
                "selection_sha256": selection_hash,
                "code_state_hash": state["code_state_hash"],
                "source_union_hash": union_semantic,
                "selected": [
                    (row["qualified_id"], row["input_fingerprint"])
                    for row in selection
                ],
            },
        )
        task_plan = build_task_plan(
            selection,
            study_hash=binding_hash,
            config_sha256=config_hash,
            selection_sha256=selection_hash,
            code_state_hash=state["code_state_hash"],
            source_union_hash=union_semantic,
        )
        _write_json(staging / "task_plan.json", task_plan)
        snapshot_manifest = _write_source_snapshot(repo_root, staging, state)
        if snapshot_manifest["source_file_count"] != state["source_file_count"]:
            raise RuntimeError("source snapshot/code-state file count mismatch")
        if code_state(repo_root) != state:
            raise RuntimeError("code state changed while freezing the source snapshot")
        static_names = (
            "static_audit_rows.jsonl",
            "static_audit_summary.json",
            "diagnostic_selection.json",
            "global_source_union.npy",
            "task_plan.json",
            "source_snapshot.zip",
            "source_snapshot_manifest.json",
        )
        freeze = {
            "schema_id": SCHEMA_ID,
            "artifact_kind": "finite-horizon-boundary-study-static-freeze",
            "status": config["status"],
            "config_sha256": config_hash,
            "inventory_sha256": inventory_hash,
            "code_state": state,
            "code_state_hash": state["code_state_hash"],
            "source_snapshot": {
                "archive": "source_snapshot.zip",
                "manifest": "source_snapshot_manifest.json",
                "source_file_count": snapshot_manifest["source_file_count"],
            },
            "selection_sha256": selection_hash,
            "static_files": {
                name: _file_record(staging / name) for name in static_names
            },
            "source_union": {
                "raw": union_raw,
                "semantic_sha256": union_semantic,
            },
            "task_plan_binding_hash": binding_hash,
            "task_count": len(task_plan),
            "thresholds_frozen_before_selected_case_execution": _json_safe(
                config["proposed_future_thresholds"]
            ),
            "candidate_policies_frozen_before_selected_case_execution": _json_safe(
                config["candidate_policies"]
            ),
            "selected_case_count": len(selection),
            "selected_qualified_ids": [row["qualified_id"] for row in selection],
            "created_before_numerical_outcomes": True,
        }
        freeze["study_hash"] = _freeze_self_hash(freeze)
        _write_json(staging / "STATIC_FREEZE.json", freeze)
        os.replace(staging, output_dir)
        return {"summary": summary, "selection": selection, "freeze": freeze}
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def extend_common_domain(
    bathymetry: np.ndarray,
    eta0: np.ndarray,
    h0: np.ndarray,
    *,
    pad_cells: int,
    perturbation_extension: str = "edge_cosine_taper",
    perturbation_taper_cells: int = 16,
) -> dict[str, Any]:
    if pad_cells <= 0:
        raise ValueError("pad_cells must be positive")
    bathy = np.asarray(bathymetry, dtype=np.float64)
    eta = np.asarray(eta0, dtype=np.float64)
    depth = np.asarray(h0, dtype=np.float64)
    if bathy.shape != eta.shape or bathy.shape != depth.shape:
        raise ValueError("bathymetry, eta0, and h0 must share one shape")
    if perturbation_extension not in ("edge_cosine_taper", "zero"):
        raise ValueError("unsupported perturbation extension")
    if (
        isinstance(perturbation_taper_cells, bool)
        or not isinstance(perturbation_taper_cells, int)
        or perturbation_taper_cells < 2
    ):
        raise ValueError("perturbation taper must contain at least two cells")
    if (
        perturbation_extension == "edge_cosine_taper"
        and perturbation_taper_cells >= pad_cells
    ):
        raise ValueError("perturbation taper must leave a zero exterior guard")
    pad = ((pad_cells, pad_cells), (pad_cells, pad_cells))
    extended_bathy = np.pad(bathy, pad, mode="edge")
    crop = (
        slice(pad_cells, pad_cells + bathy.shape[0]),
        slice(pad_cells, pad_cells + bathy.shape[1]),
    )
    if perturbation_extension == "zero":
        extended_eta = np.zeros_like(extended_bathy)
        extended_eta[crop] = eta
        taper = np.zeros_like(extended_bathy)
        taper[crop] = 1.0
        support_cells = 0
    else:
        extended_eta = np.pad(eta, pad, mode="edge")
        nx_extended, ny_extended = extended_eta.shape
        x_index = np.arange(nx_extended, dtype=np.int64)
        y_index = np.arange(ny_extended, dtype=np.int64)
        x_distance = np.maximum(
            np.maximum(pad_cells - x_index, 0),
            np.maximum(x_index - (pad_cells + bathy.shape[0] - 1), 0),
        )
        y_distance = np.maximum(
            np.maximum(pad_cells - y_index, 0),
            np.maximum(y_index - (pad_cells + bathy.shape[1] - 1), 0),
        )
        outside_layers = np.maximum(x_distance[:, None], y_distance[None, :])
        taper_coordinate = np.clip(
            (outside_layers.astype(np.float64) - 1.0)
            / float(perturbation_taper_cells - 1),
            0.0,
            1.0,
        )
        taper = 0.5 * (1.0 + np.cos(math.pi * taper_coordinate))
        taper[outside_layers == 0] = 1.0
        taper[outside_layers >= perturbation_taper_cells] = 0.0
        extended_eta *= taper
        support_cells = perturbation_taper_cells
    rest = np.maximum(-bathy, 0.0)
    extended_rest = np.maximum(-extended_bathy, 0.0)
    depth_perturbation = depth - rest
    extended_depth_perturbation = np.pad(
        depth_perturbation, pad, mode="edge"
    ) * taper
    extended_h = np.maximum(extended_rest + extended_depth_perturbation, 0.0)
    # Preserve the authoritative stored initial depth bit-for-bit after dtype
    # promotion. Reconstruct only the newly introduced exterior rest state.
    extended_h[crop] = depth
    if not np.array_equal(extended_bathy[crop], bathy):
        raise RuntimeError("padded bathymetry crop is not identical")
    if not np.array_equal(extended_eta[crop], eta):
        raise RuntimeError("padded eta crop is not identical")
    if not np.array_equal(extended_h[crop], depth):
        raise RuntimeError("padded initial-depth crop is not identical")
    def seam_jump(extended_values: np.ndarray, crop_values: np.ndarray) -> float:
        return max(
            float(
                np.max(
                    np.abs(
                        extended_values[
                            pad_cells - 1,
                            pad_cells : pad_cells + crop_values.shape[1],
                        ]
                        - crop_values[0, :]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        extended_values[
                            pad_cells + crop_values.shape[0],
                            pad_cells : pad_cells + crop_values.shape[1],
                        ]
                        - crop_values[-1, :]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        extended_values[
                            pad_cells : pad_cells + crop_values.shape[0],
                            pad_cells - 1,
                        ]
                        - crop_values[:, 0]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        extended_values[
                            pad_cells : pad_cells + crop_values.shape[0],
                            pad_cells + crop_values.shape[1],
                        ]
                        - crop_values[:, -1]
                    )
                )
            ),
        )

    eta_seam_jump = seam_jump(extended_eta, eta)
    h_seam_jump = seam_jump(extended_h, depth)
    maximum_seam_jump = max(eta_seam_jump, h_seam_jump)
    return {
        "bathymetry": extended_bathy,
        "eta0": extended_eta,
        "h0": extended_h,
        "crop": crop,
        "pad_cells": pad_cells,
        "perturbation_support_cells": support_cells,
        "seam_jump_max": maximum_seam_jump,
        "extension": {
            "bathymetry": "constant edge continuation",
            "perturbation": (
                "zero outside authoritative crop"
                if perturbation_extension == "zero"
                else "edge-matched fixed-width cosine taper to zero"
            ),
            "perturbation_extension": perturbation_extension,
            "perturbation_taper_cells": (
                0
                if perturbation_extension == "zero"
                else perturbation_taper_cells
            ),
            "eta_seam_jump_max": eta_seam_jump,
            "natural_depth_seam_jump_max": h_seam_jump,
            "seam_jump_max": maximum_seam_jump,
            "rest_state": "max(-extended_bathymetry, 0)",
            "scientific_status": (
                "explicit_assumption_only; authoritative exterior source and "
                "bathymetry parameters are unavailable in the input cache"
            ),
        },
    }


def reference_padding_cells(
    *, wave_speed_bound: float, horizon: float, dx: float, config: Mapping[str, Any]
) -> int:
    reference = config["large_domain_reference"]
    # The taper may seed a perturbation inside the padding. Its closest
    # possible outer-boundary reflection traverses two padding widths minus
    # the fixed taper support before it can re-enter the authoritative crop.
    support_cells = int(reference["perturbation_taper_cells"])
    required_path_cells = (
        float(reference["boundary_influence_safety_factor"])
        * wave_speed_bound
        * horizon
        / dx
    )
    stencil_cells = int(reference["stencil_safety_cells"])
    return max(
        support_cells + stencil_cells,
        int(math.ceil(0.5 * (required_path_cells + support_cells)))
        + stencil_cells,
    )


def reference_boundary_influence_time(
    *,
    pad_cells: int,
    perturbation_support_cells: int,
    dx: float,
    wave_speed_bound: float,
) -> float:
    path_cells = 2 * pad_cells - perturbation_support_cells
    if path_cells <= 0:
        raise ValueError("reference padding must exceed perturbation support")
    return path_cells * dx / wave_speed_bound


def boussinesq_reference_cg_max_iterations(
    shape: Sequence[int], config: Mapping[str, Any]
) -> int:
    """Scale only the padded-reference CG work budget with its axis length."""
    if len(shape) != 2 or min(int(value) for value in shape) <= 1:
        raise ValueError("Boussinesq reference shape must contain two valid axes")
    production_grid = int(config["production"]["grid"])
    base = int(
        config["large_domain_reference"][
            "boussinesq_reference_cg_base_max_iterations"
        ]
    )
    return max(base, int(math.ceil(base * max(shape) / production_grid)))


def _install_delayed_sponge(solver: Any, start_time: float) -> None:
    if start_time <= 0.0 or not solver.use_sponge:
        return
    original = solver.apply_sponge_layer
    elapsed = 0.0

    def delayed(*, dt: float | None = None) -> None:
        nonlocal elapsed
        step_dt = float(solver.dt if dt is None else dt)
        before = elapsed
        elapsed += step_dt
        active_dt = max(0.0, elapsed - start_time) - max(0.0, before - start_time)
        if active_dt > 0.0:
            original(dt=active_dt)

    solver.apply_sponge_layer = delayed


def _simulate(
    solver_name: str,
    *,
    bathymetry: np.ndarray,
    eta0: np.ndarray,
    h0: np.ndarray,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    times: np.ndarray,
    sponge_mask: np.ndarray | None = None,
    linear_solver_max_iter: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    nx, ny = bathymetry.shape
    enabled = bool(candidate["enabled"])
    solver = _solver(
        solver_name,
        nx=nx,
        ny=ny,
        cfl=float(config["production"]["cfl"][solver_name]),
        boundary="open",
        use_sponge=enabled,
        sponge_mode="elapsed_time_consistent" if enabled else "legacy_per_step",
        filter_mode="disabled",
        sponge_axes=str(candidate["axes"]),
        sponge_width=int(candidate["width"]),
        sponge_min_factor=float(candidate["minimum_factor"]),
        sponge_profile=str(candidate["profile"]),
        dx=1.0 / int(config["production"]["grid"]),
        dy=1.0 / int(config["production"]["grid"]),
    )
    if linear_solver_max_iter is not None:
        if solver_name != "boussinesq" or linear_solver_max_iter <= 0:
            raise ValueError(
                "a positive evaluation-only CG budget is valid for Boussinesq only"
            )
        solver.linear_solver_max_iter = int(linear_solver_max_iter)
    solver.set_bathymetry(np.asarray(bathymetry, dtype=np.float64))
    if solver_name == "boussinesq":
        solver.set_initial_condition(
            np.asarray(eta0, dtype=np.float64),
            eta_t0=np.zeros_like(eta0, dtype=np.float64),
        )
    else:
        solver.set_initial_condition(
            np.asarray(h0, dtype=np.float64),
            hu0=np.zeros_like(h0, dtype=np.float64),
            hv0=np.zeros_like(h0, dtype=np.float64),
        )
    if sponge_mask is not None:
        if sponge_mask.shape != (nx, ny):
            raise ValueError("study sponge mask shape mismatch")
        solver.sponge_mask = np.asarray(sponge_mask, dtype=np.float64).copy()
        solver.reset_operator_diagnostics()
    _install_delayed_sponge(solver, float(candidate.get("start_time", 0.0)))
    started = time.monotonic()
    states, emitted, dt_history, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=float(config["production"]["cfl"][solver_name]),
        include_initial_state=False,
        requested_times=np.asarray(times, dtype=np.float64),
        max_natural_steps=20_000,
        collect_natural_step_health=True,
        requested_state_dtype=np.float64,
    )
    if not np.array_equal(emitted, times):
        raise RuntimeError("requested timestamps changed")
    eta = states[:, 0] if solver_name == "boussinesq" else states[:, 0] + bathymetry
    operator = solver.get_operator_diagnostics()
    return np.asarray(eta, dtype=np.float64), {
        "runtime_s": time.monotonic() - started,
        "natural_steps": int(dt_history.size),
        "finite": bool(np.isfinite(states).all()),
        "measurement_dtype": str(states.dtype),
        "requested_output_count": int(emitted.size),
        "requested_times_exact": True,
        "requested_times": emitted.tolist(),
        "max_post_step_cfl": float(
            np.max(np.asarray(diagnostics["post_step_cfl"], dtype=np.float64))
        ),
        "cg_failure_count": int(
            np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=np.int64))
        ),
        "cg_iterations_max": int(operator.get("cg_iterations_max", 0)),
        "cg_iterations_sum": int(operator.get("cg_iterations_sum", 0)),
        "cg_residual_ratio_max": float(operator.get("cg_residual_ratio_max", 0.0)),
        "cg_max_iteration_limit": int(
            getattr(solver, "linear_solver_max_iter", 0)
        ),
        "operator": operator,
    }


def comparison_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    boundary_band_cells: int,
    absolute_floor: float,
) -> list[dict[str, float]]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("candidate/reference trajectories must share [T,X,Y] shape")
    nx, ny = left.shape[1:]
    band = min(boundary_band_cells, max(1, min(nx, ny) // 2 - 1))
    boundary = np.zeros((nx, ny), dtype=bool)
    boundary[:band, :] = True
    boundary[-band:, :] = True
    boundary[:, :band] = True
    boundary[:, -band:] = True
    interior = ~boundary
    rows: list[dict[str, float]] = []
    for index in range(left.shape[0]):
        candidate_frame = left[index]
        reference_frame = right[index]
        diff = candidate_frame - reference_frame

        def region(mask: np.ndarray) -> tuple[float, float, float, bool]:
            delta = diff[mask]
            target = reference_frame[mask]
            abs_rms = float(np.sqrt(np.mean(delta * delta)))
            target_rms = float(np.sqrt(np.mean(target * target)))
            return (
                abs_rms,
                abs_rms / max(target_rms, absolute_floor),
                target_rms,
                target_rms < absolute_floor,
            )

        full_abs = float(np.sqrt(np.mean(diff * diff)))
        reference_rms = float(np.sqrt(np.mean(reference_frame * reference_frame)))
        interior_abs, interior_rel, interior_reference_rms, interior_floor_used = (
            region(interior)
        )
        boundary_abs, boundary_rel, boundary_reference_rms, boundary_floor_used = (
            region(boundary)
        )
        reference_amplitude = float(np.max(np.abs(reference_frame)))
        left_norm = float(np.linalg.norm(candidate_frame.ravel()))
        right_norm = float(np.linalg.norm(reference_frame.ravel()))
        correlation = float(
            np.vdot(candidate_frame.ravel(), reference_frame.ravel()).real
            / max(left_norm * right_norm, absolute_floor**2)
        )
        rows.append(
            {
                "absolute_rms": full_abs,
                "relative_l2": full_abs / max(reference_rms, absolute_floor),
                "reference_rms": reference_rms,
                "relative_l2_denominator_floor_used": (
                    reference_rms < absolute_floor
                ),
                "interior_absolute_rms": interior_abs,
                "interior_relative_l2": interior_rel,
                "interior_reference_rms": interior_reference_rms,
                "interior_denominator_floor_used": interior_floor_used,
                "boundary_absolute_rms": boundary_abs,
                "boundary_relative_l2": boundary_rel,
                "boundary_reference_rms": boundary_reference_rms,
                "boundary_denominator_floor_used": boundary_floor_used,
                "amplitude_absolute_error": abs(
                    float(np.max(np.abs(candidate_frame)))
                    - float(np.max(np.abs(reference_frame)))
                ),
                "amplitude_relative_error": abs(
                    float(np.max(np.abs(candidate_frame)))
                    - float(np.max(np.abs(reference_frame)))
                )
                / max(reference_amplitude, absolute_floor),
                "reference_amplitude": reference_amplitude,
                "amplitude_denominator_floor_used": (
                    reference_amplitude < absolute_floor
                ),
                "phase_correlation_loss": max(0.0, 1.0 - correlation),
            }
        )
    return rows


def padding_control_diagnostics(
    reference_crops: Sequence[np.ndarray],
    offsets: Sequence[int],
    *,
    qualified_id: str,
    times: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(reference_crops) < 3 or len(reference_crops) != len(offsets):
        raise ValueError("padding control requires at least three aligned references")
    normalized_offsets = [int(value) for value in offsets]
    if (
        normalized_offsets[0] != 0
        or any(
            right <= left
            for left, right in zip(normalized_offsets, normalized_offsets[1:])
        )
    ):
        raise ValueError("padding-control offsets must start at zero and increase")
    expected_shape = (
        times.size,
        int(config["production"]["grid"]),
        int(config["production"]["grid"]),
    )
    crops = [np.asarray(crop, dtype=np.float64) for crop in reference_crops]
    if any(crop.shape != expected_shape for crop in crops):
        raise RuntimeError("Boussinesq padding-control crops are not coordinate aligned")
    fraction = float(
        config["proposed_future_thresholds"][
            "reference_padding_control_fraction"
        ]
    )
    allowances = {
        metric: fraction
        * float(config["proposed_future_thresholds"][threshold_name])
        for metric, threshold_name in PADDING_CONTROL_THRESHOLDS.items()
    }
    rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for pair_index, (lower, upper) in enumerate(zip(crops, crops[1:])):
        lower_offset = normalized_offsets[pair_index]
        upper_offset = normalized_offsets[pair_index + 1]
        metrics = comparison_metrics(
            lower,
            upper,
            boundary_band_cells=int(
                config["static_audit"]["scientific_interior_band_cells"]
            ),
            absolute_floor=float(
                config["proposed_future_thresholds"]["absolute_rms_floor"]
            ),
        )
        for timestamp, metric_values in zip(times, metrics, strict=True):
            adequate_by_metric = {
                metric: float(metric_values[metric]) <= allowance
                for metric, allowance in allowances.items()
            }
            rows.append(
                {
                    "qualified_id": qualified_id,
                    "solver": "boussinesq",
                    "control_pair_index": pair_index,
                    "control_pair": f"P+{lower_offset}_to_P+{upper_offset}",
                    "lower_offset_cells": lower_offset,
                    "upper_offset_cells": upper_offset,
                    "requested_time": float(timestamp),
                    **metric_values,
                    "allowances": allowances,
                    "adequate_by_metric": adequate_by_metric,
                    "adequate": all(adequate_by_metric.values()),
                }
            )
        maxima = {
            metric: max(float(item[metric]) for item in metrics)
            for metric in allowances
        }
        convergence_maxima = {
            measure: max(float(item[measure]) for item in metrics)
            for measure, _scale in PADDING_CONVERGENCE_METRICS.values()
        }
        convergence_reference_scales = {
            scale: max(float(item[scale]) for item in metrics)
            for _measure, scale in PADDING_CONVERGENCE_METRICS.values()
        }
        adequate_by_metric = {
            metric: maxima[metric] <= allowances[metric] for metric in allowances
        }
        pair_summaries.append(
            {
                "control_pair_index": pair_index,
                "control_pair": f"P+{lower_offset}_to_P+{upper_offset}",
                "lower_offset_cells": lower_offset,
                "upper_offset_cells": upper_offset,
                "maxima": maxima,
                "convergence_maxima": convergence_maxima,
                "convergence_reference_scales": convergence_reference_scales,
                "adequate_by_metric": adequate_by_metric,
                "adequate": all(adequate_by_metric.values()),
            }
        )
    previous = pair_summaries[-2]["maxima"]
    final = pair_summaries[-1]["maxima"]
    previous_convergence = pair_summaries[-2]["convergence_maxima"]
    final_convergence = pair_summaries[-1]["convergence_maxima"]
    roundoff_safety = float(
        config["proposed_future_thresholds"]["float64_roundoff_safety_factor"]
    )
    convergence_precision_tolerances = {
        metric: roundoff_safety
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(
                pair_summaries[-2]["convergence_reference_scales"][scale]
            ),
            float(pair_summaries[-1]["convergence_reference_scales"][scale]),
        )
        for metric, (_measure, scale) in PADDING_CONVERGENCE_METRICS.items()
    }
    nonincreasing = {}
    for metric in allowances:
        if metric in PADDING_CONVERGENCE_METRICS:
            measure, _scale = PADDING_CONVERGENCE_METRICS[metric]
            nonincreasing[metric] = float(final_convergence[measure]) <= (
                float(previous_convergence[measure])
                + convergence_precision_tolerances[metric]
            )
        else:
            nonincreasing[metric] = float(final[metric]) <= float(previous[metric])
    successive_error_ratio = {
        metric: float(final[metric])
        / max(float(previous[metric]), np.finfo(np.float64).tiny)
        for metric in allowances
    }
    monotonicity_required = {
        metric: metric in PADDING_CONVERGENCE_METRICS for metric in allowances
    }
    convergence_requirement_met = {
        metric: (not monotonicity_required[metric]) or nonincreasing[metric]
        for metric in allowances
    }
    summary = {
        "allowances": allowances,
        "offsets": normalized_offsets,
        "pairs": pair_summaries,
        "maxima": final,
        "adequate_by_metric": pair_summaries[-1]["adequate_by_metric"],
        "nonincreasing_by_metric": nonincreasing,
        "monotonicity_required_by_metric": monotonicity_required,
        "convergence_requirement_met_by_metric": convergence_requirement_met,
        "convergence_measure_by_metric": {
            metric: (
                PADDING_CONVERGENCE_METRICS[metric][0]
                if metric in PADDING_CONVERGENCE_METRICS
                else None
            )
            for metric in allowances
        },
        "convergence_precision_tolerance_by_metric": {
            metric: convergence_precision_tolerances.get(metric)
            for metric in allowances
        },
        "successive_error_ratio": successive_error_ratio,
        "control_description": (
            "successive frozen padding references; the largest crop is the "
            "candidate-comparison reference, and the final adjacent difference "
            "must be within allowance; norm discrepancies must also be no larger "
            "than the preceding norm discrepancies"
        ),
        "absolute_rms_floor_role": "denominator_floor_only_not_error_ceiling",
    }
    summary["adequate"] = all(summary["adequate_by_metric"].values()) and all(
        convergence_requirement_met.values()
    )
    return rows, summary


def _execute_case_task(task: Mapping[str, Any]) -> dict[str, Any]:
    config = task["config"]
    row = task["case_record"]
    _bathy, _source, _strength_array, _strength, arrays = _load_canary_arrays(row)
    bathymetry = np.asarray(arrays["bathymetry"], dtype=np.float64)
    eta0 = np.asarray(arrays["eta0"], dtype=np.float64)
    h0 = np.asarray(arrays["initial_depth"], dtype=np.float64)
    solver_name = str(task["solver"])
    times = requested_times(config)
    speed = math.sqrt(float(config["production"]["gravity"]) * float(np.max(h0)))
    dx = 1.0 / int(config["production"]["grid"])
    pads = reference_padding_cells(
        wave_speed_bound=speed,
        horizon=float(config["production"]["horizon"]),
        dx=dx,
        config=config,
    )
    reference_config = config["large_domain_reference"]
    extension_kwargs = {
        "perturbation_extension": str(
            reference_config["perturbation_extension"]
        ),
        "perturbation_taper_cells": int(
            reference_config["perturbation_taper_cells"]
        ),
    }
    extended = extend_common_domain(
        bathymetry,
        eta0,
        h0,
        pad_cells=pads,
        **extension_kwargs,
    )
    if float(extended["seam_jump_max"]) != 0.0:
        raise RuntimeError("selected reference extension changed the initial seam")
    no_sponge = {
        "enabled": False,
        "axes": "xy",
        "width": 0,
        "minimum_factor": 1.0,
        "profile": "quadratic",
    }
    base_reference, base_reference_health = _simulate(
        solver_name,
        bathymetry=extended["bathymetry"],
        eta0=extended["eta0"],
        h0=extended["h0"],
        config=config,
        candidate=no_sponge,
        times=times,
        linear_solver_max_iter=(
            boussinesq_reference_cg_max_iterations(
                extended["bathymetry"].shape, config
            )
            if solver_name == "boussinesq"
            else None
        ),
    )
    crop = extended["crop"]
    base_reference_crop = base_reference[:, crop[0], crop[1]]
    reference_crop = base_reference_crop
    padding_offsets = [0]
    padding_reference_health: list[dict[str, Any]] = [
        {"offset_cells": 0, "health": base_reference_health}
    ]
    padding_control_rows: list[dict[str, Any]] = []
    padding_control = None
    if solver_name == "boussinesq":
        padding_offsets = [
            int(value)
            for value in config["large_domain_reference"][
                "boussinesq_padding_control_offsets"
            ]
        ]
        reference_crops = [base_reference_crop]
        reference_shapes = [list(extended["bathymetry"].shape)]
        for offset in padding_offsets[1:]:
            enlarged = extend_common_domain(
                bathymetry,
                eta0,
                h0,
                pad_cells=pads + offset,
                **extension_kwargs,
            )
            enlarged_reference, enlarged_health = _simulate(
                solver_name,
                bathymetry=enlarged["bathymetry"],
                eta0=enlarged["eta0"],
                h0=enlarged["h0"],
                config=config,
                candidate=no_sponge,
                times=times,
                linear_solver_max_iter=boussinesq_reference_cg_max_iterations(
                    enlarged["bathymetry"].shape, config
                ),
            )
            enlarged_crop = enlarged["crop"]
            crop_values = enlarged_reference[
                :, enlarged_crop[0], enlarged_crop[1]
            ]
            if crop_values.shape != (times.size, *bathymetry.shape):
                raise RuntimeError("enlarged Boussinesq reference crop is misaligned")
            reference_crops.append(crop_values)
            reference_shapes.append(list(enlarged["bathymetry"].shape))
            padding_reference_health.append(
                {"offset_cells": offset, "health": enlarged_health}
            )
        reference_crop = reference_crops[-1]
        padding_control_rows, padding_control = padding_control_diagnostics(
            reference_crops,
            padding_offsets,
            qualified_id=str(row["qualified_id"]),
            times=times,
            config=config,
        )
        padding_control.update(
            {
                "base_pad_cells": pads,
                "absolute_pad_cells": [pads + value for value in padding_offsets],
                "padded_shapes": reference_shapes,
            }
        )
    rows: list[dict[str, Any]] = []
    candidate_health: dict[str, Any] = {}
    union = np.asarray(task["global_source_union"], dtype=bool)
    for candidate_name, candidate in config["static_audit"][
        "candidate_sponges"
    ].items():
        mask = _candidate_mask(
            bathymetry.shape,
            candidate,
            global_source_union=union,
        )
        trajectory, health = _simulate(
            solver_name,
            bathymetry=bathymetry,
            eta0=eta0,
            h0=h0,
            config=config,
            candidate=candidate,
            times=times,
            sponge_mask=mask,
        )
        metrics = comparison_metrics(
            trajectory,
            reference_crop,
            boundary_band_cells=int(
                config["static_audit"]["scientific_interior_band_cells"]
            ),
            absolute_floor=float(
                config["proposed_future_thresholds"]["absolute_rms_floor"]
            ),
        )
        for timestamp, metric in zip(times, metrics, strict=True):
            rows.append(
                {
                    "qualified_id": row["qualified_id"],
                    "selection_role": row["selection_role"],
                    "bathymetry_type": row["bathymetry_type"],
                    "source_type": row["source_type"],
                    "solver": solver_name,
                    "candidate": candidate_name,
                    "requested_time": float(timestamp),
                    **metric,
                }
            )
        candidate_health[str(candidate_name)] = health
    return {
        "qualified_id": row["qualified_id"],
        "input_fingerprint": row["input_fingerprint"],
        "solver": solver_name,
        "requested_times": times.tolist(),
        "pad_cells": pads,
        "comparison_pad_cells": (
            pads
            if solver_name != "boussinesq"
            else pads + padding_offsets[-1]
        ),
        "padded_shape": list(extended["bathymetry"].shape),
        "authoritative_crop": {
            "shape": list(bathymetry.shape),
            "base_origin": [pads, pads],
            "comparison_origin": [
                pads
                if solver_name != "boussinesq"
                else pads + padding_offsets[-1],
                pads
                if solver_name != "boussinesq"
                else pads + padding_offsets[-1],
            ],
        },
        "reference_boundary_influence_time_bound": reference_boundary_influence_time(
            pad_cells=pads,
            perturbation_support_cells=int(
                extended["perturbation_support_cells"]
            ),
            dx=dx,
            wave_speed_bound=speed,
        ),
        "production_horizon": float(config["production"]["horizon"]),
        "reference_return_safe": bool(
            reference_boundary_influence_time(
                pad_cells=pads,
                perturbation_support_cells=int(
                    extended["perturbation_support_cells"]
                ),
                dx=dx,
                wave_speed_bound=speed,
            )
            > float(config["production"]["horizon"])
        ),
        "extension": extended["extension"],
        "reference_health": base_reference_health,
        "base_reference_health": base_reference_health,
        "comparison_reference_health": padding_reference_health[-1]["health"],
        "padding_offsets": padding_offsets,
        "padding_reference_health": padding_reference_health,
        "reference_adequate": True if padding_control is None else padding_control["adequate"],
        "boundary_conclusion_available": True if padding_control is None else padding_control["adequate"],
        "candidate_health": candidate_health,
        "padding_control": padding_control,
        "padding_control_rows": padding_control_rows,
        "rows": rows,
        "worker": {
            "pid": os.getpid(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        },
    }


_OPERATIONAL_KEYS = {"runtime_s", "pid", "platform", "python_version", "worker", "workers"}


def _scientific_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _scientific_payload(item) for key, item in sorted(value.items())
                if str(key) not in _OPERATIONAL_KEYS}
    if isinstance(value, (list, tuple)):
        return [_scientific_payload(item) for item in value]
    return _json_safe(value)


def scientific_digest(result: Mapping[str, Any]) -> str:
    return _semantic_hash("finite-horizon-task-scientific-result", _scientific_payload(result))


def _task_directory_name(task: Mapping[str, Any]) -> str:
    return f"{int(task['ordinal']):03d}-{str(task['spec_hash'])[:16]}"


def _write_task_artifact(task_root: Path, task: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    task_root.mkdir(parents=True, exist_ok=True)
    final = task_root / _task_directory_name(task)
    if final.exists():
        raise RuntimeError(f"completed task already exists: {final.name}")
    staging = task_root / f".{final.name}.staging-{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"task staging collision: {staging.name}")
    staging.mkdir()
    digest = scientific_digest(result)
    manifest = {
        "schema_id": SCHEMA_ID, "artifact_kind": "finite-horizon-solver-case-task",
        "status": "complete", "task_id": str(task["task_id"]),
        "directory_name": _task_directory_name(task),
        "ordinal": int(task.get("ordinal", 0)), "spec_hash": str(task.get("spec_hash", "")),
        "qualified_id": str(task["qualified_id"]), "solver": str(task["solver"]),
        "scientific_digest": digest,
    }
    try:
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "result.json", result)
        sums = {name: sha256_file(staging / name) for name in ("manifest.json", "result.json")}
        (staging / "SHA256SUMS.txt").write_text(
            "".join(f"{value}  {name}\n" for name, value in sums.items()), encoding="utf-8")
        os.replace(staging, final)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _load_task_artifact(task_root, task)


def _load_task_artifact(task_root: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    directory = task_root / _task_directory_name(task)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    expected_files = {"manifest.json", "result.json", "SHA256SUMS.txt"}
    if {path.name for path in directory.iterdir()} != expected_files:
        raise RuntimeError(f"invalid task file set: {directory.name}")
    lines = (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or any(line.count("  ") != 1 for line in lines):
        raise RuntimeError("invalid task checksum manifest syntax")
    entries = [line.split("  ", 1) for line in lines]
    if len({name for _digest, name in entries}) != len(entries):
        raise RuntimeError("duplicate task checksum entry")
    checksums = {name: digest for digest, name in entries}
    if set(checksums) != {"manifest.json", "result.json"}:
        raise RuntimeError("invalid task checksum coverage")
    if any(sha256_file(directory / name) != digest for name, digest in checksums.items()):
        raise RuntimeError("task checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if manifest.get("schema_id") != SCHEMA_ID or manifest.get("status") != "complete":
        raise RuntimeError("invalid task manifest schema/status")
    checks = (("task_id", str(task["task_id"])),
              ("directory_name", _task_directory_name(task)),
              ("ordinal", int(task.get("ordinal", 0))),
              ("spec_hash", str(task.get("spec_hash", ""))),
              ("qualified_id", str(task["qualified_id"])), ("solver", str(task["solver"])))
    if any(manifest.get(key) != value for key, value in checks):
        raise RuntimeError("task identity/spec mismatch")
    candidates = set(task["config"]["static_audit"]["candidate_sponges"])
    times = requested_times(task["config"])
    if (result.get("qualified_id") != task["qualified_id"] or
            result.get("input_fingerprint") != task["input_fingerprint"] or
            result.get("solver") != task["solver"]):
        raise RuntimeError("task result case/solver mismatch")
    if result.get("requested_times") != times.tolist():
        raise RuntimeError("task requested-time identity mismatch")
    if set(result.get("candidate_health", {})) != candidates:
        raise RuntimeError("task candidate health coverage mismatch")
    for health in result["candidate_health"].values():
        if (health.get("measurement_dtype") != "float64" or
                not health.get("finite") or
                health.get("cg_failure_count") != 0 or
                health.get("requested_output_count") != times.size or
                health.get("requested_times") != times.tolist() or
                health.get("requested_times_exact") is not True):
            raise RuntimeError("task candidate health is invalid")
    reference_health = result.get("base_reference_health")
    if (not isinstance(reference_health, dict) or
            reference_health.get("measurement_dtype") != "float64" or
            not reference_health.get("finite") or
            reference_health.get("cg_failure_count") != 0 or
            reference_health.get("requested_times") != times.tolist()):
        raise RuntimeError("task base reference health is invalid")
    rows = result.get("rows", [])
    if len(rows) != len(candidates) * times.size:
        raise RuntimeError("task metric row count mismatch")
    if set(row.get("candidate") for row in rows) != candidates or sorted({row.get("requested_time") for row in rows}) != times.tolist():
        raise RuntimeError("task candidates/requested times mismatch")
    if any(sum(row.get("candidate") == candidate and row.get("requested_time") == float(timestamp)
               for row in rows) != 1 for candidate in candidates for timestamp in times):
        raise RuntimeError("task candidate/time rows are not one-to-one")
    if any(
        metric not in row or not math.isfinite(float(row[metric]))
        for row in rows
        for metric in (*METRIC_NAMES, *REFERENCE_SCALE_NAMES)
    ):
        raise RuntimeError("task metric row is missing or nonfinite")
    if any(
        not isinstance(row.get(flag), bool)
        for row in rows
        for flag in DENOMINATOR_FLOOR_FLAGS
    ):
        raise RuntimeError("task metric denominator-floor flags are invalid")
    if result.get("reference_return_safe") is not True:
        raise RuntimeError("task reference is not return-safe")
    extension = result.get("extension")
    if (
        not isinstance(extension, Mapping)
        or extension.get("perturbation_extension") != "edge_cosine_taper"
        or extension.get("seam_jump_max") != 0.0
    ):
        raise RuntimeError("task reference extension is invalid")
    is_bouss = task["solver"] == "boussinesq"
    if (result.get("padding_control") is not None) != is_bouss or bool(result.get("padding_control_rows")) != is_bouss:
        raise RuntimeError("task padding-control presence mismatch")
    expected_offsets = (
        [
            int(value)
            for value in task["config"]["large_domain_reference"][
                "boussinesq_padding_control_offsets"
            ]
        ]
        if is_bouss
        else [0]
    )
    if result.get("padding_offsets") != expected_offsets:
        raise RuntimeError("task padding-control offsets mismatch")
    if is_bouss:
        control_rows = result["padding_control_rows"]
        expected_pairs = list(zip(expected_offsets, expected_offsets[1:]))
        if (len(control_rows) != times.size * len(expected_pairs) or
                result["comparison_pad_cells"] <= result["pad_cells"]):
            raise RuntimeError("task padding-control row count/alignment mismatch")
        for pair_index, (lower, upper) in enumerate(expected_pairs):
            pair_rows = [
                row
                for row in control_rows
                if row.get("control_pair_index") == pair_index
            ]
            if (
                len(pair_rows) != times.size
                or sorted(row.get("requested_time") for row in pair_rows)
                != times.tolist()
                or any(
                    row.get("lower_offset_cells") != lower
                    or row.get("upper_offset_cells") != upper
                    for row in pair_rows
                )
            ):
                raise RuntimeError("task padding-control pair alignment mismatch")
        if any(set(row.get("allowances", {})) != set(PADDING_CONTROL_THRESHOLDS) for row in control_rows):
            raise RuntimeError("task padding-control allowances are incomplete")
    reference_health_rows = result.get("padding_reference_health")
    if (
        not isinstance(reference_health_rows, list)
        or [row.get("offset_cells") for row in reference_health_rows]
        != expected_offsets
    ):
        raise RuntimeError("task padding-reference health coverage mismatch")
    for health_row in reference_health_rows:
        health = health_row.get("health")
        if (
            not isinstance(health, dict)
            or health.get("measurement_dtype") != "float64"
            or not health.get("finite")
            or health.get("cg_failure_count") != 0
            or health.get("requested_times") != times.tolist()
        ):
            raise RuntimeError("task padding-reference health is invalid")
    if result.get("comparison_reference_health") != reference_health_rows[-1]["health"]:
        raise RuntimeError("task comparison-reference health mismatch")
    if scientific_digest(result) != manifest.get("scientific_digest"):
        raise RuntimeError("task scientific digest mismatch")
    return result


def _execute_and_persist_case_task(
    task_root: str, task: Mapping[str, Any]
) -> str:
    result = _execute_case_task(task)
    _write_task_artifact(Path(task_root), task, result)
    return _task_directory_name(task)


def execute_case_tasks(
    tasks: Sequence[Mapping[str, Any]], *, workers: int, start_method: str,
    max_in_flight: int | None = None, task_root: Path | None = None,
    resume: bool = False,
    execution_provenance: dict[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    maximum = workers if max_in_flight is None else max_in_flight
    if workers < 1 or maximum < 1 or maximum < workers or start_method != "spawn":
        raise ValueError("invalid bounded spawn execution policy")
    ordered = sorted(tasks, key=lambda task: int(task.get("ordinal", 0)))
    _validate_task_plan(ordered)
    results: dict[str, dict[str, Any]] = {}
    if task_root is not None:
        task_root.mkdir(parents=True, exist_ok=True)
        expected_directories = {_task_directory_name(task) for task in ordered}
        for path in task_root.iterdir():
            if path.name.startswith(".") and ".staging-" in path.name:
                base, pid = path.name[1:].rsplit(".staging-", 1)
                if (not path.is_dir() or base not in expected_directories or
                        not pid.isdigit()):
                    raise RuntimeError(f"unrecognized task staging artifact: {path.name}")
                if not resume:
                    raise RuntimeError("pre-existing task staging requires --resume")
                shutil.rmtree(path)
            elif not path.is_dir() or path.name not in expected_directories:
                raise RuntimeError(f"unexpected task artifact: {path.name}")
        for task in ordered:
            directory = task_root / _task_directory_name(task)
            if directory.exists():
                if not resume:
                    raise RuntimeError("pre-existing completed task requires --resume")
                results[_task_directory_name(task)] = _load_task_artifact(task_root, task)
    missing = [task for task in ordered if _task_directory_name(task) not in results]
    peak_in_flight = 0
    started = time.monotonic()

    def notify(
        event: str,
        task: Mapping[str, Any] | None = None,
        *,
        active_count: int = 0,
    ) -> None:
        if progress_callback is None:
            return
        payload: dict[str, Any] = {
            "event": event,
            "completed": len(results),
            "total": len(ordered),
            "pending": len(ordered) - len(results),
            "resumed": len(ordered) - len(missing),
            "workers": workers,
            "max_in_flight": maximum,
            "elapsed_s": time.monotonic() - started,
            "active": active_count,
        }
        if task is not None:
            payload.update(
                {
                    "ordinal": int(task.get("ordinal", 0)),
                    "task_id": str(task["task_id"]),
                    "qualified_id": str(task["qualified_id"]),
                    "solver": str(task["solver"]),
                }
            )
        progress_callback(payload)

    notify("start")

    def keep(task: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        value = dict(result) if task_root is None else _write_task_artifact(task_root, task, result)
        results[_task_directory_name(task)] = value
        notify("task_completed", task)
    if workers == 1:
        for task in missing:
            peak_in_flight = max(peak_in_flight, 1)
            keep(task, _execute_case_task(task))
    elif missing:
        context = multiprocessing.get_context(start_method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            iterator = iter(missing)
            active: dict[Any, Mapping[str, Any]] = {}

            def submit(task: Mapping[str, Any]) -> Any:
                if task_root is None:
                    return pool.submit(_execute_case_task, task)
                return pool.submit(
                    _execute_and_persist_case_task, str(task_root), task
                )

            while len(active) < maximum:
                try:
                    task = next(iterator)
                except StopIteration:
                    break
                active[submit(task)] = task
                peak_in_flight = max(peak_in_flight, len(active))
            while active:
                done, _ = wait(
                    active, timeout=30.0, return_when=FIRST_COMPLETED
                )
                if not done:
                    notify("heartbeat", active_count=len(active))
                    continue
                first_error: BaseException | None = None
                for future in sorted(
                    done, key=lambda item: int(active[item]["ordinal"])
                ):
                    task = active.pop(future)
                    try:
                        value = future.result()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    if task_root is None:
                        keep(task, value)
                    else:
                        if value != _task_directory_name(task):
                            raise RuntimeError("worker returned the wrong task identity")
                        results[value] = _load_task_artifact(task_root, task)
                        notify("task_completed", task)
                if first_error is not None:
                    for future in active:
                        future.cancel()
                    raise first_error
                for _ in range(len(done)):
                    try:
                        following = next(iterator)
                    except StopIteration:
                        break
                    active[submit(following)] = following
                    peak_in_flight = max(peak_in_flight, len(active))
    notify("complete")
    if execution_provenance is not None:
        execution_provenance.update(
            {
                "requested_workers": workers,
                "effective_workers": min(workers, max(1, len(missing))),
                "maximum_in_flight": maximum,
                "peak_in_flight": peak_in_flight,
                "resumed_completed_task_count": len(ordered) - len(missing),
                "computed_task_count": len(missing),
                "process_start_method": start_method,
            }
        )
    return [results[_task_directory_name(task)] for task in ordered]


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        source = str(row["source_type"])
        bathymetry = str(row["bathymetry_type"])
        timestamp = float(row["requested_time"])
        dimensions = (
            ("overall", "", "", -1.0),
            ("source_family", source, "", -1.0),
            ("bathymetry_family", "", bathymetry, -1.0),
            ("requested_time", "", "", timestamp),
            ("source_x_time", source, "", timestamp),
            ("bathymetry_x_time", "", bathymetry, timestamp),
        )
        for label, source_value, bathymetry_value, time_value in dimensions:
            key = (
                str(row["solver"]),
                str(row["candidate"]),
                label,
                source_value,
                bathymetry_value,
                time_value,
            )
            groups.setdefault(key, []).append(row)
    output = []
    for (
        solver,
        candidate,
        dimension,
        source_value,
        bathymetry_value,
        time_value,
    ), group in sorted(groups.items()):
        dimension_values: dict[str, Any] = {}
        if source_value:
            dimension_values["source_type"] = source_value
        if bathymetry_value:
            dimension_values["bathymetry_type"] = bathymetry_value
        if time_value >= 0.0:
            dimension_values["requested_time"] = time_value
        output.append(
            {
                "solver": solver,
                "candidate": candidate,
                "dimension": dimension,
                "dimension_values": dimension_values,
                "unique_case_count": len({str(row["qualified_id"]) for row in group}),
                "row_count": len(group),
                "metrics": {
                    metric: {
                        "maximum": float(
                            np.max(
                                values := np.asarray(
                                    [row[metric] for row in group], dtype=np.float64
                                )
                            )
                        ),
                        "median": float(np.median(values)),
                        "p95": float(np.quantile(values, 0.95)),
                    }
                    for metric in METRIC_NAMES
                },
            }
        )
    return output


def evaluate_candidate_policies(
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    global_source_union: np.ndarray,
) -> dict[str, Any]:
    thresholds = config["proposed_future_thresholds"]
    threshold_map = {
        "relative_l2": float(thresholds["shared_crop_relative_l2"]),
        "interior_relative_l2": float(thresholds["interior_relative_l2"]),
        "amplitude_relative_error": float(thresholds["amplitude_relative_error"]),
        "phase_correlation_loss": float(thresholds["phase_correlation_loss"]),
    }
    boussinesq_adequate = all(
        bool(result["reference_adequate"])
        for result in results
        if result["solver"] == "boussinesq"
    )
    production_eligible = bool(
        config["large_domain_reference"]["production_policy_eligible"]
    )
    candidate_assessments: dict[str, dict[str, Any]] = {}
    for solver in SOLVERS:
        candidate_assessments[solver] = {}
        for candidate in config["static_audit"]["candidate_sponges"]:
            subset = [
                row
                for row in rows
                if row["solver"] == solver and row["candidate"] == candidate
            ]
            maxima = {
                metric: max(float(row[metric]) for row in subset)
                for metric in threshold_map
            }
            passed_by_metric = {
                metric: maxima[metric] <= threshold
                for metric, threshold in threshold_map.items()
            }
            health_ok = all(
                result["candidate_health"][candidate]["finite"]
                and result["candidate_health"][candidate]["cg_failure_count"] == 0
                for result in results
                if result["solver"] == solver
            )
            reference_ok = solver != "boussinesq" or boussinesq_adequate
            conditional_supported = bool(
                health_ok and reference_ok and all(passed_by_metric.values())
            )
            candidate_assessments[solver][candidate] = {
                "case_count": len({str(row["qualified_id"]) for row in subset}),
                "row_count": len(subset),
                "maxima": maxima,
                "thresholds": threshold_map,
                "passed_by_metric": passed_by_metric,
                "solver_health_ok": health_ok,
                "reference_adequate": reference_ok,
                "conditional_finite_horizon_supported": conditional_supported,
                "production_reference_eligible": production_eligible,
                "finite_horizon_supported": bool(
                    conditional_supported and production_eligible
                ),
            }
    policies: dict[str, Any] = {}
    for name, mapping in config["candidate_policies"].items():
        if mapping == "padded_reference":
            health_ok = all(
                result["reference_return_safe"]
                and all(
                    item["health"]["finite"]
                    and item["health"]["cg_failure_count"] == 0
                    for item in result["padding_reference_health"]
                )
                for result in results
            )
            conditional_supported = bool(health_ok and boussinesq_adequate)
            policies[name] = {
                "kind": "larger_domain_reference_policy",
                "solver_health_ok": health_ok,
                "reference_adequate": boussinesq_adequate,
                "conditional_reference_supported": conditional_supported,
                "production_policy_eligible": production_eligible,
                "finite_horizon_supported": bool(
                    conditional_supported and production_eligible
                ),
                "extension_status": config["large_domain_reference"][
                    "extension_status"
                ],
                "discrepancy_role": "reference_itself_not_candidate_comparison",
            }
            continue
        per_solver = {
            solver: candidate_assessments[solver][str(mapping[solver])]
            for solver in SOLVERS
        }
        policies[name] = {
            "kind": "finite_domain_boundary_policy",
            "candidate_by_solver": _json_safe(mapping),
            "per_solver_supported": {
                solver: item["finite_horizon_supported"]
                for solver, item in per_solver.items()
            },
            "per_solver_conditionally_supported": {
                solver: item["conditional_finite_horizon_supported"]
                for solver, item in per_solver.items()
            },
            "conditional_finite_horizon_supported": all(
                item["conditional_finite_horizon_supported"]
                for item in per_solver.values()
            ),
            "production_reference_eligible": production_eligible,
            "finite_horizon_supported": all(
                item["finite_horizon_supported"] for item in per_solver.values()
            ),
        }
    return {
        "candidate_assessments": candidate_assessments,
        "policies": policies,
        "boussinesq_reference_adequate": boussinesq_adequate,
        "production_reference_eligible": production_eligible,
        "fixed_source_exclusion_degenerates_to_no_sponge": bool(
            np.all(global_source_union)
        ),
        "scope": (
            "12 selected training cases through t=0.175 only; no claim of a "
            "generally nonreflecting boundary"
        ),
    }


def validate_static_freeze(*, repo_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    freeze_path = output_dir / "STATIC_FREEZE.json"
    selection_path = output_dir / "diagnostic_selection.json"
    if not freeze_path.is_file():
        raise RuntimeError("static audit and selection must be frozen first")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (freeze.get("schema_id") != SCHEMA_ID or
            freeze.get("artifact_kind") != "finite-horizon-boundary-study-static-freeze" or
            freeze.get("status") != "diagnostic_unfrozen_non_decisional" or
            freeze.get("study_hash") != _freeze_self_hash(freeze)):
        raise RuntimeError("invalid static freeze schema/kind/status/self-hash")
    if sha256_file(config_path) != freeze["config_sha256"]:
        raise RuntimeError("study config changed after static freeze")
    inventory_path = repo_root / str(config["authoritative_inventory"])
    if sha256_file(inventory_path) != freeze["inventory_sha256"]:
        raise RuntimeError("authoritative inventory changed after static freeze")
    state = code_state(repo_root)
    if state != freeze["code_state"] or state["code_state_hash"] != freeze["code_state_hash"]:
        raise RuntimeError("code state changed after static freeze")
    expected_static_names = {
        "static_audit_rows.jsonl",
        "static_audit_summary.json",
        "diagnostic_selection.json",
        "global_source_union.npy",
        "task_plan.json",
        "source_snapshot.zip",
        "source_snapshot_manifest.json",
    }
    if set(freeze.get("static_files", {})) != expected_static_names:
        raise RuntimeError("static freeze file coverage mismatch")
    allowed_root_names = expected_static_names | {"STATIC_FREEZE.json"}
    unexpected_root = {
        path.name
        for path in output_dir.iterdir()
        if path.name not in allowed_root_names
        and path.name not in {".execution-staging", "execution", "SHA256SUMS.txt"}
    }
    if unexpected_root:
        raise RuntimeError(f"unexpected static artifact entries: {sorted(unexpected_root)}")
    for name, record in freeze["static_files"].items():
        path = output_dir / name
        if not path.is_file() or _file_record(path) != record:
            raise RuntimeError(f"frozen static file changed: {name}")
    snapshot_manifest = _validate_source_snapshot(output_dir, freeze)
    if (
        freeze.get("source_snapshot", {}).get("source_file_count")
        != snapshot_manifest["source_file_count"]
        or snapshot_manifest["source_file_count"]
        != freeze["code_state"]["source_file_count"]
    ):
        raise RuntimeError("source snapshot/code-state coverage mismatch")
    union = np.load(output_dir / "global_source_union.npy", allow_pickle=False)
    union_raw = hash_array(union)
    grid = int(config["production"]["grid"])
    if (union.dtype != np.bool_ or union.shape != (grid, grid) or
            union_raw != freeze["source_union"]["raw"] or
            _semantic_hash("global-significant-source-union", union_raw) != freeze["source_union"]["semantic_sha256"]):
        raise RuntimeError("global source union changed")
    expected_threads = config["execution"]["thread_environment"]
    actual_threads = {key: os.environ.get(key) for key in THREAD_KEYS}
    if actual_threads != expected_threads:
        raise RuntimeError(f"thread environment mismatch: {actual_threads}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (sha256_file(selection_path) != freeze["selection_sha256"] or
            len(selection) != freeze["selected_case_count"] or
            freeze["selected_case_count"] != 12 or
            [row["qualified_id"] for row in selection] != freeze["selected_qualified_ids"] or
            freeze["task_count"] != len(selection) * len(SOLVERS) or
            freeze["thresholds_frozen_before_selected_case_execution"] !=
            _json_safe(config["proposed_future_thresholds"]) or
            freeze["candidate_policies_frozen_before_selected_case_execution"] !=
            _json_safe(config["candidate_policies"])):
        raise RuntimeError("static selection/count/threshold binding mismatch")
    inventory = {row["qualified_id"]: row for row in _read_jsonl(inventory_path)}
    for selected in selection:
        authoritative = inventory.get(selected["qualified_id"])
        if (authoritative is None or
                authoritative["input_fingerprint"] != selected["input_fingerprint"] or
                any(selected.get(key) != authoritative.get(key) for key in (
                    "split", "scenario_id", "sample_index", "bathymetry_type",
                    "source_type", "source_strength", "bathymetry_cache_path",
                    "source_cache_path", "array_hashes",
                ))):
            raise RuntimeError("selected authoritative input/fingerprint mismatch")
        _load_canary_arrays(selected)
    expected_plan = build_task_plan(
        selection, study_hash=freeze["task_plan_binding_hash"],
        config_sha256=freeze["config_sha256"],
        selection_sha256=freeze["selection_sha256"],
        code_state_hash=freeze["code_state_hash"],
        source_union_hash=freeze["source_union"]["semantic_sha256"])
    plan = json.loads((output_dir / "task_plan.json").read_text(encoding="utf-8"))
    _validate_task_plan(plan)
    if plan != expected_plan or len(plan) != 36:
        raise RuntimeError("task plan changed or is non-contiguous")
    return {"config": config, "freeze": freeze, "selection": selection,
            "union": union, "plan": plan, "threads": actual_threads}


def run_numerical_study(
    *,
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    resume: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    validated = validate_static_freeze(repo_root=repo_root, config_path=config_path, output_dir=output_dir)
    config, freeze, selection, union = (validated[key] for key in ("config", "freeze", "selection", "union"))
    execution = output_dir / "execution"
    staging = output_dir / ".execution-staging"
    if execution.exists():
        if not resume:
            raise RuntimeError("completed execution already exists")
        report = json.loads((execution / "STUDY_RESULT.json").read_text(encoding="utf-8"))
        verify_artifact_checksums(output_dir)
        return {"report": report, "aggregates": json.loads((execution / "finite_horizon_aggregates.json").read_text()),
                "results": json.loads((execution / "finite_horizon_task_results.json").read_text())}
    if staging.exists() and not resume:
        raise RuntimeError("partial execution exists; use --resume")
    staging.mkdir(exist_ok=resume)
    execution_output_names = {
        "finite_horizon_rows.jsonl",
        "finite_horizon_task_results.json",
        "finite_horizon_aggregates.json",
        "boussinesq_padding_control_rows.jsonl",
        "boussinesq_padding_control_aggregates.json",
        "candidate_policy_assessment.json",
        "STUDY_RESULT.json",
    }
    if resume:
        for path in staging.iterdir():
            if path.name == "tasks":
                continue
            if path.is_file() and path.name in execution_output_names:
                path.unlink()
                continue
            raise RuntimeError(f"unrecognized partial execution artifact: {path.name}")
    cases = {row["qualified_id"]: row for row in selection}
    tasks = [
        {
            **spec, "config": config,
            "case_record": cases[spec["qualified_id"]],
            "global_source_union": union,
        }
        for spec in validated["plan"]
    ]
    started = time.monotonic()
    execution_provenance: dict[str, Any] = {}
    progress_every = int(config["execution"]["progress_every"])

    def configured_progress(event: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        if (
            event.get("event") != "task_completed"
            or int(event["completed"]) % progress_every == 0
            or int(event["completed"]) == int(event["total"])
        ):
            progress_callback(event)

    results = execute_case_tasks(
        tasks,
        workers=int(config["execution"]["workers"]),
        start_method=str(config["execution"]["process_start_method"]),
        max_in_flight=int(config["execution"]["max_in_flight"]),
        task_root=staging / "tasks", resume=resume,
        execution_provenance=execution_provenance,
        progress_callback=configured_progress,
    )
    duration = time.monotonic() - started
    if code_state(repo_root) != freeze["code_state"]:
        raise RuntimeError(
            "code state changed during numerical execution; completed task "
            "checkpoints were retained, but finalization is blocked"
        )
    if not all(result["reference_return_safe"] for result in results):
        raise RuntimeError("at least one padded reference is not return-safe")
    rows = [row for result in results for row in result["rows"]]
    _write_jsonl(staging / "finite_horizon_rows.jsonl", rows)
    _write_json(staging / "finite_horizon_task_results.json", results)
    aggregates = _aggregate_metrics(rows)
    _write_json(staging / "finite_horizon_aggregates.json", aggregates)
    padding_rows = [row for result in results for row in result["padding_control_rows"]]
    _write_jsonl(staging / "boussinesq_padding_control_rows.jsonl", padding_rows)
    padding_aggregates = {
        "cases": [{"qualified_id": result["qualified_id"], **result["padding_control"]}
                  for result in results if result["padding_control"] is not None],
    }
    padding_aggregates["global_adequate"] = all(item["adequate"] for item in padding_aggregates["cases"])
    metric_names = tuple(next(iter(padding_aggregates["cases"]), {"allowances": {}})["allowances"])
    padding_aggregates["overall"] = {
        "allowances": {name: min(item["allowances"][name] for item in padding_aggregates["cases"])
                       for name in metric_names},
        "maxima": {name: max(item["maxima"][name] for item in padding_aggregates["cases"])
                   for name in metric_names},
    }
    padding_aggregates["overall"]["adequate_by_metric"] = {
        name: padding_aggregates["overall"]["maxima"][name] <= padding_aggregates["overall"]["allowances"][name]
        for name in metric_names}
    padding_aggregates["overall"]["nonincreasing_by_metric"] = {
        name: all(
            item["nonincreasing_by_metric"][name]
            for item in padding_aggregates["cases"]
        )
        for name in metric_names
    }
    padding_aggregates["overall"]["monotonicity_required_by_metric"] = {
        name: name in PADDING_CONVERGENCE_METRICS for name in metric_names
    }
    padding_aggregates["overall"]["convergence_requirement_met_by_metric"] = {
        name: (
            name not in PADDING_CONVERGENCE_METRICS
            or padding_aggregates["overall"]["nonincreasing_by_metric"][name]
        )
        for name in metric_names
    }
    _write_json(staging / "boussinesq_padding_control_aggregates.json", padding_aggregates)
    policy_assessment = evaluate_candidate_policies(
        rows,
        results,
        config,
        global_source_union=union,
    )
    _write_json(staging / "candidate_policy_assessment.json", policy_assessment)
    report = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "finite-horizon-boundary-selection-study-result",
        "status": "diagnostic_unfrozen_non_decisional",
        "static_study_hash": freeze["study_hash"],
        "selected_case_count": len(selection),
        "solver_case_task_count": len(tasks),
        "candidate_count": len(config["static_audit"]["candidate_sponges"]),
        "metric_row_count": len(rows),
        "duration_s": duration,
        "workers": int(config["execution"]["workers"]),
        "execution_provenance": execution_provenance,
        "thread_environment": validated["threads"],
        "thresholds_proposed_before_outcomes": config["proposed_future_thresholds"],
        "all_references_return_safe": True,
        "all_requested_states_float64": all(
            all(
                item["health"]["measurement_dtype"] == "float64"
                for item in result["padding_reference_health"]
            )
            and all(
                health["measurement_dtype"] == "float64"
                for health in result["candidate_health"].values()
            )
            for result in results
        ),
        "boussinesq_reference_globally_adequate": padding_aggregates["global_adequate"],
        "conditional_boundary_conclusion_available": padding_aggregates[
            "global_adequate"
        ],
        "production_boundary_conclusion_available": bool(
            padding_aggregates["global_adequate"]
            and config["large_domain_reference"]["production_policy_eligible"]
        ),
        "boundary_conclusion_available": bool(
            padding_aggregates["global_adequate"]
            and config["large_domain_reference"]["production_policy_eligible"]
        ),
        "large_domain_extension_status": config["large_domain_reference"][
            "extension_status"
        ],
        "reference_inadequacy_wording": (
            None
            if padding_aggregates["global_adequate"]
            else "reference padding sequence inadequate; candidate diagnostics retained; no boundary conclusion"
        ),
        "extension_assumption_wording": (
            "padded comparison is conditional on an edge-matched fixed-width "
            "cosine taper of the perturbation and edge-continued bathymetry; "
            "authoritative exterior generator parameters are unavailable, so it "
            "is not a production-policy result"
        ),
        "code_state_stable_through_execution": True,
        "source_snapshot_sha256": freeze["static_files"]["source_snapshot.zip"][
            "sha256"
        ],
        "aggregates_sha256": sha256_file(staging / "finite_horizon_aggregates.json"),
        "rows_sha256": sha256_file(staging / "finite_horizon_rows.jsonl"),
        "candidate_policy_assessment_sha256": sha256_file(
            staging / "candidate_policy_assessment.json"
        ),
    }
    _write_json(staging / "STUDY_RESULT.json", report)
    os.replace(staging, execution)
    verify_artifact_checksums(output_dir)
    return {"report": report, "aggregates": aggregates, "results": results}


def verify_artifact_checksums(output_dir: Path) -> dict[str, str]:
    if (not (output_dir / "STATIC_FREEZE.json").is_file() or
            not (output_dir / "execution" / "STUDY_RESULT.json").is_file()):
        raise RuntimeError("checksums require a complete scientific execution")
    checksum_path = output_dir / "SHA256SUMS.txt"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksums = {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    content = "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    if checksum_path.exists():
        if checksum_path.read_text(encoding="utf-8") != content:
            raise RuntimeError("artifact checksum validation failed")
    else:
        temporary = output_dir / f".SHA256SUMS.txt.tmp-{os.getpid()}"
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, checksum_path)
    return checksums
