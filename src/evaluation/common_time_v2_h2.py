from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    candidate_requested_times,
    code_state,
    hash_array,
    sha256_file,
    stable_hash_payload,
)
from src.evaluation.buffered_crop_benchmark import (
    SOLVERS,
    run_buffered_case_detailed,
)
from src.evaluation.common_time_v2_h1 import (
    THREAD_ENV_KEYS,
    _solver_health_checks,
    _summarize_diagnostics,
    _verify_prerequisites as _verify_pre_h1_prerequisites,
    validate_h1_checksums,
)
from src.evaluation.common_time_v2_level_a import _load_canary_arrays


SCHEMA_ID = "tsunami-surrogate.common-time-v2.h2.v1"
CONFIG_SCHEMA_ID = "tsunami-surrogate.common-time-v2.h2-config.v1"
FROZEN_FILENAMES = (
    "preregistered_contract.json",
    "selected_scenarios.json",
    "task_plan.json",
)
FROZEN_CHECKSUMS = "CONTRACT_SHA256SUMS.txt"


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _validate_standard_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise RuntimeError(f"Missing checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {root.name}/{relative}")
        listed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if listed != actual:
        raise RuntimeError(f"Checksum inventory mismatch: {root}")


def _write_frozen_checksums(root: Path) -> None:
    rows = [f"{sha256_file(root / name)}  {name}" for name in FROZEN_FILENAMES]
    _write_text(root / FROZEN_CHECKSUMS, "\n".join(rows) + "\n")


def validate_frozen_checksums(root: Path) -> None:
    manifest = root / FROZEN_CHECKSUMS
    if not manifest.is_file():
        raise RuntimeError(f"Missing H2 frozen checksum manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        if relative not in FROZEN_FILENAMES:
            raise RuntimeError(f"Unexpected H2 frozen checksum entry: {relative}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"H2 frozen checksum mismatch: {relative}")
        listed.add(relative)
    if listed != set(FROZEN_FILENAMES):
        raise RuntimeError("H2 frozen checksum inventory is incomplete")
    unexpected = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and path.name not in {*FROZEN_FILENAMES, FROZEN_CHECKSUMS}
    }
    if unexpected:
        raise RuntimeError(f"Unexpected top-level files in H2 artifact: {unexpected}")


def _write_execution_checksums(execution_root: Path) -> None:
    rows = []
    for path in sorted(execution_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(
                f"{sha256_file(path)}  {path.relative_to(execution_root).as_posix()}"
            )
    _write_text(execution_root / "SHA256SUMS.txt", "\n".join(rows) + "\n")


def validate_execution_checksums(execution_root: Path) -> None:
    _validate_standard_checksums(execution_root)


def validate_h2_checksums(root: Path) -> None:
    validate_frozen_checksums(root)
    execution_root = root / "execution"
    if execution_root.exists():
        if (execution_root / "result.json").is_file():
            validate_execution_checksums(execution_root)
        elif (execution_root / "SHA256SUMS.txt").exists():
            raise RuntimeError("Partial H2 execution has a final checksum manifest")


def _environment_snapshot() -> dict[str, Any]:
    packages = sorted(
        {
            (
                str(distribution.metadata.get("Name", "")).lower(),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    package_rows = [{"name": name, "version": version} for name, version in packages]
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "package_inventory": package_rows,
        "package_inventory_hash": stable_hash_payload(
            artifact_kind="h2-python-package-inventory",
            payload=package_rows,
            schema_id=SCHEMA_ID,
        ),
    }


def _expected_candidate() -> dict[str, Any]:
    return {
        "solvers": list(SOLVERS),
        "computational_grid": 96,
        "publication_grid": 64,
        "buffer_cells_per_side": 16,
        "source_taper_cells": 8,
        "bathymetry_extension": "edge",
        "output_crop": "central",
        "dx": 1.0 / 64.0,
        "dy": 1.0 / 64.0,
        "requested_time_start": 0.0035,
        "requested_time_step": 0.0035,
        "requested_time_count": 50,
        "horizon": 0.175,
        "requested_state_dtype": "float64",
        "max_natural_steps": 20_000,
        "sponge_width_cells": 16,
        "sponge_min_factor": 0.8,
        "sponge_axes": "xy",
        "sponge_profile": "cosine",
        "sponge_time_mode": "elapsed_time_consistent",
        "sponge_reference_dt": 0.0035,
        "swe_cfl": 0.45,
        "swe_boundary": "radiation",
        "dry_tolerance": 1.0e-6,
        "max_velocity": 30.0,
        "muscl_limiter": "minmod",
        "boussinesq_cfl": 0.35,
        "boussinesq_boundary": "open",
        "boussinesq_boundary_interpretation": "zero_gradient_edge_padding",
        "boussinesq_depth_scale": 1.0,
        "boussinesq_mode": "linear_variable_depth",
        "boussinesq_filter_strength": 0.0,
        "boussinesq_filter_time_mode": "disabled",
        "boussinesq_linear_solver_tol": 1.0e-10,
        "boussinesq_linear_solver_abs_tol": 0.0,
        "boussinesq_linear_solver_max_iter": 750,
        "boussinesq_cg_failure_mode": "strict_v2",
    }


def _expected_thresholds() -> dict[str, dict[str, float]]:
    return {
        "swe_hydrostatic": {
            "trajectory_relative_l2_median": 0.12,
            "trajectory_relative_l2_p95": 0.20,
            "trajectory_relative_l2_max": 0.30,
            "per_time_normalized_rmse_p95": 0.25,
            "per_time_normalized_rmse_max": 0.50,
            "peak_amplitude_relative_error_median": 0.15,
            "peak_amplitude_relative_error_p95": 0.25,
            "peak_amplitude_relative_error_max": 0.40,
            "phase_correlation_loss_median": 0.05,
            "phase_correlation_loss_p95": 0.15,
            "phase_correlation_loss_max": 0.30,
            "family_cell_trajectory_median_max": 0.20,
            "family_cell_trajectory_case_max": 0.30,
        },
        "swe_muscl_hr": {
            "trajectory_relative_l2_median": 0.03,
            "trajectory_relative_l2_p95": 0.06,
            "trajectory_relative_l2_max": 0.10,
            "per_time_normalized_rmse_p95": 0.08,
            "per_time_normalized_rmse_max": 0.15,
            "peak_amplitude_relative_error_median": 0.05,
            "peak_amplitude_relative_error_p95": 0.10,
            "peak_amplitude_relative_error_max": 0.15,
            "phase_correlation_loss_median": 0.02,
            "phase_correlation_loss_p95": 0.05,
            "phase_correlation_loss_max": 0.10,
            "family_cell_trajectory_median_max": 0.06,
            "family_cell_trajectory_case_max": 0.10,
        },
        "boussinesq": {
            "trajectory_relative_l2_median": 0.001,
            "trajectory_relative_l2_p95": 0.0025,
            "trajectory_relative_l2_max": 0.005,
            "per_time_normalized_rmse_p95": 0.005,
            "per_time_normalized_rmse_max": 0.01,
            "peak_amplitude_relative_error_median": 0.0025,
            "peak_amplitude_relative_error_p95": 0.005,
            "peak_amplitude_relative_error_max": 0.01,
            "phase_correlation_loss_median": 0.001,
            "phase_correlation_loss_p95": 0.0025,
            "phase_correlation_loss_max": 0.005,
            "family_cell_trajectory_median_max": 0.0025,
            "family_cell_trajectory_case_max": 0.005,
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("H2 YAML must contain a mapping")
    _validate_config(payload)
    return payload


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != CONFIG_SCHEMA_ID or config.get("stage") != "H2":
        raise ValueError("H2 config schema/stage mismatch")
    if config.get("claim_scope") != (
        "total_temporal_discretization_and_production_operator_sensitivity"
    ):
        raise ValueError("H2 claim scope changed")

    expected_prerequisites = {
        "h0_contract_hash": (
            "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b"
        ),
        "level_a_contract_hash": (
            "be1af7dce1f48942e6d20a96bb06b1359655903847c7580954901e2dcfa3332b"
        ),
        "level_b_bundle_hash": (
            "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
        ),
        "h1_contract_hash": (
            "ef96c24f62a0eb0884f5384436a50802c0d8dd644946552d9c462b225334bc7d"
        ),
        "require_h0_pass": True,
        "require_level_a_decision": "pass_to_H1",
        "require_level_b_decision": "pass_to_H1",
        "require_h1_decision": "pass_to_H2",
    }
    if config.get("prerequisites") != expected_prerequisites:
        raise ValueError("H2 prerequisite identities or decisions changed")

    expected_selection = {
        "split": "train",
        "expected_split_count": 10_000,
        "count_per_cell": 4,
        "selection_seed": "common-time-v2-h2-balanced-selection-v1",
        "replay_selection_seed": "common-time-v2-h2-replay-selection-v1",
        "exclude_h1_contract_hash": expected_prerequisites["h1_contract_hash"],
        "expected_h1_exclusion_count": 30,
        "bathymetry_families": [
            "canyon",
            "continental",
            "island",
            "seamounts",
            "trench",
        ],
        "source_families": [
            "dipole",
            "fault",
            "gaussian",
            "multi-gauss",
            "okada-like",
            "rough",
        ],
    }
    if config.get("selection") != expected_selection:
        raise ValueError("H2 balanced selection or H1 exclusion policy changed")
    if config.get("candidate") != _expected_candidate():
        raise ValueError("H2 candidate differs from the passing H1 candidate")

    expected_comparison = {
        "production_cfl": {
            "swe_hydrostatic": 0.45,
            "swe_muscl_hr": 0.45,
            "boussinesq": 0.35,
        },
        "reference_cfl": {
            "swe_hydrostatic": 0.225,
            "swe_muscl_hr": 0.225,
            "boussinesq": 0.175,
        },
        "reference_cfl_factor": 0.5,
        "pair_execution_order": ["production", "reference"],
        "requested_state_dtype": "float64",
        "requested_time_comparison": "exact",
        "interpretation": (
            "total_temporal_discretization_and_production_operator_sensitivity"
        ),
    }
    if config.get("comparison") != expected_comparison:
        raise ValueError("H2 production/reference CFL comparison changed")
    for solver in SOLVERS:
        if (
            expected_comparison["reference_cfl"][solver]
            != 0.5 * expected_comparison["production_cfl"][solver]
        ):
            raise ValueError("H2 reference CFL is not exactly half production CFL")

    expected_health = {
        "require_exact_requested_times": True,
        "require_adjacent_bracket_provenance": True,
        "require_no_missing_duplicate_or_extrapolated_outputs": True,
        "require_complete_float64_natural_health": True,
        "reject_nonfinite": True,
        "min_h_tolerance": -1.0e-6,
        "max_abs_eta_limit": 5.0,
        "max_velocity_limit": 30.0,
        "max_eta_over_depth": 1.0,
        "require_cg_converged": True,
        "require_two_cg_solves_per_natural_step": True,
        "require_both_cfl_variants_healthy": True,
        "deterministic_replay_comparison": "exact_scientific_digest",
    }
    if config.get("health_gates") != expected_health:
        raise ValueError("H2 health gates changed")

    metrics = config.get("metrics", {})
    expected_metrics = {
        "trajectory_reference": "smaller_cfl_replay",
        "relative_denominator": "full_reference_trajectory_rms",
        "relative_floor_absolute_rms": 1.0e-12,
        "relative_floor_rationale": (
            "above_float64_roundoff_but_below_intended_eta_resolution"
        ),
        "per_time_relative_denominator": "full_reference_trajectory_rms",
        "boundary_band_cells": 8,
        "amplitude_metric": "peak_absolute_eta_over_full_trajectory",
        "phase_metric": "centered_waveform_correlation_loss",
        "phase_activity_floor_absolute_rms": 1.0e-10,
        "minimum_phase_applicable_fraction": 0.95,
        "stable_reduction": "sorted_values_math_fsum_linear_quantile",
        "family_cell_rule": (
            "trajectory_every_cell_median_and_max_amplitude_phase_reported"
        ),
        "bootstrap": {
            "seed": "common-time-v2-h2-paired-stratified-bootstrap-v1",
            "resamples": 2000,
            "confidence_level": 0.95,
            "method": "paired_within_family_cell",
            "decision_role": "informational_uncertainty_not_gate",
        },
    }
    if metrics != expected_metrics:
        raise ValueError("H2 metric, floor, family, or bootstrap policy changed")
    if config.get("thresholds") != _expected_thresholds():
        raise ValueError("H2 solver-specific thresholds changed")

    basis = config.get("threshold_basis", {})
    if (
        basis.get("source") != "frozen_level_a_be1af7d"
        or basis.get("h2_outcomes_viewed") is not False
        or basis.get("historical_stage_c_thresholds_inherited") is not False
        or basis.get("rationale")
        != (
            "solver_specific_envelopes_from_level_a_production_to_half_cfl_"
            "and_spatial_controls"
        )
    ):
        raise ValueError("H2 threshold provenance changed")

    if config.get("replay") != {
        "selected_case_count": 1,
        "runs_per_solver": 1,
        "comparison": "exact_scientific_digest",
    }:
        raise ValueError("H2 deterministic replay policy changed")
    if config.get("execution") != {
        "requested_workers": 8,
        "requested_max_in_flight": 8,
        "process_start_method": "spawn",
        "numerical_library_threads": 1,
        "resume_policy": "validate_then_skip",
        "corruption_policy": "fail_closed",
        "require_clean_git_at_freeze": True,
    }:
        raise ValueError("H2 execution policy changed")
    if config.get("decisions") != {
        "pass": "pass_to_common_time_v2_contract_freeze",
        "sensitivity_failure": "blocked_h2_temporal_operator_sensitivity",
        "health_failure": "blocked_h2_solver_health",
        "implementation_failure": "implementation_failure",
    }:
        raise ValueError("H2 decision vocabulary changed")
    expected_times = np.arange(1, 51, dtype=np.float64) * np.float64(0.0035)
    expected_times[-1] = np.float64(0.175)
    if not np.array_equal(candidate_requested_times(), expected_times):
        raise ValueError("H2 canonical requested-time vector changed")


def _verify_prerequisites(
    *,
    config: Mapping[str, Any],
    h0_root: Path,
    level_a_root: Path,
    level_b_bundle_root: Path,
    level_b_evaluation_root: Path,
    h1_root: Path,
) -> dict[str, Any]:
    evidence = _verify_pre_h1_prerequisites(
        config=config,
        h0_root=h0_root,
        level_a_root=level_a_root,
        level_b_bundle_root=level_b_bundle_root,
        level_b_evaluation_root=level_b_evaluation_root,
    )
    validate_h1_checksums(h1_root)
    h1_contract = _read_json(h1_root / "preregistered_contract.json")
    h1_result = _read_json(h1_root / "execution" / "result.json")
    h1_decision = _read_json(h1_root / "execution" / "decision.json")
    expected = config["prerequisites"]
    if (
        h1_root.name != expected["h1_contract_hash"]
        or h1_contract.get("contract_hash") != h1_root.name
        or h1_result.get("contract_hash") != h1_root.name
        or h1_decision.get("contract_hash") != h1_root.name
        or h1_result.get("decision") != expected["require_h1_decision"]
        or h1_decision.get("decision") != expected["require_h1_decision"]
        or h1_result.get("h1_passed") is not True
        or h1_decision.get("h1_passed") is not True
    ):
        raise RuntimeError("H1 prerequisite identity or decision mismatch")
    selected_path = h1_root / "selected_scenarios.json"
    evidence["h1"] = {
        "root": str(h1_root),
        "contract_hash": h1_root.name,
        "decision": str(h1_result["decision"]),
        "scientific_digest": stable_hash_payload(
            artifact_kind="h1-result-scientific-reference",
            payload={
                "contract_hash": h1_result["contract_hash"],
                "decision": h1_result["decision"],
                "summary": h1_result["summary"],
                "failed_task_ids": h1_result["failed_task_ids"],
                "replay_mismatches": h1_result["replay_mismatches"],
            },
            schema_id=SCHEMA_ID,
        ),
        "selected_scenarios_path": str(selected_path),
        "selected_scenarios_sha256": sha256_file(selected_path),
        "checksums_sha256": sha256_file(
            h1_root / "execution" / "SHA256SUMS.txt"
        ),
    }
    return evidence


def _selection_rank(
    record: Mapping[str, Any],
    *,
    inventory_sha256: str,
    seed: str,
    purpose: str,
) -> str:
    return stable_hash_payload(
        artifact_kind="common-time-v2-h2-selection-rank",
        payload={
            "purpose": purpose,
            "seed": seed,
            "inventory_sha256": inventory_sha256,
            "qualified_id": record["qualified_id"],
            "input_fingerprint": record["input_fingerprint"],
            "bathymetry_type": record["bathymetry_type"],
            "source_type": record["source_type"],
        },
        schema_id=SCHEMA_ID,
    )


def select_h2_scenarios(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_config: Mapping[str, Any],
    inventory_sha256: str,
    h1_selected: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = str(selection_config["split"])
    expected_count = int(selection_config["expected_split_count"])
    count_per_cell = int(selection_config["count_per_cell"])
    bathymetry_families = tuple(map(str, selection_config["bathymetry_families"]))
    source_families = tuple(map(str, selection_config["source_families"]))
    training = [dict(row) for row in rows if row.get("split") == split]
    if len(training) != expected_count:
        raise RuntimeError(
            f"H2 expected {expected_count} {split} inputs, found {len(training)}"
        )
    qualified_ids = [str(row.get("qualified_id")) for row in training]
    fingerprints = [str(row.get("input_fingerprint")) for row in training]
    if len(set(qualified_ids)) != len(qualified_ids):
        raise RuntimeError("H2 authoritative training inventory has duplicate identities")
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("H2 authoritative training inventory has duplicate inputs")

    expected_exclusions = int(selection_config["expected_h1_exclusion_count"])
    excluded_records = [dict(entry["record"]) for entry in h1_selected]
    excluded_ids = {str(record["qualified_id"]) for record in excluded_records}
    excluded_fingerprints = {
        str(record["input_fingerprint"]) for record in excluded_records
    }
    if (
        len(excluded_records) != expected_exclusions
        or len(excluded_ids) != expected_exclusions
        or len(excluded_fingerprints) != expected_exclusions
    ):
        raise RuntimeError("H2 requires exactly 30 unique H1 exclusions")
    inventory_by_id = {str(row["qualified_id"]): row for row in training}
    for record in excluded_records:
        authoritative = inventory_by_id.get(str(record["qualified_id"]))
        if authoritative is None or (
            authoritative.get("input_fingerprint") != record.get("input_fingerprint")
        ):
            raise RuntimeError("H2 H1 exclusion identity/fingerprint mismatch")

    expected_cells = {
        (bathymetry, source)
        for bathymetry in bathymetry_families
        for source in source_families
    }
    cell_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        cell: [] for cell in expected_cells
    }
    unexpected_cells: set[tuple[str, str]] = set()
    for row in training:
        if (
            str(row["qualified_id"]) in excluded_ids
            or str(row["input_fingerprint"]) in excluded_fingerprints
        ):
            continue
        cell = (str(row["bathymetry_type"]), str(row["source_type"]))
        if cell not in cell_rows:
            unexpected_cells.add(cell)
            continue
        cell_rows[cell].append(row)
    if unexpected_cells:
        raise RuntimeError(f"Unexpected H2 family cells: {sorted(unexpected_cells)}")
    insufficient = [
        cell for cell, candidates in cell_rows.items()
        if len(candidates) < count_per_cell
    ]
    if insufficient:
        raise RuntimeError(f"Insufficient H2 family cells: {sorted(insufficient)}")

    selected: list[dict[str, Any]] = []
    seed = str(selection_config["selection_seed"])
    for cell in sorted(expected_cells):
        ranked = sorted(
            (
                _selection_rank(
                    record,
                    inventory_sha256=inventory_sha256,
                    seed=seed,
                    purpose="balanced_cell_selection",
                ),
                str(record["qualified_id"]),
                record,
            )
            for record in cell_rows[cell]
        )
        for within_cell_ordinal, (rank, _qualified_id, record) in enumerate(
            ranked[:count_per_cell]
        ):
            selected.append(
                {
                    "selection_ordinal": len(selected),
                    "within_cell_ordinal": within_cell_ordinal,
                    "bathymetry_type": cell[0],
                    "source_type": cell[1],
                    "cell_candidate_count_after_h1_exclusion": len(ranked),
                    "selection_rank": rank,
                    "record": record,
                }
            )

    replay_seed = str(selection_config["replay_selection_seed"])
    replay_ranked = sorted(
        (
            _selection_rank(
                entry["record"],
                inventory_sha256=inventory_sha256,
                seed=replay_seed,
                purpose="deterministic_replay_case",
            ),
            int(entry["selection_ordinal"]),
        )
        for entry in selected
    )
    replay_selection_ordinal = replay_ranked[0][1]
    if any(
        entry["record"]["qualified_id"] in excluded_ids
        or entry["record"]["input_fingerprint"] in excluded_fingerprints
        for entry in selected
    ):
        raise RuntimeError("H2 selection overlaps H1")
    summary = {
        "split": split,
        "authoritative_split_count": len(training),
        "family_cell_count": len(expected_cells),
        "count_per_cell": count_per_cell,
        "selected_count": len(selected),
        "selected_qualified_ids": [
            entry["record"]["qualified_id"] for entry in selected
        ],
        "selected_input_fingerprints": [
            entry["record"]["input_fingerprint"] for entry in selected
        ],
        "excluded_h1_count": len(excluded_ids),
        "excluded_h1_qualified_ids": sorted(excluded_ids),
        "excluded_h1_input_fingerprints": sorted(excluded_fingerprints),
        "replay_selection_ordinal": replay_selection_ordinal,
        "replay_qualified_id": selected[replay_selection_ordinal]["record"][
            "qualified_id"
        ],
        "validation_inputs_inspected_scientifically": False,
        "test_inputs_inspected_scientifically": False,
        "eventual_training_overlap": "all H2 cases may remain in training",
    }
    return selected, summary


def _candidate_config_hash(
    candidate: Mapping[str, Any], comparison: Mapping[str, Any]
) -> str:
    return stable_hash_payload(
        artifact_kind="common-time-v2-h2-candidate-comparison-config",
        payload={"candidate": dict(candidate), "comparison": dict(comparison)},
        schema_id=SCHEMA_ID,
    )


def _task_identity(
    *,
    ordinal: int,
    run_kind: str,
    solver: str,
    selection: Mapping[str, Any],
    candidate_config_hash: str,
    production_cfl: float,
    reference_cfl: float,
    reference_primary_task_id: str | None = None,
) -> dict[str, Any]:
    record = selection["record"]
    identity = {
        "ordinal": int(ordinal),
        "run_kind": str(run_kind),
        "solver": str(solver),
        "selection_ordinal": int(selection["selection_ordinal"]),
        "qualified_id": str(record["qualified_id"]),
        "input_fingerprint": str(record["input_fingerprint"]),
        "bathymetry_type": str(record["bathymetry_type"]),
        "source_type": str(record["source_type"]),
        "candidate_config_hash": str(candidate_config_hash),
        "production_cfl": float(production_cfl),
        "reference_cfl": float(reference_cfl),
        "reference_primary_task_id": reference_primary_task_id,
    }
    identity["task_id"] = stable_hash_payload(
        artifact_kind="common-time-v2-h2-paired-task",
        payload=identity,
        schema_id=SCHEMA_ID,
    )[:24]
    return identity


def _build_tasks(
    selected: Sequence[Mapping[str, Any]],
    *,
    solvers: Sequence[str],
    replay_selection_ordinal: int,
    candidate_config_hash: str,
    comparison: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    primary_by_selection_solver: dict[tuple[int, str], str] = {}
    ordinal = 0
    for selection in selected:
        for solver in solvers:
            task = _task_identity(
                ordinal=ordinal,
                run_kind="primary",
                solver=str(solver),
                selection=selection,
                candidate_config_hash=candidate_config_hash,
                production_cfl=float(comparison["production_cfl"][solver]),
                reference_cfl=float(comparison["reference_cfl"][solver]),
            )
            tasks.append(task)
            primary_by_selection_solver[
                (int(selection["selection_ordinal"]), str(solver))
            ] = str(task["task_id"])
            ordinal += 1
    replay_selection = selected[replay_selection_ordinal]
    for solver in solvers:
        tasks.append(
            _task_identity(
                ordinal=ordinal,
                run_kind="replay",
                solver=str(solver),
                selection=replay_selection,
                candidate_config_hash=candidate_config_hash,
                production_cfl=float(comparison["production_cfl"][solver]),
                reference_cfl=float(comparison["reference_cfl"][solver]),
                reference_primary_task_id=primary_by_selection_solver[
                    (replay_selection_ordinal, str(solver))
                ],
            )
        )
        ordinal += 1
    return tasks


def _contract_hash(contract: Mapping[str, Any]) -> str:
    identity = dict(contract)
    identity.pop("contract_hash", None)
    return stable_hash_payload(
        artifact_kind="common-time-v2-h2-preregistered-contract",
        payload=identity,
        schema_id=SCHEMA_ID,
    )


def _validate_contract_identity(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("H2 contract schema mismatch")
    if contract.get("contract_hash") != _contract_hash(contract):
        raise RuntimeError("H2 contract content hash mismatch")
    selected = contract["selected_scenarios"]
    tasks = contract["tasks"]
    if len(selected) != 120 or len(tasks) != 363:
        raise RuntimeError("H2 contract must freeze 120 scenarios and 363 paired tasks")
    if sum(task["run_kind"] == "primary" for task in tasks) != 360:
        raise RuntimeError("H2 contract primary paired-task count changed")
    if sum(task["run_kind"] == "replay" for task in tasks) != 3:
        raise RuntimeError("H2 contract replay paired-task count changed")
    if [int(task["ordinal"]) for task in tasks] != list(range(len(tasks))):
        raise RuntimeError("H2 task ordering is not contiguous and frozen")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("H2 task identities are not unique")
    h1_ids = set(contract["selection_summary"]["excluded_h1_qualified_ids"])
    h2_ids = set(contract["selection_summary"]["selected_qualified_ids"])
    if h1_ids & h2_ids:
        raise RuntimeError("H2 contract overlaps passing H1")


def freeze_h2_contract(
    *,
    repo_root: Path,
    config_path: Path,
    h0_root: Path,
    level_a_root: Path,
    level_b_bundle_root: Path,
    level_b_evaluation_root: Path,
    h1_root: Path,
    output_base: Path,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    h0_root = h0_root.resolve()
    level_a_root = level_a_root.resolve()
    level_b_bundle_root = level_b_bundle_root.resolve()
    level_b_evaluation_root = level_b_evaluation_root.resolve()
    h1_root = h1_root.resolve()
    output_base = output_base.resolve()
    config = _load_config(config_path)
    current_code = code_state(repo_root)
    if config["execution"]["require_clean_git_at_freeze"] and current_code["dirty"]:
        raise RuntimeError(
            "H2 freeze requires a clean committed code state; commit or stash "
            "unrelated changes first"
        )
    prerequisite_evidence = _verify_prerequisites(
        config=config,
        h0_root=h0_root,
        level_a_root=level_a_root,
        level_b_bundle_root=level_b_bundle_root,
        level_b_evaluation_root=level_b_evaluation_root,
        h1_root=h1_root,
    )
    inventory_path = Path(prerequisite_evidence["h0"]["inventory_path"])
    inventory_sha256 = str(prerequisite_evidence["h0"]["inventory_sha256"])
    h1_selected_payload = _read_json(
        Path(prerequisite_evidence["h1"]["selected_scenarios_path"])
    )
    h1_selected = h1_selected_payload["selected_scenarios"]
    selected, selection_summary = select_h2_scenarios(
        _read_jsonl(inventory_path),
        selection_config=config["selection"],
        inventory_sha256=inventory_sha256,
        h1_selected=h1_selected,
    )
    for selection in selected:
        _load_canary_arrays(selection["record"])

    candidate_hash = _candidate_config_hash(
        config["candidate"], config["comparison"]
    )
    tasks = _build_tasks(
        selected,
        solvers=config["candidate"]["solvers"],
        replay_selection_ordinal=int(selection_summary["replay_selection_ordinal"]),
        candidate_config_hash=candidate_hash,
        comparison=config["comparison"],
    )
    contract: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-preregistered-contract",
        "stage": "H2",
        "status": "frozen_unexecuted",
        "claim_scope": config["claim_scope"],
        "scientific_outcome_viewed_before_freeze": False,
        "validation_or_test_outcomes_inspected": False,
        "prerequisites": prerequisite_evidence,
        "code_state": current_code,
        "code_state_policy": {
            "clean_committed_source_required": True,
            "freeze_current_h2_code": True,
            "solver_numerics_or_metric_change_requires_new_contract": True,
        },
        "environment": _environment_snapshot(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config,
        "candidate_config_hash": candidate_hash,
        "selection_summary": selection_summary,
        "selected_scenarios": selected,
        "tasks": tasks,
        "worker_policy": dict(config["execution"]),
        "decision_policy": dict(config["decisions"]),
        "immutability": {
            "post_result_changes_forbidden": True,
            "threshold_or_selection_change_requires_new_h2_contract": True,
            "validation_and_test_scientific_outcomes_excluded": True,
            "h2_does_not_authorize_mass_generation": True,
            "passing_h2_only_authorizes_separate_contract_freeze": True,
        },
    }
    contract["contract_hash"] = _contract_hash(contract)
    _validate_contract_identity(contract)
    output_root = output_base / contract["contract_hash"]
    selected_payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-selected-scenarios",
        "contract_hash": contract["contract_hash"],
        "selection_summary": selection_summary,
        "selected_scenarios": selected,
    }
    task_payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-task-plan",
        "contract_hash": contract["contract_hash"],
        "candidate_config_hash": candidate_hash,
        "paired_solver_execution_count": 2 * len(tasks),
        "tasks": tasks,
    }
    if output_root.exists():
        contract_path = output_root / "preregistered_contract.json"
        if not contract_path.is_file() or _read_json(contract_path) != contract:
            raise FileExistsError(
                f"Refusing to replace a different H2 artifact: {output_root}"
            )
        validate_frozen_checksums(output_root)
        return output_root
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "preregistered_contract.json", contract)
    _write_json(output_root / "selected_scenarios.json", selected_payload)
    _write_json(output_root / "task_plan.json", task_payload)
    _write_frozen_checksums(output_root)
    validate_frozen_checksums(output_root)
    return output_root


def _stable_sum_squares(values: Any) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    block_size = 4096
    return float(
        math.fsum(
            float(np.dot(block, block))
            for start in range(0, flat.size, block_size)
            for block in (flat[start : start + block_size],)
        )
    )


def _rms(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("RMS requires at least one value")
    return math.sqrt(_stable_sum_squares(array) / float(array.size))


def _stable_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(math.fsum(sorted(map(float, values))) / len(values))


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    coordinate = (len(ordered) - 1) * quantile
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return float(
        math.fsum(
            (
                (1.0 - fraction) * ordered[lower],
                fraction * ordered[upper],
            )
        )
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        raise ValueError("distribution requires finite values")
    return {
        "count": len(finite),
        "mean": _stable_mean(finite),
        "median": _linear_quantile(finite, 0.5),
        "p95": _linear_quantile(finite, 0.95),
        "max": max(finite),
        "min": min(finite),
    }


def paired_cfl_metrics(
    production: np.ndarray,
    reference: np.ndarray,
    *,
    relative_floor_absolute_rms: float,
    phase_activity_floor_absolute_rms: float,
    boundary_band_cells: int,
) -> dict[str, Any]:
    left = np.asarray(production)
    right = np.asarray(reference)
    if (
        left.dtype != np.dtype(np.float64)
        or right.dtype != np.dtype(np.float64)
    ):
        raise ValueError("H2 scientific metrics require internal float64 states")
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("H2 paired trajectories must share [T,X,Y] shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("H2 paired trajectories must be finite")
    if relative_floor_absolute_rms <= 0.0:
        raise ValueError("H2 relative floor must be positive")

    diff = left - right
    reference_rms = _rms(right)
    denominator = max(reference_rms, float(relative_floor_absolute_rms))
    absolute_rms = _rms(diff)
    per_time_absolute = [_rms(frame) for frame in diff]
    per_time_normalized = [value / denominator for value in per_time_absolute]

    nx, ny = left.shape[1:]
    band = int(boundary_band_cells)
    if band <= 0 or 2 * band >= min(nx, ny):
        raise ValueError("H2 boundary band must leave a non-empty interior")
    boundary = np.zeros((nx, ny), dtype=np.bool_)
    boundary[:band, :] = True
    boundary[-band:, :] = True
    boundary[:, :band] = True
    boundary[:, -band:] = True
    interior = ~boundary
    interior_reference_rms = _rms(right[:, interior])
    boundary_reference_rms = _rms(right[:, boundary])
    interior_absolute_rms = _rms(diff[:, interior])
    boundary_absolute_rms = _rms(diff[:, boundary])

    production_amplitude_by_time = [
        float(np.max(np.abs(frame))) for frame in left
    ]
    reference_amplitude_by_time = [
        float(np.max(np.abs(frame))) for frame in right
    ]
    production_peak = max(production_amplitude_by_time)
    reference_peak = max(reference_amplitude_by_time)
    amplitude_absolute_error = abs(production_peak - reference_peak)
    amplitude_relative_error = amplitude_absolute_error / max(
        reference_peak, float(relative_floor_absolute_rms)
    )

    left_centered = left - np.mean(left, axis=(1, 2), keepdims=True)
    right_centered = right - np.mean(right, axis=(1, 2), keepdims=True)
    left_centered_rms = _rms(left_centered)
    right_centered_rms = _rms(right_centered)
    phase_applicable = bool(
        left_centered_rms >= phase_activity_floor_absolute_rms
        and right_centered_rms >= phase_activity_floor_absolute_rms
    )
    phase_loss: float | None
    if phase_applicable:
        denominator_norm = math.sqrt(
            _stable_sum_squares(left_centered)
            * _stable_sum_squares(right_centered)
        )
        dot = float(
            math.fsum(
                float(np.dot(left_block, right_block))
                for start in range(0, left_centered.size, 4096)
                for left_block, right_block in (
                    (
                        left_centered.reshape(-1)[start : start + 4096],
                        right_centered.reshape(-1)[start : start + 4096],
                    ),
                )
            )
        )
        correlation = min(1.0, max(-1.0, dot / denominator_norm))
        phase_loss = max(0.0, 1.0 - correlation)
    else:
        correlation = None
        phase_loss = None

    return {
        "measurement_dtype": "float64",
        "trajectory_absolute_rms": absolute_rms,
        "trajectory_reference_rms": reference_rms,
        "trajectory_relative_l2": absolute_rms / denominator,
        "relative_denominator_floor_used": reference_rms
        < relative_floor_absolute_rms,
        "final_time_absolute_rms": per_time_absolute[-1],
        "final_time_normalized_rmse": per_time_normalized[-1],
        "per_time_absolute_rms": per_time_absolute,
        "per_time_normalized_rmse": per_time_normalized,
        "interior_absolute_rms": interior_absolute_rms,
        "interior_reference_rms": interior_reference_rms,
        "interior_relative_l2": interior_absolute_rms
        / max(interior_reference_rms, relative_floor_absolute_rms),
        "boundary_absolute_rms": boundary_absolute_rms,
        "boundary_reference_rms": boundary_reference_rms,
        "boundary_relative_l2": boundary_absolute_rms
        / max(boundary_reference_rms, relative_floor_absolute_rms),
        "production_peak_amplitude": production_peak,
        "reference_peak_amplitude": reference_peak,
        "peak_amplitude_absolute_error": amplitude_absolute_error,
        "peak_amplitude_relative_error": amplitude_relative_error,
        "production_amplitude_by_time": production_amplitude_by_time,
        "reference_amplitude_by_time": reference_amplitude_by_time,
        "phase_metric": "centered_waveform_correlation_loss",
        "phase_applicable": phase_applicable,
        "production_centered_rms": left_centered_rms,
        "reference_centered_rms": right_centered_rms,
        "phase_correlation": correlation,
        "phase_correlation_loss": phase_loss,
    }


def _run_variant(
    *,
    record: Mapping[str, Any],
    solver: str,
    candidate: Mapping[str, Any],
    health_gates: Mapping[str, Any],
    target_cfl: float,
    bathymetry: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    row, trajectory, details = run_buffered_case_detailed(
        record,
        solver_name=solver,
        total_grid=int(candidate["computational_grid"]),
        core_grid=int(candidate["publication_grid"]),
        source_taper_cells=int(candidate["source_taper_cells"]),
        sponge_min_factor=float(candidate["sponge_min_factor"]),
        sponge_width_cells=int(candidate["sponge_width_cells"]),
        target_cfl=float(target_cfl),
    )
    expected_times = candidate_requested_times()
    provenance_checks, provenance, natural_health, diagnostic_evidence = (
        _summarize_diagnostics(
            solver=solver,
            details=details,
            expected_times=expected_times,
            target_cfl=float(target_cfl),
        )
    )
    solver_checks, health_metrics = _solver_health_checks(
        solver=solver,
        trajectory=trajectory,
        diagnostics=details["diagnostics"],
        arrays=arrays,
        bathymetry=bathymetry,
        health_gates=health_gates,
        candidate=candidate,
    )
    expected_boundary = "open" if solver == "boussinesq" else "radiation"
    identity_checks = {
        "qualified_id_matches": row["qualified_id"] == record["qualified_id"],
        "input_fingerprint_matches": row["input_fingerprint"]
        == record["input_fingerprint"],
        "solver_matches": row["solver"] == solver,
        "target_cfl_matches": float(row["target_cfl"]) == float(target_cfl)
        and float(details["target_cfl"]) == float(target_cfl),
        "candidate_grid_matches": (
            int(row["total_grid"]) == int(candidate["computational_grid"])
            and int(row["core_grid"]) == int(candidate["publication_grid"])
            and int(row["buffer_cells"]) == int(candidate["buffer_cells_per_side"])
        ),
        "candidate_source_taper_matches": int(row["source_taper_cells"])
        == int(candidate["source_taper_cells"]),
        "candidate_sponge_matches": (
            int(row["sponge_width_cells"])
            == int(candidate["sponge_width_cells"])
            and float(row["sponge_core_min"]) == 1.0
        ),
        "candidate_boundary_matches": row["outer_boundary"] == expected_boundary,
        "source_edge_is_exact_zero": float(row["source_edge_max_abs"]) == 0.0,
        "cropped_requested_state_float64": trajectory.dtype
        == np.dtype(np.float64),
        "full_requested_state_float64": details["full_requested_state_hash"][
            "dtype"
        ]
        == np.dtype(np.float64).str,
        "row_reports_float64": row["health"]["measurement_dtype"] == "float64",
        "row_reports_finite": bool(row["health"]["finite"]),
    }
    checks = {**identity_checks, **provenance_checks, **solver_checks}
    evidence = {
        "target_cfl": float(target_cfl),
        "passed": bool(all(checks.values())),
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "checks": checks,
        "health_metrics": health_metrics,
        "diagnostic_summary": diagnostic_evidence["summary"],
        "requested_output_provenance": provenance,
        "natural_step_health": natural_health,
        "operator_diagnostics": row["health"]["operator"],
        "runtime_s": float(row["health"]["runtime_s"]),
        "array_hashes": {
            "cropped_eta_trajectory": hash_array(trajectory),
            "full_requested_state": details["full_requested_state_hash"],
            "diagnostics": diagnostic_evidence["diagnostic_array_hashes"],
        },
    }
    return trajectory, evidence


def _run_task(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    health_gates: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
    contract_hash: str,
) -> dict[str, Any]:
    started = time.monotonic()
    bathymetry, _source, _strength_array, _strength, arrays = _load_canary_arrays(
        record
    )
    production, production_evidence = _run_variant(
        record=record,
        solver=str(task["solver"]),
        candidate=candidate,
        health_gates=health_gates,
        target_cfl=float(task["production_cfl"]),
        bathymetry=bathymetry,
        arrays=arrays,
    )
    reference, reference_evidence = _run_variant(
        record=record,
        solver=str(task["solver"]),
        candidate=candidate,
        health_gates=health_gates,
        target_cfl=float(task["reference_cfl"]),
        bathymetry=bathymetry,
        arrays=arrays,
    )
    metrics = paired_cfl_metrics(
        production,
        reference,
        relative_floor_absolute_rms=float(
            metrics_config["relative_floor_absolute_rms"]
        ),
        phase_activity_floor_absolute_rms=float(
            metrics_config["phase_activity_floor_absolute_rms"]
        ),
        boundary_band_cells=int(metrics_config["boundary_band_cells"]),
    )
    pair_checks = {
        "production_variant_healthy": bool(production_evidence["passed"]),
        "reference_variant_healthy": bool(reference_evidence["passed"]),
        "requested_times_exactly_equal": (
            production_evidence["requested_output_provenance"][
                "requested_timestamps"
            ]
            == reference_evidence["requested_output_provenance"][
                "requested_timestamps"
            ]
            == candidate_requested_times().tolist()
        ),
        "production_and_reference_are_float64": (
            production.dtype == np.dtype(np.float64)
            and reference.dtype == np.dtype(np.float64)
        ),
        "paired_shapes_equal": production.shape == reference.shape,
        "metrics_finite": all(
            math.isfinite(float(value))
            for key, value in metrics.items()
            if isinstance(value, (float, int)) and key != "phase_correlation"
        ),
    }
    scientific_digest = stable_hash_payload(
        artifact_kind="common-time-v2-h2-scientific-paired-task-digest",
        payload={
            "qualified_id": task["qualified_id"],
            "input_fingerprint": task["input_fingerprint"],
            "solver": task["solver"],
            "candidate_config_hash": task["candidate_config_hash"],
            "production_cfl": task["production_cfl"],
            "reference_cfl": task["reference_cfl"],
            "production_array_hashes": production_evidence["array_hashes"],
            "reference_array_hashes": reference_evidence["array_hashes"],
            "production_checks": production_evidence["checks"],
            "reference_checks": reference_evidence["checks"],
            "production_health_metrics": production_evidence["health_metrics"],
            "reference_health_metrics": reference_evidence["health_metrics"],
            "pair_checks": pair_checks,
            "metrics": metrics,
        },
        schema_id=SCHEMA_ID,
    )
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-paired-task-result",
        "contract_hash": str(contract_hash),
        "task_id": str(task["task_id"]),
        "ordinal": int(task["ordinal"]),
        "run_kind": str(task["run_kind"]),
        "reference_primary_task_id": task["reference_primary_task_id"],
        "selection_ordinal": int(task["selection_ordinal"]),
        "qualified_id": str(task["qualified_id"]),
        "input_fingerprint": str(task["input_fingerprint"]),
        "bathymetry_type": str(task["bathymetry_type"]),
        "source_type": str(task["source_type"]),
        "solver": str(task["solver"]),
        "candidate_config_hash": str(task["candidate_config_hash"]),
        "production_cfl": float(task["production_cfl"]),
        "reference_cfl": float(task["reference_cfl"]),
        "passed_health": bool(all(pair_checks.values())),
        "failed_pair_checks": sorted(
            key for key, value in pair_checks.items() if not value
        ),
        "pair_checks": pair_checks,
        "metrics": metrics,
        "production": production_evidence,
        "reference": reference_evidence,
        "scientific_digest": scientific_digest,
        "runtime_s": float(time.monotonic() - started),
        "worker": {
            "pid": os.getpid(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "thread_environment": {
                key: os.environ.get(key) for key in THREAD_ENV_KEYS
            },
        },
    }
    identity = dict(payload)
    payload["result_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-h2-paired-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    return payload


def _task_path(root: Path, task: Mapping[str, Any]) -> Path:
    return (
        root
        / "execution"
        / "tasks"
        / f"{int(task['ordinal']):04d}-{task['task_id']}.json"
    )


def _validate_task_result(
    payload: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    contract_hash: str,
) -> None:
    for key in (
        "task_id",
        "ordinal",
        "run_kind",
        "reference_primary_task_id",
        "selection_ordinal",
        "qualified_id",
        "input_fingerprint",
        "bathymetry_type",
        "source_type",
        "solver",
        "candidate_config_hash",
        "production_cfl",
        "reference_cfl",
    ):
        if payload.get(key) != task.get(key):
            raise RuntimeError(f"H2 task result identity mismatch for {key}")
    if payload.get("contract_hash") != contract_hash:
        raise RuntimeError("H2 task result contract hash mismatch")
    identity = dict(payload)
    recorded_hash = identity.pop("result_hash", None)
    expected_hash = stable_hash_payload(
        artifact_kind="common-time-v2-h2-paired-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    if recorded_hash != expected_hash:
        raise RuntimeError(f"H2 task result hash mismatch: {task['task_id']}")
    if payload.get("passed_health") != (not payload.get("failed_pair_checks")):
        raise RuntimeError(f"H2 task health inconsistency: {task['task_id']}")


def _load_completed(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    expected_paths = {_task_path(root, task): task for task in contract["tasks"]}
    tasks_root = root / "execution" / "tasks"
    if tasks_root.exists():
        unexpected = {
            path
            for path in tasks_root.iterdir()
            if path.is_file() and path not in expected_paths
        }
        if unexpected:
            raise RuntimeError(
                f"Unexpected H2 task result files: {sorted(map(str, unexpected))}"
            )
    completed: dict[str, dict[str, Any]] = {}
    for path, task in expected_paths.items():
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_task_result(
            payload, task=task, contract_hash=str(contract["contract_hash"])
        )
        completed[str(task["task_id"])] = payload
    return completed


def _find_replay_mismatches(
    ordered: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_task_id = {str(payload["task_id"]): payload for payload in ordered}
    mismatches: list[dict[str, Any]] = []
    for payload in ordered:
        if payload["run_kind"] != "replay":
            continue
        reference_id = str(payload["reference_primary_task_id"])
        if reference_id not in by_task_id:
            raise RuntimeError(f"H2 replay reference is missing: {reference_id}")
        reference = by_task_id[reference_id]
        if (
            reference["run_kind"] != "primary"
            or reference["qualified_id"] != payload["qualified_id"]
            or reference["solver"] != payload["solver"]
            or reference["production_cfl"] != payload["production_cfl"]
            or reference["reference_cfl"] != payload["reference_cfl"]
        ):
            raise RuntimeError(f"H2 replay reference identity mismatch: {reference_id}")
        if payload["scientific_digest"] != reference["scientific_digest"]:
            mismatches.append(
                {
                    "replay_task_id": payload["task_id"],
                    "primary_task_id": reference["task_id"],
                    "qualified_id": payload["qualified_id"],
                    "solver": payload["solver"],
                    "primary_scientific_digest": reference["scientific_digest"],
                    "replay_scientific_digest": payload["scientific_digest"],
                }
            )
    return mismatches


def _bootstrap_seed(seed: str, solver: str, metric: str) -> int:
    digest = stable_hash_payload(
        artifact_kind="common-time-v2-h2-bootstrap-seed",
        payload={"seed": seed, "solver": solver, "metric": metric},
        schema_id=SCHEMA_ID,
    )
    return int(digest[:16], 16)


def _paired_stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    solver: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        cell = (str(row["bathymetry_type"]), str(row["source_type"]))
        values = cells.setdefault(cell, [])
        value = row["metrics"].get(metric)
        if value is None:
            continue
        values.append(float(value))
    if not cells:
        return {"applicable": False, "metric": metric}
    if any(not values for values in cells.values()):
        return {
            "applicable": False,
            "metric": metric,
            "reason": "at_least_one_family_cell_has_no_applicable_values",
        }
    if metric != "phase_correlation_loss" and set(map(len, cells.values())) != {4}:
        raise RuntimeError(
            f"H2 bootstrap requires four paired observations per cell: {metric}"
        )
    resamples = int(bootstrap["resamples"])
    confidence = float(bootstrap["confidence_level"])
    rng = np.random.Generator(
        np.random.PCG64(
            _bootstrap_seed(str(bootstrap["seed"]), solver, metric)
        )
    )
    medians: list[float] = []
    p95s: list[float] = []
    means: list[float] = []
    ordered_cells = [sorted(cells[cell]) for cell in sorted(cells)]
    for _ in range(resamples):
        sample: list[float] = []
        for values in ordered_cells:
            indices = rng.integers(0, len(values), size=len(values))
            sample.extend(values[int(index)] for index in indices)
        medians.append(_linear_quantile(sample, 0.5))
        p95s.append(_linear_quantile(sample, 0.95))
        means.append(_stable_mean(sample))
    tail = (1.0 - confidence) / 2.0

    def interval(values: Sequence[float]) -> list[float]:
        return [
            _linear_quantile(values, tail),
            _linear_quantile(values, 1.0 - tail),
        ]

    return {
        "applicable": True,
        "metric": metric,
        "method": bootstrap["method"],
        "seed": bootstrap["seed"],
        "derived_seed": _bootstrap_seed(
            str(bootstrap["seed"]), solver, metric
        ),
        "resamples": resamples,
        "confidence_level": confidence,
        "decision_role": bootstrap["decision_role"],
        "median_ci": interval(medians),
        "p95_ci": interval(p95s),
        "mean_ci": interval(means),
    }


def _family_cell_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        cell = (str(row["bathymetry_type"]), str(row["source_type"]))
        by_cell.setdefault(cell, []).append(row)
    summaries: list[dict[str, Any]] = []
    for cell in sorted(by_cell):
        cell_rows = by_cell[cell]
        if len(cell_rows) != 4:
            raise RuntimeError(f"H2 family cell does not contain four cases: {cell}")
        summaries.append(
            {
                "bathymetry_type": cell[0],
                "source_type": cell[1],
                "case_count": len(cell_rows),
                "qualified_ids": sorted(
                    str(row["qualified_id"]) for row in cell_rows
                ),
                "trajectory_relative_l2": _distribution(
                    [
                        float(row["metrics"]["trajectory_relative_l2"])
                        for row in cell_rows
                    ]
                ),
                "peak_amplitude_relative_error": _distribution(
                    [
                        float(row["metrics"]["peak_amplitude_relative_error"])
                        for row in cell_rows
                    ]
                ),
                "phase_correlation_loss": (
                    _distribution(
                        [
                            float(row["metrics"]["phase_correlation_loss"])
                            for row in cell_rows
                            if row["metrics"]["phase_correlation_loss"] is not None
                        ]
                    )
                    if any(
                        row["metrics"]["phase_correlation_loss"] is not None
                        for row in cell_rows
                    )
                    else {"count": 0}
                ),
            }
        )
    if len(summaries) != 30:
        raise RuntimeError("H2 family summary must contain all 30 cells")
    return summaries


def _per_time_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    times = candidate_requested_times()
    summaries: list[dict[str, Any]] = []
    for index, requested_time in enumerate(times):
        summaries.append(
            {
                "requested_time": float(requested_time),
                "absolute_rms": _distribution(
                    [
                        float(row["metrics"]["per_time_absolute_rms"][index])
                        for row in rows
                    ]
                ),
                "normalized_rmse": _distribution(
                    [
                        float(
                            row["metrics"]["per_time_normalized_rmse"][index]
                        )
                        for row in rows
                    ]
                ),
                "production_amplitude": _distribution(
                    [
                        float(
                            row["metrics"]["production_amplitude_by_time"][index]
                        )
                        for row in rows
                    ]
                ),
                "reference_amplitude": _distribution(
                    [
                        float(
                            row["metrics"]["reference_amplitude_by_time"][index]
                        )
                        for row in rows
                    ]
                ),
            }
        )
    return summaries


def _variant_health_summary(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, Any]:
    evidence = [row[variant] for row in rows]
    summary: dict[str, Any] = {
        "passed_count": sum(bool(item["passed"]) for item in evidence),
        "failed_count": sum(not bool(item["passed"]) for item in evidence),
        "runtime_s_sum": float(
            math.fsum(float(item["runtime_s"]) for item in evidence)
        ),
        "runtime_s": _distribution(
            [float(item["runtime_s"]) for item in evidence]
        ),
        "natural_steps": _distribution(
            [
                float(item["diagnostic_summary"]["total_natural_steps"])
                for item in evidence
            ]
        ),
        "max_abs_eta": max(
            float(item["health_metrics"]["max_abs_eta"]) for item in evidence
        ),
        "max_eta_over_depth": max(
            float(item["health_metrics"]["max_eta_over_depth"])
            for item in evidence
        ),
        "max_post_step_cfl": max(
            float(item["diagnostic_summary"]["max_post_step_cfl"])
            for item in evidence
        ),
    }
    if rows[0]["solver"] in {"swe_hydrostatic", "swe_muscl_hr"}:
        summary.update(
            {
                "minimum_depth": min(
                    float(item["health_metrics"]["minimum_depth"])
                    for item in evidence
                ),
                "maximum_speed": max(
                    float(item["health_metrics"]["maximum_speed"])
                    for item in evidence
                ),
                "maximum_dry_cell_count": max(
                    int(item["health_metrics"]["maximum_dry_cell_count"])
                    for item in evidence
                ),
            }
        )
    else:
        summary.update(
            {
                "cg_failure_count": sum(
                    int(item["health_metrics"]["cg_failure_count"])
                    for item in evidence
                ),
                "cg_iterations_max": max(
                    int(item["health_metrics"]["cg_iterations_max"])
                    for item in evidence
                ),
                "cg_residual_ratio_max": max(
                    float(item["health_metrics"]["cg_residual_ratio_max"])
                    for item in evidence
                ),
                "filter_application_count": sum(
                    int(
                        item["operator_diagnostics"].get(
                            "filter_applications", 0
                        )
                    )
                    for item in evidence
                ),
                "sponge_elapsed_time": _distribution(
                    [
                        float(
                            item["operator_diagnostics"].get(
                                "sponge_elapsed_time", 0.0
                            )
                        )
                        for item in evidence
                    ]
                ),
            }
        )
    return summary


def _gate(
    *,
    solver: str,
    name: str,
    observed: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "gate": name,
        "comparison": "at_or_below",
        "observed": float(observed),
        "threshold": float(threshold),
        "passed": bool(observed <= threshold),
    }


def _solver_summary_and_gates(
    solver: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != 120:
        raise RuntimeError(f"H2 expected 120 primary rows for {solver}")
    trajectory = _distribution(
        [float(row["metrics"]["trajectory_relative_l2"]) for row in rows]
    )
    per_time_values = [
        float(value)
        for row in rows
        for value in row["metrics"]["per_time_normalized_rmse"]
    ]
    per_time = _distribution(per_time_values)
    per_time_evolution = _per_time_summary(rows)
    worst_requested_time_p95 = max(
        float(entry["normalized_rmse"]["p95"])
        for entry in per_time_evolution
    )
    amplitude = _distribution(
        [
            float(row["metrics"]["peak_amplitude_relative_error"])
            for row in rows
        ]
    )
    phase_values = [
        float(row["metrics"]["phase_correlation_loss"])
        for row in rows
        if row["metrics"]["phase_correlation_loss"] is not None
    ]
    phase = _distribution(phase_values) if phase_values else {"count": 0}
    phase_fraction = len(phase_values) / len(rows)
    cells = _family_cell_summary(rows)
    worst_cell_median = max(
        float(cell["trajectory_relative_l2"]["median"]) for cell in cells
    )
    worst_cell_max = max(
        float(cell["trajectory_relative_l2"]["max"]) for cell in cells
    )
    gates = [
        _gate(
            solver=solver,
            name="trajectory_relative_l2_median",
            observed=trajectory["median"],
            threshold=thresholds["trajectory_relative_l2_median"],
        ),
        _gate(
            solver=solver,
            name="trajectory_relative_l2_p95",
            observed=trajectory["p95"],
            threshold=thresholds["trajectory_relative_l2_p95"],
        ),
        _gate(
            solver=solver,
            name="trajectory_relative_l2_max",
            observed=trajectory["max"],
            threshold=thresholds["trajectory_relative_l2_max"],
        ),
        _gate(
            solver=solver,
            name="per_time_normalized_rmse_p95",
            observed=worst_requested_time_p95,
            threshold=thresholds["per_time_normalized_rmse_p95"],
        ),
        _gate(
            solver=solver,
            name="per_time_normalized_rmse_max",
            observed=per_time["max"],
            threshold=thresholds["per_time_normalized_rmse_max"],
        ),
        _gate(
            solver=solver,
            name="peak_amplitude_relative_error_median",
            observed=amplitude["median"],
            threshold=thresholds[
                "peak_amplitude_relative_error_median"
            ],
        ),
        _gate(
            solver=solver,
            name="peak_amplitude_relative_error_p95",
            observed=amplitude["p95"],
            threshold=thresholds["peak_amplitude_relative_error_p95"],
        ),
        _gate(
            solver=solver,
            name="peak_amplitude_relative_error_max",
            observed=amplitude["max"],
            threshold=thresholds["peak_amplitude_relative_error_max"],
        ),
        _gate(
            solver=solver,
            name="family_cell_trajectory_median_max",
            observed=worst_cell_median,
            threshold=thresholds["family_cell_trajectory_median_max"],
        ),
        _gate(
            solver=solver,
            name="family_cell_trajectory_case_max",
            observed=worst_cell_max,
            threshold=thresholds["family_cell_trajectory_case_max"],
        ),
        {
            "solver": solver,
            "gate": "minimum_phase_applicable_fraction",
            "comparison": "at_or_above",
            "observed": phase_fraction,
            "threshold": float(
                metrics_config["minimum_phase_applicable_fraction"]
            ),
            "passed": phase_fraction
            >= float(metrics_config["minimum_phase_applicable_fraction"]),
        },
    ]
    if phase_values:
        gates.extend(
            [
                _gate(
                    solver=solver,
                    name="phase_correlation_loss_median",
                    observed=phase["median"],
                    threshold=thresholds[
                        "phase_correlation_loss_median"
                    ],
                ),
                _gate(
                    solver=solver,
                    name="phase_correlation_loss_p95",
                    observed=phase["p95"],
                    threshold=thresholds["phase_correlation_loss_p95"],
                ),
                _gate(
                    solver=solver,
                    name="phase_correlation_loss_max",
                    observed=phase["max"],
                    threshold=thresholds["phase_correlation_loss_max"],
                ),
            ]
        )
    bootstrap = metrics_config["bootstrap"]
    bootstrap_metrics = {
        metric: _paired_stratified_bootstrap(
            rows,
            metric=metric,
            solver=solver,
            bootstrap=bootstrap,
        )
        for metric in (
            "trajectory_relative_l2",
            "peak_amplitude_relative_error",
            "phase_correlation_loss",
        )
    }
    summary = {
        "case_count": len(rows),
        "trajectory_relative_l2": trajectory,
        "per_time_normalized_rmse": per_time,
        "worst_requested_time_normalized_rmse_p95": worst_requested_time_p95,
        "peak_amplitude_relative_error": amplitude,
        "phase_correlation_loss": phase,
        "phase_applicable_fraction": phase_fraction,
        "interior_relative_l2": _distribution(
            [float(row["metrics"]["interior_relative_l2"]) for row in rows]
        ),
        "boundary_relative_l2": _distribution(
            [float(row["metrics"]["boundary_relative_l2"]) for row in rows]
        ),
        "family_cells": cells,
        "per_time": per_time_evolution,
        "bootstrap": bootstrap_metrics,
        "production_health": _variant_health_summary(rows, "production"),
        "reference_health": _variant_health_summary(rows, "reference"),
        "gates": gates,
        "passed_sensitivity_gates": all(gate["passed"] for gate in gates),
    }
    return summary, gates


def _aggregate_results(
    ordered: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    replay_mismatches: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    primary = [row for row in ordered if row["run_kind"] == "primary"]
    failed_health = [
        str(row["task_id"]) for row in ordered if not row["passed_health"]
    ]
    by_solver: dict[str, Any] = {}
    gates: list[dict[str, Any]] = []
    for solver in SOLVERS:
        rows = [row for row in primary if row["solver"] == solver]
        summary, solver_gates = _solver_summary_and_gates(
            solver,
            rows,
            thresholds=contract["resolved_config"]["thresholds"][solver],
            metrics_config=contract["resolved_config"]["metrics"],
        )
        by_solver[solver] = summary
        gates.extend(solver_gates)
    joint_rows = list(primary)
    joint_phase_values = [
        float(row["metrics"]["phase_correlation_loss"])
        for row in joint_rows
        if row["metrics"]["phase_correlation_loss"] is not None
    ]
    summary = {
        "by_solver": by_solver,
        "joint": {
            "case_solver_pair_count": len(joint_rows),
            "trajectory_relative_l2": _distribution(
                [
                    float(row["metrics"]["trajectory_relative_l2"])
                    for row in joint_rows
                ]
            ),
            "peak_amplitude_relative_error": _distribution(
                [
                    float(row["metrics"]["peak_amplitude_relative_error"])
                    for row in joint_rows
                ]
            ),
            "phase_correlation_loss": (
                _distribution(joint_phase_values)
                if joint_phase_values
                else {"count": 0}
            ),
        },
        "health_failure_count": len(failed_health),
        "replay_count": sum(row["run_kind"] == "replay" for row in ordered),
        "replay_mismatch_count": len(replay_mismatches),
    }
    return summary, gates, failed_health


def _verify_execution_environment(
    contract: Mapping[str, Any], *, repo_root: Path
) -> None:
    current_code = code_state(repo_root)
    if current_code["code_state_hash"] != contract["code_state"]["code_state_hash"]:
        raise RuntimeError("Code state changed after H2 freeze; create a new contract")
    current_environment = _environment_snapshot()
    frozen_environment = contract["environment"]
    for key in (
        "python_version",
        "python_executable",
        "numpy_version",
        "platform",
        "machine",
        "package_inventory_hash",
    ):
        if current_environment[key] != frozen_environment[key]:
            raise RuntimeError(f"H2 execution environment changed: {key}")


def _report_text(result: Mapping[str, Any]) -> str:
    lines = [
        "# Common-time-v2 H2 result",
        "",
        f"- Contract: `{result['contract_hash']}`",
        f"- Decision: `{result['decision']}`",
        f"- Paired tasks: {result['task_count']} "
        f"({result['solver_execution_count']} solver executions)",
        f"- Failed health pairs: {len(result['failed_health_task_ids'])}",
        f"- Failed sensitivity gates: {len(result['failed_gates'])}",
        f"- Replay mismatches: {len(result['replay_mismatches'])}",
        f"- Wall duration: {float(result['wall_duration_s']):.1f} s",
        "",
        "H2 measures total temporal-discretization and production-operator "
        "sensitivity. It is not a pure temporal-order estimate. Validation and "
        "test scientific outcomes were not inspected.",
        "",
    ]
    for solver, summary in result["summary"]["by_solver"].items():
        trajectory = summary["trajectory_relative_l2"]
        lines.append(
            f"- `{solver}`: trajectory relative L2 median "
            f"{trajectory['median']:.6g}, p95 {trajectory['p95']:.6g}, "
            f"max {trajectory['max']:.6g}; sensitivity gates "
            f"{'passed' if summary['passed_sensitivity_gates'] else 'FAILED'}."
        )
    return "\n".join(lines) + "\n"


def execute_h2_contract(
    *,
    repo_root: Path,
    contract_root: Path,
    workers: int,
    max_in_flight: int | None,
    resume: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    contract_root = contract_root.resolve()
    validate_frozen_checksums(contract_root)
    contract = _read_json(contract_root / "preregistered_contract.json")
    _validate_contract_identity(contract)
    if Path(contract["config_path"]).resolve().parents[2] != repo_root:
        raise RuntimeError("H2 contract repository root mismatch")
    _verify_execution_environment(contract, repo_root=repo_root)
    prerequisites = contract["prerequisites"]
    _verify_prerequisites(
        config=contract["resolved_config"],
        h0_root=Path(prerequisites["h0"]["root"]),
        level_a_root=Path(prerequisites["level_a"]["root"]),
        level_b_bundle_root=Path(prerequisites["level_b"]["bundle_root"]),
        level_b_evaluation_root=Path(prerequisites["level_b"]["evaluation_root"]),
        h1_root=Path(prerequisites["h1"]["root"]),
    )

    worker_policy = contract["worker_policy"]
    frozen_workers = int(worker_policy["requested_workers"])
    frozen_in_flight = int(worker_policy["requested_max_in_flight"])
    effective_in_flight = frozen_in_flight if max_in_flight is None else max_in_flight
    if workers != frozen_workers or effective_in_flight != frozen_in_flight:
        raise RuntimeError(
            "H2 requires frozen workers/max-in-flight "
            f"{frozen_workers}/{frozen_in_flight}"
        )
    if workers <= 0 or effective_in_flight < workers:
        raise ValueError("H2 workers/max-in-flight must be positive and bounded")

    execution_root = contract_root / "execution"
    result_path = execution_root / "result.json"
    execution_manifest = execution_root / "SHA256SUMS.txt"
    if execution_manifest.exists():
        if not result_path.is_file():
            raise RuntimeError("H2 final checksum manifest exists without result.json")
        if not resume:
            raise FileExistsError(f"H2 result already finalized: {result_path}")
        validate_execution_checksums(execution_root)
        return result_path
    if result_path.exists() and not resume:
        raise FileExistsError(
            "Incomplete H2 finalization exists; rerun with --resume to "
            "recompute final summaries and checksums"
        )

    tasks = contract["tasks"]
    selected_by_ordinal = {
        int(entry["selection_ordinal"]): entry["record"]
        for entry in contract["selected_scenarios"]
    }
    completed = _load_completed(contract_root, contract)
    if completed and not resume:
        raise FileExistsError("Partial H2 results exist; rerun with --resume")
    pending = [task for task in tasks if str(task["task_id"]) not in completed]
    (execution_root / "tasks").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_completed = len(completed)
    if progress is not None:
        progress(
            {
                "event": "start",
                "completed": len(completed),
                "total": len(tasks),
                "pending": len(pending),
                "workers": workers,
                "max_in_flight": effective_in_flight,
            }
        )

    context = multiprocessing.get_context(str(worker_policy["process_start_method"]))
    pending_iter = iter(pending)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        active: dict[Any, Mapping[str, Any]] = {}

        def submit_until_full() -> None:
            while len(active) < effective_in_flight:
                try:
                    task = next(pending_iter)
                except StopIteration:
                    break
                record = selected_by_ordinal[int(task["selection_ordinal"])]
                future = executor.submit(
                    _run_task,
                    task,
                    record,
                    contract["resolved_config"]["candidate"],
                    contract["resolved_config"]["health_gates"],
                    contract["resolved_config"]["metrics"],
                    contract["contract_hash"],
                )
                active[future] = task

        submit_until_full()
        while active:
            done, _not_done = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                payload = future.result()
                _validate_task_result(
                    payload,
                    task=task,
                    contract_hash=str(contract["contract_hash"]),
                )
                _write_json(_task_path(contract_root, task), payload)
                completed[str(task["task_id"])] = payload
                if progress is not None:
                    elapsed = time.monotonic() - started
                    completed_this_run = len(completed) - initial_completed
                    remaining = len(tasks) - len(completed)
                    eta_s = (
                        elapsed * remaining / completed_this_run
                        if completed_this_run
                        else None
                    )
                    progress(
                        {
                            "event": "task_complete",
                            "completed": len(completed),
                            "total": len(tasks),
                            "run_kind": task["run_kind"],
                            "qualified_id": task["qualified_id"],
                            "solver": task["solver"],
                            "runtime_s": payload["runtime_s"],
                            "passed": payload["passed_health"],
                            "elapsed_s": elapsed,
                            "eta_s": eta_s,
                        }
                    )
            submit_until_full()

    ordered = [completed[str(task["task_id"])] for task in tasks]
    replay_mismatches = _find_replay_mismatches(ordered)
    summary, gates, failed_health = _aggregate_results(
        ordered,
        contract=contract,
        replay_mismatches=replay_mismatches,
    )
    failed_gates = [gate for gate in gates if not gate["passed"]]
    if failed_health or replay_mismatches:
        decision = contract["decision_policy"]["health_failure"]
    elif failed_gates:
        decision = contract["decision_policy"]["sensitivity_failure"]
    else:
        decision = contract["decision_policy"]["pass"]
    result: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-result",
        "contract_hash": contract["contract_hash"],
        "decision": decision,
        "h2_passed": decision == contract["decision_policy"]["pass"],
        "claim_scope": contract["claim_scope"],
        "task_count": len(ordered),
        "primary_task_count": sum(
            row["run_kind"] == "primary" for row in ordered
        ),
        "replay_task_count": sum(
            row["run_kind"] == "replay" for row in ordered
        ),
        "solver_execution_count": 2 * len(ordered),
        "failed_health_task_ids": failed_health,
        "failed_gates": failed_gates,
        "gates": gates,
        "replay_mismatches": replay_mismatches,
        "wall_duration_s": float(time.monotonic() - started),
        "sum_task_runtime_s": float(
            math.fsum(float(row["runtime_s"]) for row in ordered)
        ),
        "effective_workers": workers,
        "effective_max_in_flight": effective_in_flight,
        "summary": summary,
        "task_rows": [
            {
                "ordinal": row["ordinal"],
                "task_id": row["task_id"],
                "run_kind": row["run_kind"],
                "qualified_id": row["qualified_id"],
                "bathymetry_type": row["bathymetry_type"],
                "source_type": row["source_type"],
                "solver": row["solver"],
                "production_cfl": row["production_cfl"],
                "reference_cfl": row["reference_cfl"],
                "passed_health": row["passed_health"],
                "failed_pair_checks": row["failed_pair_checks"],
                "metrics": row["metrics"],
                "runtime_s": row["runtime_s"],
                "production_health_metrics": row["production"]["health_metrics"],
                "reference_health_metrics": row["reference"]["health_metrics"],
                "production_diagnostic_summary": row["production"][
                    "diagnostic_summary"
                ],
                "reference_diagnostic_summary": row["reference"][
                    "diagnostic_summary"
                ],
                "scientific_digest": row["scientific_digest"],
            }
            for row in ordered
        ],
        "operational_provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "process_start_method": str(worker_policy["process_start_method"]),
            "thread_environment": {
                key: os.environ.get(key) for key in THREAD_ENV_KEYS
            },
            "workers": sorted(
                {
                    (
                        int(row["worker"]["pid"]),
                        str(row["worker"]["python_version"]),
                        str(row["worker"]["numpy_version"]),
                        json.dumps(
                            row["worker"]["thread_environment"], sort_keys=True
                        ),
                    )
                    for row in ordered
                }
            ),
        },
        "validation_and_test_scientific_outcomes_inspected": False,
        "accepted_contract_frozen": False,
        "mass_generation_authorized": False,
    }
    _write_json(result_path, result)
    _write_json(
        execution_root / "decision.json",
        {
            "schema_id": SCHEMA_ID,
            "contract_hash": contract["contract_hash"],
            "decision": decision,
            "h2_passed": result["h2_passed"],
            "failed_health_task_ids": failed_health,
            "failed_gate_count": len(failed_gates),
            "replay_mismatch_count": len(replay_mismatches),
            "accepted_contract_frozen": False,
            "mass_generation_authorized": False,
        },
    )
    _write_text(execution_root / "REPORT.md", _report_text(result))
    _write_execution_checksums(execution_root)
    validate_execution_checksums(execution_root)
    if progress is not None:
        progress(
            {
                "event": "finalized",
                "completed": len(ordered),
                "total": len(tasks),
                "decision": decision,
                "passed": result["h2_passed"],
                "duration_s": result["wall_duration_s"],
            }
        )
    return result_path


def h2_status(contract_root: Path) -> dict[str, Any]:
    contract_root = contract_root.resolve()
    validate_frozen_checksums(contract_root)
    contract = _read_json(contract_root / "preregistered_contract.json")
    _validate_contract_identity(contract)
    completed = _load_completed(contract_root, contract)
    result_path = contract_root / "execution" / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    return {
        "contract_hash": contract["contract_hash"],
        "selected_scenario_count": len(contract["selected_scenarios"]),
        "completed": len(completed),
        "total": len(contract["tasks"]),
        "pending": len(contract["tasks"]) - len(completed),
        "failed_completed_health_tasks": sorted(
            task_id
            for task_id, payload in completed.items()
            if not payload["passed_health"]
        ),
        "finalized": result is not None,
        "decision": None if result is None else result["decision"],
    }
