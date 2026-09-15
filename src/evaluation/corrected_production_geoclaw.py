"""Freeze, run, and summarize a GeoClaw comparator for the corrected R6 data.

The historical established-solver package cannot be reused for this purpose:
it is bound to the earlier 96-cell domain and normalized 0.175 time horizon.
This module instead freezes exact 128-cell R6 inputs, recreates the 192-cell
buffered initial state, and compares GeoClaw SWE with the already-published
Hydrostatic and MUSCL-HR R6 trajectories at their 50 shared output times.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from src.data_gen.common_time_v2 import hash_array, sha256_file, stable_hash_payload
from src.data_gen.simulate_dataset import BufferedDomainConfig, _prepare_buffered_domain
from src.evaluation.common_time_v2_level_a import validate_checksums
from src.evaluation.established_solver_validation import (
    CORRECTED_PRODUCTION_GEOCLAW_EXTERNAL_RESULT_SCHEMA_ID,
    CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID,
    _comparison_metrics_v4,
    _load_external_result,
    _load_external_run_manifest,
    _validate_external_checksums,
    _write_checksums,
)
from src.evaluation.geoclaw_adapter import GeoClawEnvironment, run_geoclaw_bundle


ARTIFACT_KIND = "corrected-production-geoclaw-swe-comparison"
SOLVER_DIRECTORY = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([_json_safe(row) for row in rows])


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Corrected-production GeoClaw configuration must be a mapping")
    if payload.get("schema_id") != CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID:
        raise ValueError("Corrected-production GeoClaw schema mismatch")
    if payload.get("artifact_kind") != "corrected-production-geoclaw-swe-comparison-candidate":
        raise ValueError("Corrected-production GeoClaw artifact kind mismatch")
    production = payload.get("corrected_production")
    if not isinstance(production, Mapping):
        raise ValueError("Corrected-production GeoClaw configuration has no production contract")
    expected_shapes = {
        "master_shape": [384, 384],
        "solver_input_shape": [128, 128],
        "computational_shape": [192, 192],
        "publication_shape": [64, 64],
    }
    for key, expected in expected_shapes.items():
        if list(production.get(key, [])) != expected:
            raise ValueError(f"Corrected-production GeoClaw {key} mismatch")
    mapping = production.get("publication_mapping")
    if (
        not isinstance(mapping, Mapping)
        or list(mapping.get("crop_shape", [])) != [128, 128]
        or list(mapping.get("block_mean_reduction", [])) != [2, 2]
    ):
        raise ValueError("Corrected-production GeoClaw publication mapping mismatch")
    if int(production.get("buffer_cells", 0)) != 32:
        raise ValueError("Corrected-production GeoClaw buffer-cell mismatch")
    requested = production.get("requested_times")
    if not isinstance(requested, Mapping):
        raise ValueError("Corrected-production GeoClaw requested-time policy is missing")
    expected_times = _requested_times(requested)
    if expected_times.shape != (50,) or expected_times[0] != 8.4 or expected_times[-1] != 420.0:
        raise ValueError("Corrected-production GeoClaw requested-time grid mismatch")
    time_tolerance = float(production.get("requested_time_abs_tolerance", 0.0))
    if not math.isfinite(time_tolerance) or not 5.0e-14 < time_tolerance <= 1.0e-12:
        raise ValueError("Corrected-production GeoClaw requested-time tolerance mismatch")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping) or not bool(selection.get("outcome_blind")):
        raise ValueError("Corrected-production GeoClaw selection must be outcome-blind")
    cases = selection.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("Corrected-production GeoClaw requires exactly three frozen cases")
    indices = [int(case.get("sample_index", 0)) for case in cases if isinstance(case, Mapping)]
    if len(indices) != 3 or len(set(indices)) != 3 or any(index <= 0 for index in indices):
        raise ValueError("Corrected-production GeoClaw cases must have unique positive indices")
    if list(payload.get("inhouse_references", [])) != [
        "swe_hydrostatic",
        "swe_muscl_hr",
    ]:
        raise ValueError("Corrected-production GeoClaw in-house solver scope mismatch")
    comparator = payload.get("external_comparator")
    if not isinstance(comparator, Mapping) or comparator.get("mode") != "standard_single_level_cartesian_swe":
        raise ValueError("Corrected-production GeoClaw comparator scope mismatch")
    revisions = comparator.get("expected_revisions")
    if not isinstance(revisions, Mapping) or set(revisions) != {
        "clawpack_commit",
        "geoclaw_commit",
        "petsc_commit",
        "petsc_options_sha256",
    }:
        raise ValueError("Corrected-production GeoClaw revision pins are incomplete")
    execution = payload.get("external_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Corrected-production GeoClaw execution policy is missing")
    if execution.get("output_t0_for_initial_state_verification") is not True:
        raise ValueError("Corrected-production GeoClaw must verify t=0 mapping")
    if execution.get("initial_state_mapping") != "exact_cell_centered_custom_qinit_and_setaux":
        raise ValueError("Corrected-production GeoClaw initial-state mapping mismatch")
    if execution.get("amr_levels") != 1 or execution.get("output_format") != "ascii":
        raise ValueError("Corrected-production GeoClaw execution mode mismatch")
    return payload


def _requested_times(spec: Mapping[str, Any]) -> np.ndarray:
    count = int(spec["count"])
    start = float(spec["start"])
    step = float(spec["step"])
    values = start + step * np.arange(count, dtype=np.float64)
    values[-1] = float(spec["horizon"])
    if values[0] != start or not np.all(np.diff(values) > 0.0):
        raise ValueError("Invalid corrected-production GeoClaw requested-time grid")
    return values


def _npz_scalar(payload: Mapping[str, np.ndarray], key: str) -> str:
    return str(np.asarray(payload[key]).reshape(-1)[0])


def _sample_name(index: int) -> str:
    return f"sample_{index:06d}"


def _sample_input(
    *,
    repo_root: Path,
    production: Mapping[str, Any],
    selected: Mapping[str, Any],
    expected_times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    index = int(selected["sample_index"])
    sample_name = _sample_name(index)
    bathymetry_path = repo_root / str(production["bathymetry_root"]) / f"{sample_name}.npz"
    source_path = repo_root / str(production["source_root"]) / f"{sample_name}.npz"
    if not bathymetry_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(f"Missing frozen R6 input for {sample_name}")
    with np.load(bathymetry_path, allow_pickle=False) as payload:
        bathymetry_type = _npz_scalar(payload, "bathymetry_type")
        solver_bathymetry = np.asarray(payload["solver_bathymetry"], dtype=np.float32)
        master_bathymetry = np.asarray(payload["master_bathymetry"], dtype=np.float32)
        recorded_solver_hash = _npz_scalar(payload, "native_solver_array_sha256")
        recorded_master_hash = _npz_scalar(payload, "native_master_array_sha256")
    with np.load(source_path, allow_pickle=False) as payload:
        source_type = _npz_scalar(payload, "source_type")
        source_strength = float(np.asarray(payload["source_strength"]).reshape(-1)[0])
        solver_source = np.asarray(payload["solver_source_field"], dtype=np.float32)
        master_source = np.asarray(payload["master_source_field"], dtype=np.float32)
        recorded_solver_source_hash = _npz_scalar(payload, "native_solver_array_sha256")
        recorded_master_source_hash = _npz_scalar(payload, "native_master_array_sha256")
    if bathymetry_type != str(selected["bathymetry_type"]) or source_type != str(selected["source_type"]):
        raise RuntimeError(f"Frozen stratum mismatch for {sample_name}")
    if solver_bathymetry.shape != (128, 128) or solver_source.shape != (128, 128):
        raise RuntimeError(f"R6 solver-input shape mismatch for {sample_name}")
    if master_bathymetry.shape != (384, 384) or master_source.shape != (384, 384):
        raise RuntimeError(f"R6 master-input shape mismatch for {sample_name}")
    for label, array, recorded in (
        ("solver bathymetry", solver_bathymetry, recorded_solver_hash),
        ("solver source", solver_source, recorded_solver_source_hash),
        ("master bathymetry", master_bathymetry, recorded_master_hash),
        ("master source", master_source, recorded_master_source_hash),
    ):
        if hash_array(array)["sha256"] != recorded:
            raise RuntimeError(f"R6 {label} hash mismatch for {sample_name}")
    prepared = _prepare_buffered_domain(
        solver_bathymetry,
        solver_source,
        source_strength,
        sea_level_offset=0.0,
        config=BufferedDomainConfig(
            enabled=True,
            buffer_cells=32,
            source_taper_cells=16,
            bathymetry_extension="edge",
            output_crop="central",
        ),
        source_type=source_type,
        source_already_tapered=True,
    )
    crop = prepared["crop"]
    arrays = {
        "bathymetry": np.asarray(prepared["solver_bathymetry"], dtype=np.float64),
        "eta0": np.asarray(prepared["solver_eta0"], dtype=np.float64),
        "initial_depth": np.asarray(prepared["solver_h0"], dtype=np.float64),
        "hu0": np.zeros((192, 192), dtype=np.float64),
        "hv0": np.zeros((192, 192), dtype=np.float64),
        "requested_times": np.asarray(expected_times, dtype=np.float64),
        "output_crop": np.asarray(
            [crop[0].start, crop[0].stop, crop[1].start, crop[1].stop], dtype=np.int64
        ),
        "domain_bounds": np.asarray([0.0, 3600.0, 0.0, 3600.0], dtype=np.float64),
    }
    if arrays["bathymetry"].shape != (192, 192):
        raise RuntimeError(f"R6 buffered computation shape mismatch for {sample_name}")
    input_metadata = {
        "sample_index": index,
        "scenario_id": f"scenario_{index:06d}",
        "qualified_id": f"test:scenario_{index:06d}",
        "bathymetry_type": bathymetry_type,
        "source_type": source_type,
        "source_strength": source_strength,
        "bathymetry_path": str(bathymetry_path.relative_to(repo_root)),
        "source_path": str(source_path.relative_to(repo_root)),
        "bathymetry_file_sha256": sha256_file(bathymetry_path),
        "source_file_sha256": sha256_file(source_path),
        "master_bathymetry": hash_array(master_bathymetry),
        "master_source": hash_array(master_source),
        "solver_bathymetry": hash_array(solver_bathymetry),
        "solver_source": hash_array(solver_source),
        "buffered_bathymetry": hash_array(arrays["bathymetry"]),
        "buffered_initial_depth": hash_array(arrays["initial_depth"]),
        "buffered_nominal_eta0": hash_array(arrays["eta0"]),
        "source_edge_max_abs": float(prepared["source_edge_max_abs"]),
    }
    return arrays, input_metadata, {
        "solver_bathymetry": solver_bathymetry,
        "solver_source": solver_source,
    }


def _load_inhouse_trajectory(
    *,
    repo_root: Path,
    production: Mapping[str, Any],
    solver_name: str,
    index: int,
    expected_times: np.ndarray,
    input_metadata: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_name = _sample_name(index)
    directory = SOLVER_DIRECTORY[solver_name]
    sample_dir = repo_root / str(production["raw_root"]) / directory / "samples" / sample_name
    metadata_path = sample_dir / "meta.json"
    sample_path = sample_dir / "sample.npz"
    if not metadata_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError(f"Missing current R6 {solver_name} publication for {sample_name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract_hash") != production["contract_hash"]:
        raise RuntimeError(f"R6 contract mismatch in {metadata_path}")
    if metadata.get("qualified_id") != input_metadata["qualified_id"]:
        raise RuntimeError(f"R6 identity mismatch in {metadata_path}")
    if metadata.get("solver_name") != solver_name:
        raise RuntimeError(f"R6 solver identity mismatch in {metadata_path}")
    if metadata.get("quality_status") != "ok":
        raise RuntimeError(f"R6 quality status is not ok in {metadata_path}")
    domain = metadata.get("computational_domain", {})
    if domain.get("solver_shape") != [192, 192] or domain.get("publication_shape") != [64, 64]:
        raise RuntimeError(f"R6 computational-domain mismatch in {metadata_path}")
    lineage = metadata.get("input_lineage", {})
    if lineage.get("solver_bathymetry_sha256") != input_metadata["solver_bathymetry"]["sha256"]:
        raise RuntimeError(f"R6 solver-bathymetry lineage mismatch in {metadata_path}")
    if lineage.get("solver_source_sha256") != input_metadata["solver_source"]["sha256"]:
        raise RuntimeError(f"R6 solver-source lineage mismatch in {metadata_path}")
    with np.load(sample_path, allow_pickle=False) as payload:
        trajectory = np.asarray(payload["trajectory_eta"], dtype=np.float64)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        sample_contract = _npz_scalar(payload, "contract_hash")
    if sample_contract != production["contract_hash"] or not np.array_equal(timestamps, expected_times):
        raise RuntimeError(f"R6 requested-time mismatch in {sample_path}")
    if trajectory.shape != (expected_times.size, 64, 64) or not np.isfinite(trajectory).all():
        raise RuntimeError(f"R6 trajectory health mismatch in {sample_path}")
    return trajectory, {
        "sample_path": str(sample_path.relative_to(repo_root)),
        "sample_file_sha256": sha256_file(sample_path),
        "meta_path": str(metadata_path.relative_to(repo_root)),
        "meta_file_sha256": sha256_file(metadata_path),
        "trajectory": hash_array(trajectory),
        "timestamps": hash_array(timestamps),
        "generation_code_state_hash": metadata.get("code_state_hash"),
        "generation_resolved_config_hash": metadata.get("resolved_config_hash"),
    }


def _gauge_indices(locations: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(locations, list):
        raise ValueError("Corrected-production GeoClaw gauge locations must be a list")
    indices: list[tuple[int, int]] = []
    for location in locations:
        if not isinstance(location, list) or len(location) != 2:
            raise ValueError("Corrected-production GeoClaw gauge location is malformed")
        i = min(shape[0] - 1, max(0, int(float(location[0]) * shape[0])))
        j = min(shape[1] - 1, max(0, int(float(location[1]) * shape[1])))
        indices.append((i, j))
    return np.asarray(indices, dtype=np.int64)


def prepare_corrected_production_geoclaw_bundle(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path | None = None,
) -> Path:
    """Freeze current R6 inputs and trajectory outputs before GeoClaw is run."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = _load_config(config_path)
    production = config["corrected_production"]
    expected_times = _requested_times(production["requested_times"])
    cases: list[dict[str, Any]] = []
    staged_cases: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for selected in config["selection"]["cases"]:
        arrays, input_metadata, _original_arrays = _sample_input(
            repo_root=repo_root,
            production=production,
            selected=selected,
            expected_times=expected_times,
        )
        index = int(selected["sample_index"])
        inhouse: dict[str, np.ndarray] = {}
        inhouse_metadata: dict[str, Any] = {}
        for solver_name in config["inhouse_references"]:
            trajectory, trajectory_metadata = _load_inhouse_trajectory(
                repo_root=repo_root,
                production=production,
                solver_name=str(solver_name),
                index=index,
                expected_times=expected_times,
                input_metadata=input_metadata,
            )
            inhouse[str(solver_name)] = trajectory
            inhouse_metadata[str(solver_name)] = trajectory_metadata
        case_payload = {
            "selection": dict(selected),
            "input": input_metadata,
            "inhouse": inhouse_metadata,
            "output_crop": arrays["output_crop"].tolist(),
            "domain_bounds": arrays["domain_bounds"].tolist(),
            "requested_times": expected_times.tolist(),
        }
        case_hash = stable_hash_payload(
            artifact_kind="corrected-production-geoclaw-case",
            payload=case_payload,
            schema_id=CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID,
        )
        case_id = f"test_{index:06d}_{input_metadata['bathymetry_type']}_{input_metadata['source_type']}"
        record = {
            "case_id": case_id,
            "case_hash": case_hash,
            "boundary": "open_extrapolation",
            "external_domain": {
                "shape": [192, 192],
                "bounds": arrays["domain_bounds"].tolist(),
                "output_crop": arrays["output_crop"].tolist(),
                "publication_reduction": list(
                    production["publication_mapping"]["block_mean_reduction"]
                ),
                "boundary": "open_extrapolation",
            },
            "publication_shape": [64, 64],
            "selection": dict(selected),
            "input": input_metadata,
            "inhouse": inhouse_metadata,
        }
        arrays = {**arrays, "case_hash": np.asarray(case_hash)}
        cases.append(record)
        staged_cases.append((record, arrays, inhouse))
    requirements = [
        {
            "case_id": record["case_id"],
            "case_hash": record["case_hash"],
            "comparator_id": "geoclaw_swe",
            "comparator_version": str(config["external_comparator"]["version"]),
            "result_schema_id": CORRECTED_PRODUCTION_GEOCLAW_EXTERNAL_RESULT_SCHEMA_ID,
            "relative_path": f"{record['case_id']}/geoclaw_swe.npz",
            "required_npz_keys": [
                "schema_id",
                "case_hash",
                "comparator_id",
                "comparator_version",
                "comparator_commit",
                "clawpack_commit",
                "petsc_commit",
                "adapter_hash",
                "times",
                "actual_times",
                "eta",
                "runtime_seconds",
                "initial_state_max_abs_error",
                "requested_time_max_abs_error",
                "nominal_eta_max_abs_difference",
                "nominal_eta_consistency_floor",
                "solver_health_status",
            ],
            "eta_shape": [int(expected_times.size), 64, 64],
            "publication_reduction": record["external_domain"]["publication_reduction"],
            "requested_time_abs_tolerance": float(
                production["requested_time_abs_tolerance"]
            ),
            "computational_shape": [192, 192],
            "output_crop": record["external_domain"]["output_crop"],
            "computational_domain_bounds": record["external_domain"]["bounds"],
        }
        for record in cases
    ]
    frozen = {
        "schema_id": CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID,
        "artifact_kind": "corrected-production-geoclaw-swe-frozen-contract",
        "source_config": _json_safe(config),
        "source_config_sha256": sha256_file(config_path),
        "corrected_production_contract_hash": production["contract_hash"],
        "requested_times": expected_times.tolist(),
        "cases": cases,
        "external_results": requirements,
        "comparison_scope": config["comparison"],
    }
    bundle_hash = stable_hash_payload(
        artifact_kind="corrected-production-geoclaw-bundle",
        payload=frozen,
        schema_id=CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID,
    )
    frozen["bundle_hash"] = bundle_hash
    base = output_root or repo_root / "artifacts/corrected_production_geoclaw/bundles"
    final = base.resolve() / bundle_hash
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite corrected-production GeoClaw bundle: {final}")
    staging = final.parent / f".{bundle_hash}.staging"
    if staging.exists():
        raise FileExistsError(f"Stale corrected-production GeoClaw staging path: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(staging / "frozen_contract.json", frozen)
        _write_json(
            staging / "external_results_manifest.json",
            {
                "schema_id": CORRECTED_PRODUCTION_GEOCLAW_EXTERNAL_RESULT_SCHEMA_ID,
                "bundle_hash": bundle_hash,
                "results": requirements,
            },
        )
        for record, arrays, inhouse in staged_cases:
            case_dir = staging / "cases" / str(record["case_id"])
            case_dir.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(case_dir / "input.npz", **arrays)
            for solver_name, trajectory in inhouse.items():
                np.savez_compressed(
                    case_dir / f"inhouse_{solver_name}.npz",
                    eta=trajectory,
                    times=expected_times,
                    case_hash=np.asarray(record["case_hash"]),
                    solver_id=np.asarray(solver_name),
                )
        (staging / "README.md").write_text(
            "# Corrected-production GeoClaw SWE comparator\n\n"
            f"- Bundle hash: `{bundle_hash}`\n"
            f"- R6 contract: `{production['contract_hash']}`\n"
            "- Inputs: exact 128-cell solver inputs buffered to the 192-cell computation\n"
            "- Outputs: the central 64-cell crop at 50 requested times from 8.4 to 420.0\n"
            "- Scope: independent SWE compatibility diagnostic for Hydrostatic and MUSCL-HR\n\n"
            "This does not regenerate or replace the historical Level A/H1/H2 studies.\n",
            encoding="utf-8",
        )
        _write_checksums(staging)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def run_corrected_production_geoclaw(
    *,
    bundle_root: Path,
    external_root: Path,
    environment: GeoClawEnvironment,
    workers: int = 1,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen R6 bundle through the shared, strict GeoClaw adapter."""

    return run_geoclaw_bundle(
        bundle_root=bundle_root,
        external_root=external_root,
        environment=environment,
        workers=workers,
        resume=resume,
        comparator_ids=["geoclaw_swe"],
        progress=progress,
    )


def evaluate_corrected_production_geoclaw(
    *,
    bundle_root: Path,
    external_root: Path,
    output_root: Path,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Validate frozen external results and report descriptive inter-code metrics."""

    bundle_root = bundle_root.resolve()
    external_root = external_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite corrected-production GeoClaw evaluation: {output_root}")
    validate_checksums(bundle_root)
    frozen = json.loads((bundle_root / "frozen_contract.json").read_text(encoding="utf-8"))
    if frozen.get("schema_id") != CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID:
        raise RuntimeError("Corrected-production GeoClaw bundle schema mismatch")
    _validate_external_checksums(external_root, frozen)
    run_manifest = _load_external_run_manifest(external_root, frozen)
    times = np.asarray(frozen["requested_times"], dtype=np.float64)
    requirement_by_case = {str(row["case_id"]): row for row in frozen["external_results"]}
    comparison = frozen["comparison_scope"]
    gauges = _gauge_indices(comparison["gauges_fractional_cell_locations"], (64, 64))
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(frozen["cases"], start=1):
        case_id = str(case["case_id"])
        requirement = requirement_by_case[case_id]
        external_eta, external_metadata = _load_external_result(
            external_root / str(requirement["relative_path"]),
            requirement,
            times,
            run_manifest,
        )
        for solver_name in frozen["source_config"]["inhouse_references"]:
            with np.load(
                bundle_root / "cases" / case_id / f"inhouse_{solver_name}.npz",
                allow_pickle=False,
            ) as payload:
                inhouse_eta = np.asarray(payload["eta"], dtype=np.float64)
                inhouse_times = np.asarray(payload["times"], dtype=np.float64)
                stored_case_hash = _npz_scalar(payload, "case_hash")
            if stored_case_hash != case["case_hash"] or not np.array_equal(inhouse_times, times):
                raise RuntimeError(f"Frozen R6 in-house trajectory identity mismatch: {case_id}")
            metrics = _comparison_metrics_v4(
                inhouse_eta,
                external_eta,
                times,
                gauges,
                inactive_floor=float(comparison["inactive_external_peak_floor"]),
                per_time_signal_floor_fraction=float(comparison["per_time_signal_floor_fraction"]),
                peak_plateau_fraction=float(comparison["peak_plateau_fraction"]),
                lag_minimum_overlap_fraction=float(comparison["lag_minimum_overlap_fraction"]),
                diagnostic_boundary_band_cells=int(comparison["diagnostic_boundary_band_cells"]),
            )
            rows.append(
                {
                    "case_id": case_id,
                    "sample_index": case["input"]["sample_index"],
                    "bathymetry_type": case["input"]["bathymetry_type"],
                    "source_type": case["input"]["source_type"],
                    "inhouse_solver": solver_name,
                    "external_comparator": "geoclaw_swe",
                    "external_runtime_seconds": external_metadata["runtime_seconds"],
                    "initial_state_max_abs_error": external_metadata["initial_state_max_abs_error"],
                    "requested_time_max_abs_error": external_metadata["requested_time_max_abs_error"],
                    **metrics,
                }
            )
        if progress is not None:
            progress(f"[corrected-geoclaw-evaluate] {case_index}/{len(frozen['cases'])} {case_id}")
    output_root.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_id": CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID,
        "artifact_kind": "corrected-production-geoclaw-swe-evaluation",
        "status": "completed_descriptive_compatibility_diagnostic",
        "bundle_hash": frozen["bundle_hash"],
        "corrected_production_contract_hash": frozen["corrected_production_contract_hash"],
        "case_count": len(frozen["cases"]),
        "comparison_count": len(rows),
        "requested_time_count": int(times.size),
        "requested_time_start": float(times[0]),
        "requested_time_horizon": float(times[-1]),
        "computational_shape": [192, 192],
        "publication_shape": [64, 64],
        "external_solver_is_truth": False,
        "comparison_role": comparison["role"],
        "boundary_interpretation": comparison["boundary_interpretation"],
        "run_manifest": str((external_root / "RUN_MANIFEST.json")),
        "external_revisions": run_manifest["revisions"],
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "comparison_rows.json", rows)
    _write_csv(output_root / "comparison_rows.csv", rows)
    (output_root / "README.md").write_text(
        "# Corrected-production GeoClaw SWE comparison\n\n"
        "The completed run validates exact frozen R6 input identity, 192-cell domain, "
        "64-cell publication crop, t=0 state mapping, 50 requested output times, "
        "and finite GeoClaw output. The reported field metrics are descriptive "
        "inter-code compatibility diagnostics; GeoClaw is not treated as physical truth.\n",
        encoding="utf-8",
    )
    _write_checksums(output_root)
    return output_root
