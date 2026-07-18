from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    code_state,
    hash_array,
    sha256_file,
    stable_hash_payload,
)
from src.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    _prepare_buffered_domain,
)
from src.evaluation.common_time_v2_level_a import (
    execute_level_a,
    _load_canary_arrays,
    _trajectory_eta,
    validate_checksums,
)


SCHEMA_ID = "tsunami-surrogate.minimum-established-solver-validation.v2"
SCHEMA_ID_V3 = "tsunami-surrogate.minimum-established-solver-validation.v3"
SCHEMA_ID_V4 = "tsunami-surrogate.minimum-established-solver-validation.v4"
SUPPORTED_SCHEMA_IDS = (SCHEMA_ID, SCHEMA_ID_V3, SCHEMA_ID_V4)
HARDENED_SCHEMA_IDS = (SCHEMA_ID_V3, SCHEMA_ID_V4)
EXTERNAL_RESULT_SCHEMA_ID = (
    "tsunami-surrogate.minimum-established-solver-external-result.v2"
)
EXTERNAL_RESULT_SCHEMA_ID_V3 = (
    "tsunami-surrogate.minimum-established-solver-external-result.v3"
)
EXTERNAL_ACTUAL_TIME_ABS_TOLERANCE = 5.0e-14
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
COMPARATORS = ("geoclaw_swe", "geoclaw_sgn")


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([_json_safe(row) for row in rows])


def _write_checksums(root: Path) -> None:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _requested_times(config: Mapping[str, Any]) -> np.ndarray:
    raw = config["requested_times"]
    count = int(raw["count"])
    values = float(raw["step"]) * np.arange(1, count + 1, dtype=np.float64)
    values[-1] = float(raw["horizon"])
    if values[0] != float(raw["start"]) or not np.all(np.diff(values) > 0.0):
        raise ValueError("Invalid established-solver requested-time grid")
    return values


def _config_schema_id(config: Mapping[str, Any]) -> str:
    schema_id = str(config.get("schema_id", ""))
    if schema_id not in SUPPORTED_SCHEMA_IDS:
        raise ValueError("Established-solver validation schema mismatch")
    return schema_id


def _external_result_schema_id(schema_id: str) -> str:
    if schema_id == SCHEMA_ID:
        return EXTERNAL_RESULT_SCHEMA_ID
    if schema_id in HARDENED_SCHEMA_IDS:
        return EXTERNAL_RESULT_SCHEMA_ID_V3
    raise ValueError(f"Unsupported established-solver schema: {schema_id}")


def _validate_config(config: Mapping[str, Any]) -> None:
    schema_id = _config_schema_id(config)
    if config.get("artifact_kind") != "minimum-established-solver-validation-candidate":
        raise ValueError("Established-solver validation artifact kind mismatch")
    if config["external_comparator"].get("version") != "5.14.0":
        raise ValueError("The minimum package must pin Clawpack 5.14.0")
    if schema_id in HARDENED_SCHEMA_IDS:
        revisions = config["external_comparator"].get("expected_revisions")
        if not isinstance(revisions, Mapping) or set(revisions) != {
            "clawpack_commit",
            "geoclaw_commit",
            "petsc_commit",
            "petsc_options_sha256",
        }:
            raise ValueError("v3 external revision pins are incomplete")
        if any(
            len(str(value)) != 40
            for key, value in revisions.items()
            if key != "petsc_options_sha256"
        ) or len(str(revisions["petsc_options_sha256"])) != 64:
            raise ValueError("v3 external revision pins are malformed")
    if not bool(config["prerequisites"].get("require_level_a_pass")):
        raise ValueError("The minimum package must require a passing Level A")
    prerequisites = config["prerequisites"]
    if bool(prerequisites.get("require_unchanged_code_state")):
        raise ValueError(
            "The reviewed comparator policy must use scoped post-Level-A provenance"
        )
    if not bool(prerequisites.get("require_completed_level_a_replay")):
        raise ValueError("The minimum package must replay-validate completed Level A")
    if prerequisites.get("code_state_policy") != (
        "freeze_current_bundle_code_and_record_scoped_post_level_a_changes"
    ):
        raise ValueError("Established-solver code-state policy mismatch")
    if set(prerequisites.get("allowed_post_level_a_scopes", [])) != {
        "completed_level_a_raw_csv_newline_validator",
        "established_solver_bundle_preparation_and_evaluation",
    }:
        raise ValueError("Established-solver allowed post-Level-A scopes changed")
    if set(prerequisites.get("forbidden_post_level_a_scopes", [])) != {
        "solver_numerics",
        "dataset_generation_semantics",
        "level_a_scientific_tasks_thresholds_or_metrics",
    }:
        raise ValueError("Established-solver forbidden post-Level-A scopes changed")
    execution = config.get("external_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Established-solver external execution policy is missing")
    expected_execution = {
        "spatial_order": 2,
        "dimensional_split": "unsplit",
        "transverse_waves": 2,
        "limiter": "mc",
        "use_fwaves": True,
        "source_split": "godunov",
        "cfl_desired": 0.75,
        "cfl_max": 1.0,
        "dt_initial": 1.0e-4,
        "steps_max": 100000,
        "num_ghost": 2,
        "coordinate_system": 1,
        "sea_level": 0.0,
        "dry_tolerance": 1.0e-12,
        "friction_forcing": False,
        "coriolis_forcing": False,
        "amr_levels": 1,
        "output_format": "ascii",
        "output_t0_for_initial_state_verification": True,
        "initial_state_mapping": "exact_cell_centered_custom_qinit_and_setaux",
        "initial_free_surface_definition": (
            "natural_depth_plus_bathymetry_float64"
        ),
        "nominal_eta_consistency_floor": "four_float32_eps_scaled",
        "initial_state_abs_tolerance": 5.0e-13,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"Established-solver external execution {key} changed")
    if float(execution.get("dt_max", 0.0)) != 1.0e99:
        raise ValueError("Established-solver external execution dt_max changed")
    expected_sgn = {
        "bouss_equations": 2,
        "bouss_min_level": 1,
        "bouss_max_level": 1,
        "bouss_min_depth": 0.0,
        "bouss_solver": 3,
        "bouss_tstart": 0.0,
        "petsc_ksp_type": "gmres",
        "petsc_ksp_rtol": 1.0e-9,
        "petsc_ksp_max_it": 200,
        "mpi_processes": 2,
        "omp_threads": 1,
    }
    if schema_id in HARDENED_SCHEMA_IDS:
        expected_sgn.update(
            {
                "petsc_report_converged_reason": True,
                "petsc_fail_on_nonconvergence": True,
            }
        )
    if dict(execution.get("sgn", {})) != expected_sgn:
        raise ValueError("Established-solver SGN execution policy changed")
    _requested_times(config)

    case_ids: set[str] = set()
    categories = set(config["thresholds"]) - {"refinement"}
    for case in config.get("cases", []):
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(f"Invalid or duplicate case_id: {case_id!r}")
        case_ids.add(case_id)
        if str(case.get("category")) not in categories:
            raise ValueError(f"Missing thresholds for case {case_id}")
        grids = [int(value) for value in case.get("grids", [])]
        if not grids or any(value <= 1 for value in grids):
            raise ValueError(f"Invalid grids for case {case_id}")
        for pairing in case.get("pairings", []):
            if len(pairing) != 2 or pairing[0] not in SOLVERS or pairing[1] not in COMPARATORS:
                raise ValueError(f"Invalid pairing for case {case_id}: {pairing}")
            if (
                schema_id in HARDENED_SCHEMA_IDS
                and pairing == ["boussinesq", "geoclaw_sgn"]
            ):
                if str(case.get("generator")) != "flat_linear_mode":
                    raise ValueError(
                        "Hardened Boussinesq/SGN comparisons require the matched "
                        "constant-depth long-wave generator"
                    )
                depth = float(case.get("depth", 0.0))
                amplitude = float(case.get("amplitude", 0.0))
                mode = int(case.get("mode", 0))
                max_kh = 2.0 * math.pi * mode * depth
                if (
                    depth <= 0.0
                    or amplitude <= 0.0
                    or mode <= 0
                    or max_kh > 0.35
                    or amplitude / depth > 1.0e-3
                ):
                    raise ValueError(
                        "Hardened Boussinesq/SGN case is outside the frozen "
                        "small-amplitude long-wave regime"
                    )
        if str(case.get("generator")) == "level_a_canaries":
            buffered = case.get("buffered_domain")
            if not isinstance(buffered, Mapping):
                raise ValueError("Production canaries require buffered_domain")
            core = int(buffered.get("core_grid", 0))
            inhouse = int(buffered.get("inhouse_total_grid", 0))
            external = int(buffered.get("external_total_grid", 0))
            if core != 64 or inhouse != 96 or external <= inhouse:
                raise ValueError(
                    "Production comparison must freeze 64 crop, 96 in-house, "
                    "and a larger external grid"
                )
            if any((grid - core) % 2 for grid in (inhouse, external)):
                raise ValueError("Buffered comparison grids must have symmetric integer padding")
            sponge = buffered.get("inhouse_sponge")
            if not isinstance(sponge, Mapping) or not bool(sponge.get("enabled")):
                raise ValueError("Production comparison must use the selected in-house sponge")
            if int(sponge.get("width", 0)) > (inhouse - core) // 2:
                raise ValueError("Production sponge must remain outside the crop")
            if float(buffered.get("return_time_safety_factor", 0.0)) < 1.0:
                raise ValueError("External reference requires a return-time safety factor")

    if schema_id == SCHEMA_ID:
        required_thresholds = {
            "trajectory_relative_l2",
            "per_time_relative_l2_p95",
            "gauge_nrmse_max",
            "arrival_time_abs_max",
            "peak_relative_error_max",
            "time_to_peak_abs_max",
            "waveform_lag_steps_max",
        }
    else:
        metric_policy = config.get("metric_policy")
        if not isinstance(metric_policy, Mapping):
            raise ValueError("Hardened metric policy is missing")
        if not 0.0 < float(metric_policy.get("per_time_signal_floor_fraction", 0.0)) < 1.0:
            raise ValueError("Hardened per-time signal floor must lie in (0, 1)")
        if not 0.0 < float(metric_policy.get("peak_plateau_fraction", 0.0)) < 1.0:
            raise ValueError("Hardened peak plateau fraction must lie in (0, 1)")
        if not 0.0 < float(metric_policy.get("lag_minimum_overlap_fraction", 0.0)) <= 1.0:
            raise ValueError("Hardened lag overlap fraction must lie in (0, 1]")
        if metric_policy.get("arrival_metric") != "disabled_initially_supported_fields":
            raise ValueError(
                "Hardened protocols must not use the invalid initially-active "
                "arrival gate"
            )
        required_thresholds = {
            "trajectory_relative_l2",
            "per_time_scaled_l2_p95",
            "gauge_nrmse_max",
            "peak_relative_error_max",
            "peak_plateau_time_abs_max",
            "waveform_lag_steps_max",
        }
    for category in categories:
        values = config["thresholds"][category]
        if set(values) != required_thresholds:
            raise ValueError(f"Threshold schema mismatch for {category}")
        if any(float(value) <= 0.0 for value in values.values()):
            raise ValueError(f"Thresholds must be positive for {category}")
    if schema_id == SCHEMA_ID_V4:
        policy = config.get("decision_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("v4 decision policy is missing")
        if set(policy) != {
            "threshold_float_rel_tolerance",
            "threshold_float_abs_tolerance",
            "diagnostic_boundary_band_cells",
            "category_roles",
            "flat_analytical_verification",
        }:
            raise ValueError("v4 decision policy keys changed")
        if float(policy["threshold_float_rel_tolerance"]) != 1.0e-12:
            raise ValueError("v4 relative threshold tolerance changed")
        if float(policy["threshold_float_abs_tolerance"]) != 1.0e-15:
            raise ValueError("v4 absolute threshold tolerance changed")
        if int(policy["diagnostic_boundary_band_cells"]) != 8:
            raise ValueError("v4 diagnostic boundary band changed")
        expected_roles = {
            "flat_analytical": {
                "comparison": "descriptive_only",
                "refinement": "gate",
            },
            "matched_long_wave": {
                "comparison": "gate",
                "descriptive_metrics": ["waveform_lag_steps_max"],
                "refinement": "gate",
            },
            "production_input": {
                "comparison": "descriptive_only",
                "identity_time_finite_health": "gate",
            },
        }
        if policy["category_roles"] != expected_roles:
            raise ValueError("v4 category decision roles changed")
        verification = policy["flat_analytical_verification"]
        if verification != {
            "finest_analytical_inhouse_relative_l2_max": 0.20,
            "require_pairwise_strict_decrease": True,
            "gated_error_series": [
                "trajectory_relative_l2",
                "analytical_inhouse_relative_l2",
                "analytical_external_relative_l2",
            ],
        }:
            raise ValueError("v4 flat analytical verification policy changed")
        if (
            float(
                config["thresholds"]["flat_analytical"][
                    "trajectory_relative_l2"
                ]
            )
            != float(
                verification["finest_analytical_inhouse_relative_l2_max"]
            )
        ):
            raise ValueError(
                "v4 finest analytical limit must carry forward the existing "
                "flat trajectory limit"
            )
        expected_aggregation = {
            "require_every_gated_comparison": True,
            "require_every_descriptive_comparison": False,
            "require_every_gated_refinement": True,
            "require_every_gated_verification": True,
            "production_compatibility_metrics": "descriptive_only",
            "missing_or_invalid_external_result": "fail",
            "threshold_changes_after_external_results": "forbidden",
            "external_checksums": "exact_canonical_coverage",
            "external_manifest_binding": "required",
        }
        if config.get("aggregation") != expected_aggregation:
            raise ValueError("v4 aggregation policy changed")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Established-solver YAML must contain a mapping")
    _validate_config(payload)
    return payload


def _verify_level_a(
    repo_root: Path, level_a_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_checksums(level_a_root)
    contract = _read_json(level_a_root / "preregistered_contract.json")
    decision = _read_json(level_a_root / "execution/decision.json")
    if decision.get("decision") != "pass_to_H1" or not decision.get("level_a_passed"):
        raise RuntimeError(
            "Minimum established-solver preparation requires a fresh passing Level A; "
            f"found decision={decision.get('decision')!r}"
        )
    if decision.get("contract_hash") != contract.get("contract_hash"):
        raise RuntimeError("Level A decision/contract identity mismatch")
    worker_policy = contract.get("worker_policy", {})
    execute_level_a(
        repo_root=repo_root,
        contract_root=level_a_root,
        workers=int(worker_policy["requested_workers"]),
        max_in_flight=worker_policy["requested_max_in_flight"],
        resume=True,
    )
    current_code = code_state(repo_root)
    return contract, decision, current_code


def _case_coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx, dtype=np.float64) + 0.5) / float(nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) / float(ny)
    return np.meshgrid(x, y, indexing="ij")


def _gauge_indices(
    nx: int, ny: int, fractions: Sequence[Sequence[float]]
) -> np.ndarray:
    indices = []
    for x_fraction, y_fraction in fractions:
        i = int(np.clip(round(float(x_fraction) * nx - 0.5), 0, nx - 1))
        j = int(np.clip(round(float(y_fraction) * ny - 0.5), 0, ny - 1))
        indices.append((i, j))
    return np.asarray(indices, dtype=np.int64)


def _synthetic_arrays(case: Mapping[str, Any], nx: int, ny: int) -> dict[str, np.ndarray]:
    x, y = _case_coordinates(nx, ny)
    depth = float(case["depth"])
    amplitude = float(case["amplitude"])
    generator = str(case["generator"])
    if generator == "flat_linear_packet":
        bathymetry = -depth * np.ones((nx, ny), dtype=np.float64)
        distance = np.minimum(
            np.abs(x - float(case["center_x"])),
            1.0 - np.abs(x - float(case["center_x"])),
        )
        eta0 = amplitude * np.exp(
            -0.5 * (distance / float(case["sigma"])) ** 2
        )
        eta0 -= float(np.mean(eta0))
    elif generator == "flat_linear_mode":
        bathymetry = -depth * np.ones((nx, ny), dtype=np.float64)
        eta0 = amplitude * np.cos(
            2.0
            * np.pi
            * int(case["mode"])
            * (x - float(case.get("phase_origin", 0.25)))
        )
    elif generator == "smooth_periodic_bathymetry":
        bathymetry = -depth + float(case["bathymetry_amplitude"]) * (
            np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        )
        eta0 = amplitude * (
            np.cos(2.0 * np.pi * (x - 0.25))
            + 0.35 * np.cos(4.0 * np.pi * (x - 0.25))
        )
        eta0 -= float(np.mean(eta0))
    else:
        raise ValueError(f"Unknown synthetic generator: {generator}")
    initial_depth = np.maximum(-bathymetry + eta0, 0.0)
    zeros = np.zeros_like(eta0)
    return {
        "bathymetry": bathymetry,
        "eta0": eta0,
        "initial_depth": initial_depth,
        "hu0": zeros,
        "hv0": zeros,
        "eta_t0": zeros,
    }


def _case_record(
    *,
    case_id: str,
    category: str,
    nx: int,
    ny: int,
    boundary: str,
    arrays: Mapping[str, np.ndarray],
    pairings: Sequence[Sequence[str]],
    gauges: np.ndarray,
    source: Mapping[str, Any],
    schema_id: str = SCHEMA_ID,
) -> dict[str, Any]:
    identity = {
        "case_id": case_id,
        "category": category,
        "nx": int(nx),
        "ny": int(ny),
        "domain": [0.0, 1.0, 0.0, 1.0],
        "boundary": boundary,
        "use_sponge": False,
        "pairings": [list(pairing) for pairing in pairings],
        "gauges": gauges.tolist(),
        "source": _json_safe(source),
        "array_hashes": {name: hash_array(values) for name, values in arrays.items()},
    }
    identity["case_hash"] = stable_hash_payload(
        artifact_kind="minimum-established-solver-case",
        payload=identity,
        schema_id=schema_id,
    )
    return identity


def _build_cases(
    config: Mapping[str, Any], level_a_contract: Mapping[str, Any]
) -> list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]]:
    cases: list[
        tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = []
    schema_id = str(config.get("schema_id", SCHEMA_ID))
    fractions = config["gauges"]["fractional_cell_locations"]
    for spec in config["cases"]:
        generator = str(spec["generator"])
        if generator != "level_a_canaries":
            for nx in [int(value) for value in spec["grids"]]:
                ny = int(spec["transverse_cells"])
                arrays = _synthetic_arrays(spec, nx, ny)
                record = _case_record(
                    case_id=f"{spec['case_id']}_nx{nx}_ny{ny}",
                    category=str(spec["category"]),
                    nx=nx,
                    ny=ny,
                    boundary=str(spec["boundary"]),
                    arrays=arrays,
                    pairings=spec["pairings"],
                    gauges=_gauge_indices(nx, ny, fractions),
                    source={"generator": generator, "parameters": dict(spec)},
                    schema_id=schema_id,
                )
                cases.append((record, arrays, arrays))
            continue

        canaries = list(level_a_contract["canaries"])
        count = int(spec["count"])
        if len(canaries) < count:
            raise RuntimeError("Passing Level A contract has too few frozen canaries")
        for canary in canaries[:count]:
            bathymetry, source, _strength_array, strength, loaded = (
                _load_canary_arrays(canary)
            )
            nx, ny = bathymetry.shape
            buffered = spec["buffered_domain"]
            core = int(buffered["core_grid"])
            if (nx, ny) != (core, core):
                raise RuntimeError("Production canary does not match the frozen crop")

            def buffered_arrays(
                total_grid: int,
            ) -> tuple[dict[str, np.ndarray], list[int]]:
                prepared = _prepare_buffered_domain(
                    bathymetry,
                    source,
                    strength,
                    0.0,
                    BufferedDomainConfig(
                        enabled=True,
                        buffer_cells=(total_grid - core) // 2,
                        source_taper_cells=int(buffered["source_taper_cells"]),
                        bathymetry_extension=str(buffered["bathymetry_extension"]),
                        output_crop=str(buffered["output_crop"]),
                    ),
                )
                shape = prepared["solver_bathymetry"].shape
                zeros = np.zeros(shape, dtype=np.float64)
                crop = prepared["crop"]
                return (
                    {
                        "bathymetry": np.asarray(
                            prepared["solver_bathymetry"], dtype=np.float64
                        ),
                        "eta0": np.asarray(
                            prepared["solver_eta0"], dtype=np.float64
                        ),
                        "initial_depth": np.asarray(
                            prepared["solver_h0"], dtype=np.float64
                        ),
                        "hu0": zeros,
                        "hv0": zeros,
                        "eta_t0": zeros,
                    },
                    [crop[0].start, crop[0].stop, crop[1].start, crop[1].stop],
                )

            inhouse_total = int(buffered["inhouse_total_grid"])
            external_total = int(buffered["external_total_grid"])
            inhouse_arrays, inhouse_crop = buffered_arrays(inhouse_total)
            external_arrays, external_crop = buffered_arrays(external_total)
            if (
                inhouse_crop[1] - inhouse_crop[0] != core
                or external_crop[1] - external_crop[0] != core
            ):
                raise RuntimeError("Buffered production crop shape mismatch")
            maximum_speed = math.sqrt(
                float(config["inhouse"]["gravity"])
                * float(np.max(external_arrays["initial_depth"]))
            )
            external_buffer = (external_total - core) // 2
            round_trip_bound = 2.0 * external_buffer * (1.0 / core) / maximum_speed
            required_bound = float(config["requested_times"]["horizon"]) * float(
                buffered["return_time_safety_factor"]
            )
            if round_trip_bound <= required_bound:
                raise RuntimeError(
                    f"External production grid is not return-safe for {canary['qualified_id']}: "
                    f"{round_trip_bound} <= {required_bound}"
                )
            identity_arrays = {
                **{f"inhouse_{name}": value for name, value in inhouse_arrays.items()},
                **{f"external_{name}": value for name, value in external_arrays.items()},
            }
            record = _case_record(
                case_id=f"{spec['case_id']}_{str(canary['qualified_id']).replace(':', '_')}",
                category=str(spec["category"]),
                nx=core,
                ny=core,
                boundary=str(spec["boundary"]),
                arrays=identity_arrays,
                pairings=spec["pairings"],
                gauges=_gauge_indices(core, core, fractions),
                source={
                    "generator": generator,
                    "qualified_id": canary["qualified_id"],
                    "input_fingerprint": canary["input_fingerprint"],
                },
                schema_id=schema_id,
            )
            record["inhouse_domain"] = {
                "shape": [inhouse_total, inhouse_total],
                "dx": 1.0 / core,
                "bounds": [
                    -((inhouse_total - core) // 2) / core,
                    1.0 + ((inhouse_total - core) // 2) / core,
                    -((inhouse_total - core) // 2) / core,
                    1.0 + ((inhouse_total - core) // 2) / core,
                ],
                "output_crop": inhouse_crop,
                "boundary": str(spec["boundary"]),
                "sponge": _json_safe(buffered["inhouse_sponge"]),
            }
            record["external_domain"] = {
                "shape": [external_total, external_total],
                "dx": 1.0 / core,
                "bounds": [
                    -external_buffer / core,
                    1.0 + external_buffer / core,
                    -external_buffer / core,
                    1.0 + external_buffer / core,
                ],
                "output_crop": external_crop,
                "boundary": str(buffered["external_boundary"]),
                "sponge": str(buffered["external_sponge"]),
                "round_trip_time_bound": round_trip_bound,
                "required_round_trip_time_bound": required_bound,
            }
            case_identity = dict(record)
            case_identity.pop("case_hash", None)
            record["case_hash"] = stable_hash_payload(
                artifact_kind="minimum-established-solver-case",
                payload=case_identity,
                schema_id=schema_id,
            )
            cases.append((record, inhouse_arrays, external_arrays))
    return cases


def _run_inhouse(
    config: Mapping[str, Any],
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    solver_name: str,
) -> np.ndarray:
    domain = record.get("inhouse_domain")
    if isinstance(domain, Mapping):
        sponge = domain["sponge"]
        eta, _dt, _diagnostics, _solver = _trajectory_eta(
            solver_name,
            nx=int(domain["shape"][0]),
            ny=int(domain["shape"][1]),
            cfl=float(config["inhouse"]["cfl"][solver_name]),
            boundary=str(domain["boundary"]),
            use_sponge=bool(sponge["enabled"]),
            sponge_mode=str(sponge["time_mode"]),
            filter_mode="disabled",
            bathymetry=np.asarray(arrays["bathymetry"], dtype=np.float64),
            eta0=np.asarray(arrays["eta0"], dtype=np.float64),
            h0=np.asarray(arrays["initial_depth"], dtype=np.float64),
            hu0=np.asarray(arrays["hu0"], dtype=np.float64),
            hv0=np.asarray(arrays["hv0"], dtype=np.float64),
            eta_t0=np.asarray(arrays["eta_t0"], dtype=np.float64),
            sponge_axes=str(sponge["axes"]),
            sponge_width=int(sponge["width"]),
            sponge_min_factor=float(sponge["min_factor"]),
            sponge_profile=str(sponge["profile"]),
            dx=float(domain["dx"]),
            dy=float(domain["dx"]),
        )
        i0, i1, j0, j1 = (int(value) for value in domain["output_crop"])
        return np.asarray(eta[:, i0:i1, j0:j1], dtype=np.float64)
    eta, _dt, _diagnostics, _solver = _trajectory_eta(
        solver_name,
        nx=int(record["nx"]),
        ny=int(record["ny"]),
        cfl=float(config["inhouse"]["cfl"][solver_name]),
        boundary=str(record["boundary"]),
        use_sponge=False,
        sponge_mode="elapsed_time_consistent",
        filter_mode="disabled",
        bathymetry=np.asarray(arrays["bathymetry"], dtype=np.float64),
        eta0=np.asarray(arrays["eta0"], dtype=np.float64),
        h0=np.asarray(arrays["initial_depth"], dtype=np.float64),
        eta_t0=np.asarray(arrays["eta_t0"], dtype=np.float64),
    )
    return np.asarray(eta, dtype=np.float64)


def _run_inhouse_worker(
    args: tuple[
        Mapping[str, Any], Mapping[str, Any], Mapping[str, np.ndarray], str
    ]
) -> tuple[np.ndarray, float]:
    config, record, arrays, solver_name = args
    started = time.monotonic()
    eta = _run_inhouse(config, record, arrays, solver_name)
    return eta, time.monotonic() - started


def _external_result_relative_path(case_id: str, comparator_id: str) -> str:
    return f"{case_id}/{comparator_id}.npz"


def prepare_minimum_established_solver_validation(
    *,
    repo_root: Path,
    config_path: Path,
    level_a_root: Path,
    output_root: Path | None = None,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    level_a_root = level_a_root.resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = _load_config(config_path)
    schema_id = _config_schema_id(config)
    external_result_schema_id = _external_result_schema_id(schema_id)
    level_a_contract, level_a_decision, current_code = _verify_level_a(
        repo_root, level_a_root
    )
    if bool(config["prerequisites"]["require_unchanged_code_state"]) and (
        current_code["code_state_hash"]
        != level_a_contract["code_state"]["code_state_hash"]
    ):
        raise RuntimeError(
            "Code state changed after the passing Level A; freeze a new Level A first"
        )
    level_a_code_state = level_a_contract["code_state"]
    code_state_matches_level_a = (
        current_code["code_state_hash"] == level_a_code_state["code_state_hash"]
    )

    times = _requested_times(config)
    if not np.array_equal(
        times, np.asarray(level_a_contract["candidate_times"], dtype=np.float64)
    ):
        raise RuntimeError("Level B requested times differ from passing Level A")
    cases = _build_cases(config, level_a_contract)
    case_records = [record for record, _inhouse, _external in cases]
    external_requirements: dict[tuple[str, str], dict[str, Any]] = {}
    pairings: list[dict[str, Any]] = []
    for record in case_records:
        for solver_name, comparator_id in record["pairings"]:
            key = (str(record["case_id"]), str(comparator_id))
            required_npz_keys = [
                "schema_id",
                "case_hash",
                "comparator_id",
                "comparator_version",
                "comparator_commit",
                "times",
                "eta",
            ]
            if schema_id in HARDENED_SCHEMA_IDS:
                required_npz_keys.extend(
                    [
                        "clawpack_commit",
                        "petsc_commit",
                        "adapter_hash",
                        "actual_times",
                        "runtime_seconds",
                        "initial_state_max_abs_error",
                        "requested_time_max_abs_error",
                        "nominal_eta_max_abs_difference",
                        "nominal_eta_consistency_floor",
                        "solver_health_status",
                    ]
                )
                if comparator_id == "geoclaw_sgn":
                    required_npz_keys.extend(
                        [
                            "ksp_solve_count",
                            "ksp_iteration_max",
                            "ksp_iteration_mean",
                            "ksp_convergence_reasons",
                        ]
                    )
            external_requirements[key] = {
                "case_id": record["case_id"],
                "case_hash": record["case_hash"],
                "comparator_id": comparator_id,
                "comparator_version": config["external_comparator"]["version"],
                "result_schema_id": external_result_schema_id,
                "relative_path": _external_result_relative_path(*key),
                "required_npz_keys": required_npz_keys,
                "eta_shape": [int(times.size), int(record["nx"]), int(record["ny"])],
                "computational_shape": list(
                    record.get("external_domain", {}).get(
                        "shape", [int(record["nx"]), int(record["ny"])]
                    )
                ),
                "output_crop": list(
                    record.get("external_domain", {}).get(
                        "output_crop", [0, int(record["nx"]), 0, int(record["ny"])]
                    )
                ),
                "computational_domain_bounds": list(
                    record.get("external_domain", {}).get(
                        "bounds", record["domain"]
                    )
                ),
            }
            pairings.append(
                {
                    "pairing_id": f"{solver_name}__{comparator_id}",
                    "case_id": record["case_id"],
                    "case_hash": record["case_hash"],
                    "category": record["category"],
                    "inhouse_solver": solver_name,
                    "external_comparator": comparator_id,
                }
            )

    frozen = {
        "schema_id": schema_id,
        "artifact_kind": "minimum-established-solver-validation-frozen-contract",
        "source_config": _json_safe(config),
        "source_config_sha256": sha256_file(config_path),
        "code_state": current_code,
        "level_a": {
            "root": str(level_a_root),
            "contract_hash": level_a_contract["contract_hash"],
            "scientific_digest": level_a_decision["scientific_digest"],
            "decision": level_a_decision["decision"],
            "completed_replay_validated": True,
            "frozen_code_state_hash": level_a_code_state["code_state_hash"],
            "bundle_code_state_matches_level_a": code_state_matches_level_a,
            "code_state_policy": _json_safe(config["prerequisites"]),
        },
        "requested_times": times.tolist(),
        "cases": case_records,
        "pairings": pairings,
        "external_results": list(external_requirements.values()),
        "thresholds_frozen_before_external_results": True,
    }
    bundle_hash = stable_hash_payload(
        artifact_kind="minimum-established-solver-validation-contract",
        payload=frozen,
        schema_id=schema_id,
    )
    frozen["bundle_hash"] = bundle_hash
    base = output_root or repo_root / "artifacts/common_time_v2/level_b_minimum"
    final = base / bundle_hash
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite Level B bundle: {final}")
    staging = base / f".{bundle_hash}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(staging / "frozen_contract.json", frozen)
        _write_json(
            staging / "external_results_manifest.json",
            {
                "schema_id": external_result_schema_id,
                "bundle_hash": bundle_hash,
                "results": list(external_requirements.values()),
            },
        )
        run_specs: list[
            tuple[
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, np.ndarray],
                str,
                Path,
            ]
        ] = []
        for record, inhouse_arrays, external_arrays in cases:
            case_dir = staging / "cases" / str(record["case_id"])
            case_dir.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(
                case_dir / "input.npz",
                **external_arrays,
                requested_times=times,
                gauge_indices=np.asarray(record["gauges"], dtype=np.int64),
                case_hash=np.asarray(record["case_hash"]),
                output_crop=np.asarray(
                    record.get("external_domain", {}).get(
                        "output_crop", [0, int(record["nx"]), 0, int(record["ny"])]
                    ),
                    dtype=np.int64,
                ),
                domain_bounds=np.asarray(
                    record.get("external_domain", {}).get(
                        "bounds", record["domain"]
                    ),
                    dtype=np.float64,
                ),
            )
            if "inhouse_domain" in record:
                np.savez_compressed(
                    case_dir / "inhouse_input.npz",
                    **inhouse_arrays,
                    requested_times=times,
                    output_crop=np.asarray(
                        record["inhouse_domain"]["output_crop"], dtype=np.int64
                    ),
                    domain_bounds=np.asarray(
                        record["inhouse_domain"]["bounds"], dtype=np.float64
                    ),
                    case_hash=np.asarray(record["case_hash"]),
                )
            inhouse_solvers = sorted(
                {str(pairing[0]) for pairing in record["pairings"]}
            )
            for solver_name in inhouse_solvers:
                run_specs.append(
                    (config, record, inhouse_arrays, solver_name, case_dir)
                )

        run_total = len(run_specs)
        run_completed = 0

        def write_inhouse(
            record: Mapping[str, Any],
            solver_name: str,
            case_dir: Path,
            eta: np.ndarray,
            runtime_s: float,
        ) -> None:
            nonlocal run_completed
            np.savez_compressed(
                case_dir / f"inhouse_{solver_name}.npz",
                eta=eta,
                times=times,
                case_hash=np.asarray(record["case_hash"]),
                solver_id=np.asarray(solver_name),
            )
            run_completed += 1
            if progress is not None:
                progress(
                    f"[level-b-prepare] done {run_completed}/{run_total} "
                    f"{record['case_id']} {solver_name} runtime={runtime_s:.1f}s"
                )

        if workers == 1:
            for task_index, (task_config, record, arrays, solver_name, case_dir) in enumerate(
                run_specs, start=1
            ):
                if progress is not None:
                    progress(
                        f"[level-b-prepare] start {task_index}/{run_total} "
                        f"{record['case_id']} {solver_name}"
                    )
                eta, runtime_s = _run_inhouse_worker(
                    (task_config, record, arrays, solver_name)
                )
                write_inhouse(record, solver_name, case_dir, eta, runtime_s)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(workers, max(1, run_total)), mp_context=context
            ) as executor:
                futures = {}
                for task_index, (task_config, record, arrays, solver_name, case_dir) in enumerate(
                    run_specs, start=1
                ):
                    if progress is not None:
                        progress(
                            f"[level-b-prepare] queued {task_index}/{run_total} "
                            f"{record['case_id']} {solver_name}"
                        )
                    future = executor.submit(
                        _run_inhouse_worker,
                        (task_config, record, arrays, solver_name),
                    )
                    futures[future] = (record, solver_name, case_dir)
                for future in as_completed(futures):
                    record, solver_name, case_dir = futures[future]
                    eta, runtime_s = future.result()
                    write_inhouse(
                        record, solver_name, case_dir, eta, runtime_s
                    )
        (staging / "README.md").write_text(
            "# Frozen minimum established-solver validation bundle\n\n"
            f"- Bundle hash: `{bundle_hash}`\n"
            f"- Passing Level A: `{level_a_contract['contract_hash']}`\n"
            f"- Cases: {len(case_records)}\n"
            f"- Pairwise comparisons: {len(pairings)}\n\n"
            "Run GeoClaw in a separate environment, populate the canonical external "
            "result files listed in `external_results_manifest.json`, and evaluate "
            "them with `scripts/evaluate_established_solver_validation.py`.\n",
            encoding="utf-8",
        )
        _write_checksums(staging)
        base.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _npz_scalar(payload: Mapping[str, np.ndarray], key: str) -> str:
    return str(np.asarray(payload[key]).reshape(-1)[0])


def _load_external_run_manifest(
    external_root: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(external_root / "RUN_MANIFEST.json")
    config = frozen["source_config"]
    if manifest.get("schema_id") != (
        "tsunami-surrogate.geoclaw-external-adapter.v2"
    ):
        raise RuntimeError("External run manifest adapter schema mismatch")
    if manifest.get("bundle_hash") != frozen.get("bundle_hash"):
        raise RuntimeError("External run manifest bundle identity mismatch")
    if manifest.get("execution") != config.get("external_execution"):
        raise RuntimeError("External run manifest execution policy mismatch")
    if not str(manifest.get("adapter_hash", "")).strip():
        raise RuntimeError("External run manifest has no adapter hash")
    revisions = manifest.get("revisions")
    if not isinstance(revisions, Mapping):
        raise RuntimeError("External run manifest revisions are missing")
    expected_revisions = config["external_comparator"].get("expected_revisions", {})
    for key, expected in expected_revisions.items():
        if revisions.get(key) != expected:
            raise RuntimeError(
                f"External run manifest {key} mismatch: "
                f"{revisions.get(key)!r} != {expected!r}"
            )
    return manifest


def _validate_external_checksums(
    external_root: Path,
    frozen: Mapping[str, Any],
) -> None:
    checksum_path = external_root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise RuntimeError("External canonical checksum manifest is missing")
    expected_paths = {
        "RUN_MANIFEST.json",
        *(str(row["relative_path"]) for row in frozen["external_results"]),
    }
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or relative in recorded
            or relative not in expected_paths
        ):
            raise RuntimeError("External canonical checksum manifest is invalid")
        recorded[relative] = digest
    if set(recorded) != expected_paths:
        raise RuntimeError("External canonical checksum coverage mismatch")
    for relative, expected_digest in recorded.items():
        path = external_root / relative
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise RuntimeError(f"External checksum mismatch: {relative}")


def _load_external_result(
    path: Path,
    requirement: Mapping[str, Any],
    requested_times: np.ndarray,
    run_manifest: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing external result: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = set(requirement["required_npz_keys"]) - set(payload.files)
        if missing:
            raise RuntimeError(f"External result {path} missing keys {sorted(missing)}")
        metadata = {
            key: _npz_scalar(payload, key)
            for key in (
                "schema_id",
                "case_hash",
                "comparator_id",
                "comparator_version",
                "comparator_commit",
            )
        }
        eta = np.asarray(payload["eta"], dtype=np.float64)
        times = np.asarray(payload["times"], dtype=np.float64)
        result_schema_id = str(
            requirement.get("result_schema_id", EXTERNAL_RESULT_SCHEMA_ID)
        )
        if result_schema_id == EXTERNAL_RESULT_SCHEMA_ID_V3:
            for key in (
                "clawpack_commit",
                "petsc_commit",
                "adapter_hash",
                "solver_health_status",
            ):
                metadata[key] = _npz_scalar(payload, key)
            actual_times = np.asarray(payload["actual_times"], dtype=np.float64)
            diagnostics = {
                key: float(np.asarray(payload[key]).reshape(-1)[0])
                for key in (
                    "runtime_seconds",
                    "initial_state_max_abs_error",
                    "requested_time_max_abs_error",
                    "nominal_eta_max_abs_difference",
                    "nominal_eta_consistency_floor",
                )
            }
            if str(requirement["comparator_id"]) == "geoclaw_sgn":
                diagnostics.update(
                    {
                        "ksp_solve_count": int(
                            np.asarray(payload["ksp_solve_count"]).reshape(-1)[0]
                        ),
                        "ksp_iteration_max": int(
                            np.asarray(payload["ksp_iteration_max"]).reshape(-1)[0]
                        ),
                        "ksp_iteration_mean": float(
                            np.asarray(payload["ksp_iteration_mean"]).reshape(-1)[0]
                        ),
                    }
                )
                metadata["ksp_convergence_reasons"] = [
                    str(value)
                    for value in np.asarray(
                        payload["ksp_convergence_reasons"]
                    ).reshape(-1)
                ]
        else:
            actual_times = times
            diagnostics = {}
    expected_metadata = {
        "schema_id": str(
            requirement.get("result_schema_id", EXTERNAL_RESULT_SCHEMA_ID)
        ),
        "case_hash": str(requirement["case_hash"]),
        "comparator_id": str(requirement["comparator_id"]),
        "comparator_version": str(requirement["comparator_version"]),
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise RuntimeError(
                f"External result {path} {key} mismatch: {metadata[key]!r} != {expected!r}"
            )
    if not metadata["comparator_commit"].strip():
        raise RuntimeError(f"External result {path} has no comparator commit")
    if list(eta.shape) != [int(v) for v in requirement["eta_shape"]]:
        raise RuntimeError(f"External result {path} eta shape mismatch: {eta.shape}")
    if not np.array_equal(times, requested_times):
        raise RuntimeError(f"External result {path} requested-time mismatch")
    if actual_times.shape != requested_times.shape:
        raise RuntimeError(f"External result {path} actual-time shape mismatch")
    actual_time_error = float(
        np.max(np.abs(actual_times - requested_times))
    )
    if actual_time_error > EXTERNAL_ACTUAL_TIME_ABS_TOLERANCE:
        raise RuntimeError(
            f"External result {path} actual-time mismatch: "
            f"{actual_time_error:.3e} > "
            f"{EXTERNAL_ACTUAL_TIME_ABS_TOLERANCE:.3e}"
        )
    if not np.isfinite(eta).all():
        raise RuntimeError(f"External result {path} contains nonfinite eta")
    if expected_metadata["schema_id"] == EXTERNAL_RESULT_SCHEMA_ID_V3:
        if run_manifest is None:
            raise RuntimeError("v3 external result requires its frozen run manifest")
        manifest_revisions = run_manifest["revisions"]
        for result_key, manifest_key in (
            ("comparator_commit", "geoclaw_commit"),
            ("clawpack_commit", "clawpack_commit"),
            ("petsc_commit", "petsc_commit"),
        ):
            if metadata[result_key] != manifest_revisions[manifest_key]:
                raise RuntimeError(
                    f"External result {path} {result_key} does not match run manifest"
                )
        if metadata["adapter_hash"] != run_manifest["adapter_hash"]:
            raise RuntimeError(
                f"External result {path} adapter hash does not match run manifest"
            )
        if metadata["solver_health_status"] != "passed":
            raise RuntimeError(f"External result {path} solver health did not pass")
        if (
            not all(math.isfinite(float(value)) for value in diagnostics.values())
            or diagnostics["runtime_seconds"] < 0.0
            or diagnostics["initial_state_max_abs_error"] > 5.0e-13
            or diagnostics["requested_time_max_abs_error"] > 5.0e-14
            or diagnostics["nominal_eta_max_abs_difference"]
            > diagnostics["nominal_eta_consistency_floor"]
        ):
            raise RuntimeError(f"External result {path} diagnostics are invalid")
        if str(requirement["comparator_id"]) == "geoclaw_sgn":
            reasons = metadata["ksp_convergence_reasons"]
            if (
                diagnostics["ksp_solve_count"] <= 0
                or diagnostics["ksp_iteration_max"] < 0
                or diagnostics["ksp_iteration_mean"] < 0.0
                or not reasons
                or any(not reason.startswith("CONVERGED_") for reason in reasons)
            ):
                raise RuntimeError(f"External result {path} KSP health is invalid")
    metadata.update(diagnostics)
    return eta, metadata


def _first_arrival(times: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(np.abs(values) >= threshold)
    return None if indices.size == 0 else float(times[int(indices[0])])


def _waveform_lag_steps(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    if np.linalg.norm(aa) <= 1e-30 or np.linalg.norm(bb) <= 1e-30:
        return 0
    correlation = np.correlate(aa, bb, mode="full")
    return int(np.argmax(correlation) - (aa.size - 1))


def _comparison_metrics(
    inhouse: np.ndarray,
    external: np.ndarray,
    times: np.ndarray,
    gauges: np.ndarray,
    *,
    arrival_fraction: float,
    inactive_floor: float,
) -> dict[str, Any]:
    difference = np.asarray(inhouse, dtype=np.float64) - np.asarray(
        external, dtype=np.float64
    )
    external_norm = max(float(np.linalg.norm(external)), inactive_floor)
    trajectory_l2 = float(np.linalg.norm(difference) / external_norm)
    per_time = np.asarray(
        [
            np.linalg.norm(difference[index])
            / max(float(np.linalg.norm(external[index])), inactive_floor)
            for index in range(times.size)
        ],
        dtype=np.float64,
    )
    gauge_nrmse: list[float] = []
    arrival_errors: list[float] = []
    peak_errors: list[float] = []
    peak_time_errors: list[float] = []
    lags: list[int] = []
    active_gauges = 0
    for i, j in np.asarray(gauges, dtype=np.int64):
        candidate = np.asarray(inhouse[:, i, j], dtype=np.float64)
        reference = np.asarray(external[:, i, j], dtype=np.float64)
        peak = float(np.max(np.abs(reference)))
        if peak <= inactive_floor:
            continue
        active_gauges += 1
        gauge_nrmse.append(
            float(np.sqrt(np.mean((candidate - reference) ** 2)) / peak)
        )
        threshold = arrival_fraction * peak
        candidate_arrival = _first_arrival(times, candidate, threshold)
        reference_arrival = _first_arrival(times, reference, threshold)
        if candidate_arrival is None or reference_arrival is None:
            arrival_errors.append(float(times[-1] - times[0] + times[0]))
        else:
            arrival_errors.append(abs(candidate_arrival - reference_arrival))
        candidate_peak_index = int(np.argmax(np.abs(candidate)))
        reference_peak_index = int(np.argmax(np.abs(reference)))
        peak_errors.append(abs(abs(candidate[candidate_peak_index]) - peak) / peak)
        peak_time_errors.append(
            abs(
                float(times[candidate_peak_index])
                - float(times[reference_peak_index])
            )
        )
        lags.append(abs(_waveform_lag_steps(candidate, reference)))
    if active_gauges == 0:
        return {
            "active_gauge_count": 0,
            "trajectory_relative_l2": trajectory_l2,
            "per_time_relative_l2_p95": float(np.quantile(per_time, 0.95)),
            "gauge_nrmse_max": None,
            "arrival_time_abs_max": None,
            "peak_relative_error_max": None,
            "time_to_peak_abs_max": None,
            "waveform_lag_steps_max": None,
        }
    return {
        "active_gauge_count": active_gauges,
        "trajectory_relative_l2": trajectory_l2,
        "per_time_relative_l2_p95": float(np.quantile(per_time, 0.95)),
        "gauge_nrmse_max": max(gauge_nrmse),
        "arrival_time_abs_max": max(arrival_errors),
        "peak_relative_error_max": max(peak_errors),
        "time_to_peak_abs_max": max(peak_time_errors),
        "waveform_lag_steps_max": max(lags),
    }


def _normalized_waveform_lag_diagnostics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_overlap_fraction: float,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    minimum_overlap = max(
        2, int(math.ceil(candidate.size * minimum_overlap_fraction))
    )
    candidates: list[tuple[float, int]] = []
    for lag in range(-(candidate.size - minimum_overlap), candidate.size - minimum_overlap + 1):
        if lag < 0:
            left = candidate[:lag]
            right = reference[-lag:]
        elif lag > 0:
            left = candidate[lag:]
            right = reference[:-lag]
        else:
            left = candidate
            right = reference
        left = left - float(np.mean(left))
        right = right - float(np.mean(right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        score = 0.0 if denominator <= 1.0e-30 else float(np.dot(left, right) / denominator)
        candidates.append((score, lag))
    best_score = max(score for score, _lag in candidates)
    tied = [
        lag
        for score, lag in candidates
        if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1.0e-14)
    ]
    best_lag = min(tied, key=lambda value: (abs(value), value))
    zero_lag_score = next(score for score, lag in candidates if lag == 0)
    return {
        "lag_steps": best_lag,
        "best_correlation": best_score,
        "zero_lag_correlation": zero_lag_score,
        "score_advantage_over_zero": best_score - zero_lag_score,
    }


def _normalized_waveform_lag_steps(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_overlap_fraction: float,
) -> int:
    diagnostics = _normalized_waveform_lag_diagnostics(
        candidate,
        reference,
        minimum_overlap_fraction=minimum_overlap_fraction,
    )
    return int(diagnostics["lag_steps"])


def _peak_plateau_distance(
    times: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    plateau_fraction: float,
    inactive_floor: float,
) -> float:
    candidate_peak = float(np.max(np.abs(candidate)))
    reference_peak = float(np.max(np.abs(reference)))
    if candidate_peak <= inactive_floor or reference_peak <= inactive_floor:
        return float(times[-1] - times[0] + times[0])
    candidate_indices = np.flatnonzero(
        np.abs(candidate) >= plateau_fraction * candidate_peak
    )
    reference_indices = np.flatnonzero(
        np.abs(reference) >= plateau_fraction * reference_peak
    )
    candidate_times = times[candidate_indices]
    reference_times = times[reference_indices]
    return float(
        np.min(
            np.abs(
                candidate_times[:, np.newaxis]
                - reference_times[np.newaxis, :]
            )
        )
    )


def _comparison_metrics_v3(
    inhouse: np.ndarray,
    external: np.ndarray,
    times: np.ndarray,
    gauges: np.ndarray,
    *,
    inactive_floor: float,
    per_time_signal_floor_fraction: float,
    peak_plateau_fraction: float,
    lag_minimum_overlap_fraction: float,
) -> dict[str, Any]:
    inhouse = np.asarray(inhouse, dtype=np.float64)
    external = np.asarray(external, dtype=np.float64)
    difference = inhouse - external
    external_norm = max(float(np.linalg.norm(external)), inactive_floor)
    reference_time_rms = np.sqrt(np.mean(external**2, axis=(1, 2)))
    difference_time_rms = np.sqrt(np.mean(difference**2, axis=(1, 2)))
    reference_signal_scale = max(
        float(np.max(reference_time_rms)), inactive_floor
    )
    denominator_floor = per_time_signal_floor_fraction * reference_signal_scale
    per_time_scaled = difference_time_rms / np.maximum(
        reference_time_rms, denominator_floor
    )

    gauge_nrmse: list[float] = []
    peak_errors: list[float] = []
    peak_plateau_errors: list[float] = []
    lags: list[int] = []
    for i, j in np.asarray(gauges, dtype=np.int64):
        candidate = np.asarray(inhouse[:, i, j], dtype=np.float64)
        reference = np.asarray(external[:, i, j], dtype=np.float64)
        reference_peak = float(np.max(np.abs(reference)))
        if reference_peak <= inactive_floor:
            continue
        candidate_peak = float(np.max(np.abs(candidate)))
        gauge_nrmse.append(
            float(np.sqrt(np.mean((candidate - reference) ** 2)) / reference_peak)
        )
        peak_errors.append(abs(candidate_peak - reference_peak) / reference_peak)
        peak_plateau_errors.append(
            _peak_plateau_distance(
                times,
                candidate,
                reference,
                plateau_fraction=peak_plateau_fraction,
                inactive_floor=inactive_floor,
            )
        )
        lags.append(
            abs(
                _normalized_waveform_lag_steps(
                    candidate,
                    reference,
                    minimum_overlap_fraction=lag_minimum_overlap_fraction,
                )
            )
        )

    return {
        "active_gauge_count": len(gauge_nrmse),
        "absolute_rms": float(np.sqrt(np.mean(difference**2))),
        "absolute_linf": float(np.max(np.abs(difference))),
        "trajectory_relative_l2": float(np.linalg.norm(difference) / external_norm),
        "per_time_reference_rms_max": reference_signal_scale,
        "per_time_denominator_floor": denominator_floor,
        "per_time_scaled_l2_p95": float(np.quantile(per_time_scaled, 0.95)),
        "gauge_nrmse_max": max(gauge_nrmse) if gauge_nrmse else None,
        "arrival_metric_eligible": False,
        "arrival_time_abs_max": None,
        "peak_relative_error_max": max(peak_errors) if peak_errors else None,
        "peak_plateau_time_abs_max": (
            max(peak_plateau_errors) if peak_plateau_errors else None
        ),
        "waveform_lag_steps_max": max(lags) if lags else None,
    }


def _relative_l2(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    inactive_floor: float,
) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), inactive_floor)
    )


def _comparison_metrics_v4(
    inhouse: np.ndarray,
    external: np.ndarray,
    times: np.ndarray,
    gauges: np.ndarray,
    *,
    inactive_floor: float,
    per_time_signal_floor_fraction: float,
    peak_plateau_fraction: float,
    lag_minimum_overlap_fraction: float,
    diagnostic_boundary_band_cells: int,
) -> dict[str, Any]:
    metrics = _comparison_metrics_v3(
        inhouse,
        external,
        times,
        gauges,
        inactive_floor=inactive_floor,
        per_time_signal_floor_fraction=per_time_signal_floor_fraction,
        peak_plateau_fraction=peak_plateau_fraction,
        lag_minimum_overlap_fraction=lag_minimum_overlap_fraction,
    )
    inhouse = np.asarray(inhouse, dtype=np.float64)
    external = np.asarray(external, dtype=np.float64)
    difference = inhouse - external
    reference_time_rms = np.sqrt(np.mean(external**2, axis=(1, 2)))
    difference_time_rms = np.sqrt(np.mean(difference**2, axis=(1, 2)))
    reference_scale = max(float(np.max(reference_time_rms)), inactive_floor)
    denominator_floor = per_time_signal_floor_fraction * reference_scale
    active = reference_time_rms >= denominator_floor
    per_time_scaled = difference_time_rms / np.maximum(
        reference_time_rms, denominator_floor
    )

    candidate_flat = inhouse.reshape(-1)
    reference_flat = external.reshape(-1)
    reference_energy = float(np.dot(reference_flat, reference_flat))
    candidate_energy = float(np.dot(candidate_flat, candidate_flat))
    cross = float(np.dot(candidate_flat, reference_flat))
    amplitude_scale = (
        cross / reference_energy if reference_energy > inactive_floor**2 else 0.0
    )
    correlation_denominator = math.sqrt(candidate_energy * reference_energy)
    correlation = (
        cross / correlation_denominator
        if correlation_denominator > inactive_floor**2
        else 0.0
    )
    shape_residual = _relative_l2(
        inhouse - amplitude_scale * external,
        np.zeros_like(external),
        inactive_floor=max(math.sqrt(reference_energy), inactive_floor),
    )

    nx, ny = external.shape[1:]
    width = min(
        int(diagnostic_boundary_band_cells),
        max(0, (min(nx, ny) - 1) // 2),
    )
    ii, jj = np.indices((nx, ny))
    edge_distance = np.minimum.reduce(
        (ii, jj, nx - 1 - ii, ny - 1 - jj)
    )
    boundary_mask = edge_distance < width if width > 0 else np.zeros((nx, ny), bool)
    interior_mask = ~boundary_mask

    lag_diagnostics = []
    for i, j in np.asarray(gauges, dtype=np.int64):
        reference = np.asarray(external[:, i, j], dtype=np.float64)
        if float(np.max(np.abs(reference))) <= inactive_floor:
            continue
        lag_diagnostics.append(
            _normalized_waveform_lag_diagnostics(
                np.asarray(inhouse[:, i, j], dtype=np.float64),
                reference,
                minimum_overlap_fraction=lag_minimum_overlap_fraction,
            )
        )

    metrics.update(
        {
            "per_time_active_count": int(np.count_nonzero(active)),
            "per_time_inactive_count": int(np.count_nonzero(~active)),
            "per_time_scaled_l2_p95_active": (
                float(np.quantile(per_time_scaled[active], 0.95))
                if np.any(active)
                else None
            ),
            "field_norm_ratio": (
                math.sqrt(candidate_energy / reference_energy)
                if reference_energy > inactive_floor**2
                else None
            ),
            "optimal_amplitude_scale": amplitude_scale,
            "field_cosine_similarity": correlation,
            "shape_relative_l2_after_scale": shape_residual,
            "diagnostic_boundary_band_cells": width,
            "boundary_band_relative_l2": (
                _relative_l2(
                    inhouse[:, boundary_mask],
                    external[:, boundary_mask],
                    inactive_floor=inactive_floor,
                )
                if np.any(boundary_mask)
                else None
            ),
            "interior_relative_l2": (
                _relative_l2(
                    inhouse[:, interior_mask],
                    external[:, interior_mask],
                    inactive_floor=inactive_floor,
                )
                if np.any(interior_mask)
                else None
            ),
            "waveform_zero_lag_correlation_min": (
                min(float(row["zero_lag_correlation"]) for row in lag_diagnostics)
                if lag_diagnostics
                else None
            ),
            "waveform_best_lag_correlation_min": (
                min(float(row["best_correlation"]) for row in lag_diagnostics)
                if lag_diagnostics
                else None
            ),
            "waveform_lag_score_advantage_max": (
                max(
                    float(row["score_advantage_over_zero"])
                    for row in lag_diagnostics
                )
                if lag_diagnostics
                else None
            ),
        }
    )
    return metrics


def _flat_linear_swe_reference(
    eta0: np.ndarray,
    times: np.ndarray,
    *,
    depth: float,
    gravity: float,
) -> np.ndarray:
    eta0 = np.asarray(eta0, dtype=np.float64)
    nx = eta0.shape[0]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    frequencies = math.sqrt(gravity * depth) * np.abs(wave_numbers)
    coefficients = np.fft.fft(eta0, axis=0)
    return np.stack(
        [
            np.fft.ifft(
                coefficients * np.cos(frequencies * float(time_value))[:, None],
                axis=0,
            ).real
            for time_value in times
        ],
        axis=0,
    )


def _metrics_pass(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    if int(metrics["active_gauge_count"]) <= 0:
        return False
    return all(
        metrics[key] is not None and float(metrics[key]) <= float(limit)
        for key, limit in thresholds.items()
    )


def _v4_at_or_below(
    value: Any,
    limit: Any,
    *,
    integer: bool,
    rel_tolerance: float,
    abs_tolerance: float,
) -> bool:
    if value is None:
        return False
    if integer:
        return int(value) <= int(limit)
    metric = float(value)
    threshold = float(limit)
    return metric <= threshold or math.isclose(
        metric,
        threshold,
        rel_tol=rel_tolerance,
        abs_tol=abs_tolerance,
    )


def _v4_threshold_results(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    descriptive_metrics: Sequence[str],
    rel_tolerance: float,
    abs_tolerance: float,
) -> tuple[dict[str, dict[str, Any]], bool]:
    descriptive = set(str(value) for value in descriptive_metrics)
    results: dict[str, dict[str, Any]] = {}
    gated_passes: list[bool] = []
    for key, limit in thresholds.items():
        role = "descriptive_only" if key in descriptive else "gate"
        passed = _v4_at_or_below(
            metrics.get(key),
            limit,
            integer=key == "waveform_lag_steps_max",
            rel_tolerance=rel_tolerance,
            abs_tolerance=abs_tolerance,
        )
        results[str(key)] = {
            "decision_role": role,
            "limit": limit,
            "passed": passed,
        }
        if role == "gate":
            gated_passes.append(passed)
    active_gauges = int(metrics.get("active_gauge_count", 0))
    return results, active_gauges > 0 and all(gated_passes)


def _v4_pairwise_refinement(
    errors: Sequence[float],
    grids: Sequence[int],
    *,
    ratio_limit: float,
    require_strict_decrease: bool,
    rel_tolerance: float,
    abs_tolerance: float,
) -> dict[str, Any]:
    values = [float(value) for value in errors]
    grid_values = [int(value) for value in grids]
    pairwise_ratios = [
        values[index + 1] / max(values[index], 1.0e-30)
        for index in range(len(values) - 1)
    ]
    pairwise_orders = [
        math.log(
            max(values[index], 1.0e-30)
            / max(values[index + 1], 1.0e-30)
        )
        / math.log(float(grid_values[index + 1]) / grid_values[index])
        for index in range(len(values) - 1)
    ]
    pairwise_decreasing = all(
        values[index + 1] < values[index]
        and not math.isclose(
            values[index + 1],
            values[index],
            rel_tol=rel_tolerance,
            abs_tol=abs_tolerance,
        )
        for index in range(len(values) - 1)
    )
    overall_ratio = values[-1] / max(values[0], 1.0e-30)
    ratio_passed = _v4_at_or_below(
        overall_ratio,
        ratio_limit,
        integer=False,
        rel_tolerance=rel_tolerance,
        abs_tolerance=abs_tolerance,
    )
    return {
        "grids": grid_values,
        "errors": values,
        "pairwise_error_ratios": pairwise_ratios,
        "pairwise_orders": pairwise_orders,
        "pairwise_strictly_decreasing": pairwise_decreasing,
        "finest_to_coarsest_error_ratio": overall_ratio,
        "ratio_limit": ratio_limit,
        "passed": ratio_passed
        and (pairwise_decreasing or not require_strict_decrease),
    }


def evaluate_minimum_established_solver_validation(
    *,
    bundle_root: Path,
    external_root: Path,
    output_root: Path,
    progress: Callable[[str], None] | None = None,
) -> Path:
    bundle_root = bundle_root.resolve()
    external_root = external_root.resolve()
    output_root = output_root.resolve()
    validate_checksums(bundle_root)
    frozen = _read_json(bundle_root / "frozen_contract.json")
    schema_id = str(frozen.get("schema_id", ""))
    if schema_id not in SUPPORTED_SCHEMA_IDS:
        raise RuntimeError("Frozen Level B schema mismatch")
    bundle_identity = dict(frozen)
    recorded_hash = bundle_identity.pop("bundle_hash", None)
    expected_hash = stable_hash_payload(
        artifact_kind="minimum-established-solver-validation-contract",
        payload=bundle_identity,
        schema_id=schema_id,
    )
    if recorded_hash != expected_hash or bundle_root.name != expected_hash:
        raise RuntimeError("Frozen Level B content-addressed identity mismatch")
    config = frozen["source_config"]
    run_manifest: Mapping[str, Any] | None = None
    if schema_id in HARDENED_SCHEMA_IDS:
        run_manifest = _load_external_run_manifest(external_root, frozen)
        _validate_external_checksums(external_root, frozen)
    v4_policy = config.get("decision_policy", {}) if schema_id == SCHEMA_ID_V4 else {}
    v4_rel_tolerance = float(
        v4_policy.get("threshold_float_rel_tolerance", 0.0)
    )
    v4_abs_tolerance = float(
        v4_policy.get("threshold_float_abs_tolerance", 0.0)
    )
    times = np.asarray(frozen["requested_times"], dtype=np.float64)
    case_by_id = {str(row["case_id"]): row for row in frozen["cases"]}
    requirement_by_key = {
        (str(row["case_id"]), str(row["comparator_id"])): row
        for row in frozen["external_results"]
    }
    external_cache: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    total_pairings = len(frozen["pairings"])
    for pairing_index, pairing in enumerate(frozen["pairings"], start=1):
        case_id = str(pairing["case_id"])
        comparator_id = str(pairing["external_comparator"])
        key = (case_id, comparator_id)
        requirement = requirement_by_key[key]
        if key not in external_cache:
            external_cache[key] = _load_external_result(
                external_root / str(requirement["relative_path"]),
                requirement,
                times,
                run_manifest,
            )
        external_eta, metadata = external_cache[key]
        solver_name = str(pairing["inhouse_solver"])
        with np.load(
            bundle_root / "cases" / case_id / f"inhouse_{solver_name}.npz",
            allow_pickle=False,
        ) as payload:
            inhouse_eta = np.asarray(payload["eta"], dtype=np.float64)
            inhouse_times = np.asarray(payload["times"], dtype=np.float64)
            inhouse_case_hash = _npz_scalar(payload, "case_hash")
        case = case_by_id[case_id]
        if inhouse_case_hash != case["case_hash"] or not np.array_equal(
            inhouse_times, times
        ):
            raise RuntimeError(f"Frozen in-house result identity mismatch: {case_id}")
        if schema_id == SCHEMA_ID_V4 and (
            inhouse_eta.shape != external_eta.shape
            or not np.isfinite(inhouse_eta).all()
        ):
            raise RuntimeError(
                f"Frozen in-house result shape/health mismatch: {case_id}"
            )
        with np.load(
            bundle_root / "cases" / case_id / "input.npz", allow_pickle=False
        ) as payload:
            gauges = np.asarray(payload["gauge_indices"], dtype=np.int64)
            eta0 = (
                np.asarray(payload["eta0"], dtype=np.float64)
                if schema_id in HARDENED_SCHEMA_IDS
                else None
            )
        if schema_id == SCHEMA_ID:
            metrics = _comparison_metrics(
                inhouse_eta,
                external_eta,
                times,
                gauges,
                arrival_fraction=float(
                    config["gauges"]["arrival_fraction_of_external_peak"]
                ),
                inactive_floor=float(
                    config["gauges"]["inactive_external_peak_floor"]
                ),
            )
        elif schema_id == SCHEMA_ID_V3:
            metric_policy = config["metric_policy"]
            metrics = _comparison_metrics_v3(
                inhouse_eta,
                external_eta,
                times,
                gauges,
                inactive_floor=float(
                    config["gauges"]["inactive_external_peak_floor"]
                ),
                per_time_signal_floor_fraction=float(
                    metric_policy["per_time_signal_floor_fraction"]
                ),
                peak_plateau_fraction=float(
                    metric_policy["peak_plateau_fraction"]
                ),
                lag_minimum_overlap_fraction=float(
                    metric_policy["lag_minimum_overlap_fraction"]
                ),
            )
        else:
            metric_policy = config["metric_policy"]
            metrics = _comparison_metrics_v4(
                inhouse_eta,
                external_eta,
                times,
                gauges,
                inactive_floor=float(
                    config["gauges"]["inactive_external_peak_floor"]
                ),
                per_time_signal_floor_fraction=float(
                    metric_policy["per_time_signal_floor_fraction"]
                ),
                peak_plateau_fraction=float(
                    metric_policy["peak_plateau_fraction"]
                ),
                lag_minimum_overlap_fraction=float(
                    metric_policy["lag_minimum_overlap_fraction"]
                ),
                diagnostic_boundary_band_cells=int(
                    v4_policy["diagnostic_boundary_band_cells"]
                ),
            )
        if schema_id in HARDENED_SCHEMA_IDS:
            source = case.get("source", {})
            parameters = source.get("parameters", {})
            if (
                comparator_id == "geoclaw_swe"
                and source.get("generator")
                in {"flat_linear_packet", "flat_linear_mode"}
            ):
                analytical = _flat_linear_swe_reference(
                    np.asarray(eta0, dtype=np.float64),
                    times,
                    depth=float(parameters["depth"]),
                    gravity=float(config["inhouse"]["gravity"]),
                )
                analytical_norm = max(
                    float(np.linalg.norm(analytical)),
                    float(config["gauges"]["inactive_external_peak_floor"]),
                )
                metrics.update(
                    {
                        "analytical_inhouse_relative_l2": float(
                            np.linalg.norm(inhouse_eta - analytical)
                            / analytical_norm
                        ),
                        "analytical_external_relative_l2": float(
                            np.linalg.norm(external_eta - analytical)
                            / analytical_norm
                        ),
                    }
                )
        thresholds = config["thresholds"][str(pairing["category"])]
        external_health = (
            {
                "solver_health_status": metadata["solver_health_status"],
                "external_runtime_seconds": metadata["runtime_seconds"],
                "ksp_solve_count": metadata.get("ksp_solve_count"),
                "ksp_iteration_max": metadata.get("ksp_iteration_max"),
                "ksp_iteration_mean": metadata.get("ksp_iteration_mean"),
            }
            if schema_id in HARDENED_SCHEMA_IDS
            else {}
        )
        if schema_id == SCHEMA_ID_V4:
            category_role = v4_policy["category_roles"][
                str(pairing["category"])
            ]
            comparison_role = str(category_role["comparison"])
            descriptive_metrics = (
                list(thresholds)
                if comparison_role == "descriptive_only"
                else category_role.get("descriptive_metrics", [])
            )
            threshold_results, gated_thresholds_passed = (
                _v4_threshold_results(
                    metrics,
                    thresholds,
                    descriptive_metrics=descriptive_metrics,
                    rel_tolerance=v4_rel_tolerance,
                    abs_tolerance=v4_abs_tolerance,
                )
            )
            thresholds_satisfied = all(
                bool(result["passed"]) for result in threshold_results.values()
            )
            row_passed = (
                gated_thresholds_passed
                if comparison_role == "gate"
                else True
            )
            decision_fields = {
                "decision_role": comparison_role,
                "decision_reason": (
                    "matched-regime field and amplitude comparison"
                    if comparison_role == "gate"
                    else (
                        "flat-grid comparison supports refinement evidence"
                        if str(pairing["category"]) == "flat_analytical"
                        else (
                            "complex production comparison is a compatibility "
                            "diagnostic, not pointwise external truth"
                        )
                    )
                ),
                "threshold_results": threshold_results,
                "thresholds_satisfied": thresholds_satisfied,
                "passed": row_passed,
            }
        else:
            decision_fields = {
                "passed": _metrics_pass(metrics, thresholds),
            }
        rows.append(
            {
                **pairing,
                "nx": int(case["nx"]),
                "ny": int(case["ny"]),
                "comparator_version": metadata["comparator_version"],
                "comparator_commit": metadata["comparator_commit"],
                **external_health,
                **metrics,
                **decision_fields,
            }
        )
        if progress is not None:
            progress(
                f"[level-b-evaluate] {pairing_index}/{total_pairings} "
                f"{case_id} {solver_name} vs {comparator_id} "
                f"passed={rows[-1]['passed']}"
            )

    refinement_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    gated_pairings = set(
        str(value)
        for value in config["thresholds"]["refinement"]["gated_pairings"]
    )
    ratio_limit = float(
        config["thresholds"]["refinement"][
            "finest_to_coarsest_error_ratio_max"
        ]
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        base_case = str(row["case_id"]).split("_nx", 1)[0]
        groups.setdefault((base_case, str(row["pairing_id"])), []).append(row)
    for (base_case, pairing_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["nx"]))
        if len(ordered) < 2:
            continue
        if schema_id == SCHEMA_ID_V4:
            category = str(ordered[0]["category"])
            category_role = v4_policy["category_roles"][category]
            refinement_role = str(category_role.get("refinement", "descriptive_only"))
            error_metrics = (
                list(
                    v4_policy["flat_analytical_verification"][
                        "gated_error_series"
                    ]
                )
                if category == "flat_analytical"
                else ["trajectory_relative_l2"]
            )
            for error_metric in error_metrics:
                if any(row.get(error_metric) is None for row in ordered):
                    raise RuntimeError(
                        f"Missing v4 refinement metric {error_metric}: "
                        f"{base_case} {pairing_id}"
                    )
                refinement = _v4_pairwise_refinement(
                    [float(row[error_metric]) for row in ordered],
                    [int(row["nx"]) for row in ordered],
                    ratio_limit=ratio_limit,
                    require_strict_decrease=bool(
                        v4_policy["flat_analytical_verification"][
                            "require_pairwise_strict_decrease"
                        ]
                    ),
                    rel_tolerance=v4_rel_tolerance,
                    abs_tolerance=v4_abs_tolerance,
                )
                refinement_rows.append(
                    {
                        "base_case": base_case,
                        "pairing_id": pairing_id,
                        "error_metric": error_metric,
                        "decision_role": refinement_role,
                        **refinement,
                        "passed": (
                            bool(refinement["passed"])
                            if refinement_role == "gate"
                            else True
                        ),
                    }
                )
            if category == "flat_analytical":
                finest = ordered[-1]
                limit = float(
                    v4_policy["flat_analytical_verification"][
                        "finest_analytical_inhouse_relative_l2_max"
                    ]
                )
                value = finest.get("analytical_inhouse_relative_l2")
                passed = _v4_at_or_below(
                    value,
                    limit,
                    integer=False,
                    rel_tolerance=v4_rel_tolerance,
                    abs_tolerance=v4_abs_tolerance,
                )
                verification_rows.append(
                    {
                        "base_case": base_case,
                        "pairing_id": pairing_id,
                        "verification": (
                            "finest_analytical_inhouse_relative_l2"
                        ),
                        "finest_nx": int(finest["nx"]),
                        "value": value,
                        "limit": limit,
                        "decision_role": "gate",
                        "passed": passed,
                    }
                )
            continue

        coarse = float(ordered[0]["trajectory_relative_l2"])
        fine = float(ordered[-1]["trajectory_relative_l2"])
        ratio = fine / max(coarse, 1e-30)
        gated = pairing_id in gated_pairings
        refinement_rows.append(
            {
                "base_case": base_case,
                "pairing_id": pairing_id,
                "coarsest_nx": int(ordered[0]["nx"]),
                "finest_nx": int(ordered[-1]["nx"]),
                "coarsest_trajectory_relative_l2": coarse,
                "finest_trajectory_relative_l2": fine,
                "finest_to_coarsest_error_ratio": ratio,
                "decision_role": "gate" if gated else "descriptive_only",
                "passed": ratio <= ratio_limit if gated else True,
            }
        )
    comparison_passed = all(bool(row["passed"]) for row in rows)
    refinement_passed = all(bool(row["passed"]) for row in refinement_rows)
    verification_passed = all(
        bool(row["passed"]) for row in verification_rows
    )
    level_b_passed = (
        comparison_passed and refinement_passed and verification_passed
    )
    decision = {
        "schema_id": schema_id,
        "bundle_hash": frozen["bundle_hash"],
        "minimum_level_b_passed": level_b_passed,
        "decision": (
            "pass_to_H1"
            if level_b_passed
            else "blocked_established_solver_validation"
        ),
        "comparison_count": len(rows),
        "failed_comparisons": [row for row in rows if not row["passed"]],
        "failed_refinements": [
            row for row in refinement_rows if not row["passed"]
        ],
        "external_solver_is_truth": False,
    }
    if schema_id == SCHEMA_ID_V4:
        descriptive_failures = []
        for row in rows:
            if (
                row["decision_role"] == "descriptive_only"
                and not row["thresholds_satisfied"]
            ):
                descriptive_failures.append(
                    {
                        "case_id": row["case_id"],
                        "pairing_id": row["pairing_id"],
                        "failed_descriptive_metrics": [
                            key
                            for key, result in row["threshold_results"].items()
                            if not result["passed"]
                        ],
                    }
                )
        decision.update(
            {
                "gated_comparison_count": sum(
                    row["decision_role"] == "gate" for row in rows
                ),
                "descriptive_comparison_count": sum(
                    row["decision_role"] == "descriptive_only" for row in rows
                ),
                "descriptive_threshold_failure_count": len(
                    descriptive_failures
                ),
                "descriptive_threshold_failures": descriptive_failures,
                "verification_count": len(verification_rows),
                "failed_verifications": [
                    row for row in verification_rows if not row["passed"]
                ],
            }
        )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Level B evaluation: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "decision.json", decision)
    _write_json(output_root / "comparison_rows.json", rows)
    _write_json(output_root / "refinement_rows.json", refinement_rows)
    _write_csv(output_root / "comparison_rows.csv", rows)
    _write_csv(output_root / "refinement_rows.csv", refinement_rows)
    if schema_id == SCHEMA_ID_V4:
        _write_json(output_root / "verification_rows.json", verification_rows)
        _write_csv(output_root / "verification_rows.csv", verification_rows)
    if schema_id == SCHEMA_ID_V4:
        production_rows = [
            row for row in rows if row["category"] == "production_input"
        ]
        production_lines = [
            (
                f"- `{row['case_id']}` / `{row['pairing_id']}`: "
                f"trajectory={row['trajectory_relative_l2']:.6f}, "
                f"active-time-p95="
                f"{row['per_time_scaled_l2_p95_active']:.6f}, "
                f"norm-ratio={row['field_norm_ratio']:.6f}, "
                f"cosine={row['field_cosine_similarity']:.6f}, "
                f"boundary-band={row['boundary_band_relative_l2']:.6f}, "
                f"interior={row['interior_relative_l2']:.6f}, "
                f"legacy-limits-satisfied={row['thresholds_satisfied']}"
            )
            for row in production_rows
        ]
        report = (
            "# Minimum established-solver validation\n\n"
            f"Decision: `{decision['decision']}`\n\n"
            "GeoClaw is an independent comparator, not automatic physical "
            "truth. Version 4 gates analytical/refinement verification and "
            "matched-regime comparisons. Complex production comparisons are "
            "mandatory compatibility diagnostics; their numerical limits are "
            "still evaluated and reported below.\n\n"
            "## Decision evidence\n\n"
            f"- Gated comparisons: {decision['gated_comparison_count']}\n"
            f"- Verification gates: {decision['verification_count']}\n"
            f"- Refinement gates: "
            f"{sum(row['decision_role'] == 'gate' for row in refinement_rows)}\n"
            f"- Descriptive rows outside legacy limits: "
            f"{decision['descriptive_threshold_failure_count']}\n\n"
            "## Production compatibility diagnostics\n\n"
            + "\n".join(production_lines)
            + "\n\n"
            "These production discrepancies remain documented limitations. A "
            "`pass_to_H1` decision does not claim that the 96x96 SWE pipeline "
            "is pointwise equivalent to GeoClaw or boundary-independent.\n"
        )
    else:
        report = (
            "# Minimum established-solver validation\n\n"
            f"Decision: `{decision['decision']}`\n\n"
            "GeoClaw is an independent comparator, not automatic physical truth.\n"
        )
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    _write_checksums(output_root)
    return output_root


def established_solver_status(
    *, bundle_root: Path, external_root: Path
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    external_root = external_root.resolve()
    validate_checksums(bundle_root)
    frozen = _read_json(bundle_root / "frozen_contract.json")
    schema_id = str(frozen.get("schema_id", ""))
    run_manifest: Mapping[str, Any] | None = None
    global_error: str | None = None
    if schema_id in HARDENED_SCHEMA_IDS:
        try:
            run_manifest = _load_external_run_manifest(external_root, frozen)
            _validate_external_checksums(external_root, frozen)
        except Exception as exc:
            global_error = str(exc)
    requested_times = np.asarray(frozen["requested_times"], dtype=np.float64)
    valid: list[str] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for requirement in frozen["external_results"]:
        relative = str(requirement["relative_path"])
        path = external_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            if global_error is not None:
                raise RuntimeError(global_error)
            _load_external_result(
                path, requirement, requested_times, run_manifest
            )
        except Exception as exc:
            invalid.append({"relative_path": relative, "error": str(exc)})
        else:
            valid.append(relative)
    total = len(frozen["external_results"])
    status = {
        "bundle_hash": frozen["bundle_hash"],
        "valid": len(valid),
        "missing": len(missing),
        "invalid": len(invalid),
        "total": total,
        "complete": len(valid) == total,
        "valid_paths": valid,
        "missing_paths": missing,
        "invalid_paths": invalid,
    }
    if schema_id in HARDENED_SCHEMA_IDS:
        status["external_provenance_error"] = global_error
    return status
