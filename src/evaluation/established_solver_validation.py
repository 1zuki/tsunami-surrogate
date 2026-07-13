from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    code_state,
    hash_array,
    sha256_file,
    stable_hash_payload,
)
from src.evaluation.common_time_v2_level_a import (
    _load_canary_arrays,
    _trajectory_eta,
    validate_checksums,
)


SCHEMA_ID = "tsunami-surrogate.minimum-established-solver-validation.v1"
EXTERNAL_RESULT_SCHEMA_ID = (
    "tsunami-surrogate.minimum-established-solver-external-result.v1"
)
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


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != SCHEMA_ID:
        raise ValueError("Established-solver validation schema mismatch")
    if config.get("artifact_kind") != "minimum-established-solver-validation-candidate":
        raise ValueError("Established-solver validation artifact kind mismatch")
    if config["external_comparator"].get("version") != "5.14.0":
        raise ValueError("The minimum package must pin Clawpack 5.14.0")
    if not bool(config["prerequisites"].get("require_level_a_pass")):
        raise ValueError("The minimum package must require a passing Level A")
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

    required_thresholds = {
        "trajectory_relative_l2",
        "per_time_relative_l2_p95",
        "gauge_nrmse_max",
        "arrival_time_abs_max",
        "peak_relative_error_max",
        "time_to_peak_abs_max",
        "waveform_lag_steps_max",
    }
    for category in categories:
        values = config["thresholds"][category]
        if set(values) != required_thresholds:
            raise ValueError(f"Threshold schema mismatch for {category}")
        if any(float(value) <= 0.0 for value in values.values()):
            raise ValueError(f"Thresholds must be positive for {category}")


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
        schema_id=SCHEMA_ID,
    )
    return identity


def _build_cases(
    config: Mapping[str, Any], level_a_contract: Mapping[str, Any]
) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    cases: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
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
                )
                cases.append((record, arrays))
            continue

        canaries = list(level_a_contract["canaries"])
        count = int(spec["count"])
        if len(canaries) < count:
            raise RuntimeError("Passing Level A contract has too few frozen canaries")
        for canary in canaries[:count]:
            bathymetry, _source, _strength_array, _strength, loaded = (
                _load_canary_arrays(canary)
            )
            nx, ny = bathymetry.shape
            arrays = {
                "bathymetry": np.asarray(loaded["bathymetry"], dtype=np.float64),
                "eta0": np.asarray(loaded["eta0"], dtype=np.float64),
                "initial_depth": np.asarray(
                    loaded["initial_depth"], dtype=np.float64
                ),
                "hu0": np.zeros((nx, ny), dtype=np.float64),
                "hv0": np.zeros((nx, ny), dtype=np.float64),
                "eta_t0": np.zeros((nx, ny), dtype=np.float64),
            }
            record = _case_record(
                case_id=f"{spec['case_id']}_{str(canary['qualified_id']).replace(':', '_')}",
                category=str(spec["category"]),
                nx=nx,
                ny=ny,
                boundary=str(spec["boundary"]),
                arrays=arrays,
                pairings=spec["pairings"],
                gauges=_gauge_indices(nx, ny, fractions),
                source={
                    "generator": generator,
                    "qualified_id": canary["qualified_id"],
                    "input_fingerprint": canary["input_fingerprint"],
                },
            )
            cases.append((record, arrays))
    return cases


def _run_inhouse(
    config: Mapping[str, Any],
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    solver_name: str,
) -> np.ndarray:
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


def _external_result_relative_path(case_id: str, comparator_id: str) -> str:
    return f"{case_id}/{comparator_id}.npz"


def prepare_minimum_established_solver_validation(
    *,
    repo_root: Path,
    config_path: Path,
    level_a_root: Path,
    output_root: Path | None = None,
) -> Path:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    level_a_root = level_a_root.resolve()
    config = _load_config(config_path)
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

    times = _requested_times(config)
    if not np.array_equal(
        times, np.asarray(level_a_contract["candidate_times"], dtype=np.float64)
    ):
        raise RuntimeError("Level B requested times differ from passing Level A")
    cases = _build_cases(config, level_a_contract)
    case_records = [record for record, _arrays in cases]
    external_requirements: dict[tuple[str, str], dict[str, Any]] = {}
    pairings: list[dict[str, Any]] = []
    for record in case_records:
        for solver_name, comparator_id in record["pairings"]:
            key = (str(record["case_id"]), str(comparator_id))
            external_requirements[key] = {
                "case_id": record["case_id"],
                "case_hash": record["case_hash"],
                "comparator_id": comparator_id,
                "comparator_version": config["external_comparator"]["version"],
                "relative_path": _external_result_relative_path(*key),
                "required_npz_keys": [
                    "schema_id",
                    "case_hash",
                    "comparator_id",
                    "comparator_version",
                    "comparator_commit",
                    "times",
                    "eta",
                ],
                "eta_shape": [int(times.size), int(record["nx"]), int(record["ny"])],
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
        "schema_id": SCHEMA_ID,
        "artifact_kind": "minimum-established-solver-validation-frozen-contract",
        "source_config": _json_safe(config),
        "source_config_sha256": sha256_file(config_path),
        "code_state": current_code,
        "level_a": {
            "root": str(level_a_root),
            "contract_hash": level_a_contract["contract_hash"],
            "scientific_digest": level_a_decision["scientific_digest"],
            "decision": level_a_decision["decision"],
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
        schema_id=SCHEMA_ID,
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
                "schema_id": EXTERNAL_RESULT_SCHEMA_ID,
                "bundle_hash": bundle_hash,
                "results": list(external_requirements.values()),
            },
        )
        for record, arrays in cases:
            case_dir = staging / "cases" / str(record["case_id"])
            case_dir.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(
                case_dir / "input.npz",
                **arrays,
                requested_times=times,
                gauge_indices=np.asarray(record["gauges"], dtype=np.int64),
                case_hash=np.asarray(record["case_hash"]),
            )
            inhouse_solvers = sorted(
                {str(pairing[0]) for pairing in record["pairings"]}
            )
            for solver_name in inhouse_solvers:
                eta = _run_inhouse(config, record, arrays, solver_name)
                np.savez_compressed(
                    case_dir / f"inhouse_{solver_name}.npz",
                    eta=eta,
                    times=times,
                    case_hash=np.asarray(record["case_hash"]),
                    solver_id=np.asarray(solver_name),
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


def _load_external_result(
    path: Path,
    requirement: Mapping[str, Any],
    requested_times: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
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
    expected_metadata = {
        "schema_id": EXTERNAL_RESULT_SCHEMA_ID,
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
    if not np.isfinite(eta).all():
        raise RuntimeError(f"External result {path} contains nonfinite eta")
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


def _metrics_pass(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    if int(metrics["active_gauge_count"]) <= 0:
        return False
    return all(
        metrics[key] is not None and float(metrics[key]) <= float(limit)
        for key, limit in thresholds.items()
    )


def evaluate_minimum_established_solver_validation(
    *,
    bundle_root: Path,
    external_root: Path,
    output_root: Path,
) -> Path:
    bundle_root = bundle_root.resolve()
    external_root = external_root.resolve()
    output_root = output_root.resolve()
    validate_checksums(bundle_root)
    frozen = _read_json(bundle_root / "frozen_contract.json")
    if frozen.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("Frozen Level B schema mismatch")
    bundle_identity = dict(frozen)
    recorded_hash = bundle_identity.pop("bundle_hash", None)
    expected_hash = stable_hash_payload(
        artifact_kind="minimum-established-solver-validation-contract",
        payload=bundle_identity,
        schema_id=SCHEMA_ID,
    )
    if recorded_hash != expected_hash or bundle_root.name != expected_hash:
        raise RuntimeError("Frozen Level B content-addressed identity mismatch")
    config = frozen["source_config"]
    times = np.asarray(frozen["requested_times"], dtype=np.float64)
    case_by_id = {str(row["case_id"]): row for row in frozen["cases"]}
    requirement_by_key = {
        (str(row["case_id"]), str(row["comparator_id"])): row
        for row in frozen["external_results"]
    }
    external_cache: dict[tuple[str, str], tuple[np.ndarray, dict[str, str]]] = {}
    rows: list[dict[str, Any]] = []
    for pairing in frozen["pairings"]:
        case_id = str(pairing["case_id"])
        comparator_id = str(pairing["external_comparator"])
        key = (case_id, comparator_id)
        requirement = requirement_by_key[key]
        if key not in external_cache:
            external_cache[key] = _load_external_result(
                external_root / str(requirement["relative_path"]),
                requirement,
                times,
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
        with np.load(
            bundle_root / "cases" / case_id / "input.npz", allow_pickle=False
        ) as payload:
            gauges = np.asarray(payload["gauge_indices"], dtype=np.int64)
        metrics = _comparison_metrics(
            inhouse_eta,
            external_eta,
            times,
            gauges,
            arrival_fraction=float(
                config["gauges"]["arrival_fraction_of_external_peak"]
            ),
            inactive_floor=float(config["gauges"]["inactive_external_peak_floor"]),
        )
        thresholds = config["thresholds"][str(pairing["category"])]
        rows.append(
            {
                **pairing,
                "nx": int(case["nx"]),
                "ny": int(case["ny"]),
                "comparator_version": metadata["comparator_version"],
                "comparator_commit": metadata["comparator_commit"],
                **metrics,
                "passed": _metrics_pass(metrics, thresholds),
            }
        )

    refinement_rows: list[dict[str, Any]] = []
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
    decision = {
        "schema_id": SCHEMA_ID,
        "bundle_hash": frozen["bundle_hash"],
        "minimum_level_b_passed": comparison_passed and refinement_passed,
        "decision": (
            "pass_to_H1"
            if comparison_passed and refinement_passed
            else "blocked_established_solver_validation"
        ),
        "comparison_count": len(rows),
        "failed_comparisons": [row for row in rows if not row["passed"]],
        "failed_refinements": [
            row for row in refinement_rows if not row["passed"]
        ],
        "external_solver_is_truth": False,
    }
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Level B evaluation: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "decision.json", decision)
    _write_json(output_root / "comparison_rows.json", rows)
    _write_json(output_root / "refinement_rows.json", refinement_rows)
    _write_csv(output_root / "comparison_rows.csv", rows)
    _write_csv(output_root / "refinement_rows.csv", refinement_rows)
    (output_root / "REPORT.md").write_text(
        "# Minimum established-solver validation\n\n"
        f"Decision: `{decision['decision']}`\n\n"
        "GeoClaw is an independent comparator, not automatic physical truth.\n",
        encoding="utf-8",
    )
    _write_checksums(output_root)
    return output_root
