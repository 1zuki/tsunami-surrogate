#!/usr/bin/env python
"""Validate the current 384 -> 128 -> 192 -> 64 production contract.

This is deliberately separate from the historical common-time-v2 H0/A/B/H1/H2
chain.  It validates the data contract used by the current production rebuild,
then reruns a small, deterministic set of 192-cell buffered solver canaries
under the current 420-time-unit configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.common_time_v2 import (  # noqa: E402
    PUBLICATION_SCHEMA_ID,
    code_state,
    parse_requested_output_config,
)
from src.data_gen.simulate_dataset import (  # noqa: E402
    BufferedDomainConfig,
    _block_mean_downsample,
    _block_mean_downsample_spatial,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
    _simulate_one_local,
)
from src.utils.hashing import sha256_file  # noqa: E402


SCHEMA_ID = "tsunami-surrogate.current-production-contract-validation.v1"
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
SOLVER_DIRS = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
}
EXPECTED_LINEAGE_SCHEMA = "tsunami-surrogate.native-resolution-inputs.v1"
EXPECTED_SAMPLE_SCHEMA = "tsunami-surrogate.common-time-v2.eta-sample.v1"


class ValidationError(RuntimeError):
    """Raised when the current production contract is not proven."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"Missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"Missing JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Malformed JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValidationError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _expected_times(contract: Mapping[str, Any]) -> np.ndarray:
    requested = contract["scientific_scope"]["requested_times"]
    start = float(requested["start"])
    step = float(requested["step"])
    count = int(requested["count"])
    horizon = float(requested["horizon"])
    times = start + step * np.arange(count, dtype=np.float64)
    times[-1] = horizon
    return times


def _validate_generation_configs(
    contract: Mapping[str, Any], expected_times: np.ndarray
) -> list[dict[str, Any]]:
    scientific = contract["scientific_scope"]
    expected_hash = str(scientific["contract_hash"])
    domain = scientific["computational_domain"]
    expected_solver_shape = tuple(int(value) for value in domain["solver_shape"])
    expected_publication_shape = tuple(
        int(value) for value in domain["publication_shape"]
    )
    expected_buffer = int(domain["buffer_cells"])
    summaries: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        spec = contract["main_datasets"]["splits"][split_name]
        path = _repo_path(str(spec["generation_config"]))
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        requested = parse_requested_output_config(raw.get("requested_output"))
        if requested is None:
            raise ValidationError(f"Requested output is disabled: {path}")
        if requested.contract_hash != expected_hash:
            raise ValidationError(f"Generation contract hash mismatch: {path}")
        if not np.array_equal(requested.requested_times, expected_times):
            raise ValidationError(f"Generation timestamps mismatch: {path}")
        paired = raw.get("paired_inputs", {})
        buffered = raw.get("computational_domain", {})
        solver = raw.get("solver", {})
        solver_input = tuple(int(value) for value in paired.get("solver_shape", []))
        target = tuple(int(value) for value in paired.get("target_shape", []))
        computation = (int(solver.get("nx", -1)), int(solver.get("ny", -1)))
        if solver_input != (128, 128) or target != expected_publication_shape:
            raise ValidationError(f"Input/publication lineage mismatch: {path}")
        if computation != expected_solver_shape or int(
            buffered.get("buffer_cells", -1)
        ) != expected_buffer:
            raise ValidationError(f"Buffered computational contract mismatch: {path}")
        if raw.get("fdes", {}).get("enabled") != list(SOLVERS):
            raise ValidationError(f"Solver roster mismatch: {path}")
        profiles = raw.get("solver_profiles", {})
        expected_profiles = {
            "swe_hydrostatic": {
                "cfl": 0.1125,
                "boundary": "radiation",
                "sponge_reference_dt": 8.4,
            },
            "swe_muscl_hr": {
                "cfl": 0.225,
                "boundary": "radiation",
                "sponge_reference_dt": 8.4,
            },
            "boussinesq": {
                "cfl": 0.35,
                "boundary": "open",
                "depth_scale": 1.0,
                "sponge_reference_dt": 8.4,
                "filter_time_mode": "disabled",
                "filter_reference_dt": 8.4,
                "cg_failure_mode": "strict_v2",
            },
        }
        for solver_name, required in expected_profiles.items():
            profile = profiles.get(solver_name, {})
            if any(profile.get(key) != value for key, value in required.items()):
                raise ValidationError(
                    f"Production solver profile mismatch for {solver_name}: {path}"
                )
        summaries.append(
            {
                "split": split_name,
                "config": str(path.relative_to(ROOT)),
                "solver_input_shape": list(solver_input),
                "buffered_shape": list(computation),
                "publication_shape": list(target),
                "requested_time_count": int(expected_times.size),
                "requested_horizon": float(expected_times[-1]),
            }
        )
    return summaries


def _validate_raw_sample(
    sample_dir: Path,
    row: Mapping[str, Any],
    *,
    expected_times: np.ndarray,
    expected_contract_hash: str,
    expected_split: str,
    expected_solver: str,
    expected_code_state_hash: str,
    deep_payload_audit: bool,
) -> None:
    meta = _read_json(sample_dir / "meta.json")
    publication = _read_json(sample_dir / "publication.json")
    health = meta.get("health_summary")
    if not isinstance(health, Mapping):
        raise ValidationError(f"Raw numerical-health summary is missing: {sample_dir}")
    required_meta = {
        "schema_id": EXPECTED_SAMPLE_SCHEMA,
        "contract_hash": expected_contract_hash,
        "split": expected_split,
        "solver_name": expected_solver,
        "code_state_hash": expected_code_state_hash,
        "quality_status": "ok",
    }
    for key, value in required_meta.items():
        if meta.get(key) != value:
            raise ValidationError(
                f"Raw metadata {key} mismatch for {sample_dir}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    for key in ("requested_output_count", "covered_requested_output_count"):
        if health.get(key) != int(expected_times.size):
            raise ValidationError(
                f"Raw numerical health {key} mismatch for {sample_dir}: "
                f"{health.get(key)!r} != {int(expected_times.size)!r}"
            )
    if (
        int(health.get("nan_count", -1)) != 0
        or int(health.get("inf_count", -1)) != 0
        or not math.isfinite(float(health.get("max_post_step_cfl", math.nan)))
        or float(health.get("max_post_step_cfl", math.inf))
        > float(health.get("target_cfl", math.nan)) * 1.01
        or meta.get("quality_violations") not in ([], None)
    ):
        raise ValidationError(f"Raw numerical health failed: {sample_dir}")
    for key in (
        "operator_nan_to_num_replacement_count",
        "operator_positivity_projection_count",
        "operator_dry_projection_count",
    ):
        if int(health.get(key, 0) or 0) != 0:
            raise ValidationError(f"Raw quality counter {key} is nonzero: {sample_dir}")
    if expected_solver == "swe_muscl_hr":
        for key in (
            "operator_muscl_cell_velocity_clip_count",
            "operator_muscl_face_velocity_clip_count",
        ):
            if int(health.get(key, 0) or 0) != 0:
                raise ValidationError(f"Raw MUSCL quality counter {key} is nonzero: {sample_dir}")
    if expected_solver == "boussinesq" and (
        not bool(health.get("has_cg_diagnostics", False))
        or int(health.get("cg_failed_count", -1)) != 0
        or float(health.get("cg_converged_fraction", 0.0)) != 1.0
    ):
        raise ValidationError(f"Raw Boussinesq convergence failed: {sample_dir}")
    if meta.get("scenario_id") != row.get("scenario_id"):
        raise ValidationError(f"Scenario identity mismatch: {sample_dir}")
    domain = meta.get("computational_domain")
    lineage = meta.get("input_lineage")
    if not isinstance(domain, Mapping) or not isinstance(lineage, Mapping):
        raise ValidationError(f"Missing domain/lineage metadata: {sample_dir}")
    if (
        tuple(domain.get("solver_shape", [])) != (192, 192)
        or tuple(domain.get("publication_shape", [])) != (64, 64)
        or int(domain.get("buffer_cells", -1)) != 32
    ):
        raise ValidationError(f"192 -> 64 crop contract mismatch: {sample_dir}")
    if (
        lineage.get("schema_id") != EXPECTED_LINEAGE_SCHEMA
        or tuple(lineage.get("master_shape", [])) != (384, 384)
        or tuple(lineage.get("solver_shape", [])) != (128, 128)
        or tuple(lineage.get("target_shape", [])) != (64, 64)
        or lineage.get("downsample_method") != "block_mean_float64_v1"
        or lineage.get("solver_input") != "solver"
    ):
        raise ValidationError(f"384 -> 128 input lineage mismatch: {sample_dir}")
    if publication.get("schema_id") != PUBLICATION_SCHEMA_ID:
        raise ValidationError(f"Publication schema mismatch: {sample_dir}")
    for key, value in (
        ("contract_hash", expected_contract_hash),
        ("split", expected_split),
        ("solver_name", expected_solver),
        ("scenario_id", row.get("scenario_id")),
    ):
        if publication.get(key) != value:
            raise ValidationError(f"Publication {key} mismatch: {sample_dir}")
    listed = publication.get("files")
    if not isinstance(listed, list):
        raise ValidationError(f"Publication file inventory missing: {sample_dir}")
    required_names = {"sample.npz", "provenance.npz", "meta.json"}
    observed_names = {str(item.get("name", "")) for item in listed}
    if not required_names.issubset(observed_names):
        raise ValidationError(f"Publication file inventory incomplete: {sample_dir}")
    for item in listed:
        payload_path = sample_dir / str(item.get("name", ""))
        if not payload_path.is_file() or payload_path.stat().st_size != int(
            item.get("size_bytes", -1)
        ):
            raise ValidationError(f"Publication payload missing/size mismatch: {sample_dir}")
    if deep_payload_audit:
        for item in listed:
            payload_path = sample_dir / str(item["name"])
            if sha256_file(payload_path) != str(item.get("sha256", "")):
                raise ValidationError(
                    f"Publication payload hash mismatch: {payload_path}"
                )
        with np.load(sample_dir / "sample.npz", allow_pickle=False) as payload:
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
            trajectory = np.asarray(payload["trajectory_eta"])
            if not np.array_equal(timestamps, expected_times):
                raise ValidationError(f"Requested timestamps mismatch: {sample_dir}")
            if trajectory.shape != (expected_times.size, 64, 64):
                raise ValidationError(f"Publication shape mismatch: {sample_dir}")
            if not np.isfinite(trajectory).all():
                raise ValidationError(
                    f"Non-finite publication trajectory: {sample_dir}"
                )
        with np.load(sample_dir / "provenance.npz", allow_pickle=False) as provenance:
            for name in (
                "requested_timestamps",
                "post_step_cfl",
                "finite_state_flag",
            ):
                if name not in provenance:
                    raise ValidationError(
                        f"Missing provenance field {name}: {sample_dir}"
                    )
            if not np.array_equal(
                np.asarray(provenance["requested_timestamps"], dtype=np.float64),
                expected_times,
            ):
                raise ValidationError(
                    f"Provenance timestamps mismatch: {sample_dir}"
                )
            post_cfl = np.asarray(provenance["post_step_cfl"], dtype=np.float64)
            finite = np.asarray(provenance["finite_state_flag"], dtype=bool)
            if not np.isfinite(post_cfl).all() or not bool(finite.all()):
                raise ValidationError(
                    f"Non-finite natural-step provenance: {sample_dir}"
                )


def _validate_raw_datasets(
    contract: Mapping[str, Any],
    *,
    expected_times: np.ndarray,
    deep_payload_audit: bool,
) -> dict[str, Any]:
    expected_contract_hash = str(contract["scientific_scope"]["contract_hash"])
    split_names = {"train": "train", "val": "eval", "test": "test"}
    result: dict[str, Any] = {}
    for split_name, publication_split in split_names.items():
        print(f"[production-validation] audit split={split_name}", flush=True)
        spec = contract["main_datasets"]["splits"][split_name]
        raw_root = _repo_path(str(spec["raw_root"]))
        frozen = contract["main_datasets"]["frozen_generation_artifacts"][
            split_name
        ]
        expected_code_state_hash = str(frozen["code_state_hash"])
        for frozen_key in ("config_snapshot", "scenario_manifest", "operational_shard"):
            frozen_spec = frozen[frozen_key]
            frozen_path = _repo_path(str(frozen_spec["path"]))
            if not frozen_path.is_file() or sha256_file(frozen_path) != str(
                frozen_spec["sha256"]
            ):
                raise ValidationError(f"Frozen {frozen_key} hash mismatch: {frozen_path}")
        for frozen_spec in frozen["solver_manifests"].values():
            frozen_path = _repo_path(str(frozen_spec["path"]))
            if not frozen_path.is_file() or sha256_file(frozen_path) != str(
                frozen_spec["sha256"]
            ):
                raise ValidationError(f"Frozen solver manifest hash mismatch: {frozen_path}")
        expected_count = int(spec["count"])
        shard_path = _repo_path(str(frozen["operational_shard"]["path"]))
        shard = _read_json(shard_path)
        if (
            shard.get("schema_id")
            != "tsunami-surrogate.common-time-v2.operational-shard.v1"
            or shard.get("split") != publication_split
            or shard.get("contract_hash") != expected_contract_hash
            or shard.get("code_state_hash") != expected_code_state_hash
            or shard.get("complete") is not True
            or shard.get("solver_names") != sorted(SOLVERS)
            or len(shard.get("publications", [])) != expected_count * len(SOLVERS)
        ):
            raise ValidationError(
                f"Frozen operational shard contract mismatch: {shard_path}"
            )
        by_solver: dict[str, int] = {}
        rosters: dict[str, set[str]] = {}
        for solver_name, directory in SOLVER_DIRS.items():
            manifest_path = _repo_path(
                str(frozen["solver_manifests"][directory]["path"])
            )
            rows = _read_jsonl(manifest_path)
            if len(rows) != expected_count:
                raise ValidationError(
                    f"Raw {solver_name} count mismatch for {split_name}: "
                    f"{len(rows)} != {expected_count}"
                )
            roster: set[str] = set()
            for row in rows:
                scenario_id = str(row.get("scenario_id", ""))
                if not scenario_id or scenario_id in roster:
                    raise ValidationError(f"Duplicate raw scenario: {manifest_path}")
                roster.add(scenario_id)
                sample_dir = _repo_path(str(row.get("sample_dir", "")))
                _validate_raw_sample(
                    sample_dir,
                    row,
                    expected_times=expected_times,
                    expected_contract_hash=expected_contract_hash,
                    expected_split=publication_split,
                    expected_solver=solver_name,
                    expected_code_state_hash=expected_code_state_hash,
                    deep_payload_audit=deep_payload_audit,
                )
                if len(roster) % 1000 == 0 or len(roster) == expected_count:
                    print(
                        f"[production-validation] {split_name}:{solver_name} "
                        f"{len(roster)}/{expected_count}",
                        flush=True,
                    )
            by_solver[solver_name] = len(rows)
            rosters[solver_name] = roster
        if len({frozenset(value) for value in rosters.values()}) != 1:
            raise ValidationError(f"Solver roster mismatch for {split_name}")
        result[split_name] = {
            "raw_root": str(raw_root.relative_to(ROOT)),
            "samples_per_solver": by_solver,
            "scenario_count": len(next(iter(rosters.values()))),
        }
    return result


def _solver_from_name(name: str, cfg: Mapping[str, Any]) -> Any:
    if name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(dict(cfg))
    if name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(dict(cfg))
    return _make_boussinesq_solver_from_cfg(dict(cfg))


def _set_solver_initial_state(
    solver: Any,
    solver_name: str,
    *,
    bathymetry: np.ndarray,
    eta0: np.ndarray,
    h0: np.ndarray,
) -> None:
    solver.set_bathymetry(bathymetry)
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        solver.set_initial_condition(
            h0,
            hu0=np.zeros_like(h0),
            hv0=np.zeros_like(h0),
        )
    else:
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))


def _run_well_balanced_check(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    steps: int = 32,
) -> dict[str, Any]:
    cfg = dict(solver_cfg)
    cfg.update({"boundary": "reflective", "use_sponge": False})
    solver = _solver_from_name(solver_name, cfg)
    zero_eta = np.zeros_like(prepared["solver_bathymetry"], dtype=np.float64)
    rest_depth = np.maximum(-prepared["solver_bathymetry"], 0.0)
    _set_solver_initial_state(
        solver,
        solver_name,
        bathymetry=prepared["solver_bathymetry"],
        eta0=zero_eta,
        h0=rest_depth,
    )
    initial_eta = np.asarray(solver.compute_free_surface(), dtype=np.float64)
    max_drift = 0.0
    max_velocity_like = 0.0
    for _ in range(steps):
        dt = float(solver.suggest_dt(target_cfl=float(cfg.get("cfl", 0.45))))
        solver.dt = dt
        solver.step(dt=dt, auto_dt=False)
        eta = np.asarray(solver.compute_free_surface(), dtype=np.float64)
        max_drift = max(max_drift, float(np.max(np.abs(eta - initial_eta))))
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
            u, v = solver.compute_velocity()
            max_velocity_like = max(
                max_velocity_like,
                float(np.max(np.abs(u))),
                float(np.max(np.abs(v))),
            )
        else:
            max_velocity_like = max(
                max_velocity_like,
                float(np.max(np.abs(np.asarray(solver.eta_t, dtype=np.float64)))),
            )
    if not np.isfinite(max_drift) or not np.isfinite(max_velocity_like):
        raise ValidationError(f"Non-finite well-balanced diagnostic: {solver_name}")
    if max_drift > 1.0e-6 or max_velocity_like > 1.0e-5:
        raise ValidationError(
            f"Well-balanced check failed for {solver_name}: "
            f"eta_drift={max_drift:.6g}, velocity_like={max_velocity_like:.6g}"
        )
    return {
        "steps": int(steps),
        "boundary": "reflective",
        "use_sponge": False,
        "max_eta_drift": max_drift,
        "max_velocity_or_eta_t": max_velocity_like,
    }


def _run_conservation_check(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    steps: int = 32,
) -> dict[str, Any]:
    cfg = dict(solver_cfg)
    cfg.update({"boundary": "reflective", "use_sponge": False})
    solver = _solver_from_name(solver_name, cfg)
    _set_solver_initial_state(
        solver,
        solver_name,
        bathymetry=prepared["solver_bathymetry"],
        eta0=prepared["solver_eta0"],
        h0=prepared["solver_h0"],
    )
    area = float(cfg["dx"]) * float(cfg["dy"])
    initial_eta_integral = float(
        np.sum(np.asarray(solver.compute_free_surface(), dtype=np.float64)) * area
    )
    initial_mass = (
        float(np.sum(np.asarray(solver.h, dtype=np.float64)) * area)
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}
        else None
    )
    for _ in range(steps):
        dt = float(solver.suggest_dt(target_cfl=float(cfg.get("cfl", 0.45))))
        solver.dt = dt
        solver.step(dt=dt, auto_dt=False)
    state = np.asarray(solver.get_state(), dtype=np.float64)
    if not np.isfinite(state).all():
        raise ValidationError(f"Non-finite conservation diagnostic: {solver_name}")
    final_eta_integral = float(
        np.sum(np.asarray(solver.compute_free_surface(), dtype=np.float64)) * area
    )
    eta_integral_change = abs(final_eta_integral - initial_eta_integral)
    result: dict[str, Any] = {
        "steps": int(steps),
        "boundary": "reflective",
        "use_sponge": False,
        "finite_state": True,
        "free_surface_integral_abs_change": float(eta_integral_change),
    }
    if initial_mass is not None:
        final_mass = float(np.sum(np.asarray(solver.h, dtype=np.float64)) * area)
        relative_change = abs(final_mass - initial_mass) / max(abs(initial_mass), 1.0e-30)
        if relative_change > 1.0e-5:
            raise ValidationError(
                f"SWE conservation check failed for {solver_name}: "
                f"relative_mass_change={relative_change:.6g}"
            )
        result["mass_relative_change"] = float(relative_change)
    return result


def _run_canary_solver(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    bathymetry: np.ndarray,
    source: np.ndarray,
    source_strength: float,
    expected_times: np.ndarray,
    expected_trajectory: np.ndarray,
    *,
    max_natural_steps: int,
) -> dict[str, Any]:
    prepared = _prepare_buffered_domain(
        bathymetry,
        source,
        source_strength,
        0.0,
        # The cache source is already tapered on the 384 master grid before
        # reduction, so do not taper it a second time here.
        config=BufferedDomainConfig(
            enabled=True,
            buffer_cells=32,
            source_taper_cells=16,
            bathymetry_extension="edge",
            output_crop="central",
        ),
        source_already_tapered=True,
    )
    solver = _solver_from_name(solver_name, solver_cfg)
    solver.set_bathymetry(prepared["solver_bathymetry"])
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        _set_solver_initial_state(
            solver,
            solver_name,
            bathymetry=prepared["solver_bathymetry"],
            eta0=prepared["solver_eta0"],
            h0=prepared["solver_h0"],
        )
    else:
        _set_solver_initial_state(
            solver,
            solver_name,
            bathymetry=prepared["solver_bathymetry"],
            eta0=prepared["solver_eta0"],
            h0=prepared["solver_h0"],
        )
    states, emitted_times, _, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=250,
        save_every=5,
        auto_dt=True,
        target_cfl=float(solver_cfg.get("cfl", 0.45)),
        include_initial_state=True,
        requested_times=expected_times,
        max_natural_steps=max_natural_steps,
        collect_natural_step_health=True,
    )
    if not np.array_equal(np.asarray(emitted_times, dtype=np.float64), expected_times):
        raise ValidationError(f"Canary timestamps changed for {solver_name}")
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        eta = np.asarray(states[:, 0], dtype=np.float64)
        eta += prepared["solver_bathymetry"][None, ...]
    else:
        eta = np.asarray(states[:, 0], dtype=np.float64)
    crop = prepared["crop"]
    published = _block_mean_downsample_spatial(
        eta[..., crop[0], crop[1]], (64, 64)
    ).astype(np.float32)
    expected = np.asarray(expected_trajectory, dtype=np.float32)
    if published.shape != expected.shape:
        raise ValidationError(f"Canary publication shape changed for {solver_name}")
    difference = np.asarray(published, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    replay_max_abs = float(np.max(np.abs(difference)))
    replay_relative_l2 = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(np.asarray(expected, dtype=np.float64).ravel()), 1.0e-30)
    )
    if replay_max_abs > 5.0e-6 or replay_relative_l2 > 1.0e-5:
        raise ValidationError(
            f"Deterministic replay mismatch for {solver_name}: "
            f"max_abs={replay_max_abs:.6g}, rel_l2={replay_relative_l2:.6g}"
        )
    temporal_change = float(
        np.linalg.norm(
            np.asarray(published[-1], dtype=np.float64)
            - np.asarray(published[0], dtype=np.float64)
        )
        / max(np.linalg.norm(np.asarray(published[0], dtype=np.float64)), 1.0e-30)
    )
    if temporal_change <= 1.0e-4:
        raise ValidationError(f"Canary is effectively stationary for {solver_name}")
    post_cfl = np.asarray(diagnostics.get("post_step_cfl", []), dtype=float)
    finite_flags = np.asarray(diagnostics.get("finite_state_flag", []), dtype=bool)
    if post_cfl.size == 0 or not np.isfinite(post_cfl).all():
        raise ValidationError(f"Missing/non-finite CFL health for {solver_name}")
    if finite_flags.size == 0 or not bool(finite_flags.all()):
        raise ValidationError(f"Non-finite natural state in {solver_name} canary")
    result = {
        "solver": solver_name,
        "solver_shape": list(map(int, prepared["solver_bathymetry"].shape)),
        "publication_shape": [64, 64],
        "requested_time_count": int(expected_times.size),
        "requested_horizon": float(expected_times[-1]),
        "natural_steps": int(np.asarray(diagnostics["total_natural_steps"]).reshape(-1)[0]),
        "max_post_step_cfl": float(np.max(post_cfl)),
        "target_cfl": float(solver_cfg.get("cfl", 0.45)),
        "finite_state": True,
        "deterministic_replay_max_abs": replay_max_abs,
        "deterministic_replay_relative_l2": replay_relative_l2,
        "first_to_final_relative_change": temporal_change,
        "well_balanced": _run_well_balanced_check(
            solver_name, solver_cfg, prepared
        ),
        "conservation": _run_conservation_check(
            solver_name, solver_cfg, prepared
        ),
    }
    if solver_name == "boussinesq":
        failed = np.asarray(diagnostics.get("cg_failed_count", []), dtype=int)
        converged = np.asarray(diagnostics.get("cg_step_converged", []), dtype=bool)
        if failed.size == 0 or int(np.sum(failed)) != 0 or not bool(converged.all()):
            raise ValidationError("Boussinesq canary has a CG convergence failure")
        result["cg_failed_steps"] = int(np.sum(failed))
        result["cg_converged_fraction"] = float(np.mean(converged))
    return result


def _run_canaries(
    contract: Mapping[str, Any], expected_times: np.ndarray, canary_count: int
) -> dict[str, Any]:
    test_cfg_path = _repo_path(
        str(contract["main_datasets"]["splits"]["test"]["generation_config"])
    )
    cfg = yaml.safe_load(test_cfg_path.read_text(encoding="utf-8"))
    records = _read_jsonl(_repo_path("data/test/synthetic/scenario_manifest.jsonl"))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("bathymetry_type")), str(record.get("source_type")))
        if key in seen:
            continue
        selected.append(record)
        seen.add(key)
        if len(selected) >= canary_count:
            break
    if len(selected) < canary_count:
        selected_ids = {str(record["scenario_id"]) for record in selected}
        selected.extend(
            record
            for record in records
            if str(record["scenario_id"]) not in selected_ids
        )
        selected = selected[:canary_count]
    if len(selected) < canary_count:
        raise ValidationError("Not enough deterministic production canaries")
    outputs: list[dict[str, Any]] = []
    for record in selected:
        sample_index = int(record["sample_index"])
        with np.load(_repo_path(record["bathymetry_cache_path"]), allow_pickle=False) as b:
            master_bathymetry = np.asarray(b["master_bathymetry"], dtype=np.float32)
            bathymetry = np.asarray(b["solver_bathymetry"], dtype=np.float32)
        with np.load(_repo_path(record["source_cache_path"]), allow_pickle=False) as s:
            master_source = np.asarray(s["master_source_field"], dtype=np.float32)
            source = np.asarray(s["solver_source_field"], dtype=np.float32)
            source_strength = float(
                np.asarray(s["source_strength"]).reshape(-1)[0]
            )
        if master_bathymetry.shape != (384, 384) or bathymetry.shape != (128, 128):
            raise ValidationError(
                f"Canary input shapes do not follow 384 -> 128: {record['scenario_id']}"
            )
        if master_source.shape != (384, 384) or source.shape != (128, 128):
            raise ValidationError(
                f"Canary source shapes do not follow 384 -> 128: {record['scenario_id']}"
            )
        if not np.array_equal(bathymetry, _block_mean_downsample(master_bathymetry, (128, 128))):
            raise ValidationError(f"Canary bathymetry reduction mismatch: {record['scenario_id']}")
        if not np.array_equal(source, _block_mean_downsample(master_source, (128, 128))):
            raise ValidationError(f"Canary source reduction mismatch: {record['scenario_id']}")
        sample_result: dict[str, Any] = {
            "scenario_id": str(record["scenario_id"]),
            "sample_index": sample_index,
            "bathymetry_type": str(record.get("bathymetry_type")),
            "source_type": str(record.get("source_type")),
            "solvers": [],
        }
        for solver_name in SOLVERS:
            print(
                f"[production-validation] canary={record['scenario_id']} "
                f"solver={solver_name}",
                flush=True,
            )
            solver_cfg = dict(cfg["solver"])
            solver_cfg.update(dict(cfg["solver_profiles"][solver_name]))
            saved_path = (
                _repo_path("data/test/raw")
                / SOLVER_DIRS[solver_name]
                / "samples"
                / f"sample_{sample_index:06d}"
                / "sample.npz"
            )
            with np.load(saved_path, allow_pickle=False) as saved:
                expected_trajectory = np.asarray(
                    saved["trajectory_eta"], dtype=np.float32
                )
            sample_result["solvers"].append(
                _run_canary_solver(
                    solver_name,
                    solver_cfg,
                    bathymetry,
                    source,
                    source_strength,
                    expected_times,
                    expected_trajectory,
                    max_natural_steps=int(cfg["requested_output"]["max_natural_steps"]),
                )
            )
        outputs.append(sample_result)
    return {"canary_count": len(outputs), "results": outputs}


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_validation(
    *,
    contract_path: Path,
    output_root: Path,
    canary_count: int,
    deep_payload_audit: bool,
) -> Path:
    if output_root.exists():
        raise ValidationError(f"Refusing to reuse validation output: {output_root}")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping):
        raise ValidationError("Evaluation contract must be a mapping")
    expected_times = _expected_times(contract)
    print("[production-validation] validate generation contracts", flush=True)
    configs = _validate_generation_configs(contract, expected_times)
    print("[production-validation] validate frozen datasets", flush=True)
    datasets = _validate_raw_datasets(
        contract,
        expected_times=expected_times,
        deep_payload_audit=deep_payload_audit,
    )
    print("[production-validation] rerun solver canaries", flush=True)
    canaries = _run_canaries(contract, expected_times, canary_count)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "canary_results.json").write_text(
        json.dumps(canaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_id": SCHEMA_ID,
        "evaluation_type": "current_production_contract_validation",
        "status": "passed",
        "code_state": code_state(ROOT),
        "contract_hash": str(contract["scientific_scope"]["contract_hash"]),
        "requested_times": expected_times.tolist(),
        "production_lineage": {
            "master_shape": [384, 384],
            "solver_input_shape": [128, 128],
            "buffered_computation_shape": [192, 192],
            "publication_shape": [64, 64],
            "buffer_cells": 32,
            "solver_roster": list(SOLVERS),
        },
        "generation_configs": configs,
        "raw_datasets": datasets,
        "canaries": "canary_results.json",
        "canary_count": int(canaries["canary_count"]),
        "deep_payload_audit": bool(deep_payload_audit),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(output_root)
    return output_root / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/eval/final_v2_suite.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canary-count", type=int, default=3)
    parser.add_argument("--deep-payload-audit", action="store_true")
    args = parser.parse_args()
    if args.canary_count <= 0:
        parser.error("--canary-count must be positive")
    path = run_validation(
        contract_path=args.contract,
        output_root=args.output_root,
        canary_count=args.canary_count,
        deep_payload_audit=bool(args.deep_payload_audit),
    )
    print(f"[production-validation] passed: {path}")


if __name__ == "__main__":
    main()
