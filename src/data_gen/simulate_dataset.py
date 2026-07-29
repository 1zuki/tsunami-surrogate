from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.data_gen.common_time_v2 import (
        ETA_SAMPLE_SCHEMA_ID,
        PUBLICATION_SCHEMA_ID,
        RequestedOutputConfig,
        atomic_replace_directory,
        authoritative_input_fingerprint,
        code_state,
        hash_array,
        parse_requested_output_config,
        resolved_config_hash,
        sha256_file,
        split_qualified_identity,
        stable_hash_payload,
        validate_publication,
    )
    from src.data_gen.operational_timing import GenerationTimingRecorder
except ImportError:
    from common_time_v2 import (
        ETA_SAMPLE_SCHEMA_ID,
        PUBLICATION_SCHEMA_ID,
        RequestedOutputConfig,
        atomic_replace_directory,
        authoritative_input_fingerprint,
        code_state,
        hash_array,
        parse_requested_output_config,
        resolved_config_hash,
        sha256_file,
        split_qualified_identity,
        stable_hash_payload,
        validate_publication,
    )
    from operational_timing import GenerationTimingRecorder

try:
    from src.data_gen.generate_bathymetry import BathymetryGenerator
    from src.data_gen.generate_sources import SourceGenerator
except ImportError:
    from generate_bathymetry import BathymetryGenerator
    from generate_sources import SourceGenerator

try:
    from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver
except ImportError:
    from hydrostatic_swe import HydrostaticShallowWaterSolver

try:
    from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver
except ImportError:
    from muscl_hr_swe import MUSCLHRShallowWaterSolver

try:
    from src.solver.boussinesq import BoussinesqSolver
except ImportError:
    from boussinesq import BoussinesqSolver

FDE_ALIASES = {
    "swe_muscl": "swe_muscl_hr",  # backward-compatible
}
KNOWN_FDES = {"swe_hydrostatic", "swe_muscl_hr", "boussinesq", *FDE_ALIASES.keys()}
IMPLEMENTED_FDES = {"swe_hydrostatic", "swe_muscl_hr", "boussinesq"}
NATIVE_INPUT_SCHEMA_ID = "tsunami-surrogate.native-resolution-inputs.v1"
FDE_OUTPUT_DIRNAME = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
}

COMMON_SOLVER_KEYS = {
    "nx",
    "ny",
    "dx",
    "dy",
    "dt",
    "g",
    "cfl",
    "boundary",
    "use_sponge",
    "sponge_width",
    "sponge_min_factor",
    "sponge_time_mode",
    "sponge_reference_dt",
    "sponge_axes",
    "sponge_profile",
}
SWE_SOLVER_KEYS = COMMON_SOLVER_KEYS | {"dry_tolerance", "max_velocity"}
BOUSSINESQ_SOLVER_KEYS = COMMON_SOLVER_KEYS | {
    "alpha",
    "min_depth",
    "sea_level_offset",
    "depth_scale",
    "mode",
    "filter_strength",
    "linear_solver_tol",
    "linear_solver_abs_tol",
    "linear_solver_max_iter",
    "linear_solver_preconditioner",
    "check_finite",
    "filter_time_mode",
    "filter_reference_dt",
    "cg_failure_mode",
}


@dataclass
class QualityPolicy:
    on_violation: str
    reject_nonfinite: bool
    min_h_tolerance: float | None
    max_abs_eta_limit: float | None
    max_velocity_limit: float | None
    max_eta_over_depth: float | None
    require_cg_converged: bool


@dataclass(frozen=True)
class BufferedDomainConfig:
    """Opt-in computational padding around an unchanged publication crop."""

    enabled: bool = False
    buffer_cells: int = 0
    source_taper_cells: int = 0
    bathymetry_extension: str = "edge"
    output_crop: str = "central"

    def semantics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "buffer_cells": self.buffer_cells,
            "source_taper_cells": self.source_taper_cells,
            "bathymetry_extension": self.bathymetry_extension,
            "output_crop": self.output_crop,
        }


@dataclass(frozen=True)
class OperationalConfig:
    """Nondeterministic execution metadata kept outside scientific samples."""

    enabled: bool = True
    progress_every: int = 1
    solver_progress: bool = True
    max_in_flight: int | None = None
    cloud_provider: str | None = None
    cloud_zone: str | None = None
    machine_type: str | None = None
    storage_class: str | None = None
    hourly_cost_usd: float | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "solver_progress": self.solver_progress,
            "cloud_provider": self.cloud_provider,
            "cloud_zone": self.cloud_zone,
            "machine_type": self.machine_type,
            "storage_class": self.storage_class,
            "hourly_cost_usd": self.hourly_cost_usd,
        }


@dataclass(frozen=True)
class AuthoritativeInputsConfig:
    inventory_path: Path
    inventory_sha256: str
    h0_contract_hash: str
    require_exact_arrays: bool = True
    allow_input_generation: bool = False

    def semantics(self) -> dict[str, Any]:
        return {
            "inventory_sha256": self.inventory_sha256,
            "h0_contract_hash": self.h0_contract_hash,
            "require_exact_arrays": self.require_exact_arrays,
            "allow_input_generation": self.allow_input_generation,
        }


@dataclass(frozen=True)
class PairedInputsConfig:
    """Deterministic master-grid construction for paired native resolutions."""

    enabled: bool = False
    lineage_id: str = ""
    master_shape: tuple[int, int] = (0, 0)
    target_shape: tuple[int, int] = (0, 0)
    downsample_method: str = "block_mean_float64_v1"
    master_bathymetry_config: Path | None = None
    master_source_config: Path | None = None
    inventory_path: Path | None = None
    lineage_hash: str = ""
    target_contract_hash: str = ""

    def semantics(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "schema_id": NATIVE_INPUT_SCHEMA_ID,
            "lineage_id": self.lineage_id,
            "master_shape": list(self.master_shape),
            "target_shape": list(self.target_shape),
            "downsample_method": self.downsample_method,
            "lineage_hash": self.lineage_hash,
            "target_contract_hash": self.target_contract_hash,
        }


@dataclass
class DatasetConfig:
    """Convenience wrapper for the top-level dataset config."""

    num_samples: int
    seed: int | None
    num_workers: int
    n_steps: int
    save_every: int
    auto_dt: bool
    target_cfl: float
    include_initial_state: bool
    sea_level_offset: float
    source_strength_range: Tuple[float, float]
    output_dir: Path
    bathymetry_dir: Path
    source_dir: Path
    manifest_path: Path
    copy_configs: bool
    enabled_fdes: tuple[str, ...]
    primary_fde: str
    quality_policy: QualityPolicy
    requested_output: RequestedOutputConfig | None
    authoritative_inputs: AuthoritativeInputsConfig | None = None
    solver_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    buffered_domain: BufferedDomainConfig = field(default_factory=BufferedDomainConfig)
    paired_inputs: PairedInputsConfig = field(default_factory=PairedInputsConfig)
    paired_input_inventory_sha256: str | None = None


@dataclass
class RolloutResult:
    trajectory: np.ndarray
    trajectory_eta: np.ndarray
    timestamps: np.ndarray
    dt_history: np.ndarray
    diagnostics: Dict[str, np.ndarray] | None = None


def _block_mean_downsample(
    values: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Area-average an integer-ratio master grid with float64 accumulation."""

    master = np.asarray(values, dtype=np.float32)
    if master.ndim != 2:
        raise ValueError("paired native inputs must be two-dimensional")
    tx, ty = map(int, target_shape)
    mx, my = master.shape
    if tx <= 0 or ty <= 0 or mx % tx != 0 or my % ty != 0:
        raise ValueError(
            f"master shape {master.shape} must be integer-divisible by target "
            f"shape {target_shape}"
        )
    if master.shape == (tx, ty):
        return master.copy()
    fx, fy = mx // tx, my // ty
    reduced = master.reshape(tx, fx, ty, fy).mean(axis=(1, 3), dtype=np.float64)
    return np.asarray(reduced, dtype=np.float32)


def _cosine_source_window(shape: tuple[int, int], taper_cells: int) -> np.ndarray:
    if min(shape) <= 1:
        raise ValueError("source-window shape must contain two valid axes")
    if taper_cells < 2 or 2 * taper_cells >= min(shape):
        raise ValueError(
            "source_taper_cells must be at least 2 and leave an untapered interior"
        )
    nx, ny = shape
    x_distance = np.minimum(np.arange(nx), np.arange(nx)[::-1])
    y_distance = np.minimum(np.arange(ny), np.arange(ny)[::-1])
    edge_distance = np.minimum(x_distance[:, None], y_distance[None, :])
    coordinate = np.clip(
        edge_distance.astype(np.float64) / float(taper_cells - 1), 0.0, 1.0
    )
    return 0.5 * (1.0 - np.cos(np.pi * coordinate))


def _prepare_buffered_domain(
    bathymetry: np.ndarray,
    source_field: np.ndarray,
    source_strength: float,
    sea_level_offset: float,
    config: BufferedDomainConfig,
) -> dict[str, Any]:
    """Build solver-sized arrays while retaining a source-consistent core."""

    bathy = np.asarray(bathymetry, dtype=np.float32)
    source = np.asarray(source_field, dtype=np.float32)
    if bathy.ndim != 2 or source.shape != bathy.shape:
        raise ValueError("bathymetry and source_field must be same-shape 2-D arrays")
    if not config.enabled:
        raise ValueError("buffered-domain preparation requires enabled=true")

    window = _cosine_source_window(bathy.shape, config.source_taper_cells)
    effective_source = np.asarray(
        np.asarray(source, dtype=np.float64) * window, dtype=np.float32
    )
    eta0 = np.asarray(float(source_strength) * effective_source, dtype=np.float32)
    rest_depth = np.maximum(-bathy + sea_level_offset, 0.0)
    initial_depth = np.maximum(rest_depth + eta0, 0.0)
    free_surface0 = initial_depth + bathy

    width = config.buffer_cells
    pad = ((width, width), (width, width))
    solver_bathymetry = np.pad(
        np.asarray(bathy, dtype=np.float64), pad, mode=config.bathymetry_extension
    )
    solver_eta0 = np.zeros_like(solver_bathymetry)
    solver_rest_depth = np.maximum(
        -solver_bathymetry + sea_level_offset, 0.0
    )
    solver_h0 = solver_rest_depth.copy()
    crop = (
        slice(width, width + bathy.shape[0]),
        slice(width, width + bathy.shape[1]),
    )
    solver_eta0[crop] = np.asarray(eta0, dtype=np.float64)
    solver_h0[crop] = np.asarray(initial_depth, dtype=np.float64)

    if not np.array_equal(solver_bathymetry[crop], bathy):
        raise RuntimeError("buffer construction changed core bathymetry")
    edge_max = max(
        float(np.max(np.abs(effective_source[[0, -1], :]))),
        float(np.max(np.abs(effective_source[:, [0, -1]]))),
    )
    if edge_max != 0.0:
        raise RuntimeError("source taper did not produce an exact-zero crop edge")

    return {
        "bathymetry": bathy,
        "source_field": effective_source,
        "eta0": eta0,
        "rest_depth": rest_depth,
        "h0": initial_depth,
        "free_surface0": free_surface0,
        "solver_bathymetry": solver_bathymetry,
        "solver_eta0": solver_eta0,
        "solver_rest_depth": solver_rest_depth,
        "solver_h0": solver_h0,
        "crop": crop,
        "source_edge_max_abs": edge_max,
    }


def _crop_rollout(rollout: RolloutResult, crop: tuple[slice, slice]) -> RolloutResult:
    trajectory = np.asarray(rollout.trajectory)
    trajectory_eta = np.asarray(rollout.trajectory_eta)
    if trajectory.ndim != 4 or trajectory_eta.ndim != 3:
        raise ValueError("rollout arrays do not have the expected spatial dimensions")
    return RolloutResult(
        trajectory=trajectory[..., crop[0], crop[1]].copy(),
        trajectory_eta=trajectory_eta[..., crop[0], crop[1]].copy(),
        timestamps=rollout.timestamps,
        dt_history=rollout.dt_history,
        diagnostics=rollout.diagnostics,
    )


def _sample_output_complete(sample_dir: Path, *, requested: RequestedOutputConfig | None = None) -> bool:
    if requested is not None:
        try:
            validate_publication(
                sample_dir,
                expected_contract_hash=requested.contract_hash,
                expected_times=requested.requested_times,
            )
        except Exception:
            return False
        return True

    required_files = ("sample.npz", "rollout.npz", "trajectory_eta.npy", "meta.json")
    if not sample_dir.is_dir():
        return False
    for name in required_files:
        if not (sample_dir / name).is_file():
            return False

    return True


def _load_existing_solver_record(
    sample_dir: Path, sample_idx: int, fallback_solver_name: str
) -> Dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "sample_index": sample_idx,
        "scenario_id": str(meta.get("scenario_id", f"scenario_{sample_idx:06d}")),
        "sample_dir": str(sample_dir),
        "solver_name": str(meta.get("solver_name", fallback_solver_name)),
        "bathymetry_type": str(meta.get("bathymetry_type", "unknown")),
        "source_type": str(meta.get("source_type", "unknown")),
        "source_strength": float(meta.get("source_strength", np.nan)),
        "num_frames": int(meta.get("num_frames", 0)),
        "trajectory_shape": meta.get("trajectory_shape", []),
        "trajectory_eta_shape": meta.get("trajectory_eta_shape", []),
        "fdes_run": list(meta.get("fdes_run", [fallback_solver_name])),
        "fdes_skipped_unimplemented": list(meta.get("fdes_skipped_unimplemented", [])),
        "nan_count": int(meta.get("nan_count", 0)),
        "inf_count": int(meta.get("inf_count", 0)),
        "min_h": float(meta.get("min_h", np.nan)),
        "max_abs_eta": float(meta.get("max_abs_eta", np.nan)),
        "max_abs_velocity": float(meta.get("max_abs_velocity", np.nan)),
        "max_abs_eta_over_depth": float(meta.get("max_abs_eta_over_depth", np.nan)),
        "cg_failed_count": meta.get("cg_failed_count", None),
        "cg_converged_fraction": float(meta.get("cg_converged_fraction", np.nan)),
        "max_cg_iterations": meta.get("max_cg_iterations", None),
        "max_cg_residual_ratio": float(meta.get("max_cg_residual_ratio", np.nan)),
        "dt_min": float(meta.get("dt_min", 0.0)),
        "dt_max": float(meta.get("dt_max", 0.0)),
        "quality_status": str(meta.get("quality_status", "unknown")),
        "input_fingerprint": meta.get("input_fingerprint"),
        "resolved_config_hash": meta.get("resolved_config_hash"),
        "code_state_hash": meta.get("code_state_hash"),
        "reused_existing": True,
    }


def _compute_rollout_health(
    fde_name: str,
    rollout: RolloutResult,
    rest_depth: np.ndarray,
    effective_depth: np.ndarray | None = None,
) -> Dict[str, Any]:
    state_stack = np.asarray(rollout.trajectory, dtype=np.float32)
    eta_stack = np.asarray(rollout.trajectory_eta, dtype=np.float32)
    nan_count = int(np.isnan(state_stack).sum() + np.isnan(eta_stack).sum())
    inf_count = int(np.isinf(state_stack).sum() + np.isinf(eta_stack).sum())
    max_abs_eta_over_depth = float("nan")

    if fde_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        h_hist = np.asarray(state_stack[:, 0], dtype=np.float32)
        hu_hist = np.asarray(state_stack[:, 1], dtype=np.float32)
        hv_hist = np.asarray(state_stack[:, 2], dtype=np.float32)
        h_safe = np.maximum(h_hist, 1e-8)
        wet = h_hist > 1e-8
        u_hist = np.zeros_like(h_hist, dtype=np.float32)
        v_hist = np.zeros_like(h_hist, dtype=np.float32)
        u_hist[wet] = hu_hist[wet] / h_safe[wet]
        v_hist[wet] = hv_hist[wet] / h_safe[wet]
        max_abs_velocity = float(max(np.max(np.abs(u_hist)), np.max(np.abs(v_hist))))
        min_h = float(np.min(h_hist))
    else:
        depth_ref = rest_depth if effective_depth is None else effective_depth
        depth_ref = np.asarray(depth_ref, dtype=np.float32)
        h_hist = eta_stack + depth_ref[None, ...]
        min_h = float(np.min(h_hist))
        max_abs_velocity = float(np.nan)
        eta_over_depth = np.abs(eta_stack) / np.maximum(depth_ref[None, ...], 1e-8)
        max_abs_eta_over_depth = float(np.nanmax(eta_over_depth))

    dt_positive = rollout.dt_history[rollout.dt_history > 0.0]
    dt_min = float(np.min(dt_positive)) if dt_positive.size > 0 else 0.0
    dt_max = float(np.max(dt_positive)) if dt_positive.size > 0 else 0.0
    max_abs_eta = float(np.max(np.abs(eta_stack)))

    diagnostics = rollout.diagnostics or {}
    cg_failed = np.asarray(diagnostics.get("cg_failed_count", []), dtype=np.int32)
    cg_iterations = np.asarray(diagnostics.get("cg_max_iterations", []), dtype=np.int32)
    cg_residual_ratio = np.asarray(
        diagnostics.get("cg_max_residual_ratio", []), dtype=np.float32
    )
    has_cg_diagnostics = bool(cg_failed.size > 0)
    cg_failed_count = int(np.sum(cg_failed)) if has_cg_diagnostics else None
    cg_converged_fraction = (
        float(np.mean(cg_failed == 0)) if has_cg_diagnostics else float("nan")
    )
    max_cg_iterations = int(np.max(cg_iterations)) if cg_iterations.size > 0 else None
    max_cg_residual_ratio = (
        float(np.nanmax(cg_residual_ratio))
        if cg_residual_ratio.size > 0
        else float("nan")
    )

    return {
        "fde_name": fde_name,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min_h": min_h,
        "max_abs_eta": max_abs_eta,
        "max_abs_velocity": max_abs_velocity,
        "max_abs_eta_over_depth": max_abs_eta_over_depth,
        "has_cg_diagnostics": has_cg_diagnostics,
        "cg_failed_count": cg_failed_count,
        "cg_converged_fraction": cg_converged_fraction,
        "max_cg_iterations": max_cg_iterations,
        "max_cg_residual_ratio": max_cg_residual_ratio,
        "dt_min": dt_min,
        "dt_max": dt_max,
    }


def _quality_violations_for_health(
    health: Dict[str, Any], policy: QualityPolicy
) -> list[str]:
    violations: list[str] = []
    nan_count = int(health.get("nan_count", 0))
    inf_count = int(health.get("inf_count", 0))
    min_h = float(health.get("min_h", np.nan))
    max_abs_eta = float(health.get("max_abs_eta", np.nan))
    max_abs_velocity = float(health.get("max_abs_velocity", np.nan))
    max_abs_eta_over_depth = float(health.get("max_abs_eta_over_depth", np.nan))

    if policy.reject_nonfinite and (nan_count > 0 or inf_count > 0):
        violations.append(f"nonfinite(nan_count={nan_count}, inf_count={inf_count})")

    if (
        policy.min_h_tolerance is not None
        and np.isfinite(min_h)
        and min_h < float(policy.min_h_tolerance)
    ):
        violations.append(
            f"min_h({min_h:.6g}) < min_h_tolerance({float(policy.min_h_tolerance):.6g})"
        )

    if (
        policy.max_abs_eta_limit is not None
        and np.isfinite(max_abs_eta)
        and max_abs_eta > float(policy.max_abs_eta_limit)
    ):
        violations.append(
            f"max_abs_eta({max_abs_eta:.6g}) > max_abs_eta_limit({float(policy.max_abs_eta_limit):.6g})"
        )

    if (
        policy.max_velocity_limit is not None
        and np.isfinite(max_abs_velocity)
        and max_abs_velocity > float(policy.max_velocity_limit)
    ):
        violations.append(
            "max_abs_velocity"
            f"({max_abs_velocity:.6g}) > max_velocity_limit({float(policy.max_velocity_limit):.6g})"
        )

    if (
        policy.max_eta_over_depth is not None
        and np.isfinite(max_abs_eta_over_depth)
        and max_abs_eta_over_depth > float(policy.max_eta_over_depth)
    ):
        violations.append(
            "max_abs_eta_over_depth"
            f"({max_abs_eta_over_depth:.6g}) > max_eta_over_depth({float(policy.max_eta_over_depth):.6g})"
        )

    if (
        policy.require_cg_converged
        and str(health.get("fde_name", "")).strip().lower() == "boussinesq"
    ):
        if not bool(health.get("has_cg_diagnostics", False)):
            violations.append("cg_convergence_diagnostics_missing")
        else:
            cg_failed_count = int(health.get("cg_failed_count", 0) or 0)
            if cg_failed_count > 0:
                violations.append(f"cg_failed_count({cg_failed_count}) > 0")

    return violations


def _fde_dirname(fde_name: str) -> str:
    if fde_name in FDE_OUTPUT_DIRNAME:
        return FDE_OUTPUT_DIRNAME[fde_name]
    return str(fde_name).replace("swe_", "")


def _canonical_fde_name(name: str) -> str:
    raw = str(name).strip()
    return FDE_ALIASES.get(raw, raw)


def _seed_for_sample(run_seed: int, sample_idx: int) -> int:
    # sample_idx is 1-based; keep derivation stable across workers/runs.
    return int(run_seed + sample_idx * 10007)


def _bathymetry_file_path(bathymetry_dir: str | Path, sample_idx: int) -> Path:
    return Path(bathymetry_dir) / f"sample_{sample_idx:06d}.npz"


def _source_file_path(source_dir: str | Path, sample_idx: int) -> Path:
    return Path(source_dir) / f"sample_{sample_idx:06d}.npz"


def _resolved_solver_cfg_for_fde(
    base: Dict[str, Any], profiles: Mapping[str, Mapping[str, Any]], fde_name: str
) -> Dict[str, Any]:
    resolved = dict(base)
    profile = profiles.get(fde_name, {})
    if not isinstance(profile, Mapping):
        raise ValueError(f"solver_profiles.{fde_name} must be a mapping")
    resolved.update(dict(profile))
    return resolved


def _filter_solver_cfg(sv: Dict[str, Any], allowed: set[str]) -> Dict[str, Any]:
    return {key: sv[key] for key in allowed if key in sv}


def _make_hydrostatic_solver_from_cfg(
    sv: Dict[str, Any],
) -> HydrostaticShallowWaterSolver:
    cfg = _filter_solver_cfg(sv, SWE_SOLVER_KEYS)
    boundary = cfg.get("boundary", "open")
    return HydrostaticShallowWaterSolver(
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        dx=float(cfg["dx"]),
        dy=float(cfg["dy"]),
        dt=float(cfg["dt"]),
        g=float(cfg.get("g", 9.81)),
        cfl=float(cfg.get("cfl", 0.45)),
        dry_tolerance=float(cfg.get("dry_tolerance", 1e-6)),
        boundary=boundary,
        use_sponge=bool(cfg.get("use_sponge", True)),
        sponge_width=int(cfg.get("sponge_width", 20)),
        sponge_min_factor=float(cfg.get("sponge_min_factor", 0.9)),
        max_velocity=float(cfg.get("max_velocity", 50.0)),
        sponge_time_mode=str(cfg.get("sponge_time_mode", "legacy_per_step")),
        sponge_reference_dt=cfg.get("sponge_reference_dt", None),
        sponge_axes=str(cfg.get("sponge_axes", "xy")),
        sponge_profile=str(cfg.get("sponge_profile", "quadratic")),
    )


def _make_muscl_solver_from_cfg(sv: Dict[str, Any]) -> MUSCLHRShallowWaterSolver:
    cfg = _filter_solver_cfg(sv, SWE_SOLVER_KEYS)
    boundary = cfg.get("boundary", "open")
    return MUSCLHRShallowWaterSolver(
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        dx=float(cfg["dx"]),
        dy=float(cfg["dy"]),
        dt=float(cfg["dt"]),
        g=float(cfg.get("g", 9.81)),
        cfl=float(cfg.get("cfl", 0.45)),
        dry_tolerance=float(cfg.get("dry_tolerance", 1e-6)),
        boundary=boundary,
        use_sponge=bool(cfg.get("use_sponge", True)),
        sponge_width=int(cfg.get("sponge_width", 20)),
        sponge_min_factor=float(cfg.get("sponge_min_factor", 0.9)),
        max_velocity=float(cfg.get("max_velocity", 50.0)),
        sponge_time_mode=str(cfg.get("sponge_time_mode", "legacy_per_step")),
        sponge_reference_dt=cfg.get("sponge_reference_dt", None),
        sponge_axes=str(cfg.get("sponge_axes", "xy")),
        sponge_profile=str(cfg.get("sponge_profile", "quadratic")),
    )


def _make_boussinesq_solver_from_cfg(sv: Dict[str, Any]) -> BoussinesqSolver:
    cfg = _filter_solver_cfg(sv, BOUSSINESQ_SOLVER_KEYS)
    return BoussinesqSolver(
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        dx=float(cfg["dx"]),
        dy=float(cfg["dy"]),
        dt=float(cfg["dt"]),
        g=float(cfg.get("g", 9.81)),
        cfl=float(cfg.get("cfl", 0.35)),
        alpha=float(cfg.get("alpha", 1.0 / 3.0)),
        min_depth=float(cfg.get("min_depth", 1e-3)),
        sea_level_offset=float(cfg.get("sea_level_offset", 0.0)),
        depth_scale=float(cfg.get("depth_scale", 1.0)),
        boundary=cfg.get("boundary", "open"),
        mode=cfg.get("mode", "linear_variable_depth"),
        use_sponge=cfg["use_sponge"] if "use_sponge" in cfg else None,
        sponge_width=int(cfg.get("sponge_width", 20)),
        sponge_min_factor=float(cfg.get("sponge_min_factor", 0.9)),
        filter_strength=float(cfg.get("filter_strength", 0.0)),
        linear_solver_tol=float(cfg.get("linear_solver_tol", 1e-8)),
        linear_solver_abs_tol=float(cfg.get("linear_solver_abs_tol", 0.0)),
        linear_solver_max_iter=int(cfg.get("linear_solver_max_iter", 80)),
        linear_solver_preconditioner=str(
            cfg.get("linear_solver_preconditioner", "jacobi")
        ),
        check_finite=bool(cfg.get("check_finite", True)),
        sponge_time_mode=str(cfg.get("sponge_time_mode", "legacy_per_step")),
        sponge_reference_dt=cfg.get("sponge_reference_dt", None),
        filter_time_mode=str(cfg.get("filter_time_mode", "legacy_per_step")),
        filter_reference_dt=cfg.get("filter_reference_dt", None),
        cg_failure_mode=str(cfg.get("cg_failure_mode", "legacy_posthoc")),
        sponge_axes=str(cfg.get("sponge_axes", "xy")),
        sponge_profile=str(cfg.get("sponge_profile", "quadratic")),
    )


def _require_solver_cfl_diagnostics(solver: Any) -> None:
    if not hasattr(solver, "compute_cfl"):
        raise RuntimeError(
            f"Dense diagnostics require compute_cfl(dt=...) support, missing on {type(solver).__name__}"
        )


def _is_swe_solver(solver: Any) -> bool:
    return all(
        hasattr(solver, name)
        for name in ("h", "hu", "hv", "dry_tolerance", "compute_velocity")
    )


def _is_boussinesq_solver(solver: Any) -> bool:
    return hasattr(solver, "eta") and hasattr(solver, "eta_t")


def _append_dense_value(buffers: Dict[str, list[Any]], key: str, value: Any) -> None:
    buffers.setdefault(key, []).append(value)


def _collect_swe_dense_health(solver: Any) -> dict[str, Any]:
    h = np.asarray(solver.h, dtype=np.float64)
    u, v = solver.compute_velocity()
    speed = np.sqrt(
        np.asarray(u, dtype=np.float64) ** 2 + np.asarray(v, dtype=np.float64) ** 2
    )
    return {
        "swe_min_depth": float(np.min(h)),
        "swe_max_speed": float(np.max(speed)),
        "swe_dry_cell_count": int(np.count_nonzero(h <= float(solver.dry_tolerance))),
    }


def _collect_boussinesq_dense_health(solver: Any) -> dict[str, Any]:
    required = (
        "last_step_cg_solve_converged",
        "last_step_cg_solve_iterations",
        "last_step_cg_solve_initial_residual",
        "last_step_cg_solve_final_residual",
        "last_step_cg_solve_residual_ratio",
    )
    if any(not hasattr(solver, name) for name in required):
        missing = [name for name in required if not hasattr(solver, name)]
        raise RuntimeError(
            "Dense diagnostics require separate Boussinesq acceleration-solve records; "
            f"missing {missing} on {type(solver).__name__}"
        )

    converged = tuple(
        bool(value) for value in getattr(solver, "last_step_cg_solve_converged")
    )
    iterations = tuple(
        int(value) for value in getattr(solver, "last_step_cg_solve_iterations")
    )
    initial_residual = tuple(
        float(value) for value in getattr(solver, "last_step_cg_solve_initial_residual")
    )
    final_residual = tuple(
        float(value) for value in getattr(solver, "last_step_cg_solve_final_residual")
    )
    residual_ratio = tuple(
        float(value) for value in getattr(solver, "last_step_cg_solve_residual_ratio")
    )
    if not (
        len(converged)
        == len(iterations)
        == len(initial_residual)
        == len(final_residual)
        == len(residual_ratio)
        == 2
    ):
        raise RuntimeError(
            "Dense diagnostics require exactly two Boussinesq acceleration-solve records per step"
        )

    return {
        "cg_solve0_converged": converged[0],
        "cg_solve0_iterations": iterations[0],
        "cg_solve0_initial_residual": initial_residual[0],
        "cg_solve0_final_residual": final_residual[0],
        "cg_solve0_residual_ratio": residual_ratio[0],
        "cg_solve1_converged": converged[1],
        "cg_solve1_iterations": iterations[1],
        "cg_solve1_initial_residual": initial_residual[1],
        "cg_solve1_final_residual": final_residual[1],
        "cg_solve1_residual_ratio": residual_ratio[1],
    }


def _finalize_dense_diagnostics(
    buffers: Dict[str, list[Any]], *, float_dtype: np.dtype[Any] = np.float32
) -> Dict[str, np.ndarray]:
    bool_keys = {
        "finite_state_flag",
        "cg_step_converged",
        "cg_solve0_converged",
        "cg_solve1_converged",
        "filter_enabled",
    }
    int_keys = {
        "cg_failed_count",
        "cg_max_iterations",
        "swe_dry_cell_count",
        "cg_solve0_iterations",
        "cg_solve1_iterations",
        "natural_health_step_indices",
        "filter_application_count",
        "operator_linear_solver_factorization_count",
        "operator_linear_solver_factorization_nnz",
    }
    diagnostics: Dict[str, np.ndarray] = {}
    for key, values in buffers.items():
        if key in bool_keys:
            diagnostics[key] = np.asarray(values, dtype=np.bool_)
        elif key in int_keys:
            diagnostics[key] = np.asarray(values, dtype=np.int32)
        else:
            diagnostics[key] = np.asarray(values, dtype=float_dtype)
    return diagnostics


def _extract_requested_states_from_bracket(
    *,
    left_state: np.ndarray,
    right_state: np.ndarray,
    left_time: float,
    right_time: float,
    requested_times: np.ndarray,
    right_natural_step_index: int,
    output_dtype: Any = np.float32,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    left = np.asarray(left_state)
    right = np.asarray(right_state)
    if left.shape != right.shape:
        raise ValueError(
            "Left and right natural states must have identical shapes: "
            f"{left.shape} != {right.shape}"
        )
    if left.size == 0:
        raise ValueError("Natural states must be non-empty")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Natural states must be finite")

    left_t = float(left_time)
    right_t = float(right_time)
    if not np.isfinite(left_t) or not np.isfinite(right_t):
        raise ValueError("Natural bracket timestamps must be finite")
    if right_t <= left_t:
        raise ValueError("Natural bracket timestamps must be strictly increasing")
    if isinstance(right_natural_step_index, (bool, np.bool_)) or not isinstance(
        right_natural_step_index, (int, np.integer)
    ):
        raise TypeError("right_natural_step_index must be an integer")
    right_step = int(right_natural_step_index)
    if right_step < 1:
        raise ValueError("right_natural_step_index must be at least 1")

    queries = np.asarray(requested_times, dtype=np.float64)
    if queries.ndim != 1 or queries.size == 0:
        raise ValueError("requested_times must be a non-empty 1-D array")
    if not np.isfinite(queries).all():
        raise ValueError("requested_times must be finite")
    if np.any(np.diff(queries) <= 0.0):
        raise ValueError("requested_times must be strictly increasing")
    if np.any(queries < left_t) or np.any(queries > right_t):
        raise ValueError(
            "requested_times extend beyond the adjacent natural-state bracket"
        )

    dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("requested-state output dtype must be float32 or float64")

    output_shape = (int(queries.size),) + left.shape
    extracted = np.empty(output_shape, dtype=dtype)
    left_times = np.full(queries.shape, left_t, dtype=np.float64)
    right_times = np.full(queries.shape, right_t, dtype=np.float64)
    widths = np.full(queries.shape, right_t - left_t, dtype=np.float64)
    weights = (queries - left_t) / (right_t - left_t)
    exact_knot = np.zeros(queries.shape, dtype=np.bool_)
    step_indices = np.full(queries.shape, right_step, dtype=np.int64)

    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    for query_idx, query in enumerate(queries):
        if query == left_t:
            extracted[query_idx] = np.asarray(left, dtype=dtype)
            left_times[query_idx] = query
            right_times[query_idx] = query
            weights[query_idx] = 0.0
            widths[query_idx] = 0.0
            exact_knot[query_idx] = True
            step_indices[query_idx] = right_step - 1
        elif query == right_t:
            extracted[query_idx] = np.asarray(right, dtype=dtype)
            left_times[query_idx] = query
            right_times[query_idx] = query
            weights[query_idx] = 0.0
            widths[query_idx] = 0.0
            exact_knot[query_idx] = True
        else:
            weight = float(weights[query_idx])
            extracted[query_idx] = np.asarray(
                left64 * (1.0 - weight) + right64 * weight,
                dtype=dtype,
            )

    provenance = {
        "requested_timestamps": queries.copy(),
        "left_natural_timestamps": left_times,
        "right_natural_timestamps": right_times,
        "interpolation_weights": np.asarray(weights, dtype=np.float64),
        "bracket_widths": widths,
        "exact_knot": exact_knot,
        "natural_step_indices": step_indices,
    }
    return extracted, provenance


def _simulate_requested_times_local(
    solver: Any,
    *,
    auto_dt: bool,
    target_cfl: float,
    requested_times: np.ndarray,
    max_natural_steps: int | None,
    collect_natural_step_health: bool = False,
    requested_state_dtype: Any = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    queries = np.asarray(requested_times, dtype=np.float64)
    if queries.ndim != 1 or queries.size == 0:
        raise ValueError("requested_times must be a non-empty 1-D array")
    if not np.isfinite(queries).all():
        raise ValueError("requested_times must be finite")
    if np.any(queries <= 0.0):
        raise ValueError("requested_times must be strictly positive")
    if np.any(np.diff(queries) <= 0.0):
        raise ValueError("requested_times must be strictly increasing")
    if isinstance(max_natural_steps, (bool, np.bool_)) or not isinstance(
        max_natural_steps, (int, np.integer)
    ):
        raise ValueError("max_natural_steps must be an explicit positive integer")
    step_cap = int(max_natural_steps)
    if step_cap <= 0:
        raise ValueError("max_natural_steps must be an explicit positive integer")

    left_state = np.asarray(solver.get_state()).copy()
    if left_state.size == 0 or not np.isfinite(left_state).all():
        raise ValueError("Initial solver state must be finite and non-empty")

    extracted_chunks: list[np.ndarray] = []
    provenance_chunks: Dict[str, list[np.ndarray]] = {
        "requested_timestamps": [],
        "left_natural_timestamps": [],
        "right_natural_timestamps": [],
        "interpolation_weights": [],
        "bracket_widths": [],
        "exact_knot": [],
        "natural_step_indices": [],
    }
    natural_dt_history: list[float] = []
    cg_failed_count: list[int] = []
    cg_max_iterations: list[int] = []
    cg_max_residual_ratio: list[float] = []
    natural_health_buffers: Dict[str, list[Any]] = {}
    current_time = 0.0
    next_request = 0

    for right_step in range(1, step_cap + 1):
        left_time = current_time
        if auto_dt:
            dt = solver.suggest_dt(target_cfl=target_cfl)
            solver.dt = dt
        else:
            dt = solver.dt
        natural_dt = float(dt)
        if not np.isfinite(natural_dt) or natural_dt <= 0.0:
            raise RuntimeError(
                f"Natural timestep must be finite and positive, got {natural_dt!r}"
            )
        if collect_natural_step_health:
            _require_solver_cfl_diagnostics(solver)
            _append_dense_value(
                natural_health_buffers, "natural_health_step_indices", right_step
            )
            _append_dense_value(natural_health_buffers, "left_natural_step_times", left_time)
            _append_dense_value(natural_health_buffers, "proposed_dt", natural_dt)
            _append_dense_value(
                natural_health_buffers,
                "pre_step_cfl",
                float(solver.compute_cfl(dt=natural_dt)),
            )

        solver.step(dt=dt, auto_dt=False)
        current_time = left_time + natural_dt
        if not np.isfinite(current_time) or current_time <= left_time:
            raise RuntimeError(
                "Natural solver time failed to advance strictly after "
                f"step {right_step}: left={left_time!r}, right={current_time!r}"
            )
        natural_dt_history.append(natural_dt)

        if hasattr(solver, "last_step_cg_converged"):
            cg_failed_count.append(int(getattr(solver, "last_step_cg_failed_count", 0)))
            cg_max_iterations.append(
                int(getattr(solver, "last_step_cg_max_iterations", 0))
            )
            cg_max_residual_ratio.append(
                float(getattr(solver, "last_step_cg_max_residual_ratio", 0.0))
            )

        right_state = np.asarray(solver.get_state()).copy()
        if right_state.shape != left_state.shape:
            raise RuntimeError(
                "Solver state shape changed during requested-time rollout: "
                f"{left_state.shape} != {right_state.shape}"
            )
        if not np.isfinite(right_state).all():
            raise RuntimeError(
                f"Solver produced a non-finite state at natural step {right_step}"
            )
        if collect_natural_step_health:
            _append_dense_value(
                natural_health_buffers,
                "right_natural_step_times",
                current_time,
            )
            _append_dense_value(
                natural_health_buffers,
                "post_step_cfl",
                float(solver.compute_cfl(dt=natural_dt)),
            )
            _append_dense_value(natural_health_buffers, "finite_state_flag", True)
            if _is_swe_solver(solver):
                for key, value in _collect_swe_dense_health(solver).items():
                    _append_dense_value(natural_health_buffers, key, value)
            if hasattr(solver, "get_operator_diagnostics"):
                for key, value in solver.get_operator_diagnostics().items():
                    if value is None or isinstance(value, str):
                        continue
                    _append_dense_value(
                        natural_health_buffers, f"operator_{key}", value
                    )
            if hasattr(solver, "last_step_cg_converged"):
                _append_dense_value(
                    natural_health_buffers,
                    "cg_step_converged",
                    bool(getattr(solver, "last_step_cg_converged")),
                )
                _append_dense_value(
                    natural_health_buffers,
                    "cg_failed_count",
                    int(getattr(solver, "last_step_cg_failed_count", 0)),
                )
                _append_dense_value(
                    natural_health_buffers,
                    "cg_max_iterations",
                    int(getattr(solver, "last_step_cg_max_iterations", 0)),
                )
                _append_dense_value(
                    natural_health_buffers,
                    "cg_max_residual_ratio",
                    float(getattr(solver, "last_step_cg_max_residual_ratio", 0.0)),
                )
            if _is_boussinesq_solver(solver):
                for key, value in _collect_boussinesq_dense_health(solver).items():
                    _append_dense_value(natural_health_buffers, key, value)
                _append_dense_value(
                    natural_health_buffers,
                    "filter_enabled",
                    bool(
                        str(getattr(solver, "filter_time_mode", "legacy_per_step"))
                        != "disabled"
                        and float(getattr(solver, "filter_strength", 0.0)) > 0.0
                    ),
                )
                _append_dense_value(
                    natural_health_buffers,
                    "filter_application_count",
                    int(
                        solver.get_operator_diagnostics().get(
                            "filter_applications", 0
                        )
                        if hasattr(solver, "get_operator_diagnostics")
                        else 0
                    ),
                )
        if queries[next_request] < left_time:
            raise RuntimeError(
                "Next requested time fell behind the current natural bracket: "
                f"request={queries[next_request]!r}, left={left_time!r}"
            )

        bracket_end = next_request
        while bracket_end < queries.size and queries[bracket_end] <= current_time:
            bracket_end += 1
        if bracket_end > next_request:
            states, provenance = _extract_requested_states_from_bracket(
                left_state=left_state,
                right_state=right_state,
                left_time=left_time,
                right_time=current_time,
                requested_times=queries[next_request:bracket_end],
                right_natural_step_index=right_step,
                output_dtype=requested_state_dtype,
            )
            extracted_chunks.append(states)
            for key, values in provenance.items():
                provenance_chunks[key].append(values)
            next_request = bracket_end
            if next_request == queries.size:
                diagnostics = {
                    key: np.concatenate(chunks, axis=0)
                    for key, chunks in provenance_chunks.items()
                }
                natural_history = np.asarray(natural_dt_history, dtype=np.float64)
                diagnostics["total_natural_steps"] = np.asarray(
                    [right_step], dtype=np.int64
                )
                diagnostics["natural_dt_history"] = natural_history.copy()
                diagnostics["final_natural_timestamp"] = np.asarray(
                    [current_time], dtype=np.float64
                )
                if collect_natural_step_health:
                    diagnostics.update(
                        _finalize_dense_diagnostics(
                            natural_health_buffers, float_dtype=np.float64
                        )
                    )
                if cg_failed_count:
                    diagnostics["cg_failed_count"] = np.asarray(
                        cg_failed_count, dtype=np.int32
                    )
                    diagnostics["cg_max_iterations"] = np.asarray(
                        cg_max_iterations, dtype=np.int32
                    )
                    diagnostics["cg_max_residual_ratio"] = np.asarray(
                        cg_max_residual_ratio, dtype=np.float64
                    )
                return (
                    np.concatenate(extracted_chunks, axis=0),
                    queries.copy(),
                    natural_history,
                    diagnostics,
                )

        left_state = right_state

    next_missing = float(queries[next_request])
    raise RuntimeError(
        "Requested-time rollout exhausted max_natural_steps before full coverage: "
        f"emitted={next_request}/{queries.size}, "
        f"last_natural_time={current_time:.17g}, "
        f"next_missing_requested_time={next_missing:.17g}, cap={step_cap}"
    )


def _simulate_one_local(
    solver: Any,
    n_steps: int,
    save_every: int,
    auto_dt: bool,
    target_cfl: float,
    include_initial_state: bool,
    *,
    record_every_step: bool = False,
    dense_diagnostics: bool = False,
    requested_times: np.ndarray | None = None,
    max_natural_steps: int | None = None,
    collect_natural_step_health: bool = False,
    requested_state_dtype: Any = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if requested_times is not None:
        if record_every_step or dense_diagnostics:
            raise ValueError(
                "requested_times mode does not yet support "
                "record_every_step or dense_diagnostics"
            )
        return _simulate_requested_times_local(
            solver,
            auto_dt=auto_dt,
            target_cfl=target_cfl,
            requested_times=requested_times,
            max_natural_steps=max_natural_steps,
            collect_natural_step_health=collect_natural_step_health,
            requested_state_dtype=requested_state_dtype,
        )

    if max_natural_steps is not None:
        raise ValueError("max_natural_steps requires requested_times")

    if dense_diagnostics and not record_every_step:
        raise ValueError("dense_diagnostics=True requires record_every_step=True")

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    dt_hist: list[float] = []
    cg_failed_count: list[int] = []
    cg_max_iterations: list[int] = []
    cg_max_residual_ratio: list[float] = []
    dense_buffers: Dict[str, list[Any]] = {}
    current_time = 0.0

    if include_initial_state:
        frames.append(solver.get_state().astype(np.float32))
        timestamps.append(current_time)
        dt_hist.append(0.0)

    for step_idx in range(n_steps):
        if auto_dt:
            dt = solver.suggest_dt(target_cfl=target_cfl)
            solver.dt = dt
        else:
            dt = solver.dt

        if dense_diagnostics:
            _require_solver_cfl_diagnostics(solver)
            _append_dense_value(dense_buffers, "proposed_dt", float(dt))
            _append_dense_value(
                dense_buffers, "pre_step_cfl", float(solver.compute_cfl(dt=dt))
            )

        solver.step(dt=dt, auto_dt=False)
        if hasattr(solver, "last_step_cg_converged"):
            cg_failed_count.append(int(getattr(solver, "last_step_cg_failed_count", 0)))
            cg_max_iterations.append(
                int(getattr(solver, "last_step_cg_max_iterations", 0))
            )
            cg_max_residual_ratio.append(
                float(getattr(solver, "last_step_cg_max_residual_ratio", 0.0))
            )
        current_time += float(dt)

        if dense_diagnostics:
            _append_dense_value(
                dense_buffers, "post_step_cfl", float(solver.compute_cfl(dt=dt))
            )
            _append_dense_value(dense_buffers, "elapsed_benchmark_time", current_time)
            finite_state = bool(np.isfinite(solver.get_state()).all())
            _append_dense_value(dense_buffers, "finite_state_flag", finite_state)
            if _is_swe_solver(solver):
                for key, value in _collect_swe_dense_health(solver).items():
                    _append_dense_value(dense_buffers, key, value)
            if hasattr(solver, "last_step_cg_converged"):
                _append_dense_value(
                    dense_buffers,
                    "cg_step_converged",
                    bool(getattr(solver, "last_step_cg_converged")),
                )
                _append_dense_value(
                    dense_buffers,
                    "cg_failed_count",
                    int(getattr(solver, "last_step_cg_failed_count", 0)),
                )
                _append_dense_value(
                    dense_buffers,
                    "cg_max_iterations",
                    int(getattr(solver, "last_step_cg_max_iterations", 0)),
                )
                _append_dense_value(
                    dense_buffers,
                    "cg_max_residual_ratio",
                    float(getattr(solver, "last_step_cg_max_residual_ratio", 0.0)),
                )
            if _is_boussinesq_solver(solver):
                for key, value in _collect_boussinesq_dense_health(solver).items():
                    _append_dense_value(dense_buffers, key, value)

        if record_every_step or (step_idx + 1) % save_every == 0:
            frames.append(solver.get_state().astype(np.float32))
            timestamps.append(current_time)
            dt_hist.append(float(dt))

    if not frames:
        frames.append(solver.get_state().astype(np.float32))
        timestamps.append(current_time)
        dt_hist.append(0.0)

    diagnostics: Dict[str, np.ndarray] = {}
    if cg_failed_count:
        diagnostics["cg_failed_count"] = np.asarray(cg_failed_count, dtype=np.int32)
        diagnostics["cg_max_iterations"] = np.asarray(cg_max_iterations, dtype=np.int32)
        diagnostics["cg_max_residual_ratio"] = np.asarray(
            cg_max_residual_ratio, dtype=np.float32
        )
    if dense_diagnostics:
        diagnostics.update(_finalize_dense_diagnostics(dense_buffers))

    return (
        np.stack(frames, axis=0),
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(dt_hist, dtype=np.float32),
        diagnostics,
    )


def _run_fde_rollout(
    fde_name: str,
    solver_cfg: Dict[str, Any],
    dataset: DatasetConfig,
    bathymetry: np.ndarray,
    eta0: np.ndarray,
    h0: np.ndarray,
    *,
    requested_times: np.ndarray | None = None,
    max_natural_steps: int | None = None,
    collect_natural_step_health: bool = False,
) -> RolloutResult:

    if fde_name == "swe_hydrostatic":
        solver = _make_hydrostatic_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

        trajectory, timestamps, dt_hist, diagnostics = _simulate_one_local(
            solver=solver,
            n_steps=dataset.n_steps,
            save_every=dataset.save_every,
            auto_dt=dataset.auto_dt,
            target_cfl=dataset.target_cfl,
            include_initial_state=dataset.include_initial_state,
            requested_times=requested_times,
            max_natural_steps=max_natural_steps,
            collect_natural_step_health=collect_natural_step_health,
        )
        trajectory_eta = trajectory[:, 0] + bathymetry[None, ...]
        return RolloutResult(
            trajectory, trajectory_eta, timestamps, dt_hist, diagnostics
        )

    if fde_name == "swe_muscl_hr":
        solver = _make_muscl_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

        trajectory, timestamps, dt_hist, diagnostics = _simulate_one_local(
            solver=solver,
            n_steps=dataset.n_steps,
            save_every=dataset.save_every,
            auto_dt=dataset.auto_dt,
            target_cfl=dataset.target_cfl,
            include_initial_state=dataset.include_initial_state,
            requested_times=requested_times,
            max_natural_steps=max_natural_steps,
            collect_natural_step_health=collect_natural_step_health,
        )
        trajectory_eta = trajectory[:, 0] + bathymetry[None, ...]
        return RolloutResult(
            trajectory, trajectory_eta, timestamps, dt_hist, diagnostics
        )

    if fde_name == "boussinesq":
        solver = _make_boussinesq_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))

        trajectory, timestamps, dt_hist, diagnostics = _simulate_one_local(
            solver=solver,
            n_steps=dataset.n_steps,
            save_every=dataset.save_every,
            auto_dt=dataset.auto_dt,
            target_cfl=dataset.target_cfl,
            include_initial_state=dataset.include_initial_state,
            requested_times=requested_times,
            max_natural_steps=max_natural_steps,
            collect_natural_step_health=collect_natural_step_health,
        )
        trajectory_eta = trajectory[:, 0]
        return RolloutResult(
            trajectory, trajectory_eta, timestamps, dt_hist, diagnostics
        )

    raise NotImplementedError(f"FDE '{fde_name}' is not implemented yet")


def _generate_bathymetry_worker(
    sample_idx: int,
    run_seed: int,
    bathy_cfg_path: str,
    bathymetry_dir: str,
) -> Dict[str, Any]:

    sample_seed = _seed_for_sample(run_seed, sample_idx)
    generator = BathymetryGenerator(bathy_cfg_path)
    generator.rng = np.random.default_rng([sample_seed, 11])

    bathymetry, bathy_type = generator.generate()
    out_path = _bathymetry_file_path(bathymetry_dir, sample_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        bathymetry=bathymetry.astype(np.float32),
        bathymetry_type=np.array([str(bathy_type)], dtype="U64"),
        sample_seed=np.array([sample_seed], dtype=np.int64),
    )

    return {
        "sample_index": sample_idx,
        "bathymetry_type": str(bathy_type),
        "bathymetry_path": str(out_path),
    }


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if staging.exists():
        staging.unlink()
    try:
        with staging.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _generate_paired_bathymetry_worker(
    sample_idx: int,
    run_seed: int,
    master_config_path: str,
    bathymetry_dir: str,
    paired: PairedInputsConfig,
) -> Dict[str, Any]:
    sample_seed = _seed_for_sample(run_seed, sample_idx)
    generator = BathymetryGenerator(master_config_path)
    generator.rng = np.random.default_rng([sample_seed, 11])
    master, bathy_type = generator.generate()
    master = np.asarray(master, dtype=np.float32)
    target = _block_mean_downsample(master, paired.target_shape)
    master_hash = hash_array(master)["sha256"]
    out_path = _bathymetry_file_path(bathymetry_dir, sample_idx)
    _write_npz_atomic(
        out_path,
        bathymetry=target,
        master_bathymetry=master,
        bathymetry_type=np.asarray([str(bathy_type)], dtype="U64"),
        sample_seed=np.asarray([sample_seed], dtype=np.int64),
        native_input_schema_id=np.asarray([NATIVE_INPUT_SCHEMA_ID], dtype="U96"),
        native_lineage_id=np.asarray([paired.lineage_id], dtype="U128"),
        native_lineage_hash=np.asarray([paired.lineage_hash], dtype="U64"),
        native_target_contract_hash=np.asarray(
            [paired.target_contract_hash], dtype="U64"
        ),
        native_master_shape=np.asarray(paired.master_shape, dtype=np.int64),
        native_target_shape=np.asarray(paired.target_shape, dtype=np.int64),
        native_downsample_method=np.asarray(
            [paired.downsample_method], dtype="U64"
        ),
        native_master_array_sha256=np.asarray([master_hash], dtype="U64"),
    )
    return {
        "sample_index": sample_idx,
        "bathymetry_type": str(bathy_type),
        "bathymetry_path": str(out_path),
    }


def _generate_source_worker(
    sample_idx: int,
    run_seed: int,
    source_cfg_path: str,
    source_dir: str,
    source_strength_range: Tuple[float, float],
) -> Dict[str, Any]:
    sample_seed = _seed_for_sample(run_seed, sample_idx)
    source_generator = SourceGenerator(source_cfg_path)
    source_generator.rng = np.random.default_rng([sample_seed, 23])
    strength_rng = np.random.default_rng([sample_seed, 37])

    source_field, source_type = source_generator.generate()
    lo, hi = source_strength_range
    source_strength = float(strength_rng.uniform(lo, hi))

    out_path = _source_file_path(source_dir, sample_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        source_field=np.asarray(source_field, dtype=np.float32),
        source_type=np.array([str(source_type)], dtype="U64"),
        source_strength=np.array([source_strength], dtype=np.float32),
        sample_seed=np.array([sample_seed], dtype=np.int64),
    )
    return {
        "sample_index": sample_idx,
        "source_type": str(source_type),
        "source_strength": source_strength,
        "source_path": str(out_path),
    }


def _generate_paired_source_worker(
    sample_idx: int,
    run_seed: int,
    master_config_path: str,
    source_dir: str,
    source_strength_range: Tuple[float, float],
    paired: PairedInputsConfig,
) -> Dict[str, Any]:
    sample_seed = _seed_for_sample(run_seed, sample_idx)
    generator = SourceGenerator(master_config_path)
    generator.rng = np.random.default_rng([sample_seed, 23])
    strength_rng = np.random.default_rng([sample_seed, 37])
    master, source_type = generator.generate()
    master = np.asarray(master, dtype=np.float32)
    target = _block_mean_downsample(master, paired.target_shape)
    source_strength = float(strength_rng.uniform(*source_strength_range))
    master_hash = hash_array(master)["sha256"]
    out_path = _source_file_path(source_dir, sample_idx)
    _write_npz_atomic(
        out_path,
        source_field=target,
        master_source_field=master,
        source_type=np.asarray([str(source_type)], dtype="U64"),
        source_strength=np.asarray([source_strength], dtype=np.float32),
        sample_seed=np.asarray([sample_seed], dtype=np.int64),
        native_input_schema_id=np.asarray([NATIVE_INPUT_SCHEMA_ID], dtype="U96"),
        native_lineage_id=np.asarray([paired.lineage_id], dtype="U128"),
        native_lineage_hash=np.asarray([paired.lineage_hash], dtype="U64"),
        native_target_contract_hash=np.asarray(
            [paired.target_contract_hash], dtype="U64"
        ),
        native_master_shape=np.asarray(paired.master_shape, dtype=np.int64),
        native_target_shape=np.asarray(paired.target_shape, dtype=np.int64),
        native_downsample_method=np.asarray(
            [paired.downsample_method], dtype="U64"
        ),
        native_master_array_sha256=np.asarray([master_hash], dtype="U64"),
    )
    return {
        "sample_index": sample_idx,
        "source_type": str(source_type),
        "source_strength": source_strength,
        "source_path": str(out_path),
    }


def _requested_input_fingerprint(
    *,
    split: str,
    sample_idx: int,
    scenario_id: str,
    bathymetry: np.ndarray,
    source_field: np.ndarray,
    source_strength: float,
    rest_depth: np.ndarray,
    eta0: np.ndarray,
    initial_depth: np.ndarray,
    free_surface0: np.ndarray,
    bathymetry_type: str,
    source_type: str,
) -> str:
    return authoritative_input_fingerprint(
        split=split,
        sample_index=sample_idx,
        scenario_id=scenario_id,
        bathymetry_type=bathymetry_type,
        source_type=source_type,
        source_strength=np.asarray([source_strength], dtype=np.float32),
        arrays={
            "bathymetry": bathymetry,
            "source_field": source_field,
            "rest_depth": rest_depth,
            "eta0": eta0,
            "initial_depth": initial_depth,
            "free_surface0": free_surface0,
        },
    )


def _validate_authoritative_input(
    *,
    record: Mapping[str, Any],
    split: str,
    sample_idx: int,
    scenario_id: str,
    bathymetry: np.ndarray,
    source_field: np.ndarray,
    source_strength_array: np.ndarray,
    bathymetry_type: str,
    source_type: str,
    sea_level_offset: float,
    config: AuthoritativeInputsConfig,
) -> dict[str, Any]:
    identity = split_qualified_identity(split, scenario_id)
    for key, expected in (
        ("qualified_id", identity["qualified_id"]),
        ("scenario_id", scenario_id),
        ("split", identity["split"]),
    ):
        if str(record.get(key)) != str(expected):
            raise RuntimeError(f"Authoritative input {key} mismatch")
    if int(record.get("sample_index", -1)) != int(sample_idx):
        raise RuntimeError("Authoritative input sample_index mismatch")
    if str(record.get("bathymetry_type")) != str(bathymetry_type):
        raise RuntimeError("Authoritative input bathymetry family mismatch")
    if str(record.get("source_type")) != str(source_type):
        raise RuntimeError("Authoritative input source family mismatch")

    strength_array = np.asarray(source_strength_array)
    strength = float(strength_array.reshape(-1)[0])
    if np.float32(record.get("source_strength")) != np.float32(strength):
        raise RuntimeError("Authoritative input source strength mismatch")
    raw_bathymetry = np.asarray(bathymetry, dtype=np.float32)
    raw_source = np.asarray(source_field, dtype=np.float32)
    rest_depth = np.maximum(
        -raw_bathymetry + float(sea_level_offset), 0.0
    ).astype(np.float32, copy=False)
    eta0 = np.asarray(strength * raw_source, dtype=np.float32)
    initial_depth = np.asarray(np.maximum(rest_depth + eta0, 0.0), dtype=np.float32)
    free_surface0 = np.asarray(initial_depth + raw_bathymetry, dtype=np.float32)
    arrays = {
        "bathymetry": raw_bathymetry,
        "source_field": raw_source,
        "rest_depth": rest_depth,
        "eta0": eta0,
        "initial_depth": initial_depth,
        "free_surface0": free_surface0,
    }
    expected_hashes = record.get("array_hashes")
    if not isinstance(expected_hashes, Mapping):
        raise RuntimeError("Authoritative input array hashes are missing")
    if config.require_exact_arrays:
        for name, values in arrays.items():
            if hash_array(values) != expected_hashes.get(name):
                raise RuntimeError(f"Authoritative input array hash mismatch: {name}")
    fingerprint = authoritative_input_fingerprint(
        split=split,
        sample_index=sample_idx,
        scenario_id=scenario_id,
        bathymetry_type=bathymetry_type,
        source_type=source_type,
        source_strength=strength_array,
        arrays=arrays,
    )
    if fingerprint != str(record.get("input_fingerprint")):
        raise RuntimeError("Authoritative input fingerprint mismatch")
    return {
        "h0_contract_hash": config.h0_contract_hash,
        "inventory_sha256": config.inventory_sha256,
        "qualified_id": identity["qualified_id"],
        "input_fingerprint": fingerprint,
    }


def _requested_health_summary(
    diagnostics: Dict[str, np.ndarray], health: Dict[str, Any], *, fde_name: str
) -> dict[str, Any]:
    def _scalar(name: str, default: float = np.nan) -> float:
        values = np.asarray(diagnostics.get(name, []), dtype=np.float64).reshape(-1)
        return float(values[0]) if values.size else float(default)

    weights = np.asarray(diagnostics.get("interpolation_weights", []), dtype=np.float64)
    widths = np.asarray(diagnostics.get("bracket_widths", []), dtype=np.float64)
    exact = np.asarray(diagnostics.get("exact_knot", []), dtype=np.bool_)
    cfl = np.asarray(diagnostics.get("post_step_cfl", []), dtype=np.float64)
    summary = {
        **health,
        "requested_output_count": int(weights.size),
        "covered_requested_output_count": int(weights.size),
        "exact_knot_count": int(np.count_nonzero(exact)),
        "min_interpolation_weight": float(np.min(weights)) if weights.size else np.nan,
        "max_interpolation_weight": float(np.max(weights)) if weights.size else np.nan,
        "min_bracket_width": float(np.min(widths)) if widths.size else np.nan,
        "max_bracket_width": float(np.max(widths)) if widths.size else np.nan,
        "total_natural_steps": int(_scalar("total_natural_steps", 0.0)),
        "final_natural_timestamp": _scalar("final_natural_timestamp"),
        "max_post_step_cfl": float(np.max(cfl)) if cfl.size else np.nan,
    }
    if fde_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        natural_min_depth = np.asarray(
            diagnostics.get("swe_min_depth", []), dtype=np.float64
        )
        natural_max_speed = np.asarray(
            diagnostics.get("swe_max_speed", []), dtype=np.float64
        )
        if natural_min_depth.size:
            summary["min_h"] = min(
                float(summary.get("min_h", np.inf)),
                float(np.min(natural_min_depth)),
            )
        if natural_max_speed.size:
            summary["max_abs_velocity"] = max(
                float(summary.get("max_abs_velocity", -np.inf)),
                float(np.max(natural_max_speed)),
            )
    if fde_name == "boussinesq":
        natural_cg_failed = np.asarray(
            diagnostics.get("cg_failed_count", []), dtype=np.int64
        )
        if natural_cg_failed.size:
            summary["has_cg_diagnostics"] = True
            summary["cg_failed_count"] = int(np.sum(natural_cg_failed))
            summary["cg_converged_fraction"] = float(
                np.mean(natural_cg_failed == 0)
            )
    return summary


def _requested_dataset_semantics(dataset: DatasetConfig) -> dict[str, Any]:
    requested = dataset.requested_output
    if requested is None:
        raise RuntimeError("Requested semantics require requested_output config")
    return {
        "auto_dt": dataset.auto_dt,
        "target_cfl": dataset.target_cfl,
        "sea_level_offset": dataset.sea_level_offset,
        "max_natural_steps": requested.max_natural_steps,
        "quality_policy": dataset.quality_policy.__dict__,
        "eta_primary": requested.eta_primary,
        "debug_full_states": requested.debug_full_states,
        "buffered_domain": dataset.buffered_domain.semantics(),
        "authoritative_inputs": (
            None
            if dataset.authoritative_inputs is None
            else dataset.authoritative_inputs.semantics()
        ),
        "paired_inputs": dataset.paired_inputs.semantics(),
        "paired_input_inventory_sha256": dataset.paired_input_inventory_sha256,
    }


def _generation_resolved_config_hashes(
    dataset: DatasetConfig, solver_cfg: Mapping[str, Any]
) -> dict[str, str]:
    semantics = _requested_dataset_semantics(dataset)
    return {
        fde_name: resolved_config_hash(
            solver_name=fde_name,
            solver_config=_resolved_solver_cfg_for_fde(
                dict(solver_cfg), dataset.solver_profiles, fde_name
            ),
            dataset_semantics=semantics,
        )
        for fde_name in dataset.enabled_fdes
        if fde_name in IMPLEMENTED_FDES
    }


def _write_requested_publication(
    *,
    sample_dir: Path,
    rollout: RolloutResult,
    fde_name: str,
    dataset: DatasetConfig,
    solver_cfg: Dict[str, Any],
    sample_idx: int,
    scenario_id: str,
    bathymetry: np.ndarray,
    source_field: np.ndarray,
    source_strength: float,
    rest_depth: np.ndarray,
    eta0: np.ndarray,
    initial_depth: np.ndarray,
    free_surface0: np.ndarray,
    bathymetry_type: str,
    source_type: str,
    health: Dict[str, Any],
    quality_status: str,
    quality_violations: list[str],
    authoritative_input: Mapping[str, Any] | None = None,
    input_lineage: Mapping[str, Any] | None = None,
    source_code: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested = dataset.requested_output
    if requested is None:
        raise RuntimeError("Requested publication requires requested_output config")

    identity = split_qualified_identity(requested.split, scenario_id)
    input_fingerprint = _requested_input_fingerprint(
        split=requested.split,
        sample_idx=sample_idx,
        scenario_id=scenario_id,
        bathymetry=bathymetry,
        source_field=source_field,
        source_strength=source_strength,
        rest_depth=rest_depth,
        eta0=eta0,
        initial_depth=initial_depth,
        free_surface0=free_surface0,
        bathymetry_type=bathymetry_type,
        source_type=source_type,
    )
    config_hash = resolved_config_hash(
        solver_name=fde_name,
        solver_config=solver_cfg,
        dataset_semantics=_requested_dataset_semantics(dataset),
    )
    code = dict(source_code) if source_code is not None else code_state(ROOT)
    diagnostics = rollout.diagnostics or {}
    summary = _requested_health_summary(diagnostics, health, fde_name=fde_name)

    staging = sample_dir.with_name(f".{sample_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    if sample_dir.exists():
        raise FileExistsError(f"Refusing to overwrite requested publication: {sample_dir}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(
            staging / "sample.npz",
            bathymetry=np.asarray(bathymetry, dtype=np.float32),
            source_field=np.asarray(source_field, dtype=np.float32),
            source_strength=np.asarray([source_strength], dtype=np.float64),
            rest_depth=np.asarray(rest_depth, dtype=np.float32),
            eta0=np.asarray(eta0, dtype=np.float32),
            initial_depth=np.asarray(initial_depth, dtype=np.float32),
            free_surface0=np.asarray(free_surface0, dtype=np.float32),
            trajectory_eta=np.asarray(rollout.trajectory_eta, dtype=np.float32),
            timestamps=np.asarray(rollout.timestamps, dtype=np.float64),
            solver_name=np.asarray([fde_name], dtype="U64"),
            scenario_id=np.asarray([scenario_id], dtype="U64"),
            split=np.asarray([requested.split], dtype="U16"),
            schema_id=np.asarray([ETA_SAMPLE_SCHEMA_ID], dtype="U96"),
            contract_hash=np.asarray([requested.contract_hash], dtype="U64"),
        )
        np.savez_compressed(
            staging / "provenance.npz",
            **{key: np.asarray(value) for key, value in diagnostics.items()},
        )
        if requested.debug_full_states:
            np.savez_compressed(
                staging / "debug_full_states.npz",
                trajectory=np.asarray(rollout.trajectory, dtype=np.float32),
            )
        meta = {
            "schema_id": ETA_SAMPLE_SCHEMA_ID,
            "artifact_kind": "eta-primary-requested-output-sample",
            **identity,
            "sample_index": int(sample_idx),
            "solver_name": fde_name,
            "bathymetry_type": bathymetry_type,
            "source_type": source_type,
            "source_strength": float(source_strength),
            "input_fingerprint": input_fingerprint,
            "contract_hash": requested.contract_hash,
            "resolved_config_hash": config_hash,
            **code,
            "timestamps_shape": list(map(int, rollout.timestamps.shape)),
            "trajectory_eta_shape": list(map(int, rollout.trajectory_eta.shape)),
            "computational_domain": {
                **dataset.buffered_domain.semantics(),
                "solver_shape": [int(solver_cfg["nx"]), int(solver_cfg["ny"])],
                "publication_shape": list(map(int, bathymetry.shape)),
            },
            "debug_full_states": requested.debug_full_states,
            "quality_status": quality_status,
            "quality_violations": quality_violations,
            "health_summary": summary,
        }
        if authoritative_input is not None:
            meta["authoritative_input"] = dict(authoritative_input)
        if input_lineage is not None:
            meta["input_lineage"] = dict(input_lineage)
        with (staging / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

        payload_names = ["sample.npz", "provenance.npz", "meta.json"]
        if requested.debug_full_states:
            payload_names.append("debug_full_states.npz")
        files = [
            {
                "name": name,
                "size_bytes": int((staging / name).stat().st_size),
                "sha256": sha256_file(staging / name),
            }
            for name in sorted(payload_names)
        ]
        publication = {
            "schema_id": PUBLICATION_SCHEMA_ID,
            "artifact_kind": "requested-output-publication",
            **identity,
            "sample_index": int(sample_idx),
            "solver_name": fde_name,
            "input_fingerprint": input_fingerprint,
            "contract_hash": requested.contract_hash,
            "resolved_config_hash": config_hash,
            "code_state_hash": code["code_state_hash"],
            "quality_status": quality_status,
            "files": files,
        }
        if authoritative_input is not None:
            publication.update(
                {
                    "authoritative_input_fingerprint": str(
                        authoritative_input["input_fingerprint"]
                    ),
                    "authoritative_inventory_sha256": str(
                        authoritative_input["inventory_sha256"]
                    ),
                    "h0_contract_hash": str(
                        authoritative_input["h0_contract_hash"]
                    ),
                }
            )
        if input_lineage is not None:
            publication["input_lineage"] = dict(input_lineage)
        with (staging / "publication.json").open("w", encoding="utf-8") as handle:
            json.dump(publication, handle, indent=2, sort_keys=True)
        validate_publication(
            staging,
            expected_identity=identity,
            expected_contract_hash=requested.contract_hash,
            expected_config_hash=config_hash,
            expected_code_state_hash=code["code_state_hash"],
            expected_input_fingerprint=input_fingerprint,
            expected_authoritative_input_fingerprint=(
                None
                if authoritative_input is None
                else str(authoritative_input["input_fingerprint"])
            ),
            expected_authoritative_inventory_sha256=(
                None
                if authoritative_input is None
                else str(authoritative_input["inventory_sha256"])
            ),
            expected_times=requested.requested_times,
        )
        atomic_replace_directory(staging, sample_dir)
        return {"meta": meta, "publication": publication}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _generate_sample_worker(
    sample_idx: int,
    run_seed: int,
    dataset: DatasetConfig,
    solver_cfg: Dict[str, Any],
    source_cfg_path: str,
    config_path: str,
    bathy_cfg_path: str,
    bathymetry_dir: str,
    source_dir: str,
    fde_samples_dirs: Dict[str, str],
    authoritative_record: Mapping[str, Any] | None = None,
    source_code: Mapping[str, Any] | None = None,
    emit_solver_progress: bool = False,
    allow_override: bool = False,
) -> Dict[str, Any]:
    worker_started = time.monotonic()
    input_load_started = time.monotonic()
    bathy_path = _bathymetry_file_path(bathymetry_dir, sample_idx)
    source_path = _source_file_path(source_dir, sample_idx)

    if not bathy_path.exists():
        raise FileNotFoundError(
            f"Missing bathymetry cache for sample {sample_idx}: {bathy_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source cache for sample {sample_idx}: {source_path}"
        )

    bathymetry_master_hash: str | None = None
    source_master_hash: str | None = None
    with np.load(bathy_path, allow_pickle=False) as bathy_npz:
        bathymetry = np.asarray(bathy_npz["bathymetry"], dtype=np.float32)
        bathy_type = str(np.asarray(bathy_npz["bathymetry_type"]).reshape(-1)[0])
        if dataset.paired_inputs.enabled:
            bathymetry_master_hash = str(
                np.asarray(bathy_npz["native_master_array_sha256"]).reshape(-1)[0]
            )

    with np.load(source_path, allow_pickle=False) as src_npz:
        source_field = np.asarray(src_npz["source_field"], dtype=np.float32)
        source_type = str(np.asarray(src_npz["source_type"]).reshape(-1)[0])
        source_strength_array = np.asarray(src_npz["source_strength"])
        source_strength = float(source_strength_array.reshape(-1)[0])
        if dataset.paired_inputs.enabled:
            source_master_hash = str(
                np.asarray(src_npz["native_master_array_sha256"]).reshape(-1)[0]
            )

    scenario_id = f"scenario_{sample_idx:06d}"
    authoritative_input: dict[str, Any] | None = None
    input_lineage: dict[str, Any] | None = None
    if dataset.authoritative_inputs is not None:
        if authoritative_record is None:
            raise RuntimeError(
                f"Missing authoritative inventory row for {scenario_id}"
            )
        authoritative_input = _validate_authoritative_input(
            record=authoritative_record,
            split=(
                dataset.requested_output.split
                if dataset.requested_output is not None
                else str(authoritative_record.get("split", ""))
            ),
            sample_idx=sample_idx,
            scenario_id=scenario_id,
            bathymetry=bathymetry,
            source_field=source_field,
            source_strength_array=source_strength_array,
            bathymetry_type=bathy_type,
            source_type=source_type,
            sea_level_offset=dataset.sea_level_offset,
            config=dataset.authoritative_inputs,
        )
    if dataset.paired_inputs.enabled:
        if not dataset.paired_input_inventory_sha256:
            raise RuntimeError("paired input inventory was not frozen before rollout")
        input_lineage = {
            **(dataset.paired_inputs.semantics() or {}),
            "inventory_sha256": dataset.paired_input_inventory_sha256,
            "master_bathymetry_sha256": bathymetry_master_hash,
            "master_source_sha256": source_master_hash,
        }

    if dataset.buffered_domain.enabled:
        prepared = _prepare_buffered_domain(
            bathymetry,
            source_field,
            source_strength,
            dataset.sea_level_offset,
            dataset.buffered_domain,
        )
        bathymetry = prepared["bathymetry"]
        source_field = prepared["source_field"]
        eta0 = prepared["eta0"]
        rest_depth = prepared["rest_depth"]
        h0 = prepared["h0"]
        free_surface0 = prepared["free_surface0"]
        solver_bathymetry = prepared["solver_bathymetry"]
        solver_eta0 = prepared["solver_eta0"]
        solver_rest_depth = prepared["solver_rest_depth"]
        solver_h0 = prepared["solver_h0"]
        output_crop = prepared["crop"]
    else:
        eta0 = source_strength * source_field
        rest_depth = np.maximum(-bathymetry + dataset.sea_level_offset, 0.0)
        h0 = np.maximum(rest_depth + eta0, 0.0)
        free_surface0 = h0 + bathymetry
        solver_bathymetry = bathymetry
        solver_eta0 = eta0
        solver_rest_depth = rest_depth
        solver_h0 = h0
        output_crop = None

    runnable_fdes = [name for name in dataset.enabled_fdes if name in IMPLEMENTED_FDES]
    skipped_unimplemented = [
        name for name in dataset.enabled_fdes if name not in IMPLEMENTED_FDES
    ]

    if not runnable_fdes:
        raise RuntimeError(
            "No runnable FDE selected. Enable at least one implemented solver "
            "(swe_hydrostatic, swe_muscl_hr, boussinesq)."
        )

    solver_records: list[Dict[str, Any]] = []
    solver_timings: list[dict[str, Any]] = []
    input_load_s = max(0.0, time.monotonic() - input_load_started)

    for fde_name in runnable_fdes:
        solver_worker_started = time.monotonic()
        if emit_solver_progress:
            print(
                f"[solver-start] sample={sample_idx:06d} solver={fde_name}",
                flush=True,
            )
        solver_cfg_for_fde = _resolved_solver_cfg_for_fde(
            solver_cfg, dataset.solver_profiles, fde_name
        )
        if fde_name not in fde_samples_dirs:
            raise KeyError(f"Missing output samples directory for FDE '{fde_name}'")

        sample_dir = Path(fde_samples_dirs[fde_name]) / f"sample_{sample_idx:06d}"
        if dataset.requested_output is not None:
            if sample_dir.exists():
                validation_started = time.monotonic()
                try:
                    expected_input_fingerprint = _requested_input_fingerprint(
                        split=dataset.requested_output.split,
                        sample_idx=sample_idx,
                        scenario_id=scenario_id,
                        bathymetry=bathymetry,
                        source_field=source_field,
                        source_strength=source_strength,
                        rest_depth=rest_depth,
                        eta0=eta0,
                        initial_depth=h0,
                        free_surface0=free_surface0,
                        bathymetry_type=bathy_type,
                        source_type=source_type,
                    )
                    expected_config_hash = resolved_config_hash(
                        solver_name=fde_name,
                        solver_config=solver_cfg_for_fde,
                        dataset_semantics=_requested_dataset_semantics(
                            replace(
                                dataset,
                                target_cfl=float(
                                    solver_cfg_for_fde.get(
                                        "cfl", dataset.target_cfl
                                    )
                                ),
                            )
                        ),
                    )
                    expected_code_hash = (
                        code_state(ROOT)["code_state_hash"]
                        if source_code is None
                        else str(source_code["code_state_hash"])
                    )
                    publication = validate_publication(
                        sample_dir,
                        expected_identity=split_qualified_identity(
                            dataset.requested_output.split, scenario_id
                        ),
                        expected_contract_hash=dataset.requested_output.contract_hash,
                        expected_config_hash=expected_config_hash,
                        expected_code_state_hash=expected_code_hash,
                        expected_input_fingerprint=expected_input_fingerprint,
                        expected_times=dataset.requested_output.requested_times,
                        expected_solver_name=fde_name,
                        expected_sample_index=sample_idx,
                        expected_authoritative_input_fingerprint=(
                            None
                            if authoritative_input is None
                            else str(authoritative_input["input_fingerprint"])
                        ),
                        expected_authoritative_inventory_sha256=(
                            None
                            if authoritative_input is None
                            else str(authoritative_input["inventory_sha256"])
                        ),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Existing requested publication is corrupt or incompatible: {sample_dir}"
                    ) from exc
                solver_records.append(
                    _load_existing_solver_record(
                        sample_dir=sample_dir,
                        sample_idx=sample_idx,
                        fallback_solver_name=fde_name,
                    )
                )
                solver_records[-1]["publication_hash"] = stable_hash_payload(
                    artifact_kind="requested-output-publication-record",
                    payload=publication,
                    schema_id=PUBLICATION_SCHEMA_ID,
                )
                solver_timings.append(
                    {
                        "solver": fde_name,
                        "status": "reused",
                        "solve_s": 0.0,
                        "serialization_s": 0.0,
                        "validation_s": max(
                            0.0, time.monotonic() - validation_started
                        ),
                        "worker_s": max(
                            0.0, time.monotonic() - solver_worker_started
                        ),
                        "natural_steps": 0,
                    }
                )
                if emit_solver_progress:
                    timing = solver_timings[-1]
                    print(
                        f"[solver-done] sample={sample_idx:06d} "
                        f"solver={fde_name} status=reused "
                        f"validation={timing['validation_s']:.1f}s",
                        flush=True,
                    )
                continue
        elif sample_dir.exists() and not allow_override:
            if _sample_output_complete(sample_dir):
                try:
                    solver_records.append(
                        _load_existing_solver_record(
                            sample_dir=sample_dir,
                            sample_idx=sample_idx,
                            fallback_solver_name=fde_name,
                        )
                    )
                    continue
                except Exception:
                    shutil.rmtree(sample_dir)
            if sample_dir.exists():
                shutil.rmtree(sample_dir)

        dataset_for_fde = replace(
            dataset,
            target_cfl=float(solver_cfg_for_fde.get("cfl", dataset.target_cfl)),
        )
        solve_started = time.monotonic()
        full_rollout = _run_fde_rollout(
            fde_name=fde_name,
            solver_cfg=solver_cfg_for_fde,
            dataset=dataset_for_fde,
            bathymetry=solver_bathymetry,
            eta0=solver_eta0,
            h0=solver_h0,
            requested_times=(
                None
                if dataset.requested_output is None
                else dataset.requested_output.requested_times
            ),
            max_natural_steps=(
                None
                if dataset.requested_output is None
                else dataset.requested_output.max_natural_steps
            ),
            collect_natural_step_health=(
                False
                if dataset.requested_output is None
                else dataset.requested_output.collect_natural_step_health
            ),
        )
        solve_s = max(0.0, time.monotonic() - solve_started)

        effective_depth = None
        if fde_name == "boussinesq":
            solver_sea_level = float(solver_cfg_for_fde.get("sea_level_offset", 0.0))
            solver_depth_scale = float(solver_cfg_for_fde.get("depth_scale", 1.0))
            solver_min_depth = float(solver_cfg_for_fde.get("min_depth", 1e-3))
            effective_depth = np.maximum(
                (-solver_bathymetry + solver_sea_level) * solver_depth_scale,
                solver_min_depth,
            )

        health = _compute_rollout_health(
            fde_name=fde_name,
            rollout=full_rollout,
            rest_depth=solver_rest_depth,
            effective_depth=effective_depth,
        )
        if dataset.requested_output is not None:
            health = _requested_health_summary(
                full_rollout.diagnostics or {}, health, fde_name=fde_name
            )
        quality_violations = _quality_violations_for_health(
            health=health, policy=dataset.quality_policy
        )
        quality_status = "ok"
        if quality_violations:
            quality_status = dataset.quality_policy.on_violation
            message = (
                f"[quality] sample={sample_idx:06d} solver={fde_name} "
                f"violations={quality_violations}"
            )
            if dataset.quality_policy.on_violation == "fail":
                raise RuntimeError(message)
            print(f"{message} (continuing)")

        rollout = (
            full_rollout
            if output_crop is None
            else _crop_rollout(full_rollout, output_crop)
        )

        if dataset.requested_output is not None:
            serialization_started = time.monotonic()
            publication_result = _write_requested_publication(
                sample_dir=sample_dir,
                rollout=rollout,
                fde_name=fde_name,
                dataset=dataset_for_fde,
                solver_cfg=solver_cfg_for_fde,
                sample_idx=sample_idx,
                scenario_id=scenario_id,
                bathymetry=bathymetry,
                source_field=source_field,
                source_strength=source_strength,
                rest_depth=rest_depth,
                eta0=eta0,
                initial_depth=h0,
                free_surface0=free_surface0,
                bathymetry_type=bathy_type,
                source_type=source_type,
                health=health,
                quality_status=quality_status,
                quality_violations=quality_violations,
                authoritative_input=authoritative_input,
                input_lineage=input_lineage,
                source_code=source_code,
            )
            meta = publication_result["meta"]
            solver_records.append(
                {
                    "sample_index": sample_idx,
                    "scenario_id": scenario_id,
                    "sample_dir": str(sample_dir),
                    "solver_name": fde_name,
                    "bathymetry_type": bathy_type,
                    "source_type": source_type,
                    "source_strength": source_strength,
                    "num_frames": int(rollout.trajectory_eta.shape[0]),
                    "trajectory_shape": [],
                    "trajectory_eta_shape": list(
                        map(int, rollout.trajectory_eta.shape)
                    ),
                    "fdes_run": runnable_fdes,
                    "fdes_skipped_unimplemented": skipped_unimplemented,
                    "quality_status": quality_status,
                    "quality_violations": quality_violations,
                    "contract_hash": dataset.requested_output.contract_hash,
                    "resolved_config_hash": meta["resolved_config_hash"],
                    "input_fingerprint": meta["input_fingerprint"],
                    "code_state_hash": meta["code_state_hash"],
                    "authoritative_input_fingerprint": (
                        None
                        if authoritative_input is None
                        else authoritative_input["input_fingerprint"]
                    ),
                    "reused_existing": False,
                    **health,
                }
            )
            solver_timings.append(
                {
                    "solver": fde_name,
                    "status": "generated",
                    "solve_s": solve_s,
                    "serialization_s": max(
                        0.0, time.monotonic() - serialization_started
                    ),
                    "validation_s": 0.0,
                    "worker_s": max(
                        0.0, time.monotonic() - solver_worker_started
                    ),
                    "natural_steps": int(health.get("total_natural_steps", 0)),
                }
            )
            if emit_solver_progress:
                timing = solver_timings[-1]
                print(
                    f"[solver-done] sample={sample_idx:06d} "
                    f"solver={fde_name} status=generated "
                    f"solve={timing['solve_s']:.1f}s "
                    f"serialize={timing['serialization_s']:.1f}s "
                    f"steps={timing['natural_steps']}",
                    flush=True,
                )
            continue

        if allow_override and sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        rollout_payload = {
            "trajectory": rollout.trajectory.astype(np.float32),
            "trajectory_eta": rollout.trajectory_eta.astype(np.float32),
            "timestamps": rollout.timestamps.astype(np.float32),
            "dt_history": rollout.dt_history.astype(np.float32),
            "fde_name": np.array([fde_name], dtype="U64"),
        }
        if rollout.diagnostics:
            rollout_payload.update(
                {key: value for key, value in rollout.diagnostics.items()}
            )
        np.savez_compressed(sample_dir / "rollout.npz", **rollout_payload)
        np.save(
            sample_dir / "trajectory_eta.npy", rollout.trajectory_eta.astype(np.float32)
        )

        # keep sample.npz backward compatible for downstream preprocess/training
        np.savez_compressed(
            sample_dir / "sample.npz",
            bathymetry=bathymetry.astype(np.float32),
            source_field=source_field.astype(np.float32),
            rest_depth=rest_depth.astype(np.float32),
            eta0=eta0.astype(np.float32),
            initial_depth=h0.astype(np.float32),
            free_surface0=free_surface0.astype(np.float32),
            trajectory=rollout.trajectory.astype(np.float32),
            trajectory_eta=rollout.trajectory_eta.astype(np.float32),
            timestamps=rollout.timestamps.astype(np.float32),
            dt_history=rollout.dt_history.astype(np.float32),
            solver_name=np.array([fde_name], dtype="U64"),
            scenario_id=np.array([scenario_id], dtype="U64"),
        )

        meta = {
            "sample_index": sample_idx,
            "scenario_id": scenario_id,
            "solver_name": fde_name,
            "bathymetry_type": bathy_type,
            "source_type": source_type,
            "source_strength": source_strength,
            "num_frames": int(rollout.trajectory.shape[0]),
            "trajectory_shape": list(map(int, rollout.trajectory.shape)),
            "trajectory_eta_shape": list(map(int, rollout.trajectory_eta.shape)),
            "timestamps_shape": list(map(int, rollout.timestamps.shape)),
            "dt_history_shape": list(map(int, rollout.dt_history.shape)),
            "bathymetry_shape": list(map(int, bathymetry.shape)),
            "source_shape": list(map(int, source_field.shape)),
            "eta0_shape": list(map(int, eta0.shape)),
            "h0_shape": list(map(int, h0.shape)),
            "free_surface0_shape": list(map(int, free_surface0.shape)),
            "dataset_config_path": config_path,
            "bathymetry_config_path": bathy_cfg_path,
            "source_config_path": source_cfg_path,
            "solver": solver_cfg_for_fde,
            "computational_domain": {
                **dataset.buffered_domain.semantics(),
                "solver_shape": list(map(int, solver_bathymetry.shape)),
                "publication_shape": list(map(int, bathymetry.shape)),
            },
            "bathymetry_cache_path": str(bathy_path),
            "source_cache_path": str(source_path),
            "fdes_requested": list(dataset.enabled_fdes),
            "fdes_run": runnable_fdes,
            "fdes_skipped_unimplemented": skipped_unimplemented,
            "quality_status": quality_status,
            "quality_violations": quality_violations,
            **health,
        }
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        solver_records.append(
            {
                "sample_index": sample_idx,
                "scenario_id": scenario_id,
                "sample_dir": str(sample_dir),
                "solver_name": fde_name,
                "bathymetry_type": bathy_type,
                "source_type": source_type,
                "source_strength": source_strength,
                "num_frames": int(rollout.trajectory.shape[0]),
                "trajectory_shape": list(map(int, rollout.trajectory.shape)),
                "trajectory_eta_shape": list(map(int, rollout.trajectory_eta.shape)),
                "fdes_run": runnable_fdes,
                "fdes_skipped_unimplemented": skipped_unimplemented,
                "quality_status": quality_status,
                "quality_violations": quality_violations,
                "reused_existing": False,
                **health,
            }
        )

    fdes_run_actual = sorted(
        {
            str(rec.get("solver_name", ""))
            for rec in solver_records
            if rec.get("solver_name")
        }
    )
    scenario_record = {
        "sample_index": sample_idx,
        "scenario_id": scenario_id,
        "bathymetry_type": bathy_type,
        "source_type": source_type,
        "source_strength": source_strength,
        "bathymetry_cache_path": str(bathy_path),
        "source_cache_path": str(source_path),
        "fdes_requested": list(dataset.enabled_fdes),
        "fdes_run": fdes_run_actual if fdes_run_actual else runnable_fdes,
        "fdes_skipped_unimplemented": skipped_unimplemented,
    }
    if input_lineage is not None:
        scenario_record["input_lineage"] = input_lineage

    return {
        "sample_index": sample_idx,
        "scenario_id": scenario_id,
        "scenario_record": scenario_record,
        "solver_records": solver_records,
        "_operational": {
            "sample_index": int(sample_idx),
            "scenario_id": scenario_id,
            "worker_pid": os.getpid(),
            "input_load_s": input_load_s,
            "worker_total_s": max(0.0, time.monotonic() - worker_started),
            "solvers": solver_timings,
        },
    }


class TsunamiDatasetBuilder:
    """Generate raw tsunami surrogate samples."""

    def __init__(
        self, config_path: str, *, provenance_config_path: str | Path | None = None
    ) -> None:
        self.config_path = Path(config_path)
        self.provenance_config_path = Path(
            config_path if provenance_config_path is None else provenance_config_path
        )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Could not find {config_path}, is the path correct"
            )

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError("yaml config is empty/invalid")

        self.cfg = cfg
        self.dataset = self._parse_dataset_section(cfg)
        if (
            self.dataset.authoritative_inputs is not None
            and self.dataset.paired_inputs.enabled
        ):
            raise ValueError(
                "authoritative_inputs and paired_inputs are mutually exclusive"
            )
        self.operations = self._parse_operational_section(
            cfg, requested_workers=self.dataset.num_workers
        )
        self.source_code = code_state(ROOT)
        self.authoritative_records = self._load_authoritative_records()
        self.solver_cfg = self._parse_solver_section(cfg)
        self.bathy_cfg_path = self._require_path(cfg, ["configs", "bathymetry"])
        self.source_cfg_path = self._require_path(cfg, ["configs", "source"])
        if self.dataset.paired_inputs.enabled:
            self.dataset.paired_inputs = self._resolve_paired_input_contract(
                self.dataset.paired_inputs
            )

        self.output_dir = self.dataset.output_dir
        self.bathymetry_dir = self.dataset.bathymetry_dir
        self.source_dir = self.dataset.source_dir
        self.scenario_manifest_path = self.dataset.manifest_path
        self.fde_samples_dirs: Dict[str, Path] = {}
        self.fde_manifest_paths: Dict[str, Path] = {}

        for fde_name in self.dataset.enabled_fdes:
            if fde_name not in IMPLEMENTED_FDES:
                continue
            folder_name = _fde_dirname(fde_name)
            self.fde_samples_dirs[fde_name] = self.output_dir / folder_name / "samples"
            self.fde_manifest_paths[fde_name] = (
                self.scenario_manifest_path.parent / f"{folder_name}_manifest.jsonl"
            )

        if self.dataset.primary_fde in self.fde_samples_dirs:
            self.samples_dir = self.fde_samples_dirs[self.dataset.primary_fde]
        else:
            self.samples_dir = self.output_dir / "samples"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        for p in self.fde_samples_dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        self.bathymetry_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if self.dataset.copy_configs:
            self._copy_config_snapshot()

        self.bathy_generator = BathymetryGenerator(str(self.bathy_cfg_path))
        self.source_generator = SourceGenerator(str(self.source_cfg_path))

        self.run_seed = (
            int(np.random.SeedSequence().entropy)
            if self.dataset.seed is None
            else int(self.dataset.seed)
        )

        solver_nx = int(self.solver_cfg["nx"])
        solver_ny = int(self.solver_cfg["ny"])
        configured_input_shape = (self.bathy_generator.nx, self.bathy_generator.ny)
        configured_source_shape = (self.source_generator.nx, self.source_generator.ny)
        input_shape = (
            self.dataset.paired_inputs.target_shape
            if self.dataset.paired_inputs.enabled
            else configured_input_shape
        )
        source_shape = (
            self.dataset.paired_inputs.target_shape
            if self.dataset.paired_inputs.enabled
            else configured_source_shape
        )
        if self.dataset.paired_inputs.enabled:
            if configured_input_shape != input_shape:
                raise ValueError(
                    f"Target bathymetry config grid {configured_input_shape} must match "
                    f"paired_inputs.target_shape {input_shape}"
                )
            if configured_source_shape != source_shape:
                raise ValueError(
                    f"Target source config grid {configured_source_shape} must match "
                    f"paired_inputs.target_shape {source_shape}"
                )
            master_bathy = BathymetryGenerator(
                str(self.dataset.paired_inputs.master_bathymetry_config)
            )
            master_source = SourceGenerator(
                str(self.dataset.paired_inputs.master_source_config)
            )
            if (master_bathy.nx, master_bathy.ny) != (
                self.dataset.paired_inputs.master_shape
            ):
                raise ValueError(
                    "paired master bathymetry config does not match master_shape"
                )
            if (master_source.nx, master_source.ny) != (
                self.dataset.paired_inputs.master_shape
            ):
                raise ValueError("paired master source config does not match master_shape")
        if source_shape != input_shape:
            raise ValueError(
                f"Bathymetry grid {input_shape} must match source grid {source_shape}"
            )

        buffered = self.dataset.buffered_domain
        expected_solver_shape = input_shape
        if buffered.enabled:
            expected_solver_shape = tuple(
                axis + 2 * buffered.buffer_cells for axis in input_shape
            )
            if 2 * buffered.source_taper_cells >= min(input_shape):
                raise ValueError(
                    "computational_domain.source_taper_cells must leave an "
                    "untapered input-grid interior"
                )

        if (solver_nx, solver_ny) != expected_solver_shape:
            raise ValueError(
                f"Solver grid ({solver_nx}, {solver_ny}) must match the expected "
                f"computational grid {expected_solver_shape} for input grid {input_shape}"
            )

        if buffered.enabled:
            for fde_name in self.dataset.enabled_fdes:
                if fde_name not in IMPLEMENTED_FDES:
                    continue
                resolved = _resolved_solver_cfg_for_fde(
                    self.solver_cfg, self.dataset.solver_profiles, fde_name
                )
                if not bool(resolved.get("use_sponge", False)):
                    raise ValueError(
                        f"Buffered solver profile {fde_name} must enable the external sponge"
                    )
                if int(resolved.get("sponge_width", -1)) != buffered.buffer_cells:
                    raise ValueError(
                        f"Buffered solver profile {fde_name} must use sponge_width="
                        f"{buffered.buffer_cells} so damping remains outside the crop"
                    )
                if str(resolved.get("sponge_axes", "xy")) != "xy":
                    raise ValueError(
                        f"Buffered solver profile {fde_name} must use sponge_axes='xy'"
                    )
                if str(resolved.get("sponge_profile", "quadratic")) != "cosine":
                    raise ValueError(
                        f"Buffered solver profile {fde_name} must use sponge_profile='cosine'"
                    )

    @staticmethod
    def _require_path(cfg: Dict[str, Any], keys: list[str]) -> Path:
        node: Any = cfg
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"missing config path: {'.'.join(keys)}")
            node = node[key]
        return Path(str(node))

    def _resolve_paired_input_contract(
        self, config: PairedInputsConfig
    ) -> PairedInputsConfig:
        if (
            config.master_bathymetry_config is None
            or config.master_source_config is None
            or config.inventory_path is None
        ):
            raise ValueError("paired input paths were not resolved")
        for path in (
            config.master_bathymetry_config,
            config.master_source_config,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        generator_files = (
            ROOT / "src/data_gen/generate_bathymetry.py",
            ROOT / "src/data_gen/generate_sources.py",
            ROOT / "src/data_gen/simulate_dataset.py",
        )
        lineage_hash = stable_hash_payload(
            artifact_kind="native-resolution-master-input-lineage",
            schema_id=NATIVE_INPUT_SCHEMA_ID,
            payload={
                "lineage_id": config.lineage_id,
                "master_shape": list(config.master_shape),
                "dataset_seed": self.dataset.seed,
                "source_strength_range": list(self.dataset.source_strength_range),
                "master_bathymetry_config_sha256": sha256_file(
                    config.master_bathymetry_config
                ),
                "master_source_config_sha256": sha256_file(
                    config.master_source_config
                ),
                "generator_source_sha256": {
                    path.relative_to(ROOT).as_posix(): sha256_file(path)
                    for path in generator_files
                },
            },
        )
        target_contract_hash = stable_hash_payload(
            artifact_kind="native-resolution-target-input-contract",
            schema_id=NATIVE_INPUT_SCHEMA_ID,
            payload={
                "lineage_hash": lineage_hash,
                "target_shape": list(config.target_shape),
                "downsample_method": config.downsample_method,
            },
        )
        return replace(
            config,
            lineage_hash=lineage_hash,
            target_contract_hash=target_contract_hash,
        )

    @staticmethod
    def _parse_range(value: Any, name: str) -> Tuple[float, float]:
        arr = np.asarray(value, dtype=float)
        if arr.size != 2:
            raise ValueError(f"{name} must have 2 values [min, max]")
        if arr[0] > arr[1]:
            raise ValueError(f"{name} must have min <= max")
        return float(arr[0]), float(arr[1])

    @staticmethod
    def _parse_operational_section(
        cfg: Mapping[str, Any], *, requested_workers: int
    ) -> OperationalConfig:
        raw = cfg.get("operations", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("operations section must be a mapping")
        allowed = {
            "enabled",
            "progress_every",
            "solver_progress",
            "max_in_flight",
            "cloud_provider",
            "cloud_zone",
            "machine_type",
            "storage_class",
            "hourly_cost_usd",
        }
        unknown = sorted(set(str(key) for key in raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown operations keys: {unknown}")

        progress_every = int(raw.get("progress_every", 1))
        if progress_every <= 0:
            raise ValueError("operations.progress_every must be positive")
        max_in_flight_raw = raw.get("max_in_flight")
        max_in_flight = (
            None if max_in_flight_raw is None else int(max_in_flight_raw)
        )
        if max_in_flight is not None and max_in_flight < requested_workers:
            raise ValueError(
                "operations.max_in_flight must be at least dataset.num_workers"
            )
        hourly_cost_raw = raw.get("hourly_cost_usd")
        hourly_cost = None if hourly_cost_raw is None else float(hourly_cost_raw)
        if hourly_cost is not None and (
            not np.isfinite(hourly_cost) or hourly_cost < 0.0
        ):
            raise ValueError("operations.hourly_cost_usd must be finite and nonnegative")

        def _optional_text(name: str) -> str | None:
            value = raw.get(name)
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        return OperationalConfig(
            enabled=bool(raw.get("enabled", True)),
            progress_every=progress_every,
            solver_progress=bool(raw.get("solver_progress", True)),
            max_in_flight=max_in_flight,
            cloud_provider=_optional_text("cloud_provider"),
            cloud_zone=_optional_text("cloud_zone"),
            machine_type=_optional_text("machine_type"),
            storage_class=_optional_text("storage_class"),
            hourly_cost_usd=hourly_cost,
        )

    @staticmethod
    def _parse_authoritative_inputs_section(
        cfg: Mapping[str, Any], requested: RequestedOutputConfig | None
    ) -> AuthoritativeInputsConfig | None:
        raw = cfg.get("authoritative_inputs")
        if raw is None:
            return None
        if requested is None:
            raise ValueError(
                "authoritative_inputs requires requested_output generation"
            )
        if not isinstance(raw, Mapping):
            raise ValueError("authoritative_inputs section must be a mapping")
        allowed = {
            "inventory_path",
            "inventory_sha256",
            "h0_contract_hash",
            "require_exact_arrays",
            "allow_input_generation",
        }
        unknown = sorted(set(str(key) for key in raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown authoritative_inputs keys: {unknown}")
        for key in ("inventory_path", "inventory_sha256", "h0_contract_hash"):
            if not str(raw.get(key, "")).strip():
                raise ValueError(f"authoritative_inputs.{key} is required")
        inventory_path = Path(str(raw["inventory_path"]))
        if not inventory_path.is_absolute():
            inventory_path = ROOT / inventory_path
        require_exact = bool(raw.get("require_exact_arrays", True))
        allow_generation = bool(raw.get("allow_input_generation", False))
        if not require_exact:
            raise ValueError("authoritative_inputs.require_exact_arrays must be true")
        return AuthoritativeInputsConfig(
            inventory_path=inventory_path.resolve(),
            inventory_sha256=str(raw["inventory_sha256"]),
            h0_contract_hash=str(raw["h0_contract_hash"]),
            require_exact_arrays=require_exact,
            allow_input_generation=allow_generation,
        )

    @staticmethod
    def _parse_paired_inputs_section(cfg: Mapping[str, Any]) -> PairedInputsConfig:
        raw = cfg.get("paired_inputs")
        if raw is None:
            return PairedInputsConfig()
        if not isinstance(raw, Mapping):
            raise ValueError("paired_inputs section must be a mapping")
        allowed = {
            "enabled",
            "lineage_id",
            "master_shape",
            "target_shape",
            "downsample_method",
            "master_bathymetry_config",
            "master_source_config",
            "inventory_path",
        }
        unknown = sorted(set(str(key) for key in raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown paired_inputs keys: {unknown}")
        if not bool(raw.get("enabled", False)):
            raise ValueError("paired_inputs must be omitted unless enabled=true")

        def _shape(name: str) -> tuple[int, int]:
            values = tuple(int(value) for value in raw.get(name, ()))
            if len(values) != 2 or min(values) <= 1:
                raise ValueError(f"paired_inputs.{name} must contain two values > 1")
            return values

        lineage_id = str(raw.get("lineage_id", "")).strip()
        if not lineage_id:
            raise ValueError("paired_inputs.lineage_id is required")
        master_shape = _shape("master_shape")
        target_shape = _shape("target_shape")
        if any(master % target for master, target in zip(master_shape, target_shape)):
            raise ValueError(
                "paired_inputs.master_shape must be integer-divisible by target_shape"
            )
        method = str(
            raw.get("downsample_method", "block_mean_float64_v1")
        ).strip()
        if method != "block_mean_float64_v1":
            raise ValueError(
                "paired_inputs.downsample_method must be block_mean_float64_v1"
            )

        def _required_path(name: str) -> Path:
            value = str(raw.get(name, "")).strip()
            if not value:
                raise ValueError(f"paired_inputs.{name} is required")
            path = Path(value)
            return (path if path.is_absolute() else ROOT / path).resolve()

        return PairedInputsConfig(
            enabled=True,
            lineage_id=lineage_id,
            master_shape=master_shape,
            target_shape=target_shape,
            downsample_method=method,
            master_bathymetry_config=_required_path("master_bathymetry_config"),
            master_source_config=_required_path("master_source_config"),
            inventory_path=_required_path("inventory_path"),
        )

    def _load_authoritative_records(self) -> dict[int, dict[str, Any]]:
        config = self.dataset.authoritative_inputs
        if config is None:
            return {}
        if not config.inventory_path.is_file():
            raise FileNotFoundError(config.inventory_path)
        if sha256_file(config.inventory_path) != config.inventory_sha256:
            raise RuntimeError("Authoritative input inventory checksum mismatch")
        requested = self.dataset.requested_output
        if requested is None:
            raise RuntimeError("Authoritative inputs require requested-output mode")
        records: dict[int, dict[str, Any]] = {}
        with config.inventory_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                if str(record.get("split")) != requested.split:
                    continue
                index = int(record["sample_index"])
                if index in records:
                    raise RuntimeError(
                        f"Duplicate authoritative sample index: {index}"
                    )
                records[index] = record
        missing = [
            index
            for index in range(1, self.dataset.num_samples + 1)
            if index not in records
        ]
        if missing:
            raise RuntimeError(
                "Authoritative inventory is missing configured sample indices: "
                f"{missing[:10]}"
            )
        return records

    @staticmethod
    def _parse_buffered_domain_section(cfg: Mapping[str, Any]) -> BufferedDomainConfig:
        raw = cfg.get("computational_domain", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("computational_domain section must be a mapping")
        allowed = {
            "enabled",
            "buffer_cells",
            "source_taper_cells",
            "bathymetry_extension",
            "output_crop",
        }
        unknown = sorted(set(str(key) for key in raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown computational_domain keys: {unknown}")

        enabled = bool(raw.get("enabled", False))
        buffer_cells = int(raw.get("buffer_cells", 0))
        source_taper_cells = int(raw.get("source_taper_cells", 0))
        bathymetry_extension = str(raw.get("bathymetry_extension", "edge"))
        output_crop = str(raw.get("output_crop", "central"))
        if enabled:
            if buffer_cells <= 0:
                raise ValueError(
                    "computational_domain.buffer_cells must be positive when enabled"
                )
            if source_taper_cells < 2:
                raise ValueError(
                    "computational_domain.source_taper_cells must be at least 2"
                )
            if bathymetry_extension != "edge":
                raise ValueError(
                    "only computational_domain.bathymetry_extension='edge' is supported"
                )
            if output_crop != "central":
                raise ValueError(
                    "only computational_domain.output_crop='central' is supported"
                )
        elif buffer_cells != 0 or source_taper_cells != 0:
            raise ValueError(
                "disabled computational_domain must use zero buffer and taper cells"
            )
        return BufferedDomainConfig(
            enabled=enabled,
            buffer_cells=buffer_cells,
            source_taper_cells=source_taper_cells,
            bathymetry_extension=bathymetry_extension,
            output_crop=output_crop,
        )

    @staticmethod
    def _parse_dataset_section(cfg: Dict[str, Any]) -> DatasetConfig:
        ds = cfg.get("dataset", {})
        if not isinstance(ds, dict):
            raise ValueError("dataset section must be a mapping")

        fdes = cfg.get("fdes", {})
        if not isinstance(fdes, dict):
            raise ValueError("fdes section must be a mapping")

        enabled_fdes_raw = fdes.get("enabled", ["swe_hydrostatic"])
        if not isinstance(enabled_fdes_raw, list) or not enabled_fdes_raw:
            raise ValueError("fdes.enabled must be a non-empty list")

        enabled_fdes: list[str] = list(
            dict.fromkeys(
                _canonical_fde_name(str(name).strip()) for name in enabled_fdes_raw
            )
        )
        for name in enabled_fdes:
            if name not in KNOWN_FDES:
                raise ValueError(
                    f"Unknown FDE '{name}'. Supported names: {sorted(KNOWN_FDES)}"
                )

        primary_fde = _canonical_fde_name(
            str(fdes.get("primary", enabled_fdes[0])).strip()
        )
        if primary_fde not in enabled_fdes:
            raise ValueError("fdes.primary must be one of fdes.enabled")

        num_samples = int(ds.get("num_samples", 100))
        seed = ds.get("seed", None)
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("dataset.seed must be >= 0")

        num_workers = int(ds.get("num_workers", 1))
        n_steps = int(ds.get("n_steps", 200))
        save_every = int(ds.get("save_every", 5))
        auto_dt = bool(ds.get("auto_dt", True))
        target_cfl = float(ds.get("target_cfl", 0.45))
        include_initial_state = bool(ds.get("include_initial_state", True))
        sea_level_offset = float(ds.get("sea_level_offset", 0.0))

        source_strength_range = TsunamiDatasetBuilder._parse_range(
            ds.get("source_strength_range", [0.5, 2.0]), "dataset.source_strength_range"
        )

        output_dir = Path(ds.get("output_dir", "data/raw"))
        bathymetry_dir = Path(ds.get("bathymetry_dir", "data/bathymetry"))
        source_dir = Path(ds.get("source_dir", "data/sources"))
        manifest_path = Path(
            ds.get("manifest_path", "data/synthetic/scenario_manifest.jsonl")
        )
        copy_configs = bool(ds.get("copy_configs", True))
        quality_cfg = cfg.get("quality", {})
        if not isinstance(quality_cfg, dict):
            quality_cfg = {}

        quality_on_violation = (
            str(quality_cfg.get("on_violation", "warn")).strip().lower()
        )
        if quality_on_violation not in {"warn", "fail"}:
            raise ValueError("quality.on_violation must be one of: warn, fail")

        def _optional_float(value: Any, key: str) -> float | None:
            if value is None:
                return None
            out = float(value)
            if not np.isfinite(out):
                raise ValueError(f"quality.{key} must be finite when set")
            return out

        quality_policy = QualityPolicy(
            on_violation=quality_on_violation,
            reject_nonfinite=bool(quality_cfg.get("reject_nonfinite", True)),
            min_h_tolerance=_optional_float(
                quality_cfg.get("min_h_tolerance", None), "min_h_tolerance"
            ),
            max_abs_eta_limit=_optional_float(
                quality_cfg.get("max_abs_eta_limit", None), "max_abs_eta_limit"
            ),
            max_velocity_limit=_optional_float(
                quality_cfg.get("max_velocity_limit", None), "max_velocity_limit"
            ),
            max_eta_over_depth=_optional_float(
                quality_cfg.get("max_eta_over_depth", None), "max_eta_over_depth"
            ),
            require_cg_converged=bool(quality_cfg.get("require_cg_converged", True)),
        )

        if (
            quality_policy.max_eta_over_depth is not None
            and quality_policy.max_eta_over_depth <= 0
        ):
            raise ValueError("quality.max_eta_over_depth must be positive when set")

        if num_samples <= 0:
            raise ValueError("dataset.num_samples must be positive")
        if n_steps <= 0:
            raise ValueError("dataset.n_steps must be positive")
        if num_workers <= 0:
            raise ValueError("dataset.num_workers must be positive")
        if save_every <= 0:
            raise ValueError("dataset.save_every must be positive")
        if target_cfl <= 0:
            raise ValueError("dataset.target_cfl must be positive")

        requested_output = parse_requested_output_config(cfg.get("requested_output"))
        paired_inputs = TsunamiDatasetBuilder._parse_paired_inputs_section(cfg)
        if paired_inputs.enabled:
            if requested_output is None:
                raise ValueError("paired_inputs requires requested_output generation")
            if seed is None:
                raise ValueError("paired_inputs requires a fixed dataset.seed")
            if quality_policy.on_violation != "fail":
                raise ValueError("paired_inputs requires quality.on_violation=fail")
        return DatasetConfig(
            num_samples=num_samples,
            seed=seed,
            num_workers=num_workers,
            n_steps=n_steps,
            save_every=save_every,
            auto_dt=auto_dt,
            target_cfl=target_cfl,
            include_initial_state=include_initial_state,
            sea_level_offset=sea_level_offset,
            source_strength_range=source_strength_range,
            output_dir=output_dir,
            bathymetry_dir=bathymetry_dir,
            source_dir=source_dir,
            manifest_path=manifest_path,
            copy_configs=copy_configs,
            enabled_fdes=tuple(enabled_fdes),
            primary_fde=primary_fde,
            quality_policy=quality_policy,
            requested_output=requested_output,
            authoritative_inputs=(
                TsunamiDatasetBuilder._parse_authoritative_inputs_section(
                    cfg, requested_output
                )
            ),
            solver_profiles={
                _canonical_fde_name(str(name)): dict(profile)
                for name, profile in dict(cfg.get("solver_profiles", {})).items()
            },
            buffered_domain=TsunamiDatasetBuilder._parse_buffered_domain_section(cfg),
            paired_inputs=paired_inputs,
        )

    @staticmethod
    def _parse_solver_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
        sv = cfg.get("solver", {})
        if not isinstance(sv, dict):
            raise ValueError("solver section must be a mapping")

        required = ["nx", "ny", "dx", "dy", "dt"]
        for key in required:
            if key not in sv:
                raise KeyError(f"missing solver key: {key}")
        return sv

    def _copy_config_snapshot(self) -> None:
        snapshot_path = self.output_dir / "dataset_config.snapshot.yaml"
        with snapshot_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

    @staticmethod
    def _append_manifest(manifest_path: Path, record: Dict[str, Any]) -> None:
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _purge_manifest_indices(manifest_path: Path, indices: set[int]) -> None:
        if not indices:
            return
        if not manifest_path.exists():
            return

        keep_lines: list[str] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                except Exception:
                    keep_lines.append(line)
                    continue
                idx = rec.get("sample_index")
                if idx is None:
                    keep_lines.append(line)
                    continue
                if int(idx) in indices:
                    continue
                keep_lines.append(line)

        with manifest_path.open("w", encoding="utf-8") as f:
            for line in keep_lines:
                f.write(line if line.endswith("\n") else line + "\n")

    def _existing_sample_indices(self) -> set[int]:
        patt = re.compile(r"^sample_(\d{6})$")
        per_fde_sets: list[set[int]] = []

        for samples_dir in self.fde_samples_dirs.values():
            idxs: set[int] = set()
            if not samples_dir.exists():
                per_fde_sets.append(idxs)
                continue
            for p in samples_dir.iterdir():
                if not p.is_dir():
                    continue
                if not _sample_output_complete(p, requested=self.dataset.requested_output):
                    continue
                m = patt.match(p.name)
                if m is None:
                    continue
                idxs.add(int(m.group(1)))
            per_fde_sets.append(idxs)

        if not per_fde_sets:
            return set()
        return set.intersection(*per_fde_sets)

    def _existing_any_sample_indices(self) -> set[int]:
        patt = re.compile(r"^sample_(\d{6})$")
        out: set[int] = set()
        for samples_dir in self.fde_samples_dirs.values():
            if not samples_dir.exists():
                continue
            for p in samples_dir.iterdir():
                if not p.is_dir():
                    continue
                m = patt.match(p.name)
                if m is None:
                    continue
                out.add(int(m.group(1)))
        return out

    def _resume_rollback_start_index(self, existing_indices: set[int]) -> int:
        if not existing_indices:
            return 1
        workers = max(1, int(self.dataset.num_workers))
        n = max(existing_indices)
        start_idx = n - (n % workers)
        return max(1, int(start_idx))

    def _existing_bathymetry_indices(self) -> set[int]:
        out: set[int] = set()
        patt = re.compile(r"^sample_(\d{6})\.npz$")

        for p in self.bathymetry_dir.iterdir():
            if not p.is_file():
                continue
            m = patt.match(p.name)
            if m is None:
                continue
            out.add(int(m.group(1)))

        return out

    def _existing_source_indices(self) -> set[int]:
        out: set[int] = set()
        patt = re.compile(r"^sample_(\d{6})\.npz$")
        for p in self.source_dir.iterdir():
            if not p.is_file():
                continue
            m = patt.match(p.name)
            if m is None:
                continue
            out.add(int(m.group(1)))
        return out

    def rebuild_manifests_from_existing_outputs(self) -> None:
        patt = re.compile(r"^sample_(\d{6})$")
        scenario_rows: dict[int, Dict[str, Any]] = {}
        solver_rows: dict[str, list[Dict[str, Any]]] = {
            name: [] for name in self.fde_manifest_paths.keys()
        }

        for fde_name, samples_dir in self.fde_samples_dirs.items():
            if not samples_dir.exists():
                continue

            for sample_dir in sorted(samples_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                m = patt.match(sample_dir.name)
                if m is None:
                    continue

                sample_idx = int(m.group(1))
                meta_path = sample_dir / "meta.json"
                if not meta_path.exists():
                    continue

                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    continue

                scenario_id = str(meta.get("scenario_id", f"scenario_{sample_idx:06d}"))
                source_strength_raw = meta.get("source_strength", np.nan)
                try:
                    source_strength = float(source_strength_raw)
                except Exception:
                    source_strength = float(np.nan)

                solver_name = _canonical_fde_name(
                    str(meta.get("solver_name", fde_name))
                )
                if solver_name not in self.fde_manifest_paths:
                    solver_name = fde_name

                if sample_idx not in scenario_rows:
                    scenario_rows[sample_idx] = {
                        "sample_index": sample_idx,
                        "scenario_id": scenario_id,
                        "bathymetry_type": str(meta.get("bathymetry_type", "unknown")),
                        "source_type": str(meta.get("source_type", "unknown")),
                        "source_strength": source_strength,
                        "bathymetry_cache_path": str(
                            meta.get("bathymetry_cache_path", "")
                        ),
                        "source_cache_path": str(meta.get("source_cache_path", "")),
                        "fdes_requested": list(
                            meta.get("fdes_requested", self.dataset.enabled_fdes)
                        ),
                        "fdes_run": list(meta.get("fdes_run", [solver_name])),
                        "fdes_skipped_unimplemented": list(
                            meta.get("fdes_skipped_unimplemented", [])
                        ),
                    }
                else:
                    existing_run = set(scenario_rows[sample_idx].get("fdes_run", []))
                    existing_run.add(solver_name)
                    scenario_rows[sample_idx]["fdes_run"] = sorted(existing_run)

                srec: Dict[str, Any] = {
                    "sample_index": sample_idx,
                    "scenario_id": scenario_id,
                    "sample_dir": str(sample_dir),
                    "solver_name": solver_name,
                    "bathymetry_type": str(meta.get("bathymetry_type", "unknown")),
                    "source_type": str(meta.get("source_type", "unknown")),
                    "source_strength": source_strength,
                    "fdes_run": list(meta.get("fdes_run", [solver_name])),
                    "fdes_skipped_unimplemented": list(
                        meta.get("fdes_skipped_unimplemented", [])
                    ),
                }
                if "num_frames" in meta:
                    srec["num_frames"] = int(meta["num_frames"])
                if "trajectory_shape" in meta:
                    srec["trajectory_shape"] = meta["trajectory_shape"]
                if "trajectory_eta_shape" in meta:
                    srec["trajectory_eta_shape"] = meta["trajectory_eta_shape"]
                for key in (
                    "nan_count",
                    "inf_count",
                    "min_h",
                    "max_abs_eta",
                    "max_abs_velocity",
                    "dt_min",
                    "dt_max",
                ):
                    if key in meta:
                        srec[key] = meta[key]
                if "quality_status" in meta:
                    srec["quality_status"] = meta["quality_status"]
                if "quality_violations" in meta:
                    srec["quality_violations"] = meta["quality_violations"]

                solver_rows.setdefault(solver_name, []).append(srec)

        if not scenario_rows:
            raise RuntimeError(
                "Could not rebuild manifests: no valid sample_*/meta.json records found under output sample directories."
            )

        for path in [self.scenario_manifest_path, *self.fde_manifest_paths.values()]:
            if path.exists():
                path.unlink()

        for idx in sorted(scenario_rows.keys()):
            self._append_manifest(self.scenario_manifest_path, scenario_rows[idx])

        for solver_name, rows in solver_rows.items():
            manifest_path = self.fde_manifest_paths.get(solver_name)
            if manifest_path is None:
                continue
            rows.sort(key=lambda r: int(r["sample_index"]))
            for row in rows:
                self._append_manifest(manifest_path, row)

        print(
            f"[dataset] rebuilt manifests from existing outputs: "
            f"scenarios={len(scenario_rows)}, "
            + ", ".join(
                f"{name}={len(rows)}" for name, rows in sorted(solver_rows.items())
            )
        )

    def _phase_generate_bathymetry(
        self, indices: list[int], allow_override: bool = False
    ) -> None:
        existing = self._existing_bathymetry_indices()
        pending = (
            indices
            if allow_override
            else [idx for idx in indices if idx not in existing]
        )

        if not pending:
            print(
                "[dataset] phase 1/3 bathymetry cache already complete for this range"
            )
            return
        authoritative = self.dataset.authoritative_inputs
        if authoritative is not None and not authoritative.allow_input_generation:
            raise RuntimeError(
                "Authoritative bathymetry caches are missing; input regeneration "
                "is disabled"
            )

        print(
            f"[dataset] phase 1/3 generate bathymetry: pending={len(pending)}, "
            f"range=[{pending[0]}, {pending[-1]}], out='{self.bathymetry_dir}'"
        )

        if self.dataset.num_workers <= 1:
            done = 0
            for idx in pending:
                if self.dataset.paired_inputs.enabled:
                    rec = _generate_paired_bathymetry_worker(
                        idx,
                        self.run_seed,
                        str(
                            self.dataset.paired_inputs.master_bathymetry_config
                        ),
                        str(self.bathymetry_dir),
                        self.dataset.paired_inputs,
                    )
                else:
                    rec = _generate_bathymetry_worker(
                        idx,
                        self.run_seed,
                        str(self.bathy_cfg_path),
                        str(self.bathymetry_dir),
                    )
                done += 1
                print(
                    f"[bathy {done:06d}/{len(pending):06d}] "
                    f"sample={idx:06d} type={rec['bathymetry_type']:<11}"
                )
            return

        workers = min(self.dataset.num_workers, max(1, os.cpu_count() or 1))
        mp_ctx = get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            if self.dataset.paired_inputs.enabled:
                futures = {
                    ex.submit(
                        _generate_paired_bathymetry_worker,
                        idx,
                        self.run_seed,
                        str(
                            self.dataset.paired_inputs.master_bathymetry_config
                        ),
                        str(self.bathymetry_dir),
                        self.dataset.paired_inputs,
                    ): idx
                    for idx in pending
                }
            else:
                futures = {
                    ex.submit(
                        _generate_bathymetry_worker,
                        idx,
                        self.run_seed,
                        str(self.bathy_cfg_path),
                        str(self.bathymetry_dir),
                    ): idx
                    for idx in pending
                }

            for fut in as_completed(futures):
                rec = fut.result()
                done += 1
                print(
                    f"[bathy {done:06d}/{len(pending):06d}] "
                    f"sample={rec['sample_index']:06d} type={rec['bathymetry_type']:<11}"
                )

    def _phase_generate_sources(
        self, indices: list[int], allow_override: bool = False
    ) -> None:
        existing = self._existing_source_indices()
        pending = (
            indices
            if allow_override
            else [idx for idx in indices if idx not in existing]
        )

        if not pending:
            print("[dataset] phase 2/3 source cache already complete for this range")
            return
        authoritative = self.dataset.authoritative_inputs
        if authoritative is not None and not authoritative.allow_input_generation:
            raise RuntimeError(
                "Authoritative source caches are missing; input regeneration is "
                "disabled"
            )

        print(
            f"[dataset] phase 2/3 generate sources: pending={len(pending)}, "
            f"range=[{pending[0]}, {pending[-1]}], out='{self.source_dir}'"
        )

        if self.dataset.num_workers <= 1:
            done = 0
            for idx in pending:
                if self.dataset.paired_inputs.enabled:
                    rec = _generate_paired_source_worker(
                        idx,
                        self.run_seed,
                        str(self.dataset.paired_inputs.master_source_config),
                        str(self.source_dir),
                        self.dataset.source_strength_range,
                        self.dataset.paired_inputs,
                    )
                else:
                    rec = _generate_source_worker(
                        idx,
                        self.run_seed,
                        str(self.source_cfg_path),
                        str(self.source_dir),
                        self.dataset.source_strength_range,
                    )
                done += 1
                print(
                    f"[source {done:06d}/{len(pending):06d}] "
                    f"sample={idx:06d} type={rec['source_type']:<11} amp={rec['source_strength']:.4f}"
                )
            return

        workers = min(self.dataset.num_workers, max(1, os.cpu_count() or 1))
        mp_ctx = get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            if self.dataset.paired_inputs.enabled:
                futures = {
                    ex.submit(
                        _generate_paired_source_worker,
                        idx,
                        self.run_seed,
                        str(self.dataset.paired_inputs.master_source_config),
                        str(self.source_dir),
                        self.dataset.source_strength_range,
                        self.dataset.paired_inputs,
                    ): idx
                    for idx in pending
                }
            else:
                futures = {
                    ex.submit(
                        _generate_source_worker,
                        idx,
                        self.run_seed,
                        str(self.source_cfg_path),
                        str(self.source_dir),
                        self.dataset.source_strength_range,
                    ): idx
                    for idx in pending
                }

            for fut in as_completed(futures):
                rec = fut.result()
                done += 1
                print(
                    f"[source {done:06d}/{len(pending):06d}] "
                    f"sample={rec['sample_index']:06d} type={rec['source_type']:<11} amp={rec['source_strength']:.4f}"
                )

    def _validate_authoritative_caches(self, indices: list[int]) -> None:
        config = self.dataset.authoritative_inputs
        requested = self.dataset.requested_output
        if config is None:
            return
        if requested is None:
            raise RuntimeError("Authoritative inputs require requested-output mode")
        for sample_idx in indices:
            bathymetry_path = _bathymetry_file_path(
                str(self.bathymetry_dir), sample_idx
            )
            source_path = _source_file_path(str(self.source_dir), sample_idx)
            with np.load(bathymetry_path, allow_pickle=False) as payload:
                bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
                bathymetry_type = str(
                    np.asarray(payload["bathymetry_type"]).reshape(-1)[0]
                )
                bathymetry_seed = int(
                    np.asarray(payload["sample_seed"]).reshape(-1)[0]
                )
            with np.load(source_path, allow_pickle=False) as payload:
                source_field = np.asarray(payload["source_field"], dtype=np.float32)
                source_type = str(
                    np.asarray(payload["source_type"]).reshape(-1)[0]
                )
                source_strength = np.asarray(payload["source_strength"])
                source_seed = int(
                    np.asarray(payload["sample_seed"]).reshape(-1)[0]
                )
            expected_seed = _seed_for_sample(self.run_seed, sample_idx)
            if bathymetry_seed != expected_seed or source_seed != expected_seed:
                raise RuntimeError(
                    f"Authoritative input seed mismatch for sample {sample_idx}"
                )
            _validate_authoritative_input(
                record=self.authoritative_records[sample_idx],
                split=requested.split,
                sample_idx=sample_idx,
                scenario_id=f"scenario_{sample_idx:06d}",
                bathymetry=bathymetry,
                source_field=source_field,
                source_strength_array=source_strength,
                bathymetry_type=bathymetry_type,
                source_type=source_type,
                sea_level_offset=self.dataset.sea_level_offset,
                config=config,
            )
        print(
            f"[dataset] exact H0 input verification passed: "
            f"split={requested.split} samples={len(indices)}"
        )

    def _validate_paired_cache(
        self, sample_idx: int
    ) -> dict[str, Any]:
        paired = self.dataset.paired_inputs
        if not paired.enabled:
            raise RuntimeError("paired cache validation requires paired_inputs")
        bathymetry_path = _bathymetry_file_path(self.bathymetry_dir, sample_idx)
        source_path = _source_file_path(self.source_dir, sample_idx)
        expected_seed = _seed_for_sample(self.run_seed, sample_idx)

        def _text(payload: Mapping[str, np.ndarray], key: str) -> str:
            if key not in payload:
                raise RuntimeError(f"paired input cache is missing {key}")
            return str(np.asarray(payload[key]).reshape(-1)[0])

        def _validate_common(payload: Mapping[str, np.ndarray]) -> None:
            if _text(payload, "native_input_schema_id") != NATIVE_INPUT_SCHEMA_ID:
                raise RuntimeError("paired input schema mismatch")
            if _text(payload, "native_lineage_id") != paired.lineage_id:
                raise RuntimeError("paired input lineage id mismatch")
            if _text(payload, "native_lineage_hash") != paired.lineage_hash:
                raise RuntimeError("paired input lineage hash mismatch")
            if (
                _text(payload, "native_target_contract_hash")
                != paired.target_contract_hash
            ):
                raise RuntimeError("paired input target contract mismatch")
            if _text(payload, "native_downsample_method") != paired.downsample_method:
                raise RuntimeError("paired input downsample method mismatch")
            if tuple(np.asarray(payload["native_master_shape"], dtype=int)) != (
                paired.master_shape
            ):
                raise RuntimeError("paired input master shape mismatch")
            if tuple(np.asarray(payload["native_target_shape"], dtype=int)) != (
                paired.target_shape
            ):
                raise RuntimeError("paired input target shape mismatch")
            seed = int(np.asarray(payload["sample_seed"]).reshape(-1)[0])
            if seed != expected_seed:
                raise RuntimeError("paired input sample seed mismatch")

        with np.load(bathymetry_path, allow_pickle=False) as payload:
            _validate_common(payload)
            bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
            master_bathymetry = np.asarray(
                payload["master_bathymetry"], dtype=np.float32
            )
            bathymetry_type = _text(payload, "bathymetry_type")
            bathymetry_master_hash = hash_array(master_bathymetry)["sha256"]
            if bathymetry_master_hash != _text(
                payload, "native_master_array_sha256"
            ):
                raise RuntimeError("paired master bathymetry hash mismatch")
            expected_bathymetry = _block_mean_downsample(
                master_bathymetry, paired.target_shape
            )
            if not np.array_equal(bathymetry, expected_bathymetry):
                raise RuntimeError(
                    "paired target bathymetry is not the exact master-grid reduction"
                )

        with np.load(source_path, allow_pickle=False) as payload:
            _validate_common(payload)
            source_field = np.asarray(payload["source_field"], dtype=np.float32)
            master_source = np.asarray(
                payload["master_source_field"], dtype=np.float32
            )
            source_type = _text(payload, "source_type")
            source_strength_array = np.asarray(payload["source_strength"])
            source_strength = float(source_strength_array.reshape(-1)[0])
            source_master_hash = hash_array(master_source)["sha256"]
            if source_master_hash != _text(
                payload, "native_master_array_sha256"
            ):
                raise RuntimeError("paired master source hash mismatch")
            expected_source = _block_mean_downsample(
                master_source, paired.target_shape
            )
            if not np.array_equal(source_field, expected_source):
                raise RuntimeError(
                    "paired target source is not the exact master-grid reduction"
                )
        lo, hi = self.dataset.source_strength_range
        if not np.isfinite(source_strength) or not lo <= source_strength <= hi:
            raise RuntimeError("paired input source strength is out of range")

        rest_depth = np.maximum(
            -bathymetry + self.dataset.sea_level_offset, 0.0
        ).astype(np.float32, copy=False)
        eta0 = np.asarray(source_strength * source_field, dtype=np.float32)
        initial_depth = np.maximum(rest_depth + eta0, 0.0).astype(
            np.float32, copy=False
        )
        free_surface0 = np.asarray(initial_depth + bathymetry, dtype=np.float32)

        master_rest_depth = np.maximum(
            -master_bathymetry + self.dataset.sea_level_offset, 0.0
        ).astype(np.float32, copy=False)
        master_eta0 = np.asarray(source_strength * master_source, dtype=np.float32)
        master_initial_depth = np.maximum(
            master_rest_depth + master_eta0, 0.0
        ).astype(np.float32, copy=False)
        master_free_surface0 = np.asarray(
            master_initial_depth + master_bathymetry, dtype=np.float32
        )
        requested = self.dataset.requested_output
        if requested is None:
            raise RuntimeError("paired native inputs require requested_output")
        scenario_id = f"scenario_{sample_idx:06d}"
        target_arrays = {
            "bathymetry": bathymetry,
            "source_field": source_field,
            "rest_depth": rest_depth,
            "eta0": eta0,
            "initial_depth": initial_depth,
            "free_surface0": free_surface0,
        }
        master_arrays = {
            "bathymetry": master_bathymetry,
            "source_field": master_source,
            "rest_depth": master_rest_depth,
            "eta0": master_eta0,
            "initial_depth": master_initial_depth,
            "free_surface0": master_free_surface0,
        }
        target_fingerprint = authoritative_input_fingerprint(
            split=requested.split,
            sample_index=sample_idx,
            scenario_id=scenario_id,
            bathymetry_type=bathymetry_type,
            source_type=source_type,
            source_strength=source_strength_array,
            arrays=target_arrays,
        )
        master_fingerprint = authoritative_input_fingerprint(
            split=requested.split,
            sample_index=sample_idx,
            scenario_id=scenario_id,
            bathymetry_type=bathymetry_type,
            source_type=source_type,
            source_strength=source_strength_array,
            arrays=master_arrays,
        )
        return {
            "schema_id": NATIVE_INPUT_SCHEMA_ID,
            "lineage_id": paired.lineage_id,
            "lineage_hash": paired.lineage_hash,
            "target_contract_hash": paired.target_contract_hash,
            "master_shape": list(paired.master_shape),
            "target_shape": list(paired.target_shape),
            "downsample_method": paired.downsample_method,
            "split": requested.split,
            "qualified_id": f"{requested.split}:{scenario_id}",
            "scenario_id": scenario_id,
            "sample_index": sample_idx,
            "sample_seed": expected_seed,
            "bathymetry_type": bathymetry_type,
            "source_type": source_type,
            "source_strength": source_strength,
            "target_input_fingerprint": target_fingerprint,
            "master_input_fingerprint": master_fingerprint,
            "target_array_hashes": {
                name: hash_array(values) for name, values in target_arrays.items()
            },
            "master_array_hashes": {
                name: hash_array(values) for name, values in master_arrays.items()
            },
            "bathymetry_cache_path": str(bathymetry_path),
            "source_cache_path": str(source_path),
        }

    def _freeze_paired_input_inventory(self, indices: list[int]) -> str:
        paired = self.dataset.paired_inputs
        if not paired.enabled or paired.inventory_path is None:
            raise RuntimeError("paired input inventory path is unavailable")
        records = [self._validate_paired_cache(index) for index in indices]
        content = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
        path = paired.inventory_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    f"Frozen paired input inventory mismatch: {path}. "
                    "Use a new lineage_id and empty output paths for an intentional new lineage."
                )
        else:
            staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
            try:
                staging.write_text(content, encoding="utf-8")
                os.replace(staging, path)
            finally:
                staging.unlink(missing_ok=True)
        checksum = sha256_file(path)
        print(
            f"[dataset] paired native input inventory frozen: "
            f"samples={len(records)} sha256={checksum}"
        )
        return checksum

    def _phase_generate_rollouts(
        self, indices: list[int], allow_override: bool = False
    ) -> list[Dict[str, Any]]:
        print(
            f"[dataset] phase 3/3 run FDEs={list(self.dataset.enabled_fdes)} "
            f"on samples={len(indices)}"
        )

        records: list[Dict[str, Any]] = []
        phase_started = time.monotonic()
        recorder = getattr(self, "_operational_recorder", None)

        def record_complete(rec: Dict[str, Any], done: int) -> None:
            if recorder is not None:
                recorder.record_sample(rec)
                progress = recorder.progress(completed=done, total=len(indices))
            else:
                elapsed = max(0.0, time.monotonic() - phase_started)
                rate = done / elapsed if done > 0 and elapsed > 0.0 else None
                progress = {
                    "elapsed_s": elapsed,
                    "rate_per_s": rate,
                    "eta_s": (
                        (len(indices) - done) / rate
                        if rate is not None and done < len(indices)
                        else 0.0 if done >= len(indices) else None
                    ),
                }
            if done % self.operations.progress_every != 0 and done != len(indices):
                return
            solver_status = {
                str(row.get("solver")): str(row.get("status"))
                for row in rec.get("_operational", {}).get("solvers", [])
            }
            elapsed_s = float(progress["elapsed_s"] or 0.0)
            eta_value = progress["eta_s"]
            eta_text = "unknown" if eta_value is None else f"{float(eta_value):.1f}s"
            rate_value = progress["rate_per_s"]
            rate_text = (
                "unknown"
                if rate_value is None
                else f"{float(rate_value) * 3600.0:.2f} scenarios/h"
            )
            print(
                f"[rollout {done:06d}/{len(indices):06d}] "
                f"sample={int(rec['sample_index']):06d} "
                f"scenario={rec['scenario_id']} status={solver_status} "
                f"elapsed={elapsed_s:.1f}s rate={rate_text} ETA={eta_text}",
                flush=True,
            )

        if self.dataset.num_workers <= 1:
            done = 0
            for idx in indices:
                try:
                    rec = _generate_sample_worker(
                        sample_idx=idx,
                        run_seed=self.run_seed,
                        dataset=self.dataset,
                        solver_cfg=self.solver_cfg,
                        source_cfg_path=str(self.source_cfg_path),
                        config_path=str(self.config_path),
                        bathy_cfg_path=str(self.bathy_cfg_path),
                        bathymetry_dir=str(self.bathymetry_dir),
                        source_dir=str(self.source_dir),
                        fde_samples_dirs={
                            k: str(v) for k, v in self.fde_samples_dirs.items()
                        },
                        authoritative_record=self.authoritative_records.get(idx),
                        source_code=self.source_code,
                        emit_solver_progress=(
                            self.operations.enabled and self.operations.solver_progress
                        ),
                        allow_override=allow_override,
                    )
                except BaseException as exc:
                    if recorder is not None:
                        recorder.record_failure(idx, exc)
                    raise
                records.append(rec)
                done += 1
                record_complete(rec, done)
            return records

        workers = min(self.dataset.num_workers, max(1, os.cpu_count() or 1))
        max_in_flight = self.operations.max_in_flight or 2 * workers
        max_in_flight = min(len(indices), max(workers, int(max_in_flight)))
        mp_ctx = get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            pending = iter(indices)
            futures: dict[Any, int] = {}

            def submit_until_full() -> None:
                while len(futures) < max_in_flight:
                    try:
                        idx = next(pending)
                    except StopIteration:
                        return
                    future = ex.submit(
                        _generate_sample_worker,
                        idx,
                        self.run_seed,
                        self.dataset,
                        self.solver_cfg,
                        str(self.source_cfg_path),
                        str(self.config_path),
                        str(self.bathy_cfg_path),
                        str(self.bathymetry_dir),
                        str(self.source_dir),
                        {k: str(v) for k, v in self.fde_samples_dirs.items()},
                        self.authoritative_records.get(idx),
                        self.source_code,
                        self.operations.enabled and self.operations.solver_progress,
                        allow_override,
                    )
                    futures[future] = idx

            submit_until_full()
            while futures:
                fut = next(as_completed(tuple(futures)))
                idx = futures.pop(fut)
                try:
                    rec = fut.result()
                except BaseException as exc:
                    if recorder is not None:
                        recorder.record_failure(idx, exc)
                    raise
                records.append(rec)
                done += 1
                record_complete(rec, done)
                submit_until_full()

        return records

    def _write_operational_shard_manifest(
        self, records: list[Dict[str, Any]], indices: list[int]
    ) -> Path | None:
        requested = self.dataset.requested_output
        if requested is None or not records:
            return None
        from src.data_gen.common_time_v2 import write_operational_shard_manifest

        publication_hashes: dict[str, str] = {}
        resolved_config_hashes: dict[str, str] = {}
        solver_names: set[str] = set()
        code_hashes: set[str] = set()
        for record in records:
            scenario_id = str(record["scenario_id"])
            for solver_record in record.get("solver_records", []):
                solver = str(solver_record.get("solver_name", "unknown"))
                solver_names.add(solver)
                resolved_config_hashes[solver] = str(
                    solver_record.get("resolved_config_hash", "")
                )
                if solver_record.get("code_state_hash"):
                    code_hashes.add(str(solver_record["code_state_hash"]))
                sample_dir = Path(str(solver_record["sample_dir"]))
                publication = validate_publication(
                    sample_dir,
                    expected_identity=split_qualified_identity(
                        requested.split, scenario_id
                    ),
                    expected_contract_hash=requested.contract_hash,
                    expected_times=requested.requested_times,
                )
                key = f"{requested.split}:{scenario_id}:{solver}"
                publication_hashes[key] = stable_hash_payload(
                    artifact_kind="requested-output-publication-record",
                    payload=publication,
                    schema_id=PUBLICATION_SCHEMA_ID,
                )
        if len(code_hashes) != 1:
            raise RuntimeError("Operational shard publications do not share one code state")
        code_hash = next(iter(code_hashes))
        shard_dir = self.output_dir / "operational_shards"
        path = shard_dir / (
            f"{requested.split}_{min(indices):06d}_{max(indices):06d}.json"
        )
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            observed = {
                str(item["qualified_id"]): str(item["publication_hash"])
                for item in existing.get("publications", [])
            }
            if (
                existing.get("contract_hash") != requested.contract_hash
                or existing.get("start_index") != min(indices)
                or existing.get("stop_index") != max(indices)
                or existing.get("solver_names") != sorted(solver_names)
                or existing.get("resolved_config_hashes") != resolved_config_hashes
                or existing.get("code_state_hash") != code_hash
                or observed != publication_hashes
                or not bool(existing.get("complete", False))
            ):
                raise RuntimeError(f"Operational shard manifest mismatch: {path}")
            return path
        write_operational_shard_manifest(
            path,
            split=requested.split,
            start_index=min(indices),
            stop_index=max(indices),
            contract_hash_value=requested.contract_hash,
            publication_hashes=publication_hashes,
            complete=True,
            solver_names=sorted(solver_names),
            resolved_config_hashes=resolved_config_hashes,
            code_state_hash=code_hash,
        )
        return path

    def run(
        self,
        continue_from_last: bool = False,
        start_at: int | None = None,
        stop_at: int | None = None,
        allow_override: bool = False,
        rebuild_manifests: bool = False,
        acknowledge_provisional: bool = False,
    ) -> None:
        """Generate samples and checkpoint requested-output operational timing."""
        requested = self.dataset.requested_output
        recorder: GenerationTimingRecorder | None = None
        if (
            requested is not None
            and self.operations.enabled
            and not rebuild_manifests
        ):
            current_code = code_state(ROOT)
            if current_code != self.source_code:
                raise RuntimeError("Code state changed after dataset builder initialization")
            recorder = GenerationTimingRecorder(
                output_dir=self.output_dir,
                split=requested.split,
                contract_hash=requested.contract_hash,
                code_state_hash=str(self.source_code["code_state_hash"]),
                config_path=self.provenance_config_path,
                config_sha256=sha256_file(self.config_path),
                solver_names=self.dataset.enabled_fdes,
                requested_workers=self.dataset.num_workers,
                requested_max_in_flight=self.operations.max_in_flight,
                operational_config=self.operations.metadata(),
            )
        self._operational_recorder = recorder
        try:
            self._run_impl(
                continue_from_last=continue_from_last,
                start_at=start_at,
                stop_at=stop_at,
                allow_override=allow_override,
                rebuild_manifests=rebuild_manifests,
                acknowledge_provisional=acknowledge_provisional,
            )
            if code_state(ROOT) != self.source_code:
                raise RuntimeError("Code state changed during dataset generation")
        except BaseException as exc:
            if recorder is not None:
                recorder.finalize(status="failed", error=exc)
            raise
        else:
            if recorder is not None:
                recorder.finalize(status="complete")
        finally:
            self._operational_recorder = None

    def _run_impl(
        self,
        continue_from_last: bool = False,
        start_at: int | None = None,
        stop_at: int | None = None,
        allow_override: bool = False,
        rebuild_manifests: bool = False,
        acknowledge_provisional: bool = False,
    ) -> None:
        """generate all raw samples in three phases: bathymetry, source, and FDE rollouts."""
        if (
            self.dataset.requested_output is not None
            and self.dataset.requested_output.status == "provisional"
            and not (
                acknowledge_provisional
                or self.dataset.requested_output.acknowledged_provisional
            )
        ):
            raise RuntimeError(
                "The common-time-v2 requested-output contract is provisional. "
                "Pass --acknowledge-provisional only for explicitly approved preparation runs."
            )
        if rebuild_manifests:
            self.rebuild_manifests_from_existing_outputs()
            return
        if allow_override and (
            self.dataset.authoritative_inputs is not None
            or self.dataset.paired_inputs.enabled
        ):
            raise RuntimeError(
                "--allow-override is forbidden for frozen common-time-v2 inputs"
            )

        if start_at is not None and start_at < 1:
            raise ValueError("--start-at must be >= 1")
        if stop_at is not None and stop_at < 1:
            raise ValueError("--stop-at must be >= 1")

        total = self.dataset.num_samples
        existing_indices = self._existing_sample_indices()
        any_existing_indices = self._existing_any_sample_indices()

        if start_at is not None:
            start_idx = int(start_at)
        elif continue_from_last and self.dataset.requested_output is None:
            start_idx = self._resume_rollback_start_index(existing_indices)
        else:
            start_idx = 1

        clean_run = start_idx == 1 and not continue_from_last and start_at is None
        if clean_run and any_existing_indices and not allow_override:
            raise RuntimeError(
                "Existing sample outputs found. Use --continue, --allow-override, or --rebuild-manifests "
                "to avoid deleting/invalidating manifests."
            )
        if not clean_run:
            print(f"[dataset] resume mode: start_at={start_idx}")

        if start_idx > total:
            print(
                f"[dataset] nothing to do: start_at={start_idx} > num_samples={total}"
            )
            return

        range_stop = total if stop_at is None else min(int(stop_at), total)
        if range_stop < start_idx:
            raise ValueError("--stop-at must be >= the resolved start index")
        planned_indices = list(range(start_idx, range_stop + 1))
        recorder = getattr(self, "_operational_recorder", None)
        if recorder is not None:
            recorder.begin_range(
                start_index=start_idx,
                stop_index=range_stop,
                planned_scenarios=len(planned_indices),
                resume=continue_from_last,
                allow_override=allow_override,
            )
        if allow_override:
            to_generate = planned_indices
            existing_in_range = len(
                [idx for idx in planned_indices if idx in existing_indices]
            )
            if existing_in_range > 0:
                print(
                    f"[dataset] allow-override: regenerating {existing_in_range} existing samples in range"
                )
        else:
            to_generate = [
                idx for idx in planned_indices if idx not in existing_indices
            ]
            skipped = len(planned_indices) - len(to_generate)
            if skipped > 0:
                print(
                    f"[dataset] skipping {skipped} existing samples (already present on disk)"
                )

        if not to_generate:
            print("[dataset] no missing samples; validating completed range.")
        else:
            if clean_run and (allow_override or not any_existing_indices):
                if self.scenario_manifest_path.exists():
                    self.scenario_manifest_path.unlink()
                for path in self.fde_manifest_paths.values():
                    if path.exists():
                        path.unlink()

            print(
                f"[dataset] generation plan: samples={len(to_generate)}, "
                f"range=[{to_generate[0]}, {to_generate[-1]}], seed={self.run_seed}"
            )

        static_indices = (
            list(range(1, total + 1))
            if self.dataset.paired_inputs.enabled
            else to_generate
        )
        if static_indices:
            if recorder is not None:
                recorder.start_phase("bathymetry")
            try:
                self._phase_generate_bathymetry(
                    static_indices, allow_override=allow_override
                )
            finally:
                if recorder is not None:
                    recorder.end_phase("bathymetry")
            if recorder is not None:
                recorder.start_phase("sources")
            try:
                self._phase_generate_sources(
                    static_indices, allow_override=allow_override
                )
            finally:
                if recorder is not None:
                    recorder.end_phase("sources")

        if self.dataset.paired_inputs.enabled:
            inventory_sha256 = self._freeze_paired_input_inventory(static_indices)
            self.dataset.paired_input_inventory_sha256 = inventory_sha256

        self._validate_authoritative_caches(planned_indices)

        if recorder is not None:
            recorder.start_phase("rollouts")
        try:
            records = self._phase_generate_rollouts(
                planned_indices, allow_override=allow_override
            )
        finally:
            if recorder is not None:
                recorder.end_phase("rollouts")
        if recorder is not None:
            recorder.start_phase("publication_audit_and_shard")
        try:
            shard_path = self._write_operational_shard_manifest(
                records, planned_indices
            )
        finally:
            if recorder is not None:
                recorder.end_phase("publication_audit_and_shard")
        if recorder is not None and shard_path is not None:
            recorder.set_shard_manifest(shard_path)

        if recorder is not None:
            recorder.start_phase("manifest_update")
        records.sort(key=lambda r: int(r["sample_index"]))
        try:
            if allow_override or not clean_run:
                sample_indices = set(int(r["sample_index"]) for r in records)
                self._purge_manifest_indices(self.scenario_manifest_path, sample_indices)
                for manifest_path in self.fde_manifest_paths.values():
                    self._purge_manifest_indices(manifest_path, sample_indices)

            for rec in records:
                self._append_manifest(
                    self.scenario_manifest_path, rec["scenario_record"]
                )
                for srec in rec.get("solver_records", []):
                    solver_name = str(srec.get("solver_name", "unknown"))
                    if solver_name in self.fde_manifest_paths:
                        self._append_manifest(
                            self.fde_manifest_paths[solver_name], srec
                        )
        finally:
            if recorder is not None:
                recorder.end_phase("manifest_update")

if __name__ == "__main__":
    raise SystemExit(
        "src/data_gen/simulate_dataset.py is an internal module. "
        "Use python scripts/make_dataset.py --help."
    )
