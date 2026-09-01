#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.generate_bathymetry import BathymetryGenerator  # noqa: E402
from src.data_gen.common_time_v2 import (  # noqa: E402
    code_state,
    hash_array,
    parse_requested_output_config,
    sha256_file,
    stable_hash_payload,
)
from src.data_gen.simulate_dataset import (  # noqa: E402
    BufferedDomainConfig,
    QualityPolicy,
    RolloutResult,
    TsunamiDatasetBuilder,
    _block_mean_downsample,
    _block_mean_downsample_spatial,
    _compute_rollout_health,
    _generate_paired_source_fields,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
    _quality_violations_for_health,
    _requested_health_summary,
    _resolved_solver_cfg_for_fde,
    _seed_for_sample,
    _simulate_one_local,
)


PILOT_SCHEMA_ID = "tsunami-surrogate.aspect2400-rebuild-pilot.v1"
CONFIGS = {
    "train": ROOT / "configs/data/rebuild/dataset_train_64.yaml",
    "confirmation": ROOT
    / "configs/data/rebuild/dataset_validation_64.yaml",
    "test": ROOT / "configs/data/rebuild/dataset_test_64.yaml",
}
SOURCE_FAMILIES = (
    "gaussian",
    "multi-gauss",
    "okada-like",
    "dipole",
    "fault",
    "rough",
)
BATHYMETRY_FAMILIES = (
    "trench",
    "continental",
    "seamounts",
    "canyon",
    "island",
)
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
PUBLICATION_RESOLUTION = 64
PRODUCTION_RESOLUTION = 128
REFINEMENT_RESOLUTION = 192
MASTER_RESOLUTION = 384
RESOLUTIONS = {
    128: {"buffer": 32, "taper": 16, "solver": 192, "dx": 18.75},
    192: {"buffer": 48, "taper": 24, "solver": 288, "dx": 12.5},
    384: {"buffer": 96, "taper": 48, "solver": 576, "dx": 6.25},
}
RESOLUTION_CFL_FACTORS = {
    PRODUCTION_RESOLUTION: (1.0, 0.5),
    REFINEMENT_RESOLUTION: (1.0,),
}
TARGETED_GAUGE_DIAGNOSTIC_SAMPLES = (727, 6398)
TARGETED_GAUGE_DIAGNOSTIC_RESOLUTIONS = (
    REFINEMENT_RESOLUTION,
    MASTER_RESOLUTION,
)
GAUGE_GLOBAL_SIGNAL_FLOOR_FRACTION = 0.01
GAUGE_ARRIVAL_PEAK_FRACTION = 0.10

INPUT_GATES = {
    "min_points_per_p90_wavelength": 24.0,
    "min_temporal_samples_per_p90_period": 5.5,
    "temporal_samples_per_p90_period_p01_min": 6.0,
    "high_frequency_fraction_p95_max": 0.01,
    "high_frequency_fraction_absolute_max": 0.05,
    "kh_p90_absolute_max": 1.0,
    "energy_fraction_kh_gt_1_absolute_max": 0.10,
    "local_eta_over_depth_p95_max": 0.10,
    "local_eta_over_depth_absolute_max": 0.10,
    "amplitude_cap_fraction_max": 0.10,
    "amplitude_cap_fraction_per_family_max": 0.25,
    "rough_abs_mean_over_rms_max": 1.0e-6,
    "rough_rms_relative_error_max": 1.0e-6,
    "bathymetry_depth_min": -10.0,
    "bathymetry_depth_max": -0.75,
    "pairing_bathymetry_max_abs": 1.0e-6,
    "pairing_source_max_abs": 1.0e-7,
    "pairing_eta0_max_abs": 1.0e-7,
}
ROLLOUT_GATES = {
    "half_cfl_rel_l2_median_max": 0.01,
    "half_cfl_rel_l2_p95_max": 0.02,
    "half_cfl_rel_l2_absolute_max": 0.05,
    "refinement_rel_l2_median_max": 0.05,
    "refinement_rel_l2_p95_max": 0.10,
    "refinement_rel_l2_absolute_max": 0.15,
    "per_frame_initial_nrmse_p95_max": 0.10,
    "per_frame_initial_nrmse_absolute_max": 0.15,
    "gauge_nrmse_p95_max": 0.10,
    "gauge_nrmse_absolute_max": 0.15,
    "gauge_peak_relative_error_p95_max": 0.10,
    "gauge_peak_relative_error_absolute_max": 0.15,
    "gauge_waveform_lag_frame_p95_max": 1.0,
    "gauge_waveform_lag_frame_absolute_max": 2.0,
    "eligible_waveform_lag_gauges_min": 3,
    "trajectory_high_frequency_fraction_max": 0.05,
    "half_cfl_high_frequency_fraction_delta_max": 0.005,
    "refinement_high_frequency_fraction_excess_max": 0.005,
}
LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS = (
    "gauge_nrmse_p95",
    "gauge_nrmse_max",
    "gauge_peak_p95",
    "gauge_peak_max",
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seed: int
    count: int
    config_path: Path


def _load_split(name: str) -> SplitSpec:
    path = CONFIGS[name]
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SplitSpec(
        name=name,
        seed=int(cfg["dataset"]["seed"]),
        count=int(cfg["dataset"]["num_samples"]),
        config_path=path,
    )


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return raw


def _requested_config(cfg: dict[str, Any]) -> Any:
    requested = parse_requested_output_config(cfg.get("requested_output"))
    if requested is None:
        raise RuntimeError("rebuild pilot requires requested_output")
    return requested


def _quality_policy(cfg: dict[str, Any]) -> QualityPolicy:
    raw = cfg.get("quality", {})
    if not isinstance(raw, dict):
        raise ValueError("quality config must be a mapping")

    def _optional_float(key: str) -> float | None:
        value = raw.get(key)
        return None if value is None else float(value)

    return QualityPolicy(
        on_violation=str(raw.get("on_violation", "warn")).strip().lower(),
        reject_nonfinite=bool(raw.get("reject_nonfinite", True)),
        min_h_tolerance=_optional_float("min_h_tolerance"),
        max_abs_eta_limit=_optional_float("max_abs_eta_limit"),
        max_velocity_limit=_optional_float("max_velocity_limit"),
        max_eta_over_depth=_optional_float("max_eta_over_depth"),
        require_cg_converged=bool(raw.get("require_cg_converged", True)),
        reject_sanitization=bool(raw.get("reject_sanitization", False)),
        require_no_projection=bool(raw.get("require_no_projection", False)),
        require_no_velocity_clipping=bool(
            raw.get("require_no_velocity_clipping", False)
        ),
        max_post_step_cfl_ratio=_optional_float(
            "max_post_step_cfl_ratio"
        ),
    )


def _assert_split_contract(split: SplitSpec) -> None:
    cfg = _load_config(split.config_path)
    requested = _requested_config(cfg)
    if (
        requested.status != "accepted"
        or requested.execution_scope != "production"
        or requested.acknowledged_provisional
    ):
        raise RuntimeError(
            f"{split.name} rebuild contract is not accepted for production"
        )
    expected_times = 8.4 + 8.4 * np.arange(50, dtype=np.float64)
    expected_times[-1] = np.float64(420.0)
    if not np.array_equal(requested.requested_times, expected_times):
        raise RuntimeError(
            f"{split.name} requested times differ from the preregistered pilot"
        )
    paired = cfg["paired_inputs"]
    domain = cfg["computational_domain"]
    solver = cfg["solver"]
    expected = {
        "master_shape": [384, 384],
        "paired_solver_shape": [128, 128],
        "target_shape": [64, 64],
        "solver_input": "solver",
        "source_taper_stage": "master",
        "rough_zero_mean_rms_after_taper": True,
        "source_spectral_acceptance": {
            "enabled": True,
            "stage": "post_master_taper_solver",
            "reference_shape": [128, 128],
            "min_points_per_p90_wavelength": 32.0,
            "comparison_tolerance": 1.0e-9,
            "preserve_source_family": True,
            "max_attempts": 64,
        },
        "buffer_cells": 32,
        "source_taper_cells": 16,
        "computational_shape": [192, 192],
        "solver_spacing": [18.75, 18.75],
        "max_initial_eta_over_depth": 0.10,
        "source_strength_range": [0.15, 0.30],
    }
    observed = {
        "master_shape": list(paired["master_shape"]),
        "paired_solver_shape": list(paired["solver_shape"]),
        "target_shape": list(paired["target_shape"]),
        "solver_input": str(paired["solver_input"]),
        "source_taper_stage": str(paired["source_taper_stage"]),
        "rough_zero_mean_rms_after_taper": bool(
            paired["rough_zero_mean_rms_after_taper"]
        ),
        "source_spectral_acceptance": dict(
            paired["source_spectral_acceptance"]
        ),
        "buffer_cells": int(domain["buffer_cells"]),
        "source_taper_cells": int(domain["source_taper_cells"]),
        "computational_shape": [int(solver["nx"]), int(solver["ny"])],
        "solver_spacing": [float(solver["dx"]), float(solver["dy"])],
        "max_initial_eta_over_depth": float(
            cfg["dataset"]["max_initial_eta_over_depth"]
        ),
        "source_strength_range": [
            float(value)
            for value in cfg["dataset"]["source_strength_range"]
        ],
    }
    if observed != expected:
        raise RuntimeError(
            f"{split.name} rebuild contract drifted: {observed} != {expected}"
        )
    if tuple(cfg["fdes"]["enabled"]) != SOLVERS:
        raise RuntimeError(f"{split.name} solver roster drifted")
    if requested.max_natural_steps != 20000:
        raise RuntimeError(f"{split.name} natural-step cap drifted")
    if _quality_policy(cfg).on_violation != "fail":
        raise RuntimeError(f"{split.name} quality policy must fail closed")


def _pilot_contract() -> dict[str, Any]:
    config_paths = {
        **CONFIGS,
        "bathymetry_128": ROOT
        / "configs/data/rebuild/bathymetry_128.yaml",
        "source_128": ROOT / "configs/data/rebuild/source_128.yaml",
        "bathymetry_384": ROOT
        / "configs/data/rebuild/bathymetry_384.yaml",
        "source_384": ROOT / "configs/data/rebuild/source_384.yaml",
    }
    payload = {
        "schema_id": PILOT_SCHEMA_ID,
        "code_state": code_state(ROOT),
        "configs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(config_paths.items())
        },
        "input_gates": INPUT_GATES,
        "rollout_gates": ROLLOUT_GATES,
        "promotion_policy": {
            "local_amplitude_role": "reported_resolution_diagnostic_nonblocking",
            "local_amplitude_checks": list(
                LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS
            ),
            "gauge_position_policy": "fixed_5x5_lattice",
            "rationale": (
                "the publication target is a full cell-averaged field; local "
                "point-amplitude refinement remains disclosed separately "
                "because the low-order Hydrostatic reference has a genuine "
                "resolution-dependent dissipative tail"
            ),
            "confirmation_is_independent": True,
        },
        "selection_policy": {
            "bathymetry_families": list(BATHYMETRY_FAMILIES),
            "source_families": list(SOURCE_FAMILIES),
            "selection_uses_solver_outputs": False,
            "train_cases_per_family_cell": 2,
            "confirmation_cases_per_family_cell": 1,
            "mini_cases_per_source_family": {
                "spectrum_temporal_extreme": 1,
                "amplitude_bathymetry_extreme": 1,
            },
        },
        "resolution_policy": {
            "shared_master_resolution": MASTER_RESOLUTION,
            "publication_resolution": PUBLICATION_RESOLUTION,
            "production_resolution": PRODUCTION_RESOLUTION,
            "refinement_resolution": REFINEMENT_RESOLUTION,
            "shared_master_downsample_method": "block_mean_float64_v1",
            "resolution_cfl_factors": {
                str(resolution): list(factors)
                for resolution, factors in RESOLUTION_CFL_FACTORS.items()
            },
        },
        "final_test_seed": 911,
    }
    contract_hash = stable_hash_payload(
        artifact_kind="aspect2400-rebuild-pilot-contract",
        payload=payload,
        schema_id=PILOT_SCHEMA_ID,
    )
    return {**payload, "contract_hash": contract_hash}


def _bind_pilot_root(root: Path, contract: dict[str, Any]) -> None:
    path = root / "pilot_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(
                "pilot root is bound to a different code/config contract; "
                "use a fresh output root"
            )
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            "non-empty pilot root has no contract; use a fresh output root"
        )
    root.mkdir(parents=True, exist_ok=True)
    _write_json(path, contract)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    try:
        staging.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    try:
        with staging.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _safe_rel_l2(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-30))


def _amplitude_cap_applied(
    resolved_strength: float,
    sampled_strength: float,
) -> bool:
    return bool(
        np.float32(resolved_strength) < np.float32(sampled_strength)
    )


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    fraction: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("spectral weights are empty or non-finite")
    index = int(np.searchsorted(cumulative, fraction * total))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _spectral_metrics(
    source_field: np.ndarray,
    eta0: np.ndarray,
    bathymetry: np.ndarray,
    *,
    dx: float,
) -> dict[str, float]:
    source = np.asarray(source_field, dtype=np.float64)
    centered = source - float(np.mean(source))
    power = np.abs(np.fft.fft2(centered)) ** 2
    kx = 2.0 * np.pi * np.fft.fftfreq(source.shape[0], d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(source.shape[1], d=dx)
    k_grid = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    mask = k_grid > 0.0
    k_values = k_grid[mask]
    weights = power[mask]
    k90 = _weighted_quantile(k_values, weights, 0.90)
    depth = np.maximum(-np.asarray(bathymetry, dtype=np.float64), 1.0e-4)
    eta = np.asarray(eta0, dtype=np.float64)
    eta_energy = eta * eta
    effective_depth = float(
        np.sum(depth * eta_energy) / (np.sum(eta_energy) + 1.0e-30)
    )
    total_power = float(np.sum(weights))
    return {
        "points_per_p90_wavelength": float(
            2.0 * np.pi / max(k90 * dx, 1.0e-30)
        ),
        "high_frequency_fraction": float(
            np.sum(weights[(k_values * dx) > (np.pi / 2.0)])
            / total_power
        ),
        "kh_p90": float(k90 * effective_depth),
        "energy_fraction_kh_gt_1": float(
            np.sum(weights[(k_values * effective_depth) > 1.0])
            / total_power
        ),
        "effective_depth": effective_depth,
    }


def _downsample_shared_master(
    values: np.ndarray,
    target_resolution: int,
) -> np.ndarray:
    master = np.asarray(values)
    if master.shape != (MASTER_RESOLUTION, MASTER_RESOLUTION):
        raise ValueError(
            "shared-master downsampling requires the 384-grid master input"
        )
    if (
        target_resolution <= 1
        or MASTER_RESOLUTION % target_resolution != 0
    ):
        raise ValueError(
            "target resolution must integer-divide the shared master"
        )
    return _block_mean_downsample(
        master,
        (target_resolution, target_resolution),
    )


def _generate_inputs(split: SplitSpec, sample_index: int) -> dict[str, Any]:
    cfg = yaml.safe_load(split.config_path.read_text(encoding="utf-8"))
    sample_seed = _seed_for_sample(split.seed, sample_index)
    paired = TsunamiDatasetBuilder._parse_paired_inputs_section(cfg)
    buffered = TsunamiDatasetBuilder._parse_buffered_domain_section(cfg)

    bathy_generator = BathymetryGenerator(
        str(ROOT / "configs/data/rebuild/bathymetry_384.yaml")
    )
    bathy_generator.rng = np.random.default_rng([sample_seed, 11])
    bathymetry_master, bathymetry_type = bathy_generator.generate()
    bathymetry_master = np.asarray(
        bathymetry_master, dtype=np.float32
    )

    source_fields = _generate_paired_source_fields(
        sample_seed=sample_seed,
        master_config_path=str(
            ROOT / "configs/data/rebuild/source_384.yaml"
        ),
        paired=paired,
        buffered_domain=buffered,
    )
    raw_source_master = source_fields["raw_master"]
    source_master = source_fields["master"]
    source_type = str(source_fields["source_type"])
    sampled_strength = float(
        np.float32(
            np.random.default_rng([sample_seed, 37]).uniform(
                *cfg["dataset"]["source_strength_range"]
            )
        )
    )
    depth_master = np.maximum(
        -np.asarray(bathymetry_master, dtype=np.float64), 1.0e-4
    )
    unit_ratio = float(
        np.max(
            np.abs(np.asarray(source_master, dtype=np.float64))
            / depth_master
        )
    )
    cap = float(cfg["dataset"]["max_initial_eta_over_depth"])
    strength = float(
        np.float32(
            min(
                sampled_strength,
                cap / max(unit_ratio, 1.0e-30),
            )
        )
    )
    eta0_master = np.asarray(
        strength * source_master, dtype=np.float32
    )
    bathymetry_384 = np.asarray(bathymetry_master, dtype=np.float32)
    bathymetry_64 = _downsample_shared_master(bathymetry_master, 64)
    bathymetry_128 = _downsample_shared_master(
        bathymetry_master, 128
    )
    bathymetry_192 = _downsample_shared_master(
        bathymetry_master, 192
    )
    source_384 = np.asarray(source_master, dtype=np.float32)
    source_64 = np.asarray(source_fields["target"], dtype=np.float32)
    source_128 = np.asarray(source_fields["solver"], dtype=np.float32)
    source_192 = _downsample_shared_master(source_master, 192)
    raw_source_128 = _downsample_shared_master(raw_source_master, 128)
    eta0_384 = np.asarray(eta0_master, dtype=np.float32)
    eta0_64 = np.asarray(strength * source_64, dtype=np.float32)
    eta0_128 = np.asarray(strength * source_128, dtype=np.float32)
    eta0_192 = np.asarray(strength * source_192, dtype=np.float32)

    return {
        "split": split.name,
        "sample_index": sample_index,
        "sample_seed": sample_seed,
        "bathymetry_type": str(bathymetry_type),
        "source_type": str(source_type),
        "source_strength": strength,
        "sampled_source_strength": sampled_strength,
        "source_generation_attempt_index": source_fields["attempt_index"],
        "source_generation_attempt_count": source_fields["attempt_count"],
        "source_spectral_points_per_p90_wavelength": source_fields[
            "points_per_p90_wavelength"
        ],
        "bathymetry_master": bathymetry_master,
        "source_master": source_master,
        "raw_source_master": raw_source_master,
        "eta0_master": eta0_master,
        "bathymetry_384": bathymetry_384,
        "bathymetry_64": bathymetry_64,
        "bathymetry_128": bathymetry_128,
        "bathymetry_192": bathymetry_192,
        "source_384": source_384,
        "source_64": source_64,
        "source_128": source_128,
        "source_192": source_192,
        "raw_source_128": raw_source_128,
        "eta0_384": eta0_384,
        "eta0_64": eta0_64,
        "eta0_128": eta0_128,
        "eta0_192": eta0_192,
    }


def _scan_one(task: tuple[SplitSpec, int]) -> dict[str, Any]:
    split, sample_index = task
    inputs = _generate_inputs(split, sample_index)
    cfg = _load_config(split.config_path)
    spectrum = _spectral_metrics(
        inputs["source_128"],
        inputs["eta0_128"],
        inputs["bathymetry_128"],
        dx=float(cfg["solver"]["dx"]),
    )
    requested = _requested_config(cfg)
    requested_step = float(np.min(np.diff(requested.requested_times)))
    wavelength = (
        float(spectrum["points_per_p90_wavelength"])
        * float(cfg["solver"]["dx"])
    )
    period = wavelength / math.sqrt(
        float(cfg["solver"].get("g", 9.81))
        * float(spectrum["effective_depth"])
    )
    depth = np.maximum(
        -np.asarray(inputs["bathymetry_128"], dtype=np.float64), 1.0e-4
    )
    source_128 = np.asarray(inputs["source_128"], dtype=np.float64)
    source_master = np.asarray(
        inputs["source_master"], dtype=np.float64
    )
    raw_source_master = np.asarray(
        inputs["raw_source_master"], dtype=np.float64
    )
    source_rms = float(np.sqrt(np.mean(source_master * source_master)))
    raw_source_rms = float(
        np.sqrt(np.mean(raw_source_master * raw_source_master))
    )
    eta_abs = np.abs(np.asarray(inputs["eta0_128"], dtype=np.float64))
    support_threshold = float(np.percentile(eta_abs, 90))
    support = eta_abs >= support_threshold
    grad_x, grad_y = np.gradient(
        np.asarray(inputs["bathymetry_128"], dtype=np.float64),
        float(cfg["solver"]["dx"]),
        float(cfg["solver"]["dy"]),
    )
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return {
        "split": split.name,
        "sample_index": sample_index,
        "sample_seed": inputs["sample_seed"],
        "bathymetry_type": inputs["bathymetry_type"],
        "source_type": inputs["source_type"],
        "source_strength": inputs["source_strength"],
        "sampled_source_strength": inputs["sampled_source_strength"],
        "amplitude_cap_applied": _amplitude_cap_applied(
            inputs["source_strength"],
            inputs["sampled_source_strength"],
        ),
        "source_generation_attempt_index": inputs[
            "source_generation_attempt_index"
        ],
        "source_generation_attempt_count": inputs[
            "source_generation_attempt_count"
        ],
        "accepted_source_points_per_p90_wavelength": inputs[
            "source_spectral_points_per_p90_wavelength"
        ],
        "source_rms": source_rms,
        "raw_source_rms": raw_source_rms,
        "source_peak": float(np.max(np.abs(source_128))),
        "temporal_samples_per_p90_period": float(
            period / requested_step
        ),
        "bathymetry_gradient_p99": float(
            np.percentile(gradient, 99)
        ),
        "source_support_min_depth": float(np.min(depth[support])),
        "local_eta_over_depth_max": float(
            np.max(
                np.abs(np.asarray(inputs["eta0_128"], dtype=np.float64))
                / depth
            )
        ),
        "master_eta_over_depth_max": float(
            np.max(
                np.abs(
                    np.asarray(inputs["eta0_master"], dtype=np.float64)
                )
                / np.maximum(
                    -np.asarray(
                        inputs["bathymetry_master"],
                        dtype=np.float64,
                    ),
                    1.0e-4,
                )
            )
        ),
        **{
            f"bathymetry_{name}_{bound}": float(reducer(values))
            for name, values in (
                ("master", inputs["bathymetry_master"]),
                ("64", inputs["bathymetry_64"]),
                ("128", inputs["bathymetry_128"]),
                ("192", inputs["bathymetry_192"]),
            )
            for bound, reducer in (("min", np.min), ("max", np.max))
        },
        "rough_abs_mean_over_rms": (
            float(abs(np.mean(source_master)) / max(source_rms, 1.0e-30))
            if inputs["source_type"] == "rough"
            else 0.0
        ),
        "rough_rms_relative_error": (
            float(
                abs(source_rms - raw_source_rms)
                / max(raw_source_rms, 1.0e-30)
            )
            if inputs["source_type"] == "rough"
            else 0.0
        ),
        "pairing_bathymetry_max_abs": float(
            np.max(
                np.abs(
                    inputs["bathymetry_64"]
                    - _block_mean_downsample(
                        inputs["bathymetry_128"], (64, 64)
                    )
                )
            )
        ),
        "pairing_source_max_abs": float(
            np.max(
                np.abs(
                    inputs["source_64"]
                    - _block_mean_downsample(inputs["source_128"], (64, 64))
                )
            )
        ),
        "pairing_eta0_max_abs": float(
            np.max(
                np.abs(
                    inputs["eta0_64"]
                    - _block_mean_downsample(inputs["eta0_128"], (64, 64))
                )
            )
        ),
        "refinement_pairing_bathymetry_max_abs": float(
            np.max(
                np.abs(
                    inputs["bathymetry_64"]
                    - _block_mean_downsample(
                        inputs["bathymetry_192"], (64, 64)
                    )
                )
            )
        ),
        "refinement_pairing_source_max_abs": float(
            np.max(
                np.abs(
                    inputs["source_64"]
                    - _block_mean_downsample(
                        inputs["source_192"], (64, 64)
                    )
                )
            )
        ),
        "refinement_pairing_eta0_max_abs": float(
            np.max(
                np.abs(
                    inputs["eta0_64"]
                    - _block_mean_downsample(
                        inputs["eta0_192"], (64, 64)
                    )
                )
            )
        ),
        "refinement_source_edge_max_abs": max(
            float(
                np.max(
                    np.abs(inputs["source_192"][[0, -1], :])
                )
            ),
            float(
                np.max(
                    np.abs(inputs["source_192"][:, [0, -1]])
                )
            ),
        ),
        **spectrum,
    }


def _percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def _input_gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hf = [float(row["high_frequency_fraction"]) for row in rows]
    local = [float(row["local_eta_over_depth_max"]) for row in rows]
    master_local = [
        float(row["master_eta_over_depth_max"]) for row in rows
    ]
    temporal = [
        float(row["temporal_samples_per_p90_period"]) for row in rows
    ]
    cap_fraction = float(
        np.mean(
            [
                str(row["amplitude_cap_applied"]).strip().lower()
                in {"1", "true", "yes"}
                for row in rows
            ]
        )
    )
    source_attempts = [
        int(row["source_generation_attempt_count"]) for row in rows
    ]
    accepted_source_points = [
        float(row["accepted_source_points_per_p90_wavelength"])
        for row in rows
    ]
    bathymetry_pairing_max = max(
        float(row["pairing_bathymetry_max_abs"]) for row in rows
    )
    source_pairing_max = max(
        float(row["pairing_source_max_abs"]) for row in rows
    )
    eta0_pairing_max = max(
        float(row["pairing_eta0_max_abs"]) for row in rows
    )
    refinement_bathymetry_pairing_max = max(
        float(row["refinement_pairing_bathymetry_max_abs"])
        for row in rows
    )
    refinement_source_pairing_max = max(
        float(row["refinement_pairing_source_max_abs"])
        for row in rows
    )
    refinement_eta0_pairing_max = max(
        float(row["refinement_pairing_eta0_max_abs"])
        for row in rows
    )
    refinement_source_edge_max = max(
        float(row["refinement_source_edge_max_abs"])
        for row in rows
    )
    decision = {
        "pairing_within_float32": (
            bathymetry_pairing_max
            <= INPUT_GATES["pairing_bathymetry_max_abs"]
            and source_pairing_max
            <= INPUT_GATES["pairing_source_max_abs"]
            and eta0_pairing_max
            <= INPUT_GATES["pairing_eta0_max_abs"]
            and refinement_bathymetry_pairing_max
            <= INPUT_GATES["pairing_bathymetry_max_abs"]
            and refinement_source_pairing_max
            <= INPUT_GATES["pairing_source_max_abs"]
            and refinement_eta0_pairing_max
            <= INPUT_GATES["pairing_eta0_max_abs"]
            and refinement_source_edge_max == 0.0
        ),
        "pairing_bathymetry_max_abs": bathymetry_pairing_max,
        "pairing_source_max_abs": source_pairing_max,
        "pairing_eta0_max_abs": eta0_pairing_max,
        "refinement_pairing_bathymetry_max_abs": (
            refinement_bathymetry_pairing_max
        ),
        "refinement_pairing_source_max_abs": (
            refinement_source_pairing_max
        ),
        "refinement_pairing_eta0_max_abs": refinement_eta0_pairing_max,
        "refinement_source_edge_max_abs": refinement_source_edge_max,
        "points_per_p90_wavelength_min": min(
            float(row["points_per_p90_wavelength"]) for row in rows
        ),
        "temporal_samples_per_p90_period_min": min(temporal),
        "temporal_samples_per_p90_period_p01": _percentile(
            temporal, 1
        ),
        "high_frequency_fraction_p95": _percentile(hf, 95),
        "high_frequency_fraction_max": max(hf),
        "kh_p90_max": max(float(row["kh_p90"]) for row in rows),
        "energy_fraction_kh_gt_1_max": max(
            float(row["energy_fraction_kh_gt_1"]) for row in rows
        ),
        "local_eta_over_depth_p95": _percentile(local, 95),
        "local_eta_over_depth_max": max(local),
        "master_eta_over_depth_max": max(master_local),
        "bathymetry_min": min(
            float(row[f"bathymetry_{resolution}_min"])
            for row in rows
            for resolution in ("master", "64", "128", "192")
        ),
        "bathymetry_max": max(
            float(row[f"bathymetry_{resolution}_max"])
            for row in rows
            for resolution in ("master", "64", "128", "192")
        ),
        "amplitude_cap_fraction": cap_fraction,
        "source_resampled_fraction": float(
            np.mean([attempts > 1 for attempts in source_attempts])
        ),
        "source_generation_attempt_count_max": max(source_attempts),
        "accepted_source_points_per_p90_wavelength_min": min(
            accepted_source_points
        ),
        "rough_abs_mean_over_rms_max": max(
            float(row["rough_abs_mean_over_rms"]) for row in rows
        ),
        "rough_rms_relative_error_max": max(
            float(row["rough_rms_relative_error"]) for row in rows
        ),
    }
    decision["per_source_family"] = {
        source_type: {
            "count": len(family),
            "points_per_p90_wavelength_min": min(
                float(row["points_per_p90_wavelength"])
                for row in family
            ),
            "temporal_samples_per_p90_period_min": min(
                float(row["temporal_samples_per_p90_period"])
                for row in family
            ),
            "local_eta_over_depth_p95": _percentile(
                [
                    float(row["local_eta_over_depth_max"])
                    for row in family
                ],
                95,
            ),
            "amplitude_cap_fraction": float(
                np.mean(
                    [
                        str(row["amplitude_cap_applied"])
                        .strip()
                        .lower()
                        in {"1", "true", "yes"}
                        for row in family
                    ]
                )
            ),
        }
        for source_type in SOURCE_FAMILIES
        for family in [
            [row for row in rows if row["source_type"] == source_type]
        ]
        if family
    }
    decision["amplitude_cap_fraction_per_family_max"] = max(
        float(summary["amplitude_cap_fraction"])
        for summary in decision["per_source_family"].values()
    )
    checks = {
        "source_spectral_acceptance": (
            decision[
                "accepted_source_points_per_p90_wavelength_min"
            ]
            + 1.0e-9
            >= 32.0
            and decision["source_generation_attempt_count_max"] <= 64
        ),
        "pairing_within_float32": decision["pairing_within_float32"],
        "resolution": (
            decision["points_per_p90_wavelength_min"] + 1.0e-12
            >= INPUT_GATES["min_points_per_p90_wavelength"]
        ),
        "temporal_resolution_min": (
            decision["temporal_samples_per_p90_period_min"]
            >= INPUT_GATES["min_temporal_samples_per_p90_period"]
        ),
        "temporal_resolution_p01": (
            decision["temporal_samples_per_p90_period_p01"]
            >= INPUT_GATES[
                "temporal_samples_per_p90_period_p01_min"
            ]
        ),
        "spectrum_p95": (
            decision["high_frequency_fraction_p95"]
            <= INPUT_GATES["high_frequency_fraction_p95_max"]
        ),
        "spectrum_max": (
            decision["high_frequency_fraction_max"]
            <= INPUT_GATES["high_frequency_fraction_absolute_max"]
        ),
        "kh_p90": (
            decision["kh_p90_max"] <= INPUT_GATES["kh_p90_absolute_max"]
        ),
        "kh_energy": (
            decision["energy_fraction_kh_gt_1_max"]
            <= INPUT_GATES["energy_fraction_kh_gt_1_absolute_max"]
        ),
        "amplitude_p95": (
            decision["local_eta_over_depth_p95"]
            <= INPUT_GATES["local_eta_over_depth_p95_max"] * (1.0 + 1.0e-6)
        ),
        "amplitude_max": (
            decision["local_eta_over_depth_max"]
            <= INPUT_GATES["local_eta_over_depth_absolute_max"]
            * (1.0 + 1.0e-6)
        ),
        "master_amplitude_max": (
            decision["master_eta_over_depth_max"]
            <= INPUT_GATES["local_eta_over_depth_absolute_max"]
            * (1.0 + 1.0e-6)
        ),
        "bathymetry_bounds": (
            decision["bathymetry_min"]
            >= INPUT_GATES["bathymetry_depth_min"] - 1.0e-6
            and decision["bathymetry_max"]
            <= INPUT_GATES["bathymetry_depth_max"] + 1.0e-6
        ),
        "amplitude_cap_fraction": (
            decision["amplitude_cap_fraction"]
            <= INPUT_GATES["amplitude_cap_fraction_max"]
        ),
        "amplitude_cap_fraction_per_family": (
            decision["amplitude_cap_fraction_per_family_max"]
            <= INPUT_GATES[
                "amplitude_cap_fraction_per_family_max"
            ]
        ),
        "rough_zero_mean": (
            decision["rough_abs_mean_over_rms_max"]
            <= INPUT_GATES["rough_abs_mean_over_rms_max"]
        ),
        "rough_rms_preserved": (
            decision["rough_rms_relative_error_max"]
            <= INPUT_GATES["rough_rms_relative_error_max"]
        ),
    }
    return {**decision, "checks": checks, "passed": all(checks.values())}


def _rank(values: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return 1.0 - ranks if reverse else ranks


def _select_cases(
    rows: list[dict[str, Any]],
    *,
    cases_per_cell: int = 2,
) -> list[dict[str, Any]]:
    if cases_per_cell not in {1, 2}:
        raise ValueError("cases_per_cell must be one or two")
    selected: list[dict[str, Any]] = []
    for bathymetry_type in BATHYMETRY_FAMILIES:
        for source_type in SOURCE_FAMILIES:
            candidates = [
                row
                for row in rows
                if row["bathymetry_type"] == bathymetry_type
                and row["source_type"] == source_type
            ]
            if not candidates:
                raise RuntimeError(
                    f"missing input cell {(bathymetry_type, source_type)}"
                )
            hf = np.asarray(
                [float(row["high_frequency_fraction"]) for row in candidates]
            )
            kh = np.asarray([float(row["kh_p90"]) for row in candidates])
            local = np.asarray(
                [float(row["local_eta_over_depth_max"]) for row in candidates]
            )
            ppw = np.asarray(
                [
                    float(row["points_per_p90_wavelength"])
                    for row in candidates
                ]
            )
            temporal = np.asarray(
                [
                    float(row["temporal_samples_per_p90_period"])
                    for row in candidates
                ]
            )
            gradients = np.asarray(
                [
                    float(row["bathymetry_gradient_p99"])
                    for row in candidates
                ]
            )
            support_depth = np.asarray(
                [
                    float(row["source_support_min_depth"])
                    for row in candidates
                ]
            )
            cap_applied = np.asarray(
                [
                    float(
                        str(row["amplitude_cap_applied"]).strip().lower()
                        in {"1", "true", "yes"}
                    )
                    for row in candidates
                ]
            )
            spectrum_scores = (
                _rank(hf)
                + _rank(kh)
                + _rank(ppw, reverse=True)
                + _rank(temporal, reverse=True)
            )
            interaction_scores = (
                _rank(local)
                + _rank(gradients)
                + _rank(support_depth, reverse=True)
                + cap_applied
            )
            if cases_per_cell == 1:
                combined = spectrum_scores + interaction_scores
                index = int(np.argmax(combined))
                winner = dict(candidates[index])
                winner["selection_reason"] = "combined_extreme"
                winner["selection_score"] = float(combined[index])
                selected.append(winner)
                continue
            spectrum_index = int(np.argmax(spectrum_scores))
            interaction_order = np.argsort(interaction_scores)[::-1]
            interaction_index = next(
                int(index)
                for index in interaction_order
                if int(index) != spectrum_index
            )
            for reason, index, score in (
                (
                    "spectrum_temporal_extreme",
                    spectrum_index,
                    spectrum_scores[spectrum_index],
                ),
                (
                    "amplitude_bathymetry_extreme",
                    interaction_index,
                    interaction_scores[interaction_index],
                ),
            ):
                winner = dict(candidates[index])
                winner["selection_reason"] = reason
                winner["selection_score"] = float(score)
                selected.append(winner)
    return sorted(selected, key=lambda row: int(row["sample_index"]))


def _select_mini_cases(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mini: list[dict[str, Any]] = []
    for source_type in SOURCE_FAMILIES:
        family = [
            row for row in selected if row["source_type"] == source_type
        ]
        tails = [
            row
            for row in family
            if row["selection_reason"] == "spectrum_temporal_extreme"
        ]
        controls = [
            row
            for row in family
            if row["selection_reason"] == "amplitude_bathymetry_extreme"
        ]
        if not tails or not controls:
            raise RuntimeError(
                f"mini selection is missing source family {source_type}"
            )
        tail = min(
            tails,
            key=lambda row: (
                float(row["points_per_p90_wavelength"]),
                float(row["temporal_samples_per_p90_period"]),
                int(row["sample_index"]),
            ),
        )
        control = max(
            controls,
            key=lambda row: (
                float(row["points_per_p90_wavelength"]),
                float(row["temporal_samples_per_p90_period"]),
                -int(row["sample_index"]),
            ),
        )
        mini.extend((tail, control))
    if len({int(row["sample_index"]) for row in mini}) != 12:
        raise RuntimeError("mini selection did not produce 12 distinct cases")
    return sorted(mini, key=lambda row: int(row["sample_index"]))


def _solver_factory(name: str, cfg: dict[str, Any]) -> Any:
    if name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(cfg)
    if name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(cfg)
    if name == "boussinesq":
        return _make_boussinesq_solver_from_cfg(cfg)
    raise KeyError(name)


def _rollout_paths(
    root: Path,
    split: str,
    sample_index: int,
    solver: str,
    resolution: int,
    cfl_factor: float,
) -> tuple[Path, Path]:
    factor = "full" if cfl_factor == 1.0 else "half"
    directory = (
        root
        / split
        / "rollouts"
        / f"sample_{sample_index:06d}"
        / solver
        / f"res{resolution}_{factor}"
    )
    return directory / "trajectory.npz", directory / "metrics.json"


def _run_rollout(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(task["root"]))
    split = _load_split(str(task["split"]))
    sample_index = int(task["sample_index"])
    solver_name = str(task["solver"])
    resolution = int(task["resolution"])
    cfl_factor = float(task["cfl_factor"])
    pilot_contract_hash = str(task["pilot_contract_hash"])
    trajectory_path, metrics_path = _rollout_paths(
        root,
        split.name,
        sample_index,
        solver_name,
        resolution,
        cfl_factor,
    )

    inputs = _generate_inputs(split, sample_index)
    dataset_cfg = _load_config(split.config_path)
    requested_config = _requested_config(dataset_cfg)
    solver_cfg = dict(dataset_cfg["solver"])
    profile = {
        str(name): dict(values)
        for name, values in dataset_cfg["solver_profiles"].items()
    }
    shape = RESOLUTIONS[resolution]
    solver_cfg.update(
        {
            "nx": shape["solver"],
            "ny": shape["solver"],
            "dx": shape["dx"],
            "dy": shape["dx"],
            "sponge_width": shape["buffer"],
        }
    )
    resolved = _resolved_solver_cfg_for_fde(
        solver_cfg, profile, solver_name
    )
    resolved["cfl"] = float(resolved["cfl"]) * cfl_factor
    task_identity = {
        "pilot_contract_hash": pilot_contract_hash,
        "split": split.name,
        "split_seed": split.seed,
        "sample_index": sample_index,
        "sample_seed": inputs["sample_seed"],
        "bathymetry_type": inputs["bathymetry_type"],
        "source_type": inputs["source_type"],
        "solver": solver_name,
        "resolution": resolution,
        "cfl_factor": cfl_factor,
        "resolved_solver_config_hash": stable_hash_payload(
            artifact_kind="pilot-resolved-solver-config",
            payload=resolved,
            schema_id=PILOT_SCHEMA_ID,
        ),
        "requested_contract_hash": requested_config.contract_hash,
        "input_hashes": {
            name: hash_array(inputs[f"{name}_{resolution}"])
            for name in ("bathymetry", "source", "eta0")
        },
    }
    if trajectory_path.is_file() or metrics_path.is_file():
        if not trajectory_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(
                f"incomplete cached rollout artifact: {trajectory_path.parent}"
            )
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("task_identity") != task_identity:
            raise RuntimeError(
                f"cached rollout identity mismatch: {metrics_path}"
            )
        with np.load(trajectory_path, allow_pickle=False) as payload:
            required = (
                "trajectory_eta",
                "timestamps",
                "eta0",
                "bathymetry",
                "source_field",
            )
            missing = [name for name in required if name not in payload]
            if missing:
                raise RuntimeError(
                    f"cached rollout arrays missing {missing}: {trajectory_path}"
                )
            observed_hashes = {
                name: hash_array(payload[name]) for name in required
            }
        if cached.get("artifact_hashes") != observed_hashes:
            raise RuntimeError(
                f"cached rollout hash mismatch: {trajectory_path}"
            )
        return cached

    buffered = BufferedDomainConfig(
        enabled=True,
        buffer_cells=int(shape["buffer"]),
        source_taper_cells=int(shape["taper"]),
        bathymetry_extension="edge",
        output_crop="central",
    )
    prepared = _prepare_buffered_domain(
        inputs[f"bathymetry_{resolution}"],
        inputs[f"source_{resolution}"],
        inputs["source_strength"],
        0.0,
        buffered,
        source_type=inputs["source_type"],
        source_already_tapered=True,
    )

    solver = _solver_factory(solver_name, resolved)
    solver.set_bathymetry(prepared["solver_bathymetry"])
    if solver_name == "boussinesq":
        solver.set_initial_condition(
            prepared["solver_eta0"],
            eta_t0=np.zeros_like(prepared["solver_eta0"]),
        )
    else:
        solver.set_initial_condition(
            prepared["solver_h0"],
            hu0=np.zeros_like(prepared["solver_h0"]),
            hv0=np.zeros_like(prepared["solver_h0"]),
        )

    requested = requested_config.requested_times
    started = time.perf_counter()
    states, timestamps, dt_history, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=float(resolved["cfl"]),
        include_initial_state=False,
        requested_times=requested,
        max_natural_steps=requested_config.max_natural_steps,
        collect_natural_step_health=(
            requested_config.collect_natural_step_health
        ),
        requested_state_dtype=np.float32,
    )
    crop = prepared["crop"]
    eta_full = (
        np.asarray(states[:, 0])
        if solver_name == "boussinesq"
        else states[:, 0] + prepared["solver_bathymetry"][None, ...]
    )
    eta = np.asarray(eta_full[:, crop[0], crop[1]])
    full_rollout = RolloutResult(
        trajectory=np.asarray(states),
        trajectory_eta=eta_full,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        dt_history=np.asarray(dt_history, dtype=np.float64),
        diagnostics=diagnostics,
    )
    effective_depth = None
    if solver_name == "boussinesq":
        effective_depth = np.maximum(
            (
                -prepared["solver_bathymetry"]
                + float(resolved.get("sea_level_offset", 0.0))
            )
            * float(resolved.get("depth_scale", 1.0)),
            float(resolved.get("min_depth", 1.0e-3)),
        )
    health = _compute_rollout_health(
        fde_name=solver_name,
        rollout=full_rollout,
        rest_depth=prepared["solver_rest_depth"],
        effective_depth=effective_depth,
    )
    health = _requested_health_summary(
        diagnostics,
        health,
        fde_name=solver_name,
        target_cfl=float(resolved["cfl"]),
    )
    violations = _quality_violations_for_health(
        health,
        _quality_policy(dataset_cfg),
    )
    arrays = {
        "trajectory_eta": eta,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "eta0": np.asarray(prepared["eta0"], dtype=np.float32),
        "bathymetry": np.asarray(
            prepared["bathymetry"], dtype=np.float32
        ),
        "source_field": np.asarray(
            prepared["source_field"], dtype=np.float32
        ),
    }
    artifact_hashes = {
        name: hash_array(values) for name, values in arrays.items()
    }
    metrics = {
        "split": split.name,
        "sample_index": sample_index,
        "bathymetry_type": inputs["bathymetry_type"],
        "source_type": inputs["source_type"],
        "solver": solver_name,
        "resolution": resolution,
        "cfl_factor": cfl_factor,
        "task_identity": task_identity,
        "artifact_hashes": artifact_hashes,
        "quality_status": "ok" if not violations else "fail",
        "quality_violations": violations,
        "runtime_s": float(time.perf_counter() - started),
        **health,
    }
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    staging = trajectory_path.with_name(
        f".{trajectory_path.stem}.staging-{os.getpid()}.npz"
    )
    try:
        np.savez_compressed(staging, **arrays)
        os.replace(staging, trajectory_path)
    finally:
        staging.unlink(missing_ok=True)
    _write_json(metrics_path, metrics)
    return metrics


def _load_eta(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["trajectory_eta"]),
            np.asarray(payload["timestamps"], dtype=np.float64),
            np.asarray(payload["eta0"]),
        )


def _publication_downsample(values: np.ndarray) -> np.ndarray:
    reduced = _block_mean_downsample_spatial(
        values,
        (PUBLICATION_RESOLUTION, PUBLICATION_RESOLUTION),
    )
    return np.asarray(reduced, dtype=np.float32)


def _high_frequency_fraction(field: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("spectrum field must be a finite 2-D array")
    coefficients = dctn(
        values - float(np.mean(values)),
        type=2,
        norm="ortho",
    )
    power = np.abs(coefficients) ** 2
    kx = np.pi * np.arange(values.shape[0], dtype=np.float64) / float(
        values.shape[0]
    )
    ky = np.pi * np.arange(values.shape[1], dtype=np.float64) / float(
        values.shape[1]
    )
    k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    mask = k > 0.0
    weights = power[mask]
    total = float(np.sum(weights))
    if not np.isfinite(total) or total < 0.0:
        raise RuntimeError("trajectory spectrum has invalid total power")
    if total == 0.0:
        return 0.0
    fraction = float(
        np.sum(weights[k[mask] > np.pi / 2.0]) / total
    )
    if not np.isfinite(fraction):
        raise RuntimeError("trajectory spectrum fraction is non-finite")
    return fraction


def _select_gauge_positions(
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    if len(shape) != 2 or min(shape) <= 4:
        raise ValueError("gauge grid must contain two spatial axes")
    fractions = tuple(index / 6.0 for index in range(1, 6))
    return [
        (
            int(round(fx * (shape[0] - 1))),
            int(round(fy * (shape[1] - 1))),
        )
        for fx in fractions
        for fy in fractions
    ]


def _threshold_crossing_frame(
    signal: np.ndarray,
    threshold: float,
) -> float | None:
    values = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.abs(np.asarray(signal, dtype=np.float64)),
        ]
    )
    hits = np.flatnonzero(values >= threshold)
    if not hits.size:
        return None
    index = int(hits[0])
    if index == 0:
        return 0.0
    previous = float(values[index - 1])
    current = float(values[index])
    if current <= previous:
        return float(index)
    fraction = float(
        np.clip(
            (threshold - previous) / (current - previous),
            0.0,
            1.0,
        )
    )
    return float(index - 1) + fraction


def _waveform_lag_metrics(
    candidate: np.ndarray,
    target: np.ndarray,
    *,
    max_lag_frames: int = 10,
) -> tuple[float, float] | None:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if (
        candidate_values.ndim != 1
        or target_values.shape != candidate_values.shape
        or not np.isfinite(candidate_values).all()
        or not np.isfinite(target_values).all()
    ):
        raise ValueError("waveform lag requires aligned finite 1-D signals")
    candidate_values = candidate_values - float(
        np.mean(candidate_values)
    )
    target_values = target_values - float(np.mean(target_values))
    maximum = min(
        int(max_lag_frames),
        max(0, candidate_values.size // 4),
    )
    best: tuple[float, int, int, int] | None = None
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            left = candidate_values[-lag:]
            right = target_values[:lag]
        elif lag > 0:
            left = candidate_values[:-lag]
            right = target_values[lag:]
        else:
            left = candidate_values
            right = target_values
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-30:
            continue
        correlation = float(np.dot(left, right) / denominator)
        rank = (correlation, -abs(lag), -lag, lag)
        if best is None or rank > best:
            best = rank
    if best is None:
        return None
    return float(abs(best[-1])), float(best[0])


def _trajectory_spectrum_metrics(
    trajectory: np.ndarray,
    eta0: np.ndarray,
    *,
    split: str,
    sample_index: int,
    solver: str,
    resolution: int,
    cfl_factor: float,
) -> dict[str, Any]:
    frames = np.asarray(trajectory, dtype=np.float64)
    initial = np.asarray(eta0, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[1:] != initial.shape:
        raise ValueError("trajectory spectrum shape mismatch")
    initial_fraction = _high_frequency_fraction(initial)
    fractions = [_high_frequency_fraction(frame) for frame in frames]
    maximum = max(fractions)
    return {
        "split": split,
        "sample_index": sample_index,
        "solver": solver,
        "resolution": resolution,
        "cfl_factor": cfl_factor,
        "initial_high_frequency_fraction": initial_fraction,
        "trajectory_high_frequency_fraction_max": maximum,
        "trajectory_high_frequency_growth": maximum - initial_fraction,
    }


def _comparison_metrics(
    left: np.ndarray,
    right: np.ndarray,
    left_eta0: np.ndarray,
    right_eta0: np.ndarray,
    *,
    kind: str,
    split: str,
    sample_index: int,
    solver: str,
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"comparison shape mismatch: {left.shape} != {right.shape}")
    if left_eta0.shape != right_eta0.shape or left.shape[1:] != left_eta0.shape:
        raise ValueError("comparison initial-state shape mismatch")
    diff = np.asarray(left - right, dtype=np.float64)
    frame_rmse = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
    initial_scale = float(
        np.sqrt(np.mean(np.asarray(right_eta0, dtype=np.float64) ** 2))
    )
    if not np.isfinite(initial_scale) or initial_scale <= 0.0:
        raise RuntimeError(
            "framewise comparison requires a finite positive initial RMS"
        )
    frame_initial_nrmse = frame_rmse / initial_scale

    target_delta_field = np.asarray(
        right - right_eta0[None, ...], dtype=np.float64
    )
    candidate_delta_field = np.asarray(
        left - left_eta0[None, ...], dtype=np.float64
    )
    gauge_positions = _select_gauge_positions(left.shape[1:])
    gauge_nrmse: list[float] = []
    gauge_peak_error: list[float] = []
    gauge_arrival_frames: list[float] = []
    gauge_waveform_lag_frames: list[float] = []
    gauge_waveform_correlations: list[float] = []
    gauge_time_to_peak_frames: list[float] = []
    global_target_peak = float(np.max(np.abs(target_delta_field)))
    signal_floor = max(
        1.0e-8,
        GAUGE_GLOBAL_SIGNAL_FLOOR_FRACTION * global_target_peak,
    )
    eligible_target_peak_min = (
        signal_floor / GAUGE_ARRIVAL_PEAK_FRACTION
    )
    for x, y in gauge_positions:
        target_delta = np.asarray(
            target_delta_field[:, x, y], dtype=np.float64
        )
        candidate_delta = np.asarray(
            candidate_delta_field[:, x, y], dtype=np.float64
        )
        target_peak = float(np.max(np.abs(target_delta)))
        if target_peak < eligible_target_peak_min:
            continue
        gauge_nrmse.append(
            float(
                np.sqrt(np.mean((candidate_delta - target_delta) ** 2))
                / (np.sqrt(np.mean(target_delta * target_delta)) + 1.0e-30)
            )
        )
        candidate_peak = float(np.max(np.abs(candidate_delta)))
        gauge_peak_error.append(
            abs(candidate_peak - target_peak) / target_peak
        )
        lag = _waveform_lag_metrics(candidate_delta, target_delta)
        if lag is not None:
            gauge_waveform_lag_frames.append(lag[0])
            gauge_waveform_correlations.append(lag[1])
        gauge_time_to_peak_frames.append(
            float(
                abs(
                    int(np.argmax(np.abs(candidate_delta)))
                    - int(np.argmax(np.abs(target_delta)))
                )
            )
        )
        threshold = GAUGE_ARRIVAL_PEAK_FRACTION * target_peak
        target_arrival = _threshold_crossing_frame(
            target_delta, threshold
        )
        candidate_arrival = _threshold_crossing_frame(
            candidate_delta, threshold
        )
        if target_arrival is not None:
            gauge_arrival_frames.append(
                float(
                    left.shape[0]
                    if candidate_arrival is None
                    else abs(candidate_arrival - target_arrival)
                )
            )

    return {
        "split": split,
        "sample_index": sample_index,
        "solver": solver,
        "kind": kind,
        "trajectory_rel_l2": _safe_rel_l2(left, right),
        "per_frame_normalization": "target_initial_rms",
        "target_initial_rms": initial_scale,
        "per_frame_initial_nrmse_p95": _percentile(
            frame_initial_nrmse.tolist(), 95
        ),
        "per_frame_initial_nrmse_max": float(
            np.max(frame_initial_nrmse)
        ),
        "gauge_nrmse_p95": (
            _percentile(gauge_nrmse, 95) if gauge_nrmse else 0.0
        ),
        "gauge_nrmse_max": max(gauge_nrmse, default=0.0),
        "gauge_peak_relative_error_p95": (
            _percentile(gauge_peak_error, 95) if gauge_peak_error else 0.0
        ),
        "gauge_peak_relative_error_max": max(
            gauge_peak_error, default=0.0
        ),
        "gauge_arrival_frame_p95": (
            _percentile(gauge_arrival_frames, 95)
            if gauge_arrival_frames
            else 0.0
        ),
        "gauge_arrival_frame_max": max(gauge_arrival_frames, default=0.0),
        "eligible_arrival_gauges": len(gauge_arrival_frames),
        "gauge_waveform_lag_frame_p95": (
            _percentile(gauge_waveform_lag_frames, 95)
            if gauge_waveform_lag_frames
            else 0.0
        ),
        "gauge_waveform_lag_frame_max": max(
            gauge_waveform_lag_frames, default=0.0
        ),
        "gauge_waveform_correlation_p05": (
            _percentile(gauge_waveform_correlations, 5)
            if gauge_waveform_correlations
            else 0.0
        ),
        "gauge_waveform_correlation_min": min(
            gauge_waveform_correlations, default=0.0
        ),
        "eligible_waveform_lag_gauges": len(gauge_waveform_lag_frames),
        "gauge_time_to_peak_frame_p95": (
            _percentile(gauge_time_to_peak_frames, 95)
            if gauge_time_to_peak_frames
            else 0.0
        ),
        "gauge_time_to_peak_frame_max": max(
            gauge_time_to_peak_frames, default=0.0
        ),
        "predefined_gauge_count": len(gauge_positions),
        "eligible_waveform_gauges": len(gauge_nrmse),
        "gauge_position_policy": "fixed_5x5_lattice",
        "gauge_signal_floor": signal_floor,
        "gauge_eligible_target_peak_min": eligible_target_peak_min,
        "gauge_eligibility_policy": (
            "target_peak_times_arrival_fraction_at_least_global_signal_floor"
        ),
        "gauge_arrival_threshold_policy": "fraction_of_target_local_peak",
        "gauge_arrival_peak_fraction": GAUGE_ARRIVAL_PEAK_FRACTION,
        "gauge_timing_gate_policy": (
            "integer_lag_maximizing_mean_centered_waveform_correlation"
        ),
        "gauge_timing_max_search_lag_frames": 10,
    }


def _compare_split(
    root: Path,
    split: str,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    for scenario in selected:
        sample_index = int(scenario["sample_index"])
        for solver in SOLVERS:
            paths = {}
            for resolution, factors in RESOLUTION_CFL_FACTORS.items():
                for factor in factors:
                    trajectory_path, _ = _rollout_paths(
                        root,
                        split,
                        sample_index,
                        solver,
                        resolution,
                        factor,
                    )
                    paths[(resolution, factor)] = trajectory_path
            eta_production, time_production, eta0_production = _load_eta(
                paths[(PRODUCTION_RESOLUTION, 1.0)]
            )
            (
                eta_production_half,
                time_production_half,
                eta0_production_half,
            ) = _load_eta(
                paths[(PRODUCTION_RESOLUTION, 0.5)]
            )
            (
                eta_refinement,
                time_refinement,
                eta0_refinement,
            ) = _load_eta(
                paths[(REFINEMENT_RESOLUTION, 1.0)]
            )
            if not (
                np.array_equal(
                    time_production, time_production_half
                )
                and np.array_equal(
                    time_production, time_refinement
                )
            ):
                raise RuntimeError("pilot timestamps are not exactly aligned")
            if not np.array_equal(
                eta0_production, eta0_production_half
            ):
                raise RuntimeError(
                    "production full/half-CFL initial states are not identical"
                )
            for resolution, factor, trajectory, eta0 in (
                (
                    PRODUCTION_RESOLUTION,
                    1.0,
                    eta_production,
                    eta0_production,
                ),
                (
                    PRODUCTION_RESOLUTION,
                    0.5,
                    eta_production_half,
                    eta0_production_half,
                ),
                (
                    REFINEMENT_RESOLUTION,
                    1.0,
                    eta_refinement,
                    eta0_refinement,
                ),
            ):
                spectra.append(
                    _trajectory_spectrum_metrics(
                        trajectory,
                        eta0,
                        split=split,
                        sample_index=sample_index,
                        solver=solver,
                        resolution=resolution,
                        cfl_factor=factor,
                    )
                )
            eta_production_64 = _publication_downsample(eta_production)
            eta_production_half_64 = _publication_downsample(
                eta_production_half
            )
            eta_refinement_64 = _publication_downsample(eta_refinement)
            eta0_production_64 = _publication_downsample(eta0_production)
            eta0_production_half_64 = _publication_downsample(
                eta0_production_half
            )
            eta0_refinement_64 = _publication_downsample(eta0_refinement)
            if (
                np.max(
                    np.abs(
                        eta0_production_64 - eta0_refinement_64
                    )
                )
                > 1.0e-7
            ):
                raise RuntimeError(
                    "production/refinement publication initial states differ"
                )
            rows.append(
                _comparison_metrics(
                    eta_production_half_64,
                    eta_production_64,
                    eta0_production_half_64,
                    eta0_production_64,
                    kind="half_cfl_production",
                    split=split,
                    sample_index=sample_index,
                    solver=solver,
                )
            )
            rows.append(
                _comparison_metrics(
                    eta_production_64,
                    eta_refinement_64,
                    eta0_production_64,
                    eta0_refinement_64,
                    kind="refinement_publication_128_to_192",
                    split=split,
                    sample_index=sample_index,
                    solver=solver,
                )
            )
    return rows, spectra


def _rollout_gate_summary(
    health: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    spectra: list[dict[str, Any]],
) -> dict[str, Any]:
    health_checks = {
        "all_finite": all(
            int(row["nan_count"]) == 0 and int(row["inf_count"]) == 0
            for row in health
        ),
        "production_quality_passed": all(
            row.get("quality_status") == "ok"
            and not row.get("quality_violations")
            for row in health
        ),
        "cfl_bounded": all(
            float(row["max_post_step_cfl"])
            <= float(row["target_cfl"]) * 1.01
            for row in health
        ),
    }
    half = [
        row for row in comparisons if str(row["kind"]).startswith("half_cfl")
    ]
    refinement = [
        row
        for row in comparisons
        if row["kind"] == "refinement_publication_128_to_192"
    ]
    half_rel = [float(row["trajectory_rel_l2"]) for row in half]
    refine_rel = [float(row["trajectory_rel_l2"]) for row in refinement]
    spectrum_lookup: dict[
        tuple[str, int, str, int, float],
        dict[str, Any],
    ] = {}
    for row in spectra:
        key = (
            str(row["split"]),
            int(row["sample_index"]),
            str(row["solver"]),
            int(row["resolution"]),
            float(row["cfl_factor"]),
        )
        if key in spectrum_lookup:
            raise RuntimeError(f"duplicate trajectory spectrum row: {key}")
        spectrum_lookup[key] = row
    half_spectrum_deltas: list[float] = []
    refinement_spectrum_excesses: list[float] = []
    scenario_solver_keys = {
        (str(row["split"]), int(row["sample_index"]), str(row["solver"]))
        for row in spectra
    }
    for split_name, sample_index, solver_name in scenario_solver_keys:
        production_full = spectrum_lookup.get(
            (
                split_name,
                sample_index,
                solver_name,
                PRODUCTION_RESOLUTION,
                1.0,
            )
        )
        production_half = spectrum_lookup.get(
            (
                split_name,
                sample_index,
                solver_name,
                PRODUCTION_RESOLUTION,
                0.5,
            )
        )
        refinement_full = spectrum_lookup.get(
            (
                split_name,
                sample_index,
                solver_name,
                REFINEMENT_RESOLUTION,
                1.0,
            )
        )
        if (
            production_full is None
            or production_half is None
            or refinement_full is None
        ):
            raise RuntimeError(
                "trajectory spectrum rows are incomplete for "
                f"{(split_name, sample_index, solver_name)}"
            )
        production_fraction = float(
            production_full["trajectory_high_frequency_fraction_max"]
        )
        half_spectrum_deltas.append(
            abs(
                production_fraction
                - float(
                    production_half[
                        "trajectory_high_frequency_fraction_max"
                    ]
                )
            )
        )
        refinement_spectrum_excesses.append(
            float(
                refinement_full[
                    "trajectory_high_frequency_fraction_max"
                ]
            )
            - production_fraction
        )
    summary = {
        "health_checks": health_checks,
        "half_cfl_rel_l2_median": float(np.median(half_rel)),
        "half_cfl_rel_l2_p95": _percentile(half_rel, 95),
        "half_cfl_rel_l2_max": max(half_rel),
        "refinement_rel_l2_median": float(np.median(refine_rel)),
        "refinement_rel_l2_p95": _percentile(refine_rel, 95),
        "refinement_rel_l2_max": max(refine_rel),
        "per_frame_initial_nrmse_p95": max(
            float(row["per_frame_initial_nrmse_p95"])
            for row in refinement
        ),
        "per_frame_initial_nrmse_max": max(
            float(row["per_frame_initial_nrmse_max"])
            for row in refinement
        ),
        "gauge_nrmse_p95": max(
            float(row["gauge_nrmse_p95"]) for row in refinement
        ),
        "gauge_nrmse_max": max(
            float(row["gauge_nrmse_max"]) for row in refinement
        ),
        "gauge_peak_relative_error_p95": max(
            float(row["gauge_peak_relative_error_p95"])
            for row in refinement
        ),
        "gauge_peak_relative_error_max": max(
            float(row["gauge_peak_relative_error_max"])
            for row in refinement
        ),
        "gauge_arrival_frame_p95": max(
            float(row["gauge_arrival_frame_p95"]) for row in refinement
        ),
        "gauge_arrival_frame_max": max(
            float(row["gauge_arrival_frame_max"]) for row in refinement
        ),
        "gauge_waveform_lag_frame_p95": max(
            float(row["gauge_waveform_lag_frame_p95"])
            for row in refinement
        ),
        "gauge_waveform_lag_frame_max": max(
            float(row["gauge_waveform_lag_frame_max"])
            for row in refinement
        ),
        "gauge_waveform_correlation_p05": min(
            float(row["gauge_waveform_correlation_p05"])
            for row in refinement
        ),
        "gauge_waveform_correlation_min": min(
            float(row["gauge_waveform_correlation_min"])
            for row in refinement
        ),
        "eligible_arrival_gauges_min": min(
            int(row["eligible_arrival_gauges"]) for row in refinement
        ),
        "eligible_waveform_lag_gauges_min": min(
            int(row["eligible_waveform_lag_gauges"])
            for row in refinement
        ),
        "eligible_waveform_gauges_min": min(
            int(row["eligible_waveform_gauges"]) for row in refinement
        ),
        "gauge_time_to_peak_frame_p95": max(
            float(row["gauge_time_to_peak_frame_p95"])
            for row in refinement
        ),
        "gauge_time_to_peak_frame_max": max(
            float(row["gauge_time_to_peak_frame_max"])
            for row in refinement
        ),
        "gauge_position_policy": "fixed_5x5_lattice",
        "gauge_case_aggregation": "maximum_of_case_p95",
        "trajectory_high_frequency_fraction_max": max(
            float(row["trajectory_high_frequency_fraction_max"])
            for row in spectra
        ),
        "trajectory_high_frequency_growth_max": max(
            float(row["trajectory_high_frequency_growth"])
            for row in spectra
        ),
        "half_cfl_high_frequency_fraction_delta_max": max(
            half_spectrum_deltas
        ),
        "refinement_high_frequency_fraction_excess_max": max(
            refinement_spectrum_excesses
        ),
    }
    all_checks = {
        **health_checks,
        "half_cfl_median": (
            summary["half_cfl_rel_l2_median"]
            <= ROLLOUT_GATES["half_cfl_rel_l2_median_max"]
        ),
        "half_cfl_p95": (
            summary["half_cfl_rel_l2_p95"]
            <= ROLLOUT_GATES["half_cfl_rel_l2_p95_max"]
        ),
        "half_cfl_max": (
            summary["half_cfl_rel_l2_max"]
            <= ROLLOUT_GATES["half_cfl_rel_l2_absolute_max"]
        ),
        "refinement_median": (
            summary["refinement_rel_l2_median"]
            <= ROLLOUT_GATES["refinement_rel_l2_median_max"]
        ),
        "refinement_p95": (
            summary["refinement_rel_l2_p95"]
            <= ROLLOUT_GATES["refinement_rel_l2_p95_max"]
        ),
        "refinement_max": (
            summary["refinement_rel_l2_max"]
            <= ROLLOUT_GATES["refinement_rel_l2_absolute_max"]
        ),
        "per_frame_initial_p95": (
            summary["per_frame_initial_nrmse_p95"]
            <= ROLLOUT_GATES[
                "per_frame_initial_nrmse_p95_max"
            ]
        ),
        "per_frame_initial_max": (
            summary["per_frame_initial_nrmse_max"]
            <= ROLLOUT_GATES[
                "per_frame_initial_nrmse_absolute_max"
            ]
        ),
        "gauge_nrmse_p95": (
            summary["gauge_nrmse_p95"]
            <= ROLLOUT_GATES["gauge_nrmse_p95_max"]
        ),
        "gauge_nrmse_max": (
            summary["gauge_nrmse_max"]
            <= ROLLOUT_GATES["gauge_nrmse_absolute_max"]
        ),
        "gauge_peak_p95": (
            summary["gauge_peak_relative_error_p95"]
            <= ROLLOUT_GATES["gauge_peak_relative_error_p95_max"]
        ),
        "gauge_peak_max": (
            summary["gauge_peak_relative_error_max"]
            <= ROLLOUT_GATES["gauge_peak_relative_error_absolute_max"]
        ),
        "gauge_waveform_lag_p95": (
            summary["gauge_waveform_lag_frame_p95"]
            <= ROLLOUT_GATES["gauge_waveform_lag_frame_p95_max"]
        ),
        "gauge_waveform_lag_max": (
            summary["gauge_waveform_lag_frame_max"]
            <= ROLLOUT_GATES["gauge_waveform_lag_frame_absolute_max"]
        ),
        "gauge_waveform_lag_coverage": (
            summary["eligible_waveform_lag_gauges_min"]
            >= ROLLOUT_GATES["eligible_waveform_lag_gauges_min"]
        ),
        "trajectory_spectrum_absolute": (
            summary["trajectory_high_frequency_fraction_max"]
            <= ROLLOUT_GATES["trajectory_high_frequency_fraction_max"]
        ),
        "trajectory_spectrum_half_cfl": (
            summary["half_cfl_high_frequency_fraction_delta_max"]
            <= ROLLOUT_GATES[
                "half_cfl_high_frequency_fraction_delta_max"
            ]
        ),
        "trajectory_spectrum_refinement": (
            summary["refinement_high_frequency_fraction_excess_max"]
            <= ROLLOUT_GATES[
                "refinement_high_frequency_fraction_excess_max"
            ]
        ),
    }
    diagnostic_checks = {
        name: all_checks[name]
        for name in LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS
    }
    promotion_checks = {
        name: passed
        for name, passed in all_checks.items()
        if name not in LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS
    }
    return {
        **summary,
        "checks": all_checks,
        "promotion_checks": promotion_checks,
        "diagnostic_checks": diagnostic_checks,
        "local_amplitude_diagnostic_passed": all(
            diagnostic_checks.values()
        ),
        "promotion_policy": (
            "full_field_health_timing_and_spectrum_with_reported_"
            "local_amplitude_resolution_diagnostics"
        ),
        "passed": all(promotion_checks.values()),
    }


def _scan_split(
    split: SplitSpec,
    root: Path,
    *,
    workers: int,
    pilot_contract_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cases_per_cell = 2 if split.name == "train" else 1
    rows_path = root / split.name / "input_scan.csv"
    selected_path = root / split.name / "selected_cases.json"
    manifest_path = root / split.name / "input_scan_manifest.json"
    cached = [
        rows_path.is_file(),
        selected_path.is_file(),
        manifest_path.is_file(),
    ]
    if any(cached) and not all(cached):
        raise RuntimeError(
            f"incomplete cached input scan for split {split.name}"
        )
    if all(cached):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = {
            "pilot_contract_hash": pilot_contract_hash,
            "split": split.name,
            "seed": split.seed,
            "scenario_count": split.count,
            "input_scan_sha256": sha256_file(rows_path),
            "selected_cases_sha256": sha256_file(selected_path),
        }
        if manifest != expected_manifest:
            raise RuntimeError(
                f"cached input scan provenance mismatch for {split.name}"
            )
        with rows_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        indices = [int(row["sample_index"]) for row in rows]
        if (
            len(rows) != split.count
            or len(set(indices)) != split.count
            or sorted(indices) != list(range(1, split.count + 1))
        ):
            raise RuntimeError(
                f"cached input scan coverage mismatch for {split.name}"
            )
    else:
        tasks = [(split, index) for index in range(1, split.count + 1)]
        rows = []
        if workers <= 1:
            for done, task in enumerate(tasks, 1):
                rows.append(_scan_one(task))
                if done % 100 == 0 or done == len(tasks):
                    print(
                        f"[scan {split.name}] {done}/{len(tasks)}",
                        flush=True,
                    )
        else:
            context = get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as executor:
                futures = {
                    executor.submit(_scan_one, task): task[1]
                    for task in tasks
                }
                for done, future in enumerate(as_completed(futures), 1):
                    rows.append(future.result())
                    if done % 100 == 0 or done == len(tasks):
                        print(
                            f"[scan {split.name}] {done}/{len(tasks)}",
                            flush=True,
                        )
        rows.sort(key=lambda row: int(row["sample_index"]))
        selected = _select_cases(
            rows,
            cases_per_cell=cases_per_cell,
        )
        _write_csv(rows_path, rows)
        _write_json(
            selected_path,
            {
                "split": split.name,
                "seed": split.seed,
                "selection_uses_solver_outputs": False,
                "cases_per_bathymetry_source_cell": cases_per_cell,
                "cases": selected,
            },
        )
        _write_json(
            manifest_path,
            {
                "pilot_contract_hash": pilot_contract_hash,
                "split": split.name,
                "seed": split.seed,
                "scenario_count": split.count,
                "input_scan_sha256": sha256_file(rows_path),
                "selected_cases_sha256": sha256_file(selected_path),
            },
        )
    gate = _input_gate_summary(rows)
    gate.update(
        {
            "pilot_contract_hash": pilot_contract_hash,
            "split": split.name,
            "seed": split.seed,
            "scenario_count": len(rows),
            "thresholds": INPUT_GATES,
        }
    )
    _write_json(root / split.name / "input_gate.json", gate)
    if isinstance(selected, dict):
        selected_rows = list(selected["cases"])
    else:
        selected_rows = list(selected)
    if len(selected_rows) != len(BATHYMETRY_FAMILIES) * len(
        SOURCE_FAMILIES
    ) * cases_per_cell:
        raise RuntimeError(
            f"selected-case count mismatch for {split.name}"
        )
    return rows, selected_rows, gate


def _run_split(
    split: SplitSpec,
    root: Path,
    selected: list[dict[str, Any]],
    *,
    workers: int,
    pilot_contract_hash: str,
    artifact_subdir: str | None = None,
) -> dict[str, Any]:
    tasks = [
        {
            "root": str(root),
            "split": split.name,
            "sample_index": int(row["sample_index"]),
            "solver": solver,
            "resolution": resolution,
            "cfl_factor": factor,
            "pilot_contract_hash": pilot_contract_hash,
        }
        for row in selected
        for solver in SOLVERS
        for resolution, factors in RESOLUTION_CFL_FACTORS.items()
        for factor in factors
    ]
    health: list[dict[str, Any]] = []
    if workers <= 1:
        for done, task in enumerate(tasks, 1):
            row = _run_rollout(task)
            health.append(row)
            print(
                f"[rollout {split.name}] {done}/{len(tasks)} "
                f"sample={row['sample_index']:06d} solver={row['solver']} "
                f"res={row['resolution']} cfl={row['cfl_factor']}",
                flush=True,
            )
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(_run_rollout, task): task for task in tasks
            }
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                health.append(row)
                print(
                    f"[rollout {split.name}] {done}/{len(tasks)} "
                    f"sample={row['sample_index']:06d} solver={row['solver']} "
                    f"res={row['resolution']} cfl={row['cfl_factor']}",
                    flush=True,
                )
    health.sort(
        key=lambda row: (
            int(row["sample_index"]),
            str(row["solver"]),
            int(row["resolution"]),
            float(row["cfl_factor"]),
        )
    )
    artifact_root = root / split.name
    if artifact_subdir is not None:
        artifact_root = artifact_root / artifact_subdir
    _write_csv(artifact_root / "rollout_health.csv", health)
    comparisons, spectra = _compare_split(root, split.name, selected)
    _write_csv(artifact_root / "comparison_metrics.csv", comparisons)
    _write_csv(artifact_root / "trajectory_spectra.csv", spectra)
    gate = _rollout_gate_summary(health, comparisons, spectra)
    gate.update(
        {
            "pilot_contract_hash": pilot_contract_hash,
            "split": split.name,
            "selected_case_count": len(selected),
            "rollout_count": len(health),
            "comparison_count": len(comparisons),
            "spectrum_count": len(spectra),
            "thresholds": ROLLOUT_GATES,
            "artifact_subdir": artifact_subdir,
        }
    )
    _write_json(artifact_root / "rollout_gate.json", gate)
    return gate


def _targeted_gauge_diagnostic_contract() -> dict[str, Any]:
    config_paths = {
        "train": CONFIGS["train"],
        "bathymetry_384": ROOT
        / "configs/data/rebuild/bathymetry_384.yaml",
        "source_384": ROOT / "configs/data/rebuild/source_384.yaml",
    }
    payload = {
        "schema_id": PILOT_SCHEMA_ID,
        "kind": "targeted-gauge-refinement-diagnostic",
        "code_state": code_state(ROOT),
        "configs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(config_paths.items())
        },
        "split": "train",
        "sample_indices": list(TARGETED_GAUGE_DIAGNOSTIC_SAMPLES),
        "solvers": list(SOLVERS),
        "resolutions": list(TARGETED_GAUGE_DIAGNOSTIC_RESOLUTIONS),
        "publication_resolution": PUBLICATION_RESOLUTION,
        "shared_master_downsample_method": "block_mean_float64_v1",
    }
    contract_hash = stable_hash_payload(
        artifact_kind="targeted-gauge-refinement-diagnostic",
        payload=payload,
        schema_id=PILOT_SCHEMA_ID,
    )
    return {**payload, "contract_hash": contract_hash}


def _bind_targeted_diagnostic_root(
    root: Path,
    contract: dict[str, Any],
) -> None:
    path = root / "targeted_gauge_contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(
                "targeted diagnostic root is bound to a different "
                "code/config contract; use a fresh output root"
            )
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            "non-empty targeted diagnostic root has no matching contract; "
            "use a fresh output root"
        )
    root.mkdir(parents=True, exist_ok=True)
    _write_json(path, contract)


def _run_targeted_gauge_diagnostic(
    root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Compare 192 and 384 publication fields for known gauge outliers.

    This deliberately uses a fresh root and does not alter the frozen pilot
    contract, gate thresholds, or selected pilot cases.
    """
    train = _load_split("train")
    _assert_split_contract(train)
    contract = _targeted_gauge_diagnostic_contract()
    _bind_targeted_diagnostic_root(root, contract)
    contract_hash = str(contract["contract_hash"])
    tasks = [
        {
            "root": str(root),
            "split": train.name,
            "sample_index": sample_index,
            "solver": solver,
            "resolution": resolution,
            "cfl_factor": 1.0,
            "pilot_contract_hash": contract_hash,
        }
        for sample_index in TARGETED_GAUGE_DIAGNOSTIC_SAMPLES
        for solver in SOLVERS
        for resolution in TARGETED_GAUGE_DIAGNOSTIC_RESOLUTIONS
    ]
    health: list[dict[str, Any]] = []
    if workers <= 1:
        for done, task in enumerate(tasks, 1):
            row = _run_rollout(task)
            health.append(row)
            print(
                f"[targeted gauge] {done}/{len(tasks)} "
                f"sample={row['sample_index']:06d} solver={row['solver']} "
                f"res={row['resolution']}",
                flush=True,
            )
    else:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(_run_rollout, task): task for task in tasks
            }
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                health.append(row)
                print(
                    f"[targeted gauge] {done}/{len(tasks)} "
                    f"sample={row['sample_index']:06d} "
                    f"solver={row['solver']} res={row['resolution']}",
                    flush=True,
                )
    health.sort(
        key=lambda row: (
            int(row["sample_index"]),
            str(row["solver"]),
            int(row["resolution"]),
        )
    )
    comparisons: list[dict[str, Any]] = []
    for sample_index in TARGETED_GAUGE_DIAGNOSTIC_SAMPLES:
        for solver in SOLVERS:
            production_path, _ = _rollout_paths(
                root,
                train.name,
                sample_index,
                solver,
                REFINEMENT_RESOLUTION,
                1.0,
            )
            reference_path, _ = _rollout_paths(
                root,
                train.name,
                sample_index,
                solver,
                MASTER_RESOLUTION,
                1.0,
            )
            eta_192, timestamps_192, eta0_192 = _load_eta(production_path)
            eta_384, timestamps_384, eta0_384 = _load_eta(reference_path)
            if not np.array_equal(timestamps_192, timestamps_384):
                raise RuntimeError(
                    "targeted gauge diagnostic timestamps are not exactly "
                    "aligned"
                )
            eta_192_64 = _publication_downsample(eta_192)
            eta_384_64 = _publication_downsample(eta_384)
            eta0_192_64 = _publication_downsample(eta0_192)
            eta0_384_64 = _publication_downsample(eta0_384)
            if (
                np.max(np.abs(eta0_192_64 - eta0_384_64))
                > 1.0e-7
            ):
                raise RuntimeError(
                    "targeted gauge diagnostic publication initial states "
                    "differ"
                )
            comparisons.append(
                _comparison_metrics(
                    eta_192_64,
                    eta_384_64,
                    eta0_192_64,
                    eta0_384_64,
                    kind="targeted_refinement_publication_192_to_384",
                    split=train.name,
                    sample_index=sample_index,
                    solver=solver,
                )
            )
    health_checks = {
        "all_finite": all(
            int(row["nan_count"]) == 0 and int(row["inf_count"]) == 0
            for row in health
        ),
        "quality_passed": all(
            str(row["quality_status"]) == "ok" for row in health
        ),
        "cfl_bounded": all(
            float(row["max_post_step_cfl"])
            <= float(row["target_cfl"]) * 1.01 + 1.0e-12
            for row in health
        ),
    }
    summary = {
        "artifact_kind": "targeted-gauge-refinement-diagnostic",
        "status": "completed",
        "contract_hash": contract_hash,
        "sample_indices": list(TARGETED_GAUGE_DIAGNOSTIC_SAMPLES),
        "solvers": list(SOLVERS),
        "comparison_kind": "publication_192_to_384",
        "rollout_count": len(health),
        "comparison_count": len(comparisons),
        "health_checks": health_checks,
        "trajectory_rel_l2": {
            "max": max(
                float(row["trajectory_rel_l2"]) for row in comparisons
            ),
            "by_sample_solver": [
                {
                    "sample_index": int(row["sample_index"]),
                    "solver": str(row["solver"]),
                    "value": float(row["trajectory_rel_l2"]),
                }
                for row in comparisons
            ],
        },
        "gauge_threshold_arrival_frame_diagnostic": {
            "max": max(
                float(row["gauge_arrival_frame_max"])
                for row in comparisons
            ),
            "by_sample_solver": [
                {
                    "sample_index": int(row["sample_index"]),
                    "solver": str(row["solver"]),
                    "p95": float(row["gauge_arrival_frame_p95"]),
                    "max": float(row["gauge_arrival_frame_max"]),
                }
                for row in comparisons
            ],
        },
        "gauge_waveform_lag_frame": {
            "max": max(
                float(row["gauge_waveform_lag_frame_max"])
                for row in comparisons
            ),
            "by_sample_solver": [
                {
                    "sample_index": int(row["sample_index"]),
                    "solver": str(row["solver"]),
                    "p95": float(row["gauge_waveform_lag_frame_p95"]),
                    "max": float(row["gauge_waveform_lag_frame_max"]),
                }
                for row in comparisons
            ],
        },
        "gauge_waveform_correlation": {
            "min": min(
                float(row["gauge_waveform_correlation_min"])
                for row in comparisons
            ),
            "by_sample_solver": [
                {
                    "sample_index": int(row["sample_index"]),
                    "solver": str(row["solver"]),
                    "p05": float(row["gauge_waveform_correlation_p05"]),
                    "min": float(row["gauge_waveform_correlation_min"]),
                }
                for row in comparisons
            ],
        },
        "submitted_artifacts_modified": False,
    }
    _write_csv(root / train.name / "targeted_rollout_health.csv", health)
    _write_csv(
        root / train.name / "targeted_comparison_metrics.csv",
        comparisons,
    )
    _write_json(root / "targeted_gauge_summary.json", summary)
    return summary


def _path_contains_artifacts(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_dir():
        return True
    return any(
        child.is_symlink() or not child.is_dir()
        for child in path.rglob("*")
    )


def _assert_final_test_unopened(root: Path) -> list[str]:
    cfg = _load_config(CONFIGS["test"])
    dataset = cfg["dataset"]
    paired = cfg["paired_inputs"]
    declared = [
        Path(str(dataset[key]))
        for key in (
            "bathymetry_dir",
            "source_dir",
            "output_dir",
            "manifest_path",
        )
    ]
    declared.append(Path(str(paired["inventory_path"])))
    paths = [
        path if path.is_absolute() else (ROOT / path).resolve()
        for path in declared
    ]
    paths.extend(
        [
            (root / "test").resolve(),
            (root.parent / "data/test").resolve(),
        ]
    )
    artifacts = sorted(
        {str(path) for path in paths if _path_contains_artifacts(path)}
    )
    if artifacts:
        raise RuntimeError(
            "final-test artifacts already exist; the unopened-test boundary "
            f"cannot be asserted: {artifacts}"
        )
    return sorted({str(path) for path in paths})


def _write_train_promotion(
    root: Path,
    *,
    pilot_contract_hash: str,
) -> dict[str, Any]:
    payload = {
        "pilot_contract_hash": pilot_contract_hash,
        "split": "train",
        "status": "passed",
        "input_gate_sha256": sha256_file(
            root / "train/input_gate.json"
        ),
        "rollout_gate_sha256": sha256_file(
            root / "train/rollout_gate.json"
        ),
        "selected_cases_sha256": sha256_file(
            root / "train/selected_cases.json"
        ),
    }
    _write_json(root / "train/promotion.json", payload)
    return payload


def _require_train_promotion(
    root: Path,
    *,
    expected: dict[str, Any],
) -> None:
    path = root / "train/promotion.json"
    if not path.is_file():
        raise RuntimeError(
            "confirmation scan requires a passed train promotion artifact"
        )
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("train promotion artifact changed before confirmation")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered aspect-2400 rebuild input, refinement, "
            "and half-CFL pilot without touching submitted artifacts."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--workers", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scan-only", action="store_true")
    mode.add_argument("--mini", action="store_true")
    mode.add_argument("--targeted-gauge-diagnostic", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.targeted_gauge_diagnostic:
        root = (
            args.output_root
            or Path(
                "/tmp/tsunami-surrogate-rebuild/"
                "aspect2400-roughgrf-r1/targeted-gauge-r2"
            )
        ).resolve()
        _run_targeted_gauge_diagnostic(root, workers=args.workers)
        return
    root = (
        args.output_root
        or Path(
            "/mnt/Windows/tsunami-surrogate-rebuild/"
            "aspect2400-roughgrf-r1/pilot-r22"
        )
    ).resolve()
    for name in ("train", "confirmation", "test"):
        _assert_split_contract(_load_split(name))
    contract = _pilot_contract()
    _bind_pilot_root(root, contract)
    final_test_paths = _assert_final_test_unopened(root)
    pilot_contract_hash = str(contract["contract_hash"])
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "artifact_kind": "aspect2400-roughgrf-rebuild-pilot",
        "created_date": "2026-09-01",
        "output_root": str(root),
        "pilot_contract_hash": pilot_contract_hash,
        "train_seed": _load_split("train").seed,
        "confirmation_seed": _load_split("confirmation").seed,
        "final_test_seed": 911,
        "final_test_opened": False,
        "final_test_paths_checked_without_artifacts": final_test_paths,
        "input_gates": INPUT_GATES,
        "rollout_gates": ROLLOUT_GATES,
        "submitted_artifacts_modified": False,
    }

    train = _load_split("train")
    _, train_selected, train_input_gate = _scan_split(
        train,
        root,
        workers=args.workers,
        pilot_contract_hash=pilot_contract_hash,
    )
    summary["train_input_gate"] = train_input_gate
    if not train_input_gate["passed"]:
        summary.update(
            {
                "status": "failed_train_input_gate",
                "wall_time_s": time.perf_counter() - started,
            }
        )
        _write_json(root / "summary.json", summary)
        raise SystemExit(2)
    if args.scan_only:
        summary.update(
            {
                "status": "train_input_gate_passed_scan_only",
                "wall_time_s": time.perf_counter() - started,
            }
        )
        _write_json(root / "summary.json", summary)
        return
    if args.mini:
        mini_selected = _select_mini_cases(train_selected)
        _write_json(
            root / "train/mini/selected_cases.json",
            {
                "pilot_contract_hash": pilot_contract_hash,
                "selection_uses_solver_outputs": False,
                "cases": mini_selected,
            },
        )
        mini_gate = _run_split(
            train,
            root,
            mini_selected,
            workers=args.workers,
            pilot_contract_hash=pilot_contract_hash,
            artifact_subdir="mini",
        )
        summary["train_mini_rollout_gate"] = mini_gate
        summary.update(
            {
                "status": (
                    "train_mini_rollout_gate_passed"
                    if mini_gate["passed"]
                    else "failed_train_mini_rollout_gate"
                ),
                "wall_time_s": time.perf_counter() - started,
            }
        )
        _write_json(root / "summary.json", summary)
        if not mini_gate["passed"]:
            raise SystemExit(3)
        return

    train_rollout_gate = _run_split(
        train,
        root,
        train_selected,
        workers=args.workers,
        pilot_contract_hash=pilot_contract_hash,
    )
    summary["train_rollout_gate"] = train_rollout_gate
    if not train_rollout_gate["passed"]:
        summary.update(
            {
                "status": "failed_train_rollout_gate",
                "wall_time_s": time.perf_counter() - started,
            }
        )
        _write_json(root / "summary.json", summary)
        raise SystemExit(3)
    train_promotion = _write_train_promotion(
        root,
        pilot_contract_hash=pilot_contract_hash,
    )
    summary["train_promotion"] = train_promotion

    confirmation = _load_split("confirmation")
    _require_train_promotion(root, expected=train_promotion)
    _, confirmation_selected, confirmation_input_gate = _scan_split(
        confirmation,
        root,
        workers=args.workers,
        pilot_contract_hash=pilot_contract_hash,
    )
    summary["confirmation_input_gate"] = confirmation_input_gate
    if not confirmation_input_gate["passed"]:
        summary.update(
            {
                "status": "failed_confirmation_input_gate",
                "wall_time_s": time.perf_counter() - started,
            }
        )
        _write_json(root / "summary.json", summary)
        raise SystemExit(4)

    confirmation_rollout_gate = _run_split(
        confirmation,
        root,
        confirmation_selected,
        workers=args.workers,
        pilot_contract_hash=pilot_contract_hash,
    )
    summary["confirmation_rollout_gate"] = confirmation_rollout_gate
    summary.update(
        {
            "status": (
                "passed"
                if confirmation_rollout_gate["passed"]
                else "failed_confirmation_rollout_gate"
            ),
            "wall_time_s": time.perf_counter() - started,
        }
    )
    _write_json(root / "summary.json", summary)
    if not confirmation_rollout_gate["passed"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
