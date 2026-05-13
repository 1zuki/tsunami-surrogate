from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    from src.solver.muscl_swe import MUSCLShallowWaterSolver
except ImportError:
    from muscl_swe import MUSCLShallowWaterSolver

try:
    from src.solver.boussinesq import BoussinesqSolver
except ImportError:
    from boussinesq import BoussinesqSolver

KNOWN_FDES = {"swe_hydrostatic", "swe_muscl", "boussinesq"}
IMPLEMENTED_FDES = {"swe_hydrostatic", "swe_muscl", "boussinesq"}
FDE_OUTPUT_DIRNAME = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl": "muscl",
    "boussinesq": "boussinesq",
}

COMMON_SOLVER_KEYS = {"nx", "ny", "dx", "dy", "dt", "g", "cfl", "boundary", "use_sponge", "sponge_width", "sponge_min_factor"}
SWE_SOLVER_KEYS = COMMON_SOLVER_KEYS | {"dry_tolerance", "max_velocity"}
BOUSSINESQ_SOLVER_KEYS = COMMON_SOLVER_KEYS | {
    "alpha",
    "min_depth",
    "sea_level_offset",
    "mode",
    "filter_strength",
    "linear_solver_tol",
    "linear_solver_max_iter",
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

@dataclass
class RolloutResult:
    trajectory: np.ndarray
    trajectory_eta: np.ndarray
    timestamps: np.ndarray
    dt_history: np.ndarray


def _fde_dirname(fde_name: str) -> str:
    if fde_name in FDE_OUTPUT_DIRNAME:
        return FDE_OUTPUT_DIRNAME[fde_name]
    return str(fde_name).replace("swe_", "")

def _seed_for_sample(run_seed: int, sample_idx: int) -> int:
    # sample_idx is 1-based; keep derivation stable across workers/runs.
    return int(run_seed + sample_idx * 10007)

def _bathymetry_file_path(bathymetry_dir: str | Path, sample_idx: int) -> Path:
    return Path(bathymetry_dir) / f"sample_{sample_idx:06d}.npz"

def _source_file_path(source_dir: str | Path, sample_idx: int) -> Path:
    return Path(source_dir) / f"sample_{sample_idx:06d}.npz"

def _filter_solver_cfg(sv: Dict[str, Any], allowed: set[str]) -> Dict[str, Any]:
    return {key: sv[key] for key in allowed if key in sv}

def _make_hydrostatic_solver_from_cfg(sv: Dict[str, Any]) -> HydrostaticShallowWaterSolver:
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
    )


def _make_muscl_solver_from_cfg(sv: Dict[str, Any]) -> MUSCLShallowWaterSolver:
    cfg = _filter_solver_cfg(sv, SWE_SOLVER_KEYS)
    boundary = cfg.get("boundary", "open")
    return MUSCLShallowWaterSolver(
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
        boundary=cfg.get("boundary", "open"),
        mode=cfg.get("mode", "linear_variable_depth"),
        use_sponge=cfg["use_sponge"] if "use_sponge" in cfg else None,
        sponge_width=int(cfg.get("sponge_width", 20)),
        sponge_min_factor=float(cfg.get("sponge_min_factor", 0.9)),
        filter_strength=float(cfg.get("filter_strength", 0.0)),
        linear_solver_tol=float(cfg.get("linear_solver_tol", 1e-8)),
        linear_solver_max_iter=int(cfg.get("linear_solver_max_iter", 80)),
    )

def _simulate_one_local(
    solver: Any,
    n_steps: int,
    save_every: int,
    auto_dt: bool,
    target_cfl: float,
    include_initial_state: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    dt_hist: list[float] = []
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

        solver.step(dt=dt, auto_dt=False)
        current_time += float(dt)

        if (step_idx + 1) % save_every == 0:
            frames.append(solver.get_state().astype(np.float32))
            timestamps.append(current_time)
            dt_hist.append(float(dt))

    if not frames:
        frames.append(solver.get_state().astype(np.float32))
        timestamps.append(current_time)
        dt_hist.append(0.0)

    return (
        np.stack(frames, axis=0),
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(dt_hist, dtype=np.float32),
    )

def _run_fde_rollout(
    fde_name: str,
    solver_cfg: Dict[str, Any],
    dataset: DatasetConfig,
    bathymetry: np.ndarray,
    eta0: np.ndarray,
    h0: np.ndarray,
) -> RolloutResult:

    if fde_name == "swe_hydrostatic":
        solver = _make_hydrostatic_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

        trajectory, timestamps, dt_hist = _simulate_one_local(
            solver=solver,
            n_steps=dataset.n_steps,
            save_every=dataset.save_every,
            auto_dt=dataset.auto_dt,
            target_cfl=dataset.target_cfl,
            include_initial_state=dataset.include_initial_state,
        )
        trajectory_eta = trajectory[:, 0] + bathymetry[None, ...]
        return RolloutResult(trajectory, trajectory_eta, timestamps, dt_hist)

    if fde_name == "swe_muscl":
        solver = _make_muscl_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

        trajectory, timestamps, dt_hist = _simulate_one_local(
            solver=solver,
            n_steps=dataset.n_steps,
            save_every=dataset.save_every,
            auto_dt=dataset.auto_dt,
            target_cfl=dataset.target_cfl,
            include_initial_state=dataset.include_initial_state,
        )
        trajectory_eta = trajectory[:, 0] + bathymetry[None, ...]
        return RolloutResult(trajectory, trajectory_eta, timestamps, dt_hist)

    if fde_name == "boussinesq":
        solver = _make_boussinesq_solver_from_cfg(solver_cfg)
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))

        trajectory, timestamps, dt_hist = _simulate_one_local(
            solver=solver,
            n_steps=dataset.n_steps,
            save_every=dataset.save_every,
            auto_dt=dataset.auto_dt,
            target_cfl=dataset.target_cfl,
            include_initial_state=dataset.include_initial_state,
        )
        trajectory_eta = trajectory[:, 0]
        return RolloutResult(trajectory, trajectory_eta, timestamps, dt_hist)

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
    allow_override: bool = False,
) -> Dict[str, Any]:
    bathy_path = _bathymetry_file_path(bathymetry_dir, sample_idx)
    source_path = _source_file_path(source_dir, sample_idx)

    if not bathy_path.exists():
        raise FileNotFoundError(f"Missing bathymetry cache for sample {sample_idx}: {bathy_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source cache for sample {sample_idx}: {source_path}")

    with np.load(bathy_path) as bathy_npz:
        bathymetry = np.asarray(bathy_npz["bathymetry"], dtype=np.float32)
        bathy_type = str(np.asarray(bathy_npz["bathymetry_type"]).reshape(-1)[0])

    with np.load(source_path) as src_npz:
        source_field = np.asarray(src_npz["source_field"], dtype=np.float32)
        source_type = str(np.asarray(src_npz["source_type"]).reshape(-1)[0])
        source_strength = float(np.asarray(src_npz["source_strength"]).reshape(-1)[0])

    eta0 = source_strength * source_field
    rest_depth = np.maximum(-bathymetry + dataset.sea_level_offset, 0.0)
    h0 = np.maximum(rest_depth + eta0, 0.0)
    free_surface0 = h0 + bathymetry

    runnable_fdes = [name for name in dataset.enabled_fdes if name in IMPLEMENTED_FDES]
    skipped_unimplemented = [name for name in dataset.enabled_fdes if name not in IMPLEMENTED_FDES]

    if not runnable_fdes:
        raise RuntimeError(
            "No runnable FDE selected. Enable at least one implemented solver (swe_hydrostatic, swe_muscl, boussinesq)."
        )

    scenario_id = f"scenario_{sample_idx:06d}"
    solver_records: list[Dict[str, Any]] = []

    for fde_name in runnable_fdes:
        if fde_name not in fde_samples_dirs:
            raise KeyError(f"Missing output samples directory for FDE '{fde_name}'")

        rollout = _run_fde_rollout(
            fde_name=fde_name,
            solver_cfg=solver_cfg,
            dataset=dataset,
            bathymetry=bathymetry,
            eta0=eta0,
            h0=h0,
        )

        sample_dir = Path(fde_samples_dirs[fde_name]) / f"sample_{sample_idx:06d}"
        if allow_override and sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            sample_dir / "rollout.npz",
            trajectory=rollout.trajectory.astype(np.float32),
            trajectory_eta=rollout.trajectory_eta.astype(np.float32),
            timestamps=rollout.timestamps.astype(np.float32),
            dt_history=rollout.dt_history.astype(np.float32),
            fde_name=np.array([fde_name], dtype="U64"),
        )
        np.save(sample_dir / "trajectory_eta.npy", rollout.trajectory_eta.astype(np.float32))

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
            "solver": solver_cfg,
            "bathymetry_cache_path": str(bathy_path),
            "source_cache_path": str(source_path),
            "fdes_requested": list(dataset.enabled_fdes),
            "fdes_run": runnable_fdes,
            "fdes_skipped_unimplemented": skipped_unimplemented,
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
        "fdes_run": runnable_fdes,
        "fdes_skipped_unimplemented": skipped_unimplemented,
    }

    return {
        "sample_index": sample_idx,
        "scenario_id": scenario_id,
        "scenario_record": scenario_record,
        "solver_records": solver_records,
    }

class TsunamiDatasetBuilder:
    """Generate raw tsunami surrogate samples."""

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Could not find {config_path}, is the path correct")

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError("yaml config is empty/invalid")

        self.cfg = cfg
        self.dataset = self._parse_dataset_section(cfg)
        self.solver_cfg = self._parse_solver_section(cfg)
        self.bathy_cfg_path = self._require_path(cfg, ["configs", "bathymetry"])
        self.source_cfg_path = self._require_path(cfg, ["configs", "source"])

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
            self.fde_manifest_paths[fde_name] = self.scenario_manifest_path.parent / f"{folder_name}_manifest.jsonl"

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

        self.run_seed = int(np.random.SeedSequence().entropy) if self.dataset.seed is None else int(self.dataset.seed)

        solver_nx = int(self.solver_cfg["nx"])
        solver_ny = int(self.solver_cfg["ny"])

        if self.bathy_generator.nx != solver_nx or self.bathy_generator.ny != solver_ny:
            raise ValueError(
                f"Bathymetry grid ({self.bathy_generator.nx}, {self.bathy_generator.ny}) "
                f"must match solver grid ({solver_nx}, {solver_ny})"
            )

        if self.source_generator.nx != solver_nx or self.source_generator.ny != solver_ny:
            raise ValueError(
                f"Source grid ({self.source_generator.nx}, {self.source_generator.ny}) "
                f"must match solver grid ({solver_nx}, {solver_ny})"
            )

    @staticmethod
    def _require_path(cfg: Dict[str, Any], keys: list[str]) -> Path:
        node: Any = cfg
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"missing config path: {'.'.join(keys)}")
            node = node[key]
        return Path(str(node))

    @staticmethod
    def _parse_range(value: Any, name: str) -> Tuple[float, float]:
        arr = np.asarray(value, dtype=float)
        if arr.size != 2:
            raise ValueError(f"{name} must have 2 values [min, max]")
        if arr[0] > arr[1]:
            raise ValueError(f"{name} must have min <= max")
        return float(arr[0]), float(arr[1])

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

        enabled_fdes: list[str] = [str(name).strip() for name in enabled_fdes_raw]
        for name in enabled_fdes:
            if name not in KNOWN_FDES:
                raise ValueError(
                    f"Unknown FDE '{name}'. Supported names: {sorted(KNOWN_FDES)}"
                )

        primary_fde = str(fdes.get("primary", enabled_fdes[0])).strip()
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
        source_dir = Path(ds.get("source_dir", "data/source"))
        manifest_path = Path(ds.get("manifest_path", "data/synthetic/scenario_manifest.jsonl"))
        copy_configs = bool(ds.get("copy_configs", True))

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
                m = patt.match(p.name)
                if m is None:
                    continue
                idxs.add(int(m.group(1)))
            per_fde_sets.append(idxs)

        if not per_fde_sets:
            return set()
        return set.intersection(*per_fde_sets)

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

    def _phase_generate_bathymetry(self, indices: list[int], allow_override: bool = False) -> None:
        existing = self._existing_bathymetry_indices()
        pending = indices if allow_override else [idx for idx in indices if idx not in existing]

        if not pending:
            print("[dataset] phase 1/3 bathymetry cache already complete for this range")
            return

        print(
            f"[dataset] phase 1/3 generate bathymetry: pending={len(pending)}, "
            f"range=[{pending[0]}, {pending[-1]}], out='{self.bathymetry_dir}'"
        )

        if self.dataset.num_workers <= 1:
            done = 0
            for idx in pending:
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

    def _phase_generate_sources(self, indices: list[int], allow_override: bool = False) -> None:
        existing = self._existing_source_indices()
        pending = indices if allow_override else [idx for idx in indices if idx not in existing]

        if not pending:
            print("[dataset] phase 2/3 source cache already complete for this range")
            return

        print(
            f"[dataset] phase 2/3 generate sources: pending={len(pending)}, "
            f"range=[{pending[0]}, {pending[-1]}], out='{self.source_dir}'"
        )

        if self.dataset.num_workers <= 1:
            done = 0
            for idx in pending:
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

    def _phase_generate_rollouts(self, indices: list[int], allow_override: bool = False) -> list[Dict[str, Any]]:
        print(
            f"[dataset] phase 3/3 run FDEs={list(self.dataset.enabled_fdes)} "
            f"on samples={len(indices)}"
        )

        records: list[Dict[str, Any]] = []
        if self.dataset.num_workers <= 1:
            done = 0
            for idx in indices:
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
                    fde_samples_dirs={k: str(v) for k, v in self.fde_samples_dirs.items()},
                    allow_override=allow_override,
                )
                records.append(rec)
                done += 1
                solver_names = [str(s.get("solver_name", "unknown")) for s in rec.get("solver_records", [])]
                print(
                    f"[{done:06d}/{len(indices):06d}] sample={idx:06d} "
                    f"scenario={rec['scenario_id']} solvers={solver_names}"
                )
            return records

        workers = min(self.dataset.num_workers, max(1, os.cpu_count() or 1))
        mp_ctx = get_context("spawn")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            futures = {
                ex.submit(
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
                    allow_override,
                ): idx
                for idx in indices
            }

            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                done += 1
                solver_names = [str(s.get("solver_name", "unknown")) for s in rec.get("solver_records", [])]
                print(
                    f"[{done:06d}/{len(indices):06d}] sample={rec['sample_index']:06d} "
                    f"scenario={rec['scenario_id']} solvers={solver_names}"
                )

        return records

    def run(
        self,
        continue_from_last: bool = False,
        start_at: int | None = None,
        allow_override: bool = False,
    ) -> None:
        """generate all raw samples in three phases: bathymetry, source, and FDE rollouts."""
        if start_at is not None and start_at < 1:
            raise ValueError("--start-at must be >= 1")

        total = self.dataset.num_samples
        existing_indices = self._existing_sample_indices()

        if start_at is not None:
            start_idx = int(start_at)
        elif continue_from_last:
            start_idx = self._resume_rollback_start_index(existing_indices)
        else:
            start_idx = 1

        if start_idx == 1 and not continue_from_last and start_at is None:
            if self.scenario_manifest_path.exists():
                self.scenario_manifest_path.unlink()
            for path in self.fde_manifest_paths.values():
                if path.exists():
                    path.unlink()
        else:
            print(f"[dataset] resume mode: start_at={start_idx}")

        if start_idx > total:
            print(f"[dataset] nothing to do: start_at={start_idx} > num_samples={total}")
            return

        planned_indices = list(range(start_idx, total + 1))
        if allow_override:
            to_generate = planned_indices
            existing_in_range = len([idx for idx in planned_indices if idx in existing_indices])
            if existing_in_range > 0:
                print(f"[dataset] allow-override: regenerating {existing_in_range} existing samples in range")
        else:
            to_generate = [idx for idx in planned_indices if idx not in existing_indices]
            skipped = len(planned_indices) - len(to_generate)
            if skipped > 0:
                print(f"[dataset] skipping {skipped} existing samples (already present on disk)")

        if not to_generate:
            print("[dataset] nothing new to generate.")
            return

        print(
            f"[dataset] generation plan: samples={len(to_generate)}, "
            f"range=[{to_generate[0]}, {to_generate[-1]}], seed={self.run_seed}"
        )

        self._phase_generate_bathymetry(to_generate, allow_override=allow_override)
        self._phase_generate_sources(to_generate, allow_override=allow_override)
        records = self._phase_generate_rollouts(to_generate, allow_override=allow_override)

        records.sort(key=lambda r: int(r["sample_index"]))
        if allow_override:
            sample_indices = set(int(r["sample_index"]) for r in records)
            self._purge_manifest_indices(self.scenario_manifest_path, sample_indices)
            for manifest_path in self.fde_manifest_paths.values():
                self._purge_manifest_indices(manifest_path, sample_indices)

        for rec in records:
            self._append_manifest(self.scenario_manifest_path, rec["scenario_record"])
            for srec in rec.get("solver_records", []):
                solver_name = str(srec.get("solver_name", "unknown"))
                if solver_name in self.fde_manifest_paths:
                    self._append_manifest(self.fde_manifest_paths[solver_name], srec)

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="generate raw tsunami surrogate samples")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/dataset.yaml",
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_from_last",
        action="store_true",
        help="Resume from the largest existing sample index instead of starting at 1.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=None,
        help="Explicit 1-based sample index to start generating from exist.",
    )
    parser.add_argument(
        "--allow-override",
        action="store_true",
        help="Regenerate samples even if their output directories already exist.",
    )
    return parser

def main() -> None:
    args = _build_argparser().parse_args()
    builder = TsunamiDatasetBuilder(args.config)
    builder.run(
        continue_from_last=bool(args.continue_from_last),
        start_at=args.start_at,
        allow_override=bool(args.allow_override),
    )

if __name__ == "__main__":
    main()
