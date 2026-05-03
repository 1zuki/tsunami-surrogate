from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import yaml

try:
    from src.data_gen.generate_bathymetry import BathymetryGenerator
    from src.data_gen.generate_sources import SourceGenerator
except ImportError:
    from generate_bathymetry import BathymetryGenerator
    from generate_sources import SourceGenerator

try:
    from src.solver.shallow_water import ShallowWaterSolver
except ImportError: 
    from shallow_water import ShallowWaterSolver


@dataclass
class DatasetConfig:
    """ convenience wrapper for the top-level dataset config """
    num_samples: int
    n_steps: int
    save_every: int
    auto_dt: bool
    target_cfl: float
    include_initial_state: bool
    sea_level_offset: float
    source_strength_range: Tuple[float, float]
    output_dir: Path
    manifest_path: Path
    copy_configs: bool

class TsunamiDatasetBuilder:
    """ generate raw tsunami surrogate samples """

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
        self.samples_dir = self.output_dir / "samples"
        self.manifest_path = self.dataset.manifest_path

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # keep a copy of the config for reproducibility
        if self.dataset.copy_configs:
            self._copy_config_snapshot()

        self.bathy_generator = BathymetryGenerator(str(self.bathy_cfg_path))
        self.source_generator = SourceGenerator(str(self.source_cfg_path))
        
        # validate grid consistency
        solver_nx = int(self.solver_cfg["nx"])
        solver_ny = int(self.solver_cfg["ny"])
        
        if self.bathy_generator.nx != solver_nx or self.bathy_generator.ny != solver_ny:
            raise ValueError(f"Bathymetry grid ({self.bathy_generator.nx}, {self.bathy_generator.ny}) "
                             f"must match solver grid ({solver_nx}, {solver_ny})")
        
        if self.source_generator.nx != solver_nx or self.source_generator.ny != solver_ny:
            raise ValueError(f"Source grid ({self.source_generator.nx}, {self.source_generator.ny}) "
                             f"must match solver grid ({solver_nx}, {solver_ny})")

    # config helper
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

        num_samples = int(ds.get("num_samples", 100))
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
        manifest_path = Path(ds.get("manifest_path", "data/synthetic/manifest.jsonl"))
        copy_configs = bool(ds.get("copy_configs", True))

        if num_samples <= 0:
            raise ValueError("dataset.num_samples must be positive")

        if n_steps <= 0:
            raise ValueError("dataset.n_steps must be positive")

        if save_every <= 0:
            raise ValueError("dataset.save_every must be positive")

        if target_cfl <= 0:
            raise ValueError("dataset.target_cfl must be positive")

        return DatasetConfig(
            num_samples=num_samples,
            n_steps=n_steps,
            save_every=save_every,
            auto_dt=auto_dt,
            target_cfl=target_cfl,
            include_initial_state=include_initial_state,
            sea_level_offset=sea_level_offset,
            source_strength_range=source_strength_range,
            output_dir=output_dir,
            manifest_path=manifest_path,
            copy_configs=copy_configs
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

    # solver
    def _make_solver(self) -> ShallowWaterSolver:
        sv = self.solver_cfg
        boundary = sv.get("boundary", "open")

        return ShallowWaterSolver(
            nx=int(sv["nx"]),
            ny=int(sv["ny"]),
            dx=float(sv["dx"]),
            dy=float(sv["dy"]),
            dt=float(sv["dt"]),
            g=float(sv.get("g", 9.81)),
            cfl=float(sv.get("cfl", 0.45)),
            dry_tolerance=float(sv.get("dry_tolerance", 1e-6)),
            boundary=boundary,
            use_sponge=bool(sv.get("use_sponge", True)),
            sponge_width=int(sv.get("sponge_width", 20)),
            sponge_min_factor=float(sv.get("sponge_min_factor", 0.9))
        )

    def _sample_strength(self) -> float:
        lo, hi = self.dataset.source_strength_range

        return float(self.bathy_generator.rng.uniform(lo, hi))

    def _build_initial_conditions(self,  bathymetry: np.ndarray, source_field: np.ndarray,
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ return rest depth, scaled source, initial depth, free surface, and source type field """
        sea_level_offset = self.dataset.sea_level_offset
        source_strength = self._sample_strength()

        # source generator returns a dimensionless field -> treat it as a free-surface
        # displacement template and scale it into physical units.
        eta0 = source_strength * source_field

        # rest depth from bathymetry: bathymetry is negative below sea level
        rest_depth = np.maximum(-bathymetry + sea_level_offset, 0.0)
        h0 = np.maximum(rest_depth + eta0, 0.0)
        free_surface0 = h0 + bathymetry

        return rest_depth, eta0, h0, free_surface0, np.array([source_strength], dtype=float)

    def _simulate_one(self, solver: ShallowWaterSolver, n_steps: int, save_every: int, auto_dt: bool,
                      target_cfl: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ run one sample and return trajectory, timestamps, dt history """
        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        dt_hist: list[float] = []

        current_time = 0.0

        if self.dataset.include_initial_state:
            state = solver.get_state().astype(np.float32)
            frames.append(state)
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
                state = solver.get_state().astype(np.float32)
                frames.append(state)
                timestamps.append(current_time)
                dt_hist.append(float(dt))

        if not frames:
            state = solver.get_state().astype(np.float32)
            frames.append(state)
            timestamps.append(current_time)
            dt_hist.append(0.0)

        trajectory = np.stack(frames, axis=0)
        timestamps_arr = np.asarray(timestamps, dtype=np.float32)
        dt_hist_arr = np.asarray(dt_hist, dtype=np.float32)

        return trajectory, timestamps_arr, dt_hist_arr

    def _sample_meta(self, sample_idx: int, bathy_type: str, source_type: str, source_strength: float,
            trajectory: np.ndarray, timestamps: np.ndarray, dt_hist: np.ndarray, bathymetry: np.ndarray,
            source_field: np.ndarray, eta0: np.ndarray, h0: np.ndarray, free_surface0: np.ndarray
        ) -> Dict[str, Any]:

        return {
            "sample_index": sample_idx,
            "bathymetry_type": bathy_type,
            "source_type": source_type,
            "source_strength": source_strength,
            "num_frames": int(trajectory.shape[0]),
            "trajectory_shape": list(map(int, trajectory.shape)),
            "timestamps_shape": list(map(int, timestamps.shape)),
            "dt_history_shape": list(map(int, dt_hist.shape)),
            "bathymetry_shape": list(map(int, bathymetry.shape)),
            "source_shape": list(map(int, source_field.shape)),
            "eta0_shape": list(map(int, eta0.shape)),
            "h0_shape": list(map(int, h0.shape)),
            "free_surface0_shape": list(map(int, free_surface0.shape)),
            "dataset_config_path": str(self.config_path),
            "bathymetry_config_path": str(self.bathy_cfg_path),
            "source_config_path": str(self.source_cfg_path),
            "solver": self.solver_cfg
        }

    def _save_sample(self, sample_idx: int, bathymetry: np.ndarray, bathy_type: str,
            source_field: np.ndarray, source_type: str, rest_depth: np.ndarray,
            eta0: np.ndarray, h0: np.ndarray, free_surface0: np.ndarray, trajectory: np.ndarray,
            timestamps: np.ndarray, dt_hist: np.ndarray, source_strength: float
        ) -> None:

        sample_dir = self.samples_dir / f"sample_{sample_idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            sample_dir / "sample.npz",
            bathymetry=bathymetry.astype(np.float32),
            source_field=source_field.astype(np.float32),
            rest_depth=rest_depth.astype(np.float32),
            eta0=eta0.astype(np.float32),
            initial_depth=h0.astype(np.float32),
            free_surface0=free_surface0.astype(np.float32),
            trajectory=trajectory.astype(np.float32),
            timestamps=timestamps.astype(np.float32),
            dt_history=dt_hist.astype(np.float32)
        )

        meta = self._sample_meta(
            sample_idx=sample_idx,
            bathy_type=bathy_type,
            source_type=source_type,
            source_strength=source_strength,
            trajectory=trajectory,
            timestamps=timestamps,
            dt_hist=dt_hist,
            bathymetry=bathymetry,
            source_field=source_field,
            eta0=eta0,
            h0=h0,
            free_surface0=free_surface0
        )

        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _append_manifest(self, record: Dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # generate loop
    def run(self) -> None:
        """ generate all raw samples """
        # start fresh manifest for each run
        if self.manifest_path.exists():
            self.manifest_path.unlink()

        for idx in range(1, self.dataset.num_samples + 1):
            bathymetry, bathy_type = self.bathy_generator.generate()
            source_field, source_type = self.source_generator.generate()

            rest_depth, eta0, h0, free_surface0, strength_arr = self._build_initial_conditions(
                bathymetry=bathymetry,
                source_field=source_field,
            )

            source_strength = float(strength_arr[0])

            solver = self._make_solver()
            solver.set_bathymetry(bathymetry)
            solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

            trajectory, timestamps, dt_hist = self._simulate_one(
                solver=solver,
                n_steps=self.dataset.n_steps,
                save_every=self.dataset.save_every,
                auto_dt=self.dataset.auto_dt,
                target_cfl=self.dataset.target_cfl,
            )

            self._save_sample(
                sample_idx=idx,
                bathymetry=bathymetry,
                bathy_type=bathy_type,
                source_field=source_field,
                source_type=source_type,
                rest_depth=rest_depth,
                eta0=eta0,
                h0=h0,
                free_surface0=free_surface0,
                trajectory=trajectory,
                timestamps=timestamps,
                dt_hist=dt_hist,
                source_strength=source_strength,
            )

            manifest_record = {
                "sample_index": idx,
                "sample_dir": str(self.samples_dir / f"sample_{idx:06d}"),
                "bathy_type": bathy_type,
                "source_type": source_type,
                "num_frames": int(trajectory.shape[0]),
                "trajectory_shape": list(map(int, trajectory.shape)),
                "source_strength": source_strength,
            }

            self._append_manifest(manifest_record)

            print(f"[{idx:06d}/{self.dataset.num_samples:06d}] "
                  f"bathy={bathy_type:<11} source={source_type:<11} "
                  f"frames={trajectory.shape[0]}")

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="generate raw tsunami surrogate samples")
    parser.add_argument("--config",
                        type=str,
                        default="configs/data/dataset.yaml",
                        help="Path to the dataset YAML config.")
    return parser

def main() -> None:
    args = _build_argparser().parse_args()
    builder = TsunamiDatasetBuilder(args.config)
    builder.run()

if __name__ == "__main__":
    main()
    
