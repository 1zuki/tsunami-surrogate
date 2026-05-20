from __future__ import annotations
import json
import pathlib
import random
import numpy as np
import yaml
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

@dataclass
class PreprocessConfig:
    raw_dir: pathlib.Path
    processed_dir: pathlib.Path
    manifest_path: pathlib.Path

    split_train: float
    split_val: float
    split_test: float
    seed: int

    use_bathymetry: bool
    use_source: bool
    use_initial_depth: bool
    use_initial_surface: bool
    use_solver_id: bool

    target_mode: str # next_step / multi_step / final_state
    target_variable: str # eta / depth / state
    forecast_steps: int
    stride: int

    norm_method: str # standardize / minmax
    norm_channels: Dict[str, bool]
    eps: float

    save_format: str # npy
    compress: bool
    include_meta: bool

    export_eval_arrays: bool
    eval_input_order: List[str]
    eval_inputs_name: str
    eval_targets_name: str
    eval_ids_name: str
    eval_archive_name: str
    eval_manifest_name: str

class TsunamiPreprocessor:
    def __init__(self, config_path: str) -> None:
        self.config_path = pathlib.Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Could not find {config_path}, is the path correct")

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if cfg is None:
            raise ValueError("yaml config is empty/invalid")

        raw_dir = pathlib.Path(cfg.get("raw_dir", "data/raw/hydrostatic/samples"))
        processed_dir = pathlib.Path(cfg.get("processed_dir", "data/processed"))
        manifest_path = pathlib.Path(cfg.get("manifest_path", "data/synthetic/hydrostatic_manifest.jsonl"))

        split_cfg = cfg.get("split", cfg.get("spilt", {}))
        train_ratio = float(split_cfg.get("train", 0.7))
        val_ratio = float(split_cfg.get("val", 0.15))
        test_ratio = float(split_cfg.get("test", 0.15))
        seed = int(split_cfg.get("seed", 42))

        fde_cfg = cfg.get("fde", {})

        if not isinstance(fde_cfg, dict):
            fde_cfg = {}

        fde_mode = str(fde_cfg.get("mode", "legacy")).strip().lower()

        input_cfg = cfg.get("input", cfg.get("intput", {}))
        use_bathy = bool(input_cfg.get("use_bathymetry", True))
        use_src = bool(input_cfg.get("use_source", True))
        use_init_depth = bool(input_cfg.get("use_initial_depth", True))
        use_init_surface = bool(input_cfg.get("use_initial_surface", False))
        if "use_solver_id" in input_cfg:
            use_solver_id = bool(input_cfg.get("use_solver_id"))
        else:
            use_solver_id = (fde_mode == "multifidelity")

        target_cfg = cfg.get("target", {})
        target_mode = str(target_cfg.get("mode", "next_step"))
        target_variable = str(target_cfg.get("variable", "eta")).strip().lower()
        if target_mode.strip().lower() == "rollout":
            target_mode = "multi_step"
        forecast_steps = int(target_cfg.get("forecast_steps", 10))
        stride = int(target_cfg.get("stride", 1))

        if target_variable not in ("eta", "depth", "state"):
            raise ValueError("target.variable must be one of: eta, depth, state")

        norm_cfg = cfg.get("normalization", {})
        norm_method = str(norm_cfg.get("method", "standardize"))
        norm_channels = {
            "bathymetry": bool(norm_cfg.get("channels", {}).get("bathymetry", True)),
            "source": bool(norm_cfg.get("channels", {}).get("source", True)),
            "solver_id": bool(norm_cfg.get("channels", {}).get("solver_id", False)),
            "trajectory": bool(norm_cfg.get("channels", {}).get("trajectory", True)),
        }
        eps = float(norm_cfg.get("eps", 1e-6))
 
        saving_cfg = cfg.get("saving", {})
        save_format = str(saving_cfg.get("format", "npy"))
        compress = bool(saving_cfg.get("compress", True))
        include_meta = bool(saving_cfg.get("include_meta", True))

        eval_cfg = cfg.get("eval_export", {})
        export_eval_arrays = bool(eval_cfg.get("enabled", True))
        eval_input_order = list(eval_cfg.get("input_order", ["bathymetry", "source", "initial_depth", "initial_surface"]))
        eval_inputs_name = str(eval_cfg.get("inputs_name", "inputs.npy"))
        eval_targets_name = str(eval_cfg.get("targets_name", "targets.npy"))
        eval_ids_name = str(eval_cfg.get("ids_name", "sample_id.npy"))
        eval_archive_name = str(eval_cfg.get("archive_name", "eval_dataset.npz"))
        eval_manifest_name = str(eval_cfg.get("manifest_name", "eval_manifest.json"))

        raw_cfg = cfg.get("raw", {})
        self.scenario_manifest_path: pathlib.Path | None = None
        self.fde_manifest_paths: Dict[str, pathlib.Path] = {}
        self.fde_raw_dirs: Dict[str, pathlib.Path] = {}
        if isinstance(raw_cfg, dict):
            scenario_manifest = raw_cfg.get("scenario_manifest")
            if scenario_manifest:
                self.scenario_manifest_path = pathlib.Path(str(scenario_manifest))

            for key, value in dict(raw_cfg.get("fde_manifests", {})).items():
                self.fde_manifest_paths[self._canonical_fde_name(str(key))] = pathlib.Path(str(value))
            for key, value in dict(raw_cfg.get("raw_dirs", {})).items():
                self.fde_raw_dirs[self._canonical_fde_name(str(key))] = pathlib.Path(str(value))

        self.fde_mode = fde_mode
        self.fde_targets = [self._canonical_fde_name(str(v)) for v in list(fde_cfg.get("targets", []))]

        if not self.fde_targets and self.fde_manifest_paths:
            self.fde_targets = [next(iter(self.fde_manifest_paths.keys()))]

        target_field = fde_cfg.get("target_field", None)
        if target_field is not None:
            target_variable = str(target_field).strip().lower()

        total = train_ratio + val_ratio + test_ratio

        if not np.isclose(total, 1.0):
            raise ValueError("splits ratio must sum to 1")

        self.cfg = PreprocessConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            manifest_path=manifest_path,
            split_train=train_ratio,
            split_val=val_ratio,
            split_test=test_ratio,
            seed=seed,
            use_bathymetry=use_bathy,
            use_source=use_src,
            use_initial_depth=use_init_depth,
            use_initial_surface=use_init_surface,
            use_solver_id=use_solver_id,
            target_mode=target_mode,
            target_variable=target_variable,
            forecast_steps=forecast_steps,
            stride=stride,
            norm_method=norm_method,
            norm_channels=norm_channels,
            eps=eps,
            save_format=save_format,
            compress=compress,
            include_meta=include_meta,
            export_eval_arrays=export_eval_arrays,
            eval_input_order=eval_input_order,
            eval_inputs_name=eval_inputs_name,
            eval_targets_name=eval_targets_name,
            eval_ids_name=eval_ids_name,
            eval_archive_name=eval_archive_name,
            eval_manifest_name=eval_manifest_name,
        )

        # create output dir
        self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)

        # placeholders
        self._mean: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._target_min: float = 0.0
        self._target_max: float = 1.0
        solver_vocab = self.fde_targets if self.fde_targets else sorted(set(self.fde_manifest_paths.keys()))

        if not solver_vocab:
            solver_vocab = ["hydrostatic", "muscl_hr", "boussinesq"]
        
        self._solver_id_map: Dict[str, float] = {name: float(i) for i, name in enumerate(solver_vocab)}

    def load_manifest(self) -> List[Dict[str, Any]]:
        """ load the raw manifest file and sample metadata """
        if not self.cfg.manifest_path.is_file():
            raise FileNotFoundError(f"Could not find {self.cfg.manifest_path}")
        
        records: List[Dict[str, Any]] = []

        with self.cfg.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    records.append(json.loads(line))

        return records

    def load_sample(self, sample_dir: Union[str, pathlib.Path]) -> Dict[str, Any]:
        """
        load one raw sample:
        - bathymetry
        - source_field
        - trajectory
        - timestamps
        - meta.json
        """
        sample_dir = pathlib.Path(sample_dir)
        npz_path = sample_dir / "sample.npz"
        meta_path = sample_dir / "meta.json"

        if not npz_path.is_file():
            raise FileNotFoundError(f"missing {npz_path}")

        data = dict(np.load(npz_path))

        for k, v in data.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.number):
                data[k] = arr.astype(np.float32)
            else:
                data[k] = arr

        meta: Dict[str, Any] = {}

        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        data["meta"] = meta
        data["sample_dir"] = str(sample_dir)

        return data

    def build_example(self, raw_sample: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """ convert one raw sample into model-ready IO tensors """
        X = self.select_input_channels(raw_sample)
        Y = self.select_target_channels(raw_sample)

        return X, Y

    def select_input_channels(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        build X from the raw arrays

        Possible inputs:
        - bathymetry only
        - bathymetry + source
        - bathymetry + source + initial state
        """
        X: Dict[str, np.ndarray] = {}

        if self.cfg.use_bathymetry:
            if "bathymetry" not in sample:
                raise KeyError("bathymetry not in sample")
            X["bathymetry"] = sample["bathymetry"]

        if self.cfg.use_source:
            if "source_field" not in sample:
                raise KeyError("source field not present in sample")
            X["source"] = sample["source_field"]

        if self.cfg.use_initial_depth:
            if "initial_depth" not in sample:
                raise KeyError("initial depth not present in sample")
            X["initial_depth"] = sample["initial_depth"]

        if self.cfg.use_initial_surface:
            if "initial_surface" in sample:
                X["initial_surface"] = sample["initial_surface"]
            elif "free_surface0" in sample:
                X["initial_surface"] = sample["free_surface0"]
            else:
                raise KeyError("initial free surface not present in sample (expected initial_surface or free_surface0)")

        if self.cfg.use_solver_id:
            raw_meta = sample.get("meta", {})
            meta: Dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            solver_name = self._canonical_fde_name(
                str(meta.get("solver_name", meta.get("primary_fde", "unknown")))
            )
            solver_code = float(self._solver_id_map.get(solver_name, -1.0))

            if X:
                ref = next(iter(X.values()))
            elif "bathymetry" in sample:
                ref = sample["bathymetry"]
            elif "source_field" in sample:
                ref = sample["source_field"]
            elif "initial_depth" in sample:
                ref = sample["initial_depth"]
            else:
                raise KeyError("could not infer shape for solver_id channel")

            X["solver_id"] = np.full_like(np.asarray(ref, dtype=np.float32), solver_code, dtype=np.float32)

        return X

    def select_target_channels(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        build Y from the trajectory

        Possible targets:
        - next-step prediction
        - multi-step rollout
        - final-state forecast
        """
        traj: np.ndarray = sample["trajectory"]
        traj = np.asarray(traj, dtype=np.float32)

        if traj.ndim not in (3, 4):
            raise ValueError(f"trajectory must have shape [T,H,W] or [T,C,H,W], got {traj.shape}")

        variable = self.cfg.target_variable
        if variable == "state":
            target_source = traj
        else:
            if variable == "eta" and "trajectory_eta" in sample:
                target_source = np.asarray(sample["trajectory_eta"], dtype=np.float32)
                if target_source.ndim != 3:
                    raise ValueError(
                        f"trajectory_eta must have shape [T,H,W], got {target_source.shape}"
                    )
            else:
                if traj.ndim == 4:
                    depth_frames = traj[:, 0]
                else:
                    depth_frames = traj

                if variable == "depth":
                    target_source = depth_frames
                else:  # eta
                    if "bathymetry" not in sample:
                        raise KeyError("bathymetry is required to build eta targets")
                    bathy = np.asarray(sample["bathymetry"], dtype=np.float32)
                    target_source = depth_frames + bathy[None, ...]

        if self.cfg.target_mode == "next_step":
            index = min(max(1, self.cfg.stride), target_source.shape[0] - 1)
            selected = target_source[index]
            if variable in ("eta", "depth"):
                return selected[None, ...]
            return selected

        if self.cfg.target_mode == "multi_step":
            end = 1 + self.cfg.forecast_steps * self.cfg.stride

            return target_source[1: end: self.cfg.stride]

        if self.cfg.target_mode == "final_state":
            selected = target_source[-1]
            if variable in ("eta", "depth"):
                return selected[None, ...]
            return selected
        
        raise ValueError(f"Unsupported target mode: {self.cfg.target_mode}")

    def fit_normalizer(self, train_samples: List[Tuple[Dict[str, np.ndarray], np.ndarray]]) -> None:
        """ compute normalization statistics from training data only """
        sums: Dict[str, float] = {}
        sq_sums: Dict[str, float] = {}
        mins: Dict[str, float] = {}
        maxs: Dict[str, float] = {}
        counts: Dict[str, float] = {}

        target_sum = 0.0
        target_sq_sum = 0.0
        target_count = 0.0
        target_min = np.inf
        target_max = -np.inf

        for X, Y in train_samples:
            for name, arr in X.items():
                if not self.cfg.norm_channels.get(name, False):
                    continue
            
                flat = arr.ravel()
                sums[name] = sums.get(name, 0.0) + flat.sum()
                sq_sums[name] = sq_sums.get(name, 0.0) + (flat ** 2).sum()
                mins[name] = min(mins.get(name, np.inf), flat.min())
                maxs[name] = max(maxs.get(name, -np.inf), flat.max())
                counts[name] = counts.get(name, 0) + flat.size

            if self.cfg.norm_channels.get("trajectory", False):
                y_flat = np.asarray(Y, dtype=np.float32).ravel()
                target_sum += float(y_flat.sum())
                target_sq_sum += float((y_flat ** 2).sum())
                target_count += float(y_flat.size)
                target_min = min(target_min, float(y_flat.min()))
                target_max = max(target_max, float(y_flat.max()))

        for name in sums:
            if self.cfg.norm_method == "standardize":
                mean = sums[name] / counts[name] 
                var = sq_sums[name] / counts[name] - mean ** 2
                std = np.sqrt(max(var, self.cfg.eps))
                
                self._mean[name] = float(mean)
                self._stds[name] = float(std)

            elif self.cfg.norm_method == "minmax":
                self._mean[name] = float(mins[name])
                self._stds[name] = float(maxs[name] - mins[name] + self.cfg.eps)
            
            else:
                raise ValueError(f"Unsupported normalization method: {self.cfg.norm_method}")

        if self.cfg.norm_channels.get("trajectory", False):
            if target_count <= 0:
                raise ValueError("trajectory normalization is enabled but no target samples were found.")

            if self.cfg.norm_method == "standardize":
                mean = target_sum / target_count
                var = target_sq_sum / target_count - mean ** 2
                std = np.sqrt(max(var, self.cfg.eps))
                self._target_mean = float(mean)
                self._target_std = float(std)
                self._target_min = float(target_min)
                self._target_max = float(target_max)

            elif self.cfg.norm_method == "minmax":
                self._target_mean = float(target_min)
                self._target_std = float(target_max - target_min + self.cfg.eps)
                self._target_min = float(target_min)
                self._target_max = float(target_max)

            else:
                raise ValueError(f"Unsupported normalization method: {self.cfg.norm_method}")

    def normalize_sample(self, X: Dict[str, np.ndarray], Y: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """ apply normalization using training statistics """
        X_norm : Dict[str, np.ndarray] = {}
        
        for name, arr in X.items():
            if not self.cfg.norm_channels.get(name, False):
                X_norm[name] = arr
                continue

            if name not in self._mean:
                raise KeyError(f"Normalization statistic for {name} not available")

            if self.cfg.norm_method == "standardize":
                X_norm[name] = (arr - self._mean[name]) / self._stds[name]

            else:
                X_norm[name] = (arr - self._mean[name]) / self._stds[name]

        Y_norm = np.asarray(Y, dtype=np.float32)
        if self.cfg.norm_channels.get("trajectory", False):
            Y_norm = (Y_norm - self._target_mean) / self._target_std

        return X_norm, Y_norm

    @staticmethod
    def _meta_string(meta: Dict[str, Any], key: str, default: str = "unknown") -> str:
        value = meta.get(key, default)
        if value is None:
            return default

        return str(value)

    @staticmethod
    def _meta_float(meta: Dict[str, Any], key: str, default: float = np.nan) -> float:
        value = meta.get(key, default)
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _canonical_fde_name(name: str) -> str:
        n = str(name).strip().lower()
        mapping = {
            "swe_hydrostatic": "hydrostatic",
            "hydrostatic": "hydrostatic",
            "swe_muscl": "muscl_hr",
            "swe_muscl_hr": "muscl_hr",
            "muscl": "muscl_hr",
            "muscl_hr": "muscl_hr",
            "boussinesq": "boussinesq",
        }
        return mapping.get(n, n)

    @staticmethod
    def _load_manifest_path(manifest_path: pathlib.Path) -> List[Dict[str, Any]]:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Could not find {manifest_path}")

        records: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _scenario_id_from_record(rec: Dict[str, Any]) -> str:
        if "scenario_id" in rec and rec["scenario_id"] is not None:
            return str(rec["scenario_id"])
        if "sample_index" in rec:
            try:
                return f"scenario_{int(rec['sample_index']):06d}"
            except Exception:
                pass
        if "sample_dir" in rec:
            name = pathlib.Path(str(rec["sample_dir"])).name
            if name.startswith("sample_"):
                suffix = name.split("sample_", 1)[-1]
                if suffix.isdigit():
                    return f"scenario_{int(suffix):06d}"
        return "scenario_unknown"

    def _split_scenario_ids(self, records: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str], Set[str]]:
        scenario_ids = sorted({self._scenario_id_from_record(rec) for rec in records})
        random.seed(self.cfg.seed)
        random.shuffle(scenario_ids)

        n = len(scenario_ids)
        n_train = int(self.cfg.split_train * n)
        n_val = int(self.cfg.split_val * n)

        train_ids = set(scenario_ids[:n_train])
        val_ids = set(scenario_ids[n_train:n_train + n_val])
        test_ids = set(scenario_ids[n_train + n_val:])
        return train_ids, val_ids, test_ids

    def _records_for_scenarios(
        self,
        records: List[Dict[str, Any]],
        scenario_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        return [rec for rec in records if self._scenario_id_from_record(rec) in scenario_ids]

    def split_dataset(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """ split raw samples into train / val / test """
        random.seed(self.cfg.seed)
        shuffled = records.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(self.cfg.split_train * n)
        n_val = int(self.cfg.split_val * n)

        train = shuffled[: n_train]
        val = shuffled[n_train: n_train + n_val]
        test = shuffled[n_train + n_val :]

        return train, val, test

    def _process_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[Dict[str, Any]], List[str]]:
        Xs: List[Dict[str, np.ndarray]] = []
        Ys: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        sample_ids: List[str] = []

        for rec in records:
            raw = self.load_sample(rec["sample_dir"])
            X, Y = self.build_example(raw)
            Xs.append(X)
            Ys.append(Y)
            if "sample_index" in rec:
                sample_id = f"sample_{int(rec['sample_index']):06d}"
            else:
                sample_id = pathlib.Path(rec["sample_dir"]).name
            sample_ids.append(sample_id)

            merged_meta: Dict[str, Any] = {}
            merged_meta.update(rec if isinstance(rec, dict) else {})
            raw_meta = raw.get("meta", {})
            if isinstance(raw_meta, dict):
                merged_meta.update(raw_meta)

            merged_meta.setdefault("source_type", "unknown")
            merged_meta.setdefault("bathymetry_type", "unknown")
            merged_meta.setdefault("source_strength", np.nan)
            merged_meta.setdefault("scenario_id", merged_meta.get("scenario_id", sample_id))
            merged_meta.setdefault("solver_name", merged_meta.get("solver_name", "unknown"))
            metas.append(merged_meta)

        return Xs, Ys, metas, sample_ids

    def _normalize_and_save(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
        output_dir: pathlib.Path,
    ) -> None:
        X_train_raw, Y_train_raw, meta_train, ids_train = self._process_records(train_records)
        X_val_raw, Y_val_raw, meta_val, ids_val = self._process_records(val_records)
        X_test_raw, Y_test_raw, meta_test, ids_test = self._process_records(test_records)

        self.fit_normalizer(list(zip(X_train_raw, Y_train_raw)))

        def _normalize(X_raw: List[Dict[str, np.ndarray]], Y_raw: List[np.ndarray]) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]:
            X_norm: List[Dict[str, np.ndarray]] = []
            Y_norm: List[np.ndarray] = []
            for X, Y in zip(X_raw, Y_raw):
                X_n, Y_n = self.normalize_sample(X, Y)
                X_norm.append(X_n)
                Y_norm.append(Y_n)
            return X_norm, Y_norm

        X_train, Y_train = _normalize(X_train_raw, Y_train_raw)
        X_val, Y_val = _normalize(X_val_raw, Y_val_raw)
        X_test, Y_test = _normalize(X_test_raw, Y_test_raw)

        original_processed_dir = self.cfg.processed_dir
        self.cfg.processed_dir = output_dir
        self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        self._save_normalization_stats()
        self.save_split("train", X_train, Y_train, meta_train, ids_train)
        self.save_split("val", X_val, Y_val, meta_val, ids_val)
        self.save_split("test", X_test, Y_test, meta_test, ids_test)
        self.cfg.processed_dir = original_processed_dir

    def _resolved_eval_input_order(self, sample_inputs: Dict[str, np.ndarray]) -> List[str]:
        preferred = [name for name in self.cfg.eval_input_order if name in sample_inputs]
        extras = [name for name in sample_inputs.keys() if name not in preferred]
        return preferred + extras

    def save_split(self, split_name: str, X: List[Dict[str, np.ndarray]], Y: List[np.ndarray],
                   meta_list: List[Dict[str, Any]], sample_ids: List[str]) -> None:
        """ save processed arrays and metadata """
        out_dir = self.cfg.processed_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(X) == 0 or len(Y) == 0:
            if self.cfg.include_meta:
                with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
                    f.write("")
            return

        # save each input channel
        channel_names = list(X[0].keys())

        for channel in channel_names:
            stacked = np.stack([x[channel] for x in X], axis=0)
            npz_path = out_dir / f"X_{channel}.npz"

            if self.cfg.compress:
                np.savez_compressed(npz_path, data=stacked)
            
            else:
                np.savez(npz_path, data=stacked)

        # save targets
        Y_arr = np.stack(Y, axis=0)
        Y_path = out_dir / "Y.npy"
        np.save(Y_path, Y_arr)

        if self.cfg.export_eval_arrays:
            input_order = self._resolved_eval_input_order(X[0])
            eval_inputs = np.stack(
                [np.stack([sample[channel] for channel in input_order], axis=0) for sample in X],
                axis=0,
            ).astype(np.float32)
            eval_targets = Y_arr.astype(np.float32)
            eval_ids = np.asarray(sample_ids, dtype=np.str_)
            source_types = np.asarray(
                [self._meta_string(meta, "source_type", "unknown") for meta in meta_list],
                dtype=np.str_,
            )
            bathymetry_types = np.asarray(
                [self._meta_string(meta, "bathymetry_type", "unknown") for meta in meta_list],
                dtype=np.str_,
            )
            source_strengths = np.asarray(
                [self._meta_float(meta, "source_strength", np.nan) for meta in meta_list],
                dtype=np.float32,
            )
            scenario_ids = np.asarray(
                [
                    self._meta_string(meta, "scenario_id", default=sample_ids[i])
                    for i, meta in enumerate(meta_list)
                ],
                dtype=np.str_,
            )
            solver_names = np.asarray(
                [
                    self._meta_string(
                        meta,
                        "solver_name",
                        default=self._meta_string(meta, "primary_fde", "unknown"),
                    )
                    for meta in meta_list
                ],
                dtype=np.str_,
            )

            np.save(out_dir / self.cfg.eval_inputs_name, eval_inputs)
            np.save(out_dir / self.cfg.eval_targets_name, eval_targets)
            np.save(out_dir / self.cfg.eval_ids_name, eval_ids)
            np.savez_compressed(
                out_dir / self.cfg.eval_archive_name,
                inputs=eval_inputs,
                targets=eval_targets,
                sample_id=eval_ids,
                source_id=source_types,
                source_type=source_types,
                bathymetry_type=bathymetry_types,
                source_strength=source_strengths,
                scenario_id=scenario_ids,
                solver_name=solver_names,
                target_variable=np.asarray([self.cfg.target_variable], dtype=np.str_),
                target_mean=np.asarray([self._target_mean], dtype=np.float32),
                target_std=np.asarray([self._target_std], dtype=np.float32),
                target_min=np.asarray([self._target_min], dtype=np.float32),
                target_max=np.asarray([self._target_max], dtype=np.float32),
            )

            eval_manifest = {
                "split": split_name,
                "inputs_name": self.cfg.eval_inputs_name,
                "targets_name": self.cfg.eval_targets_name,
                "ids_name": self.cfg.eval_ids_name,
                "archive_name": self.cfg.eval_archive_name,
                "input_order": input_order,
                "target_mode": self.cfg.target_mode,
                "target_variable": self.cfg.target_variable,
                "normalized_targets": bool(self.cfg.norm_channels.get("trajectory", False)),
                "inputs_shape": list(map(int, eval_inputs.shape)),
                "targets_shape": list(map(int, eval_targets.shape)),
            }
            with (out_dir / self.cfg.eval_manifest_name).open("w", encoding="utf-8") as f:
                json.dump(eval_manifest, f, indent=2)

        if self.cfg.include_meta:
            meta_path = out_dir / "meta.jsonl"

            with meta_path.open("w", encoding="utf-8") as f:
                for m in meta_list:
                    f.write(json.dumps(m) + "\n")

    def _save_normalization_stats(self) -> None:
        stats = {
            "method": self.cfg.norm_method,
            "eps": float(self.cfg.eps),
            "inputs": {
                name: {
                    "offset": float(self._mean[name]),
                    "scale": float(self._stds[name]),
                }
                for name in sorted(self._mean.keys())
            },
            "targets": {
                "enabled": bool(self.cfg.norm_channels.get("trajectory", False)),
                "variable": self.cfg.target_variable,
                "offset": float(self._target_mean),
                "scale": float(self._target_std),
                "min": float(self._target_min),
                "max": float(self._target_max),
            },
        }
        with (self.cfg.processed_dir / "normalization_stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def run(self) -> None:
        mode = self.fde_mode
        if mode in {"single", "separate_all", "multifidelity"} and self.fde_manifest_paths:
            requested = self.fde_targets if self.fde_targets else list(self.fde_manifest_paths.keys())
            targets = [name for name in requested if name in self.fde_manifest_paths]
            if not targets:
                raise ValueError(
                    f"No valid fde.targets found for mode={mode}. Available: {sorted(self.fde_manifest_paths.keys())}"
                )

            # Build one shared scenario split for fair cross-FDE comparison.
            if self.scenario_manifest_path is not None and self.scenario_manifest_path.exists():
                split_source = self._load_manifest_path(self.scenario_manifest_path)
            else:
                split_source = self._load_manifest_path(self.fde_manifest_paths[targets[0]])
            train_ids, val_ids, test_ids = self._split_scenario_ids(split_source)

            if mode == "single":
                fde_name = targets[0]
                records = self._load_manifest_path(self.fde_manifest_paths[fde_name])
                train_records = self._records_for_scenarios(records, train_ids)
                val_records = self._records_for_scenarios(records, val_ids)
                test_records = self._records_for_scenarios(records, test_ids)
                out_dir = self.cfg.processed_dir / fde_name
                print(f"[preprocess] mode=single fde={fde_name} out={out_dir}")
                self._normalize_and_save(train_records, val_records, test_records, out_dir)
                return

            if mode == "separate_all":
                for fde_name in targets:
                    records = self._load_manifest_path(self.fde_manifest_paths[fde_name])
                    train_records = self._records_for_scenarios(records, train_ids)
                    val_records = self._records_for_scenarios(records, val_ids)
                    test_records = self._records_for_scenarios(records, test_ids)
                    out_dir = self.cfg.processed_dir / fde_name
                    print(f"[preprocess] mode=separate_all fde={fde_name} out={out_dir}")
                    self._normalize_and_save(train_records, val_records, test_records, out_dir)
                return

            # multifidelity
            train_records: List[Dict[str, Any]] = []
            val_records: List[Dict[str, Any]] = []
            test_records: List[Dict[str, Any]] = []
            for fde_name in targets:
                records = self._load_manifest_path(self.fde_manifest_paths[fde_name])
                train_records.extend(self._records_for_scenarios(records, train_ids))
                val_records.extend(self._records_for_scenarios(records, val_ids))
                test_records.extend(self._records_for_scenarios(records, test_ids))

            out_dir = self.cfg.processed_dir / "multifidelity"
            print(f"[preprocess] mode=multifidelity targets={targets} out={out_dir}")
            self._normalize_and_save(train_records, val_records, test_records, out_dir)
            return

        # Legacy mode (single manifest path + one processed root).
        print("Loading manifest")
        manifest = self.load_manifest()
        print("Splitting dataset")
        train_records, val_records, test_records = self.split_dataset(manifest)
        print("Building dataset")
        self._normalize_and_save(train_records, val_records, test_records, self.cfg.processed_dir)

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-process raw tsunami surrogate data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/preprocess.yaml",
        help="path to the preprocessing yaml configuration.",
    )
    return parser

def main() -> None:
    args = _build_argparser().parse_args()
    preproc = TsunamiPreprocessor(args.config)
    preproc.run()

if __name__ == "__main__":
    main()
