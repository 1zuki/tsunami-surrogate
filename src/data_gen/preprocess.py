from __future__ import annotations

import json
import os
import pathlib
import random
import shutil
import sys
import uuid
import numpy as np
import yaml
import argparse

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.hashing import sha256_file, stable_json_sha256

NORMALIZATION_STATS_SCHEMA_ID = "tsunami-surrogate.normalization-stats.v2"
PROCESSED_MANIFEST_SCHEMA_ID = "tsunami-surrogate.processed-dataset.v2"


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
    norm_reference_stats_path: Optional[pathlib.Path]
    eps: float

    save_format: str # npy
    compress: bool
    include_meta: bool
    sharded: bool
    shard_size: int
    write_legacy_eval_archive: bool
    publication_mode: str # replace / merge_split

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
        norm_reference_stats_path_raw = norm_cfg.get("reference_stats_path", None)
        norm_reference_stats_path: Optional[pathlib.Path] = None
        if norm_reference_stats_path_raw:
            norm_reference_stats_path = pathlib.Path(str(norm_reference_stats_path_raw))
        eps = float(norm_cfg.get("eps", 1e-6))
 
        saving_cfg = cfg.get("saving", {})
        save_format = str(saving_cfg.get("format", "npy"))
        compress = bool(saving_cfg.get("compress", True))
        include_meta = bool(saving_cfg.get("include_meta", True))
        sharded = bool(saving_cfg.get("sharded", False))
        shard_size = int(saving_cfg.get("shard_size", 128))
        if shard_size <= 0:
            raise ValueError("saving.shard_size must be a positive integer")
        write_legacy_eval_archive = bool(
            saving_cfg.get(
                "write_legacy_eval_archive",
                saving_cfg.get(
                    "write_legacy_test_archive",
                    saving_cfg.get(
                        "write_legacy_val_archive",
                        saving_cfg.get("write_legacy_train_archive", False),
                    ),
                ),
            )
        )
        publication_mode = str(
            saving_cfg.get("publication_mode", "replace")
        ).strip().lower()
        if publication_mode not in {"replace", "merge_split"}:
            raise ValueError(
                "saving.publication_mode must be one of: replace, merge_split"
            )

        eval_cfg = cfg.get(
            "eval_export",
            cfg.get("test_export", cfg.get("val_export", cfg.get("train_export", {}))),
        )
        export_eval_arrays = bool(eval_cfg.get("enabled", True))
        eval_input_order = list(eval_cfg.get("input_order", ["bathymetry", "source", "initial_depth", "initial_surface"]))
        eval_inputs_name = str(eval_cfg.get("inputs_name", "inputs.npy"))
        eval_targets_name = str(eval_cfg.get("targets_name", "targets.npy"))
        eval_ids_name = str(eval_cfg.get("ids_name", "sample_id.npy"))
        eval_archive_name = str(eval_cfg.get("archive_name", "eval_dataset.npz"))
        eval_manifest_name = str(eval_cfg.get("manifest_name", "eval_manifest.json"))

        raw_cfg = cfg.get("raw", {})
        self.expected_publication_split: str | None = None
        self.scenario_manifest_path: pathlib.Path | None = None
        self.fde_manifest_paths: Dict[str, pathlib.Path] = {}
        self.fde_raw_dirs: Dict[str, pathlib.Path] = {}
        if isinstance(raw_cfg, dict):
            publication_split = raw_cfg.get("publication_split")
            if publication_split is not None:
                self.expected_publication_split = str(
                    publication_split
                ).strip()
                if not self.expected_publication_split:
                    raise ValueError("raw.publication_split must not be empty")
            scenario_manifest = raw_cfg.get("scenario_manifest")
            if scenario_manifest:
                self.scenario_manifest_path = pathlib.Path(str(scenario_manifest))

            for key, value in dict(raw_cfg.get("fde_manifests", {})).items():
                self.fde_manifest_paths[self._canonical_fde_name(str(key))] = pathlib.Path(str(value))
            for key, value in dict(raw_cfg.get("raw_dirs", {})).items():
                self.fde_raw_dirs[self._canonical_fde_name(str(key))] = pathlib.Path(str(value))

        self.fde_mode = fde_mode
        self.fde_targets = [self._canonical_fde_name(str(v)) for v in list(fde_cfg.get("targets", []))]
        self.fde_norm_reference_paths: Dict[str, pathlib.Path] = {}
        raw_norm_map = norm_cfg.get("reference_stats_by_fde", {})
        if isinstance(raw_norm_map, dict):
            for key, value in raw_norm_map.items():
                if value:
                    self.fde_norm_reference_paths[self._canonical_fde_name(str(key))] = pathlib.Path(str(value))

        if not self.fde_targets and self.fde_manifest_paths:
            self.fde_targets = [next(iter(self.fde_manifest_paths.keys()))]

        target_field = fde_cfg.get("target_field", None)
        if target_field is not None:
            target_variable = str(target_field).strip().lower()

        total = train_ratio + val_ratio + test_ratio

        if not np.isclose(total, 1.0):
            raise ValueError("splits ratio must sum to 1")
        if min(train_ratio, val_ratio, test_ratio) < 0.0:
            raise ValueError("split ratios must be non-negative")
        active_splits = [
            name
            for name, ratio in (
                ("train", train_ratio),
                ("val", val_ratio),
                ("test", test_ratio),
            )
            if ratio > 0.0
        ]
        if publication_mode == "merge_split":
            if len(active_splits) != 1:
                raise ValueError(
                    "saving.publication_mode=merge_split requires exactly one "
                    "non-zero split"
                )
            if active_splits[0] == "train":
                raise ValueError(
                    "Training preprocessing must use publication_mode=replace "
                    "because fitting new normalization statistics invalidates "
                    "existing validation/test splits"
                )

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
            norm_reference_stats_path=norm_reference_stats_path,
            eps=eps,
            save_format=save_format,
            compress=compress,
            include_meta=include_meta,
            sharded=sharded,
            shard_size=shard_size,
            write_legacy_eval_archive=write_legacy_eval_archive,
            publication_mode=publication_mode,
            export_eval_arrays=export_eval_arrays,
            eval_input_order=eval_input_order,
            eval_inputs_name=eval_inputs_name,
            eval_targets_name=eval_targets_name,
            eval_ids_name=eval_ids_name,
            eval_archive_name=eval_archive_name,
            eval_manifest_name=eval_manifest_name,
        )

        # placeholders
        self._mean: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._target_min: float = 0.0
        self._target_max: float = 1.0
        self._active_norm_reference_path: Optional[pathlib.Path] = None
        self._loaded_normalization_payload: Dict[str, Any] = {}
        self._normalization_stats_sha256: Optional[str] = None
        self._normalization_provenance: Dict[str, Any] = {}
        self._preprocess_config_sha256 = sha256_file(self.config_path)
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

    def load_sample(
        self,
        sample_dir: Union[str, pathlib.Path],
        *,
        expected_record: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        load one raw sample:
        - bathymetry
        - source_field
        - trajectory or eta-primary requested target
        - timestamps
        - meta.json
        """
        sample_dir = pathlib.Path(sample_dir)
        npz_path = sample_dir / "sample.npz"
        meta_path = sample_dir / "meta.json"
        publication_path = sample_dir / "publication.json"

        if not npz_path.is_file():
            raise FileNotFoundError(f"missing {npz_path}")

        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        payload_schema = ""
        payload_contract = ""
        try:
            with np.load(npz_path, allow_pickle=False) as payload:
                if "schema_id" in payload:
                    payload_schema = str(
                        np.asarray(payload["schema_id"]).reshape(-1)[0]
                    )
                if "contract_hash" in payload:
                    payload_contract = str(
                        np.asarray(payload["contract_hash"]).reshape(-1)[0]
                    )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not inspect sample payload: {npz_path}") from exc

        expects_v2_publication = bool(
            (expected_record or {}).get("contract_hash")
            or meta.get("contract_hash")
            or payload_contract
            or payload_schema.startswith("tsunami-surrogate.common-time-v2.")
        )
        if expects_v2_publication and not publication_path.is_file():
            raise RuntimeError(
                "Common-time-v2 preprocessing requires publication.json: "
                f"{sample_dir}"
            )

        if publication_path.is_file():
            try:
                from src.data_gen.common_time_v2 import (
                    CONTRACT_SCHEMA_ID,
                    ETA_SAMPLE_SCHEMA_ID,
                    validate_candidate_times,
                    validate_publication,
                )
            except ImportError:
                from common_time_v2 import (  # type: ignore
                    CONTRACT_SCHEMA_ID,
                    ETA_SAMPLE_SCHEMA_ID,
                    validate_candidate_times,
                    validate_publication,
                )

            expected_identity = None
            expected_solver_name = None
            expected_sample_index = None
            if expected_record is not None:
                scenario_id = self._scenario_id_from_record(expected_record)
                if self.expected_publication_split is not None:
                    expected_identity = {
                        "split": self.expected_publication_split,
                        "scenario_id": scenario_id,
                    }
                expected_solver_name = expected_record.get("solver_name")
                if expected_record.get("sample_index") is not None:
                    expected_sample_index = int(expected_record["sample_index"])

            publication = validate_publication(
                sample_dir,
                expected_identity=expected_identity,
                expected_contract_hash=(
                    expected_record.get("contract_hash")
                    if expected_record is not None
                    else meta.get("contract_hash")
                ),
                expected_config_hash=(
                    expected_record.get("resolved_config_hash")
                    if expected_record is not None
                    else meta.get("resolved_config_hash")
                ),
                expected_code_state_hash=(
                    expected_record.get("code_state_hash")
                    if expected_record is not None
                    else meta.get("code_state_hash")
                ),
                expected_input_fingerprint=(
                    expected_record.get("input_fingerprint")
                    if expected_record is not None
                    else meta.get("input_fingerprint")
                ),
                expected_solver_name=expected_solver_name,
                expected_sample_index=expected_sample_index,
                expected_authoritative_input_fingerprint=(
                    expected_record.get("authoritative_input_fingerprint")
                    if expected_record is not None
                    else meta.get("authoritative_input_fingerprint")
                ),
            )
            if meta.get("schema_id") != ETA_SAMPLE_SCHEMA_ID:
                raise RuntimeError("Unsupported common-time-v2 sample schema")
            if publication.get("contract_hash") != meta.get("contract_hash"):
                raise RuntimeError("Publication/meta contract hash mismatch")
            for key in (
                "scenario_id",
                "sample_index",
                "solver_name",
                "resolved_config_hash",
                "code_state_hash",
                "input_fingerprint",
            ):
                expected = publication.get(key)
                observed = meta.get(key)
                if expected is not None and observed != expected:
                    raise RuntimeError(
                        f"Publication/meta {key} mismatch"
                    )
            with np.load(npz_path, allow_pickle=False) as payload:
                if "timestamps" not in payload:
                    raise RuntimeError("common-time-v2 sample is missing timestamps")
                validate_candidate_times(payload["timestamps"])
                schema_values = np.asarray(payload.get("schema_id", [])).reshape(-1)
                if not schema_values or str(schema_values[0]) != ETA_SAMPLE_SCHEMA_ID:
                    raise RuntimeError("sample.npz common-time-v2 schema mismatch")
                contract_values = np.asarray(payload.get("contract_hash", [])).reshape(-1)
                if not contract_values or str(contract_values[0]) != meta.get(
                    "contract_hash"
                ):
                    raise RuntimeError("sample.npz common-time-v2 contract mismatch")
            meta.setdefault("contract_schema_id", CONTRACT_SCHEMA_ID)

        data = dict(np.load(npz_path, allow_pickle=False))
        for k, v in data.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.number):
                data[k] = (
                    arr.astype(np.float64)
                    if k in {"timestamps", "source_strength"}
                    else arr.astype(np.float32)
                )
            else:
                data[k] = arr

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
        variable = self.cfg.target_variable
        if variable == "eta" and "trajectory_eta" in sample:
            target_source = np.asarray(sample["trajectory_eta"], dtype=np.float32)
            if target_source.ndim != 3:
                raise ValueError(
                    f"trajectory_eta must have shape [T,H,W], got {target_source.shape}"
                )
        else:
            if "trajectory" not in sample:
                raise KeyError(
                    f"trajectory is required for target.variable={variable!r}"
                )
            traj = np.asarray(sample["trajectory"], dtype=np.float32)
            if traj.ndim not in (3, 4):
                raise ValueError(
                    f"trajectory must have shape [T,H,W] or [T,C,H,W], got {traj.shape}"
                )

            if variable == "state":
                target_source = traj
            else:
                depth_frames = traj[:, 0] if traj.ndim == 4 else traj
                if variable == "depth":
                    target_source = depth_frames
                else:
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
            if sample.get("meta", {}).get("schema_id") == (
                "tsunami-surrogate.common-time-v2.eta-sample.v1"
            ):
                end = self.cfg.forecast_steps * self.cfg.stride
                return target_source[0:end:self.cfg.stride]
            end = 1 + self.cfg.forecast_steps * self.cfg.stride
            return target_source[1:end:self.cfg.stride]

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

    def fit_normalizer_from_records(self, train_records: List[Dict[str, Any]]) -> None:
        """Compute normalization stats without retaining all training samples."""
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

        for rec in train_records:
            X, Y, _, _ = self._record_to_example(rec)
            for name, arr in X.items():
                if not self.cfg.norm_channels.get(name, False):
                    continue

                flat = np.asarray(arr, dtype=np.float32).ravel()
                sums[name] = sums.get(name, 0.0) + float(flat.sum())
                sq_sums[name] = sq_sums.get(name, 0.0) + float((flat ** 2).sum())
                mins[name] = min(mins.get(name, np.inf), float(flat.min()))
                maxs[name] = max(maxs.get(name, -np.inf), float(flat.max()))
                counts[name] = counts.get(name, 0.0) + float(flat.size)

            if self.cfg.norm_channels.get("trajectory", False):
                y_flat = np.asarray(Y, dtype=np.float32).ravel()
                target_sum += float(y_flat.sum())
                target_sq_sum += float((y_flat ** 2).sum())
                target_count += float(y_flat.size)
                target_min = min(target_min, float(y_flat.min()))
                target_max = max(target_max, float(y_flat.max()))

        for name in sums:
            if counts[name] <= 0:
                continue

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

    def _first_sample_inputs(self, record_groups: List[List[Dict[str, Any]]]) -> Optional[Dict[str, np.ndarray]]:
        for records in record_groups:
            if not records:
                continue
            X, _, _, _ = self._record_to_example(records[0])
            return X
        return None

    def _normalization_enabled(self) -> bool:
        return any(bool(v) for v in self.cfg.norm_channels.values())

    def _resolve_normalization_reference_for_run(
        self,
        output_dir: pathlib.Path,
        train_records: List[Dict[str, Any]],
        norm_reference_stats_path: Optional[pathlib.Path],
    ) -> Optional[pathlib.Path]:
        if norm_reference_stats_path is not None:
            return norm_reference_stats_path

        if train_records:
            return None

        if self._normalization_enabled():
            raise ValueError(
                "No training records are available to fit normalization statistics. "
                "Set normalization.reference_stats_path or "
                "normalization.reference_stats_by_fde explicitly; implicit path-based "
                "normalization-stat discovery is disabled."
            )

        return None

    def _load_normalizer_from_stats_file(
        self,
        stats_path: pathlib.Path,
        sample_inputs: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if not stats_path.is_file():
            raise FileNotFoundError(f"Normalization reference stats not found: {stats_path}")

        with stats_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid normalization stats in {stats_path}: expected object")

        method = str(payload.get("method", "")).strip().lower()
        if method != self.cfg.norm_method.strip().lower():
            raise ValueError(
                f"Normalization method mismatch for {stats_path}: "
                f"stats={method!r}, config={self.cfg.norm_method!r}"
            )

        inputs = payload.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"Invalid normalization stats in {stats_path}: expected object at 'inputs'")

        self._mean = {}
        self._stds = {}
        for name, spec in inputs.items():
            if not isinstance(spec, dict):
                continue
            if "offset" not in spec or "scale" not in spec:
                continue

            offset = float(spec["offset"])
            scale = float(spec["scale"])
            if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Invalid normalization statistic for {name!r} in {stats_path}: "
                    f"offset={offset}, scale={scale}"
                )
            self._mean[str(name)] = offset
            self._stds[str(name)] = scale

        targets = payload.get("targets", {})
        trajectory_normalized = bool(self.cfg.norm_channels.get("trajectory", False))
        if trajectory_normalized and not isinstance(targets, dict):
            raise ValueError(
                f"Normalization stats {stats_path} is missing target statistics"
            )
        if isinstance(targets, dict):
            if trajectory_normalized:
                missing_target_fields = [
                    name for name in ("offset", "scale") if name not in targets
                ]
                if missing_target_fields:
                    raise ValueError(
                        f"Normalization stats {stats_path} is missing target fields: "
                        f"{missing_target_fields}"
                    )
            target_variable = str(targets.get("variable", "")).strip().lower()
            if target_variable and target_variable != self.cfg.target_variable:
                raise ValueError(
                    f"Normalization target mismatch for {stats_path}: "
                    f"stats={target_variable!r}, config={self.cfg.target_variable!r}"
                )
            self._target_mean = float(targets.get("offset", 0.0))
            self._target_std = float(targets.get("scale", 1.0))
            self._target_min = float(targets.get("min", 0.0))
            self._target_max = float(targets.get("max", 1.0))
        else:
            self._target_mean = 0.0
            self._target_std = 1.0
            self._target_min = 0.0
            self._target_max = 1.0

        if sample_inputs is not None:
            required = [k for k in sample_inputs.keys() if self.cfg.norm_channels.get(k, False)]
            missing = [k for k in required if k not in self._mean]
            if missing:
                raise KeyError(
                    f"Normalization stats {stats_path} missing required input channels: {missing}. "
                    f"Available={sorted(self._mean.keys())}"
                )

        if trajectory_normalized and self._target_std <= 0.0:
            raise ValueError(
                f"Normalization stats {stats_path} has invalid target scale={self._target_std}. "
                "Expected a positive target scale."
            )
        if not all(
            np.isfinite(value)
            for value in (
                self._target_mean,
                self._target_std,
                self._target_min,
                self._target_max,
            )
        ):
            raise ValueError(f"Normalization target statistics are non-finite in {stats_path}")

        self._loaded_normalization_payload = payload

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

    def _source_manifest_provenance(
        self, solver_names: Set[str]
    ) -> Dict[str, Dict[str, str]]:
        manifests: Dict[str, Dict[str, str]] = {}
        for solver_name in sorted(solver_names):
            canonical = self._canonical_fde_name(solver_name)
            path = self.fde_manifest_paths.get(canonical)
            if path is None and len(solver_names) == 1:
                path = self.cfg.manifest_path
            if path is not None and path.is_file():
                manifests[canonical] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
        return manifests

    def _source_lineage(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not records:
            return {
                "status": "empty",
                "sample_count": 0,
                "sample_identity_sha256": stable_json_sha256([]),
                "source_manifests": {},
            }

        rows: List[Dict[str, Any]] = []
        v2_count = 0
        contract_hashes: Set[str] = set()
        code_state_hashes: Set[str] = set()
        solver_config_hashes: Dict[str, Set[str]] = {}
        solver_names: Set[str] = set()
        required_v2 = (
            "contract_hash",
            "resolved_config_hash",
            "code_state_hash",
            "input_fingerprint",
        )

        for record in records:
            scenario_id = self._scenario_id_from_record(record)
            solver_name = self._canonical_fde_name(
                str(record.get("solver_name", record.get("primary_fde", "unknown")))
            )
            solver_names.add(solver_name)
            row: Dict[str, Any] = {
                "sample_index": record.get("sample_index"),
                "scenario_id": scenario_id,
                "solver_name": solver_name,
            }
            if record.get("contract_hash"):
                v2_count += 1
                missing = [key for key in required_v2 if not record.get(key)]
                if missing:
                    raise RuntimeError(
                        f"Common-time-v2 preprocessing record {scenario_id} is "
                        f"missing lineage fields: {missing}"
                    )
                contract_hash = str(record["contract_hash"])
                config_hash = str(record["resolved_config_hash"])
                code_hash = str(record["code_state_hash"])
                contract_hashes.add(contract_hash)
                code_state_hashes.add(code_hash)
                solver_config_hashes.setdefault(solver_name, set()).add(config_hash)
                row.update(
                    {
                        "contract_hash": contract_hash,
                        "resolved_config_hash": config_hash,
                        "code_state_hash": code_hash,
                        "input_fingerprint": str(record["input_fingerprint"]),
                    }
                )
            rows.append(row)

        if v2_count not in {0, len(records)}:
            raise RuntimeError("Refusing to preprocess mixed legacy and common-time-v2 records")
        if len(contract_hashes) > 1:
            raise RuntimeError(
                f"Refusing mixed common-time-v2 contracts: {sorted(contract_hashes)}"
            )
        if len(code_state_hashes) > 1:
            raise RuntimeError(
                f"Refusing mixed common-time-v2 code states: {sorted(code_state_hashes)}"
            )
        for solver_name, hashes in solver_config_hashes.items():
            if len(hashes) > 1:
                raise RuntimeError(
                    f"Refusing mixed resolved configs for {solver_name}: {sorted(hashes)}"
                )

        rows.sort(
            key=lambda row: (
                str(row["scenario_id"]),
                str(row["solver_name"]),
                int(row["sample_index"] or 0),
            )
        )
        lineage: Dict[str, Any] = {
            "status": "bound_common_time_v2" if v2_count else "legacy_unbound",
            "sample_count": len(rows),
            "sample_identity_sha256": stable_json_sha256(rows),
            "solver_names": sorted(solver_names),
            "source_manifests": self._source_manifest_provenance(solver_names),
        }
        if v2_count:
            lineage.update(
                {
                    "contract_hash": next(iter(contract_hashes)),
                    "code_state_hash": next(iter(code_state_hashes)),
                    "resolved_config_hashes": {
                        name: next(iter(hashes))
                        for name, hashes in sorted(solver_config_hashes.items())
                    },
                }
            )
        return lineage

    def _processed_provenance(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if self._normalization_stats_sha256 is None:
            raise RuntimeError("Normalization statistics were not frozen before saving")
        return {
            "schema_id": PROCESSED_MANIFEST_SCHEMA_ID,
            "preprocess_config": {
                "path": str(self.config_path),
                "sha256": self._preprocess_config_sha256,
            },
            "source_lineage": self._source_lineage(records),
            "normalization": {
                "path": "../normalization_stats.json",
                "sha256": self._normalization_stats_sha256,
                **self._normalization_provenance,
            },
        }

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

    def _unique_scenario_ids(
        self,
        records: List[Dict[str, Any]],
        *,
        label: str,
    ) -> Set[str]:
        ids = [self._scenario_id_from_record(record) for record in records]
        if "scenario_unknown" in ids:
            raise RuntimeError(f"{label} contains an unidentifiable scenario")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{label} contains duplicate scenario IDs")
        return set(ids)

    def _validate_solver_rosters(
        self,
        *,
        targets: List[str],
        split_source: List[Dict[str, Any]],
        records_by_target: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        expected = self._unique_scenario_ids(
            split_source,
            label="scenario manifest",
        )
        for target in targets:
            records = records_by_target[target]
            observed = self._unique_scenario_ids(
                records,
                label=f"{target} manifest",
            )
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            if missing or extra:
                raise RuntimeError(
                    f"{target} manifest scenario roster mismatch: "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
            for record in records:
                solver_name = self._canonical_fde_name(
                    str(record.get("solver_name", target))
                )
                if solver_name != target:
                    raise RuntimeError(
                        f"{target} manifest contains solver_name "
                        f"{record.get('solver_name')!r}"
                    )

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

    def _record_to_example(
        self,
        rec: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any], str]:
        raw = self.load_sample(
            rec["sample_dir"],
            expected_record=rec,
        )
        X, Y = self.build_example(raw)

        if "sample_index" in rec:
            sample_id = f"sample_{int(rec['sample_index']):06d}"
        else:
            sample_id = pathlib.Path(rec["sample_dir"]).name

        merged_meta: Dict[str, Any] = {}
        raw_meta = raw.get("meta", {})
        if isinstance(raw_meta, dict):
            merged_meta.update(raw_meta)
        merged_meta.update(rec if isinstance(rec, dict) else {})

        merged_meta.setdefault("source_type", "unknown")
        merged_meta.setdefault("bathymetry_type", "unknown")
        merged_meta.setdefault("source_strength", np.nan)
        merged_meta.setdefault("scenario_id", merged_meta.get("scenario_id", sample_id))
        merged_meta.setdefault("solver_name", merged_meta.get("solver_name", "unknown"))
        return X, Y, merged_meta, sample_id

    def _process_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[Dict[str, Any]], List[str]]:
        Xs: List[Dict[str, np.ndarray]] = []
        Ys: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        sample_ids: List[str] = []
        v2_contract_hashes: Set[str] = set()

        for rec in records:
            X, Y, merged_meta, sample_id = self._record_to_example(rec)
            contract_hash = merged_meta.get("contract_hash")
            if contract_hash:
                v2_contract_hashes.add(str(contract_hash))
            Xs.append(X)
            Ys.append(Y)
            sample_ids.append(sample_id)
            metas.append(merged_meta)

        if len(v2_contract_hashes) > 1:
            raise RuntimeError(
                "Refusing to preprocess mixed common-time-v2 contracts: "
                f"{sorted(v2_contract_hashes)}"
            )
        return Xs, Ys, metas, sample_ids

    def _normalize_and_save(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
        output_dir: pathlib.Path,
        norm_reference_stats_path: Optional[pathlib.Path] = None,
    ) -> None:
        output_dir = pathlib.Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = output_dir.parent / (
            f".{output_dir.name}.staging-{uuid.uuid4().hex}"
        )
        if staging_dir.exists():
            raise RuntimeError(f"Unexpected preprocessing staging path: {staging_dir}")

        try:
            split_to_merge: Optional[str] = None
            if self.cfg.publication_mode == "merge_split":
                active_splits = [
                    name
                    for name, records in (
                        ("train", train_records),
                        ("val", val_records),
                        ("test", test_records),
                    )
                    if records
                ]
                if len(active_splits) != 1:
                    raise RuntimeError(
                        "merge_split preprocessing expected exactly one populated "
                        f"split, found {active_splits}"
                    )
                split_to_merge = active_splits[0]
                if norm_reference_stats_path is None:
                    raise ValueError(
                        "merge_split preprocessing requires an explicit training "
                        "normalization reference"
                    )
                published_stats = output_dir / "normalization_stats.json"
                if not published_stats.is_file():
                    raise FileNotFoundError(
                        "Training normalization statistics are missing from "
                        f"{published_stats}. Run train preprocessing first."
                    )
                if not norm_reference_stats_path.is_file():
                    raise FileNotFoundError(
                        "Normalization reference stats not found: "
                        f"{norm_reference_stats_path}"
                    )
                if sha256_file(published_stats) != sha256_file(
                    norm_reference_stats_path
                ):
                    raise RuntimeError(
                        "merge_split normalization reference does not match the "
                        f"published training statistics in {output_dir}"
                    )

            self._write_normalized_dataset(
                train_records,
                val_records,
                test_records,
                staging_dir,
                norm_reference_stats_path=norm_reference_stats_path,
            )
            if split_to_merge is None:
                self._publish_processed_directory(staging_dir, output_dir)
            else:
                self._publish_processed_split(
                    staging_dir,
                    output_dir,
                    split_to_merge,
                )
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

    def _write_normalized_dataset(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
        output_dir: pathlib.Path,
        norm_reference_stats_path: Optional[pathlib.Path] = None,
    ) -> None:
        effective_norm_ref = self._resolve_normalization_reference_for_run(
            output_dir=output_dir,
            train_records=train_records,
            norm_reference_stats_path=norm_reference_stats_path,
        )

        if self.cfg.sharded:
            self._active_norm_reference_path = None
            if effective_norm_ref is not None:
                self._active_norm_reference_path = effective_norm_ref
                example_inputs = self._first_sample_inputs([train_records, val_records, test_records])
                self._load_normalizer_from_stats_file(effective_norm_ref, sample_inputs=example_inputs)
            elif train_records:
                self.fit_normalizer_from_records(train_records)

            original_processed_dir = self.cfg.processed_dir
            self.cfg.processed_dir = output_dir
            self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)
            self._save_normalization_stats(train_records)
            write_empty_splits = bool(train_records)
            for split_name, records in (
                ("train", train_records),
                ("val", val_records),
                ("test", test_records),
            ):
                if records or write_empty_splits:
                    self.save_split_sharded(split_name, records)
            self.cfg.processed_dir = original_processed_dir
            self._active_norm_reference_path = None
            return

        X_train_raw, Y_train_raw, meta_train, ids_train = self._process_records(train_records)
        X_val_raw, Y_val_raw, meta_val, ids_val = self._process_records(val_records)
        X_test_raw, Y_test_raw, meta_test, ids_test = self._process_records(test_records)

        self._active_norm_reference_path = None
        if effective_norm_ref is not None:
            self._active_norm_reference_path = effective_norm_ref
            example_inputs = X_train_raw[0] if X_train_raw else None
            if example_inputs is None:
                example_inputs = self._first_sample_inputs([val_records, test_records])
            self._load_normalizer_from_stats_file(effective_norm_ref, sample_inputs=example_inputs)
        elif train_records:
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
        self._save_normalization_stats(train_records)
        write_empty_splits = bool(train_records)
        for split_name, X_split, Y_split, meta_split, ids_split in (
            ("train", X_train, Y_train, meta_train, ids_train),
            ("val", X_val, Y_val, meta_val, ids_val),
            ("test", X_test, Y_test, meta_test, ids_test),
        ):
            if X_split or write_empty_splits:
                self.save_split(split_name, X_split, Y_split, meta_split, ids_split)
        self.cfg.processed_dir = original_processed_dir
        self._active_norm_reference_path = None

    @staticmethod
    def _publish_processed_directory(
        staging_dir: pathlib.Path,
        output_dir: pathlib.Path,
    ) -> None:
        backup_dir = output_dir.parent / (
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        moved_existing = False
        try:
            if output_dir.exists():
                os.replace(output_dir, backup_dir)
                moved_existing = True
            os.replace(staging_dir, output_dir)
        except Exception:
            if moved_existing and backup_dir.exists() and not output_dir.exists():
                os.replace(backup_dir, output_dir)
            raise
        else:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

    @staticmethod
    def _publish_processed_split(
        staging_dir: pathlib.Path,
        output_dir: pathlib.Path,
        split_name: str,
    ) -> None:
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported processed split: {split_name}")
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"Processed output root does not exist: {output_dir}"
            )

        staging_stats = staging_dir / "normalization_stats.json"
        output_stats = output_dir / "normalization_stats.json"
        if not staging_stats.is_file() or not output_stats.is_file():
            raise FileNotFoundError(
                "Split publication requires staged and published "
                "normalization_stats.json files"
            )
        if sha256_file(staging_stats) != sha256_file(output_stats):
            raise RuntimeError(
                "Refusing to publish a split with normalization statistics "
                "that differ from the training dataset"
            )

        staged_split = staging_dir / split_name
        output_split = output_dir / split_name
        if not staged_split.is_dir():
            raise FileNotFoundError(
                f"Staged processed split does not exist: {staged_split}"
            )
        unexpected_splits = [
            name
            for name in ("train", "val", "test")
            if name != split_name and (staging_dir / name).exists()
        ]
        if unexpected_splits:
            raise RuntimeError(
                "merge_split staging contains unexpected sibling splits: "
                f"{unexpected_splits}"
            )

        backup_dir = output_dir / (
            f".{split_name}.backup-{uuid.uuid4().hex}"
        )
        moved_existing = False
        try:
            if output_split.exists():
                os.replace(output_split, backup_dir)
                moved_existing = True
            os.replace(staged_split, output_split)
        except Exception:
            if (
                moved_existing
                and backup_dir.exists()
                and not output_split.exists()
            ):
                os.replace(backup_dir, output_split)
            raise
        else:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.rmtree(staging_dir)

    def _resolved_eval_input_order(self, sample_inputs: Dict[str, np.ndarray]) -> List[str]:
        preferred = [name for name in self.cfg.eval_input_order if name in sample_inputs]
        extras = [name for name in sample_inputs.keys() if name not in preferred]
        return preferred + extras

    def _split_metadata_arrays(
        self,
        meta_list: List[Dict[str, Any]],
        sample_ids: List[str],
    ) -> Dict[str, np.ndarray]:
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
        return {
            "source_id": source_types,
            "source_type": source_types,
            "bathymetry_type": bathymetry_types,
            "source_strength": source_strengths,
            "scenario_id": scenario_ids,
            "solver_name": solver_names,
        }

    def _save_npz_payload(self, path: pathlib.Path, payload: Dict[str, Any]) -> None:
        if self.cfg.compress:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)

    def _write_eval_manifest(self, out_dir: pathlib.Path, payload: Dict[str, Any]) -> None:
        manifest_names = [self.cfg.eval_manifest_name]
        if self.cfg.eval_manifest_name != "eval_manifest.json":
            manifest_names.append("eval_manifest.json")

        for name in manifest_names:
            with (out_dir / name).open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    @staticmethod
    def _hash_split_artifacts(
        out_dir: pathlib.Path,
        artifact_names: Set[str],
    ) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for name in sorted(artifact_names):
            path = out_dir / name
            if not path.is_file():
                raise RuntimeError(f"Expected processed artifact was not written: {path}")
            hashes[name] = sha256_file(path)
        return hashes

    def _clear_generated_split_outputs(self, out_dir: pathlib.Path) -> None:
        known_files = {
            self.cfg.eval_inputs_name,
            self.cfg.eval_targets_name,
            self.cfg.eval_ids_name,
            self.cfg.eval_archive_name,
            self.cfg.eval_manifest_name,
            "eval_manifest.json",
            "Y.npy",
            "meta.jsonl",
            "shards_manifest.json",
        }
        for name in known_files:
            path = out_dir / name
            if path.is_file():
                path.unlink()

        for path in out_dir.glob("X_*.npz"):
            if path.is_file():
                path.unlink()

        shard_dir = out_dir / "shards"
        if shard_dir.is_dir():
            for path in shard_dir.glob("shard_*.npz"):
                if path.is_file():
                    path.unlink()

    def _write_shard(
        self,
        out_dir: pathlib.Path,
        shard_dir: pathlib.Path,
        shard_idx: int,
        X: List[Dict[str, np.ndarray]],
        Y: List[np.ndarray],
        meta_list: List[Dict[str, Any]],
        sample_ids: List[str],
        input_order: List[str],
    ) -> Dict[str, Any]:
        eval_inputs = np.stack(
            [np.stack([sample[channel] for channel in input_order], axis=0) for sample in X],
            axis=0,
        ).astype(np.float32)
        eval_targets = np.stack(Y, axis=0).astype(np.float32)
        eval_ids = np.asarray(sample_ids, dtype=np.str_)
        metadata = self._split_metadata_arrays(meta_list, sample_ids)

        shard_path = shard_dir / f"shard_{shard_idx:05d}.npz"
        payload: Dict[str, Any] = {
            "inputs": eval_inputs,
            "targets": eval_targets,
            "sample_id": eval_ids,
            "target_variable": np.asarray([self.cfg.target_variable], dtype=np.str_),
            "target_mean": np.asarray([self._target_mean], dtype=np.float32),
            "target_std": np.asarray([self._target_std], dtype=np.float32),
            "target_min": np.asarray([self._target_min], dtype=np.float32),
            "target_max": np.asarray([self._target_max], dtype=np.float32),
            "input_order": np.asarray(input_order, dtype=np.str_),
        }
        payload.update(metadata)
        self._save_npz_payload(shard_path, payload)

        return {
            "file": str(shard_path.relative_to(out_dir)),
            "sha256": sha256_file(shard_path),
            "num_samples": int(eval_inputs.shape[0]),
            "inputs_shape": list(map(int, eval_inputs.shape)),
            "targets_shape": list(map(int, eval_targets.shape)),
        }

    def save_split_sharded(self, split_name: str, records: List[Dict[str, Any]]) -> None:
        """Save one split as bounded-size training/evaluation shards."""
        out_dir = self.cfg.processed_dir / split_name
        shard_dir = out_dir / "shards"
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        self._clear_generated_split_outputs(out_dir)

        if self.cfg.write_legacy_eval_archive:
            print(
                "[preprocess] saving.write_legacy_eval_archive is ignored in sharded mode; "
                "writing only bounded shard archives."
            )

        shards: List[Dict[str, Any]] = []
        input_order: List[str] = []
        total_samples = 0
        first_inputs_shape: Optional[List[int]] = None
        first_targets_shape: Optional[List[int]] = None

        X_chunk: List[Dict[str, np.ndarray]] = []
        Y_chunk: List[np.ndarray] = []
        meta_chunk: List[Dict[str, Any]] = []
        id_chunk: List[str] = []

        meta_file = None
        if self.cfg.include_meta:
            meta_file = (out_dir / "meta.jsonl").open("w", encoding="utf-8")

        def flush_chunk() -> None:
            nonlocal total_samples, first_inputs_shape, first_targets_shape
            if not X_chunk:
                return

            shard_info = self._write_shard(
                out_dir=out_dir,
                shard_dir=shard_dir,
                shard_idx=len(shards),
                X=X_chunk,
                Y=Y_chunk,
                meta_list=meta_chunk,
                sample_ids=id_chunk,
                input_order=input_order,
            )
            shards.append(shard_info)
            total_samples += int(shard_info["num_samples"])
            if first_inputs_shape is None:
                first_inputs_shape = list(shard_info["inputs_shape"])
                first_targets_shape = list(shard_info["targets_shape"])
            X_chunk.clear()
            Y_chunk.clear()
            meta_chunk.clear()
            id_chunk.clear()

        try:
            for rec in records:
                X_raw, Y_raw, meta, sample_id = self._record_to_example(rec)
                X_norm, Y_norm = self.normalize_sample(X_raw, Y_raw)

                if not input_order:
                    input_order = self._resolved_eval_input_order(X_norm)

                X_chunk.append(X_norm)
                Y_chunk.append(Y_norm)
                meta_chunk.append(meta)
                id_chunk.append(sample_id)

                if meta_file is not None:
                    meta_file.write(json.dumps(meta) + "\n")

                if len(X_chunk) >= self.cfg.shard_size:
                    flush_chunk()

            flush_chunk()
        finally:
            if meta_file is not None:
                meta_file.close()

        provenance = self._processed_provenance(records)
        shard_manifest = {
            "version": 2,
            "schema_id": PROCESSED_MANIFEST_SCHEMA_ID,
            "split": split_name,
            "sharded": True,
            "num_samples": int(total_samples),
            "num_shards": int(len(shards)),
            "shard_size": int(self.cfg.shard_size),
            "shards": shards,
            "input_order": input_order,
            "target_mode": self.cfg.target_mode,
            "target_variable": self.cfg.target_variable,
            "normalized_targets": bool(self.cfg.norm_channels.get("trajectory", False)),
            "target_mean": float(self._target_mean),
            "target_std": float(self._target_std),
            "target_min": float(self._target_min),
            "target_max": float(self._target_max),
            "provenance": provenance,
        }
        with (out_dir / "shards_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(shard_manifest, f, indent=2)

        eval_manifest = {
            "schema_id": PROCESSED_MANIFEST_SCHEMA_ID,
            "split": split_name,
            "sharded": True,
            "shards_manifest": "shards_manifest.json",
            "input_order": input_order,
            "target_mode": self.cfg.target_mode,
            "target_variable": self.cfg.target_variable,
            "normalized_targets": bool(self.cfg.norm_channels.get("trajectory", False)),
            "num_samples": int(total_samples),
            "num_shards": int(len(shards)),
            "inputs_shape": first_inputs_shape,
            "targets_shape": first_targets_shape,
            "provenance": provenance,
        }
        self._write_eval_manifest(out_dir, eval_manifest)

    def save_split(self, split_name: str, X: List[Dict[str, np.ndarray]], Y: List[np.ndarray],
                   meta_list: List[Dict[str, Any]], sample_ids: List[str]) -> None:
        """ save processed arrays and metadata """
        out_dir = self.cfg.processed_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        self._clear_generated_split_outputs(out_dir)

        if len(X) == 0 or len(Y) == 0:
            if self.cfg.include_meta:
                with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
                    f.write("")
            return

        # save each input channel
        channel_names = list(X[0].keys())
        artifact_names: Set[str] = {"Y.npy"}

        for channel in channel_names:
            stacked = np.stack([x[channel] for x in X], axis=0)
            npz_path = out_dir / f"X_{channel}.npz"
            artifact_names.add(npz_path.name)

            if self.cfg.compress:
                np.savez_compressed(npz_path, data=stacked)
            
            else:
                np.savez(npz_path, data=stacked)

        # save targets
        Y_arr = np.stack(Y, axis=0)
        Y_path = out_dir / "Y.npy"
        np.save(Y_path, Y_arr)

        if self.cfg.include_meta:
            meta_path = out_dir / "meta.jsonl"
            with meta_path.open("w", encoding="utf-8") as f:
                for meta in meta_list:
                    f.write(json.dumps(meta) + "\n")
            artifact_names.add(meta_path.name)

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
            artifact_names.update(
                {
                    self.cfg.eval_inputs_name,
                    self.cfg.eval_targets_name,
                    self.cfg.eval_ids_name,
                    self.cfg.eval_archive_name,
                }
            )

            eval_manifest = {
                "schema_id": PROCESSED_MANIFEST_SCHEMA_ID,
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
                "artifacts": self._hash_split_artifacts(
                    out_dir,
                    artifact_names,
                ),
                "provenance": self._processed_provenance(meta_list),
            }
            self._write_eval_manifest(out_dir, eval_manifest)

    def _save_normalization_stats(
        self, fit_records: List[Dict[str, Any]]
    ) -> None:
        stats_path = self.cfg.processed_dir / "normalization_stats.json"
        reference: Dict[str, Any] | None = None
        if self._active_norm_reference_path is not None:
            reference_path = self._active_norm_reference_path.resolve()
            reference = {
                "mode": "explicit_reference",
                "path": str(self._active_norm_reference_path),
                "sha256": sha256_file(reference_path),
                "source": self._loaded_normalization_payload.get("source"),
            }
            if stats_path.resolve() == reference_path:
                self._normalization_stats_sha256 = reference["sha256"]
                self._normalization_provenance = {
                    "mode": "explicit_reference_in_place",
                    "reference": reference,
                }
                return
            if self.cfg.publication_mode == "merge_split":
                shutil.copyfile(reference_path, stats_path)
                self._normalization_stats_sha256 = reference["sha256"]
                self._normalization_provenance = {
                    "mode": "explicit_reference",
                    "reference": reference,
                }
                return

        source = (
            reference
            if reference is not None
            else {
                "mode": "fit_training_records",
                "lineage": self._source_lineage(fit_records),
            }
        )
        stats = {
            "schema_id": NORMALIZATION_STATS_SCHEMA_ID,
            "method": self.cfg.norm_method,
            "eps": float(self.cfg.eps),
            "preprocess_config": {
                "path": str(self.config_path),
                "sha256": self._preprocess_config_sha256,
            },
            "source": source,
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
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        self._normalization_stats_sha256 = sha256_file(stats_path)
        self._normalization_provenance = {
            "mode": str(source["mode"]),
            "source": source,
        }

    def run(self) -> None:
        mode = self.fde_mode
        if mode in {"single", "separate_all", "multifidelity"} and self.fde_manifest_paths:
            requested = self.fde_targets if self.fde_targets else list(self.fde_manifest_paths.keys())
            missing_targets = [
                name for name in requested if name not in self.fde_manifest_paths
            ]
            if missing_targets:
                raise ValueError(
                    "Missing requested FDE manifests: "
                    + ", ".join(sorted(missing_targets))
                )
            targets = list(requested)
            if not targets:
                raise ValueError(
                    f"No valid fde.targets found for mode={mode}. Available: {sorted(self.fde_manifest_paths.keys())}"
                )

            # Build one shared scenario split for fair cross-FDE comparison.
            if self.scenario_manifest_path is not None and self.scenario_manifest_path.exists():
                split_source = self._load_manifest_path(self.scenario_manifest_path)
            else:
                split_source = self._load_manifest_path(self.fde_manifest_paths[targets[0]])
            records_by_target = {
                target: self._load_manifest_path(self.fde_manifest_paths[target])
                for target in targets
            }
            self._validate_solver_rosters(
                targets=targets,
                split_source=split_source,
                records_by_target=records_by_target,
            )
            train_ids, val_ids, test_ids = self._split_scenario_ids(split_source)

            if mode == "single":
                fde_name = targets[0]
                records = records_by_target[fde_name]
                train_records = self._records_for_scenarios(records, train_ids)
                val_records = self._records_for_scenarios(records, val_ids)
                test_records = self._records_for_scenarios(records, test_ids)
                out_dir = self.cfg.processed_dir / fde_name
                norm_ref = self.fde_norm_reference_paths.get(fde_name, self.cfg.norm_reference_stats_path)
                print(f"[preprocess] mode=single fde={fde_name} out={out_dir}")
                self._normalize_and_save(
                    train_records,
                    val_records,
                    test_records,
                    out_dir,
                    norm_reference_stats_path=norm_ref,
                )
                return

            if mode == "separate_all":
                for fde_name in targets:
                    records = records_by_target[fde_name]
                    train_records = self._records_for_scenarios(records, train_ids)
                    val_records = self._records_for_scenarios(records, val_ids)
                    test_records = self._records_for_scenarios(records, test_ids)
                    out_dir = self.cfg.processed_dir / fde_name
                    norm_ref = self.fde_norm_reference_paths.get(fde_name, self.cfg.norm_reference_stats_path)
                    print(f"[preprocess] mode=separate_all fde={fde_name} out={out_dir}")
                    self._normalize_and_save(
                        train_records,
                        val_records,
                        test_records,
                        out_dir,
                        norm_reference_stats_path=norm_ref,
                    )
                return

            # multifidelity
            train_records: List[Dict[str, Any]] = []
            val_records: List[Dict[str, Any]] = []
            test_records: List[Dict[str, Any]] = []
            for fde_name in targets:
                records = records_by_target[fde_name]
                train_records.extend(self._records_for_scenarios(records, train_ids))
                val_records.extend(self._records_for_scenarios(records, val_ids))
                test_records.extend(self._records_for_scenarios(records, test_ids))

            out_dir = self.cfg.processed_dir / "multifidelity"
            print(f"[preprocess] mode=multifidelity targets={targets} out={out_dir}")
            self._normalize_and_save(
                train_records,
                val_records,
                test_records,
                out_dir,
                norm_reference_stats_path=self.cfg.norm_reference_stats_path,
            )
            return

        # Legacy mode (single manifest path + one processed root).
        print("Loading manifest")
        manifest = self.load_manifest()
        print("Splitting dataset")
        train_records, val_records, test_records = self.split_dataset(manifest)
        print("Building dataset")
        self._normalize_and_save(
            train_records,
            val_records,
            test_records,
            self.cfg.processed_dir,
            norm_reference_stats_path=self.cfg.norm_reference_stats_path,
        )

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
