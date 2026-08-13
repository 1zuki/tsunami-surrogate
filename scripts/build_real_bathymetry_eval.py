#!/usr/bin/env python
"""Build hydrostatic evaluation datasets on raw real-bathymetry patches.

The raw files under data/real_bathymetry contain bathymetry only. This script
keeps that bathymetry fixed, generates synthetic sources, runs the hydrostatic
reference solver, and preprocesses the result with the main hydrostatic training
normalization stats. The output is therefore a synthetic-source / real-bathymetry
transfer set, not real-event validation data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen.preprocess import TsunamiPreprocessor
from src.data_gen.simulate_dataset import _seed_for_sample
from src.utils.hashing import sha256_file


REAL_BATHYMETRY_LINEAGE_SCHEMA_ID = (
    "tsunami-surrogate.real-bathymetry-input.v1"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    os.replace(staging, path)


def _suite_dirs(raw_root: Path, selected: list[str] | None) -> list[Path]:
    suites = [p for p in sorted(raw_root.iterdir()) if p.is_dir()]
    if selected:
        keep = set(selected)
        suites = [p for p in suites if p.name in keep]
        missing = keep.difference({p.name for p in suites})
        if missing:
            raise FileNotFoundError(
                f"Missing requested real-bathymetry suites: {sorted(missing)}"
            )
    if not suites:
        raise FileNotFoundError(f"No suite directories found under {raw_root}")
    return suites


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _jsonable_npz_metadata(payload: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in payload.files:
        if key == "bathymetry":
            continue
        values = np.asarray(payload[key])
        if values.dtype == object:
            continue
        metadata[key] = values.tolist()
    return metadata


def _write_prepared_input(
    *,
    target: Path,
    bathymetry: np.ndarray,
    bathymetry_type: str,
    sample_seed: int,
    lineage: dict[str, Any],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    bathymetry = np.asarray(bathymetry, dtype=np.float32)
    lineage_payload = {
        **lineage,
        "schema_id": REAL_BATHYMETRY_LINEAGE_SCHEMA_ID,
        "bathymetry_array_sha256": _array_sha256(bathymetry),
        "bathymetry_shape": [int(v) for v in bathymetry.shape],
        "bathymetry_dtype": str(bathymetry.dtype),
    }
    lineage_json = json.dumps(
        lineage_payload, sort_keys=True, separators=(",", ":")
    )

    if target.is_file():
        with np.load(target, allow_pickle=False) as existing:
            same = (
                np.array_equal(existing["bathymetry"], bathymetry)
                and str(
                    np.asarray(existing["bathymetry_type"]).reshape(-1)[0]
                )
                == bathymetry_type
                and int(np.asarray(existing["sample_seed"]).reshape(-1)[0])
                == int(sample_seed)
                and str(
                    np.asarray(existing["input_lineage_json"]).reshape(-1)[0]
                )
                == lineage_json
            )
        if not same:
            raise FileExistsError(
                f"Prepared real-bathymetry input is incompatible: {target}"
            )
        return

    staging = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with staging.open("wb") as handle:
        np.savez_compressed(
            handle,
            bathymetry=bathymetry,
            bathymetry_type=np.asarray([bathymetry_type], dtype="U96"),
            sample_seed=np.asarray([sample_seed], dtype=np.int64),
            input_lineage_json=np.asarray([lineage_json], dtype="U4096"),
        )
    os.replace(staging, target)


def _prepare_direct_suite(
    *,
    source_dir: Path,
    target_dir: Path,
    suite_name: str,
    seed: int,
) -> None:
    for index, path in enumerate(_validate_bathymetry_files(source_dir), start=1):
        with np.load(path, allow_pickle=False) as payload:
            bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
            bathymetry_type = str(
                np.asarray(payload["bathymetry_type"]).reshape(-1)[0]
            )
            metadata = _jsonable_npz_metadata(payload)
        _write_prepared_input(
            target=target_dir / path.name,
            bathymetry=bathymetry,
            bathymetry_type=bathymetry_type,
            sample_seed=_seed_for_sample(seed, index),
            lineage={
                "suite": suite_name,
                "sample_index": index,
                "source_kind": "preserved_static_input",
                "source_artifact": str(path),
                "source_artifact_sha256": sha256_file(path),
                "source_metadata": metadata,
                "h0_authoritative": False,
            },
        )


def _prepare_recovered_suite(
    *,
    legacy_raw_root: Path,
    target_dir: Path,
    suite_name: str,
    seed: int,
) -> None:
    sample_dir = (
        legacy_raw_root
        / suite_name
        / "hydrostatic"
        / "samples"
        / "sample_000001"
    )
    sample_path = sample_dir / "sample.npz"
    meta_path = sample_dir / "meta.json"
    if not sample_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Cannot recover static bathymetry for {suite_name}: {sample_dir}"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(sample_path, allow_pickle=False) as payload:
        bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
    _write_prepared_input(
        target=target_dir / "sample_000001.npz",
        bathymetry=bathymetry,
        bathymetry_type=str(metadata.get("bathymetry_type", suite_name)),
        sample_seed=_seed_for_sample(seed, 1),
        lineage={
            "suite": suite_name,
            "sample_index": 1,
            "source_kind": "recovered_from_legacy_rollout_static_array",
            "source_artifact": str(sample_path),
            "source_artifact_sha256": sha256_file(sample_path),
            "source_meta_artifact": str(meta_path),
            "source_meta_sha256": sha256_file(meta_path),
            "legacy_bathymetry_cache_path": metadata.get(
                "bathymetry_cache_path"
            ),
            "h0_authoritative": False,
        },
    )


def _prepare_input_suites(
    *,
    raw_root: Path,
    legacy_raw_root: Path,
    input_root: Path,
    seed: int,
    selected: list[str] | None,
) -> list[Path]:
    requested = (
        set(selected)
        if selected
        else {
            "main_morphology_suite_10",
            "main_morphology",
            "appendix_coastline_fully_wet",
        }
    )
    unknown = requested - {
        "main_morphology_suite_10",
        "main_morphology",
        "appendix_coastline_stress",
        "appendix_coastline_fully_wet",
    }
    if unknown:
        raise ValueError(f"Unknown real-bathymetry suites: {sorted(unknown)}")

    if "main_morphology_suite_10" in requested:
        _prepare_direct_suite(
            source_dir=raw_root / "main_morphology_suite_10",
            target_dir=input_root / "main_morphology_suite_10",
            suite_name="main_morphology_suite_10",
            seed=seed,
        )
    for suite_name in sorted(
        requested
        & {
            "main_morphology",
            "appendix_coastline_stress",
            "appendix_coastline_fully_wet",
        }
    ):
        _prepare_recovered_suite(
            legacy_raw_root=legacy_raw_root,
            target_dir=input_root / suite_name,
            suite_name=suite_name,
            seed=seed,
        )
    return [input_root / name for name in sorted(requested)]


def _rescale_to_range(arr: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    in_min = float(np.nanmin(arr))
    in_max = float(np.nanmax(arr))
    if not np.isfinite(in_min) or not np.isfinite(in_max) or in_max <= in_min:
        raise ValueError(
            f"Cannot rescale invalid bathymetry range [{in_min}, {in_max}]"
        )
    scaled = (arr - in_min) / (in_max - in_min)
    return (out_min + scaled * (out_max - out_min)).astype(np.float32)


def _derive_fully_wet_suite(
    source_dir: Path, target_dir: Path, bathymetry_type: str
) -> None:
    files = _validate_bathymetry_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            bathy = np.asarray(data["bathymetry"], dtype=np.float32)
            sample_seed = (
                np.asarray(data["sample_seed"])
                if "sample_seed" in data
                else np.asarray([0], dtype=np.int64)
            )
        fully_wet = _rescale_to_range(bathy, -10.0, -0.75)
        np.savez_compressed(
            target_dir / path.name,
            bathymetry=fully_wet,
            bathymetry_type=np.asarray([bathymetry_type], dtype="U64"),
            sample_seed=sample_seed,
            derived_from=np.asarray([str(path)], dtype="U256"),
            derivation=np.asarray(
                ["linear_rescale_to_fully_wet_-10_-0.75"], dtype="U64"
            ),
        )


def _validate_bathymetry_files(suite_dir: Path) -> list[Path]:
    files = sorted(suite_dir.glob("sample_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No sample_*.npz bathymetry files found in {suite_dir}"
        )

    for expected, path in enumerate(files, start=1):
        expected_name = f"sample_{expected:06d}.npz"
        if path.name != expected_name:
            raise ValueError(
                f"{suite_dir} must be consecutively numbered from sample_000001.npz; "
                f"expected {expected_name}, found {path.name}."
            )
        with np.load(path, allow_pickle=True) as data:
            if "bathymetry" not in data:
                raise KeyError(f"{path} has no 'bathymetry' array")
            bathy = np.asarray(data["bathymetry"])
            if bathy.ndim != 2:
                raise ValueError(
                    f"{path} bathymetry must be 2D, got shape {bathy.shape}"
                )
    return files


def _dataset_config(
    base: dict[str, Any],
    suite_name: str,
    suite_dir: Path,
    num_samples: int,
    out_root: Path,
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["configs"] = dict(cfg.get("configs", {}))
    cfg["configs"]["source"] = (
        "configs/data/real_bathymetry/real_bathymetry_source.yaml"
    )
    cfg.pop("authoritative_inputs", None)
    cfg.pop("paired_inputs", None)
    cfg.setdefault("dataset", {})
    cfg["dataset"] = dict(cfg["dataset"])
    cfg["dataset"].update(
        {
            "num_samples": int(num_samples),
            "seed": int(seed),
            "num_workers": int(num_workers),
            "bathymetry_dir": str(suite_dir),
            "source_dir": str(out_root / "sources" / suite_name),
            "output_dir": str(out_root / "raw" / suite_name),
            "manifest_path": str(
                out_root / "manifests" / suite_name / "scenario_manifest.jsonl"
            ),
            "copy_configs": True,
        }
    )
    cfg["fdes"] = {"enabled": ["swe_hydrostatic"], "primary": "swe_hydrostatic"}
    cfg["quality"] = dict(cfg.get("quality", {}))
    cfg["quality"]["on_violation"] = "fail"
    if suite_name == "appendix_coastline_stress":
        cfg["quality"]["max_abs_eta_limit"] = 5.01
        cfg["quality"]["require_no_projection"] = False
    cfg.setdefault("operations", {})
    cfg["operations"] = dict(cfg["operations"])
    cfg["operations"]["max_in_flight"] = max(1, int(num_workers))
    return cfg


def _preprocess_config(
    suite_name: str, out_root: Path, processed_root: Path, train_stats: Path
) -> dict[str, Any]:
    manifest_root = out_root / "manifests" / suite_name
    raw_root = out_root / "raw" / suite_name
    return {
        "raw": {
            "publication_split": "test",
            "scenario_manifest": str(manifest_root / "scenario_manifest.jsonl"),
            "fde_manifests": {
                "hydrostatic": str(manifest_root / "hydrostatic_manifest.jsonl")
            },
            "raw_dirs": {"hydrostatic": str(raw_root / "hydrostatic" / "samples")},
        },
        "processed_dir": str(processed_root / suite_name),
        "fde": {"mode": "single", "targets": ["hydrostatic"], "target_field": "eta"},
        "split": {"train": 0, "val": 0, "test": 1, "seed": 42},
        "input": {
            "use_bathymetry": True,
            "use_source": True,
            "use_initial_depth": True,
            "use_initial_surface": False,
            "use_solver_id": False,
        },
        "target": {
            "mode": "multi_step",
            "variable": "eta",
            "forecast_steps": 50,
            "stride": 1,
        },
        "normalization": {
            "method": "standardize",
            "reference_stats_by_fde": {"hydrostatic": str(train_stats)},
            "channels": {
                "bathymetry": True,
                "source": True,
                "solver_id": False,
                "trajectory": True,
            },
            "eps": 1e-6,
        },
        "saving": {
            "format": "npy",
            "compress": True,
            "include_meta": True,
            "sharded": False,
            "shard_size": 128,
            "write_legacy_test_archive": False,
        },
        "test_export": {
            "enabled": True,
            "input_order": ["bathymetry", "source", "initial_depth", "initial_surface"],
            "inputs_name": "inputs.npy",
            "targets_name": "targets.npy",
            "ids_name": "sample_id.npy",
            "archive_name": "eval_dataset.npz",
            "manifest_name": "eval_manifest.json",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        default="data/real_bathymetry_raw",
        help="Directory containing the preserved ten-crop morphology suite.",
    )
    p.add_argument(
        "--legacy-raw-root",
        default="data/real_bathymetry/raw",
        help="Old rollout root used only to recover preserved static inputs.",
    )
    p.add_argument(
        "--input-root", default="data/real_bathymetry_inputs_v2"
    )
    p.add_argument(
        "--base-config",
        default="configs/data/dataset_test.yaml",
    )
    p.add_argument("--out-root", default="data/real_bathymetry_v2")
    p.add_argument(
        "--processed-root", default="data/processed_real_bathymetry_v2"
    )
    p.add_argument("--config-out", default="configs/data/real_bathymetry_v2")
    p.add_argument(
        "--train-stats", default="data/processed/hydrostatic/normalization_stats.json"
    )
    p.add_argument(
        "--suite", action="append", help="Restrict to one suite name; repeatable."
    )
    p.add_argument("--seed", type=int, default=367)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--allow-override", action="store_true")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    legacy_raw_root = Path(args.legacy_raw_root)
    input_root = Path(args.input_root)
    base_config_path = Path(args.base_config)
    out_root = Path(args.out_root)
    processed_root = Path(args.processed_root)
    config_out = Path(args.config_out)
    train_stats = Path(args.train_stats)

    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if not train_stats.exists():
        raise FileNotFoundError(train_stats)

    base = _load_yaml(base_config_path)
    processed_paths: list[str] = []
    suite_dirs = _prepare_input_suites(
        raw_root=raw_root,
        legacy_raw_root=legacy_raw_root,
        input_root=input_root,
        seed=int(args.seed),
        selected=args.suite,
    )

    for suite_dir in suite_dirs:
        bathy_files = _validate_bathymetry_files(suite_dir)
        suite_name = suite_dir.name

        ds_cfg = _dataset_config(
            base=base,
            suite_name=suite_name,
            suite_dir=suite_dir,
            num_samples=len(bathy_files),
            out_root=out_root,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        ds_cfg_path = config_out / f"{suite_name}_dataset.yaml"
        _write_yaml(ds_cfg_path, ds_cfg)

        pp_cfg = _preprocess_config(
            suite_name=suite_name,
            out_root=out_root,
            processed_root=processed_root,
            train_stats=train_stats,
        )
        pp_cfg_path = config_out / f"{suite_name}_preprocess.yaml"
        _write_yaml(pp_cfg_path, pp_cfg)

        if not args.skip_generate:
            print(
                f"[real-bath] generate hydrostatic raw suite={suite_name} n={len(bathy_files)}"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts/make_dataset.py"),
                "--config",
                str(ds_cfg_path),
                "--continue",
            ]
            if args.allow_override:
                command.append("--allow-override")
            subprocess.run(command, check=True, cwd=ROOT)

        if not args.skip_preprocess:
            print(f"[real-bath] preprocess suite={suite_name}")
            TsunamiPreprocessor(str(pp_cfg_path)).run()

        processed_paths.append(
            str(processed_root / suite_name / "hydrostatic" / "test")
        )

    print("[real-bath] processed test paths:")
    for path in processed_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
