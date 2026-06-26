#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import save_json


def _as_array(data: Dict[str, np.ndarray], key: str, n: int, default: Any) -> np.ndarray:
    if key in data:
        arr = np.asarray(data[key])
        if arr.shape and arr.shape[0] == n:
            return arr
    return np.asarray([default] * n)


def _as_str_array(data: Dict[str, np.ndarray], key: str, n: int, default: str = "unknown") -> np.ndarray:
    arr = _as_array(data, key, n, default)
    return np.asarray(arr).astype(str)


def _as_float_array(data: Dict[str, np.ndarray], key: str, n: int, default: float = np.nan) -> np.ndarray:
    arr = _as_array(data, key, n, default)
    out = np.full((n,), float(default), dtype=np.float64)
    for i, v in enumerate(arr):
        try:
            out[i] = float(v)
        except Exception:
            out[i] = float(default)
    return out


def _apply_filters(meta: Dict[str, np.ndarray], filters: Dict[str, Any]) -> np.ndarray:
    n = int(meta["source_type"].shape[0])
    mask = np.ones((n,), dtype=bool)

    def in_list(values: np.ndarray, allowed: Iterable[str]) -> np.ndarray:
        allowed_set = {str(x) for x in allowed}
        return np.asarray([str(v) in allowed_set for v in values], dtype=bool)

    if "source_type_in" in filters:
        mask &= in_list(meta["source_type"], filters["source_type_in"])
    if "source_type_not_in" in filters:
        mask &= ~in_list(meta["source_type"], filters["source_type_not_in"])

    if "bathymetry_type_in" in filters:
        mask &= in_list(meta["bathymetry_type"], filters["bathymetry_type_in"])
    if "bathymetry_type_not_in" in filters:
        mask &= ~in_list(meta["bathymetry_type"], filters["bathymetry_type_not_in"])

    if "solver_name_in" in filters:
        mask &= in_list(meta["solver_name"], filters["solver_name_in"])
    if "solver_name_not_in" in filters:
        mask &= ~in_list(meta["solver_name"], filters["solver_name_not_in"])

    strength = meta["source_strength"]
    if "source_strength_min" in filters:
        mask &= strength >= float(filters["source_strength_min"])
    if "source_strength_max" in filters:
        mask &= strength <= float(filters["source_strength_max"])

    if "scenario_id_in" in filters:
        mask &= in_list(meta["scenario_id"], filters["scenario_id_in"])
    if "scenario_id_not_in" in filters:
        mask &= ~in_list(meta["scenario_id"], filters["scenario_id_not_in"])

    return mask


def _subset_npz(payload: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = int(mask.shape[0])
    out: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.shape and arr.shape[0] == n:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def _load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _load_sharded_payload(path: Path) -> Dict[str, np.ndarray]:
    manifest_path = path / "shards_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks: Dict[str, list[np.ndarray]] = {}
    constants: Dict[str, np.ndarray] = {}

    for shard in manifest.get("shards", []):
        shard_file = shard.get("file")
        if not shard_file:
            raise KeyError(f"Shard entry in {manifest_path} is missing a file path.")
        shard_path = path / str(shard_file)
        with np.load(shard_path, allow_pickle=True) as data:
            if "inputs" in data:
                n = int(data["inputs"].shape[0])
            elif "x" in data:
                n = int(data["x"].shape[0])
            else:
                raise KeyError(f"Shard {shard_path} must contain inputs or x")

            for key in data.files:
                arr = np.asarray(data[key])
                if arr.shape and arr.shape[0] == n:
                    chunks.setdefault(key, []).append(arr)
                elif key not in constants:
                    constants[key] = arr

    payload = dict(constants)
    for key, arrays in chunks.items():
        payload[key] = np.concatenate(arrays, axis=0)

    return payload


def _load_payload(path: Path) -> Dict[str, np.ndarray]:
    if path.name == "eval_dataset.npz" and (path.parent / "shards_manifest.json").is_file():
        return _load_sharded_payload(path.parent)
    if path.is_dir() and (path / "shards_manifest.json").is_file():
        return _load_sharded_payload(path)
    if path.is_dir() and (path / "eval_dataset.npz").is_file():
        return _load_npz_payload(path / "eval_dataset.npz")
    return _load_npz_payload(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Build OOD evaluation suites from a flat or sharded processed dataset.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_dataset = Path(str(cfg.get("input_dataset", "")).strip())
    output_root = Path(str(cfg.get("output_root", "")).strip())
    suites = cfg.get("suites", [])
    min_samples_raw = cfg.get("min_samples", None)
    min_samples_action = str(cfg.get("min_samples_action", "fail")).strip().lower()

    if not input_dataset:
        raise KeyError("config is missing required key: input_dataset")
    if not output_root:
        raise KeyError("config is missing required key: output_root")
    if not isinstance(suites, list) or not suites:
        raise ValueError("config.suites must be a non-empty list")
    if min_samples_action not in {"warn", "fail"}:
        raise ValueError("min_samples_action must be one of: warn, fail")
    
    min_samples_default: int | None = None
    
    if min_samples_raw is not None:
        min_samples_default = int(min_samples_raw)
        if min_samples_default < 0:
            raise ValueError("min_samples must be >= 0")
    
    if not input_dataset.exists() and not (
        input_dataset.name == "eval_dataset.npz" and (input_dataset.parent / "shards_manifest.json").is_file()
    ):
        raise FileNotFoundError(input_dataset)

    payload = _load_payload(input_dataset)

    if "inputs" not in payload and "x" in payload:
        payload["inputs"] = payload["x"]
    if "targets" not in payload and "y" in payload:
        payload["targets"] = payload["y"]
    if "inputs" not in payload or "targets" not in payload:
        raise KeyError("input_dataset must contain keys: inputs/targets or x/y")
    n = int(payload["inputs"].shape[0])
    if int(payload["targets"].shape[0]) != n:
        raise ValueError("inputs and targets must have matching sample dimension")

    meta = {
        "source_type": _as_str_array(payload, "source_type", n),
        "bathymetry_type": _as_str_array(payload, "bathymetry_type", n),
        "solver_name": _as_str_array(payload, "solver_name", n),
        "scenario_id": _as_str_array(payload, "scenario_id", n),
        "sample_id": _as_str_array(payload, "sample_id", n),
        "source_strength": _as_float_array(payload, "source_strength", n),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[ood] input={input_dataset} n={n}")
    source_vals, source_counts = np.unique(meta["source_type"], return_counts=True)
    bathy_vals, bathy_counts = np.unique(meta["bathymetry_type"], return_counts=True)
    print(
        "[ood] available source_type counts: "
        + ", ".join(f"{v}={int(c)}" for v, c in zip(source_vals.tolist(), source_counts.tolist()))
    )
    print(
        "[ood] available bathymetry_type counts: "
        + ", ".join(f"{v}={int(c)}" for v, c in zip(bathy_vals.tolist(), bathy_counts.tolist()))
    )
    print(
        f"[ood] source_strength range: min={float(np.nanmin(meta['source_strength'])):.4f} "
        f"max={float(np.nanmax(meta['source_strength'])):.4f}"
    )

    for i, suite in enumerate(suites):
        suite_cfg = suite if isinstance(suite, dict) else {}
        label = str(suite_cfg.get("label", f"suite_{i}")).strip()
        if not label:
            raise ValueError(f"suites[{i}] has empty label")
        filters = dict(suite_cfg.get("filters", {}))
        max_samples = suite_cfg.get("max_samples")

        mask = _apply_filters(meta, filters)
        idx = np.flatnonzero(mask)
        if max_samples is not None:
            max_samples_int = int(max_samples)
            if max_samples_int < 1:
                raise ValueError(f"suites[{i}].max_samples must be >= 1")
            idx = idx[:max_samples_int]
            mask = np.zeros_like(mask)
            mask[idx] = True

        selected_count = int(mask.sum())
        suite_min_samples_raw = suite_cfg.get("min_samples", min_samples_default)
        
        if suite_min_samples_raw is not None:
            suite_min_samples = int(suite_min_samples_raw)
            if suite_min_samples < 0:
                raise ValueError(f"suites[{i}].min_samples must be >= 0")
        
            if selected_count < suite_min_samples:
                msg = (
                    f"[ood] suite '{label}' selected too few samples: {selected_count} < {suite_min_samples}. "
                    "Relax filters or increase source dataset size."
                )
                if min_samples_action == "fail":
                    raise ValueError(msg)
                print(f"[ood][warn] {msg}")

        subset = _subset_npz(payload, mask)
        out_dir = output_root / label
        out_npz = out_dir / "eval_dataset.npz"
        out_manifest = out_dir / "eval_manifest.json"

        if out_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists for suite '{label}': {out_dir}. "
                "Use --overwrite to replace."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **subset)

        summary = {
            "label": label,
            "input_dataset": str(input_dataset),
            "num_selected": selected_count,
            "num_total": n,
            "filters": filters,
            "max_samples": None if max_samples is None else int(max_samples),
            "min_samples_required": (
                None
                if suite_cfg.get("min_samples", min_samples_default) is None
                else int(suite_cfg.get("min_samples", min_samples_default))
            ),
            "selected_source_types": sorted({str(x) for x in meta["source_type"][mask]}),
            "selected_bathymetry_types": sorted({str(x) for x in meta["bathymetry_type"][mask]}),
            "selected_solver_names": sorted({str(x) for x in meta["solver_name"][mask]}),
        }
        save_json(summary, out_manifest)
        print(f"[ood] {label:<32} selected={selected_count:5d} -> {out_npz}")

    print(f"[ood] done. suites_root={output_root}")


if __name__ == "__main__":
    main()
