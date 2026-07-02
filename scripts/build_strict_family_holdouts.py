#!/usr/bin/env python
"""Build true held-out-family hydrostatic datasets from processed shards.

The existing OOD suites filter the normalized test set only. This script creates
strict train/val/test splits where a source or bathymetry family is absent from
training and validation, then refits normalization statistics from the filtered
training split only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


META_KEYS = (
    "sample_id",
    "source_id",
    "source_type",
    "bathymetry_type",
    "source_strength",
    "scenario_id",
    "solver_name",
)


class RunningStats:
    def __init__(self) -> None:
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, arr: np.ndarray) -> None:
        x = np.asarray(arr, dtype=np.float64)
        if x.size == 0:
            return
        self.sum += float(x.sum())
        self.sum_sq += float((x * x).sum())
        self.count += int(x.size)
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))

    def standardize(self, eps: float) -> dict[str, float]:
        if self.count <= 0:
            raise ValueError("Cannot compute stats from zero values.")
        mean = self.sum / float(self.count)
        var = self.sum_sq / float(self.count) - mean * mean
        scale = math.sqrt(max(var, float(eps)))
        return {
            "offset": float(mean),
            "scale": float(scale),
            "min": float(self.min),
            "max": float(self.max),
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_json(path)


def _input_offset_scale(stats: Mapping[str, Any], name: str) -> tuple[float, float]:
    spec = stats.get("inputs", {}).get(name, {})
    if not isinstance(spec, Mapping):
        return 0.0, 1.0
    offset = float(spec.get("offset", 0.0))
    scale = float(spec.get("scale", 1.0))
    if not math.isfinite(offset) or not math.isfinite(scale) or scale <= 0.0:
        return 0.0, 1.0
    return offset, scale


def _target_offset_scale(stats: Mapping[str, Any]) -> tuple[float, float]:
    spec = stats.get("targets", {})
    if not isinstance(spec, Mapping) or not bool(spec.get("enabled", True)):
        return 0.0, 1.0
    offset = float(spec.get("offset", 0.0))
    scale = float(spec.get("scale", 1.0))
    if not math.isfinite(offset) or not math.isfinite(scale) or scale <= 0.0:
        return 0.0, 1.0
    return offset, scale


def _split_manifest(split_dir: Path) -> dict[str, Any]:
    manifest_path = split_dir / "shards_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return _load_json(manifest_path)


def _iter_shards(
    source_root: Path, split: str
) -> Iterable[tuple[Path, dict[str, Any]]]:
    split_dir = source_root / split
    manifest = _split_manifest(split_dir)
    for shard in manifest.get("shards", []):
        shard_file = shard.get("file")
        if not shard_file:
            raise KeyError(
                f"Shard entry in {split_dir / 'shards_manifest.json'} is missing file."
            )
        yield split_dir / str(shard_file), manifest


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _row_count(payload: Mapping[str, np.ndarray]) -> int:
    if "inputs" in payload:
        return int(payload["inputs"].shape[0])
    if "x" in payload:
        return int(payload["x"].shape[0])
    raise KeyError("Shard must contain inputs or x.")


def _str_array(
    payload: Mapping[str, np.ndarray], key: str, n: int, default: str
) -> np.ndarray:
    if key not in payload:
        return np.asarray([default] * n, dtype=np.str_)
    arr = np.asarray(payload[key]).reshape(-1)
    if arr.shape[0] != n:
        return np.asarray([default] * n, dtype=np.str_)
    return arr.astype(str)


def _float_array(
    payload: Mapping[str, np.ndarray], key: str, n: int, default: float = np.nan
) -> np.ndarray:
    if key not in payload:
        return np.full((n,), float(default), dtype=np.float32)
    arr = np.asarray(payload[key]).reshape(-1)
    if arr.shape[0] != n:
        return np.full((n,), float(default), dtype=np.float32)
    return arr.astype(np.float32)


def _input_order(
    payload: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> list[str]:
    if "input_order" in payload:
        return [str(v) for v in np.asarray(payload["input_order"]).reshape(-1).tolist()]
    order = manifest.get("input_order")
    if isinstance(order, list) and all(isinstance(v, str) for v in order):
        return list(order)
    return ["bathymetry", "source", "initial_depth"]


def _heldout_mask(
    payload: Mapping[str, np.ndarray], key: str, value: str
) -> np.ndarray:
    n = _row_count(payload)
    arr = _str_array(payload, key, n, default="")
    return arr == str(value)


def _selected_mask(
    payload: Mapping[str, np.ndarray], spec: Mapping[str, Any], split_kind: str
) -> np.ndarray:
    heldout = _heldout_mask(payload, str(spec["key"]), str(spec["value"]))
    if split_kind in {"train", "val", "test_id"}:
        return ~heldout
    if split_kind in {"test_heldout", "test"}:
        return heldout
    raise ValueError(f"Unknown split kind: {split_kind}")


def _denorm_inputs(
    inputs: np.ndarray,
    input_order: list[str],
    reference_stats: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for idx, name in enumerate(input_order):
        arr = np.asarray(inputs[:, idx], dtype=np.float32)
        offset, scale = _input_offset_scale(reference_stats, name)
        out[name] = arr * np.float32(scale) + np.float32(offset)
    return out


def _denorm_targets(
    targets: np.ndarray, reference_stats: Mapping[str, Any]
) -> np.ndarray:
    offset, scale = _target_offset_scale(reference_stats)
    return np.asarray(targets, dtype=np.float32) * np.float32(scale) + np.float32(
        offset
    )


def _fit_holdout_stats(
    source_root: Path,
    spec: Mapping[str, Any],
    reference_stats: Mapping[str, Any],
    eps: float,
) -> dict[str, Any]:
    reference_input_stats = reference_stats.get("inputs", {})
    if not isinstance(reference_input_stats, Mapping):
        reference_input_stats = {}
    input_stats: dict[str, RunningStats] = {}
    target_stats = RunningStats()
    selected = 0

    for shard_path, manifest in _iter_shards(source_root, "train"):
        payload = _load_shard(shard_path)
        mask = _selected_mask(payload, spec, "train")
        if not np.any(mask):
            continue
        order = _input_order(payload, manifest)
        inputs = np.asarray(payload["inputs"], dtype=np.float32)[mask]
        targets = np.asarray(payload["targets"], dtype=np.float32)[mask]
        phys_inputs = _denorm_inputs(inputs, order, reference_stats)
        phys_targets = _denorm_targets(targets, reference_stats)

        for name in order:
            if name in phys_inputs and name in reference_input_stats:
                input_stats.setdefault(name, RunningStats()).update(phys_inputs[name])
        target_stats.update(phys_targets)
        selected += int(mask.sum())

    if selected <= 0:
        raise ValueError(
            f"Holdout {spec['label']} has zero training samples after filtering."
        )

    fitted_inputs = {
        name: {
            "offset": values["offset"],
            "scale": values["scale"],
        }
        for name, values in (
            (name, stats.standardize(eps))
            for name, stats in input_stats.items()
            if stats.count > 0
        )
    }
    fitted_target = target_stats.standardize(eps)
    return {
        "method": "standardize",
        "eps": float(eps),
        "reference_stats_path": None,
        "inputs": fitted_inputs,
        "targets": {
            "enabled": True,
            "variable": "eta",
            "offset": fitted_target["offset"],
            "scale": fitted_target["scale"],
            "min": fitted_target["min"],
            "max": fitted_target["max"],
        },
    }


def _normalize_inputs(
    phys_inputs: Mapping[str, np.ndarray],
    input_order: list[str],
    fitted_stats: Mapping[str, Any],
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for name in input_order:
        arr = np.asarray(phys_inputs[name], dtype=np.float32)
        spec = fitted_stats.get("inputs", {}).get(name)
        if isinstance(spec, Mapping):
            offset = np.float32(float(spec["offset"]))
            scale = np.float32(float(spec["scale"]))
            arr = (arr - offset) / scale
        channels.append(arr)
    return np.stack(channels, axis=1).astype(np.float32)


def _normalize_targets(
    targets_phys: np.ndarray, fitted_stats: Mapping[str, Any]
) -> np.ndarray:
    spec = fitted_stats["targets"]
    offset = np.float32(float(spec["offset"]))
    scale = np.float32(float(spec["scale"]))
    return ((np.asarray(targets_phys, dtype=np.float32) - offset) / scale).astype(
        np.float32
    )


def _subset_meta(
    payload: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    n = int(mask.shape[0])
    meta = {
        "sample_id": _str_array(payload, "sample_id", n, default="")[mask],
        "source_id": _str_array(payload, "source_id", n, default="unknown")[mask],
        "source_type": _str_array(payload, "source_type", n, default="unknown")[mask],
        "bathymetry_type": _str_array(payload, "bathymetry_type", n, default="unknown")[
            mask
        ],
        "source_strength": _float_array(payload, "source_strength", n)[mask],
        "scenario_id": _str_array(payload, "scenario_id", n, default="")[mask],
        "solver_name": _str_array(payload, "solver_name", n, default="unknown")[mask],
    }
    empty_ids = meta["sample_id"] == ""
    if np.any(empty_ids):
        meta["sample_id"][empty_ids] = np.asarray(
            [f"sample_{i:06d}" for i in np.flatnonzero(empty_ids)], dtype=np.str_
        )
    empty_scenarios = meta["scenario_id"] == ""
    meta["scenario_id"][empty_scenarios] = meta["sample_id"][empty_scenarios]
    return meta


def _append_meta_jsonl(
    handle: Any, meta: Mapping[str, np.ndarray], count: int, spec: Mapping[str, Any]
) -> None:
    for i in range(count):
        row = {
            "sample_id": str(meta["sample_id"][i]),
            "scenario_id": str(meta["scenario_id"][i]),
            "source_type": str(meta["source_type"][i]),
            "bathymetry_type": str(meta["bathymetry_type"][i]),
            "source_strength": float(meta["source_strength"][i]),
            "solver_name": str(meta["solver_name"][i]),
            "strict_holdout_label": str(spec["label"]),
            "strict_holdout_key": str(spec["key"]),
            "strict_holdout_value": str(spec["value"]),
        }
        handle.write(json.dumps(row) + "\n")


def _concat_meta(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
        for key in META_KEYS
    }


def _write_shard(
    out_dir: Path,
    split_name: str,
    shard_idx: int,
    inputs: np.ndarray,
    targets: np.ndarray,
    meta: Mapping[str, np.ndarray],
    input_order: list[str],
    fitted_stats: Mapping[str, Any],
) -> dict[str, Any]:
    shard_dir = out_dir / split_name / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{shard_idx:05d}.npz"
    target_spec = fitted_stats["targets"]
    payload = {
        "inputs": inputs.astype(np.float32),
        "targets": targets.astype(np.float32),
        "sample_id": meta["sample_id"].astype(np.str_),
        "target_variable": np.asarray(["eta"], dtype=np.str_),
        "target_mean": np.asarray([float(target_spec["offset"])], dtype=np.float32),
        "target_std": np.asarray([float(target_spec["scale"])], dtype=np.float32),
        "target_min": np.asarray([float(target_spec["min"])], dtype=np.float32),
        "target_max": np.asarray([float(target_spec["max"])], dtype=np.float32),
        "input_order": np.asarray(input_order, dtype=np.str_),
        "source_id": meta["source_id"].astype(np.str_),
        "source_type": meta["source_type"].astype(np.str_),
        "bathymetry_type": meta["bathymetry_type"].astype(np.str_),
        "source_strength": meta["source_strength"].astype(np.float32),
        "scenario_id": meta["scenario_id"].astype(np.str_),
        "solver_name": meta["solver_name"].astype(np.str_),
    }
    np.savez_compressed(shard_path, **payload)
    return {
        "file": str(shard_path.relative_to(out_dir / split_name)),
        "num_samples": int(inputs.shape[0]),
        "inputs_shape": list(map(int, inputs.shape)),
        "targets_shape": list(map(int, targets.shape)),
    }


def _write_split_manifests(
    out_dir: Path,
    split_name: str,
    shards: list[dict[str, Any]],
    input_order: list[str],
    fitted_stats: Mapping[str, Any],
    shard_size: int,
) -> None:
    split_dir = out_dir / split_name
    n = int(sum(int(shard["num_samples"]) for shard in shards))
    first = shards[0] if shards else {}
    target_spec = fitted_stats["targets"]
    shard_manifest = {
        "version": 1,
        "split": split_name,
        "sharded": True,
        "num_samples": n,
        "num_shards": int(len(shards)),
        "shard_size": int(shard_size),
        "shards": shards,
        "input_order": input_order,
        "target_mode": "multi_step",
        "target_variable": "eta",
        "normalized_targets": True,
        "target_mean": float(target_spec["offset"]),
        "target_std": float(target_spec["scale"]),
        "target_min": float(target_spec["min"]),
        "target_max": float(target_spec["max"]),
    }
    eval_manifest = {
        "split": split_name,
        "sharded": True,
        "shards_manifest": "shards_manifest.json",
        "input_order": input_order,
        "target_mode": "multi_step",
        "target_variable": "eta",
        "normalized_targets": True,
        "num_samples": n,
        "num_shards": int(len(shards)),
        "inputs_shape": first.get("inputs_shape"),
        "targets_shape": first.get("targets_shape"),
    }
    _save_json(split_dir / "shards_manifest.json", shard_manifest)
    _save_json(split_dir / "eval_manifest.json", eval_manifest)


def _build_split(
    source_root: Path,
    source_split: str,
    out_dir: Path,
    out_split: str,
    spec: Mapping[str, Any],
    split_kind: str,
    reference_stats: Mapping[str, Any],
    fitted_stats: Mapping[str, Any],
    shard_size: int,
) -> dict[str, Any]:
    split_dir = out_dir / out_split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    input_order: list[str] | None = None
    input_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    meta_chunks: list[dict[str, np.ndarray]] = []
    total = 0
    heldout_count = 0
    source_counts: dict[str, int] = {}
    bathy_counts: dict[str, int] = {}

    meta_handle = (split_dir / "meta.jsonl").open("w", encoding="utf-8")

    def flush() -> None:
        nonlocal total
        if not input_chunks:
            return
        inputs = np.concatenate(input_chunks, axis=0)
        targets = np.concatenate(target_chunks, axis=0)
        meta = _concat_meta(meta_chunks)
        shard_info = _write_shard(
            out_dir=out_dir,
            split_name=out_split,
            shard_idx=len(shards),
            inputs=inputs,
            targets=targets,
            meta=meta,
            input_order=input_order or ["bathymetry", "source", "initial_depth"],
            fitted_stats=fitted_stats,
        )
        shards.append(shard_info)
        total += int(inputs.shape[0])
        input_chunks.clear()
        target_chunks.clear()
        meta_chunks.clear()

    try:
        for shard_path, manifest in _iter_shards(source_root, source_split):
            payload = _load_shard(shard_path)
            mask = _selected_mask(payload, spec, split_kind)
            if not np.any(mask):
                continue
            order = _input_order(payload, manifest)
            if input_order is None:
                input_order = order
            elif input_order != order:
                raise ValueError(
                    f"Input order mismatch in {shard_path}: {order} != {input_order}"
                )

            selected_inputs = np.asarray(payload["inputs"], dtype=np.float32)[mask]
            selected_targets = np.asarray(payload["targets"], dtype=np.float32)[mask]
            phys_inputs = _denorm_inputs(selected_inputs, order, reference_stats)
            phys_targets = _denorm_targets(selected_targets, reference_stats)
            norm_inputs = _normalize_inputs(phys_inputs, order, fitted_stats)
            norm_targets = _normalize_targets(phys_targets, fitted_stats)
            meta = _subset_meta(payload, mask)
            count = int(norm_inputs.shape[0])

            heldout_values = meta[str(spec["key"])]
            heldout_count += int(np.sum(heldout_values == str(spec["value"])))
            for value in meta["source_type"]:
                source_counts[str(value)] = source_counts.get(str(value), 0) + 1
            for value in meta["bathymetry_type"]:
                bathy_counts[str(value)] = bathy_counts.get(str(value), 0) + 1

            _append_meta_jsonl(meta_handle, meta, count, spec)

            start = 0
            while start < count:
                remaining_in_shard = shard_size - sum(
                    chunk.shape[0] for chunk in input_chunks
                )
                end = min(count, start + remaining_in_shard)
                input_chunks.append(norm_inputs[start:end])
                target_chunks.append(norm_targets[start:end])
                meta_chunks.append(
                    {key: value[start:end] for key, value in meta.items()}
                )
                if sum(chunk.shape[0] for chunk in input_chunks) >= shard_size:
                    flush()
                start = end

        flush()
    finally:
        meta_handle.close()

    if input_order is None:
        input_order = ["bathymetry", "source", "initial_depth"]
    _write_split_manifests(
        out_dir, out_split, shards, input_order, fitted_stats, shard_size
    )

    return {
        "split": out_split,
        "source_split": source_split,
        "kind": split_kind,
        "num_samples": int(total),
        "heldout_count": int(heldout_count),
        "source_type_counts": dict(sorted(source_counts.items())),
        "bathymetry_type_counts": dict(sorted(bathy_counts.items())),
    }


def _build_one_holdout(
    source_root: Path,
    output_root: Path,
    spec: Mapping[str, Any],
    reference_stats: Mapping[str, Any],
    reference_stats_path: Path,
    shard_size: int,
    eps: float,
    overwrite: bool,
) -> dict[str, Any]:
    label = str(spec["label"])
    out_dir = output_root / label
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists; pass --overwrite to replace."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted_stats = _fit_holdout_stats(source_root, spec, reference_stats, eps)
    _save_json(out_dir / "normalization_stats.json", fitted_stats)

    split_specs = [
        ("train", "train", "train"),
        ("val", "val", "val"),
        ("test", "test", "test"),
        ("test_id", "test", "test_id"),
        ("test_heldout", "test", "test_heldout"),
    ]
    split_summaries = []
    for out_split, source_split, kind in split_specs:
        split_summaries.append(
            _build_split(
                source_root=source_root,
                source_split=source_split,
                out_dir=out_dir,
                out_split=out_split,
                spec=spec,
                split_kind=kind,
                reference_stats=reference_stats,
                fitted_stats=fitted_stats,
                shard_size=shard_size,
            )
        )

    summary_by_split = {row["split"]: row for row in split_summaries}
    sanity = {
        "train_heldout_count": int(summary_by_split["train"]["heldout_count"]),
        "val_heldout_count": int(summary_by_split["val"]["heldout_count"]),
        "test_id_heldout_count": int(summary_by_split["test_id"]["heldout_count"]),
        "test_heldout_count": int(summary_by_split["test_heldout"]["heldout_count"]),
        "test_heldout_total": int(summary_by_split["test_heldout"]["num_samples"]),
        "normalization_from_train_only": True,
    }
    sanity["passed"] = bool(
        sanity["train_heldout_count"] == 0
        and sanity["val_heldout_count"] == 0
        and sanity["test_id_heldout_count"] == 0
        and sanity["test_heldout_count"] == sanity["test_heldout_total"]
        and sanity["test_heldout_total"] > 0
    )
    manifest = {
        "label": label,
        "holdout_key": str(spec["key"]),
        "holdout_value": str(spec["value"]),
        "source_root": str(source_root),
        "output_dir": str(out_dir),
        "reference_stats_path": str(reference_stats_path),
        "normalization_policy": "refit_from_filtered_train_only",
        "shard_size": int(shard_size),
        "splits": split_summaries,
        "sanity_checks": sanity,
    }
    _save_json(out_dir / "holdout_manifest.json", manifest)
    if not sanity["passed"]:
        raise RuntimeError(f"Sanity checks failed for {label}: {sanity}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict held-out-family FNO datasets."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    source_root = Path(str(cfg.get("source_root", "data/processed/hydrostatic")))
    output_root = Path(
        str(cfg.get("output_root", "data/processed_strict_holdout/hydrostatic"))
    )
    reference_stats_path = Path(
        str(cfg.get("reference_stats", source_root / "normalization_stats.json"))
    )
    shard_size = int(cfg.get("shard_size", 128))
    eps = float(cfg.get("eps", 1e-6))
    holdouts = list(cfg.get("holdouts", []))

    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not holdouts:
        raise ValueError("config.holdouts must be a non-empty list.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    reference_stats = _load_stats(reference_stats_path)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for spec in holdouts:
        if not isinstance(spec, Mapping):
            raise TypeError(f"Invalid holdout spec: {spec!r}")
        for key in ("label", "key", "value"):
            if key not in spec:
                raise KeyError(f"Holdout spec missing {key}: {spec!r}")
        print(
            f"[strict-holdout] building {spec['label']} ({spec['key']}={spec['value']})"
        )
        manifest = _build_one_holdout(
            source_root=source_root,
            output_root=output_root,
            spec=spec,
            reference_stats=reference_stats,
            reference_stats_path=reference_stats_path,
            shard_size=shard_size,
            eps=eps,
            overwrite=bool(args.overwrite),
        )
        manifests.append(manifest)
        sanity = manifest["sanity_checks"]
        print(
            f"[strict-holdout] {manifest['label']}: "
            f"train={manifest['splits'][0]['num_samples']} "
            f"val={manifest['splits'][1]['num_samples']} "
            f"test_id={manifest['splits'][3]['num_samples']} "
            f"test_heldout={manifest['splits'][4]['num_samples']} "
            f"sanity={sanity['passed']}"
        )

    index = {
        "config_path": str(cfg_path),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "reference_stats_path": str(reference_stats_path),
        "num_holdouts": len(manifests),
        "holdouts": [
            {
                "label": m["label"],
                "holdout_key": m["holdout_key"],
                "holdout_value": m["holdout_value"],
                "output_dir": m["output_dir"],
                "sanity_checks": m["sanity_checks"],
            }
            for m in manifests
        ],
    }
    _save_json(output_root / "strict_holdout_index.json", index)
    print(f"[strict-holdout] done -> {output_root}")


if __name__ == "__main__":
    main()
