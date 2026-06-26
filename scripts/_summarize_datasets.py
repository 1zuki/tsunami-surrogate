#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_meta(path: Path) -> Dict[str, Any]:
    counters = {
        "source_type": Counter(),
        "bathymetry_type": Counter(),
        "solver_name": Counter(),
        "quality_status": Counter(),
    }
    source_strength_min = math.inf
    source_strength_max = -math.inf
    n = 0

    if not path.is_file():
        return {"meta_rows": 0}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            for key, counter in counters.items():
                value = row.get(key)
                if value is not None:
                    counter[str(value)] += 1
            strength = row.get("source_strength")
            if strength is not None:
                try:
                    value = float(strength)
                except Exception:
                    continue
                if math.isfinite(value):
                    source_strength_min = min(source_strength_min, value)
                    source_strength_max = max(source_strength_max, value)

    out: Dict[str, Any] = {
        "meta_rows": n,
        "source_type_counts": dict(sorted(counters["source_type"].items())),
        "bathymetry_type_counts": dict(sorted(counters["bathymetry_type"].items())),
        "solver_name_counts": dict(sorted(counters["solver_name"].items())),
        "quality_status_counts": dict(sorted(counters["quality_status"].items())),
    }
    if source_strength_min < math.inf:
        out["source_strength_min"] = source_strength_min
        out["source_strength_max"] = source_strength_max
    return out


def _summarize_split(path: Path) -> Dict[str, Any]:
    manifest = _load_json(path / "shards_manifest.json")
    shards = list(manifest.get("shards", []))
    first = shards[0] if shards else {}
    out: Dict[str, Any] = {
        "path": str(path),
        "num_samples": int(manifest.get("num_samples", 0)),
        "num_shards": int(manifest.get("num_shards", len(shards))),
        "shard_size": manifest.get("shard_size"),
        "inputs_shape": first.get("inputs_shape"),
        "targets_shape": first.get("targets_shape"),
    }
    out.update(_summarize_meta(path / "meta.jsonl"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize processed sharded dataset manifests and metadata."
    )
    p.add_argument("--processed-root", type=str, default="data/processed")
    p.add_argument("--output", type=str, default="results/dataset_summary.json")
    args = p.parse_args()

    root = ROOT / args.processed_root
    datasets: Dict[str, Any] = {}
    totals: Dict[str, int] = {}

    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        splits: Dict[str, Any] = {}
        for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            if (split_dir / "shards_manifest.json").is_file():
                splits[split_dir.name] = _summarize_split(split_dir)
        if splits:
            datasets[dataset_dir.name] = splits
            totals[dataset_dir.name] = sum(
                int(row.get("num_samples", 0)) for row in splits.values()
            )

    out = {
        "evaluation_type": "dataset_summary",
        "processed_root": str(root),
        "datasets": datasets,
        "total_samples_by_dataset": totals,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"dataset summary datasets={len(datasets)} -> {output_path}")


if __name__ == "__main__":
    main()
