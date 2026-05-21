#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import save_json


def _load_eta(sample_dir: Path, field: str) -> np.ndarray:
    sample_npz = sample_dir / "sample.npz"
    rollout_npz = sample_dir / "rollout.npz"

    if sample_npz.exists():
        with np.load(sample_npz) as z:
            if field in z:
                return np.asarray(z[field], dtype=np.float64)
    if rollout_npz.exists():
        with np.load(rollout_npz) as z:
            if field in z:
                return np.asarray(z[field], dtype=np.float64)
    raise KeyError(f"Could not find '{field}' in {sample_npz} or {rollout_npz}")


def _iter_sample_map(samples_dir: Path) -> Dict[int, Path]:
    patt = re.compile(r"^sample_(\d{6})$")
    out: Dict[int, Path] = {}
    for p in sorted(samples_dir.iterdir()):
        if not p.is_dir():
            continue
        m = patt.match(p.name)
        if m is None:
            continue
        out[int(m.group(1))] = p
    return out


def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = a - b
    denom = np.sqrt(np.mean(b * b))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rel_l2 = float(rmse / max(denom, 1e-30))
    return {"rmse": rmse, "mae": mae, "max_abs": max_abs, "rel_l2": rel_l2}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two solver rollout folders in physical eta units.")
    p.add_argument("--solver-a-dir", type=str, required=True)
    p.add_argument("--solver-b-dir", type=str, required=True)
    p.add_argument("--field", type=str, default="trajectory_eta")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default="results/solver_physical_comparison.json")
    args = p.parse_args()

    dir_a = Path(args.solver_a_dir)
    dir_b = Path(args.solver_b_dir)
    if not dir_a.exists():
        raise FileNotFoundError(dir_a)
    if not dir_b.exists():
        raise FileNotFoundError(dir_b)

    map_a = _iter_sample_map(dir_a)
    map_b = _iter_sample_map(dir_b)
    shared_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    if args.max_samples is not None:
        shared_ids = shared_ids[: int(args.max_samples)]
    if not shared_ids:
        raise RuntimeError("No shared sample ids between solver directories")

    rows: List[Dict[str, Any]] = []
    per_t_rmse: List[np.ndarray] = []
    per_t_mae: List[np.ndarray] = []

    skipped_shape = 0
    skipped_nonfinite = 0

    for sid in shared_ids:
        eta_a = _load_eta(map_a[sid], args.field)
        eta_b = _load_eta(map_b[sid], args.field)

        if eta_a.ndim != 3 or eta_b.ndim != 3:
            skipped_shape += 1
            continue
        if eta_a.shape[1:] != eta_b.shape[1:]:
            skipped_shape += 1
            continue

        t = min(eta_a.shape[0], eta_b.shape[0])
        eta_a = eta_a[:t]
        eta_b = eta_b[:t]

        if not np.isfinite(eta_a).all() or not np.isfinite(eta_b).all():
            skipped_nonfinite += 1
            continue

        row = {"sample_index": int(sid), "num_frames_compared": int(t), **_metrics(eta_a, eta_b)}
        rows.append(row)

        diff = eta_a - eta_b
        per_t_rmse.append(np.sqrt(np.mean(diff * diff, axis=(1, 2))))
        per_t_mae.append(np.mean(np.abs(diff), axis=(1, 2)))

    if not rows:
        raise RuntimeError("No valid comparable samples after shape/finite checks")

    agg: Dict[str, Any] = {}
    for key in ("rmse", "mae", "max_abs", "rel_l2"):
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        agg[key] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "max": float(np.max(vals)),
        }

    min_t = min(arr.shape[0] for arr in per_t_rmse)
    rmse_stack = np.stack([arr[:min_t] for arr in per_t_rmse], axis=0)
    mae_stack = np.stack([arr[:min_t] for arr in per_t_mae], axis=0)

    out = {
        "solver_a_dir": str(dir_a),
        "solver_b_dir": str(dir_b),
        "field": args.field,
        "num_shared_samples": int(len(shared_ids)),
        "num_compared_samples": int(len(rows)),
        "num_skipped_shape_mismatch": int(skipped_shape),
        "num_skipped_nonfinite": int(skipped_nonfinite),
        "aggregate_metrics": agg,
        "per_timestep_mean": {
            "rmse": np.mean(rmse_stack, axis=0).tolist(),
            "mae": np.mean(mae_stack, axis=0).tolist(),
        },
        "per_sample": rows,
    }

    save_json(out, args.output)
    print(
        "[compare] "
        f"shared={len(shared_ids)} compared={len(rows)} "
        f"rmse_mean={out['aggregate_metrics']['rmse']['mean']:.6e} "
        f"rel_l2_mean={out['aggregate_metrics']['rel_l2']['mean']:.6e} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
