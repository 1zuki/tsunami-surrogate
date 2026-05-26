#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

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


def _load_optional_1d(sample_dir: Path, key: str) -> Optional[np.ndarray]:
    sample_npz = sample_dir / "sample.npz"
    rollout_npz = sample_dir / "rollout.npz"

    for path in (sample_npz, rollout_npz):
        if not path.exists():
            continue
        with np.load(path) as z:
            if key not in z:
                continue
            arr = np.asarray(z[key], dtype=np.float64).reshape(-1)
            if arr.size == 0:
                continue
            return arr

    return None


def _load_meta(sample_dir: Path) -> Dict[str, Any]:
    for name in ("meta.json", "metadata.json"):
        path = sample_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    return {}


def _quality_status(meta: Dict[str, Any]) -> Optional[str]:
    raw = meta.get("quality_status", None)
    if raw is not None:
        text = str(raw).strip().lower()
        if text:
            return text

    nan_count = meta.get("nan_count", None)
    inf_count = meta.get("inf_count", None)

    if nan_count is not None and inf_count is not None:
        try:
            return "ok" if int(nan_count) == 0 and int(inf_count) == 0 else "bad"
        except Exception:
            return None

    return None


def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = a - b
    denom = np.sqrt(np.mean(b * b))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rel_l2 = float(rmse / max(denom, 1e-30))
    return {"rmse": rmse, "mae": mae, "max_abs": max_abs, "rel_l2": rel_l2}


def _normalize_spectrum(power: np.ndarray) -> np.ndarray:
    power = np.asarray(power, dtype=np.float64)
    total = float(np.sum(power))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(power, 1.0 / max(power.size, 1), dtype=np.float64)

    return power / total


def _mean_power_spectrum(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 3:
        raise ValueError(f"Expected eta shape [T,H,W], got {eta.shape}")
    frames = eta.shape[0]
    if frames == 0:
        raise ValueError("eta has zero frames")

    power_acc = None
    for t in range(frames):
        f = np.fft.rfft2(eta[t], norm="ortho")
        power = np.abs(f) ** 2
        power_acc = power if power_acc is None else (power_acc + power)

    return power_acc / float(frames)


def _spectral_metrics(eta_a: np.ndarray, eta_b: np.ndarray) -> Dict[str, float]:
    pa = _normalize_spectrum(_mean_power_spectrum(eta_a))
    pb = _normalize_spectrum(_mean_power_spectrum(eta_b))
    diff = pa - pb
    spectral_rmse = float(np.sqrt(np.mean(diff * diff)))
    spectral_l1 = float(np.mean(np.abs(diff)))

    eps = 1e-12
    pa_c = np.clip(pa, eps, None)
    pb_c = np.clip(pb, eps, None)
    pa_c = pa_c / np.sum(pa_c)
    pb_c = pb_c / np.sum(pb_c)
    m = 0.5 * (pa_c + pb_c)
    js_div = 0.5 * np.sum(pa_c * np.log(pa_c / m)) + 0.5 * np.sum(pb_c * np.log(pb_c / m))

    return {
        "spectral_rmse": spectral_rmse,
        "spectral_l1": spectral_l1,
        "spectral_js_divergence": float(js_div),
    }


def _first_arrival_index(abs_eta: np.ndarray, threshold_abs: float) -> np.ndarray:
    crossed = abs_eta >= float(threshold_abs)
    has_cross = crossed.any(axis=0)
    first = np.argmax(crossed, axis=0).astype(np.int32)
    first[~has_cross] = -1

    return first

def _arrival_details(
    eta_a: np.ndarray,
    eta_b: np.ndarray,
    arrival_threshold_fraction: float,
    timestamps_a: Optional[np.ndarray],
    timestamps_b: Optional[np.ndarray],
) -> Dict[str, Any]:
    abs_a = np.abs(np.asarray(eta_a, dtype=np.float64))
    abs_b = np.abs(np.asarray(eta_b, dtype=np.float64))
    shared_peak = float(max(np.max(abs_a), np.max(abs_b), 0.0))
    thr = float(arrival_threshold_fraction) * shared_peak

    first_a = _first_arrival_index(abs_a, threshold_abs=thr)
    first_b = _first_arrival_index(abs_b, threshold_abs=thr)
    valid = (first_a >= 0) & (first_b >= 0)
    n_valid = int(np.sum(valid))
    n_total = int(valid.size)
    coverage = float(n_valid / max(n_total, 1))

    metrics: Dict[str, float] = {
        "arrival_threshold_fraction": float(arrival_threshold_fraction),
        "arrival_threshold_abs": float(thr),
        "arrival_valid_fraction": coverage,
        "arrival_valid_points": float(n_valid),
    }

    if n_valid == 0:
        metrics.update(
            {
                "arrival_mean_abs_diff_steps": float("nan"),
                "arrival_p90_abs_diff_steps": float("nan"),
                "arrival_max_abs_diff_steps": float("nan"),
            }
        )
        return {
            "metrics": metrics,
            "first_a": first_a,
            "first_b": first_b,
            "valid": valid,
            "step_diff_full": np.full(first_a.shape, np.nan, dtype=np.float64),
        }

    step_diff = np.abs(first_a[valid] - first_b[valid]).astype(np.float64)
    metrics["arrival_mean_abs_diff_steps"] = float(np.mean(step_diff))
    metrics["arrival_p90_abs_diff_steps"] = float(np.percentile(step_diff, 90))
    metrics["arrival_max_abs_diff_steps"] = float(np.max(step_diff))

    if timestamps_a is not None and timestamps_b is not None:
        if timestamps_a.ndim == 1 and timestamps_b.ndim == 1:
            ta = timestamps_a
            tb = timestamps_b
            if ta.size > 0 and tb.size > 0:
                ia = first_a[valid].astype(np.int64)
                ib = first_b[valid].astype(np.int64)
                ia = np.clip(ia, 0, ta.size - 1)
                ib = np.clip(ib, 0, tb.size - 1)
                sec_diff = np.abs(ta[ia] - tb[ib]).astype(np.float64)
                metrics["arrival_mean_abs_diff_seconds"] = float(np.mean(sec_diff))
                metrics["arrival_p90_abs_diff_seconds"] = float(np.percentile(sec_diff, 90))
                metrics["arrival_max_abs_diff_seconds"] = float(np.max(sec_diff))

    step_diff_full = np.full(first_a.shape, np.nan, dtype=np.float64)
    step_diff_full[valid] = np.abs(first_a[valid] - first_b[valid]).astype(np.float64)
    return {
        "metrics": metrics,
        "first_a": first_a,
        "first_b": first_b,
        "valid": valid,
        "step_diff_full": step_diff_full,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two solver rollout folders in physical eta units.")
    p.add_argument("--solver-a-dir", type=str, required=True)
    p.add_argument("--solver-b-dir", type=str, required=True)
    p.add_argument("--field", type=str, default="trajectory_eta")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--arrival-threshold-fraction",
        type=float,
        default=0.05,
        help="Arrival is first time |eta| exceeds this fraction of shared sample peak.",
    )
    p.add_argument(
        "--require-quality-ok",
        action="store_true",
        help=(
            "Compare only samples where both solvers have meta quality_status='ok'. "
            "If status is missing, behavior follows --missing-quality-action."
        ),
    )
    p.add_argument(
        "--missing-quality-action",
        type=str,
        default="include",
        choices=("include", "skip", "fail"),
        help=(
            "How to handle samples with missing quality_status when --require-quality-ok is set: "
            "include=allow, skip=drop, fail=raise."
        ),
    )
    p.add_argument(
        "--save-arrival-maps",
        action="store_true",
        help="Save aggregated arrival-time maps (npz) alongside scalar comparison metrics.",
    )
    p.add_argument(
        "--arrival-maps-output",
        type=str,
        default=None,
        help="Optional explicit output path for arrival maps npz. Defaults to <output_stem>_arrival_maps.npz.",
    )
    p.add_argument("--output", type=str, default="results/solver_physical_comparison.json")
    args = p.parse_args()
    if args.arrival_threshold_fraction < 0.0:
        raise ValueError("--arrival-threshold-fraction must be >= 0")

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
    arrival_maps: Dict[str, np.ndarray] | None = None

    skipped_shape = 0
    skipped_nonfinite = 0
    skipped_quality = 0

    for sid in shared_ids:
        meta_a = _load_meta(map_a[sid])
        meta_b = _load_meta(map_b[sid])
        status_a = _quality_status(meta_a)
        status_b = _quality_status(meta_b)

        if args.require_quality_ok:
            known_bad = (
                (status_a is not None and status_a != "ok")
                or (status_b is not None and status_b != "ok")
            )
            if known_bad:
                skipped_quality += 1
                continue
            
            missing = (status_a is None) or (status_b is None)
            if missing:
                if args.missing_quality_action == "fail":
                    raise ValueError(
                        f"Sample {sid:06d} missing quality_status: "
                        f"solver_a={status_a!r}, solver_b={status_b!r}"
                    )
                if args.missing_quality_action == "skip":
                    skipped_quality += 1
                    continue

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
        ts_a = _load_optional_1d(map_a[sid], "timestamps")
        ts_b = _load_optional_1d(map_b[sid], "timestamps")
        if ts_a is not None:
            ts_a = ts_a[:t]
        if ts_b is not None:
            ts_b = ts_b[:t]

        if not np.isfinite(eta_a).all() or not np.isfinite(eta_b).all():
            skipped_nonfinite += 1
            continue

        row = {
            "sample_index": int(sid),
            "num_frames_compared": int(t),
            "quality_status_a": status_a,
            "quality_status_b": status_b,
            **_metrics(eta_a, eta_b),
        }
        row.update(_spectral_metrics(eta_a, eta_b))
        arrival = _arrival_details(
            eta_a,
            eta_b,
            arrival_threshold_fraction=float(args.arrival_threshold_fraction),
            timestamps_a=ts_a,
            timestamps_b=ts_b,
        )
        row.update(arrival["metrics"])
        rows.append(row)

        if args.save_arrival_maps:
            first_a = np.asarray(arrival["first_a"], dtype=np.float64)
            first_b = np.asarray(arrival["first_b"], dtype=np.float64)
            valid = np.asarray(arrival["valid"], dtype=bool)
            step_diff_full = np.asarray(arrival["step_diff_full"], dtype=np.float64)

            if arrival_maps is None:
                h, w = first_a.shape
                arrival_maps = {
                    "count_samples": np.zeros((h, w), dtype=np.float64),
                    "count_valid_pair": np.zeros((h, w), dtype=np.float64),
                    "count_valid_a": np.zeros((h, w), dtype=np.float64),
                    "count_valid_b": np.zeros((h, w), dtype=np.float64),
                    "sum_first_a": np.zeros((h, w), dtype=np.float64),
                    "sum_first_b": np.zeros((h, w), dtype=np.float64),
                    "sum_abs_diff_steps": np.zeros((h, w), dtype=np.float64),
                }

            assert arrival_maps is not None
            arrival_maps["count_samples"] += 1.0
            valid_a = first_a >= 0
            valid_b = first_b >= 0
            arrival_maps["count_valid_pair"] += valid.astype(np.float64)
            arrival_maps["count_valid_a"] += valid_a.astype(np.float64)
            arrival_maps["count_valid_b"] += valid_b.astype(np.float64)
            arrival_maps["sum_first_a"] += np.where(valid_a, first_a, 0.0)
            arrival_maps["sum_first_b"] += np.where(valid_b, first_b, 0.0)
            arrival_maps["sum_abs_diff_steps"] += np.where(valid, step_diff_full, 0.0)

        diff = eta_a - eta_b
        per_t_rmse.append(np.sqrt(np.mean(diff * diff, axis=(1, 2))))
        per_t_mae.append(np.mean(np.abs(diff), axis=(1, 2)))

    if not rows:
        raise RuntimeError("No valid comparable samples after shape/finite checks")

    agg: Dict[str, Any] = {}
    for key in (
        "rmse",
        "mae",
        "max_abs",
        "rel_l2",
        "spectral_rmse",
        "spectral_l1",
        "spectral_js_divergence",
        "arrival_valid_fraction",
        "arrival_mean_abs_diff_steps",
        "arrival_p90_abs_diff_steps",
        "arrival_max_abs_diff_steps",
        "arrival_mean_abs_diff_seconds",
        "arrival_p90_abs_diff_seconds",
        "arrival_max_abs_diff_seconds",
    ):
        vals = np.asarray([float(r[key]) for r in rows if key in r], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
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
        "quality_filter": {
            "require_quality_ok": bool(args.require_quality_ok),
            "missing_quality_action": str(args.missing_quality_action),
        },
        "arrival_threshold_fraction": float(args.arrival_threshold_fraction),
        "num_skipped_shape_mismatch": int(skipped_shape),
        "num_skipped_nonfinite": int(skipped_nonfinite),
        "num_skipped_quality_filter": int(skipped_quality),
        "aggregate_metrics": agg,
        "per_timestep_mean": {
            "rmse": np.mean(rmse_stack, axis=0).tolist(),
            "mae": np.mean(mae_stack, axis=0).tolist(),
        },
        "per_sample": rows,
    }

    if args.save_arrival_maps and arrival_maps is not None:
        cnt_samples = np.maximum(arrival_maps["count_samples"], 1.0)
        cnt_valid_pair = np.maximum(arrival_maps["count_valid_pair"], 1.0)
        cnt_valid_a = np.maximum(arrival_maps["count_valid_a"], 1.0)
        cnt_valid_b = np.maximum(arrival_maps["count_valid_b"], 1.0)

        coverage_map = arrival_maps["count_valid_pair"] / cnt_samples
        mean_first_a = np.where(
            arrival_maps["count_valid_a"] > 0.0,
            arrival_maps["sum_first_a"] / cnt_valid_a,
            -1.0,
        )
        mean_first_b = np.where(
            arrival_maps["count_valid_b"] > 0.0,
            arrival_maps["sum_first_b"] / cnt_valid_b,
            -1.0,
        )
        mean_abs_diff_steps = np.where(
            arrival_maps["count_valid_pair"] > 0.0,
            arrival_maps["sum_abs_diff_steps"] / cnt_valid_pair,
            np.nan,
        )

        arrival_maps_path = Path(args.arrival_maps_output) if args.arrival_maps_output else Path(args.output).with_name(
            Path(args.output).stem + "_arrival_maps.npz"
        )
        arrival_maps_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrival_maps_path,
            coverage_map=coverage_map.astype(np.float32),
            mean_arrival_step_a=mean_first_a.astype(np.float32),
            mean_arrival_step_b=mean_first_b.astype(np.float32),
            mean_abs_diff_steps=mean_abs_diff_steps.astype(np.float32),
            valid_pair_count=arrival_maps["count_valid_pair"].astype(np.float32),
            sample_count=arrival_maps["count_samples"].astype(np.float32),
        )
        out["arrival_maps_path"] = str(arrival_maps_path)

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
