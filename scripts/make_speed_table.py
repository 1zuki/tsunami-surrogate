#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_json, save_json


def _read_payload(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    payload["_path"] = str(path)
    return payload


def _time_per_sample(payload: Dict[str, Any]) -> float:
    if str(payload.get("evaluation_type", "")) == "solver_speed_benchmark" and "rollout_time_per_sample_s" in payload:
        return float(payload["rollout_time_per_sample_s"])
    if "time_per_sample_mean_s" in payload:
        return float(payload["time_per_sample_mean_s"])
    if "rollout_time_per_sample_s" in payload:
        return float(payload["rollout_time_per_sample_s"])
    if "mean_seconds" in payload:
        # legacy eval_speed output was per batch; this fallback is lossy.
        return float(payload["mean_seconds"])
    raise KeyError(f"No per-sample timing field found in {payload.get('_path')}")


def _samples_per_second(payload: Dict[str, Any]) -> float:
    tps = _time_per_sample(payload)
    return float(1.0 / max(tps, 1e-12))


def _method_name(payload: Dict[str, Any]) -> str:
    for key in ("method", "solver_name", "model_name"):
        if key in payload:
            return str(payload[key])
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate solver/model runtime JSON files into a speed table CSV.")
    p.add_argument("--solver", action="append", default=[], help="Path to solver speed JSON. Repeat flag for multiple.")
    p.add_argument("--model", action="append", default=[], help="Path to model speed JSON. Repeat flag for multiple.")
    p.add_argument(
        "--baseline-solver-method",
        type=str,
        default=None,
        help="Optional baseline solver method name. Defaults to first --solver entry.",
    )
    p.add_argument("--output", type=str, default="results/speed/speed_table.csv")
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    if not args.solver:
        raise ValueError("At least one --solver payload is required")
    if not args.model:
        raise ValueError("At least one --model payload is required")

    solver_payloads = [_read_payload(path) for path in args.solver]
    model_payloads = [_read_payload(path) for path in args.model]

    baseline_payload = solver_payloads[0]
    if args.baseline_solver_method:
        for row in solver_payloads:
            if _method_name(row) == str(args.baseline_solver_method):
                baseline_payload = row
                break
        else:
            raise ValueError(f"baseline solver '{args.baseline_solver_method}' not found in --solver files")

    baseline_solver_time = _time_per_sample(baseline_payload)
    baseline_method = _method_name(baseline_payload)

    rows: List[Dict[str, Any]] = []
    for payload in solver_payloads + model_payloads:
        tps = _time_per_sample(payload)
        row = {
            "method": _method_name(payload),
            "kind": str(payload.get("evaluation_type", "unknown")),
            "device": str(payload.get("device", "unknown")),
            "precision": str(
                payload.get(
                    "precision_label",
                    payload.get("precision", payload.get("precision_actual", payload.get("precision_requested", "unknown"))),
                )
            ),
            "allow_tf32": payload.get("allow_tf32", None),
            "batch_size": payload.get("batch_size", None),
            "time_per_sample_mean_s": float(tps),
            "samples_per_second": float(_samples_per_second(payload)),
            "speedup_vs_baseline_solver_cpu": float(baseline_solver_time / max(tps, 1e-12)),
            "source_file": str(payload.get("_path")),
        }
        rows.append(row)

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "kind",
        "device",
        "precision",
        "allow_tf32",
        "batch_size",
        "time_per_sample_mean_s",
        "samples_per_second",
        "speedup_vs_baseline_solver_cpu",
        "source_file",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "baseline_solver_method": baseline_method,
        "baseline_solver_time_per_sample_s": float(baseline_solver_time),
        "warning": (
            "Speedup denominator is CPU NumPy reference solver rollout time. "
            "Interpret as implementation-level acceleration under stated hardware."
        ),
        "rows": rows,
        "output_csv": str(out_csv),
    }

    out_json = Path(args.output_json) if args.output_json else out_csv.with_suffix(".json")
    save_json(summary, out_json)
    print(summary)


if __name__ == "__main__":
    main()
