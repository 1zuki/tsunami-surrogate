#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]

MODEL_TO_SOLVER = {
    "fno": "swe_hydrostatic",
    "ffno": "swe_hydrostatic",
    "cnn": "swe_hydrostatic",
    "unet": "swe_hydrostatic",
    "convlstm": "swe_hydrostatic",
    "fno_modes8": "swe_hydrostatic",
    "fno_modes20": "swe_hydrostatic",
    "ufno": "swe_hydrostatic",
    "wno": "swe_hydrostatic",
    "fno_muscl_hr": "swe_muscl_hr",
    "fno_boussinesq": "boussinesq",
}


def _load(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate model and solver speed JSONs into paper speedup rows."
    )
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output", type=str, default="results/speed_table.json")
    p.add_argument("--csv-output", type=str, default="results/speed_table.csv")
    args = p.parse_args()

    results_dir = ROOT / args.results_dir
    rows: list[Dict[str, Any]] = []
    missing: list[str] = []

    for model, solver in MODEL_TO_SOLVER.items():
        model_path = results_dir / f"speed_{model}.json"
        # Fallback: model speed JSON is written natively under experiments/<model>/eval/
        # and only mirrored into results/ by consolidation, which may run after this step.
        if not model_path.is_file():
            model_path = ROOT / "experiments" / model / "eval" / "speed.json"
        solver_path = results_dir / f"solver_speed_{solver}.json"
        model_payload = _load(model_path)
        solver_payload = _load(solver_path)

        if model_payload is None:
            missing.append(str(model_path))
            continue
        if solver_payload is None:
            missing.append(str(solver_path))
            continue

        model_time = float(model_payload["time_per_sample_mean_s"])
        solver_time = float(
            solver_payload.get(
                "rollout_time_per_sample_s",
                solver_payload.get("time_per_sample_mean_s", 0.0),
            )
        )
        speedup = float(solver_time / model_time) if model_time > 0.0 else None

        rows.append(
            {
                "model": model,
                "model_method": model_payload.get("method", model),
                "solver": solver,
                "model_time_per_sample_s": model_time,
                "solver_rollout_time_per_sample_s": solver_time,
                "speedup_vs_solver_rollout": speedup,
                "model_samples_per_second": float(
                    model_payload.get("samples_per_second", 0.0)
                ),
                "solver_samples_per_second": float(
                    solver_payload.get("samples_per_second", 0.0)
                ),
                "model_device": model_payload.get("device"),
                "model_precision": model_payload.get("precision"),
                "solver_precision": solver_payload.get("precision_actual"),
            }
        )

    out = {
        "evaluation_type": "speed_table",
        "notes": "speedup_vs_solver_rollout = solver rollout_time_per_sample_s / model time_per_sample_mean_s",
        "rows": rows,
        "missing_inputs": missing,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = ROOT / args.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_method",
        "solver",
        "model_time_per_sample_s",
        "solver_rollout_time_per_sample_s",
        "speedup_vs_solver_rollout",
        "model_samples_per_second",
        "solver_samples_per_second",
        "model_device",
        "model_precision",
        "solver_precision",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"speed table rows={len(rows)} missing={len(missing)} -> {output_path}")


if __name__ == "__main__":
    main()
