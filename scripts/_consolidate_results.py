#!/usr/bin/env python
"""Mirror every per-eval JSON into results/ with flat names and merge into one file.

Per-eval JSONs are produced natively by each eval script (in experiments/<model>/eval/
and results/solver_speed_*.json). This script copies them to results/<descriptive>.json
and also writes results/all_results.json: a single dict keyed by eval type -> model -> data,
so the whole evaluation can be read at a glance.
"""

from __future__ import annotations
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ACCURACY_MODELS = ["fno", "cnn", "unet", "fno_muscl_hr", "fno_boussinesq"]
FNO_MODELS = ["fno", "fno_muscl_hr", "fno_boussinesq"]
SOLVERS = ["swe_hydrostatic", "swe_muscl_hr", "boussinesq"]

# (eval-script output filename, group key in all_results, results/ flat-name prefix)
PER_MODEL_EVALS = [
    ("metrics.json", "accuracy", "accuracy"),
    ("speed.json", "speed", "speed"),
    ("ood_generalization.json", "ood_generalization", "ood_generalization"),
    (
        "resolution_transfer_proxy.json",
        "resolution_transfer_proxy",
        "resolution_transfer_proxy",
    ),
    ("perframe.json", "perframe", "perframe"),
]

PAPER_MODEL_EVALS = [
    ("eval_ood_suites/ood_generalization.json", "ood_suites", "ood_suites"),
    (
        "eval_resolution_proxy/resolution_transfer_proxy.json",
        "resolution_transfer_proxy",
        "resolution_transfer_proxy",
    ),
]

DIRECT_RESULTS = [
    ("solver_compare_hydro_vs_muscl_hr.json", "solver_comparison", "hydro_vs_muscl_hr"),
    ("solver_compare_muscl_hr_vs_hydro.json", "solver_comparison", "muscl_hr_vs_hydro"),
    (
        "solver_compare_hydro_vs_boussinesq.json",
        "solver_comparison",
        "hydro_vs_boussinesq",
    ),
    (
        "solver_compare_muscl_hr_vs_boussinesq.json",
        "solver_comparison",
        "muscl_hr_vs_boussinesq",
    ),
    (
        "solver_compare_boussinesq_vs_hydrostatic.json",
        "solver_comparison",
        "boussinesq_vs_hydrostatic",
    ),
    (
        "solver_compare_boussinesq_vs_muscl_hr.json",
        "solver_comparison",
        "boussinesq_vs_muscl_hr",
    ),
    (
        "emulator_superiority_hydro_to_muscl_hr.json",
        "emulator_superiority",
        "hydro_to_muscl_hr",
    ),
    (
        "emulator_superiority_muscl_hr_to_hydro.json",
        "emulator_superiority",
        "muscl_hr_to_hydro",
    ),
    (
        "emulator_superiority_hydro_to_boussinesq.json",
        "emulator_superiority",
        "hydro_to_boussinesq",
    ),
    (
        "emulator_superiority_muscl_hr_to_boussinesq.json",
        "emulator_superiority",
        "muscl_hr_to_boussinesq",
    ),
    (
        "emulator_superiority_boussinesq_to_hydrostatic.json",
        "emulator_superiority",
        "boussinesq_to_hydrostatic",
    ),
    (
        "emulator_superiority_boussinesq_to_muscl_hr.json",
        "emulator_superiority",
        "boussinesq_to_muscl_hr",
    ),
    (
        "native_resolution_transfer_matrix_fno_hydrostatic.json",
        "native_resolution_transfer_matrix",
        "fno_hydrostatic",
    ),
]

SINGLE_RESULTS = [
    ("speed_table.json", "speed_table"),
    ("dataset_summary.json", "dataset_summary"),
]

EXPERIMENT_RESULTS = [
    (
        "experiments/fno/eval_real_bathymetry/real_resolution.json",
        "real_bathymetry",
        "fno",
    ),
    (
        "experiments/fno_window5_hydrostatic/eval_ood_suites/window_rollout_suites.json",
        "window5_ood",
        "hydrostatic",
    ),
    (
        "experiments/fno_window5_hydrostatic/eval_crossres_native/window_rollout_suites.json",
        "window5_crossres_native",
        "hydrostatic",
    ),
    (
        "experiments/fno_window5_hydrostatic/eval_real_bathymetry/window_rollout_suites.json",
        "window5_real_bathymetry",
        "hydrostatic",
    ),
]


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    merged: dict = {}

    for fname, group, prefix in PER_MODEL_EVALS:
        merged.setdefault(group, {})
        for m in ACCURACY_MODELS:
            src = ROOT / "experiments" / m / "eval" / fname
            if src.is_file():
                shutil.copyfile(src, RESULTS / f"{prefix}_{m}.json")
                data = load(src)
                if data is not None:
                    merged[group][m] = data

    for rel_path, group, prefix in PAPER_MODEL_EVALS:
        merged.setdefault(group, {})
        for m in FNO_MODELS:
            src = ROOT / "experiments" / m / rel_path
            if src.is_file():
                shutil.copyfile(src, RESULTS / f"{prefix}_{m}.json")
                data = load(src)
                if data is not None:
                    merged[group][m] = data

    # solver speeds already write directly into results/
    merged.setdefault("solver_speed", {})
    for s in SOLVERS:
        src = RESULTS / f"solver_speed_{s}.json"
        if src.is_file():
            data = load(src)
            if data is not None:
                merged["solver_speed"][s] = data

    for fname, group, key in DIRECT_RESULTS:
        src = RESULTS / fname
        if src.is_file():
            data = load(src)
            if data is not None:
                merged.setdefault(group, {})[key] = data

    for fname, group in SINGLE_RESULTS:
        src = RESULTS / fname
        if src.is_file():
            data = load(src)
            if data is not None:
                merged[group] = data

    for rel_path, group, key in EXPERIMENT_RESULTS:
        src = ROOT / rel_path
        if src.is_file():
            data = load(src)
            if data is not None:
                merged.setdefault(group, {})[key] = data

    # prune empty groups so the merged file only shows what actually ran
    merged = {k: v for k, v in merged.items() if v}

    out = RESULTS / "all_results.json"
    out.write_text(json.dumps(merged, indent=2, sort_keys=True))

    # short console summary
    print(f"consolidated -> {out}")
    for group, payload in merged.items():
        if group in {"speed_table", "dataset_summary"}:
            print(f"  {group}: included")
        elif (
            isinstance(payload, dict)
            and payload
            and all(isinstance(k, str) for k in payload.keys())
        ):
            print(f"  {group}: {', '.join(sorted(payload))}")
        else:
            print(f"  {group}: included")


if __name__ == "__main__":
    main()
