#!/usr/bin/env python
"""Select new reference-sensitivity cases from metadata and estimate rollout cost."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def select(rows: list[dict], settings: dict) -> tuple[list[dict], list[float]]:
    excluded = set(settings["exclude_indices"])
    beds = sorted({r["bathymetry_type"] for r in rows})
    sources = sorted({r["source_type"] for r in rows})
    if len(beds) != 5 or len(sources) != 6:
        raise ValueError("Expected five bathymetry and six source families")
    candidates = [r for r in rows if r["sample_index"] not in excluded]
    cuts = np.quantile([r["source_strength"] for r in candidates], [1 / 3, 2 / 3])
    selected = []
    for i, bed in enumerate(beds):
        for j, source in enumerate(sources):
            strength_bin = (i + j) % 3
            pool = [r for r in candidates if r["bathymetry_type"] == bed
                    and r["source_type"] == source
                    and int(np.searchsorted(cuts, r["source_strength"], side="right")) == strength_bin]
            if not pool:
                raise ValueError(f"No candidate for {bed}/{source}/strength tercile {strength_bin}")
            # Hash ordering is reproducible without depending on manifest row order
            # or a library-specific random-number sequence. No model errors are read.
            chosen = min(pool, key=lambda r: hashlib.sha256(
                f"{settings['selection_seed']}:{r['scenario_id']}".encode()).digest())
            selected.append({k: chosen[k] for k in (
                "sample_index", "scenario_id", "bathymetry_type", "source_type", "source_strength")})
            selected[-1].update(strength_tercile=strength_bin + 1, eligible_in_stratum=len(pool))
    if len(selected) != settings["scenario_count"] or len({r["sample_index"] for r in selected}) != 30:
        raise ValueError("Expected thirty distinct cases")
    return selected, cuts.tolist()


def timing(study: dict, count: int, workers: int) -> dict:
    groups = {}
    for solver in ("swe_hydrostatic", "swe_muscl_hr"):
        per_variant = {}
        for variant in ("coarse", "baseline", "fine"):
            values = [v["elapsed_s"] for k, v in study["rollouts"].items()
                      if k.endswith(f"__{solver}__{variant}")]
            if len(values) != 6 or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("Expected six completed positive timings per solver/grid")
            per_variant[variant] = {"count": len(values), "mean_seconds": float(np.mean(values)),
                                    "minimum_seconds": min(values), "maximum_seconds": max(values)}
        summed = count * sum(v["mean_seconds"] for v in per_variant.values()) / 3600
        groups[solver] = {"observed_seconds": per_variant, "projected_sum_task_hours": summed,
                          "idealized_four_worker_hours": summed / workers,
                          "planning_wall_hours": [1.25 * summed / workers, 1.75 * summed / workers]}
    return {"basis": "Observed elapsed times from six completed cases per solver/grid on this machine; these are not isolated CPU timings.",
            "workers": workers, "solvers": groups,
            "uncertainty": "1.25–1.75 allowance for different cases, memory pressure, scheduling and I/O; this is a planning range, not a confidence interval.",
            "prediction_evaluation_allowance_hours": [0.25, 0.5],
            "prediction_allowance_basis": "Unmeasured planning allowance; four models, three seeds, thirty cases. Benchmark a small batch before launch.",
            "memory_note": "Four simultaneous fine-grid jobs may require fewer workers if memory is insufficient; serial execution takes approximately the summed task hours."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--completed-study", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/eval/reference_sensitivity_30.yaml"))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    settings = yaml.safe_load(a.config.read_text())
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines()]
    if len(rows) != 2500 or len({r["sample_index"] for r in rows}) != 2500:
        raise ValueError("Expected the 2500-case test manifest")
    roster, cuts = select(rows, settings)
    study = json.loads(a.completed_study.read_text())
    if not study["complete_full_planned_study"]:
        raise ValueError("Timing estimates require a completed study")
    inputs = [("manifest", a.manifest), ("completed_study", a.completed_study),
              ("config", a.config), ("planner", Path(__file__))]
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs}
    plan = {"status": "planned_not_executed", "settings": settings, "source_sha256": hashes,
            "strength_tercile_boundaries": cuts, "roster": roster,
            "hydrostatic_rollout_count": 90, "both_swe_rollout_count": 180,
            "timing": timing(study, len(roster), settings["max_workers"]),
            "interpretation": "Post-review exploratory extension on an already examined test distribution. Case selection uses metadata only. Thirty family-balanced cases do not establish population-wide or within-cell ranking uncertainty."}
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (a.output_dir / "roster.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(roster[0]))
        writer.writeheader()
        writer.writerows(roster)
    print(json.dumps(plan["timing"], indent=2))


if __name__ == "__main__":
    main()
