#!/usr/bin/env python
"""Summarize the bounded amplitude and requested-output sensitivity checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-study", type=Path, required=True)
    p.add_argument("--checks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sources = {}

    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    periodic = []
    identities = []
    for amplitude, path in (
        (0.001, a.original_study / "periodic_linear/periodic_linear_convergence.json"),
        (0.0001, a.checks / "amplitude_1e-4/periodic_linear_convergence.json"),
        (0.00001, a.checks / "amplitude_1e-5/periodic_linear_convergence.json"),
    ):
        data = read(path)
        if not (data["amplitude"] == amplitude and data["final_time"] == 0.5
                and data["ny"] == 8 and data["all_rollouts_finite"]):
            raise ValueError("Incompatible periodic-wave settings")
        identities.append(data["code_sha256"])
        rows = [r for r in data["rows"] if r["grid"] in (128, 256, 512)
                and r["target_cfl"] == 0.1125]
        if len(rows) != 6:
            raise ValueError("Expected two solvers and three grids per amplitude")
        periodic.extend({"amplitude": amplitude, "reused_original_run": amplitude == 0.001,
                         **row} for row in rows)
    if not all(identity == identities[0] for identity in identities):
        raise ValueError("Periodic-wave sources differ across amplitudes")
    temporal = []
    checkpoint_root = a.original_study / "benchmark_numerical/checkpoints"
    eta = []
    for solver in ("swe_hydrostatic", "swe_muscl_hr"):
        path = checkpoint_root / f"scenario_000001__{solver}__common_cfl.npz"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            eta.append(data["eta"].astype(np.float64))
    gap = float(np.sqrt(np.mean((eta[0] - eta[1])**2)))
    for solver, name in (("swe_hydrostatic", "hydrostatic"), ("swe_muscl_hr", "muscl")):
        data = read(a.checks / f"temporal_{name}.json")
        if data["solver"] != solver or data["case_id"] != "scenario_000001":
            raise ValueError("Temporal-output scenario or solver mismatch")
        if data["landing"]["health"]["exact_requested_knots"] != 50:
            raise ValueError("Not all output times were reached exactly")
        temporal.append({"solver": solver, "comparison": data["comparison"],
                         "rmse_fraction_of_same_case_common_cfl_solver_gap": data["comparison"]["trajectory_rmse"] / gap,
                         "natural_elapsed_s": data["natural"]["elapsed_s"],
                         "landing_elapsed_s": data["landing"]["elapsed_s"],
                         "natural_health": data["natural"]["health"],
                         "landing_health": data["landing"]["health"]})
    result = {"status": "completed_bounded_checks", "source_sha256": sources,
              "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "new_rollouts": 16, "reused_periodic_rows": 6,
              "periodic_amplitude_rows": periodic, "temporal_comparisons": temporal,
              "same_case_common_cfl_solver_gap_rmse": gap,
              "interpretation": "The amplitude sweep probes nonlinear-reference mismatch and residual numerical effects. Output landing changes integration as well as extraction. One temporal-test case does not establish population-wide interpolation accuracy."}
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved twelve new amplitude runs and four new temporal-output runs to {a.output}")


if __name__ == "__main__":
    main()
