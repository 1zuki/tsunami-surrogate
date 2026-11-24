#!/usr/bin/env python
"""Compare natural-step interpolation with steps landing on requested times."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_benchmark_numerical_study as numerical


class LandingSolver:
    """Limit adaptive steps to the next output time without changing the solver."""
    def __init__(self, solver, times):
        self.solver = solver
        self.times = times
        self.current_time = 0.0
        self.next_request = 0

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def suggest_dt(self, *, target_cfl):
        while (self.next_request < len(self.times)
               and self.current_time >= self.times[self.next_request]):
            self.next_request += 1
        natural = float(self.solver.suggest_dt(target_cfl=target_cfl))
        if self.next_request == len(self.times):
            return natural
        return min(natural, float(self.times[self.next_request]) - self.current_time)

    def step(self, *, dt, auto_dt):
        self.solver.step(dt=dt, auto_dt=auto_dt)
        self.current_time += dt


def run(task: dict) -> dict:
    if task["variant"] != "common_cfl" or task["core"] != 128:
        raise ValueError("Use a completed-study common-CFL task on the 128-cell core")
    natural_eta, natural = numerical._solve(task)
    original = numerical.FACTORIES[task["solver"]]
    numerical.FACTORIES[task["solver"]] = lambda cfg: LandingSolver(original(cfg), task["times"])
    try:
        landing_eta, landing = numerical._solve(copy.deepcopy(task))
    finally:
        numerical.FACTORIES[task["solver"]] = original
    if landing["health"]["exact_requested_knots"] != len(task["times"]):
        raise RuntimeError("Landing run did not reach every requested time exactly")
    return {"case_id": task["case"]["row"]["scenario_id"], "solver": task["solver"],
            "task_sha256": numerical.stable_json_sha256(task),
            "comparison": numerical._metrics(natural_eta, landing_eta),
            "natural": natural, "landing": landing,
            "interpretation": "Empirical sensitivity to output-only interpolation versus output-aligned integration. Shortening steps also changes time integration; this is not an isolated interpolation-error estimate."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-plan", type=Path, required=True)
    p.add_argument("--case-index", type=int, default=1)
    p.add_argument("--solver", choices=list(numerical.FACTORIES), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.study_plan.read_text())
    case = next(r for r in plan["roster"] if r["index"] == a.case_index)
    tasks = [t for t in plan["tasks"] if t["variant"] == "common_cfl"
             and t["solver"] == a.solver
             and t["id"].startswith(case["row"]["scenario_id"] + "__")]
    if len(tasks) != 1:
        raise ValueError("Expected one matching planned task")
    records = plan["code_and_input_files"]
    dataset_file = next(r for r in records if r["path"].endswith("/configs/data/dataset_test.yaml"))
    for record in records:
        if record["path"].startswith(str(numerical.ROOT)):
            numerical._check_file(record)
    task = {**tasks[0], "case": case,
            "dataset": numerical._read_yaml(numerical._check_file(dataset_file)),
            "times": plan["requested_timestamps"]}
    result = run(task)
    result["study_plan_sha256"] = hashlib.sha256(a.study_plan.read_bytes()).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
