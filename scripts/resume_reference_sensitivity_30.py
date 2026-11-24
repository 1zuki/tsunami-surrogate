#!/usr/bin/env python
"""Resume the thirty-case study without importing neural libraries in workers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Spawned numerical workers execute this module, so keep torch-dependent imports
# inside main. Numerical task definitions and checkpoint identities stay unchanged.
from scripts import run_benchmark_numerical_study as numerical


def restore_tasks(plan: dict, dataset: dict) -> list[dict]:
    by_id = {row["row"]["scenario_id"]: row for row in plan["roster"]}
    tasks = []
    for meta in plan["tasks"]:
        case_id = meta["id"].split("__", 1)[0]
        tasks.append({**meta, "dataset": dataset, "case": by_id[case_id],
                      "times": plan["requested_timestamps"]})
    if len(tasks) != 90 or len({t["id"] for t in tasks}) != 90:
        raise ValueError("Expected ninety unique planned rollouts")
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=range(1, 13), default=10)
    args = p.parse_args()
    root = args.run_root.resolve()
    os.chdir(ROOT)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        old_status = json.loads((root / "status.json").read_text())

        def status(phase, **extra):
            payload = {"phase": phase, "pid": os.getpid(), "workers": args.workers,
                       "started_utc": old_status["started_utc"],
                       "resumed_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                       "updated_utc": datetime.now(timezone.utc).isoformat(),
                       "resume_elapsed_s": time.time() - started, **extra}
            tmp = root / ".resume-status.tmp"
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            tmp.replace(root / "status.json")
            print(json.dumps(payload), flush=True)

        try:
            status("validating_resume")
            from scripts import run_reference_sensitivity_30 as original
            scoring = original.scoring
            np = original.np
            plan = scoring._json(root / "execution_plan.json")
            proof = scoring._json(root / "prediction_provenance.json")
            files = dict(proof["source_sha256"])
            for record in plan["code_and_input_files"]:
                scoring._record(files, Path(record["path"]), record["sha256"])
            for path, digest in list(files.items()):
                scoring._record(files, Path(path), digest)
            scoring._record(files, Path(__file__))
            scoring._record(files, root / "execution_plan.json")
            scoring._record(files, root / "prediction_provenance.json")
            expected_predictions = {f"{model}_seed_{seed}" for model in scoring.MODEL_NAMES for seed in scoring.SEEDS}
            if {path.stem for path in (root / "predictions").glob("*.npz")} != expected_predictions:
                raise ValueError("Resume requires all twelve validated prediction sets")
            dataset = numerical._read_yaml(ROOT / "configs/data/dataset_test.yaml")
            tasks = restore_tasks(plan, dataset)
            existing = len(list((root / "numerical/checkpoints").glob("*.npz")))
            event = {"resumed_utc": datetime.now(timezone.utc).isoformat(), "workers": args.workers,
                     "completed_checkpoints_at_resume": existing,
                     "resume_script_sha256": files[str(Path(__file__).resolve())]}
            numerical._write_json(root / f"resume-{int(started)}.json", event)
            status("numerical_rollouts", total_rollouts=90, completed_at_resume=existing)
            summary = numerical._execute(root / "numerical", plan, tasks, resume=True, workers=args.workers)
            status("scoring_fixed_predictions", total_rollouts=summary["rollout_count"])
            torch = original.torch
            torch.set_num_threads(1)
            _, archived, _, _, checks = scoring._test_inputs(ROOT / scoring.TEST_PATH, plan, files, None)
            fields = {variant: {} for variant in scoring.RESOLUTION}
            for task in tasks:
                path = root / "numerical/checkpoints" / f"{task['id']}.npz"
                eta, result = numerical._read_checkpoint(path, task, numerical.stable_json_sha256(plan))
                if result != summary["rollouts"][task["id"]]:
                    raise ValueError("Checkpoint and numerical summary disagree")
                fields[task["variant"]][task["case"]["row"]["scenario_id"]] = eta
                scoring._record(files, path)
            scenarios = [case["row"]["scenario_id"] for case in plan["roster"]]
            targets = {reference: np.stack([values[s] for s in scenarios])
                       for reference, values in {"archived": archived, **fields}.items()}
            members = []
            for model in scoring.MODEL_NAMES:
                for seed in scoring.SEEDS:
                    with np.load(root / "predictions" / f"{model}_seed_{seed}.npz", allow_pickle=False) as payload:
                        prediction = payload["prediction"]
                    members.append({"model": model, "seed": seed, "prediction": scoring.hash_array(prediction),
                                    "scores": scoring._scores(prediction, targets)})
            for path, digest in list(files.items()):
                scoring._record(files, Path(path), digest)
            numerical._write_json(root / "reference_resolution_sensitivity_30.json", {
                "schema_id": "tsunami-surrogate.reference-sensitivity-thirty.v1", "status": "completed",
                "scope": {"case_count": 30, "roster": plan["roster"], "rollout_count": 90,
                          "models": list(scoring.MODEL_NAMES), "seeds": list(scoring.SEEDS),
                          "requested_timestamps": plan["requested_timestamps"]},
                "members": members, "paired_summary": original.paired_summary(members),
                "rankings": scoring._rankings(members, scenarios), "source_sha256": files,
                "archive_checks": checks, "runtime": {"workers": args.workers,
                    "prediction_device": "cuda", "predictions_reused": True,
                    "resume_elapsed_s": time.time() - started},
                "interpretation": "Unchanged validated predictions and completed numerical checkpoints were reused. Thirty additional family-balanced cases remain separate from the original six; uncertainty summaries describe the selected scenarios rather than a design-based population interval."})
            status("completed", total_rollouts=90, completed_checkpoints=12)
        except BaseException as error:
            status("failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
