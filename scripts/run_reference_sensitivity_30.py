#!/usr/bin/env python
"""Run the predeclared thirty-case Hydrostatic reference-sensitivity extension."""
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
from scripts import run_benchmark_numerical_study as numerical
from scripts import eval_reference_resolution_sensitivity as scoring
from scripts.plan_reference_sensitivity import select

import numpy as np
import torch


def prepare(selection_path: Path, original_plan: Path, dataset_root: Path):
    selection = scoring._json(selection_path)
    old = scoring._json(original_plan)
    study = numerical._read_yaml(ROOT / "configs/eval/benchmark_numerical_study.yaml")
    dataset = numerical._read_yaml(ROOT / "configs/data/dataset_test.yaml")
    manifest = dataset_root / "synthetic/scenario_manifest.jsonl"
    inventory = dataset_root / "synthetic/native_input_inventory.jsonl"
    files = {}
    scoring._record(files, selection_path)
    scoring._record(files, original_plan)
    scoring._record(files, manifest, selection["source_sha256"]["manifest"])
    scoring._record(files, ROOT / "configs/eval/reference_sensitivity_30.yaml", selection["source_sha256"]["config"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    observed, cuts = select(rows, selection["settings"])
    if observed != selection["roster"] or cuts != selection["strength_tercile_boundaries"]:
        raise ValueError("The predeclared thirty-case selection changed")
    if len(rows) != 2500 or len({r["scenario_id"] for r in rows}) != 2500:
        raise ValueError("Expected the unique 2500-case test manifest")
    # Keep every numerical implementation bound to the completed six-case study.
    for record in old["code_and_input_files"]:
        if Path(record["path"]).is_relative_to(ROOT):
            scoring._record(files, Path(record["path"]), record["sha256"])
    inventory_hash = scoring._record(files, inventory)
    by_index = {r["sample_index"]: r for r in rows}
    roster = []
    for chosen in selection["roster"]:
        index = chosen["sample_index"]
        row = by_index[index]
        lineage = row["input_lineage"]
        if (lineage["inventory_sha256"] != inventory_hash
                or lineage["lineage_id"] != dataset["paired_inputs"]["lineage_id"]
                or lineage["solver_shape"] != [128, 128]):
            raise ValueError(f"Input identity mismatch for {row['scenario_id']}")
        sample = f"sample_{index:06d}"
        case = {"index": index, "bathymetry": row["bathymetry_type"], "source": row["source_type"],
                "boundary_controls": False, "row": row,
                "inputs": {kind: numerical._file_record(dataset_root / kind / f"{sample}.npz")
                           for kind in ("bathymetry", "sources")},
                "publications": {"swe_hydrostatic": {
                    kind: numerical._file_record(dataset_root / "raw/hydrostatic/samples" / sample / f"{kind}.npz")
                    for kind in ("sample", "provenance")}}}
        numerical._load_arrays(case, 128, lake_rest=False)
        for record in (*case["inputs"].values(), *case["publications"]["swe_hydrostatic"].values()):
            scoring._record(files, Path(record["path"]), record["sha256"])
        roster.append(case)
    requested = numerical.parse_requested_output_config(dataset["requested_output"])
    if requested is None or requested.contract_hash != old["requested_contract_hash"]:
        raise ValueError("Output times differ from the original benchmark")
    shared = {"study": study, "dataset": dataset, "times": requested.requested_times.tolist(),
              "contract_hash": requested.contract_hash, "max_natural_steps": requested.max_natural_steps}
    tasks = [numerical._make_task(variant, "swe_hydrostatic", case, shared, core)
             for case in roster for variant, core in (("coarse", 64), ("baseline", 128), ("fine", 256))]
    if len(tasks) != 90 or len({t["id"] for t in tasks}) != 90:
        raise ValueError("Expected ninety distinct Hydrostatic rollouts")
    sources = [Path(__file__), ROOT / "scripts/plan_reference_sensitivity.py",
               ROOT / "scripts/eval_reference_resolution_sensitivity.py",
               ROOT / "scripts/eval_suite_preflight.py", ROOT / "src/data/dataset.py",
               ROOT / "src/evaluation/normalization_bridge.py", ROOT / "src/training/checkpointing.py",
               ROOT / "tsunami_surrogate/__init__.py", ROOT / "src/utils/device.py"]
    sources.extend(sorted((ROOT / "src/models").rglob("*.py")))
    for source in sources:
        scoring._record(files, source)
    plan = {"schema_id": numerical.SCHEMA, "study_name": "thirty-case-hydrostatic-extension",
            "roster": roster, "requested_timestamps": shared["times"],
            "requested_contract_hash": requested.contract_hash,
            "complete_full_scope": True, "full_scope_rollout_count": 90,
            "selection_plan_sha256": files[str(selection_path.resolve())],
            "code_and_input_files": [{"path": path, "sha256": digest} for path, digest in sorted(files.items())],
            "tasks": [{k: v for k, v in task.items() if k not in {"case", "dataset", "times"}} for task in tasks],
            "interpretation": "Thirty additional family-balanced cases, Hydrostatic only, fixed raster inputs, 64/128/256 solver cores and common 64-cell outputs. This is a separate 90-rollout extension, not the original 92-rollout study or a reproduction campaign."}
    return plan, tasks, files


def paired_summary(members: list[dict], bootstrap_samples=10000) -> dict:
    models = sorted({r["model"] for r in members})
    rng = np.random.default_rng(20261125)
    output = {}
    means = {}
    count = len(members[0]["scores"]["baseline"]["relative_l2_by_case"])
    samples = rng.integers(0, count, size=(bootstrap_samples, count))
    for model in models:
        rows = [r for r in members if r["model"] == model]
        if sorted(r["seed"] for r in rows) != list(scoring.SEEDS):
            raise ValueError("Expected three fixed seeds per model")
        means[model] = {ref: np.mean([r["scores"][ref]["relative_l2_by_case"] for r in rows], axis=0)
                        for ref in scoring.REFERENCES}
        baseline, fine = means[model]["baseline"], means[model]["fine"]
        delta = fine - baseline
        percent = 100 * delta / baseline
        output[model] = {"baseline_mean_relative_l2": float(baseline.mean()),
                         "fine_mean_relative_l2": float(fine.mean()),
                         "mean_paired_absolute_change": float(delta.mean()),
                         "mean_paired_percent_change": float(percent.mean()),
                         "relative_change_in_aggregate_percent": float(100 * (fine.mean() / baseline.mean() - 1)),
                         "paired_absolute_changes_by_case": delta.tolist(),
                         "paired_percent_changes_by_case": percent.tolist(),
                         "descriptive_bootstrap_absolute_change_95": np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist()}
    pairs = []
    for i, left in enumerate(models):
        for right in models[i + 1:]:
            b = means[left]["baseline"] - means[right]["baseline"]
            f = means[left]["fine"] - means[right]["fine"]
            pairs.append({"models": [left, right], "reversed_case_indices": np.flatnonzero(b * f < 0).tolist(),
                          "fraction_reversed": float(np.mean(b * f < 0))})
    return {"models": output, "seed_averaged_pairwise_reversals": pairs,
            "interpretation": "Seeds are averaged within each scenario. Bootstrap intervals describe variation across thirty equally weighted family combinations; one scenario per combination cannot supply a design-based population interval. This is a post-review exploratory extension."}


def execute(args):
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(phase, **extra):
            payload = {"phase": phase, "pid": os.getpid(), "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                       "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.time() - started, **extra}
            temporary = args.output / ".status.tmp"
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(args.output / "status.json")
            print(json.dumps(payload), flush=True)

        try:
            status("validating_inputs")
            plan, tasks, files = prepare(args.selection.resolve(), args.original_plan.resolve(), args.dataset_root.resolve())
            numerical._write_json(args.output / "execution_plan.json", plan)
            scoring._record(files, args.preflight)
            completion = scoring._read_preflight(args.preflight)
            preflight_source = ROOT / "scripts/eval_suite_preflight.py"
            proof = scoring._json(args.preflight)
            scoring._record(files, preflight_source, proof["source_files"][str(preflight_source)])
            inputs, archived, stats_hash, target_stats, archive_checks = scoring._test_inputs(args.test_dataset.resolve(), plan, files, None)
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            device = scoring.resolve_device(args.device)
            predictions = args.output / "predictions"
            predictions.mkdir()
            proofs = []
            status("fixed_predictions", completed_checkpoints=0, total_checkpoints=12, device=str(device))
            for label in scoring.MODEL_NAMES:
                for seed in scoring.SEEDS:
                    saved, proof = scoring._checkpoint(ROOT, label, seed, stats_hash, files, None, completion)
                    model = scoring.build_model(saved["config"])
                    model.load_state_dict(saved["model_state"], strict=True)
                    model.to(device).eval()
                    del saved
                    prediction = scoring._predict(model, inputs, args.batch_size, device, target_stats)
                    if not np.isfinite(prediction).all():
                        raise ValueError("Non-finite fixed prediction")
                    path = predictions / f"{label}_seed_{seed}.npz"
                    numerical._write_atomic(path, lambda stream: np.savez_compressed(stream, prediction=prediction))
                    scoring._record(files, path)
                    proofs.append(proof)
                    del model, prediction
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    status("fixed_predictions", completed_checkpoints=len(proofs), total_checkpoints=12, device=str(device))
            numerical._write_json(args.output / "prediction_provenance.json", {
                "source_sha256": files, "checkpoints": proofs,
                "target_affine_transform": {"offset": target_stats[0], "scale": target_stats[1]},
                "input_tensor": scoring.hash_array(inputs.numpy()), "archive_checks": archive_checks})
            del inputs
            status("numerical_rollouts", total_rollouts=90, workers=args.workers)
            study_root = args.output / "numerical"
            summary = numerical._execute(study_root, plan, tasks, resume=False, workers=args.workers)
            status("scoring_fixed_predictions", total_rollouts=summary["rollout_count"])
            fields = {v: {} for v in scoring.RESOLUTION}
            for task in tasks:
                path = study_root / "checkpoints" / f"{task['id']}.npz"
                eta, result = numerical._read_checkpoint(path, task, numerical.stable_json_sha256(plan))
                if result != summary["rollouts"][task["id"]]:
                    raise ValueError("Checkpoint and numerical summary disagree")
                fields[task["variant"]][task["case"]["row"]["scenario_id"]] = eta
                scoring._record(files, path)
            scenarios = [case["row"]["scenario_id"] for case in plan["roster"]]
            targets = {ref: np.stack([values[s] for s in scenarios]) for ref, values in {"archived": archived, **fields}.items()}
            members = []
            for label in scoring.MODEL_NAMES:
                for seed in scoring.SEEDS:
                    path = predictions / f"{label}_seed_{seed}.npz"
                    scoring._record(files, path)
                    with np.load(path, allow_pickle=False) as payload:
                        prediction = payload["prediction"]
                    members.append({"model": label, "seed": seed, "prediction": scoring.hash_array(prediction),
                                    "scores": scoring._scores(prediction, targets)})
            for path, digest in list(files.items()):
                scoring._record(files, Path(path), digest)
            numerical._write_json(args.output / "reference_resolution_sensitivity_30.json", {
                "schema_id": "tsunami-surrogate.reference-sensitivity-thirty.v1", "status": "completed",
                "scope": {"case_count": 30, "roster": plan["roster"], "rollout_count": 90,
                          "models": list(scoring.MODEL_NAMES), "seeds": list(scoring.SEEDS),
                          "requested_timestamps": plan["requested_timestamps"]},
                "runtime": {"device": str(device), "batch_size": args.batch_size,
                            "torch": torch.__version__, "numpy": np.__version__,
                            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                            "workers": args.workers, "elapsed_s": time.time() - started},
                "members": members, "paired_summary": paired_summary(members),
                "rankings": scoring._rankings(members, scenarios), "source_sha256": files,
                "interpretation": "Each saved prediction is reused unchanged against every reference. Thirty additional family-balanced scenarios are separate from the original six. Numerical refinement changes both spatial and adaptive temporal resolution on fixed raster inputs; no population-wide or formal convergence claim follows."})
            status("completed", total_rollouts=90, completed_checkpoints=12)
        except BaseException as error:
            status("failed", error_type=type(error).__name__, error=str(error))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=ROOT / "docs/reference-sensitivity-30-20261125/plan.json")
    p.add_argument("--original-plan", type=Path, default=Path("/mnt/Windows/Dataset/verdict_revision_20261124/benchmark_numerical/plan.json"))
    p.add_argument("--dataset-root", type=Path, default=Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test"))
    p.add_argument("--test-dataset", type=Path, default=ROOT / scoring.TEST_PATH)
    p.add_argument("--preflight", type=Path, default=Path("/mnt/Windows/Dataset/verdict_revision_20261124/reference_resolution_checkpoint_preflight.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    if a.execute:
        execute(a)
    else:
        plan, _, _ = prepare(a.selection.resolve(), a.original_plan.resolve(), a.dataset_root.resolve())
        numerical._write_json(a.output, plan)
        print("Validated thirty scenarios and ninety tasks; no rollouts executed.")


if __name__ == "__main__":
    main()
