"""Validate and package the completed manuscript revision evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tsunami_surrogate.evaluation.geoclaw_adapter import _adapter_hash
from tsunami_surrogate.utils.hashing import stable_json_sha256

from scripts import build_core_checkpoint_release as core
from scripts import eval_reference_resolution_sensitivity as sensitivity
from scripts import run_benchmark_numerical_study as numerical
from scripts import summarize_matched_geoclaw_comparison as matched

NUMERICAL = Path("benchmark_numerical")
REFERENCE_RESOLUTION = Path("reference_resolution_sensitivity.json")
REFERENCE_RESOLUTION_PROOF = Path("reference_resolution_checkpoint_preflight.json")
MATCHED_GEOCLAW = Path("matched_geoclaw_comparison.json")
COARSE_BUNDLE = Path(
    "artifacts/corrected_production_geoclaw/bundles/"
    "c49bae5d7577935fe266cd4659f1c26dc953cf97a389abc97fcdac2183427498"
)
GEO_CASES = {
    "test_000001_continental_gaussian",
    "test_000004_trench_rough",
    "test_000034_seamounts_fault",
}
GEO_SOLVERS = {"swe_hydrostatic", "swe_muscl_hr"}
SOURCES = (
    "LICENSE",
    "requirements.txt",
    "pytest.ini",
    "tsunami_surrogate/__init__.py",
    "configs/data/dataset_test.yaml",
    "configs/eval/benchmark_numerical_study.yaml",
    "configs/eval/corrected_production_geoclaw_validation.yaml",
    "scripts/build_core_checkpoint_release.py",
    "scripts/build_revision_evidence.py",
    "scripts/eval_reference_resolution_sensitivity.py",
    "scripts/eval_trajectory_calibration.py",
    "scripts/eval_seed_metrics.py",
    "scripts/plot_benchmark_numerical_study.py",
    "scripts/plot_trajectory_reliability.py",
    "scripts/prepare_geoclaw_refinement.py",
    "scripts/run_benchmark_numerical_study.py",
    "scripts/run_corrected_production_geoclaw.py",
    "scripts/run_periodic_linear_convergence.py",
    "scripts/summarize_geoclaw_resolution.py",
    "scripts/summarize_matched_geoclaw_comparison.py",
    "scripts/summarize_leading_training_budget.py",
    "scripts/summarize_reference_error_decomposition.py",
    "scripts/summarize_trajectory_calibration.py",
    "src/evaluation/corrected_production_geoclaw.py",
    "src/evaluation/geoclaw_adapter.py",
    "src/evaluation/boussinesq_boundary.py",
    "src/data_gen/operational_timing.py",
    "src/data_gen/generate_bathymetry.py",
    "src/data_gen/generate_sources.py",
    "tests/test_benchmark_numerical_study.py",
    "tests/test_geoclaw_refinement.py",
    "tests/test_matched_geoclaw_comparison.py",
    "tests/test_periodic_linear_convergence.py",
    "tests/test_reference_resolution_sensitivity.py",
    "tests/test_revision_evidence.py",
    "tests/test_trajectory_calibration.py",
)


@dataclass
class Inventory:
    files: dict[Path, tuple[Path, str]] = field(default_factory=dict)
    states: dict[Path, tuple[int, ...]] = field(default_factory=dict)

    def record(
        self, root: Path, relative: Path | str, name: Path | str, expected=None
    ) -> Path:
        path = core.safe_path(root, relative)
        name = Path(name)
        core.safe_path(REPO, name)
        state = core.file_state(path)
        if path in self.states:
            core.require_unchanged(path, self.states[path])
        self.states[path] = state
        checksum = core.digest(path, expected)
        entry = (path, checksum)
        if self.files.setdefault(name, entry) != entry:
            raise ValueError(f"Conflicting evidence archive path: {name}")
        return path

    def unchanged(self) -> None:
        for path, state in self.states.items():
            core.require_unchanged(path, state)


def document(path: Path):
    return json.loads(path.read_bytes())


def require_final(root: Path, relative: Path) -> None:
    if not core.safe_path(root, relative).is_file():
        raise ValueError(f"Study completion report missing: {relative}")


def check_numerical_payload(
    plan: dict, final: dict, tasks: list[dict], directory: Path, *, count: int
) -> set[str]:
    ids = {task["id"] for task in tasks}
    if (
        plan["schema_id"] != numerical.SCHEMA
        or final["schema_id"] != numerical.SCHEMA
        or not plan["complete_full_scope"]
        or not final["complete_full_planned_study"]
        or len(tasks) != count
        or len(ids) != count
        or plan["full_scope_rollout_count"] != count
        or final["rollout_count"] != count
        or final["full_scope_rollout_count"] != count
        or set(final["rollouts"]) != ids
    ):
        raise ValueError(
            f"Numerical study must contain the full prescribed {count} rollouts"
        )
    names = {f"{task_id}.npz" for task_id in ids}
    observed = {path.name for path in directory.iterdir()}
    if observed != names:
        raise ValueError(
            f"Numerical checkpoint inventory mismatch: missing={sorted(names - observed)}, extra={sorted(observed - names)}"
        )
    plan_hash = stable_json_sha256(plan)
    if final["plan_sha256"] != plan_hash:
        raise ValueError("Numerical summary plan SHA-256 mismatch")
    entries = {
        task["id"]: numerical._read_checkpoint(
            core.safe_path(directory, f"{task['id']}.npz"), task, plan_hash
        )
        for task in tasks
    }
    if numerical._summary(plan, entries, plan_hash) != final:
        raise ValueError("Numerical summary differs from validated checkpoints")
    return names


def numerical_inputs(root: Path, inventory: Inventory) -> str:
    final_name = NUMERICAL / "benchmark_numerical_study.json"
    plan_name = NUMERICAL / "plan.json"
    final_path = inventory.record(root, final_name, final_name)
    plan = document(inventory.record(root, plan_name, plan_name))
    regenerated, tasks = numerical._plan(
        numerical.DEFAULT_CONFIG, None, None, numerical.VARIANTS
    )
    if regenerated != plan:
        raise ValueError(
            "Current numerical source, data identities, runtime, or full plan differ from the recorded execution"
        )
    # Archive execution-bound source and settings, while retaining only hashes
    # for the externally released raw samples referenced in the plan.
    for row in plan["code_and_input_files"]:
        path = Path(row["path"])
        if path.is_relative_to(REPO):
            name = path.relative_to(REPO)
            inventory.record(REPO, name, name, row["sha256"])
    directory = core.safe_path(root, NUMERICAL / "checkpoints")
    for task in tasks:
        name = NUMERICAL / "checkpoints" / f"{task['id']}.npz"
        inventory.record(root, name, name)
    check_numerical_payload(plan, document(final_path), tasks, directory, count=92)
    return inventory.files[final_name][1]


def checked_tree(
    root: Path, output: Path, required: set[str], inventory: Inventory
) -> None:
    manifest = inventory.record(root, "SHA256SUMS.txt", output / "SHA256SUMS.txt")
    records = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        if name in records or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError(f"Invalid checksum entry in {manifest}: {name}")
        records[name] = checksum
    if set(records) != required:
        raise ValueError(f"Unexpected checksum inventory in {manifest}")
    for name, sha in records.items():
        inventory.record(root, name, output / name, sha)


def geoclaw_inputs(root: Path, inventory: Inventory) -> None:
    frozen = {"README.md", "external_results_manifest.json", "frozen_contract.json"}
    frozen.update(
        f"cases/{case}/{name}.npz"
        for case in GEO_CASES
        for name in ("input", "inhouse_swe_hydrostatic", "inhouse_swe_muscl_hr")
    )
    external = {"RUN_MANIFEST.json", *(f"{case}/geoclaw_swe.npz" for case in GEO_CASES)}
    evaluation = {
        "README.md",
        "comparison_rows.csv",
        "comparison_rows.json",
        "summary.json",
    }
    dirs = (
        (192, "geoclaw_external", "geoclaw_evaluation"),
        (384, "geoclaw_refined_external", "geoclaw_refined_evaluation"),
    )
    for grid, external_dir, evaluation_dir in dirs:
        bundle_dir = COARSE_BUNDLE if grid == 192 else Path("geoclaw_refined_bundle_v2")
        bundle = core.safe_path(REPO if grid == 192 else root, bundle_dir)
        frozen_prefix = Path(f"geoclaw_frozen_{grid}")
        checked_tree(bundle, frozen_prefix, frozen, inventory)
        checked_tree(
            core.safe_path(root, external_dir), Path(external_dir), external, inventory
        )
        checked_tree(
            core.safe_path(root, evaluation_dir),
            Path(evaluation_dir),
            evaluation,
            inventory,
        )
        contract = document(bundle / "frozen_contract.json")
        manifest = document(root / external_dir / "RUN_MANIFEST.json")
        summary = document(root / evaluation_dir / "summary.json")
        results = document(root / evaluation_dir / "comparison_rows.json")
        shape = [grid, grid]
        if (
            {row["case_id"] for row in contract["cases"]} != GEO_CASES
            or len(contract["cases"]) != len(GEO_CASES)
            or any(
                row["computational_shape"] != shape
                for row in contract["external_results"]
            )
            or {row["relative_path"] for row in contract["external_results"]}
            != external - {"RUN_MANIFEST.json"}
            or summary["computational_shape"] != shape
            or summary["comparison_count"] != 6
            or len(results) != 6
            or {(row["case_id"], row["inhouse_solver"]) for row in results}
            != {(case, solver) for case in GEO_CASES for solver in GEO_SOLVERS}
            or manifest["bundle_hash"] != contract["bundle_hash"]
            or summary["bundle_hash"] != contract["bundle_hash"]
            or manifest["adapter_hash"]
            != _adapter_hash(
                execution=manifest["execution"],
                revisions=manifest["revisions"],
                bundle_schema_id=contract["schema_id"],
            )
        ):
            raise ValueError(f"GeoClaw {grid} archived evidence is inconsistent")
    name = Path("geoclaw_resolution_summary_final.json")
    summary = document(inventory.record(root, name, name))
    inventory.record(
        REPO,
        "scripts/summarize_geoclaw_resolution.py",
        "scripts/summarize_geoclaw_resolution.py",
        summary["script_sha256"],
    )
    archived = {source: sha for source, sha in inventory.files.values()}
    if (
        summary["coarse_shape"] != [192, 192]
        or summary["fine_shape"] != [384, 384]
        or {row["case_id"] for row in summary["rows"]} != GEO_CASES
        or any(
            archived.get(Path(path)) != sha
            for path, sha in summary["source_files"].items()
        )
    ):
        raise ValueError(
            "GeoClaw resolution summary does not describe the packaged evidence"
        )


def periodic_inputs(root: Path, inventory: Inventory) -> None:
    prefix = Path("periodic_linear")
    report_name = prefix / "periodic_linear_convergence.json"
    report = document(inventory.record(root, report_name, report_name))
    expected = {
        (name, grid, cfl)
        for name in GEO_SOLVERS
        for grid in (32, 64, 128, 256, 512)
        for cfl in (0.1125, 0.225, 0.45)
    }
    if (
        not report["all_rollouts_finite"]
        or len(report["rows"]) != 30
        or {
            (row["solver_name"], row["grid"], row["target_cfl"])
            for row in report["rows"]
        }
        != expected
    ):
        raise ValueError(
            "Periodic wave study must contain the complete finite 30-rollout grid"
        )
    for path, sha in report["code_sha256"].items():
        inventory.record(REPO, path, path, sha)
    for name in ("rows", "spatial_orders", "cfl_sensitivity"):
        path = prefix / f"periodic_linear_{name}.csv"
        inventory.record(root, path, path)


def trajectory_inputs(root: Path, inventory: Inventory) -> str:
    raw = document(core.safe_path(root, core.TRAJECTORY_RAW))
    preflight = core.PAPER / "preflight_report.json"
    historical = document(inventory.record(REPO, preflight, preflight))
    checkpoints = {}
    for member in historical["ensemble"]["members"]:
        name = Path(member["checkpoint"])
        checkpoints[name] = core.Checkpoint(
            name, member["seed"], member["best_epoch"], member["best_checkpoint_sha256"]
        )
    evidence, _ = core.trajectory_evidence(root, checkpoints)
    for path, digest in evidence.values():
        inventory.record(root, path.name, path.name, digest)
    for path, sha in raw["provenance"]["source_files"].items():
        name = Path(path).relative_to(REPO)
        inventory.record(REPO, name, name, sha)
    summary = document(root / core.TRAJECTORY_RESULT)
    for path, sha in summary["postprocessing"]["config_dependencies"].items():
        name = Path(path).relative_to(REPO)
        inventory.record(REPO, name, name, sha)
    return inventory.files[Path(core.TRAJECTORY_RESULT)][1]


def leading_model_inputs(root: Path, inventory: Inventory) -> None:
    seeds = {18, 36, 67}
    budget_file = Path("leading_training_budget_final.json")
    budget = document(inventory.record(root, budget_file, budget_file))
    if len(budget["rows"]) != 12 or {
        (row["model"], row["seed"]) for row in budget["rows"]
    } != {(model, seed) for model in core.SEED_MODELS for seed in seeds}:
        raise ValueError(
            "Training budget must contain four leading models and three seeds"
        )
    script = "scripts/summarize_leading_training_budget.py"
    inventory.record(REPO, script, script, budget["script_sha256"])
    for row in budget["rows"]:
        if (
            row["optimizer_updates"]
            != row["updates_per_epoch"] * row["completed_epochs"]
        ):
            raise ValueError("Training budget optimizer counts disagree")
        for path, sha in row["source_files"].items():
            # Preserve source optimizer SHA from the summary. Keep the actual
            # weights and optimizer state out of this evidence-only archive.
            if not path.endswith(".pt"):
                name = Path(path).relative_to(REPO)
                inventory.record(REPO, name, name, sha)
    for model in core.SEED_MODELS:
        name = Path(f"leading_models/{model}_hydrostatic.json")
        result = document(inventory.record(root, name, name))
        if (
            result["num_samples"] != 2500
            or set(result["training_seeds"]) != seeds
            or result["member_count"] != 3
            or result["label"] != f"{model}_hydrostatic"
            or result["reference"] != "hydrostatic"
        ):
            raise ValueError(f"Leading model seed inputs incomplete: {name}")


def decomposition_inputs(root: Path, inventory: Inventory) -> None:
    name = Path("reference_error_decomposition.json")
    result = document(inventory.record(root, name, name))
    script = "scripts/summarize_reference_error_decomposition.py"
    inventory.record(REPO, script, script, result["script_sha256"])
    expected = {
        (left, right)
        for left in core.SOLVERS
        for right in core.SOLVERS
        if left != right
    }
    rows = result["cross_reference"]
    centroid = result["anonymous_reference_centroid"]
    members = centroid["member_equal_reference_rmse"]
    if (
        len(rows) != 6
        or {(row["model_solver"], row["benchmark_solver"]) for row in rows} != expected
        or any(
            not math.isclose(
                row["rho_squared"],
                1 + row["normalized_surrogate_error"] + row["normalized_alignment"],
            )
            for row in rows
        )
        or len(members) != 3
        or not math.isclose(
            centroid["model_to_centroid_rms_from_mean_mse"] ** 2
            + centroid["reference_centroid_rms"] ** 2,
            sum(value**2 for value in members) / 3,
        )
    ):
        raise ValueError(
            "Reference error decomposition is incomplete or numerically inconsistent"
        )
    for path, sha in result["source_files"].items():
        relative = Path(path).relative_to(REPO)
        inventory.record(REPO, relative, relative, sha)


def validate_sensitivity_metrics(result: dict, plan: dict, proof: dict) -> None:
    scope = result["scope"]
    cases = [
        {
            "scenario_id": case["row"]["scenario_id"],
            "sample_index": case["index"],
            "bathymetry": case["bathymetry"],
            "source": case["source"],
        }
        for case in plan["roster"]
    ]
    scenario_ids = [row["scenario_id"] for row in cases]
    if (
        result["schema_id"] != sensitivity.SCHEMA
        or scope["study_plan_sha256"] != stable_json_sha256(plan)
        or scope["study_rollout_count"] != sensitivity.ROLLOUT_COUNT
        or scope["scenarios"] != cases
        or scope["case_count"] != len(sensitivity.CASE_INDICES)
        or tuple(row["sample_index"] for row in cases) != sensitivity.CASE_INDICES
        or len(set(scenario_ids)) != len(sensitivity.CASE_INDICES)
        or scope["requested_timestamps"] != plan["requested_timestamps"]
        or scope["reference_core_grid_cells"]
        != {"archived": 128, **sensitivity.RESOLUTION}
        or scope["publication_grid_cells"] != sensitivity.SHAPE[-1]
        or scope["models"] != list(sensitivity.MODEL_NAMES)
        or scope["seeds"] != list(sensitivity.SEEDS)
        or result["checkpoint_preflight"]["status"] != "passed"
    ):
        raise ValueError(
            "Reference-resolution result differs from the prescribed six-case numerical plan"
        )
    members = result["members"]
    expected = {
        (model, seed) for model in sensitivity.MODEL_NAMES for seed in sensitivity.SEEDS
    }
    if (
        len(members) != len(expected)
        or {(row["model"], row["seed"]) for row in members} != expected
    ):
        raise ValueError(
            "Reference-resolution results must include twelve distinct leading checkpoints"
        )
    reported_proof = result["checkpoint_preflight"]["checkpoints"]
    if (
        len(reported_proof) != len(proof)
        or {row["checkpoint"]: row for row in reported_proof} != proof
    ):
        raise ValueError(
            "Reference-resolution results refer to another completion proof"
        )

    def finite(value) -> bool:
        if isinstance(value, dict):
            return all(finite(child) for child in value.values())
        if isinstance(value, list):
            return all(finite(child) for child in value)
        return type(value) not in (float, int) or math.isfinite(value)

    if not finite(result):
        raise ValueError("Reference-resolution metrics contain nonfinite values")
    for row in members:
        scores = row["scores"]
        if set(scores) != set(sensitivity.REFERENCES):
            raise ValueError(
                "Each reference-resolution checkpoint requires all four reference fields"
            )
        for metric in scores.values():
            rmse = metric["rmse_by_case"]
            relative = metric["relative_l2_by_case"]
            if (
                len(rmse) != len(cases)
                or len(relative) != len(cases)
                or any(
                    type(value) not in (float, int) or value < 0
                    for value in (*rmse, *relative)
                )
                or not math.isclose(
                    metric["global_rmse"],
                    math.sqrt(sum(x * x for x in rmse) / len(cases)),
                    rel_tol=1e-12,
                )
                or not math.isclose(
                    metric["mean_per_case_relative_l2"],
                    sum(relative) / len(cases),
                    rel_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Incomplete or inconsistent six-case reference-resolution metrics"
                )
    if result["rankings"] != sensitivity._rankings(members, scenario_ids):
        raise ValueError(
            "Reference-resolution rankings differ from the recorded member metrics"
        )


def reference_resolution_inputs(root: Path, inventory: Inventory) -> None:
    report_path = inventory.record(root, REFERENCE_RESOLUTION, REFERENCE_RESOLUTION)
    proof_path = inventory.record(
        root, REFERENCE_RESOLUTION_PROOF, core.REFERENCE_RESOLUTION_PREFLIGHT
    )
    proof = sensitivity._read_preflight(proof_path)
    plan_path = core.safe_path(root, NUMERICAL / "plan.json")
    plan = document(plan_path)
    result = document(report_path)
    validate_sensitivity_metrics(result, plan, proof)
    expected = {
        Path(name): sha for name, sha in document(proof_path)["source_files"].items()
    }
    proof_source = REPO / "scripts/eval_suite_preflight.py"
    if proof_source not in expected:
        raise ValueError(
            "Reference-resolution proof has no checkpoint-preflight source fingerprint"
        )
    expected[proof_path] = inventory.files[core.REFERENCE_RESOLUTION_PREFLIGHT][1]
    for name, (path, sha) in inventory.files.items():
        if name.is_relative_to(NUMERICAL):
            expected[path] = sha
    train, test = REPO / sensitivity.TRAIN_PATH, REPO / sensitivity.TEST_PATH
    stats = train.parent / "normalization_stats.json"
    expected[stats] = result["target_affine_transform"]["training_normalization_sha256"]
    for case in plan["roster"]:
        publication = case["publications"][sensitivity.SOLVER]["sample"]
        path = Path(publication["path"])
        expected[path] = publication["sha256"]
    required = {
        *expected,
        plan_path,
        REPO / "scripts/eval_reference_resolution_sensitivity.py",
        REPO / "src/training/checkpointing.py",
        REPO / "src/evaluation/normalization_bridge.py",
        REPO / "src/data/dataset.py",
        REPO / "src/data_gen/common_time.py",
        REPO / "src/utils/hashing.py",
        REPO / "src/utils/device.py",
        REPO / "tsunami_surrogate/__init__.py",
        REPO / sensitivity.STUDY_CONFIG,
        REPO / sensitivity.DATASET_CONFIG,
        REPO / "scripts/run_benchmark_numerical_study.py",
        *(REPO / "src/models").rglob("*.py"),
        *(
            test / name
            for name in ("eval_manifest.json", "shards_manifest.json", "meta.jsonl")
        ),
    }
    manual_receipts = set()
    for name, row in proof.items():
        path = REPO / name
        expected[path] = row["best_checkpoint_sha256"]
        expected[REPO / row["last_checkpoint"]] = row["last_checkpoint_sha256"]
        for artifact, sha in row["training_data_artifacts"].items():
            candidate = stats if artifact == stats.name else train / artifact
            if expected.setdefault(candidate, sha) != sha:
                raise ValueError(
                    f"Inconsistent training artifact provenance: {candidate}"
                )
        required.update(
            (path.with_name("history.json"), path.with_name("config_resolved.yaml"))
        )
        receipt = path.with_name("manual_completion.json")
        if receipt.is_file():
            manual_receipts.add(receipt)
    required.update(expected)
    cohorts = {case["row"]["scenario_id"] for case in plan["roster"]}
    selected_rows = set()
    with (test / "meta.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if json.loads(line)["scenario_id"] in cohorts:
                selected_rows.add(index)
    if len(selected_rows) != len(cohorts):
        raise ValueError(
            "Reference-resolution test metadata does not name the fixed six cases"
        )
    lower = 0
    for shard in document(test / "shards_manifest.json")["shards"]:
        upper = lower + shard["num_samples"]
        if any(lower <= index < upper for index in selected_rows):
            expected[test / shard["file"]] = shard["sha256"]
        lower = upper
    required.update(expected)
    sources = validate_sensitivity_sources(
        result["source_sha256"], expected, required, manual_receipts
    )
    for path, sha in sources.items():
        if (
            path.is_relative_to(REPO)
            and not path.is_relative_to(REPO / "data")
            and path.suffix != ".pt"
        ):
            name = path.relative_to(REPO)
            inventory.record(REPO, name, name, sha)
        else:
            path = core.safe_path(path.parent, path.name)
            if path not in inventory.states:
                inventory.states[path] = core.file_state(path)
                core.digest(path, sha)
            elif path not in expected:
                core.digest(path, sha)


def validate_sensitivity_sources(
    recorded: dict[str, str],
    expected: dict[Path, str],
    required: set[Path],
    optional: set[Path],
) -> dict[Path, str]:
    sources = {Path(name): sha for name, sha in recorded.items()}
    if (
        len(sources) != len(recorded)
        or not required <= sources.keys()
        or not sources.keys() <= required | optional
        or any(sources.get(path) != sha for path, sha in expected.items())
    ):
        raise ValueError(
            "Reference-resolution result omits or disagrees with its frozen source fingerprints"
        )
    return sources


def matching_metrics(recorded, computed) -> bool:
    """Allow reduction roundoff from different BLAS thread counts, not changed results."""
    if isinstance(computed, float):
        return type(recorded) in (float, int) and math.isclose(recorded, computed, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(computed, dict):
        return isinstance(recorded, dict) and recorded.keys() == computed.keys() and all(
            matching_metrics(recorded[key], value) for key, value in computed.items()
        )
    if isinstance(computed, list):
        return isinstance(recorded, list) and len(recorded) == len(computed) and all(
            matching_metrics(left, right) for left, right in zip(recorded, computed)
        )
    return recorded == computed


def matched_geoclaw_inputs(root: Path, inventory: Inventory) -> None:
    result = document(inventory.record(root, MATCHED_GEOCLAW, MATCHED_GEOCLAW))
    files = {}
    plan, dataset, _, entries = matched._study(root / NUMERICAL, files)
    summary = document(root / "geoclaw_resolution_summary_final.json")
    bundles, fields = {}, {}
    for grid, bundle, external in (
        (192, REPO / COARSE_BUNDLE, root / "geoclaw_external"),
        (384, root / "geoclaw_refined_bundle_v2", root / "geoclaw_refined_external"),
    ):
        bundles[grid], _, fields[grid] = matched._external(
            bundle, external, grid, set(summary["source_files"].values()), files
        )
    scenarios = {case["index"]: case for case in plan["roster"]}
    case_maps = {grid: {case["selection"]["sample_index"]: case for case in bundle["cases"]}
                 for grid, bundle in bundles.items()}
    times = np.asarray(plan["requested_timestamps"])
    for index in matched.CASE_INDICES:
        case = scenarios[index]
        baseline = matched._initial_arrays(case, dataset, None, files)
        case_id = case_maps[192][index]["case_id"]
        with (np.load(REPO / COARSE_BUNDLE / "cases" / case_id / "input.npz", allow_pickle=False) as parent,
              np.load(root / "geoclaw_refined_bundle_v2/cases" / case_id / "input.npz", allow_pickle=False) as fine):
            matched._verify_arrays(baseline, parent, fine, times, case_maps[192][index], case_maps[384][index])
        for kind, record in case["inputs"].items():
            path = Path(record["path"])
            inventory.record(path.parent, path.name, Path("matched_geoclaw_inputs") / kind / path.name, record["sha256"])
    rows, changes = matched._compare_cases(scenarios, case_maps, entries, fields)
    if (result["schema_id"] != matched.SCHEMA
        or result["study_plan_sha256"] != stable_json_sha256(plan)
        or result["scope"] != {"requested_times": times.tolist(), "publication_shape": [64, 64], "full_study_rollouts": 92, "case_count": 3}
        or not matching_metrics(result["rows"], rows)
        or not matching_metrics(result["paired_changes"], changes)):
        raise ValueError("Matched GeoClaw metrics differ from the verified paired fields")
    known = {path: sha for path, sha in inventory.files.values()}
    for name, sha in result["source_sha256"].items():
        path = Path(name)
        if path in known:
            if known[path] != sha:
                raise ValueError(f"Matched GeoClaw source differs: {path}")
        elif path.is_relative_to(REPO):
            inventory.record(REPO, path.relative_to(REPO), path.relative_to(REPO), sha)
        else:
            raise ValueError(f"Unpackaged matched GeoClaw source: {path}")
    if any(result["source_sha256"].get(name) != sha for name, sha in files.items()):
        raise ValueError("Matched GeoClaw result omits its verified source identities")


def figure_inputs(
    inventory: Inventory, stem: str, script_name: str, result_sha: str
) -> None:
    base = Path("paper/figures") / stem
    sidecar = base.with_suffix(".json")
    metadata = document(inventory.record(REPO, sidecar, sidecar))
    if metadata["source_sha256"] != result_sha:
        raise ValueError(
            f"Manuscript figure sidecar differs from frozen evidence: {sidecar}"
        )
    script = f"scripts/{script_name}.py"
    inventory.record(
        REPO,
        script,
        script,
        metadata.get("plot_script_sha256") or metadata["script_sha256"],
    )
    for suffix in (".pdf", ".png"):
        name = base.with_suffix(suffix)
        inventory.record(REPO, name, name)


def plan(root: Path, pdf: Path | None = None) -> Inventory:
    unfinished = []
    for path in (
        NUMERICAL / "benchmark_numerical_study.json",
        REFERENCE_RESOLUTION,
        REFERENCE_RESOLUTION_PROOF,
        MATCHED_GEOCLAW,
    ):
        try:
            require_final(root, path)
        except ValueError as exc:
            unfinished.append(str(exc))
    if unfinished:
        raise ValueError("\n".join(unfinished))
    inventory = Inventory()
    numerical_sha = numerical_inputs(root, inventory)
    reference_resolution_inputs(root, inventory)
    geoclaw_inputs(root, inventory)
    matched_geoclaw_inputs(root, inventory)
    periodic_inputs(root, inventory)
    trajectory_sha = trajectory_inputs(root, inventory)
    leading_model_inputs(root, inventory)
    decomposition_inputs(root, inventory)
    figure_inputs(
        inventory,
        "full_horizon_numerical_study",
        "plot_benchmark_numerical_study",
        numerical_sha,
    )
    figure_inputs(
        inventory,
        "trajectory_reliability",
        "plot_trajectory_reliability",
        trajectory_sha,
    )
    for name in SOURCES:
        inventory.record(REPO, name, name)
    if pdf is not None:
        path = pdf.absolute()
        inventory.record(path.parent, path.name, "paper/manuscript.pdf")
    inventory.unchanged()
    return inventory


def build(output: Path, inventory: Inventory) -> str:
    if os.path.lexists(output):
        raise FileExistsError(output)
    hashes = {str(name): sha for name, (_, sha) in sorted(inventory.files.items())}
    manifest = {
        "schema_id": "tsunami-surrogate.revision-evidence.v1",
        "files": hashes,
        "coverage": {
            "numerical_rollouts": 92,
            "matched_geoclaw_comparisons": 24,
            "periodic_rollouts": 30,
            "trajectory_validation": 1000,
            "trajectory_test": 2500,
        },
        "source_provenance": (
            "Full-horizon numerical sources match their recorded execution fingerprints. "
            "Trajectory scoring and correction have separate historical snapshots. "
            "Other sources are the packaged checkout. Historical absolute paths in original records "
            "describe their source environments."
        ),
        "verify": "From the extracted archive directory, run sha256sum -c SHA256SUMS.txt.",
        "scope": "Numerical and analysis evidence for the finite benchmark. "
        "Raw production samples and model checkpoints are distributed separately.",
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.TemporaryDirectory(
        prefix=".revision-evidence-", dir=output.parent
    ) as temp:
        staging = Path(temp) / "evidence.tar"
        with tarfile.open(staging, "w", format=tarfile.PAX_FORMAT) as archive:
            for name, (source, sha) in sorted(inventory.files.items()):
                core.add_file(archive, source, name, sha)
            core.add_bytes(archive, "revision-evidence.json", payload)
            hashes["revision-evidence.json"] = hashlib.sha256(payload).hexdigest()
            sums = "".join(f"{sha}  {name}\n" for name, sha in sorted(hashes.items()))
            core.add_bytes(archive, "SHA256SUMS.txt", sums.encode())
        inventory.unchanged()
        os.link(staging, output)
    return core.digest(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing completed revision evidence",
    )
    parser.add_argument("--paper-pdf", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--plan", action="store_true")
    output.add_argument("--build", type=Path, metavar="NEW_ARCHIVE.tar")
    args = parser.parse_args()
    if args.build is not None and os.path.lexists(args.build):
        parser.error(f"Archive already exists: {args.build}")
    try:
        inventory = plan(args.root.resolve(), args.paper_pdf)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Revision evidence is incomplete or invalid: {exc}\n")
    if args.build is not None:
        print(f"{build(args.build.absolute(), inventory)}  {args.build}")
    else:
        print(
            json.dumps(
                {
                    "file_count": len(inventory.files),
                    "input_bytes": sum(
                        inventory.states[path][2]
                        for path, _ in inventory.files.values()
                    ),
                    "paths": sorted(map(str, inventory.files)),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
