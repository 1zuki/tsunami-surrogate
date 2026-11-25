"""Build and replay the frozen core common-time benchmark release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsunami_surrogate.utils.hashing import sha256_file

SCHEMA = "tsunami-surrogate.core-checkpoint-release.v1"
PAPER = Path("evaluation_runs/final-v2-full-nonspeed-20260912-r6")
POOLED = Path("evaluation_runs/pooled-reference-ablation-r6/primary_equal_updates.json")
SEED_RESULTS = Path("results/reviewer_revision/seed_metrics")
DATA = Path("data/processed")
SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")
SEEDS = (18, 36, 67)
ENSEMBLE_SEEDS = (11, 22, 33, 44, 55, 66, 77)
SEED_MODELS = {"fno": 64, "ffno": 64, "convlstm": 8, "ufno": 8}
GROUPS = ("direct", "multiseed", "reference", "pooled", "ensemble", "trajectory")
CONTRACT = "configs/eval/benchmark_suite.yaml"
MANIFEST = "core-release.json"
SUMS = "SHA256SUMS.txt"
DATA_DOI = "https://doi.org/10.5281/zenodo.22997509"
TRAJECTORY_DIR = Path("evidence/trajectory")
TRAJECTORY_RESULT = "trajectory_calibration_physical.json"
TRAJECTORY_RAW = "trajectory_calibration.json"
TRAJECTORY_SOURCE = "trajectory_execution_source.py"
TRAJECTORY_SUMMARY_SOURCE = "trajectory_postprocessing_source.py"
TRAJECTORY_LINEAGE = "trajectory_lineage_audit.json"
REFERENCE_RESOLUTION_PREFLIGHT = Path(
    "evidence/reference_resolution/checkpoint_preflight.json"
)
SCRIPT_FILES = (
    "eval_accuracy",
    "eval_ensemble_calibration",
    "eval_pooled_reference_ablation",
    "eval_suite_preflight",
    "eval_trajectory_calibration",
    "eval_calibration",
    "eval_reference_analysis",
    "eval_seed_metrics",
    "eval_slices",
    "eval_window_rollout",
    "eval_window_suites",
)
SUPPLEMENTARY_SOURCE = (
    Path("scripts/eval_reference_resolution_sensitivity.py"),
    Path("scripts/plot_trajectory_reliability.py"),
    Path("scripts/prepare_geoclaw_refinement.py"),
    Path("scripts/run_benchmark_numerical_study.py"),
    Path("scripts/run_periodic_linear_convergence.py"),
    Path("scripts/run_production_dense_diagnostic.py"),
    Path("scripts/summarize_trajectory_calibration.py"),
    Path("configs/eval/benchmark_numerical_study.yaml"),
    Path("tests/test_benchmark_numerical_study.py"),
    Path("tests/test_core_checkpoint_release.py"),
    Path("tests/test_geoclaw_refinement.py"),
    Path("tests/test_periodic_linear_convergence.py"),
    Path("tests/test_production_dense_diagnostic.py"),
    Path("tests/test_reference_resolution_sensitivity.py"),
    Path("tests/test_source_checkout_imports.py"),
    Path("tests/test_trajectory_calibration.py"),
)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    seed: int
    epoch: int | None
    expected_sha256: str | None
    proof_sha256: str | None = None

    @property
    def resolved_config(self) -> Path:
        return self.path.with_name("config_resolved.yaml")


def safe_path(root: Path, name: Path | str) -> Path:
    relative = Path(name)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or any(character in str(relative) for character in "\r\n\\")
        or path.resolve() != path
    ):
        raise ValueError(f"Unsafe release path: {name}")
    return path


def file_state(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def require_unchanged(path: Path, state: tuple[int, ...]) -> None:
    if file_state(path) != state:
        raise ValueError(f"File changed during release operation: {path}")


def digest(path: Path, expected: str | None = None) -> str:
    state = file_state(path)
    actual = sha256_file(path)
    require_unchanged(path, state)
    if expected is not None and actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, found {actual}"
        )
    return actual


def read_json(root: Path, name: Path, state: dict[Path, tuple[int, ...]]) -> dict:
    path = safe_path(root, name)
    state[name] = file_state(path)
    document = json.loads(path.read_bytes())
    require_unchanged(path, state[name])
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object: {name}")
    return document


def checkpoint_args(paths) -> list[str]:
    return [arg for path in paths for arg in ("--checkpoint", str(path))]


def replay_step(script: str, args: list[str], evidence, labels: list[str]) -> dict:
    return {
        "script": f"scripts/{script}.py",
        "arguments": args,
        "archived_evidence": [str(path) for path in evidence],
        "paper_table_labels": [label for label in labels if label.startswith("tab:")],
        "paper_figure_labels": [label for label in labels if label.startswith("fig:")],
    }


def plan(
    root: Path,
) -> tuple[dict, dict[Path, Checkpoint], dict[Path, str | None], dict]:
    states: dict[Path, tuple[int, ...]] = {}
    run_manifest = PAPER / "run_manifest.json"
    completion_manifest = PAPER / "completion_manifest.json"
    preflight_report = PAPER / "preflight_report.json"
    completed = read_json(root, completion_manifest, states)
    run = read_json(root, run_manifest, states)
    preflight = read_json(root, preflight_report, states)
    if completed["status"] != "validated" or preflight["status"] != "passed":
        raise ValueError("The archived evaluation run must have completed validation.")
    if any(record["run_id"] != PAPER.name for record in (completed, run)):
        raise ValueError("Archived evaluation run ID does not match its directory.")
    recorded_files = {
        PAPER / row["path"]: row["sha256"] for row in completed["artifacts"]
    }
    checkpoints: dict[Path, Checkpoint] = {}
    files: dict[Path, str | None] = {
        completion_manifest: None,
        run_manifest: completed["run_manifest_sha256"],
        preflight_report: run["preflight_report_sha256"],
        PAPER / "parameter_counts.json": None,
        POOLED: None,
        Path("LICENSE"): None,
        Path("pytest.ini"): None,
        Path("requirements.txt"): None,
        Path("tsunami_surrogate/__init__.py"): None,
        Path("scripts/build_core_checkpoint_release.py"): None,
        **{Path(f"scripts/{script}.py"): None for script in SCRIPT_FILES},
    }
    git = ["git", "-C", str(root)]
    source = subprocess.check_output(git + ["ls-files", "-z", "--", "src", "configs"])
    files.update(
        {Path(os.fsdecode(name)): None for name in source.split(b"\0") if name}
    )
    files.update(
        {path: None for path in SUPPLEMENTARY_SOURCE if (root / path).is_file()}
    )
    recipes: dict[str, list[dict]] = {group: [] for group in GROUPS}
    training = {
        row["checkpoint"]: row
        for row in [*preflight["checkpoints"], *preflight["ensemble"]["members"]]
    }

    def remember(
        path: Path, seed: int, epoch: int | None, sha256: str | None = None
    ) -> None:
        checkpoint = Checkpoint(path, seed, epoch, sha256)
        if path in checkpoints and checkpoints[path] != checkpoint:
            raise ValueError(f"Inconsistent archived checkpoint identity: {path}")
        checkpoints[path] = checkpoint
        files.setdefault(checkpoint.resolved_config, None)

    for row in run["cells"]:
        if row["group"] != "direct_accuracy":
            continue
        label = row["id"].split(":", 1)[1]
        checkpoint = training[row["checkpoint"]]
        if checkpoint["best_checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError(f"Conflicting historical SHA-256 for {row['checkpoint']}")
        remember(
            Path(row["checkpoint"]),
            checkpoint["seed"],
            checkpoint["best_epoch"],
            row["checkpoint_sha256"],
        )
        files[Path(row["config"])] = row["config_sha256"]
        evidence = PAPER / row["path"]
        recipes["direct"].append(
            replay_step(
                "eval_accuracy",
                [
                    "--config",
                    row["config"],
                    "--checkpoint",
                    row["checkpoint"],
                    "--output",
                    f"{{output}}/direct/{label}.json",
                ],
                [evidence],
                ["tab:accuracy"],
            )
        )

    for label, batch_size in SEED_MODELS.items():
        evidence = SEED_RESULTS / f"{label}_hydrostatic.json"
        row = read_json(root, evidence, states)
        if tuple(row["training_seeds"]) != SEEDS:
            raise ValueError(f"Unexpected training seeds in {evidence}")
        for member in row["members"]:
            path = Path(member["checkpoint"])
            historical = training.get(str(path))
            remember(
                path,
                member["training_seed"],
                member["checkpoint_epoch"],
                historical["best_checkpoint_sha256"] if historical else None,
            )
        recipes["multiseed"].append(
            replay_step(
                "eval_seed_metrics",
                [
                    "--config",
                    row["config_path"],
                    *checkpoint_args(row["checkpoints"]),
                    "--dataset",
                    row["dataset_path"],
                    "--label",
                    row["label"],
                    "--reference",
                    row["reference"],
                    "--expected-seeds",
                    *map(str, SEEDS),
                    "--batch-size",
                    str(batch_size),
                    "--bootstrap-seed",
                    str(row["bootstrap"]["seed"]),
                    "--bootstrap-resamples",
                    str(row["bootstrap"]["resamples"]),
                    "--output",
                    f"{{output}}/multiseed/{label}.json",
                ],
                [evidence],
                ["tab:multiseed-architecture"],
            )
        )

    reference_args = []
    for solver in SOLVERS:
        label = "fno" if solver == "hydrostatic" else f"fno_{solver}"
        path = Path(f"experiments/{label}/seed_18/best.pt")
        if path not in checkpoints:
            raise ValueError(f"Missing direct FNO checkpoint for {solver}")
        reference_args += [
            "--model",
            f"{solver}|configs/model/{label}.yaml|{path}|{DATA / solver / 'test'}",
        ]
        for suffix in ("json", "npz"):
            evidence = PAPER / f"paper_evidence/arrival_maps/fno_{solver}.{suffix}"
            files[evidence] = recorded_files[evidence]
    recipes["reference"].append(
        replay_step(
            "eval_reference_analysis",
            [
                "--contract",
                CONTRACT,
                *reference_args,
                "--bootstrap-seed",
                "20260813",
                "--bootstrap-resamples",
                "2000",
                "--output-dir",
                "{output}/reference",
            ],
            [
                PAPER / f"paper_evidence/reference_analysis/{metric}.json"
                for metric in ("solver_gap", "cross_reference")
            ],
            ["tab:solvergap", "tab:superiority", "tab:ratio-decomposition"],
        )
    )

    pooled = read_json(root, POOLED, states)
    pooled_args = []
    for solver in SOLVERS:
        pooled_args.extend(
            ["--reference-dataset", f"{solver}|{pooled['reference_datasets'][solver]}"]
        )
    for label, row in pooled["models"].items():
        path = Path(row["checkpoint"])
        seed = int(path.parent.name.removeprefix("seed_"))
        if seed not in SEEDS:
            raise ValueError(f"Unexpected pooled training seed in {path}")
        remember(path, seed, None, row["checkpoint_sha256"])
        stats = row["model_normalization_stats"]
        files[Path(stats)] = row["model_normalization_stats_sha256"]
        model = [
            label,
            row["config"],
            str(path),
            stats,
            row["target_solver_for_input_rebasing"],
        ]
        if row["conditioned_solver"] is not None:
            model.append(row["conditioned_solver"])
        pooled_args += ["--model", "|".join(model)]
    recipes["pooled"].append(
        replay_step(
            "eval_pooled_reference_ablation",
            [
                "--contract",
                CONTRACT,
                *pooled_args,
                "--bootstrap-seed",
                str(pooled["bootstrap"]["seed"]),
                "--bootstrap-resamples",
                str(pooled["bootstrap"]["resamples"]),
                "--output",
                "{output}/pooled/primary_equal_updates.json",
            ],
            [POOLED],
            ["tab:pooled-reference"],
        )
    )

    ensemble_paths = []
    for seed in ENSEMBLE_SEEDS:
        path = Path(f"experiments/ensemble/member_{seed}/best.pt")
        member = training[str(path)]
        remember(path, seed, member["best_epoch"], member["best_checkpoint_sha256"])
        ensemble_paths.append(path)
    ensemble_args = [
        "--config",
        "configs/model/fno.yaml",
        *checkpoint_args(ensemble_paths),
    ]
    recipes["ensemble"] = [
        replay_step(
            "eval_calibration",
            [
                *ensemble_args,
                "--val-dataset",
                str(DATA / "hydrostatic/val"),
                "--test-dataset",
                str(DATA / "hydrostatic/test"),
                "--suite",
                "source_holdout_multi_gauss=source_type_in:multi-gauss",
                "--suite",
                "bathymetry_holdout_trench=bathymetry_type_in:trench",
                "--suite",
                "source_strength_extreme_high=source_strength_min:0.285",
                "--output",
                "{output}/ensemble/calibration.json",
            ],
            [PAPER / "paper_evidence/ensemble/calibration.json"],
            ["tab:uncertainty", "tab:app-ensemble-calibration"],
        ),
        replay_step(
            "eval_seed_metrics",
            [
                *ensemble_args,
                "--dataset",
                str(DATA / "hydrostatic/test"),
                "--bootstrap-seed",
                "20260813",
                "--bootstrap-resamples",
                "2000",
                "--output",
                "{output}/ensemble/seed_stability.json",
            ],
            [PAPER / "paper_evidence/ensemble/seed_stability.json"],
            ["tab:app-ensemble"],
        ),
    ]
    for steps in recipes.values():
        for step in steps:
            for archived in step["archived_evidence"]:
                path = Path(archived)
                files[path] = recorded_files.get(path)

    recipes["trajectory"] = [
        replay_step(
            "eval_trajectory_calibration",
            [
                *ensemble_args,
                "--val-dataset",
                str(DATA / "hydrostatic/val"),
                "--test-dataset",
                str(DATA / "hydrostatic/test"),
                "--batch-size",
                "2",
                "--std-floor",
                "1e-6",
                "--output",
                "{output}/trajectory/trajectory_calibration_physical.json",
            ],
            [TRAJECTORY_DIR / TRAJECTORY_RESULT],
            ["tab:trajectory-coverage", "fig:trajectory-reliability"],
        )
    ]

    metadata = {
        "schema_id": SCHEMA,
        "source_checkout_base_commit": subprocess.check_output(
            git + ["rev-parse", "HEAD"], text=True
        ).strip(),
        "historical_evaluation_code": run["code_state"],
        "historical_evaluation_run": PAPER.name,
        "processed_data_doi": DATA_DOI,
        "recipes": recipes,
    }
    return metadata, checkpoints, files, states


def trajectory_evidence(
    source_dir: Path, checkpoints: dict[Path, Checkpoint]
) -> tuple[dict[Path, tuple[Path, str]], dict[Path, str]]:
    """Verify the linkage between the historical run and its physical summary."""
    states = {}
    raw = read_json(root=source_dir, name=Path(TRAJECTORY_RAW), state=states)
    physical = read_json(root=source_dir, name=Path(TRAJECTORY_RESULT), state=states)
    audit = read_json(root=source_dir, name=Path(TRAJECTORY_LINEAGE), state=states)
    provenance = raw["provenance"]
    postprocessing = physical["postprocessing"]
    checksums = {
        TRAJECTORY_RAW: digest(
            safe_path(source_dir, TRAJECTORY_RAW),
            postprocessing["inference_result_sha256"],
        ),
        TRAJECTORY_SOURCE: digest(
            safe_path(source_dir, TRAJECTORY_SOURCE), provenance["script_sha256"]
        ),
        TRAJECTORY_SUMMARY_SOURCE: digest(
            safe_path(source_dir, TRAJECTORY_SUMMARY_SOURCE),
            postprocessing["script_sha256"],
        ),
        TRAJECTORY_RESULT: digest(safe_path(source_dir, TRAJECTORY_RESULT)),
        TRAJECTORY_LINEAGE: digest(safe_path(source_dir, TRAJECTORY_LINEAGE)),
    }
    if any(
        result["status"] != "completed_full_splits"
        or result["calibration_fit_split"] != "validation"
        for result in (raw, physical)
    ):
        raise ValueError(
            "Trajectory evidence must cover the full validation and test splits."
        )
    if (
        raw["thresholds"] != physical["thresholds"]
        or physical["provenance"] != provenance
        or postprocessing["score_and_thresholds_changed"] is not False
        or physical["groups"]["source_strength>=0.285"]["scenario_count"] != 177
        or audit["status"] != "passed"
    ):
        raise ValueError(
            "The trajectory evidence must contain unchanged bands and the physical risk summary."
        )
    original_checkpoints = {
        Path(*Path(name).parts[-4:]): sha256
        for name, sha256 in provenance["checkpoints"].items()
    }
    expected_checkpoints = {
        path: entry.expected_sha256
        for path, entry in checkpoints.items()
        if path.parent.parent == Path("experiments/ensemble")
    }
    if (
        len(expected_checkpoints) != len(ENSEMBLE_SEEDS)
        or original_checkpoints != expected_checkpoints
    ):
        raise ValueError(
            "The archived trajectory score was computed with a different ensemble."
        )

    dataset_files = {}
    for split, directory, count in (
        ("validation", "val", 1000),
        ("test", "test", 2500),
    ):
        source_rows, summary_rows = raw["records"][split], physical["records"][split]
        if (
            len(source_rows) != count
            or len(summary_rows) != count
            or any(
                (before["scenario_id"], before["score"])
                != (after["scenario_id"], after["score"])
                or after["rel_l2_normalized"] != before["rel_l2"]
                for before, after in zip(source_rows, summary_rows, strict=True)
            )
        ):
            raise ValueError(
                f"The physical summary changed the {split} scenario scores or roster."
            )
        base = DATA / "hydrostatic" / directory
        recorded = {
            Path(name).name: sha256 for name, sha256 in provenance[split].items()
        }
        if (
            set(recorded)
            != {"eval_manifest.json", "shards_manifest.json", "meta.jsonl"}
            or recorded["meta.jsonl"] != audit["splits"][split]["metadata_sha256"]
            or audit["splits"][split]["rows"] != count
        ):
            raise ValueError(
                f"The archived trajectory lineage audit covers a different {split} roster."
            )
        for name, sha256 in recorded.items():
            dataset_files[base / name] = sha256
    intersections = audit["split_pair_intersection_counts"]
    split_pairs = {"train__validation", "train__test", "validation__test"}
    if set(intersections) != split_pairs or any(
        row["master_input_hash_pair"] != 0 for row in intersections.values()
    ):
        raise ValueError(
            "The archived trajectory lineage audit found overlapping inputs."
        )
    for name, state in states.items():
        require_unchanged(safe_path(source_dir, name), state)

    external = {
        TRAJECTORY_DIR / name: (safe_path(source_dir, name), sha256)
        for name, sha256 in checksums.items()
    }
    return external, dataset_files


def reference_resolution_preflight(
    root: Path, proof: Path, checkpoints: dict[Path, Checkpoint]
) -> tuple[str, dict[Path, Checkpoint], dict[Path, str]]:
    """Bind completed-training proof to original weights before stripping them."""
    import torch

    proof = safe_path(proof.parent, proof.name)
    state = file_state(proof)
    payload = proof.read_bytes()
    proof_sha = hashlib.sha256(payload).hexdigest()
    record = json.loads(payload)
    if record.get("status") != "passed":
        raise ValueError("Reference-resolution checkpoint preflight did not pass")
    rows = record["checkpoints"]
    expected = {
        Path(f"experiments/{model}/seed_{seed}/best.pt")
        for model in SEED_MODELS
        for seed in SEEDS
    }
    if (
        len(rows) != len(expected)
        or {Path(row["checkpoint"]) for row in rows} != expected
    ):
        raise ValueError(
            "Reference-resolution proof must cover exactly the twelve prescribed checkpoints"
        )

    sources = {}
    for name, expected_sha in record["source_files"].items():
        relative = (
            Path(name).relative_to(root) if Path(name).is_absolute() else Path(name)
        )
        if relative.parts[0] not in ("scripts", "src", "tsunami_surrogate"):
            raise ValueError(f"Unexpected preflight source path: {relative}")
        sources[relative] = digest(safe_path(root, relative), expected_sha)
    if Path("scripts/eval_suite_preflight.py") not in sources:
        raise ValueError(
            "Reference-resolution proof lacks its checkpoint-preflight source digest"
        )

    bound = {}
    for row in rows:
        name = Path(row["checkpoint"])
        entry = checkpoints[name]
        expected_sha = row["best_checkpoint_sha256"]
        completed = {
            "horizon": row["configured_epochs"] > 0
            and row["last_epoch"] >= row["configured_epochs"],
            "early_stopping": row["patience"] > 0
            and row["early_count"] >= row["patience"],
            "manual_resource_stop": row.get("manual_completion") is not None,
        }
        if (
            not completed.get(row["completion"], False)
            or not 1 <= row["best_epoch"] <= row["last_epoch"]
            or row["training_data_identity_strength"] != "content_bound"
            or row["seed"] != entry.seed
            or row["best_epoch"] != entry.epoch
            or (
                entry.expected_sha256 is not None
                and expected_sha != entry.expected_sha256
            )
        ):
            raise ValueError(
                f"Reference-resolution proof disagrees with checkpoint metadata: {name}"
            )
        path = safe_path(root, name)
        source_state = file_state(path)
        source_hash = digest(path, expected_sha)
        saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        if saved["config"]["seed"] != entry.seed or saved["epoch"] != entry.epoch:
            raise ValueError(
                f"Original checkpoint seed/epoch differs from its proof: {name}"
            )
        require_unchanged(path, source_state)
        bound[name] = replace(entry, proof_sha256=source_hash)
    require_unchanged(proof, state)
    return proof_sha, bound, sources


def archive_member(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o644
    member.mtime = 0
    return member


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    archive.addfile(archive_member(name, len(payload)), io.BytesIO(payload))


def add_file(archive: tarfile.TarFile, path: Path, name: Path, expected=None) -> str:
    state = file_state(path)
    sha256 = hashlib.sha256()

    class HashingReader(io.BufferedReader):
        def read(self, size: int = -1) -> bytes:
            data = super().read(size)
            sha256.update(data)
            return data

    with HashingReader(path.open("rb", buffering=0)) as handle:
        archive.addfile(archive_member(str(name), state[2]), handle)
    require_unchanged(path, state)
    actual = sha256.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, found {actual}"
        )
    return actual


def freeze_checkpoint(root: Path, checkpoint: Checkpoint, destination: Path) -> dict:
    import torch

    source = safe_path(root, checkpoint.path)
    state = file_state(source)
    # mmap keeps optimizer tensors from being paged into memory while exporting.
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    require_unchanged(source, state)
    source_hash = digest(source, checkpoint.proof_sha256 or checkpoint.expected_sha256)
    config = payload["config"]
    if config["seed"] != checkpoint.seed or (
        checkpoint.epoch is not None and payload["epoch"] != checkpoint.epoch
    ):
        raise ValueError(
            f"Checkpoint seed/epoch does not match archived evidence: {checkpoint.path}"
        )
    fields = (
        "model_state",
        "model_signature",
        "data_provenance",
        "training_contract",
        "config",
        "epoch",
        "metrics",
    )
    torch.save({name: payload[name] for name in fields}, destination)
    require_unchanged(source, state)
    return {
        "path": str(checkpoint.path),
        "resolved_config": str(checkpoint.resolved_config),
        "training_seed": checkpoint.seed,
        "epoch": payload["epoch"],
        "source_checkpoint_sha256": source_hash,
        "historical_checkpoint_sha256": checkpoint.expected_sha256,
        "size_bytes": destination.stat().st_size,
    }


def dataset_inventory(root: Path, states: dict) -> tuple[dict[str, str], list[Path]]:
    identity: dict[str, str] = {}
    snapshots: set[Path] = set()
    splits = [(solver, "test") for solver in SOLVERS] + [("hydrostatic", "val")]
    for solver, split in splits:
        base = DATA / solver / split
        manifest = base / "shards_manifest.json"
        shards = read_json(root, manifest, states)["shards"]
        for row in shards:
            # The processed release already records shard digests.
            shard = base / row["file"]
            safe_path(root, shard)
            identity[str(shard)] = row["sha256"]
        manifest_files = {
            manifest,
            base / "eval_manifest.json",
            base.parent / "normalization_stats.json",
        }
        snapshots.update(manifest_files)
        for name in manifest_files | {base / "meta.jsonl"}:
            path = safe_path(root, name)
            states.setdefault(name, file_state(path))
            identity[str(name)] = digest(path)
    return identity, sorted(snapshots)


RELEASE_README = f"""# Frozen common-time benchmark checkpoints

Use this bundle to replay the core manuscript metrics from frozen inference
weights. `{MANIFEST}` maps manuscript tables and figures to evaluation commands,
archived result files, and original checkpoint hashes where available. The
archive includes model states, resolved training settings, evaluation source,
normalization statistics, and derived arrival-time maps. The trajectory
group replays simultaneous coverage and physical relative errors. Its archived
evidence includes the original scenario records, physical risk summary, lineage
audit, and source snapshots from both steps. The archive omits dense
predictions and optimizer state. Source file checksums describe this packaged
checkout. The historical evaluation code identity is recorded separately.

1. Verify the archive against its published SHA-256 before extracting into a
   dedicated directory.
2. Download and verify the [processed benchmark]({DATA_DOI}) using its own
   release manifest and SHA-256 files. Restore `data/processed` under the
   extracted bundle. The required data are the three reference test splits
   and the Hydrostatic validation split. The pooled model normalization
   statistics come with this bundle.
3. From the extracted directory, install the dependencies with
   `python -m pip install -r requirements.txt`.
4. Run `python scripts/build_core_checkpoint_release.py --replay replay-results --device cuda`.
   Use `--device cpu` for CPU inference, or `--group` to replay one metric
   group. The launcher verifies the bundle and benchmark data, then writes new
   metric JSON files. Floating-point results can vary with the hardware and
   software environment.
5. To regenerate the whole-trajectory figure after its metric replay, run
   `python scripts/plot_trajectory_reliability.py --input replay-results/trajectory/trajectory_calibration_physical.json --output trajectory-reliability.pdf`.
6. If this bundle includes `{REFERENCE_RESOLUTION_PREFLIGHT}`, extract the
   separate numerical evidence archive and run
   `python scripts/eval_reference_resolution_sensitivity.py --release-root . --study /path/to/revision-evidence/benchmark_numerical --device cuda --batch-size 2 --output reference-resolution-results.json`.
   This scores fixed predictions from twelve checkpoints on six prescribed
   cases against four Hydrostatic reference fields. It does not retrain the
   models or test native neural-network resolution transfer.

The numerical reference trajectories and the checkpoint metrics characterize
the finite benchmark defined in the manuscript.
"""


def build(
    root: Path,
    output: Path,
    metadata,
    checkpoints,
    files,
    states,
    *,
    evidence=None,
    expected_data=None,
) -> None:
    if os.path.lexists(output):
        raise FileExistsError(output)
    data_files, snapshots = dataset_inventory(root, states)
    for name, expected in (expected_data or {}).items():
        if data_files.get(str(name)) != expected:
            raise ValueError(f"Replay data differ from the trajectory evidence: {name}")
    external = evidence or {}
    external_states = {path: file_state(path) for path, _ in external.values()}
    for name in [*files, *checkpoints]:
        states.setdefault(name, file_state(safe_path(root, name)))
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-release-", dir=output.parent
    ) as scratch:
        temp = Path(scratch)
        tar_path = temp / "bundle.tar"
        export_path = temp / "inference.pt"
        entries: dict[str, str] = {}
        released = []
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for name, expected in sorted(files.items()):
                entries[str(name)] = add_file(
                    archive, safe_path(root, name), name, expected
                )
            for name, (source, expected) in sorted(external.items()):
                if str(name) in entries:
                    raise ValueError(f"Duplicate archived evidence path: {name}")
                entries[str(name)] = add_file(archive, source, name, expected)
            for name in snapshots:
                stored = Path("data_identity") / name
                entries[str(stored)] = add_file(
                    archive, safe_path(root, name), stored, data_files[str(name)]
                )
            for name, checkpoint in sorted(checkpoints.items()):
                row = freeze_checkpoint(root, checkpoint, export_path)
                row["sha256"] = entries[str(name)] = add_file(
                    archive, export_path, name
                )
                released.append(row)
            add_bytes(archive, "RELEASE_README.md", RELEASE_README.encode())
            entries["RELEASE_README.md"] = hashlib.sha256(
                RELEASE_README.encode()
            ).hexdigest()
            metadata = {
                **metadata,
                "checkpoints": released,
                "files": entries,
                "external_processed_data": data_files,
            }
            manifest = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
            add_bytes(archive, MANIFEST, manifest)
            entries[MANIFEST] = hashlib.sha256(manifest).hexdigest()
            sums = "".join(
                f"{checksum}  {name}\n" for name, checksum in sorted(entries.items())
            )
            add_bytes(archive, SUMS, sums.encode())
        for name, state in states.items():
            require_unchanged(safe_path(root, name), state)
        for path, state in external_states.items():
            require_unchanged(path, state)
        # Linking on the same filesystem publishes a completed archive without replacing an existing file.
        os.link(tar_path, output)
    print(f"{digest(output)}  {output}")


def verify(root: Path) -> dict:
    manifest = safe_path(root, MANIFEST)
    checksum_lines = safe_path(root, SUMS).read_text().splitlines()
    calculated = {}
    for line in checksum_lines:
        checksum, name = line.split("  ", 1)
        if name in calculated:
            raise ValueError(f"Duplicate checksum entry: {name}")
        calculated[name] = digest(safe_path(root, name), checksum)
    if MANIFEST not in calculated:
        raise ValueError(f"Missing checksum for {MANIFEST}")
    metadata = json.loads(manifest.read_bytes())
    if metadata["schema_id"] != SCHEMA:
        raise ValueError(f"Unsupported release schema: {metadata['schema_id']}")
    calculated.pop(MANIFEST)
    if calculated != metadata["files"]:
        raise ValueError("The manifest and archive checksum inventory disagree.")
    return metadata


def replay(root: Path, output: Path, device: str, group: str | None) -> None:
    if os.path.lexists(output):
        raise FileExistsError(output)
    metadata = verify(root)
    for name, checksum in sorted(metadata["external_processed_data"].items()):
        digest(safe_path(root, name), checksum)
    output.mkdir()
    for selected in GROUPS if group is None else (group,):
        for step in metadata["recipes"][selected]:
            script = safe_path(root, step["script"])
            args = [
                value.replace("{output}", str(output)) for value in step["arguments"]
            ]
            subprocess.run(
                [sys.executable, str(script), *args, "--device", device],
                cwd=root,
                check=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--build", type=Path, metavar="ARCHIVE.tar")
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--verify", action="store_true")
    actions.add_argument("--replay", type=Path, metavar="OUTPUT_DIR")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument(
        "--trajectory-evidence",
        type=Path,
        metavar="DIR",
        help="Directory containing the frozen trajectory run, physical summary, source files, and lineage audit",
    )
    parser.add_argument(
        "--reference-resolution-preflight",
        type=Path,
        metavar="JSON",
        help="Completed-training checkpoint proof for the fixed-prediction resolution analysis",
    )
    args = parser.parse_args()
    if args.group is not None and args.replay is None:
        parser.error("--group requires --replay")
    if args.trajectory_evidence is not None and (
        args.verify or args.replay is not None
    ):
        parser.error(
            "--trajectory-evidence is only available when planning or building"
        )
    if args.build is not None and args.trajectory_evidence is None:
        parser.error("--build requires --trajectory-evidence DIR")
    if args.reference_resolution_preflight is not None and (
        args.verify or args.replay is not None
    ):
        parser.error(
            "--reference-resolution-preflight is only available when planning or building"
        )
    if args.verify:
        verify(ROOT)
        print(f"Verified files listed in {SUMS}")
        return
    if args.replay is not None:
        replay(ROOT, args.replay.absolute(), args.device, args.group)
        return
    metadata, checkpoints, files, states = plan(ROOT)
    evidence, expected_data = {}, {}
    if args.trajectory_evidence is not None:
        evidence, expected_data = trajectory_evidence(
            args.trajectory_evidence.resolve(), checkpoints
        )
    if args.reference_resolution_preflight is not None:
        proof = args.reference_resolution_preflight.absolute()
        proof_sha, bound, proof_sources = reference_resolution_preflight(
            ROOT, proof, checkpoints
        )
        evidence[REFERENCE_RESOLUTION_PREFLIGHT] = (proof, proof_sha)
        checkpoints.update(bound)
        files.update(proof_sources)
    if args.build is not None:
        build(
            ROOT,
            args.build.absolute(),
            metadata,
            checkpoints,
            files,
            states,
            evidence=evidence,
            expected_data=expected_data,
        )
        return
    for name, expected in expected_data.items():
        digest(safe_path(ROOT, name), expected)
    sizes = {str(name): file_state(safe_path(ROOT, name))[2] for name in checkpoints}
    for name in files:
        file_state(safe_path(ROOT, name))
    print(
        json.dumps(
            {
                "checkpoint_count": len(checkpoints),
                "original_checkpoint_bytes": sum(sizes.values()),
                "archive_size_estimate_bytes": sum(sizes.values()) // 3,
                "archive_size_estimate_basis": "Adam checkpoints contain weights and two moment tensors",
                "included_source_and_evidence_files": len(files) + len(evidence),
                "trajectory_evidence_verified": args.trajectory_evidence is not None,
                "reference_resolution_preflight_verified": REFERENCE_RESOLUTION_PREFLIGHT
                in evidence,
                "replay_groups": list(metadata["recipes"]),
                "checkpoints": sizes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
