#!/usr/bin/env python
"""Compare fixed Hydrostatic surrogate predictions against four frozen targets."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_benchmark_numerical_study as numerical
from scripts.eval_suite_preflight import validate_completed_checkpoint

import numpy as np
import torch
import yaml

from tsunami_surrogate.data.dataset import ShardedTsunamiDataset
from tsunami_surrogate.data_gen.common_time_v2 import hash_array, parse_requested_output_config
from tsunami_surrogate.evaluation.normalization_bridge import load_standardization_spec
from tsunami_surrogate.models import build_model, model_config_signature
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.utils.hashing import sha256_file, stable_json_sha256


SCHEMA = "tsunami-surrogate.reference-resolution-sensitivity.v1"
MODEL_NAMES = {"fno": "fno2d", "ffno": "ffno2d", "ufno": "ufno2d", "convlstm": "convlstm"}
SEEDS = (18, 36, 67)
REFERENCES = ("archived", "coarse", "baseline", "fine")
RESOLUTION = {"coarse": 64, "baseline": 128, "fine": 256}
SOLVER = "swe_hydrostatic"
SHAPE = (50, 64, 64)
INPUT_ORDER = ("bathymetry", "source", "initial_depth")
CASE_INDICES = (1, 4, 5, 13, 28, 34)
ROLLOUT_COUNT = 92
TEST_PATH = Path("data/processed/hydrostatic/test")
TRAIN_PATH = Path("data/processed/hydrostatic/train")
STUDY_CONFIG = Path("configs/eval/benchmark_numerical_study.yaml")
DATASET_CONFIG = Path("configs/data/dataset_test.yaml")
RELEASE_PREFLIGHT = Path("evidence/reference_resolution/checkpoint_preflight.json")


def _record(files: dict[str, str], path: Path, expected: str | None = None) -> str:
    path = path.resolve()
    digest = sha256_file(path)
    prior = files.setdefault(str(path), digest)
    if digest != prior or (expected is not None and digest != expected):
        raise ValueError(f"Unexpected SHA-256 for {path}")
    return digest


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_study(study_root: Path, files: dict[str, str]) -> tuple[dict, dict]:
    plan_path = study_root / "plan.json"
    final_path = study_root / "benchmark_numerical_study.json"
    if not plan_path.is_file() or not final_path.is_file():
        raise ValueError("The full 92-rollout numerical study must be finalized")
    _record(files, plan_path)
    _record(files, final_path)
    plan = _json(plan_path)
    summary = _json(final_path)
    plan_hash = stable_json_sha256(plan)
    if (
        plan.get("schema_id") != numerical.SCHEMA or summary.get("schema_id") != numerical.SCHEMA
        or plan.get("complete_full_scope") is not True
        or summary.get("complete_full_planned_study") is not True
        or summary.get("plan_sha256") != plan_hash
        or plan.get("full_scope_rollout_count") != ROLLOUT_COUNT
        or summary.get("full_scope_rollout_count") != ROLLOUT_COUNT
        or summary.get("rollout_count") != ROLLOUT_COUNT
    ):
        raise ValueError("Expected the finalized, complete, matching 92-rollout study")
    cases = plan["roster"]
    if tuple(case["index"] for case in cases) != CASE_INDICES:
        raise ValueError("Study case roster differs from the prescribed cohort")
    case_by_id = {case["row"]["scenario_id"]: case for case in cases}
    if len(case_by_id) != len(CASE_INDICES):
        raise ValueError("Study roster contains duplicate scenario IDs")

    inputs = plan["code_and_input_files"]
    for relative in (STUDY_CONFIG, DATASET_CONFIG, Path("scripts/run_benchmark_numerical_study.py")):
        records = [row for row in inputs if Path(row["path"]).parts[-len(relative.parts):] == relative.parts]
        if len(records) != 1:
            raise ValueError(f"Expected a unique study plan digest for {relative}")
        _record(files, ROOT / relative, records[0]["sha256"])
    dataset_cfg = yaml.safe_load((ROOT / DATASET_CONFIG).read_text(encoding="utf-8"))
    requested = parse_requested_output_config(dataset_cfg["requested_output"])
    if (
        requested is None or requested.contract_hash != plan["requested_contract_hash"]
        or not np.array_equal(requested.requested_times, plan["requested_timestamps"])
        or summary["requested_timestamps"] != plan["requested_timestamps"]
        or len(requested.requested_times) != SHAPE[0] or requested.requested_times[-1] != 420
    ):
        raise ValueError("Study and accepted test output contracts differ")

    tasks = {task["id"]: task for task in plan["tasks"]}
    if len(tasks) != ROLLOUT_COUNT or len(plan["tasks"]) != ROLLOUT_COUNT or set(tasks) != set(summary["rollouts"]):
        raise ValueError("Numerical study task and summary roster do not match")
    checkpoint_dir = study_root / "checkpoints"
    if {path.stem for path in checkpoint_dir.glob("*.npz")} != set(tasks):
        raise ValueError("Numerical study checkpoint roster does not match the completed plan")
    for task_id in tasks:
        _record(files, checkpoint_dir / f"{task_id}.npz")

    fields = {variant: {} for variant in RESOLUTION}
    for variant, core in RESOLUTION.items():
        for case_id, case in case_by_id.items():
            task_id = f"{case_id}__{SOLVER}__{variant}"
            meta = tasks.get(task_id)
            if (
                meta is None or meta["core"] != core or meta["publication_core"] != 64
                or meta["contract_hash"] != requested.contract_hash
                or meta["variant"] != variant or meta["solver"] != SOLVER
            ):
                raise ValueError(f"Missing or mismatched Hydrostatic comparison field: {task_id}")
            task = {**meta, "dataset": dataset_cfg, "case": case, "times": plan["requested_timestamps"]}
            eta, rollout = numerical._read_checkpoint(checkpoint_dir / f"{task_id}.npz", task, plan_hash)
            if rollout != summary["rollouts"][task_id]:
                raise ValueError(f"Final summary and checkpoint results differ: {task_id}")
            fields[variant][case_id] = eta
    return plan, fields


def _physical(normalized: np.ndarray, offset_scale: tuple[float, float]) -> np.ndarray:
    offset, scale = offset_scale
    if not np.isfinite([offset, scale]).all() or scale <= 0:
        raise ValueError("Normalization offset and scale must be finite with positive scale")
    return np.asarray(normalized, dtype=np.float64) * scale + offset


def _test_inputs(
    dataset_path: Path, plan: dict, files: dict[str, str], release: dict | None,
) -> tuple[torch.Tensor, dict, str, tuple[float, float], list]:
    manifest_path = dataset_path / "eval_manifest.json"
    meta_path = dataset_path / "meta.jsonl"
    shard_manifest_path = dataset_path / "shards_manifest.json"
    stats_path = dataset_path.parent / "normalization_stats.json"

    def record(path: Path, release_name: Path) -> str:
        expected = None if release is None else release["external_processed_data"][release_name.as_posix()]
        return _record(files, path, expected)

    for path in (manifest_path, meta_path, shard_manifest_path):
        record(path, TEST_PATH / path.name)
    stats_hash = record(stats_path, TEST_PATH.parent / stats_path.name)
    manifest = _json(manifest_path)
    provenance = manifest["provenance"]
    lineage = provenance["source_lineage"]
    if (
        manifest["split"] != "test" or manifest["target_variable"] != "eta"
        or manifest["target_mode"] != "multi_step" or manifest["normalized_targets"] is not True
        or manifest["num_samples"] != 2500 or tuple(manifest["input_order"]) != INPUT_ORDER
        or lineage["contract_hash"] != plan["requested_contract_hash"]
        or lineage["solver_names"] != ["hydrostatic"] or provenance["normalization"]["sha256"] != stats_hash
    ):
        raise ValueError("Expected normalized common-time-v2 Hydrostatic test inputs and targets")
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    indices = {row["scenario_id"]: index for index, row in enumerate(rows)}
    dataset = ShardedTsunamiDataset(dataset_path)
    if len(indices) != len(rows) or len(rows) != len(dataset) or len(dataset) != manifest["num_samples"]:
        raise ValueError("Processed test metadata must give a unique scenario-to-row mapping")

    stats = load_standardization_spec(stats_path)
    if stats.target_variable != "eta":
        raise ValueError("Training normalization must describe elevation")
    target_stats = stats.target
    seen_shards = set()
    xs = []
    targets = {}
    archived_field_checks = []
    for case in plan["roster"]:
        identity = case["row"]
        scenario = identity["scenario_id"]
        index = indices[scenario]
        row = rows[index]
        comparison_metadata = {
            "scenario_id": scenario,
            "bathymetry_type": identity["bathymetry_type"],
            "source_type": identity["source_type"],
            "solver_name": SOLVER,
        }
        input_lineage = identity["input_lineage"]
        lineage_keys = ("lineage_id", "solver_bathymetry_sha256", "solver_source_sha256")
        if (
            any(row.get(key) != value for key, value in comparison_metadata.items())
            or row["qualified_id"] != f"test:{scenario}"
            or row["sample_index"] != identity["sample_index"] or row["split"] != "test"
            or row["contract_hash"] != plan["requested_contract_hash"]
            or tuple(row["trajectory_eta_shape"]) != SHAPE
            or row["timestamps_shape"] != [SHAPE[0]]
            or row["source_strength"] != identity["source_strength"]
            or any(row["input_lineage"].get(key) != input_lineage[key] for key in lineage_keys)
        ):
            raise ValueError(f"Processed metadata differs from the frozen scenario: {scenario}")

        shard = dataset.shards[dataset.shard_index_for_sample(index)]
        shard_file = Path(shard["file"])
        if shard_file not in seen_shards:
            shard_path = dataset_path / shard_file
            if record(shard_path, TEST_PATH / shard_file) != shard["sha256"]:
                raise ValueError(f"Processed shard disagrees with the manifest: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as payload:
                observed_stats = (payload["target_mean"].item(), payload["target_std"].item())
                if (
                    observed_stats != tuple(float(np.float32(value)) for value in target_stats)
                    or tuple(payload["input_order"].tolist()) != INPUT_ORDER
                    or payload["target_variable"].item() != "eta"
                ):
                    raise ValueError(f"Processed shard disagrees with normalization or channel order: {shard_path}")
            seen_shards.add(shard_file)

        sample = dataset[index]
        if (
            any(sample.get(key) != value for key, value in comparison_metadata.items())
            or sample["sample_id"] != f"sample_{identity['sample_index']:06d}"
            or sample["source_strength"] != float(np.float32(identity["source_strength"]))
            or tuple(sample["x"].shape) != (len(INPUT_ORDER), *SHAPE[-2:])
            or tuple(sample["y"].shape) != SHAPE
            or not torch.isfinite(sample["x"]).all() or not torch.isfinite(sample["y"]).all()
        ):
            raise ValueError(f"Processed shard row does not match the frozen scenario: {scenario}")
        xs.append(sample["x"])
        physical = _physical(sample["y"].numpy(), target_stats)
        targets[scenario] = physical

        # The local repository also has the raw archive, which supplies the full
        # black-box output vector and frame timestamps for a direct check.
        if release is None:
            publication = case["publications"][SOLVER]["sample"]
            raw_path = Path(publication["path"])
            _record(files, raw_path, publication["sha256"])
            with np.load(raw_path, allow_pickle=False) as payload:
                raw = payload["trajectory_eta"]
                if not np.array_equal(payload["timestamps"], plan["requested_timestamps"]) or raw.shape != SHAPE:
                    raise ValueError(f"Archived output shape or timestamps differ: {scenario}")
            difference = float(np.max(np.abs(raw - physical)))
            if not np.allclose(raw, physical, rtol=2e-7, atol=1e-8):
                raise ValueError(f"Processed physical target differs from frozen publication: {scenario}")
            archived_field_checks.append({"scenario_id": scenario, "max_abs_difference": difference})
    return torch.stack(xs), targets, stats_hash, target_stats, archived_field_checks


def _read_preflight(path: Path) -> dict:
    report = _json(path)
    if report.get("status") != "passed":
        raise ValueError(f"Training preflight did not pass: {path}")
    rows = report["checkpoints"]
    indexed = {row["checkpoint"]: row for row in rows}
    expected = {f"experiments/{label}/seed_{seed}/best.pt" for label in MODEL_NAMES for seed in SEEDS}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Completion proof must cover the twelve prescribed checkpoints exactly")
    for row in rows:
        eligible = {
            "horizon": row["configured_epochs"] > 0 and row["last_epoch"] >= row["configured_epochs"],
            "early_stopping": row["patience"] > 0 and row["early_count"] >= row["patience"],
            "manual_resource_stop": row.get("manual_completion") is not None,
        }
        if not eligible.get(row.get("completion"), False) or not 1 <= row["best_epoch"] <= row["last_epoch"]:
            raise ValueError(f"Completion proof contains an incomplete training run: {row['checkpoint']}")
    return indexed


def _release_preflights(root: Path, files: dict[str, str]) -> tuple[dict, dict]:
    from scripts.build_core_checkpoint_release import MANIFEST, SUMS, verify

    _record(files, root / MANIFEST)
    _record(files, root / SUMS)
    release = verify(root)
    expected = release["files"].get(RELEASE_PREFLIGHT.as_posix())
    if expected is None:
        raise ValueError(f"Core manifest does not bind the required evidence: {RELEASE_PREFLIGHT}")
    _record(files, root / RELEASE_PREFLIGHT, expected)
    return release, _read_preflight(root / RELEASE_PREFLIGHT)


def _checkpoint(
    root: Path, label: str, seed: int, stats_hash: str, files: dict[str, str],
    release: dict | None, completion: dict,
) -> tuple[dict, dict]:
    relative = Path(f"experiments/{label}/seed_{seed}/best.pt")
    path = root / relative
    config_path = path.with_name("config_resolved.yaml")
    config_relative = config_path.relative_to(root).as_posix()
    if release is None:
        last = path.parent / "checkpoints/last.pt"
        _record(files, path.parent / "history.json")
        manual_completion = path.parent / "manual_completion.json"
        if manual_completion.is_file():
            _record(files, manual_completion)
        if completion:
            preflight = completion[relative.as_posix()]
        else:
            _record(files, path)
            _record(files, last)
            preflight = validate_completed_checkpoint(path, expected_seed=seed)
        expected_digest = preflight["best_checkpoint_sha256"]
        _record(files, last, preflight["last_checkpoint_sha256"])
        history = _json(path.parent / "history.json")
        if [entry["epoch"] for entry in history] != list(range(1, preflight["last_epoch"] + 1)):
            raise ValueError(f"Training history does not match completion evidence: {relative}")
    else:
        matches = [row for row in release["checkpoints"] if row["path"] == relative.as_posix()]
        if len(matches) != 1:
            raise ValueError(f"Release lacks a unique checkpoint entry: {relative}")
        entry = matches[0]
        preflight = completion.get(relative.as_posix())
        if preflight is None:
            raise ValueError(f"Release lacks completion evidence for {relative} at {RELEASE_PREFLIGHT}")
        if (
            entry["source_checkpoint_sha256"] != preflight["best_checkpoint_sha256"]
            or entry["epoch"] != preflight["best_epoch"] or entry["training_seed"] != seed
            or entry["resolved_config"] != config_relative
            or entry["sha256"] != release["files"][relative.as_posix()]
        ):
            raise ValueError(f"Release checkpoint completion metadata differs: {relative}")
        expected_digest = entry["sha256"]

    _record(files, path, expected_digest)
    _record(files, config_path, None if release is None else release["files"][config_relative])
    saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    config = saved["config"]
    artifacts = preflight["training_data_artifacts"]
    saved_training = saved["data_provenance"]["datasets"]["train_path"]
    train_path = Path(config["data"]["train_path"])
    if train_path.name == "eval_dataset.npz":
        train_path = train_path.parent
    train_root = root / train_path
    if (
        seed != preflight["seed"] or seed != config["seed"]
        or saved["epoch"] != preflight["best_epoch"]
        or config["model"]["name"] != MODEL_NAMES[label]
        or config["model"].get("in_channels") != len(INPUT_ORDER)
        or config["model"].get("out_channels") != SHAPE[0]
        or train_path != TRAIN_PATH or saved_training["configured_path"] != config["data"]["train_path"]
        or saved_training["artifacts"] != artifacts
        or artifacts.get("normalization_stats.json") != stats_hash
        or saved["model_signature"] != model_config_signature(config)
        or preflight["training_data_identity_strength"] != "content_bound"
    ):
        raise ValueError(f"Checkpoint configuration, completed training or normalization differs: {relative}")

    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if resolved_config["seed"] != seed or model_config_signature(resolved_config) != saved["model_signature"]:
        raise ValueError(f"Saved model differs from resolved training configuration: {relative}")
    if release is None:
        for name, fingerprint in artifacts.items():
            candidate = train_root / name
            if name == "normalization_stats.json" and not candidate.is_file():
                candidate = train_root.parent / name
            _record(files, candidate, fingerprint)
    return saved, preflight


@torch.inference_mode()
def _predict(model, inputs: torch.Tensor, batch_size: int, device: torch.device,
             offset_scale: tuple[float, float]) -> np.ndarray:
    prediction = torch.cat([model(batch.to(device)).detach().cpu() for batch in inputs.split(batch_size)])
    return _physical(prediction.numpy(), offset_scale)


def _scores(prediction: np.ndarray, targets: dict[str, np.ndarray]) -> dict:
    if set(targets) != set(REFERENCES) or prediction.ndim != 4:
        raise ValueError("Expected aligned [case, time, x, y] predictions and four target fields")
    if not np.isfinite(prediction).all():
        raise ValueError("Model predictions must be finite")
    rows = {}
    for reference in REFERENCES:
        target = targets[reference]
        if prediction.shape != target.shape or not np.isfinite(target).all():
            raise ValueError(f"Invalid reference shape or values: {reference}")
        norm = np.linalg.norm(target.reshape(len(target), -1), axis=1)
        if np.any(norm == 0):
            raise ValueError(f"Physical reference norm is zero: {reference}")
        squared = np.sum((prediction - target)**2, axis=(1, 2, 3))
        rows[reference] = {
            "rmse_by_case": np.sqrt(squared / np.prod(target.shape[1:])).tolist(),
            "relative_l2_by_case": (np.sqrt(squared) / norm).tolist(),
            "global_rmse": float(np.sqrt(squared.sum() / target.size)),
            "mean_per_case_relative_l2": float(np.mean(np.sqrt(squared) / norm)),
        }
    return rows


def _alignment(prediction: np.ndarray, baseline: np.ndarray, fine: np.ndarray) -> dict:
    error = prediction - baseline
    delta = baseline - fine
    axes = tuple(range(1, prediction.ndim))
    error_squared = np.mean(error**2, axis=axes)
    delta_squared = np.mean(delta**2, axis=axes)
    cross = 2 * np.mean(error * delta, axis=axes)
    fine_squared = np.mean((prediction - fine)**2, axis=axes)
    residual = fine_squared - (error_squared + delta_squared + cross)
    scale = np.maximum.reduce((error_squared, delta_squared, abs(cross), fine_squared))
    if np.any(abs(residual) > 32 * np.finfo(np.float64).eps * scale):
        raise RuntimeError("The aligned MSE decomposition identity failed")
    denominator = 2 * np.sqrt(error_squared * delta_squared)
    cosine = [float(term / norm) if norm else None for term, norm in zip(cross, denominator)]
    terms = {
        "baseline_mse": error_squared,
        "reference_shift_mse": delta_squared,
        "twice_error_shift_inner_product_per_cell": cross,
        "fine_mse": fine_squared,
        "identity_residual": residual,
    }
    return {
        "by_case": {name: value.tolist() for name, value in terms.items()} | {"cosine_alignment": cosine},
        "global": {name: float(np.mean(value)) for name, value in terms.items()},
    }


def _pairwise_changes(base: dict[str, float], candidate: dict[str, float]) -> dict:
    def sign(a: float, b: float) -> int:
        return int(a > b) - int(a < b)

    changes = []
    ties = []
    for left, right in combinations(MODEL_NAMES, 2):
        a = sign(base[left], base[right])
        b = sign(candidate[left], candidate[right])
        if a != b and a * b != 0:
            changes.append([left, right])
        elif a != b:
            ties.append([left, right])
    return {"strict_pairwise_reversals": changes, "tie_status_changes": ties}


def _rankings(members: list[dict], scenario_ids: list[str]) -> dict:
    by_seed = {}
    for seed in SEEDS:
        selected = {row["model"]: row["scores"] for row in members if row["seed"] == seed}
        by_seed[str(seed)] = _rank_rows(selected, scenario_ids)
    seed_mean = {}
    for model in MODEL_NAMES:
        selected = [row["scores"] for row in members if row["model"] == model]
        seed_mean[model] = {
            reference: {
                key: np.mean([row[reference][key] for row in selected], axis=0).tolist()
                for key in selected[0][reference]
            }
            for reference in REFERENCES
        }
    return {"by_seed": by_seed, "model_means_across_seeds": seed_mean,
            "mean_across_seeds": _rank_rows(seed_mean, scenario_ids)}


def _rank_rows(scores: dict, scenario_ids: list[str]) -> dict:
    def ranks(base_values: dict, values: dict) -> dict:
        return {
            "order_lowest_first": sorted(values, key=lambda model: (values[model], model)),
            "ranks": {model: 1 + sum(other < value for other in values.values())
                      for model, value in values.items()},
            **_pairwise_changes(base_values, values),
        }

    by_reference = {}
    metric = "mean_per_case_relative_l2"
    for reference in REFERENCES:
        row = {}
        for aggregate_key in (metric, "global_rmse"):
            base_values = {model: values["baseline"][aggregate_key] for model, values in scores.items()}
            values = {model: result[reference][aggregate_key] for model, result in scores.items()}
            row[aggregate_key] = ranks(base_values, values)
        row["by_case"] = {}
        for index, scenario in enumerate(scenario_ids):
            key = "relative_l2_by_case"
            base_values = {model: values["baseline"][key][index] for model, values in scores.items()}
            values = {model: result[reference][key][index] for model, result in scores.items()}
            row["by_case"][scenario] = ranks(base_values, values)
        by_reference[reference] = row
    return by_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, help="Processed Hydrostatic test directory")
    parser.add_argument("--release-root", type=Path, help="Extracted verified core checkpoint bundle")
    parser.add_argument("--checkpoint-preflight", type=Path, help="Reuse a local completed-checkpoint preflight")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= len(CASE_INDICES):
        parser.error(f"--batch-size must be between 1 and {len(CASE_INDICES)}")
    if args.release_root is not None and args.checkpoint_preflight is not None:
        parser.error("Release mode uses the completion proof bound to the core manifest")
    output = args.output.resolve()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    files = {}
    plan, numerical_fields = _load_study(args.study.resolve(), files)
    execution_root = args.release_root.resolve() if args.release_root is not None else ROOT
    dataset = (args.test_dataset or execution_root / TEST_PATH).resolve()
    preflight_path = args.checkpoint_preflight.resolve() if args.checkpoint_preflight else None
    os.chdir(ROOT)
    release, completion = (None, {}) if args.release_root is None else _release_preflights(execution_root, files)
    if preflight_path is not None:
        _record(files, preflight_path)
        completion = _read_preflight(preflight_path)
        preflight_source = ROOT / "scripts/eval_suite_preflight.py"
        source_digests = _json(preflight_path)["source_files"]
        source_digest = source_digests.get(str(preflight_source))
        if source_digest is None:
            raise ValueError("Local completion proof lacks the checkpoint-preflight source digest")
        _record(files, preflight_source, source_digest)
    sources = [
        Path(__file__).resolve(), ROOT / "scripts/eval_suite_preflight.py",
        ROOT / "tsunami_surrogate/__init__.py", ROOT / "src/training/checkpointing.py",
        ROOT / "src/evaluation/normalization_bridge.py", ROOT / "src/data/dataset.py",
        ROOT / "src/data_gen/common_time_v2.py", ROOT / "src/utils/hashing.py",
        ROOT / "src/utils/device.py",
    ]
    sources.extend(sorted((ROOT / "src/models").rglob("*.py")))
    if args.release_root is not None:
        sources.append(ROOT / "scripts/build_core_checkpoint_release.py")
    for source in sources:
        _record(files, source)
    inputs, archived, stats_hash, target_stats, archival_checks = _test_inputs(dataset, plan, files, release)
    scenarios = [case["row"]["scenario_id"] for case in plan["roster"]]
    targets = {name: np.stack([fields[scenario] for scenario in scenarios])
               for name, fields in {"archived": archived, **numerical_fields}.items()}

    torch.set_num_threads(1)
    device = resolve_device(args.device)
    members = []
    proofs = []
    for label in MODEL_NAMES:
        for seed in SEEDS:
            saved, preflight = _checkpoint(execution_root, label, seed, stats_hash, files, release, completion)
            model = build_model(saved["config"])
            model.load_state_dict(saved["model_state"], strict=True)
            model.to(device).eval()
            del saved
            prediction = _predict(model, inputs, args.batch_size, device, target_stats)
            scores = _scores(prediction, targets)
            members.append({
                "model": label, "seed": seed, "prediction": hash_array(prediction), "scores": scores,
                "fine_minus_baseline": {
                    key: (np.asarray(scores["fine"][key]) - np.asarray(scores["baseline"][key])).tolist()
                    for key in scores["baseline"]
                },
                "error_alignment": _alignment(prediction, targets["baseline"], targets["fine"]),
            })
            proofs.append(preflight)
            del model, prediction
            print(f"Scored {label} seed {seed} on {len(scenarios)} cases", flush=True)

    for path, digest in files.items():
        _record(files, Path(path), digest)
    result = {
        "schema_id": SCHEMA,
        "scope": {
            "scenarios": [{"scenario_id": case["row"]["scenario_id"], "sample_index": case["index"],
                           "bathymetry": case["bathymetry"], "source": case["source"]}
                          for case in plan["roster"]],
            "requested_timestamps": plan["requested_timestamps"],
            "study_plan_sha256": stable_json_sha256(plan), "study_rollout_count": ROLLOUT_COUNT,
            "case_count": len(scenarios), "models": list(MODEL_NAMES), "seeds": list(SEEDS),
            "reference_core_grid_cells": {"archived": 128, **RESOLUTION}, "publication_grid_cells": 64,
        },
        "runtime": {"device": str(device), "batch_size": args.batch_size, "torch": str(torch.__version__),
                    "numpy": np.__version__, "python": platform.python_version(),
                    "release_checkpoint_mode": release is not None},
        "target_affine_transform": {"offset": target_stats[0], "scale": target_stats[1],
                                    "training_normalization_sha256": stats_hash},
        "input_tensor": hash_array(inputs.numpy()),
        "targets": {name: hash_array(values) for name, values in targets.items()},
        "archived_to_baseline_global_rmse": float(np.sqrt(np.mean((targets["archived"] - targets["baseline"])**2))),
        "baseline_to_fine_global_rmse": float(np.sqrt(np.mean((targets["baseline"] - targets["fine"])**2))),
        "archive_processed_target_checks": archival_checks,
        "source_sha256": files,
        "checkpoint_preflight": {"status": "passed", "checkpoints": proofs},
        "members": members,
        "rankings": _rankings(members, scenarios),
        "interpretation": (
            "Each checkpoint produces one physical prediction per case, reused for all reference comparisons. "
            "RMSE pools cells and times across the six cases. Relative L2 divides each case by its own "
            "physical reference norm before averaging. Pairwise rankings compare the same cases and seeds, "
            "using the new float64 baseline as the comparison point. Affine scaling is bound to the "
            "training normalization. Archived processed fields retain archival rounding. The nested "
            "reference changes computational resolution and natural time steps for a fixed raster input. "
            "Observed rank changes describe these checkpoints and six prescribed cases only."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    numerical._write_json(output, result)


if __name__ == "__main__":
    main()
