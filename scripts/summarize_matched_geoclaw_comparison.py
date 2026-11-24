#!/usr/bin/env python
"""Compare new in-house and GeoClaw rollouts on paired frozen grid levels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_benchmark_numerical_study as numerical

import numpy as np
import yaml

from tsunami_surrogate.data_gen.common_time_v2 import hash_array, parse_requested_output_config, stable_hash_payload
from tsunami_surrogate.data_gen.simulate_dataset import BufferedDomainConfig, _prepare_buffered_domain
from tsunami_surrogate.evaluation.common_time_v2_level_a import validate_checksums
from tsunami_surrogate.evaluation.established_solver_validation import (
    CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID,
    _load_external_result,
    _load_external_run_manifest,
    _validate_external_checksums,
)
from tsunami_surrogate.evaluation.geoclaw_adapter import _adapter_hash
from tsunami_surrogate.utils.hashing import sha256_file, stable_json_sha256


SCHEMA = "tsunami-surrogate.matched-geoclaw-resolution-comparison.v1"
CASE_INDICES = (1, 4, 34)
SOLVERS = ("swe_hydrostatic", "swe_muscl_hr")
CORE_GRIDS = {"baseline": 128, "fine": 256}
GEO_GRIDS = (192, 384)
SHAPE = (50, 64, 64)
FIELDS = ("bathymetry", "eta0", "initial_depth", "hu0", "hv0")
DATASET_CONFIG = Path("configs/data/dataset_test.yaml")
STUDY_CONFIG = Path("configs/eval/benchmark_numerical_study.yaml")
STUDY_COUNT = 92


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bind(files: dict[str, str], path: Path, expected: str | None = None) -> str:
    path = path.resolve()
    digest = sha256_file(path)
    retained = files.setdefault(str(path), digest)
    if digest != retained or (expected is not None and digest != expected):
        raise ValueError(f"SHA-256 mismatch: {path}")
    return digest


def _study(root: Path, files: dict[str, str]) -> tuple[dict, dict, dict, dict]:
    final_path = root / "benchmark_numerical_study.json"
    plan_path = root / "plan.json"
    if not final_path.is_file() or not plan_path.is_file():
        raise ValueError("The full 92-rollout numerical study must be finalized")
    _bind(files, final_path)
    _bind(files, plan_path)
    final = _json(final_path)
    plan = _json(plan_path)
    digest = stable_json_sha256(plan)
    if (
        plan.get("schema_id") != numerical.SCHEMA or final.get("schema_id") != numerical.SCHEMA
        or plan.get("complete_full_scope") is not True or final.get("complete_full_planned_study") is not True
        or plan.get("full_scope_rollout_count") != STUDY_COUNT
        or final.get("full_scope_rollout_count") != STUDY_COUNT
        or final.get("rollout_count") != STUDY_COUNT or final.get("plan_sha256") != digest
    ):
        raise ValueError("Expected a complete matching 92-rollout numerical study")

    original_repo = Path(plan["code_and_input_files"][0]["path"]).parents[2]
    sources = {
        Path(row["path"]).relative_to(original_repo): row["sha256"]
        for row in plan["code_and_input_files"] if Path(row["path"]).is_relative_to(original_repo)
    }
    required = {DATASET_CONFIG, STUDY_CONFIG, Path("scripts/run_benchmark_numerical_study.py")}
    if not required <= set(sources):
        raise ValueError("Numerical plan lacks execution code or dataset settings")
    for path, digest_expected in sources.items():
        _bind(files, ROOT / path, digest_expected)
    dataset = yaml.safe_load((ROOT / DATASET_CONFIG).read_text(encoding="utf-8"))
    requested = parse_requested_output_config(dataset["requested_output"])
    if (
        requested is None or requested.contract_hash != plan["requested_contract_hash"]
        or not np.array_equal(requested.requested_times, plan["requested_timestamps"])
        or len(requested.requested_times) != SHAPE[0] or requested.requested_times[-1] != 420
    ):
        raise ValueError("Numerical outputs differ from the accepted test time contract")

    by_scenario = {row["row"]["scenario_id"]: row for row in plan["roster"]}
    if len(by_scenario) != 6 or len(plan["roster"]) != 6:
        raise ValueError("Expected six unique numerical scenarios")
    tasks = {row["id"]: row for row in plan["tasks"]}
    if len(tasks) != STUDY_COUNT or len(plan["tasks"]) != STUDY_COUNT:
        raise ValueError("Numerical task roster is incomplete or contains repeats")
    directory = root / "checkpoints"
    expected_names = {f"{name}.npz" for name in tasks}
    if {path.name for path in directory.iterdir()} != expected_names:
        raise ValueError("Numerical checkpoint inventory does not match the completed plan")
    entries = {}
    for task_id, metadata in tasks.items():
        case_id = task_id.split("__", 1)[0]
        case = None if metadata["variant"] == "moving_equilibrium" else by_scenario[case_id]
        task = {**metadata, "dataset": dataset, "case": case, "times": plan["requested_timestamps"]}
        path = directory / f"{task_id}.npz"
        _bind(files, path)
        entries[task_id] = numerical._read_checkpoint(path, task, digest)
    if numerical._summary(plan, entries, digest) != final:
        raise ValueError("Numerical summary differs from the verified checkpoints")
    return plan, dataset, tasks, entries


def _external(
    bundle: Path, results: Path, grid: int, summary_digests: set[str], files: dict[str, str],
) -> tuple[dict, dict, dict]:
    bundle_json = bundle / "frozen_contract.json"
    _bind(files, bundle_json)
    frozen = _json(bundle_json)
    expected_kind = ("corrected-production-geoclaw-swe-frozen-contract" if grid == 192
                     else "nested-grid-geoclaw-swe-comparison")
    if (
        frozen.get("schema_id") != CORRECTED_PRODUCTION_GEOCLAW_SCHEMA_ID
        or frozen.get("artifact_kind") != expected_kind
    ):
        raise ValueError(f"Unexpected GeoClaw {grid} bundle schema or role")
    bundle_kind = "corrected-production-geoclaw-bundle" if grid == 192 else "nested-geoclaw-bundle"
    preimage = {key: value for key, value in frozen.items() if key != "bundle_hash"}
    digest = stable_hash_payload(artifact_kind=bundle_kind, payload=preimage, schema_id=frozen["schema_id"])
    if digest != frozen["bundle_hash"]:
        raise ValueError(f"GeoClaw {grid} bundle fingerprint does not match its content")

    for directory in (bundle, results):
        index = directory / "SHA256SUMS.txt"
        _bind(files, index)
        for line in index.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid frozen artifact path: {name}")
            _bind(files, directory / relative, expected)
    validate_checksums(bundle)
    _validate_external_checksums(results, frozen)
    manifest = _load_external_run_manifest(results, frozen)
    if manifest["adapter_hash"] != _adapter_hash(
        execution=manifest["execution"], revisions=manifest["revisions"], bundle_schema_id=frozen["schema_id"],
    ):
        raise ValueError("Current GeoClaw adapter source does not match the recorded run")
    transport_manifest = _json(bundle / "external_results_manifest.json")
    if (
        transport_manifest["bundle_hash"] != digest
        or transport_manifest["results"] != frozen["external_results"]
        or len(frozen["cases"]) != len(CASE_INDICES)
        or len(frozen["external_results"]) != len(CASE_INDICES)
    ):
        raise ValueError(f"Unexpected GeoClaw {grid} case or result manifest")
    bound = [bundle_json, bundle / "SHA256SUMS.txt", results / "SHA256SUMS.txt", results / "RUN_MANIFEST.json"]
    bound.extend(results / row["relative_path"] for row in frozen["external_results"])
    if any(files[str(path.resolve())] not in summary_digests for path in bound):
        raise ValueError(f"GeoClaw {grid} artifacts do not match the completed resolution summary")
    times = np.asarray(frozen["requested_times"])
    fields = {
        row["case_id"]: _load_external_result(results / row["relative_path"], row, times, manifest)[0]
        for row in frozen["external_results"]
    }
    if len(fields) != len(CASE_INDICES):
        raise ValueError("GeoClaw result IDs are repeated")
    return frozen, manifest, fields


def _initial_arrays(case: dict, dataset: dict, input_root: Path | None, files: dict[str, str]) -> dict:
    local = {**case, "inputs": {}}
    for kind, record in case["inputs"].items():
        original = Path(record["path"])
        path = original if input_root is None else input_root / kind / original.name
        local["inputs"][kind] = {"path": str(path.resolve()), "sha256": _bind(files, path, record["sha256"])}
    bathymetry, source, strength = numerical._load_arrays(local, CORE_GRIDS["baseline"], lake_rest=False)
    initialized = _prepare_buffered_domain(
        bathymetry, source, strength, dataset["dataset"]["sea_level_offset"],
        BufferedDomainConfig(**dataset["computational_domain"]), source_already_tapered=True,
    )
    depth = initialized["solver_h0"]
    return {
        "bathymetry": initialized["solver_bathymetry"], "eta0": initialized["solver_eta0"],
        "initial_depth": depth, "hu0": np.zeros_like(depth), "hv0": np.zeros_like(depth),
    }


def _verify_arrays(
    baseline: dict, parent: dict, fine: dict, times: np.ndarray, base_case: dict, fine_case: dict,
) -> None:
    for key in FIELDS:
        if not np.array_equal(parent[key], baseline[key]):
            raise ValueError(f"Frozen GeoClaw 192 initial field differs from the in-house baseline: {key}")
        if not np.array_equal(fine[key], numerical._resample(baseline[key], 128, 256)):
            raise ValueError(f"Frozen GeoClaw 384 initial field differs from the in-house refinement: {key}")
        if fine_case["refined_initial_arrays"][key] != hash_array(fine[key]):
            raise ValueError(f"GeoClaw 384 initial field is not bound to its case record: {key}")
    for arrays, case, shape in ((parent, base_case, 192), (fine, fine_case, 384)):
        domain = case["external_domain"]
        width = shape // 6
        crop = [width, shape - width, width, shape - width]
        if (
            domain["shape"] != [shape, shape] or arrays["bathymetry"].shape != (shape, shape)
            or domain["output_crop"] != crop or not np.array_equal(arrays["output_crop"], crop)
            or domain["bounds"] != [0.0, 3600.0, 0.0, 3600.0]
            or not np.array_equal(arrays["domain_bounds"], domain["bounds"])
            or domain["publication_reduction"] != [shape // 96] * 2
            or domain["boundary"] != "open_extrapolation"
            or case["publication_shape"] != list(SHAPE[-2:])
            or arrays["case_hash"].item() != case["case_hash"]
            or not np.array_equal(arrays["requested_times"], times)
        ):
            raise ValueError(f"GeoClaw {shape} physical domain, crop or output time mismatch")


def _metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if left.shape != SHAPE or right.shape != SHAPE or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Expected finite paired 50×64×64 fields")
    diff = left.astype(np.float64) - right
    norm = np.linalg.norm(right)
    if not norm:
        raise ValueError("GeoClaw reference trajectory has zero norm")
    return float(np.sqrt(np.mean(diff**2))), float(np.linalg.norm(diff) / norm)


def _compare_cases(
    scenarios: dict, case_maps: dict, entries: dict, geo_fields: dict,
) -> tuple[list[dict], list[dict]]:
    rows = []
    paired_changes = []
    contrasts = (
        ("inhouse_refinement_at_geoclaw_192", ("baseline", 192), ("fine", 192)),
        ("inhouse_refinement_at_geoclaw_384", ("baseline", 384), ("fine", 384)),
        ("geoclaw_refinement_at_inhouse_192", ("baseline", 192), ("baseline", 384)),
        ("geoclaw_refinement_at_inhouse_384", ("fine", 192), ("fine", 384)),
        ("both_refined", ("baseline", 192), ("fine", 384)),
    )
    for solver in SOLVERS:
        changes = {label: [] for label, _, _ in contrasts}
        for index in CASE_INDICES:
            scenario_id = scenarios[index]["row"]["scenario_id"]
            case_id = case_maps[192][index]["case_id"]
            metrics = {}
            for variant, core in CORE_GRIDS.items():
                field, _ = entries[f"{scenario_id}__{solver}__{variant}"]
                for grid in GEO_GRIDS:
                    rmse, relative = _metrics(field, geo_fields[grid][case_id])
                    metrics[variant, grid] = (rmse, relative)
                    rows.append({
                        "case_id": case_id, "scenario_id": scenario_id, "inhouse_solver": solver,
                        "inhouse_variant": variant, "inhouse_core_cells": core,
                        "inhouse_computational_cells": core * 3 // 2, "geoclaw_computational_cells": grid,
                        "rmse": rmse, "relative_l2_geoclaw_denominator": relative,
                    })
            for label, before, after in contrasts:
                initial, initial_relative = metrics[before]
                final, final_relative = metrics[after]
                changes[label].append({
                    "case_id": case_id, "before_rmse": initial, "after_rmse": final,
                    "rmse_delta": final - initial,
                    "rmse_fraction_change": (final - initial) / initial if initial else None,
                    "relative_l2_delta": final_relative - initial_relative,
                })
        for label, cases in changes.items():
            delta = [row["rmse_delta"] for row in cases]
            paired_changes.append({
                "solver": solver, "contrast": label, "by_case": cases,
                "median_rmse_delta": float(np.median(delta)), "min_rmse_delta": min(delta),
                "max_rmse_delta": max(delta), "case_count_decreased_rmse": sum(value < 0 for value in delta),
                "case_count": len(cases),
            })
    return rows, paired_changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--geoclaw-summary", type=Path, required=True)
    parser.add_argument("--coarse-bundle", type=Path, required=True)
    parser.add_argument("--coarse-external", type=Path, required=True)
    parser.add_argument("--fine-bundle", type=Path, required=True)
    parser.add_argument("--fine-external", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Optional relocated root containing bathymetry/ and sources/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    files = {}
    sources = [
        Path(__file__).resolve(), ROOT / "tsunami_surrogate/__init__.py",
        ROOT / "scripts/summarize_geoclaw_resolution.py",
        ROOT / "src/evaluation/common_time_v2_level_a.py",
        ROOT / "src/evaluation/established_solver_validation.py",
        ROOT / "src/evaluation/geoclaw_adapter.py",
    ]
    for path in sources:
        _bind(files, path)
    _bind(files, args.geoclaw_summary)
    summary = _json(args.geoclaw_summary)
    _bind(files, ROOT / "scripts/summarize_geoclaw_resolution.py", summary["script_sha256"])
    if summary["coarse_shape"] != [192, 192] or summary["fine_shape"] != [384, 384]:
        raise ValueError("Expected the 192/384 GeoClaw resolution summary")
    plan, dataset, tasks, entries = _study(args.study, files)
    times = np.asarray(plan["requested_timestamps"])
    geo_paths = {192: (args.coarse_bundle, args.coarse_external), 384: (args.fine_bundle, args.fine_external)}
    bundles = {}
    geo_fields = {}
    manifests = {}
    digests = set(summary["source_files"].values())
    for grid, paths in geo_paths.items():
        bundles[grid], manifests[grid], geo_fields[grid] = _external(*paths, grid, digests, files)

    parent, fine = bundles[192], bundles[384]
    refinement = fine["refinement"]
    if (
        refinement["factor"] != 2 or refinement["parent_bundle"] != parent["bundle_hash"]
        or refinement["parent_manifest_sha256"] != files[str((args.coarse_bundle / "frozen_contract.json").resolve())]
        or any(bundle["corrected_production_contract_hash"] != plan["requested_contract_hash"] for bundle in bundles.values())
        or any(not np.array_equal(bundle["requested_times"], times) for bundle in bundles.values())
        or not np.array_equal(summary["requested_times"], times)
        or manifests[192]["execution"] != manifests[384]["execution"]
        or manifests[192]["revisions"] != manifests[384]["revisions"]
    ):
        raise ValueError("GeoClaw grids do not share the recorded parent, execution settings, and output contract")
    scenarios = {case["index"]: case for case in plan["roster"]}
    case_maps = {
        grid: {case["selection"]["sample_index"]: case for case in bundle["cases"]}
        for grid, bundle in bundles.items()
    }
    if any(set(case_map) != set(CASE_INDICES) for case_map in case_maps.values()):
        raise ValueError("GeoClaw case roster differs from the prescribed three scenarios")
    if {row["case_id"] for row in summary["rows"]} != set(geo_fields[192]):
        raise ValueError("GeoClaw summary and bundle case roster differ")

    verified_inputs = []
    for index in CASE_INDICES:
        case = scenarios[index]
        old = case_maps[192][index]
        new = case_maps[384][index]
        case_id = old["case_id"]
        lineage = case["row"]["input_lineage"]
        parent_metadata = old["input"]
        if (
            new["case_id"] != case_id or new["parent_case_hash"] != old["case_hash"]
            or new["input"] != parent_metadata or old["selection"] != new["selection"]
            or old["selection"] != {"sample_index": index, "bathymetry_type": case["bathymetry"], "source_type": case["source"]}
            or parent_metadata["scenario_id"] != case["row"]["scenario_id"]
            or parent_metadata["source_strength"] != case["row"]["source_strength"]
            or parent_metadata["bathymetry_file_sha256"] != case["inputs"]["bathymetry"]["sha256"]
            or parent_metadata["source_file_sha256"] != case["inputs"]["sources"]["sha256"]
            or any(parent_metadata[key]["sha256"] != lineage[f"{key}_sha256"]
                   for key in ("solver_bathymetry", "solver_source", "master_bathymetry", "master_source"))
        ):
            raise ValueError(f"GeoClaw and in-house frozen identities differ: {case_id}")
        baseline = _initial_arrays(case, dataset, args.input_root, files)
        with (
            np.load(args.coarse_bundle / "cases" / case_id / "input.npz", allow_pickle=False) as coarse_arrays,
            np.load(args.fine_bundle / "cases" / case_id / "input.npz", allow_pickle=False) as fine_arrays,
        ):
            _verify_arrays(baseline, coarse_arrays, fine_arrays, times, old, new)
        for grid, variant in zip(GEO_GRIDS, CORE_GRIDS):
            requirement = next(row for row in bundles[grid]["external_results"] if row["case_id"] == case_id)
            domain_case = case_maps[grid][index]
            domain = domain_case["external_domain"]
            if (
                requirement["case_hash"] != domain_case["case_hash"]
                or requirement["comparator_id"] != "geoclaw_swe"
                or requirement["computational_shape"] != domain["shape"]
                or requirement["computational_domain_bounds"] != domain["bounds"]
                or requirement["publication_reduction"] != domain["publication_reduction"]
                or requirement["output_crop"] != domain["output_crop"] or requirement["eta_shape"] != list(SHAPE)
            ):
                raise ValueError(f"GeoClaw output requirement differs from the checked domain: {case_id}")
            for solver in SOLVERS:
                task = tasks[f"{case['row']['scenario_id']}__{solver}__{variant}"]
                cfg = task["solver_config"]
                core = CORE_GRIDS[variant]
                if (
                    task["solver"] != solver or task["variant"] != variant
                    or task["baseline_core"] != 128 or task["core"] != core
                    or task["publication_core"] != 64 or task["buffer_cells"] != grid // 6
                    or task["contract_hash"] != plan["requested_contract_hash"]
                    or cfg["nx"] != grid or cfg["ny"] != grid
                    or cfg["dx"] * grid != 3600 or cfg["dy"] * grid != 3600
                ):
                    raise ValueError(f"Numerical task domain differs from checked GeoClaw grid: {case_id}")
        verified_inputs.append({
            "case_id": case_id, "scenario_id": case["row"]["scenario_id"],
            "original_numerical_input_records": case["inputs"],
            "baseline_initial_array_sha256": {key: hash_array(value)["sha256"] for key, value in baseline.items()},
            "refined_initial_array_sha256": {key: new["refined_initial_arrays"][key]["sha256"] for key in FIELDS},
        })

    rows, paired_changes = _compare_cases(scenarios, case_maps, entries, geo_fields)
    for path, digest in files.items():
        _bind(files, Path(path), digest)
    result = {
        "schema_id": SCHEMA,
        "scope": {"requested_times": times.tolist(), "publication_shape": list(SHAPE[-2:]),
                  "full_study_rollouts": STUDY_COUNT, "case_count": len(CASE_INDICES)},
        "study_plan_sha256": stable_json_sha256(plan),
        "geoclaw_bundle_hashes": {str(grid): bundles[grid]["bundle_hash"] for grid in GEO_GRIDS},
        "archived_geoclaw_preparation_script_sha256": refinement["preparation_script_sha256"],
        "verified_input_pairs": verified_inputs, "rows": rows, "paired_changes": paired_changes,
        "source_sha256": files,
        "interpretation": (
            "Three paired cases on the shared 64×64 output grid at all 50 requested times. "
            "Both in-house levels use newly computed float64 trajectories. GeoClaw uses open "
            "extrapolation and the in-house solvers use their production radiation-plus-sponge "
            "profiles. Refinement preserves the frozen cell averages and physical domain while "
            "changing spatial resolution and natural time steps. Relative L2 uses the named "
            "GeoClaw trajectory norm, which changes with resolution. Changes describe inter-code "
            "agreement in this finite-horizon experiment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    numerical._write_json(args.output, result)
    print(f"Verified {len(rows)} matched comparisons in {args.output}")


if __name__ == "__main__":
    main()
