#!/usr/bin/env python
"""Restore physical risk metrics from saved trajectory scores and verified targets.

No model inference is performed. Simultaneous scores and fitted thresholds are
retained unchanged; the nonzero target offset is restored in relative L2.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tsunami_surrogate.data.dataset import _make_dataset
from tsunami_surrogate.evaluation.target_scaling import load_target_denorm

from scripts.eval_trajectory_calibration import (
    _assert_disjoint_lineage,
    _config_identity,
    _dataset_identity,
    _risk_coverage,
    _sha256,
    _summarize,
)


def _validate_dataset_binding(
    observed: dict[str, str], expected: dict[str, str]
) -> None:
    observed_by_name = {Path(path).name: digest for path, digest in observed.items()}
    expected_by_name = {Path(path).name: digest for path, digest in expected.items()}
    if (
        len(observed_by_name) != len(observed)
        or len(expected_by_name) != len(expected)
        or observed_by_name != expected_by_name
    ):
        raise ValueError("Target dataset bytes differ from the saved inference inputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--execution-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model/fno.yaml"))
    parser.add_argument(
        "--val-dataset", type=Path, default=Path("data/processed/hydrostatic/val")
    )
    parser.add_argument(
        "--test-dataset", type=Path, default=Path("data/processed/hydrostatic/test")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hash = _sha256(Path(__file__))
    original_bytes = args.input.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    original = json.loads(original_bytes)
    provenance = original["provenance"]
    if str(args.config.resolve()) not in provenance["config"]:
        raise ValueError("Configuration was not recorded by the inference run")
    if _sha256(args.execution_source) != provenance["script_sha256"]:
        raise ValueError("Saved inference source does not match its recorded hash")
    for group in ("config", "checkpoints", "source_files", "validation", "test"):
        if any(
            _sha256(Path(path)) != digest for path, digest in provenance[group].items()
        ):
            raise ValueError(f"Inference inputs no longer match: {group}")
    result = copy.deepcopy(original)
    lineage = _assert_disjoint_lineage(args.val_dataset, args.test_dataset)
    torch.set_num_threads(1)
    for name, path, split, count in (
        ("validation", args.val_dataset, "val", 1000),
        ("test", args.test_dataset, "test", 2500),
    ):
        _validate_dataset_binding(
            _dataset_identity(path, split, count), provenance[name]
        )
        offset, scale = load_target_denorm(path)
        rows = result["records"][name]
        if len(rows) != count:
            raise ValueError("Saved inference did not cover the full expected split")
        by_id = {row["scenario_id"]: row for row in rows}
        if len(by_id) != count:
            raise ValueError("Duplicate inference identities")
        seen = set()
        for batch in DataLoader(
            _make_dataset(str(path)), batch_size=16, shuffle=False, num_workers=0
        ):
            shifted = batch["y"].to(torch.float64) + float(offset) / float(scale)
            rms = shifted.square().flatten(start_dim=1).mean(dim=1).sqrt().numpy()
            for index, scenario_id in enumerate(batch["scenario_id"]):
                if scenario_id not in by_id or scenario_id in seen:
                    raise ValueError("Target and inference scenario rosters differ")
                row = by_id[scenario_id]
                for key in ("bathymetry_type", "source_type"):
                    if row[key] != batch[key][index]:
                        raise ValueError("Target and inference metadata differ")
                if float(row["source_strength"]) != float(
                    batch["source_strength"][index]
                ):
                    raise ValueError("Source strength mismatch")
                row["rel_l2_normalized"] = row["rel_l2"]
                row["rel_l2"] = float(row["rmse"] / max(float(rms[index]), 1e-12))
                row["rmse_physical"] = float(row["rmse"] * abs(scale))
                seen.add(scenario_id)
        if seen != set(by_id):
            raise ValueError("Incomplete target norm audit")
        print(f"Restored physical risks for {name}: {len(seen)} scenarios", flush=True)
    rows = result["records"]["test"]
    groups = {"all": rows}
    for key in ("source_type", "bathymetry_type"):
        for value in sorted({row[key] for row in rows}):
            groups[f"{key}={value}"] = [row for row in rows if row[key] == value]
    groups["source_strength>=0.285"] = [
        row for row in rows if row["source_strength"] >= 0.285
    ]
    result["groups"] = {
        name: _summarize(group, result["thresholds"]) for name, group in groups.items()
    }
    result["risk_coverage"] = _risk_coverage(rows, result["thresholds"])
    result["risk_metric"] = (
        "relative L2 after reversing training normalization; rmse is normalized and rmse_physical is denormalized"
    )
    result["lineage_check"] = lineage
    result["postprocessing"] = {
        "inference_result": str(args.input.resolve()),
        "inference_result_sha256": original_hash,
        "execution_source": str(args.execution_source.resolve()),
        "script_sha256": source_hash,
        "config_dependencies": _config_identity(args.config),
        "score_and_thresholds_changed": False,
        "method": "physical relative L2 = normalized RMSE / RMS(normalized target + training offset / training scale)",
    }
    if _sha256(args.input) != original_hash or _sha256(Path(__file__)) != source_hash:
        raise RuntimeError("Input scores or summarizer source changed during analysis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
