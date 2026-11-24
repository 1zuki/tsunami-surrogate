#!/usr/bin/env python
"""Calibrate simultaneous elevation bands on validation trajectories only.

The score is the maximum standardized absolute error over every requested
space-time cell in a scenario. Coverage therefore refers to whole numerical
trajectories. Validation was also used for model selection, so this analysis
does not supply an independent-calibration coverage guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from tsunami_surrogate.evaluation.target_scaling import load_target_denorm
from tsunami_surrogate.utils.config import load_config
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.utils.seed import seed_everything

from scripts.eval_ensemble_calibration import _load_ensemble, _loader_for_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_scores(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    std_floor: float,
    target_offset_over_scale: float = 0.0,
) -> dict[str, np.ndarray]:
    if mean.shape != target.shape or variance.shape != target.shape:
        raise ValueError("Ensemble fields and targets must have identical shapes")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("Standard-deviation floor must be positive and finite")
    if not all(torch.isfinite(value).all() for value in (mean, variance, target)):
        raise ValueError("Non-finite ensemble or target field")
    if torch.any(variance < 0):
        raise ValueError("Negative ensemble variance")
    error = (target - mean).to(torch.float64).flatten(start_dim=1)
    std = variance.to(torch.float64).clamp_min(std_floor**2).sqrt().flatten(start_dim=1)
    return {
        "score": (error.abs() / std).amax(dim=1).cpu().numpy(),
        "rmse": error.square().mean(dim=1).sqrt().cpu().numpy(),
        "rel_l2": (
            torch.linalg.vector_norm(error, dim=1)
            / torch.linalg.vector_norm(
                (target.to(torch.float64) + target_offset_over_scale).flatten(
                    start_dim=1
                ),
                dim=1,
            ).clamp_min(1e-12)
        )
        .cpu()
        .numpy(),
        "mean_std": std.mean(dim=1).cpu().numpy(),
    }


def _calibration_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Calibration requires a nonempty finite score vector")
    if np.any(values < 0) or not 0 < coverage < 1:
        raise ValueError("Invalid scores or nominal coverage")
    rank = math.ceil((values.size + 1) * coverage)
    if rank > values.size:
        raise ValueError(
            "Too few validation scenarios for a finite band at this coverage"
        )
    return float(np.sort(values)[rank - 1])


def _summarize(
    rows: list[dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty scenario group")
    scores = np.asarray([row["score"] for row in rows])
    return {
        "scenario_count": len(rows),
        "mean_rel_l2": float(np.mean([row["rel_l2"] for row in rows])),
        "simultaneous_coverage": {
            p: float(np.mean(scores <= q)) for p, q in thresholds.items()
        },
        "mean_band_half_width": {
            p: float(q * np.mean([row["mean_std_physical"] for row in rows]))
            for p, q in thresholds.items()
        },
    }


def _risk_coverage(
    rows: list[dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows, key=lambda row: (row["mean_std_physical"], row["scenario_id"])
    )
    return [
        {
            "requested_retained_fraction": fraction,
            "retained_fraction": max(1, math.ceil(fraction * len(ordered)))
            / len(ordered),
            **_summarize(
                ordered[: max(1, math.ceil(fraction * len(ordered)))], thresholds
            ),
        }
        for fraction in (0.25, 0.5, 0.75, 0.9, 1.0)
    ]


@torch.no_grad()
def _score_loader(
    ensemble: Any,
    loader: Any,
    device: torch.device,
    *,
    std_floor: float,
    target_scale: float,
    target_offset: float,
    delay: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(loader, 1):
        remaining = (
            len(batch["x"])
            if limit is None
            else min(len(batch["x"]), limit - len(rows))
        )
        if remaining <= 0:
            break
        output = ensemble(batch["x"][:remaining].to(device))
        values = _trajectory_scores(
            output["mean"],
            output["variance"],
            batch["y"][:remaining].to(device),
            std_floor,
            target_offset / target_scale,
        )
        for index in range(remaining):
            rows.append(
                {
                    "scenario_id": str(batch["scenario_id"][index]),
                    "source_type": str(batch["source_type"][index]),
                    "bathymetry_type": str(batch["bathymetry_type"][index]),
                    "source_strength": float(batch["source_strength"][index]),
                    **{key: float(value[index]) for key, value in values.items()},
                    "mean_std_physical": float(values["mean_std"][index])
                    * abs(target_scale),
                    "rmse_physical": float(values["rmse"][index]) * abs(target_scale),
                }
            )
        if batch_number % 25 == 0:
            print(f"scenarios={len(rows)}", flush=True)
        if delay:
            time.sleep(delay)
    if not rows or len({row["scenario_id"] for row in rows}) != len(rows):
        raise ValueError("Scenario roster is empty or contains duplicate identifiers")
    return rows


def _dataset_identity(path: Path, split: str, expected_count: int) -> dict[str, str]:
    manifest = json.loads((path / "eval_manifest.json").read_text())
    lineage = manifest.get("provenance", {}).get("source_lineage", {})
    if (
        manifest.get("split") != split
        or manifest.get("num_samples") != expected_count
        or manifest.get("target_variable") != "eta"
        or lineage.get("solver_names") != ["hydrostatic"]
        or not lineage.get("contract_hash")
    ):
        raise ValueError(
            "Unexpected trajectory-calibration dataset population or reference"
        )
    files = [
        path / name
        for name in ("eval_manifest.json", "shards_manifest.json", "meta.jsonl")
    ]
    found = {str(file.resolve()): _sha256(file) for file in files if file.is_file()}
    if not found:
        raise ValueError(f"No dataset provenance manifests at {path}")
    return found


def _config_identity(
    path: Path, active: frozenset[Path] = frozenset()
) -> dict[str, str]:
    path = path.resolve()
    if path in active:
        raise ValueError("Recursive configuration defaults")
    data = yaml.safe_load(path.read_text()) or {}
    found = {str(path): _sha256(path)}
    for default in data.get("defaults", []) or []:
        found.update(_config_identity(path.parent / default, active | {path}))
    return found


def _assert_disjoint_lineage(validation: Path, test: Path) -> dict[str, Any]:
    identities = {}
    for name, path in (("validation", validation), ("test", test)):
        rows = [
            json.loads(line) for line in (path / "meta.jsonl").read_text().splitlines()
        ]
        pairs = {
            (
                row["input_lineage"]["master_bathymetry_sha256"],
                row["input_lineage"]["master_source_sha256"],
            )
            for row in rows
        }
        if not pairs or len(pairs) != len(rows):
            raise ValueError("Repeated or empty master-input identities within a split")
        identities[name] = pairs
    if identities["validation"] & identities["test"]:
        raise ValueError("Calibration and test scenarios share master inputs")
    return {
        "validation_unique": len(identities["validation"]),
        "test_unique": len(identities["test"]),
        "shared_master_input_pairs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/fno.yaml")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--val-dataset", default="data/processed/hydrostatic/val")
    parser.add_argument("--test-dataset", default="data/processed/hydrostatic/test")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-6,
        help="Standard-deviation floor in normalized target units",
    )
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Diagnostic subset only; never a full-data result",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if (
        args.batch_size <= 0
        or args.delay < 0
        or (args.max_samples is not None and args.max_samples <= 0)
    ):
        parser.error(
            "Batch size and sample limit must be positive; delay must be nonnegative"
        )
    if Path(args.val_dataset).resolve() == Path(args.test_dataset).resolve():
        raise ValueError("Calibration and test data must be separate splits")
    if not math.isfinite(args.std_floor) or args.std_floor <= 0:
        parser.error("Standard-deviation floor must be positive and finite")
    cfg = load_config(args.config)
    cfg["data"] = {**cfg.get("data", {}), "num_workers": 0}
    seed_everything(20261124)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    denorm = {
        name: load_target_denorm(path)
        for name, path in (
            ("validation", args.val_dataset),
            ("test", args.test_dataset),
        )
    }
    if (
        any(value is None for value in denorm.values())
        or denorm["validation"] != denorm["test"]
    ):
        raise ValueError(
            "Both splits require identical training-derived target normalization"
        )
    provenance = {
        "config": _config_identity(Path(args.config)),
        "checkpoints": {
            str(Path(path).resolve()): _sha256(Path(path)) for path in args.checkpoint
        },
        "validation": _dataset_identity(Path(args.val_dataset), "val", 1000),
        "test": _dataset_identity(Path(args.test_dataset), "test", 2500),
        "script_sha256": _sha256(Path(__file__)),
        "source_files": {
            str(path.resolve()): _sha256(path)
            for path in sorted(
                {
                    *ROOT.joinpath("src/models").rglob("*.py"),
                    *ROOT.joinpath("src/data").rglob("*.py"),
                    *ROOT.joinpath("src/utils").rglob("*.py"),
                    ROOT / "src/training/checkpointing.py",
                    ROOT / "src/evaluation/target_scaling.py",
                    ROOT / "scripts/eval_ensemble_calibration.py",
                    ROOT / "tsunami_surrogate/__init__.py",
                }
            )
        },
    }
    lineage_check = _assert_disjoint_lineage(
        Path(args.val_dataset), Path(args.test_dataset)
    )
    if len(provenance["checkpoints"]) < 2 or len(
        set(provenance["checkpoints"].values())
    ) != len(args.checkpoint):
        raise ValueError("Ensemble requires distinct checkpoints")
    ensemble = _load_ensemble(cfg, [Path(path) for path in args.checkpoint], device)
    records = {}
    thresholds = None
    for name, split, path in (
        ("validation", "val", args.val_dataset),
        ("test", "test", args.test_dataset),
    ):
        print(f"split={name}", flush=True)
        records[name] = _score_loader(
            ensemble,
            _loader_for_path(cfg, split, Path(path), args.batch_size),
            device,
            std_floor=args.std_floor,
            target_scale=float(denorm[name][1]),
            target_offset=float(denorm[name][0]),
            delay=args.delay,
            limit=args.max_samples,
        )
        if args.max_samples is None and len(records[name]) != (
            1000 if name == "validation" else 2500
        ):
            raise RuntimeError(
                "Full split evaluation did not cover its expected population"
            )
        if name == "validation":
            thresholds = {
                str(p): _calibration_quantile(
                    np.asarray([r["score"] for r in records[name]]), p
                )
                for p in (0.5, 0.8, 0.9, 0.95)
            }
    groups = {"all": records["test"]}
    for key in ("source_type", "bathymetry_type"):
        for value in sorted({row[key] for row in records["test"]}):
            groups[f"{key}={value}"] = [
                row for row in records["test"] if row[key] == value
            ]
    high = [row for row in records["test"] if row["source_strength"] >= 0.285]
    if high:
        groups["source_strength>=0.285"] = high
    result = {
        "evaluation_type": "simultaneous_trajectory_calibration",
        "status": "diagnostic_subset"
        if args.max_samples is not None
        else "completed_full_splits",
        "score_definition": "max over all requested times and spatial cells of abs(target-mean)/max(std,std_floor)",
        "calibration_fit_split": "validation",
        "quantile_rule": "ceil((n_validation+1)*nominal_coverage) order statistic",
        "interpretation": "Empirical numerical-reference trajectory coverage. Validation was previously used for model selection; no independent-calibration or subgroup coverage guarantee is claimed.",
        "calibration_set_previously_used_for_model_selection": True,
        "lineage_check": lineage_check,
        "numerics": {
            "torch_version": str(torch.__version__),
            "default_dtype": str(torch.get_default_dtype()),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "std_floor_normalized": args.std_floor,
        "thresholds": thresholds,
        "risk_metric": "relative L2 after reversing training normalization; rmse is retained in normalized units and rmse_physical is denormalized",
        "retention_policy": "Retrospective within-test-cohort ranking by mean predicted standard deviation; no deployment cutoff is fitted",
        "provenance": provenance,
        "device": str(device),
        "groups": {name: _summarize(rows, thresholds) for name, rows in groups.items()},
        "risk_coverage": _risk_coverage(records["test"], thresholds),
        "records": records,
    }
    # Check immutable source inputs again before publishing a complete result.
    for mapping in (
        provenance["config"],
        provenance["checkpoints"],
        provenance["validation"],
        provenance["test"],
        provenance["source_files"],
    ):
        if any(_sha256(Path(path)) != expected for path, expected in mapping.items()):
            raise RuntimeError("An input changed during evaluation")
    if _sha256(Path(__file__)) != provenance["script_sha256"]:
        raise RuntimeError("Evaluation script changed during execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
