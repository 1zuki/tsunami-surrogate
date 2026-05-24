#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.target_scaling import load_target_denorm, resolve_dataset_npz
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels


def _suite_test_loader(cfg: Dict[str, Any], dataset_path: str, batch_size: int):
    resolved_dataset_path = resolve_dataset_npz(dataset_path)
    local_cfg = dict(cfg)
    local_data = dict(local_cfg.get("data", {}))
    local_data["test_path"] = str(resolved_dataset_path)
    local_data["batch_size"] = batch_size
    local_cfg["data"] = local_data

    loaders = create_dataloaders(local_cfg)
    test_loader = loaders.get("test")

    if test_loader is None:
        raise KeyError(
            "No test dataloader could be created for suite dataset_path: "
            f"{dataset_path} (resolved: {resolved_dataset_path})"
        )
    
    validate_model_io_channels(local_cfg, loaders, preferred_splits=("test",))
    return test_loader


def _dataset_size_from_loader(loader) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _target_signature(dataset_path: str | Path) -> Dict[str, Any]:
    npz_path = resolve_dataset_npz(dataset_path)
    denorm = load_target_denorm(npz_path)
    manifest_path = npz_path.with_name("eval_manifest.json")
    normalized_targets = None
    
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if isinstance(manifest.get("normalized_targets"), bool):
                normalized_targets = bool(manifest["normalized_targets"])
        except Exception:
            normalized_targets = None

    if denorm is not None:
        offset, scale = denorm
        return {
            "dataset_path": str(npz_path),
            "normalized_targets": True if normalized_targets is None else bool(normalized_targets),
            "target_offset": float(offset),
            "target_scale": float(scale),
        }

    return {
        "dataset_path": str(npz_path),
        "normalized_targets": False if normalized_targets is None else bool(normalized_targets),
        "target_offset": None,
        "target_scale": None,
    }


def _signatures_match(reference: Dict[str, Any], candidate: Dict[str, Any], tol: float) -> bool:
    ref_norm = bool(reference.get("normalized_targets", False))
    cand_norm = bool(candidate.get("normalized_targets", False))
    if ref_norm != cand_norm:
        return False

    ref_off = reference.get("target_offset")
    ref_scale = reference.get("target_scale")
    cand_off = candidate.get("target_offset")
    cand_scale = candidate.get("target_scale")

    if ref_off is None or ref_scale is None or cand_off is None or cand_scale is None:
        return True

    return bool(
        abs(float(ref_off) - float(cand_off)) <= tol
        and abs(float(ref_scale) - float(cand_scale)) <= tol
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a single trained model across multiple real-resolution processed datasets."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    rr_cfg = eval_cfg.get("real_resolution", cfg.get("real_resolution", {}))
    suites = list(rr_cfg.get("suites", []))
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))
    normalization_policy = str(rr_cfg.get("normalization_policy", "require_target_stats_match")).strip().lower()
    mismatch_action = str(rr_cfg.get("normalization_mismatch", "fail")).strip().lower()
    normalization_tol = float(rr_cfg.get("normalization_tol", 1e-6))

    if not suites:
        raise ValueError(
            "No real-resolution suites configured. "
            "Add `eval.real_resolution.suites` with `label` + `path` entries."
        )
    if normalization_policy not in {"require_target_stats_match", "ignore"}:
        raise ValueError("real_resolution.normalization_policy must be one of: require_target_stats_match, ignore")
    if mismatch_action not in {"warn", "fail"}:
        raise ValueError("real_resolution.normalization_mismatch must be one of: warn, fail")
    if normalization_policy == "ignore" and report_physical:
        print(
            "[eval_full_resolution][warn] "
            "real_resolution.normalization_policy=ignore can make physical denormalized metrics misleading. "
            "Disabling report_physical_metrics for this run."
        )
        report_physical = False

    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    result_rows: List[Dict[str, Any]] = []
    reference_signature: Dict[str, Any] | None = None
    if normalization_policy == "require_target_stats_match":
        ref_path = rr_cfg.get("normalization_reference_path", cfg.get("data", {}).get("train_path"))
        if not ref_path:
            raise KeyError(
                "Normalization policy requires a reference dataset path. "
                "Set real_resolution.normalization_reference_path or data.train_path."
            )
        reference_signature = _target_signature(str(ref_path))
        print(f"[eval_full_resolution] normalization reference: {reference_signature}")

    for i, suite in enumerate(suites):
        suite_cfg = suite if isinstance(suite, dict) else {}
        label = str(suite_cfg.get("label", f"suite_{i}")).strip()
        dataset_path = str(suite_cfg.get("path", "")).strip()

        if not dataset_path:
            raise KeyError(f"real_resolution.suites[{i}] is missing required key: path")
        
        batch_size = int(
            suite_cfg.get("batch_size", eval_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 8)))
        )

        loader = _suite_test_loader(cfg, dataset_path=dataset_path, batch_size=batch_size)
        suite_signature = _target_signature(dataset_path)
        if normalization_policy == "require_target_stats_match" and reference_signature is not None:
            if not _signatures_match(reference_signature, suite_signature, tol=normalization_tol):
                msg = (
                    f"Normalization signature mismatch for suite '{label}'. "
                    f"reference={reference_signature}, suite={suite_signature}, tol={normalization_tol}"
                )
                if mismatch_action == "fail":
                    raise ValueError(msg)
                print(f"[eval_full_resolution][warn] {msg}")

        metrics = evaluate_accuracy(model, loader, device)
        row: Dict[str, Any] = {
            "label": label,
            "dataset_path": dataset_path,
            "num_samples": _dataset_size_from_loader(loader),
            "normalized_targets": suite_signature["normalized_targets"],
            **{k: float(v) for k, v in metrics.items()},
        }
        if suite_signature.get("target_offset") is not None:
            row["target_offset"] = float(suite_signature["target_offset"])
        if suite_signature.get("target_scale") is not None:
            row["target_scale"] = float(suite_signature["target_scale"])

        if report_physical:
            denorm = load_target_denorm(Path(dataset_path))
            if denorm is not None:
                metrics_phys = evaluate_accuracy(model, loader, device, target_denorm=denorm)
                row.update({f"{k}_physical": float(v) for k, v in metrics_phys.items()})

        result_rows.append(row)

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"

    summary: Dict[str, Any] = {
        "evaluation_type": "native_real_resolution_benchmark",
        "rows": result_rows,
        "normalization_policy": normalization_policy,
        "normalization_mismatch": mismatch_action,
        "normalization_tol": normalization_tol,
        "normalization_reference": reference_signature,
    }
    print(summary)
    save_json(summary, f"{output_dir}/real_resolution.json")


if __name__ == "__main__":
    main()
