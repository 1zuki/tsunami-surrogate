#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.target_scaling import (
    load_target_denorm,
    resolve_dataset_npz,
    signatures_match,
    target_signature,
)
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.utils.seed import seed_everything


def _suite_test_loader(cfg: Dict[str, Any], dataset_path: str, batch_size: int):
    resolved_dataset_path = resolve_dataset_npz(dataset_path)
    local_cfg = dict(cfg)
    local_data = {
        "test_path": str(resolved_dataset_path),
        "batch_size": int(batch_size),
        "num_workers": 0,
    }
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


def _checkpoint_train_path(ckpt: Dict[str, Any]) -> str | None:
    raw_cfg = ckpt.get("config", {})
    if not isinstance(raw_cfg, dict):
        return None
    data_cfg = raw_cfg.get("data", raw_cfg.get("dataset", {}))
    if not isinstance(data_cfg, dict):
        return None

    train_path = data_cfg.get("train_path")
    if train_path:
        return str(train_path)
    fallback_path = data_cfg.get("path")
    if fallback_path:
        return str(fallback_path)
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a single trained model across multiple real-resolution processed datasets."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    rr_cfg = eval_cfg.get("real_resolution", cfg.get("real_resolution", {}))
    suites = list(rr_cfg.get("suites", []))
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))
    normalization_policy = str(rr_cfg.get("normalization_policy", "require_target_stats_match")).strip().lower()
    mismatch_action = str(rr_cfg.get("normalization_mismatch", "fail")).strip().lower()
    checkpoint_mismatch_action = str(rr_cfg.get("checkpoint_reference_mismatch", "fail")).strip().lower()
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
    if checkpoint_mismatch_action not in {"warn", "fail"}:
        raise ValueError("real_resolution.checkpoint_reference_mismatch must be one of: warn, fail")
    if normalization_policy == "ignore" and report_physical:
        print(
            "[eval_full_resolution][warn] "
            "real_resolution.normalization_policy=ignore can make physical denormalized metrics misleading. "
            "Disabling report_physical_metrics for this run."
        )
        report_physical = False

    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    checkpoint_payload = load_checkpoint(args.checkpoint, model, map_location=device)

    result_rows: List[Dict[str, Any]] = []
    reference_signature: Dict[str, Any] | None = None
    checkpoint_train_signature: Dict[str, Any] | None = None
    checkpoint_train_path: str | None = None
    if normalization_policy == "require_target_stats_match":
        ref_path = rr_cfg.get("normalization_reference_path", cfg.get("data", {}).get("train_path"))
        if not ref_path:
            raise KeyError(
                "Normalization policy requires a reference dataset path. "
                "Set real_resolution.normalization_reference_path or data.train_path."
            )
        reference_signature = target_signature(str(ref_path))
        if (
            bool(reference_signature.get("normalized_targets", False))
            and (
                reference_signature.get("target_offset") is None
                or reference_signature.get("target_scale") is None
            )
        ):
            raise ValueError(
                "Normalization reference declares normalized targets but has no target_mean/target_std stats: "
                f"{reference_signature.get('dataset_path')}"
            )
        print(f"[eval_full_resolution] normalization reference: {reference_signature}")

        checkpoint_train_path = str(rr_cfg.get("checkpoint_train_path", "")).strip() or _checkpoint_train_path(checkpoint_payload)
        if not checkpoint_train_path:
            msg = (
                "Could not resolve checkpoint training dataset path from checkpoint config. "
                "Set real_resolution.checkpoint_train_path explicitly to validate native-resolution claims."
            )
            if checkpoint_mismatch_action == "fail":
                raise ValueError(msg)
            print(f"[eval_full_resolution][warn] {msg}")
        else:
            try:
                checkpoint_train_signature = target_signature(checkpoint_train_path)
            except FileNotFoundError as e:
                msg = (
                    f"Checkpoint training dataset not found: {checkpoint_train_path}. "
                    "Provide real_resolution.checkpoint_train_path or regenerate the training split."
                )
                if checkpoint_mismatch_action == "fail":
                    raise FileNotFoundError(msg) from e
                print(f"[eval_full_resolution][warn] {msg}")
                checkpoint_train_signature = None

            if checkpoint_train_signature is not None and reference_signature is not None:
                if not signatures_match(reference_signature, checkpoint_train_signature, tol=normalization_tol):
                    msg = (
                        "Checkpoint training normalization signature does not match the evaluation reference. "
                        f"checkpoint={checkpoint_train_signature}, reference={reference_signature}, tol={normalization_tol}. "
                        "Use a checkpoint trained on the same normalization reference for paper-safe native-resolution claims."
                    )
                    if checkpoint_mismatch_action == "fail":
                        raise ValueError(msg)
                    print(f"[eval_full_resolution][warn] {msg}")

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
        suite_signature = target_signature(dataset_path)
        if (
            bool(suite_signature.get("normalized_targets", False))
            and (suite_signature.get("target_offset") is None or suite_signature.get("target_scale") is None)
        ):
            msg = (
                f"Suite '{label}' declares normalized targets but has missing target stats: "
                f"{suite_signature.get('dataset_path')}"
            )
            if mismatch_action == "fail":
                raise ValueError(msg)
            print(f"[eval_full_resolution][warn] {msg}")
        if normalization_policy == "require_target_stats_match" and reference_signature is not None:
            if not signatures_match(reference_signature, suite_signature, tol=normalization_tol):
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
        "config_path": args.config,
        "checkpoint": args.checkpoint,
        "rows": result_rows,
        "normalization_policy": normalization_policy,
        "normalization_mismatch": mismatch_action,
        "checkpoint_reference_mismatch": checkpoint_mismatch_action,
        "normalization_tol": normalization_tol,
        "normalization_reference": reference_signature,
        "checkpoint_train_path": checkpoint_train_path,
        "checkpoint_train_signature": checkpoint_train_signature,
    }
    print(summary)
    output_path = (
        Path(args.output)
        if args.output
        else Path(output_dir) / "real_resolution.json"
    )
    save_json(summary, output_path)


if __name__ == "__main__":
    main()
