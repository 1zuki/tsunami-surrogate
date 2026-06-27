#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.accuracy import evaluate_accuracy
from src.evaluation.target_scaling import load_target_denorm, resolve_dataset_npz, signatures_match, target_signature
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.training.metrics import MetricAccumulator
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.utils.seed import seed_everything


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _solver_metric_mean(payload: Dict[str, Any], metric: str) -> float:
    agg = payload.get("aggregate_metrics", {})
    if metric not in agg:
        raise KeyError(f"Metric '{metric}' not found in solver comparison aggregate_metrics")
    row = agg[metric]
    if not isinstance(row, dict) or "mean" not in row:
        raise KeyError(f"Metric '{metric}' in solver comparison has no 'mean'")
    return float(row["mean"])


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


def _input_norm_signature(dataset_path: str | Path) -> Dict[str, Any] | None:
    npz_path = resolve_dataset_npz(dataset_path)
    split_dir = npz_path.parent
    if split_dir.name == "shards":
        split_dir = split_dir.parent
    processed_root = split_dir.parent
    stats_path = processed_root / "normalization_stats.json"
    manifest_path = split_dir / "eval_manifest.json"

    input_order: list[str] = []
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            order = manifest_payload.get("input_order", [])
            if isinstance(order, list):
                input_order = [str(x) for x in order]
        except Exception:
            input_order = []

    channels: Dict[str, Dict[str, float]] = {}
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            inputs_raw = payload.get("inputs", {})
            if isinstance(inputs_raw, dict):
                for name, stats in inputs_raw.items():
                    if not isinstance(stats, dict):
                        continue
                    if "offset" not in stats or "scale" not in stats:
                        continue
                    channels[str(name)] = {
                        "offset": float(stats["offset"]),
                        "scale": float(stats["scale"]),
                    }
        except Exception:
            channels = {}

    if not channels and not input_order:
        return None

    return {
        "dataset_path": str(npz_path),
        "normalization_stats_path": str(stats_path),
        "input_order": input_order,
        "channels": channels,
    }


def _compare_input_signatures(
    prediction_sig: Dict[str, Any] | None,
    eval_sig: Dict[str, Any] | None,
    tol: float,
) -> Dict[str, Any]:
    if prediction_sig is None or eval_sig is None:
        return {
            "checked": False,
            "compatible": False,
            "reason": "missing_input_signature",
            "shared_channels": [],
            "mismatch_channels": [],
            "unscaled_shared_channels": [],
            "mixed_scaled_channels": [],
        }

    pred_channels = dict(prediction_sig.get("channels", {}))
    eval_channels = dict(eval_sig.get("channels", {}))
    pred_order = [str(x) for x in prediction_sig.get("input_order", [])]
    eval_order = [str(x) for x in eval_sig.get("input_order", [])]
    shared_order = sorted(set(pred_order) & set(eval_order))
    shared = sorted(set(pred_channels.keys()) & set(eval_channels.keys()))

    if not shared:
        if shared_order and all((ch not in pred_channels and ch not in eval_channels) for ch in shared_order):
            return {
                "checked": True,
                "compatible": True,
                "reason": "shared_channels_unscaled",
                "shared_channels": [],
                "mismatch_channels": [],
                "unscaled_shared_channels": shared_order,
                "mixed_scaled_channels": [],
            }
        return {
            "checked": False,
            "compatible": False,
            "reason": "no_shared_channels",
            "shared_channels": [],
            "mismatch_channels": [],
            "unscaled_shared_channels": [],
            "mixed_scaled_channels": [],
        }

    mismatches: list[str] = []
    for ch in shared:
        pa = pred_channels[ch]
        ea = eval_channels[ch]
        if (
            abs(float(pa["offset"]) - float(ea["offset"])) > tol
            or abs(float(pa["scale"]) - float(ea["scale"])) > tol
        ):
            mismatches.append(ch)

    unscaled_shared = [ch for ch in shared_order if ch not in pred_channels and ch not in eval_channels]
    mixed_scaled = [ch for ch in shared_order if (ch in pred_channels) ^ (ch in eval_channels)]

    if mixed_scaled:
        mismatches.extend([f"{ch}(scaled-vs-unscaled)" for ch in mixed_scaled])

    mismatches = sorted(set(mismatches))

    return {
        "checked": True,
        "compatible": len(mismatches) == 0,
        "reason": "ok" if len(mismatches) == 0 else "channel_mismatch",
        "shared_channels": shared,
        "mismatch_channels": mismatches,
        "unscaled_shared_channels": unscaled_shared,
        "mixed_scaled_channels": mixed_scaled,
    }


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


@torch.no_grad()
def _evaluate_metrics(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    pred_denorm: tuple[float, float] | None = None,
    target_denorm: tuple[float, float] | None = None,
) -> Dict[str, float]:
    model.eval()
    metrics_acc = MetricAccumulator()

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = _model_output(model, x)
        pred_eval = pred
        y_eval = y

        if pred_denorm is not None:
            pred_eval = pred_eval * float(pred_denorm[1]) + float(pred_denorm[0])
        if target_denorm is not None:
            y_eval = y_eval * float(target_denorm[1]) + float(target_denorm[0])

        metrics_acc.update(pred_eval, y_eval)

    return metrics_acc.compute()


def main() -> None:
    p = argparse.ArgumentParser(description="Compute emulator-superiority ratio against solver-vs-solver error.")
    p.add_argument("--config", required=True, help="YAML config for ratio evaluation")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    model_cfg_path = str(cfg.get("model_config", "")).strip()
    checkpoint_path = str(cfg.get("checkpoint", "")).strip()
    solver_compare_path = Path(str(cfg.get("solver_compare_path", "")).strip())
    eval_cfg = dict(cfg.get("evaluation", {}))
    ratio_cfg = dict(cfg.get("ratio", {}))
    norm_cfg = dict(cfg.get("normalization", {}))

    if not model_cfg_path:
        raise KeyError("config requires model_config")
    if not checkpoint_path:
        raise KeyError("config requires checkpoint")
    if not solver_compare_path:
        raise KeyError("config requires solver_compare_path")
    if not solver_compare_path.exists():
        raise FileNotFoundError(solver_compare_path)

    dataset_path = str(eval_cfg.get("dataset_path", "")).strip()
    if not dataset_path:
        raise KeyError("config requires evaluation.dataset_path")

    batch_size = int(eval_cfg.get("batch_size", 8))
    report_physical = bool(eval_cfg.get("report_physical_metrics", True))
    numerator_metric = str(ratio_cfg.get("numerator_metric", "rmse_physical_separate_denorm"))
    denominator_metric = str(ratio_cfg.get("denominator_metric", "rmse"))
    output_path = Path(str(cfg.get("output_path", "results/emulator_superiority.json")))

    if (not report_physical) and ("physical" in numerator_metric):
        raise ValueError(
            "ratio.numerator_metric requests physical-space metrics, but evaluation.report_physical_metrics=false."
        )
    mismatch_action = str(norm_cfg.get("on_signature_mismatch", "fail")).strip().lower()
    if mismatch_action not in {"warn", "fail"}:
        raise ValueError("normalization.on_signature_mismatch must be one of: warn, fail")
    
    input_mismatch_action = str(norm_cfg.get("on_input_stats_mismatch", mismatch_action)).strip().lower()
    if input_mismatch_action not in {"warn", "fail"}:
        raise ValueError("normalization.on_input_stats_mismatch must be one of: warn, fail")
    
    check_input_stats = bool(norm_cfg.get("check_input_stats", True))
    signature_tol = float(norm_cfg.get("tol", 1e-6))

    model_cfg = load_config(model_cfg_path)
    if args.device is not None:
        model_cfg["device"] = args.device
    seed_everything(int(model_cfg.get("seed", cfg.get("seed", 42))))
    data_cfg = dict(model_cfg.get("data", {}))
    data_cfg["test_path"] = dataset_path
    data_cfg["batch_size"] = batch_size
    model_cfg["data"] = data_cfg

    device = resolve_device(model_cfg.get("device", "auto"))
    loaders = create_dataloaders(model_cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError("Could not create test loader for evaluation.dataset_path")

    validate_model_io_channels(model_cfg, loaders, preferred_splits=("test",))

    model = build_model(model_cfg).to(device)
    checkpoint_payload = load_checkpoint(checkpoint_path, model, map_location=device)
    metrics = evaluate_accuracy(model, test_loader, device)
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["num_samples"] = float(_dataset_num_samples(test_loader))
    metrics["dataset_path"] = str(dataset_path)

    target_denorm_path = str(norm_cfg.get("target_denorm_path", "")).strip() or dataset_path
    checkpoint_train_path = _checkpoint_train_path(checkpoint_payload)
    prediction_denorm_path = str(norm_cfg.get("prediction_denorm_path", "")).strip() or checkpoint_train_path

    eval_signature = target_signature(dataset_path)
    prediction_signature = None
    if prediction_denorm_path:
        try:
            prediction_signature = target_signature(prediction_denorm_path)
        except FileNotFoundError as e:
            msg = (
                f"prediction_denorm_path not found: {prediction_denorm_path}. "
                "Set normalization.prediction_denorm_path to an existing train/eval dataset archive."
            )
            if mismatch_action == "fail":
                raise FileNotFoundError(msg) from e
            print(f"[eval_emulator_superiority][warn] {msg}")

    signature_mismatch = (
        prediction_signature is not None
        and not signatures_match(prediction_signature, eval_signature, tol=signature_tol)
    )
    prediction_input_signature = _input_norm_signature(prediction_denorm_path) if prediction_denorm_path else None
    eval_input_signature = _input_norm_signature(dataset_path)
    input_check = _compare_input_signatures(prediction_input_signature, eval_input_signature, tol=signature_tol)

    if check_input_stats and (not input_check["checked"] or not input_check["compatible"]):
        msg = (
            "Cross-solver input normalization stats are not verified compatible. "
            f"check={input_check}, prediction_input_signature={prediction_input_signature}, "
            f"eval_input_signature={eval_input_signature}."
        )
        if input_mismatch_action == "fail":
            raise ValueError(msg)
        print(f"[eval_emulator_superiority][warn] {msg}")

    target_denorm = None
    prediction_denorm = None
    if report_physical:
        try:
            target_denorm = load_target_denorm(target_denorm_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"target_denorm_path not found: {target_denorm_path}. "
                "Set normalization.target_denorm_path to an existing eval dataset archive."
            ) from e
        if prediction_denorm_path:
            try:
                prediction_denorm = load_target_denorm(prediction_denorm_path)
            except FileNotFoundError as e:
                msg = (
                    f"prediction_denorm_path not found: {prediction_denorm_path}. "
                    "Set normalization.prediction_denorm_path to an existing train/eval dataset archive."
                )
                if mismatch_action == "fail":
                    raise FileNotFoundError(msg) from e
                print(f"[eval_emulator_superiority][warn] {msg}")
                
    if target_denorm is not None:
        phys = evaluate_accuracy(model, test_loader, device, target_denorm=target_denorm)
        for k, v in phys.items():
            metrics[f"{k}_physical"] = float(v)
        metrics["target_offset"] = float(target_denorm[0])
        metrics["target_scale"] = float(target_denorm[1])

    if report_physical and prediction_denorm is not None and target_denorm is not None:
        separate = _evaluate_metrics(
            model,
            test_loader,
            device,
            pred_denorm=prediction_denorm,
            target_denorm=target_denorm,
        )
        for k, v in separate.items():
            metrics[f"{k}_physical_separate_denorm"] = float(v)
        metrics["prediction_offset"] = float(prediction_denorm[0])
        metrics["prediction_scale"] = float(prediction_denorm[1])

    unsafe_metrics = {
        "mae",
        "rmse",
        "rel_l2",
        "max_error",
        "mae_physical",
        "rmse_physical",
        "rel_l2_physical",
        "max_error_physical",
    }
    if prediction_signature is None and numerator_metric in unsafe_metrics:
        msg = (
            "Could not verify prediction-side normalization signature. "
            "Set normalization.prediction_denorm_path or keep numerator_metric on *_physical_separate_denorm."
        )
        if mismatch_action == "fail":
            raise ValueError(msg)
        print(f"[eval_emulator_superiority][warn] {msg}")

    if signature_mismatch:
        msg = (
            "Cross-solver normalization mismatch detected between checkpoint-train signature "
            f"{prediction_signature} and eval-target signature {eval_signature}. "
            "Use dedicated cross-solver processed datasets or use *_physical_separate_denorm metrics."
        )
        if numerator_metric in unsafe_metrics:
            if mismatch_action == "fail":
                raise ValueError(msg)
            print(f"[eval_emulator_superiority][warn] {msg}")

    if numerator_metric.endswith("_physical_separate_denorm") and numerator_metric not in metrics:
        raise ValueError(
            f"Requested numerator_metric '{numerator_metric}' requires both prediction and target denormalization stats. "
            "Check normalization.prediction_denorm_path / normalization.target_denorm_path and ensure eval_dataset.npz "
            "contains target_mean + target_std."
        )

    with solver_compare_path.open("r", encoding="utf-8") as f:
        solver_payload = json.load(f)

    solver_mean = _solver_metric_mean(solver_payload, denominator_metric)

    if numerator_metric not in metrics:
        raise KeyError(
            f"numerator_metric '{numerator_metric}' not found in model metrics. "
            f"Available={sorted(metrics.keys())}"
        )
    numerator = float(metrics[numerator_metric])
    ratio = float(numerator / solver_mean) if abs(solver_mean) > 0 else float("inf")

    out = {
        "evaluation_type": "emulator_superiority_ratio",
        "model_config": model_cfg_path,
        "checkpoint": checkpoint_path,
        "model_metrics": metrics,
        "solver_compare_path": str(solver_compare_path),
        "solver_denominator_metric": denominator_metric,
        "solver_denominator_mean": solver_mean,
        "emulator_numerator_metric": numerator_metric,
        "emulator_numerator_value": numerator,
        "ratio": ratio,
        "normalization": {
            "dataset_signature": eval_signature,
            "prediction_signature": prediction_signature,
            "signature_mismatch": bool(signature_mismatch),
            "prediction_denorm_path": prediction_denorm_path,
            "target_denorm_path": target_denorm_path,
            "checkpoint_train_path": checkpoint_train_path,
            "tolerance": signature_tol,
            "on_signature_mismatch": mismatch_action,
            "check_input_stats": bool(check_input_stats),
            "on_input_stats_mismatch": input_mismatch_action,
            "prediction_input_signature": prediction_input_signature,
            "eval_input_signature": eval_input_signature,
            "input_check": input_check,
        },
        "interpretation": "ratio < 1 means emulator error is lower than solver-A vs solver-B disagreement.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, output_path)
    print(out)


if __name__ == "__main__":
    main()
