#!/usr/bin/env python
"""Evaluate direct, seeded-window, or ensemble metrics on v2 metadata slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_window_suites import _evaluate_suite
from src.data.dataset import _make_dataset
from src.evaluation.calibration import interval_calibration
from src.evaluation.target_scaling import load_target_denorm
from src.evaluation.uncertainty import ErrorUncertaintyCorrelationAccumulator
from src.models import build_model
from src.models.ensemble import EnsemblePredictor
from src.training.checkpointing import load_checkpoint
from src.training.metrics import MetricAccumulator, compute_metrics
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything


def _parse_suite(raw: str) -> tuple[str, str, Any]:
    if "=" not in raw or ":" not in raw:
        raise ValueError(
            "Suite must use LABEL=FILTER:VALUE, for example "
            "source_holdout_multi_gauss=source_type_in:multi-gauss"
        )
    label, expression = raw.split("=", 1)
    key, value = expression.split(":", 1)
    label = label.strip()
    key = key.strip()
    value = value.strip()
    if not label or not key or not value:
        raise ValueError(f"Invalid suite expression: {raw!r}")
    if key in {
        "source_type_in",
        "source_type_not_in",
        "bathymetry_type_in",
        "bathymetry_type_not_in",
    }:
        parsed: Any = [item.strip() for item in value.split(",") if item.strip()]
        if not parsed:
            raise ValueError(f"Suite filter has no values: {raw!r}")
    elif key in {"source_strength_min", "source_strength_max"}:
        parsed = float(value)
    else:
        raise ValueError(f"Unsupported v2 slice filter: {key}")
    return label, key, parsed


def _select_indices(dataset: Any, key: str, value: Any) -> list[int]:
    item_key = (
        key.removesuffix("_in")
        .removesuffix("_not")
        .removesuffix("_min")
        .removesuffix("_max")
    )
    indices: list[int] = []
    for index in range(len(dataset)):
        item = dataset[index]
        observed = item.get(item_key, "unknown")
        if key.endswith("_in"):
            selected = str(observed) in {str(v) for v in value}
        elif key.endswith("_not_in"):
            selected = str(observed) not in {str(v) for v in value}
        elif key.endswith("_min"):
            selected = float(observed) >= float(value)
        elif key.endswith("_max"):
            selected = float(observed) <= float(value)
        else:
            selected = False
        if selected:
            indices.append(index)
    return indices


def _loader(dataset: Any, indices: Iterable[int], batch_size: int) -> DataLoader:
    subset = Subset(dataset, [int(index) for index in indices])
    if len(subset) == 0:
        raise ValueError("The requested v2 slice contains zero samples")
    return DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=0)


def _model_from_checkpoint(
    config_path: str, checkpoint_path: str, device: torch.device
):
    cfg = load_config(config_path)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return cfg, model


def _ensemble_from_checkpoints(
    config_path: str,
    checkpoints: list[str],
    device: torch.device,
) -> EnsemblePredictor:
    members = []
    for checkpoint in checkpoints:
        try:
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        cfg = payload.get("config") if isinstance(payload, dict) else None
        cfg = cfg if isinstance(cfg, dict) else load_config(config_path)
        cfg["device"] = str(device)
        model = build_model(cfg).to(device)
        load_checkpoint(checkpoint, model, map_location=device)
        members.append(model)
    return EnsemblePredictor(members).to(device).eval()


@torch.no_grad()
def _ensemble_metrics(
    ensemble: EnsemblePredictor,
    loader: DataLoader,
    device: torch.device,
    denorm: tuple[float, float] | None,
) -> dict[str, float]:
    levels = [0.5, 0.8, 0.9, 0.95]
    sums: dict[str, float] = {}
    count = 0
    correlation = ErrorUncertaintyCorrelationAccumulator()
    physical_correlation = ErrorUncertaintyCorrelationAccumulator()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        output = ensemble(x)
        row = interval_calibration(output["mean"], output["variance"], y, levels)
        correlation.update(output["mean"], output["variance"], y)
        if denorm is not None:
            offset, scale = denorm
            mean = output["mean"] * float(scale) + float(offset)
            target = y * float(scale) + float(offset)
            variance = output["variance"] * float(scale * scale)
            physical = interval_calibration(mean, variance, target, levels)
            physical_correlation.update(mean, variance, target)
            row.update(
                {f"{key}_physical": float(value) for key, value in physical.items()}
            )
        batch_size = int(x.shape[0])
        count += batch_size
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    if count <= 0:
        raise ValueError("Ensemble slice loader was empty")
    result = {key: value / float(count) for key, value in sums.items()}
    result["error_uncertainty_corr"] = correlation.compute()
    if denorm is not None:
        result["error_uncertainty_corr_physical"] = physical_correlation.compute()
    result["num_samples"] = int(count)
    return result


def _group_key(filter_key: str) -> str:
    if "bathymetry" in filter_key:
        return "bathymetry_type"
    return "source_type"


@torch.no_grad()
def _direct_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    denorm: tuple[float, float] | None,
) -> dict[str, dict[str, float]]:
    sample_sums = {
        "mae": 0.0,
        "rmse": 0.0,
        "rel_l2": 0.0,
        "max_error": 0.0,
    }
    physical_sample_sums = dict.fromkeys(sample_sums, 0.0)
    global_metrics = MetricAccumulator()
    global_physical_metrics = MetricAccumulator() if denorm is not None else None
    count = 0
    for batch in loader:
        x = batch["x"].to(device)
        target = batch["y"].to(device)
        prediction = model(x)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        elif isinstance(prediction, dict):
            prediction = prediction.get("mean", next(iter(prediction.values())))
        global_metrics.update(prediction, target)
        physical_prediction = None
        physical_target = None
        if denorm is not None:
            offset, scale = denorm
            physical_prediction = prediction * float(scale) + float(offset)
            physical_target = target * float(scale) + float(offset)
            assert global_physical_metrics is not None
            global_physical_metrics.update(
                physical_prediction,
                physical_target,
            )
        for index in range(int(x.shape[0])):
            metrics = compute_metrics(
                prediction[index : index + 1],
                target[index : index + 1],
            )
            for key in sample_sums:
                sample_sums[key] += float(metrics[key])
            if physical_prediction is not None and physical_target is not None:
                physical = compute_metrics(
                    physical_prediction[index : index + 1],
                    physical_target[index : index + 1],
                )
                for key in physical_sample_sums:
                    physical_sample_sums[key] += float(physical[key])
            count += 1
    if count <= 0:
        raise ValueError("The requested v2 slice contains zero samples")
    return {
        "metrics": {key: value / float(count) for key, value in sample_sums.items()},
        "metrics_physical": (
            {key: value / float(count) for key, value in physical_sample_sums.items()}
            if denorm is not None
            else {}
        ),
        "global_field_metrics": global_metrics.compute(),
        "global_field_metrics_physical": (
            global_physical_metrics.compute()
            if global_physical_metrics is not None
            else {}
        ),
    }


def _group_rows(
    model: torch.nn.Module,
    dataset: Any,
    group_key: str,
    device: torch.device,
    batch_size: int,
    denorm: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    labels = sorted(
        {str(dataset[index].get(group_key, "unknown")) for index in range(len(dataset))}
    )
    rows: list[dict[str, Any]] = []
    for label in labels:
        indices = [
            index
            for index in range(len(dataset))
            if str(dataset[index].get(group_key, "unknown")) == label
        ]
        loader = _loader(dataset, indices, batch_size)
        metrics = _direct_metrics(model, loader, device, denorm)
        rows.append(
            {
                "label": label,
                "group_key": group_key,
                "diagnostic_kind": "metadata_subgroup",
                "num_samples": int(len(indices)),
                **metrics,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--ensemble-checkpoint", action="append", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--group-by", choices=["source_type", "bathymetry_type"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.suite and not args.group_by:
        raise ValueError("Provide at least one --suite or --group-by")
    if args.group_by and args.window:
        raise ValueError("--group-by cannot be combined with --window")
    if args.checkpoint is None and not args.ensemble_checkpoint:
        raise ValueError(
            "Provide --checkpoint or at least two --ensemble-checkpoint values"
        )
    if args.checkpoint is not None and args.ensemble_checkpoint:
        raise ValueError("Use either --checkpoint or --ensemble-checkpoint, not both")
    if args.ensemble_checkpoint and len(args.ensemble_checkpoint) < 2:
        raise ValueError("Ensemble slice evaluation requires at least two members")

    cfg = load_config(args.config)
    cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)
    dataset = _make_dataset(args.dataset)
    denorm = load_target_denorm(args.dataset)
    parsed_suites = [_parse_suite(raw) for raw in (args.suite or [])]

    if args.ensemble_checkpoint:
        ensemble = _ensemble_from_checkpoints(
            args.config,
            [str(path) for path in args.ensemble_checkpoint],
            device,
        )
        if args.group_by:
            raise ValueError(
                "--group-by is currently supported for direct model metrics only"
            )
        rows = []
        for label, key, value in parsed_suites:
            indices = _select_indices(dataset, key, value)
            metrics = _ensemble_metrics(
                ensemble,
                _loader(dataset, indices, args.batch_size),
                device,
                denorm,
            )
            rows.append(
                {
                    "label": label,
                    "filter": {key: value},
                    "diagnostic_kind": "metadata_subgroup",
                    "dataset_path": str(args.dataset),
                    **metrics,
                }
            )
        result = {
            "evaluation_type": "v2_ensemble_slice_metrics",
            "config_path": str(args.config),
            "checkpoints": [str(path) for path in args.ensemble_checkpoint],
            "dataset_path": str(args.dataset),
            "diagnostic_kind": "metadata_subgroup",
            "rows": rows,
        }
    else:
        model_cfg, model = _model_from_checkpoint(
            args.config,
            str(args.checkpoint),
            device,
        )
        rows = []
        if args.group_by:
            rows = _group_rows(
                model,
                dataset,
                args.group_by,
                device,
                args.batch_size,
                denorm,
            )
        for label, key, value in parsed_suites:
            indices = _select_indices(dataset, key, value)
            loader = _loader(dataset, indices, args.batch_size)
            row = {
                "label": label,
                "filter": {key: value},
                "diagnostic_kind": "metadata_subgroup",
                "dataset_path": str(args.dataset),
                "num_samples": int(len(indices)),
            }
            if args.window:
                window_data = dict(model_cfg.get("data", {}))
                row["window_metrics"] = _evaluate_suite(
                    model,
                    loader,
                    device,
                    int(window_data.get("window_K", 5)),
                    bool(window_data.get("window_include_source", True)),
                    bool(window_data.get("window_prev", True)),
                    denorm,
                )
            else:
                row.update(_direct_metrics(model, loader, device, denorm))
            rows.append(row)
        result = {
            "evaluation_type": (
                "v2_slice_group_metrics"
                if args.group_by
                else ("v2_window_slice_metrics" if args.window else "v2_slice_metrics")
            ),
            "config_path": str(args.config),
            "checkpoint_path": str(args.checkpoint),
            "dataset_path": str(args.dataset),
            "group_by": args.group_by,
            "diagnostic_kind": "metadata_subgroup",
            "aggregation": (
                "seeded_window_rollout"
                if args.window
                else "mean_of_per_scenario_metrics"
            ),
            "seeded_with_true_first_frame": bool(args.window),
            "window_K": (
                int(model_cfg.get("data", {}).get("window_K", 5))
                if args.window
                else None
            ),
            "rows": rows,
        }

    if denorm is not None:
        result["target_offset"] = float(denorm[0])
        result["target_scale"] = float(denorm[1])
    save_json(result, args.output)
    print(f"[v2-slices] rows={len(result['rows'])} -> {args.output}")


if __name__ == "__main__":
    main()
