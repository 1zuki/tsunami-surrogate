#!/usr/bin/env python
"""Evaluate fixed virtual-gauge waveform, arrival, and peak metrics on v2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.target_scaling import load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything
from scripts.eval_suite_preflight import _expected_times, load_suite_contract


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output.get("mean", next(iter(output.values())))
    return output


def _plateau_time_to_peak(
    values: np.ndarray,
    times: np.ndarray,
    fraction: float,
) -> float:
    peak = float(np.max(values))
    trough = float(np.min(values))
    tolerance = max(1.0e-12, (1.0 - float(fraction)) * (peak - trough))
    indices = np.flatnonzero(values >= peak - tolerance)
    return float(np.mean(times[indices])) if indices.size else float(times[0])


def _arrival_time(
    values: np.ndarray,
    times: np.ndarray,
    threshold: float,
) -> float | None:
    indices = np.flatnonzero(np.abs(values) >= float(threshold))
    return float(times[indices[0]]) if indices.size else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gauge", action="append", required=True, help="ROW,COL")
    parser.add_argument("--arrival-threshold-fraction", type=float, default=0.10)
    parser.add_argument("--peak-plateau-fraction", type=float, default=0.99)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--contract",
        default="configs/eval/final_v2_suite.yaml",
        help="Evaluation-suite contract that defines the shared requested times.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gauges: list[tuple[int, int]] = []
    for raw in args.gauge:
        parts = raw.split(",")
        if len(parts) != 2:
            raise ValueError(f"Gauge must use ROW,COL, got {raw!r}")
        gauges.append((int(parts[0]), int(parts[1])))
    if not gauges:
        raise ValueError("At least one gauge is required")

    contract = load_suite_contract(args.contract)
    times = _expected_times(contract)
    cfg = load_config(args.config)
    cfg["device"] = args.device
    cfg["data"] = {
        "test_path": args.dataset,
        "batch_size": int(args.batch_size),
        "num_workers": 0,
    }
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)
    loaders = create_dataloaders(cfg)
    loader = loaders.get("test")
    if loader is None:
        raise KeyError("Could not build the v2 wave-metrics test loader")
    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    denorm = load_target_denorm(args.dataset)
    if denorm is None:
        raise ValueError("Wave metrics require normalized v2 targets with statistics")
    offset, scale = denorm
    per_gauge: dict[str, dict[str, Any]] = {
        f"{row},{col}": {
            "waveform_nrmse": [],
            "arrival_time_abs_error": [],
            "peak_elevation_abs_error": [],
            "peak_elevation_relative_error": [],
            "time_to_peak_abs_error": [],
            "arrival_eligible_target_count": 0,
            "arrival_compared_count": 0,
            "arrival_missing_prediction_count": 0,
            "arrival_ineligible_initially_active_count": 0,
            "arrival_inactive_target_count": 0,
            "peak_relative_ineligible_near_zero_count": 0,
        }
        for row, col in gauges
    }
    sample_count = 0
    arrival_ineligible_initially_active = 0
    arrival_missing_prediction = 0

    with torch.no_grad():
        for batch in loader:
            prediction = _model_output(model, batch["x"].to(device)).cpu().numpy()
            target = batch["y"].cpu().numpy()
            prediction = prediction * float(scale) + float(offset)
            target = target * float(scale) + float(offset)
            for batch_index in range(prediction.shape[0]):
                sample_count += 1
                for row, col in gauges:
                    if not (
                        0 <= row < prediction.shape[2]
                        and 0 <= col < prediction.shape[3]
                    ):
                        raise IndexError(
                            f"Gauge {(row, col)} is outside {prediction.shape[2:]}"
                        )
                    key = f"{row},{col}"
                    pred_values = prediction[batch_index, :, row, col]
                    target_values = target[batch_index, :, row, col]
                    target_amplitude = float(np.max(np.abs(target_values)))
                    if target_amplitude <= 1.0e-12:
                        per_gauge[key]["arrival_inactive_target_count"] += 1
                        per_gauge[key]["peak_relative_ineligible_near_zero_count"] += 1
                        continue
                    waveform_rmse = float(
                        np.sqrt(np.mean((pred_values - target_values) ** 2))
                    )
                    per_gauge[key]["waveform_nrmse"].append(
                        waveform_rmse / target_amplitude
                    )
                    pred_peak_elevation = float(np.max(pred_values))
                    target_peak_elevation = float(np.max(target_values))
                    peak_abs_error = abs(pred_peak_elevation - target_peak_elevation)
                    per_gauge[key]["peak_elevation_abs_error"].append(peak_abs_error)
                    peak_relative_floor = max(
                        1.0e-12,
                        1.0e-6 * target_amplitude,
                    )
                    if abs(target_peak_elevation) > peak_relative_floor:
                        per_gauge[key]["peak_elevation_relative_error"].append(
                            peak_abs_error / abs(target_peak_elevation)
                        )
                    else:
                        per_gauge[key]["peak_relative_ineligible_near_zero_count"] += 1
                    pred_peak_time = _plateau_time_to_peak(
                        pred_values,
                        times,
                        args.peak_plateau_fraction,
                    )
                    target_peak_time = _plateau_time_to_peak(
                        target_values,
                        times,
                        args.peak_plateau_fraction,
                    )
                    per_gauge[key]["time_to_peak_abs_error"].append(
                        abs(pred_peak_time - target_peak_time)
                    )
                    pred_arrival = _arrival_time(
                        pred_values,
                        times,
                        args.arrival_threshold_fraction * target_amplitude,
                    )
                    target_arrival = _arrival_time(
                        target_values,
                        times,
                        args.arrival_threshold_fraction * target_amplitude,
                    )
                    if target_arrival is None:
                        per_gauge[key]["arrival_inactive_target_count"] += 1
                    elif target_arrival <= float(times[0]):
                        arrival_ineligible_initially_active += 1
                        per_gauge[key]["arrival_ineligible_initially_active_count"] += 1
                    elif pred_arrival is None:
                        arrival_missing_prediction += 1
                        per_gauge[key]["arrival_eligible_target_count"] += 1
                        per_gauge[key]["arrival_missing_prediction_count"] += 1
                    else:
                        per_gauge[key]["arrival_eligible_target_count"] += 1
                        per_gauge[key]["arrival_compared_count"] += 1
                        per_gauge[key]["arrival_time_abs_error"].append(
                            abs(pred_arrival - target_arrival)
                        )

    aggregates: dict[str, Any] = {}
    for metric in (
        "waveform_nrmse",
        "arrival_time_abs_error",
        "peak_elevation_abs_error",
        "peak_elevation_relative_error",
        "time_to_peak_abs_error",
    ):
        aggregates[metric] = _summary(
            [value for row in per_gauge.values() for value in row[metric]]
        )

    result = {
        "evaluation_type": "v2_gauge_waveform_peak_metrics",
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "dataset_path": str(args.dataset),
        "num_samples": int(sample_count),
        "common_time_v2": {
            "requested_times": times.tolist(),
            "horizon": float(times[-1]),
            "frame_count": int(times.size),
        },
        "gauges": [{"row": int(row), "col": int(col)} for row, col in gauges],
        "arrival_threshold_fraction": float(args.arrival_threshold_fraction),
        "arrival_policy": (
            "shared absolute threshold equal to the configured fraction of "
            "the target-gauge peak; initially active target gauges are "
            "reported as ineligible"
        ),
        "arrival_ineligible_initially_active": int(arrival_ineligible_initially_active),
        "arrival_missing_prediction": int(arrival_missing_prediction),
        "peak_plateau_fraction": float(args.peak_plateau_fraction),
        "target_units": "physical",
        "aggregates": aggregates,
        "per_gauge": {
            key: {
                **{
                    metric: _summary(values)
                    for metric, values in metrics.items()
                    if isinstance(values, list)
                },
                "arrival_counts": {
                    name: int(value)
                    for name, value in metrics.items()
                    if name.startswith("arrival_") and name.endswith("_count")
                },
                "peak_relative_ineligible_near_zero_count": int(
                    metrics["peak_relative_ineligible_near_zero_count"]
                ),
            }
            for key, metrics in per_gauge.items()
        },
    }
    save_json(result, args.output)
    print(f"[v2-wave-metrics] samples={sample_count} -> {args.output}")


if __name__ == "__main__":
    main()
