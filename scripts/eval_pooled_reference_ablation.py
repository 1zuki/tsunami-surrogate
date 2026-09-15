#!/usr/bin/env python
"""Evaluate paired R6 pooled-reference ablation checkpoints in physical units."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import _make_dataset
from src.evaluation.normalization_bridge import (
    EvaluationNormalizationBridge,
    load_input_order,
    load_standardization_spec,
)
from src.evaluation.target_scaling import load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.hashing import sha256_file
from src.utils.io import save_json
from src.utils.seed import seed_everything
from scripts.eval_suite_preflight import _expected_times, load_suite_contract


SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")


def _canonical_solver(name: str) -> str:
    aliases = {
        "swe_hydrostatic": "hydrostatic",
        "hydrostatic": "hydrostatic",
        "swe_muscl": "muscl_hr",
        "swe_muscl_hr": "muscl_hr",
        "muscl": "muscl_hr",
        "muscl_hr": "muscl_hr",
        "boussinesq": "boussinesq",
    }
    value = str(name).strip().lower()
    return aliases.get(value, value)


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output.get("mean", next(iter(output.values())))
    return output


def _parse_spec(raw: str) -> tuple[str, str, str, str, str, str | None]:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) not in {4, 5, 6} or any(not part for part in parts[:4]):
        raise ValueError(
            "Model specs must use LABEL|CONFIG|CHECKPOINT|STATS_PATH"
            "[|TARGET_SOLVER[|SOLVER_ID]]."
        )
    target_solver = "hydrostatic" if len(parts) == 4 else _canonical_solver(parts[4])
    conditioned_solver = None if len(parts) < 6 else _canonical_solver(parts[5])
    return parts[0], parts[1], parts[2], parts[3], target_solver, conditioned_solver


def _stats_path_for_dataset(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    candidates = (
        path / "normalization_stats.json",
        path.parent / "normalization_stats.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find normalization_stats.json for {dataset_path}"
    )


def _dataset_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path) / "shards_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object manifest in {manifest_path}")
    return payload


def _source_contract_hash(path: str | Path) -> str:
    manifest = _dataset_manifest(path)
    provenance = manifest.get("provenance")
    lineage = provenance.get("source_lineage") if isinstance(provenance, Mapping) else None
    value = lineage.get("contract_hash") if isinstance(lineage, Mapping) else None
    if not value:
        raise ValueError(f"Dataset lacks common-time-v2 contract lineage: {path}")
    return str(value)


def _input_order(path: str | Path) -> list[str]:
    manifest = _dataset_manifest(path)
    values = manifest.get("input_order")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Dataset lacks input_order: {path}")
    return [str(value) for value in values]


class _PairedReferenceDataset(Dataset):
    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        input_stats: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
        input_order: list[str] | None = None,
    ) -> None:
        self.datasets = dict(datasets)
        self.input_stats = input_stats
        self.input_order = list(input_order or ["bathymetry", "source", "initial_depth"])
        lengths = {name: len(dataset) for name, dataset in self.datasets.items()}
        if set(self.datasets) != set(SOLVERS) or len(set(lengths.values())) != 1:
            raise ValueError(f"Reference datasets must be complete and equally sized: {lengths}")
        self.length = int(next(iter(lengths.values())))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = {name: dataset[index] for name, dataset in self.datasets.items()}
        reference = rows["hydrostatic"]
        scenario_id = str(reference["scenario_id"])
        x = reference["x"]
        for name, row in rows.items():
            if str(row["scenario_id"]) != scenario_id:
                raise ValueError(
                    f"Scenario roster mismatch at index {index}: {name}"
                )
            candidate = row["x"]
            reference = x
            if self.input_stats is not None:
                candidate_channels = []
                reference_channels = []
                # The paired input tensors carry the same channel ordering in
                # all accepted v2 roots; rebase each channel to physical units.
                for channel_index, channel in enumerate(self.input_order):
                    ref_offset, ref_scale = self.input_stats["hydrostatic"][channel]
                    cand_offset, cand_scale = self.input_stats[name][channel]
                    reference_channels.append(
                        reference[:, channel_index] * ref_scale + ref_offset
                    )
                    candidate_channels.append(
                        candidate[:, channel_index] * cand_scale + cand_offset
                    )
                reference = torch.stack(reference_channels, dim=0)
                candidate = torch.stack(candidate_channels, dim=0)
            if not torch.allclose(candidate, reference, atol=1.0e-6, rtol=0.0):
                raise ValueError(f"Input identity mismatch at index {index}: {name}")
        return {
            "x": x,
            "scenario_id": scenario_id,
            **{f"y_{name}": rows[name]["y"] for name in SOLVERS},
        }



def _bootstrap_rmse(values: list[float], seed: int, resamples: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty metric")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(resamples), array.size))
    samples = np.sqrt(np.mean(array[indices], axis=1))
    return {
        "global_field_rmse": float(math.sqrt(float(np.mean(array)))),
        "ci_lower": float(np.percentile(samples, 2.5)),
        "ci_upper": float(np.percentile(samples, 97.5)),
    }


def _phase_amplitude_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[float], list[float]]:
    pred_peak = prediction.abs().amax(dim=(1, 2, 3))
    target_peak = target.abs().amax(dim=(1, 2, 3))
    amplitude = (pred_peak - target_peak).abs() / (target_peak + 1.0e-12)
    pred_time = prediction.abs().mean(dim=(2, 3)).argmax(dim=1)
    target_time = target.abs().mean(dim=(2, 3)).argmax(dim=1)
    phase = (pred_time - target_time).abs()
    return amplitude.cpu().tolist(), phase.cpu().tolist()


def _mean_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _domain_arrival_time(
    values: np.ndarray,
    times: np.ndarray,
    threshold_fraction: float = 0.10,
) -> float | None:
    if values.ndim != 3 or times.ndim != 1 or values.shape[0] != times.size:
        raise ValueError("arrival diagnostic expects [T,H,W] values and [T] times")
    amplitude = np.max(np.abs(values), axis=(1, 2))
    peak = float(np.max(amplitude))
    if peak <= 1.0e-12:
        return None
    indices = np.flatnonzero(amplitude >= float(threshold_fraction) * peak)
    return float(times[int(indices[0])]) if indices.size else None


def _arrival_time_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    times: np.ndarray,
) -> tuple[list[float], int]:
    errors: list[float] = []
    missing = 0
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for pred_values, target_values in zip(prediction_np, target_np):
        pred_time = _domain_arrival_time(pred_values, times)
        target_time = _domain_arrival_time(target_values, times)
        if pred_time is None or target_time is None:
            missing += 1
        else:
            errors.append(abs(pred_time - target_time))
    return errors, missing


def _load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = load_config(config_path)
    model = build_model(cfg).to(device).eval()
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    return model, checkpoint


def _solver_channel_value(solver: str) -> float:
    return float(SOLVERS.index(_canonical_solver(solver)))


def _model_prediction(
    *,
    model: torch.nn.Module,
    x_reference: torch.Tensor,
    bridge: EvaluationNormalizationBridge,
    target: torch.Tensor,
    conditioned_solver: str | None,
) -> torch.Tensor:
    x_model, _ = bridge.transform(x_reference, target)
    if conditioned_solver is not None:
        if x_model.shape[1] != 3:
            raise ValueError(
                "Conditioned pooled model requires three physical input channels "
                "before appending solver ID"
            )
        solver_channel = torch.full(
            (x_model.shape[0], 1, x_model.shape[2], x_model.shape[3]),
            _solver_channel_value(conditioned_solver),
            dtype=x_model.dtype,
            device=x_model.device,
        )
        x_model = torch.cat([x_model, solver_channel], dim=1)
        if x_model.shape[1] != 4:
            raise AssertionError("conditioned solver-ID channel construction failed")
    elif x_model.shape[1] != 3:
        raise ValueError(
            "Anonymous pooled model must have exactly three input channels"
        )
    prediction = _model_output(model, x_model)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    offset, scale = bridge.model_target_denorm
    return prediction * float(scale) + float(offset)


def _physical_bathymetry(
    x_reference: torch.Tensor,
    *,
    input_order: list[str],
    input_stats: Mapping[str, tuple[float, float]],
) -> torch.Tensor:
    try:
        index = input_order.index("bathymetry")
        offset, scale = input_stats["bathymetry"]
    except (ValueError, KeyError) as exc:
        raise ValueError("Reference inputs must include bathymetry statistics") from exc
    return x_reference[:, index] * float(scale) + float(offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dataset", action="append", required=True, help="SOLVER|PATH")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract", default="configs/eval/final_v2_suite.yaml")
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    reference_paths: dict[str, str] = {}
    for raw in args.reference_dataset:
        parts = raw.split("|", 1)
        if len(parts) != 2:
            raise ValueError("--reference-dataset must use SOLVER|PATH")
        solver = _canonical_solver(parts[0])
        if solver not in SOLVERS or solver in reference_paths:
            raise ValueError(f"Invalid or duplicate reference solver: {solver!r}")
        reference_paths[solver] = parts[1]
    if set(reference_paths) != set(SOLVERS):
        raise ValueError(f"Exactly {SOLVERS} reference datasets are required")

    contract = load_suite_contract(args.contract)
    expected_contract = str(contract["scientific_scope"]["contract_hash"])
    observed_hashes = {solver: _source_contract_hash(path) for solver, path in reference_paths.items()}
    if set(observed_hashes.values()) != {expected_contract}:
        raise ValueError(
            f"Reference dataset contract mismatch: expected={expected_contract}, observed={observed_hashes}"
        )
    reference_orders = {solver: _input_order(path) for solver, path in reference_paths.items()}
    if len({tuple(order) for order in reference_orders.values()}) != 1:
        raise ValueError(f"Reference input-order mismatch: {reference_orders}")

    specs = [_parse_spec(raw) for raw in args.model]
    labels = [label for label, *_ in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Model labels must be unique")
    for label, _, _, stats_path, target_solver, conditioned_solver in specs:
        if target_solver not in SOLVERS:
            raise ValueError(f"Model {label!r} uses unknown target solver {target_solver!r}")
        if conditioned_solver is not None and conditioned_solver not in SOLVERS:
            raise ValueError(f"Model {label!r} uses unknown solver ID {conditioned_solver!r}")
        if not Path(stats_path).is_file():
            raise FileNotFoundError(stats_path)

    seed_everything(42)
    device = resolve_device(args.device)
    datasets = {solver: _make_dataset(path) for solver, path in reference_paths.items()}
    input_stats_by_solver = {
        solver: load_standardization_spec(_stats_path_for_dataset(path)).inputs
        for solver, path in reference_paths.items()
    }
    denorm = {solver: load_target_denorm(path) for solver, path in reference_paths.items()}
    if any(value is None for value in denorm.values()):
        raise ValueError("Reference datasets must carry normalized target statistics")
    loader = DataLoader(
        _PairedReferenceDataset(
            datasets,
            input_stats=input_stats_by_solver,
            input_order=reference_orders["hydrostatic"],
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    loaded_models: dict[str, tuple[torch.nn.Module, EvaluationNormalizationBridge, str, str | None, dict[str, Any]]] = {}
    input_order = load_input_order(reference_paths["hydrostatic"])
    input_stats = load_standardization_spec(
        _stats_path_for_dataset(reference_paths["hydrostatic"])
    )
    for label, config_path, checkpoint_path, stats_path, target_solver, conditioned_solver in specs:
        model, checkpoint = _load_model(config_path, checkpoint_path, device)
        target_stats = load_standardization_spec(
            _stats_path_for_dataset(reference_paths[target_solver])
        )
        model_stats = load_standardization_spec(stats_path)
        source_stats = type(input_stats)(
            path=input_stats.path,
            inputs=input_stats.inputs,
            target=target_stats.target,
            target_variable=target_stats.target_variable,
        )
        bridge = EvaluationNormalizationBridge(source_stats, model_stats, input_order)
        expected_channels = 4 if conditioned_solver is not None else 3
        if int(model._tsunami_model_config_signature["in_channels"]) != expected_channels:
            raise ValueError(
                f"Model {label!r} declares "
                f"{model._tsunami_model_config_signature['in_channels']} input "
                f"channels; expected {expected_channels}"
            )
        loaded_models[label] = (model, bridge, target_solver, conditioned_solver, checkpoint)

    per_model: dict[str, dict[str, Any]] = {
        label: {
            "per_reference_mse": {solver: [] for solver in SOLVERS},
            "equal_solver_mse": [],
            "nearest_reference": Counter(),
            "nearest_reference_mse": [],
            "phase_frame_abs_error": {solver: [] for solver in SOLVERS},
            "peak_amplitude_relative_error": {solver: [] for solver in SOLVERS},
            "arrival_time_abs_error": {solver: [] for solver in SOLVERS},
            "arrival_missing_count": {solver: 0 for solver in SOLVERS},
            "max_abs_eta": 0.0,
            "min_depth": math.inf,
        }
        for label in labels
    }
    centroid_mse: list[float] = []
    centroid_per_reference: dict[str, list[float]] = {solver: [] for solver in SOLVERS}
    solver_gap_mse: dict[str, list[float]] = {
        f"{left}_to_{right}": []
        for index, left in enumerate(SOLVERS)
        for right in SOLVERS[index + 1 :]
    }
    scenario_count = 0

    times = _expected_times(contract)
    with torch.no_grad():
        for batch in loader:
            x_reference = batch["x"].to(device)
            targets = {
                solver: batch[f"y_{solver}"].to(device) * float(denorm[solver][1])
                + float(denorm[solver][0])
                for solver in SOLVERS
            }
            bathymetry = _physical_bathymetry(
                x_reference,
                input_order=input_order,
                input_stats=input_stats.inputs,
            )
            centroid = sum(targets.values()) / float(len(SOLVERS))
            scenario_count += int(x_reference.shape[0])
            for solver in SOLVERS:
                centroid_per_reference[solver].extend(
                    (centroid - targets[solver]).square().mean(dim=(1, 2, 3)).cpu().tolist()
                )
            centroid_mse.extend(
                sum(
                    (centroid - targets[solver]).square().mean(dim=(1, 2, 3))
                    for solver in SOLVERS
                ).div(float(len(SOLVERS))).cpu().tolist()
            )
            for key in solver_gap_mse:
                left, right = key.split("_to_")
                solver_gap_mse[key].extend(
                    (targets[left] - targets[right]).square().mean(dim=(1, 2, 3)).cpu().tolist()
                )

            for label, (model, bridge, target_solver, conditioned_solver, _) in loaded_models.items():
                prediction = _model_prediction(
                    model=model,
                    x_reference=x_reference,
                    bridge=bridge,
                    target=batch[f"y_{target_solver}"].to(device),
                    conditioned_solver=conditioned_solver,
                )
                if not bool(torch.isfinite(prediction).all().item()):
                    raise FloatingPointError(f"Nonfinite physical output from {label!r}")
                per_model[label]["max_abs_eta"] = max(
                    float(per_model[label]["max_abs_eta"]),
                    float(prediction.abs().amax().cpu()),
                )
                per_model[label]["min_depth"] = min(
                    float(per_model[label]["min_depth"]),
                    float((prediction - bathymetry[:, None]).amin().cpu()),
                )
                reference_mse = []
                for solver in SOLVERS:
                    mse = (prediction - targets[solver]).square().mean(dim=(1, 2, 3))
                    per_model[label]["per_reference_mse"][solver].extend(mse.cpu().tolist())
                    reference_mse.append(mse)
                    amplitude, phase = _phase_amplitude_error(prediction, targets[solver])
                    per_model[label]["peak_amplitude_relative_error"][solver].extend(amplitude)
                    per_model[label]["phase_frame_abs_error"][solver].extend(phase)
                    arrival_errors, arrival_missing = _arrival_time_errors(
                        prediction,
                        targets[solver],
                        times,
                    )
                    per_model[label]["arrival_time_abs_error"][solver].extend(
                        arrival_errors
                    )
                    per_model[label]["arrival_missing_count"][solver] += int(
                        arrival_missing
                    )
                stacked = torch.stack(reference_mse, dim=1)
                per_model[label]["equal_solver_mse"].extend(stacked.mean(dim=1).cpu().tolist())
                nearest_values, nearest_indices = stacked.min(dim=1)
                per_model[label]["nearest_reference_mse"].extend(nearest_values.cpu().tolist())
                per_model[label]["nearest_reference"].update(
                    SOLVERS[int(index)] for index in nearest_indices.cpu().tolist()
                )

    result_models: dict[str, Any] = {}
    for model_index, (label, config_path, checkpoint_path, stats_path, target_solver, conditioned_solver) in enumerate(specs):
        rows = per_model[label]
        result_models[label] = {
            "config": config_path,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_normalization_stats": stats_path,
            "model_normalization_stats_sha256": sha256_file(stats_path),
            "target_solver_for_input_rebasing": target_solver,
            "conditioned_solver": conditioned_solver,
            "per_reference_rmse": {
                solver: _bootstrap_rmse(values, args.bootstrap_seed + 100 * model_index + index, args.bootstrap_resamples)
                for index, (solver, values) in enumerate(rows["per_reference_mse"].items())
            },
            "equal_solver_rmse": _bootstrap_rmse(rows["equal_solver_mse"], args.bootstrap_seed + 100 * model_index + 10, args.bootstrap_resamples),
            "nearest_reference_distance_rmse": _bootstrap_rmse(rows["nearest_reference_mse"], args.bootstrap_seed + 100 * model_index + 11, args.bootstrap_resamples),
            "nearest_reference_identity_frequency": {
                solver: {
                    "count": int(rows["nearest_reference"][solver]),
                    "fraction": float(rows["nearest_reference"][solver] / max(1, scenario_count)),
                }
                for solver in SOLVERS
            },
            "phase_frame_abs_error": {
                solver: _mean_summary(values)
                for solver, values in rows["phase_frame_abs_error"].items()
            },
            "peak_amplitude_relative_error": {
                solver: _mean_summary(values)
                for solver, values in rows["peak_amplitude_relative_error"].items()
            },
            "arrival_time_abs_error": {
                solver: _mean_summary(values)
                for solver, values in rows["arrival_time_abs_error"].items()
            },
            "arrival_missing_prediction_or_target_count": {
                solver: int(value)
                for solver, value in rows["arrival_missing_count"].items()
            },
            "physical_health": {
                "outputs_finite": True,
                "max_abs_eta": float(rows["max_abs_eta"]),
                "min_depth": float(rows["min_depth"]),
                "depth_definition": "eta - bathymetry on the paired physical benchmark grid",
            },
        }

    result = {
        "schema_id": "tsunami-surrogate.pooled-reference-ablation-evaluation.v1",
        "evaluation_type": "pooled_reference_ablation",
        "interpretation_boundary": (
            "All quantities are paired, physical-space benchmark discrepancies. "
            "They do not identify physical truth, numerical fidelity, or physical superiority."
        ),
        "contract_hash": expected_contract,
        "common_time_v2": {
            "requested_times": times.tolist(),
            "frame_count": int(times.size),
            "horizon": float(times[-1]),
        },
        "reference_datasets": reference_paths,
        "reference_dataset_contract_hashes": observed_hashes,
        "reference_input_order": reference_orders["hydrostatic"],
        "reference_normalization_stats": {
            solver: str(_stats_path_for_dataset(path))
            for solver, path in reference_paths.items()
        },
        "num_paired_scenarios": int(scenario_count),
        "bootstrap": {
            "seed": int(args.bootstrap_seed),
            "resamples": int(args.bootstrap_resamples),
            "confidence_level": 0.95,
        },
        "oracle_centroid": {
            "definition": "Per-scenario arithmetic mean of all three physical-space reference trajectories.",
            "equal_solver_rmse": _bootstrap_rmse(centroid_mse, args.bootstrap_seed + 1, args.bootstrap_resamples),
            "per_reference_rmse": {
                solver: _bootstrap_rmse(values, args.bootstrap_seed + 10 + index, args.bootstrap_resamples)
                for index, (solver, values) in enumerate(centroid_per_reference.items())
            },
        },
        "solver_gap_rmse": {
            key: _bootstrap_rmse(values, args.bootstrap_seed + 20 + index, args.bootstrap_resamples)
            for index, (key, values) in enumerate(solver_gap_mse.items())
        },
        "models": result_models,
    }
    save_json(result, args.output)
    print(f"[pooled-reference-ablation] paired_scenarios={scenario_count} -> {args.output}")


if __name__ == "__main__":
    main()
