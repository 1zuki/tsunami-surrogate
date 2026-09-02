#!/usr/bin/env python
"""Fresh common-time-v2 solver-gap and cross-reference analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import _make_dataset
from src.evaluation.target_scaling import load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.seed import seed_everything
from scripts.eval_suite_preflight import _expected_times, load_suite_contract


SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")


class _TriDataset(Dataset):
    def __init__(self, datasets: dict[str, Any]):
        self.datasets = datasets
        lengths = {name: len(dataset) for name, dataset in datasets.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Reference dataset lengths differ: {lengths}")
        self.length = int(next(iter(lengths.values())))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        items = {name: dataset[index] for name, dataset in self.datasets.items()}
        reference = items["hydrostatic"]
        for name, item in items.items():
            if str(item["scenario_id"]) != str(reference["scenario_id"]):
                raise ValueError(
                    "Scenario roster mismatch at index "
                    f"{index}: hydrostatic={reference['scenario_id']} "
                    f"{name}={item['scenario_id']}"
                )
            if not torch.allclose(item["x"], reference["x"], atol=1.0e-6, rtol=0.0):
                raise ValueError(f"Input identity mismatch at index {index} for {name}")
        return {
            "x": reference["x"],
            "scenario_id": str(reference["scenario_id"]),
            **{f"y_{name}": item["y"] for name, item in items.items()},
        }


class _Stats:
    def __init__(self) -> None:
        self.mse: list[float] = []
        self.mae: list[float] = []
        self.rmse: list[float] = []
        self.rel_l2: list[float] = []
        self.max_error: list[float] = []

    def add(self, left: torch.Tensor, right: torch.Tensor) -> None:
        diff = (left - right).to(torch.float64)
        target = right.to(torch.float64)
        dims = tuple(range(1, diff.ndim))
        mse = diff.square().mean(dim=dims).cpu().numpy()
        mae = diff.abs().mean(dim=dims).cpu().numpy()
        rel = (
            (
                torch.linalg.vector_norm(diff.flatten(start_dim=1), dim=1)
                / (
                    torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1)
                    + 1.0e-12
                )
            )
            .cpu()
            .numpy()
        )
        maximum = diff.abs().amax(dim=dims).cpu().numpy()
        self.mse.extend(float(value) for value in mse)
        self.mae.extend(float(value) for value in mae)
        self.rmse.extend(float(value) for value in np.sqrt(mse))
        self.rel_l2.extend(float(value) for value in rel)
        self.max_error.extend(float(value) for value in maximum)


def _bootstrap(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty metric")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(resamples), array.size))
    means = np.mean(array[indices], axis=1, dtype=np.float64)
    return {
        "mean": float(np.mean(array, dtype=np.float64)),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
    }


def _summary(stats: _Stats, seed: int, resamples: int) -> dict[str, Any]:
    global_rmse = math.sqrt(float(np.mean(np.asarray(stats.mse, dtype=np.float64))))
    return {
        "num_samples": len(stats.mse),
        "global_field_rmse": global_rmse,
        "scenario_mae_mean": _bootstrap(stats.mae, seed, resamples),
        "scenario_rmse_mean": _bootstrap(stats.rmse, seed + 1, resamples),
        "scenario_rel_l2_mean": _bootstrap(stats.rel_l2, seed + 2, resamples),
        "scenario_max_error_mean": _bootstrap(stats.max_error, seed + 3, resamples),
        "scenario_mse_mean": _bootstrap(stats.mse, seed + 4, resamples),
    }


def _parse_model(raw: str) -> tuple[str, str, str, str]:
    parts = raw.split("|")
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise ValueError(
            "Model must use SOLVER|CONFIG|CHECKPOINT|DATASET, for example "
            "hydrostatic|configs/model/fno.yaml|experiments/fno/best.pt|"
            "data/processed/hydrostatic/test"
        )
    return tuple(part.strip() for part in parts)  # type: ignore[return-value]


def _load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
):
    cfg = load_config(config_path)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device).eval()
    load_checkpoint(checkpoint_path, model, map_location=device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--contract",
        default="configs/eval/final_v2_suite.yaml",
        help="Evaluation-suite contract that defines the shared requested times.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Optional training-seed label for replicated reference analyses.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract = load_suite_contract(args.contract)
    requested_times = _expected_times(contract)

    specs = [_parse_model(raw) for raw in args.model]
    if len(specs) != len(SOLVERS):
        raise ValueError(
            f"Reference analysis requires exactly {len(SOLVERS)} model specs"
        )
    observed = {solver for solver, *_ in specs}
    if observed != set(SOLVERS):
        raise ValueError(
            f"Reference analysis requires exactly {SOLVERS}, got {sorted(observed)}"
        )
    spec_by_solver = {
        solver: (config, checkpoint, dataset)
        for solver, config, checkpoint, dataset in specs
    }

    seed_everything(42)
    device = resolve_device(args.device)
    datasets = {solver: _make_dataset(spec_by_solver[solver][2]) for solver in SOLVERS}
    denorm = {
        solver: load_target_denorm(spec_by_solver[solver][2]) for solver in SOLVERS
    }
    if any(value is None for value in denorm.values()):
        raise ValueError(
            "All v2 reference datasets require target denormalization statistics"
        )

    loader = DataLoader(
        _TriDataset(datasets),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )
    gap_stats = {
        (left, right): _Stats()
        for left in SOLVERS
        for right in SOLVERS
        if left != right
    }
    models = {
        solver: _load_model(
            spec_by_solver[solver][0],
            spec_by_solver[solver][1],
            device,
        )
        for solver in SOLVERS
    }
    numerator_stats = {
        (model_solver, target_solver): _Stats()
        for model_solver in SOLVERS
        for target_solver in SOLVERS
        if model_solver != target_solver
    }
    control_stats = {solver: _Stats() for solver in SOLVERS}
    numerator_mse = {key: [] for key in numerator_stats}
    denominator_mse = {key: [] for key in numerator_stats}
    scenario_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            scenario_count += int(x.shape[0])
            targets = {
                solver: batch[f"y_{solver}"].to(device) * float(denorm[solver][1])
                + float(denorm[solver][0])
                for solver in SOLVERS
            }
            predictions = {}
            for solver, model in models.items():
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                elif isinstance(output, dict):
                    output = output.get("mean", next(iter(output.values())))
                predictions[solver] = output * float(denorm[solver][1]) + float(
                    denorm[solver][0]
                )
                control_stats[solver].add(predictions[solver], targets[solver])

            for left in SOLVERS:
                for right in SOLVERS:
                    if left == right:
                        continue
                    gap_stats[(left, right)].add(targets[left], targets[right])
                    key = (left, right)
                    numerator_stats[key].add(predictions[left], targets[right])
                    numerator_diff = predictions[left].to(torch.float64) - targets[
                        right
                    ].to(torch.float64)
                    denominator_diff = targets[left].to(torch.float64) - targets[
                        right
                    ].to(torch.float64)
                    numerator_mse[key].extend(
                        float(value)
                        for value in numerator_diff.square()
                        .flatten(start_dim=1)
                        .mean(dim=1)
                        .cpu()
                    )
                    denominator_mse[key].extend(
                        float(value)
                        for value in denominator_diff.square()
                        .flatten(start_dim=1)
                        .mean(dim=1)
                        .cpu()
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_rows: list[dict[str, Any]] = []
    for left in SOLVERS:
        for right in SOLVERS:
            if left == right:
                continue
            gap_rows.append(
                {
                    "solver_a": left,
                    "solver_b": right,
                    "relative_l2_reference": right,
                    "metrics": _summary(
                        gap_stats[(left, right)],
                        args.bootstrap_seed,
                        args.bootstrap_resamples,
                    ),
                }
            )

    gap_result = {
        "evaluation_type": "v2_solver_gap",
        "training_seed": args.training_seed,
        "output_mode": "common_time_v2_processed",
        "common_time_v2": {
            "requested_times": requested_times.tolist(),
            "horizon": float(requested_times[-1]),
            "frame_count": int(requested_times.size),
        },
        "num_samples": int(scenario_count),
        "dataset_paths": {solver: spec_by_solver[solver][2] for solver in SOLVERS},
        "configs": [spec_by_solver[solver][0] for solver in SOLVERS],
        "checkpoints": [spec_by_solver[solver][1] for solver in SOLVERS],
        "bootstrap": {
            "seed": int(args.bootstrap_seed),
            "resamples": int(args.bootstrap_resamples),
            "confidence_level": 0.95,
        },
        "pairs": gap_rows,
    }
    save_json(gap_result, output_dir / "solver_gap.json")

    cross_rows: list[dict[str, Any]] = []
    for model_solver in SOLVERS:
        for target_solver in SOLVERS:
            if model_solver == target_solver:
                continue
            key = (model_solver, target_solver)
            numerator = numerator_stats[key]
            numerator_rmse = math.sqrt(float(np.mean(numerator_mse[key])))
            denominator_rmse = math.sqrt(float(np.mean(denominator_mse[key])))
            if denominator_rmse <= 0.0:
                raise ValueError(
                    "Cross-reference discrepancy ratio is undefined because "
                    f"the {model_solver}->{target_solver} solver-gap "
                    "denominator is zero"
                )
            rho = numerator_rmse / denominator_rmse
            rng = np.random.default_rng(args.bootstrap_seed + len(cross_rows))
            indices = rng.integers(
                0,
                len(numerator_mse[key]),
                size=(int(args.bootstrap_resamples), len(numerator_mse[key])),
            )
            numerator_samples = np.sqrt(
                np.mean(np.asarray(numerator_mse[key])[indices], axis=1)
            )
            denominator_samples = np.sqrt(
                np.mean(np.asarray(denominator_mse[key])[indices], axis=1)
            )
            if bool(np.any(denominator_samples <= 0.0)):
                raise ValueError(
                    "A paired bootstrap resample produced a zero "
                    f"{model_solver}->{target_solver} solver-gap denominator"
                )
            rho_samples = numerator_samples / denominator_samples
            cross_rows.append(
                {
                    "model_solver": model_solver,
                    "benchmark_solver": target_solver,
                    "num_samples": len(numerator_mse[key]),
                    "numerator": _summary(
                        numerator,
                        args.bootstrap_seed,
                        args.bootstrap_resamples,
                    ),
                    "denominator_solver_gap": _summary(
                        gap_stats[key],
                        args.bootstrap_seed + 1,
                        args.bootstrap_resamples,
                    ),
                    "same_reference_control": _summary(
                        control_stats[model_solver],
                        args.bootstrap_seed + 2,
                        args.bootstrap_resamples,
                    ),
                    "rho": {
                        "point_estimate": float(rho),
                        "ci_lower": float(np.percentile(rho_samples, 2.5)),
                        "ci_upper": float(np.percentile(rho_samples, 97.5)),
                    },
                    "interpretation": (
                        "cross-reference emulator discrepancy ratio; this is a "
                        "benchmark-relative diagnostic, not physical superiority"
                    ),
                }
            )

    cross_result = {
        "evaluation_type": "v2_cross_reference_discrepancy",
        "training_seed": args.training_seed,
        "output_mode": "common_time_v2_processed",
        "common_time_v2": gap_result["common_time_v2"],
        "num_samples": int(scenario_count),
        "dataset_paths": {solver: spec_by_solver[solver][2] for solver in SOLVERS},
        "configs": [spec_by_solver[solver][0] for solver in SOLVERS],
        "checkpoints": [spec_by_solver[solver][1] for solver in SOLVERS],
        "model_specs": {
            solver: {
                "config": spec_by_solver[solver][0],
                "checkpoint": spec_by_solver[solver][1],
                "dataset": spec_by_solver[solver][2],
            }
            for solver in SOLVERS
        },
        "bootstrap": gap_result["bootstrap"],
        "directions": cross_rows,
    }
    save_json(cross_result, output_dir / "cross_reference.json")
    print(f"[v2-reference-analysis] samples={scenario_count} -> {output_dir}")


if __name__ == "__main__":
    main()
