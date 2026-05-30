#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from typing import Any, Dict
import torch
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.models.ensemble import EnsemblePredictor
from src.training.checkpointing import load_checkpoint
from src.evaluation.calibration import interval_calibration
from src.evaluation.uncertainty import error_uncertainty_correlation
from src.evaluation.target_scaling import load_target_denorm, resolve_eval_dataset_path
from src.utils.io import save_json


def _evaluate_uncertainty_loader(
    ensemble: EnsemblePredictor,
    loader: Any,
    device: torch.device,
    levels: list[float],
    target_denorm: tuple[float, float] | None = None,
) -> Dict[str, float]:
    weighted_sums: Dict[str, float] = {}
    total_samples = 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        out = ensemble(x)
        row = interval_calibration(out["mean"], out["variance"], y, levels)
        row["error_uncertainty_corr"] = error_uncertainty_correlation(out["mean"], out["variance"], y)

        if target_denorm is not None:
            offset, scale = float(target_denorm[0]), float(target_denorm[1])
            mean_p = out["mean"] * scale + offset
            y_p = y * scale + offset
            var_p = out["variance"] * (scale * scale)
            row_physical = interval_calibration(mean_p, var_p, y_p, levels)
            row_physical["error_uncertainty_corr"] = error_uncertainty_correlation(mean_p, var_p, y_p)
            for key, value in row_physical.items():
                row[f"{key}_physical"] = float(value)

        n = int(x.shape[0])
        total_samples += n
        for k, v in row.items():
            weighted_sums[k] = weighted_sums.get(k, 0.0) + float(v) * n

    if total_samples <= 0:
        raise ValueError("test loader had zero batches")

    return {k: v / float(total_samples) for k, v in weighted_sums.items()}


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _build_suite_loader(cfg: Dict[str, Any], test_path: str, batch_size: int):
    local_cfg = dict(cfg)
    local_data = dict(local_cfg.get("data", {}))
    local_data["test_path"] = test_path
    local_data["batch_size"] = batch_size
    local_cfg["data"] = local_data
    loaders = create_dataloaders(local_cfg)
    test_loader = loaders.get("test")

    if test_loader is None:
        raise KeyError(f"No test dataloader could be built for suite path: {test_path}")
    n = _dataset_num_samples(test_loader)
    if n == 0:
        raise ValueError(f"Suite dataset has zero samples: {test_path}")
    validate_model_io_channels(local_cfg, loaders, preferred_splits=("test",))

    return test_loader


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument(
        '--checkpoint',
        action='append',
        default=None,
        help='Optional ensemble checkpoint path. Repeat this flag to pass multiple members.',
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    ckpts = list(cfg.get('uncertainty', {}).get('ensemble_checkpoints', []))
    if args.checkpoint:
        ckpts = [str(p) for p in args.checkpoint]
    if not ckpts:
        fallback = eval_cfg.get("checkpoint")
        if fallback:
            ckpts = [str(fallback)]

    if not ckpts:
        print(
            "No uncertainty checkpoints configured. "
            "Set uncertainty.ensemble_checkpoints, pass --checkpoint, or set eval.checkpoint."
        )
        return
    min_members = int(cfg.get("uncertainty", {}).get("min_ensemble_members", 2))
    if len(ckpts) < min_members:
        raise ValueError(
            f"Uncertainty evaluation requires at least {min_members} ensemble checkpoints, got {len(ckpts)}. "
            "Single-member runs produce degenerate ensemble variance."
        )

    data_cfg = dict(cfg.get("data", {}))
    dataset_cfg = cfg.get("dataset", {})
    if not data_cfg and isinstance(dataset_cfg, dict):
        dataset_path = dataset_cfg.get("path")
        if dataset_path:
            data_cfg["test_path"] = dataset_path
        if "batch_size" in dataset_cfg:
            data_cfg["batch_size"] = dataset_cfg["batch_size"]
    dataset_path = eval_cfg.get("dataset_path")
    if dataset_path:
        data_cfg["test_path"] = dataset_path
    if "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]
    cfg["data"] = data_cfg

    device = resolve_device(cfg.get('device', 'auto'))
    members = []

    for path in ckpts:
        model = build_model(torch.load(path, map_location='cpu').get('config', cfg)).to(device)
        load_checkpoint(path, model, map_location=device)
        members.append(model)

    ensemble = EnsemblePredictor(members).to(device).eval()
    uncertainty_cfg = cfg.get("uncertainty", {})
    levels = uncertainty_cfg.get("interval_levels", [0.5, 0.8, 0.9, 0.95])
    report_physical = bool(uncertainty_cfg.get("report_physical_metrics", True))
    suite_cfg = uncertainty_cfg.get("suites", eval_cfg.get("uncertainty", {}).get("suites", []))
    skip_empty_suites = bool(uncertainty_cfg.get("skip_empty_suites", True))

    if suite_cfg:
        suite_rows: Dict[str, Dict[str, float]] = {}
        skipped_labels: list[str] = []
        for i, suite in enumerate(list(suite_cfg)):
            suite_dict = suite if isinstance(suite, dict) else {}
            label = str(suite_dict.get("label", f"suite_{i}"))
            suite_path = str(suite_dict.get("path", "")).strip()
            if not suite_path:
                raise KeyError(f"uncertainty.suites[{i}] is missing required key: path")
            batch_size = int(
                suite_dict.get(
                    "batch_size",
                    eval_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 8)),
                )
            )
            try:
                loader = _build_suite_loader(cfg, suite_path, batch_size)
            except ValueError as e:
                if skip_empty_suites and "zero samples" in str(e):
                    print(f"[eval_uncertainty] skipping empty suite '{label}': {e}")
                    skipped_labels.append(label)
                    continue
                raise

            denorm = load_target_denorm(str(suite_path)) if report_physical else None
            row = _evaluate_uncertainty_loader(
                ensemble=ensemble,
                loader=loader,
                device=device,
                levels=levels,
                target_denorm=denorm,
            )
            row["num_samples"] = float(_dataset_num_samples(loader))
            if denorm is not None:
                row["target_offset"] = float(denorm[0])
                row["target_scale"] = float(denorm[1])
            suite_rows[label] = row

        if not suite_rows:
            skipped_txt = ", ".join(skipped_labels) if skipped_labels else "none"
            raise ValueError(
                "All configured uncertainty suites are empty after filtering, so evaluation cannot proceed. "
                f"Skipped suites: [{skipped_txt}]"
            )
        mean_results: Dict[str, Any] = {
            "evaluation_type": "ood_uncertainty_suites",
            "suites": suite_rows,
        }
        output_name = "uncertainty_ood.json"
    else:
        loaders = create_dataloaders(cfg)
        test_loader = loaders.get("test")
        if test_loader is None:
            print("No test loader found; uncertainty eval skipped gracefully.")
            return
        validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))
        denorm = None
        if report_physical:
            resolved_dataset_path = resolve_eval_dataset_path(cfg, split="test")
            if resolved_dataset_path is not None:
                denorm = load_target_denorm(resolved_dataset_path)
        mean_results = _evaluate_uncertainty_loader(
            ensemble=ensemble,
            loader=test_loader,
            device=device,
            levels=levels,
            target_denorm=denorm,
        )
        mean_results = {
            "evaluation_type": "in_distribution_uncertainty",
            "num_samples": float(_dataset_num_samples(test_loader)),
            **mean_results,
        }
        if denorm is not None:
            mean_results["target_offset"] = float(denorm[0])
            mean_results["target_scale"] = float(denorm[1])
        output_name = "uncertainty.json"

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    print(mean_results)
    save_json(mean_results, f"{output_dir}/{output_name}")


if __name__ == '__main__':
    main()
