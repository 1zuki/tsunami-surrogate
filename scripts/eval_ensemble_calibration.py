#!/usr/bin/env python
"""Appendix ensemble reliability and scalar variance-inflation diagnostic.

This is post-hoc calibration only: a single multiplicative variance factor is
fit on validation elements and then applied unchanged to test/OOD elements.
Coverage is marginal over correlated pixels and frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.models import build_model
from src.models.ensemble import EnsemblePredictor
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.model_io import validate_model_io_channels
from src.utils.seed import seed_everything


DEFAULT_OUTPUT_DIR = Path("results/reviewer_validation/ensemble_calibration")
DEFAULT_FIGURE = Path("paper/figures/ensemble_reliability.pdf")
DEFAULT_CHECKPOINTS = [
    Path(f"experiments/ensemble/member_{seed}/best.pt")
    for seed in (11, 22, 33, 44, 55, 66, 77)
]
DEFAULT_SUITES = [
    ("test", Path("data/processed/hydrostatic/test")),
    (
        "source_holdout_multi_gauss",
        Path("data/processed_ood/hydrostatic/source_holdout_multi_gauss"),
    ),
    (
        "bathymetry_holdout_trench",
        Path("data/processed_ood/hydrostatic/bathymetry_holdout_trench"),
    ),
    (
        "source_strength_extreme_high",
        Path("data/processed_ood/hydrostatic/source_strength_extreme_high"),
    ),
]
SUMMARY_FALLBACK_INDIST = Path("results/uncertainty_hydrostatic_indist_m7.json")
SUMMARY_FALLBACK_OOD = Path("results/uncertainty_hydrostatic_ood_m7.json")


def _nominal_levels() -> list[float]:
    return [round(0.05 * i, 2) for i in range(1, 20)]


def _headline_levels() -> list[float]:
    return [0.50, 0.80, 0.90, 0.95]


def _z_for_central_coverage(p: float, device: torch.device) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(
        math.sqrt(2.0)
        * torch.erfinv(torch.tensor(p, dtype=torch.float64, device=device)).item()
    )


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _loader_for_path(cfg: dict[str, Any], split_name: str, path: Path, batch_size: int):
    local_cfg = dict(cfg)
    data_cfg = dict(local_cfg.get("data", {}))
    data_cfg["batch_size"] = int(batch_size)
    data_cfg["num_workers"] = int(data_cfg.get("num_workers", 0))
    data_cfg["train_path"] = None
    data_cfg["val_path"] = None
    data_cfg["test_path"] = None
    if split_name == "val":
        data_cfg["val_path"] = str(path)
    else:
        data_cfg["test_path"] = str(path)
    local_cfg["data"] = data_cfg
    loaders = create_dataloaders(local_cfg)
    loader = loaders.get("val" if split_name == "val" else "test")
    if loader is None:
        raise KeyError(f"No dataloader could be built for {split_name} path: {path}")
    if _dataset_num_samples(loader) == 0:
        raise ValueError(f"Dataset has zero samples: {path}")
    validate_model_io_channels(
        local_cfg, loaders, preferred_splits=("val", "test", "train")
    )
    return loader


def _load_ensemble(
    cfg: dict[str, Any],
    checkpoint_paths: list[Path],
    device: torch.device,
) -> EnsemblePredictor:
    members = []
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected ensemble checkpoint: {path}")
        ckpt_cpu = torch.load(path, map_location="cpu")
        model_cfg = ckpt_cpu.get("config", cfg)
        model = build_model(model_cfg).to(device)
        load_checkpoint(path, model, map_location=device)
        model.eval()
        members.append(model)
    return EnsemblePredictor(members).to(device).eval()


@torch.no_grad()
def _fit_gamma(
    ensemble: EnsemblePredictor,
    loader: Any,
    device: torch.device,
    eps: float,
    max_samples: int | None,
) -> dict[str, float]:
    ratio_sum = 0.0
    mse_sum = 0.0
    var_sum = 0.0
    n_elements = 0
    n_samples = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if max_samples is not None:
            remaining = int(max_samples) - n_samples
            if remaining <= 0:
                break
            if int(x.shape[0]) > remaining:
                x = x[:remaining]
                y = y[:remaining]
        out = ensemble(x)
        var = torch.clamp(out["variance"], min=eps)
        err_sq = (y - out["mean"]).square()
        ratio_sum += float((err_sq / var).sum().detach().cpu())
        mse_sum += float(err_sq.sum().detach().cpu())
        var_sum += float(var.sum().detach().cpu())
        n_elements += int(err_sq.numel())
        n_samples += int(x.shape[0])
    if n_elements <= 0:
        raise RuntimeError("No validation elements were evaluated while fitting gamma.")
    gamma_raw = ratio_sum / float(n_elements)
    gamma = max(1.0, float(gamma_raw))
    return {
        "gamma_raw": float(gamma_raw),
        "gamma": float(gamma),
        "std_scale": float(math.sqrt(gamma)),
        "n_elements": int(n_elements),
        "n_samples": int(n_samples),
        "mse": float(mse_sum / float(n_elements)),
        "mean_variance": float(var_sum / float(n_elements)),
    }


@torch.no_grad()
def _evaluate_loader(
    ensemble: EnsemblePredictor,
    loader: Any,
    device: torch.device,
    levels: list[float],
    gamma: float,
    eps: float,
    max_samples: int | None,
) -> dict[str, Any]:
    z_values = {p: _z_for_central_coverage(p, device) for p in levels}
    raw_inside = {p: 0 for p in levels}
    calibrated_inside = {p: 0 for p in levels}
    n_elements = 0
    n_samples = 0
    abs_err_sum = 0.0
    mse_sum = 0.0
    var_sum = 0.0
    corr_sum = 0.0
    corr_n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if max_samples is not None:
            remaining = int(max_samples) - n_samples
            if remaining <= 0:
                break
            if int(x.shape[0]) > remaining:
                x = x[:remaining]
                y = y[:remaining]
        out = ensemble(x)
        mean = out["mean"]
        var = torch.clamp(out["variance"], min=eps)
        std = torch.sqrt(var)
        abs_err = torch.abs(y - mean)
        cal_std = std * math.sqrt(float(gamma))
        for p, z in z_values.items():
            raw_inside[p] += int((abs_err <= z * std).sum().detach().cpu())
            calibrated_inside[p] += int((abs_err <= z * cal_std).sum().detach().cpu())
        n_elements += int(abs_err.numel())
        n_samples += int(x.shape[0])
        abs_err_sum += float(abs_err.sum().detach().cpu())
        mse_sum += float(abs_err.square().sum().detach().cpu())
        var_sum += float(var.sum().detach().cpu())

        # Batch-level Pearson correlation avoids storing all elements.
        std_flat = std.reshape(-1).float()
        err_flat = abs_err.reshape(-1).float()
        if std_flat.numel() >= 2:
            std_centered = std_flat - std_flat.mean()
            err_centered = err_flat - err_flat.mean()
            denom = torch.sqrt((std_centered * std_centered).sum()) * torch.sqrt(
                (err_centered * err_centered).sum()
            )
            if float(denom.detach().cpu()) > 1e-12:
                corr_sum += float(
                    ((std_centered * err_centered).sum() / denom).detach().cpu()
                )
                corr_n += 1

    if n_elements <= 0:
        raise RuntimeError("No elements were evaluated.")

    rows = []
    for p in levels:
        rows.append(
            {
                "nominal": float(p),
                "raw_coverage": float(raw_inside[p] / float(n_elements)),
                "calibrated_coverage": float(calibrated_inside[p] / float(n_elements)),
            }
        )
    return {
        "n_elements": int(n_elements),
        "n_samples": int(n_samples),
        "mean_abs_error": float(abs_err_sum / float(n_elements)),
        "mse": float(mse_sum / float(n_elements)),
        "mean_variance": float(var_sum / float(n_elements)),
        "error_uncertainty_corr_batch_mean": float(corr_sum / corr_n)
        if corr_n > 0
        else 0.0,
        "coverage": rows,
    }


def _parse_suite_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return DEFAULT_SUITES
    suites: list[tuple[str, Path]] = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        elif ":" in value:
            label, path = value.split(":", 1)
        else:
            p = Path(value)
            label, path = p.name, value
        label = label.strip()
        if not label:
            raise ValueError(f"Suite label is empty in {value!r}")
        suites.append((label, Path(path.strip())))
    return suites


def _coverage_at(
    rows: list[dict[str, float]], nominal: float, key: str
) -> float | None:
    for row in rows:
        if abs(float(row["nominal"]) - float(nominal)) < 1e-9:
            value = row.get(key)
            if value is None:
                return None
            return float(value)
    return None


def _write_reliability_csv(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "nominal",
        "raw_coverage",
        "calibrated_coverage",
        "n_samples",
        "n_elements",
        "calibration_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, res in results.get("datasets", {}).items():
            for row in res.get("coverage", []):
                writer.writerow(
                    {
                        "dataset": label,
                        "nominal": row.get("nominal", ""),
                        "raw_coverage": row.get("raw_coverage", ""),
                        "calibrated_coverage": row.get("calibrated_coverage", ""),
                        "n_samples": res.get("n_samples", ""),
                        "n_elements": res.get("n_elements", ""),
                        "calibration_status": results.get("calibration_status", ""),
                    }
                )


def _write_coverage_table_csv(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "nominal",
        "raw_coverage",
        "calibrated_coverage",
        "raw_gap",
        "calibrated_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, res in results.get("datasets", {}).items():
            rows = res.get("coverage", [])
            for p in _headline_levels():
                raw = _coverage_at(rows, p, "raw_coverage")
                cal = _coverage_at(rows, p, "calibrated_coverage")
                writer.writerow(
                    {
                        "dataset": label,
                        "nominal": p,
                        "raw_coverage": raw,
                        "calibrated_coverage": cal,
                        "raw_gap": None if raw is None else raw - p,
                        "calibrated_gap": None if cal is None else cal - p,
                    }
                )


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _plot(results: dict[str, Any], output: Path, png_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = results.get("datasets", {})
    calibrated_available = (
        results.get("calibration_status") == "fit_validation_scalar_inflation"
    )
    if calibrated_available:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
        panels = [
            ("raw_coverage", "Raw ensemble variance"),
            ("calibrated_coverage", "After scalar inflation"),
        ]
    else:
        fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
        axes = [ax]
        panels = [("raw_coverage", "Raw summary coverage")]

    for ax, (coverage_key, title) in zip(axes, panels):
        ax.plot(
            [0, 1], [0, 1], color="0.2", linestyle="--", linewidth=1.0, label="nominal"
        )
        for label, res in datasets.items():
            rows = sorted(res.get("coverage", []), key=lambda r: float(r["nominal"]))
            if not rows:
                continue
            x = [float(r["nominal"]) for r in rows]
            y = [float(r.get(coverage_key, float("nan"))) for r in rows]
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.2, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Nominal central Gaussian coverage")
        ax.set_ylabel("Empirical marginal coverage")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    fig.savefig(output, bbox_inches="tight")
    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fallback_from_summaries(
    indist_path: Path,
    ood_path: Path,
    levels: list[float],
) -> dict[str, Any]:
    if not indist_path.is_file() or not ood_path.is_file():
        missing = [str(p) for p in (indist_path, ood_path) if not p.is_file()]
        raise FileNotFoundError(
            f"Cannot run summary fallback; missing summary JSON(s): {missing}"
        )

    datasets: dict[str, Any] = {}
    indist = _load_json(indist_path)
    rows = []
    for p in levels:
        key = f"coverage_{int(round(p * 100))}"
        if key in indist:
            rows.append(
                {
                    "nominal": float(p),
                    "raw_coverage": float(indist[key]),
                    "calibrated_coverage": None,
                }
            )
    datasets["test_summary"] = {
        "n_samples": int(float(indist.get("num_samples", -1))),
        "n_elements": None,
        "coverage": rows,
        "source_summary": str(indist_path),
    }

    ood = _load_json(ood_path)
    for label, suite in ood.get("suites", {}).items():
        rows = []
        for p in levels:
            key = f"coverage_{int(round(p * 100))}"
            if key in suite:
                rows.append(
                    {
                        "nominal": float(p),
                        "raw_coverage": float(suite[key]),
                        "calibrated_coverage": None,
                    }
                )
        datasets[str(label)] = {
            "n_samples": int(float(suite.get("num_samples", -1))),
            "n_elements": None,
            "coverage": rows,
            "source_summary": str(ood_path),
        }
    return {
        "diagnostic": "ensemble_reliability_summary_fallback",
        "calibration_status": "unavailable_no_per_element_predictions",
        "note": (
            "Only existing headline summary coverage values were available; scalar calibration was not fit "
            "and calibrated intervals are intentionally absent."
        ),
        "nominal_levels": levels,
        "datasets": datasets,
    }


def _run_full(
    args: argparse.Namespace, levels: list[float], suites: list[tuple[str, Path]]
) -> dict[str, Any]:
    cfg = load_config(args.config)
    cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))
    batch_size = int(args.batch_size)
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["batch_size"] = batch_size
    data_cfg["num_workers"] = int(args.num_workers)
    cfg["data"] = data_cfg
    device = resolve_device(args.device)
    checkpoint_paths = (
        [Path(p) for p in args.checkpoint] if args.checkpoint else DEFAULT_CHECKPOINTS
    )
    if len(checkpoint_paths) < 2:
        raise ValueError(
            "At least two ensemble checkpoints are required for reliability evaluation."
        )

    ensemble = _load_ensemble(cfg, checkpoint_paths, device)
    val_loader = _loader_for_path(cfg, "val", Path(args.val_path), batch_size)
    fit = _fit_gamma(
        ensemble,
        val_loader,
        device,
        eps=float(args.variance_eps),
        max_samples=args.max_val_samples,
    )
    gamma = float(fit["gamma"])

    datasets: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    eval_specs = [("validation", Path(args.val_path)), *suites]
    for label, path in eval_specs:
        try:
            loader = _loader_for_path(
                cfg, "val" if label == "validation" else "test", path, batch_size
            )
        except Exception as e:
            if args.skip_missing_suites and label != "validation":
                skipped.append(
                    {
                        "label": label,
                        "path": str(path),
                        "reason": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            raise
        res = _evaluate_loader(
            ensemble,
            loader,
            device,
            levels=levels,
            gamma=gamma,
            eps=float(args.variance_eps),
            max_samples=args.max_eval_samples
            if label != "validation"
            else args.max_val_samples,
        )
        res["path"] = str(path)
        datasets[label] = res
        raw90 = _coverage_at(res["coverage"], 0.90, "raw_coverage")
        cal90 = _coverage_at(res["coverage"], 0.90, "calibrated_coverage")
        print(
            f"{label}: raw90={raw90:.4f} calibrated90={cal90:.4f} samples={res['n_samples']}"
        )

    return {
        "diagnostic": "ensemble_reliability_scalar_variance_inflation",
        "calibration_status": "fit_validation_scalar_inflation",
        "evaluation_scope": (
            "limited"
            if args.max_val_samples is not None or args.max_eval_samples is not None
            else "full"
        ),
        "calibration_fit_split": "validation",
        "coverage_definition": "central Gaussian marginal per-element coverage over correlated pixels and frames",
        "variance_eps": float(args.variance_eps),
        "max_val_samples": args.max_val_samples,
        "max_eval_samples": args.max_eval_samples,
        "nominal_levels": levels,
        "headline_levels": _headline_levels(),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "fit": fit,
        "datasets": datasets,
        "skipped_suites": skipped,
        "device": str(device),
        "batch_size": batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/fno.yaml")
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--val-path", default="data/processed/hydrostatic/val")
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite as label=path. Repeatable.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--variance-eps", type=float, default=1.0e-12)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--png-output", type=Path, default=None)
    parser.set_defaults(skip_missing_suites=True)
    parser.add_argument(
        "--no-skip-missing-suites",
        dest="skip_missing_suites",
        action="store_false",
        help="Fail instead of recording unavailable optional OOD suites.",
    )
    parser.add_argument("--summary-fallback", action="store_true")
    parser.add_argument(
        "--fallback-indist-json", type=Path, default=SUMMARY_FALLBACK_INDIST
    )
    parser.add_argument("--fallback-ood-json", type=Path, default=SUMMARY_FALLBACK_OOD)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    levels = _nominal_levels()
    if args.smoke:
        args.max_val_samples = args.max_val_samples or 4
        args.max_eval_samples = args.max_eval_samples or 4
        args.batch_size = min(int(args.batch_size), 2)

    output_dir = Path(args.output_dir)
    json_path = output_dir / "ensemble_calibration.json"
    reliability_csv = output_dir / "reliability_curve.csv"
    table_csv = output_dir / "coverage_table.csv"
    png_output = (
        args.png_output
        if args.png_output is not None
        else args.figure_output.with_suffix(".png")
    )

    checkpoint_paths = (
        [Path(p) for p in args.checkpoint] if args.checkpoint else DEFAULT_CHECKPOINTS
    )
    missing_checkpoints = [str(p) for p in checkpoint_paths if not p.is_file()]
    if args.summary_fallback or missing_checkpoints:
        if missing_checkpoints and not args.summary_fallback:
            print(
                f"[warn] missing ensemble checkpoints; using summary fallback: {missing_checkpoints}"
            )
        results = _fallback_from_summaries(
            args.fallback_indist_json, args.fallback_ood_json, _headline_levels()
        )
    else:
        suites = _parse_suite_specs(args.suite)
        results = _run_full(args, levels, suites)

    results["json_path"] = str(json_path)
    results["reliability_csv_path"] = str(reliability_csv)
    results["coverage_table_csv_path"] = str(table_csv)
    results["figure_path"] = str(args.figure_output)
    results["png_path"] = str(png_output)
    _save_json(results, json_path)
    _write_reliability_csv(results, reliability_csv)
    _write_coverage_table_csv(results, table_csv)
    _plot(
        results,
        Path(args.figure_output),
        Path(png_output) if png_output is not None else None,
    )

    if results.get("calibration_status") == "fit_validation_scalar_inflation":
        fit = results["fit"]
        print(
            f"gamma_raw={fit['gamma_raw']:.6f} gamma={fit['gamma']:.6f} "
            f"std_scale={fit['std_scale']:.6f}"
        )
    else:
        print(f"calibration_status={results.get('calibration_status')}")
    print(f"saved_json={json_path}")
    print(f"saved_csv={reliability_csv}")
    print(f"saved_table_csv={table_csv}")
    print(f"saved_pdf={args.figure_output}")
    print(f"saved_png={png_output}")


if __name__ == "__main__":
    main()
