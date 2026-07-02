#!/usr/bin/env python
"""Physics-aware error diagnostics for surrogate rollouts.

The standard evaluator reports aggregate pixel-space rel-L2/RMSE. This script adds
three compact paper-facing diagnostics:

1. free-surface / mass-proxy integral error;
2. low/mid/high Fourier-band relative error;
3. per-sample rel-L2 stratified by source strength and bathymetry gradient.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.target_scaling import (
    apply_target_denorm,
    load_target_denorm,
    resolve_eval_dataset_path,
)
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.utils.seed import seed_everything


EPS = 1e-12


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


@torch.no_grad()
def _rollout_window_model(
    model: torch.nn.Module,
    x_static: torch.Tensor,
    y: torch.Tensor,
    *,
    window_k: int,
    include_source: bool,
    use_prev: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    bathy = x_static[:, 0]
    source = x_static[:, 1]
    eta_t = y[:, 0]
    eta_prev = y[:, 0]
    target = y[:, 1:]
    target_len = int(target.shape[1])
    preds: list[torch.Tensor] = []
    produced = 0

    while produced < target_len:
        chans = [bathy, source] if include_source else [bathy]
        chans.append(eta_t)
        if use_prev:
            chans.append(eta_prev)
        win_x = torch.stack(chans, dim=1)
        win_pred = _model_output(model, win_x)
        preds.append(win_pred)
        eta_prev = win_pred[:, -2] if win_pred.shape[1] >= 2 else win_pred[:, -1]
        eta_t = win_pred[:, -1]
        produced += int(win_pred.shape[1])

    pred = torch.cat(preds, dim=1)[:, :target_len]
    return pred, target


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalization_stats_path(dataset_path: Path) -> Optional[Path]:
    p = dataset_path
    if p.name == "eval_dataset.npz":
        p = p.parent
    candidates = []
    for parent in [p, *p.parents]:
        candidates.append(parent / "normalization_stats.json")
        if parent == ROOT:
            break
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_input_order(dataset_path: Path) -> list[str]:
    p = dataset_path
    if p.name == "eval_dataset.npz":
        p = p.parent
    candidates = [
        p / "eval_manifest.json",
        p.parent / "eval_manifest.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            data = _load_json(candidate)
        except Exception:
            continue
        order = data.get("input_order")
        if isinstance(order, list) and all(isinstance(v, str) for v in order):
            return list(order)
    return ["bathymetry", "source", "initial_depth"]


def _input_offset_scale(stats: Mapping[str, Any], name: str) -> tuple[float, float]:
    inputs = stats.get("inputs", {})
    spec = inputs.get(name, {}) if isinstance(inputs, Mapping) else {}
    if not isinstance(spec, Mapping):
        return 0.0, 1.0
    offset = float(spec.get("offset", 0.0))
    scale = float(spec.get("scale", 1.0))
    if not math.isfinite(offset) or not math.isfinite(scale) or scale == 0.0:
        return 0.0, 1.0
    return offset, scale


def _as_float_tensor(values: Any, *, device: torch.device, n: int) -> torch.Tensor:
    if torch.is_tensor(values):
        out = values.to(device=device, dtype=torch.float64).reshape(-1)
    else:
        out = torch.as_tensor(values, dtype=torch.float64, device=device).reshape(-1)
    if out.numel() >= n:
        return out[:n]
    if out.numel() != n:
        return torch.full((n,), float("nan"), dtype=torch.float64, device=device)
    return out


def _make_spectral_masks(
    height: int,
    width: int,
    device: torch.device,
    low_cutoff: float,
    mid_cutoff: float,
) -> dict[str, torch.Tensor]:
    kx = torch.fft.fftfreq(height, d=1.0, device=device) * float(height)
    ky = torch.fft.rfftfreq(width, d=1.0, device=device) * float(width)
    radius = torch.sqrt(kx[:, None] * kx[:, None] + ky[None, :] * ky[None, :])
    return {
        f"low_0_{low_cutoff:g}": radius <= float(low_cutoff),
        f"mid_{low_cutoff:g}_{mid_cutoff:g}": (radius > float(low_cutoff))
        & (radius <= float(mid_cutoff)),
        f"high_{mid_cutoff:g}_plus": radius > float(mid_cutoff),
    }


def _empty_spectral_sums(mask_names: Iterable[str]) -> dict[str, dict[str, float]]:
    return {name: {"sum_sq_err": 0.0, "sum_sq_target": 0.0} for name in mask_names}


def _update_spectral_sums(
    sums: dict[str, dict[str, float]],
    masks: dict[str, torch.Tensor],
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    diff = (pred - target).to(torch.float32)
    target32 = target.to(torch.float32)
    err_fft = torch.fft.rfft2(diff, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target32, dim=(-2, -1), norm="ortho")
    err_energy = err_fft.real.square() + err_fft.imag.square()
    target_energy = target_fft.real.square() + target_fft.imag.square()
    for name, mask in masks.items():
        sums[name]["sum_sq_err"] += float(err_energy[..., mask].sum().detach().cpu())
        sums[name]["sum_sq_target"] += float(
            target_energy[..., mask].sum().detach().cpu()
        )


def _empty_integral_sums() -> dict[str, float]:
    return {
        "eta_sum_abs_err": 0.0,
        "eta_sum_sq_err": 0.0,
        "eta_sum_sq_target": 0.0,
        "eta_max_abs_err": 0.0,
        "mass_sum_abs_err": 0.0,
        "mass_sum_sq_err": 0.0,
        "mass_sum_sq_target": 0.0,
        "mass_max_abs_err": 0.0,
        "n": 0.0,
    }


def _update_integral_sums(
    sums: dict[str, float],
    pred_eta: torch.Tensor,
    target_eta: torch.Tensor,
    bathymetry: torch.Tensor,
) -> None:
    pred_eta_mean = pred_eta.to(torch.float64).mean(dim=(-2, -1))
    target_eta_mean = target_eta.to(torch.float64).mean(dim=(-2, -1))
    eta_diff = pred_eta_mean - target_eta_mean

    bathy_mean = bathymetry.to(torch.float64).mean(dim=(-2, -1))[:, None]
    pred_depth_mean = pred_eta_mean - bathy_mean
    target_depth_mean = target_eta_mean - bathy_mean
    mass_diff = pred_depth_mean - target_depth_mean

    sums["eta_sum_abs_err"] += float(eta_diff.abs().sum().detach().cpu())
    sums["eta_sum_sq_err"] += float(eta_diff.square().sum().detach().cpu())
    sums["eta_sum_sq_target"] += float(target_eta_mean.square().sum().detach().cpu())
    sums["eta_max_abs_err"] = max(
        sums["eta_max_abs_err"], float(eta_diff.abs().max().detach().cpu())
    )
    sums["mass_sum_abs_err"] += float(mass_diff.abs().sum().detach().cpu())
    sums["mass_sum_sq_err"] += float(mass_diff.square().sum().detach().cpu())
    sums["mass_sum_sq_target"] += float(target_depth_mean.square().sum().detach().cpu())
    sums["mass_max_abs_err"] = max(
        sums["mass_max_abs_err"], float(mass_diff.abs().max().detach().cpu())
    )
    sums["n"] += float(eta_diff.numel())


def _bathymetry_gradient_rms(bathymetry: torch.Tensor) -> torch.Tensor:
    bx = bathymetry[..., :, 1:] - bathymetry[..., :, :-1]
    by = bathymetry[..., 1:, :] - bathymetry[..., :-1, :]
    return torch.sqrt(
        bx.to(torch.float64).square().mean(dim=(-2, -1))
        + by.to(torch.float64).square().mean(dim=(-2, -1))
        + EPS
    )


def _per_sample_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).to(torch.float64).flatten(start_dim=1)
    tgt = target.to(torch.float64).flatten(start_dim=1)
    return torch.linalg.vector_norm(diff, dim=1) / (
        torch.linalg.vector_norm(tgt, dim=1) + EPS
    )


def _summarize_array(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
    }


def _quantile_bins(
    values: np.ndarray,
    errors: np.ndarray,
    *,
    num_bins: int,
    label: str,
) -> list[dict[str, float | int | str]]:
    valid = np.isfinite(values) & np.isfinite(errors)
    values = values[valid]
    errors = errors[valid]
    if values.size == 0:
        return []

    num_bins = max(1, min(int(num_bins), int(values.size)))
    edges = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if edges.size == 1:
        edges = np.asarray([edges[0], edges[0]], dtype=np.float64)

    rows: list[dict[str, float | int | str]] = []
    for i in range(edges.size - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == edges.size - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        if not np.any(mask):
            continue
        e = errors[mask]
        v = values[mask]
        rows.append(
            {
                "bin": f"{label}_q{i + 1}",
                "count": int(e.size),
                "value_min": float(np.min(v)),
                "value_max": float(np.max(v)),
                "value_mean": float(np.mean(v)),
                "rel_l2_mean": float(np.mean(e)),
                "rel_l2_median": float(np.median(e)),
                "rel_l2_p90": float(np.percentile(e, 90)),
            }
        )
    return rows


def _write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "scenario_id",
        "source_type",
        "bathymetry_type",
        "source_strength",
        "bathymetry_gradient_rms",
        "rel_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate physics-aware surrogate error diagnostics."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--per-sample-output", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-bins", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--low-mode-cutoff", type=float, default=8.0)
    p.add_argument("--mid-mode-cutoff", type=float, default=16.0)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))

    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    data_cfg = dict(cfg.get("data", {}))
    dataset_cfg = cfg.get("dataset", {})
    if not data_cfg and isinstance(dataset_cfg, dict):
        dataset_path = dataset_cfg.get("path")
        if dataset_path:
            data_cfg["test_path"] = dataset_path
        if "batch_size" in dataset_cfg:
            data_cfg["batch_size"] = dataset_cfg["batch_size"]

    if eval_cfg.get("dataset_path"):
        data_cfg["test_path"] = eval_cfg["dataset_path"]
    if "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]

    is_window_model = bool(data_cfg.get("windowed", False))
    test_path = data_cfg.get("test_path") or eval_cfg.get("dataset_path")
    if not test_path:
        raise KeyError(
            "Set eval.dataset_path or data.test_path for physics diagnostics."
        )
    loader_data_cfg = dict(data_cfg)
    loader_data_cfg.pop("train_path", None)
    loader_data_cfg.pop("val_path", None)
    loader_data_cfg["test_path"] = test_path
    loader_data_cfg["windowed"] = False
    loader_data_cfg["num_workers"] = int(args.num_workers)
    cfg_for_loader = dict(cfg)
    cfg_for_loader["data"] = loader_data_cfg

    device = resolve_device(cfg.get("device", "auto"))
    loaders = create_dataloaders(cfg_for_loader)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError(
            "No test dataloader found. Set eval.dataset_path or data.test_path."
        )
    if not is_window_model:
        validate_model_io_channels(
            cfg, loaders, preferred_splits=("test", "val", "train")
        )

    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    resolved_dataset_path = resolve_eval_dataset_path(cfg_for_loader, split="test")
    if resolved_dataset_path is None:
        raise KeyError("Could not resolve evaluation dataset path.")
    target_denorm = None
    if bool(eval_cfg.get("report_physical_metrics", True)):
        try:
            target_denorm = load_target_denorm(resolved_dataset_path)
        except Exception:
            target_denorm = None

    stats_path = _normalization_stats_path(Path(resolved_dataset_path))
    stats = _load_json(stats_path) if stats_path is not None else {}
    input_order = _read_input_order(Path(resolved_dataset_path))
    bathy_idx = input_order.index("bathymetry") if "bathymetry" in input_order else 0
    bathy_offset, bathy_scale = _input_offset_scale(stats, "bathymetry")

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    output_path = (
        Path(args.output)
        if args.output
        else Path(output_dir) / "physics_diagnostics.json"
    )
    per_sample_path = (
        Path(args.per_sample_output)
        if args.per_sample_output
        else Path(output_dir) / "physics_diagnostics_per_sample.csv"
    )

    integral_sums = _empty_integral_sums()
    spectral_sums: Optional[dict[str, dict[str, float]]] = None
    spectral_masks: Optional[dict[str, torch.Tensor]] = None
    per_sample_rows: list[dict[str, Any]] = []
    total_seen = 0

    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if args.max_samples is not None:
            remaining = int(args.max_samples) - total_seen
            if remaining <= 0:
                break
            if x.shape[0] > remaining:
                x = x[:remaining]
                y = y[:remaining]

        if is_window_model:
            pred, target = _rollout_window_model(
                model,
                x,
                y,
                window_k=int(data_cfg.get("window_K", 5)),
                include_source=bool(data_cfg.get("window_include_source", True)),
                use_prev=bool(data_cfg.get("window_prev", True)),
            )
            evaluation_mode = "seeded_window_rollout"
        else:
            pred = _model_output(model, x)
            target = y
            evaluation_mode = "direct_single_pass"

        pred_eval = apply_target_denorm(pred, target_denorm)
        target_eval = apply_target_denorm(target, target_denorm)
        bathymetry = x[:, bathy_idx] * float(bathy_scale) + float(bathy_offset)

        if spectral_masks is None:
            h, w = int(target_eval.shape[-2]), int(target_eval.shape[-1])
            spectral_masks = _make_spectral_masks(
                h,
                w,
                device,
                low_cutoff=float(args.low_mode_cutoff),
                mid_cutoff=float(args.mid_mode_cutoff),
            )
            spectral_sums = _empty_spectral_sums(spectral_masks.keys())

        assert spectral_masks is not None and spectral_sums is not None
        _update_integral_sums(integral_sums, pred_eval, target_eval, bathymetry)
        _update_spectral_sums(spectral_sums, spectral_masks, pred_eval, target_eval)

        rel_l2 = _per_sample_rel_l2(pred_eval, target_eval)
        gradient = _bathymetry_gradient_rms(bathymetry)
        source_strength = _as_float_tensor(
            batch.get("source_strength", []),
            device=device,
            n=int(x.shape[0]),
        )

        sample_ids = list(batch.get("sample_id", [""] * int(x.shape[0])))
        scenario_ids = list(batch.get("scenario_id", [""] * int(x.shape[0])))
        source_types = list(batch.get("source_type", ["unknown"] * int(x.shape[0])))
        bathy_types = list(batch.get("bathymetry_type", ["unknown"] * int(x.shape[0])))

        if args.max_samples is not None and len(sample_ids) > x.shape[0]:
            sample_ids = sample_ids[: x.shape[0]]
            scenario_ids = scenario_ids[: x.shape[0]]
            source_types = source_types[: x.shape[0]]
            bathy_types = bathy_types[: x.shape[0]]

        rel_l2_np = rel_l2.detach().cpu().numpy()
        gradient_np = gradient.detach().cpu().numpy()
        source_strength_np = source_strength.detach().cpu().numpy()
        for i in range(int(x.shape[0])):
            per_sample_rows.append(
                {
                    "sample_id": str(sample_ids[i]),
                    "scenario_id": str(scenario_ids[i]),
                    "source_type": str(source_types[i]),
                    "bathymetry_type": str(bathy_types[i]),
                    "source_strength": float(source_strength_np[i]),
                    "bathymetry_gradient_rms": float(gradient_np[i]),
                    "rel_l2": float(rel_l2_np[i]),
                }
            )

        total_seen += int(x.shape[0])

    if spectral_sums is None:
        raise RuntimeError("No batches were evaluated.")

    n_integral = max(float(integral_sums["n"]), 1.0)
    spectral = {}
    for name, sums in spectral_sums.items():
        spectral[name] = {
            "rel_l2": float(
                math.sqrt(sums["sum_sq_err"]) / (math.sqrt(sums["sum_sq_target"]) + EPS)
            ),
            "sum_sq_err": float(sums["sum_sq_err"]),
            "sum_sq_target": float(sums["sum_sq_target"]),
        }

    source_strength_values = np.asarray(
        [r["source_strength"] for r in per_sample_rows], dtype=np.float64
    )
    gradient_values = np.asarray(
        [r["bathymetry_gradient_rms"] for r in per_sample_rows], dtype=np.float64
    )
    rel_l2_values = np.asarray([r["rel_l2"] for r in per_sample_rows], dtype=np.float64)

    out: Dict[str, Any] = {
        "evaluation_type": "physics_diagnostics",
        "evaluation_mode": evaluation_mode,
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(resolved_dataset_path),
        "num_samples_seen": int(total_seen),
        "num_samples_dataset": int(_dataset_num_samples(test_loader)),
        "target_units": "physical" if target_denorm is not None else "normalized",
        "target_offset": float(target_denorm[0]) if target_denorm is not None else None,
        "target_scale": float(target_denorm[1]) if target_denorm is not None else None,
        "normalization_stats_path": str(stats_path) if stats_path is not None else "",
        "input_order": input_order,
        "diagnostics": {
            "free_surface_integral": {
                "description": "Spatial-mean eta time-series error; equivalent to free-surface integral error up to constant cell area.",
                "mae": float(integral_sums["eta_sum_abs_err"] / n_integral),
                "rmse": float(math.sqrt(integral_sums["eta_sum_sq_err"] / n_integral)),
                "rel_l2": float(
                    math.sqrt(integral_sums["eta_sum_sq_err"])
                    / (math.sqrt(integral_sums["eta_sum_sq_target"]) + EPS)
                ),
                "max_abs_error": float(integral_sums["eta_max_abs_err"]),
            },
            "mass_proxy_integral": {
                "description": "Spatial-mean water-depth h=eta-b time-series error for fully wet cases.",
                "mae": float(integral_sums["mass_sum_abs_err"] / n_integral),
                "rmse": float(math.sqrt(integral_sums["mass_sum_sq_err"] / n_integral)),
                "rel_l2": float(
                    math.sqrt(integral_sums["mass_sum_sq_err"])
                    / (math.sqrt(integral_sums["mass_sum_sq_target"]) + EPS)
                ),
                "max_abs_error": float(integral_sums["mass_max_abs_err"]),
            },
            "spectral_band_rel_l2": spectral,
            "sample_rel_l2_summary": _summarize_array(rel_l2_values),
            "source_strength_bins": _quantile_bins(
                source_strength_values,
                rel_l2_values,
                num_bins=int(args.num_bins),
                label="source_strength",
            ),
            "bathymetry_gradient_bins": _quantile_bins(
                gradient_values,
                rel_l2_values,
                num_bins=int(args.num_bins),
                label="bathymetry_gradient",
            ),
        },
        "per_sample_csv": str(per_sample_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, output_path)
    _write_per_sample_csv(per_sample_path, per_sample_rows)
    spectral_text = ", ".join(f"{k}: {v['rel_l2']:.4g}" for k, v in spectral.items())
    print(
        f"[physics] mode={evaluation_mode} n={total_seen} "
        f"eta_integral_rel_l2={out['diagnostics']['free_surface_integral']['rel_l2']:.4g} "
        f"spectral={{{spectral_text}}} -> {output_path}"
    )


if __name__ == "__main__":
    main()
