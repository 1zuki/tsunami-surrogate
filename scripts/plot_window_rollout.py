#!/usr/bin/env python
"""Plot an autoregressive windowed-model rollout for one test scenario.

The model is evaluated in the same seeded autoregressive mode as
scripts/eval_window_rollout.py: frame 0 is given, then the model predicts chunks
of K future eta frames. The figure shows bathymetry/source plus reference eta,
predicted eta, and absolute error at selected trajectory frames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_window_rollout import rollout_trajectory
from src.data.dataset import ShardedTsunamiDataset
from src.evaluation.target_scaling import apply_target_denorm, load_target_denorm
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.seed import seed_everything


DEFAULT_CONFIG = "configs/model/ffno_window5_hydrostatic.yaml"
DEFAULT_CHECKPOINT = "experiments/ffno_window5_hydrostatic/best.pt"
DEFAULT_PROCESSED = "data/processed/hydrostatic/test"
DEFAULT_STATS = "data/processed/hydrostatic/normalization_stats.json"
DEFAULT_OUTPUT = "paper/figures/window_rollout.pdf"
DEFAULT_SAMPLE_INDEX = 856


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _denorm_input(arr: np.ndarray, stats: dict[str, Any], name: str) -> np.ndarray:
    channel = stats.get("inputs", {}).get(name)
    if not channel:
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32) * float(channel["scale"]) + float(
        channel["offset"]
    )


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return float(
        np.linalg.norm((pred64 - target64).ravel())
        / (np.linalg.norm(target64.ravel()) + 1e-12)
    )


def _find_sample_index(dataset: ShardedTsunamiDataset, sample_id: str) -> int:
    for idx in range(len(dataset)):
        item = dataset[idx]
        if (
            str(item.get("sample_id", "")) == sample_id
            or str(item.get("scenario_id", "")) == sample_id
        ):
            return int(idx)
    raise LookupError(
        f"No sample_id/scenario_id {sample_id!r} found in {len(dataset)} samples."
    )


def _draw_panel(fig, ax, panel: dict[str, Any]) -> None:
    im = ax.imshow(
        panel["array"],
        origin="upper",
        cmap=panel["cmap"],
        norm=panel.get("norm"),
        vmin=panel.get("vmin"),
        vmax=panel.get("vmax"),
    )
    ax.set_title(panel["title"], fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)


def _plot(
    bathymetry: np.ndarray,
    source: np.ndarray,
    target_full: np.ndarray,
    pred_full: np.ndarray,
    frames: list[int],
    frame_rel_l2: list[float],
    meta: dict[str, Any],
    global_rel_l2: float,
    output_path: Path,
    png_output_path: Path | None,
) -> None:
    n_rows = 1 + len(frames)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(10.2, 3.25 * n_rows), constrained_layout=True
    )

    _draw_panel(
        fig, axes[0, 0], {"array": bathymetry, "title": "Bathymetry", "cmap": "terrain"}
    )
    _draw_panel(
        fig, axes[0, 1], {"array": source, "title": "Initial source", "cmap": "RdBu_r"}
    )

    ax_text = axes[0, 2]
    ax_text.axis("off")
    metrics_text = (
        f"{meta['scenario_id']}\n"
        f"{meta['bathymetry_type']} / {meta['source_type']}\n"
        f"window K = {meta['window_K']}\n"
        f"rollout rel-L2 = {global_rel_l2:.3f}\n"
        f"frames: {', '.join(str(f) for f in frames)}"
    )
    ax_text.text(
        0.5,
        0.5,
        metrics_text,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        transform=ax_text.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="0.95", edgecolor="0.6"),
    )

    for row, frame in enumerate(frames, start=1):
        ref = target_full[frame]
        pred = pred_full[frame]
        err = np.abs(pred - ref)
        eta_vmax = max(float(np.nanmax(np.abs(ref))), float(np.nanmax(np.abs(pred))))
        eta_vmax = eta_vmax if eta_vmax > 0.0 else 1.0
        err_vmax = float(np.nanmax(err)) if float(np.nanmax(err)) > 0.0 else 1.0
        eta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-eta_vmax, vmax=eta_vmax)

        _draw_panel(
            fig,
            axes[row, 0],
            {
                "array": ref,
                "title": "Reference eta" if row == 1 else "",
                "cmap": "RdBu_r",
                "norm": eta_norm,
            },
        )
        _draw_panel(
            fig,
            axes[row, 1],
            {
                "array": pred,
                "title": "Window rollout eta" if row == 1 else "",
                "cmap": "RdBu_r",
                "norm": eta_norm,
            },
        )
        _draw_panel(
            fig,
            axes[row, 2],
            {
                "array": err,
                "title": "|error|" if row == 1 else "",
                "cmap": "magma",
                "vmin": 0.0,
                "vmax": err_vmax,
            },
        )
        label = "seed" if frame == 0 else f"rel-L2={frame_rel_l2[row - 1]:.2f}"
        axes[row, 0].set_ylabel(f"frame {frame}\n{label}", fontsize=10, labelpad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if png_output_path is not None:
        png_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--stats", default=DEFAULT_STATS)
    p.add_argument("--sample-index", type=int, default=DEFAULT_SAMPLE_INDEX)
    p.add_argument(
        "--sample-id",
        default=None,
        help="Exact sample_id or scenario_id. Overrides --sample-index.",
    )
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[1, 10, 25, 49],
        help="Actual trajectory frame numbers to plot.",
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--png-output", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    data_cfg = dict(cfg.get("data", {}))
    K = int(data_cfg.get("window_K", 5))
    include_source = bool(data_cfg.get("window_include_source", True))
    use_prev = bool(data_cfg.get("window_prev", True))

    dataset = ShardedTsunamiDataset(args.processed_path)
    sample_index = (
        _find_sample_index(dataset, args.sample_id)
        if args.sample_id
        else int(args.sample_index)
    )
    item = dataset[sample_index]
    x = item["x"].unsqueeze(0)
    y = item["y"].unsqueeze(0)
    _, T = y.shape[:2]

    frames = [int(f) for f in args.frames]
    for frame in frames:
        if frame < 0 or frame >= T:
            raise ValueError(f"frame {frame} out of range [0, {T - 1}]")

    device = resolve_device(args.device)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    with torch.no_grad():
        pred_tail = rollout_trajectory(
            model,
            x.to(device),
            y[:, 0].to(device),
            int(T),
            K,
            include_source,
            use_prev,
            device,
        )
    target_tail = y[:, 1:].to(device)
    pred_full = torch.cat([y[:, :1].to(device), pred_tail], dim=1)

    target_denorm = None
    try:
        target_denorm = load_target_denorm(args.processed_path)
    except Exception:
        target_denorm = None
    target_phys = (
        apply_target_denorm(y.to(device), target_denorm)[0].detach().cpu().numpy()
    )
    pred_phys = apply_target_denorm(pred_full, target_denorm)[0].detach().cpu().numpy()

    stats = _load_json(Path(args.stats))
    x_np = item["x"].detach().cpu().numpy()
    bathymetry = _denorm_input(x_np[0], stats, "bathymetry")
    source = _denorm_input(x_np[1], stats, "source")

    frame_rel_l2 = [_rel_l2(pred_phys[f], target_phys[f]) for f in frames]
    global_rel_l2 = _rel_l2(pred_phys[1:], target_phys[1:])
    meta = {
        "sample_index": int(sample_index),
        "scenario_id": item["scenario_id"],
        "bathymetry_type": item["bathymetry_type"],
        "source_type": item["source_type"],
        "source_strength": float(item["source_strength"]),
        "window_K": int(K),
    }

    output = Path(args.output)
    png_output = (
        Path(args.png_output) if args.png_output else output.with_suffix(".png")
    )
    _plot(
        bathymetry,
        source,
        target_phys,
        pred_phys,
        frames,
        frame_rel_l2,
        meta,
        global_rel_l2,
        output,
        png_output,
    )

    print(f"sample_index={sample_index}")
    print(f"scenario_id={meta['scenario_id']}")
    print(
        f"bathymetry_type={meta['bathymetry_type']} source_type={meta['source_type']}"
    )
    print(f"source_strength={meta['source_strength']:.4f}")
    print(f"window_K={K}")
    print(f"rollout_rel_l2_physical={global_rel_l2:.4f}")
    for frame, rel in zip(frames, frame_rel_l2):
        print(f"frame {frame}: rel_l2={rel:.4f}")
    print(f"saved_pdf={output}")
    print(f"saved_png={png_output}")


if __name__ == "__main__":
    main()
