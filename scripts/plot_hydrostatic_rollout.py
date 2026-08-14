#!/usr/bin/env python
"""Plot the qualitative single-pass FNO rollout diagnostic on the hydrostatic target.

Produces a multi-panel snapshot for one test scenario (default: a continental
bathymetry + Okada-like source sample): the de-normalized bathymetry and source
on the top row, then the reference eta, the single-pass FNO eta, and the
absolute error at an early, mid, and late frame of the 50-step rollout.

The sample is read from the sharded processed test split; the FNO prediction is
computed from the model checkpoint. All eta/error panels are in de-normalized
physical surface-elevation units.
"""

from __future__ import annotations

import argparse
import glob
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

from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config


DEFAULT_PROCESSED = "data/processed/hydrostatic/test"
DEFAULT_STATS = "data/processed/hydrostatic/normalization_stats.json"
DEFAULT_OUTPUT = "paper/figures/hydrostatic_rollout.pdf"
# Default sample: continental + Okada-like, near-median single-pass FNO error
# with clearly visible wave amplitude (scenario_000857).
DEFAULT_SAMPLE_INDEX = 856


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _denorm_input(arr: np.ndarray, stats: dict[str, Any], name: str) -> np.ndarray:
    channel = stats.get("inputs", {}).get(name)
    if not channel:
        return arr
    return arr * float(channel["scale"]) + float(channel["offset"])


def _denorm_target(arr: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    target = stats["targets"]
    return arr * float(target["scale"]) + float(target["offset"])


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return float(
        np.linalg.norm((pred64 - target64).ravel())
        / (np.linalg.norm(target64.ravel()) + 1e-12)
    )


def _load_sharded_sample(
    processed_path: str, sample_index: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (inputs[3,H,W], targets[T,H,W], meta) for one global sample index."""
    shards = sorted(glob.glob(str(Path(processed_path) / "shards" / "*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shards under {processed_path}/shards")
    offset = 0
    for sp in shards:
        d = np.load(sp, allow_pickle=True)
        n = int(d["inputs"].shape[0])
        if sample_index < offset + n:
            li = sample_index - offset
            inputs = d["inputs"][li].astype(np.float32)
            targets = d["targets"][li].astype(np.float32)
            meta = {
                "scenario_id": str(d["scenario_id"][li]),
                "bathymetry_type": str(d["bathymetry_type"][li]),
                "source_type": str(d["source_type"][li]),
                "source_strength": float(d["source_strength"][li]),
            }
            return inputs, targets, meta
        offset += n
    raise IndexError(f"sample_index {sample_index} out of range (total {offset})")


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
    target_phys: np.ndarray,
    pred_phys: np.ndarray,
    frames: list[int],
    frame_rel_l2: list[float],
    meta: dict[str, Any],
    global_rel_l2: float,
    output_path: Path,
    png_output_path: Path | None,
) -> None:
    """Layout: row 0 = bathymetry, source, metrics text box; one row per frame
    below, each showing reference eta | FNO eta | absolute error.

    Because the wavefield amplitude decays strongly over the rollout (sponge
    damping + open-boundary radiation), each frame row uses its own symmetric
    eta scale, shared between the reference and prediction panels so that the
    visual match at that frame is meaningful; the error panel uses the same
    per-row magnitude.
    """
    n_rows = 1 + len(frames)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(10.0, 3.3 * n_rows), constrained_layout=True
    )

    # Top row: bathymetry, source, then a metrics text box.
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
        f"full rel-$L_2$ = {global_rel_l2:.3f}\n"
        f"frames: {', '.join(str(f) for f in frames)}"
    )
    ax_text.text(
        0.5,
        0.5,
        metrics_text,
        ha="center",
        va="center",
        fontsize=11,
        family="monospace",
        transform=ax_text.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="0.95", edgecolor="0.6"),
    )

    col_titles = ["Reference $\\eta$", "FNO $\\eta$", "$|$error$|$"]
    for r, frame in enumerate(frames, start=1):
        ref = target_phys[frame]
        pred = pred_phys[frame]
        err = np.abs(pred - ref)
        eta_vmax = max(float(np.nanmax(np.abs(ref))), float(np.nanmax(np.abs(pred))))
        eta_vmax = eta_vmax if eta_vmax > 0 else 1.0
        eta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-eta_vmax, vmax=eta_vmax)
        err_vmax = float(np.nanmax(err)) if float(np.nanmax(err)) > 0 else 1.0

        title_ref = col_titles[0] if r == 1 else ""
        title_pred = col_titles[1] if r == 1 else ""
        title_err = col_titles[2] if r == 1 else ""
        _draw_panel(
            fig,
            axes[r, 0],
            {"array": ref, "title": title_ref, "cmap": "RdBu_r", "norm": eta_norm},
        )
        _draw_panel(
            fig,
            axes[r, 1],
            {"array": pred, "title": title_pred, "cmap": "RdBu_r", "norm": eta_norm},
        )
        _draw_panel(
            fig,
            axes[r, 2],
            {
                "array": err,
                "title": title_err,
                "cmap": "magma",
                "vmin": 0.0,
                "vmax": err_vmax,
            },
        )
        axes[r, 0].set_ylabel(
            f"frame {frame}\nrel-$L_2$={frame_rel_l2[r - 1]:.2f}",
            fontsize=10,
            labelpad=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if png_output_path is not None:
        fig.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/model/fno.yaml")
    p.add_argument("--checkpoint", default="experiments/fno/best.pt")
    p.add_argument("--processed-path", default=DEFAULT_PROCESSED)
    p.add_argument("--stats", default=DEFAULT_STATS)
    p.add_argument("--sample-index", type=int, default=DEFAULT_SAMPLE_INDEX)
    p.add_argument(
        "--frames",
        type=int,
        nargs=3,
        default=[0, 24, 49],
        help="Early, mid, late 0-based frame indices into the 50-frame rollout.",
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--png-output", default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    stats = json.load(open(args.stats))
    inputs, targets, meta = _load_sharded_sample(args.processed_path, args.sample_index)

    cfg = load_config(args.config)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(inputs[np.newaxis]).to(device)
        pred = _model_output(model, x)[0].detach().cpu().numpy()

    if pred.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {pred.shape} does not match targets {targets.shape}"
        )

    bathymetry = _denorm_input(inputs[0], stats, "bathymetry")
    source = _denorm_input(inputs[1], stats, "source")
    target_phys = _denorm_target(targets, stats)
    pred_phys = _denorm_target(pred, stats)

    frames = [int(f) for f in args.frames]
    for f in frames:
        if f < 0 or f >= targets.shape[0]:
            raise ValueError(f"frame {f} out of range [0, {targets.shape[0] - 1}]")

    output = Path(args.output)
    png_output = (
        Path(args.png_output) if args.png_output else output.with_suffix(".png")
    )
    frame_rel_l2 = [_rel_l2(pred_phys[f], target_phys[f]) for f in frames]
    global_rel_l2 = _rel_l2(pred_phys, target_phys)
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

    print(f"scenario_id={meta['scenario_id']}")
    print(
        f"bathymetry_type={meta['bathymetry_type']} source_type={meta['source_type']}"
    )
    print(f"source_strength={meta['source_strength']:.4f}")
    print(f"global_rel_l2_physical={global_rel_l2:.4f}")
    for f in frames:
        print(f"frame {f}: rel_l2={_rel_l2(pred_phys[f], target_phys[f]):.4f}")
    print(f"saved_pdf={output}")
    print(f"saved_png={png_output}")


if __name__ == "__main__":
    main()
