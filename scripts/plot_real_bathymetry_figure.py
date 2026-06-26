#!/usr/bin/env python
"""Plot the real-bathymetry qualitative diagnostics.

Produces two figures:

* ``--main-output`` -- a 5-panel snapshot for the ``main_morphology`` suite
  (bathymetry | source | reference eta | Direct-FNO |error| | Window-FNO-5 |error|).
  This is the main-paper morphology-transfer diagnostic.
* ``--appendix-output`` -- a grid showing all three suites (offshore morphology,
  coastline stress / wet-dry, coastline fully wet) with bathymetry, source, and
  reference eta per suite. This is the appendix stress-test overview.
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
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config


PROCESSED_ROOT = "data/processed_real_bathymetry"
DEFAULT_MAIN_OUTPUT = "paper/figures/real_bathymetry_offshore.pdf"
DEFAULT_APPENDIX_OUTPUT = "paper/figures/real_bathymetry_suites.pdf"

# (suite key, human label) for the appendix overview, in display order.
APPENDIX_SUITES = [
    ("main_morphology", "Offshore morphology"),
    ("appendix_coastline_stress", "Coastline wet-dry"),
    ("appendix_coastline_fully_wet", "Coastline fully wet"),
]


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _load_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module]:
    cfg = load_config(config_path)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return cfg, model


def _positive_vmax(*arrays: np.ndarray) -> float:
    vmax = max(float(np.nanmax(np.abs(arr))) for arr in arrays)
    return vmax if vmax > 0.0 else 1.0


def _suite_paths(processed_root: str, suite: str) -> tuple[Path, Path]:
    base = Path(processed_root) / suite / "hydrostatic"
    return base / "test", base / "normalization_stats.json"


def _load_suite_inputs(
    data_dir: Path, stats: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bathymetry, source, targets) de-normalized for one suite."""
    inputs = np.load(data_dir / "inputs.npy").astype(np.float32)
    targets = np.load(data_dir / "targets.npy").astype(np.float32)

    if inputs.shape[:2] != (1, 3):
        raise ValueError(f"Expected inputs shape [1, 3, H, W], got {inputs.shape}")
    if targets.ndim != 4 or targets.shape[0] != 1:
        raise ValueError(f"Expected targets shape [1, T, H, W], got {targets.shape}")

    bathymetry = _denorm_input(inputs[0, 0], stats, "bathymetry")
    source = _denorm_input(inputs[0, 1], stats, "source")
    target_phys = _denorm_target(targets[0], stats)
    return bathymetry, source, target_phys


def _draw_panel(fig, ax, panel: dict[str, Any]) -> None:
    im = ax.imshow(
        panel["array"],
        origin="upper",
        cmap=panel["cmap"],
        norm=panel.get("norm"),
        vmin=panel.get("vmin"),
        vmax=panel.get("vmax"),
    )
    ax.set_title(panel["title"], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(panel["label"], fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _save(fig, output_path: Path, png_output_path: Path | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if png_output_path is not None:
        png_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_main(
    bathymetry: np.ndarray,
    source: np.ndarray,
    target_frame: np.ndarray,
    window_pred_frame: np.ndarray,
    direct_error: np.ndarray,
    window_error: np.ndarray,
    frame_index: int,
    direct_rel_l2: float,
    window_rel_l2: float,
    output_path: Path,
    png_output_path: Path | None,
) -> None:
    eta_vmax = _positive_vmax(target_frame, window_pred_frame)
    err_vmax = _positive_vmax(direct_error, window_error)
    eta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-eta_vmax, vmax=eta_vmax)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2), constrained_layout=True)

    panels = [
        {
            "array": bathymetry,
            "title": "Bathymetry crop",
            "cmap": "terrain",
            "label": "de-normalized bathymetry",
        },
        {
            "array": source,
            "title": "Source field",
            "cmap": "viridis",
            "label": "de-normalized source",
        },
        {
            "array": target_frame,
            "title": f"Target eta, frame {frame_index}",
            "cmap": "RdBu_r",
            "norm": eta_norm,
            "label": "de-normalized eta",
        },
        {
            "array": direct_error,
            "title": f"Direct-FNO |error|\nrel-L2={direct_rel_l2:.3f}",
            "cmap": "magma",
            "vmin": 0.0,
            "vmax": err_vmax,
            "label": "absolute eta error",
        },
        {
            "array": window_error,
            "title": f"Window-FNO-5 |error|\nrel-L2={window_rel_l2:.3f}",
            "cmap": "magma",
            "vmin": 0.0,
            "vmax": err_vmax,
            "label": "absolute eta error",
        },
        {
            "array": window_pred_frame,
            "title": f"Window-FNO-5 eta, frame {frame_index}",
            "cmap": "RdBu_r",
            "norm": eta_norm,
            "label": "de-normalized eta",
        },
    ]

    for ax, panel in zip(axes.ravel(), panels):
        _draw_panel(fig, ax, panel)

    _save(fig, output_path, png_output_path)


def _plot_appendix(
    rows: list[dict[str, Any]],
    frame_index: int,
    output_path: Path,
    png_output_path: Path | None,
) -> None:
    """One row per suite: bathymetry | source | reference eta at frame_index."""
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(11.0, 3.4 * n_rows), constrained_layout=True
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, row in enumerate(rows):
        target_frame = row["target_phys"][frame_index]
        eta_vmax = _positive_vmax(target_frame)
        panels = [
            {
                "array": row["bathymetry"],
                "title": "Bathymetry crop",
                "cmap": "terrain",
                "label": "de-normalized bathymetry",
            },
            {
                "array": row["source"],
                "title": "Source field",
                "cmap": "viridis",
                "label": "de-normalized source",
            },
            {
                "array": target_frame,
                "title": f"Reference eta, frame {frame_index}",
                "cmap": "RdBu_r",
                "norm": TwoSlopeNorm(vcenter=0.0, vmin=-eta_vmax, vmax=eta_vmax),
                "label": "de-normalized eta",
            },
        ]
        for c, panel in enumerate(panels):
            _draw_panel(fig, axes[r, c], panel)
        axes[r, 0].set_ylabel(row["label"], fontsize=11, rotation=90, labelpad=10)

    _save(fig, output_path, png_output_path)


def _compute_main_errors(
    data_dir: Path,
    stats: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    frame: int,
) -> dict[str, Any]:
    inputs = np.load(data_dir / "inputs.npy").astype(np.float32)
    targets = np.load(data_dir / "targets.npy").astype(np.float32)
    if inputs.shape[:2] != (1, 3):
        raise ValueError(f"Expected inputs shape [1, 3, H, W], got {inputs.shape}")
    if targets.ndim != 4 or targets.shape[0] != 1:
        raise ValueError(f"Expected targets shape [1, T, H, W], got {targets.shape}")
    if frame <= 0 or frame >= targets.shape[1]:
        raise ValueError(
            f"frame-index must be in [1, {targets.shape[1] - 1}] for the window rollout"
        )

    x = torch.from_numpy(inputs).to(device)
    y = torch.from_numpy(targets).to(device)

    direct_cfg, direct_model = _load_model(
        Path(args.direct_config), Path(args.direct_checkpoint), device
    )
    window_cfg, window_model = _load_model(
        Path(args.window_config), Path(args.window_checkpoint), device
    )

    data_cfg = dict(window_cfg.get("data", {}))
    K = int(data_cfg.get("window_K", 5))
    include_source = bool(data_cfg.get("window_include_source", True))
    use_prev = bool(data_cfg.get("window_prev", True))

    with torch.no_grad():
        direct_pred = _model_output(direct_model, x)
        window_pred = rollout_trajectory(
            window_model,
            x,
            y[:, 0],
            targets.shape[1],
            K,
            include_source,
            use_prev,
            device,
        )

    if tuple(direct_pred.shape) != tuple(targets.shape):
        raise ValueError(
            f"Direct prediction shape {tuple(direct_pred.shape)} does not match targets {targets.shape}"
        )

    expected_window_shape = (
        targets.shape[0],
        targets.shape[1] - 1,
        targets.shape[2],
        targets.shape[3],
    )
    if tuple(window_pred.shape) != expected_window_shape:
        raise ValueError(
            f"Window prediction shape {tuple(window_pred.shape)} does not match {expected_window_shape}"
        )

    direct_norm = direct_pred.detach().cpu().numpy()
    window_norm = window_pred.detach().cpu().numpy()

    target_phys = _denorm_target(targets[0], stats)
    direct_phys = _denorm_target(direct_norm[0], stats)
    window_phys = _denorm_target(window_norm[0], stats)

    bathymetry = _denorm_input(inputs[0, 0], stats, "bathymetry")
    source = _denorm_input(inputs[0, 1], stats, "source")
    target_frame = target_phys[frame]
    window_pred_frame = window_phys[frame - 1]
    direct_error = np.abs(direct_phys[frame] - target_frame)
    window_error = np.abs(window_pred_frame - target_frame)

    return {
        "bathymetry": bathymetry,
        "source": source,
        "target_frame": target_frame,
        "window_pred_frame": window_pred_frame,
        "direct_error": direct_error,
        "window_error": window_error,
        "direct_rel_l2": _rel_l2(direct_phys, target_phys),
        "window_rel_l2": _rel_l2(window_phys, target_phys[1:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default=PROCESSED_ROOT)
    parser.add_argument("--main-suite", default="main_morphology")
    parser.add_argument("--direct-config", default="configs/model/fno.yaml")
    parser.add_argument("--direct-checkpoint", default="experiments/fno/best.pt")
    parser.add_argument(
        "--window-config", default="configs/model/fno_window5_hydrostatic.yaml"
    )
    parser.add_argument(
        "--window-checkpoint", default="experiments/fno_window5_hydrostatic/best.pt"
    )
    parser.add_argument("--main-output", default=DEFAULT_MAIN_OUTPUT)
    parser.add_argument("--appendix-output", default=DEFAULT_APPENDIX_OUTPUT)
    parser.add_argument(
        "--main-png-output",
        default=None,
        help="Optional PNG path. Defaults to main output with .png suffix.",
    )
    parser.add_argument(
        "--appendix-png-output",
        default=None,
        help="Optional PNG path. Defaults to appendix output with .png suffix.",
    )
    parser.add_argument("--frame-index", type=int, default=49)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-appendix", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    frame = int(args.frame_index)

    if not args.skip_main:
        data_dir, stats_path = _suite_paths(args.processed_root, args.main_suite)
        stats = _load_json(stats_path)
        res = _compute_main_errors(data_dir, stats, args, device, frame)
        main_output = Path(args.main_output)
        main_png = (
            Path(args.main_png_output)
            if args.main_png_output
            else main_output.with_suffix(".png")
        )
        _plot_main(
            res["bathymetry"],
            res["source"],
            res["target_frame"],
            res["window_pred_frame"],
            res["direct_error"],
            res["window_error"],
            frame,
            res["direct_rel_l2"],
            res["window_rel_l2"],
            main_output,
            main_png,
        )
        print(f"main_direct_rel_l2_de_normalized={res['direct_rel_l2']:.6f}")
        print(f"main_window_rel_l2_de_normalized={res['window_rel_l2']:.6f}")
        print(f"saved_main_pdf={main_output}")
        print(f"saved_main_png={main_png}")

    if not args.skip_appendix:
        rows: list[dict[str, Any]] = []
        for suite, label in APPENDIX_SUITES:
            data_dir, stats_path = _suite_paths(args.processed_root, suite)
            if not data_dir.exists():
                print(f"[warn] skipping appendix suite (missing): {data_dir}")
                continue
            stats = _load_json(stats_path)
            bathymetry, source, target_phys = _load_suite_inputs(data_dir, stats)
            rows.append(
                {
                    "label": label,
                    "bathymetry": bathymetry,
                    "source": source,
                    "target_phys": target_phys,
                }
            )
        if not rows:
            raise RuntimeError("No appendix suites found to plot.")
        appendix_output = Path(args.appendix_output)
        appendix_png = (
            Path(args.appendix_png_output)
            if args.appendix_png_output
            else appendix_output.with_suffix(".png")
        )
        _plot_appendix(rows, frame, appendix_output, appendix_png)
        print(f"appendix_suites={[r['label'] for r in rows]}")
        print(f"saved_appendix_pdf={appendix_output}")
        print(f"saved_appendix_png={appendix_png}")

    print(f"device={device}")
    print(f"frame_index={frame}")


if __name__ == "__main__":
    main()
