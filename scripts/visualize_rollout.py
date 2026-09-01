#!/usr/bin/env python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.evaluation.visualize import run_visualization


def main():
    p = argparse.ArgumentParser(description="Visualize one tsunami sample: bathymetry, rollout, prediction, and uncertainty.")
    p.add_argument("--config", required=True, help="Model config YAML (e.g. configs/model/fno.yaml)")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path (e.g. experiments/fno/best.pt)")
    p.add_argument(
        "--processed-path",
        default="auto",
        help=(
            "Processed split directory or eval_dataset.npz path. The default "
            "uses the selected model config's test dataset."
        ),
    )
    p.add_argument(
        "--raw-dir",
        default="auto",
        help=(
            "Raw sample directory for physical bathymetry/timestamps. "
            "The default resolves data/<split>/raw/<reference>/samples from "
            "--processed-path."
        ),
    )
    p.add_argument("--sample-id", default=None, help="Exact sample id (e.g. sample_000123). Overrides --sample-index.")
    p.add_argument(
        "--sample-index",
        type=int,
        default=1,
        help=(
            "1-based sample number when --sample-id is not set "
            "(1 selects sample_000001)."
        ),
    )
    p.add_argument("--mc-samples", type=int, default=0, help="MC forward passes for epistemic uncertainty proxy.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="torch device (auto/cpu/cuda)")
    p.add_argument("--interval", type=int, default=120, help="Animation interval in milliseconds")
    p.add_argument("--repeat", action="store_true", help="Loop animation")
    p.add_argument("--elev", type=float, default=35.0, help="3D camera elevation")
    p.add_argument("--azim", type=float, default=-60.0, help="3D camera azimuth")
    p.add_argument(
        "--wave-scale",
        type=float,
        default=None,
        help="Vertical multiplier for wave eta in 3D plots (default: auto)",
    )
    p.add_argument(
        "--wave-3d-mode",
        choices=["eta", "overlay"],
        default="eta",
        help="3D rendering mode: eta-only surface or bathymetry overlay (default: eta)",
    )
    p.add_argument(
        "--eta-limit",
        type=float,
        default=None,
        help=(
            "Fixed symmetric eta bound used by every 2D and 3D wave frame. "
            "The default is the full-rollout maximum absolute target/prediction value."
        ),
    )
    p.add_argument("--max-frames", type=int, default=None, help="Cap number of animated timesteps")
    p.add_argument(
        "--cache-dpi",
        type=int,
        default=80,
        help="DPI used while pre-rendering the in-memory frame cache (default: 80)",
    )
    p.add_argument("--save", default=None, help="Optional output file (.gif/.mp4). If omitted, opens interactive window.")
    args = p.parse_args()
    if args.sample_id is None and args.sample_index < 1:
        p.error("--sample-index must be >= 1 (1 selects sample_000001)")

    run_visualization(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        processed_path=args.processed_path,
        raw_dir=args.raw_dir,
        sample_id=args.sample_id,
        sample_index=args.sample_index,
        mc_samples=args.mc_samples,
        device=args.device,
        interval_ms=args.interval,
        repeat=args.repeat,
        elev=args.elev,
        azim=args.azim,
        wave_scale=args.wave_scale,
        wave_3d_mode=args.wave_3d_mode,
        eta_limit=args.eta_limit,
        max_frames=args.max_frames,
        cache_dpi=args.cache_dpi,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
