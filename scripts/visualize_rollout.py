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
        default="data/processed/test",
        help="Processed split directory or eval_dataset.npz path (default: data/processed/test)",
    )
    p.add_argument("--raw-dir", default="data/raw/samples", help="Raw sample directory for physical bathymetry/timestamps")
    p.add_argument("--sample-id", default=None, help="Exact sample id (e.g. sample_000123). Overrides --sample-index.")
    p.add_argument("--sample-index", type=int, default=0, help="0-based sample index when --sample-id is not set.")
    p.add_argument("--mc-samples", type=int, default=0, help="MC forward passes for epistemic uncertainty proxy.")
    p.add_argument("--device", default="auto", help="torch device (auto/cpu/cuda)")
    p.add_argument("--interval", type=int, default=120, help="Animation interval in milliseconds")
    p.add_argument("--repeat", action="store_true", help="Loop animation")
    p.add_argument("--elev", type=float, default=35.0, help="3D camera elevation")
    p.add_argument("--azim", type=float, default=-60.0, help="3D camera azimuth")
    p.add_argument("--max-frames", type=int, default=None, help="Cap number of animated timesteps")
    p.add_argument("--save", default=None, help="Optional output file (.gif/.mp4). If omitted, opens interactive window.")
    args = p.parse_args()

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
        max_frames=args.max_frames,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
