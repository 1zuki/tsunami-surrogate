#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen.simulate_dataset import TsunamiDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw tsunami dataset via full simulator pipeline.")
    parser.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("dataset", {})
    ds = cfg["dataset"]
    if args.num_samples is not None:
        ds["num_samples"] = int(args.num_samples)
    if args.n_steps is not None:
        ds["n_steps"] = int(args.n_steps)
    if args.save_every is not None:
        ds["save_every"] = int(args.save_every)
    if args.num_workers is not None:
        ds["num_workers"] = int(args.num_workers)
    if args.output_dir is not None:
        ds["output_dir"] = str(args.output_dir)
    if args.manifest_path is not None:
        ds["manifest_path"] = str(args.manifest_path)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False)
        tmp_cfg = Path(tf.name)

    try:
        builder = TsunamiDatasetBuilder(str(tmp_cfg))
        builder.run()
        print(f"Dataset generation complete using {cfg_path}")
    finally:
        try:
            tmp_cfg.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
