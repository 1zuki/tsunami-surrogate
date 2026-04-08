from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np

from src.solver.source_modelling import random_source
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import plot_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and preview a random tsunami source field.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default="data/bathymetry/source_preview.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sim = config.get("simulation", {})
    field, meta = random_source(config, rng, int(sim.get("nx", 32)), int(sim.get("ny", 32)), float(sim.get("dx", 1.0)), float(sim.get("dy", 1.0)))
    plot_fields({"Initial disturbance": field}, args.output, title=str(meta), cmap=config.get("visualization", {}).get("cmap_wave", "RdBu_r"))


if __name__ == "__main__":
    main()
