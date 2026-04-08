from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from src.evaluation._common import benchmark_model, benchmark_solver, load_checkpoint_and_model, make_eval_loader, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark surrogate inference speed against the simulator.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--solver-samples", type=int, default=8)
    args = parser.parse_args()

    config, model, stats, device, state = load_checkpoint_and_model(args.config, args.checkpoint)
    loader = make_eval_loader(config, split=args.split, return_meta=False)
    model_stats = benchmark_model(model, loader, device)
    solver_stats = benchmark_solver(config, loader, stats, n_samples=args.solver_samples)
    speedup = solver_stats["mean_sample_seconds"] / max(model_stats["mean_sample_seconds"], 1e-12)
    summary = {
        "model": model_stats,
        "solver": solver_stats,
        "speedup_vs_solver": float(speedup),
    }
    out_dir = Path(config.get("paths", {}).get("output_root", "results/default_run")) / "eval_speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / "speed.json")
    print(summary)


if __name__ == "__main__":
    main()
