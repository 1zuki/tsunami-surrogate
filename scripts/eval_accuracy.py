#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.accuracy import evaluate_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)

    args = p.parse_args()
    cfg = load_config(args.config)

    if "data" not in cfg:
        cfg["data"] = {
            "test_path": cfg["eval"]["dataset_path"],
            "batch_size": cfg["eval"].get("batch_size", 8),
        }

    device = resolve_device(cfg.get("device", "auto"))
    loaders = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate_accuracy(model, loaders.get("test") or list(loaders.values())[-1], device)
    print(metrics)
    save_json(metrics, f"{cfg.get('eval', {}).get('output_dir', 'experiments/eval_accuracy')}/metrics.json")


if __name__ == "__main__":
    main()
