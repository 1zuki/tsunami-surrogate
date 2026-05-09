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
from src.evaluation.benchmark import benchmark_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)

    args = p.parse_args()
    cfg = load_config(args.config)
    cfg['data'] = {'test_path': cfg['eval']['dataset_path'], 'batch_size': cfg['eval'].get('batch_size', 8)}
    device = resolve_device(cfg.get('device', 'auto'))
    loaders = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = benchmark_inference(model, loaders['test'], device, cfg['eval'].get('warmup_steps', 5), cfg['eval'].get('timed_steps', 20))
    print(metrics)
    save_json(metrics, f"{cfg['eval'].get('output_dir', 'experiments/eval_speed')}/speed.json")


if __name__ == '__main__':
    main()
