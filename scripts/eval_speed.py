#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from tsunami_surrogate.utils.config import load_config
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.utils.io import save_json
from tsunami_surrogate.data.dataset import create_dataloaders
from tsunami_surrogate.models import build_model
from tsunami_surrogate.training.checkpointing import load_checkpoint
from tsunami_surrogate.evaluation.benchmark import benchmark_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg['data'] = {'path': cfg['eval']['dataset_path'], 'batch_size': cfg['eval'].get('batch_size', 8), 'split': {'type': 'iid'}}
    device = resolve_device(cfg.get('device', 'auto'))
    loaders = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = benchmark_inference(model, loaders['test'], device, cfg['eval'].get('warmup_steps', 5), cfg['eval'].get('timed_steps', 20))
    print(metrics)
    save_json(metrics, f"{cfg['eval'].get('output_dir', 'experiments/eval_speed')}/speed.json")


if __name__ == '__main__':
    main()
