#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
import torch
from tsunami_surrogate.utils.config import load_config
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.data.dataset import create_dataloaders
from tsunami_surrogate.models import build_model
from tsunami_surrogate.training.checkpointing import load_checkpoint
from tsunami_surrogate.evaluation.visualize import save_prediction_triplet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default='experiments/figures/sample_prediction.png')
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    loaders = create_dataloaders(cfg)
    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    batch = next(iter(loaders['test']))
    with torch.no_grad():
        pred = model(batch['x'].to(device)).cpu()
    save_prediction_triplet(batch['x'], pred, batch['y'], args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
