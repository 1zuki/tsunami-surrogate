#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import torch
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.visualize import save_prediction_triplet


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default='experiments/figures/sample_prediction.png')
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError("No test dataloader found. Set eval.dataset_path or data.test_path.")
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))
    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    batch = next(iter(test_loader))

    with torch.no_grad():
        pred = _model_output(model, batch['x'].to(device)).cpu()

    save_prediction_triplet(batch['x'], pred, batch['y'], args.out)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
