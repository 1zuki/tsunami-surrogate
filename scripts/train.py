#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from tsunami_surrogate.utils.config import load_config
from tsunami_surrogate.utils.seed import seed_everything
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.utils.experiment import init_run
from tsunami_surrogate.data.dataset import create_dataloaders
from tsunami_surrogate.models import build_model
from tsunami_surrogate.training.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    out = init_run(cfg.get('output_dir', 'experiments/default'), cfg)
    cfg['output_dir'] = str(out)
    device = resolve_device(cfg.get('device', 'auto'))
    loaders = create_dataloaders(cfg)
    model = build_model(cfg)
    trainer = Trainer(model, loaders, cfg, device)
    trainer.fit()


if __name__ == '__main__':
    main()
