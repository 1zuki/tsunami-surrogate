#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from copy import deepcopy
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
    seeds = cfg.get('ensemble', {}).get('seeds', [cfg.get('seed', 42)])
    device = resolve_device(cfg.get('device', 'auto'))
    for seed in seeds:
        member_cfg = deepcopy(cfg)
        member_cfg['seed'] = int(seed)
        member_cfg['output_dir'] = cfg.get('ensemble', {}).get('member_dir_template', 'experiments/ensemble/member_{seed}').format(seed=seed)
        seed_everything(int(seed))
        init_run(member_cfg['output_dir'], member_cfg)
        loaders = create_dataloaders(member_cfg)
        model = build_model(member_cfg)
        Trainer(model, loaders, member_cfg, device).fit()


if __name__ == '__main__':
    main()
