#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from tsunami_surrogate.utils.config import load_config
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.data.dataset import create_dataloaders
from tsunami_surrogate.models import build_model
from tsunami_surrogate.evaluation.generalization_suite import evaluate_by_regime
from tsunami_surrogate.utils.io import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    loaders = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    # In normal use, load a checkpoint before this benchmark. Here it is import-safe and smoke-testable.
    result = evaluate_by_regime(model, loaders['test'], device, key='source_id')
    print(result)
    save_json(result, f"{cfg.get('output_dir', 'experiments/ood')}/ood_by_source.json")


if __name__ == '__main__':
    main()
