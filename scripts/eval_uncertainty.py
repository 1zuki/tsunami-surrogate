#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import torch
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.models.ensemble import EnsemblePredictor
from src.training.checkpointing import load_checkpoint
from src.evaluation.calibration import interval_calibration
from src.evaluation.uncertainty import error_uncertainty_correlation
from src.utils.io import save_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    ckpts = cfg.get('uncertainty', {}).get('ensemble_checkpoints', [])

    if not ckpts:
        print('No ensemble_checkpoints configured; uncertainty eval skipped gracefully.')
        return

    cfg['data'] = {'path': cfg['eval']['dataset_path'], 'batch_size': cfg['eval'].get('batch_size', 8), 'split': {'type': 'iid'}}
    device = resolve_device(cfg.get('device', 'auto'))
    members = []

    for path in ckpts:
        model = build_model(torch.load(path, map_location='cpu').get('config', cfg)).to(device)
        load_checkpoint(path, model, map_location=device)
        members.append(model)

    ensemble = EnsemblePredictor(members).to(device).eval()
    loaders = create_dataloaders(cfg)
    levels = cfg.get('uncertainty', {}).get('interval_levels', [0.5, 0.8, 0.9, 0.95])
    results = []

    for batch in loaders['test']:
        x, y = batch['x'].to(device), batch['y'].to(device)
        out = ensemble(x)
        row = interval_calibration(out['mean'], out['variance'], y, levels)
        row['error_uncertainty_corr'] = error_uncertainty_correlation(out['mean'], out['variance'], y)
        results.append(row)

    mean_results = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
    print(mean_results)
    save_json(mean_results, f"{cfg['eval'].get('output_dir', 'experiments/eval_uncertainty')}/uncertainty.json")


if __name__ == '__main__':
    main()
