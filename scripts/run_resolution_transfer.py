#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from torch.utils.data import DataLoader
from tsunami_surrogate.utils.config import load_config
from tsunami_surrogate.utils.device import resolve_device
from tsunami_surrogate.data.multires_dataset import MultiResolutionDataset
from tsunami_surrogate.models import build_model
from tsunami_surrogate.training.metrics import compute_metrics
from tsunami_surrogate.utils.io import save_json
import torch


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    resolutions = cfg.get('resolution_transfer', {}).get('eval_resolutions', [32, 64])
    ds = MultiResolutionDataset(cfg['data']['path'], resolutions)
    loader = DataLoader(ds, batch_size=cfg['data'].get('batch_size', 8))
    model = build_model(cfg).to(device).eval()
    rows = {}
    for res in resolutions:
        sums, n = {'mae':0.0,'rmse':0.0,'rel_l2':0.0,'max_error':0.0}, 0
        for batch in loader:
            x = batch[f'x_{res}'].to(device)
            y = batch[f'y_{res}'].to(device)
            pred = model(x)
            metrics = compute_metrics(pred, y)
            for k, v in metrics.items():
                sums[k] += v * x.size(0)
            n += x.size(0)
        rows[str(res)] = {k: v / max(1, n) for k, v in sums.items()}
    print(rows)
    save_json(rows, f"{cfg.get('output_dir', 'experiments/crossres')}/resolution_transfer.json")


if __name__ == '__main__':
    main()
