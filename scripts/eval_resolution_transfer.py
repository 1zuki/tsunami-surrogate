#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.multires_dataset import MultiResolutionDataset
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.training.metrics import compute_metrics
from src.utils.io import save_json
import torch


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)

    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get('device', 'auto'))
    resolutions = cfg.get('resolution_transfer', {}).get('eval_resolutions', [32, 64])
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    data_path = Path(eval_cfg.get("dataset_path", data_cfg.get("path", "")))
    if not str(data_path):
        raise KeyError(
            "Resolution-transfer dataset path is missing. Set `eval.dataset_path` (preferred) "
            "or `data.path` in the config."
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Resolution-transfer dataset not found: {data_path}. "
            "Prepare the dedicated cross-resolution data first (e.g., train_32/test_64)."
        )
    ds = MultiResolutionDataset(data_path, resolutions)
    loader = DataLoader(ds, batch_size=eval_cfg.get('batch_size', data_cfg.get('batch_size', 8)))
    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    rows = {}

    for res in resolutions:
        sums, n = {'mae':0.0,'rmse':0.0,'rel_l2':0.0,'max_error':0.0}, 0

        for batch in loader:
            x = batch[f'x_{res}'].to(device)
            y = batch[f'y_{res}'].to(device)
            pred = _model_output(model, x)
            metrics = compute_metrics(pred, y)

            for k, v in metrics.items():
                sums[k] += v * x.size(0)

            n += x.size(0)

        rows[str(res)] = {k: v / max(1, n) for k, v in sums.items()}

    print(rows)
    save_json(rows, f"{cfg.get('output_dir', 'experiments/crossres')}/resolution_transfer.json")


if __name__ == '__main__':
    main()
