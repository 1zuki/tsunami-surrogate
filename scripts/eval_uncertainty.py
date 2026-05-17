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
from src.models.ensemble import EnsemblePredictor
from src.training.checkpointing import load_checkpoint
from src.evaluation.calibration import interval_calibration
from src.evaluation.uncertainty import error_uncertainty_correlation
from src.utils.io import save_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument(
        '--checkpoint',
        action='append',
        default=None,
        help='Optional ensemble checkpoint path. Repeat this flag to pass multiple members.',
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    ckpts = list(cfg.get('uncertainty', {}).get('ensemble_checkpoints', []))
    if args.checkpoint:
        ckpts = [str(p) for p in args.checkpoint]
    if not ckpts:
        fallback = eval_cfg.get("checkpoint")
        if fallback:
            ckpts = [str(fallback)]

    if not ckpts:
        print(
            "No uncertainty checkpoints configured. "
            "Set uncertainty.ensemble_checkpoints, pass --checkpoint, or set eval.checkpoint."
        )
        return
    min_members = int(cfg.get("uncertainty", {}).get("min_ensemble_members", 2))
    if len(ckpts) < min_members:
        raise ValueError(
            f"Uncertainty evaluation requires at least {min_members} ensemble checkpoints, got {len(ckpts)}. "
            "Single-member runs produce degenerate ensemble variance."
        )

    data_cfg = dict(cfg.get("data", {}))
    dataset_cfg = cfg.get("dataset", {})
    if not data_cfg and isinstance(dataset_cfg, dict):
        dataset_path = dataset_cfg.get("path")
        if dataset_path:
            data_cfg["test_path"] = dataset_path
        if "batch_size" in dataset_cfg:
            data_cfg["batch_size"] = dataset_cfg["batch_size"]
    dataset_path = eval_cfg.get("dataset_path")
    if dataset_path:
        data_cfg["test_path"] = dataset_path
    if "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]
    cfg["data"] = data_cfg

    device = resolve_device(cfg.get('device', 'auto'))
    members = []

    for path in ckpts:
        model = build_model(torch.load(path, map_location='cpu').get('config', cfg)).to(device)
        load_checkpoint(path, model, map_location=device)
        members.append(model)

    ensemble = EnsemblePredictor(members).to(device).eval()
    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        print("No test loader found; uncertainty eval skipped gracefully.")
        return
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))
    levels = cfg.get('uncertainty', {}).get('interval_levels', [0.5, 0.8, 0.9, 0.95])
    results = []

    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        out = ensemble(x)
        row = interval_calibration(out['mean'], out['variance'], y, levels)
        row['error_uncertainty_corr'] = error_uncertainty_correlation(out['mean'], out['variance'], y)
        results.append(row)

    if not results:
        print("Test loader had zero batches; uncertainty eval skipped gracefully.")
        return

    mean_results = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    print(mean_results)
    save_json(mean_results, f"{output_dir}/uncertainty.json")


if __name__ == '__main__':
    main()
