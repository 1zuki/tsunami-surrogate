#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.evaluation.benchmark import benchmark_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)

    args = p.parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
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
    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test")
    
    if test_loader is None:
        raise KeyError("No test dataloader found. Set eval.dataset_path or data.test_path.")
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))
    
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = benchmark_inference(
        model,
        test_loader,
        device,
        eval_cfg.get('warmup_steps', 5),
        eval_cfg.get('timed_steps', 20),
    )
    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    print(metrics)
    save_json(metrics, f"{output_dir}/speed.json")


if __name__ == '__main__':
    main()
