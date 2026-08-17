#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from copy import deepcopy
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.device import resolve_device
from src.utils.experiment import init_run
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.train import Trainer


def resolve_training_seeds(cfg):
    if 'seeds' not in cfg:
        return [int(cfg.get('seed', 42))], False

    raw_seeds = cfg['seeds']
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError('seeds must be a non-empty list of integers')
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise ValueError('seeds must be a non-empty list of integers')
    if len(set(raw_seeds)) != len(raw_seeds):
        raise ValueError('seeds must not contain duplicates')

    return list(raw_seeds), True


def seed_output_dir(base_output_dir, seed, list_mode):
    base = Path(base_output_dir)
    if not list_mode:
        return base
    return base / f'{base.name}_seed_{seed}'


def train_one(cfg, device, resume_path=None):
    seed_everything(int(cfg.get('seed', 42)))
    out = init_run(cfg.get('output_dir', 'experiments/default'), cfg, fresh=resume_path is None)
    cfg['output_dir'] = str(out)
    loaders = create_dataloaders(cfg)
    validate_model_io_channels(cfg, loaders, preferred_splits=("train", "val", "test"))
    split_sizes = {name: len(loader.dataset) for name, loader in loaders.items()}
    print(f"[train] split sizes: {split_sizes}")
    save_json({
        "split_sizes": split_sizes,
        "data_limits": {
            key: cfg.get("data", {}).get(key)
            for key in (
                "n_samples",
                "train_samples",
                "val_samples",
                "test_samples",
                "n_train_samples",
                "n_val_samples",
                "n_test_samples",
            )
            if key in cfg.get("data", {})
        },
    }, out / "split_sizes.json")
    train_n = split_sizes.get("train", 0)
    if train_n < 100:
        print(
            f"[train][warning] train split has only {train_n} samples. "
            "For stable surrogate learning, this is usually too small and can collapse to near-mean predictions."
        )
    model = build_model(cfg)
    trainer = Trainer(model, loaders, cfg, device)
    trainer.fit(resume_path=resume_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None,
                   help="Path to a checkpoint to resume from (e.g. experiments/fno/checkpoints/last.pt). "
                        "Restores model, optimizer, scheduler, early-stopping state, and history.")
    p.add_argument('--seed', type=int, default=None,
                   help="Run only this seed from a multi-seed config. "
                        "Useful for independent launch and resume commands.")
    args = p.parse_args()
    cfg = load_config(args.config)

    try:
        seeds, list_mode = resolve_training_seeds(cfg)
    except ValueError as exc:
        p.error(str(exc))

    if args.seed is not None:
        if args.seed not in seeds:
            p.error(f'--seed must be one of the configured seeds: {seeds}')
        seeds = [args.seed]

    if args.resume is not None and list_mode and len(seeds) > 1:
        p.error('--resume cannot be used with multiple seeds; keep only the seed being resumed')

    base_output_dir = cfg.get('output_dir', 'experiments/default')
    device = resolve_device(cfg.get('device', 'auto'))
    for index, seed in enumerate(seeds, start=1):
        run_cfg = deepcopy(cfg)
        run_cfg['seed'] = int(seed)
        run_output_dir = seed_output_dir(base_output_dir, seed, list_mode)
        run_cfg['output_dir'] = str(run_output_dir)
        if list_mode:
            eval_cfg = deepcopy(run_cfg.get('eval', {}))
            eval_cfg['output_dir'] = str(run_output_dir / 'eval')
            run_cfg['eval'] = eval_cfg
        print(
            f'[train] run {index}/{len(seeds)} seed={seed} '
            f'output_dir={run_cfg["output_dir"]}'
        )
        train_one(run_cfg, device, resume_path=args.resume)


if __name__ == '__main__':
    main()
