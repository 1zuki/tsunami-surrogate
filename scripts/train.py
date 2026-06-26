#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.utils.device import resolve_device
from src.utils.experiment import init_run
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.train import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None,
                   help="Path to a checkpoint to resume from (e.g. experiments/fno/checkpoints/last.pt). "
                        "Restores model, optimizer, scheduler, early-stopping state, and history.")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    out = init_run(cfg.get('output_dir', 'experiments/default'), cfg, fresh=args.resume is None)
    cfg['output_dir'] = str(out)
    device = resolve_device(cfg.get('device', 'auto'))
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
    trainer.fit(resume_path=args.resume)


if __name__ == '__main__':
    main()
