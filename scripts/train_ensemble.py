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
from src.utils.model_io import validate_model_io_channels
from src.data.dataset import create_dataloaders
from src.models import build_model
from src.training.train import Trainer


def _require_fresh_member(output_dir: str | Path) -> None:
    output = Path(output_dir)
    existing = [
        path
        for path in (
            output / "history.json",
            output / "best.pt",
            output / "checkpoints" / "last.pt",
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing ensemble member artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def _parse_seeds(raw: str | None, configured: object) -> list[int]:
    values = (
        configured
        if raw is None
        else [part.strip() for part in raw.split(",") if part.strip()]
    )
    if not isinstance(values, list) or not values:
        raise ValueError(
            "Ensemble seeds must be configured explicitly or passed with --seeds"
        )
    if raw is None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError("Configured ensemble seeds must be integers")
        seeds = list(values)
    else:
        try:
            seeds = [int(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("Ensemble seeds must be integers") from exc
    if len(seeds) != len(set(seeds)):
        raise ValueError("Ensemble seeds must not contain duplicates")
    return seeds


def _parse_resume_members(values: list[str]) -> dict[int, Path]:
    resumes: dict[int, Path] = {}
    for value in values:
        seed_text, separator, checkpoint = value.partition("=")
        if not separator or not seed_text.strip() or not checkpoint.strip():
            raise ValueError(
                "--resume-member must use SEED=CHECKPOINT syntax"
            )
        seed = int(seed_text)
        if seed in resumes:
            raise ValueError(f"Duplicate resume checkpoint for seed {seed}")
        resumes[seed] = Path(checkpoint)
    return resumes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument(
        '--seeds',
        default=None,
        help='Comma-separated explicit member seeds. Overrides ensemble.seeds.',
    )
    p.add_argument(
        '--resume-member',
        action='append',
        default=[],
        metavar='SEED=CHECKPOINT',
        help='Resume one selected member from its own checkpoint. Repeatable.',
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    try:
        seeds = _parse_seeds(
            args.seeds,
            cfg.get('ensemble', {}).get('seeds'),
        )
        resume_members = _parse_resume_members(args.resume_member)
    except ValueError as exc:
        p.error(str(exc))
    unexpected_resumes = sorted(set(resume_members) - set(seeds))
    if unexpected_resumes:
        p.error(
            '--resume-member seeds must also be selected by --seeds/config: '
            + ', '.join(str(seed) for seed in unexpected_resumes)
        )
    device = resolve_device(cfg.get('device', 'auto'))

    plans = []
    for seed in seeds:
        member_cfg = deepcopy(cfg)
        member_cfg['seed'] = int(seed)
        member_cfg['output_dir'] = cfg.get('ensemble', {}).get('member_dir_template', 'experiments/ensemble/member_{seed}').format(seed=seed)
        resume_path = resume_members.get(int(seed))
        if resume_path is None:
            _require_fresh_member(member_cfg['output_dir'])
        elif not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        plans.append((seed, member_cfg, resume_path))

    for seed, member_cfg, resume_path in plans:
        seed_everything(int(seed))
        init_run(
            member_cfg['output_dir'],
            member_cfg,
            fresh=resume_path is None,
        )
        loaders = create_dataloaders(member_cfg)
        validate_model_io_channels(member_cfg, loaders, preferred_splits=("train", "val", "test"))
        model = build_model(member_cfg)
        Trainer(model, loaders, member_cfg, device).fit(
            resume_path=resume_path
        )


if __name__ == '__main__':
    main()
