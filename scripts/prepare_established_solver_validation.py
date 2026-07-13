#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.established_solver_validation import (
    prepare_minimum_established_solver_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze minimum independent GeoClaw validation inputs and in-house outputs"
        )
    )
    parser.add_argument("--level-a-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/minimum_established_solver_validation.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    path = prepare_minimum_established_solver_validation(
        repo_root=ROOT,
        config_path=args.config,
        level_a_root=args.level_a_root,
        output_root=args.output_root,
    )
    print(path)


if __name__ == "__main__":
    main()
