#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

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
        default=Path(
            "configs/eval/minimum_established_solver_validation_v4.yaml"
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.workers > 1:
        invalid = {
            key: value
            for key in THREAD_ENV_KEYS
            if (value := os.environ.get(key)) not in (None, "1")
        }
        if invalid:
            details = ", ".join(
                f"{key}={value}" for key, value in sorted(invalid.items())
            )
            parser.error(
                "multiprocessing requires single-thread numerical backends; "
                f"found {details}"
            )
        for key in THREAD_ENV_KEYS:
            os.environ.setdefault(key, "1")
    path = prepare_minimum_established_solver_validation(
        repo_root=ROOT,
        config_path=args.config,
        level_a_root=args.level_a_root,
        output_root=args.output_root,
        workers=args.workers,
        progress=(
            None
            if args.quiet_progress
            else lambda message: print(message, flush=True)
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
