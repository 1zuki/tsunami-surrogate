#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _configure_worker_threads(workers: int) -> None:
    if workers <= 1:
        return
    invalid = {
        key: value
        for key in THREAD_ENV_KEYS
        if (value := os.environ.get(key)) not in (None, "1")
    }
    if invalid:
        details = ", ".join(f"{key}={value}" for key, value in sorted(invalid.items()))
        raise SystemExit(
            "Level A multiprocessing requires single-thread numerical backends; "
            f"found {details}"
        )
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute frozen common-time-v2 Level A contract"
    )
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_in_flight is not None and args.max_in_flight <= 0:
        parser.error("--max-in-flight must be positive")
    _configure_worker_threads(args.workers)

    sys.path.insert(0, str(ROOT))
    from src.evaluation.common_time_v2_level_a import execute_level_a

    path = execute_level_a(
        repo_root=ROOT,
        contract_root=args.contract_root,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        resume=args.resume,
    )
    print(path)


if __name__ == "__main__":
    main()
