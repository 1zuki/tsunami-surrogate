#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.aligned_comparison import MODE_SAVED_INDEX_LEGACY

from scripts.compare_solvers_aligned import main as aligned_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated wrapper for the old saved-index solver comparison path. "
            "Common-time comparisons must use scripts/compare_solvers_aligned.py."
        )
    )
    parser.add_argument("--mode", required=True, choices=(MODE_SAVED_INDEX_LEGACY,))
    parser.add_argument(
        "--legacy-initial-frame", required=True, choices=("include", "exclude")
    )
    args, remaining = parser.parse_known_args(argv)

    forwarded = [
        "--mode",
        args.mode,
        "--legacy-initial-frame",
        args.legacy_initial_frame,
    ]
    forwarded.extend(remaining)
    aligned_main(forwarded)


if __name__ == "__main__":
    main()
