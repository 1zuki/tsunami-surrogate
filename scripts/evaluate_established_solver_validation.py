#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.established_solver_validation import (
    evaluate_minimum_established_solver_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical GeoClaw results against a frozen minimum bundle"
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    path = evaluate_minimum_established_solver_validation(
        bundle_root=args.bundle_root,
        external_root=args.external_root,
        output_root=args.output_root,
        progress=(
            None
            if args.quiet_progress
            else lambda message: print(message, flush=True)
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
