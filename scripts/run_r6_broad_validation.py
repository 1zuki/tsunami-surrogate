#!/usr/bin/env python
"""Freeze and execute the R6-only broad numerical-validation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.r6_broad_validation import (  # noqa: E402
    execute_r6_broad_validation,
    preflight_r6_broad_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/r6_broad_validation.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--production-validation-summary",
        type=Path,
        help="Reuse the just-created current-production H0 audit from the suite run.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        result = preflight_r6_broad_validation(
            repo_root=ROOT,
            config_path=args.config,
            output_root=args.output_root,
        )
        print(result)
        return

    result = execute_r6_broad_validation(
        repo_root=ROOT,
        config_path=args.config,
        output_root=args.output_root,
        resume=bool(args.resume),
        production_validation_summary=args.production_validation_summary,
    )
    print(result)


if __name__ == "__main__":
    main()
