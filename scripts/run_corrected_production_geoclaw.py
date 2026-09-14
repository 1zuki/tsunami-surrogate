#!/usr/bin/env python
"""Freeze, run, or evaluate the current-contract GeoClaw SWE comparator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.corrected_production_geoclaw import (  # noqa: E402
    evaluate_corrected_production_geoclaw,
    prepare_corrected_production_geoclaw_bundle,
    run_corrected_production_geoclaw,
)
from src.evaluation.geoclaw_adapter import GeoClawEnvironment  # noqa: E402


def _progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="freeze R6 inputs and references")
    prepare.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/corrected_production_geoclaw_validation.yaml"),
    )
    prepare.add_argument("--output-root", type=Path)

    run = subcommands.add_parser("run", help="run a frozen bundle with GeoClaw")
    run.add_argument("--bundle-root", type=Path, required=True)
    run.add_argument("--external-root", type=Path, required=True)
    run.add_argument("--claw-root", type=Path, required=True)
    run.add_argument("--petsc-dir", type=Path, required=True)
    run.add_argument("--petsc-arch", default="arch-linux-c-opt")
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--resume", action="store_true")

    evaluate = subcommands.add_parser("evaluate", help="summarize a completed GeoClaw run")
    evaluate.add_argument("--bundle-root", type=Path, required=True)
    evaluate.add_argument("--external-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare_corrected_production_geoclaw_bundle(
                repo_root=ROOT,
                config_path=args.config,
                output_root=args.output_root,
            )
        )
    elif args.command == "run":
        if args.workers <= 0:
            parser.error("--workers must be positive")
        result = run_corrected_production_geoclaw(
            bundle_root=args.bundle_root,
            external_root=args.external_root,
            environment=GeoClawEnvironment(
                claw_root=args.claw_root,
                petsc_dir=args.petsc_dir,
                petsc_arch=args.petsc_arch,
                python_executable=args.python,
            ),
            workers=args.workers,
            resume=args.resume,
            progress=_progress,
        )
        print(
            f"bundle={result['bundle_hash']} adapter={result['adapter_hash']} "
            f"executed={result['executed']} skipped={result['skipped']}"
        )
    else:
        print(
            evaluate_corrected_production_geoclaw(
                bundle_root=args.bundle_root,
                external_root=args.external_root,
                output_root=args.output_root,
                progress=_progress,
            )
        )


if __name__ == "__main__":
    main()
