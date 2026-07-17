#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.geoclaw_adapter import GeoClawEnvironment, run_geoclaw_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen established-solver cases with GeoClaw SWE and SGN"
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--claw-root", type=Path, required=True)
    parser.add_argument("--petsc-dir", type=Path, required=True)
    parser.add_argument("--petsc-arch", default="arch-linux-c-opt")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--comparator",
        action="append",
        choices=("geoclaw_swe", "geoclaw_sgn"),
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    summary = run_geoclaw_bundle(
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
        case_ids=args.case_id,
        comparator_ids=args.comparator,
        max_tasks=args.max_tasks,
        progress=(
            None if args.quiet_progress else lambda message: print(message, flush=True)
        ),
    )
    print(
        f"bundle={summary['bundle_hash']} adapter={summary['adapter_hash']} "
        f"executed={summary['executed']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
