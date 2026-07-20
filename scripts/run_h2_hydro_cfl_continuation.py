#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.common_time_v2_h1 import THREAD_ENV_KEYS
from src.evaluation.h2_hydro_cfl_continuation import (
    continuation_status,
    execute_continuation,
    freeze_continuation,
    validate_continuation_checksums,
)


DEFAULT_CONFIG = Path("configs/eval/h2_hydro_cfl_continuation_v1.yaml")
DEFAULT_SOURCE = Path(
    "artifacts/common_time_v2/h2_diagnostics/swe_cfl_refinement_v1/"
    "1a29fea6ee28afb528e844abb55a6988249c09ed507e655e5aad1f657e2da138"
)
DEFAULT_OUTPUT_BASE = Path(
    "artifacts/common_time_v2/h2_diagnostics/hydro_cfl_continuation_v1"
)


def _configure_threads(workers: int) -> None:
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
            "Multiprocessing requires single-thread numerical backends; "
            f"found {details}"
        )
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")


def _duration(seconds: object) -> str:
    value = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(value)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _show_progress(event: dict[str, object]) -> None:
    if event["event"] == "start":
        print(
            "[H2-HYDRO-CFL] start "
            f"completed={event['completed']}/{event['total']} "
            f"pending={event['pending']} workers={event['workers']} "
            "(each task runs CFL 0.1125 and 0.05625)",
            flush=True,
        )
    elif event["event"] == "task_complete":
        eta = "unknown" if event["eta_s"] is None else _duration(event["eta_s"])
        print(
            f"[H2-HYDRO-CFL] {event['completed']}/{event['total']} "
            f"{event['qualified_id']} task={float(event['runtime_s']):.1f}s "
            f"order={float(event['observed_order']):.3f} "
            f"ratio={float(event['contraction_ratio']):.3f} "
            f"per_time_max={float(event['per_time_max']):.6f} "
            f"elapsed={_duration(event['elapsed_s'])} ETA={eta} "
            f"checks={'pass' if event['passed'] else 'FAIL'}",
            flush=True,
        )
    else:
        print(
            "[H2-HYDRO-CFL] finalized "
            f"healthy={event['healthy']} "
            f"screening_passed={event['screening_passed']} "
            f"duration={_duration(event['duration_s'])}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, execute, resume, and verify the three-case Hydro "
            "quarter-to-eighth CFL continuation."
        )
    )
    parser.add_argument("action", choices=("freeze", "execute", "status", "checksums"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-diagnostic-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--continuation-root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    if args.action == "freeze":
        if args.continuation_root is not None:
            parser.error("--continuation-root is not used by freeze")
        root = freeze_continuation(
            repo_root=ROOT,
            config_path=args.config,
            source_diagnostic_root=args.source_diagnostic_root,
            output_base=args.output_base,
        )
        print(root)
        return
    if args.continuation_root is None:
        parser.error(f"{args.action} requires --continuation-root")
    if args.action == "status":
        print(
            json.dumps(
                continuation_status(args.continuation_root),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.action == "checksums":
        validate_continuation_checksums(args.continuation_root)
        print(f"checksums valid: {args.continuation_root}")
        return
    _configure_threads(args.workers)
    result = execute_continuation(
        repo_root=ROOT,
        continuation_root=args.continuation_root,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        resume=args.resume,
        progress=None if args.quiet_progress else _show_progress,
    )
    print(result)


if __name__ == "__main__":
    main()
