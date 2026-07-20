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
from src.evaluation.h2_swe_cfl_diagnostic import (
    diagnostic_status,
    execute_diagnostic,
    freeze_diagnostic,
    validate_diagnostic_checksums,
)


DEFAULT_CONFIG = Path("configs/eval/h2_swe_cfl_refinement_v1.yaml")
DEFAULT_SOURCE_H2 = Path(
    "artifacts/common_time_v2/h2/"
    "b0a91373ea8dc6ba4304a2b2d319cbeb551d5e211279b8fb799228b811058be9"
)
DEFAULT_OUTPUT_BASE = Path(
    "artifacts/common_time_v2/h2_diagnostics/swe_cfl_refinement_v1"
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


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _show_progress(event: dict[str, object]) -> None:
    if event["event"] == "start":
        print(
            "[H2-SWE-CFL] start "
            f"completed={event['completed']}/{event['total']} "
            f"pending={event['pending']} workers={event['workers']} "
            f"max_in_flight={event['max_in_flight']} "
            "(each task runs production, half, and quarter CFL)",
            flush=True,
        )
    elif event["event"] == "task_complete":
        eta = "unknown" if event["eta_s"] is None else _duration(event["eta_s"])
        print(
            f"[H2-SWE-CFL] {event['completed']}/{event['total']} "
            f"{event['qualified_id']} {event['solver']} "
            f"task={float(event['runtime_s']):.1f}s "
            f"order={_format_optional(event['observed_order'])} "
            f"ratio={_format_optional(event['contraction_ratio'])} "
            f"elapsed={_duration(event['elapsed_s'])} ETA={eta} "
            f"checks={'pass' if event['passed'] else 'FAIL'}",
            flush=True,
        )
    else:
        print(
            "[H2-SWE-CFL] finalized "
            f"healthy={event['healthy']} "
            f"failed_screening_gates={event['failed_screening_gate_count']} "
            f"duration={_duration(event['duration_s'])}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, run, resume, and inspect the non-decisional H2 SWE "
            "production/half/quarter CFL diagnostic."
        )
    )
    parser.add_argument("action", choices=("freeze", "execute", "status", "checksums"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-h2-root", type=Path, default=DEFAULT_SOURCE_H2)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--diagnostic-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    if args.action == "freeze":
        if args.diagnostic_root is not None:
            parser.error("--diagnostic-root is not used by freeze")
        root = freeze_diagnostic(
            repo_root=ROOT,
            config_path=args.config,
            source_h2_root=args.source_h2_root,
            output_base=args.output_base,
        )
        print(root)
        return

    if args.diagnostic_root is None:
        parser.error(f"{args.action} requires --diagnostic-root")
    if args.action == "status":
        print(
            json.dumps(
                diagnostic_status(args.diagnostic_root), indent=2, sort_keys=True
            )
        )
        return
    if args.action == "checksums":
        validate_diagnostic_checksums(args.diagnostic_root)
        print(f"checksums valid: {args.diagnostic_root}")
        return

    _configure_threads(args.workers)
    path = execute_diagnostic(
        repo_root=ROOT,
        diagnostic_root=args.diagnostic_root,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        resume=args.resume,
        progress=None if args.quiet_progress else _show_progress,
    )
    print(path)


if __name__ == "__main__":
    main()
