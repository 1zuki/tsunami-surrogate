#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.common_time_v2_h2 import (
    THREAD_ENV_KEYS,
    execute_h2_contract,
    freeze_h2_contract,
    h2_status,
    validate_h2_checksums,
)


DEFAULT_CONFIG = Path("configs/eval/common_time_v2_h2.yaml")
DEFAULT_H0 = Path(
    "artifacts/common_time_v2/h0/"
    "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b"
)
DEFAULT_LEVEL_A = Path(
    "artifacts/common_time_v2/level_a/"
    "be1af7dce1f48942e6d20a96bb06b1359655903847c7580954901e2dcfa3332b"
)
DEFAULT_LEVEL_B = Path(
    "artifacts/common_time_v2/level_b_minimum/"
    "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
)
DEFAULT_LEVEL_B_EVALUATION = Path(
    "artifacts/common_time_v2/level_b_minimum_evaluation/"
    "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
)
DEFAULT_H1 = Path(
    "artifacts/common_time_v2/h1/"
    "ef96c24f62a0eb0884f5384436a50802c0d8dd644946552d9c462b225334bc7d"
)
DEFAULT_OUTPUT_BASE = Path("artifacts/common_time_v2/h2")


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
            "H2 multiprocessing requires single-thread numerical backends; "
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
            "[H2] start "
            f"completed={event['completed']}/{event['total']} "
            f"pending={event['pending']} workers={event['workers']} "
            f"max_in_flight={event['max_in_flight']} "
            "(each task runs production and half CFL)",
            flush=True,
        )
    elif event["event"] == "task_complete":
        eta = (
            "unknown"
            if event["eta_s"] is None
            else _duration(event["eta_s"])
        )
        print(
            f"[H2] {event['completed']}/{event['total']} "
            f"{event['run_kind']} {event['qualified_id']} {event['solver']} "
            f"pair={float(event['runtime_s']):.1f}s "
            f"elapsed={_duration(event['elapsed_s'])} ETA={eta} "
            f"health={'pass' if event['passed'] else 'FAIL'}",
            flush=True,
        )
    else:
        print(
            f"[H2] finalized decision={event['decision']} "
            f"duration={_duration(event['duration_s'])}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze, execute, resume, and verify the common-time-v2 H2 "
            "paired-CFL scientific pilot."
        )
    )
    parser.add_argument("action", choices=("freeze", "execute", "status", "checksums"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--h0-root", type=Path, default=DEFAULT_H0)
    parser.add_argument("--level-a-root", type=Path, default=DEFAULT_LEVEL_A)
    parser.add_argument("--level-b-root", type=Path, default=DEFAULT_LEVEL_B)
    parser.add_argument(
        "--level-b-evaluation-root",
        type=Path,
        default=DEFAULT_LEVEL_B_EVALUATION,
    )
    parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--contract-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    if args.action == "freeze":
        if args.contract_root is not None:
            parser.error("--contract-root is not used by freeze")
        path = freeze_h2_contract(
            repo_root=ROOT,
            config_path=args.config,
            h0_root=args.h0_root,
            level_a_root=args.level_a_root,
            level_b_bundle_root=args.level_b_root,
            level_b_evaluation_root=args.level_b_evaluation_root,
            h1_root=args.h1_root,
            output_base=args.output_base,
        )
        print(path)
        return

    if args.contract_root is None:
        parser.error(f"{args.action} requires --contract-root")
    if args.action == "status":
        print(json.dumps(h2_status(args.contract_root), indent=2, sort_keys=True))
        return
    if args.action == "checksums":
        validate_h2_checksums(args.contract_root)
        print(f"checksums valid: {args.contract_root}")
        return

    _configure_threads(args.workers)
    result = execute_h2_contract(
        repo_root=ROOT,
        contract_root=args.contract_root,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        resume=args.resume,
        progress=None if args.quiet_progress else _show_progress,
    )
    print(result)


if __name__ == "__main__":
    main()
