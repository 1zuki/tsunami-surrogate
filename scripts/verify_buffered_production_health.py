#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.buffered_production_health import (
    THREAD_ENV_KEYS,
    audit_health_contract,
    execute_health_contract,
    health_status,
    validate_checksums,
)


DEFAULT_INVENTORY = Path(
    "artifacts/common_time_v2/h0/"
    "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b/"
    "h0_input_inventory.jsonl"
)
DEFAULT_OUTPUT = Path("artifacts/common_time_v2/buffered_production_health_v1")


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
        raise SystemExit(f"Multiprocessing requires single-thread backends; found {details}")
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")


def _show_progress(event: dict[str, object]) -> None:
    if event["event"] == "start":
        print(
            "[96-health] start "
            f"completed={event['completed']}/{event['total']} "
            f"pending={event['pending']} workers={event['workers']}",
            flush=True,
        )
    elif event["event"] == "task_complete":
        print(
            f"[96-health] {event['completed']}/{event['total']} "
            f"{event['qualified_id']} {event['solver']} "
            f"solver={float(event['runtime_s']):.1f}s "
            f"elapsed={float(event['elapsed_s']):.1f}s "
            f"health={'pass' if event['health_passed'] else 'FAIL'}",
            flush=True,
        )
    else:
        print(
            f"[96-health] finalized passed={event['passed']} "
            f"duration={float(event['duration_s']):.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and run the resumable 96x96 production health check."
    )
    parser.add_argument("action", choices=("audit", "execute", "status", "checksums"))
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    if args.action == "audit":
        path = audit_health_contract(
            repo_root=ROOT,
            inventory_path=args.inventory,
            output_root=args.output_root,
        )
        print(path)
        return
    if args.action == "status":
        print(json.dumps(health_status(args.output_root), indent=2, sort_keys=True))
        return
    if args.action == "checksums":
        validate_checksums(args.output_root)
        print(f"checksums valid: {args.output_root}")
        return
    _configure_threads(args.workers)
    path = execute_health_contract(
        repo_root=ROOT,
        output_root=args.output_root,
        workers=args.workers,
        resume=args.resume,
        progress=None if args.quiet_progress else _show_progress,
    )
    print(path)


if __name__ == "__main__":
    main()
