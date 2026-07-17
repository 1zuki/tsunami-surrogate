#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.finite_horizon_boundary_study import (
    run_numerical_study,
    run_static_audit,
    verify_artifact_checksums,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the non-decisional finite-horizon boundary study"
    )
    parser.add_argument("mode", choices=("audit", "execute", "checksums"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/finite_horizon_boundary_study.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/common_time_v2/boundary_contract_study/finite_horizon_v5"
        ),
    )
    parser.add_argument("--resume", action="store_true", help="resume execute only")
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="suppress execute progress on stderr",
    )
    args = parser.parse_args()
    if args.resume and args.mode != "execute":
        parser.error("--resume is valid only with execute")
    repo_root = Path.cwd().resolve()
    if args.mode == "audit":
        result = run_static_audit(
            repo_root=repo_root,
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
        )
        print(json.dumps(result["freeze"], indent=2, sort_keys=True))
    elif args.mode == "execute":
        def show_progress(event: dict[str, object]) -> None:
            elapsed = float(event["elapsed_s"])
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            kind = str(event["event"])
            completed = int(event["completed"])
            total = int(event["total"])
            if kind == "start":
                print(
                    "[finite-horizon] start "
                    f"completed={completed}/{total} resumed={event['resumed']} "
                    f"pending={event['pending']} workers={event['workers']} "
                    f"max_in_flight={event['max_in_flight']}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "task_completed":
                percent = 100.0 * completed / max(total, 1)
                print(
                    "[finite-horizon] progress "
                    f"completed={completed}/{total} ({percent:5.1f}%) "
                    f"elapsed={clock} solver={event['solver']} "
                    f"case={event['qualified_id']} task={event['task_id']}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "heartbeat":
                print(
                    "[finite-horizon] progress "
                    f"completed={completed}/{total} active={event['active']} "
                    f"pending={event['pending']} elapsed={clock}",
                    file=sys.stderr,
                    flush=True,
                )
            elif kind == "complete":
                print(
                    "[finite-horizon] tasks complete "
                    f"completed={completed}/{total} elapsed={clock}; finalizing",
                    file=sys.stderr,
                    flush=True,
                )

        result = run_numerical_study(
            repo_root=repo_root,
            config_path=args.config.resolve(),
            output_dir=args.output.resolve(),
            resume=args.resume,
            progress_callback=None if args.quiet_progress else show_progress,
        )
        print(json.dumps(result["report"], indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                verify_artifact_checksums(args.output.resolve()),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
