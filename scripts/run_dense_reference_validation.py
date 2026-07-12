#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.cli_progress import (
    ScenarioProgressLogger,
    resolve_progress_every,
)
from src.evaluation.dense_reference_validation import (
    mapping_arg,
    run_dense_reference_validation,
)
from src.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay natural-step dense solver trajectories for the frozen common-time "
            "validation scenarios and gate on legacy-knot reproduction before any "
            "interpolation evidence is reported."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/eval/dense_reference_validation.yaml",
        help="Dense reference validation YAML config.",
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=("smoke", "dense_validation"),
        help="Explicit suite selector. 'smoke' is implementation-only and does not permit manuscript claims.",
    )
    parser.add_argument(
        "--audit-artifact",
        default=None,
        help="Optional explicit paired-reference audit artifact JSON.",
    )
    parser.add_argument(
        "--scenario-selection",
        default=None,
        help="Optional explicit common-time validation scenario-selection JSON.",
    )
    parser.add_argument(
        "--processed-test-root",
        action="append",
        default=[],
        help="Override processed solver test root as solver=PATH.",
    )
    parser.add_argument(
        "--raw-test-root",
        action="append",
        default=[],
        help="Override raw solver sample root as solver=PATH.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional explicit output root. Suite artifacts are written under OUTPUT_ROOT/<suite>/.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Log deterministic progress every N scenarios. Defaults by suite.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress start/progress messages while keeping final artifact output.",
    )
    args = parser.parse_args(argv)

    config_path = (
        ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    )
    config = load_config(config_path)
    progress_every = resolve_progress_every(args.suite, args.progress_every)
    progress_logger = ScenarioProgressLogger(
        label="dense-reference-validation",
        progress_every=progress_every,
        quiet=args.quiet,
    )
    if not args.quiet:
        print(
            f"[dense-reference-validation] start suite={args.suite} "
            f"progress_every={progress_every} config={config_path}"
        )
    summary = run_dense_reference_validation(
        config,
        suite_name=args.suite,
        config_path=config_path,
        audit_artifact_path=args.audit_artifact,
        scenario_selection_path=args.scenario_selection,
        processed_root_overrides=mapping_arg(args.processed_test_root),
        raw_root_overrides=mapping_arg(args.raw_test_root),
        output_root_override=args.output_root,
        progress_callback=progress_logger,
    )

    print(
        f"[dense-reference-validation] suite={summary['suite']['name']} "
        f"status={summary['status']} "
        f"scenario_count={summary['counts']['scenario_count']} "
        f"eligible_for_interpolation={summary['counts']['eligible_for_interpolation_count']}"
    )
    print(
        "[dense-reference-validation] artifacts="
        f"{summary['artifacts_written']['suite_output_dir']}"
    )


if __name__ == "__main__":
    main()
