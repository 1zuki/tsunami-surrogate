#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.aligned_comparison import (
    MODE_COMMON_TIME,
    compare_solver_scenarios,
    iter_paired_raw_reference_samples,
    resolve_suite_contract,
    require_explicit_mode,
    write_jsonl,
)
from src.evaluation.cli_progress import ScenarioProgressLogger, resolve_progress_every
from src.utils.config import load_config
from src.utils.io import get_git_commit, save_json


def _default_output_path(
    *,
    solver_a: str,
    solver_b: str,
    suite: str,
    mode: str,
) -> Path:
    stem = f"{solver_a}_vs_{solver_b}_{suite}"
    if mode != MODE_COMMON_TIME:
        stem = f"{stem}_{mode.replace('-', '_')}"
    return (
        ROOT
        / "results"
        / "common_time_validation"
        / "aligned_solver_comparison"
        / f"{stem}.json"
    )


def _iter_with_progress(
    paired_scenarios,
    *,
    logger: ScenarioProgressLogger,
    total: int,
):
    completed = 0
    for paired in paired_scenarios:
        yield paired
        completed += 1
        logger(completed, total, str(paired.get("scenario_id", "")) or None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired solver references with explicit common-time or "
            "saved-index-legacy alignment semantics."
        )
    )
    parser.add_argument("--config", default="configs/eval/common_time_alignment.yaml")
    parser.add_argument(
        "--mode", required=True, choices=(MODE_COMMON_TIME, "saved-index-legacy")
    )
    parser.add_argument(
        "--suite", required=True, choices=("smoke", "dense_validation", "full")
    )
    parser.add_argument("--solver-a", required=True)
    parser.add_argument("--solver-b", required=True)
    parser.add_argument("--solver-a-dir", required=True)
    parser.add_argument("--solver-b-dir", required=True)
    parser.add_argument("--audit-artifact", default=None)
    parser.add_argument("--scenario-selection", default=None)
    parser.add_argument("--dense-validation-decision", default=None)
    parser.add_argument(
        "--legacy-initial-frame", choices=("include", "exclude"), default=None
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--num-resamples", type=int, default=None)
    parser.add_argument("--confidence-level", type=float, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Log deterministic progress every N paired scenarios. Defaults by suite.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress start/progress messages while keeping final artifact output.",
    )
    args = parser.parse_args(argv)

    mode = require_explicit_mode(args.mode)
    config = load_config(args.config)
    alignment_cfg = dict(config.get("alignment", {}))
    if not alignment_cfg:
        raise KeyError(f"{args.config} is missing an alignment section")

    audit_artifact_path = args.audit_artifact or str(
        dict(config.get("audit", {})).get(
            "results_dir",
            "results/common_time_validation/audit",
        )
    )
    if Path(audit_artifact_path).is_dir():
        audit_artifact_path = str(
            Path(audit_artifact_path) / "paired_reference_audit.json"
        )
    selection_path = args.scenario_selection or str(
        dict(config.get("selection", {})).get(
            "output_path",
            "configs/eval/common_time_validation_scenarios.json",
        )
    )
    dense_validation_decision = args.dense_validation_decision or (
        "results/common_time_validation/dense_reference_validation/dense_validation/decision.json"
    )

    bootstrap_cfg = dict(dict(alignment_cfg).get("aggregation", {})).get(
        "bootstrap", {}
    )
    bootstrap_seed = int(
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else bootstrap_cfg.get("seed", 20260711)
    )
    num_resamples = int(
        args.num_resamples
        if args.num_resamples is not None
        else bootstrap_cfg.get("num_resamples", 10000)
    )
    confidence_level = float(
        args.confidence_level
        if args.confidence_level is not None
        else bootstrap_cfg.get("confidence_level", 0.95)
    )

    contract = resolve_suite_contract(
        alignment_cfg=alignment_cfg,
        audit_artifact_path=audit_artifact_path,
        scenario_selection_path=selection_path,
        suite_name=args.suite,
        dense_validation_decision_path=dense_validation_decision,
        require_full_suite_dense_decision=(mode == MODE_COMMON_TIME),
        dense_fallback_policy="unsupported",
    )
    progress_every = resolve_progress_every(args.suite, args.progress_every)
    progress_logger = ScenarioProgressLogger(
        label="aligned-solver-comparison",
        progress_every=progress_every,
        quiet=args.quiet,
    )
    if not args.quiet:
        print(
            f"[aligned-solver-comparison] start mode={mode} suite={args.suite} "
            f"solver_a={args.solver_a} solver_b={args.solver_b} "
            f"progress_every={progress_every}"
        )
    summary = compare_solver_scenarios(
        contract=contract,
        solver_a_name=args.solver_a,
        solver_b_name=args.solver_b,
        paired_scenarios=_iter_with_progress(
            iter_paired_raw_reference_samples(
                contract=contract,
                left_root=args.solver_a_dir,
                right_root=args.solver_b_dir,
            ),
            logger=progress_logger,
            total=len(contract.ordered_scenario_ids),
        ),
        mode=mode,
        bootstrap_seed=bootstrap_seed,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        initial_frame_policy=args.legacy_initial_frame,
        git_commit=get_git_commit(),
        script_path=str(Path(__file__).resolve()),
    )

    output_path = (
        Path(args.output_path)
        if args.output_path
        else _default_output_path(
            solver_a=args.solver_a,
            solver_b=args.solver_b,
            suite=args.suite,
            mode=mode,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_metrics = summary.pop("scenario_metrics")
    scenario_metrics_path = output_path.with_name(
        f"{output_path.stem}_scenario_metrics.jsonl"
    )
    write_jsonl(scenario_metrics, scenario_metrics_path)
    summary["artifacts_written"] = {
        "summary_json": str(output_path),
        "scenario_metrics_jsonl": str(scenario_metrics_path),
    }
    save_json(summary, output_path)
    print(
        f"[aligned-solver-comparison] mode={mode} suite={args.suite} "
        f"solver_a={args.solver_a} solver_b={args.solver_b}"
    )
    print(f"[aligned-solver-comparison] artifacts={output_path}")


if __name__ == "__main__":
    main()
