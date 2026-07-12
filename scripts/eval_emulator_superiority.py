#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.aligned_comparison import (
    MODE_COMMON_TIME,
    build_processed_input_lookup,
    build_emulator_superiority_metric_row,
    align_positive_time_series,
    evaluate_emulator_superiority_metric_rows,
    iter_paired_raw_reference_samples,
    load_model_input_order,
    prediction_positive_timestamps,
    resolve_suite_contract,
    validate_common_time_solver_comparison_artifact,
    verify_common_raw_identity,
    verify_reconstructed_input_match,
    write_jsonl,
)
from src.evaluation.alignment import align_elevation_series
from src.evaluation.cli_progress import ScenarioProgressLogger, resolve_progress_every
from src.evaluation.normalization_bridge import (
    denormalize_model_target,
    load_standardization_spec,
    normalize_raw_inputs_for_model,
)
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.io import get_git_commit, save_json
from src.utils.seed import seed_everything


def _ensure_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _ensure_mapping(payload, label=str(path))


def _checkpoint_train_path(ckpt: Mapping[str, Any]) -> str | None:
    raw_cfg = ckpt.get("config", {})
    if not isinstance(raw_cfg, Mapping):
        return None
    data_cfg = raw_cfg.get("data", raw_cfg.get("dataset", {}))
    if not isinstance(data_cfg, Mapping):
        return None
    train_path = data_cfg.get("train_path")
    if train_path:
        return str(train_path)
    fallback = data_cfg.get("path")
    if fallback:
        return str(fallback)
    return None


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _direction_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    direction = cfg.get("direction")
    if direction is None:
        raise KeyError("config requires a direction mapping")
    return _ensure_mapping(direction, label="direction")


def _alignment_contract_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    section = cfg.get("alignment_contract")
    if section is None:
        raise KeyError("config requires an alignment_contract mapping")
    return _ensure_mapping(section, label="alignment_contract")


def _resolve_output_path(cfg: Mapping[str, Any], direction_name: str) -> Path:
    configured = str(cfg.get("output_path", "")).strip()
    if configured:
        return Path(configured)
    return (
        ROOT
        / "results"
        / "common_time_validation"
        / "emulator_superiority"
        / f"{direction_name}.json"
    )


def _resolve_direction_runtime_paths(
    direction_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    configured_paths = {
        "checkpoint": str(direction_cfg.get("checkpoint", "")).strip(),
        "model_raw_root": str(direction_cfg.get("model_raw_root", "")).strip(),
        "benchmark_raw_root": str(direction_cfg.get("benchmark_raw_root", "")).strip(),
        "model_processed_test_path": str(
            direction_cfg.get("model_processed_test_path", "")
        ).strip(),
        "model_normalization_stats_path": str(
            direction_cfg.get("model_normalization_stats_path", "")
        ).strip(),
    }
    override_values = {
        "checkpoint": args.checkpoint,
        "model_raw_root": args.model_raw_root,
        "benchmark_raw_root": args.benchmark_raw_root,
        "model_processed_test_path": args.processed_test_path,
        "model_normalization_stats_path": args.normalization_stats_path,
    }
    cli_overrides = {
        key: str(value).strip()
        for key, value in override_values.items()
        if value is not None and str(value).strip()
    }
    effective_paths = {
        key: cli_overrides.get(key, configured_paths[key]) for key in configured_paths
    }
    return {
        "configured_paths": configured_paths,
        "effective_paths": effective_paths,
        "cli_overrides": cli_overrides,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate benchmark-specific emulator superiority with raw model-A inputs, "
            "model-A normalization, and independent common-time alignment."
        )
    )
    parser.add_argument("--config", required=True, help="YAML config for the direction")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--model-raw-root", default=None)
    parser.add_argument("--benchmark-raw-root", default=None)
    parser.add_argument("--processed-test-path", default=None)
    parser.add_argument("--normalization-stats-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Log deterministic progress every N evaluated scenarios. Defaults by suite.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress start/progress messages while keeping final artifact output.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    direction_cfg = _direction_cfg(cfg)
    contract_cfg = _alignment_contract_cfg(cfg)
    eval_cfg = _ensure_mapping(cfg.get("evaluation", {}), label="evaluation")
    runtime_paths = _resolve_direction_runtime_paths(direction_cfg, args)

    model_config_path = str(direction_cfg.get("model_config", "")).strip()
    checkpoint_path = runtime_paths["effective_paths"]["checkpoint"]
    direction_name = str(direction_cfg.get("name", "")).strip()
    model_solver = str(direction_cfg.get("model_solver", "")).strip()
    benchmark_solver = str(direction_cfg.get("benchmark_solver", "")).strip()
    model_raw_root = runtime_paths["effective_paths"]["model_raw_root"]
    benchmark_raw_root = runtime_paths["effective_paths"]["benchmark_raw_root"]
    model_processed_test_path = runtime_paths["effective_paths"][
        "model_processed_test_path"
    ]
    model_normalization_stats_path = runtime_paths["effective_paths"][
        "model_normalization_stats_path"
    ]
    if not model_config_path or not checkpoint_path:
        raise KeyError("direction.model_config and direction.checkpoint are required")
    if not direction_name or not model_solver or not benchmark_solver:
        raise KeyError(
            "direction.name, direction.model_solver, and direction.benchmark_solver are required"
        )
    if not model_raw_root or not benchmark_raw_root:
        raise KeyError(
            "direction.model_raw_root and direction.benchmark_raw_root are required"
        )
    if not model_processed_test_path or not model_normalization_stats_path:
        raise KeyError(
            "direction.model_processed_test_path and direction.model_normalization_stats_path are required"
        )

    alignment_config_path = str(contract_cfg.get("config_path", "")).strip()
    if not alignment_config_path:
        raise KeyError("alignment_contract.config_path is required")
    alignment_config = load_config(alignment_config_path)
    alignment_cfg = _ensure_mapping(
        alignment_config.get("alignment"), label="alignment"
    )

    contract = resolve_suite_contract(
        alignment_cfg=alignment_cfg,
        audit_artifact_path=str(contract_cfg.get("audit_artifact_path", "")).strip(),
        scenario_selection_path=str(
            contract_cfg.get("scenario_selection_path", "")
        ).strip()
        or None,
        suite_name=str(contract_cfg.get("suite", "")).strip(),
        dense_validation_decision_path=str(
            contract_cfg.get("dense_validation_decision_path", "")
        ).strip()
        or None,
        require_full_suite_dense_decision=bool(
            contract_cfg.get("require_full_suite_dense_decision", True)
        ),
        dense_fallback_policy=str(
            contract_cfg.get("dense_fallback_policy", "unsupported")
        ),
    )
    progress_every = resolve_progress_every(contract.suite_name, args.progress_every)
    progress_logger = ScenarioProgressLogger(
        label="emulator-superiority",
        progress_every=progress_every,
        quiet=args.quiet,
    )

    deprecated_solver_compare_path = str(
        contract_cfg.get(
            "reference_solver_comparison_path",
            contract_cfg.get("solver_compare_path", ""),
        )
    ).strip()
    if deprecated_solver_compare_path:
        validate_common_time_solver_comparison_artifact(
            _load_json_mapping(deprecated_solver_compare_path),
            contract=contract,
        )

    output_path = _resolve_output_path(cfg, direction_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = int(eval_cfg.get("batch_size", 8))
    reconstruction_atol = float(eval_cfg.get("reconstruction_atol", 1.0e-6))
    bootstrap_seed = int(eval_cfg.get("bootstrap_seed", 20260711))
    num_resamples = int(eval_cfg.get("num_resamples", 10000))
    confidence_level = float(eval_cfg.get("confidence_level", 0.95))

    model_cfg = load_config(model_config_path)
    if args.device is not None:
        model_cfg["device"] = args.device
    seed_everything(int(model_cfg.get("seed", cfg.get("seed", 42))))
    device = resolve_device(model_cfg.get("device", "auto"))
    if not args.quiet:
        print(
            f"[emulator-superiority] start direction={direction_name} "
            f"suite={contract.suite_name} device={device} batch_size={batch_size} "
            f"progress_every={progress_every}"
        )

    model = build_model(model_cfg).to(device)
    checkpoint_payload = load_checkpoint(checkpoint_path, model, map_location=device)
    checkpoint_train_path = _checkpoint_train_path(checkpoint_payload)
    checkpoint_train_order_path = None
    if checkpoint_train_path:
        candidate = Path(checkpoint_train_path)
        if candidate.exists():
            checkpoint_train_order_path = candidate

    model_input_order = load_model_input_order(
        processed_test_path=model_processed_test_path,
        checkpoint_train_path=checkpoint_train_order_path,
    )
    processed_lookup = build_processed_input_lookup(model_processed_test_path)
    if tuple(processed_lookup.input_order) != tuple(model_input_order):
        raise ValueError(
            "Processed lookup input_order does not match the model input_order control: "
            f"{processed_lookup.input_order!r} != {model_input_order!r}"
        )
    model_stats = load_standardization_spec(model_normalization_stats_path)

    scenario_metric_rows: list[dict[str, Any]] = []
    reconstruction_checked = 0
    reconstruction_max_abs_diff = 0.0
    identity_verified = 0
    batch_inputs: list[np.ndarray] = []
    batch_rows: list[dict[str, Any]] = []

    def _consume_pending_batch() -> None:
        if not batch_inputs:
            return
        x_batch = torch.from_numpy(np.stack(batch_inputs, axis=0)).to(device)
        pred_batch = _model_output(model, x_batch).detach().cpu().numpy()
        for batch_index, batch_row in enumerate(batch_rows):
            pred_model = np.asarray(pred_batch[batch_index], dtype=np.float64)
            pred_physical = np.asarray(
                denormalize_model_target(pred_model, model_stats),
                dtype=np.float64,
            )
            pred_ts = prediction_positive_timestamps(
                batch_row["left"]["timestamps"],
                expected_output_channels=pred_physical.shape[0],
            )
            pred_aligned = align_positive_time_series(
                pred_physical,
                pred_ts,
                common_time_grid=contract.common_time_grid,
                endpoint_tolerance=contract.endpoint_tolerance,
            )
            ref_a_aligned = align_elevation_series(
                batch_row["left"]["trajectory_eta"],
                batch_row["left"]["timestamps"],
                mode=MODE_COMMON_TIME,
                common_time_grid=contract.common_time_grid,
                endpoint_tolerance=contract.endpoint_tolerance,
            )
            ref_b_aligned = align_elevation_series(
                batch_row["right"]["trajectory_eta"],
                batch_row["right"]["timestamps"],
                mode=MODE_COMMON_TIME,
                common_time_grid=contract.common_time_grid,
                endpoint_tolerance=contract.endpoint_tolerance,
            )
            scenario_metric_rows.append(
                build_emulator_superiority_metric_row(
                    scenario_id=str(batch_row["scenario_id"]),
                    bathymetry_type=str(batch_row["bathymetry_type"]),
                    source_type=str(batch_row["source_type"]),
                    source_strength=float(batch_row["source_strength"]),
                    pred_aligned=pred_aligned,
                    ref_a_aligned=ref_a_aligned,
                    ref_b_aligned=ref_b_aligned,
                )
            )
            progress_logger(
                len(scenario_metric_rows),
                len(contract.ordered_scenario_ids),
                str(batch_row["scenario_id"]),
            )
        batch_inputs.clear()
        batch_rows.clear()

    model.eval()
    with torch.no_grad():
        for paired in iter_paired_raw_reference_samples(
            contract=contract,
            left_root=model_raw_root,
            right_root=benchmark_raw_root,
        ):
            verify_common_raw_identity(
                scenario_id=str(paired["scenario_id"]),
                left_sample=paired["left"],
                right_sample=paired["right"],
            )
            identity_verified += 1
            raw_inputs = {
                "bathymetry": paired["left"]["bathymetry"],
                "source_field": paired["left"]["source_field"],
                "source": paired["left"]["source_field"],
                "initial_depth": paired["left"]["initial_depth"],
                "eta0": paired["left"]["eta0"],
                "free_surface0": paired["left"]["free_surface0"],
                "initial_surface": paired["left"]["free_surface0"],
            }
            reconstructed = normalize_raw_inputs_for_model(
                raw_inputs,
                input_order=model_input_order,
                model_stats=model_stats,
            )
            reconstruction = verify_reconstructed_input_match(
                scenario_id=str(paired["scenario_id"]),
                reconstructed_input=reconstructed,
                lookup=processed_lookup,
                atol=reconstruction_atol,
            )
            reconstruction_checked += 1
            reconstruction_max_abs_diff = max(
                reconstruction_max_abs_diff,
                float(reconstruction["max_abs_diff"]),
            )

            batch_inputs.append(reconstructed)
            batch_rows.append(paired)
            if len(batch_inputs) < batch_size:
                continue
            _consume_pending_batch()

        _consume_pending_batch()

    result = evaluate_emulator_superiority_metric_rows(
        contract=contract,
        direction_name=direction_name,
        model_solver_name=model_solver,
        benchmark_solver_name=benchmark_solver,
        scenario_metric_rows=scenario_metric_rows,
        bootstrap_seed=bootstrap_seed,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        git_commit=get_git_commit(),
        script_path=str(Path(__file__).resolve()),
    )
    scenario_metrics = result.pop("scenario_metrics")
    scenario_metrics_path = output_path.with_name(
        f"{output_path.stem}_scenario_metrics.jsonl"
    )
    write_jsonl(scenario_metrics, scenario_metrics_path)
    result["model"] = {
        "model_config": model_config_path,
        "checkpoint": checkpoint_path,
        "model_input_order": list(model_input_order),
        "model_processed_test_path": str(model_processed_test_path),
        "model_normalization_stats_path": str(model_normalization_stats_path),
        "model_raw_root": str(model_raw_root),
        "benchmark_raw_root": str(benchmark_raw_root),
        "checkpoint_train_path": checkpoint_train_path,
        "prediction_channel_timestamp_assignment": "reference_a_positive_timestamps",
        "configured_paths": runtime_paths["configured_paths"],
        "effective_paths": runtime_paths["effective_paths"],
    }
    result["provenance"]["cli_overrides"] = runtime_paths["cli_overrides"]
    result["reconstruction_control"] = {
        "status": "pass",
        "checked_scenario_count": int(reconstruction_checked),
        "checked_channel_count": int(reconstruction_checked * len(model_input_order)),
        "max_abs_diff": float(reconstruction_max_abs_diff),
        "atol": float(reconstruction_atol),
    }
    result["raw_identity_control"] = {
        "status": "pass",
        "verified_scenario_count": int(identity_verified),
        "shared_fields": ["bathymetry", "source_field", "initial_depth"],
    }
    result["artifacts_written"] = {
        "summary_json": str(output_path),
        "scenario_metrics_jsonl": str(scenario_metrics_path),
    }
    save_json(result, output_path)
    print(
        f"[emulator-superiority] direction={direction_name} "
        f"suite={contract.suite_name} "
        f"classification={result['benchmark_specific_superiority']['classification']} "
        f"rho={result['metrics']['rho']:.6g}"
    )
    print(f"[emulator-superiority] artifacts={output_path}")


if __name__ == "__main__":
    main()
