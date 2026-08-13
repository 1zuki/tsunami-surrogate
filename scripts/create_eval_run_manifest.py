#!/usr/bin/env python
"""Create the exact required-cell manifest for one isolated evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_suite_preflight import load_suite_contract


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    os.replace(staging, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_preflight(
    cells: list[dict[str, Any]],
    preflight: Mapping[str, Any],
) -> None:
    config_index = {
        str(row["config"]): row
        for row in preflight.get("evaluation_configs", [])
        if isinstance(row, Mapping) and row.get("config")
    }
    checkpoint_rows = [
        row for row in preflight.get("checkpoints", []) if isinstance(row, Mapping)
    ]
    ensemble = preflight.get("ensemble")
    if isinstance(ensemble, Mapping):
        checkpoint_rows.extend(
            row
            for row in ensemble.get("members", [])
            if isinstance(row, Mapping) and row.get("status") == "complete"
        )
    checkpoint_index = {
        str(row["checkpoint"]): row for row in checkpoint_rows if row.get("checkpoint")
    }
    known_dataset_paths = {
        str(path)
        for row in preflight.get("evaluation_configs", [])
        if isinstance(row, Mapping)
        for path in row.get("dataset_paths", [])
    }
    known_dataset_paths.update(
        str(row["processed_root"])
        for row in preflight.get("datasets", [])
        if isinstance(row, Mapping) and row.get("processed_root")
    )
    paper = preflight.get("paper_evidence")
    if isinstance(paper, Mapping):
        for group in (
            "direct_models",
            "window_models",
            "reference_models",
            "wave_models",
        ):
            known_dataset_paths.update(
                str(row["dataset"])
                for row in paper.get(group, [])
                if isinstance(row, Mapping) and row.get("dataset")
            )
        proxy = paper.get("proxy_resolution")
        if isinstance(proxy, Mapping) and proxy.get("dataset"):
            known_dataset_paths.add(str(proxy["dataset"]))
        native = paper.get("native_transfer")
        if isinstance(native, Mapping):
            known_dataset_paths.update(str(path) for path in native.get("datasets", []))
        ensemble_summary = paper.get("ensemble")
        if isinstance(ensemble_summary, Mapping):
            known_dataset_paths.update(
                str(ensemble_summary[key])
                for key in ("val_dataset", "test_dataset")
                if ensemble_summary.get(key)
            )

    for cell in cells:
        config = cell.get("config")
        if config is not None:
            summary = config_index.get(str(config))
            if summary is None:
                raise ValueError(
                    f"Preflight has no config binding for cell {cell['id']}"
                )
            cell["config_sha256"] = str(summary["config_sha256"])
            if "dataset_paths" not in cell:
                cell["expected_dataset_paths"] = list(summary.get("dataset_paths", []))
        configs = cell.get("configs")
        if isinstance(configs, list):
            hashes: list[str] = []
            config_dataset_paths: set[str] = set()
            for config_path in configs:
                summary = config_index.get(str(config_path))
                if summary is None:
                    raise ValueError(
                        "Preflight has no config binding for "
                        f"cell {cell['id']}: {config_path}"
                    )
                hashes.append(str(summary["config_sha256"]))
                config_dataset_paths.update(
                    str(path) for path in summary.get("dataset_paths", [])
                )
            cell["config_sha256s"] = hashes
            if "dataset_paths" not in cell:
                cell["expected_dataset_paths"] = sorted(config_dataset_paths)
        checkpoint = cell.get("checkpoint")
        if checkpoint is not None:
            summary = checkpoint_index.get(str(checkpoint))
            if summary is None:
                raise ValueError(
                    f"Preflight has no checkpoint binding for cell {cell['id']}"
                )
            cell["checkpoint_sha256"] = str(summary["best_checkpoint_sha256"])
        checkpoints = cell.get("checkpoints")
        if isinstance(checkpoints, list):
            hashes: list[str] = []
            for checkpoint_path in checkpoints:
                summary = checkpoint_index.get(str(checkpoint_path))
                if summary is None:
                    raise ValueError(
                        "Preflight has no ensemble checkpoint binding for "
                        f"cell {cell['id']}: {checkpoint_path}"
                    )
                hashes.append(str(summary["best_checkpoint_sha256"]))
            cell["checkpoint_sha256s"] = hashes
        dataset_paths = cell.get("dataset_paths")
        if isinstance(dataset_paths, list):
            unknown = sorted(
                str(path)
                for path in dataset_paths
                if str(path) not in known_dataset_paths
            )
            if unknown:
                raise ValueError(
                    f"Preflight has no dataset binding for cell {cell['id']}: "
                    + ", ".join(unknown)
                )
            cell["expected_dataset_paths"] = [str(path) for path in dataset_paths]


def _accuracy_cell(
    *,
    cell_id: str,
    group: str,
    path: str,
    config: str,
    checkpoint: str,
    num_samples: int,
) -> dict[str, Any]:
    return {
        "id": cell_id,
        "group": group,
        "path": path,
        "evaluation_type": "accuracy",
        "config": config,
        "checkpoint": checkpoint,
        "num_samples": num_samples,
        "required_keys": [
            "mae",
            "rmse",
            "rel_l2",
            "max_error",
            "mae_physical",
            "rmse_physical",
            "rel_l2_physical",
            "checkpoint_epoch",
        ],
        "require_physical_metrics": True,
    }


def _paper_evidence_cells(
    contract: Mapping[str, Any],
    *,
    test_count: int,
    preflight: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    paper = contract["paper_evidence"]
    paper_preflight = (
        preflight.get("paper_evidence", {}) if isinstance(preflight, Mapping) else {}
    )
    slice_ids = [str(row["id"]) for row in paper["metadata_slices"]]
    bootstrap = paper["bootstrap"]
    cells: list[dict[str, Any]] = [
        {
            "id": "paper_evidence:numerical_evidence",
            "group": "paper_numerical_evidence",
            "path": "paper_evidence/numerical_evidence.json",
            "evaluation_type": "v2_numerical_evidence_index",
            "required_keys": [
                "contract_path",
                "contract_sha256",
                "rows",
            ],
            "row_count": len(contract.get("accepted_numerical_artifacts", [])),
            "row_identity_key": "id",
            "row_identities": [
                str(row["id"])
                for row in contract.get("accepted_numerical_artifacts", [])
            ],
            "row_required_keys": [
                "root",
                "decision_path",
                "decision_sha256",
                "decision",
                "checksum_files",
            ],
        }
    ]

    slice_counts = (
        paper_preflight.get("slice_counts", {})
        if isinstance(paper_preflight, Mapping)
        else {}
    )
    for row in paper["direct_slice_models"]:
        reference = str(row["reference"])
        cell = {
            "id": f"paper_slice_direct:{row['id']}",
            "group": "paper_slice_direct",
            "path": f"paper_evidence/slices/direct/{row['id']}.json",
            "evaluation_type": "v2_slice_metrics",
            "config": str(row["config"]),
            "checkpoint": str(row["checkpoint"]),
            "dataset_paths": [str(row["dataset"])],
            "row_count": len(slice_ids),
            "row_identity_key": "label",
            "row_identities": slice_ids,
            "row_required_keys": [
                "filter",
                "diagnostic_kind",
                "dataset_path",
                "num_samples",
                "metrics.rel_l2",
                "metrics_physical.rel_l2",
            ],
            "expected_values": {
                "diagnostic_kind": "metadata_subgroup",
                "aggregation": "mean_of_per_scenario_metrics",
                "group_by": None,
                "seeded_with_true_first_frame": False,
            },
        }
        counts = slice_counts.get(reference)
        if isinstance(counts, Mapping):
            cell["row_sample_counts"] = dict(counts)
        cells.append(cell)

    hydro_model = next(
        row for row in paper["direct_slice_models"] if row["reference"] == "hydrostatic"
    )
    group_counts = (
        paper_preflight.get("group_counts", {})
        if isinstance(paper_preflight, Mapping)
        else {}
    )
    for group_key in ("source_type", "bathymetry_type"):
        counts = group_counts.get(group_key, {})
        cell = {
            "id": f"paper_slice_group:{group_key}",
            "group": "paper_slice_group",
            "path": f"paper_evidence/slices/groups/{group_key}.json",
            "evaluation_type": "v2_slice_group_metrics",
            "config": str(hydro_model["config"]),
            "checkpoint": str(hydro_model["checkpoint"]),
            "dataset_paths": [str(hydro_model["dataset"])],
            "row_count": len(counts) if isinstance(counts, Mapping) else None,
            "row_identity_key": "label",
            "row_identities": (
                sorted(str(label) for label in counts)
                if isinstance(counts, Mapping)
                else []
            ),
            "row_required_keys": [
                "group_key",
                "diagnostic_kind",
                "num_samples",
                "metrics.rel_l2",
                "metrics_physical.rel_l2",
            ],
            "expected_values": {
                "diagnostic_kind": "metadata_subgroup",
                "aggregation": "mean_of_per_scenario_metrics",
                "group_by": group_key,
                "seeded_with_true_first_frame": False,
            },
        }
        if isinstance(counts, Mapping):
            cell["row_sample_counts"] = dict(counts)
        cells.append(cell)

    for row in paper["window_slice_models"]:
        cell = {
            "id": f"paper_slice_window:{row['id']}",
            "group": "paper_slice_window",
            "path": f"paper_evidence/slices/window/{row['id']}.json",
            "evaluation_type": "v2_window_slice_metrics",
            "config": str(row["config"]),
            "checkpoint": str(row["checkpoint"]),
            "dataset_paths": [str(row["dataset"])],
            "row_count": len(slice_ids),
            "row_identity_key": "label",
            "row_identities": slice_ids,
            "row_required_keys": [
                "filter",
                "diagnostic_kind",
                "dataset_path",
                "num_samples",
                "window_metrics.rel_l2",
                "window_metrics.rel_l2_physical",
                "window_metrics.num_predicted_frames",
            ],
            "expected_values": {
                "diagnostic_kind": "metadata_subgroup",
                "aggregation": "seeded_window_rollout",
                "seeded_with_true_first_frame": True,
                "window_K": 5,
            },
            "seeded_window_rollout": True,
        }
        counts = slice_counts.get("hydrostatic")
        if isinstance(counts, Mapping):
            cell["row_sample_counts"] = dict(counts)
        cells.append(cell)

    proxy = paper["proxy_resolution"]
    cells.append(
        {
            "id": f"paper_proxy_resolution:{proxy['id']}",
            "group": "paper_proxy_resolution",
            "path": "paper_evidence/resolution/proxy_hydrostatic.json",
            "evaluation_type": "proxy_resolution_transfer",
            "config": str(proxy["config"]),
            "checkpoint": str(proxy["checkpoint"]),
            "dataset_paths": [str(proxy["dataset"])],
            "required_keys": [
                "dataset_num_samples",
                "eval_resolutions",
                "rows",
            ],
            "expected_values": {
                "dataset_num_samples": test_count,
                "eval_resolutions": [int(value) for value in proxy["grids"]],
            },
            "row_collection_type": "mapping",
            "row_count": len(proxy["grids"]),
            "row_identities": [str(value) for value in proxy["grids"]],
            "row_required_keys": [
                "num_samples",
                "rel_l2",
                "rel_l2_physical",
            ],
            "row_num_samples": test_count,
        }
    )

    native = paper["native_transfer"]
    grids = [int(value) for value in native["grids"]]
    cells.append(
        {
            "id": "paper_native_resolution_transfer:muscl_hr",
            "group": "paper_native_resolution_transfer",
            "path": "paper_evidence/resolution/native_muscl_hr.json",
            "evaluation_type": "v2_native_resolution_transfer_matrix",
            "configs": [str(value) for value in native["configs"]],
            "checkpoints": [str(value) for value in native["checkpoints"]],
            "dataset_paths": [str(value) for value in native["datasets"]],
            "required_keys": [
                "reference",
                "grids",
                "matrix_rel_l2_physical",
                "rows",
                "normalization_policy",
            ],
            "expected_values": {
                "reference": "muscl_hr",
                "grids": grids,
            },
            "row_count": len(grids) * len(grids),
            "row_identity_keys": ["train_grid", "test_grid"],
            "row_identities": [
                [train_grid, test_grid] for train_grid in grids for test_grid in grids
            ],
            "row_required_keys": [
                "config_path",
                "checkpoint_path",
                "dataset_path",
                "num_samples",
                "metrics.rel_l2",
                "metrics_physical.rel_l2",
                "normalization_bridge.enabled",
            ],
            "row_bindings": [
                {
                    "identity": [train_grid, test_grid],
                    "config_path": str(native["configs"][train_index]),
                    "checkpoint_path": str(native["checkpoints"][train_index]),
                    "dataset_path": str(native["datasets"][test_index]),
                }
                for train_index, train_grid in enumerate(grids)
                for test_index, test_grid in enumerate(grids)
            ],
            "row_num_samples": int(contract["native_muscl"][0]["counts"]["test"]),
        }
    )

    reference_models = paper["reference_analysis"]["models"]
    configs = [str(row["config"]) for row in reference_models]
    checkpoints = [str(row["checkpoint"]) for row in reference_models]
    datasets = [str(row["dataset"]) for row in reference_models]
    directed_pairs = [
        [left, right]
        for left in ("hydrostatic", "muscl_hr", "boussinesq")
        for right in ("hydrostatic", "muscl_hr", "boussinesq")
        if left != right
    ]
    common_reference = {
        "configs": configs,
        "checkpoints": checkpoints,
        "dataset_paths": datasets,
        "num_samples": test_count,
        "expected_values": {
            "output_mode": "common_time_v2_processed",
            "bootstrap.seed": int(bootstrap["seed"]),
            "bootstrap.resamples": int(bootstrap["resamples"]),
            "bootstrap.confidence_level": float(bootstrap["confidence_level"]),
            "common_time_v2.horizon": 0.175,
            "common_time_v2.frame_count": 50,
        },
    }
    cells.append(
        {
            "id": "paper_reference_analysis:solver_gap",
            "group": "paper_reference_analysis",
            "path": "paper_evidence/reference_analysis/solver_gap.json",
            "evaluation_type": "v2_solver_gap",
            "required_keys": ["common_time_v2", "bootstrap", "pairs"],
            "rows_key": "pairs",
            "row_count": len(directed_pairs),
            "row_identity_keys": ["solver_a", "solver_b"],
            "row_identities": directed_pairs,
            "row_required_keys": [
                "relative_l2_reference",
                "metrics.num_samples",
                "metrics.global_field_rmse",
                "metrics.scenario_rel_l2_mean.mean",
                "metrics.scenario_rel_l2_mean.ci_lower",
                "metrics.scenario_rel_l2_mean.ci_upper",
            ],
            "row_num_samples": test_count,
            **common_reference,
        }
    )
    cells.append(
        {
            "id": "paper_reference_analysis:cross_reference",
            "group": "paper_reference_analysis",
            "path": "paper_evidence/reference_analysis/cross_reference.json",
            "evaluation_type": "v2_cross_reference_discrepancy",
            "required_keys": [
                "common_time_v2",
                "bootstrap",
                "model_specs",
                "directions",
            ],
            "rows_key": "directions",
            "row_count": len(directed_pairs),
            "row_identity_keys": ["model_solver", "benchmark_solver"],
            "row_identities": directed_pairs,
            "row_required_keys": [
                "num_samples",
                "numerator.global_field_rmse",
                "denominator_solver_gap.global_field_rmse",
                "same_reference_control.global_field_rmse",
                "rho.point_estimate",
                "rho.ci_lower",
                "rho.ci_upper",
                "interpretation",
            ],
            "row_num_samples": test_count,
            **common_reference,
        }
    )

    wave = paper["wave_metrics"]
    gauge_values = [{"row": int(row), "col": int(col)} for row, col in wave["gauges"]]
    for row in wave["models"]:
        model_id = str(row["id"])
        cells.extend(
            [
                {
                    "id": f"paper_arrival_map:{model_id}",
                    "group": "paper_arrival_map",
                    "path": (f"paper_evidence/arrival_maps/{model_id}.json"),
                    "evaluation_type": "arrival_map_model_vs_target",
                    "config": str(row["config"]),
                    "checkpoint": str(row["checkpoint"]),
                    "dataset_paths": [str(row["dataset"])],
                    "num_samples": test_count,
                    "required_keys": [
                        "arrival_map_shape",
                        "arrival_valid_fraction_mean",
                        "arrival_mean_abs_diff_steps_map_mean",
                        "sample_mean_abs_diff_steps",
                        "num_samples_compared",
                        "maps_path",
                    ],
                    "expected_values": {
                        "arrival_threshold_fraction": float(
                            wave["arrival_threshold_fraction"]
                        ),
                        "target_units": "physical",
                        "num_samples_compared": test_count,
                    },
                    "companion_path_fields": {
                        "maps_path": (f"paper_evidence/arrival_maps/{model_id}.npz")
                    },
                },
                {
                    "id": f"paper_arrival_map_npz:{model_id}",
                    "group": "companion_artifacts",
                    "path": (f"paper_evidence/arrival_maps/{model_id}.npz"),
                    "file_only": True,
                    "npz_required_keys": [
                        "coverage_map",
                        "mean_arrival_step_pred",
                        "mean_arrival_step_target",
                        "mean_abs_diff_steps",
                        "valid_pair_count",
                        "sample_count",
                    ],
                    "npz_allow_nonfinite_keys": [
                        "mean_abs_diff_steps",
                    ],
                },
                {
                    "id": f"paper_wave_metrics:{model_id}",
                    "group": "paper_wave_metrics",
                    "path": (f"paper_evidence/wave_metrics/{model_id}.json"),
                    "evaluation_type": "v2_gauge_waveform_peak_metrics",
                    "config": str(row["config"]),
                    "checkpoint": str(row["checkpoint"]),
                    "dataset_paths": [str(row["dataset"])],
                    "num_samples": test_count,
                    "required_keys": [
                        "common_time_v2",
                        "gauges",
                        "aggregates.waveform_nrmse",
                        "aggregates.arrival_time_abs_error",
                        "aggregates.peak_elevation_abs_error",
                        "aggregates.peak_elevation_relative_error",
                        "aggregates.time_to_peak_abs_error",
                        "per_gauge",
                    ],
                    "expected_values": {
                        "gauges": gauge_values,
                        "arrival_threshold_fraction": float(
                            wave["arrival_threshold_fraction"]
                        ),
                        "peak_plateau_fraction": float(wave["peak_plateau_fraction"]),
                        "common_time_v2.horizon": 0.175,
                        "common_time_v2.frame_count": 50,
                    },
                    "require_physical_target_units": True,
                },
            ]
        )

    ensemble = contract["ensemble"]
    ensemble_contract = paper["ensemble"]
    ensemble_checkpoints = [
        str(ensemble["checkpoint_template"]).format(seed=int(seed))
        for seed in ensemble["required_members"]
    ]
    ensemble_slice = {
        "id": "paper_ensemble:slices",
        "group": "paper_ensemble",
        "path": "paper_evidence/ensemble/slices.json",
        "evaluation_type": "v2_ensemble_slice_metrics",
        "config": str(ensemble_contract["config"]),
        "checkpoints": ensemble_checkpoints,
        "dataset_paths": [str(ensemble_contract["test_dataset"])],
        "row_count": len(slice_ids),
        "row_identity_key": "label",
        "row_identities": slice_ids,
        "row_required_keys": [
            "filter",
            "diagnostic_kind",
            "dataset_path",
            "num_samples",
            "coverage_90",
            "coverage_90_physical",
            "nll",
            "nll_physical",
            "error_uncertainty_corr",
            "error_uncertainty_corr_physical",
        ],
        "expected_values": {
            "diagnostic_kind": "metadata_subgroup",
        },
    }
    counts = slice_counts.get("hydrostatic")
    if isinstance(counts, Mapping):
        ensemble_slice["row_sample_counts"] = dict(counts)
    cells.append(ensemble_slice)
    cells.append(
        {
            "id": "paper_ensemble:calibration",
            "group": "paper_ensemble",
            "path": "paper_evidence/ensemble/calibration.json",
            "evaluation_type": "v2_ensemble_calibration",
            "config": str(ensemble_contract["config"]),
            "checkpoints": ensemble_checkpoints,
            "dataset_paths": [
                str(ensemble_contract["val_dataset"]),
                str(ensemble_contract["test_dataset"]),
            ],
            "required_keys": [
                "nominal_levels",
                "fit.gamma",
                "fit.n_samples",
                "datasets.validation",
                "datasets.test",
                *[f"datasets.{slice_id}" for slice_id in slice_ids],
                "datasets.validation.error_uncertainty_corr",
                "datasets.test.error_uncertainty_corr",
            ],
            "expected_values": {
                "calibration_fit_split": "validation",
            },
        }
    )
    cells.append(
        {
            "id": "paper_ensemble:seed_stability",
            "group": "paper_ensemble",
            "path": "paper_evidence/ensemble/seed_stability.json",
            "evaluation_type": "v2_seed_stability",
            "config": str(ensemble_contract["config"]),
            "checkpoints": ensemble_checkpoints,
            "dataset_paths": [str(ensemble_contract["test_dataset"])],
            "num_samples": test_count,
            "required_keys": [
                "member_count",
                "bootstrap",
                "members",
                "seed_summary",
                "interpretation",
            ],
            "expected_values": {
                "member_count": len(ensemble_checkpoints),
                "bootstrap.seed": int(bootstrap["seed"]),
                "bootstrap.resamples": int(bootstrap["resamples"]),
                "bootstrap.confidence_level": float(bootstrap["confidence_level"]),
            },
            "rows_key": "members",
            "row_count": len(ensemble_checkpoints),
            "row_identity_key": "checkpoint",
            "row_identities": ensemble_checkpoints,
            "row_required_keys": [
                "checkpoint_epoch",
                "metrics.mae.mean",
                "metrics.rmse.mean",
                "metrics.rel_l2.mean",
                "metrics.max_error.mean",
            ],
        }
    )
    return cells


def build_manifest(
    contract: Mapping[str, Any],
    *,
    run_id: str,
    include_ensemble: bool,
    include_real_bathymetry: bool,
    include_speed: bool,
    include_paper_evidence: bool = False,
    rerun_numerical_validation: bool = False,
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    test_count = int(contract["main_datasets"]["splits"]["test"]["count"])

    for model in contract.get("direct_models", []):
        model_id = str(model["id"])
        base = f"direct/{model_id}"
        cells.append(
            _accuracy_cell(
                cell_id=model_id,
                group="direct_accuracy",
                path=f"{base}/metrics.json",
                config=str(model["config"]),
                checkpoint=str(model["checkpoint"]),
                num_samples=test_count,
            )
        )
        cells[-1]["id"] = f"direct_accuracy:{model_id}"
        if "perframe" in model.get("analyses", []):
            cells.append(
                {
                    "id": f"direct_perframe:{model_id}",
                    "group": "direct_perframe",
                    "path": f"{base}/perframe.json",
                    "evaluation_type": "per_frame_error",
                    "config": str(model["config"]),
                    "checkpoint": str(model["checkpoint"]),
                    "num_samples": test_count,
                    "required_keys": [
                        "num_frames",
                        "per_frame",
                        "per_frame_physical",
                    ],
                    "expected_values": {"num_frames": 50},
                }
            )
        if "physics" in model.get("analyses", []):
            cells.append(
                {
                    "id": f"direct_physics:{model_id}",
                    "group": "direct_physics",
                    "path": f"{base}/physics_diagnostics.json",
                    "evaluation_type": "physics_diagnostics",
                    "config": str(model["config"]),
                    "checkpoint": str(model["checkpoint"]),
                    "num_samples": test_count,
                    "required_keys": ["diagnostics", "target_units"],
                    "require_physical_target_units": True,
                }
            )
            cells.append(
                {
                    "id": f"direct_physics_csv:{model_id}",
                    "group": "companion_artifacts",
                    "path": f"{base}/physics_diagnostics_per_sample.csv",
                    "file_only": True,
                }
            )

    for model in contract.get("window_models", []):
        model_id = str(model["id"])
        base = f"window/{model_id}"
        cells.append(
            {
                "id": f"conditional_window_accuracy:{model_id}",
                "group": "conditional_window_accuracy",
                "path": f"{base}/metrics.json",
                "evaluation_type": "conditional_seeded_window_rollout",
                "config": str(model["config"]),
                "checkpoint": str(model["checkpoint"]),
                "num_samples": test_count,
                "required_keys": [
                    "mae",
                    "rmse",
                    "rel_l2",
                    "mae_physical",
                    "rmse_physical",
                    "rel_l2_physical",
                    "num_predicted_frames",
                ],
                "require_physical_metrics": True,
                "seeded_window_rollout": True,
                "expected_values": {"num_predicted_frames": 49},
            }
        )
        cells.append(
            {
                "id": f"conditional_window_perframe:{model_id}",
                "group": "conditional_window_perframe",
                "path": f"{base}/perframe.json",
                "evaluation_type": "window_rollout_perframe",
                "config": str(model["config"]),
                "checkpoint": str(model["checkpoint"]),
                "num_samples": test_count,
                "required_keys": ["per_frame", "num_frames"],
                "seeded_window_rollout": True,
                "expected_values": {"num_frames": 49},
            }
        )

    for row in contract.get("sample_scaling", []):
        cells.append(
            _accuracy_cell(
                cell_id=str(row["id"]),
                group="sample_scaling",
                path=f"sample_scaling/{row['id']}/metrics.json",
                config=str(row["config"]),
                checkpoint=str(row["checkpoint"]),
                num_samples=test_count,
            )
        )
        cells[-1]["id"] = f"sample_scaling:{row['id']}"

    for row in contract.get("native_muscl", []):
        cells.append(
            _accuracy_cell(
                cell_id=str(row["id"]),
                group="native_muscl_diagonal",
                path=f"native_muscl/{row['id']}/metrics.json",
                config=str(row["config"]),
                checkpoint=str(row["checkpoint"]),
                num_samples=int(row["counts"]["test"]),
            )
        )
        cells[-1]["id"] = f"native_muscl:{row['id']}"

    for row in contract.get("strict_holdouts", []):
        holdout_manifest = _read_json(ROOT / str(row["manifest"]))
        counts = {
            str(split.get("kind", split.get("split"))): int(split.get("num_samples", 0))
            for split in holdout_manifest.get("splits", [])
            if isinstance(split, Mapping)
        }
        label = str(row["id"])
        base = f"strict_holdout/{label}"
        specs = [
            (
                "id",
                str(row["config_id"]),
                str(row["checkpoint"]),
                counts["test_id"],
                f"{base}/eval_id/metrics.json",
            ),
            (
                "heldout",
                str(row["config_heldout"]),
                str(row["checkpoint"]),
                counts["test_heldout"],
                f"{base}/eval_heldout/metrics.json",
            ),
            (
                "full_on_heldout",
                str(row["config_full"]),
                str(contract["strict_holdout_full_checkpoint"]),
                counts["test_heldout"],
                f"{base}/full_on_heldout/metrics.json",
            ),
        ]
        for kind, config, checkpoint, count, path in specs:
            cell = _accuracy_cell(
                cell_id=f"strict_holdout_accuracy:{label}:{kind}",
                group="strict_holdout_accuracy",
                path=path,
                config=config,
                checkpoint=checkpoint,
                num_samples=count,
            )
            if kind == "full_on_heldout":
                cell["require_normalization_bridge"] = True
            cells.append(cell)
        for kind, config, count, path in (
            (
                "id",
                str(row["config_id"]),
                counts["test_id"],
                f"{base}/eval_id",
            ),
            (
                "heldout",
                str(row["config_heldout"]),
                counts["test_heldout"],
                f"{base}/eval_heldout",
            ),
        ):
            cells.append(
                {
                    "id": f"strict_holdout_perframe:{label}:{kind}",
                    "group": "strict_holdout_perframe",
                    "path": f"{path}/perframe.json",
                    "evaluation_type": "per_frame_error",
                    "config": config,
                    "checkpoint": str(row["checkpoint"]),
                    "num_samples": count,
                    "required_keys": [
                        "num_frames",
                        "per_frame",
                        "per_frame_physical",
                    ],
                    "expected_values": {"num_frames": 50},
                }
            )
            cells.append(
                {
                    "id": f"strict_holdout_physics:{label}:{kind}",
                    "group": "strict_holdout_physics",
                    "path": f"{path}/physics_diagnostics.json",
                    "evaluation_type": "physics_diagnostics",
                    "config": config,
                    "checkpoint": str(row["checkpoint"]),
                    "num_samples": count,
                    "required_keys": ["diagnostics", "target_units"],
                    "require_physical_target_units": True,
                }
            )
            cells.append(
                {
                    "id": f"strict_holdout_physics_csv:{label}:{kind}",
                    "group": "companion_artifacts",
                    "path": f"{path}/physics_diagnostics_per_sample.csv",
                    "file_only": True,
                }
            )
    cells.append(
        {
            "id": "strict_holdout_summary",
            "group": "strict_holdout_summary",
            "path": "strict_holdout/strict_holdout_summary.json",
            "required_keys": [
                "expected_holdouts",
                "missing_outputs",
                "rows",
            ],
            "row_count": len(contract.get("strict_holdouts", [])),
            "row_identity_key": "label",
            "row_identities": [
                str(row["id"]) for row in contract.get("strict_holdouts", [])
            ],
            "require_empty_keys": ["missing_outputs"],
        }
    )
    cells.append(
        {
            "id": "strict_holdout_summary_csv",
            "group": "companion_artifacts",
            "path": "strict_holdout/strict_holdout_summary.csv",
            "file_only": True,
        }
    )

    if include_real_bathymetry:
        real = contract.get("real_bathymetry", {})
        suite_counts = {
            str(key): int(value) for key, value in real.get("suites", {}).items()
        }
        for row in real.get("direct", []):
            cells.append(
                {
                    "id": f"real_bathymetry_direct:{row['id']}",
                    "group": "real_bathymetry_direct",
                    "path": f"real_bathymetry/direct/{row['id']}.json",
                    "evaluation_type": "native_real_resolution_benchmark",
                    "config": str(row["config"]),
                    "checkpoint": str(row["checkpoint"]),
                    "required_keys": ["rows", "normalization_reference"],
                    "row_count": len(suite_counts),
                    "row_sample_counts": suite_counts,
                    "row_required_keys": [
                        "mae_physical",
                        "rmse_physical",
                        "rel_l2_physical",
                    ],
                }
            )
        for row in real.get("window", []):
            cells.append(
                {
                    "id": f"real_bathymetry_window:{row['id']}",
                    "group": "real_bathymetry_conditional_window",
                    "path": f"real_bathymetry/window/{row['id']}.json",
                    "evaluation_type": "window_rollout_suites",
                    "config": str(row["config"]),
                    "checkpoint": str(row["checkpoint"]),
                    "required_keys": ["rows", "window_K"],
                    "row_count": len(suite_counts),
                    "row_sample_counts": suite_counts,
                    "row_required_keys": [
                        "mae_physical",
                        "rmse_physical",
                        "rel_l2_physical",
                    ],
                    "seeded_window_rollout": True,
                }
            )

    cells.extend(
        [
            {
                "id": "main",
                "group": "dataset_summary",
                "path": "dataset_summary.json",
                "evaluation_type": "dataset_summary",
                "required_keys": ["datasets", "total_samples_by_dataset"],
                "dataset_totals": {
                    str(reference): sum(
                        int(split["count"])
                        for split in contract["main_datasets"]["splits"].values()
                    )
                    for reference in ("hydrostatic", "muscl_hr", "boussinesq")
                },
            },
            {
                "id": "parameter_counts",
                "group": "parameter_counts",
                "path": "parameter_counts.json",
                "evaluation_type": "parameter_counts",
                "required_keys": ["rows", "missing_checkpoints"],
                "require_empty_keys": ["missing_checkpoints"],
                "row_count": len(contract.get("direct_models", []))
                + len(contract.get("window_models", [])),
                "row_identity_key": "model",
                "row_identities": [
                    str(row["id"])
                    for row in (
                        list(contract.get("direct_models", []))
                        + list(contract.get("window_models", []))
                    )
                ],
            },
            {
                "id": "parameter_counts_csv",
                "group": "companion_artifacts",
                "path": "parameter_counts.csv",
                "file_only": True,
            },
        ]
    )

    if include_ensemble:
        ensemble = contract["ensemble"]
        checkpoints = [
            str(ensemble["checkpoint_template"]).format(seed=int(seed))
            for seed in ensemble["required_members"]
        ]
        for row in ensemble.get("configs", []):
            cells.append(
                {
                    "id": f"uncertainty:{row['id']}",
                    "group": "uncertainty",
                    "path": f"ensemble/{row['id']}.json",
                    "evaluation_type": "in_distribution_uncertainty",
                    "config": str(row["config"]),
                    "checkpoints": checkpoints,
                    "num_samples": test_count,
                    "required_keys": [
                        "coverage_50",
                        "coverage_80",
                        "coverage_90",
                        "coverage_95",
                        "nll",
                        "error_uncertainty_corr",
                        "coverage_50_physical",
                        "coverage_80_physical",
                        "coverage_90_physical",
                        "coverage_95_physical",
                        "nll_physical",
                        "error_uncertainty_corr_physical",
                    ],
                }
            )

    if include_speed:
        for model in contract.get("direct_models", []):
            cells.append(
                {
                    "id": f"model_speed:{model['id']}",
                    "group": "model_speed",
                    "path": f"speed/speed_{model['id']}.json",
                    "evaluation_type": "model_speed_benchmark",
                    "config": str(model["config"]),
                    "checkpoint": str(model["checkpoint"]),
                    "required_keys": [
                        "time_per_sample_mean_s",
                        "samples_per_second",
                        "hardware",
                    ],
                }
            )
        for solver in ("swe_hydrostatic", "swe_muscl_hr", "boussinesq"):
            cells.append(
                {
                    "id": f"solver_speed:{solver}",
                    "group": "solver_speed",
                    "path": f"speed/solver_speed_{solver}.json",
                    "evaluation_type": "solver_speed_benchmark",
                    "required_keys": [
                        "rollout_time_per_sample_s",
                        "num_scenarios",
                    ],
                }
            )
        cells.append(
            {
                "id": "speed_table",
                "group": "speed_table",
                "path": "speed/speed_table.json",
                "evaluation_type": "speed_table",
                "required_keys": ["rows", "missing_inputs"],
                "require_empty_keys": ["missing_inputs"],
                "row_count": len(contract.get("direct_models", [])),
                "row_identity_key": "model",
                "row_identities": [
                    str(row["id"]) for row in contract.get("direct_models", [])
                ],
            }
        )
        cells.append(
            {
                "id": "speed_table_csv",
                "group": "companion_artifacts",
                "path": "speed/speed_table.csv",
                "file_only": True,
            }
        )

    if include_paper_evidence:
        if not include_ensemble:
            raise ValueError("Paper evidence requires the seven-member ensemble")
        cells.extend(
            _paper_evidence_cells(
                contract,
                test_count=test_count,
                preflight=preflight,
            )
        )

    if rerun_numerical_validation:
        cells.extend(
            [
                {
                    "id": "numerical_validation:summary",
                    "group": "numerical_validation_rerun",
                    "path": "numerical_validation/summary.json",
                    "evaluation_type": ("v2_numerical_validation_regression_chain"),
                    "required_keys": [
                        "status",
                        "interpretation",
                        "code_state.code_state_hash",
                        "external_revisions",
                        "stages",
                        "archive_path",
                        "archive_sha256",
                        "archive_size_bytes",
                    ],
                    "expected_values": {"status": "passed"},
                    "row_count": 5,
                    "rows_key": "stages",
                    "row_identity_key": "id",
                    "row_identities": [
                        "h0",
                        "level_a",
                        "minimum_established_solver",
                        "h1",
                        "h2_v2",
                    ],
                    "row_required_keys": [
                        "artifact_hash",
                        "decision",
                    ],
                    "companion_path_fields": {
                        "archive_path": "numerical_validation/chain.tar.zst",
                    },
                    "companion_sha256_fields": {
                        "archive_sha256": "numerical_validation/chain.tar.zst",
                    },
                },
                {
                    "id": "numerical_validation:archive",
                    "group": "companion_artifacts",
                    "path": "numerical_validation/chain.tar.zst",
                    "file_only": True,
                },
            ]
        )

    if preflight is not None:
        _bind_preflight(cells, preflight)

    manifest = {
        "schema_id": "tsunami-surrogate.evaluation-run-manifest.v1",
        "suite_id": contract["suite_id"],
        "run_id": run_id,
        "include_ensemble": bool(include_ensemble),
        "include_real_bathymetry": bool(include_real_bathymetry),
        "include_speed": bool(include_speed),
        "include_paper_evidence": bool(include_paper_evidence),
        "rerun_numerical_validation": bool(rerun_numerical_validation),
        "cells": cells,
    }
    if preflight is not None:
        manifest["preflight_report"] = "preflight_report.json"
        manifest["code_state"] = dict(preflight.get("code_state", {}))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/eval/final_v2_suite.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--include-ensemble", action="store_true")
    parser.add_argument("--include-real-bathymetry", action="store_true")
    parser.add_argument("--include-speed", action="store_true")
    parser.add_argument("--include-paper-evidence", action="store_true")
    parser.add_argument("--rerun-numerical-validation", action="store_true")
    args = parser.parse_args()

    contract = load_suite_contract(args.contract)
    preflight_path = Path(args.preflight_report)
    preflight = _read_json(preflight_path)
    manifest = build_manifest(
        contract,
        run_id=args.run_id,
        include_ensemble=bool(args.include_ensemble),
        include_real_bathymetry=bool(args.include_real_bathymetry),
        include_speed=bool(args.include_speed),
        include_paper_evidence=bool(args.include_paper_evidence),
        rerun_numerical_validation=bool(args.rerun_numerical_validation),
        preflight=preflight,
    )
    manifest["preflight_report_sha256"] = _sha256(preflight_path)
    _write_atomic(Path(args.output), manifest)
    print(f"[eval-manifest] cells={len(manifest['cells'])} -> {args.output}")


if __name__ == "__main__":
    main()
