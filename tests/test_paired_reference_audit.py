from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT_MODULE = _load_script_module(
    "scripts/audit_paired_reference_data.py",
    "audit_paired_reference_data_module",
)
SELECTION_MODULE = _load_script_module(
    "scripts/select_common_time_validation_scenarios.py",
    "select_common_time_validation_scenarios_module",
)

SOLVER_KEYS = ("hydrostatic", "muscl_hr", "boussinesq")
SOLVER_RUNTIME_NAMES = {
    "hydrostatic": "swe_hydrostatic",
    "muscl_hr": "swe_muscl_hr",
    "boussinesq": "boussinesq",
}
BASE_SOLVER_CFG = {
    "dx": 0.015625,
    "dy": 0.015625,
    "g": 9.81,
    "boundary": "open",
    "use_sponge": True,
    "sponge_width": 20,
    "sponge_min_factor": 0.9,
    "dry_tolerance": "1e-6",
    "max_velocity": 30.0,
    "alpha": 1.0 / 3.0,
    "filter_strength": 0.01,
    "linear_solver_tol": "1e-6",
    "linear_solver_max_iter": 500,
    "min_depth": "1e-4",
    "depth_scale": 1.0,
    "mode": "linear_variable_depth",
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _scenario_id(sample_index: int) -> str:
    return f"scenario_{sample_index:06d}"


def _compute_common_arrays(
    *,
    bathymetry: np.ndarray,
    source_field: np.ndarray,
    source_strength: float,
) -> dict[str, np.ndarray]:
    bathymetry_values = np.asarray(bathymetry, dtype=np.float32)
    source_values = np.asarray(source_field, dtype=np.float32)
    eta0 = np.asarray(np.float32(source_strength) * source_values, dtype=np.float32)
    rest_depth = np.asarray(
        np.maximum(-np.asarray(bathymetry_values, dtype=np.float64), 0.0),
        dtype=np.float32,
    )
    initial_depth = np.asarray(
        np.maximum(
            np.asarray(rest_depth, dtype=np.float64)
            + np.asarray(eta0, dtype=np.float64),
            0.0,
        ),
        dtype=np.float32,
    )
    free_surface0 = np.asarray(
        np.asarray(initial_depth, dtype=np.float64)
        + np.asarray(bathymetry_values, dtype=np.float64),
        dtype=np.float32,
    )
    return {
        "bathymetry": bathymetry_values,
        "source_field": source_values,
        "eta0": eta0,
        "rest_depth": rest_depth,
        "initial_depth": initial_depth,
        "free_surface0": free_surface0,
    }


def _make_audit_config(
    tmp_path: Path,
    *,
    processed_root_paths: dict[str, Path],
    scenario_manifest_path: Path,
    expected_scenario_count: int,
    common_time_grid: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "schema_id": "tsunami-surrogate.alignment.v1",
        "alignment": {
            "mode": "common-time",
            "field": "trajectory_eta",
            "elevation_semantics": "sea_level_offset_relative_surface_elevation",
            "time_semantics": "solver_benchmark_time",
            "initial_frame_treatment": "require_saved_zero_frame_but_exclude_zero_from_common_grid",
            "aggregation": {
                "global_metric": "equal_scenario_weight_field_rmse",
            },
            "common_time_grid": {
                "values": common_time_grid or [0.004, 0.008],
                "endpoint_tolerance": 1.0e-6,
            },
        },
        "audit": {
            "expected_scenario_count": int(expected_scenario_count),
            "processed_test_roots": {
                solver_key: str(path)
                for solver_key, path in processed_root_paths.items()
            },
            "raw_test_solver_roots": {},
            "scenario_manifest_path": str(scenario_manifest_path),
            "results_dir": str(tmp_path / "results"),
            "equality": {
                "array_atol": 1.0e-6,
                "scalar_atol": 1.0e-12,
            },
            "timestamp": {
                "initial_zero_tolerance": 1.0e-7,
            },
            "clipping": {
                "block_true_pre_clipping": True,
                "roundoff_residual_atol": 1.0e-6,
            },
            "reconstruction_control": {
                "enabled": True,
                "float32_atol": 1.0e-6,
            },
        },
    }


def _write_processed_hydrostatic_dataset(
    processed_root: Path,
    *,
    scenarios: list[dict[str, Any]],
    input_order: list[str],
    input_stats: dict[str, tuple[float, float]],
    input_overrides: dict[str, np.ndarray] | None = None,
) -> Path:
    input_overrides = input_overrides or {}
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scenario_ids: list[str] = []
    sample_ids: list[str] = []
    source_types: list[str] = []
    bathymetry_types: list[str] = []
    source_strengths: list[float] = []
    solver_names: list[str] = []

    for spec in scenarios:
        sample_index = int(spec["sample_index"])
        scenario_id = _scenario_id(sample_index)
        common_arrays = _compute_common_arrays(
            bathymetry=np.asarray(spec["bathymetry"], dtype=np.float32),
            source_field=np.asarray(spec["source_field"], dtype=np.float32),
            source_strength=float(np.float32(spec["source_strength"])),
        )
        channel_map = {
            "bathymetry": common_arrays["bathymetry"],
            "source": common_arrays["source_field"],
            "initial_depth": common_arrays["initial_depth"],
            "initial_surface": common_arrays["free_surface0"],
        }
        sample_channels = []
        for channel_name in input_order:
            values = np.asarray(channel_map[channel_name], dtype=np.float32)
            if channel_name in input_stats:
                offset, scale = input_stats[channel_name]
                values = (values - float(offset)) / float(scale)
            sample_channels.append(np.asarray(values, dtype=np.float32))
        reconstructed = np.stack(sample_channels, axis=0).astype(np.float32, copy=False)
        reconstructed = np.asarray(
            input_overrides.get(scenario_id, reconstructed),
            dtype=np.float32,
        )
        inputs.append(reconstructed)
        targets.append(np.zeros((1,) + reconstructed.shape[1:], dtype=np.float32))
        scenario_ids.append(scenario_id)
        sample_ids.append(f"sample_{sample_index:06d}")
        source_types.append(str(spec["source_type"]))
        bathymetry_types.append(str(spec["bathymetry_type"]))
        source_strengths.append(float(np.float32(spec["source_strength"])))
        solver_names.append(SOLVER_RUNTIME_NAMES["hydrostatic"])

    shard_dir = processed_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard_00000.npz"
    np.savez_compressed(
        shard_path,
        inputs=np.stack(inputs, axis=0).astype(np.float32),
        targets=np.stack(targets, axis=0).astype(np.float32),
        sample_id=np.asarray(sample_ids, dtype=np.str_),
        source_id=np.asarray(source_types, dtype=np.str_),
        source_type=np.asarray(source_types, dtype=np.str_),
        bathymetry_type=np.asarray(bathymetry_types, dtype=np.str_),
        source_strength=np.asarray(source_strengths, dtype=np.float32),
        scenario_id=np.asarray(scenario_ids, dtype=np.str_),
        solver_name=np.asarray(solver_names, dtype=np.str_),
        target_variable=np.asarray(["eta"], dtype=np.str_),
        target_mean=np.asarray([0.0], dtype=np.float32),
        target_std=np.asarray([1.0], dtype=np.float32),
        target_min=np.asarray([0.0], dtype=np.float32),
        target_max=np.asarray([0.0], dtype=np.float32),
        input_order=np.asarray(input_order, dtype=np.str_),
    )
    (processed_root / "shards_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "split": "test",
                "sharded": True,
                "num_samples": len(scenarios),
                "num_shards": 1,
                "shard_size": len(scenarios),
                "shards": [
                    {
                        "file": str(shard_path.relative_to(processed_root)),
                        "num_samples": len(scenarios),
                        "inputs_shape": list(map(int, np.stack(inputs, axis=0).shape)),
                        "targets_shape": list(
                            map(int, np.stack(targets, axis=0).shape)
                        ),
                    }
                ],
                "input_order": list(input_order),
                "target_mode": "multi_step",
                "target_variable": "eta",
                "normalized_targets": True,
                "target_mean": 0.0,
                "target_std": 1.0,
                "target_min": 0.0,
                "target_max": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (processed_root / "eval_manifest.json").write_text(
        json.dumps(
            {
                "split": "test",
                "sharded": True,
                "shards_manifest": "shards_manifest.json",
                "input_order": list(input_order),
                "target_mode": "multi_step",
                "target_variable": "eta",
                "normalized_targets": True,
                "num_samples": len(scenarios),
                "num_shards": 1,
                "inputs_shape": list(map(int, np.stack(inputs, axis=0).shape)),
                "targets_shape": list(map(int, np.stack(targets, axis=0).shape)),
            }
        ),
        encoding="utf-8",
    )

    stats_path = processed_root.parent / "normalization_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "method": "standardize",
                "inputs": {
                    name: {
                        "offset": float(offset),
                        "scale": float(scale),
                    }
                    for name, (offset, scale) in sorted(input_stats.items())
                },
                "targets": {
                    "enabled": True,
                    "variable": "eta",
                    "offset": 0.0,
                    "scale": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return stats_path


def _build_fixture(
    tmp_path: Path,
    *,
    scenarios: list[dict[str, Any]],
    solver_cfg_overrides: dict[str, dict[str, Any]] | None = None,
    sample_array_overrides: dict[tuple[str, int], dict[str, np.ndarray]] | None = None,
    rows_mutator: Callable[[dict[str, list[dict[str, Any]]]], None] | None = None,
    common_time_grid: list[float] | None = None,
) -> dict[str, Any]:
    processed_root_paths = {
        solver_key: tmp_path / "processed" / solver_key / "test"
        for solver_key in SOLVER_KEYS
    }
    raw_root = tmp_path / "raw"
    bathymetry_cache_dir = tmp_path / "bathymetry_cache"
    source_cache_dir = tmp_path / "source_cache"

    scenario_manifest_rows: list[dict[str, Any]] = []
    rows_by_solver: dict[str, list[dict[str, Any]]] = {
        solver_key: [] for solver_key in SOLVER_KEYS
    }
    solver_cfg_overrides = solver_cfg_overrides or {}
    sample_array_overrides = sample_array_overrides or {}

    for spec in scenarios:
        sample_index = int(spec["sample_index"])
        scenario_id = _scenario_id(sample_index)
        source_strength = float(np.float32(spec["source_strength"]))
        common_arrays = _compute_common_arrays(
            bathymetry=np.asarray(spec["bathymetry"], dtype=np.float32),
            source_field=np.asarray(spec["source_field"], dtype=np.float32),
            source_strength=source_strength,
        )
        bathymetry_cache_path = bathymetry_cache_dir / f"sample_{sample_index:06d}.npz"
        source_cache_path = source_cache_dir / f"sample_{sample_index:06d}.npz"
        bathymetry_cache_path.parent.mkdir(parents=True, exist_ok=True)
        source_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bathymetry_cache_path,
            bathymetry=common_arrays["bathymetry"],
            bathymetry_type=np.asarray([spec["bathymetry_type"]], dtype=np.str_),
        )
        np.savez_compressed(
            source_cache_path,
            source_field=common_arrays["source_field"],
            source_type=np.asarray([spec["source_type"]], dtype=np.str_),
            source_strength=np.asarray([source_strength], dtype=np.float32),
        )
        scenario_manifest_rows.append(
            {
                "sample_index": sample_index,
                "scenario_id": scenario_id,
                "bathymetry_type": spec["bathymetry_type"],
                "source_type": spec["source_type"],
                "source_strength": source_strength,
                "bathymetry_cache_path": str(bathymetry_cache_path),
                "source_cache_path": str(source_cache_path),
            }
        )

        for solver_key in SOLVER_KEYS:
            solver_cfg = dict(BASE_SOLVER_CFG)
            solver_cfg.update(solver_cfg_overrides.get(solver_key, {}))
            sample_dir = (
                raw_root / solver_key / "samples" / f"sample_{sample_index:06d}"
            )
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_arrays = {
                name: np.asarray(values, dtype=np.float32)
                for name, values in common_arrays.items()
            }
            for field_name, override in sample_array_overrides.get(
                (solver_key, sample_index),
                {},
            ).items():
                sample_arrays[field_name] = np.asarray(override, dtype=np.float32)

            timestamps = np.asarray(spec["timestamps"], dtype=np.float32)
            np.savez_compressed(
                sample_dir / "sample.npz",
                bathymetry=sample_arrays["bathymetry"],
                source_field=sample_arrays["source_field"],
                eta0=sample_arrays["eta0"],
                rest_depth=sample_arrays["rest_depth"],
                initial_depth=sample_arrays["initial_depth"],
                free_surface0=sample_arrays["free_surface0"],
                timestamps=timestamps,
                scenario_id=np.asarray([scenario_id], dtype=np.str_),
                solver_name=np.asarray(
                    [SOLVER_RUNTIME_NAMES[solver_key]], dtype=np.str_
                ),
            )
            np.savez_compressed(
                sample_dir / "rollout.npz",
                timestamps=timestamps,
                fde_name=np.asarray([SOLVER_RUNTIME_NAMES[solver_key]], dtype=np.str_),
            )
            meta = {
                "sample_index": sample_index,
                "scenario_id": scenario_id,
                "solver_name": SOLVER_RUNTIME_NAMES[solver_key],
                "bathymetry_type": spec["bathymetry_type"],
                "source_type": spec["source_type"],
                "source_strength": source_strength,
                "num_frames": int(timestamps.size),
                "solver": solver_cfg,
            }
            (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            rows_by_solver[solver_key].append(
                {
                    "sample_index": sample_index,
                    "scenario_id": scenario_id,
                    "sample_dir": str(sample_dir),
                    "solver_name": SOLVER_RUNTIME_NAMES[solver_key],
                    "bathymetry_type": spec["bathymetry_type"],
                    "source_type": spec["source_type"],
                    "source_strength": source_strength,
                }
            )

    if rows_mutator is not None:
        rows_mutator(rows_by_solver)

    for solver_key, rows in rows_by_solver.items():
        _write_jsonl(processed_root_paths[solver_key] / "meta.jsonl", rows)

    scenario_manifest_path = tmp_path / "scenario_manifest.jsonl"
    _write_jsonl(scenario_manifest_path, scenario_manifest_rows)
    return _make_audit_config(
        tmp_path,
        processed_root_paths=processed_root_paths,
        scenario_manifest_path=scenario_manifest_path,
        expected_scenario_count=len(scenarios),
        common_time_grid=common_time_grid,
    )


def _run_audit(config: dict[str, Any]) -> dict[str, Any]:
    return AUDIT_MODULE.run_paired_reference_audit(config)


def _selection_audit_artifact(*, eligible_per_cell: int) -> dict[str, Any]:
    bathymetry_types = ["canyon", "continental", "island", "seamounts", "trench"]
    source_types = ["dipole", "fault", "gaussian", "multi-gauss", "okada-like", "rough"]
    eligible_scenarios = []
    family_cells = []
    counter = 1
    for bathymetry_type in bathymetry_types:
        for source_type in source_types:
            family_cells.append(
                {
                    "bathymetry_type": bathymetry_type,
                    "source_type": source_type,
                    "eligible_count": eligible_per_cell,
                }
            )
            for index in range(eligible_per_cell):
                eligible_scenarios.append(
                    {
                        "scenario_id": f"scenario_{counter:06d}",
                        "bathymetry_type": bathymetry_type,
                        "source_type": source_type,
                        "source_strength": 0.5 + 0.01 * index,
                    }
                )
                counter += 1
    return {
        "schema_id": "tsunami-surrogate.alignment.v1",
        "artifact_kind": "paired-reference-audit",
        "status": "pass",
        "audit_hash": "audit-hash-for-tests",
        "eligible_scenarios": eligible_scenarios,
        "family_cells": family_cells,
    }


def test_audit_accepts_and_records_solver_specific_differences(tmp_path: Path) -> None:
    scenarios = [
        {
            "sample_index": 1,
            "bathymetry_type": "seamounts",
            "source_type": "okada-like",
            "source_strength": 0.83,
            "bathymetry": np.asarray(
                [[-9.4543228, -9.0681038], [-1.0, -0.5]],
                dtype=np.float32,
            ),
            "source_field": np.asarray(
                [[0.0158, 0.0166], [0.01, -0.02]],
                dtype=np.float32,
            ),
            "timestamps": np.asarray([0.0, 0.004, 0.008, 0.012], dtype=np.float32),
        },
        {
            "sample_index": 2,
            "bathymetry_type": "trench",
            "source_type": "fault",
            "source_strength": 0.42,
            "bathymetry": np.asarray(
                [[-2.0, -1.0], [-0.75, -0.5]],
                dtype=np.float32,
            ),
            "source_field": np.asarray(
                [[0.01, -0.03], [0.04, 0.02]],
                dtype=np.float32,
            ),
            "timestamps": np.asarray([0.0, 0.004, 0.008, 0.012], dtype=np.float32),
        },
    ]
    config = _build_fixture(
        tmp_path,
        scenarios=scenarios,
        solver_cfg_overrides={
            "hydrostatic": {"max_velocity": 30.0},
            "muscl_hr": {"max_velocity": 25.0},
            "boussinesq": {"alpha": 0.25, "linear_solver_max_iter": 400},
        },
    )

    summary = _run_audit(config)

    assert summary["status"] == "pass"
    assert summary["reconstruction_control"]["status"] == "not_checked"
    assert (
        "Processed evaluation dataset is unavailable"
        in summary["reconstruction_control"]["reason"]
    )
    assert summary["solver_specific_settings"]["hydrostatic"]["max_velocity"][
        "unique_values"
    ]
    assert summary["solver_specific_settings"]["boussinesq"]["alpha"]["unique_values"]


def test_audit_roundoff_only_residual_is_nonblocking(tmp_path: Path) -> None:
    config = _build_fixture(
        tmp_path,
        scenarios=[
            {
                "sample_index": 1,
                "bathymetry_type": "canyon",
                "source_type": "gaussian",
                "source_strength": 0.9,
                "bathymetry": np.asarray(
                    [[-9.4543228, -9.0681038], [-0.5, -0.25]],
                    dtype=np.float32,
                ),
                "source_field": np.asarray(
                    [[0.0158, 0.0166], [0.01, -0.01]],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.008, 0.012], dtype=np.float32),
            },
        ],
    )

    summary = _run_audit(config)

    assert summary["status"] == "pass"
    assert summary["clipping_audit"]["true_pre_clipping_scenario_count"] == 0
    assert (
        summary["clipping_audit"]["free_surface0_minus_eta0_residual"][
            "nonzero_cell_count"
        ]
        > 0
    )
    assert (
        summary["clipping_audit"]["free_surface0_minus_eta0_residual"][
            "exceeds_tolerance_cell_count"
        ]
        == 0
    )


def test_audit_fails_on_true_pre_clipping(tmp_path: Path) -> None:
    config = _build_fixture(
        tmp_path,
        scenarios=[
            {
                "sample_index": 2,
                "bathymetry_type": "continental",
                "source_type": "dipole",
                "source_strength": 1.0,
                "bathymetry": np.asarray(
                    [[-0.01, -1.0], [-0.5, -0.25]],
                    dtype=np.float32,
                ),
                "source_field": np.asarray(
                    [[-10.0, 0.01], [0.02, 0.03]],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.008, 0.012], dtype=np.float32),
            },
        ],
    )

    summary = _run_audit(config)

    assert summary["status"] == "fail"
    assert summary["clipping_audit"]["true_pre_clipping_scenario_count"] == 1
    assert summary["clipping_audit"]["true_pre_clipping_cell_count"] >= 1
    assert any(issue["check"] == "true_pre_clipping" for issue in summary["issues"])


def test_audit_fails_on_common_field_mismatch(tmp_path: Path) -> None:
    mismatch_bathymetry = np.asarray(
        [[-8.0, -8.0], [-1.0, -0.5]],
        dtype=np.float32,
    )
    config = _build_fixture(
        tmp_path,
        scenarios=[
            {
                "sample_index": 1,
                "bathymetry_type": "seamounts",
                "source_type": "rough",
                "source_strength": 0.5,
                "bathymetry": np.asarray(
                    [[-9.4543228, -9.0681038], [-1.0, -0.5]],
                    dtype=np.float32,
                ),
                "source_field": np.asarray(
                    [[0.0158, 0.0166], [0.01, -0.02]],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.008, 0.012], dtype=np.float32),
            }
        ],
        sample_array_overrides={
            ("muscl_hr", 1): {"bathymetry": mismatch_bathymetry},
        },
    )

    summary = _run_audit(config)

    assert summary["status"] == "fail"
    assert any(
        issue["check"] == "common_field_equality" and issue.get("field") == "bathymetry"
        for issue in summary["issues"]
    )


def test_audit_fails_on_timestamp_coverage_gap(tmp_path: Path) -> None:
    config = _build_fixture(
        tmp_path,
        scenarios=[
            {
                "sample_index": 1,
                "bathymetry_type": "island",
                "source_type": "fault",
                "source_strength": 0.5,
                "bathymetry": np.asarray(
                    [[-2.0, -1.0], [-0.5, -0.25]],
                    dtype=np.float32,
                ),
                "source_field": np.asarray(
                    [[0.01, 0.02], [0.03, 0.04]],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.006], dtype=np.float32),
            }
        ],
        common_time_grid=[0.004, 0.008],
    )

    summary = _run_audit(config)

    assert summary["status"] == "fail"
    assert any(issue["check"] == "timestamp_coverage" for issue in summary["issues"])


def test_audit_fails_on_duplicate_and_order_mismatch(tmp_path: Path) -> None:
    def _mutate_rows(rows_by_solver: dict[str, list[dict[str, Any]]]) -> None:
        rows_by_solver["muscl_hr"] = list(reversed(rows_by_solver["muscl_hr"]))
        rows_by_solver["hydrostatic"][1]["scenario_id"] = rows_by_solver["hydrostatic"][
            0
        ]["scenario_id"]

    config = _build_fixture(
        tmp_path,
        scenarios=[
            {
                "sample_index": 1,
                "bathymetry_type": "canyon",
                "source_type": "dipole",
                "source_strength": 0.4,
                "bathymetry": np.asarray(
                    [[-2.0, -1.0], [-0.5, -0.25]],
                    dtype=np.float32,
                ),
                "source_field": np.asarray(
                    [[0.01, 0.02], [0.03, 0.04]],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
            },
            {
                "sample_index": 2,
                "bathymetry_type": "trench",
                "source_type": "rough",
                "source_strength": 0.7,
                "bathymetry": np.asarray(
                    [[-3.0, -1.5], [-0.75, -0.5]],
                    dtype=np.float32,
                ),
                "source_field": np.asarray(
                    [[-0.01, 0.02], [0.01, -0.03]],
                    dtype=np.float32,
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
            },
        ],
        rows_mutator=_mutate_rows,
    )

    summary = _run_audit(config)

    assert summary["status"] == "fail"
    checks = {issue["check"] for issue in summary["issues"]}
    assert "duplicate_scenario_ids" in checks
    assert "ordered_mapping_mismatch" in checks


def test_audit_reconstruction_control_passes_for_sharded_processed_dataset(
    tmp_path: Path,
) -> None:
    scenarios = [
        {
            "sample_index": 1,
            "bathymetry_type": "canyon",
            "source_type": "fault",
            "source_strength": 0.45,
            "bathymetry": np.asarray(
                [[-3.0, -1.0], [-0.5, -0.25]],
                dtype=np.float32,
            ),
            "source_field": np.asarray(
                [[0.01, 0.02], [0.03, 0.04]],
                dtype=np.float32,
            ),
            "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
        },
        {
            "sample_index": 2,
            "bathymetry_type": "island",
            "source_type": "gaussian",
            "source_strength": 0.72,
            "bathymetry": np.asarray(
                [[-2.0, -1.5], [-0.75, -0.5]],
                dtype=np.float32,
            ),
            "source_field": np.asarray(
                [[-0.01, 0.02], [0.01, -0.03]],
                dtype=np.float32,
            ),
            "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
        },
    ]
    config = _build_fixture(tmp_path, scenarios=scenarios)
    processed_root = Path(config["audit"]["processed_test_roots"]["hydrostatic"])
    stats_path = _write_processed_hydrostatic_dataset(
        processed_root,
        scenarios=scenarios,
        input_order=["bathymetry", "source", "initial_depth"],
        input_stats={
            "bathymetry": (-2.5, 1.5),
            "source": (0.01, 0.05),
        },
    )
    config["audit"]["reconstruction_control"].update(
        {
            "processed_dataset_path": str(processed_root),
            "normalization_stats_path": str(stats_path),
        }
    )

    summary = _run_audit(config)

    assert summary["status"] == "pass"
    reconstruction = summary["reconstruction_control"]
    assert reconstruction["status"] == "pass"
    assert reconstruction["checked_scenario_count"] == len(scenarios)
    assert reconstruction["checked_channel_count"] == len(scenarios) * 3
    assert reconstruction["mismatch_count"] == 0
    assert reconstruction["max_abs_diff"] == pytest.approx(0.0)
    assert reconstruction["provenance"]["normalization_stats_path"] == str(stats_path)
    assert reconstruction["input_order"] == ["bathymetry", "source", "initial_depth"]


def test_audit_reconstruction_control_fails_on_processed_input_mismatch(
    tmp_path: Path,
) -> None:
    scenarios = [
        {
            "sample_index": 1,
            "bathymetry_type": "trench",
            "source_type": "rough",
            "source_strength": 0.61,
            "bathymetry": np.asarray(
                [[-2.5, -1.0], [-0.5, -0.25]],
                dtype=np.float32,
            ),
            "source_field": np.asarray(
                [[0.02, -0.01], [0.01, 0.03]],
                dtype=np.float32,
            ),
            "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
        }
    ]
    config = _build_fixture(tmp_path, scenarios=scenarios)
    processed_root = Path(config["audit"]["processed_test_roots"]["hydrostatic"])
    bad_inputs = {
        "scenario_000001": np.asarray(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.001]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        )
    }
    stats_path = _write_processed_hydrostatic_dataset(
        processed_root,
        scenarios=scenarios,
        input_order=["bathymetry", "source", "initial_depth"],
        input_stats={
            "bathymetry": (-2.5, 1.5),
            "source": (0.01, 0.05),
        },
        input_overrides=bad_inputs,
    )
    config["audit"]["reconstruction_control"].update(
        {
            "processed_dataset_path": str(processed_root),
            "normalization_stats_path": str(stats_path),
            "float32_atol": 1.0e-6,
        }
    )

    summary = _run_audit(config)

    assert summary["status"] == "fail"
    reconstruction = summary["reconstruction_control"]
    assert reconstruction["status"] == "fail"
    assert reconstruction["mismatch_count"] >= 1
    assert reconstruction["max_abs_diff"] > 1.0e-6
    assert any(
        issue["check"] == "reconstruction_control" for issue in summary["issues"]
    )


def test_selection_is_deterministic_and_balanced_for_primary_policy() -> None:
    audit_artifact = _selection_audit_artifact(eligible_per_cell=5)

    first = SELECTION_MODULE.select_common_time_validation_scenarios(
        audit_artifact,
        policy_name="primary",
        seed=20260711,
    )
    second = SELECTION_MODULE.select_common_time_validation_scenarios(
        audit_artifact,
        policy_name="primary",
        seed=20260711,
    )

    assert first == second
    assert first["dense_validation"]["count"] == 120
    assert first["smoke"]["count"] == 12
    assert all(cell["selected_count"] == 4 for cell in first["family_cells"])


def test_selection_fails_when_a_family_cell_is_undersized() -> None:
    audit_artifact = _selection_audit_artifact(eligible_per_cell=4)
    audit_artifact["eligible_scenarios"] = [
        row
        for row in audit_artifact["eligible_scenarios"]
        if not (
            row["bathymetry_type"] == "canyon"
            and row["source_type"] == "dipole"
            and row["scenario_id"].endswith("000004")
        )
    ]

    with pytest.raises(ValueError, match="cannot be satisfied"):
        SELECTION_MODULE.select_common_time_validation_scenarios(
            audit_artifact,
            policy_name="primary",
            seed=20260711,
        )


def test_selection_labels_reduced_policy_explicitly() -> None:
    audit_artifact = _selection_audit_artifact(eligible_per_cell=2)

    selected = SELECTION_MODULE.select_common_time_validation_scenarios(
        audit_artifact,
        policy_name="reduced",
        seed=20260711,
    )

    assert selected["selection_policy"]["label"] == "reduced"
    assert selected["dense_validation"]["count"] == 60
    assert all(cell["selected_count"] == 2 for cell in selected["family_cells"])
