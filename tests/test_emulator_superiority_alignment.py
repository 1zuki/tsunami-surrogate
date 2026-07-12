import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.aligned_comparison import (
    MODE_COMMON_TIME,
    MODE_SAVED_INDEX_LEGACY,
    align_positive_time_series,
    build_processed_input_lookup,
    build_emulator_superiority_metric_row,
    compare_solver_scenarios,
    evaluate_emulator_superiority_metric_rows,
    evaluate_emulator_superiority_scenarios,
    prediction_positive_timestamps,
    resolve_suite_contract,
    validate_common_time_solver_comparison_artifact,
    verify_reconstructed_input_match,
)
from src.evaluation.alignment import SCHEMA_ID, stable_hash_scenario_ids


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _alignment_cfg() -> dict:
    return {
        "mode": MODE_COMMON_TIME,
        "field": "trajectory_eta",
        "elevation_semantics": "sea_level_offset_relative_surface_elevation",
        "time_semantics": "solver_benchmark_time",
        "initial_frame_treatment": "require_saved_zero_frame_but_exclude_zero_from_common_grid",
        "aggregation": {"global_metric": "equal_scenario_weight_field_rmse"},
        "common_time_grid": {
            "endpoint_tolerance": 1.0e-6,
            "values": [0.004, 0.008],
        },
    }


def _contract(tmp_path: Path, ordered_ids: tuple[str, ...]) -> Any:
    audit_hash = "audit-hash"
    rows = [
        {
            "scenario_id": scenario_id,
            "bathymetry_type": f"bathy_{index}",
            "source_type": f"source_{index}",
            "source_strength": float(index + 1),
        }
        for index, scenario_id in enumerate(ordered_ids)
    ]
    audit = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "paired-reference-audit",
        "status": "pass",
        "audit_hash": audit_hash,
        "alignment": {
            "mode": MODE_COMMON_TIME,
            "field": "trajectory_eta",
            "elevation_semantics": "sea_level_offset_relative_surface_elevation",
            "time_semantics": "solver_benchmark_time",
            "initial_frame_treatment": "require_saved_zero_frame_but_exclude_zero_from_common_grid",
            "aggregation": {"global_metric": "equal_scenario_weight_field_rmse"},
            "common_time_grid": [0.004, 0.008],
        },
        "scenario_order": {
            "ordered_scenario_ids": list(ordered_ids),
            "ordered_scenario_hash": stable_hash_scenario_ids(list(ordered_ids)),
        },
        "eligible_scenarios": rows,
    }
    selection = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-validation-scenarios",
        "audit_hash": audit_hash,
        "dense_validation": {
            "label": "dense_reference_validation",
            "ordered_scenarios": rows,
            "ordered_scenario_ids": list(ordered_ids),
            "list_hash": stable_hash_scenario_ids(list(ordered_ids)),
        },
        "smoke": {
            "label": "implementation_only_smoke",
            "ordered_scenarios": rows[:1],
            "ordered_scenario_ids": [ordered_ids[0]],
            "list_hash": stable_hash_scenario_ids([ordered_ids[0]]),
        },
    }
    audit_path = tmp_path / "audit.json"
    selection_path = tmp_path / "selection.json"
    _write_json(audit_path, audit)
    _write_json(selection_path, selection)
    return resolve_suite_contract(
        alignment_cfg=_alignment_cfg(),
        audit_artifact_path=audit_path,
        scenario_selection_path=selection_path,
        suite_name="dense_validation",
        dense_validation_decision_path=None,
        require_full_suite_dense_decision=False,
        dense_fallback_policy="unsupported",
    )


def _write_processed_dataset(
    path: Path,
    *,
    inputs: np.ndarray,
    scenario_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=np.zeros((len(scenario_ids), 1, 1, 1), dtype=np.float32),
        input_order=np.asarray(
            ["bathymetry", "source", "initial_depth"], dtype=np.str_
        ),
        scenario_id=np.asarray(scenario_ids, dtype=np.str_),
    )


def test_prediction_timestamps_follow_reference_a_positive_times() -> None:
    pred_ts = prediction_positive_timestamps(
        np.asarray([0.0, 0.004, 0.009], dtype=np.float64),
        expected_output_channels=2,
    )
    np.testing.assert_allclose(pred_ts, np.asarray([0.004, 0.009], dtype=np.float64))

    pred_aligned = align_positive_time_series(
        np.asarray([[[1.0]], [[2.0]]], dtype=np.float64),
        pred_ts,
        common_time_grid=np.asarray([0.004, 0.008], dtype=np.float64),
        endpoint_tolerance=1.0e-6,
    )
    np.testing.assert_allclose(
        pred_aligned[:, 0, 0],
        np.asarray([1.0, 1.8], dtype=np.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_verify_reconstructed_input_match_passes_and_fails(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed" / "eval_dataset.npz"
    expected = np.asarray([[[[2.0]], [[3.0]], [[4.0]]]], dtype=np.float32)
    _write_processed_dataset(
        dataset_path,
        inputs=expected,
        scenario_ids=["scenario_000001"],
    )
    lookup = build_processed_input_lookup(dataset_path)

    passed = verify_reconstructed_input_match(
        scenario_id="scenario_000001",
        reconstructed_input=expected[0],
        lookup=lookup,
        atol=1.0e-6,
    )
    assert passed["max_abs_diff"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="reconstruction mismatch"):
        verify_reconstructed_input_match(
            scenario_id="scenario_000001",
            reconstructed_input=np.asarray(
                [[[2.0]], [[3.0]], [[4.1]]],
                dtype=np.float32,
            ),
            lookup=lookup,
            atol=1.0e-6,
        )


def test_emulator_superiority_ratio_and_bootstrap_are_deterministic(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, ("scenario_000001", "scenario_000002"))
    scenario_rows = [
        {
            "scenario_id": "scenario_000001",
            "bathymetry_type": "bathy_0",
            "source_type": "source_0",
            "source_strength": 1.0,
            "pred_aligned": np.asarray([[[2.0]], [[2.0]]], dtype=np.float64),
            "ref_a_aligned": np.asarray([[[1.0]], [[1.0]]], dtype=np.float64),
            "ref_b_aligned": np.asarray([[[3.0]], [[3.0]]], dtype=np.float64),
        },
        {
            "scenario_id": "scenario_000002",
            "bathymetry_type": "bathy_1",
            "source_type": "source_1",
            "source_strength": 2.0,
            "pred_aligned": np.asarray([[[4.0]], [[4.0]]], dtype=np.float64),
            "ref_a_aligned": np.asarray([[[3.0]], [[3.0]]], dtype=np.float64),
            "ref_b_aligned": np.asarray([[[5.0]], [[5.0]]], dtype=np.float64),
        },
    ]

    first = evaluate_emulator_superiority_scenarios(
        contract=contract,
        direction_name="toy_direction",
        model_solver_name="solver_a",
        benchmark_solver_name="solver_b",
        scenario_rows=scenario_rows,
        bootstrap_seed=7,
        num_resamples=16,
        confidence_level=0.95,
        git_commit="test",
    )
    second = evaluate_emulator_superiority_scenarios(
        contract=contract,
        direction_name="toy_direction",
        model_solver_name="solver_a",
        benchmark_solver_name="solver_b",
        scenario_rows=scenario_rows,
        bootstrap_seed=7,
        num_resamples=16,
        confidence_level=0.95,
        git_commit="test",
    )

    assert first["metrics"]["numerator_global_field_rmse"] == pytest.approx(1.0)
    assert first["metrics"]["denominator_global_field_rmse"] == pytest.approx(2.0)
    assert first["metrics"][
        "same_reference_control_global_field_rmse"
    ] == pytest.approx(1.0)
    assert first["metrics"]["rho"] == pytest.approx(0.5)
    assert first["benchmark_specific_superiority"]["classification"] == (
        "supported_benchmark_specific_superiority"
    )
    assert first["bootstrap"] == second["bootstrap"]
    assert (
        first["benchmark_specific_superiority"]
        == second["benchmark_specific_superiority"]
    )


def test_precomputed_metric_reduction_matches_tensor_helper(tmp_path: Path) -> None:
    contract = _contract(tmp_path, ("scenario_000001", "scenario_000002"))
    tensor_rows = [
        {
            "scenario_id": "scenario_000001",
            "bathymetry_type": "bathy_0",
            "source_type": "source_0",
            "source_strength": 1.0,
            "pred_aligned": np.asarray([[[2.0]], [[2.0]]], dtype=np.float64),
            "ref_a_aligned": np.asarray([[[1.0]], [[1.0]]], dtype=np.float64),
            "ref_b_aligned": np.asarray([[[3.0]], [[3.0]]], dtype=np.float64),
        },
        {
            "scenario_id": "scenario_000002",
            "bathymetry_type": "bathy_1",
            "source_type": "source_1",
            "source_strength": 2.0,
            "pred_aligned": np.asarray([[[4.0]], [[4.0]]], dtype=np.float64),
            "ref_a_aligned": np.asarray([[[3.0]], [[3.0]]], dtype=np.float64),
            "ref_b_aligned": np.asarray([[[5.0]], [[5.0]]], dtype=np.float64),
        },
    ]
    scalar_rows = [
        build_emulator_superiority_metric_row(
            scenario_id=row["scenario_id"],
            bathymetry_type=row["bathymetry_type"],
            source_type=row["source_type"],
            source_strength=row["source_strength"],
            pred_aligned=row["pred_aligned"],
            ref_a_aligned=row["ref_a_aligned"],
            ref_b_aligned=row["ref_b_aligned"],
        )
        for row in tensor_rows
    ]
    assert all("pred_aligned" not in row for row in scalar_rows)
    assert all("ref_a_aligned" not in row for row in scalar_rows)
    assert all("ref_b_aligned" not in row for row in scalar_rows)

    tensor_summary = evaluate_emulator_superiority_scenarios(
        contract=contract,
        direction_name="toy_direction",
        model_solver_name="solver_a",
        benchmark_solver_name="solver_b",
        scenario_rows=tensor_rows,
        bootstrap_seed=7,
        num_resamples=16,
        confidence_level=0.95,
        git_commit="test",
        script_path="tensor-helper",
    )
    scalar_summary = evaluate_emulator_superiority_metric_rows(
        contract=contract,
        direction_name="toy_direction",
        model_solver_name="solver_a",
        benchmark_solver_name="solver_b",
        scenario_metric_rows=scalar_rows,
        bootstrap_seed=7,
        num_resamples=16,
        confidence_level=0.95,
        git_commit="test",
        script_path="tensor-helper",
    )

    tensor_summary["created_at_utc"] = "normalized"
    scalar_summary["created_at_utc"] = "normalized"
    assert scalar_summary == tensor_summary


def test_emulator_superiority_flags_zero_denominator(tmp_path: Path) -> None:
    contract = _contract(tmp_path, ("scenario_000001",))
    scenario_rows = [
        {
            "scenario_id": "scenario_000001",
            "bathymetry_type": "bathy_0",
            "source_type": "source_0",
            "source_strength": 1.0,
            "pred_aligned": np.asarray([[[1.0]], [[1.0]]], dtype=np.float64),
            "ref_a_aligned": np.asarray([[[2.0]], [[2.0]]], dtype=np.float64),
            "ref_b_aligned": np.asarray([[[2.0]], [[2.0]]], dtype=np.float64),
        }
    ]

    summary = evaluate_emulator_superiority_scenarios(
        contract=contract,
        direction_name="toy_direction",
        model_solver_name="solver_a",
        benchmark_solver_name="solver_b",
        scenario_rows=scenario_rows,
        bootstrap_seed=7,
        num_resamples=8,
        confidence_level=0.95,
    )

    assert summary["metrics"]["denominator_global_field_rmse"] == pytest.approx(0.0)
    assert math.isinf(summary["metrics"]["rho"])
    assert summary["benchmark_specific_superiority"]["classification"] == (
        "invalid_zero_denominator"
    )


def test_legacy_solver_comparison_artifact_is_rejected_by_common_time_ratio(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, ("scenario_000001",))
    legacy_summary = compare_solver_scenarios(
        contract=contract,
        solver_a_name="solver_a",
        solver_b_name="solver_b",
        paired_scenarios=[
            {
                "scenario_id": "scenario_000001",
                "bathymetry_type": "bathy_0",
                "source_type": "source_0",
                "source_strength": 1.0,
                "left": {
                    "trajectory_eta": np.asarray(
                        [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float64
                    ),
                    "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float64),
                },
                "right": {
                    "trajectory_eta": np.asarray(
                        [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float64
                    ),
                    "timestamps": np.asarray([0.0, 0.004, 0.008], dtype=np.float64),
                },
            }
        ],
        mode=MODE_SAVED_INDEX_LEGACY,
        bootstrap_seed=1,
        num_resamples=4,
        confidence_level=0.95,
        initial_frame_policy="include",
    )

    with pytest.raises(
        ValueError, match="Legacy saved-index solver comparison artifacts"
    ):
        validate_common_time_solver_comparison_artifact(
            legacy_summary, contract=contract
        )
