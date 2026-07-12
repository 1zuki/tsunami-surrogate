from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.alignment import (
    MODE_COMMON_TIME,
    MODE_SAVED_INDEX_LEGACY,
    SCHEMA_ID,
    align_elevation_series,
    compute_equal_scenario_global_rmse,
    generate_paired_bootstrap_indices,
    stable_hash_scenario_ids,
    summarize_paired_bootstrap,
    validate_alignment_compatibility,
    validate_common_time_grid,
    validate_timestamps,
)


def test_common_time_alignment_is_exact_for_affine_time_series() -> None:
    timestamps = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    slope = np.asarray([[0.5, -1.0], [1.5, 2.0]], dtype=np.float64)
    elevation = np.stack([base + t * slope for t in timestamps], axis=0)
    common_time_grid = np.asarray([0.25, 0.75], dtype=np.float64)

    aligned = align_elevation_series(
        elevation,
        timestamps,
        mode=MODE_COMMON_TIME,
        common_time_grid=common_time_grid,
    )

    expected = np.stack([base + t * slope for t in common_time_grid], axis=0)
    np.testing.assert_allclose(aligned, expected, atol=1.0e-12, rtol=0.0)


def test_common_time_alignment_accepts_narrow_endpoint_tolerance_without_extrapolation() -> (
    None
):
    timestamps = np.asarray([0.0, 0.3999996], dtype=np.float64)
    elevation = np.asarray(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=np.float64,
    )

    aligned = align_elevation_series(
        elevation,
        timestamps,
        mode=MODE_COMMON_TIME,
        common_time_grid=np.asarray([0.4], dtype=np.float64),
        endpoint_tolerance=1.0e-6,
    )

    np.testing.assert_allclose(aligned[0], elevation[-1], atol=0.0, rtol=0.0)


def test_common_time_alignment_rejects_extrapolation_beyond_tolerance() -> None:
    with pytest.raises(ValueError, match="without extrapolation"):
        align_elevation_series(
            np.asarray([[[1.0]], [[2.0]]], dtype=np.float64),
            np.asarray([0.0, 0.399], dtype=np.float64),
            mode=MODE_COMMON_TIME,
            common_time_grid=np.asarray([0.4], dtype=np.float64),
            endpoint_tolerance=1.0e-6,
        )


def test_validate_timestamps_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        validate_timestamps(np.asarray([[0.0, 0.5]], dtype=np.float64))


def test_validate_common_time_grid_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        validate_common_time_grid(np.asarray([[0.004, 0.008]], dtype=np.float64))


@pytest.mark.parametrize(
    ("elevation", "timestamps", "match"),
    [
        (
            np.asarray([[[1.0]], [[2.0]]], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            "strictly increasing",
        ),
        (
            np.asarray([[[1.0]], [[2.0]]], dtype=np.float64),
            np.asarray([0.0, np.nan], dtype=np.float64),
            "finite",
        ),
        (
            np.asarray([[[1.0]], [[np.inf]]], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
            "finite",
        ),
        (
            np.asarray([[[1.0]], [[2.0]]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            "must match",
        ),
    ],
)
def test_alignment_validation_fails_loudly(
    elevation: np.ndarray,
    timestamps: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        align_elevation_series(
            elevation,
            timestamps,
            mode=MODE_COMMON_TIME,
            common_time_grid=np.asarray([0.5], dtype=np.float64),
        )


def test_saved_index_legacy_mode_requires_explicit_indices() -> None:
    timestamps = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    elevation = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float64)

    aligned = align_elevation_series(
        elevation,
        timestamps,
        mode=MODE_SAVED_INDEX_LEGACY,
        frame_indices=np.asarray([0, 2], dtype=np.int64),
    )

    np.testing.assert_allclose(
        aligned[:, 0, 0], np.asarray([1.0, 3.0], dtype=np.float64)
    )


def test_saved_index_legacy_mode_rejects_2d_frame_indices() -> None:
    timestamps = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    elevation = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float64)

    with pytest.raises(ValueError, match="1-D"):
        align_elevation_series(
            elevation,
            timestamps,
            mode=MODE_SAVED_INDEX_LEGACY,
            frame_indices=np.asarray([[0, 2]], dtype=np.int64),
        )


def test_alignment_compatibility_checks_exact_metadata_contract() -> None:
    base_metadata = {
        "schema_id": SCHEMA_ID,
        "mode": MODE_COMMON_TIME,
        "ordered_scenario_ids": ["scenario_000001", "scenario_000002"],
        "common_time_grid": [0.004, 0.008],
        "field": "trajectory_eta",
        "elevation_semantics": "sea_level_offset_relative_surface_elevation",
        "time_semantics": "solver_benchmark_time",
        "initial_frame_treatment": "exclude_zero_from_common_grid",
        "aggregation": {"global_metric": "equal_scenario_weight_field_rmse"},
    }
    validate_alignment_compatibility(base_metadata, dict(base_metadata))

    changed = dict(base_metadata)
    changed["ordered_scenario_ids"] = ["scenario_000002", "scenario_000001"]
    with pytest.raises(ValueError, match="ordered_scenario_ids"):
        validate_alignment_compatibility(base_metadata, changed)


def test_paired_bootstrap_is_deterministic() -> None:
    indices_a = generate_paired_bootstrap_indices(
        num_scenarios=4,
        num_resamples=8,
        seed=123,
    )
    indices_b = generate_paired_bootstrap_indices(
        num_scenarios=4,
        num_resamples=8,
        seed=123,
    )
    assert np.array_equal(indices_a, indices_b)

    summary_a = summarize_paired_bootstrap(
        {
            "rmse": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "mae": np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float64),
        },
        bootstrap_indices=indices_a,
    )
    summary_b = summarize_paired_bootstrap(
        {
            "rmse": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "mae": np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float64),
        },
        bootstrap_indices=indices_b,
    )

    assert summary_a == summary_b


def test_stable_scenario_hash_depends_on_order() -> None:
    left = stable_hash_scenario_ids(["scenario_000001", "scenario_000002"])
    right = stable_hash_scenario_ids(["scenario_000001", "scenario_000002"])
    changed = stable_hash_scenario_ids(["scenario_000002", "scenario_000001"])

    assert left == right
    assert left != changed


def test_equal_scenario_global_rmse_uses_equal_scenario_weight() -> None:
    rmse = compute_equal_scenario_global_rmse(
        [
            {"scenario_id": "scenario_000001", "mse": 1.0},
            {"scenario_id": "scenario_000002", "mse": 9.0},
        ]
    )

    assert math.isclose(rmse, math.sqrt(5.0))
