from __future__ import annotations

import numpy as np

from scripts.run_geoclaw_discrepancy_ablation import (
    _aggregate,
    _health_summary,
    _nested_refine_2x,
    _nested_restrict_2x,
    _validate_variant_specs,
    _variant_specs,
)


def test_health_summary_reports_actual_minimum_depth() -> None:
    diagnostics = {
        "post_step_cfl": np.asarray([0.05, 0.1]),
        "requested_timestamps": np.asarray([0.1, 0.2]),
        "total_natural_steps": np.asarray([2]),
        "swe_min_depth": np.asarray([0.8, 0.6]),
        "operator_nan_to_num_replacement_count": np.zeros(2),
        "operator_positivity_projection_count": np.zeros(2),
        "operator_dry_projection_count": np.zeros(2),
    }
    summary = _health_summary(
        trajectory=np.zeros((2, 2, 2)),
        diagnostics=diagnostics,
        target_cfl=0.1,
        min_depth_tolerance=-1.0e-6,
        solver_name="swe_hydrostatic",
    )
    assert summary["passed"]
    assert summary["minimum_depth"] == 0.6


def test_health_summary_rejects_nonfinite_counter_and_negative_depth() -> None:
    diagnostics = {
        "post_step_cfl": np.asarray([0.1]),
        "requested_timestamps": np.asarray([0.1]),
        "total_natural_steps": np.asarray([1]),
        "swe_min_depth": np.asarray([-2.0e-6]),
        "operator_nan_to_num_replacement_count": np.asarray([np.nan]),
        "operator_positivity_projection_count": np.zeros(1),
        "operator_dry_projection_count": np.zeros(1),
    }
    summary = _health_summary(
        trajectory=np.zeros((1, 2, 2)),
        diagnostics=diagnostics,
        target_cfl=0.1,
        min_depth_tolerance=-1.0e-6,
        solver_name="swe_hydrostatic",
    )
    assert not summary["passed"]
    assert "operator_nan_to_num_replacement_count_nonfinite" in summary["violations"]
    assert any(value.startswith("minimum_depth=") for value in summary["violations"])


def _case() -> dict:
    return {
        "case_id": "production_input_canaries_train_scenario_000002",
        "inhouse_domain": {
            "shape": [96, 96],
            "bounds": [-0.25, 1.25, -0.25, 1.25],
            "dx": 1.0 / 64.0,
            "output_crop": [16, 80, 16, 80],
            "sponge": {
                "enabled": True,
                "width": 16,
                "min_factor": 0.8,
                "axes": "xy",
                "profile": "cosine",
                "time_mode": "elapsed_time_consistent",
                "reference_dt": 0.0035,
            },
        },
        "external_domain": {
            "shape": [192, 192],
            "bounds": [-1.0, 2.0, -1.0, 2.0],
            "dx": 1.0 / 64.0,
            "output_crop": [64, 128, 64, 128],
        },
    }


def test_nested_refinement_restriction_round_trip() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    refined = _nested_refine_2x(values)
    restricted = _nested_restrict_2x(refined[None, ...])[0]
    np.testing.assert_array_equal(restricted, values)


def test_variant_protocol_separates_extent_from_resolution() -> None:
    variants = _variant_specs(_case())
    _validate_variant_specs(variants)
    by_id = {row["variant_id"]: row for row in variants}
    coarse = by_id["production_96_open_no_sponge"]
    extended = by_id["extended_192_open_no_sponge"]
    refined = by_id["refined_192_open_no_sponge"]
    assert extended["dx"] == coarse["dx"]
    assert extended["bounds"] != coarse["bounds"]
    assert refined["bounds"] == coarse["bounds"]
    assert refined["dx"] == 0.5 * coarse["dx"]
    assert refined["output_restriction"] == 2


def test_aggregate_reports_signed_controlled_contrasts() -> None:
    variants = _variant_specs(_case())
    cases = []
    for index, scale in enumerate((1.0, 1.1, 0.9)):
        rows = []
        for solver, solver_factor in (
            ("swe_hydrostatic", 1.0),
            ("swe_muscl_hr", 0.5),
        ):
            for variant_index, variant in enumerate(variants):
                value = scale * solver_factor * (1.0 - 0.05 * variant_index)
                rows.append(
                    {
                        "solver": solver,
                        "variant_id": variant["variant_id"],
                        "target_cfl": 0.1,
                        "metrics": {
                            key: value
                            for key in (
                                "trajectory_relative_l2",
                                "absolute_rms",
                                "per_time_scaled_l2_p95_active",
                                "field_norm_ratio",
                                "field_cosine_similarity",
                                "shape_relative_l2_after_scale",
                                "boundary_band_relative_l2",
                                "interior_relative_l2",
                            )
                        },
                        "health": {
                            "natural_steps": 10,
                            "max_post_step_cfl": 0.1,
                            "minimum_depth": 1.0,
                        },
                    }
                )
        cases.append(
            {
                "case_id": f"case-{index}",
                "rows": rows,
            }
        )
    summary = _aggregate(cases, variants)
    sponge = next(
        row
        for row in summary["controlled_contrasts"]
        if row["solver"] == "swe_hydrostatic"
        and row["isolated_dimension"] == "sponge"
    )
    assert sponge["delta_after_minus_before"]["mean"] < 0.0
    assert sponge["closer_to_geoclaw_count"] == 3
    assert all(
        row["muscl_closer_count"] == 3
        for row in summary["solver_formulation_contrasts"]
    )
