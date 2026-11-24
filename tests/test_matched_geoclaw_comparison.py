import numpy as np
import pytest

from scripts import summarize_matched_geoclaw_comparison as comparison


def test_metrics_use_named_reference_norm_and_reject_invalid_fields():
    left = np.full(comparison.SHAPE, 2.0)
    right = np.full(comparison.SHAPE, 4.0)
    assert comparison._metrics(left, right) == pytest.approx((2.0, 0.5))
    with pytest.raises(ValueError, match="zero norm"):
        comparison._metrics(left, np.zeros_like(right))
    right[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite paired"):
        comparison._metrics(left, right)


def test_comparison_covers_both_grids_and_retains_worsening_cases():
    scenarios, cases, entries, fields = {}, {192: {}, 384: {}}, {}, {192: {}, 384: {}}
    for index in comparison.CASE_INDICES:
        scenario, case = f"scenario_{index}", f"case_{index}"
        scenarios[index] = {"row": {"scenario_id": scenario}}
        for grid in comparison.GEO_GRIDS:
            cases[grid][index] = {"case_id": case}
            fields[grid][case] = np.full(comparison.SHAPE, grid / 192 + 2)
        for solver in comparison.SOLVERS:
            for variant in comparison.CORE_GRIDS:
                value = 1.0 if variant == "baseline" else (2.0 if solver == "swe_hydrostatic" else 0.0)
                entries[f"{scenario}__{solver}__{variant}"] = (np.full(comparison.SHAPE, value), {})
    rows, changes = comparison._compare_cases(scenarios, cases, entries, fields)
    assert len(rows) == 24
    assert len({(r["case_id"], r["inhouse_solver"], r["inhouse_variant"], r["geoclaw_computational_cells"]) for r in rows}) == 24
    selected = {r["solver"]: r for r in changes if r["contrast"] == "inhouse_refinement_at_geoclaw_384"}
    assert selected["swe_hydrostatic"]["case_count_decreased_rmse"] == 3
    assert selected["swe_muscl_hr"]["case_count_decreased_rmse"] == 0
    assert selected["swe_muscl_hr"]["median_rmse_delta"] == 1.0


def test_unfinished_numerical_study_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be finalized"):
        comparison._study(tmp_path, {})


def test_refined_inputs_must_repeat_parent_cell_averages():
    baseline = {key: np.ones((192, 192)) for key in comparison.FIELDS}
    parent = {key: value.copy() for key, value in baseline.items()}
    fine = {key: np.ones((384, 384)) for key in comparison.FIELDS}
    fine["bathymetry"][0, 0] = 2
    with pytest.raises(ValueError, match="differs from the in-house refinement"):
        comparison._verify_arrays(baseline, parent, fine, np.array([420]), {}, {})
