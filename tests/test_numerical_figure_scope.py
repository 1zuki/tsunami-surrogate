import json
import sys

import numpy as np
import pytest

from scripts import plot_benchmark_numerical_study as plot


def test_manuscript_figure_rejects_partial_study_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "partial.json"
    source.write_text(
        json.dumps(
            {
                "complete_full_planned_study": False,
                "rollout_count": 12,
                "full_scope_rollout_count": 92,
                "rollouts": {},
            }
        )
    )
    output = tmp_path / "figure.pdf"
    monkeypatch.setattr(
        sys, "argv", ["plot", "--input", str(source), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="complete planned study"):
        plot.main()
    assert not output.exists()


def test_comparison_selection_requires_distinct_case_coverage():
    row = {
        "category": "spatial_self_difference",
        "case_id": "one",
        "left": "one__swe_hydrostatic__coarse",
        "reference": "one__swe_hydrostatic__baseline",
    }
    with pytest.raises(ValueError, match="Incomplete comparison roster"):
        plot._selected(
            {"comparisons": [row, row]},
            "spatial_self_difference",
            "swe_hydrostatic",
            "coarse",
            "baseline",
            2,
        )


def test_case_range_is_descriptive_and_rejects_nonfinite_values():
    assert plot._range([1, 3, 2]) == {
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "count": 3,
    }
    with pytest.raises(ValueError, match="Non-finite"):
        plot._range([1, np.nan])
