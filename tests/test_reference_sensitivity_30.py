import numpy as np
import pytest

from scripts.run_reference_sensitivity_30 import paired_summary


def members():
    # Each model has thirty scenarios and three seeds with different offsets.
    rows = []
    for model, shift in (("fno", 0.1), ("ufno", 0.2)):
        for seed in (18, 36, 67):
            baseline = np.arange(1, 31, dtype=float) / 100 + shift + seed / 1000
            rows.append({"model": model, "seed": seed, "scores": {
                ref: {"relative_l2_by_case": (baseline + (0.02 if ref == "fine" else 0)).tolist()}
                for ref in ("archived", "coarse", "baseline", "fine")}})
    return rows


def test_paired_summary_averages_seeds_before_scenario_bootstrap():
    result = paired_summary(members(), bootstrap_samples=100)
    fno = result["models"]["fno"]
    assert fno["mean_paired_absolute_change"] == pytest.approx(0.02)
    assert fno["descriptive_bootstrap_absolute_change_95"] == pytest.approx([0.02, 0.02])
    assert len(fno["paired_absolute_changes_by_case"]) == 30
    assert result["seed_averaged_pairwise_reversals"][0]["fraction_reversed"] == 0


def test_paired_summary_rejects_incomplete_training_seed_roster():
    with pytest.raises(ValueError, match="three fixed seeds"):
        paired_summary(members()[1:], bootstrap_samples=100)
