import numpy as np
import pytest
import torch

from scripts.eval_trajectory_calibration import (
    _calibration_quantile,
    _summarize,
    _trajectory_scores,
)


def test_simultaneous_score_detects_single_failed_cell():
    mean = torch.zeros((2, 3, 2, 2))
    target = mean.clone()
    target[0, 2, 1, 1] = 5
    target[1] = 1
    scores = _trajectory_scores(mean, torch.ones_like(mean), target, 1e-6)
    np.testing.assert_array_equal(scores["score"], [5, 1])
    assert scores["rmse"][0] < 5


def test_quantile_uses_finite_sample_order_statistic():
    assert _calibration_quantile(np.arange(1, 11), 0.8) == 9
    with pytest.raises(ValueError, match="Too few"):
        _calibration_quantile(np.array([1, 2]), 0.95)


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([-1.0])])
def test_quantile_rejects_invalid_calibration(values):
    with pytest.raises(ValueError):
        _calibration_quantile(values, 0.9)


def test_coverage_counts_trajectories_not_pixels():
    rows = [{"score": s, "rel_l2": 0.2, "mean_std_physical": 0.01} for s in (1, 2, 20)]
    result = _summarize(rows, {"0.9": 2})
    assert result["simultaneous_coverage"]["0.9"] == 2 / 3
    assert result["mean_band_half_width"]["0.9"] == 0.02


def test_zero_variance_floor_and_nonfinite_rejection():
    mean = torch.zeros((1, 2, 1, 1))
    scores = _trajectory_scores(mean, mean, torch.ones_like(mean), 0.5)
    assert scores["score"].item() == 2
    with pytest.raises(ValueError, match="Non-finite"):
        _trajectory_scores(mean, mean, mean + float("nan"), 0.5)


def test_relative_l2_restores_nonzero_target_offset():
    mean = torch.zeros((1, 2, 1, 1))
    target = torch.ones_like(mean)
    # A normalized error of one with eta = 2*y + 4 gives physical rel-L2 2/6.
    scores = _trajectory_scores(mean, torch.ones_like(mean), target, 1e-6, 2.0)
    assert scores["rel_l2"].item() == pytest.approx(1 / 3)
    assert scores["score"].item() == 1


def test_target_norm_recovery_requires_same_inference_dataset_bytes():
    from scripts.summarize_trajectory_calibration import _validate_dataset_binding

    _validate_dataset_binding(
        {"/restored/meta.jsonl": "abc"}, {"/original/meta.jsonl": "abc"}
    )
    with pytest.raises(ValueError, match="dataset bytes"):
        _validate_dataset_binding(
            {"/restored/meta.jsonl": "new"}, {"/original/meta.jsonl": "abc"}
        )
