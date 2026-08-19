from __future__ import annotations

import pytest

from scripts.eval_v2_seed_metrics import _validate_training_seeds


def _rows(*seeds: int) -> list[dict[str, int]]:
    return [{"training_seed": seed} for seed in seeds]


def test_training_seed_validation_accepts_expected_unique_order() -> None:
    assert _validate_training_seeds(_rows(18, 36, 67), [18, 36, 67]) == [
        18,
        36,
        67,
    ]


def test_training_seed_validation_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="not unique"):
        _validate_training_seeds(_rows(18, 36, 36), [18, 36, 67])


def test_training_seed_validation_rejects_wrong_order() -> None:
    with pytest.raises(ValueError, match="requested order"):
        _validate_training_seeds(_rows(18, 67, 36), [18, 36, 67])
