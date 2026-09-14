from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.corrected_production_geoclaw import _load_config, _requested_times


def test_corrected_production_geoclaw_config_uses_the_r6_time_construction() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _load_config(
        root / "configs/eval/corrected_production_geoclaw_validation.yaml"
    )
    times = _requested_times(config["corrected_production"]["requested_times"])

    assert times.shape == (50,)
    np.testing.assert_array_equal(
        times,
        8.4 + 8.4 * np.arange(50, dtype=np.float64),
    )
    assert times[-1] == 420.0
    assert config["selection"]["outcome_blind"] is True
    assert config["corrected_production"]["requested_time_abs_tolerance"] == 1.0e-12
    assert config["corrected_production"]["publication_mapping"] == {
        "crop_shape": [128, 128],
        "block_mean_reduction": [2, 2],
    }
    assert [case["sample_index"] for case in config["selection"]["cases"]] == [
        1,
        34,
        4,
    ]
