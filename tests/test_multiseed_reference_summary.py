from __future__ import annotations

import pytest

from scripts.summarize_v2_multiseed_reference import summarize_payloads


def _payload(seed: int, rho: float) -> dict:
    summary = {"global_field_rmse": 0.2}
    return {
        "evaluation_type": "v2_cross_reference_discrepancy",
        "training_seed": seed,
        "num_samples": 10,
        "dataset_paths": {"hydrostatic": "test-a", "muscl_hr": "test-b"},
        "common_time_v2": {"frame_count": 50},
        "directions": [
            {
                "model_solver": "hydrostatic",
                "benchmark_solver": "muscl_hr",
                "rho": {
                    "point_estimate": rho,
                    "ci_lower": rho - 0.1,
                    "ci_upper": rho + 0.1,
                },
                "numerator": {"global_field_rmse": 0.3 + rho},
                "same_reference_control": {"global_field_rmse": 0.1 + rho},
                "denominator_solver_gap": summary,
            }
        ],
    }


def test_multiseed_reference_summary_preserves_seed_and_scenario_uncertainty() -> None:
    result = summarize_payloads(
        [_payload(18, 0.9), _payload(36, 1.0), _payload(67, 1.1)],
        ["seed18.json", "seed36.json", "seed67.json"],
    )

    assert result["training_seeds"] == [18, 36, 67]
    assert result["seed_count"] == 3
    row = result["directions"][0]
    assert row["rho"]["seed_mean"] == pytest.approx(1.0)
    assert row["rho"]["seed_sample_std"] == pytest.approx(0.1)
    assert [item["seed"] for item in row["rho"]["by_seed"]] == [18, 36, 67]
    assert row["denominator_solver_gap_global_field_rmse"][
        "max_abs_seed_difference"
    ] == pytest.approx(0.0)
