from __future__ import annotations

from src.evaluation.buffered_production_health import _row_passes, _task_identity


def test_health_task_identity_is_deterministic() -> None:
    record = {
        "qualified_id": "train:scenario_000001",
        "input_fingerprint": "abc123",
    }
    first = _task_identity(record, "swe_hydrostatic", 0)
    second = _task_identity(dict(reversed(list(record.items()))), "swe_hydrostatic", 0)
    assert first == second


def test_health_gate_covers_exact_96_to_64_contract() -> None:
    payload = {
        "trajectory_finite": True,
        "trajectory_shape": [50, 64, 64],
        "row": {
            "source_edge_max_abs": 0.0,
            "sponge_core_min": 1.0,
            "health": {
                "finite": True,
                "requested_times_exact": True,
                "measurement_dtype": "float64",
                "cg_failure_count": 0,
            },
        },
    }
    assert _row_passes(payload)
    payload["trajectory_shape"] = [49, 64, 64]
    assert not _row_passes(payload)
