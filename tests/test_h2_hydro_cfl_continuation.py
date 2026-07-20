from __future__ import annotations

from pathlib import Path

import pytest

from src.data_gen.common_time_v2 import stable_hash_payload
from src.evaluation.h2_hydro_cfl_continuation import (
    SCHEMA_ID,
    _load_config,
    _screening,
    _validate_task_result,
)


def test_continuation_config_freezes_three_worst_hydro_cases() -> None:
    config = _load_config(
        Path("configs/eval/h2_hydro_cfl_continuation_v1.yaml")
    )
    assert config["solver"] == "swe_hydrostatic"
    assert config["cfl_pair"] == {
        "candidate_production": 0.1125,
        "candidate_reference": 0.05625,
    }
    assert len(config["cases"]) == 3
    assert config["selection_is_outcome_targeted"] is True


def test_continuation_screening_preserves_frozen_max_thresholds() -> None:
    rows = [
        {
            "quarter_to_eighth_metrics": {
                "trajectory_relative_l2": trajectory,
                "per_time_normalized_rmse": [per_time] * 50,
            }
        }
        for trajectory, per_time in (
            (0.05, 0.25),
            (0.08, 0.20),
            (0.10, 0.18),
        )
    ]
    gates = _screening(
        rows,
        {
            "trajectory_relative_l2_max": 0.30,
            "per_time_normalized_rmse_max": 0.50,
        },
    )
    assert all(gate["passed"] for gate in gates)
    rows[0]["quarter_to_eighth_metrics"]["per_time_normalized_rmse"][0] = 0.51
    gates = _screening(
        rows,
        {
            "trajectory_relative_l2_max": 0.30,
            "per_time_normalized_rmse_max": 0.50,
        },
    )
    assert not next(
        gate
        for gate in gates
        if gate["gate"] == "per_time_normalized_rmse_max"
    )["passed"]


def test_continuation_task_identity_and_hash_fail_closed() -> None:
    task = {
        "task_id": "task",
        "ordinal": 0,
        "qualified_id": "train:scenario_009741",
        "input_fingerprint": "fingerprint",
        "selection_role": "worst",
        "candidate_production_cfl": 0.1125,
        "candidate_reference_cfl": 0.05625,
        "source_diagnostic_task_id": "source",
    }
    payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-hydro-cfl-continuation-task-result",
        "study_hash": "study",
        **task,
        "passed_health_and_replay": True,
        "failed_checks": [],
    }
    payload["result_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-h2-hydro-cfl-continuation-task-result",
        payload=payload,
        schema_id=SCHEMA_ID,
    )
    _validate_task_result(payload, task=task, study_hash="study")
    payload["candidate_reference_cfl"] = 0.1
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_task_result(payload, task=task, study_hash="study")
