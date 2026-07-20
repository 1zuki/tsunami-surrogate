from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from src.data_gen.common_time_v2 import stable_hash_payload
from src.evaluation.h2_swe_cfl_diagnostic import (
    SCHEMA_ID,
    _build_tasks,
    _load_config,
    _numeric_mismatches,
    _validate_task_result,
    observed_order,
    validate_frozen_checksums,
)


def _metric_payload(value: float) -> dict[str, object]:
    return {
        "measurement_dtype": "float64",
        "trajectory_absolute_rms": value,
        "trajectory_relative_l2": value,
        "per_time_absolute_rms": [value] * 50,
        "per_time_normalized_rmse": [value] * 50,
        "phase_applicable": True,
        "phase_correlation_loss": value,
    }


def _synthetic_source(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    selected = []
    rows = []
    task_index = 0
    for index, case in enumerate(config["cases"]):
        record = {
            "qualified_id": case["qualified_id"],
            "input_fingerprint": case["input_fingerprint"],
            "bathymetry_type": case["bathymetry_type"],
            "source_type": case["source_type"],
            "split": "train",
            "scenario_id": str(case["qualified_id"]).split(":", 1)[1],
            "sample_index": index,
        }
        selected.append({"selection_ordinal": index, "record": record})
        for solver in config["solvers"]:
            rows.append(
                {
                    "task_id": f"source-{task_index}",
                    "run_kind": "primary",
                    "qualified_id": case["qualified_id"],
                    "solver": solver,
                    "scientific_digest": f"digest-{task_index}",
                    "metrics": _metric_payload(0.01 + index * 0.001),
                }
            )
            task_index += 1
    return (
        {
            "candidate_config_hash": "candidate",
            "selected_scenarios": selected,
        },
        {"task_rows": rows},
    )


def test_config_freezes_eight_cases_and_geometric_cfl_ladders() -> None:
    config = _load_config(Path("configs/eval/h2_swe_cfl_refinement_v1.yaml"))
    assert len(config["cases"]) == 8
    assert config["selection_is_outcome_targeted"] is True
    assert config["validation_or_test_outcomes_inspected"] is False
    for solver in config["solvers"]:
        assert config["cfl_ladder"][solver] == [0.45, 0.225, 0.1125]


def test_task_plan_is_deterministic_and_covers_each_case_solver_pair() -> None:
    config = _load_config(Path("configs/eval/h2_swe_cfl_refinement_v1.yaml"))
    source_contract, source_result = _synthetic_source(config)
    first = _build_tasks(config, source_contract, source_result)
    second = _build_tasks(
        deepcopy(config), deepcopy(source_contract), deepcopy(source_result)
    )
    assert first == second
    assert len(first) == 16
    assert [task["ordinal"] for task in first] == list(range(16))
    assert len({task["task_id"] for task in first}) == 16
    assert {
        (task["qualified_id"], task["solver"]) for task in first
    } == {
        (case["qualified_id"], solver)
        for case in config["cases"]
        for solver in config["solvers"]
    }


def test_task_plan_fails_closed_on_source_identity_mismatch() -> None:
    config = _load_config(Path("configs/eval/h2_swe_cfl_refinement_v1.yaml"))
    source_contract, source_result = _synthetic_source(config)
    source_contract["selected_scenarios"][0]["record"][
        "input_fingerprint"
    ] = "changed"
    with pytest.raises(RuntimeError, match="disagrees with source H2"):
        _build_tasks(config, source_contract, source_result)


def test_observed_order_and_metric_replay_tolerance() -> None:
    assert observed_order(4.0, 2.0) == pytest.approx(1.0)
    assert observed_order(4.0, 1.0) == pytest.approx(2.0)
    assert observed_order(0.0, 0.0) is None
    frozen = {
        "scalar": 0.1,
        "nested": {"values": [0.2, 0.3], "flag": True},
    }
    changed_order = {
        "nested": {"flag": True, "values": [0.2 + 1.0e-16, 0.3]},
        "scalar": 0.1,
    }
    assert (
        _numeric_mismatches(
            changed_order, frozen, rtol=2.0e-15, atol=1.0e-18
        )
        == []
    )
    changed_order["nested"]["values"][0] = 0.21
    assert _numeric_mismatches(
        changed_order, frozen, rtol=2.0e-15, atol=1.0e-18
    ) == ["metrics.nested.values[0]"]


def test_task_result_hash_and_identity_fail_closed() -> None:
    task = {
        "task_id": "task",
        "ordinal": 0,
        "solver": "swe_hydrostatic",
        "qualified_id": "train:scenario_000001",
        "input_fingerprint": "fingerprint",
        "bathymetry_type": "island",
        "source_type": "rough",
        "selection_role": "worst",
        "cfl_ladder": [0.45, 0.225, 0.1125],
        "source_h2_task_id": "source",
    }
    payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-swe-cfl-diagnostic-task-result",
        "study_hash": "study",
        **task,
        "passed_health_and_replay": True,
        "failed_checks": [],
    }
    payload["result_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-h2-swe-cfl-diagnostic-task-result",
        payload=payload,
        schema_id=SCHEMA_ID,
    )
    _validate_task_result(payload, task=task, study_hash="study")
    payload["qualified_id"] = "train:scenario_changed"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _validate_task_result(payload, task=task, study_hash="study")


def test_frozen_checksums_ignore_resumable_execution(tmp_path: Path) -> None:
    names = ("diagnostic_contract.json", "task_plan.json")
    for name in names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    manifest = "\n".join(
        f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    )
    (tmp_path / "CONTRACT_SHA256SUMS.txt").write_text(
        manifest + "\n", encoding="utf-8"
    )
    (tmp_path / "execution" / "tasks").mkdir(parents=True)
    (tmp_path / "execution" / "tasks" / "partial.json").write_text(
        "{}\n", encoding="utf-8"
    )
    validate_frozen_checksums(tmp_path)
    (tmp_path / "task_plan.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_frozen_checksums(tmp_path)
