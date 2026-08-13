from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import buffered_crop_benchmark
from src.evaluation.common_time_v2_h1 import (
    SCHEMA_ID,
    _build_tasks,
    _find_replay_mismatches,
    _summarize_diagnostics,
    _task_identity,
    _validate_task_result,
    _validate_config,
    select_h1_scenarios,
    validate_frozen_checksums,
)
from src.utils.config import load_config


BATHYMETRY_FAMILIES = ("canyon", "continental", "island", "seamounts", "trench")
SOURCE_FAMILIES = (
    "dipole",
    "fault",
    "gaussian",
    "multi-gauss",
    "okada-like",
    "rough",
)


def _record(index: int, bathymetry: str, source: str) -> dict[str, object]:
    scenario_id = f"scenario_{index:06d}"
    return {
        "split": "train",
        "sample_index": index,
        "scenario_id": scenario_id,
        "qualified_id": f"train:{scenario_id}",
        "input_fingerprint": f"{index:064x}",
        "bathymetry_type": bathymetry,
        "source_type": source,
    }


def _selection_config(expected_count: int) -> dict[str, object]:
    return {
        "split": "train",
        "expected_split_count": expected_count,
        "count_per_cell": 1,
        "selection_seed": "selection-seed",
        "replay_selection_seed": "replay-seed",
        "bathymetry_families": list(BATHYMETRY_FAMILIES),
        "source_families": list(SOURCE_FAMILIES),
    }


def _balanced_rows(per_cell: int = 2) -> list[dict[str, object]]:
    rows = []
    index = 1
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            for _ in range(per_cell):
                rows.append(_record(index, bathymetry, source))
                index += 1
    return rows


def _valid_details(solver: str) -> tuple[np.ndarray, dict[str, object]]:
    requested = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    steps = requested.size
    diagnostics: dict[str, np.ndarray] = {
        "requested_timestamps": requested.copy(),
        "left_natural_timestamps": requested.copy(),
        "right_natural_timestamps": requested.copy(),
        "interpolation_weights": np.zeros(steps, dtype=np.float64),
        "bracket_widths": np.zeros(steps, dtype=np.float64),
        "exact_knot": np.ones(steps, dtype=np.bool_),
        "natural_step_indices": np.arange(1, steps + 1, dtype=np.int64),
        "total_natural_steps": np.asarray([steps], dtype=np.int64),
        "natural_dt_history": np.full(steps, 0.1, dtype=np.float64),
        "final_natural_timestamp": np.asarray([0.3], dtype=np.float64),
        "natural_health_step_indices": np.arange(1, steps + 1, dtype=np.int32),
        "left_natural_step_times": np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
        "right_natural_step_times": requested.copy(),
        "proposed_dt": np.full(steps, 0.1, dtype=np.float64),
        "pre_step_cfl": np.full(steps, 0.4, dtype=np.float64),
        "post_step_cfl": np.full(steps, 0.41, dtype=np.float64),
        "finite_state_flag": np.ones(steps, dtype=np.bool_),
    }
    if solver in {"swe_hydrostatic", "swe_muscl_hr"}:
        diagnostics.update(
            {
                "swe_min_depth": np.ones(steps, dtype=np.float64),
                "swe_max_speed": np.full(steps, 0.2, dtype=np.float64),
                "swe_dry_cell_count": np.zeros(steps, dtype=np.int32),
            }
        )
    else:
        diagnostics.update(
            {
                "cg_step_converged": np.ones(steps, dtype=np.bool_),
                "cg_failed_count": np.zeros(steps, dtype=np.int32),
                "cg_max_iterations": np.full(steps, 4, dtype=np.int32),
                "cg_max_residual_ratio": np.full(steps, 1.0e-11),
                "cg_solve0_converged": np.ones(steps, dtype=np.bool_),
                "cg_solve0_iterations": np.full(steps, 4, dtype=np.int32),
                "cg_solve0_initial_residual": np.ones(steps),
                "cg_solve0_final_residual": np.full(steps, 1.0e-11),
                "cg_solve0_residual_ratio": np.full(steps, 1.0e-11),
                "cg_solve1_converged": np.ones(steps, dtype=np.bool_),
                "cg_solve1_iterations": np.full(steps, 4, dtype=np.int32),
                "cg_solve1_initial_residual": np.ones(steps),
                "cg_solve1_final_residual": np.full(steps, 1.0e-11),
                "cg_solve1_residual_ratio": np.full(steps, 1.0e-11),
                "filter_enabled": np.zeros(steps, dtype=np.bool_),
                "filter_application_count": np.zeros(steps, dtype=np.int32),
            }
        )
    details = {
        "natural_dt_history": np.full(steps, 0.1, dtype=np.float64),
        "diagnostics": diagnostics,
    }
    return requested, details


def test_balanced_selection_is_order_independent_and_training_only() -> None:
    rows = _balanced_rows()
    config = _selection_config(len(rows))
    first, first_summary = select_h1_scenarios(
        rows,
        selection_config=config,
        inventory_sha256="inventory",
    )
    second, second_summary = select_h1_scenarios(
        [dict(reversed(list(row.items()))) for row in reversed(rows)],
        selection_config=config,
        inventory_sha256="inventory",
    )
    assert first == second
    assert first_summary == second_summary
    assert len(first) == 30
    assert len(
        {
            (entry["bathymetry_type"], entry["source_type"])
            for entry in first
        }
    ) == 30
    assert all(entry["record"]["split"] == "train" for entry in first)


def test_balanced_selection_fails_on_duplicate_or_missing_inventory() -> None:
    rows = _balanced_rows()
    config = _selection_config(len(rows))
    duplicate = deepcopy(rows)
    duplicate[1]["qualified_id"] = duplicate[0]["qualified_id"]
    with pytest.raises(RuntimeError, match="duplicate"):
        select_h1_scenarios(
            duplicate,
            selection_config=config,
            inventory_sha256="inventory",
        )
    with pytest.raises(RuntimeError, match="expected"):
        select_h1_scenarios(
            rows[:-1],
            selection_config=config,
            inventory_sha256="inventory",
        )


def test_task_plan_has_90_primary_and_three_replay_tasks() -> None:
    rows = _balanced_rows(per_cell=1)
    selected, summary = select_h1_scenarios(
        rows,
        selection_config=_selection_config(len(rows)),
        inventory_sha256="inventory",
    )
    tasks = _build_tasks(
        selected,
        solvers=("swe_hydrostatic", "swe_muscl_hr", "boussinesq"),
        replay_selection_ordinal=summary["replay_selection_ordinal"],
        candidate_config_hash="candidate",
    )
    assert len(tasks) == 93
    assert sum(task["run_kind"] == "primary" for task in tasks) == 90
    assert sum(task["run_kind"] == "replay" for task in tasks) == 3
    assert [task["ordinal"] for task in tasks] == list(range(93))
    assert len({task["task_id"] for task in tasks}) == 93
    assert all(
        task["reference_primary_task_id"]
        for task in tasks
        if task["run_kind"] == "replay"
    )


def test_h1_allows_fresh_content_addressed_prerequisite_identities() -> None:
    config = load_config("configs/eval/common_time_v2_h1.yaml")
    config["prerequisites"].update(
        {
            "h0_contract_hash": "a" * 64,
            "level_a_contract_hash": "b" * 64,
            "level_b_bundle_hash": "c" * 64,
        }
    )
    _validate_config(config)

    config["prerequisites"]["h0_contract_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_config(config)


@pytest.mark.parametrize(
    ("solver", "target_cfl"),
    (
        ("swe_hydrostatic", 0.45),
        ("swe_muscl_hr", 0.45),
        ("boussinesq", 0.35),
    ),
)
def test_requested_provenance_and_dense_health_pass_closed_contract(
    solver: str, target_cfl: float
) -> None:
    requested, details = _valid_details(solver)
    if solver == "boussinesq":
        details["diagnostics"]["pre_step_cfl"][:] = 0.3
    checks, provenance, natural_health, evidence = _summarize_diagnostics(
        solver=solver,
        details=details,
        expected_times=requested,
        target_cfl=target_cfl,
    )
    assert all(checks.values())
    assert provenance["requested_timestamps"] == requested.tolist()
    assert len(natural_health["natural_dt_history"]) == 3
    assert evidence["summary"]["total_natural_steps"] == 3


def test_requested_provenance_rejects_extrapolation_and_incomplete_cfl() -> None:
    requested, details = _valid_details("swe_hydrostatic")
    details["diagnostics"]["left_natural_timestamps"][1] = 0.21
    details["diagnostics"]["post_step_cfl"] = np.asarray([0.4, 0.4])
    checks, _provenance, _natural_health, _evidence = _summarize_diagnostics(
        solver="swe_hydrostatic",
        details=details,
        expected_times=requested,
        target_cfl=0.45,
    )
    assert not checks["brackets_contain_requests"]
    assert not checks["natural_history_complete"]


def test_replay_comparison_is_exact_and_identity_checked() -> None:
    primary = {
        "task_id": "primary",
        "run_kind": "primary",
        "reference_primary_task_id": None,
        "qualified_id": "train:scenario_000001",
        "solver": "swe_hydrostatic",
        "scientific_digest": "same",
    }
    replay = {
        "task_id": "replay",
        "run_kind": "replay",
        "reference_primary_task_id": "primary",
        "qualified_id": "train:scenario_000001",
        "solver": "swe_hydrostatic",
        "scientific_digest": "same",
    }
    assert _find_replay_mismatches([primary, replay]) == []
    replay["scientific_digest"] = "different"
    assert len(_find_replay_mismatches([primary, replay])) == 1
    replay["solver"] = "swe_muscl_hr"
    with pytest.raises(RuntimeError, match="identity"):
        _find_replay_mismatches([primary, replay])


def test_task_result_hash_detects_corruption() -> None:
    selection = {
        "selection_ordinal": 0,
        "record": _record(1, "canyon", "dipole"),
    }
    task = _task_identity(
        ordinal=0,
        run_kind="primary",
        solver="swe_hydrostatic",
        selection=selection,
        candidate_config_hash="candidate",
    )
    payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h1-task-result",
        "contract_hash": "contract",
        **task,
        "passed": True,
        "failed_checks": [],
    }
    identity = dict(payload)
    from src.data_gen.common_time_v2 import stable_hash_payload

    payload["result_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-h1-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    _validate_task_result(payload, task=task, contract_hash="contract")
    payload["passed"] = False
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _validate_task_result(payload, task=task, contract_hash="contract")


def test_existing_buffered_runner_api_discards_only_detailed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_row = {"health": "ok"}
    expected_trajectory = np.ones((1, 2, 2), dtype=np.float64)

    def fake_detailed(*args: object, **kwargs: object) -> tuple[object, ...]:
        return expected_row, expected_trajectory, {"diagnostics": "private"}

    monkeypatch.setattr(
        buffered_crop_benchmark, "run_buffered_case_detailed", fake_detailed
    )
    row, trajectory = buffered_crop_benchmark.run_buffered_case(
        {}, solver_name="swe_hydrostatic", total_grid=96
    )
    assert row is expected_row
    assert trajectory is expected_trajectory


def test_frozen_checksum_manifest_ignores_resumable_execution_tree(
    tmp_path: Path,
) -> None:
    for name in (
        "preregistered_contract.json",
        "selected_scenarios.json",
        "task_plan.json",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    # File checksums hash bytes directly; build the manifest the same way.
    import hashlib

    rows = [
        f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}"
        for name in (
            "preregistered_contract.json",
            "selected_scenarios.json",
            "task_plan.json",
        )
    ]
    (tmp_path / "CONTRACT_SHA256SUMS.txt").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "execution" / "tasks").mkdir(parents=True)
    (tmp_path / "execution" / "tasks" / "partial.json").write_text(
        json.dumps({"partial": True}), encoding="utf-8"
    )
    validate_frozen_checksums(tmp_path)
    (tmp_path / "task_plan.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_frozen_checksums(tmp_path)
