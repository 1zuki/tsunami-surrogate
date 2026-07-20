from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.data_gen.common_time_v2 import stable_hash_payload
from src.evaluation.common_time_v2_h2 import (
    SCHEMA_ID,
    _begin_execution_timing,
    _build_tasks,
    _checkpoint_execution_timing,
    _find_replay_mismatches,
    _load_config,
    _paired_stratified_bootstrap,
    _solver_summary_and_gates,
    _task_identity,
    _validate_task_result,
    _expected_thresholds,
    paired_cfl_metrics,
    select_h2_scenarios,
    validate_frozen_checksums,
)


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


def _inventory(
    per_cell: int = 6,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    h1_selected: list[dict[str, object]] = []
    index = 1
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            cell: list[dict[str, object]] = []
            for _ in range(per_cell):
                record = _record(index, bathymetry, source)
                rows.append(record)
                cell.append(record)
                index += 1
            h1_selected.append({"record": deepcopy(cell[0])})
    return rows, h1_selected


def _selection_config(expected_count: int) -> dict[str, object]:
    return {
        "split": "train",
        "expected_split_count": expected_count,
        "count_per_cell": 4,
        "selection_seed": "selection-seed",
        "replay_selection_seed": "replay-seed",
        "exclude_h1_contract_hash": "h1",
        "expected_h1_exclusion_count": 30,
        "bathymetry_families": list(BATHYMETRY_FAMILIES),
        "source_families": list(SOURCE_FAMILIES),
    }


def test_h2_selection_is_balanced_order_independent_and_excludes_h1() -> None:
    rows, h1_selected = _inventory()
    config = _selection_config(len(rows))
    first, first_summary = select_h2_scenarios(
        rows,
        selection_config=config,
        inventory_sha256="inventory",
        h1_selected=h1_selected,
    )
    second, second_summary = select_h2_scenarios(
        [dict(reversed(list(row.items()))) for row in reversed(rows)],
        selection_config=config,
        inventory_sha256="inventory",
        h1_selected=list(reversed(h1_selected)),
    )
    assert first == second
    assert first_summary == second_summary
    assert len(first) == 120
    h1_ids = {entry["record"]["qualified_id"] for entry in h1_selected}
    assert not h1_ids & {
        entry["record"]["qualified_id"] for entry in first
    }
    counts: dict[tuple[str, str], int] = {}
    for entry in first:
        cell = (entry["bathymetry_type"], entry["source_type"])
        counts[cell] = counts.get(cell, 0) + 1
    assert set(counts.values()) == {4}
    assert len(counts) == 30


def test_h2_selection_fails_closed_on_bad_h1_exclusion() -> None:
    rows, h1_selected = _inventory()
    config = _selection_config(len(rows))
    duplicate = deepcopy(h1_selected)
    duplicate[1] = deepcopy(duplicate[0])
    with pytest.raises(RuntimeError, match="30 unique"):
        select_h2_scenarios(
            rows,
            selection_config=config,
            inventory_sha256="inventory",
            h1_selected=duplicate,
        )
    mismatched = deepcopy(h1_selected)
    mismatched[0]["record"]["input_fingerprint"] = "wrong"
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        select_h2_scenarios(
            rows,
            selection_config=config,
            inventory_sha256="inventory",
            h1_selected=mismatched,
        )


def test_h2_v2_selection_excludes_prior_viewed_h2() -> None:
    rows, h1_selected = _inventory(per_cell=10)
    prior_h2_selected: list[dict[str, object]] = []
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = {}
    h1_ids = {
        str(entry["record"]["qualified_id"]) for entry in h1_selected
    }
    for row in rows:
        if row["qualified_id"] in h1_ids:
            continue
        cell = (str(row["bathymetry_type"]), str(row["source_type"]))
        by_cell.setdefault(cell, []).append(row)
    for cell in sorted(by_cell):
        for record in by_cell[cell][:4]:
            prior_h2_selected.append({"record": deepcopy(record)})
    config = _selection_config(len(rows))
    config.update(
        {
            "selection_seed": "selection-v2",
            "replay_selection_seed": "replay-v2",
            "exclude_prior_h2_contract_hash": "prior-h2",
            "expected_prior_h2_exclusion_count": 120,
        }
    )
    selected, summary = select_h2_scenarios(
        rows,
        selection_config=config,
        inventory_sha256="inventory",
        h1_selected=h1_selected,
        prior_h2_selected=prior_h2_selected,
    )
    selected_ids = {
        str(entry["record"]["qualified_id"]) for entry in selected
    }
    prior_ids = {
        str(entry["record"]["qualified_id"]) for entry in prior_h2_selected
    }
    assert len(selected) == 120
    assert not selected_ids & prior_ids
    assert summary["excluded_prior_h2_count"] == 120


def test_h2_task_plan_has_360_primary_and_three_replay_pairs() -> None:
    rows, h1_selected = _inventory()
    selected, summary = select_h2_scenarios(
        rows,
        selection_config=_selection_config(len(rows)),
        inventory_sha256="inventory",
        h1_selected=h1_selected,
    )
    comparison = {
        "production_cfl": {
            "swe_hydrostatic": 0.45,
            "swe_muscl_hr": 0.45,
            "boussinesq": 0.35,
        },
        "reference_cfl": {
            "swe_hydrostatic": 0.225,
            "swe_muscl_hr": 0.225,
            "boussinesq": 0.175,
        },
    }
    tasks = _build_tasks(
        selected,
        solvers=("swe_hydrostatic", "swe_muscl_hr", "boussinesq"),
        replay_selection_ordinal=summary["replay_selection_ordinal"],
        candidate_config_hash="candidate",
        comparison=comparison,
    )
    assert len(tasks) == 363
    assert sum(task["run_kind"] == "primary" for task in tasks) == 360
    assert sum(task["run_kind"] == "replay" for task in tasks) == 3
    assert [task["ordinal"] for task in tasks] == list(range(363))
    assert len({task["task_id"] for task in tasks}) == 363
    assert all(
        task["reference_cfl"] == 0.5 * task["production_cfl"]
        for task in tasks
    )


def test_paired_metrics_use_float64_trajectory_scale_and_phase_activity() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    wave = np.sin(x)[:, None] * np.cos(x)[None, :]
    reference = np.stack(
        [(1.0 + 0.1 * index) * wave for index in range(3)]
    ).astype(np.float64)
    identical = paired_cfl_metrics(
        reference.copy(),
        reference,
        relative_floor_absolute_rms=1.0e-12,
        phase_activity_floor_absolute_rms=1.0e-10,
        boundary_band_cells=2,
    )
    assert identical["trajectory_relative_l2"] == 0.0
    assert identical["peak_amplitude_relative_error"] == 0.0
    assert identical["phase_applicable"]
    assert identical["phase_correlation_loss"] == pytest.approx(0.0, abs=1.0e-15)

    shifted = np.roll(reference, shift=1, axis=1)
    changed = paired_cfl_metrics(
        shifted,
        reference,
        relative_floor_absolute_rms=1.0e-12,
        phase_activity_floor_absolute_rms=1.0e-10,
        boundary_band_cells=2,
    )
    assert changed["trajectory_relative_l2"] > 0.0
    assert changed["peak_amplitude_relative_error"] == pytest.approx(0.0)
    assert changed["phase_correlation_loss"] > 0.0


def test_paired_metrics_fail_closed_on_float32_and_handle_zero_signal() -> None:
    zeros = np.zeros((3, 8, 8), dtype=np.float64)
    metrics = paired_cfl_metrics(
        zeros,
        zeros,
        relative_floor_absolute_rms=1.0e-12,
        phase_activity_floor_absolute_rms=1.0e-10,
        boundary_band_cells=2,
    )
    assert metrics["relative_denominator_floor_used"]
    assert metrics["trajectory_relative_l2"] == 0.0
    assert not metrics["phase_applicable"]
    assert metrics["phase_correlation_loss"] is None
    with pytest.raises(ValueError, match="float64"):
        paired_cfl_metrics(
            zeros.astype(np.float32),
            zeros.astype(np.float32),
            relative_floor_absolute_rms=1.0e-12,
            phase_activity_floor_absolute_rms=1.0e-10,
            boundary_band_cells=2,
        )


def _bootstrap_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    value = 0.001
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            for _ in range(4):
                rows.append(
                    {
                        "bathymetry_type": bathymetry,
                        "source_type": source,
                        "metrics": {
                            "trajectory_relative_l2": value,
                            "peak_amplitude_relative_error": 2.0 * value,
                            "phase_correlation_loss": 0.5 * value,
                        },
                    }
                )
                value += 0.0001
    return rows


def _summary_rows(solver: str, value: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            for _ in range(4):
                health_metrics: dict[str, object] = {
                    "max_abs_eta": 0.01,
                    "max_eta_over_depth": 0.02,
                }
                operator: dict[str, object] = {"sponge_elapsed_time": 0.175}
                if solver == "boussinesq":
                    health_metrics.update(
                        {
                            "cg_failure_count": 0,
                            "cg_iterations_max": 10,
                            "cg_residual_ratio_max": 1.0e-12,
                        }
                    )
                    operator["filter_applications"] = 0
                else:
                    health_metrics.update(
                        {
                            "minimum_depth": 0.5,
                            "maximum_speed": 0.2,
                            "maximum_dry_cell_count": 0,
                        }
                    )
                variant = {
                    "passed": True,
                    "runtime_s": 1.0,
                    "diagnostic_summary": {
                        "total_natural_steps": 100,
                        "max_post_step_cfl": 0.4,
                    },
                    "health_metrics": health_metrics,
                    "operator_diagnostics": operator,
                }
                rows.append(
                    {
                        "qualified_id": f"train:scenario_{index:06d}",
                        "bathymetry_type": bathymetry,
                        "source_type": source,
                        "solver": solver,
                        "metrics": {
                            "trajectory_relative_l2": value,
                            "per_time_absolute_rms": [value] * 50,
                            "per_time_normalized_rmse": [value] * 50,
                            "peak_amplitude_relative_error": value,
                            "production_amplitude_by_time": [1.0] * 50,
                            "reference_amplitude_by_time": [1.0] * 50,
                            "phase_correlation_loss": value,
                            "interior_relative_l2": value,
                            "boundary_relative_l2": value,
                        },
                        "production": deepcopy(variant),
                        "reference": deepcopy(variant),
                    }
                )
                index += 1
    return rows


def test_paired_stratified_bootstrap_is_deterministic() -> None:
    bootstrap = {
        "seed": "bootstrap",
        "resamples": 50,
        "confidence_level": 0.95,
        "method": "paired_within_family_cell",
        "decision_role": "informational_uncertainty_not_gate",
    }
    first = _paired_stratified_bootstrap(
        _bootstrap_rows(),
        metric="trajectory_relative_l2",
        solver="swe_hydrostatic",
        bootstrap=bootstrap,
    )
    second = _paired_stratified_bootstrap(
        list(reversed(_bootstrap_rows())),
        metric="trajectory_relative_l2",
        solver="swe_hydrostatic",
        bootstrap=bootstrap,
    )
    assert first == second
    assert first["applicable"]
    assert len(first["median_ci"]) == 2


@pytest.mark.parametrize(
    ("solver", "value"),
    (
        ("swe_hydrostatic", 0.01),
        ("swe_muscl_hr", 0.005),
        ("boussinesq", 1.0e-4),
    ),
)
def test_h2_solver_aggregation_covers_cells_times_health_and_gates(
    solver: str, value: float
) -> None:
    metrics_config = {
        "minimum_phase_applicable_fraction": 0.95,
        "bootstrap": {
            "seed": "bootstrap",
            "resamples": 10,
            "confidence_level": 0.95,
            "method": "paired_within_family_cell",
            "decision_role": "informational_uncertainty_not_gate",
        },
    }
    summary, gates = _solver_summary_and_gates(
        solver,
        _summary_rows(solver, value),
        thresholds=_expected_thresholds()[solver],
        metrics_config=metrics_config,
    )
    assert summary["case_count"] == 120
    assert len(summary["family_cells"]) == 30
    assert len(summary["per_time"]) == 50
    assert summary["production_health"]["passed_count"] == 120
    assert all(gate["passed"] for gate in gates)


def test_h2_per_time_gate_uses_worst_requested_time_not_pooled_p95() -> None:
    rows = _summary_rows("swe_hydrostatic", 0.01)
    for row in rows:
        row["metrics"]["per_time_normalized_rmse"][17] = 0.26
    metrics_config = {
        "minimum_phase_applicable_fraction": 0.95,
        "bootstrap": {
            "seed": "bootstrap",
            "resamples": 10,
            "confidence_level": 0.95,
            "method": "paired_within_family_cell",
            "decision_role": "informational_uncertainty_not_gate",
        },
    }
    summary, gates = _solver_summary_and_gates(
        "swe_hydrostatic",
        rows,
        thresholds=_expected_thresholds()["swe_hydrostatic"],
        metrics_config=metrics_config,
    )
    gate = next(
        item
        for item in gates
        if item["gate"] == "per_time_normalized_rmse_p95"
    )
    assert summary["per_time_normalized_rmse"]["p95"] == pytest.approx(0.01)
    assert summary["worst_requested_time_normalized_rmse_p95"] == pytest.approx(
        0.26
    )
    assert not gate["passed"]


def test_h2_replay_and_task_hash_fail_closed() -> None:
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
        production_cfl=0.45,
        reference_cfl=0.225,
    )
    payload = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-v2-h2-paired-task-result",
        "contract_hash": "contract",
        **task,
        "passed_health": True,
        "failed_pair_checks": [],
    }
    identity = dict(payload)
    payload["result_hash"] = stable_hash_payload(
        artifact_kind="common-time-v2-h2-paired-task-result",
        payload=identity,
        schema_id=SCHEMA_ID,
    )
    _validate_task_result(payload, task=task, contract_hash="contract")
    payload["passed_health"] = False
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _validate_task_result(payload, task=task, contract_hash="contract")

    primary = {
        "task_id": "primary",
        "run_kind": "primary",
        "reference_primary_task_id": None,
        "qualified_id": "train:scenario_000001",
        "solver": "swe_hydrostatic",
        "production_cfl": 0.45,
        "reference_cfl": 0.225,
        "scientific_digest": "same",
    }
    replay = {
        **primary,
        "task_id": "replay",
        "run_kind": "replay",
        "reference_primary_task_id": "primary",
    }
    assert _find_replay_mismatches([primary, replay]) == []
    replay["scientific_digest"] = "different"
    assert len(_find_replay_mismatches([primary, replay])) == 1


def test_h2_config_and_frozen_checksum_contract() -> None:
    config = _load_config(Path("configs/eval/common_time_v2_h2.yaml"))
    assert config["selection"]["count_per_cell"] == 4
    assert config["comparison"]["reference_cfl_factor"] == 0.5
    assert config["threshold_basis"]["h2_outcomes_viewed"] is False
    v2 = _load_config(Path("configs/eval/common_time_v2_h2_v2.yaml"))
    assert v2["comparison"]["production_cfl"] == {
        "swe_hydrostatic": 0.1125,
        "swe_muscl_hr": 0.225,
        "boussinesq": 0.35,
    }
    assert v2["thresholds"] == config["thresholds"]
    assert v2["selection"]["expected_prior_h2_exclusion_count"] == 120


def test_h2_resume_timing_accumulates_successful_checkpoint_time(
    tmp_path: Path,
) -> None:
    first_base, first_invocation = _begin_execution_timing(tmp_path)
    assert first_base == 0.0
    assert first_invocation == 1
    first_total = _checkpoint_execution_timing(
        tmp_path,
        cumulative_before_invocation_s=first_base,
        invocation_count=first_invocation,
        invocation_elapsed_s=12.5,
        active=True,
    )
    assert first_total == 12.5
    second_base, second_invocation = _begin_execution_timing(tmp_path)
    assert second_base == 12.5
    assert second_invocation == 2
    second_total = _checkpoint_execution_timing(
        tmp_path,
        cumulative_before_invocation_s=second_base,
        invocation_count=second_invocation,
        invocation_elapsed_s=7.25,
        active=False,
    )
    assert second_total == 19.75


def test_h2_frozen_checksums_ignore_resumable_execution(
    tmp_path: Path,
) -> None:
    names = (
        "preregistered_contract.json",
        "selected_scenarios.json",
        "task_plan.json",
    )
    for name in names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    rows = [
        f"{hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (tmp_path / "CONTRACT_SHA256SUMS.txt").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "execution" / "tasks").mkdir(parents=True)
    (tmp_path / "execution" / "tasks" / "partial.json").write_text(
        "{}\n", encoding="utf-8"
    )
    validate_frozen_checksums(tmp_path)
    (tmp_path / "task_plan.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_frozen_checksums(tmp_path)
