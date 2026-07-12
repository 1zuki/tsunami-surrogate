import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.aligned_comparison import (
    MODE_COMMON_TIME,
    MODE_SAVED_INDEX_LEGACY,
    compare_solver_scenarios,
    iter_paired_raw_reference_samples,
    require_explicit_mode,
    resolve_suite_contract,
)
from src.evaluation.alignment import SCHEMA_ID, stable_hash_scenario_ids


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _alignment_cfg() -> dict:
    return {
        "mode": MODE_COMMON_TIME,
        "field": "trajectory_eta",
        "elevation_semantics": "sea_level_offset_relative_surface_elevation",
        "time_semantics": "solver_benchmark_time",
        "initial_frame_treatment": "require_saved_zero_frame_but_exclude_zero_from_common_grid",
        "aggregation": {
            "global_metric": "equal_scenario_weight_field_rmse",
            "bootstrap": {
                "seed": 123,
                "num_resamples": 16,
                "confidence_level": 0.95,
            },
        },
        "common_time_grid": {
            "endpoint_tolerance": 1.0e-6,
            "values": [0.004, 0.008],
        },
    }


def _audit_artifact(ordered_ids: tuple[str, ...], audit_hash: str) -> dict:
    alignment = _alignment_cfg()
    eligible = []
    for index, scenario_id in enumerate(ordered_ids):
        eligible.append(
            {
                "scenario_id": scenario_id,
                "bathymetry_type": f"bathy_{index % 2}",
                "source_type": f"source_{index % 2}",
                "source_strength": float(index + 1),
            }
        )
    return {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "paired-reference-audit",
        "status": "pass",
        "audit_hash": audit_hash,
        "alignment": {
            "mode": MODE_COMMON_TIME,
            "field": alignment["field"],
            "elevation_semantics": alignment["elevation_semantics"],
            "time_semantics": alignment["time_semantics"],
            "initial_frame_treatment": alignment["initial_frame_treatment"],
            "aggregation": alignment["aggregation"],
            "common_time_grid": alignment["common_time_grid"]["values"],
        },
        "scenario_order": {
            "ordered_scenario_ids": list(ordered_ids),
            "ordered_scenario_hash": stable_hash_scenario_ids(list(ordered_ids)),
        },
        "eligible_scenarios": eligible,
    }


def _selection_artifact(ordered_ids: tuple[str, ...], audit_hash: str) -> dict:
    rows = []
    for index, scenario_id in enumerate(ordered_ids):
        rows.append(
            {
                "scenario_id": scenario_id,
                "bathymetry_type": f"bathy_{index % 2}",
                "source_type": f"source_{index % 2}",
                "source_strength": float(index + 1),
            }
        )
    return {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-validation-scenarios",
        "audit_hash": audit_hash,
        "dense_validation": {
            "label": "dense_reference_validation",
            "ordered_scenarios": rows,
            "ordered_scenario_ids": list(ordered_ids),
            "list_hash": stable_hash_scenario_ids(list(ordered_ids)),
        },
        "smoke": {
            "label": "implementation_only_smoke",
            "ordered_scenarios": rows[:1],
            "ordered_scenario_ids": [ordered_ids[0]],
            "list_hash": stable_hash_scenario_ids([ordered_ids[0]]),
        },
    }


def _dense_validation_artifacts(
    root: Path,
    *,
    audit_hash: str,
    dense_ids: tuple[str, ...],
    status: str = "pass",
) -> Path:
    decision_path = root / "dense_validation" / "decision.json"
    summary_path = decision_path.with_name("summary.json")
    decision = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "dense-reference-validation-decision",
        "status": status,
        "suite": {"name": "dense_validation", "label": "dense_reference_validation"},
    }
    summary = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "dense-reference-validation",
        "status": status,
        "inputs": {"audit_hash": audit_hash},
        "scenario_order": {
            "ordered_scenario_ids": list(dense_ids),
            "ordered_scenario_hash": stable_hash_scenario_ids(list(dense_ids)),
        },
        "alignment": {"common_time_grid": [0.004, 0.008]},
    }
    _write_json(decision_path, decision)
    _write_json(summary_path, summary)
    return decision_path


def _contract(tmp_path: Path, *, suite: str = "dense_validation") -> tuple:
    ordered_ids = ("scenario_000001", "scenario_000002")
    audit_hash = "audit-hash"
    audit_path = tmp_path / "paired_reference_audit.json"
    selection_path = tmp_path / "common_time_validation_scenarios.json"
    _write_json(audit_path, _audit_artifact(ordered_ids, audit_hash))
    _write_json(selection_path, _selection_artifact(ordered_ids, audit_hash))
    decision_path = _dense_validation_artifacts(
        tmp_path / "dense_reference_validation",
        audit_hash=audit_hash,
        dense_ids=ordered_ids,
    )
    contract = resolve_suite_contract(
        alignment_cfg=_alignment_cfg(),
        audit_artifact_path=audit_path,
        scenario_selection_path=selection_path,
        suite_name=suite,
        dense_validation_decision_path=decision_path,
        require_full_suite_dense_decision=(suite == "full"),
        dense_fallback_policy="unsupported",
    )
    return contract, audit_path, selection_path, decision_path


def _paired_metric_rows() -> list[dict]:
    return [
        {
            "scenario_id": "scenario_000001",
            "bathymetry_type": "bathy_0",
            "source_type": "source_0",
            "source_strength": 1.0,
            "left": {
                "trajectory_eta": np.asarray(
                    [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float64
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.010], dtype=np.float64),
            },
            "right": {
                "trajectory_eta": np.asarray(
                    [[[0.0]], [[2.0]], [[2.0]]], dtype=np.float64
                ),
                "timestamps": np.asarray([0.0, 0.006, 0.010], dtype=np.float64),
            },
        },
        {
            "scenario_id": "scenario_000002",
            "bathymetry_type": "bathy_1",
            "source_type": "source_1",
            "source_strength": 2.0,
            "left": {
                "trajectory_eta": np.asarray(
                    [[[0.0]], [[1.0]], [[2.0]]], dtype=np.float64
                ),
                "timestamps": np.asarray([0.0, 0.004, 0.010], dtype=np.float64),
            },
            "right": {
                "trajectory_eta": np.asarray(
                    [[[0.0]], [[0.5]], [[1.5]]], dtype=np.float64
                ),
                "timestamps": np.asarray([0.0, 0.006, 0.010], dtype=np.float64),
            },
        },
    ]


def _write_raw_sample(
    root: Path,
    *,
    sample_index: int,
    scenario_id: str,
    trajectory_eta: np.ndarray,
    timestamps: np.ndarray | None,
) -> None:
    sample_dir = root / f"sample_{sample_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": np.asarray([scenario_id]),
        "trajectory_eta": np.asarray(trajectory_eta, dtype=np.float32),
        "bathymetry": np.zeros((1, 1), dtype=np.float32),
        "source_field": np.zeros((1, 1), dtype=np.float32),
        "initial_depth": np.ones((1, 1), dtype=np.float32),
        "eta0": np.zeros((1, 1), dtype=np.float32),
        "free_surface0": np.zeros((1, 1), dtype=np.float32),
    }
    if timestamps is not None:
        payload["timestamps"] = np.asarray(timestamps, dtype=np.float32)
    np.savez_compressed(sample_dir / "sample.npz", **payload)


def test_require_explicit_mode_rejects_missing_mode() -> None:
    with pytest.raises(ValueError, match="Explicit mode is required"):
        require_explicit_mode(None)


def test_common_time_solver_comparison_matches_hand_computed_metrics(
    tmp_path: Path,
) -> None:
    contract, _, _, _ = _contract(tmp_path)
    summary = compare_solver_scenarios(
        contract=contract,
        solver_a_name="solver_a",
        solver_b_name="solver_b",
        paired_scenarios=_paired_metric_rows(),
        mode=MODE_COMMON_TIME,
        bootstrap_seed=123,
        num_resamples=16,
        confidence_level=0.95,
        git_commit="test",
    )

    rows = summary["scenario_metrics"]
    assert rows[0]["rmse"] == pytest.approx(1.0 / 3.0)
    assert rows[1]["rmse"] == pytest.approx(2.0 / 3.0)
    assert summary["aggregate_metrics"]["global_field_rmse"] == pytest.approx(
        math.sqrt(5.0 / 18.0)
    )
    assert summary["aggregate_metrics"]["scenario_mae_mean"] == pytest.approx(0.5)
    assert summary["per_time_metrics"][0]["field_rmse"] == pytest.approx(
        math.sqrt(5.0 / 18.0)
    )
    assert summary["per_time_metrics"][1]["field_rmse"] == pytest.approx(
        math.sqrt(5.0 / 18.0)
    )


def test_common_time_solver_comparison_rejects_extrapolation_instead_of_truncating(
    tmp_path: Path,
) -> None:
    contract, _, _, _ = _contract(tmp_path)
    bad_rows = _paired_metric_rows()
    bad_rows[1]["right"]["timestamps"] = np.asarray(
        [0.0, 0.004, 0.007], dtype=np.float64
    )

    with pytest.raises(ValueError, match="without extrapolation"):
        compare_solver_scenarios(
            contract=contract,
            solver_a_name="solver_a",
            solver_b_name="solver_b",
            paired_scenarios=bad_rows,
            mode=MODE_COMMON_TIME,
            bootstrap_seed=123,
            num_resamples=8,
            confidence_level=0.95,
        )


def test_saved_index_legacy_requires_explicit_initial_frame_policy(
    tmp_path: Path,
) -> None:
    contract, _, _, _ = _contract(tmp_path)

    with pytest.raises(ValueError, match="initial_frame_policy"):
        compare_solver_scenarios(
            contract=contract,
            solver_a_name="solver_a",
            solver_b_name="solver_b",
            paired_scenarios=_paired_metric_rows(),
            mode=MODE_SAVED_INDEX_LEGACY,
            bootstrap_seed=1,
            num_resamples=4,
            confidence_level=0.95,
        )


def test_saved_index_legacy_rejects_frame_mismatch_without_truncation(
    tmp_path: Path,
) -> None:
    contract, _, _, _ = _contract(tmp_path)
    rows = _paired_metric_rows()
    rows[1]["right"]["trajectory_eta"] = np.asarray(
        [[[0.0]], [[0.5]]], dtype=np.float64
    )
    rows[1]["right"]["timestamps"] = np.asarray([0.0, 0.006], dtype=np.float64)

    with pytest.raises(ValueError, match="equal frame counts"):
        compare_solver_scenarios(
            contract=contract,
            solver_a_name="solver_a",
            solver_b_name="solver_b",
            paired_scenarios=rows,
            mode=MODE_SAVED_INDEX_LEGACY,
            bootstrap_seed=1,
            num_resamples=4,
            confidence_level=0.95,
            initial_frame_policy="include",
        )


def test_compare_solver_scenarios_rejects_shuffled_ids(tmp_path: Path) -> None:
    contract, _, _, _ = _contract(tmp_path)
    rows = list(reversed(_paired_metric_rows()))

    with pytest.raises(ValueError, match="exact ordered scenario ids"):
        compare_solver_scenarios(
            contract=contract,
            solver_a_name="solver_a",
            solver_b_name="solver_b",
            paired_scenarios=rows,
            mode=MODE_COMMON_TIME,
            bootstrap_seed=1,
            num_resamples=4,
            confidence_level=0.95,
        )


def test_iter_paired_raw_reference_samples_fails_when_timestamps_are_missing(
    tmp_path: Path,
) -> None:
    contract, _, _, _ = _contract(tmp_path)
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_raw_sample(
        left_root,
        sample_index=1,
        scenario_id="scenario_000001",
        trajectory_eta=np.zeros((3, 1, 1), dtype=np.float32),
        timestamps=None,
    )
    _write_raw_sample(
        right_root,
        sample_index=1,
        scenario_id="scenario_000001",
        trajectory_eta=np.zeros((3, 1, 1), dtype=np.float32),
        timestamps=np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
    )
    _write_raw_sample(
        left_root,
        sample_index=2,
        scenario_id="scenario_000002",
        trajectory_eta=np.zeros((3, 1, 1), dtype=np.float32),
        timestamps=np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
    )
    _write_raw_sample(
        right_root,
        sample_index=2,
        scenario_id="scenario_000002",
        trajectory_eta=np.zeros((3, 1, 1), dtype=np.float32),
        timestamps=np.asarray([0.0, 0.004, 0.008], dtype=np.float32),
    )

    iterator = iter_paired_raw_reference_samples(
        contract=contract,
        left_root=left_root,
        right_root=right_root,
    )
    with pytest.raises(KeyError, match="timestamps"):
        list(iterator)


def test_full_suite_requires_passing_dense_decision(tmp_path: Path) -> None:
    ordered_ids = ("scenario_000001", "scenario_000002")
    audit_hash = "audit-hash"
    audit_path = tmp_path / "paired_reference_audit.json"
    selection_path = tmp_path / "common_time_validation_scenarios.json"
    _write_json(audit_path, _audit_artifact(ordered_ids, audit_hash))
    _write_json(selection_path, _selection_artifact(ordered_ids, audit_hash))

    with pytest.raises(
        FileNotFoundError, match="requires a passing dense validation decision"
    ):
        resolve_suite_contract(
            alignment_cfg=_alignment_cfg(),
            audit_artifact_path=audit_path,
            scenario_selection_path=selection_path,
            suite_name="full",
            dense_validation_decision_path=tmp_path / "missing" / "decision.json",
            require_full_suite_dense_decision=True,
            dense_fallback_policy="unsupported",
        )

    failed_decision = _dense_validation_artifacts(
        tmp_path / "dense_reference_validation",
        audit_hash=audit_hash,
        dense_ids=ordered_ids,
        status="fail",
    )
    with pytest.raises(ValueError, match="did not pass"):
        resolve_suite_contract(
            alignment_cfg=_alignment_cfg(),
            audit_artifact_path=audit_path,
            scenario_selection_path=selection_path,
            suite_name="full",
            dense_validation_decision_path=failed_decision,
            require_full_suite_dense_decision=True,
            dense_fallback_policy="unsupported",
        )


def test_resolve_suite_contract_rejects_audit_hash_mismatch(tmp_path: Path) -> None:
    ordered_ids = ("scenario_000001", "scenario_000002")
    audit_path = tmp_path / "paired_reference_audit.json"
    selection_path = tmp_path / "common_time_validation_scenarios.json"
    _write_json(audit_path, _audit_artifact(ordered_ids, "audit-a"))
    _write_json(selection_path, _selection_artifact(ordered_ids, "audit-b"))

    with pytest.raises(ValueError, match="audit_hash"):
        resolve_suite_contract(
            alignment_cfg=_alignment_cfg(),
            audit_artifact_path=audit_path,
            scenario_selection_path=selection_path,
            suite_name="dense_validation",
            dense_validation_decision_path=None,
            require_full_suite_dense_decision=False,
            dense_fallback_policy="unsupported",
        )
