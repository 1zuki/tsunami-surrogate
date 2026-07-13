from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import stable_hash_payload
from src.evaluation.established_solver_validation import (
    EXTERNAL_RESULT_SCHEMA_ID,
    SCHEMA_ID,
    _comparison_metrics,
    _load_external_result,
    _validate_config,
    _verify_level_a,
    _write_checksums,
    evaluate_minimum_established_solver_validation,
)


def test_candidate_config_is_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/eval/minimum_established_solver_validation.yaml").read_text(
            encoding="utf-8"
        )
    )
    _validate_config(config)


def test_failed_level_a_cannot_prepare_minimum_package(tmp_path: Path) -> None:
    contract = {
        "contract_hash": "level-a-contract",
        "source_config": {},
        "code_state": {"code_state_hash": "old"},
    }
    decision = {
        "contract_hash": "level-a-contract",
        "decision": "blocked_boundary_behavior",
        "level_a_passed": False,
    }
    (tmp_path / "execution").mkdir()
    (tmp_path / "preregistered_contract.json").write_text(
        json.dumps(contract) + "\n", encoding="utf-8"
    )
    (tmp_path / "execution/decision.json").write_text(
        json.dumps(decision) + "\n", encoding="utf-8"
    )
    _write_checksums(tmp_path)
    with pytest.raises(RuntimeError, match="requires a fresh passing Level A"):
        _verify_level_a(Path(__file__).resolve().parents[1], tmp_path)


def test_identical_fields_pass_metric_identity() -> None:
    times = np.arange(1, 6, dtype=np.float64) * 0.1
    x = np.arange(8, dtype=np.float64)[:, None]
    field = np.stack(
        [np.sin(x + time) * np.ones((1, 4)) for time in times], axis=0
    )
    metrics = _comparison_metrics(
        field,
        field.copy(),
        times,
        np.asarray([[1, 2], [4, 2], [7, 2]], dtype=np.int64),
        arrival_fraction=0.1,
        inactive_floor=1.0e-12,
    )
    assert metrics["active_gauge_count"] == 3
    assert metrics["trajectory_relative_l2"] == 0.0
    assert metrics["per_time_relative_l2_p95"] == 0.0
    assert metrics["gauge_nrmse_max"] == 0.0
    assert metrics["arrival_time_abs_max"] == 0.0
    assert metrics["peak_relative_error_max"] == 0.0
    assert metrics["time_to_peak_abs_max"] == 0.0
    assert metrics["waveform_lag_steps_max"] == 0


def test_external_result_identity_and_shape_are_strict(tmp_path: Path) -> None:
    times = np.asarray([0.1, 0.2], dtype=np.float64)
    requirement = {
        "case_hash": "case-hash",
        "comparator_id": "geoclaw_swe",
        "comparator_version": "5.14.0",
        "eta_shape": [2, 3, 4],
        "required_npz_keys": [
            "schema_id",
            "case_hash",
            "comparator_id",
            "comparator_version",
            "comparator_commit",
            "times",
            "eta",
        ],
    }
    path = tmp_path / "result.npz"
    np.savez_compressed(
        path,
        schema_id=np.asarray(EXTERNAL_RESULT_SCHEMA_ID),
        case_hash=np.asarray("case-hash"),
        comparator_id=np.asarray("geoclaw_swe"),
        comparator_version=np.asarray("5.14.0"),
        comparator_commit=np.asarray("abc123"),
        times=times,
        eta=np.zeros((2, 3, 4), dtype=np.float64),
    )
    eta, metadata = _load_external_result(path, requirement, times)
    assert eta.shape == (2, 3, 4)
    assert metadata["comparator_commit"] == "abc123"

    np.savez_compressed(
        path,
        schema_id=np.asarray(EXTERNAL_RESULT_SCHEMA_ID),
        case_hash=np.asarray("wrong"),
        comparator_id=np.asarray("geoclaw_swe"),
        comparator_version=np.asarray("5.14.0"),
        comparator_commit=np.asarray("abc123"),
        times=times,
        eta=np.zeros((2, 3, 4), dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="case_hash mismatch"):
        _load_external_result(path, requirement, times)


def test_evaluator_accepts_complete_identical_fixture(tmp_path: Path) -> None:
    times = np.asarray([0.1, 0.2], dtype=np.float64)
    eta = np.stack(
        [
            np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float64),
            np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        ]
    )
    config = {
        "gauges": {
            "arrival_fraction_of_external_peak": 0.1,
            "inactive_external_peak_floor": 1.0e-12,
        },
        "thresholds": {
            "flat_analytical": {
                "trajectory_relative_l2": 1.0e-12,
                "per_time_relative_l2_p95": 1.0e-12,
                "gauge_nrmse_max": 1.0e-12,
                "arrival_time_abs_max": 1.0e-12,
                "peak_relative_error_max": 1.0e-12,
                "time_to_peak_abs_max": 1.0e-12,
                "waveform_lag_steps_max": 1,
            },
            "refinement": {
                "finest_to_coarsest_error_ratio_max": 1.05,
                "gated_pairings": ["swe_hydrostatic__geoclaw_swe"],
            },
        },
    }
    case = {
        "case_id": "fixture_nx2_ny2",
        "case_hash": "fixture-case-hash",
        "category": "flat_analytical",
        "nx": 2,
        "ny": 2,
    }
    requirement = {
        "case_id": case["case_id"],
        "case_hash": case["case_hash"],
        "comparator_id": "geoclaw_swe",
        "comparator_version": "5.14.0",
        "relative_path": f"{case['case_id']}/geoclaw_swe.npz",
        "required_npz_keys": [
            "schema_id",
            "case_hash",
            "comparator_id",
            "comparator_version",
            "comparator_commit",
            "times",
            "eta",
        ],
        "eta_shape": [2, 2, 2],
    }
    frozen = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "minimum-established-solver-validation-frozen-contract",
        "source_config": config,
        "requested_times": times.tolist(),
        "cases": [case],
        "pairings": [
            {
                "pairing_id": "swe_hydrostatic__geoclaw_swe",
                "case_id": case["case_id"],
                "case_hash": case["case_hash"],
                "category": case["category"],
                "inhouse_solver": "swe_hydrostatic",
                "external_comparator": "geoclaw_swe",
            }
        ],
        "external_results": [requirement],
    }
    bundle_hash = stable_hash_payload(
        artifact_kind="minimum-established-solver-validation-contract",
        payload=frozen,
        schema_id=SCHEMA_ID,
    )
    frozen["bundle_hash"] = bundle_hash
    bundle = tmp_path / bundle_hash
    case_root = bundle / "cases" / case["case_id"]
    case_root.mkdir(parents=True)
    (bundle / "frozen_contract.json").write_text(
        json.dumps(frozen) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        case_root / "input.npz",
        gauge_indices=np.asarray([[0, 1], [1, 1]], dtype=np.int64),
    )
    np.savez_compressed(
        case_root / "inhouse_swe_hydrostatic.npz",
        eta=eta,
        times=times,
        case_hash=np.asarray(case["case_hash"]),
    )
    _write_checksums(bundle)

    external = tmp_path / "external" / case["case_id"]
    external.mkdir(parents=True)
    np.savez_compressed(
        external / "geoclaw_swe.npz",
        schema_id=np.asarray(EXTERNAL_RESULT_SCHEMA_ID),
        case_hash=np.asarray(case["case_hash"]),
        comparator_id=np.asarray("geoclaw_swe"),
        comparator_version=np.asarray("5.14.0"),
        comparator_commit=np.asarray("abc123"),
        times=times,
        eta=eta,
    )
    output = evaluate_minimum_established_solver_validation(
        bundle_root=bundle,
        external_root=tmp_path / "external",
        output_root=tmp_path / "evaluation",
    )
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["minimum_level_b_passed"] is True
    assert decision["decision"] == "pass_to_H1"
