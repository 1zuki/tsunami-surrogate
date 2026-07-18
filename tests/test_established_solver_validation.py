from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import stable_hash_payload
from src.data_gen.simulate_dataset import BufferedDomainConfig, _prepare_buffered_domain
from src.evaluation.buffered_crop_benchmark import prepare_buffered_case
from src.evaluation.established_solver_validation import (
    EXTERNAL_RESULT_SCHEMA_ID,
    EXTERNAL_RESULT_SCHEMA_ID_V3,
    SCHEMA_ID,
    SCHEMA_ID_V3,
    SCHEMA_ID_V4,
    _comparison_metrics,
    _comparison_metrics_v3,
    _comparison_metrics_v4,
    _build_cases,
    _flat_linear_swe_reference,
    _load_external_result,
    _normalized_waveform_lag_steps,
    _v4_at_or_below,
    _v4_pairwise_refinement,
    _v4_threshold_results,
    _validate_external_checksums,
    _validate_config,
    _verify_level_a,
    _write_checksums,
    established_solver_status,
    evaluate_minimum_established_solver_validation,
)
from src.evaluation.geoclaw_adapter import _write_external_checksums


def test_candidate_config_is_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/eval/minimum_established_solver_validation.yaml").read_text(
            encoding="utf-8"
        )
    )
    _validate_config(config)


def test_v3_candidate_config_matches_boussinesq_only_in_long_wave_regime() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs/eval/minimum_established_solver_validation_v3.yaml"
        ).read_text(encoding="utf-8")
    )
    _validate_config(config)
    assert config["schema_id"] == SCHEMA_ID_V3
    bouss_cases = [
        case
        for case in config["cases"]
        if ["boussinesq", "geoclaw_sgn"] in case["pairings"]
    ]
    assert len(bouss_cases) == 1
    case = bouss_cases[0]
    assert case["generator"] == "flat_linear_mode"
    assert 2.0 * np.pi * case["mode"] * case["depth"] <= 0.35
    assert case["amplitude"] / case["depth"] <= 1.0e-3

    invalid = json.loads(json.dumps(config))
    invalid["cases"][0]["pairings"].append(["boussinesq", "geoclaw_sgn"])
    with pytest.raises(ValueError, match="matched constant-depth long-wave"):
        _validate_config(invalid)


def test_v4_candidate_preserves_v3_thresholds_and_declares_roles() -> None:
    root = Path(__file__).resolve().parents[1]
    v3 = yaml.safe_load(
        (
            root
            / "configs/eval/minimum_established_solver_validation_v3.yaml"
        ).read_text(encoding="utf-8")
    )
    v4 = yaml.safe_load(
        (
            root
            / "configs/eval/minimum_established_solver_validation_v4.yaml"
        ).read_text(encoding="utf-8")
    )
    _validate_config(v4)
    assert v4["schema_id"] == SCHEMA_ID_V4
    assert v4["thresholds"] == v3["thresholds"]
    roles = v4["decision_policy"]["category_roles"]
    assert roles["flat_analytical"]["comparison"] == "descriptive_only"
    assert roles["matched_long_wave"]["comparison"] == "gate"
    assert roles["matched_long_wave"]["descriptive_metrics"] == [
        "waveform_lag_steps_max"
    ]
    assert roles["production_input"]["comparison"] == "descriptive_only"

    invalid = json.loads(json.dumps(v4))
    invalid["aggregation"]["require_every_descriptive_comparison"] = True
    with pytest.raises(ValueError, match="aggregation policy changed"):
        _validate_config(invalid)


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


def test_v3_metrics_bound_low_signal_denominators_and_stabilize_peak_time() -> None:
    times = np.arange(1, 7, dtype=np.float64) * 0.1
    reference_signal = np.asarray([0.1, 1.0, 0.995, 0.2, 0.01, 0.0])
    candidate_signal = np.asarray([0.1, 0.995, 1.0, 0.2, 0.01, 1.0e-3])
    reference = reference_signal[:, None, None]
    candidate = candidate_signal[:, None, None]
    metrics = _comparison_metrics_v3(
        candidate,
        reference,
        times,
        np.asarray([[0, 0]], dtype=np.int64),
        inactive_floor=1.0e-12,
        per_time_signal_floor_fraction=0.05,
        peak_plateau_fraction=0.99,
        lag_minimum_overlap_fraction=0.5,
    )
    assert metrics["per_time_denominator_floor"] == pytest.approx(0.05)
    assert metrics["per_time_scaled_l2_p95"] < 0.02
    assert metrics["peak_plateau_time_abs_max"] == 0.0
    assert metrics["arrival_metric_eligible"] is False
    assert metrics["arrival_time_abs_max"] is None


def test_v3_lag_uses_normalized_overlap_and_prefers_zero_on_ties() -> None:
    waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False))
    assert (
        _normalized_waveform_lag_steps(
            waveform,
            waveform,
            minimum_overlap_fraction=0.5,
        )
        == 0
    )


def test_v4_float_threshold_tolerance_is_tight_and_integer_limits_are_exact() -> None:
    assert _v4_at_or_below(
        0.007000000000000006,
        0.007,
        integer=False,
        rel_tolerance=1.0e-12,
        abs_tolerance=1.0e-15,
    )
    assert not _v4_at_or_below(
        0.0070001,
        0.007,
        integer=False,
        rel_tolerance=1.0e-12,
        abs_tolerance=1.0e-15,
    )
    assert not _v4_at_or_below(
        3,
        2,
        integer=True,
        rel_tolerance=1.0,
        abs_tolerance=1.0,
    )


def test_v4_standing_mode_lag_can_be_descriptive_without_hiding_field_gates() -> None:
    metrics = {
        "active_gauge_count": 1,
        "trajectory_relative_l2": 0.01,
        "per_time_scaled_l2_p95": 0.01,
        "gauge_nrmse_max": 0.01,
        "peak_relative_error_max": 0.01,
        "peak_plateau_time_abs_max": 0.0,
        "waveform_lag_steps_max": 3,
    }
    thresholds = {
        "trajectory_relative_l2": 0.20,
        "per_time_scaled_l2_p95": 0.20,
        "gauge_nrmse_max": 0.15,
        "peak_relative_error_max": 0.15,
        "peak_plateau_time_abs_max": 0.007,
        "waveform_lag_steps_max": 2,
    }
    results, passed = _v4_threshold_results(
        metrics,
        thresholds,
        descriptive_metrics=["waveform_lag_steps_max"],
        rel_tolerance=1.0e-12,
        abs_tolerance=1.0e-15,
    )
    assert passed is True
    assert results["waveform_lag_steps_max"] == {
        "decision_role": "descriptive_only",
        "limit": 2,
        "passed": False,
    }
    metrics["trajectory_relative_l2"] = 0.21
    _results, passed = _v4_threshold_results(
        metrics,
        thresholds,
        descriptive_metrics=["waveform_lag_steps_max"],
        rel_tolerance=1.0e-12,
        abs_tolerance=1.0e-15,
    )
    assert passed is False


def test_v4_metrics_separate_active_time_amplitude_shape_and_regions() -> None:
    times = np.arange(1, 7, dtype=np.float64) * 0.1
    reference = np.ones((6, 8, 8), dtype=np.float64)
    candidate = 0.8 * reference
    metrics = _comparison_metrics_v4(
        candidate,
        reference,
        times,
        np.asarray([[2, 2], [5, 5]], dtype=np.int64),
        inactive_floor=1.0e-12,
        per_time_signal_floor_fraction=0.05,
        peak_plateau_fraction=0.99,
        lag_minimum_overlap_fraction=0.5,
        diagnostic_boundary_band_cells=2,
    )
    assert metrics["per_time_active_count"] == 6
    assert metrics["per_time_inactive_count"] == 0
    assert metrics["field_norm_ratio"] == pytest.approx(0.8)
    assert metrics["optimal_amplitude_scale"] == pytest.approx(0.8)
    assert metrics["field_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["shape_relative_l2_after_scale"] == pytest.approx(0.0)
    assert metrics["boundary_band_relative_l2"] == pytest.approx(0.2)
    assert metrics["interior_relative_l2"] == pytest.approx(0.2)


def test_v4_refinement_requires_every_pair_to_decrease() -> None:
    passing = _v4_pairwise_refinement(
        [0.4, 0.2, 0.1],
        [32, 64, 128],
        ratio_limit=1.05,
        require_strict_decrease=True,
        rel_tolerance=1.0e-12,
        abs_tolerance=1.0e-15,
    )
    assert passing["passed"] is True
    assert passing["pairwise_orders"] == pytest.approx([1.0, 1.0])

    nonmonotonic = _v4_pairwise_refinement(
        [0.4, 0.45, 0.1],
        [32, 64, 128],
        ratio_limit=1.05,
        require_strict_decrease=True,
        rel_tolerance=1.0e-12,
        abs_tolerance=1.0e-15,
    )
    assert nonmonotonic["finest_to_coarsest_error_ratio"] < 1.05
    assert nonmonotonic["passed"] is False


def test_flat_linear_swe_reference_preserves_initial_mode_at_zero_time() -> None:
    nx, ny = 32, 4
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    eta0 = np.cos(2.0 * np.pi * x)[:, None] * np.ones((1, ny))
    reference = _flat_linear_swe_reference(
        eta0,
        np.asarray([0.0], dtype=np.float64),
        depth=1.0,
        gravity=9.81,
    )
    np.testing.assert_allclose(reference[0], eta0, rtol=0.0, atol=5.0e-16)


def test_production_cases_use_dataset_exact_source_preparation(monkeypatch) -> None:
    bathymetry = np.full((64, 64), -1.0, dtype=np.float32)
    source = np.random.default_rng(17).normal(size=(64, 64)).astype(np.float32)
    strength = float(np.float32(0.7312345))
    monkeypatch.setattr(
        "src.evaluation.established_solver_validation._load_canary_arrays",
        lambda _canary: (
            bathymetry,
            source,
            np.asarray([strength], dtype=np.float32),
            strength,
            {},
        ),
    )
    buffered = {
        "core_grid": 64,
        "inhouse_total_grid": 96,
        "external_total_grid": 192,
        "source_taper_cells": 8,
        "bathymetry_extension": "edge",
        "output_crop": "central",
        "inhouse_sponge": {
            "enabled": True,
            "width": 16,
            "min_factor": 0.8,
            "axes": "xy",
            "profile": "cosine",
            "time_mode": "elapsed_time_consistent",
            "reference_dt": 0.0035,
        },
        "external_sponge": "disabled",
        "external_boundary": "open_extrapolation",
        "return_time_safety_factor": 1.1,
    }
    config = {
        "gauges": {"fractional_cell_locations": [[0.5, 0.5]]},
        "inhouse": {"gravity": 9.81},
        "requested_times": {"horizon": 0.175},
        "cases": [
            {
                "case_id": "production_input_canaries",
                "category": "production_input",
                "generator": "level_a_canaries",
                "count": 1,
                "grids": [64],
                "boundary": "radiation",
                "buffered_domain": buffered,
                "pairings": [["swe_hydrostatic", "geoclaw_swe"]],
            }
        ],
    }
    cases = _build_cases(
        config,
        {
            "canaries": [
                {"qualified_id": "train:test", "input_fingerprint": "x"}
            ]
        },
    )
    _record, inhouse, _external = cases[0]
    expected = _prepare_buffered_domain(
        bathymetry,
        source,
        strength,
        0.0,
        BufferedDomainConfig(
            enabled=True,
            buffer_cells=16,
            source_taper_cells=8,
            bathymetry_extension="edge",
            output_crop="central",
        ),
    )
    assert np.array_equal(inhouse["eta0"], expected["solver_eta0"])

    old_eta0 = np.asarray(strength * source, dtype=np.float32)
    old = prepare_buffered_case(
        bathymetry,
        old_eta0,
        buffer_cells=16,
        source_taper_cells=8,
    )
    assert not np.array_equal(inhouse["eta0"], old["eta0"])


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


def test_v3_external_result_is_bound_to_manifest_and_ksp_health(
    tmp_path: Path,
) -> None:
    times = np.asarray([0.1, 0.2], dtype=np.float64)
    requirement = {
        "case_hash": "case-hash",
        "comparator_id": "geoclaw_sgn",
        "comparator_version": "5.14.0",
        "result_schema_id": EXTERNAL_RESULT_SCHEMA_ID_V3,
        "eta_shape": [2, 1, 1],
        "required_npz_keys": [
            "schema_id",
            "case_hash",
            "comparator_id",
            "comparator_version",
            "comparator_commit",
            "clawpack_commit",
            "petsc_commit",
            "adapter_hash",
            "times",
            "actual_times",
            "eta",
            "runtime_seconds",
            "initial_state_max_abs_error",
            "requested_time_max_abs_error",
            "nominal_eta_max_abs_difference",
            "nominal_eta_consistency_floor",
            "solver_health_status",
            "ksp_solve_count",
            "ksp_iteration_max",
            "ksp_iteration_mean",
            "ksp_convergence_reasons",
        ],
    }
    manifest = {
        "adapter_hash": "adapter",
        "revisions": {
            "geoclaw_commit": "geo",
            "clawpack_commit": "claw",
            "petsc_commit": "petsc",
        },
    }
    path = tmp_path / "result.npz"

    def write(
        reason: str = "CONVERGED_RTOL",
        *,
        actual_times: np.ndarray = times,
    ) -> None:
        time_error = float(np.max(np.abs(actual_times - times)))
        np.savez_compressed(
            path,
            schema_id=np.asarray(EXTERNAL_RESULT_SCHEMA_ID_V3),
            case_hash=np.asarray("case-hash"),
            comparator_id=np.asarray("geoclaw_sgn"),
            comparator_version=np.asarray("5.14.0"),
            comparator_commit=np.asarray("geo"),
            clawpack_commit=np.asarray("claw"),
            petsc_commit=np.asarray("petsc"),
            adapter_hash=np.asarray("adapter"),
            times=times,
            actual_times=actual_times,
            eta=np.zeros((2, 1, 1), dtype=np.float64),
            runtime_seconds=np.asarray(1.0),
            initial_state_max_abs_error=np.asarray(0.0),
            requested_time_max_abs_error=np.asarray(time_error),
            nominal_eta_max_abs_difference=np.asarray(0.0),
            nominal_eta_consistency_floor=np.asarray(1.0e-7),
            solver_health_status=np.asarray("passed"),
            ksp_solve_count=np.asarray(2),
            ksp_iteration_max=np.asarray(7),
            ksp_iteration_mean=np.asarray(6.0),
            ksp_convergence_reasons=np.asarray([reason]),
        )

    write()
    eta, metadata = _load_external_result(
        path, requirement, times, manifest
    )
    assert eta.shape == (2, 1, 1)
    assert metadata["ksp_solve_count"] == 2

    roundoff_times = times.copy()
    roundoff_times[-1] = np.nextafter(roundoff_times[-1], np.inf)
    write(actual_times=roundoff_times)
    _load_external_result(path, requirement, times, manifest)

    invalid_times = times.copy()
    invalid_times[-1] += 1.0e-12
    write(actual_times=invalid_times)
    with pytest.raises(RuntimeError, match="actual-time mismatch"):
        _load_external_result(path, requirement, times, manifest)

    write("DIVERGED_ITS")
    with pytest.raises(RuntimeError, match="KSP health"):
        _load_external_result(path, requirement, times, manifest)


def test_v3_external_checksum_manifest_has_exact_coverage(
    tmp_path: Path,
) -> None:
    frozen = {
        "external_results": [{"relative_path": "case/geoclaw_swe.npz"}]
    }
    (tmp_path / "case").mkdir()
    (tmp_path / "RUN_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    result = tmp_path / "case/geoclaw_swe.npz"
    result.write_bytes(b"canonical-result")
    _write_external_checksums(tmp_path, frozen)
    _validate_external_checksums(tmp_path, frozen)

    result.write_bytes(b"corrupted-result")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _validate_external_checksums(tmp_path, frozen)


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
    status = established_solver_status(
        bundle_root=bundle,
        external_root=tmp_path / "external",
    )
    assert status["complete"] is True
    assert status["valid"] == status["total"] == 1
    output = evaluate_minimum_established_solver_validation(
        bundle_root=bundle,
        external_root=tmp_path / "external",
        output_root=tmp_path / "evaluation",
    )
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["minimum_level_b_passed"] is True
    assert decision["decision"] == "pass_to_H1"
