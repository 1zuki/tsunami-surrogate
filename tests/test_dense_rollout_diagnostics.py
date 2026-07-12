from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_gen.simulate_dataset import _simulate_one_local
from src.evaluation.alignment import (
    SCHEMA_ID,
    stable_hash_payload,
    stable_hash_scenario_ids,
)
from src.evaluation.dense_reference_validation import (
    FDE_NAME_BY_SOLVER_KEY,
    SOLVER_ORDER,
    build_solver_from_legacy_sample,
    compute_legacy_knot_reproduction_metrics,
    evaluate_dense_reference_criteria,
    resolve_validation_suite,
    run_dense_reference_validation,
    safe_ratio,
)
from src.utils.config import load_config


def _sample_arrays(nx: int = 6, ny: int = 6) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bathymetry = -(1.0 + 0.15 * np.cos(np.pi * xx) * np.cos(0.5 * np.pi * yy)).astype(
        np.float32
    )
    eta0 = (0.015 * np.exp(-3.5 * (xx * xx + yy * yy))).astype(np.float32)
    rest_depth = np.maximum(-bathymetry, 0.0).astype(np.float32)
    initial_depth = np.maximum(rest_depth + eta0, 0.0).astype(np.float32)
    free_surface0 = (initial_depth + bathymetry).astype(np.float32)
    source_field = np.where(
        (xx * xx + yy * yy) < 0.4,
        np.float32(0.5),
        np.float32(-0.1),
    ).astype(np.float32)
    return {
        "bathymetry": bathymetry,
        "source_field": source_field,
        "eta0": eta0,
        "rest_depth": rest_depth,
        "initial_depth": initial_depth,
        "free_surface0": free_surface0,
    }


def _stored_solver_cfg() -> dict[str, object]:
    return {
        "nx": 6,
        "ny": 6,
        "dx": 0.2,
        "dy": 0.2,
        "dt": 0.01,
        "g": 9.81,
        "cfl": 0.4,
        "dry_tolerance": "1e-6",
        "max_velocity": 30.0,
        "boundary": "periodic",
        "use_sponge": False,
        "sponge_width": 0,
        "sponge_min_factor": 0.9,
        "alpha": 1.0 / 3.0,
        "min_depth": "1e-4",
        "sea_level_offset": 0.0,
        "depth_scale": 1.0,
        "mode": "linear_variable_depth",
        "filter_strength": 0.0,
        "linear_solver_tol": "1e-10",
        "linear_solver_max_iter": 80,
        "check_finite": True,
    }


def _build_solver(
    solver_key: str, arrays: dict[str, np.ndarray], cfg: dict[str, object] | None = None
):
    return build_solver_from_legacy_sample(
        solver_key,
        stored_solver_cfg=_stored_solver_cfg() if cfg is None else dict(cfg),
        sample_arrays=arrays,
    )


def _dense_diag_keys() -> set[str]:
    return {
        "proposed_dt",
        "pre_step_cfl",
        "post_step_cfl",
        "elapsed_benchmark_time",
        "finite_state_flag",
    }


def _create_dense_validation_fixture(tmp_path: Path) -> dict[str, object]:
    scenario_id = "scenario_000001"
    sample_index = 1
    dataset_cfg_path = tmp_path / "dataset.yaml"
    dataset_cfg_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  n_steps: 8",
                "  save_every: 4",
                "  auto_dt: false",
                "  target_cfl: 0.4",
                "  include_initial_state: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    arrays = _sample_arrays()
    processed_roots: dict[str, str] = {}
    raw_roots: dict[str, str] = {}
    common_row = {
        "scenario_id": scenario_id,
        "bathymetry_type": "synthetic_basin",
        "source_type": "synthetic_gaussian",
        "source_strength": 1.0,
    }

    for solver_key in SOLVER_ORDER:
        raw_root = tmp_path / "raw" / solver_key / "samples"
        sample_dir = raw_root / "sample_000001"
        sample_dir.mkdir(parents=True, exist_ok=True)
        solver_cfg = _stored_solver_cfg()
        solver = _build_solver(solver_key, arrays, solver_cfg)
        trajectory, timestamps, dt_history, diagnostics = _simulate_one_local(
            solver=solver,
            n_steps=8,
            save_every=4,
            auto_dt=False,
            target_cfl=0.4,
            include_initial_state=True,
        )
        trajectory_eta = (
            trajectory[:, 0]
            if solver_key == "boussinesq"
            else trajectory[:, 0] + arrays["bathymetry"][None, ...]
        ).astype(np.float32)

        rollout_payload = {
            "trajectory": trajectory.astype(np.float32),
            "trajectory_eta": trajectory_eta,
            "timestamps": timestamps.astype(np.float32),
            "dt_history": dt_history.astype(np.float32),
            "fde_name": np.asarray([FDE_NAME_BY_SOLVER_KEY[solver_key]], dtype="U64"),
        }
        rollout_payload.update(diagnostics)
        np.savez_compressed(sample_dir / "rollout.npz", **rollout_payload)
        np.save(sample_dir / "trajectory_eta.npy", trajectory_eta)
        np.savez_compressed(
            sample_dir / "sample.npz",
            bathymetry=arrays["bathymetry"],
            source_field=arrays["source_field"],
            rest_depth=arrays["rest_depth"],
            eta0=arrays["eta0"],
            initial_depth=arrays["initial_depth"],
            free_surface0=arrays["free_surface0"],
            trajectory=trajectory.astype(np.float32),
            trajectory_eta=trajectory_eta,
            timestamps=timestamps.astype(np.float32),
            dt_history=dt_history.astype(np.float32),
            solver_name=np.asarray([FDE_NAME_BY_SOLVER_KEY[solver_key]], dtype="U64"),
            scenario_id=np.asarray([scenario_id], dtype="U64"),
        )

        meta = {
            "sample_index": sample_index,
            "scenario_id": scenario_id,
            "solver_name": FDE_NAME_BY_SOLVER_KEY[solver_key],
            "bathymetry_type": common_row["bathymetry_type"],
            "source_type": common_row["source_type"],
            "source_strength": common_row["source_strength"],
            "num_frames": int(trajectory.shape[0]),
            "solver": solver_cfg,
            "dataset_config_path": str(dataset_cfg_path),
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )

        processed_root = tmp_path / "processed" / solver_key / "test"
        processed_root.mkdir(parents=True, exist_ok=True)
        processed_roots[solver_key] = str(processed_root)
        raw_roots[solver_key] = str(raw_root)
        with (processed_root / "meta.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "sample_index": sample_index,
                        "scenario_id": scenario_id,
                        "sample_dir": str(sample_dir),
                        "solver_name": FDE_NAME_BY_SOLVER_KEY[solver_key],
                        "num_frames": int(trajectory.shape[0]),
                    }
                )
                + "\n"
            )

    audit_artifact = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "paired-reference-audit",
        "status": "pass",
        "audit_hash": stable_hash_payload({"fixture": "dense-reference-validation"}),
        "config": {
            "audit": {
                "processed_test_roots": processed_roots,
                "raw_test_solver_roots": raw_roots,
            }
        },
    }
    audit_path = tmp_path / "paired_reference_audit.json"
    audit_path.write_text(json.dumps(audit_artifact, indent=2) + "\n", encoding="utf-8")

    ordered_scenarios = [common_row]
    selection = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-validation-scenarios",
        "audit_hash": audit_artifact["audit_hash"],
        "dense_validation": {
            "label": "dense_reference_validation",
            "count": 1,
            "ordered_scenarios": ordered_scenarios,
            "ordered_scenario_ids": [scenario_id],
            "list_hash": stable_hash_scenario_ids([scenario_id]),
        },
        "smoke": {
            "label": "implementation_only_smoke",
            "count": 1,
            "ordered_scenarios": ordered_scenarios,
            "ordered_scenario_ids": [scenario_id],
            "list_hash": stable_hash_scenario_ids([scenario_id]),
        },
    }
    selection_path = tmp_path / "common_time_validation_scenarios.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    config_path = tmp_path / "dense_reference_validation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_id: tsunami-surrogate.alignment.v1",
                "alignment:",
                "  common_time_grid:",
                "    endpoint_tolerance: 1.0e-6",
                "    values:",
                "      - 0.04",
                "      - 0.08",
                "dense_reference_validation:",
                f"  audit_artifact: {audit_path}",
                f"  scenario_selection_path: {selection_path}",
                f"  results_root: {tmp_path / 'results'}",
                "  expected_natural_steps: 8",
                "  legacy_knot_stride: 4",
                "  replay_control:",
                "    n_steps: 8",
                "    save_every: 4",
                "    auto_dt: false",
                "    target_cfl: 0.4",
                "    include_initial_state: true",
                "  tolerances:",
                "    timestamp_abs: 5.0e-8",
                "    eta_max_abs: 1.0e-6",
                "    reproduction_relative_rmse: 1.0e-6",
                "    interpolation_horizon: 0.08",
                "  criteria:",
                "    aggregate_ratio_max: 0.10",
                "    scenario_ratio_median_max: 0.10",
                "    scenario_ratio_p95_max: 0.25",
                "    family_cell_median_max: 0.10",
                "    family_cell_fraction_above_p95_max: 0.5",
                "    interp_to_rms_aggregate_max: 0.01",
                "    interp_to_rms_median_max: 0.01",
                "    interp_to_rms_p95_max: 0.025",
                "  bootstrap:",
                "    seed: 123",
                "    num_resamples: 64",
                "    confidence_level: 0.95",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "config_path": config_path,
        "audit_path": audit_path,
        "selection_path": selection_path,
        "scenario_id": scenario_id,
        "dataset_config_path": dataset_cfg_path,
    }


def test_default_simulate_one_local_behavior_remains_sparse_by_default() -> None:
    arrays = _sample_arrays()

    hydro_solver = _build_solver("hydrostatic", arrays)
    hydro_trajectory, hydro_timestamps, hydro_dt_history, hydro_diagnostics = (
        _simulate_one_local(
            solver=hydro_solver,
            n_steps=4,
            save_every=2,
            auto_dt=False,
            target_cfl=0.4,
            include_initial_state=True,
        )
    )
    assert hydro_trajectory.shape[0] == 3
    assert np.array_equal(
        hydro_timestamps, np.asarray([0.0, 0.02, 0.04], dtype=np.float32)
    )
    assert np.array_equal(
        hydro_dt_history, np.asarray([0.0, 0.01, 0.01], dtype=np.float32)
    )
    assert hydro_diagnostics == {}

    bouss_solver = _build_solver("boussinesq", arrays)
    _, _, _, bouss_diagnostics = _simulate_one_local(
        solver=bouss_solver,
        n_steps=2,
        save_every=1,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
    )
    assert set(bouss_diagnostics) == {
        "cg_failed_count",
        "cg_max_iterations",
        "cg_max_residual_ratio",
    }


def test_dense_rollout_diagnostics_cover_all_three_solvers() -> None:
    arrays = _sample_arrays()

    for solver_key in SOLVER_ORDER:
        solver = _build_solver(solver_key, arrays)
        trajectory, timestamps, dt_history, diagnostics = _simulate_one_local(
            solver=solver,
            n_steps=3,
            save_every=2,
            auto_dt=True,
            target_cfl=0.4,
            include_initial_state=True,
            record_every_step=True,
            dense_diagnostics=True,
        )

        assert trajectory.shape[0] == 4
        assert timestamps.shape == (4,)
        assert dt_history.shape == (4,)
        assert np.all(np.isfinite(dt_history[1:]))
        assert np.all(dt_history[1:] > 0.0)
        assert _dense_diag_keys().issubset(diagnostics)
        assert diagnostics["proposed_dt"].shape == (3,)
        assert np.all(np.isfinite(diagnostics["proposed_dt"]))
        assert np.all(diagnostics["proposed_dt"] > 0.0)
        assert np.all(np.isfinite(diagnostics["pre_step_cfl"]))
        assert np.all(np.isfinite(diagnostics["post_step_cfl"]))
        assert np.all(np.diff(diagnostics["elapsed_benchmark_time"]) > 0.0)
        np.testing.assert_allclose(
            diagnostics["elapsed_benchmark_time"],
            timestamps[1:],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["elapsed_benchmark_time"],
            np.cumsum(dt_history[1:], dtype=np.float32),
            atol=1.0e-7,
            rtol=0.0,
        )
        assert diagnostics["finite_state_flag"].dtype == np.bool_
        assert np.all(diagnostics["finite_state_flag"])

        if solver_key in {"hydrostatic", "muscl_hr"}:
            assert "swe_min_depth" in diagnostics
            assert "swe_max_speed" in diagnostics
            assert "swe_dry_cell_count" in diagnostics
            assert np.all(np.isfinite(diagnostics["swe_min_depth"]))
            assert np.all(np.isfinite(diagnostics["swe_max_speed"]))
            assert np.all(diagnostics["swe_dry_cell_count"] >= 0)
        else:
            assert "cg_solve0_converged" in diagnostics
            assert "cg_solve1_converged" in diagnostics
            assert "cg_solve0_iterations" in diagnostics
            assert "cg_solve1_iterations" in diagnostics
            assert "cg_solve0_initial_residual" in diagnostics
            assert "cg_solve1_initial_residual" in diagnostics
            assert diagnostics["cg_solve0_converged"].shape == (3,)
            assert diagnostics["cg_solve1_converged"].shape == (3,)


def test_dense_rollout_elapsed_benchmark_time_matches_cumulative_solver_time() -> None:
    arrays = _sample_arrays()
    solver = _build_solver("hydrostatic", arrays)

    _, timestamps, dt_history, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=3,
        save_every=2,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
        record_every_step=True,
        dense_diagnostics=True,
    )

    np.testing.assert_allclose(
        timestamps,
        np.asarray([0.0, 0.01, 0.02, 0.03], dtype=np.float32),
    )
    np.testing.assert_allclose(
        diagnostics["elapsed_benchmark_time"],
        np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics["elapsed_benchmark_time"],
        timestamps[1:],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics["elapsed_benchmark_time"],
        np.cumsum(dt_history[1:], dtype=np.float64),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_dense_rollout_time_summary_avoids_float32_accumulation_drift() -> None:
    from src.evaluation.dense_reference_validation import (
        summarize_dense_rollout_diagnostics,
    )

    step_count = 250
    dt = np.full(step_count, np.float32(0.00071), dtype=np.float32)
    elapsed = np.cumsum(dt, dtype=np.float64).astype(np.float32)
    float32_cumulative = np.cumsum(dt, dtype=np.float32)
    assert (
        np.max(
            np.abs(elapsed.astype(np.float64) - float32_cumulative.astype(np.float64))
        )
        > 1.0e-7
    )

    diagnostics = {
        "proposed_dt": dt,
        "pre_step_cfl": np.full(step_count, 0.45, dtype=np.float32),
        "post_step_cfl": np.full(step_count, 0.45, dtype=np.float32),
        "elapsed_benchmark_time": elapsed,
        "finite_state_flag": np.ones(step_count, dtype=np.bool_),
    }
    dt_history = np.concatenate((np.asarray([0.0], dtype=np.float32), dt))

    summary = summarize_dense_rollout_diagnostics(
        diagnostics,
        dense_dt_history=dt_history,
    )

    assert summary["benchmark_time_matches_cumulative_dt"]
    assert summary["benchmark_time_to_cumulative_dt_max_abs_diff"] < 1.0e-7


def test_dense_rollout_diagnostics_propagate_boussinesq_cg_failure() -> None:
    arrays = _sample_arrays()
    failing_cfg = _stored_solver_cfg()
    failing_cfg["linear_solver_tol"] = "1e-14"
    failing_cfg["linear_solver_max_iter"] = 1
    solver = _build_solver("boussinesq", arrays, cfg=failing_cfg)

    _, _, _, diagnostics = _simulate_one_local(
        solver=solver,
        n_steps=1,
        save_every=1,
        auto_dt=False,
        target_cfl=0.4,
        include_initial_state=True,
        record_every_step=True,
        dense_diagnostics=True,
    )

    assert diagnostics["cg_failed_count"][0] > 0
    assert not diagnostics["cg_step_converged"][0]
    assert (
        not diagnostics["cg_solve0_converged"][0]
        or not diagnostics["cg_solve1_converged"][0]
    )


def test_reproduction_metrics_pass_and_fail_with_zero_denominator_logic() -> None:
    dense_timestamps = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    dense_eta = np.asarray(
        [[[0.0]], [[0.1]], [[0.2]], [[0.3]], [[0.4]]], dtype=np.float64
    )
    legacy_timestamps = dense_timestamps[[0, 2, 4]]
    legacy_eta = dense_eta[[0, 2, 4]]

    passed = compute_legacy_knot_reproduction_metrics(
        dense_timestamps=dense_timestamps,
        dense_trajectory_eta=dense_eta,
        legacy_timestamps=legacy_timestamps,
        legacy_trajectory_eta=legacy_eta,
        expected_natural_steps=4,
        legacy_knot_stride=2,
        timestamp_abs_tolerance=5.0e-8,
        eta_max_abs_tolerance=1.0e-6,
        relative_rmse_tolerance=1.0e-6,
    )
    assert passed["pass"]
    assert passed["relative_rmse"] == 0.0

    dense_eta_bad = dense_eta.copy()
    dense_eta_bad[2, 0, 0] += 5.0e-5
    failed = compute_legacy_knot_reproduction_metrics(
        dense_timestamps=dense_timestamps,
        dense_trajectory_eta=dense_eta_bad,
        legacy_timestamps=legacy_timestamps,
        legacy_trajectory_eta=legacy_eta,
        expected_natural_steps=4,
        legacy_knot_stride=2,
        timestamp_abs_tolerance=5.0e-8,
        eta_max_abs_tolerance=1.0e-6,
        relative_rmse_tolerance=1.0e-6,
    )
    assert not failed["pass"]
    assert any("eta_max_abs_diff" in item for item in failed["issues"])

    zero_legacy = np.zeros((3, 1, 1), dtype=np.float64)
    zero_dense = np.zeros((5, 1, 1), dtype=np.float64)
    zero_pass = compute_legacy_knot_reproduction_metrics(
        dense_timestamps=dense_timestamps,
        dense_trajectory_eta=zero_dense,
        legacy_timestamps=legacy_timestamps,
        legacy_trajectory_eta=zero_legacy,
        expected_natural_steps=4,
        legacy_knot_stride=2,
        timestamp_abs_tolerance=5.0e-8,
        eta_max_abs_tolerance=1.0e-6,
        relative_rmse_tolerance=1.0e-6,
    )
    assert zero_pass["relative_rmse"] == 0.0

    zero_dense_bad = zero_dense.copy()
    zero_dense_bad[2, 0, 0] = 1.0e-3
    zero_fail = compute_legacy_knot_reproduction_metrics(
        dense_timestamps=dense_timestamps,
        dense_trajectory_eta=zero_dense_bad,
        legacy_timestamps=legacy_timestamps,
        legacy_trajectory_eta=zero_legacy,
        expected_natural_steps=4,
        legacy_knot_stride=2,
        timestamp_abs_tolerance=5.0e-8,
        eta_max_abs_tolerance=1.0e-6,
        relative_rmse_tolerance=1.0e-6,
    )
    assert np.isinf(zero_fail["relative_rmse"])


def test_criteria_and_suite_semantics_cover_zero_gap_and_family_guardrail() -> None:
    selection = {
        "smoke": {
            "label": "implementation_only_smoke",
            "ordered_scenarios": [{"scenario_id": "scenario_000001"}],
            "ordered_scenario_ids": ["scenario_000001"],
            "list_hash": stable_hash_scenario_ids(["scenario_000001"]),
        },
        "dense_validation": {
            "label": "dense_reference_validation",
            "ordered_scenarios": [{"scenario_id": "scenario_000001"}],
            "ordered_scenario_ids": ["scenario_000001"],
            "list_hash": stable_hash_scenario_ids(["scenario_000001"]),
        },
    }
    smoke = resolve_validation_suite(selection, "smoke")
    dense = resolve_validation_suite(selection, "dense_validation")
    assert smoke.manuscript_claims_allowed is False
    assert dense.manuscript_claims_allowed is True

    decision = evaluate_dense_reference_criteria(
        suite_spec=dense,
        zero_extrapolation_failures=0,
        reproduction_failure_count=0,
        aggregate_interp_to_gap_ratio=safe_ratio(1.0, 0.0),
        scenario_ratio_median=0.12,
        scenario_ratio_p95=0.30,
        family_summaries=[
            {
                "bathymetry_type": "synthetic_basin",
                "source_type": "synthetic_gaussian",
                "scenario_ratio_median": 0.12,
                "scenario_ratio_fraction_above_p95_threshold": 0.75,
            }
        ],
        aggregate_interp_to_rms_ratio=0.02,
        scenario_rms_ratio_median=0.02,
        scenario_rms_ratio_p95=0.03,
        thresholds={
            "aggregate_ratio_max": 0.10,
            "scenario_ratio_median_max": 0.10,
            "scenario_ratio_p95_max": 0.25,
            "family_cell_median_max": 0.10,
            "family_cell_fraction_above_p95_max": 0.5,
            "interp_to_rms_aggregate_max": 0.01,
            "interp_to_rms_median_max": 0.01,
            "interp_to_rms_p95_max": 0.025,
        },
    )
    assert decision["status"] == "fail"
    assert not decision["criteria"][
        "aggregate_interp_rmse_over_smallest_aggregate_solver_gap"
    ]["pass"]
    assert not decision["criteria"]["family_cell_ratio_guardrail"]["pass"]


def test_run_dense_reference_validation_end_to_end_with_artifact_schema(
    tmp_path: Path,
) -> None:
    fixture = _create_dense_validation_fixture(tmp_path)
    config = load_config(fixture["config_path"])

    summary = run_dense_reference_validation(
        config,
        suite_name="dense_validation",
        config_path=fixture["config_path"],
    )

    assert summary["schema_id"] == SCHEMA_ID
    assert summary["artifact_kind"] == "dense-reference-validation"
    assert summary["suite"]["name"] == "dense_validation"
    assert summary["suite"]["manuscript_claims_allowed"] is True
    assert summary["inputs"]["selection_audit_hash_match"] is True
    assert summary["alignment"]["time_semantics"] == "solver_benchmark_time"
    assert summary["counts"]["scenario_count"] == 1
    assert summary["counts"]["eligible_for_interpolation_count"] == 1
    assert set(summary["solver_health_summaries"]) == set(SOLVER_ORDER)
    assert Path(summary["artifacts_written"]["scenario_metrics_jsonl"]).is_file()
    assert Path(summary["artifacts_written"]["decision_json"]).is_file()
    assert Path(summary["artifacts_written"]["summary_json"]).is_file()

    scenario_lines = (
        Path(summary["artifacts_written"]["scenario_metrics_jsonl"])
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(scenario_lines) == 1
    scenario_record = json.loads(scenario_lines[0])
    assert scenario_record["scenario_id"] == fixture["scenario_id"]
    assert scenario_record["all_legacy_reproduction_pass"] is True
    assert scenario_record["eligible_for_interpolation"] is True
    assert set(scenario_record["solver_results"]) == set(SOLVER_ORDER)
    assert set(scenario_record["pairwise_dense_solver_gaps"]) == {
        "hydrostatic__vs__muscl_hr",
        "hydrostatic__vs__boussinesq",
        "muscl_hr__vs__boussinesq",
    }
    for solver_key in SOLVER_ORDER:
        health_summary = summary["solver_health_summaries"][solver_key]
        assert "elapsed_benchmark_time_final_max" not in health_summary
        assert health_summary["benchmark_time_final_max"] > 0.0
        assert health_summary["benchmark_time_dense_timestamp_mismatch_count"] == 0
        assert health_summary["benchmark_time_cumulative_dt_mismatch_count"] == 0
        assert health_summary["benchmark_time_to_dense_timestamps_max_abs_diff"] == 0.0
        assert health_summary["benchmark_time_to_cumulative_dt_max_abs_diff"] <= 1.0e-7
        solver_result = scenario_record["solver_results"][solver_key]
        assert solver_result["reproduction"]["pass"] is True
        assert solver_result["reproduction"]["legacy_knot_indices"] == [0, 4, 8]
        assert solver_result["reproduction"]["timestamp_float32_hash_match"] is True
        assert (
            solver_result["reproduction"]["trajectory_eta_float32_hash_match"] is True
        )
        assert solver_result["interpolation"]["zero_extrapolation"] is True
        assert (
            solver_result["health_summary"]["benchmark_time_matches_dense_timestamps"]
            is True
        )
        assert (
            solver_result["health_summary"]["benchmark_time_matches_cumulative_dt"]
            is True
        )
        assert solver_result["stored_solver_config_hash"] == stable_hash_payload(
            solver_result["stored_solver_config"]
        )
        assert solver_result[
            "effective_constructor_config_hash"
        ] == stable_hash_payload(solver_result["effective_constructor_config"])
        assert (
            solver_result["dataset_rollout_control"]["status"]
            == "historical_config_recovered_matches_explicit_replay_control"
        )
        assert solver_result["dataset_rollout_control"]["source"] == (
            "historical_dataset_config_verified_by_explicit_replay_control"
        )
        assert (
            solver_result["dataset_rollout_control"][
                "historical_dataset_config_available"
            ]
            is True
        )
        assert solver_result["dataset_rollout_control"][
            "resolved_dataset_config_path"
        ] == str(fixture["dataset_config_path"])
        assert solver_result["dataset_rollout_control"][
            "original_dataset_config_path"
        ] == str(fixture["dataset_config_path"])
        assert (
            solver_result["dataset_rollout_control"][
                "historical_recovered_replay_control"
            ]
            == solver_result["dataset_rollout_control"]["replay_control"]
        )
        assert (
            solver_result["dataset_rollout_control"][
                "historical_recovered_replay_control_hash"
            ]
            == solver_result["dataset_rollout_control"]["replay_control_hash"]
        )


def test_run_dense_reference_validation_missing_historical_config_uses_explicit_replay_control(
    tmp_path: Path,
) -> None:
    fixture = _create_dense_validation_fixture(tmp_path)
    Path(fixture["dataset_config_path"]).unlink()
    config = load_config(fixture["config_path"])

    summary = run_dense_reference_validation(
        config,
        suite_name="dense_validation",
        config_path=fixture["config_path"],
        output_root_override=str(tmp_path / "missing-historical-results"),
    )

    scenario_lines = (
        Path(summary["artifacts_written"]["scenario_metrics_jsonl"])
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    scenario_record = json.loads(scenario_lines[0])
    assert scenario_record["all_legacy_reproduction_pass"] is True
    assert scenario_record["dataset_control_hash_match"] is True
    hashes = set()
    for solver_key in SOLVER_ORDER:
        dataset_control = scenario_record["solver_results"][solver_key][
            "dataset_rollout_control"
        ]
        hashes.add(dataset_control["hash"])
        assert (
            dataset_control["status"]
            == "historical_config_missing_explicit_replay_control"
        )
        assert dataset_control["source"] == "explicit_replay_control_fallback"
        assert dataset_control["historical_dataset_config_available"] is False
        assert dataset_control["resolved_dataset_config_path"] is None
        assert dataset_control["original_dataset_config_path"] == str(
            fixture["dataset_config_path"]
        )
        assert (
            "Could not resolve dataset_config_path"
            in dataset_control["historical_dataset_config_error"]
        )
        assert dataset_control["historical_recovered_replay_control"] is None
        assert dataset_control["historical_recovered_replay_control_hash"] is None
    assert len(hashes) == 1


def test_run_dense_reference_validation_historical_replay_control_mismatch_fails_loudly(
    tmp_path: Path,
) -> None:
    fixture = _create_dense_validation_fixture(tmp_path)
    config = load_config(fixture["config_path"])
    config["dense_reference_validation"]["replay_control"]["target_cfl"] = 0.41

    with pytest.raises(ValueError, match="Historical dataset replay control mismatch"):
        run_dense_reference_validation(
            config,
            suite_name="dense_validation",
            config_path=fixture["config_path"],
            output_root_override=str(tmp_path / "mismatch-results"),
        )


def test_run_dense_reference_validation_missing_historical_config_without_explicit_replay_control_fails(
    tmp_path: Path,
) -> None:
    fixture = _create_dense_validation_fixture(tmp_path)
    Path(fixture["dataset_config_path"]).unlink()
    config = load_config(fixture["config_path"])
    del config["dense_reference_validation"]["replay_control"]

    with pytest.raises(KeyError, match="replay_control"):
        run_dense_reference_validation(
            config,
            suite_name="dense_validation",
            config_path=fixture["config_path"],
            output_root_override=str(tmp_path / "missing-replay-control-results"),
        )


def test_run_dense_reference_validation_smoke_marks_suite_as_implementation_only(
    tmp_path: Path,
) -> None:
    fixture = _create_dense_validation_fixture(tmp_path)
    config = load_config(fixture["config_path"])

    summary = run_dense_reference_validation(
        config,
        suite_name="smoke",
        config_path=fixture["config_path"],
        output_root_override=str(tmp_path / "smoke-results"),
    )

    assert summary["suite"]["name"] == "smoke"
    assert summary["suite"]["manuscript_claims_allowed"] is False
    assert summary["suite"]["purpose"] == "implementation_only_smoke"
    assert summary["status"] == "pass"

    decision = json.loads(
        Path(summary["artifacts_written"]["decision_json"]).read_text(encoding="utf-8")
    )
    assert decision["decision_scope"] == "implementation_only"
    assert decision["criteria"]["zero_extrapolation"]["gating"] is True
    assert decision["criteria"]["all_legacy_reproduction_pass"]["gating"] is True
    assert (
        decision["criteria"][
            "aggregate_interp_rmse_over_smallest_aggregate_solver_gap"
        ]["gating"]
        is False
    )


def test_smoke_decision_fails_when_implementation_gate_fails() -> None:
    suite = resolve_validation_suite(
        {
            "smoke": {
                "label": "implementation_only_smoke",
                "ordered_scenarios": [{"scenario_id": "scenario_000001"}],
                "ordered_scenario_ids": ["scenario_000001"],
            }
        },
        "smoke",
    )
    decision = evaluate_dense_reference_criteria(
        suite_spec=suite,
        zero_extrapolation_failures=0,
        reproduction_failure_count=1,
        aggregate_interp_to_gap_ratio=0.0,
        scenario_ratio_median=0.0,
        scenario_ratio_p95=0.0,
        family_summaries=[],
        aggregate_interp_to_rms_ratio=0.0,
        scenario_rms_ratio_median=0.0,
        scenario_rms_ratio_p95=0.0,
        thresholds={
            "aggregate_ratio_max": 0.10,
            "scenario_ratio_median_max": 0.10,
            "scenario_ratio_p95_max": 0.25,
            "family_cell_median_max": 0.10,
            "family_cell_fraction_above_p95_max": 0.5,
            "interp_to_rms_aggregate_max": 0.01,
            "interp_to_rms_median_max": 0.01,
            "interp_to_rms_p95_max": 0.025,
        },
    )
    assert decision["decision_scope"] == "implementation_only"
    assert decision["status"] == "fail"
    assert decision["criteria"]["all_legacy_reproduction_pass"]["gating"] is True


def test_run_dense_reference_validation_skips_partial_interpolation_failures(
    tmp_path: Path,
) -> None:
    fixture = _create_dense_validation_fixture(tmp_path)
    config = load_config(fixture["config_path"])
    config["dense_reference_validation"]["common_time_grid"] = {
        "endpoint_tolerance": 1.0e-6,
        "values": [0.04, 0.12],
    }

    summary = run_dense_reference_validation(
        config,
        suite_name="dense_validation",
        config_path=fixture["config_path"],
        output_root_override=str(tmp_path / "zero-extrap-results"),
    )

    assert summary["status"] == "fail"
    assert summary["counts"]["eligible_for_interpolation_count"] == 0
    assert summary["counts"]["zero_extrapolation_failure_count"] == len(SOLVER_ORDER)
    assert summary["pairwise_dense_solver_gap_aggregate"]["rmse_by_pair"] == {}

    scenario_lines = (
        Path(summary["artifacts_written"]["scenario_metrics_jsonl"])
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    scenario_record = json.loads(scenario_lines[0])
    assert scenario_record["eligible_for_interpolation"] is False
    assert scenario_record["zero_extrapolation"] is False
    assert (
        scenario_record["interpolation_gate_reason"]
        == "incomplete_solver_interpolation"
    )
