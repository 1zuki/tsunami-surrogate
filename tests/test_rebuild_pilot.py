from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_rebuild_pilot import (
    BATHYMETRY_FAMILIES,
    CONFIGS,
    GAUGE_ARRIVAL_PEAK_FRACTION,
    LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS,
    PUBLICATION_RESOLUTION,
    RESOLUTION_CFL_FACTORS,
    SOURCE_FAMILIES,
    SplitSpec,
    _amplitude_cap_applied,
    _assert_split_contract,
    _bind_pilot_root,
    _block_mean_downsample,
    _comparison_metrics,
    _downsample_shared_master,
    _high_frequency_fraction,
    _path_contains_artifacts,
    _pilot_contract,
    _rollout_gate_summary,
    _select_gauge_positions,
    _select_cases,
    _select_mini_cases,
    _waveform_lag_metrics,
    TARGETED_GAUGE_DIAGNOSTIC_RESOLUTIONS,
    TARGETED_GAUGE_DIAGNOSTIC_SAMPLES,
    _threshold_crossing_frame,
)


def test_rebuild_configs_freeze_weak_amplitude_and_resolved_sources() -> None:
    for name in ("train", "confirmation", "test"):
        cfg = yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))
        assert cfg["dataset"]["source_strength_range"] == [0.15, 0.30]
        assert cfg["dataset"]["max_initial_eta_over_depth"] == 0.10
        assert cfg["requested_output"]["status"] == "accepted"
        assert cfg["requested_output"]["execution_scope"] == "production"
        assert cfg["requested_output"]["acknowledge_provisional"] is False
        assert cfg["paired_inputs"]["source_spectral_acceptance"] == {
            "enabled": True,
            "stage": "post_master_taper_solver",
            "reference_shape": [128, 128],
            "min_points_per_p90_wavelength": 32.0,
            "comparison_tolerance": 1.0e-9,
            "preserve_source_family": True,
            "max_attempts": 64,
        }

    source = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/data/rebuild/source_384.yaml"
        ).read_text(encoding="utf-8")
    )
    assert source["noise"]["enabled"] is False
    assert source["gaussian"]["sigma_range"][0] == 0.10
    assert source["dipole"]["sigma_range"][0] == 0.105
    assert source["fault"]["width_range"][0] == 0.20
    assert source["okada"]["width_range"][0] == 0.20
    assert source["okada"]["depth_range"][0] == 0.06


def test_amplitude_cap_classification_uses_stored_float32_values() -> None:
    raw_sampled = 0.234567891234
    stored_resolved = float(np.float32(raw_sampled))

    assert not _amplitude_cap_applied(stored_resolved, raw_sampled)
    assert _amplitude_cap_applied(0.20, raw_sampled)


def test_split_contract_checks_both_solver_tiers(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIGS["train"].read_text(encoding="utf-8"))
    cfg["paired_inputs"]["solver_shape"] = [64, 64]
    path = tmp_path / "bad_solver_tier.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="rebuild contract drifted"):
        _assert_split_contract(
            SplitSpec(
                name="train",
                seed=42,
                count=10000,
                config_path=path,
            )
        )


def test_refinement_resampling_preserves_publication_means_and_source_edge() -> None:
    rng = np.random.default_rng(42)
    field = rng.normal(size=(384, 384)).astype(np.float32)
    field[:6, :] = 0.0
    field[-6:, :] = 0.0
    field[:, :6] = 0.0
    field[:, -6:] = 0.0

    production = _downsample_shared_master(field, 128)
    refined = _downsample_shared_master(field, 192)
    coarse_publication = _block_mean_downsample(
        production,
        (PUBLICATION_RESOLUTION, PUBLICATION_RESOLUTION),
    )
    refined_publication = _block_mean_downsample(
        refined,
        (PUBLICATION_RESOLUTION, PUBLICATION_RESOLUTION),
    )

    assert refined.shape == (192, 192)
    assert np.count_nonzero(refined[[0, -1], :]) == 0
    assert np.count_nonzero(refined[:, [0, -1]]) == 0
    np.testing.assert_allclose(
        refined_publication,
        coarse_publication,
        rtol=0.0,
        atol=2.0e-7,
    )


def test_constant_field_has_finite_zero_high_frequency_fraction() -> None:
    field = np.ones((16, 16), dtype=np.float64)
    assert _high_frequency_fraction(field) == 0.0


def test_boundary_compatible_spectrum_distinguishes_ramp_and_checkerboard() -> None:
    coordinate = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
    ramp = np.repeat(coordinate[:, None], 64, axis=1)
    checkerboard = np.indices((64, 64)).sum(axis=0) % 2

    assert _high_frequency_fraction(ramp) < 1.0e-4
    assert _high_frequency_fraction(checkerboard) > 0.9


def test_threshold_crossing_uses_subframe_interpolation() -> None:
    signal = np.asarray([0.0, 0.25, 0.75, 1.0])
    assert _threshold_crossing_frame(signal, 0.5) == pytest.approx(2.5)


def test_waveform_lag_detects_shift_without_amplitude_threshold_bias() -> None:
    target = np.sin(np.linspace(0.0, 4.0 * np.pi, 50))
    candidate = np.concatenate([np.zeros(2), 1.5 * target[:-2]])

    lag = _waveform_lag_metrics(candidate, target)

    assert lag is not None
    assert lag[0] == pytest.approx(2.0)
    assert lag[1] > 0.99


def test_gauge_positions_use_fixed_5x5_interior_lattice() -> None:
    positions = _select_gauge_positions((64, 64))

    assert len(positions) == 25
    assert len(set(positions)) == 25
    assert {x for x, _ in positions} == {10, 21, 32, 42, 52}
    assert {y for _, y in positions} == {10, 21, 32, 42, 52}


def test_fixed_gauges_detect_two_frame_arrival_shift() -> None:
    eta0 = np.zeros((64, 64), dtype=np.float64)
    positions = _select_gauge_positions((64, 64))
    for x, y in positions:
        eta0[x, y] = 1.0
    target = np.repeat(eta0[None, ...], 50, axis=0)
    candidate = target.copy()
    for x, y in positions:
        target[10:, x, y] += 1.0
        candidate[12:, x, y] += 1.0

    metrics = _comparison_metrics(
        candidate,
        target,
        eta0,
        eta0,
        kind="synthetic",
        split="train",
        sample_index=1,
        solver="swe_hydrostatic",
    )

    assert metrics["eligible_arrival_gauges"] == len(positions)
    assert metrics["gauge_arrival_frame_p95"] == pytest.approx(2.0)
    assert metrics["gauge_arrival_frame_max"] == pytest.approx(2.0)


def test_weak_gauge_cannot_raise_arrival_threshold_to_its_local_peak() -> None:
    eta0 = np.zeros((64, 64), dtype=np.float64)
    eta0[0, 0] = 1.0
    target = np.repeat(eta0[None, ...], 50, axis=0)
    candidate = target.copy()
    strong, weak = _select_gauge_positions((64, 64))[:2]
    target[10:, strong[0], strong[1]] = 1.0
    candidate[10:, strong[0], strong[1]] = 1.0
    target[10:, weak[0], weak[1]] = 0.02
    candidate[30:, weak[0], weak[1]] = 0.02

    metrics = _comparison_metrics(
        candidate,
        target,
        eta0,
        eta0,
        kind="synthetic",
        split="train",
        sample_index=1,
        solver="swe_muscl_hr",
    )

    assert metrics["eligible_arrival_gauges"] == 1
    assert metrics["gauge_arrival_frame_max"] == pytest.approx(0.0)
    assert metrics["gauge_arrival_peak_fraction"] == pytest.approx(
        GAUGE_ARRIVAL_PEAK_FRACTION
    )
    assert metrics["gauge_eligible_target_peak_min"] == pytest.approx(0.1)


def test_framewise_error_uses_stable_initial_rms() -> None:
    eta0 = np.ones((16, 16), dtype=np.float64)
    target = np.full((4, 16, 16), 1.0e-3, dtype=np.float64)
    candidate = target + 1.0e-4

    metrics = _comparison_metrics(
        candidate,
        target,
        eta0,
        eta0,
        kind="synthetic",
        split="train",
        sample_index=1,
        solver="swe_hydrostatic",
    )

    assert metrics["per_frame_normalization"] == "target_initial_rms"
    assert metrics["per_frame_initial_nrmse_max"] == pytest.approx(
        1.0e-4
    )


def test_spectrum_gate_uses_cfl_and_refinement_stability() -> None:
    health = [
        {
            "nan_count": 0,
            "inf_count": 0,
            "quality_status": "ok",
            "quality_violations": [],
            "max_post_step_cfl": 0.2,
            "target_cfl": 0.2,
        }
    ]
    common = {
        "trajectory_rel_l2": 0.01,
        "per_frame_initial_nrmse_p95": 0.01,
        "per_frame_initial_nrmse_max": 0.02,
        "gauge_nrmse_p95": 0.01,
        "gauge_nrmse_max": 0.02,
        "gauge_peak_relative_error_p95": 0.01,
        "gauge_peak_relative_error_max": 0.02,
        "gauge_arrival_frame_p95": 0.0,
        "gauge_arrival_frame_max": 0.0,
        "gauge_waveform_lag_frame_p95": 0.0,
        "gauge_waveform_lag_frame_max": 0.0,
        "gauge_waveform_correlation_p05": 1.0,
        "gauge_waveform_correlation_min": 1.0,
        "eligible_arrival_gauges": 5,
        "eligible_waveform_lag_gauges": 5,
        "eligible_waveform_gauges": 5,
        "gauge_time_to_peak_frame_p95": 0.0,
        "gauge_time_to_peak_frame_max": 0.0,
    }
    comparisons = [
        {**common, "kind": "half_cfl_production"},
        {**common, "kind": "refinement_publication_128_to_192"},
    ]
    spectra = [
        {
            "split": "train",
            "sample_index": 1,
            "solver": "swe_muscl_hr",
            "resolution": 128,
            "cfl_factor": 1.0,
            "trajectory_high_frequency_fraction_max": 0.040,
            "trajectory_high_frequency_growth": 0.040,
        },
        {
            "split": "train",
            "sample_index": 1,
            "solver": "swe_muscl_hr",
            "resolution": 128,
            "cfl_factor": 0.5,
            "trajectory_high_frequency_fraction_max": 0.041,
            "trajectory_high_frequency_growth": 0.041,
        },
        {
            "split": "train",
            "sample_index": 1,
            "solver": "swe_muscl_hr",
            "resolution": 192,
            "cfl_factor": 1.0,
            "trajectory_high_frequency_fraction_max": 0.030,
            "trajectory_high_frequency_growth": 0.030,
        },
    ]

    gate = _rollout_gate_summary(health, comparisons, spectra)

    assert gate["passed"]
    assert gate["local_amplitude_diagnostic_passed"]
    assert gate[
        "half_cfl_high_frequency_fraction_delta_max"
    ] == pytest.approx(0.001)
    assert gate[
        "refinement_high_frequency_fraction_excess_max"
    ] == pytest.approx(-0.010)


def test_local_amplitude_tail_is_reported_without_overriding_field_gate() -> None:
    health = [
        {
            "nan_count": 0,
            "inf_count": 0,
            "quality_status": "ok",
            "quality_violations": [],
            "max_post_step_cfl": 0.2,
            "target_cfl": 0.2,
        }
    ]
    common = {
        "trajectory_rel_l2": 0.01,
        "per_frame_initial_nrmse_p95": 0.01,
        "per_frame_initial_nrmse_max": 0.02,
        "gauge_nrmse_p95": 0.20,
        "gauge_nrmse_max": 0.25,
        "gauge_peak_relative_error_p95": 0.20,
        "gauge_peak_relative_error_max": 0.25,
        "gauge_arrival_frame_p95": 0.0,
        "gauge_arrival_frame_max": 0.0,
        "gauge_waveform_lag_frame_p95": 0.0,
        "gauge_waveform_lag_frame_max": 0.0,
        "gauge_waveform_correlation_p05": 0.99,
        "gauge_waveform_correlation_min": 0.99,
        "eligible_arrival_gauges": 5,
        "eligible_waveform_lag_gauges": 5,
        "eligible_waveform_gauges": 5,
        "gauge_time_to_peak_frame_p95": 0.0,
        "gauge_time_to_peak_frame_max": 0.0,
    }
    comparisons = [
        {**common, "kind": "half_cfl_production"},
        {**common, "kind": "refinement_publication_128_to_192"},
    ]
    spectra = [
        {
            "split": "train",
            "sample_index": 1,
            "solver": "swe_hydrostatic",
            "resolution": resolution,
            "cfl_factor": factor,
            "trajectory_high_frequency_fraction_max": 0.01,
            "trajectory_high_frequency_growth": 0.01,
        }
        for resolution, factor in ((128, 1.0), (128, 0.5), (192, 1.0))
    ]

    gate = _rollout_gate_summary(health, comparisons, spectra)

    assert gate["passed"]
    assert not gate["local_amplitude_diagnostic_passed"]
    assert set(gate["diagnostic_checks"]) == set(
        LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS
    )
    assert not any(gate["diagnostic_checks"].values())
    assert all(
        name not in gate["promotion_checks"]
        for name in LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS
    )

    comparisons[1]["gauge_waveform_lag_frame_max"] = 3.0
    blocked = _rollout_gate_summary(health, comparisons, spectra)
    assert not blocked["passed"]
    assert not blocked["promotion_checks"]["gauge_waveform_lag_max"]


def test_pilot_contract_freezes_nonblocking_local_amplitude_policy() -> None:
    policy = _pilot_contract()["promotion_policy"]
    assert policy["local_amplitude_checks"] == list(
        LOCAL_AMPLITUDE_DIAGNOSTIC_CHECKS
    )
    assert policy["gauge_position_policy"] == "fixed_5x5_lattice"
    assert policy["confirmation_is_independent"] is True


def test_case_selection_uses_two_distinct_extremes_per_family_cell() -> None:
    rows = []
    sample_index = 1
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            for offset in range(3):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "bathymetry_type": bathymetry,
                        "source_type": source,
                        "high_frequency_fraction": 0.01 * offset,
                        "kh_p90": 0.02 * offset,
                        "local_eta_over_depth_max": 0.03 * (2 - offset),
                        "points_per_p90_wavelength": 10.0 + offset,
                        "temporal_samples_per_p90_period": 3.0 + offset,
                        "bathymetry_gradient_p99": 0.04 * (2 - offset),
                        "source_support_min_depth": 1.0 + offset,
                        "amplitude_cap_applied": offset == 0,
                    }
                )
                sample_index += 1

    selected = _select_cases(rows)

    assert len(selected) == 60
    assert {
        row["selection_reason"] for row in selected
    } == {
        "spectrum_temporal_extreme",
        "amplitude_bathymetry_extreme",
    }
    assert len({int(row["sample_index"]) for row in selected}) == 60


def test_mini_selection_keeps_one_tail_and_control_per_source_family() -> None:
    rows = []
    sample_index = 1
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            for reason, ppw in (
                ("spectrum_temporal_extreme", 32.0 + sample_index),
                ("amplitude_bathymetry_extreme", 64.0 + sample_index),
            ):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "bathymetry_type": bathymetry,
                        "source_type": source,
                        "selection_reason": reason,
                        "points_per_p90_wavelength": ppw,
                        "temporal_samples_per_p90_period": ppw / 4.0,
                    }
                )
                sample_index += 1

    selected = _select_mini_cases(rows)

    assert len(selected) == 12
    for source in SOURCE_FAMILIES:
        family = [row for row in selected if row["source_type"] == source]
        assert {row["selection_reason"] for row in family} == {
            "spectrum_temporal_extreme",
            "amplitude_bathymetry_extreme",
        }


def test_confirmation_case_selection_uses_one_case_per_family_cell() -> None:
    rows = []
    sample_index = 1
    for bathymetry in BATHYMETRY_FAMILIES:
        for source in SOURCE_FAMILIES:
            for offset in range(2):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "bathymetry_type": bathymetry,
                        "source_type": source,
                        "high_frequency_fraction": 0.01 * offset,
                        "kh_p90": 0.02 * offset,
                        "local_eta_over_depth_max": 0.03 * offset,
                        "points_per_p90_wavelength": 12.0 - offset,
                        "temporal_samples_per_p90_period": 4.0 - offset,
                        "bathymetry_gradient_p99": 0.04 * offset,
                        "source_support_min_depth": 2.0 - offset,
                        "amplitude_cap_applied": offset == 1,
                    }
                )
                sample_index += 1

    selected = _select_cases(rows, cases_per_cell=1)

    assert len(selected) == 30
    assert {
        row["selection_reason"] for row in selected
    } == {"combined_extreme"}


def test_pilot_root_rejects_unbound_or_different_artifacts(
    tmp_path: Path,
) -> None:
    contract = {"schema_id": "test", "contract_hash": "a"}
    clean = tmp_path / "clean"
    _bind_pilot_root(clean, contract)
    _bind_pilot_root(clean, contract)
    with pytest.raises(RuntimeError, match="different code/config contract"):
        _bind_pilot_root(
            clean,
            {"schema_id": "test", "contract_hash": "b"},
        )

    unbound = tmp_path / "unbound"
    unbound.mkdir()
    (unbound / "artifact.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="has no contract"):
        _bind_pilot_root(unbound, contract)


def test_empty_directory_scaffolding_is_not_a_final_test_artifact(
    tmp_path: Path,
) -> None:
    scaffold = tmp_path / "raw/hydrostatic/samples"
    scaffold.mkdir(parents=True)
    assert not _path_contains_artifacts(tmp_path / "raw")

    (scaffold / "sample.npz").write_bytes(b"artifact")
    assert _path_contains_artifacts(tmp_path / "raw")


def test_targeted_gauge_diagnostic_uses_the_two_outliers_and_384_reference() -> None:
    assert TARGETED_GAUGE_DIAGNOSTIC_SAMPLES == (727, 6398)
    assert TARGETED_GAUGE_DIAGNOSTIC_RESOLUTIONS == (192, 384)
    assert 384 not in RESOLUTION_CFL_FACTORS
