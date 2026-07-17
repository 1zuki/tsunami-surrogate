from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from src.evaluation.boussinesq_boundary import (
    SpectralPacketSpec,
    build_reference_packet,
    directional_states,
    discrete_dispersion,
    discrete_energy,
    evolve_reference,
    packet_timing,
)
from src.evaluation.common_time_v2_level_a import (
    _boussinesq_reference_boundary_metrics,
    _boussinesq_reference_crop,
    _boussinesq_reference_refinement_error,
    _boussinesq_h0_exposure_metrics,
    _boussinesq_spectral_packet_bundle,
    _build_level_a_task_plan,
    _execute_level_a_task_plan,
    _make_level_a_task,
    _resolved_boundary_packet_spec,
    _scientific_digest,
)


def _level_a_config() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/eval/common_time_v2_level_a.yaml").read_text(encoding="utf-8")
    )


def _packet_config() -> dict[str, object]:
    config = _level_a_config()
    return _resolved_boundary_packet_spec(config["boundary_packet"], "boussinesq")


def test_discrete_dispersion_matches_face_flux_symbol_and_group_derivative() -> None:
    dx = 0.125
    k = np.asarray([0.25, 0.75, 1.5, 2.5], dtype=np.float64)
    omega, phase, group = discrete_dispersion(k, dx=dx, depth=1.0)
    kd = 2.0 * np.sin(0.5 * k * dx) / dx
    expected = np.sqrt(9.81 * kd**2 / (1.0 + kd**2 / 3.0))
    np.testing.assert_allclose(omega, expected, rtol=1.0e-15, atol=0.0)
    np.testing.assert_allclose(phase, omega / k, rtol=1.0e-15, atol=0.0)
    step = 1.0e-6
    upper = discrete_dispersion(k + step, dx=dx, depth=1.0)[0]
    lower = discrete_dispersion(k - step, dx=dx, depth=1.0)[0]
    np.testing.assert_allclose(group, (upper - lower) / (2.0 * step), rtol=2.0e-8)


def test_reference_packet_is_deterministic_zero_mode_free_and_one_way() -> None:
    spec = SpectralPacketSpec()
    finite_a, reference_a, metadata_a = build_reference_packet(spec)
    finite_b, reference_b, metadata_b = build_reference_packet(spec)
    np.testing.assert_array_equal(finite_a, finite_b)
    np.testing.assert_array_equal(reference_a, reference_b)
    assert metadata_a == metadata_b
    assert abs(float(np.mean(reference_a[0]))) < 1.0e-20
    right, left = directional_states(reference_a[None], dx=spec.dx, depth=spec.depth)
    right_energy = discrete_energy(right[0], dx=spec.dx, dy=spec.dy, depth=spec.depth)
    left_energy = discrete_energy(left[0], dx=spec.dx, dy=spec.dy, depth=spec.depth)
    assert right_energy / left_energy < 1.0e-24
    assert np.max(np.abs(finite_a[0])) == pytest.approx(spec.amplitude)


def test_spectral_support_uses_group_velocity_and_fails_short_reference() -> None:
    spec = SpectralPacketSpec()
    _finite, _reference, metadata = build_reference_packet(spec)
    timing = packet_timing(spec, metadata)
    assert metadata["group_velocity_min"] < metadata["group_velocity_max"]
    assert timing["leading_edge_arrival_time"] < timing["trailing_edge_exit_time"]
    assert timing["observation_end_time"] > timing["trailing_edge_exit_time"]
    assert timing["reference_safe"] is True

    unsafe = replace(spec, reference_length=64.0)
    _finite, _reference, metadata = build_reference_packet(unsafe)
    assert packet_timing(unsafe, metadata)["reference_safe"] is False


def test_large_reference_crop_is_coordinate_aligned_and_propagates_finitely() -> None:
    spec = SpectralPacketSpec()
    finite, reference, metadata = build_reference_packet(spec)
    initial_crop = _boussinesq_reference_crop(
        reference,
        np.asarray([0.0]),
        spec=spec,
        metadata=metadata,
    )[0]
    np.testing.assert_allclose(initial_crop, finite, rtol=0.0, atol=2.0e-19)
    evolved = evolve_reference(reference, np.asarray([0.0, 0.175]), spec=spec)
    assert evolved.shape == (2, 2, reference.shape[1], spec.ny)
    assert np.isfinite(evolved).all()


def test_reference_refinement_is_below_proposed_uncertainty_allowance() -> None:
    packet = _packet_config()
    spec, _finite, reference, metadata, _timing = _boussinesq_spectral_packet_bundle(
        packet, role="production_horizon"
    )
    error = _boussinesq_reference_refinement_error(
        spec=spec,
        coarse_reference=reference,
        coarse_metadata=metadata,
        times=np.asarray([0.0035, 0.0875, 0.175]),
    )
    assert math.isfinite(error)
    assert error < 2.0e-3


def test_h0_exposure_detects_initial_sponge_overlap_and_reachability() -> None:
    eta = np.zeros((8, 8), dtype=np.float64)
    eta[0, 4] = 1.0
    sponge = np.ones_like(eta)
    sponge[:2] = 0.9
    metrics = _boussinesq_h0_exposure_metrics(
        eta,
        np.ones_like(eta),
        sponge,
        horizon=0.175,
    )
    assert metrics["significant_source_overlaps_sponge"] is True
    assert metrics["conservative_boundary_reachable"] is True
    assert metrics["initial_sponge_energy_fraction"] == 1.0


def test_reflection_metric_fails_closed_when_exit_horizon_is_missing() -> None:
    packet = _packet_config()
    spec, _finite, reference, metadata, timing = _boussinesq_spectral_packet_bundle(
        packet, role="reflection"
    )
    full_times = np.asarray(timing["requested_times"], dtype=np.float64)
    short_times = full_times[full_times < float(timing["trailing_edge_exit_time"])]
    exact = _boussinesq_reference_crop(
        reference,
        short_times,
        spec=spec,
        metadata=metadata,
    )
    metrics = _boussinesq_reference_boundary_metrics(
        candidate=exact,
        timestamps=short_times,
        packet_spec=packet,
        role="reflection",
        sponge_width=spec.nx // 8,
        reference_refinement_error_ratio=0.0,
        uncertainty_fraction=0.1,
        reflected_energy_ceiling=0.02,
        production_error_ceiling=0.10,
        precision_floor_safety_factor=64.0,
    )
    assert metrics["measurement_temporally_separated"] is False
    assert metrics["reflection_metrics_valid"] is False
    assert metrics["spectral_exit_horizon_achieved"] is False


def test_production_boundary_task_is_serial_parallel_deterministic(
    tmp_path: Path,
) -> None:
    config = _level_a_config()
    canaries = [
        {
            "qualified_id": f"train:scenario_{index:06d}",
            "input_fingerprint": str(index),
        }
        for index in range(6)
    ]
    tasks = _build_level_a_task_plan(
        config,
        canaries,
        contract_hash="protocol-test",
        code_state_hash="protocol-code",
    )
    task = next(
        task
        for task in tasks
        if task["kind"] == "boundary"
        and task["spec"]["solver"] == "boussinesq"
        and task["spec"]["boundary_role"] == "production_horizon"
        and task["spec"]["variant"] == "zero_gradient_no_sponge"
    )
    task = _make_level_a_task(
        ordinal=0,
        task_id=task["task_id"],
        kind=task["kind"],
        spec=task["spec"],
        contract_hash=task["contract_hash"],
        code_state_hash=task["code_state_hash"],
    )
    serial, _ = _execute_level_a_task_plan(
        [task], tasks_root=tmp_path / "serial", workers=1
    )
    parallel, _ = _execute_level_a_task_plan(
        [task], tasks_root=tmp_path / "parallel", workers=2, max_in_flight=1
    )
    assert serial[0]["row"] | {"runtime_s": 0.0} == parallel[0]["row"] | {
        "runtime_s": 0.0
    }
    np.testing.assert_array_equal(serial[0]["trajectory"], parallel[0]["trajectory"])
    assert _scientific_digest(serial) == _scientific_digest(parallel)
