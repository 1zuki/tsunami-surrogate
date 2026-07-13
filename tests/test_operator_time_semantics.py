from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.solver.boussinesq import BoussinesqSolver
from src.solver.hydrostatic_swe import ShallowWaterSolver
from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver
from src.solver.operator_time import filter_coefficient, sponge_factor


def _swe(cls=ShallowWaterSolver, **kwargs):
    solver = cls(
        nx=8,
        ny=8,
        dx=0.125,
        dy=0.125,
        dt=0.01,
        boundary="open",
        use_sponge=True,
        sponge_width=2,
        sponge_min_factor=0.9,
        **kwargs,
    )
    bathymetry = -np.ones((8, 8), dtype=float)
    eta = np.zeros((8, 8), dtype=float)
    eta[2:6, 2:6] = 1.0e-3
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(1.0 + eta)
    return solver


def test_default_sponge_is_explicit_legacy_and_numerically_identical() -> None:
    default = _swe()
    explicit = _swe(sponge_time_mode="legacy_per_step")
    for _ in range(4):
        default.step(dt=0.01)
        explicit.step(dt=0.01)
    np.testing.assert_array_equal(default.get_state(), explicit.get_state())


def test_elapsed_sponge_accumulates_by_elapsed_time() -> None:
    mask = np.asarray([[0.9, 1.0]], dtype=float)
    half = sponge_factor(
        mask,
        dt=0.005,
        mode="elapsed_time_consistent",
        reference_dt=0.01,
    )
    np.testing.assert_allclose(half * half, mask, rtol=1e-15, atol=1e-15)
    with pytest.raises(ValueError, match="reference_dt"):
        _swe(sponge_time_mode="elapsed_time_consistent")


def test_filter_modes_have_explicit_reference_semantics() -> None:
    assert filter_coefficient(
        0.01, dt=0.0035, mode="elapsed_time_consistent", reference_dt=0.0035
    ) == pytest.approx(0.01)
    assert filter_coefficient(
        0.01, dt=0.00175, mode="elapsed_time_consistent", reference_dt=0.0035
    ) == pytest.approx(0.005)
    assert filter_coefficient(
        0.2, dt=1.0, mode="disabled", reference_dt=None
    ) == 0.0
    with pytest.raises(ValueError, match=r"\[0, 0.25\]"):
        filter_coefficient(
            0.2, dt=0.01, mode="elapsed_time_consistent", reference_dt=0.001
        )


def test_muscl_counters_do_not_change_state() -> None:
    baseline = _swe(MUSCLHRShallowWaterSolver)
    observed = _swe(MUSCLHRShallowWaterSolver)
    baseline.step(dt=0.005)
    observed.step(dt=0.005)
    before_snapshot = observed.get_state().copy()
    diagnostics = observed.get_operator_diagnostics()
    observed.reset_operator_diagnostics()
    np.testing.assert_array_equal(baseline.get_state(), before_snapshot)
    np.testing.assert_array_equal(observed.get_state(), before_snapshot)
    assert diagnostics["muscl_limiter_total_count"] > 0
    assert diagnostics["nan_to_num_replacement_count"] == 0
    assert observed.get_operator_diagnostics()["muscl_limiter_total_count"] == 0


def test_elapsed_operator_diagnostics_record_reference_rates() -> None:
    solver = _swe(
        sponge_time_mode="elapsed_time_consistent",
        sponge_reference_dt=0.01,
    )
    diagnostics = solver.get_operator_diagnostics()
    rates = -np.log(solver.sponge_mask) / 0.01
    assert diagnostics["sponge_reference_decay_rate_min"] == pytest.approx(
        float(np.min(rates))
    )
    assert diagnostics["sponge_reference_decay_rate_max"] == pytest.approx(
        float(np.max(rates))
    )

    boussinesq = BoussinesqSolver(
        nx=8,
        ny=8,
        dx=0.125,
        dy=0.125,
        dt=0.001,
        boundary="open",
        use_sponge=True,
        sponge_width=2,
        sponge_min_factor=0.9,
        sponge_time_mode="elapsed_time_consistent",
        sponge_reference_dt=0.01,
        filter_strength=0.01,
        filter_time_mode="elapsed_time_consistent",
        filter_reference_dt=0.005,
    )
    diagnostics = boussinesq.get_operator_diagnostics()
    assert diagnostics["filter_reference_coefficient_rate"] == pytest.approx(2.0)


def test_boussinesq_default_modes_match_explicit_legacy() -> None:
    common = dict(
        nx=8,
        ny=8,
        dx=0.125,
        dy=0.125,
        dt=0.001,
        boundary="periodic",
        use_sponge=False,
        filter_strength=0.01,
    )
    default = BoussinesqSolver(**common)
    explicit = BoussinesqSolver(
        **common,
        sponge_time_mode="legacy_per_step",
        filter_time_mode="legacy_per_step",
        cg_failure_mode="legacy_posthoc",
    )
    bathymetry = -np.ones((8, 8), dtype=float)
    eta = np.cos(2 * np.pi * np.arange(8)[:, None] / 8) * 1.0e-4
    eta = np.broadcast_to(eta, (8, 8)).copy()
    for solver in (default, explicit):
        solver.set_bathymetry(bathymetry)
        solver.set_initial_condition(eta)
        solver.step(dt=0.001)
    np.testing.assert_array_equal(default.get_state(), explicit.get_state())


def test_strict_cg_failure_does_not_assign_failed_step(monkeypatch) -> None:
    solver = BoussinesqSolver(
        nx=8,
        ny=8,
        dx=0.125,
        dy=0.125,
        dt=0.001,
        boundary="periodic",
        use_sponge=False,
        cg_failure_mode="strict_v2",
    )
    solver.set_bathymetry(-np.ones((8, 8), dtype=float))
    initial = np.ones((8, 8), dtype=float) * 1.0e-4
    solver.set_initial_condition(initial)

    def fail(_eta=None):
        solver.last_cg_converged = False
        solver.last_cg_iterations = 1
        solver.last_cg_initial_residual = 1.0
        solver.last_cg_final_residual = 1.0
        return np.ones_like(solver.eta)

    monkeypatch.setattr(solver, "solve_acceleration", fail)
    before = solver.get_state().copy()
    with pytest.raises(RuntimeError, match="solve 0"):
        solver.step(dt=0.001)
    np.testing.assert_array_equal(solver.get_state(), before)


def test_strict_second_cg_failure_does_not_assign_failed_step(monkeypatch) -> None:
    solver = BoussinesqSolver(
        nx=8,
        ny=8,
        dx=0.125,
        dy=0.125,
        dt=0.001,
        boundary="periodic",
        use_sponge=False,
        cg_failure_mode="strict_v2",
    )
    solver.set_bathymetry(-np.ones((8, 8), dtype=float))
    solver.set_initial_condition(np.ones((8, 8), dtype=float) * 1.0e-4)
    calls = 0

    def fail_second(_eta=None):
        nonlocal calls
        calls += 1
        solver.last_cg_converged = calls == 1
        solver.last_cg_iterations = calls
        solver.last_cg_initial_residual = 1.0
        solver.last_cg_final_residual = 0.0 if calls == 1 else 1.0
        return np.ones_like(solver.eta)

    monkeypatch.setattr(solver, "solve_acceleration", fail_second)
    before = solver.get_state().copy()
    with pytest.raises(RuntimeError, match="solve 1"):
        solver.step(dt=0.001)
    np.testing.assert_array_equal(solver.get_state(), before)


def test_legacy_cg_failure_assigns_step_and_records_both_solves(monkeypatch) -> None:
    solver = BoussinesqSolver(
        nx=8,
        ny=8,
        dx=0.125,
        dy=0.125,
        dt=0.001,
        boundary="periodic",
        use_sponge=False,
        cg_failure_mode="legacy_posthoc",
    )
    solver.set_bathymetry(-np.ones((8, 8), dtype=float))
    solver.set_initial_condition(np.ones((8, 8), dtype=float) * 1.0e-4)
    calls = 0

    def fail_second(_eta=None):
        nonlocal calls
        calls += 1
        solver.last_cg_converged = calls == 1
        solver.last_cg_iterations = calls
        solver.last_cg_initial_residual = 2.0
        solver.last_cg_final_residual = 0.0 if calls == 1 else 1.0
        return np.ones_like(solver.eta)

    monkeypatch.setattr(solver, "solve_acceleration", fail_second)
    before = solver.get_state().copy()
    solver.step(dt=0.001)
    assert not np.array_equal(solver.get_state(), before)
    assert solver.last_step_cg_solve_converged == (True, False)
    diagnostics = solver.get_operator_diagnostics()
    assert diagnostics["cg_solve_count"] == 2
    assert diagnostics["cg_failure_count"] == 1
    assert diagnostics["cg_initial_residual_min"] == pytest.approx(2.0)
    assert diagnostics["cg_initial_residual_max"] == pytest.approx(2.0)
    assert diagnostics["cg_final_residual_min"] == pytest.approx(0.0)
    assert diagnostics["cg_final_residual_max"] == pytest.approx(1.0)
