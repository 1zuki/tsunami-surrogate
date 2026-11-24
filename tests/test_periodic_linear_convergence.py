import numpy as np
import pytest

from scripts import run_periodic_linear_convergence as wave


def test_mode_fit_wraps_against_unwrapped_reference_phase():
    kx = 2.0 * np.pi * (np.arange(32) + 0.5) / 32
    phase_exact = 9.84
    values = 0.001 * np.cos(kx[:, None] - phase_exact - 0.02)
    amplitude, _, phase_error = wave.measure_mode(values, kx, phase_exact)
    assert amplitude == pytest.approx(0.001)
    assert phase_error == pytest.approx(0.02)


@pytest.mark.parametrize("solver_name", wave.FACTORIES)
def test_periodic_wave_has_finite_phase_and_no_operator_corrections(solver_name):
    result = wave.run_wave(solver_name, grid=16, ny=8, cfl=0.225, amplitude=0.001, final_time=0.02)
    assert result["steps"] > 0
    assert 0 < result["relative_l2_error"] < 0.2
    assert abs(result["phase_error"]) < 0.1
    assert all(count == 0 for count in result["operator_health_counts"].values())


def test_order_requires_positive_errors_and_scale():
    assert wave.empirical_order(0.04, 0.01, 2.0) == pytest.approx(2.0)
    assert wave.empirical_order(0.02, 0.04, 2.0) == pytest.approx(-1.0)
    assert wave.empirical_order(0.01, 0.0, 2.0) is None
