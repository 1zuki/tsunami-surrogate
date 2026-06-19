from __future__ import annotations

import numpy as np

from src.data_gen.simulate_dataset import (
    QualityPolicy,
    RolloutResult,
    _compute_rollout_health,
    _quality_violations_for_health,
)


def test_boussinesq_health_uses_effective_depth_for_min_h_and_eta_ratio() -> None:
    eta = np.asarray(
        [
            [[0.12, -0.15], [0.02, -0.03]],
            [[0.20, -0.06], [0.04, -0.01]],
        ],
        dtype=np.float32,
    )
    eta_t = np.zeros_like(eta)
    trajectory = np.stack([eta, eta_t], axis=1)
    rollout = RolloutResult(
        trajectory=trajectory,
        trajectory_eta=eta,
        timestamps=np.asarray([0.0, 1.0], dtype=np.float32),
        dt_history=np.asarray([0.0, 1.0], dtype=np.float32),
        diagnostics={
            "cg_failed_count": np.asarray([0], dtype=np.int32),
            "cg_max_iterations": np.asarray([12], dtype=np.int32),
            "cg_max_residual_ratio": np.asarray([1.0e-8], dtype=np.float32),
        },
    )

    rest_depth = np.full((2, 2), 10.0, dtype=np.float32)
    effective_depth = np.full((2, 2), 0.1, dtype=np.float32)

    health = _compute_rollout_health(
        fde_name="boussinesq",
        rollout=rollout,
        rest_depth=rest_depth,
        effective_depth=effective_depth,
    )

    assert np.isclose(health["min_h"], -0.05)
    assert np.isclose(health["max_abs_eta_over_depth"], 2.0)
    assert health["cg_failed_count"] == 0
    assert np.isclose(health["cg_converged_fraction"], 1.0)


def test_quality_flags_boussinesq_eta_ratio_and_cg_failure() -> None:
    policy = QualityPolicy(
        on_violation="fail",
        reject_nonfinite=True,
        min_h_tolerance=-1.0e-6,
        max_abs_eta_limit=10.0,
        max_velocity_limit=50.0,
        max_eta_over_depth=1.0,
        require_cg_converged=True,
    )
    health = {
        "fde_name": "boussinesq",
        "nan_count": 0,
        "inf_count": 0,
        "min_h": 0.02,
        "max_abs_eta": 0.2,
        "max_abs_velocity": float("nan"),
        "max_abs_eta_over_depth": 1.4,
        "has_cg_diagnostics": True,
        "cg_failed_count": 1,
    }

    violations = _quality_violations_for_health(health, policy)

    assert any("max_abs_eta_over_depth" in item for item in violations)
    assert any("cg_failed_count" in item for item in violations)
