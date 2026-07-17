from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    RolloutResult,
    TsunamiDatasetBuilder,
    _crop_rollout,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _prepare_buffered_domain,
    _resolved_solver_cfg_for_fde,
)
from src.solver.operator_time import build_sponge_mask


def _config() -> BufferedDomainConfig:
    return BufferedDomainConfig(
        enabled=True,
        buffer_cells=16,
        source_taper_cells=8,
        bathymetry_extension="edge",
        output_crop="central",
    )


def test_96_domain_preserves_64_core_and_zeroes_external_source() -> None:
    bathymetry = -np.linspace(0.75, 10.0, 64 * 64, dtype=np.float32).reshape(
        64, 64
    )
    source = np.ones((64, 64), dtype=np.float32)
    prepared = _prepare_buffered_domain(
        bathymetry,
        source,
        source_strength=0.5,
        sea_level_offset=0.0,
        config=_config(),
    )
    crop = prepared["crop"]

    assert prepared["solver_bathymetry"].shape == (96, 96)
    assert prepared["solver_eta0"].shape == (96, 96)
    np.testing.assert_array_equal(prepared["solver_bathymetry"][crop], bathymetry)
    np.testing.assert_array_equal(
        prepared["solver_eta0"][crop], prepared["eta0"]
    )
    assert np.count_nonzero(prepared["solver_eta0"][:16]) == 0
    assert np.count_nonzero(prepared["solver_eta0"][-16:]) == 0
    assert np.count_nonzero(prepared["solver_eta0"][:, :16]) == 0
    assert np.count_nonzero(prepared["solver_eta0"][:, -16:]) == 0
    assert prepared["source_edge_max_abs"] == 0.0


def test_16_cell_cosine_sponge_is_exactly_one_on_64_crop() -> None:
    mask = build_sponge_mask(
        nx=96,
        ny=96,
        width=16,
        min_factor=0.8,
        axes="xy",
        profile="cosine",
    )
    np.testing.assert_array_equal(mask[16:80, 16:80], np.ones((64, 64)))
    assert float(np.min(mask)) == 0.8


def test_rollout_crop_preserves_time_and_diagnostics_without_dtype_change() -> None:
    trajectory = np.arange(2 * 3 * 96 * 96, dtype=np.float64).reshape(2, 3, 96, 96)
    eta = trajectory[:, 0]
    timestamps = np.asarray([0.0035, 0.007], dtype=np.float64)
    dt_history = np.asarray([0.001, 0.001], dtype=np.float64)
    diagnostics = {"natural_step_indices": np.asarray([1, 2], dtype=np.int64)}
    rollout = RolloutResult(trajectory, eta, timestamps, dt_history, diagnostics)

    cropped = _crop_rollout(rollout, (slice(16, 80), slice(16, 80)))

    assert cropped.trajectory.shape == (2, 3, 64, 64)
    assert cropped.trajectory_eta.shape == (2, 64, 64)
    assert cropped.trajectory.dtype == np.float64
    np.testing.assert_array_equal(cropped.trajectory, trajectory[..., 16:80, 16:80])
    assert cropped.timestamps is timestamps
    assert cropped.dt_history is dt_history
    assert cropped.diagnostics is diagnostics


def test_provisional_config_resolves_the_96_to_64_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    builder = TsunamiDatasetBuilder(
        str(root / "configs/data/common_time_v2_provisional.yaml")
    )

    assert (builder.solver_cfg["nx"], builder.solver_cfg["ny"]) == (96, 96)
    assert (builder.solver_cfg["dx"], builder.solver_cfg["dy"]) == (
        1.0 / 64.0,
        1.0 / 64.0,
    )
    assert builder.dataset.buffered_domain == _config()

    factories = {
        "swe_hydrostatic": _make_hydrostatic_solver_from_cfg,
        "swe_muscl_hr": _make_muscl_solver_from_cfg,
        "boussinesq": _make_boussinesq_solver_from_cfg,
    }
    for solver_name, factory in factories.items():
        resolved = _resolved_solver_cfg_for_fde(
            builder.solver_cfg, builder.dataset.solver_profiles, solver_name
        )
        solver = factory(resolved)
        assert solver.sponge_width == 16
        assert solver.sponge_profile == "cosine"
        np.testing.assert_array_equal(
            solver.sponge_mask[16:80, 16:80], np.ones((64, 64))
        )
        if solver_name == "boussinesq":
            assert solver.boundary_x == solver.boundary_y == "open"
            assert solver.filter_time_mode == "disabled"
            assert solver.linear_solver_tol == 1.0e-10
            assert solver.linear_solver_max_iter == 750
        else:
            assert solver.boundary_x == solver.boundary_y == "radiation"
