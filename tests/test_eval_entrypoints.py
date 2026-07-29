from __future__ import annotations

import numpy as np

from scripts.eval_accuracy import _resolve_metrics_output_path
from scripts.eval_solver_speed import _prepare_scenario, _rollout_solver
from src.data_gen.common_time_v2 import parse_requested_output_config
from src.data_gen.simulate_dataset import BufferedDomainConfig


def test_accuracy_output_override_is_exact():
    cfg = {
        "output_dir": "experiments/fno",
        "eval": {"output_dir": "experiments/fno/eval"},
    }

    assert str(_resolve_metrics_output_path(cfg, "results/seed_18.json")) == (
        "results/seed_18.json"
    )
    assert str(_resolve_metrics_output_path(cfg, None)) == (
        "experiments/fno/eval/metrics.json"
    )


def test_solver_speed_prepares_production_buffer_shape_and_taper():
    bathymetry = np.full((64, 64), -1.0, dtype=np.float32)
    source = np.ones((64, 64), dtype=np.float32)
    scenario = {
        "sample_index": 1,
        "bathymetry": bathymetry,
        "source_field": source,
        "source_strength": 0.5,
    }
    buffered = BufferedDomainConfig(
        enabled=True,
        buffer_cells=16,
        source_taper_cells=8,
        bathymetry_extension="edge",
        output_crop="central",
    )

    prepared = _prepare_scenario(
        scenario, sea_level_offset=0.0, buffered_domain=buffered
    )

    assert prepared["input_shape"] == (64, 64)
    assert prepared["solver_shape"] == (96, 96)
    assert prepared["solver_bathymetry"].shape == (96, 96)
    assert prepared["solver_eta0"].shape == (96, 96)
    assert np.max(np.abs(prepared["solver_eta0"][:16])) == 0.0
    assert np.max(np.abs(prepared["solver_eta0"][-16:])) == 0.0
    assert prepared["solver_eta0"][16, 16] == 0.0
    assert prepared["solver_eta0"][23, 23] == 0.5


def test_solver_speed_requested_mode_uses_common_time_contract(monkeypatch):
    requested = parse_requested_output_config(
        {
            "enabled": True,
            "status": "accepted",
            "execution_scope": "production",
            "split": "train",
            "max_natural_steps": 20000,
            "collect_natural_step_health": True,
            "eta_primary": True,
        }
    )
    assert requested is not None
    captured = {}

    def fake_requested_rollout(solver, **kwargs):
        captured.update(kwargs)
        times = np.asarray(kwargs["requested_times"], dtype=np.float64)
        diagnostics = {"total_natural_steps": np.asarray([634], dtype=np.int64)}
        return (
            np.zeros((times.size, 1, 2, 2), dtype=np.float32),
            times,
            np.full(634, 0.175 / 634, dtype=np.float64),
            diagnostics,
        )

    monkeypatch.setattr(
        "scripts.eval_solver_speed._simulate_requested_times_local",
        fake_requested_rollout,
    )

    summary = _rollout_solver(
        object(),
        n_steps=250,
        auto_dt=True,
        target_cfl=0.35,
        requested_output=requested,
    )

    assert np.array_equal(captured["requested_times"], requested.requested_times)
    assert captured["max_natural_steps"] == 20000
    assert captured["collect_natural_step_health"] is True
    assert summary["natural_steps"] == 634
    assert summary["published_frames"] == 50
    assert summary["final_requested_time"] == 0.175
