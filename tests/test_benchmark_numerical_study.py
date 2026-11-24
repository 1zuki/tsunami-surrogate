from pathlib import Path

import numpy as np
import pytest

from scripts import run_benchmark_numerical_study as study


def test_nested_resolution_preserves_frozen_cell_averages():
    baseline = np.arange(36, dtype=np.float64).reshape(6, 6)
    fine = study._resample(baseline, old_core=4, new_core=8)
    coarse = study._resample(baseline, old_core=4, new_core=2)
    np.testing.assert_array_equal(study._resample(fine, old_core=8, new_core=4), baseline)
    np.testing.assert_array_equal(coarse, baseline.reshape(3, 2, 3, 2).mean(axis=(1, 3)))
    assert np.mean(coarse) == np.mean(baseline) == np.mean(fine)


def test_steady_reference_satisfies_subcritical_bernoulli_branch():
    bed = -2.0 + 0.2 * np.cos(2.0 * np.pi * np.arange(128) / 128.0)
    depth = study._steady_depth(bed, discharge=0.2, head=0.0, gravity=9.81)
    energy = depth + bed + 0.2**2 / (2.0 * 9.81 * depth**2)
    np.testing.assert_allclose(energy, 0.0, atol=1e-15)
    assert np.max(0.2 / (depth * np.sqrt(9.81 * depth))) < 1.0
    with pytest.raises(ValueError, match="subcritical"):
        study._steady_depth(bed, discharge=100.0, head=0.0, gravity=9.81)


def test_trajectory_metrics_report_inactive_frames_without_dividing_by_zero():
    candidate = np.array([[[1.0, 0.0]], [[1.0, 3.0]]])
    reference = np.array([[[0.0, 0.0]], [[2.0, 2.0]]])
    metrics = study._metrics(candidate, reference)
    assert metrics["trajectory_relative_l2"] == pytest.approx(np.sqrt(3 / 8))
    assert metrics["trajectory_rmse"] == pytest.approx(np.sqrt(3 / 4))
    assert metrics["per_time_relative_l2"] == [None, 0.5]


def test_state_budget_includes_kinetic_bed_and_hydrostatic_terms():
    initial = np.array([[[2.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
    later = np.array([[[2.0, 0.0]], [[2.0, 0.0]], [[0.0, 0.0]]])
    bed = -initial[0]
    cfg = {"dx": 2.0, "dy": 2.0, "g": 10.0, "dry_tolerance": 1.0e-6}
    budget = study._state_budget(later[None], initial, bed, cfg)
    assert budget["initial_water_volume"] == 12
    assert budget["initial_mechanical_energy_per_density"] == -100
    assert budget["water_volume_change_by_time"] == [-4.0]
    assert budget["mechanical_energy_per_density_change_by_time"] == [24.0]


def test_checkpoint_resume_validates_plan_and_coverage(tmp_path: Path, monkeypatch):
    task = {
        "id": "scenario_000001__swe_hydrostatic__baseline",
        "core": 4,
        "moving_equilibrium": {"transverse_cells": 4},
        "variant": "baseline",
        "publication_core": 2,
        "times": [8.4, 16.8],
    }
    plan = {
        "requested_timestamps": task["times"],
        "complete_full_scope": False,
        "full_scope_rollout_count": 72,
        "roster": [],
        "code_and_input_files": [],
    }
    solves = []

    def solve(task):
        solves.append(task["id"])
        return np.zeros((2, 2, 2), dtype=np.float64), {"quality": "fixture"}

    monkeypatch.setattr(study, "_solve", solve)
    output = tmp_path / "result"
    original = study._execute(output, plan, [task], resume=False, workers=1)
    assert original["rollout_count"] == 1
    assert original["complete_full_planned_study"] is False
    assert study._execute(output, plan, [task], resume=True, workers=1) == original
    assert solves == [task["id"]]
    with pytest.raises(ValueError, match="frozen plan"):
        study._execute(output, {**plan, "full_scope_rollout_count": 73}, [task], resume=True, workers=1)
    with pytest.raises(FileExistsError):
        study._execute(output, plan, [task], resume=False, workers=1)

    (output / "checkpoints/extra.npz").touch()
    with pytest.raises(ValueError, match="Unexpected checkpoint"):
        study._execute(output, plan, [task], resume=True, workers=1)


def test_checkpoint_rejects_changed_trajectory(tmp_path: Path):
    task = {
        "id": "test",
        "core": 4,
        "moving_equilibrium": {"transverse_cells": 4},
        "variant": "baseline",
        "publication_core": 2,
        "times": [8.4],
    }
    study._write_checkpoint(task, tmp_path, "plan", np.ones((1, 2, 2)), {"quality": "fixture"})
    saved = tmp_path / "test.npz"
    with np.load(saved, allow_pickle=False) as payload:
        record = payload["record"]
        record_sha256 = payload["record_sha256"]
    np.savez(saved, eta=np.zeros((1, 2, 2)), record=record, record_sha256=record_sha256)
    with pytest.raises(ValueError, match="does not match"):
        study._read_checkpoint(saved, task, "plan")
