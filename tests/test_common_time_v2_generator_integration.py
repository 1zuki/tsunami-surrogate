from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import sha256_file, validate_operational_shard
from src.data_gen.preprocess import TsunamiPreprocessor
from src.data_gen.simulate_dataset import TsunamiDatasetBuilder


def _write_fixture_configs(tmp_path: Path, *, requested: bool) -> Path:
    bathy = {
        "nx": 8,
        "ny": 8,
        "seed": 1,
        "base": {"slope_range": [0.0, 0.02], "kind": "slope"},
        "gaussian": {
            "enabled": False,
            "range": [0, 0],
            "amp_range": [0, 0],
            "sigma_range": [0.1, 0.1],
        },
        "ridges": {
            "enabled": False,
            "range": [0, 0],
            "amp_range": [0, 0],
            "len_scale": [0.1, 0.1],
        },
        "noise": {"enabled": False, "scale_range": [0, 0], "smoothing_sigma": [1, 1]},
        "normalization": {"depth_min": -1.1, "depth_max": -0.9},
        "terrain": {"warp_scale": 0.0, "warp_sigma": 1.0, "bias_strength": 0.02},
    }
    source = {
        "nx": 8,
        "ny": 8,
        "seed": 2,
        "source_type": ["gaussian"],
        "gaussian": {
            "enabled": True,
            "amp_range": [0.01, 0.01],
            "sigma_range": [0.15, 0.15],
            "num_range": [1, 1],
        },
        "multi": {"enabled": False, "num_sources": [1, 1]},
        "dipole": {"enabled": False},
        "fault": {"enabled": False},
        "rough": {"enabled": False},
        "okada": {"enabled": False},
        "noise": {"enabled": False, "scale_range": [0, 0], "smoothing_sigma": [1, 1]},
        "normalization": {"mode": "none", "clip_output": True, "height_scale": [-1, 1]},
    }
    bathy_path = tmp_path / "bathy.yaml"
    source_path = tmp_path / "source.yaml"
    bathy_path.write_text(yaml.safe_dump(bathy), encoding="utf-8")
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    cfg = {
        "configs": {"bathymetry": str(bathy_path), "source": str(source_path)},
        "dataset": {
            "num_samples": 2,
            "seed": 17,
            "num_workers": 1,
            "n_steps": 6,
            "save_every": 2,
            "auto_dt": True,
            "target_cfl": 0.45,
            "include_initial_state": True,
            "sea_level_offset": 0.0,
            "source_strength_range": [0.05, 0.05],
            "bathymetry_dir": str(tmp_path / "cache" / "bathymetry"),
            "source_dir": str(tmp_path / "cache" / "sources"),
            "output_dir": str(tmp_path / "raw"),
            "manifest_path": str(tmp_path / "manifests" / "scenario.jsonl"),
            "copy_configs": False,
        },
        "fdes": {
            "enabled": ["swe_hydrostatic", "swe_muscl_hr", "boussinesq"],
            "primary": "swe_hydrostatic",
        },
        "quality": {
            "on_violation": "fail",
            "reject_nonfinite": True,
            "max_abs_eta_limit": 5.0,
            "max_eta_over_depth": 1.0,
            "require_cg_converged": True,
        },
        "solver": {
            "nx": 8,
            "ny": 8,
            "dx": 0.125,
            "dy": 0.125,
            "dt": 0.001,
            "g": 1.0,
            "cfl": 0.45,
            "dry_tolerance": 1e-6,
            "max_velocity": 30.0,
            "boundary": "periodic",
            "use_sponge": False,
            "alpha": 1 / 3,
            "min_depth": 1e-4,
            "depth_scale": 1.0,
            "mode": "linear_variable_depth",
            "filter_strength": 0.0,
            "linear_solver_tol": 1e-8,
            "linear_solver_max_iter": 100,
            "check_finite": True,
        },
    }
    if requested:
        cfg["requested_output"] = {
            "enabled": True,
            "split": "train",
            "start": 0.0035,
            "step": 0.0035,
            "count": 50,
            "horizon": 0.175,
            "max_natural_steps": 1000,
            "collect_natural_step_health": True,
            "eta_primary": True,
            "debug_full_states": False,
            "acknowledge_provisional": True,
        }
        cfg["solver_profiles"] = {
            "swe_hydrostatic": {
                "cfl": 0.45,
                "sponge_time_mode": "elapsed_time_consistent",
                "sponge_reference_dt": 0.0035,
            },
            "swe_muscl_hr": {
                "cfl": 0.45,
                "sponge_time_mode": "elapsed_time_consistent",
                "sponge_reference_dt": 0.0035,
            },
            "boussinesq": {
                "cfl": 0.35,
                "sponge_time_mode": "elapsed_time_consistent",
                "sponge_reference_dt": 0.0035,
                "filter_time_mode": "disabled",
                "filter_reference_dt": 0.0035,
                "cg_failure_mode": "strict_v2",
            },
        }
    path = tmp_path / ("requested.yaml" if requested else "legacy.yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _publications(root: Path) -> dict[str, tuple[str, int]]:
    out = {}
    for path in sorted(root.glob("*/samples/sample_*/publication.json")):
        out[str(path.relative_to(root))] = (sha256_file(path), path.stat().st_mtime_ns)
    return out


def test_real_requested_range_interrupt_resume_is_stable_and_fail_closed(
    tmp_path: Path,
) -> None:
    config = _write_fixture_configs(tmp_path, requested=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/make_dataset.py",
            "--config",
            str(config),
            "--stop-at",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    before = _publications(tmp_path / "raw")
    assert len(before) == 3

    subprocess.run(
        [
            sys.executable,
            "scripts/make_dataset.py",
            "--config",
            str(config),
            "--stop-at",
            "2",
            "--continue",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    after = _publications(tmp_path / "raw")
    assert len(after) == 6
    assert all(after[key] == value for key, value in before.items())
    shard = tmp_path / "raw" / "operational_shards" / "train_000001_000002.json"
    payload = json.loads(shard.read_text(encoding="utf-8"))
    validate_operational_shard(
        shard,
        expected_contract_hash=payload["contract_hash"],
        expected_publication_hashes={
            row["qualified_id"]: row["publication_hash"]
            for row in payload["publications"]
        },
        expected_split="train",
        expected_start_index=1,
        expected_stop_index=2,
        expected_solver_names=["boussinesq", "swe_hydrostatic", "swe_muscl_hr"],
        expected_config_hashes=payload["resolved_config_hashes"],
        expected_code_state_hash=payload["code_state_hash"],
    )

    time.sleep(0.01)
    subprocess.run(
        [
            sys.executable,
            "scripts/make_dataset.py",
            "--config",
            str(config),
            "--stop-at",
            "2",
            "--continue",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    assert _publications(tmp_path / "raw") == after

    sample = (
        tmp_path / "raw" / "hydrostatic" / "samples" / "sample_000001" / "sample.npz"
    )
    sample.write_bytes(sample.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt or incompatible"):
        TsunamiDatasetBuilder(str(config)).run(continue_from_last=True, stop_at=2)
    assert sample.exists()


def test_full_legacy_generation_and_preprocess_contract(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_fixture_configs(tmp_path, requested=False)
    import src.data_gen.simulate_dataset as simulate_dataset

    def forbidden(*_args, **_kwargs):
        raise AssertionError("requested-output path was invoked")

    monkeypatch.setattr(simulate_dataset, "_simulate_requested_times_local", forbidden)
    builder = TsunamiDatasetBuilder(str(config))
    builder.run()
    for folder in ("hydrostatic", "muscl_hr", "boussinesq"):
        sample_dir = tmp_path / "raw" / folder / "samples" / "sample_000001"
        assert {
            "sample.npz",
            "rollout.npz",
            "trajectory_eta.npy",
            "meta.json",
        }.issubset({p.name for p in sample_dir.iterdir()})
        with np.load(sample_dir / "sample.npz", allow_pickle=False) as payload:
            assert payload["timestamps"].dtype == np.float32
            assert payload["dt_history"].dtype == np.float32
            assert payload["trajectory_eta"].shape[0] == 4
            np.testing.assert_array_equal(payload["timestamps"][0], np.float32(0.0))

    preprocess = {
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "manifest_path": str(tmp_path / "manifests" / "scenario.jsonl"),
        "raw": {
            "scenario_manifest": str(tmp_path / "manifests" / "scenario.jsonl"),
            "fde_manifests": {
                name: str(tmp_path / "manifests" / f"{folder}_manifest.jsonl")
                for name, folder in (
                    ("swe_hydrostatic", "hydrostatic"),
                    ("swe_muscl_hr", "muscl_hr"),
                    ("boussinesq", "boussinesq"),
                )
            },
        },
        "fde": {
            "mode": "separate_all",
            "targets": ["swe_hydrostatic", "swe_muscl_hr", "boussinesq"],
        },
        "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 3},
        "input": {
            "use_bathymetry": True,
            "use_source": True,
            "use_initial_depth": True,
        },
        "target": {
            "mode": "multi_step",
            "variable": "eta",
            "forecast_steps": 3,
            "stride": 1,
        },
        "normalization": {
            "method": "standardize",
            "channels": {"bathymetry": True, "source": True, "trajectory": True},
        },
        "saving": {"sharded": False, "include_meta": True},
        "eval_export": {"enabled": True},
    }
    pp = tmp_path / "preprocess.yaml"
    pp.write_text(yaml.safe_dump(preprocess, sort_keys=False), encoding="utf-8")
    TsunamiPreprocessor(str(pp)).run()
    for name in ("hydrostatic", "muscl_hr", "boussinesq"):
        root = tmp_path / "processed" / name
        targets = np.load(root / "train" / "Y.npy")
        assert targets.shape == (2, 3, 8, 8)
        assert (root / "normalization_stats.json").is_file()

    before = {
        path: sha256_file(path)
        for path in (tmp_path / "raw").glob("*/samples/sample_*/*")
        if path.is_file()
    }
    builder.run(continue_from_last=True, start_at=1, stop_at=2)
    assert before == {path: sha256_file(path) for path in before}
