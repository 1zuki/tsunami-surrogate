from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data_gen.common_time_v2 import (
    candidate_requested_times,
    parse_requested_output_config,
    validate_operational_shard,
    validate_publication,
    write_operational_shard_manifest,
)
from src.data_gen.simulate_dataset import (
    DatasetConfig,
    QualityPolicy,
    RolloutResult,
    _write_requested_publication,
)


def _dataset(tmp_path: Path, *, debug: bool = False) -> DatasetConfig:
    requested = parse_requested_output_config(
        {
            "enabled": True,
            "split": "train",
            "start": 0.0035,
            "step": 0.0035,
            "count": 50,
            "horizon": 0.175,
            "max_natural_steps": 1000,
            "collect_natural_step_health": True,
            "eta_primary": True,
            "debug_full_states": debug,
            "acknowledge_provisional": True,
        }
    )
    assert requested is not None
    policy = QualityPolicy(
        on_violation="fail",
        reject_nonfinite=True,
        min_h_tolerance=-1e-6,
        max_abs_eta_limit=5.0,
        max_velocity_limit=30.0,
        max_eta_over_depth=1.0,
        require_cg_converged=True,
    )
    return DatasetConfig(
        num_samples=1,
        seed=1,
        num_workers=1,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=0.45,
        include_initial_state=False,
        sea_level_offset=0.0,
        source_strength_range=(0.4, 0.8),
        output_dir=tmp_path / "out",
        bathymetry_dir=tmp_path / "bathy",
        source_dir=tmp_path / "sources",
        manifest_path=tmp_path / "manifest.jsonl",
        copy_configs=False,
        enabled_fdes=("swe_hydrostatic",),
        primary_fde="swe_hydrostatic",
        quality_policy=policy,
        requested_output=requested,
    )


def _publish(tmp_path: Path, *, debug: bool = False) -> tuple[Path, dict]:
    dataset = _dataset(tmp_path, debug=debug)
    times = candidate_requested_times()
    bathy = np.full((2, 2), -1.0, dtype=np.float32)
    source = np.full((2, 2), 0.1, dtype=np.float32)
    rest = np.ones((2, 2), dtype=np.float32)
    eta0 = np.full((2, 2), 0.05, dtype=np.float32)
    h0 = rest + eta0
    free = h0 + bathy
    trajectory = np.zeros((50, 3, 2, 2), dtype=np.float32)
    trajectory[:, 0] = h0
    diagnostics = {
        "requested_timestamps": times,
        "left_natural_timestamps": times - 0.001,
        "right_natural_timestamps": times + 0.001,
        "interpolation_weights": np.full(50, 0.5, dtype=np.float64),
        "bracket_widths": np.full(50, 0.002, dtype=np.float64),
        "exact_knot": np.zeros(50, dtype=np.bool_),
        "natural_step_indices": np.arange(1, 51, dtype=np.int64),
        "natural_dt_history": np.full(51, 0.0035, dtype=np.float64),
        "total_natural_steps": np.asarray([51], dtype=np.int64),
        "final_natural_timestamp": np.asarray([0.1785], dtype=np.float64),
        "post_step_cfl": np.full(51, 0.4, dtype=np.float64),
    }
    rollout = RolloutResult(
        trajectory=trajectory,
        trajectory_eta=trajectory[:, 0] + bathy,
        timestamps=times,
        dt_history=diagnostics["natural_dt_history"],
        diagnostics=diagnostics,
    )
    sample_dir = tmp_path / "sample_000001"
    result = _write_requested_publication(
        sample_dir=sample_dir,
        rollout=rollout,
        fde_name="swe_hydrostatic",
        dataset=dataset,
        solver_cfg={"nx": 2, "ny": 2, "dx": 0.5, "dy": 0.5, "dt": 0.1},
        sample_idx=1,
        scenario_id="scenario_000001",
        bathymetry=bathy,
        source_field=source,
        source_strength=0.5,
        rest_depth=rest,
        eta0=eta0,
        initial_depth=h0,
        free_surface0=free,
        bathymetry_type="slope",
        source_type="gaussian",
        health={"nan_count": 0, "inf_count": 0},
        quality_status="ok",
        quality_violations=[],
    )
    return sample_dir, result


def test_eta_primary_publication_and_optional_debug_state(tmp_path: Path) -> None:
    sample_dir, result = _publish(tmp_path)
    assert (sample_dir / "sample.npz").is_file()
    assert (sample_dir / "provenance.npz").is_file()
    assert not (sample_dir / "rollout.npz").exists()
    assert not (sample_dir / "debug_full_states.npz").exists()
    with np.load(sample_dir / "sample.npz", allow_pickle=False) as payload:
        assert "trajectory" not in payload
        assert payload["trajectory_eta"].shape == (50, 2, 2)
        assert payload["timestamps"].dtype == np.float64
        np.testing.assert_array_equal(
            payload["timestamps"], candidate_requested_times()
        )
    validate_publication(
        sample_dir,
        expected_contract_hash=result["publication"]["contract_hash"],
        expected_times=candidate_requested_times(),
    )

    debug_dir, _ = _publish(tmp_path / "debug", debug=True)
    assert (debug_dir / "debug_full_states.npz").is_file()


def test_publication_rejects_corruption_and_semantic_mismatch(tmp_path: Path) -> None:
    sample_dir, result = _publish(tmp_path)
    with pytest.raises(RuntimeError, match="contract_hash mismatch"):
        validate_publication(sample_dir, expected_contract_hash="bad")
    sample_path = sample_dir / "sample.npz"
    sample_path.write_bytes(sample_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        validate_publication(
            sample_dir,
            expected_contract_hash=result["publication"]["contract_hash"],
        )


def test_operational_shard_rejects_interrupted_or_mismatched_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.json"
    publications = {"train:scenario_000001:swe_hydrostatic": "abc"}
    write_operational_shard_manifest(
        path,
        split="train",
        start_index=1,
        stop_index=1,
        contract_hash_value="contract",
        publication_hashes=publications,
        complete=False,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_operational_shard(
            path,
            expected_contract_hash="contract",
            expected_publication_hashes=publications,
        )

    complete_path = tmp_path / "complete.json"
    write_operational_shard_manifest(
        complete_path,
        split="train",
        start_index=1,
        stop_index=1,
        contract_hash_value="contract",
        publication_hashes=publications,
        complete=True,
    )
    validate_operational_shard(
        complete_path,
        expected_contract_hash="contract",
        expected_publication_hashes=publications,
    )
    with pytest.raises(RuntimeError, match="contract hash mismatch"):
        validate_operational_shard(
            complete_path,
            expected_contract_hash="different",
            expected_publication_hashes=publications,
        )


def test_publication_refuses_overwrite(tmp_path: Path) -> None:
    _publish(tmp_path)
    with pytest.raises(FileExistsError):
        _publish(tmp_path)
