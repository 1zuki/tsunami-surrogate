from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data_gen.common_time_v2 import (
    authoritative_input_fingerprint,
    candidate_requested_times,
    hash_array,
    parse_requested_output_config,
    validate_operational_shard,
    validate_publication,
    write_operational_shard_manifest,
)
from src.data_gen.simulate_dataset import (
    AuthoritativeInputsConfig,
    DatasetConfig,
    QualityPolicy,
    RolloutResult,
    _validate_authoritative_input,
    _write_requested_publication,
)
from src.data_gen.operational_timing import (
    GenerationTimingRecorder,
    effective_worker_count,
    summarize_generation_timings,
    validate_generation_timing,
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


def test_generation_timing_is_separate_and_fails_closed_on_shard_corruption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw"
    output.mkdir()
    shard = output / "operational_shards" / "train_000001_000001.json"
    write_operational_shard_manifest(
        shard,
        split="train",
        start_index=1,
        stop_index=1,
        contract_hash_value="contract",
        publication_hashes={"train:scenario_000001:swe_hydrostatic": "publication"},
        complete=True,
        solver_names=["swe_hydrostatic"],
        resolved_config_hashes={"swe_hydrostatic": "config"},
        code_state_hash="code",
    )
    recorder = GenerationTimingRecorder(
        output_dir=output,
        split="train",
        contract_hash="contract",
        code_state_hash="code",
        config_path=tmp_path / "config.yaml",
        config_sha256="config-sha256",
        solver_names=["swe_hydrostatic"],
        requested_workers=1,
        requested_max_in_flight=1,
        operational_config={"storage_class": "test"},
    )
    recorder.begin_range(
        start_index=1,
        stop_index=1,
        planned_scenarios=1,
        resume=False,
        allow_override=False,
    )
    recorder.record_sample(
        {
            "_operational": {
                "sample_index": 1,
                "scenario_id": "scenario_000001",
                "worker_pid": 1,
                "input_load_s": 0.01,
                "worker_total_s": 0.25,
                "solvers": [
                    {
                        "solver": "swe_hydrostatic",
                        "status": "generated",
                        "solve_s": 0.2,
                        "serialization_s": 0.04,
                        "validation_s": 0.0,
                        "worker_s": 0.24,
                        "natural_steps": 12,
                    }
                ],
            }
        }
    )
    recorder.set_shard_manifest(shard)
    path = recorder.finalize(status="complete")
    assert path is not None
    payload = validate_generation_timing(path)
    assert payload["counts"]["generated_solver_rollouts"] == 1
    assert payload["per_solver"]["swe_hydrostatic"]["natural_steps"] == 12
    summary = summarize_generation_timings(output)
    assert summary["complete_invocations"] == 1
    assert summary["counts"]["accepted_solver_rollouts"] == 1
    assert summary["accepted_artifacts"] == {
        "unique_shards": 1,
        "unique_scenarios": 1,
        "unique_solver_rollouts": 1,
    }
    assert summary["aggregate_solver_worker_hours"] > 0.0

    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        validate_generation_timing(path)


def test_effective_worker_count_matches_execution_limits() -> None:
    assert effective_worker_count(10, 20, logical_cpu_count=8) == 8
    assert effective_worker_count(10, 3, logical_cpu_count=8) == 3
    assert effective_worker_count(10, 0, logical_cpu_count=8) == 0
    with pytest.raises(ValueError, match="requested_workers"):
        effective_worker_count(0, 3, logical_cpu_count=8)


def test_authoritative_input_validation_is_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    bathymetry = np.full((2, 2), -1.0, dtype=np.float32)
    source = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    strength = np.asarray([0.5], dtype=np.float32)
    rest_depth = np.maximum(-bathymetry, 0.0).astype(np.float32)
    eta0 = np.asarray(float(strength[0]) * source, dtype=np.float32)
    initial_depth = np.asarray(np.maximum(rest_depth + eta0, 0.0), dtype=np.float32)
    free_surface0 = np.asarray(initial_depth + bathymetry, dtype=np.float32)
    arrays = {
        "bathymetry": bathymetry,
        "source_field": source,
        "rest_depth": rest_depth,
        "eta0": eta0,
        "initial_depth": initial_depth,
        "free_surface0": free_surface0,
    }
    record = {
        "split": "train",
        "scenario_id": "scenario_000001",
        "qualified_id": "train:scenario_000001",
        "sample_index": 1,
        "bathymetry_type": "slope",
        "source_type": "gaussian",
        "source_strength": float(strength[0]),
        "array_hashes": {name: hash_array(value) for name, value in arrays.items()},
        "input_fingerprint": authoritative_input_fingerprint(
            split="train",
            sample_index=1,
            scenario_id="scenario_000001",
            bathymetry_type="slope",
            source_type="gaussian",
            source_strength=strength,
            arrays=arrays,
        ),
    }
    config = AuthoritativeInputsConfig(
        inventory_path=tmp_path / "inventory.jsonl",
        inventory_sha256="inventory",
        h0_contract_hash="h0",
    )
    provenance = _validate_authoritative_input(
        record=record,
        split="train",
        sample_idx=1,
        scenario_id="scenario_000001",
        bathymetry=bathymetry,
        source_field=source,
        source_strength_array=strength,
        bathymetry_type="slope",
        source_type="gaussian",
        sea_level_offset=0.0,
        config=config,
    )
    assert provenance["input_fingerprint"] == record["input_fingerprint"]

    corrupted = source.copy()
    corrupted[0, 0] += np.float32(0.01)
    with pytest.raises(RuntimeError, match="array hash mismatch: source_field"):
        _validate_authoritative_input(
            record=record,
            split="train",
            sample_idx=1,
            scenario_id="scenario_000001",
            bathymetry=bathymetry,
            source_field=corrupted,
            source_strength_array=strength,
            bathymetry_type="slope",
            source_type="gaussian",
            sea_level_offset=0.0,
            config=config,
        )
