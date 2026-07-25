from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import (
    build_candidate_contract,
    contract_hash,
    validate_generation_contract_artifact,
)
from src.data_gen.generation_sequence import (
    STAGES,
    _load_yaml,
    _validate_candidate_policy,
    _validate_requested_time_provenance,
    execution_policy,
    freeze_generation_contract,
    preflight_stage,
    resolve_stage_config,
    validate_stage_prerequisites,
    write_stage_attestation,
)
from src.data_gen.simulate_dataset import (
    TsunamiDatasetBuilder,
    _generation_resolved_config_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/data/dataset.yaml"


def _policy() -> dict[str, object]:
    return execution_policy(
        workers=8,
        max_in_flight=8,
        cloud_provider="google-cloud",
        cloud_zone="asia-southeast1-b",
        machine_type="c4-highcpu-test",
        storage_class="persistent-disk-ssd",
        hourly_cost_usd=1.25,
    )


def _resolved_rehearsal(tmp_path: Path) -> dict[str, object]:
    return resolve_stage_config(
        base_config=BASE_CONFIG,
        stage=STAGES["rehearsal"],
        input_root=tmp_path / "inputs",
        output_dir=tmp_path / "rehearsal" / "raw",
        manifest_path=tmp_path / "rehearsal" / "manifests" / "scenario.jsonl",
        policy=_policy(),
    )


def _verification(
    resolved: dict[str, object], code_state_hash: str
) -> dict[str, object]:
    dataset = TsunamiDatasetBuilder._parse_dataset_section(resolved)
    solver = TsunamiDatasetBuilder._parse_solver_section(resolved)
    return {
        "verified": True,
        "stage": "rehearsal",
        "scenario_count": 1,
        "solver_publication_count": 3,
        "split": "train",
        "start_index": 1,
        "stop_index": 1,
        "generation_contract_hash": None,
        "generated_solver_rollouts": 3,
        "reused_solver_rollouts": 0,
        "output_contract_hash": contract_hash(
            build_candidate_contract(status="provisional")
        ),
        "code_state_hash": code_state_hash,
        "resolved_config_hashes": _generation_resolved_config_hashes(
            dataset, solver
        ),
        "snapshot_sha256": "snapshot",
        "operational_shard_sha256": "shard",
        "operational_timing_sha256": "timing",
        "publication_set_hash": "publications",
        "machine_fingerprint": {
            "platform": "linux",
            "machine": "x86_64",
            "python_version": "3.10",
            "numpy_version": "2.2.6",
            "cpu_model": "test-cpu",
            "logical_cpu_count": 8,
            "memory_total_bytes": 1024,
            "thread_environment": {},
            "cloud_provider": "google-cloud",
            "cloud_zone": "asia-southeast1-b",
            "machine_type": "c4-highcpu-test",
            "storage_class": "persistent-disk-ssd",
            "storage_total_bytes": 2048,
        },
        "trajectory_values_inspected": False,
    }


def test_legacy_config_is_rejected_before_generation() -> None:
    legacy = _load_yaml(
        ROOT / "configs/data/legacy/dataset_saved_step_v1.yaml"
    )
    with pytest.raises(ValueError, match="not common-time-v2"):
        _validate_candidate_policy(legacy)


def test_requested_time_provenance_requires_adjacent_natural_steps(
    tmp_path: Path,
) -> None:
    requested = np.arange(1, 51, dtype=np.float64) * 0.0035
    provenance_path = tmp_path / "provenance.npz"
    np.savez_compressed(
        provenance_path,
        requested_timestamps=requested,
        left_natural_timestamps=requested,
        right_natural_timestamps=requested,
        interpolation_weights=np.zeros(50, dtype=np.float64),
        bracket_widths=np.zeros(50, dtype=np.float64),
        exact_knot=np.ones(50, dtype=np.bool_),
        natural_step_indices=np.arange(1, 51, dtype=np.int64),
        natural_dt_history=np.full(50, 0.0035, dtype=np.float64),
        total_natural_steps=np.asarray([50], dtype=np.int64),
    )
    with np.load(provenance_path, allow_pickle=False) as provenance:
        _validate_requested_time_provenance(
            provenance, expected_times=requested
        )

    with np.load(provenance_path, allow_pickle=False) as provenance:
        corrupted = {name: provenance[name] for name in provenance.files}
    corrupted["natural_step_indices"] = np.arange(2, 52, dtype=np.int64)
    np.savez_compressed(provenance_path, **corrupted)
    with np.load(provenance_path, allow_pickle=False) as provenance:
        with pytest.raises(RuntimeError, match="missing natural step"):
            _validate_requested_time_provenance(
                provenance, expected_times=requested
            )


def test_mass_stage_requires_accepted_generation_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an accepted generation contract"):
        resolve_stage_config(
            base_config=BASE_CONFIG,
            stage=STAGES["train-1"],
            input_root=tmp_path / "inputs",
            output_dir=tmp_path / "raw",
            manifest_path=tmp_path / "manifest.jsonl",
            policy=_policy(),
        )


def test_preflight_rejects_stale_solver_manifest_before_input_scan(
    tmp_path: Path,
) -> None:
    resolved = _resolved_rehearsal(tmp_path)
    manifest_path = tmp_path / "rehearsal" / "manifests" / "scenario.jsonl"
    solver_manifest = manifest_path.parent / "hydrostatic_manifest.jsonl"
    solver_manifest.parent.mkdir(parents=True)
    solver_manifest.write_text('{"sample_index": 1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Manifest already contains"):
        preflight_stage(
            resolved_config=resolved,
            stage=STAGES["rehearsal"],
            input_root=tmp_path / "inputs",
            output_dir=tmp_path / "rehearsal" / "raw",
            manifest_path=manifest_path,
        )


def test_rehearsal_freezes_contract_and_mass_config_binds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = _resolved_rehearsal(tmp_path)
    fake_state = {
        "git_commit": "commit",
        "dirty": False,
        "source_inventory_hash": "source",
        "source_file_count": 1,
        "code_state_hash": "code-state",
    }
    artifact_dir = freeze_generation_contract(
        rehearsal_config=resolved,
        rehearsal_verification=_verification(resolved, "code-state"),
        artifact_root=tmp_path / "contracts",
        repo_state=fake_state,
    )
    contract_path = artifact_dir / "generation_contract.json"
    artifact = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_generation_contract_artifact(
        contract_path, expected_hash=artifact["contract_hash"]
    )

    production = resolve_stage_config(
        base_config=BASE_CONFIG,
        stage=STAGES["train-1"],
        input_root=tmp_path / "inputs",
        output_dir=tmp_path / "train" / "raw",
        manifest_path=tmp_path / "train" / "manifest.jsonl",
        policy=_policy(),
        generation_contract_path=contract_path,
        generation_contract_hash_value=artifact["contract_hash"],
    )
    assert production["requested_output"]["status"] == "accepted"
    assert production["requested_output"]["execution_scope"] == "production"
    assert production["generation_contract"]["contract_hash"] == artifact[
        "contract_hash"
    ]
    parsed = TsunamiDatasetBuilder._parse_dataset_section(production)
    assert parsed.seed == 42
    assert parsed.num_samples == 10000
    assert parsed.generation_contract is not None

    monkeypatch.setattr(
        "src.data_gen.simulate_dataset.code_state", lambda _: fake_state
    )
    resolved_path = tmp_path / "accepted-production.yaml"
    resolved_path.write_text(
        yaml.safe_dump(production, sort_keys=False), encoding="utf-8"
    )
    builder = TsunamiDatasetBuilder(str(resolved_path))
    assert builder.dataset.generation_contract is not None
    assert builder.dataset.generation_contract.contract_hash == artifact[
        "contract_hash"
    ]


def test_generation_contract_corruption_fails_closed(tmp_path: Path) -> None:
    resolved = _resolved_rehearsal(tmp_path)
    artifact_dir = freeze_generation_contract(
        rehearsal_config=resolved,
        rehearsal_verification=_verification(resolved, "code-state"),
        artifact_root=tmp_path / "contracts",
        repo_state={"dirty": False, "code_state_hash": "code-state"},
    )
    contract_path = artifact_dir / "generation_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["decision"]["mass_generation_authorized"] = False
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        validate_generation_contract_artifact(contract_path)


def test_stage_order_is_enforced_by_content_hashed_attestations(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="Missing prerequisite"):
        validate_stage_prerequisites(
            stage=STAGES["train-1"],
            attestation_paths=[],
            generation_contract_hash_value="contract",
        )
    validation_attestation = write_stage_attestation(
        output_dir=tmp_path,
        verification={
            "verified": True,
            "stage": "validation",
            "generation_contract_hash": "contract",
        },
    )
    observed = validate_stage_prerequisites(
        stage=STAGES["train-1"],
        attestation_paths=[validation_attestation],
        generation_contract_hash_value="contract",
    )
    assert set(observed) == {"validation"}

    with pytest.raises(RuntimeError, match="Final-test generation remains blocked"):
        validate_stage_prerequisites(
            stage=STAGES["test"],
            attestation_paths=[],
            generation_contract_hash_value="contract",
        )
