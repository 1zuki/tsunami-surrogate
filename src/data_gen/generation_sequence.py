from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.data_gen.common_time_v2 import (
    ACCEPTED_STATUS,
    GENERATION_CONTRACT_SCHEMA_ID,
    PUBLICATION_SCHEMA_ID,
    build_candidate_contract,
    candidate_requested_times,
    code_state,
    contract_hash,
    generation_contract_hash,
    hash_array,
    sha256_file,
    split_qualified_identity,
    stable_hash_payload,
    validate_generation_contract_artifact,
    validate_operational_shard,
    validate_publication,
)
from src.data_gen.operational_timing import (
    THREAD_ENV_KEYS,
    machine_snapshot,
    validate_generation_timing,
)
from src.data_gen.simulate_dataset import (
    AuthoritativeInputsConfig,
    BufferedDomainConfig,
    TsunamiDatasetBuilder,
    _generation_resolved_config_hashes,
    _prepare_buffered_domain,
    _seed_for_sample,
    _validate_authoritative_input,
)


ROOT = Path(__file__).resolve().parents[2]
H0_CONTRACT_HASH = "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b"
H0_INVENTORY_SHA256 = (
    "c4f34bc504ca60d4b1c74fc88b7c6c239ae1dbbaa0f15f3bf2c501038eb438e3"
)
VALIDATION_EVIDENCE = {
    "level_a_contract_hash": (
        "be1af7dce1f48942e6d20a96bb06b1359655903847c7580954901e2dcfa3332b"
    ),
    "established_solver_evaluation_identity": (
        "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
    ),
    "h1_contract_hash": (
        "ef96c24f62a0eb0884f5384436a50802c0d8dd644946552d9c462b225334bc7d"
    ),
    "h2_contract_hash": (
        "46f4b22cb10f259ecae496060ce508fe86cb5898ac22b73518132b97a333e03a"
    ),
}
SOLVER_FOLDERS = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
}
REQUIRED_PROVENANCE = {
    "requested_timestamps",
    "left_natural_timestamps",
    "right_natural_timestamps",
    "interpolation_weights",
    "bracket_widths",
    "exact_knot",
    "natural_step_indices",
    "natural_dt_history",
    "total_natural_steps",
}
STAGE_ATTESTATION_SCHEMA_ID = (
    "tsunami-surrogate.common-time-v2.generation-stage-attestation.v1"
)
STAGE_PREREQUISITES = {
    "rehearsal": (),
    "validation": (),
    "train-1": ("validation",),
    "train-2": ("validation", "train-1"),
    "test": ("validation", "train-1", "train-2"),
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    split: str
    seed: int
    num_samples: int
    start_index: int
    stop_index: int
    requires_generation_contract: bool

    @property
    def scenario_count(self) -> int:
        return self.stop_index - self.start_index + 1


STAGES = {
    "rehearsal": StageSpec("rehearsal", "train", 42, 10000, 1, 1, False),
    "validation": StageSpec("validation", "eval", 69, 1000, 1, 1000, True),
    "train-1": StageSpec("train-1", "train", 42, 10000, 1, 5000, True),
    "train-2": StageSpec(
        "train-2", "train", 42, 10000, 5001, 10000, True
    ),
    "test": StageSpec("test", "test", 367, 2500, 1, 2500, True),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected mapping in {path}")
    return dict(raw)


def execution_policy(
    *,
    workers: int,
    max_in_flight: int,
    cloud_provider: str,
    cloud_zone: str,
    machine_type: str,
    storage_class: str,
    hourly_cost_usd: float | None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_in_flight < workers:
        raise ValueError("max_in_flight must be at least workers")
    values = {
        "cloud_provider": cloud_provider,
        "cloud_zone": cloud_zone,
        "machine_type": machine_type,
        "storage_class": storage_class,
    }
    for name, value in values.items():
        text = str(value).strip()
        lowered = text.lower()
        if (
            not text
            or lowered in {"local", "unknown", "todo", "placeholder"}
            or "replace" in lowered
            or "set_actual" in lowered
        ):
            raise ValueError(f"{name} must identify the real cloud allocation")
        values[name] = text
    if hourly_cost_usd is not None and (
        not np.isfinite(hourly_cost_usd) or hourly_cost_usd < 0.0
    ):
        raise ValueError("hourly_cost_usd must be finite and nonnegative")
    return {
        "num_workers": int(workers),
        "max_in_flight": int(max_in_flight),
        "solver_progress": True,
        **values,
        "hourly_cost_usd": hourly_cost_usd,
    }


def _validate_candidate_policy(raw: Mapping[str, Any]) -> None:
    requested = raw.get("requested_output")
    if not isinstance(requested, Mapping) or not bool(requested.get("enabled", False)):
        raise ValueError("Config is not common-time-v2 requested-output generation")
    authoritative = raw.get("authoritative_inputs")
    if not isinstance(authoritative, Mapping):
        raise ValueError("Config is missing authoritative_inputs")
    expected_authoritative = {
        "inventory_sha256": H0_INVENTORY_SHA256,
        "h0_contract_hash": H0_CONTRACT_HASH,
        "require_exact_arrays": True,
        "allow_input_generation": False,
    }
    for key, expected in expected_authoritative.items():
        if authoritative.get(key) != expected:
            raise ValueError(f"authoritative_inputs.{key} mismatch")
    domain = raw.get("computational_domain")
    expected_domain = {
        "enabled": True,
        "buffer_cells": 16,
        "source_taper_cells": 8,
        "bathymetry_extension": "edge",
        "output_crop": "central",
    }
    if domain != expected_domain:
        raise ValueError("computational_domain does not match accepted 96-to-64 policy")
    fdes = raw.get("fdes")
    if not isinstance(fdes, Mapping) or set(fdes.get("enabled", [])) != set(
        SOLVER_FOLDERS
    ):
        raise ValueError("All three accepted solvers must be enabled")
    expected_profiles = {
        "swe_hydrostatic": {
            "cfl": 0.1125,
            "boundary": "radiation",
            "sponge_time_mode": "elapsed_time_consistent",
            "sponge_reference_dt": 0.0035,
        },
        "swe_muscl_hr": {
            "cfl": 0.225,
            "boundary": "radiation",
            "sponge_time_mode": "elapsed_time_consistent",
            "sponge_reference_dt": 0.0035,
        },
        "boussinesq": {
            "cfl": 0.35,
            "boundary": "open",
            "depth_scale": 1.0,
            "sponge_time_mode": "elapsed_time_consistent",
            "sponge_reference_dt": 0.0035,
            "filter_time_mode": "disabled",
            "filter_reference_dt": 0.0035,
            "cg_failure_mode": "strict_v2",
        },
    }
    if raw.get("solver_profiles") != expected_profiles:
        raise ValueError("solver_profiles do not match accepted H2 policy")
    solver = raw.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("solver section is missing")
    expected_solver = {
        "nx": 96,
        "ny": 96,
        "dx": 0.015625,
        "dy": 0.015625,
        "use_sponge": True,
        "sponge_width": 16,
        "sponge_min_factor": 0.8,
        "sponge_axes": "xy",
        "sponge_profile": "cosine",
        "filter_strength": 0.0,
        "linear_solver_tol": 1.0e-10,
        "linear_solver_abs_tol": 0.0,
        "linear_solver_max_iter": 750,
    }
    for key, expected in expected_solver.items():
        if solver.get(key) != expected:
            raise ValueError(f"solver.{key} mismatch")


def resolve_stage_config(
    *,
    base_config: Path,
    stage: StageSpec,
    input_root: Path,
    output_dir: Path,
    manifest_path: Path,
    policy: Mapping[str, Any],
    generation_contract_path: Path | None = None,
    generation_contract_hash_value: str | None = None,
) -> dict[str, Any]:
    raw = _load_yaml(base_config)
    _validate_candidate_policy(raw)
    dataset = dict(raw.get("dataset", {}))
    dataset.update(
        {
            "num_samples": stage.num_samples,
            "seed": stage.seed,
            "num_workers": int(policy["num_workers"]),
            "bathymetry_dir": str((input_root / "bathymetry").resolve()),
            "source_dir": str((input_root / "sources").resolve()),
            "output_dir": str(output_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "copy_configs": True,
        }
    )
    raw["dataset"] = dataset
    requested = dict(raw["requested_output"])
    requested["split"] = stage.split
    requested["acknowledge_provisional"] = False
    if stage.requires_generation_contract:
        if generation_contract_path is None or generation_contract_hash_value is None:
            raise ValueError(
                f"Stage {stage.name} requires an accepted generation contract"
            )
        validate_generation_contract_artifact(
            generation_contract_path,
            expected_hash=generation_contract_hash_value,
        )
        requested["status"] = ACCEPTED_STATUS
        requested["execution_scope"] = "production"
        raw["generation_contract"] = {
            "path": str(generation_contract_path.resolve()),
            "contract_hash": generation_contract_hash_value,
        }
    else:
        requested["status"] = "provisional"
        requested["execution_scope"] = "preparation-only"
        raw.pop("generation_contract", None)
    raw["requested_output"] = requested
    raw["operations"] = {
        "enabled": True,
        "progress_every": 1,
        "solver_progress": bool(policy["solver_progress"]),
        "max_in_flight": int(policy["max_in_flight"]),
        "cloud_provider": policy["cloud_provider"],
        "cloud_zone": policy["cloud_zone"],
        "machine_type": policy["machine_type"],
        "storage_class": policy["storage_class"],
        "hourly_cost_usd": policy["hourly_cost_usd"],
    }
    return raw


def _inventory_records(
    inventory_path: Path, *, split: str
) -> dict[int, dict[str, Any]]:
    if sha256_file(inventory_path) != H0_INVENTORY_SHA256:
        raise RuntimeError("H0 inventory checksum mismatch")
    records: dict[int, dict[str, Any]] = {}
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if record.get("split") != split:
                continue
            index = int(record["sample_index"])
            if index in records:
                raise RuntimeError(f"Duplicate H0 inventory row for {split}:{index}")
            records[index] = record
    return records


def _validate_input_cache(
    *,
    input_root: Path,
    stage: StageSpec,
    inventory_path: Path,
) -> tuple[dict[int, dict[str, Any]], int]:
    records = _inventory_records(inventory_path, split=stage.split)
    if len(records) != stage.num_samples:
        raise RuntimeError(
            f"H0 inventory has {len(records)} {stage.split} rows; "
            f"expected {stage.num_samples}"
        )
    config = AuthoritativeInputsConfig(
        inventory_path=inventory_path,
        inventory_sha256=H0_INVENTORY_SHA256,
        h0_contract_hash=H0_CONTRACT_HASH,
    )
    audited = 0
    for index in range(stage.start_index, stage.stop_index + 1):
        bathymetry_path = input_root / "bathymetry" / f"sample_{index:06d}.npz"
        source_path = input_root / "sources" / f"sample_{index:06d}.npz"
        if not bathymetry_path.is_file():
            raise FileNotFoundError(bathymetry_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with np.load(bathymetry_path, allow_pickle=False) as payload:
            bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
            bathymetry_type = str(np.asarray(payload["bathymetry_type"]).reshape(-1)[0])
            bathymetry_seed = int(np.asarray(payload["sample_seed"]).reshape(-1)[0])
        with np.load(source_path, allow_pickle=False) as payload:
            source_field = np.asarray(payload["source_field"], dtype=np.float32)
            source_type = str(np.asarray(payload["source_type"]).reshape(-1)[0])
            source_strength = np.asarray(payload["source_strength"])
            source_seed = int(np.asarray(payload["sample_seed"]).reshape(-1)[0])
        expected_seed = _seed_for_sample(stage.seed, index)
        if bathymetry_seed != expected_seed or source_seed != expected_seed:
            raise RuntimeError(f"Static-input seed mismatch for sample {index}")
        _validate_authoritative_input(
            record=records[index],
            split=stage.split,
            sample_idx=index,
            scenario_id=f"scenario_{index:06d}",
            bathymetry=bathymetry,
            source_field=source_field,
            source_strength_array=source_strength,
            bathymetry_type=bathymetry_type,
            source_type=source_type,
            sea_level_offset=0.0,
            config=config,
        )
        audited += 1
    return records, audited


def _selected_output_paths(output_dir: Path, stage: StageSpec) -> list[Path]:
    existing: list[Path] = []
    for folder in SOLVER_FOLDERS.values():
        samples = output_dir / folder / "samples"
        for index in range(stage.start_index, stage.stop_index + 1):
            candidate = samples / f"sample_{index:06d}"
            if candidate.exists():
                existing.append(candidate)
            existing.extend(
                sorted(samples.glob(f".sample_{index:06d}.staging-*"))
            )
    shard = output_dir / "operational_shards" / (
        f"{stage.split}_{stage.start_index:06d}_{stage.stop_index:06d}.json"
    )
    if shard.exists():
        existing.append(shard)
    return existing


def _validate_requested_time_provenance(
    provenance: Any, *, expected_times: np.ndarray
) -> None:
    missing = REQUIRED_PROVENANCE - set(provenance.files)
    if missing:
        raise RuntimeError(f"Requested-time provenance missing: {sorted(missing)}")
    expected = np.asarray(expected_times, dtype=np.float64)
    requested = np.asarray(provenance["requested_timestamps"], dtype=np.float64)
    left = np.asarray(provenance["left_natural_timestamps"], dtype=np.float64)
    right = np.asarray(provenance["right_natural_timestamps"], dtype=np.float64)
    weights = np.asarray(provenance["interpolation_weights"], dtype=np.float64)
    widths = np.asarray(provenance["bracket_widths"], dtype=np.float64)
    exact = np.asarray(provenance["exact_knot"], dtype=np.bool_)
    step_indices = np.asarray(provenance["natural_step_indices"], dtype=np.int64)
    arrays = (requested, left, right, weights, widths, exact, step_indices)
    if any(values.shape != expected.shape for values in arrays):
        raise RuntimeError("Requested-time provenance shape mismatch")
    if not np.array_equal(requested, expected):
        raise RuntimeError("Requested-time provenance grid mismatch")
    if not all(np.isfinite(values).all() for values in (left, right, weights, widths)):
        raise RuntimeError("Requested-time provenance contains nonfinite values")
    if np.any(step_indices < 1) or np.any(np.diff(step_indices) < 0):
        raise RuntimeError("Requested-time natural-step indices are invalid")

    natural_dt = np.asarray(provenance["natural_dt_history"], dtype=np.float64)
    total_raw = np.asarray(provenance["total_natural_steps"]).reshape(-1)
    if natural_dt.ndim != 1 or natural_dt.size == 0:
        raise RuntimeError("Natural-step history must be a nonempty vector")
    if total_raw.size != 1 or int(total_raw[0]) != natural_dt.size:
        raise RuntimeError("Natural-step count does not match its history")
    if not np.isfinite(natural_dt).all() or np.any(natural_dt <= 0.0):
        raise RuntimeError("Natural-step history contains invalid timesteps")
    if np.any(step_indices > natural_dt.size):
        raise RuntimeError("Requested output references a missing natural step")

    tolerance = 1.0e-14
    if not np.allclose(widths, right - left, rtol=0.0, atol=tolerance):
        raise RuntimeError("Requested-time bracket widths are inconsistent")
    if np.any(~exact & (left >= requested)) or np.any(~exact & (right <= requested)):
        raise RuntimeError("Requested output is not strictly inside its natural bracket")
    if np.any(exact & ((left != requested) | (right != requested))):
        raise RuntimeError("Exact-knot requested output has a nonexact bracket")
    expected_weights = np.zeros(expected.shape, dtype=np.float64)
    nonexact = ~exact
    expected_weights[nonexact] = (
        (requested[nonexact] - left[nonexact]) / widths[nonexact]
    )
    if not np.allclose(weights, expected_weights, rtol=0.0, atol=tolerance):
        raise RuntimeError("Requested-time interpolation weights are inconsistent")

    natural_times = np.cumsum(natural_dt, dtype=np.float64)
    indexed_right = natural_times[step_indices - 1]
    indexed_left = np.zeros(expected.shape, dtype=np.float64)
    later = step_indices > 1
    indexed_left[later] = natural_times[step_indices[later] - 2]
    if not np.allclose(
        right[nonexact], indexed_right[nonexact], rtol=0.0, atol=tolerance
    ) or not np.allclose(
        left[nonexact], indexed_left[nonexact], rtol=0.0, atol=tolerance
    ):
        raise RuntimeError("Requested output does not use adjacent natural steps")
    if not np.allclose(
        requested[exact], indexed_right[exact], rtol=0.0, atol=tolerance
    ):
        raise RuntimeError("Exact-knot output does not match its natural step")


def preflight_stage(
    *,
    resolved_config: Mapping[str, Any],
    stage: StageSpec,
    input_root: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    _validate_candidate_policy(resolved_config)
    dataset = resolved_config["dataset"]
    parsed_dataset = TsunamiDatasetBuilder._parse_dataset_section(
        dict(resolved_config)
    )
    parsed_solver = TsunamiDatasetBuilder._parse_solver_section(
        dict(resolved_config)
    )
    parsed_operations = TsunamiDatasetBuilder._parse_operational_section(
        resolved_config, requested_workers=parsed_dataset.num_workers
    )
    if stage.requires_generation_contract:
        binding = parsed_dataset.generation_contract
        if binding is None:
            raise RuntimeError("Accepted stage has no generation-contract binding")
        artifact = binding.artifact
        if artifact.get("code_state_hash") != code_state(ROOT).get("code_state_hash"):
            raise RuntimeError("Generation contract code-state hash mismatch")
        expected_h0 = {
            "h0_contract_hash": H0_CONTRACT_HASH,
            "inventory_sha256": H0_INVENTORY_SHA256,
            "require_exact_arrays": True,
            "allow_input_generation": False,
        }
        if artifact.get("h0") != expected_h0:
            raise RuntimeError("Generation contract H0 binding mismatch")
        if artifact.get("validation_evidence") != VALIDATION_EVIDENCE:
            raise RuntimeError("Generation contract validation evidence mismatch")
        policy = artifact.get("split_policy", {}).get(stage.split, {})
        if policy != {"seed": stage.seed, "num_samples": stage.num_samples}:
            raise RuntimeError("Generation contract split policy mismatch")
        shard_policy = artifact.get("shard_policy", {}).get(stage.name, {})
        if shard_policy != {
            "split": stage.split,
            "start": stage.start_index,
            "stop": stage.stop_index,
        }:
            raise RuntimeError("Generation contract shard policy mismatch")
        observed_execution = {
            "num_workers": parsed_dataset.num_workers,
            "max_in_flight": parsed_operations.max_in_flight,
            **parsed_operations.metadata(),
        }
        if artifact.get("execution_policy") != observed_execution:
            raise RuntimeError("Generation contract execution policy mismatch")
        observed_hashes = _generation_resolved_config_hashes(
            parsed_dataset, parsed_solver
        )
        if artifact.get("resolved_config_hashes") != observed_hashes:
            raise RuntimeError("Generation contract resolved configuration mismatch")
    input_root_resolved = input_root.resolve()
    output_resolved = output_dir.resolve()
    try:
        output_resolved.relative_to(input_root_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must not be inside the authoritative cache root")
    existing = _selected_output_paths(output_dir, stage)
    if existing:
        raise RuntimeError(
            "Selected output range is not fresh; first existing path: "
            f"{existing[0]}"
        )
    selected = set(range(stage.start_index, stage.stop_index + 1))
    manifest_paths = [manifest_path]
    manifest_paths.extend(
        manifest_path.parent / f"{folder}_manifest.jsonl"
        for folder in SOLVER_FOLDERS.values()
    )
    for candidate_manifest in manifest_paths:
        if not candidate_manifest.is_file():
            continue
        with candidate_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text and int(json.loads(text).get("sample_index", -1)) in selected:
                    raise RuntimeError(
                        "Manifest already contains the selected fresh range: "
                        f"{candidate_manifest}"
                    )
    for timing_path in sorted((output_dir / "operational_runs").glob("*.json")):
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if (
            timing.get("split") == stage.split
            and int(timing.get("start_index") or -1) == stage.start_index
            and int(timing.get("stop_index") or -1) == stage.stop_index
        ):
            raise RuntimeError(
                "Operational timing already exists for the selected fresh range: "
                f"{timing_path}"
            )
    authoritative = resolved_config["authoritative_inputs"]
    inventory_path = Path(str(authoritative["inventory_path"]))
    if not inventory_path.is_absolute():
        inventory_path = ROOT / inventory_path
    _, audited = _validate_input_cache(
        input_root=input_root,
        stage=stage,
        inventory_path=inventory_path.resolve(),
    )
    return {
        "stage": stage.name,
        "split": stage.split,
        "seed": stage.seed,
        "num_samples": stage.num_samples,
        "start_index": stage.start_index,
        "stop_index": stage.stop_index,
        "input_cache_records_audited": audited,
        "fresh_output_range": True,
        "requested_output_enabled": True,
        "authoritative_h0_bound": True,
        "trajectory_values_inspected": False,
        "output_dir": str(output_resolved),
        "manifest_path": str(manifest_path.resolve()),
        "workers": int(dataset["num_workers"]),
    }


def _publication_hash(publication: Mapping[str, Any]) -> str:
    return stable_hash_payload(
        artifact_kind="requested-output-publication-record",
        payload=publication,
        schema_id=PUBLICATION_SCHEMA_ID,
    )


def frozen_machine_fingerprint(machine: Mapping[str, Any]) -> dict[str, Any]:
    storage = machine.get("storage")
    if not isinstance(storage, Mapping):
        raise RuntimeError("Operational timing machine storage metadata is missing")
    return {
        "platform": machine.get("platform"),
        "machine": machine.get("machine"),
        "python_version": machine.get("python_version"),
        "numpy_version": machine.get("numpy_version"),
        "cpu_model": machine.get("cpu_model"),
        "logical_cpu_count": machine.get("logical_cpu_count"),
        "memory_total_bytes": machine.get("memory_total_bytes"),
        "thread_environment": machine.get("thread_environment"),
        "cloud_provider": machine.get("cloud_provider"),
        "cloud_zone": machine.get("cloud_zone"),
        "machine_type": machine.get("machine_type"),
        "storage_class": storage.get("class"),
        "storage_total_bytes": storage.get("total_bytes"),
    }


def stage_attestation_hash(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("attestation_hash", None)
    return stable_hash_payload(
        artifact_kind="verified-generation-stage-attestation",
        payload=content,
        schema_id=STAGE_ATTESTATION_SCHEMA_ID,
    )


def write_stage_attestation(
    *, output_dir: Path, verification: Mapping[str, Any]
) -> Path:
    if not bool(verification.get("verified", False)):
        raise ValueError("Only a verified generation stage can be attested")
    payload = {
        "schema_id": STAGE_ATTESTATION_SCHEMA_ID,
        "artifact_kind": "verified-generation-stage-attestation",
        "stage": verification["stage"],
        "generation_contract_hash": verification.get("generation_contract_hash"),
        "verification": dict(verification),
    }
    payload["attestation_hash"] = stage_attestation_hash(payload)
    path = (
        output_dir
        / "stage_attestations"
        / f"{payload['stage']}_{payload['attestation_hash']}.json"
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Stage attestation mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    staging.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, path)
    return path


def validate_stage_prerequisites(
    *,
    stage: StageSpec,
    attestation_paths: list[Path],
    generation_contract_hash_value: str | None,
) -> dict[str, str]:
    if stage.name == "test":
        raise RuntimeError(
            "Final-test generation remains blocked until preprocessing, training, "
            "checkpoint-selection, and evaluation protocols are frozen"
        )
    required = set(STAGE_PREREQUISITES[stage.name])
    observed: dict[str, str] = {}
    for path in attestation_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_id") != STAGE_ATTESTATION_SCHEMA_ID:
            raise RuntimeError(f"Stage attestation schema mismatch: {path}")
        if payload.get("artifact_kind") != "verified-generation-stage-attestation":
            raise RuntimeError(f"Stage attestation kind mismatch: {path}")
        if payload.get("attestation_hash") != stage_attestation_hash(payload):
            raise RuntimeError(f"Stage attestation content hash mismatch: {path}")
        prerequisite_stage = str(payload.get("stage", ""))
        if prerequisite_stage in observed:
            raise RuntimeError(f"Duplicate prerequisite stage: {prerequisite_stage}")
        if payload.get("generation_contract_hash") != generation_contract_hash_value:
            raise RuntimeError(
                f"Prerequisite generation contract mismatch: {prerequisite_stage}"
            )
        verification = payload.get("verification")
        if not isinstance(verification, Mapping) or not bool(
            verification.get("verified", False)
        ):
            raise RuntimeError(f"Prerequisite stage is not verified: {path}")
        if verification.get("stage") != prerequisite_stage:
            raise RuntimeError(f"Prerequisite stage identity mismatch: {path}")
        if (
            verification.get("generation_contract_hash")
            != generation_contract_hash_value
        ):
            raise RuntimeError(f"Prerequisite verification contract mismatch: {path}")
        observed[prerequisite_stage] = str(payload["attestation_hash"])
    missing = required - set(observed)
    extra = set(observed) - required
    if missing:
        raise RuntimeError(f"Missing prerequisite stage attestations: {sorted(missing)}")
    if extra:
        raise RuntimeError(f"Unexpected prerequisite stage attestations: {sorted(extra)}")
    return observed


def verify_stage(
    *,
    resolved_config: Mapping[str, Any],
    stage: StageSpec,
    input_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    requested_raw = resolved_config["requested_output"]
    expected_output_contract = build_candidate_contract(
        status=str(requested_raw["status"])
    )
    expected_output_hash = contract_hash(expected_output_contract)
    generation_binding = resolved_config.get("generation_contract")
    expected_generation_hash = (
        None
        if not isinstance(generation_binding, Mapping)
        else str(generation_binding["contract_hash"])
    )
    authoritative = resolved_config["authoritative_inputs"]
    inventory_path = Path(str(authoritative["inventory_path"]))
    if not inventory_path.is_absolute():
        inventory_path = ROOT / inventory_path
    inventory = _inventory_records(inventory_path.resolve(), split=stage.split)
    expected_code_hash = str(code_state(ROOT)["code_state_hash"])
    publications: dict[str, str] = {}
    config_hashes: dict[str, set[str]] = {name: set() for name in SOLVER_FOLDERS}
    for index in range(stage.start_index, stage.stop_index + 1):
        scenario_id = f"scenario_{index:06d}"
        identity = split_qualified_identity(stage.split, scenario_id)
        static_hashes: dict[str, str] | None = None
        input_fingerprints: set[str] = set()
        with np.load(
            input_root / "bathymetry" / f"sample_{index:06d}.npz",
            allow_pickle=False,
        ) as payload:
            cached_bathymetry = np.asarray(payload["bathymetry"], dtype=np.float32)
        with np.load(
            input_root / "sources" / f"sample_{index:06d}.npz",
            allow_pickle=False,
        ) as payload:
            cached_source = np.asarray(payload["source_field"], dtype=np.float32)
            cached_strength = float(np.asarray(payload["source_strength"]).reshape(-1)[0])
        prepared = _prepare_buffered_domain(
            cached_bathymetry,
            cached_source,
            cached_strength,
            0.0,
            BufferedDomainConfig(
                enabled=True,
                buffer_cells=16,
                source_taper_cells=8,
                bathymetry_extension="edge",
                output_crop="central",
            ),
        )
        expected_static_hashes = {
            "bathymetry": hash_array(prepared["bathymetry"]),
            "source_field": hash_array(prepared["source_field"]),
            "rest_depth": hash_array(prepared["rest_depth"]),
            "eta0": hash_array(prepared["eta0"]),
            "initial_depth": hash_array(prepared["h0"]),
            "free_surface0": hash_array(prepared["free_surface0"]),
        }
        for solver_name, folder in SOLVER_FOLDERS.items():
            sample_dir = output_dir / folder / "samples" / f"sample_{index:06d}"
            publication = validate_publication(
                sample_dir,
                expected_identity=identity,
                expected_contract_hash=expected_output_hash,
                expected_code_state_hash=expected_code_hash,
                expected_times=candidate_requested_times(),
                expected_solver_name=solver_name,
                expected_sample_index=index,
                expected_authoritative_input_fingerprint=str(
                    inventory[index]["input_fingerprint"]
                ),
                expected_authoritative_inventory_sha256=H0_INVENTORY_SHA256,
                expected_generation_contract_hash=expected_generation_hash,
            )
            if publication.get("generation_contract_hash") != expected_generation_hash:
                raise RuntimeError("Publication generation-contract binding mismatch")
            if publication.get("quality_status") != "ok":
                raise RuntimeError(
                    f"Publication quality is not accepted for {identity['qualified_id']}:{solver_name}"
                )
            meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
            expected_domain = {
                "enabled": True,
                "buffer_cells": 16,
                "source_taper_cells": 8,
                "bathymetry_extension": "edge",
                "output_crop": "central",
                "solver_shape": [96, 96],
                "publication_shape": [64, 64],
            }
            if meta.get("computational_domain") != expected_domain:
                raise RuntimeError("Published computational-domain provenance mismatch")
            if meta.get("generation_contract_hash") != expected_generation_hash:
                raise RuntimeError("Published metadata generation-contract mismatch")
            if (
                meta.get("authoritative_input", {}).get("h0_contract_hash")
                != H0_CONTRACT_HASH
            ):
                raise RuntimeError("Published H0 contract binding mismatch")
            with np.load(sample_dir / "sample.npz", allow_pickle=False) as payload:
                if payload["trajectory_eta"].shape != (50, 64, 64):
                    raise RuntimeError("Published trajectory shape mismatch")
                stored_generation_hash = str(
                    np.asarray(payload["generation_contract_hash"]).reshape(-1)[0]
                )
                if stored_generation_hash != (expected_generation_hash or ""):
                    raise RuntimeError(
                        "Published sample generation-contract binding mismatch"
                    )
                current_static = {
                    name: hash_array(payload[name])
                    for name in (
                        "bathymetry",
                        "source_field",
                        "rest_depth",
                        "eta0",
                        "initial_depth",
                        "free_surface0",
                    )
                }
            if current_static != expected_static_hashes:
                raise RuntimeError(
                    "Published static arrays do not match the H0-bound buffered "
                    "source transformation"
                )
            if static_hashes is None:
                static_hashes = current_static
            elif static_hashes != current_static:
                raise RuntimeError("Published static arrays differ across solvers")
            with np.load(
                sample_dir / "provenance.npz", allow_pickle=False
            ) as provenance:
                _validate_requested_time_provenance(
                    provenance, expected_times=candidate_requested_times()
                )
            input_fingerprints.add(str(publication["input_fingerprint"]))
            config_hashes[solver_name].add(str(publication["resolved_config_hash"]))
            key = f"{stage.split}:{scenario_id}:{solver_name}"
            publications[key] = _publication_hash(publication)
        if len(input_fingerprints) != 1:
            raise RuntimeError("Published input fingerprints differ across solvers")

    expected_config_hashes = {
        name: next(iter(hashes)) for name, hashes in config_hashes.items()
    }
    if any(len(hashes) != 1 for hashes in config_hashes.values()):
        raise RuntimeError("Resolved configuration changed within the selected range")
    shard_path = output_dir / "operational_shards" / (
        f"{stage.split}_{stage.start_index:06d}_{stage.stop_index:06d}.json"
    )
    shard = validate_operational_shard(
        shard_path,
        expected_contract_hash=expected_output_hash,
        expected_publication_hashes=publications,
        expected_split=stage.split,
        expected_start_index=stage.start_index,
        expected_stop_index=stage.stop_index,
        expected_solver_names=sorted(SOLVER_FOLDERS),
        expected_config_hashes=expected_config_hashes,
        expected_code_state_hash=expected_code_hash,
        expected_generation_contract_hash=expected_generation_hash,
    )
    if shard.get("generation_contract_hash") != expected_generation_hash:
        raise RuntimeError("Operational shard generation-contract mismatch")
    timing_candidates = []
    for path in sorted((output_dir / "operational_runs").glob("*.json")):
        payload = validate_generation_timing(path)
        if (
            payload.get("split") == stage.split
            and int(payload.get("start_index", -1)) == stage.start_index
            and int(payload.get("stop_index", -1)) == stage.stop_index
        ):
            timing_candidates.append((path, payload))
    if len(timing_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one operational timing record for the fresh stage; "
            f"found {len(timing_candidates)}"
        )
    timing_path, timing = timing_candidates[0]
    if timing.get("generation_contract_hash") != expected_generation_hash:
        raise RuntimeError("Operational timing generation-contract mismatch")
    expected_rollouts = stage.scenario_count * len(SOLVER_FOLDERS)
    counts = timing["counts"]
    if int(counts["generated_solver_rollouts"]) != expected_rollouts:
        raise RuntimeError("Fresh stage did not generate every solver rollout")
    if int(counts["reused_solver_rollouts"]) != 0:
        raise RuntimeError("Fresh stage reused solver trajectories")
    if int(counts["accepted_solver_rollouts"]) != expected_rollouts:
        raise RuntimeError("Fresh stage did not accept every solver rollout")
    if int(counts["failed_scenarios"]) != 0:
        raise RuntimeError("Fresh stage contains failed scenarios")
    machine_fingerprint = frozen_machine_fingerprint(timing["machine"])
    if expected_generation_hash is not None:
        contract_path = Path(str(generation_binding["path"]))
        contract_artifact = validate_generation_contract_artifact(
            contract_path, expected_hash=expected_generation_hash
        )
        if contract_artifact.get("machine_fingerprint") != machine_fingerprint:
            raise RuntimeError("Generation machine differs from the frozen rehearsal")
    snapshot_path = output_dir / "dataset_config.snapshot.yaml"
    if not snapshot_path.is_file():
        raise RuntimeError("Resolved dataset configuration snapshot is missing")
    snapshot = _load_yaml(snapshot_path)
    if snapshot != dict(resolved_config):
        raise RuntimeError("Dataset configuration snapshot differs from resolved config")
    return {
        "stage": stage.name,
        "split": stage.split,
        "start_index": stage.start_index,
        "stop_index": stage.stop_index,
        "scenario_count": stage.scenario_count,
        "solver_publication_count": len(publications),
        "output_contract_hash": expected_output_hash,
        "generation_contract_hash": expected_generation_hash,
        "code_state_hash": expected_code_hash,
        "resolved_config_hashes": expected_config_hashes,
        "snapshot_sha256": sha256_file(snapshot_path),
        "operational_shard_sha256": sha256_file(shard_path),
        "operational_timing_sha256": sha256_file(timing_path),
        "machine_fingerprint": machine_fingerprint,
        "publication_set_hash": stable_hash_payload(
            artifact_kind="verified-generation-publication-set",
            payload=publications,
            schema_id=GENERATION_CONTRACT_SCHEMA_ID,
        ),
        "generated_solver_rollouts": expected_rollouts,
        "reused_solver_rollouts": 0,
        "trajectory_values_inspected": False,
        "verified": True,
    }


def execute_stage(
    *,
    resolved_config: Mapping[str, Any],
    stage: StageSpec,
) -> None:
    workers = int(resolved_config["dataset"]["num_workers"])
    if workers > 1:
        invalid = {
            key: value
            for key in THREAD_ENV_KEYS
            if (value := os.environ.get(key)) not in (None, "1")
        }
        if invalid:
            details = ", ".join(
                f"{key}={value}" for key, value in sorted(invalid.items())
            )
            raise RuntimeError(
                "Multiprocess generation requires single-thread numerical "
                f"backends; found {details}"
            )
        for key in THREAD_ENV_KEYS:
            os.environ.setdefault(key, "1")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(dict(resolved_config), handle, sort_keys=False)
        temporary_config = Path(handle.name)
    try:
        builder = TsunamiDatasetBuilder(
            str(temporary_config),
            provenance_config_path=(
                Path(str(resolved_config["dataset"]["output_dir"]))
                / "dataset_config.snapshot.yaml"
            ),
        )
        if stage.requires_generation_contract:
            builder.output_dir.mkdir(parents=True, exist_ok=True)
            observed_machine = frozen_machine_fingerprint(
                machine_snapshot(builder.output_dir, builder.operations.metadata())
            )
            binding = builder.dataset.generation_contract
            if binding is None:
                raise RuntimeError("Accepted stage has no generation-contract binding")
            if binding.artifact.get("machine_fingerprint") != observed_machine:
                raise RuntimeError(
                    "Generation machine differs from the frozen rehearsal"
                )
        builder.run(
            start_at=stage.start_index,
            stop_at=stage.stop_index,
            acknowledge_provisional=not stage.requires_generation_contract,
        )
    finally:
        temporary_config.unlink(missing_ok=True)


def freeze_generation_contract(
    *,
    rehearsal_config: Mapping[str, Any],
    rehearsal_verification: Mapping[str, Any],
    artifact_root: Path,
    repo_state: Mapping[str, Any] | None = None,
) -> Path:
    if not bool(rehearsal_verification.get("verified", False)):
        raise ValueError("A passing rehearsal verification is required")
    if rehearsal_verification.get("stage") != "rehearsal":
        raise ValueError("Only the fresh one-sample rehearsal can freeze the contract")
    state = dict(code_state(ROOT) if repo_state is None else repo_state)
    if bool(state.get("dirty", True)):
        raise RuntimeError("Refusing to freeze a generation contract from dirty code")
    dataset = TsunamiDatasetBuilder._parse_dataset_section(dict(rehearsal_config))
    solver_cfg = TsunamiDatasetBuilder._parse_solver_section(dict(rehearsal_config))
    resolved_hashes = _generation_resolved_config_hashes(dataset, solver_cfg)
    rehearsal_machine = rehearsal_verification.get("machine_fingerprint")
    if not isinstance(rehearsal_machine, Mapping) or not rehearsal_machine:
        raise RuntimeError("Rehearsal machine fingerprint is missing")
    expected_rehearsal_hash = contract_hash(
        build_candidate_contract(status="provisional")
    )
    verification_checks = {
        "output_contract_hash": expected_rehearsal_hash,
        "generation_contract_hash": None,
        "code_state_hash": str(state["code_state_hash"]),
        "resolved_config_hashes": resolved_hashes,
        "generated_solver_rollouts": 3,
        "reused_solver_rollouts": 0,
        "scenario_count": 1,
        "solver_publication_count": 3,
        "split": "train",
        "start_index": 1,
        "stop_index": 1,
        "trajectory_values_inspected": False,
    }
    for key, expected in verification_checks.items():
        if rehearsal_verification.get(key) != expected:
            raise RuntimeError(f"Rehearsal verification {key} mismatch")
    policy = TsunamiDatasetBuilder._parse_operational_section(
        rehearsal_config, requested_workers=dataset.num_workers
    )
    execution = {
        "num_workers": dataset.num_workers,
        "max_in_flight": policy.max_in_flight,
        **policy.metadata(),
    }
    accepted_output = build_candidate_contract(status=ACCEPTED_STATUS)
    payload: dict[str, Any] = {
        "schema_id": GENERATION_CONTRACT_SCHEMA_ID,
        "artifact_kind": "accepted-generation-contract",
        "accepted_output_contract": accepted_output,
        "accepted_output_contract_hash": contract_hash(accepted_output),
        "h0": {
            "h0_contract_hash": H0_CONTRACT_HASH,
            "inventory_sha256": H0_INVENTORY_SHA256,
            "require_exact_arrays": True,
            "allow_input_generation": False,
        },
        "validation_evidence": dict(VALIDATION_EVIDENCE),
        "code_state_hash": str(state["code_state_hash"]),
        "code_state": state,
        "resolved_config_hashes": resolved_hashes,
        "execution_policy": execution,
        "machine_fingerprint": dict(rehearsal_machine),
        "split_policy": {
            "train": {"seed": 42, "num_samples": 10000},
            "eval": {"seed": 69, "num_samples": 1000},
            "test": {"seed": 367, "num_samples": 2500},
        },
        "shard_policy": {
            "rehearsal": {"split": "train", "start": 1, "stop": 1},
            "validation": {"split": "eval", "start": 1, "stop": 1000},
            "train-1": {"split": "train", "start": 1, "stop": 5000},
            "train-2": {"split": "train", "start": 5001, "stop": 10000},
            "test": {"split": "test", "start": 1, "stop": 2500},
        },
        "rehearsal_verification": dict(rehearsal_verification),
        "decision": {
            "accepted_contract_frozen": True,
            "mass_generation_authorized": True,
            "preprocessing_authorized": False,
            "final_test_scientific_inspection_authorized": False,
        },
    }
    payload["contract_hash"] = generation_contract_hash(payload)
    final = artifact_root.resolve() / str(payload["contract_hash"])
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite generation contract: {final}")
    staging = final.with_name(f".{final.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        contract_path = staging / "generation_contract.json"
        contract_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "REPORT.md").write_text(
            "# Accepted common-time-v2 generation contract\n\n"
            f"- Contract hash: `{payload['contract_hash']}`\n"
            "- Fresh cloud rehearsal verified: yes\n"
            "- Exact H0 inputs required: yes\n"
            "- Mass generation authorized: yes\n"
            "- Preprocessing authorized: no\n",
            encoding="utf-8",
        )
        with (staging / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
            for path in sorted(staging.iterdir()):
                if path.is_file():
                    handle.write(f"{sha256_file(path)}  {path.name}\n")
        artifact_root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        validate_generation_contract_artifact(
            final / "generation_contract.json",
            expected_hash=str(payload["contract_hash"]),
        )
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
