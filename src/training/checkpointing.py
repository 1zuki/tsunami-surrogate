from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping
import torch

from src.models.signature import model_config_signature
from src.utils.hashing import sha256_file


DATA_PROVENANCE_SCHEMA_ID = "tsunami-surrogate.checkpoint-data-provenance.v1"
PROCESSED_MANIFEST_SCHEMA_ID = "tsunami-surrogate.processed-dataset.v2"


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_dataset_location(configured: Path, config_key: str) -> Path:
    if configured.name == "eval_dataset.npz" and (
        configured.parent / "shards_manifest.json"
    ).is_file():
        return configured.parent

    if config_key == "path" and configured.is_dir():
        train_dir = configured / "train"
        if (
            (train_dir / "shards_manifest.json").is_file()
            or (train_dir / "eval_dataset.npz").is_file()
            or any(train_dir.glob("*.npz"))
        ):
            return train_dir

    return configured


def _dataset_artifacts(path: str | Path, config_key: str) -> Dict[str, Any]:
    configured = Path(path)
    resolved = _resolve_dataset_location(configured, config_key)

    artifacts: Dict[str, str] = {}
    identity_strength = "unavailable"
    if resolved.is_file():
        artifacts[resolved.name] = sha256_file(resolved)
        root = resolved.parent
        identity_strength = "content_bound"
    else:
        root = resolved

    flat_dataset = root / "eval_dataset.npz"
    if (
        not (root / "shards_manifest.json").is_file()
        and flat_dataset.is_file()
    ):
        artifacts[flat_dataset.name] = sha256_file(flat_dataset)
        identity_strength = "content_bound"

    for name in ("shards_manifest.json", "eval_manifest.json"):
        candidate = root / name
        if candidate.is_file():
            artifacts[name] = sha256_file(candidate)
            payload = _read_json(candidate)
            if payload and payload.get("schema_id") == PROCESSED_MANIFEST_SCHEMA_ID:
                if name == "shards_manifest.json":
                    shards = payload.get("shards", [])
                    if isinstance(shards, list) and shards and all(
                        isinstance(shard, Mapping)
                        and shard.get("file")
                        and shard.get("sha256")
                        for shard in shards
                    ):
                        identity_strength = "content_bound"
                elif isinstance(payload.get("artifacts"), Mapping):
                    identity_strength = "content_bound"
            elif identity_strength == "unavailable":
                identity_strength = "manifest_only"

    normalization_candidates = (
        root / "normalization_stats.json",
        root.parent / "normalization_stats.json",
    )
    normalization_path = next(
        (candidate for candidate in normalization_candidates if candidate.is_file()),
        None,
    )
    if normalization_path is not None:
        artifacts["normalization_stats.json"] = sha256_file(normalization_path)

    return {
        "configured_path": str(configured),
        "resolved_path": str(resolved),
        "artifacts": artifacts,
        "identity_strength": identity_strength,
    }


def capture_data_provenance(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    if not isinstance(data_cfg, Mapping):
        return {"schema_id": DATA_PROVENANCE_SCHEMA_ID, "datasets": {}}

    datasets: Dict[str, Any] = {}
    for key in ("train_path", "path"):
        value = data_cfg.get(key)
        if value:
            datasets[key] = _dataset_artifacts(str(value), key)
            break
    return {"schema_id": DATA_PROVENANCE_SCHEMA_ID, "datasets": datasets}


def _training_dataset_entry(provenance: Mapping[str, Any]) -> Mapping[str, Any] | None:
    datasets = provenance.get("datasets", {})
    if not isinstance(datasets, Mapping):
        return None
    for key in ("train_path", "path"):
        entry = datasets.get(key)
        if isinstance(entry, Mapping):
            return entry
    return None


def _artifact_hashes(entry: Mapping[str, Any]) -> Dict[str, str]:
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return {}
    return {str(name): str(value) for name, value in artifacts.items()}


def _validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    *,
    validate_training_data: bool,
) -> Dict[str, str]:
    compatibility = {
        "model_config": "unavailable",
        "training_data": "unavailable",
    }

    runtime_signature = getattr(model, "_tsunami_model_config_signature", None)
    checkpoint_signature = checkpoint.get("model_signature")
    signature_source = "stored"
    if checkpoint_signature is None:
        checkpoint_cfg = checkpoint.get("config")
        if isinstance(checkpoint_cfg, Mapping):
            checkpoint_signature = model_config_signature(checkpoint_cfg)
            signature_source = "derived"
    if isinstance(runtime_signature, Mapping) and isinstance(
        checkpoint_signature, Mapping
    ):
        if dict(runtime_signature) != dict(checkpoint_signature):
            raise ValueError(
                "Checkpoint/model configuration mismatch. "
                f"checkpoint={checkpoint_signature}, runtime={runtime_signature}"
            )
        compatibility["model_config"] = (
            "match" if signature_source == "stored" else "derived_match"
        )

    checkpoint_data = checkpoint.get("data_provenance")
    if isinstance(checkpoint_data, Mapping):
        compatibility["training_data"] = "not_checked"
    if not validate_training_data:
        return compatibility

    runtime_cfg = getattr(model, "_tsunami_runtime_config", None)
    if isinstance(checkpoint_data, Mapping) and isinstance(runtime_cfg, Mapping):
        runtime_data = capture_data_provenance(runtime_cfg)
        expected_entry = _training_dataset_entry(checkpoint_data)
        observed_entry = _training_dataset_entry(runtime_data)
        if expected_entry is None:
            compatibility["training_data"] = "unavailable"
            return compatibility
        if observed_entry is None:
            raise ValueError(
                "Cannot validate checkpoint training-data provenance because the "
                "runtime configuration has no train_path or data.path."
            )

        expected = _artifact_hashes(expected_entry)
        observed = _artifact_hashes(observed_entry)
        if not expected:
            compatibility["training_data"] = "unavailable"
            return compatibility
        if not observed:
            raise ValueError(
                "Cannot validate checkpoint training-data provenance because no "
                "runtime dataset identity artifacts were found."
            )
        if expected != observed:
            raise ValueError(
                "Checkpoint training-data provenance mismatch. "
                f"checkpoint={expected}, runtime={observed}"
            )

        strengths = {
            str(expected_entry.get("identity_strength", "unavailable")),
            str(observed_entry.get("identity_strength", "unavailable")),
        }
        compatibility["training_data"] = (
            "match" if strengths == {"content_bound"} else "manifest_match"
        )
    return compatibility


def save_checkpoint(path, model, optimizer, epoch, metrics, cfg, scheduler=None, trainer_state=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model_signature = getattr(model, "_tsunami_model_config_signature", None)
    if model_signature is None:
        model_signature = model_config_signature(cfg)
    data_provenance = getattr(
        model,
        "_tsunami_checkpoint_data_provenance",
        None,
    )
    if not isinstance(data_provenance, Mapping):
        data_provenance = capture_data_provenance(cfg)
        model._tsunami_checkpoint_data_provenance = data_provenance

    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'trainer_state': trainer_state,
        'epoch': epoch,
        'metrics': metrics,
        'config': cfg,
        'model_signature': model_signature,
        'data_provenance': data_provenance,
    }, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location='cpu',
    *,
    validate_training_data: bool = False,
):
    ckpt = torch.load(path, map_location=map_location)
    ckpt["compatibility"] = _validate_checkpoint_compatibility(
        ckpt,
        model,
        validate_training_data=validate_training_data,
    )
    model.load_state_dict(ckpt['model_state'])

    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if scheduler is not None and ckpt.get('scheduler_state') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])

    return ckpt
