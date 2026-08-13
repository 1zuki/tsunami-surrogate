from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.models import build_model
from src.training.checkpointing import (
    capture_data_provenance,
    load_checkpoint,
    save_checkpoint,
)


def _fno_config(*, padding: int) -> dict:
    return {
        "model": {
            "name": "fno2d",
            "in_channels": 3,
            "out_channels": 1,
            "modes1": 2,
            "modes2": 2,
            "width": 4,
            "depth": 1,
            "padding": padding,
            "use_grid": True,
        }
    }


def _cnn_config(train_path: Path) -> dict:
    return {
        "model": {
            "name": "cnn",
            "in_channels": 3,
            "out_channels": 1,
            "width": 4,
        },
        "data": {"train_path": str(train_path)},
    }


def _cnn_config_with_validation(
    train_path: Path, val_path: Path
) -> dict:
    cfg = _cnn_config(train_path)
    cfg["data"]["val_path"] = str(val_path)
    return cfg


def test_model_signature_rejects_padding_mismatch_with_compatible_weights(
    tmp_path: Path,
) -> None:
    checkpoint_cfg = _fno_config(padding=6)
    checkpoint_model = build_model(checkpoint_cfg)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        checkpoint_model,
        optimizer=None,
        epoch=1,
        metrics={},
        cfg=checkpoint_cfg,
    )

    runtime_cfg = _fno_config(padding=2)
    runtime_model = build_model(runtime_cfg)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(checkpoint_path, runtime_model)


def test_legacy_checkpoint_derives_model_signature_without_rewriting(
    tmp_path: Path,
) -> None:
    cfg = _fno_config(padding=6)
    model = build_model(cfg)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "scheduler_state": None,
            "trainer_state": None,
            "epoch": 3,
            "metrics": {},
            "config": cfg,
        },
        checkpoint_path,
    )
    before = checkpoint_path.read_bytes()

    loaded = load_checkpoint(checkpoint_path, build_model(cfg))

    assert loaded["compatibility"]["model_config"] == "derived_match"
    assert loaded["compatibility"]["training_data"] == "unavailable"
    assert checkpoint_path.read_bytes() == before


def test_training_data_is_strict_on_resume_but_not_for_ood_evaluation(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    train_path.write_bytes(b"training-data-v1")
    checkpoint_cfg = _cnn_config(train_path)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        build_model(checkpoint_cfg),
        optimizer=None,
        epoch=1,
        metrics={},
        cfg=checkpoint_cfg,
    )

    ood_path = tmp_path / "ood.npz"
    ood_path.write_bytes(b"intentional-ood-data")
    ood_cfg = _cnn_config(ood_path)
    evaluation_load = load_checkpoint(checkpoint_path, build_model(ood_cfg))
    assert evaluation_load["compatibility"]["training_data"] == "not_checked"

    with pytest.raises(ValueError, match="training-data provenance mismatch"):
        load_checkpoint(
            checkpoint_path,
            build_model(ood_cfg),
            validate_training_data=True,
        )


def test_data_path_provenance_resolves_the_actual_train_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "eval_dataset.npz").write_bytes(b"train")

    provenance = capture_data_provenance({"data": {"path": str(root)}})
    entry = provenance["datasets"]["path"]

    assert Path(entry["resolved_path"]) == train_dir
    assert entry["identity_strength"] == "content_bound"
    assert set(entry["artifacts"]) == {"eval_dataset.npz"}


def test_checkpoint_data_provenance_detects_in_place_training_data_change(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    train_path.write_bytes(b"before")
    cfg = _cnn_config(train_path)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        build_model(cfg),
        optimizer=None,
        epoch=1,
        metrics={},
        cfg=cfg,
    )

    train_path.write_bytes(b"after")
    with pytest.raises(ValueError, match="training-data provenance mismatch"):
        load_checkpoint(
            checkpoint_path,
            build_model(deepcopy(cfg)),
            validate_training_data=True,
        )


def test_checkpoint_data_provenance_binds_validation_data(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    train_path.write_bytes(b"train")
    val_path.write_bytes(b"validation-before")
    cfg = _cnn_config_with_validation(train_path, val_path)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        build_model(cfg),
        optimizer=None,
        epoch=1,
        metrics={},
        cfg=cfg,
    )

    val_path.write_bytes(b"validation-after")
    with pytest.raises(ValueError, match="validation-data provenance mismatch"):
        load_checkpoint(
            checkpoint_path,
            build_model(deepcopy(cfg)),
            validate_training_data=True,
        )
