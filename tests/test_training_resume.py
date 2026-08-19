from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.train import Trainer, _as_cpu_rng_state, _restore_rng_state
from src.utils.seed import (
    make_torch_generator,
    make_worker_init_fn,
    seed_everything,
    seed_worker,
)


class _DictDataset(Dataset):
    def __init__(self) -> None:
        values = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
        self.x = values.reshape(8, 1, 2, 2)
        self.y = 0.4 * self.x

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


class _RandomScaleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        random_scale = 0.9 + 0.05 * random.random() + 0.05 * np.random.random()
        dropped = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        return dropped * self.weight * random_scale


def test_dataloader_worker_init_is_spawn_picklable_and_deterministic() -> None:
    callback = pickle.loads(pickle.dumps(make_worker_init_fn(36)))

    callback(4)
    observed = (random.random(), float(np.random.random()), float(torch.rand(())))

    seed_worker(4, 36)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert observed == expected


def test_spawn_dataloader_can_read_first_batch() -> None:
    loader = DataLoader(
        _DictDataset(),
        batch_size=2,
        num_workers=1,
        worker_init_fn=make_worker_init_fn(36),
        multiprocessing_context="spawn",
    )

    batch = next(iter(loader))

    assert batch["x"].shape == (2, 1, 2, 2)
    assert batch["y"].shape == (2, 1, 2, 2)


def _loaders(
    seed: int,
    batch_size: int = 2,
    val_batch_size: int = 4,
) -> dict[str, DataLoader]:
    dataset = _DictDataset()
    return {
        "train": DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=make_torch_generator(seed),
        ),
        "val": DataLoader(
            dataset, batch_size=val_batch_size, shuffle=False
        ),
    }


def _config(output_dir: Path, epochs: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_identity = output_dir / "train.identity"
    val_identity = output_dir / "val.identity"
    train_identity.write_bytes(b"train")
    val_identity.write_bytes(b"validation")
    return {
        "output_dir": str(output_dir),
        "seed": 17,
        "data": {
            "train_path": str(train_identity),
            "val_path": str(val_identity),
        },
        "train": {
            "epochs": epochs,
            "lr": 1.0e-2,
            "weight_decay": 0.0,
            "loss": "mse",
            "checkpoint_metric": "val_rel_l2",
            "early_stopping": {"patience": 10, "mode": "min"},
        },
    }


def _run(output_dir: Path, epochs: int, resume_path: Path | None = None):
    seed_everything(17)
    model = _RandomScaleModel()
    trainer = Trainer(
        model,
        _loaders(seed=17),
        _config(output_dir, epochs),
        device=torch.device("cpu"),
    )
    history = trainer.fit(resume_path=resume_path)
    return model, history


def test_interrupted_training_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    full_model, full_history = _run(tmp_path / "full", epochs=2)

    split_dir = tmp_path / "split"
    _run(split_dir, epochs=1)
    resume_path = split_dir / "checkpoints" / "last.pt"

    # Deliberately disturb all RNGs before resuming; checkpoint restoration must
    # still reproduce the uninterrupted second epoch.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed_model = _RandomScaleModel()
    resumed_trainer = Trainer(
        resumed_model,
        _loaders(seed=17),
        _config(split_dir, epochs=2),
        device=torch.device("cpu"),
    )
    resumed_history = resumed_trainer.fit(resume_path=resume_path)

    assert resumed_history == full_history
    for name, expected in full_model.state_dict().items():
        assert torch.equal(resumed_model.state_dict()[name], expected)

    # The trainer state must remain loadable under PyTorch's default
    # weights-only policy.
    payload = torch.load(split_dir / "checkpoints" / "last.pt", map_location="cpu")
    assert isinstance(payload["trainer_state"]["rng_state"]["numpy"], dict)


def test_resume_rejects_changed_training_loader_contract(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    _run(split_dir, epochs=1)
    resume_path = split_dir / "checkpoints" / "last.pt"

    trainer = Trainer(
        _RandomScaleModel(),
        _loaders(seed=17, batch_size=4),
        _config(split_dir, epochs=2),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="DataLoader contract mismatch"):
        trainer.fit(resume_path=resume_path)


def test_resume_rejects_changed_training_contract(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    _run(split_dir, epochs=1)
    resume_path = split_dir / "checkpoints" / "last.pt"

    cfg = _config(split_dir, epochs=2)
    cfg["train"]["loss"] = "relative_l2"
    trainer = Trainer(
        _RandomScaleModel(),
        _loaders(seed=17),
        cfg,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="training contract mismatch"):
        trainer.fit(resume_path=resume_path)


def test_resume_rejects_changed_validation_loader_contract(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "split"
    _run(split_dir, epochs=1)
    resume_path = split_dir / "checkpoints" / "last.pt"

    trainer = Trainer(
        _RandomScaleModel(),
        _loaders(seed=17, val_batch_size=2),
        _config(split_dir, epochs=2),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="Validation DataLoader contract"):
        trainer.fit(resume_path=resume_path)


def test_resume_requires_own_last_checkpoint(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    _run(split_dir, epochs=1)

    trainer = Trainer(
        _RandomScaleModel(),
        _loaders(seed=17),
        _config(split_dir, epochs=2),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="own checkpoints/last.pt"):
        trainer.fit(resume_path=split_dir / "best.pt")


def test_rng_restore_normalizes_loaded_states_to_cpu(monkeypatch) -> None:
    cpu_state = torch.arange(16, dtype=torch.uint8)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        torch,
        "set_rng_state",
        lambda state: observed.setdefault("cpu", state),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: observed.setdefault("cuda", states),
    )

    _restore_rng_state(
        {
            "torch_cpu": cpu_state.clone(),
            "torch_cuda": [cpu_state.clone()],
        }
    )

    assert observed["cpu"].device.type == "cpu"
    assert observed["cpu"].dtype == torch.uint8
    assert observed["cuda"][0].device.type == "cpu"
    assert observed["cuda"][0].dtype == torch.uint8


@pytest.mark.parametrize(
    "value",
    [
        torch.arange(8, dtype=torch.uint8),
        list(range(8)),
    ],
)
def test_cpu_rng_state_normalization_accepts_tensor_and_legacy_list(value) -> None:
    normalized = _as_cpu_rng_state(value)

    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.uint8
    assert normalized.is_contiguous()
