from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.train import Trainer
from src.utils.seed import make_torch_generator, seed_everything


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


def _loaders(seed: int, batch_size: int = 2) -> dict[str, DataLoader]:
    dataset = _DictDataset()
    return {
        "train": DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=make_torch_generator(seed),
        ),
        "val": DataLoader(dataset, batch_size=4, shuffle=False),
    }


def _config(output_dir: Path, epochs: int) -> dict:
    return {
        "output_dir": str(output_dir),
        "seed": 17,
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
