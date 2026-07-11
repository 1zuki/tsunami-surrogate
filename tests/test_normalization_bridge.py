import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.normalization_bridge import (
    load_evaluation_normalization_bridge,
)


def _write_stats(
    path: Path,
    *,
    bathymetry: tuple[float, float],
    source: tuple[float, float],
    target: tuple[float, float],
) -> None:
    payload = {
        "method": "standardize",
        "inputs": {
            "bathymetry": {"offset": bathymetry[0], "scale": bathymetry[1]},
            "source": {"offset": source[0], "scale": source[1]},
        },
        "targets": {
            "enabled": True,
            "variable": "eta",
            "offset": target[0],
            "scale": target[1],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "eval_dataset.npz",
        inputs=np.zeros((1, 3, 2, 2), dtype=np.float32),
        targets=np.zeros((1, 1, 2, 2), dtype=np.float32),
        input_order=np.asarray(
            ["bathymetry", "source", "initial_depth"], dtype=np.str_
        ),
    )


def test_bridge_rebases_inputs_and_targets_to_model_stats(tmp_path):
    dataset = tmp_path / "test"
    _write_dataset(dataset)
    dataset_stats = tmp_path / "dataset_stats.json"
    model_stats = tmp_path / "model_stats.json"
    _write_stats(
        dataset_stats,
        bathymetry=(10.0, 2.0),
        source=(-1.0, 4.0),
        target=(5.0, 3.0),
    )
    _write_stats(
        model_stats,
        bathymetry=(8.0, 4.0),
        source=(1.0, 2.0),
        target=(2.0, 6.0),
    )

    bridge = load_evaluation_normalization_bridge(dataset, dataset_stats, model_stats)
    x_dataset = torch.tensor([[[[2.0]], [[2.0]], [[0.25]]]], dtype=torch.float32)
    y_dataset = torch.tensor([[[[2.0]]]], dtype=torch.float32)

    x_model, y_model = bridge.transform(x_dataset, y_dataset)

    expected_x = torch.tensor([[[[1.5]], [[3.0]], [[0.25]]]], dtype=torch.float32)
    expected_y = torch.tensor([[[[1.5]]]], dtype=torch.float32)
    assert torch.allclose(x_model, expected_x)
    assert torch.allclose(y_model, expected_y)
    assert bridge.model_target_denorm == (2.0, 6.0)
    assert bridge.metadata()["comparison_normalization"] == "model_training_stats"


def test_bridge_fails_loudly_on_input_channel_mismatch(tmp_path):
    dataset = tmp_path / "test"
    _write_dataset(dataset)
    dataset_stats = tmp_path / "dataset_stats.json"
    model_stats = tmp_path / "model_stats.json"
    _write_stats(
        dataset_stats,
        bathymetry=(0.0, 1.0),
        source=(0.0, 1.0),
        target=(0.0, 1.0),
    )
    _write_stats(
        model_stats,
        bathymetry=(0.0, 1.0),
        source=(0.0, 1.0),
        target=(0.0, 1.0),
    )
    bridge = load_evaluation_normalization_bridge(dataset, dataset_stats, model_stats)

    with pytest.raises(ValueError, match="Input channel mismatch"):
        bridge.transform(
            torch.zeros((1, 2, 2, 2), dtype=torch.float32),
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        )


def test_bridge_fails_loudly_when_stats_are_missing(tmp_path):
    dataset = tmp_path / "test"
    _write_dataset(dataset)

    with pytest.raises(FileNotFoundError):
        load_evaluation_normalization_bridge(
            dataset,
            tmp_path / "missing_dataset_stats.json",
            tmp_path / "missing_model_stats.json",
        )
