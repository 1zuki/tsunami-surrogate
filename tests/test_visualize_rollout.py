from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import visualize
from src.evaluation.visualize import (
    CachedRolloutPlayer,
    RolloutFigure,
    VisualRollout,
    prepare_visual_rollout,
)
from src.evaluation.window_rollout import rollout_trajectory


class _DirectModel(torch.nn.Module):
    def __init__(self, frames: int) -> None:
        super().__init__()
        self.frames = int(frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (x.shape[0], self.frames, x.shape[2], x.shape[3]),
            dtype=x.dtype,
            device=x.device,
        )


class _IncrementWindowModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta_t = x[:, -2]
        return torch.stack([eta_t + 1.0, eta_t + 2.0], dim=1)


def _write_processed_sample(
    root: Path,
    *,
    solver_name: str,
    targets: np.ndarray,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> Path:
    processed = root / "data" / "processed" / solver_name / "test"
    shards = processed / "shards"
    shards.mkdir(parents=True)
    inputs = np.zeros((1, 3, targets.shape[-2], targets.shape[-1]), dtype=np.float32)
    payload = {
        "inputs": inputs,
        "targets": targets[None].astype(np.float32),
        "sample_id": np.asarray(["sample_000001"]),
        "scenario_id": np.asarray(["scenario_000001"]),
        "solver_name": np.asarray(
            ["swe_hydrostatic" if solver_name == "hydrostatic" else solver_name]
        ),
        "source_id": np.asarray(["gaussian"]),
        "source_type": np.asarray(["gaussian"]),
        "bathymetry_type": np.asarray(["smooth"]),
        "source_strength": np.asarray([1.0], dtype=np.float32),
        "input_order": np.asarray(
            ["bathymetry", "source", "initial_depth"]
        ),
    }
    if target_mean is not None and target_std is not None:
        payload["target_mean"] = np.asarray([target_mean], dtype=np.float32)
        payload["target_std"] = np.asarray([target_std], dtype=np.float32)
    np.savez(shards / "shard_00000.npz", **payload)
    (processed / "shards_manifest.json").write_text(
        json.dumps(
            {
                "num_samples": 1,
                "input_order": ["bathymetry", "source", "initial_depth"],
                "shards": [
                    {
                        "file": "shards/shard_00000.npz",
                        "num_samples": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (processed.parent / "normalization_stats.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "bathymetry": {"offset": -5.0, "scale": 2.0},
                    "source": {"offset": 0.0, "scale": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    return processed


def _write_raw_sample(
    root: Path,
    *,
    solver_name: str,
    bathymetry: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    sample_dir = (
        root
        / "data"
        / "test"
        / "raw"
        / solver_name
        / "samples"
        / "sample_000001"
    )
    sample_dir.mkdir(parents=True)
    np.savez(
        sample_dir / "sample.npz",
        bathymetry=np.asarray(bathymetry, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def test_window_rollout_reconstructs_full_future_sequence() -> None:
    model = _IncrementWindowModel()
    x = torch.zeros((1, 3, 4, 4))
    seed = torch.zeros((1, 4, 4))

    pred = rollout_trajectory(
        model,
        x,
        seed,
        T=6,
        K=2,
        include_source=True,
        use_prev=True,
        device="cpu",
    )

    assert pred.shape == (1, 5, 4, 4)
    assert torch.equal(
        pred[0, :, 0, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


def test_user_facing_sample_index_is_one_based() -> None:
    assert visualize._zero_based_sample_index(1) == 0
    assert visualize._zero_based_sample_index(7) == 6
    with pytest.raises(ValueError, match="must be >= 1"):
        visualize._zero_based_sample_index(0)


def test_prepare_visual_rollout_uses_boussinesq_raw_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    targets = np.zeros((3, 4, 4), dtype=np.float32)
    processed = _write_processed_sample(
        tmp_path,
        solver_name="boussinesq",
        targets=targets,
        target_mean=2.0,
        target_std=3.0,
    )
    bathymetry = -np.arange(16, dtype=np.float32).reshape(4, 4)
    timestamps = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    _write_raw_sample(
        tmp_path,
        solver_name="boussinesq",
        bathymetry=bathymetry,
        timestamps=timestamps,
    )

    monkeypatch.setattr(
        visualize,
        "load_config",
        lambda _path: {
            "device": "cpu",
            "data": {"windowed": False},
            "eval": {"dataset_path": str(processed)},
        },
    )
    monkeypatch.setattr(visualize, "build_model", lambda _cfg: _DirectModel(3))
    monkeypatch.setattr(
        visualize,
        "load_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    rollout = prepare_visual_rollout(
        "unused.yaml",
        "unused.pt",
        "auto",
        raw_dir="auto",
        device="cpu",
    )

    assert rollout.reference_name == "boussinesq"
    assert rollout.used_raw_bathymetry is True
    assert np.array_equal(rollout.bathymetry, bathymetry)
    assert np.allclose(rollout.timestamps, timestamps)
    assert np.allclose(rollout.target, 2.0)
    assert np.allclose(rollout.prediction, 2.0)


def test_prepare_visual_rollout_supports_windowed_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    targets = np.arange(6, dtype=np.float32)[:, None, None] * np.ones(
        (6, 4, 4),
        dtype=np.float32,
    )
    processed = _write_processed_sample(
        tmp_path,
        solver_name="hydrostatic",
        targets=targets,
    )
    _write_raw_sample(
        tmp_path,
        solver_name="hydrostatic",
        bathymetry=-np.ones((4, 4), dtype=np.float32),
        timestamps=np.arange(1, 7, dtype=np.float64),
    )

    monkeypatch.setattr(
        visualize,
        "load_config",
        lambda _path: {
            "device": "cpu",
            "data": {
                "windowed": True,
                "window_K": 2,
                "window_prev": True,
                "window_include_source": True,
            },
        },
    )
    monkeypatch.setattr(
        visualize,
        "build_model",
        lambda _cfg: _IncrementWindowModel(),
    )
    monkeypatch.setattr(
        visualize,
        "load_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    rollout = prepare_visual_rollout(
        "unused.yaml",
        "unused.pt",
        processed,
        raw_dir="auto",
        device="cpu",
    )

    assert rollout.prediction_mode == "seeded-window K=2"
    assert rollout.seeded_frames == 1
    assert np.array_equal(rollout.prediction, rollout.target)
    assert rollout.metrics["global_rel_l2"] == 0.0


def test_cached_player_uses_fixed_eta_scale_and_steps() -> None:
    target = np.stack(
        [
            np.full((4, 4), -0.5, dtype=np.float32),
            np.full((4, 4), 0.75, dtype=np.float32),
        ]
    )
    prediction = target * 0.8
    metrics = visualize._compute_metrics(prediction, target)
    rollout = VisualRollout(
        sample_id="sample_000001",
        reference_name="boussinesq",
        prediction_mode="direct",
        seeded_frames=0,
        bathymetry=-np.ones((4, 4), dtype=np.float32),
        target=target,
        prediction=prediction,
        uncertainty_std=None,
        timestamps=np.asarray([0.1, 0.2], dtype=np.float32),
        metrics=metrics,
        uncertainty_metrics={},
        target_denorm=(0.0, 1.0),
        used_raw_bathymetry=True,
        notes=(),
    )

    figure = RolloutFigure(
        rollout,
        interval_ms=100,
        wave_scale=1.0,
        eta_limit=2.0,
    )
    frames = figure.cache_frames(max_frames=2, dpi=40)

    assert len(frames) == 2
    assert all(frame.startswith(b"\x89PNG") for frame in frames)
    assert figure.im_true.get_clim() == (-2.0, 2.0)
    assert np.allclose(figure.ax_true_3d.get_zlim(), (-2.0, 2.0))

    player = CachedRolloutPlayer(
        frames,
        interval_ms=100,
        repeat=False,
        controls=True,
    )
    player._step()
    assert player.index == 1
    player._back()
    assert player.index == 0

    plt.close(figure.fig)
    plt.close(player.fig)
