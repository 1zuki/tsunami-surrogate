import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.target_scaling import apply_target_denorm, load_target_denorm


def _write_eval_dataset(path: Path, target_mean: float, target_std: float) -> None:
    x = np.zeros((2, 3, 4, 4), dtype=np.float32)
    y = np.zeros((2, 5, 4, 4), dtype=np.float32)
    np.savez_compressed(
        path,
        inputs=x,
        targets=y,
        target_mean=np.asarray([target_mean], dtype=np.float32),
        target_std=np.asarray([target_std], dtype=np.float32),
    )


def test_load_target_denorm_respects_manifest_flag(tmp_path):
    split = tmp_path / "test"
    split.mkdir(parents=True, exist_ok=True)
    _write_eval_dataset(split / "eval_dataset.npz", target_mean=1.5, target_std=2.0)

    with (split / "eval_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"normalized_targets": True}, f)

    denorm = load_target_denorm(split)
    assert denorm is not None
    assert denorm[0] == 1.5
    assert denorm[1] == 2.0


def test_load_target_denorm_returns_none_when_manifest_disables_normalized_targets(tmp_path):
    split = tmp_path / "test"
    split.mkdir(parents=True, exist_ok=True)
    _write_eval_dataset(split / "eval_dataset.npz", target_mean=1.5, target_std=2.0)

    with (split / "eval_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"normalized_targets": False}, f)

    assert load_target_denorm(split) is None


def test_apply_target_denorm_identity_and_scaling():
    x = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    assert torch.equal(apply_target_denorm(x, None), x)
    y = apply_target_denorm(x, (0.5, 2.0))
    assert torch.allclose(y, torch.tensor([[2.5, -1.5]], dtype=torch.float32))
