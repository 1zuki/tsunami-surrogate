from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.cluster_run_state import classify_run


def _write_run(
    run_dir: Path,
    *,
    seed: int = 18,
    epoch: int = 2,
    epochs: int = 5,
    early_count: int = 0,
    patience: int = 3,
) -> None:
    (run_dir / "checkpoints").mkdir(parents=True)
    config = {
        "seed": seed,
        "train": {
            "epochs": epochs,
            "early_stopping": {"patience": patience},
        },
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    history = [{"epoch": value, "val_rel_l2": 1.0 / value} for value in range(1, epoch + 1)]
    (run_dir / "history.json").write_text(
        json.dumps(history) + "\n", encoding="utf-8"
    )
    payload = {
        "config": config,
        "epoch": epoch,
        "metrics": history[-1],
        "trainer_state": {
            "epoch": epoch,
            "early_count": early_count,
        },
    }
    torch.save(payload, run_dir / "checkpoints" / "last.pt")
    torch.save(payload, run_dir / "best.pt")


def test_empty_run_starts_fresh(tmp_path: Path) -> None:
    assert classify_run(tmp_path / "absent", 18)["action"] == "fresh"


def test_complete_artifacts_resume_from_own_last_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    result = classify_run(run_dir, 18)

    assert result["action"] == "resume"
    assert result["checkpoint"] == (run_dir / "checkpoints/last.pt").as_posix()


def test_completed_run_is_skipped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, epoch=5, epochs=5)

    assert classify_run(run_dir, 18)["action"] == "skip"


def test_partial_run_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "history.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="partial run artifacts"):
        classify_run(run_dir, 18)


def test_wrong_seed_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, seed=36)

    with pytest.raises(ValueError, match="checkpoint seed mismatch"):
        classify_run(run_dir, 18)


def test_changed_generated_config_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        "seed: 18\ntrain:\n  epochs: 6\n  early_stopping:\n    patience: 3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the generated config"):
        classify_run(run_dir, 18, changed)
