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


def test_completed_run_reuses_provenance_only_config_difference(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, epoch=5, epochs=5)
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        (
            "seed: 18\n"
            "train:\n"
            "  epochs: 5\n"
            "  early_stopping:\n"
            "    patience: 3\n"
            "output_dir: experiments/pooled/seed_18\n"
            "eval:\n"
            "  output_dir: experiments/pooled/seed_18/eval\n"
            "cluster_suite:\n"
            "  suite_id: pooled_reference_ablation_r6\n"
        ),
        encoding="utf-8",
    )

    result = classify_run(run_dir, 18, changed)

    assert result["action"] == "skip"
    assert "reusing compatible completed artifact" in result["reason"]


@pytest.mark.parametrize("completed", [False, True])
def test_scientific_eval_change_fails_closed(tmp_path: Path, completed: bool) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, epoch=5 if completed else 2, epochs=5)
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    config["eval"] = {"dataset_path": "different/test"}
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the generated config"):
        classify_run(run_dir, 18, changed)


def test_incomplete_run_rejects_provenance_only_config_change(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    config["output_dir"] = "different/output"
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the generated config"):
        classify_run(run_dir, 18, changed)


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
