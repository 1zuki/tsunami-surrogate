from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest

import scripts.train_kaggle_convlstm as kaggle


def _write_dataset(root: Path) -> None:
    (root / "normalization_stats.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "normalization_stats.json").write_text(
        json.dumps({"inputs": {}, "targets": {}}),
        encoding="utf-8",
    )
    for split, count in kaggle.EXPECTED_SPLIT_COUNTS.items():
        split_root = root / split
        shard = split_root / "shards" / "shard_00000.npz"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"npz")
        (split_root / "shards_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "sharded": True,
                    "split": split,
                    "input_order": kaggle.EXPECTED_INPUT_ORDER,
                    "target_variable": "eta",
                    "normalized_targets": True,
                    "num_samples": count,
                    "shards": [
                        {
                            "file": "shards/shard_00000.npz",
                            "num_samples": count,
                            "inputs_shape": [count, 3, 64, 64],
                            "targets_shape": [count, 50, 64, 64],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


def test_validate_dataset_root_accepts_uploaded_sharded_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tsunami-surrogate-hydrostatic"
    _write_dataset(root)

    summary = kaggle.validate_dataset_root(root)

    assert summary["splits"]["train"]["num_samples"] == 10_000
    assert summary["splits"]["val"]["num_samples"] == 1_000
    assert summary["splits"]["test"]["num_samples"] == 2_500


def test_validate_dataset_root_rejects_missing_uploaded_shard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tsunami-surrogate-hydrostatic"
    _write_dataset(root)
    (root / "train" / "shards" / "shard_00000.npz").unlink()

    with pytest.raises(FileNotFoundError, match="processed shard"):
        kaggle.validate_dataset_root(root)


def test_runtime_config_preserves_portable_paths() -> None:
    cfg, run_dir = kaggle.build_runtime_config(
        36,
        Path("experiments/multiseed_v2/convlstm_hydrostatic"),
        batch_size=4,
        num_workers=2,
    )

    assert cfg["seed"] == 36
    assert cfg["device"] == "cuda"
    assert cfg["data"]["train_path"] == (
        "data/processed/hydrostatic/train/eval_dataset.npz"
    )
    assert run_dir.as_posix().endswith("convlstm_hydrostatic_seed_36")
    assert cfg["output_dir"] == run_dir.as_posix()
    assert cfg["eval"]["output_dir"] == (run_dir / "eval").as_posix()


def test_canonical_data_location_uses_uploaded_dataset_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    uploaded = tmp_path / "input" / "tsunami-surrogate-hydrostatic"
    uploaded.mkdir(parents=True)
    monkeypatch.setattr(kaggle, "ROOT", root)

    canonical = kaggle.ensure_canonical_data_location(uploaded)

    assert canonical.is_symlink()
    assert canonical.resolve() == uploaded.resolve()


def test_classify_run_is_fail_closed_and_resume_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert kaggle.classify_run(run_dir, "auto") == "fresh"

    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text("seed: 36\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no safe resume checkpoint"):
        kaggle.classify_run(run_dir, "auto")

    for relative in kaggle.RESUME_ARTIFACTS:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    assert kaggle.classify_run(run_dir, "auto") == "resume"

    (run_dir / "kaggle_run_status.json").write_text(
        json.dumps({"state": "completed"}),
        encoding="utf-8",
    )
    assert kaggle.classify_run(run_dir, "auto") == "skip"


def test_last_log_line_reports_latest_nonempty_line(tmp_path: Path) -> None:
    log = tmp_path / "seed.log"
    log.write_text("first\n\nlatest\n", encoding="utf-8")

    assert kaggle._last_log_line(log) == "latest"


def test_archive_preserves_repository_relative_experiment_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    output_root = Path("experiments/multiseed_v2/convlstm_hydrostatic")
    artifact = root / output_root / "convlstm_hydrostatic_seed_36" / "best.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"checkpoint")
    monkeypatch.setattr(kaggle, "ROOT", root)

    archive_path = kaggle.archive_outputs(
        output_root,
        tmp_path / "convlstm_multiseed_v2.tar.gz",
    )

    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert (
        "experiments/multiseed_v2/convlstm_hydrostatic/"
        "convlstm_hydrostatic_seed_36/best.pt"
    ) in names
