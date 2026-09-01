from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.dataset import TsunamiDataset
from src.data_gen.preprocess import (
    PROCESSED_MANIFEST_SCHEMA_ID,
    TsunamiPreprocessor,
)
from src.utils.hashing import sha256_file


def _config(tmp_path: Path) -> Path:
    payload = {
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "manifest_path": str(tmp_path / "manifest.jsonl"),
        "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 7},
        "input": {
            "use_bathymetry": True,
            "use_source": True,
            "use_initial_depth": False,
        },
        "target": {
            "mode": "multi_step",
            "variable": "eta",
            "forecast_steps": 1,
            "stride": 1,
        },
        "normalization": {
            "method": "standardize",
            "channels": {
                "bathymetry": True,
                "source": True,
                "trajectory": True,
            },
        },
        "saving": {
            "sharded": False,
            "include_meta": True,
        },
        "eval_export": {"enabled": True},
    }
    path = tmp_path / "preprocess.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _stats_payload() -> dict:
    return {
        "method": "standardize",
        "inputs": {
            "bathymetry": {"offset": -1.0, "scale": 0.5},
            "source": {"offset": 0.0, "scale": 0.25},
        },
        "targets": {
            "enabled": True,
            "variable": "eta",
            "offset": 0.0,
            "scale": 0.1,
            "min": -1.0,
            "max": 1.0,
        },
    }


def test_no_train_records_requires_an_explicit_normalization_reference(
    tmp_path: Path,
) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    with pytest.raises(ValueError, match="implicit path-based"):
        preprocessor._resolve_normalization_reference_for_run(
            output_dir=tmp_path / "processed",
            train_records=[],
            norm_reference_stats_path=None,
        )


def test_solver_rosters_must_match_exactly(tmp_path: Path) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    split_source = [
        {"scenario_id": "scenario_000001"},
        {"scenario_id": "scenario_000002"},
    ]
    records = {
        "hydrostatic": [
            {
                "scenario_id": "scenario_000001",
                "solver_name": "swe_hydrostatic",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="roster mismatch"):
        preprocessor._validate_solver_rosters(
            targets=["hydrostatic"],
            split_source=split_source,
            records_by_target=records,
        )


def test_v2_sample_requires_publication_record(tmp_path: Path) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    sample_dir = tmp_path / "sample_000001"
    sample_dir.mkdir()
    np.savez(sample_dir / "sample.npz", trajectory_eta=np.zeros((1, 2, 2)))
    (sample_dir / "meta.json").write_text(
        json.dumps({"contract_hash": "common-time-v2-contract"}),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Common-time-v2 preprocessing requires publication.json",
    ):
        preprocessor.load_sample(
            sample_dir,
            expected_record={
                "scenario_id": "scenario_000001",
                "contract_hash": "common-time-v2-contract",
            },
        )


def test_v2_payload_schema_requires_publication_when_metadata_is_stripped(
    tmp_path: Path,
) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    sample_dir = tmp_path / "sample_000001"
    sample_dir.mkdir()
    np.savez(
        sample_dir / "sample.npz",
        trajectory_eta=np.zeros((1, 2, 2)),
        schema_id=np.asarray(
            ["tsunami-surrogate.common-time-v2.eta-sample.v1"]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Common-time-v2 preprocessing requires publication.json",
    ):
        preprocessor.load_sample(sample_dir)


def test_processed_directory_publication_restores_existing_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".processed.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected"):
        TsunamiPreprocessor._publish_processed_directory(staging, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "old"


def test_processed_split_publication_preserves_siblings_and_training_stats(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    stats = output / "normalization_stats.json"
    stats.write_text(json.dumps(_stats_payload(), indent=2), encoding="utf-8")
    stats_before = stats.read_bytes()
    for split_name in ("train", "val", "test"):
        split_dir = output / split_name
        split_dir.mkdir()
        (split_dir / "marker.txt").write_text(
            f"old-{split_name}",
            encoding="utf-8",
        )

    staging = tmp_path / ".processed.staging"
    staging.mkdir()
    (staging / "normalization_stats.json").write_bytes(stats_before)
    staged_val = staging / "val"
    staged_val.mkdir()
    (staged_val / "marker.txt").write_text("new-val", encoding="utf-8")

    TsunamiPreprocessor._publish_processed_split(staging, output, "val")

    assert stats.read_bytes() == stats_before
    assert (output / "train" / "marker.txt").read_text() == "old-train"
    assert (output / "val" / "marker.txt").read_text() == "new-val"
    assert (output / "test" / "marker.txt").read_text() == "old-test"
    assert not staging.exists()


def test_processed_split_publication_restores_existing_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    stats_payload = json.dumps(_stats_payload(), indent=2)
    (output / "normalization_stats.json").write_text(
        stats_payload,
        encoding="utf-8",
    )
    old_val = output / "val"
    old_val.mkdir()
    (old_val / "marker.txt").write_text("old-val", encoding="utf-8")

    staging = tmp_path / ".processed.staging"
    staging.mkdir()
    (staging / "normalization_stats.json").write_text(
        stats_payload,
        encoding="utf-8",
    )
    new_val = staging / "val"
    new_val.mkdir()
    (new_val / "marker.txt").write_text("new-val", encoding="utf-8")

    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected split publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected split"):
        TsunamiPreprocessor._publish_processed_split(
            staging,
            output,
            "val",
        )

    assert (output / "val" / "marker.txt").read_text() == "old-val"


def test_merge_split_copies_training_stats_without_rewriting_them(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["split"] = {"train": 0.0, "val": 1.0, "test": 0.0, "seed": 7}
    payload["saving"]["publication_mode"] = "merge_split"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    preprocessor = TsunamiPreprocessor(str(config_path))
    reference = tmp_path / "training-normalization-stats.json"
    reference.write_text(
        json.dumps(_stats_payload(), indent=2),
        encoding="utf-8",
    )
    reference_before = reference.read_bytes()
    staging = tmp_path / "staging"
    staging.mkdir()

    preprocessor.cfg.processed_dir = staging
    preprocessor._active_norm_reference_path = reference
    preprocessor._loaded_normalization_payload = _stats_payload()
    preprocessor._save_normalization_stats([])

    assert (staging / "normalization_stats.json").read_bytes() == reference_before
    assert preprocessor._normalization_stats_sha256 == sha256_file(reference)


def test_merge_split_requires_training_preprocessing_first(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["split"] = {"train": 0.0, "val": 1.0, "test": 0.0, "seed": 7}
    payload["saving"]["publication_mode"] = "merge_split"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    preprocessor = TsunamiPreprocessor(str(config_path))
    with pytest.raises(
        FileNotFoundError,
        match="Run train preprocessing first",
    ):
        preprocessor._normalize_and_save(
            [],
            [{"scenario_id": "scenario_000001"}],
            [],
            tmp_path / "published",
            norm_reference_stats_path=tmp_path / "missing-stats.json",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(method="minmax"), "method mismatch"),
        (
            lambda payload: payload["targets"].update(variable="depth"),
            "target mismatch",
        ),
        (
            lambda payload: payload["targets"].update(scale=-0.1),
            "positive target scale",
        ),
        (
            lambda payload: payload["inputs"].pop("source"),
            "missing required input channels",
        ),
    ],
)
def test_explicit_normalization_reference_is_strictly_validated(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    payload = _stats_payload()
    mutation(payload)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, KeyError), match=message):
        preprocessor._load_normalizer_from_stats_file(
            stats_path,
            sample_inputs={
                "bathymetry": np.zeros((2, 2), dtype=np.float32),
                "source": np.zeros((2, 2), dtype=np.float32),
            },
        )


def test_same_file_normalization_reference_is_not_overwritten(tmp_path: Path) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    root = tmp_path / "processed"
    root.mkdir(exist_ok=True)
    stats_path = root / "normalization_stats.json"
    stats_path.write_text(json.dumps(_stats_payload(), indent=2), encoding="utf-8")
    before = stats_path.read_bytes()

    preprocessor.cfg.processed_dir = root
    preprocessor._active_norm_reference_path = stats_path
    preprocessor._loaded_normalization_payload = _stats_payload()
    preprocessor._save_normalization_stats([])

    assert stats_path.read_bytes() == before
    assert preprocessor._normalization_stats_sha256 == sha256_file(stats_path)
    assert (
        preprocessor._normalization_provenance["mode"]
        == "explicit_reference_in_place"
    )


def test_flat_v2_outputs_bind_lineage_and_reject_archive_corruption(
    tmp_path: Path,
) -> None:
    preprocessor = TsunamiPreprocessor(str(_config(tmp_path)))
    root = tmp_path / "processed"
    root.mkdir(exist_ok=True)
    stats_path = root / "normalization_stats.json"
    stats_path.write_text(json.dumps(_stats_payload()), encoding="utf-8")
    preprocessor._normalization_stats_sha256 = sha256_file(stats_path)
    preprocessor._normalization_provenance = {"mode": "test"}

    inputs = {
        "bathymetry": np.full((2, 2), -1.0, dtype=np.float32),
        "source": np.ones((2, 2), dtype=np.float32),
    }
    target = np.zeros((1, 2, 2), dtype=np.float32)
    metadata = {
        "sample_index": 1,
        "scenario_id": "scenario_000001",
        "solver_name": "swe_hydrostatic",
        "contract_hash": "contract",
        "resolved_config_hash": "config",
        "code_state_hash": "code",
        "input_fingerprint": "input",
    }
    preprocessor.save_split(
        "train",
        [inputs],
        [target],
        [metadata],
        ["sample_000001"],
    )

    split_dir = root / "train"
    manifest = json.loads(
        (split_dir / "eval_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_id"] == PROCESSED_MANIFEST_SCHEMA_ID
    assert manifest["provenance"]["source_lineage"]["status"] == "bound_common_time_v2"
    assert manifest["artifacts"]["eval_dataset.npz"] == sha256_file(
        split_dir / "eval_dataset.npz"
    )
    assert len(TsunamiDataset(split_dir)) == 1

    archive = split_dir / "eval_dataset.npz"
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="dataset hash mismatch"):
        TsunamiDataset(split_dir)
