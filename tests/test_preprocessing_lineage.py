from __future__ import annotations

import json
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
