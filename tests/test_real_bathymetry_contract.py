from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts.build_real_bathymetry_eval import (
    REAL_BATHYMETRY_LINEAGE_SCHEMA_ID,
    _dataset_config,
    _write_paired_external_input,
)
from src.data_gen.simulate_dataset import TsunamiDatasetBuilder, _block_mean_downsample


ROOT = Path(__file__).resolve().parents[1]


def test_external_real_bathymetry_is_bound_to_the_paired_contract(
    tmp_path: Path,
) -> None:
    base = yaml.safe_load(
        (ROOT / "configs/data/dataset_test.yaml").read_text(encoding="utf-8")
    )
    suite_dir = tmp_path / "inputs" / "main_morphology_suite_10"
    config = _dataset_config(
        base=base,
        suite_name="main_morphology_suite_10",
        suite_dir=suite_dir,
        num_samples=1,
        out_root=tmp_path / "raw",
        seed=367,
        num_workers=1,
    )
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    builder = TsunamiDatasetBuilder(str(config_path))

    external = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    target = suite_dir / "sample_000001.npz"
    _write_paired_external_input(
        target=target,
        bathymetry=external,
        bathymetry_type="external",
        sample_seed=123,
        lineage={
            "suite": "main_morphology_suite_10",
            "source_kind": "test_external_raster",
        },
        paired=builder.dataset.paired_inputs,
    )

    with np.load(target, allow_pickle=False) as payload:
        assert payload["bathymetry"].shape == (64, 64)
        assert payload["master_bathymetry"].shape == (384, 384)
        assert payload["solver_bathymetry"].shape == (128, 128)
        np.testing.assert_array_equal(
            payload["bathymetry"],
            _block_mean_downsample(payload["master_bathymetry"], (64, 64)),
        )
        np.testing.assert_array_equal(
            payload["solver_bathymetry"],
            _block_mean_downsample(payload["master_bathymetry"], (128, 128)),
        )
        lineage = json.loads(str(payload["input_lineage_json"][0]))
        assert lineage["schema_id"] == REAL_BATHYMETRY_LINEAGE_SCHEMA_ID
        assert lineage["derivation"] == (
            "piecewise_constant_prolongation_64_to_384_then_block_mean"
        )
        assert lineage["paired_lineage_hash"] == (
            builder.dataset.paired_inputs.lineage_hash
        )
