from pathlib import Path

import pytest

from tsunami_surrogate.data_gen.preprocess import TsunamiPreprocessor
from tsunami_surrogate.utils.config import load_config


def test_nested_defaults_merge_without_changing_data_paths(tmp_path):
    (tmp_path / "base.yaml").write_text("data:\n  path: data/train\n  batch_size: 8\n")
    (tmp_path / "child").mkdir()
    child = tmp_path / "child/config.yaml"
    child.write_text("defaults: [../base.yaml]\ndata:\n  batch_size: 2\n")
    assert load_config(child) == {"data": {"path": "data/train", "batch_size": 2}}


def test_defaults_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\n")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("text", ["[one, two]", "[]", "false", "0", "defaults: base.yaml", "defaults: [1]"])
def test_invalid_configuration_structure_is_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("kind", ["anonymous", "conditioned"])
@pytest.mark.parametrize("split", ["train", "eval", "test"])
def test_pooled_preprocessor_uses_inherited_settings(kind, split):
    # Exercise the constructor, rather than just the standalone YAML loader.
    processor = TsunamiPreprocessor(
        f"configs/data/pooled_reference_ablation/preprocess_{split}_{kind}.yaml"
    )
    cfg = processor.cfg
    assert processor.fde_mode == "multifidelity"
    assert cfg.forecast_steps == 50
    assert cfg.sharded
    assert cfg.processed_dir == Path("data/processed/pooled_reference_ablation")
    assert cfg.use_solver_id == (kind == "conditioned")
    assert processor.expected_publication_split == split


def test_dataset_entrypoint_uses_inherited_generation_settings(tmp_path):
    from scripts.make_dataset import _load_config
    (tmp_path / "base.yaml").write_text("dataset:\n  num_samples: 2\nrequested_output:\n  count: 50\n")
    child = tmp_path / "child.yaml"
    child.write_text("defaults: [base.yaml]\ndataset:\n  num_samples: 1\n")
    assert _load_config(child) == {"dataset": {"num_samples": 1}, "requested_output": {"count": 50}}


def test_public_configs_load_without_personal_absolute_paths():
    import re

    def check(value):
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            assert not value.startswith(("/home/", "/mnt/", "/datastore/")), value
            assert not re.match(r"^[A-Za-z]:[\\/]", value), value

    for path in Path("configs").rglob("*.yaml"):
        if "generated" not in path.parts:
            check(load_config(path))
