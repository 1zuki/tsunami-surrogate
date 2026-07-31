from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_sample_scaling import _build_config, _resolve_single_seed


def test_multi_seed_base_requires_explicit_sample_scaling_seed() -> None:
    cfg = {"seeds": [18, 36, 67]}

    with pytest.raises(ValueError, match="requires exactly one seed"):
        _resolve_single_seed(cfg, None)


def test_explicit_seed_normalizes_config_to_single_seed() -> None:
    cfg = {"seed": 7, "seeds": [18, 36, 67]}

    seed = _resolve_single_seed(cfg, 42)

    assert seed == 42
    assert cfg["seed"] == 42
    assert "seeds" not in cfg


def test_build_config_uses_direct_single_run_output(tmp_path: Path) -> None:
    cfg, run_dir, config_path = _build_config(
        "configs/model/fno.yaml",
        tmp_path / "sample_scaling",
        100,
        "cuda",
        None,
        None,
        42,
    )

    assert cfg["seed"] == 42
    assert "seeds" not in cfg
    assert cfg["data"]["n_samples"] == 100
    assert cfg["output_dir"] == str(run_dir)
    assert cfg["sample_scaling"]["seed"] == 42
    assert config_path.is_file()
