from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.train as train_script
from src.utils.config import load_config


FIVE_SEED_CONFIGS = (
    "configs/model/fno.yaml",
    "configs/model/ffno.yaml",
)

THREE_SEED_CONFIGS = tuple(
    str(path)
    for path in sorted(Path("configs/model").glob("*.yaml"))
    if path.name not in {"fno.yaml", "ffno.yaml", "fno_ensemble_m8.yaml"}
)


def test_single_seed_preserves_existing_output_directory() -> None:
    seeds, list_mode = train_script.resolve_training_seeds(
        {"seed": 42, "output_dir": "experiments/fno"}
    )

    assert seeds == [42]
    assert list_mode is False
    assert (
        train_script.seed_output_dir("experiments/fno", seeds[0], list_mode)
        .as_posix()
        == "experiments/fno"
    )


def test_seed_list_uses_isolated_run_directories() -> None:
    seeds, list_mode = train_script.resolve_training_seeds(
        {"seed": 42, "seeds": [18, 36], "output_dir": "experiments/fno"}
    )

    assert seeds == [18, 36]
    assert list_mode is True
    assert [
        train_script.seed_output_dir("experiments/fno", seed, list_mode).as_posix()
        for seed in seeds
    ] == [
        "experiments/fno/fno_seed_18",
        "experiments/fno/fno_seed_36",
    ]


@pytest.mark.parametrize("seeds", [[], [18, 18], [18, "36"], 18])
def test_invalid_seed_lists_are_rejected(seeds) -> None:
    with pytest.raises(ValueError):
        train_script.resolve_training_seeds({"seeds": seeds})


def test_main_runs_seed_list_sequentially(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "fno.yaml"
    config_path.write_text(
        "output_dir: experiments/fno\nseeds: [18, 36]\ndevice: cpu\n",
        encoding="utf-8",
    )
    observed = []

    monkeypatch.setattr(train_script, "resolve_device", lambda device: device)
    monkeypatch.setattr(
        train_script,
        "train_one",
        lambda cfg, device, resume_path=None: observed.append(
            (cfg["seed"], cfg["output_dir"], device, resume_path)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path)],
    )

    train_script.main()

    assert observed == [
        (18, "experiments/fno/fno_seed_18", "cpu", None),
        (36, "experiments/fno/fno_seed_36", "cpu", None),
    ]


@pytest.mark.parametrize("config_path", FIVE_SEED_CONFIGS)
def test_headline_configs_use_five_frozen_seeds(config_path) -> None:
    cfg = load_config(config_path)

    assert cfg["seeds"] == [18, 36, 67, 72, 154]


@pytest.mark.parametrize("config_path", THREE_SEED_CONFIGS)
def test_secondary_configs_use_three_frozen_seeds(config_path) -> None:
    cfg = load_config(config_path)

    assert cfg["seeds"] == [18, 36, 67]


def test_uncertainty_ensemble_keeps_its_member_seed_protocol() -> None:
    cfg = load_config("configs/model/fno_ensemble_m8.yaml")

    assert "seeds" not in cfg
    assert cfg["ensemble"]["seeds"] == [44, 55, 66, 77]
