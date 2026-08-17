from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.train as train_script
from scripts.train_ensemble import (
    _parse_resume_members,
    _parse_seeds,
    _require_fresh_member,
)
from src.utils.config import load_config
from src.utils.experiment import init_run


LOCAL_SINGLE_SEED_CONFIGS = (
    "configs/model/fno.yaml",
    "configs/model/ffno.yaml",
    "configs/model/cnn.yaml",
    "configs/model/unet.yaml",
    "configs/model/convlstm.yaml",
    "configs/model/ufno.yaml",
    "configs/model/wno.yaml",
    "configs/model/fno_modes8.yaml",
    "configs/model/fno_modes20.yaml",
)

FINAL_V2_MULTISEED_CONFIGS = (
    "configs/model/multiseed/fno_hydrostatic.yaml",
    "configs/model/multiseed/fno_muscl_hr.yaml",
    "configs/model/multiseed/fno_boussinesq.yaml",
    "configs/model/multiseed/ffno_hydrostatic.yaml",
    "configs/model/multiseed/unet_hydrostatic.yaml",
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


def test_main_can_select_one_seed_from_seed_list(tmp_path, monkeypatch) -> None:
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
        ["train.py", "--config", str(config_path), "--seed", "36"],
    )

    train_script.main()

    assert observed == [
        (36, "experiments/fno/fno_seed_36", "cpu", None),
    ]


def test_main_rejects_unconfigured_seed(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "fno.yaml"
    config_path.write_text(
        "output_dir: experiments/fno\nseeds: [18, 36]\ndevice: cpu\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--seed", "67"],
    )

    with pytest.raises(SystemExit):
        train_script.main()


def test_multi_seed_runs_isolate_evaluation_outputs(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "fno.yaml"
    config_path.write_text(
        "output_dir: experiments/fno\n"
        "seeds: [18, 36]\n"
        "device: cpu\n"
        "eval:\n"
        "  output_dir: experiments/fno/eval\n",
        encoding="utf-8",
    )
    observed = []

    monkeypatch.setattr(train_script, "resolve_device", lambda device: device)
    monkeypatch.setattr(
        train_script,
        "train_one",
        lambda cfg, device, resume_path=None: observed.append(
            (
                cfg["seed"],
                cfg["output_dir"],
                cfg["eval"]["output_dir"],
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path)],
    )

    train_script.main()

    assert observed == [
        (
            18,
            "experiments/fno/fno_seed_18",
            "experiments/fno/fno_seed_18/eval",
        ),
        (
            36,
            "experiments/fno/fno_seed_36",
            "experiments/fno/fno_seed_36/eval",
        ),
    ]


@pytest.mark.parametrize("config_path", LOCAL_SINGLE_SEED_CONFIGS)
def test_local_model_configs_use_seed_18(config_path) -> None:
    cfg = load_config(config_path)

    assert cfg["seed"] == 18
    assert "seeds" not in cfg


@pytest.mark.parametrize("config_path", FINAL_V2_MULTISEED_CONFIGS)
def test_final_v2_multiseed_configs_add_only_missing_seeds(config_path) -> None:
    cfg = load_config(config_path)

    assert cfg["seed"] == 18
    assert cfg["seeds"] == [36, 67]
    assert cfg["output_dir"].startswith("experiments/multiseed_v2/")
    assert cfg["eval"]["output_dir"].startswith("experiments/multiseed_v2/")


def test_uncertainty_ensemble_keeps_its_member_seed_protocol() -> None:
    cfg = load_config("configs/model/fno_ensemble.yaml")

    assert "seeds" not in cfg
    assert cfg["ensemble"]["seeds"] == [11, 22, 33, 44, 55, 66, 77]


def test_ordinary_fno_does_not_implicitly_select_ensemble_members() -> None:
    cfg = load_config("configs/model/fno.yaml")

    assert "seeds" not in cfg.get("ensemble", {})


def test_ensemble_cli_selection_and_resume_parsing() -> None:
    assert _parse_seeds("44,55", None) == [44, 55]
    assert _parse_resume_members(["44=experiments/ensemble/member_44/checkpoints/last.pt"]) == {
        44: Path("experiments/ensemble/member_44/checkpoints/last.pt")
    }


@pytest.mark.parametrize("seeds", [[11.0], [True], ["11"]])
def test_configured_ensemble_seeds_must_be_actual_integers(seeds) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        _parse_seeds(None, seeds)


def test_ensemble_training_refuses_existing_member_artifacts(
    tmp_path: Path,
) -> None:
    member = tmp_path / "member_11"
    member.mkdir()
    (member / "history.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _require_fresh_member(member)


def test_fresh_training_refuses_existing_run_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "history.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        init_run(output, {"output_dir": str(output)}, fresh=True)


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("configs/model").glob("*.yaml")),
)
def test_model_config_top_level_seed_lists_are_well_formed(
    config_path: Path,
) -> None:
    cfg = load_config(config_path)
    if "seeds" not in cfg:
        return

    seeds = cfg["seeds"]
    assert isinstance(seeds, list) and seeds
    assert all(
        isinstance(seed, int) and not isinstance(seed, bool)
        for seed in seeds
    )
    assert len(seeds) == len(set(seeds))
