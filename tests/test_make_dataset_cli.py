from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.make_dataset import _apply_overrides, _build_parser, _load_config


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/make_dataset.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_is_flat_and_keeps_the_original_entrypoint_shape() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--split" not in result.stdout
    assert "production" not in result.stdout
    assert "prepare-inputs" not in result.stdout
    assert "legacy-v1" not in result.stdout


def test_config_owns_dataset_identity_and_layout(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    expected = {
        "dataset": {
            "seed": 123,
            "num_samples": 17,
            "num_workers": 1,
            "bathymetry_dir": "data/custom/bathymetry",
            "source_dir": "data/custom/sources",
            "output_dir": "data/custom/raw",
            "manifest_path": "data/custom/synthetic/scenario_manifest.jsonl",
        },
        "requested_output": {"enabled": True, "split": "custom"},
        "operations": {"max_in_flight": 7},
    }
    config_path.write_text(yaml.safe_dump(expected), encoding="utf-8")
    args = _build_parser().parse_args(["--config", str(config_path)])
    cfg = _load_config(config_path)
    before = deepcopy(cfg)

    _apply_overrides(cfg, args)

    assert cfg == before
    assert cfg["operations"]["max_in_flight"] == 7
    assert not hasattr(args, "split")


def test_common_time_configs_define_split_seed_count_and_paths() -> None:
    expected = {
        "dataset.yaml": ("train", 42, 10000),
        "dataset_eval.yaml": ("eval", 69, 1000),
        "dataset_test.yaml": ("test", 367, 2500),
    }
    for name, (split, seed, count) in expected.items():
        cfg = _load_config(ROOT / "configs/data" / name)
        dataset = cfg["dataset"]
        root = f"data/{split}"
        assert cfg["requested_output"]["split"] == split
        assert dataset["seed"] == seed
        assert dataset["num_samples"] == count
        assert dataset["bathymetry_dir"] == f"{root}/bathymetry"
        assert dataset["source_dir"] == f"{root}/sources"
        assert dataset["output_dir"] == f"{root}/raw"
        assert dataset["manifest_path"] == (
            f"{root}/synthetic/scenario_manifest.jsonl"
        )


def test_saved_step_configs_do_not_target_canonical_split_data() -> None:
    canonical = tuple(Path("data") / split for split in ("train", "eval", "test"))
    for path in sorted((ROOT / "configs/data").rglob("*.yaml")):
        cfg = _load_config(path)
        dataset = cfg.get("dataset")
        if not isinstance(dataset, dict) or "output_dir" not in dataset:
            continue
        requested = cfg.get("requested_output")
        if isinstance(requested, dict) and requested.get("enabled") is True:
            continue
        output = Path(str(dataset["output_dir"]))
        assert not any(
            output == root or root in output.parents for root in canonical
        ), f"saved-step config targets canonical split data: {path}"


def test_internal_simulation_module_refuses_direct_execution() -> None:
    result = subprocess.run(
        [sys.executable, "src/data_gen/simulate_dataset.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "internal module" in result.stderr
