from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_yaml

from scripts.prepare_cluster_training_suite import (
    DEFAULT_MANIFEST,
    ROOT,
    build_sbatch_command,
    prepare_suite,
)


def test_real_legacy_dev_manifest_prepares_one_seed_per_job(
    tmp_path: Path,
) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    assert suite.suite_id == "legacy_dev_full_v1"
    assert suite.classification == "legacy_dev_only"
    assert len(suite.runs) == 33
    assert suite.max_concurrent == 5
    assert {entry["name"] for entry in suite.disabled_entries} == {
        "fno_native_res32",
        "fno_native_res64",
        "fno_native_res128",
    }

    labels = [run.label for run in suite.runs]
    outputs = [run.output_dir for run in suite.runs]
    assert len(labels) == len(set(labels))
    assert len(outputs) == len(set(outputs))

    for run in suite.runs:
        cfg = load_yaml(run.config_path)
        assert "seeds" not in cfg
        assert cfg["seed"] == run.seed
        assert cfg["device"] == "cuda"
        assert cfg["output_dir"] == run.output_dir
        assert cfg["cluster_suite"]["classification"] == "legacy_dev_only"


def test_sample_scaling_entries_preserve_historical_single_seed(
    tmp_path: Path,
) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )
    by_label = {run.label: run for run in suite.runs}

    expected = {
        "fno_sample_100_seed_42": 100,
        "fno_sample_250_seed_42": 250,
    }
    for label, n_samples in expected.items():
        run = by_label[label]
        cfg = load_yaml(run.config_path)
        assert cfg["seed"] == 42
        assert cfg["data"]["n_samples"] == n_samples
        assert cfg["sample_scaling"]["requested_train_samples"] == n_samples


def test_submit_command_uses_smoke_dependency_and_account_concurrency(
    tmp_path: Path,
) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    cmd = build_sbatch_command(suite, afterok=55523, root=ROOT)

    assert cmd[0] == "sbatch"
    assert "--array=0-32%5" in cmd
    assert "--dependency=afterok:55523" in cmd
    assert "GPU_HELPER_VERIFIED=1" in " ".join(cmd)
    assert cmd[-1] == "slurm/train_suite_array.slurm"


def test_submit_command_can_reduce_array_concurrency(tmp_path: Path) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    cmd = build_sbatch_command(
        suite,
        afterok=55523,
        max_concurrent=3,
        root=ROOT,
    )

    assert "--array=0-32%3" in cmd


@pytest.mark.parametrize("max_concurrent", [0, 6])
def test_submit_command_rejects_invalid_array_concurrency(
    tmp_path: Path,
    max_concurrent: int,
) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    with pytest.raises(ValueError, match="manifest limit"):
        build_sbatch_command(
            suite,
            afterok=55523,
            max_concurrent=max_concurrent,
            root=ROOT,
        )


@pytest.mark.parametrize(
    "script_name",
    [
        "gpu_helper_probe.slurm",
        "train_fno.slurm",
        "train_suite_array.slurm",
    ],
)
def test_slurm_scripts_preserve_project_python_before_system_paths(
    script_name: str,
) -> None:
    text = (ROOT / "slurm" / script_name).read_text(encoding="utf-8")

    assert 'export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"' in text


@pytest.mark.parametrize(
    "script_name",
    [
        "gpu_helper_probe.slurm",
        "train_fno.slurm",
        "train_suite_array.slurm",
    ],
)
def test_slurm_scripts_reject_known_gpu_helper_typo(
    script_name: str,
) -> None:
    text = (ROOT / "slurm" / script_name).read_text(encoding="utf-8")

    assert 'grep -Fq "nvidia-smi-i "' in text
    assert "nvidia-smi -i" in text
