from __future__ import annotations

from pathlib import Path

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
