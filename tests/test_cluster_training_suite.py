from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_yaml

from scripts.prepare_cluster_training_suite import (
    DEFAULT_MANIFEST,
    ROOT,
    build_finalizer_command,
    build_sbatch_command,
    prepare_suite,
)

LEGACY_MANIFEST = Path("configs/cluster/legacy_dev_suite.yaml")


def test_real_legacy_dev_manifest_prepares_one_seed_per_job(
    tmp_path: Path,
) -> None:
    suite = prepare_suite(
        LEGACY_MANIFEST,
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
        LEGACY_MANIFEST,
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
        LEGACY_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    cmd = build_sbatch_command(suite, afterok=55523, root=ROOT)

    assert cmd[0] == "sbatch"
    assert "--parsable" in cmd
    assert "--array=0-32%5" in cmd
    assert "--dependency=afterok:55523" in cmd
    assert cmd[-1] == "slurm/train_suite_array.slurm"


def test_submit_command_can_reduce_array_concurrency(tmp_path: Path) -> None:
    suite = prepare_suite(
        LEGACY_MANIFEST,
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


def test_final_rebuild_manifest_has_frozen_seed_policy(tmp_path: Path) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    assert suite.suite_id == "final_rebuild_multiseed_r1"
    assert suite.classification == "final_rebuild_training"
    assert suite.max_concurrent == 5
    assert len(suite.runs) == 60

    seeds_by_model: dict[str, set[int]] = {}
    for run in suite.runs:
        model = run.label.rsplit("_seed_", 1)[0]
        seeds_by_model.setdefault(model, set()).add(run.seed)

    major = {
        "fno_hydrostatic",
        "ffno_hydrostatic",
        "unet_hydrostatic",
        "convlstm_hydrostatic",
    }
    for model in major:
        assert seeds_by_model[model] == {18, 36, 67, 72, 154}

    three_seed = {
        "cnn_hydrostatic",
        "ufno_hydrostatic",
        "wno_hydrostatic",
        "fno_modes8_hydrostatic",
        "fno_modes20_hydrostatic",
        "fno_muscl_hr",
        "fno_boussinesq",
        "fno_window5_hydrostatic",
        "ffno_window5_hydrostatic",
    }
    for model in three_seed:
        assert seeds_by_model[model] == {18, 36, 67}

    assert seeds_by_model["ensemble_fno"] == {11, 22, 33, 44, 55, 66, 77}
    ensemble_cfg = load_yaml(
        next(run.config_path for run in suite.runs if run.label == "ensemble_fno_seed_11")
    )
    assert ensemble_cfg["cluster_suite"]["role"] == "uncertainty_ensemble"


def test_final_subset_runs_match_full_fno_seed(tmp_path: Path) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )
    subset_runs = [run for run in suite.runs if "fno_sample_" in run.label]

    assert len(subset_runs) == 6
    for run in subset_runs:
        cfg = load_yaml(run.config_path)
        assert run.seed == 18
        assert cfg["seed"] == 18
        assert cfg["sample_scaling"]["protocol"] == (
            "matched_seed_nested_training_subset"
        )


def test_final_submit_and_finalizer_commands(tmp_path: Path) -> None:
    suite = prepare_suite(
        DEFAULT_MANIFEST,
        tmp_path,
        root=ROOT,
        check_data=False,
    )

    submit = build_sbatch_command(suite, afterok=66555, root=ROOT)
    finalizer = build_finalizer_command(suite, array_job_id=70001)

    assert "--array=0-59%5" in submit
    assert "--dependency=afterok:66555" in submit
    assert "--dependency=afterany:70001" in finalizer
    assert finalizer[-1] == "slurm/finalize_training_suite.slurm"


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


@pytest.mark.parametrize(
    "script_name",
    [
        "gpu_helper_probe.slurm",
        "train_fno.slurm",
        "train_suite_array.slurm",
        "finalize_training_suite.slurm",
    ],
)
def test_slurm_jobs_use_tsunami_display_name(script_name: str) -> None:
    text = (ROOT / "slurm" / script_name).read_text(encoding="utf-8")

    assert "#SBATCH --job-name=tsunami" in text


def test_array_runner_preserves_exit_code_and_per_run_logs() -> None:
    text = (ROOT / "slurm/train_suite_array.slurm").read_text(
        encoding="utf-8"
    )

    assert "set -euo pipefail" in text
    assert 'tee -a "$RUN_LOG"' in text
    assert 'tee -a "$RUN_ERR"' in text
    assert "TRAIN_EXIT=${PIPESTATUS[0]}" in text
