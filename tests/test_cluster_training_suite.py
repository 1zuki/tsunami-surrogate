from __future__ import annotations

from pathlib import Path

import pytest

from tsunami_surrogate.utils.config import load_yaml

from scripts.prepare_cluster_training_suite import (
    DEFAULT_MANIFEST,
    ROOT,
    build_finalizer_command,
    build_sbatch_command,
    prepare_suite,
)

def test_current_manifest_prepares_unique_single_seed_jobs(tmp_path: Path) -> None:
    suite = prepare_suite(DEFAULT_MANIFEST, tmp_path, root=ROOT, check_data=False)
    assert len({run.label for run in suite.runs}) == len(suite.runs)
    assert len({run.output_dir for run in suite.runs}) == len(suite.runs)
    for run in suite.runs:
        cfg = load_yaml(run.config_path)
        assert "seeds" not in cfg
        assert cfg["seed"] == run.seed
        assert cfg["device"] == "cuda"
        assert cfg["output_dir"] == run.output_dir


def test_retired_suite_cannot_be_submitted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported suite classification"):
        prepare_suite(Path("tests/fixtures/configs/legacy_dev_suite.yaml"),
                      tmp_path, root=ROOT, check_data=False)


def test_submission_uses_configured_resources_and_dependency(tmp_path: Path) -> None:
    suite = prepare_suite(DEFAULT_MANIFEST, tmp_path, root=ROOT, check_data=False)
    cmd = build_sbatch_command(suite, afterok=55523, root=ROOT)
    assert "--array=0-59%3" in cmd
    assert "--dependency=afterok:55523" in cmd
    assert "--gres=gpu:2g.35gb:1" in cmd
    assert "--qos=mig2-35" in cmd
    assert "--cpus-per-task=4" in cmd
    assert "--mem=32G" in cmd
    assert f"--chdir={ROOT}" in cmd
    reduced = build_sbatch_command(suite, afterok=55523, max_concurrent=2, root=ROOT)
    assert "--array=0-59%2" in reduced


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
    assert suite.max_concurrent == 3
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
    by_label = {run.label: run for run in suite.runs}
    assert by_label["fno_hydrostatic_seed_18"].output_dir == (
        "experiments/fno/seed_18"
    )
    assert by_label["fno_sample_100_seed_18"].output_dir == (
        "experiments/sample_scaling/n_000100/seed_18"
    )
    assert by_label["ensemble_fno_seed_11"].output_dir == (
        "experiments/ensemble/member_11"
    )
    ensemble_cfg = load_yaml(by_label["ensemble_fno_seed_11"].config_path)
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

    assert "--array=0-59%3" in submit
    assert "--dependency=afterok:66555" in submit
    assert "--dependency=afterany:70001" in finalizer
    assert finalizer[-1] == "slurm/finalize_training_suite.slurm"


@pytest.mark.parametrize(
    "script_name",
    [
        "gpu_environment_probe.slurm",
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
        "gpu_environment_probe.slurm",
        "train_fno.slurm",
        "train_suite_array.slurm",
        "finalize_training_suite.slurm",
    ],
)
def test_slurm_jobs_use_tsunami_display_name(script_name: str) -> None:
    text = (ROOT / "slurm" / script_name).read_text(encoding="utf-8")

    assert "#SBATCH --job-name=tsunami" in text


def test_finalizer_can_archive_prior_array_logs() -> None:
    text = (ROOT / "slurm/finalize_training_suite.slurm").read_text(
        encoding="utf-8"
    )

    assert 'PRIOR_ARRAY_JOB_IDS="${PRIOR_ARRAY_JOB_IDS:-}"' in text
    assert 'ARRAY_JOB_IDS="$ARRAY_JOB_ID"' in text
    assert 'sacct -j "$ARRAY_JOB_IDS"' in text


def test_array_runner_preserves_exit_code_and_per_run_logs() -> None:
    text = (ROOT / "slurm/train_suite_array.slurm").read_text(
        encoding="utf-8"
    )

    assert "set -euo pipefail" in text
    assert 'tee -a "$RUN_LOG"' in text
    assert 'tee -a "$RUN_ERR"' in text
    assert "TRAIN_EXIT=${PIPESTATUS[0]}" in text


@pytest.mark.parametrize("key,value", [
    ("max_concurrent", 4), ("cpus_per_job", 0),
    ("gpus_per_job", 0), ("time_limit", "72:99:00"),
    ("mps_per_job", 2), ("gpu_type", ""),
])
def test_invalid_scheduler_resources_are_rejected(key, value):
    from scripts.prepare_cluster_training_suite import _validate_resources
    cfg = load_yaml(ROOT / DEFAULT_MANIFEST)["resources"]
    cfg[key] = value
    with pytest.raises(ValueError):
        _validate_resources(cfg)


@pytest.mark.parametrize("script", ["train_fno.slurm", "train_suite_array.slurm"])
def test_job_startup_and_training_failure_are_preserved(tmp_path, script):
    import os
    import shutil
    import subprocess

    (tmp_path / "tsunami_surrogate").mkdir()
    (tmp_path / "requirements.txt").touch()
    (tmp_path / "slurm").mkdir()
    shutil.copyfile(ROOT / "slurm/environment.sh", tmp_path / "slurm/environment.sh")
    bindir = tmp_path / ".venv/bin"
    bindir.mkdir(parents=True)
    (bindir / "activate").write_text('export PATH="$ENV_PREFIX/bin:$PATH"\n')
    python = bindir / "python"
    python.write_text("#!/bin/bash\n"
                      "if [[ \"$*\" == *cluster_run_state.py* ]]; then\n"
                      "  printf 'fresh\\tnew\\t-\\n'\n"
                      "elif [[ \"$*\" == *scripts/train.py* ]]; then\n"
                      "  echo training-output; echo training-error >&2; exit 7\n"
                      "fi\n")
    python.chmod(0o755)
    gpu = bindir / "nvidia-smi"
    gpu.write_text("#!/bin/bash\nexit 0\n")
    gpu.chmod(0o755)
    config = tmp_path / "config.yaml"
    config.write_text("seed: 18\n")
    runs = tmp_path / "runs.tsv"
    runs.write_text(f"fno_seed_18\t{config}\t{tmp_path / 'output'}\t18\n")
    env = dict(os.environ, PROJECT_ROOT=str(tmp_path), ENV_PREFIX=str(bindir.parent),
               CONFIG_PATH=str(config), RUNS_FILE=str(runs), SUITE_ID="test",
               SLURM_ARRAY_TASK_ID="0")
    env.pop("BEST_GPU", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(["bash", str(ROOT / "slurm" / script)],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 7, result.stdout + result.stderr
    assert "training-output" in result.stdout
    assert "training-error" in result.stderr
    if script == "train_suite_array.slurm":
        assert "training-output" in (tmp_path / "output/cluster_console.log").read_text()
