from __future__ import annotations

from pathlib import Path
import subprocess
import sys


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


def test_cli_has_one_explicit_generation_entrypoint() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "production" in result.stdout
    assert "generate" in result.stdout
    assert "legacy-v1" in result.stdout

    old_invocation = _run("--config", "configs/data/dataset.yaml")
    assert old_invocation.returncode != 0
    assert "invalid choice" in old_invocation.stderr


def test_common_time_and_legacy_modes_cannot_be_mixed() -> None:
    legacy_as_current = _run(
        "generate",
        "--config",
        "configs/data/legacy/dataset_saved_step_v1.yaml",
    )
    assert legacy_as_current.returncode != 0
    assert "requires common-time requested_output" in legacy_as_current.stderr

    current_as_legacy = _run(
        "legacy-v1", "--config", "configs/data/dataset.yaml"
    )
    assert current_as_legacy.returncode != 0
    assert "refuses requested-output configs" in current_as_legacy.stderr


def test_production_rejects_unresolved_cloud_labels(tmp_path: Path) -> None:
    result = _run(
        "production",
        "--stage",
        "rehearsal",
        "--input-root",
        str(tmp_path / "inputs"),
        "--run-root",
        str(tmp_path / "run"),
        "--workers",
        "1",
        "--max-in-flight",
        "1",
        "--cloud-provider",
        "google-cloud",
        "--cloud-zone",
        "REPLACE_WITH_ZONE",
        "--machine-type",
        "c4-highcpu-8",
        "--storage-class",
        "pd-ssd",
    )
    assert result.returncode != 0
    assert "cloud_zone must identify the real cloud allocation" in result.stderr


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
