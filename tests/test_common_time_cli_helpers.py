import argparse
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import compare_solvers_aligned as aligned_cli
from scripts import eval_emulator_superiority as emulator_cli
from scripts import run_dense_reference_validation as dense_cli
from src.evaluation.cli_progress import (
    ScenarioProgressLogger,
    default_progress_every_for_suite,
    resolve_progress_every,
)


def test_resolve_direction_runtime_paths_prefers_cli_overrides() -> None:
    direction_cfg = {
        "checkpoint": "config/model.ckpt",
        "model_raw_root": "data/raw/model",
        "benchmark_raw_root": "data/raw/benchmark",
        "model_processed_test_path": "data/processed/model/test.npz",
        "model_normalization_stats_path": "data/processed/model/stats.json",
    }
    args = argparse.Namespace(
        checkpoint="/override/model.ckpt",
        model_raw_root="/override/model-raw",
        benchmark_raw_root=None,
        processed_test_path="/override/test.npz",
        normalization_stats_path=None,
    )

    resolved = emulator_cli._resolve_direction_runtime_paths(direction_cfg, args)

    assert resolved["configured_paths"]["checkpoint"] == "config/model.ckpt"
    assert resolved["effective_paths"]["checkpoint"] == "/override/model.ckpt"
    assert resolved["effective_paths"]["model_raw_root"] == "/override/model-raw"
    assert resolved["effective_paths"]["benchmark_raw_root"] == "data/raw/benchmark"
    assert (
        resolved["effective_paths"]["model_processed_test_path"] == "/override/test.npz"
    )
    assert (
        resolved["effective_paths"]["model_normalization_stats_path"]
        == "data/processed/model/stats.json"
    )
    assert resolved["cli_overrides"] == {
        "checkpoint": "/override/model.ckpt",
        "model_raw_root": "/override/model-raw",
        "model_processed_test_path": "/override/test.npz",
    }


def test_resolve_direction_runtime_paths_uses_config_values_without_cli_overrides() -> (
    None
):
    direction_cfg = {
        "checkpoint": "config/model.ckpt",
        "model_raw_root": "data/raw/model",
        "benchmark_raw_root": "data/raw/benchmark",
        "model_processed_test_path": "data/processed/model/test.npz",
        "model_normalization_stats_path": "data/processed/model/stats.json",
    }
    args = argparse.Namespace(
        checkpoint=None,
        model_raw_root=None,
        benchmark_raw_root=None,
        processed_test_path=None,
        normalization_stats_path=None,
    )

    resolved = emulator_cli._resolve_direction_runtime_paths(direction_cfg, args)

    assert resolved["configured_paths"] == resolved["effective_paths"]
    assert resolved["cli_overrides"] == {}


def test_progress_helper_defaults_and_quiet_behavior() -> None:
    assert default_progress_every_for_suite("smoke") == 1
    assert default_progress_every_for_suite("dense_validation") == 10
    assert default_progress_every_for_suite("full") == 25
    assert resolve_progress_every("full", None) == 25
    with pytest.raises(ValueError, match="positive integer"):
        resolve_progress_every("full", 0)

    messages: list[str] = []
    logger = ScenarioProgressLogger(
        label="test",
        progress_every=2,
        quiet=False,
        emit=messages.append,
    )
    logger(1, 3, "scenario_000001")
    logger(2, 3, "scenario_000002")
    logger(3, 3, "scenario_000003")
    assert messages == [
        "[test] progress completed=2/3 last_scenario=scenario_000002",
        "[test] progress completed=3/3 last_scenario=scenario_000003",
    ]

    quiet_messages: list[str] = []
    quiet_logger = ScenarioProgressLogger(
        label="test",
        progress_every=1,
        quiet=True,
        emit=quiet_messages.append,
    )
    quiet_logger(1, 1, "scenario_000001")
    assert quiet_messages == []


def test_dense_cli_progress_default_and_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dense_cli, "load_config", lambda path: {"stub": True})

    def fake_run_dense_reference_validation(config, **kwargs):
        kwargs["progress_callback"](1, 2, "scenario_000001")
        kwargs["progress_callback"](2, 2, "scenario_000002")
        return {
            "suite": {"name": kwargs["suite_name"]},
            "status": "pass",
            "counts": {
                "scenario_count": 2,
                "eligible_for_interpolation_count": 2,
            },
            "artifacts_written": {
                "suite_output_dir": "results/common_time_validation/dense_reference_validation/smoke",
            },
        }

    monkeypatch.setattr(
        dense_cli,
        "run_dense_reference_validation",
        fake_run_dense_reference_validation,
    )

    default_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", default_stdout)
    dense_cli.main(["--config", "dummy.yaml", "--suite", "smoke"])
    default_output = default_stdout.getvalue()
    assert (
        "[dense-reference-validation] start suite=smoke progress_every=1"
        in default_output
    )
    assert (
        "[dense-reference-validation] progress completed=1/2 last_scenario=scenario_000001"
        in default_output
    )
    assert (
        "[dense-reference-validation] progress completed=2/2 last_scenario=scenario_000002"
        in default_output
    )
    assert (
        "[dense-reference-validation] artifacts=results/common_time_validation/dense_reference_validation/smoke"
        in default_output
    )

    quiet_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", quiet_stdout)
    dense_cli.main(["--config", "dummy.yaml", "--suite", "smoke", "--quiet"])
    quiet_output = quiet_stdout.getvalue()
    assert "[dense-reference-validation] start" not in quiet_output
    assert "[dense-reference-validation] progress" not in quiet_output
    assert (
        "[dense-reference-validation] suite=smoke status=pass scenario_count=2 eligible_for_interpolation=2"
        in quiet_output
    )
    assert (
        "[dense-reference-validation] artifacts=results/common_time_validation/dense_reference_validation/smoke"
        in quiet_output
    )


def test_aligned_cli_progress_default_and_quiet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        aligned_cli,
        "load_config",
        lambda path: {
            "alignment": {"aggregation": {"bootstrap": {}}},
            "audit": {},
            "selection": {},
        },
    )
    monkeypatch.setattr(
        aligned_cli,
        "resolve_suite_contract",
        lambda **kwargs: SimpleNamespace(
            ordered_scenario_ids=("scenario_000001", "scenario_000002")
        ),
    )
    monkeypatch.setattr(
        aligned_cli,
        "iter_paired_raw_reference_samples",
        lambda **kwargs: iter(
            [
                {
                    "scenario_id": "scenario_000001",
                    "bathymetry_type": "bathy_a",
                    "source_type": "source_a",
                    "source_strength": 1.0,
                    "left": {},
                    "right": {},
                },
                {
                    "scenario_id": "scenario_000002",
                    "bathymetry_type": "bathy_b",
                    "source_type": "source_b",
                    "source_strength": 2.0,
                    "left": {},
                    "right": {},
                },
            ]
        ),
    )

    def fake_compare_solver_scenarios(*, paired_scenarios, **kwargs):
        assert [row["scenario_id"] for row in paired_scenarios] == [
            "scenario_000001",
            "scenario_000002",
        ]
        return {"scenario_metrics": [], "status": "pass"}

    monkeypatch.setattr(
        aligned_cli, "compare_solver_scenarios", fake_compare_solver_scenarios
    )
    monkeypatch.setattr(aligned_cli, "write_jsonl", lambda rows, path: None)
    monkeypatch.setattr(aligned_cli, "save_json", lambda payload, path: None)
    monkeypatch.setattr(aligned_cli, "get_git_commit", lambda: "test-commit")

    output_path = tmp_path / "aligned.json"

    default_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", default_stdout)
    aligned_cli.main(
        [
            "--config",
            "dummy.yaml",
            "--mode",
            "common-time",
            "--suite",
            "smoke",
            "--solver-a",
            "solver_a",
            "--solver-b",
            "solver_b",
            "--solver-a-dir",
            "left",
            "--solver-b-dir",
            "right",
            "--output-path",
            str(output_path),
        ]
    )
    default_output = default_stdout.getvalue()
    assert (
        "[aligned-solver-comparison] start mode=common-time suite=smoke solver_a=solver_a solver_b=solver_b progress_every=1"
        in default_output
    )
    assert (
        "[aligned-solver-comparison] progress completed=1/2 last_scenario=scenario_000001"
        in default_output
    )
    assert (
        "[aligned-solver-comparison] progress completed=2/2 last_scenario=scenario_000002"
        in default_output
    )
    assert f"[aligned-solver-comparison] artifacts={output_path}" in default_output

    quiet_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", quiet_stdout)
    aligned_cli.main(
        [
            "--config",
            "dummy.yaml",
            "--mode",
            "common-time",
            "--suite",
            "smoke",
            "--solver-a",
            "solver_a",
            "--solver-b",
            "solver_b",
            "--solver-a-dir",
            "left",
            "--solver-b-dir",
            "right",
            "--output-path",
            str(output_path),
            "--quiet",
        ]
    )
    quiet_output = quiet_stdout.getvalue()
    assert "[aligned-solver-comparison] start" not in quiet_output
    assert "[aligned-solver-comparison] progress" not in quiet_output
    assert f"[aligned-solver-comparison] artifacts={output_path}" in quiet_output
