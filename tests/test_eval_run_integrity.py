from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts._consolidate_results import (
    ConsolidationError,
    _validate_live_bindings,
    consolidate,
)
from scripts.cleanup_legacy_results import replacement_patterns
from scripts.create_eval_run_manifest import build_manifest
from scripts.eval_arrival_maps import _accumulator, _update_acc
from scripts.eval_suite_preflight import load_suite_contract
from src.evaluation.uncertainty import (
    ErrorUncertaintyCorrelationAccumulator,
)


def _suite_contract() -> dict:
    return {
        "suite_id": "test-suite",
        "main_datasets": {"splits": {"test": {"count": 3}}},
        "direct_models": [
            {
                "id": "fno",
                "config": "configs/model/fno.yaml",
                "checkpoint": "experiments/fno/best.pt",
                "analyses": ["accuracy", "perframe", "physics"],
            }
        ],
        "window_models": [
            {
                "id": "fno_window5",
                "config": "configs/model/fno_window5_hydrostatic.yaml",
                "checkpoint": "experiments/fno_window5_hydrostatic/best.pt",
            }
        ],
        "sample_scaling": [
            {
                "id": "n_000100",
                "config": "experiments/sample_scaling/configs/fno_n_000100.yaml",
                "checkpoint": "experiments/sample_scaling/n_000100/best.pt",
            }
        ],
        "native_muscl": [
            {
                "id": "res32",
                "config": "configs/model/fno_res32_muscl_hr.yaml",
                "checkpoint": "experiments/fno_res32_muscl_hr/best.pt",
                "counts": {"test": 2},
            }
        ],
        "strict_holdouts": [],
        "real_bathymetry": {
            "suites": {"main": 2},
            "direct": [
                {
                    "id": "fno",
                    "config": "configs/eval/real_bathymetry_hydrostatic.yaml",
                    "checkpoint": "experiments/fno/best.pt",
                }
            ],
            "window": [
                {
                    "id": "fno_window5",
                    "config": (
                        "configs/eval/window5_real_bathymetry_hydrostatic.yaml"
                    ),
                    "checkpoint": (
                        "experiments/fno_window5_hydrostatic/best.pt"
                    ),
                }
            ],
        },
        "ensemble": {
            "required_members": [11, 22],
            "checkpoint_template": "experiments/ensemble/member_{seed}/best.pt",
            "configs": [
                {
                    "id": "indist",
                    "config": "configs/eval/uncertainty_indist_hydrostatic.yaml",
                }
            ],
        },
    }


def test_run_manifest_has_exact_unique_cell_membership() -> None:
    manifest = build_manifest(
        _suite_contract(),
        run_id="test-run",
        include_ensemble=True,
        include_real_bathymetry=True,
        include_speed=True,
    )
    cells = manifest["cells"]
    ids = [cell["id"] for cell in cells]
    paths = [cell["path"] for cell in cells]

    assert set(ids) == {
        "direct_accuracy:fno",
        "direct_perframe:fno",
        "direct_physics:fno",
        "direct_physics_csv:fno",
        "conditional_window_accuracy:fno_window5",
        "conditional_window_perframe:fno_window5",
        "sample_scaling:n_000100",
        "native_muscl:res32",
        "strict_holdout_summary",
        "strict_holdout_summary_csv",
        "real_bathymetry_direct:fno",
        "real_bathymetry_window:fno_window5",
        "main",
        "parameter_counts",
        "parameter_counts_csv",
        "uncertainty:indist",
        "model_speed:fno",
        "solver_speed:swe_hydrostatic",
        "solver_speed:swe_muscl_hr",
        "solver_speed:boussinesq",
        "speed_table",
        "speed_table_csv",
    }
    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))


def test_consolidation_rejects_a_missing_required_cell(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": "tsunami-surrogate.evaluation-run-manifest.v1",
                "suite_id": "test-suite",
                "run_id": "test-run",
                "cells": [
                    {"id": "present", "group": "test", "path": "present.json"},
                    {"id": "missing", "group": "test", "path": "missing.json"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "present.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConsolidationError, match="Missing required result"):
        consolidate(
            run_root=run_root,
            manifest_path=manifest_path,
            output_path=run_root / "all_results.json",
            completion_manifest_path=run_root / "completion_manifest.json",
        )

    assert not (run_root / "all_results.json").exists()
    assert not (run_root / "completion_manifest.json").exists()


def test_cleanup_patterns_only_cover_validated_replacement_groups() -> None:
    completion = {
        "_validated_run_manifest": {
            "cells": [
                {"group": "direct_accuracy"},
                {"group": "real_bathymetry_direct"},
                {"group": "dataset_summary"},
            ]
        }
    }

    patterns = set(replacement_patterns(completion))

    assert "accuracy_*.json" in patterns
    assert "real_bathymetry_*.json" in patterns
    assert "dataset_summary.json" in patterns
    assert "all_results.json" in patterns
    assert "ood_suites_*.json" not in patterns
    assert "solver_compare_*.json" not in patterns
    assert "speed_*.json" not in patterns
    assert "uncertainty_*.json" not in patterns


def test_live_manifest_declares_paper_and_numerical_rerun_cells() -> None:
    contract = load_suite_contract("configs/eval/final_v2_suite.yaml")
    manifest = build_manifest(
        contract,
        run_id="paper-test",
        include_ensemble=True,
        include_real_bathymetry=True,
        include_speed=True,
        include_paper_evidence=True,
        rerun_numerical_validation=True,
    )
    ids = {str(cell["id"]) for cell in manifest["cells"]}

    assert {
        "paper_evidence:numerical_evidence",
        "paper_reference_analysis:solver_gap",
        "paper_reference_analysis:cross_reference",
        "paper_wave_metrics:fno_hydrostatic",
        "paper_ensemble:calibration",
        "paper_ensemble:seed_stability",
        "numerical_validation:summary",
        "numerical_validation:archive",
    }.issubset(ids)


def test_consolidation_rejects_companion_checksum_mismatch(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    archive = run_root / "chain.tar.zst"
    archive.write_bytes(b"archive")
    summary = run_root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "evaluation_type": "test-summary",
                "archive_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": "tsunami-surrogate.evaluation-run-manifest.v1",
                "suite_id": "test-suite",
                "run_id": "test-run",
                "cells": [
                    {
                        "id": "summary",
                        "group": "test",
                        "path": "summary.json",
                        "evaluation_type": "test-summary",
                        "companion_sha256_fields": {
                            "archive_sha256": "chain.tar.zst"
                        },
                    },
                    {
                        "id": "archive",
                        "group": "companion_artifacts",
                        "path": "chain.tar.zst",
                        "file_only": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConsolidationError,
        match="Companion checksum mismatch",
    ):
        consolidate(
            run_root=run_root,
            manifest_path=manifest_path,
            output_path=run_root / "all_results.json",
            completion_manifest_path=run_root / "completion_manifest.json",
        )


def test_live_bindings_reject_commit_change_after_preflight(
    monkeypatch,
) -> None:
    def fake_check_output(command, **_kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "new-commit\n"
        if command[:2] == ["git", "ls-files"]:
            return b""
        raise AssertionError(command)

    monkeypatch.setattr(
        "scripts._consolidate_results.subprocess.check_output",
        fake_check_output,
    )

    with pytest.raises(ConsolidationError, match="Git commit changed"):
        _validate_live_bindings(
            {
                "code_state": {
                    "git_commit": "preflight-commit",
                    "evaluation_tree_sha256": "unused",
                },
                "cells": [],
            }
        )


def test_arrival_map_threshold_is_target_only_and_shapes_fail_closed() -> None:
    accumulator = _accumulator(1, 1)
    prediction = np.asarray([[[100.0]], [[0.0]]], dtype=np.float64)
    target = np.asarray([[[0.0]], [[1.0]]], dtype=np.float64)

    sample_mean, difference = _update_acc(
        accumulator,
        prediction,
        target,
        threshold_fraction=0.5,
    )

    assert sample_mean == pytest.approx(1.0)
    assert difference[0, 0] == pytest.approx(1.0)
    assert accumulator["count_valid_target"][0, 0] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="shapes differ"):
        _update_acc(
            _accumulator(1, 1),
            np.zeros((2, 1, 1), dtype=np.float64),
            np.zeros((3, 1, 1), dtype=np.float64),
            threshold_fraction=0.5,
        )


def test_streaming_uncertainty_correlation_is_batch_partition_invariant() -> None:
    mean = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    variance = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0, 5.0, 0.0], dtype=torch.float64)

    whole = ErrorUncertaintyCorrelationAccumulator()
    whole.update(mean, variance, target)

    partitioned = ErrorUncertaintyCorrelationAccumulator()
    partitioned.update(mean[:1], variance[:1], target[:1])
    partitioned.update(mean[1:3], variance[1:3], target[1:3])
    partitioned.update(mean[3:], variance[3:], target[3:])

    assert partitioned.compute() == pytest.approx(
        whole.compute(),
        abs=1.0e-12,
    )
