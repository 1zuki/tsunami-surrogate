import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import eval_reference_resolution_sensitivity as evaluation


def test_fixed_predictions_are_denormalized_once_and_reused():
    class CountingPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return 2 * batch

    model = CountingPredictor().eval()
    inputs = torch.arange(1, 4, dtype=torch.float32).reshape(3, 1, 1, 1)
    prediction = evaluation._predict(model, inputs, 2, torch.device("cpu"), (-2, 3))
    np.testing.assert_equal(prediction[:, 0, 0, 0], [4, 10, 16])
    targets = {name: prediction + shift for name, shift in zip(
        evaluation.REFERENCES, (1, 2, 3, 4),
    )}
    scores = evaluation._scores(prediction, targets)
    assert model.calls == 2
    assert scores["archived"]["global_rmse"] == 1.0
    assert scores["fine"]["global_rmse"] == 4.0
    assert scores["archived"]["mean_per_case_relative_l2"] == pytest.approx(
        np.mean(1.0 / np.array([5, 11, 17]))
    )
    assert prediction.dtype == np.float64


def test_alignment_preserves_signed_interaction():
    prediction = np.array([4.0, 4.0]).reshape(2, 1, 1, 1)
    baseline = np.full_like(prediction, 3.0)
    fine = np.array([2.0, 4.0]).reshape(2, 1, 1, 1)
    result = evaluation._alignment(prediction, baseline, fine)
    terms = result["by_case"]
    assert terms["baseline_mse"] == [1.0, 1.0]
    assert terms["reference_shift_mse"] == [1.0, 1.0]
    assert terms["twice_error_shift_inner_product_per_cell"] == [2.0, -2.0]
    assert terms["fine_mse"] == [4.0, 0.0]
    assert terms["identity_residual"] == [0.0, 0.0]
    assert terms["cosine_alignment"] == [1.0, -1.0]


def test_rank_stability_distinguishes_reversals_from_ties():
    baseline = {"fno": 1.0, "ffno": 2.0, "ufno": 3.0, "convlstm": 4.0}
    fine = {"fno": 1.5, "ffno": 1.0, "ufno": 4.0, "convlstm": 4.0}
    assert evaluation._pairwise_changes(baseline, fine) == {
        "strict_pairwise_reversals": [["fno", "ffno"]],
        "tie_status_changes": [["ufno", "convlstm"]],
    }


def _study_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(evaluation, "CASE_INDICES", (1,))
    monkeypatch.setattr(evaluation, "ROLLOUT_COUNT", 3)
    cfg = yaml.safe_load((evaluation.ROOT / evaluation.DATASET_CONFIG).read_text())
    request = evaluation.parse_requested_output_config(cfg["requested_output"])
    times = request.requested_times.tolist()
    case = {"index": 1, "row": {"scenario_id": "scenario_000001"}}
    sources = [
        evaluation.STUDY_CONFIG, evaluation.DATASET_CONFIG,
        Path("scripts/run_benchmark_numerical_study.py"),
    ]
    keys = {variant: f"scenario_000001__swe_hydrostatic__{variant}" for variant in evaluation.RESOLUTION}
    tasks = [
        {"id": keys[variant], "solver": evaluation.SOLVER, "variant": variant, "core": core,
         "publication_core": 64, "contract_hash": request.contract_hash,
         "moving_equilibrium": {"transverse_cells": 8}}
        for variant, core in evaluation.RESOLUTION.items()
    ]
    plan = {
        "schema_id": evaluation.numerical.SCHEMA,
        "complete_full_scope": True,
        "full_scope_rollout_count": len(tasks),
        "roster": [case],
        "requested_contract_hash": request.contract_hash,
        "requested_timestamps": times,
        "tasks": tasks,
        "code_and_input_files": [
            {"path": f"/other/machine/{path}", "sha256": evaluation.sha256_file(evaluation.ROOT / path)}
            for path in sources
        ],
    }
    plan_hash = evaluation.stable_json_sha256(plan)
    summary = {
        "schema_id": evaluation.numerical.SCHEMA,
        "complete_full_planned_study": True,
        "full_scope_rollout_count": len(tasks),
        "rollout_count": len(tasks),
        "plan_sha256": plan_hash,
        "requested_timestamps": times,
        "rollouts": {task_id: {"fixture": True} for task_id in keys.values()},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "benchmark_numerical_study.json").write_text(json.dumps(summary))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for task in tasks:
        evaluation.numerical._write_checkpoint(
            {**task, "dataset": cfg, "case": case, "times": times},
            checkpoints, plan_hash,
            np.full(evaluation.SHAPE, task["core"], dtype=np.float64),
            summary["rollouts"][task["id"]],
        )
    return summary, keys


def test_missing_or_partial_study_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="must be finalized"):
        evaluation._load_study(tmp_path, {})
    summary, _ = _study_fixture(tmp_path, monkeypatch)
    summary["complete_full_planned_study"] = False
    (tmp_path / "benchmark_numerical_study.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete, matching"):
        evaluation._load_study(tmp_path, {})


def test_full_study_binds_fingerprints_and_exact_coverage(tmp_path, monkeypatch):
    _, keys = _study_fixture(tmp_path, monkeypatch)
    _, fields = evaluation._load_study(tmp_path, {})
    assert {variant: field["scenario_000001"].mean() for variant, field in fields.items()} == evaluation.RESOLUTION

    fine_file = tmp_path / "checkpoints" / f"{keys['fine']}.npz"
    with np.load(fine_file, allow_pickle=False) as checkpoint:
        record = json.loads(checkpoint["record"].item())
        eta = checkpoint["eta"]
    record["task_sha256"] = "different"
    np.savez(fine_file, eta=eta, record=json.dumps(record),
             record_sha256=evaluation.stable_json_sha256(record))
    with pytest.raises(ValueError, match="exact study"):
        evaluation._load_study(tmp_path, {})
    fine_file.unlink()
    with pytest.raises(ValueError, match="checkpoint roster"):
        evaluation._load_study(tmp_path, {})
