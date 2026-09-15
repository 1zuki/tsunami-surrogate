from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from scripts.eval_pooled_reference_ablation import (
    _PairedReferenceDataset,
    _bootstrap_rmse,
    _domain_arrival_time,
    _model_prediction,
)
from src.data.dataset import BalancedSolverBatchSampler
from src.evaluation.normalization_bridge import (
    EvaluationNormalizationBridge,
    StandardizationSpec,
)
from src.training.checkpointing import training_contract_signature
from src.utils.config import load_config


SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")


class _Dataset(Dataset):
    def __init__(self, solvers: list[str]) -> None:
        self.items = [
            {
                "x": torch.full((3, 2, 2), float(index)),
                "y": torch.zeros((1, 2, 2)),
                "solver_name": solver,
                "scenario_id": f"scenario_{index:06d}",
            }
            for index, solver in enumerate(solvers)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


class _ConstantModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _bridge() -> EvaluationNormalizationBridge:
    inputs = {
        "bathymetry": (0.0, 1.0),
        "source": (0.0, 1.0),
        "initial_depth": (0.0, 1.0),
    }
    dataset_stats = StandardizationSpec(
        path=Path("dataset_stats.json"),
        inputs=inputs,
        target=(0.0, 1.0),
        target_variable="eta",
    )
    model_stats = StandardizationSpec(
        path=Path("model_stats.json"),
        inputs=inputs,
        target=(0.0, 1.0),
        target_variable="eta",
    )
    return EvaluationNormalizationBridge(
        dataset_stats,
        model_stats,
        ["bathymetry", "source", "initial_depth"],
    )


def test_balanced_solver_sampler_is_deterministic_and_equal() -> None:
    dataset = _Dataset(
        ["swe_hydrostatic"] * 4 + ["muscl_hr"] * 4 + ["boussinesq"] * 4
    )
    sampler = BalancedSolverBatchSampler(
        dataset,
        batch_size=6,
        seed=17,
        solvers=list(SOLVERS),
        batches_per_epoch=2,
    )
    first = list(sampler)
    sampler.set_epoch(0)
    second = list(sampler)
    assert first == second
    assert len(first) == 2
    for batch in first:
        counts = {solver: 0 for solver in SOLVERS}
        for index in batch:
            raw = dataset[index]["solver_name"]
            canonical = "hydrostatic" if raw == "swe_hydrostatic" else raw
            counts[canonical] += 1
        assert counts == {solver: 2 for solver in SOLVERS}


def test_balanced_solver_sampler_rejects_unbalanced_roster() -> None:
    dataset = _Dataset(["hydrostatic"] * 3 + ["muscl_hr"] * 2 + ["boussinesq"] * 3)
    with pytest.raises(ValueError, match="equal per-solver"):
        BalancedSolverBatchSampler(
            dataset,
            batch_size=3,
            seed=1,
            solvers=list(SOLVERS),
        )


def test_conditioned_prediction_appends_requested_solver_id() -> None:
    x = torch.zeros((2, 3, 2, 2))
    target = torch.zeros((2, 1, 2, 2))

    class _InspectModel(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            assert values.shape[1] == 4
            assert torch.all(values[:, 3] == 2.0)
            return values[:, :1]

    output = _model_prediction(
        model=_InspectModel(),
        x_reference=x,
        bridge=_bridge(),
        target=target,
        conditioned_solver="boussinesq",
    )
    assert output.shape == target.shape


def test_paired_reference_dataset_rejects_input_mismatch() -> None:
    datasets = {solver: _Dataset([solver]) for solver in SOLVERS}
    datasets["muscl_hr"].items[0]["scenario_id"] = "scenario_000000"
    datasets["hydrostatic"].items[0]["scenario_id"] = "scenario_000000"
    datasets["boussinesq"].items[0]["scenario_id"] = "scenario_000000"
    datasets["muscl_hr"].items[0]["x"] = torch.ones((3, 2, 2))
    with pytest.raises(ValueError, match="Input identity mismatch"):
        _PairedReferenceDataset(datasets)[0]


def test_bootstrap_rmse_is_deterministic() -> None:
    values = [1.0, 4.0, 9.0]
    assert _bootstrap_rmse(values, 7, 32) == _bootstrap_rmse(values, 7, 32)


def test_domain_arrival_time_uses_first_threshold_crossing() -> None:
    times = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    values = np.asarray(
        [np.zeros((2, 2)), np.full((2, 2), 0.2), np.full((2, 2), 1.0)]
    )
    assert _domain_arrival_time(values, times, threshold_fraction=0.1) == 2.0
    assert _domain_arrival_time(np.zeros_like(values), times) is None


def test_pooled_configs_preserve_isolation_and_budget() -> None:
    anonymous = load_config("configs/model/pooled_reference_ablation/fno_anonymous.yaml")
    conditioned = load_config("configs/model/pooled_reference_ablation/fno_conditioned.yaml")
    same_epoch = load_config("configs/model/pooled_reference_ablation/fno_anonymous_same_epoch.yaml")
    assert anonymous["data"]["train_path"].startswith("data/processed/pooled_reference_ablation/")
    assert anonymous["model"]["in_channels"] == 3
    assert conditioned["model"]["in_channels"] == 4
    assert anonymous["train"]["max_train_batches"] == 157
    assert same_epoch["train"]["max_train_batches"] == 476
    assert anonymous["data"]["balanced_solver_sampling"]["solvers"] == list(SOLVERS)


def test_training_contract_binds_pooled_sampling_and_update_budget() -> None:
    cfg = load_config("configs/model/pooled_reference_ablation/fno_anonymous.yaml")
    contract = training_contract_signature(cfg)
    assert contract["max_train_batches"] == 157
    assert contract["data_view"]["balanced_solver_sampling"]["batches_per_epoch"] == 157


def test_pooled_preprocess_configs_bind_train_only_stats() -> None:
    train = load_config("configs/data/pooled_reference_ablation/preprocess_train_anonymous.yaml")
    val = load_config("configs/data/pooled_reference_ablation/preprocess_eval_anonymous.yaml")
    test = load_config("configs/data/pooled_reference_ablation/preprocess_test_anonymous.yaml")
    assert train["fde"]["mode"] == "multifidelity"
    assert train["split"] == {"train": 1, "val": 0, "test": 0, "seed": 42}
    for cfg in (val, test):
        assert cfg["normalization"]["reference_stats_path"].endswith(
            "anonymous/normalization_stats.json"
        )
        assert cfg["saving"]["publication_mode"] == "merge_split"


def test_pooled_suite_reuses_existing_hydrostatic_specialists() -> None:
    cfg = load_config("configs/cluster/pooled_reference_ablation_suite.yaml")
    entries = {entry["name"]: entry for entry in cfg["entries"]}
    assert not entries["fno_hydrostatic"]["enabled"]
    assert entries["fno_muscl_hr"]["seeds"] == [18, 36, 67, 72, 154]
    assert entries["fno_boussinesq"]["seeds"] == [18, 36, 67, 72, 154]
