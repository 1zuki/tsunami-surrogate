from __future__ import annotations

import copy
import json
import tarfile

import numpy as np
import pytest

from scripts import build_revision_evidence as bundle


def test_matched_metrics_allow_roundoff_but_reject_changed_results_and_scope():
    expected = {"rmse": [0.001, 0.002], "case_count": 3}
    assert bundle.matching_metrics({"rmse": [0.001 + 1e-18, 0.002], "case_count": 3}, expected)
    assert not bundle.matching_metrics({"rmse": [0.0011, 0.002], "case_count": 3}, expected)
    assert not bundle.matching_metrics({"rmse": [0.001, 0.002], "case_count": 2}, expected)
    assert not bundle.matching_metrics({"rmse": [float("nan"), 0.002], "case_count": 3}, expected)


def test_plan_reports_missing_numerical_and_sensitivity_evidence(tmp_path):
    with pytest.raises(ValueError) as failure:
        bundle.plan(tmp_path)
    assert "benchmark_numerical/benchmark_numerical_study.json" in str(failure.value)
    assert "reference_resolution_sensitivity.json" in str(failure.value)
    assert "reference_resolution_checkpoint_preflight.json" in str(failure.value)
    assert "production_replay_30" not in str(failure.value)


def test_plan_ignores_partial_canceled_production_replay(tmp_path, monkeypatch):
    numerical = tmp_path / "benchmark_numerical"
    numerical.mkdir()
    (numerical / "benchmark_numerical_study.json").write_text("{}")
    (tmp_path / bundle.REFERENCE_RESOLUTION).write_text("{}")
    (tmp_path / bundle.REFERENCE_RESOLUTION_PROOF).write_text("{}")
    (tmp_path / bundle.MATCHED_GEOCLAW).write_text("{}")
    canceled = tmp_path / "production_replay_30"
    canceled.mkdir()
    (canceled / "production_dense_rows.jsonl").write_text("{}\n")
    (tmp_path / "reproduction_scenarios_30.json").write_text("{}")

    for name in (
        "numerical_inputs",
        "reference_resolution_inputs",
        "geoclaw_inputs",
        "matched_geoclaw_inputs",
        "periodic_inputs",
        "trajectory_inputs",
        "leading_model_inputs",
        "decomposition_inputs",
        "figure_inputs",
    ):
        monkeypatch.setattr(bundle, name, lambda *_: None)
    monkeypatch.setattr(bundle, "SOURCES", ())
    assert bundle.plan(tmp_path).files == {}


def _numerical_payload(tmp_path):
    task = {
        "id": "toy",
        "variant": "baseline",
        "core": 2,
        "publication_core": 2,
        "moving_equilibrium": {"transverse_cells": 2},
        "times": [1],
    }
    plan = {
        "schema_id": bundle.numerical.SCHEMA,
        "rollout_count": 1,
        "full_scope_rollout_count": 1,
        "complete_full_scope": True,
        "roster": [],
        "requested_timestamps": task["times"],
    }
    eta = np.ones((1, 2, 2), dtype=np.float64)
    entry = {"note": "test rollout"}
    fingerprint = bundle.stable_json_sha256(plan)
    report = bundle.numerical._summary(plan, {"toy": (eta, entry)}, fingerprint)
    bundle.numerical._write_checkpoint(task, tmp_path, fingerprint, eta, entry)
    return plan, report, [task]


def test_numerical_payload_requires_full_checkpoint_inventory(tmp_path):
    plan, report, tasks = _numerical_payload(tmp_path)
    check = bundle.check_numerical_payload
    assert check(plan, report, tasks, tmp_path, count=1) == {"toy.npz"}
    report["complete_full_planned_study"] = False
    with pytest.raises(ValueError, match="full prescribed"):
        check(plan, report, tasks, tmp_path, count=1)
    report["complete_full_planned_study"] = True
    (tmp_path / "extra.npz").write_bytes(b"not part of the study")
    with pytest.raises(ValueError, match="extra=.*extra.npz"):
        check(plan, report, tasks, tmp_path, count=1)


def test_numerical_payload_rejects_missing_and_mismatched_checkpoints(tmp_path):
    plan, report, tasks = _numerical_payload(tmp_path)
    check = bundle.check_numerical_payload
    actual = tmp_path / "toy.npz"
    moved = tmp_path / "moved"
    actual.rename(moved)
    with pytest.raises(ValueError, match="missing=.*toy.npz"):
        check(plan, report, tasks, tmp_path, count=1)
    moved.rename(actual)
    tasks[0]["times"] = [2]
    with pytest.raises(ValueError, match="does not match this exact study"):
        check(plan, report, tasks, tmp_path, count=1)


def test_archive_inventory_rejects_traversal_and_symlinks(tmp_path):
    toy = tmp_path / "toy.txt"
    toy.write_text("example")
    (tmp_path / "link.txt").symlink_to(toy)
    inventory = bundle.Inventory()
    for pathname, name in (("link.txt", "bound.txt"), ("toy.txt", "../escape.txt")):
        with pytest.raises(ValueError, match="Unsafe release path"):
            inventory.record(tmp_path, pathname, name)


def test_archive_build_is_deterministic_and_preserves_existing_file(tmp_path):
    toy = tmp_path / "toy.txt"
    toy.write_text("example")
    inventory = bundle.Inventory()
    inventory.record(tmp_path, "toy.txt", "example/toy.txt")
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    assert bundle.build(first, inventory) == bundle.build(second, inventory)
    with pytest.raises(FileExistsError):
        bundle.build(first, inventory)
    with tarfile.open(first) as archive:
        assert all(
            member.isfile() and not member.name.startswith("/")
            for member in archive.getmembers()
        )
        manifest = json.load(archive.extractfile("revision-evidence.json"))
        assert manifest["files"] == {"example/toy.txt": bundle.core.digest(toy)}
        assert "production_scenarios" not in manifest["coverage"]
        assert "production_replay_rows" not in manifest["coverage"]
    toy.write_text("mutated")
    unpublished = tmp_path / "unpublished.tar"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bundle.build(unpublished, inventory)
    assert not unpublished.exists()


def _sensitivity_payload():
    module = bundle.sensitivity
    cases = [
        {
            "scenario_id": f"scenario_{index:06d}",
            "sample_index": index,
            "bathymetry": "test-bed",
            "source": "test-source",
        }
        for index in module.CASE_INDICES
    ]
    plan = {
        "roster": [
            {
                "row": row,
                "index": row["sample_index"],
                "bathymetry": row["bathymetry"],
                "source": row["source"],
            }
            for row in cases
        ],
        "requested_timestamps": [8.4, 420.0],
    }
    members = []
    checkpoints = {}
    metric = {
        "rmse_by_case": [1.0] * len(cases),
        "relative_l2_by_case": [2.0] * len(cases),
        "global_rmse": 1.0,
        "mean_per_case_relative_l2": 2.0,
    }
    for model in module.MODEL_NAMES:
        for seed in module.SEEDS:
            name = f"experiments/{model}/seed_{seed}/best.pt"
            checkpoints[name] = {"checkpoint": name}
            members.append(
                {
                    "model": model,
                    "seed": seed,
                    "scores": {
                        variant: copy.deepcopy(metric) for variant in module.REFERENCES
                    },
                }
            )
    result = {
        "schema_id": module.SCHEMA,
        "scope": {
            "study_plan_sha256": bundle.stable_json_sha256(plan),
            "study_rollout_count": module.ROLLOUT_COUNT,
            "case_count": len(cases),
            "scenarios": cases,
            "requested_timestamps": plan["requested_timestamps"],
            "reference_core_grid_cells": {"archived": 128, **module.RESOLUTION},
            "publication_grid_cells": module.SHAPE[-1],
            "models": list(module.MODEL_NAMES),
            "seeds": list(module.SEEDS),
        },
        "checkpoint_preflight": {
            "status": "passed",
            "checkpoints": list(checkpoints.values()),
        },
        "members": members,
        "rankings": module._rankings(members, [row["scenario_id"] for row in cases]),
    }
    return result, plan, checkpoints


def test_reference_resolution_requires_complete_finite_matching_metrics():
    validate = bundle.validate_sensitivity_metrics
    result, plan, proof = _sensitivity_payload()
    validate(result, plan, proof)
    member = result["members"].pop()
    with pytest.raises(ValueError, match="twelve distinct"):
        validate(result, plan, proof)
    result["members"].append(member)
    reference = member["scores"].pop("fine")
    with pytest.raises(ValueError, match="all four"):
        validate(result, plan, proof)
    member["scores"]["fine"] = reference
    reference["rmse_by_case"].pop()
    with pytest.raises(ValueError, match="six-case reference-resolution metrics"):
        validate(result, plan, proof)
    reference["rmse_by_case"].append(float("inf"))
    with pytest.raises(ValueError, match="nonfinite"):
        validate(result, plan, proof)


def test_reference_resolution_rejects_other_plan_and_preflight():
    validate = bundle.validate_sensitivity_metrics
    result, plan, proof = _sensitivity_payload()
    with pytest.raises(ValueError, match="numerical plan"):
        validate(result, {**plan, "artifact": "another study"}, proof)
    result["checkpoint_preflight"]["checkpoints"][0] = {"checkpoint": "another.pt"}
    with pytest.raises(ValueError, match="another completion proof"):
        validate(result, plan, proof)


def test_reference_resolution_requires_prescribed_source_fingerprints(tmp_path):
    bound = tmp_path / "checkpoint.npz"
    unbound = tmp_path / "extra.py"
    sources = {str(bound): "frozen"}
    assert bundle.validate_sensitivity_sources(
        sources, {bound: "frozen"}, {bound}, set()
    ) == {
        bound: "frozen",
    }
    for supplied in ({}, {str(bound): "changed"}, {**sources, str(unbound): "extra"}):
        with pytest.raises(ValueError, match="frozen source fingerprints"):
            bundle.validate_sensitivity_sources(
                supplied, {bound: "frozen"}, {bound}, set()
            )
