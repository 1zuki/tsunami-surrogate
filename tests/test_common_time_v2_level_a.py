from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data_gen.common_time_v2 import authoritative_input_fingerprint, hash_array
from src.evaluation.common_time_v2_level_a import (
    TASK_ARTIFACT_SCHEMA_ID,
    _build_level_a_task_plan,
    _execute_level_a_task_plan,
    _load_task_artifact,
    _make_level_a_task,
    _operational_provenance,
    _scientific_digest,
    _task_directory_name,
    _write_checksums,
    _bootstrap_canary_aggregates,
    _decision_from_gates,
    _group_speed_gate,
    _preflight_canaries,
    _select_canaries,
    _temporal_refinement_gate,
    _universal_health_gate,
    preregister_level_a,
    validate_checksums,
)


def test_canary_selection_covers_source_families_deterministically() -> None:
    rows = []
    for index, source in enumerate(("a", "b", "c", "d", "e", "f"), start=1):
        rows.append(
            {
                "split": "train",
                "qualified_id": f"train:scenario_{index:06d}",
                "scenario_id": f"scenario_{index:06d}",
                "sample_index": index,
                "bathymetry_type": f"bath-{index}",
                "source_type": source,
                "source_strength": 1.0,
                "input_fingerprint": str(index),
                "bathymetry_cache_path": "missing",
                "source_cache_path": "missing",
                "raw_sample_paths": {},
                "array_hashes": {},
            }
        )
    selected = _select_canaries(rows, 6)
    assert [row["source_type"] for row in selected] == ["a", "b", "c", "d", "e", "f"]


def test_decision_precedence_is_fail_closed() -> None:
    assert _decision_from_gates([]) == "pass_to_H1"
    assert (
        _decision_from_gates([{"category": "blocked_convergence", "passed": False}])
        == "blocked_convergence"
    )
    assert (
        _decision_from_gates(
            [
                {"category": "blocked_convergence", "passed": False},
                {"category": "implementation_failure", "passed": False},
            ]
        )
        == "implementation_failure"
    )


def test_preregistration_is_content_addressed_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = preregister_level_a(
        repo_root=repo,
        config_path=repo / "configs/eval/common_time_v2_level_a.yaml",
        output_root=tmp_path,
    )
    validate_checksums(root)
    payload = json.loads(
        (root / "preregistered_contract.json").read_text(encoding="utf-8")
    )
    assert root.name == payload["contract_hash"]
    assert payload["thresholds_frozen_before_execution"] is True
    with pytest.raises(FileExistsError):
        preregister_level_a(
            repo_root=repo,
            config_path=repo / "configs/eval/common_time_v2_level_a.yaml",
            output_root=tmp_path,
        )


def test_canary_preflight_validates_complete_fingerprint_before_execution(
    tmp_path: Path,
) -> None:
    bathymetry = np.full((2, 2), -1.0, dtype=np.float32)
    source = np.full((2, 2), 0.25, dtype=np.float32)
    strength = np.asarray([0.5], dtype=np.float32)
    rest = np.maximum(-bathymetry, 0.0).astype(np.float32)
    eta0 = np.asarray(float(strength[0]) * source, dtype=np.float32)
    initial = np.maximum(rest + eta0, 0.0).astype(np.float32)
    free = (initial + bathymetry).astype(np.float32)
    arrays = {
        "bathymetry": bathymetry,
        "source_field": source,
        "rest_depth": rest,
        "eta0": eta0,
        "initial_depth": initial,
        "free_surface0": free,
    }
    bathy_path = tmp_path / "bathymetry.npz"
    source_path = tmp_path / "source.npz"
    np.savez(
        bathy_path,
        bathymetry=bathymetry,
        bathymetry_type=np.asarray(["slope"]),
    )
    np.savez(
        source_path,
        source_field=source,
        source_strength=strength,
        source_type=np.asarray(["gaussian"]),
    )
    row = {
        "split": "train",
        "qualified_id": "train:scenario_000001",
        "scenario_id": "scenario_000001",
        "sample_index": 1,
        "bathymetry_type": "slope",
        "source_type": "gaussian",
        "source_strength": float(strength[0]),
        "bathymetry_cache_path": str(bathy_path),
        "source_cache_path": str(source_path),
        "array_hashes": {name: hash_array(values) for name, values in arrays.items()},
    }
    row["input_fingerprint"] = authoritative_input_fingerprint(
        split="train",
        sample_index=1,
        scenario_id="scenario_000001",
        bathymetry_type="slope",
        source_type="gaussian",
        source_strength=strength,
        arrays=arrays,
    )
    assert _preflight_canaries([row]) == []

    row["array_hashes"] = dict(row["array_hashes"])
    row["array_hashes"]["eta0"] = dict(row["array_hashes"]["eta0"])
    row["array_hashes"]["eta0"]["sha256"] = "0" * 64
    issues = _preflight_canaries([row])
    assert len(issues) == 1
    assert "derived eta0 hash mismatch" in issues[0]


def test_temporal_refinement_uses_three_trajectory_discrepancies_and_floor() -> None:
    exact = np.zeros((2, 2, 2), dtype=np.float64)
    production = exact + 4.0e-3
    half = exact + 1.0e-3
    quarter = exact + 2.5e-4
    result = _temporal_refinement_gate(
        [production, half, quarter], minimum_order=1.5, floor=1.0e-6
    )
    assert result["production_to_half_trajectory_rms"] == pytest.approx(3.0e-3)
    assert result["half_to_quarter_trajectory_rms"] == pytest.approx(7.5e-4)
    assert result["observed_order"] == pytest.approx(2.0)
    assert result["passed"] is True

    floor_result = _temporal_refinement_gate(
        [exact + 2e-7, exact + 1e-7, exact],
        minimum_order=9.0,
        floor=1.0e-6,
    )
    assert floor_result["both_below_floor"] is True
    assert floor_result["passed"] is True


def test_group_speed_gate_uses_measured_frequency_slope() -> None:
    rows = [
        {"wavenumber": 1.0, "measured_omega": 2.0, "expected_omega": 2.0},
        {"wavenumber": 2.0, "measured_omega": 4.0, "expected_omega": 4.0},
        {"wavenumber": 3.0, "measured_omega": 6.0, "expected_omega": 6.0},
    ]
    result = _group_speed_gate(rows, relative_error_limit=1.0e-12)
    assert result["measured_group_speed"] == pytest.approx(2.0)
    assert result["group_speed_relative_error"] == pytest.approx(0.0, abs=1e-14)
    assert result["passed"] is True


def test_canary_bootstrap_is_seeded_and_descriptive() -> None:
    rows = [
        {
            "component": "production_amplitude_canary",
            "qualified_id": f"train:scenario_{index:06d}",
            "solver": "boussinesq",
            "amplitude_growth": float(index),
            "max_eta_over_depth": float(index) / 10.0,
            "runtime_s": float(index) / 100.0,
        }
        for index in range(1, 7)
    ]
    first = _bootstrap_canary_aggregates(rows, seed=20260712, resamples=2000)
    second = _bootstrap_canary_aggregates(rows, seed=20260712, resamples=2000)
    assert first == second
    assert len(first) == 3
    assert all(row["decision_role"] == "descriptive_only" for row in first)
    assert all(row["resamples"] == 2000 for row in first)


def test_universal_health_gate_fails_missing_or_replaced_diagnostics() -> None:
    universal = {
        "exact_output_count": 50,
        "cg_failure_count": 0,
        "nan_to_num_replacement_count": 0,
        "require_finite": True,
    }
    good = {
        "component": "analytical_mode",
        "output_count": 50,
        "requested_times_exact": True,
        "finite": True,
        "cg_failure_count": 0,
        "operator": {"nan_to_num_replacement_count": 0},
    }
    assert _universal_health_gate([good], universal=universal)["passed"] is True
    bad = dict(good)
    bad["operator"] = {"nan_to_num_replacement_count": 1}
    result = _universal_health_gate([bad], universal=universal)
    assert result["passed"] is False
    assert result["failure_count"] == 1


def _fixture_tasks() -> list[dict[str, object]]:
    specs = [
        {"name": "slow", "value": 1.0, "delay_s": 0.05},
        {"name": "fast", "value": 2.0, "delay_s": 0.0},
        {"name": "middle", "value": 3.0, "delay_s": 0.01},
    ]
    return [
        _make_level_a_task(
            ordinal=index,
            task_id=f"fixture/{spec['name']}",
            kind="fixture",
            spec=spec,
            contract_hash="fixture-contract",
            code_state_hash="fixture-code",
        )
        for index, spec in enumerate(specs)
    ]


def _task_file_state(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (
            __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_production_task_plan_has_stable_rollout_granularity() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = __import__("yaml").safe_load(
        (repo / "configs/eval/common_time_v2_level_a.yaml").read_text(encoding="utf-8")
    )
    canaries = [
        {
            "qualified_id": f"train:scenario_{index:06d}",
            "input_fingerprint": str(index),
        }
        for index in range(6)
    ]
    first = _build_level_a_task_plan(
        config,
        canaries,
        contract_hash="contract",
        code_state_hash="code",
    )
    second = _build_level_a_task_plan(
        config,
        canaries,
        contract_hash="contract",
        code_state_hash="code",
    )
    assert first == second
    counts = {
        kind: sum(task["kind"] == kind for task in first)
        for kind in ("analytical", "operator", "boundary", "conservation", "canary")
    }
    assert counts == {
        "analytical": 27,
        "operator": 14,
        "boundary": 6,
        "conservation": 6,
        "canary": 18,
    }
    for solver in ("swe_hydrostatic", "swe_muscl_hr", "boussinesq"):
        roles = [
            task["spec"]["role"]
            for task in first
            if task["kind"] == "analytical" and task["spec"]["solver"] == solver
        ]
        assert roles.count("spatial") == 3
        assert roles.count("temporal") == 3
        assert roles.count("modal") == 3


def _fixture_scientific_outputs(
    payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    rows = [
        {
            "component": payload["row"]["component"],
            "name": payload["row"]["name"],
            "value": payload["row"]["value"],
            "trajectory_hash": payload["result"]["array_hashes"]["trajectory"],
        }
        for payload in payloads
    ]
    aggregate = {
        "row_count": len(rows),
        "ordered_names": [row["name"] for row in rows],
        "value_sum": sum(float(row["value"]) for row in rows),
        "trajectory_hashes": [row["trajectory_hash"] for row in rows],
    }
    decision = {
        "decision": "pass" if aggregate["value_sum"] == 6.0 else "fail",
        "passed": aggregate["value_sum"] == 6.0,
    }
    return rows, aggregate, decision


def test_task_execution_is_serial_parallel_scientifically_equivalent(
    tmp_path: Path,
) -> None:
    tasks = _fixture_tasks()
    serial, serial_provenance = _execute_level_a_task_plan(
        tasks, tasks_root=tmp_path / "serial", workers=1
    )
    parallel, parallel_provenance = _execute_level_a_task_plan(
        tasks, tasks_root=tmp_path / "parallel", workers=2
    )
    assert [row["task"]["task_id"] for row in serial] == [
        "fixture/slow",
        "fixture/fast",
        "fixture/middle",
    ]
    serial_rows, serial_aggregate, serial_decision = _fixture_scientific_outputs(serial)
    parallel_rows, parallel_aggregate, parallel_decision = _fixture_scientific_outputs(
        parallel
    )
    assert serial_rows == parallel_rows
    assert serial_aggregate == parallel_aggregate
    assert serial_decision == parallel_decision
    assert serial_decision == {"decision": "pass", "passed": True}
    assert _scientific_digest(serial) == _scientific_digest(parallel)
    assert serial_provenance["requested_workers"] == 1
    assert parallel_provenance["requested_workers"] == 2
    assert parallel_provenance["process_start_method"] == "spawn"
    assert set(parallel_provenance["thread_environment"]) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }


def test_task_failure_preserves_completed_artifacts_and_resumes_missing(
    tmp_path: Path,
) -> None:
    tasks = _fixture_tasks()
    tasks[1] = _make_level_a_task(
        ordinal=1,
        task_id="fixture/fail",
        kind="fixture",
        spec={"name": "fail", "value": 2.0, "fail": True, "delay_s": 0.2},
        contract_hash="fixture-contract",
        code_state_hash="fixture-code",
    )
    root = tmp_path / "tasks"
    with pytest.raises(RuntimeError, match="fixture/fail"):
        _execute_level_a_task_plan(tasks, tasks_root=root, workers=1)
    assert (root / f"000-{tasks[0]['task_spec_hash'][:16]}").is_dir()
    assert not (root / f"001-{tasks[1]['task_spec_hash'][:16]}").exists()
    completed_state = _task_file_state(root)
    tasks[1] = _make_level_a_task(
        ordinal=1,
        task_id="fixture/fail",
        kind="fixture",
        spec={"name": "fail", "value": 2.0},
        contract_hash="fixture-contract",
        code_state_hash="fixture-code",
    )
    resumed, provenance = _execute_level_a_task_plan(
        tasks, tasks_root=root, workers=1, resume=True
    )
    assert [payload["task"]["task_id"] for payload in resumed] == [
        "fixture/slow",
        "fixture/fail",
        "fixture/middle",
    ]
    assert provenance["effective_workers"] == 1
    after = _task_file_state(root)
    for relative, state in completed_state.items():
        assert after[relative] == state


def test_parallel_worker_failure_preserves_completed_siblings(tmp_path: Path) -> None:
    tasks = _fixture_tasks()
    tasks[0] = _make_level_a_task(
        ordinal=0,
        task_id="fixture/first",
        kind="fixture",
        spec={"name": "first", "value": 1.0, "delay_s": 0.0},
        contract_hash="fixture-contract",
        code_state_hash="fixture-code",
    )
    tasks[1] = _make_level_a_task(
        ordinal=1,
        task_id="fixture/fail",
        kind="fixture",
        spec={"name": "fail", "value": 2.0, "fail": True, "delay_s": 0.2},
        contract_hash="fixture-contract",
        code_state_hash="fixture-code",
    )
    tasks[2] = _make_level_a_task(
        ordinal=2,
        task_id="fixture/last",
        kind="fixture",
        spec={"name": "last", "value": 3.0, "delay_s": 0.3},
        contract_hash="fixture-contract",
        code_state_hash="fixture-code",
    )
    root = tmp_path / "parallel-failure"
    with pytest.raises(RuntimeError, match="fixture/fail"):
        _execute_level_a_task_plan(tasks, tasks_root=root, workers=2)
    assert (root / f"000-{tasks[0]['task_spec_hash'][:16]}").is_dir()
    assert not (root / f"001-{tasks[1]['task_spec_hash'][:16]}").exists()


def test_task_resume_rejects_raw_and_semantic_corruption(tmp_path: Path) -> None:
    tasks = _fixture_tasks()
    raw_root = tmp_path / "raw"
    _execute_level_a_task_plan(tasks, tasks_root=raw_root, workers=1)
    healthy = _task_file_state(raw_root)
    first = raw_root / f"000-{tasks[0]['task_spec_hash'][:16]}" / "result.json"
    first.write_text(first.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _execute_level_a_task_plan(tasks, tasks_root=raw_root, workers=1, resume=True)
    assert (
        _task_file_state(raw_root)[f"001-{tasks[1]['task_spec_hash'][:16]}/result.json"]
        == healthy[f"001-{tasks[1]['task_spec_hash'][:16]}/result.json"]
    )

    semantic_root = tmp_path / "semantic"
    _execute_level_a_task_plan(tasks, tasks_root=semantic_root, workers=1)
    task_root = semantic_root / f"000-{tasks[0]['task_spec_hash'][:16]}"
    manifest = json.loads((task_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_id"] = TASK_ARTIFACT_SCHEMA_ID + ".wrong"
    (task_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(task_root)
    with pytest.raises(RuntimeError, match="Invalid Level A task manifest"):
        _execute_level_a_task_plan(
            tasks, tasks_root=semantic_root, workers=1, resume=True
        )


def test_task_resume_rejects_result_spec_mismatch_with_valid_checksums(
    tmp_path: Path,
) -> None:
    tasks = _fixture_tasks()
    root = tmp_path / "spec-mismatch"
    _execute_level_a_task_plan(tasks, tasks_root=root, workers=1)
    task_root = root / _task_directory_name(tasks[0])
    result_path = task_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["row"]["name"] = "wrong"
    result["scientific_hash"] = _scientific_digest(
        {"row": result["row"], "array_hashes": result["array_hashes"]}
    )
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = task_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_hash"] = result["scientific_hash"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(task_root)
    with pytest.raises(RuntimeError, match="result/spec mismatch"):
        _execute_level_a_task_plan(tasks, tasks_root=root, workers=1, resume=True)


def test_task_checksum_manifest_requires_exact_coverage(tmp_path: Path) -> None:
    tasks = _fixture_tasks()
    root = tmp_path / "coverage"
    _execute_level_a_task_plan(tasks, tasks_root=root, workers=1)
    task_root = root / _task_directory_name(tasks[0])
    checksum_path = task_root / "SHA256SUMS.txt"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checksum_path.write_text(
        "\n".join(line for line in lines if not line.endswith("  result.json")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checksum coverage mismatch"):
        _execute_level_a_task_plan(tasks, tasks_root=root, workers=1, resume=True)


def test_task_completed_resume_is_noop_and_extra_artifacts_fail(tmp_path: Path) -> None:
    tasks = _fixture_tasks()
    root = tmp_path / "tasks"
    _execute_level_a_task_plan(tasks, tasks_root=root, workers=1)
    before = _task_file_state(root)
    resumed, provenance = _execute_level_a_task_plan(
        tasks, tasks_root=root, workers=2, resume=True
    )
    assert _task_file_state(root) == before
    assert [payload["task"]["task_id"] for payload in resumed] == [
        task["task_id"] for task in tasks
    ]
    assert provenance["effective_workers"] == 0
    with pytest.raises(FileExistsError):
        _execute_level_a_task_plan(tasks, tasks_root=root, workers=1)
    (root / "unexpected").mkdir()
    with pytest.raises(RuntimeError, match="Unexpected or partial"):
        _execute_level_a_task_plan(tasks, tasks_root=root, workers=1, resume=True)


def test_scientific_digest_excludes_only_operational_fields() -> None:
    first = {
        "rows": [{"value": 1.0, "runtime_s": 1.0}],
        "elapsed_s": 2.0,
        "operational_provenance": _operational_provenance(
            requested_workers=1, effective_workers=1
        ),
    }
    second = {
        "rows": [{"value": 1.0, "runtime_s": 9.0}],
        "elapsed_s": 10.0,
        "operational_provenance": _operational_provenance(
            requested_workers=2, effective_workers=2
        ),
    }
    assert _scientific_digest(first) == _scientific_digest(second)
    second["rows"][0]["value"] = 1.1
    assert _scientific_digest(first) != _scientific_digest(second)
