from __future__ import annotations

import csv
import io
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
    _boussinesq_directional_components,
    _boussinesq_directional_rate,
    _boussinesq_spectral_packet_bundle,
    _boundary_initial_conditions,
    _boundary_metrics,
    _boundary_timing,
    _csv_text,
    _decision_from_gates,
    _derived_replay_equal,
    _group_speed_gate,
    _hydro_clean_temporal_metrics,
    _hydro_spatial_control_metrics,
    _invariant_metrics,
    _operator_discrepancy_metrics,
    _packet,
    _preflight_canaries,
    _recomputed_rows_equal,
    _resolved_boundary_packet_spec,
    _run_float64_conservation,
    _select_canaries,
    _solver,
    _stable_sum,
    _temporal_refinement_gate,
    _universal_health_gate,
    _validate_boundary_packet_spec,
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
    task_plan = json.loads((root / "task_plan.json").read_text(encoding="utf-8"))
    assert len(task_plan) == 86
    assert len(payload["task_blueprint"]) == 86
    assert payload["worker_policy"]["requested_workers"] == 8
    assert payload["worker_policy"]["requested_max_in_flight"] == 8
    assert payload["worker_policy"]["process_start_method"] == "spawn"
    assert set(payload["execution_environment"]["thread_environment"].values()) == {
        "1"
    }
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


def test_csv_replay_compares_canonical_raw_bytes(tmp_path: Path) -> None:
    expected = _csv_text([{"nested": {"b": 2, "a": 1}, "value": 3.0}])
    path = tmp_path / "rows.csv"
    path.write_text(expected, encoding="utf-8", newline="")
    assert b"\r\n" in path.read_bytes()
    assert path.read_bytes() == expected.encode("utf-8")
    assert path.read_text(encoding="utf-8") != expected


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


@pytest.mark.parametrize(
    "solver_name", ["swe_hydrostatic", "swe_muscl_hr", "boussinesq"]
)
def test_x_only_sponge_preserves_quasi_1d_interior(solver_name: str) -> None:
    x_only = _solver(
        solver_name,
        nx=128,
        ny=4,
        cfl=0.2,
        boundary="open",
        use_sponge=True,
        sponge_mode="elapsed_time_consistent",
        sponge_axes="x",
    )
    assert x_only.sponge_axes == "x"
    assert not np.all(x_only.sponge_mask < 1.0)
    assert np.all(x_only.sponge_mask[16:-16] == 1.0)
    assert np.all(x_only.sponge_mask[:, 0] == x_only.sponge_mask[:, -1])

    production_default = _solver(
        solver_name,
        nx=128,
        ny=4,
        cfl=0.2,
        boundary="open",
        use_sponge=True,
        sponge_mode="elapsed_time_consistent",
    )
    assert production_default.sponge_axes == "xy"
    assert np.all(production_default.sponge_mask < 1.0)


def test_boundary_packet_windows_cover_incident_and_separated_reflection() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = __import__("yaml").safe_load(
        (repo / "configs/eval/common_time_v2_level_a.yaml").read_text(
            encoding="utf-8"
        )
    )
    boundary = config["boundary_packet"]
    for solver in ("swe_hydrostatic", "swe_muscl_hr"):
        spec = _resolved_boundary_packet_spec(boundary, solver)
        _validate_boundary_packet_spec(spec, nx=128, sponge_width=24)
        timing = _boundary_timing(solver, spec)
        assert timing["prearrival_times"][-1] < timing["leading_edge_arrival_time"]
        assert timing["postexit_times"][0] >= timing["trailing_edge_exit_time"]

    boussinesq = _resolved_boundary_packet_spec(boundary, "boussinesq")
    _packet_spec, _finite, _reference, _metadata, timing = (
        _boussinesq_spectral_packet_bundle(boussinesq, role="reflection")
    )
    assert timing["prearrival_times"][-1] < timing["leading_edge_arrival_time"]
    assert timing["postexit_times"][0] >= timing["trailing_edge_exit_time"]
    assert timing["reference_safe"] is True

    invalid = _resolved_boundary_packet_spec(boundary, "swe_hydrostatic")
    invalid["incident_window"] = [0.70, 0.90]
    with pytest.raises(ValueError, match="enough initial packet energy"):
        _validate_boundary_packet_spec(invalid, nx=128, sponge_width=24)

    invalid = _resolved_boundary_packet_spec(boundary, "swe_hydrostatic")
    invalid["reflected_window"] = [0.00, 0.20]
    with pytest.raises(ValueError, match="cannot contain"):
        _validate_boundary_packet_spec(invalid, nx=128, sponge_width=24)


def test_boundary_metrics_require_packet_exit_and_use_characteristics() -> None:
    nx, ny = 128, 4
    repo = Path(__file__).resolve().parents[1]
    config = __import__("yaml").safe_load(
        (repo / "configs/eval/common_time_v2_level_a.yaml").read_text(
            encoding="utf-8"
        )
    )
    spec = _resolved_boundary_packet_spec(
        config["boundary_packet"], "swe_hydrostatic"
    )
    timing = _boundary_timing("swe_hydrostatic", spec)
    times = np.asarray(timing["requested_times"], dtype=np.float64)
    initial = _boundary_initial_conditions(
        "swe_hydrostatic", nx=nx, ny=ny, spec=spec
    )
    bathymetry = -np.ones((nx, ny), dtype=np.float64)
    baseline = np.zeros((times.size, 3, nx, ny), dtype=np.float64)
    baseline[:, 0] = 1.0
    prearrival = times < float(timing["leading_edge_arrival_time"])
    baseline[prearrival, 0] = np.asarray(initial["h0"])
    baseline[prearrival, 1] = np.asarray(initial["hu0"])
    candidate = baseline.copy()
    reflected = np.arange(nx) / nx <= 0.50
    first_post = int(
        np.flatnonzero(times >= float(timing["trailing_edge_exit_time"]))[0]
    )
    reflected_eta = 0.01 * float(np.max(initial["eta0"]))
    candidate[first_post, 0, reflected] += reflected_eta
    candidate[first_post, 1, reflected] += np.sqrt(9.81) * reflected_eta
    metrics = _boundary_metrics(
        solver_name="swe_hydrostatic",
        baseline=baseline,
        candidate=candidate,
        initial_conditions=initial,
        bathymetry=bathymetry,
        spec=spec,
        timing=timing,
        timestamps=times,
    )
    assert metrics["measurement_temporally_separated"] is True
    assert metrics["packet_exit_achieved"] is True
    assert metrics["reflection_metrics_valid"] is True
    assert metrics["reflected_amplitude_ratio"] == pytest.approx(0.01)

    uncleared = baseline.copy()
    postexit = times >= float(timing["trailing_edge_exit_time"])
    uncleared[postexit, 0] = np.asarray(initial["h0"])
    uncleared[postexit, 1] = np.asarray(initial["hu0"])
    uncleared_metrics = _boundary_metrics(
        solver_name="swe_hydrostatic",
        baseline=baseline,
        candidate=uncleared,
        initial_conditions=initial,
        bathymetry=bathymetry,
        spec=spec,
        timing=timing,
        timestamps=times,
    )
    assert uncleared_metrics["packet_exit_achieved"] is False
    assert uncleared_metrics["reflection_metrics_valid"] is False

    with pytest.raises(ValueError, match="timestamps"):
        _boundary_metrics(
            solver_name="swe_hydrostatic",
            baseline=baseline[:-1],
            candidate=candidate[:-1],
            initial_conditions=initial,
            bathymetry=bathymetry,
            spec=spec,
            timing=timing,
            timestamps=times[:-1],
        )


def test_boussinesq_spectral_directional_decomposition_separates_packet() -> None:
    nx, ny = 64, 4
    x = np.arange(nx, dtype=np.float64)[:, None] / nx
    eta = 1.0e-5 * np.cos(4.0 * np.pi * x) * np.ones((1, ny))
    eta_t = _boussinesq_directional_rate(eta, depth=1.0, direction="left")
    states = np.stack([eta, eta_t], axis=0)[None, ...]
    rightgoing, leftgoing = _boussinesq_directional_components(states, depth=1.0)
    np.testing.assert_allclose(rightgoing, 0.0, rtol=0.0, atol=2.0e-20)
    np.testing.assert_allclose(leftgoing[0], eta, rtol=1.0e-14, atol=2.0e-20)


def test_float32_requested_state_contaminates_conservation_measurement() -> None:
    eta0 = _packet(64, 4, center=0.5, sigma=0.06)
    h0 = 1.0 + eta0
    exact_initial = _stable_sum(h0)
    internal = _invariant_metrics(
        [exact_initial, _stable_sum(h0.copy())],
        normalization_scale=exact_initial,
        roundoff_floor_absolute=0.0,
    )
    serialized = _invariant_metrics(
        [exact_initial, _stable_sum(h0.astype(np.float32))],
        normalization_scale=exact_initial,
        roundoff_floor_absolute=0.0,
    )
    assert internal["normalized_drift"] == 0.0
    assert serialized["normalized_drift"] > 1.0e-10


def test_float64_natural_state_conservation_reports_precision_floor() -> None:
    row = _run_float64_conservation(
        "swe_hydrostatic",
        nx=16,
        ny=4,
        cfl=0.45,
        boundary="periodic",
        safety_factor=8.0,
    )
    assert row["measurement_dtype"] == "float64"
    assert row["measurement_grid"] == "internal_natural_states"
    assert row["invariant_name"] == "total_water_depth"
    assert row["roundoff_floor_absolute"] > 0.0
    assert row["normalized_drift"] <= max(
        1.0e-10, row["roundoff_floor_normalized"]
    )


def test_operator_metrics_isolate_sponge_and_interior_regions() -> None:
    reference = np.ones((3, 16, 2), dtype=np.float64)
    changed = reference.copy()
    sponge = np.zeros(16, dtype=bool)
    sponge[:2] = True
    sponge[-2:] = True
    interior = ~sponge
    changed[:, sponge] *= 0.9
    metrics = _operator_discrepancy_metrics(
        changed,
        reference,
        sponge_region=sponge,
        interior_region=interior,
    )
    assert metrics["trajectory_relative_l2"] > 0.0
    assert metrics["sponge_trajectory_relative_l2"] > 0.0
    assert metrics["interior_trajectory_relative_l2"] == 0.0
    assert metrics["interior_final_time_relative_l2"] == 0.0


def test_hydro_temporal_and_spatial_controls_use_absolute_float64_refinement() -> None:
    shape = (3, 64, 2)
    pattern = np.broadcast_to(
        np.sin(2.0 * np.pi * np.arange(64) / 64)[None, :, None], shape
    )
    exact = 1.0e-5 * pattern
    temporal = [exact + scale * pattern for scale in (8.0e-7, 4.0e-7, 2.0e-7, 1.0e-7)]
    temporal_metrics = _hydro_clean_temporal_metrics(
        temporal,
        minimum_order=0.7,
        precision_floor_safety_factor=64.0,
    )
    assert temporal_metrics["passed"] is True
    np.testing.assert_allclose(temporal_metrics["pairwise_orders"], [1.0, 1.0])
    assert all(
        row["absolute_rms"] > 0.0
        and row["relative_l2"] > 0.0
        for row in temporal_metrics["reference_errors"]
    )

    spatial = {
        grid: np.full((3, grid, 2), scale, dtype=np.float64)
        for grid, scale in ((32, 4.0e-7), (64, 2.0e-7), (128, 1.0e-7))
    }
    spatial_metrics = _hydro_spatial_control_metrics(
        spatial,
        minimum_order=0.7,
        precision_floor_safety_factor=64.0,
    )
    assert spatial_metrics["passed"] is True
    assert spatial_metrics["observed_order"] == pytest.approx(1.0)


def test_derived_replay_tolerates_only_machine_scale_reduction_noise() -> None:
    stored = {
        "component": "operator_sensitivity_summary",
        "elapsed_no_filter_cfl_relative_l2": 8.534534241662949e-5,
        "legacy_cfl_relative_l2": 0.5933710254981277,
    }
    recomputed = {
        **stored,
        "elapsed_no_filter_cfl_relative_l2": 8.534534241662957e-5,
        "legacy_cfl_relative_l2": 0.5933710254981279,
    }
    assert _derived_replay_equal(stored, recomputed)
    materially_changed = dict(recomputed)
    materially_changed["legacy_cfl_relative_l2"] += 1.0e-10
    assert not _derived_replay_equal(stored, materially_changed)
    assert _recomputed_rows_equal([stored], [recomputed])

    direct_stored = {"component": "conservation_health", "value": 1.0}
    direct_changed = {
        "component": "conservation_health",
        "value": float(np.nextafter(1.0, 2.0)),
    }
    assert not _recomputed_rows_equal([direct_stored], [direct_changed])


def test_csv_nested_mappings_regenerate_independent_of_key_order() -> None:
    first = {
        "component": "operator_sensitivity_summary",
        "metrics": {"trajectory": 0.1, "regions": {"sponge": 0.2, "interior": 0.0}},
    }
    reordered = {
        "metrics": {"regions": {"interior": 0.0, "sponge": 0.2}, "trajectory": 0.1},
        "component": "operator_sensitivity_summary",
    }
    regenerated = json.loads(json.dumps(first, sort_keys=True))

    expected = _csv_text([first])
    assert _csv_text([reordered]) == expected
    assert _csv_text([regenerated]) == expected

    parsed = next(csv.DictReader(io.StringIO(expected)))
    assert json.loads(parsed["metrics"]) == first["metrics"]


def _fixture_tasks(count: int = 3) -> list[dict[str, object]]:
    base_specs = [
        {"name": "slow", "value": 1.0, "delay_s": 0.05},
        {"name": "fast", "value": 2.0, "delay_s": 0.0},
        {"name": "middle", "value": 3.0, "delay_s": 0.01},
    ]
    specs = [
        (
            dict(base_specs[index])
            if index < len(base_specs)
            else {"name": f"extra-{index}", "value": float(index + 1)}
        )
        for index in range(count)
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
    reordered = json.loads(json.dumps(config))
    candidates = reordered["boundary_packet"]["candidates"]
    reordered["boundary_packet"]["candidates"] = dict(
        reversed(list(candidates.items()))
    )
    assert first == _build_level_a_task_plan(
        reordered,
        canaries,
        contract_hash="contract",
        code_state_hash="code",
    )
    counts = {
        kind: sum(task["kind"] == kind for task in first)
        for kind in ("analytical", "operator", "boundary", "conservation", "canary")
    }
    assert counts == {
        "analytical": 27,
        "operator": 19,
        "boundary": 16,
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
    boussinesq_boundary_roles = [
        task["spec"]["boundary_role"]
        for task in first
        if task["kind"] == "boundary"
        and task["spec"]["solver"] == "boussinesq"
    ]
    assert boussinesq_boundary_roles.count("reflection") == 3
    assert boussinesq_boundary_roles.count("production_horizon") == 3
    canary_specs = [task["spec"] for task in first if task["kind"] == "canary"]
    assert len(canary_specs) == 18
    assert all(spec["computational_grid"] == 96 for spec in canary_specs)
    assert all(spec["publication_grid"] == 64 for spec in canary_specs)
    assert all(spec["buffer_cells"] == 16 for spec in canary_specs)
    assert all(spec["source_taper_cells"] == 8 for spec in canary_specs)
    assert all(spec["sponge_axes"] == "xy" for spec in canary_specs)
    assert all(spec["sponge_width"] == 16 for spec in canary_specs)
    assert all(spec["sponge_profile"] == "cosine" for spec in canary_specs)
    assert {
        spec["solver"]: spec["boundary"] for spec in canary_specs[:3]
    } == {
        "swe_hydrostatic": "radiation",
        "swe_muscl_hr": "radiation",
        "boussinesq": "open_zero_gradient_edge_padding",
    }


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
        tasks,
        tasks_root=tmp_path / "parallel",
        workers=2,
        max_in_flight=2,
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
    assert parallel_provenance["requested_max_in_flight"] == 2
    assert parallel_provenance["effective_max_in_flight"] == 2
    assert parallel_provenance["peak_in_flight_futures"] == 2
    assert parallel_provenance["process_start_method"] == "spawn"
    assert set(parallel_provenance["thread_environment"]) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }


def test_task_execution_reports_resume_aware_progress(tmp_path: Path) -> None:
    tasks = _fixture_tasks()
    events: list[dict[str, object]] = []
    _execute_level_a_task_plan(
        tasks,
        tasks_root=tmp_path / "progress",
        workers=1,
        progress_callback=lambda event: events.append(dict(event)),
    )
    assert events[0]["event"] == "start"
    assert events[0]["completed"] == 0
    task_events = [event for event in events if event["event"] == "task_completed"]
    assert [event["completed"] for event in task_events] == [1, 2, 3]
    assert events[-1]["event"] == "complete"
    assert events[-1]["completed"] == 3

    resumed: list[dict[str, object]] = []
    _execute_level_a_task_plan(
        tasks,
        tasks_root=tmp_path / "progress",
        workers=1,
        resume=True,
        progress_callback=lambda event: resumed.append(dict(event)),
    )
    assert [event["event"] for event in resumed] == ["start", "complete"]
    assert resumed[0]["completed"] == 3


def test_float64_scientific_tasks_are_serial_parallel_equivalent(
    tmp_path: Path,
) -> None:
    tasks = [
        _make_level_a_task(
            ordinal=index,
            task_id=f"analytical/{solver}",
            kind="analytical",
            spec={
                "role": "modal",
                "solver": solver,
                "grid": 8,
                "ny": 4,
                "mode": 1,
                "cfl": 0.2,
                "amplitude": 1.0e-5,
                "reconstruction_limiter": "minmod",
            },
            contract_hash="float64-equivalence",
            code_state_hash="test-code",
        )
        for index, solver in enumerate(("swe_hydrostatic", "swe_muscl_hr"))
    ]
    serial, _ = _execute_level_a_task_plan(
        tasks, tasks_root=tmp_path / "scientific-serial", workers=1
    )
    parallel, _ = _execute_level_a_task_plan(
        tasks,
        tasks_root=tmp_path / "scientific-parallel",
        workers=2,
        max_in_flight=2,
    )
    assert all(payload["trajectory"].dtype == np.float64 for payload in serial)
    assert all(payload["trajectory"].dtype == np.float64 for payload in parallel)
    assert _scientific_digest(serial) == _scientific_digest(parallel)


def test_task_execution_uses_default_bounded_window(tmp_path: Path) -> None:
    tasks = _fixture_tasks(7)
    serial, _ = _execute_level_a_task_plan(
        tasks, tasks_root=tmp_path / "serial-default", workers=1
    )
    parallel, provenance = _execute_level_a_task_plan(
        tasks, tasks_root=tmp_path / "parallel-default", workers=2
    )
    assert [payload["task"]["task_id"] for payload in parallel] == [
        task["task_id"] for task in tasks
    ]
    assert _scientific_digest(serial) == _scientific_digest(parallel)
    assert provenance["requested_max_in_flight"] is None
    assert provenance["effective_max_in_flight"] == 4
    assert 0 < provenance["peak_in_flight_futures"] <= 4


@pytest.mark.parametrize("max_in_flight", [0, -1])
def test_task_execution_rejects_nonpositive_in_flight_before_artifacts(
    tmp_path: Path, max_in_flight: int
) -> None:
    root = tmp_path / f"invalid-{max_in_flight}"
    with pytest.raises(ValueError, match="max_in_flight must be positive"):
        _execute_level_a_task_plan(
            _fixture_tasks(),
            tasks_root=root,
            workers=2,
            max_in_flight=max_in_flight,
        )
    assert not root.exists()


def test_task_execution_rejects_window_below_effective_workers_before_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "below-workers"
    with pytest.raises(ValueError, match="effective worker count"):
        _execute_level_a_task_plan(
            _fixture_tasks(),
            tasks_root=root,
            workers=2,
            max_in_flight=1,
        )
    assert root.is_dir()
    assert not any(root.iterdir())


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
        _execute_level_a_task_plan(
            tasks, tasks_root=root, workers=2, max_in_flight=2
        )
    first_root = root / f"000-{tasks[0]['task_spec_hash'][:16]}"
    assert first_root.is_dir()
    assert not (root / f"001-{tasks[1]['task_spec_hash'][:16]}").exists()
    completed_state = _task_file_state(first_root)
    tasks[1] = _make_level_a_task(
        ordinal=1,
        task_id="fixture/fail",
        kind="fixture",
        spec={"name": "fail", "value": 2.0},
        contract_hash="fixture-contract",
        code_state_hash="fixture-code",
    )
    resumed, provenance = _execute_level_a_task_plan(
        tasks,
        tasks_root=root,
        workers=2,
        max_in_flight=2,
        resume=True,
    )
    assert [payload["task"]["task_id"] for payload in resumed] == [
        "fixture/first",
        "fixture/fail",
        "fixture/last",
    ]
    assert provenance["effective_workers"] == 1
    assert provenance["effective_max_in_flight"] == 1
    assert _task_file_state(first_root) == completed_state


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
        tasks,
        tasks_root=root,
        workers=2,
        max_in_flight=2,
        resume=True,
    )
    assert _task_file_state(root) == before
    assert [payload["task"]["task_id"] for payload in resumed] == [
        task["task_id"] for task in tasks
    ]
    assert provenance["effective_workers"] == 0
    assert provenance["requested_max_in_flight"] == 2
    assert provenance["effective_max_in_flight"] == 0
    assert provenance["peak_in_flight_futures"] == 0
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
            requested_workers=1,
            effective_workers=1,
            requested_max_in_flight=2,
            effective_max_in_flight=0,
            peak_in_flight_futures=0,
        ),
    }
    second = {
        "rows": [{"value": 1.0, "runtime_s": 9.0}],
        "elapsed_s": 10.0,
        "operational_provenance": _operational_provenance(
            requested_workers=2,
            effective_workers=2,
            requested_max_in_flight=4,
            effective_max_in_flight=4,
            peak_in_flight_futures=3,
        ),
    }
    assert _scientific_digest(first) == _scientific_digest(second)
    second["rows"][0]["value"] = 1.1
    assert _scientific_digest(first) != _scientific_digest(second)
