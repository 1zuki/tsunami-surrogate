from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import (
    authoritative_input_fingerprint,
    code_state,
    hash_array,
    sha256_file,
)
import src.evaluation.finite_horizon_boundary_study as study
from src.evaluation.finite_horizon_boundary_study import (
    _load_task_artifact,
    audit_source_geometry,
    build_task_plan,
    comparison_metrics,
    _aggregate_metrics,
    evaluate_candidate_policies,
    execute_case_tasks,
    padding_control_diagnostics,
    scientific_digest,
    validate_static_freeze,
    verify_artifact_checksums,
    extend_common_domain,
    load_config,
    perturbation_edge_diagnostics,
    reference_boundary_influence_time,
    reference_padding_cells,
    significant_source_mask,
)


CONFIG_PATH = Path("configs/eval/finite_horizon_boundary_study.yaml")


def _fixture() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    bathymetry = -np.ones((8, 8), dtype=np.float32)
    eta0 = np.zeros((8, 8), dtype=np.float32)
    eta0[1:3, 3:5] = np.asarray([[0.2, 0.1], [0.1, 0.05]], dtype=np.float32)
    h0 = (-bathymetry + eta0).astype(np.float32)
    row: dict[str, object] = {
        "split": "train",
        "qualified_id": "train:scenario_000001",
        "scenario_id": "scenario_000001",
        "sample_index": 1,
        "bathymetry_type": "basin",
        "source_type": "gaussian",
        "input_fingerprint": "fixture",
    }
    return row, {
        "bathymetry": bathymetry,
        "eta0": eta0,
        "initial_depth": h0,
    }


def _short_config() -> dict[str, object]:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["production"].update(
        {
            "horizon": 0.001,
            "requested_time_start": 0.0005,
            "requested_time_step": 0.0005,
            "requested_time_count": 2,
            "grid": 8,
        }
    )
    config["static_audit"]["scientific_interior_band_cells"] = 1
    config["large_domain_reference"].update(
        {
            "stencil_safety_cells": 1,
            "perturbation_taper_cells": 2,
            "boussinesq_padding_control_offsets": [0, 2, 4],
        }
    )
    return config


def test_boussinesq_reference_cg_budget_scales_with_padded_axis() -> None:
    config = load_config(CONFIG_PATH)
    assert study.boussinesq_reference_cg_max_iterations((64, 64), config) == 500
    assert study.boussinesq_reference_cg_max_iterations((212, 212), config) == 1657
    assert study.boussinesq_reference_cg_max_iterations((228, 228), config) == 1782


def _authoritative_case(
    root: Path, *, index: int = 1, grid: int = 8
) -> dict[str, object]:
    bathymetry = -np.ones((grid, grid), dtype=np.float32)
    source = np.zeros((grid, grid), dtype=np.float32)
    source[grid // 2 - 1 : grid // 2 + 1, grid // 2 - 1 : grid // 2 + 1] = (
        np.asarray([[0.08, 0.04], [0.02, -0.01]], dtype=np.float32)
    )
    strength_array = np.asarray([0.5], dtype=np.float32)
    strength = float(strength_array[0])
    rest = np.maximum(-bathymetry, 0.0).astype(np.float32)
    eta0 = np.asarray(strength * source, dtype=np.float32)
    initial_depth = np.maximum(rest + eta0, 0.0).astype(np.float32)
    free_surface0 = (initial_depth + bathymetry).astype(np.float32)
    arrays = {
        "bathymetry": bathymetry,
        "source_field": source,
        "rest_depth": rest,
        "eta0": eta0,
        "initial_depth": initial_depth,
        "free_surface0": free_surface0,
    }
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    bathymetry_path = cache / "bathymetry.npz"
    source_path = cache / "source.npz"
    if not bathymetry_path.exists():
        np.savez_compressed(
            bathymetry_path,
            bathymetry=bathymetry,
            bathymetry_type=np.asarray(["basin"], dtype="U64"),
        )
        np.savez_compressed(
            source_path,
            source_field=source,
            source_type=np.asarray(["gaussian"], dtype="U64"),
            source_strength=strength_array,
        )
    scenario_id = f"scenario_{index:06d}"
    row: dict[str, object] = {
        "split": "train",
        "qualified_id": f"train:{scenario_id}",
        "scenario_id": scenario_id,
        "sample_index": index,
        "bathymetry_type": "basin",
        "source_type": "gaussian",
        "source_strength": strength,
        "bathymetry_cache_path": str(bathymetry_path),
        "source_cache_path": str(source_path),
        "array_hashes": {name: hash_array(values) for name, values in arrays.items()},
        "selection_role": "test_fixture",
        "static_risk": {},
    }
    row["input_fingerprint"] = authoritative_input_fingerprint(
        split="train",
        sample_index=index,
        scenario_id=scenario_id,
        bathymetry_type="basin",
        source_type="gaussian",
        source_strength=strength_array,
        arrays=arrays,
    )
    return row


def _expanded_tasks(
    case: dict[str, object], config: dict[str, object], union: np.ndarray
) -> list[dict[str, object]]:
    plan = build_task_plan(
        [case],
        study_hash="a" * 64,
        config_sha256="b" * 64,
        selection_sha256="c" * 64,
        code_state_hash="d" * 64,
        source_union_hash="e" * 64,
    )
    return [
        {
            **task,
            "config": config,
            "case_record": case,
            "global_source_union": union,
        }
        for task in plan
    ]


def _miniature_static_freeze(
    root: Path,
) -> tuple[Path, Path, list[dict[str, object]]]:
    config = _short_config()
    inventory = [_authoritative_case(root, index=index) for index in range(1, 13)]
    inventory_path = root / "inventory.jsonl"
    study._write_jsonl(inventory_path, inventory)
    config["authoritative_inventory"] = "inventory.jsonl"
    config_path = root / "configs" / "eval" / "study.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_path = root / "src" / "fixture.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    selection = [dict(row) for row in inventory]
    output = root / "artifact"
    output.mkdir()
    study._write_jsonl(
        output / "static_audit_rows.jsonl",
        [{"qualified_id": row["qualified_id"]} for row in selection],
    )
    study._write_json(
        output / "static_audit_summary.json",
        {"authoritative_count": 13_500, "split_counts": {"train": 10_000, "val": 1_000, "test": 2_500}},
    )
    study._write_json(output / "diagnostic_selection.json", selection)
    union = np.ones((8, 8), dtype=bool)
    np.save(output / "global_source_union.npy", union, allow_pickle=False)
    state = code_state(root)
    config_hash = sha256_file(config_path)
    inventory_hash = sha256_file(inventory_path)
    selection_hash = sha256_file(output / "diagnostic_selection.json")
    union_raw = hash_array(union)
    union_semantic = study._semantic_hash(
        "global-significant-source-union", union_raw
    )
    binding_hash = study._semantic_hash(
        "finite-horizon-static-bindings",
        {
            "config_sha256": config_hash,
            "inventory_sha256": inventory_hash,
            "selection_sha256": selection_hash,
            "code_state_hash": state["code_state_hash"],
            "source_union_hash": union_semantic,
            "selected": [
                (row["qualified_id"], row["input_fingerprint"])
                for row in selection
            ],
        },
    )
    task_plan = build_task_plan(
        selection,
        study_hash=binding_hash,
        config_sha256=config_hash,
        selection_sha256=selection_hash,
        code_state_hash=state["code_state_hash"],
        source_union_hash=union_semantic,
    )
    study._write_json(output / "task_plan.json", task_plan)
    snapshot_manifest = study._write_source_snapshot(root, output, state)
    static_names = (
        "static_audit_rows.jsonl",
        "static_audit_summary.json",
        "diagnostic_selection.json",
        "global_source_union.npy",
        "task_plan.json",
        "source_snapshot.zip",
        "source_snapshot_manifest.json",
    )
    freeze = {
        "schema_id": study.SCHEMA_ID,
        "artifact_kind": "finite-horizon-boundary-study-static-freeze",
        "status": config["status"],
        "config_sha256": config_hash,
        "inventory_sha256": inventory_hash,
        "code_state": state,
        "code_state_hash": state["code_state_hash"],
        "source_snapshot": {
            "archive": "source_snapshot.zip",
            "manifest": "source_snapshot_manifest.json",
            "source_file_count": snapshot_manifest["source_file_count"],
        },
        "selection_sha256": selection_hash,
        "static_files": {
            name: study._file_record(output / name) for name in static_names
        },
        "source_union": {"raw": union_raw, "semantic_sha256": union_semantic},
        "task_plan_binding_hash": binding_hash,
        "task_count": 36,
        "thresholds_frozen_before_selected_case_execution": config[
            "proposed_future_thresholds"
        ],
        "candidate_policies_frozen_before_selected_case_execution": config[
            "candidate_policies"
        ],
        "selected_case_count": 12,
        "selected_qualified_ids": [row["qualified_id"] for row in selection],
        "created_before_numerical_outcomes": True,
    }
    freeze["study_hash"] = study._freeze_self_hash(freeze)
    study._write_json(output / "STATIC_FREEZE.json", freeze)
    return config_path, output, selection


def test_significant_source_mask_is_deterministic() -> None:
    _row, arrays = _fixture()
    first, energy_first = significant_source_mask(arrays["eta0"], energy_tail=1.0e-6)
    second, energy_second = significant_source_mask(
        arrays["eta0"].copy(), energy_tail=1.0e-6
    )
    assert np.array_equal(first, second)
    assert energy_first == energy_second


def test_static_geometry_audit_is_key_order_independent_and_json_stable() -> None:
    config = load_config(CONFIG_PATH)
    row, arrays = _fixture()
    union, _ = significant_source_mask(arrays["eta0"], energy_tail=1.0e-6)
    first = audit_source_geometry(row, arrays, config, global_source_union=union)
    reversed_arrays = dict(reversed(list(arrays.items())))
    second = audit_source_geometry(
        row, reversed_arrays, config, global_source_union=union
    )
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["split"] == "train"
    assert first["authoritative_split"] == "train"
    assert first["edge_metrics"]["left"]["minimum_support_distance"] == 0.1875


def test_validation_alias_preserves_authoritative_qualified_identity() -> None:
    config = load_config(CONFIG_PATH)
    row, arrays = _fixture()
    row["split"] = "eval"
    row["qualified_id"] = "eval:scenario_000001"
    union, _ = significant_source_mask(arrays["eta0"], energy_tail=1.0e-6)
    result = audit_source_geometry(row, arrays, config, global_source_union=union)
    assert result["split"] == "val"
    assert result["authoritative_split"] == "eval"
    assert result["qualified_id"] == "eval:scenario_000001"


def test_padded_extension_preserves_crop_and_removes_zero_extension_seam() -> None:
    _row, arrays = _fixture()
    eta0 = arrays["eta0"].copy()
    eta0[0, 3] = np.float32(0.2)
    h0 = (-arrays["bathymetry"] + eta0).astype(np.float32)
    extended = extend_common_domain(
        arrays["bathymetry"],
        eta0,
        h0,
        pad_cells=6,
        perturbation_taper_cells=3,
    )
    crop = extended["crop"]
    assert extended["bathymetry"].dtype == np.float64
    assert np.array_equal(extended["bathymetry"][crop], arrays["bathymetry"])
    assert np.array_equal(extended["eta0"][crop], eta0)
    assert np.array_equal(extended["h0"][crop], h0)
    assert extended["seam_jump_max"] == 0.0
    assert extended["perturbation_support_cells"] == 3
    assert np.array_equal(extended["eta0"][5, 6:14], eta0[0, :])
    assert np.array_equal(extended["eta0"][14, 6:14], eta0[-1, :])
    assert np.count_nonzero(extended["eta0"][:3, :]) == 0
    assert np.count_nonzero(extended["eta0"][-3:, :]) == 0

    old_zero = extend_common_domain(
        arrays["bathymetry"],
        eta0,
        h0,
        pad_cells=6,
        perturbation_extension="zero",
        perturbation_taper_cells=3,
    )
    assert old_zero["seam_jump_max"] > 0.0
    outside = np.ones_like(old_zero["eta0"], dtype=bool)
    outside[old_zero["crop"]] = False
    assert np.count_nonzero(old_zero["eta0"][outside]) == 0
    assert np.allclose(
        old_zero["h0"][outside], -old_zero["bathymetry"][outside]
    )


def test_edge_diagnostic_marks_old_zero_extension_incompatible() -> None:
    _row, arrays = _fixture()
    eta0 = arrays["eta0"].copy()
    eta0[0, 3] = np.float32(0.2)
    result = perturbation_edge_diagnostics(
        eta0, absolute_floor=1.0e-7
    )
    assert result["edge_max_abs"] == pytest.approx(0.2)
    assert result["zero_extension_seam_jump_max"] == result["edge_max_abs"]
    assert not result["zero_extension_compatible_at_absolute_floor"]
    assert result["selected_extension_seam_jump_max"] == 0.0


def test_reference_padding_bound_excludes_reflected_return() -> None:
    config = load_config(CONFIG_PATH)
    speed = 9.9
    horizon = float(config["production"]["horizon"])
    dx = 1.0 / int(config["production"]["grid"])
    pad = reference_padding_cells(
        wave_speed_bound=speed, horizon=horizon, dx=dx, config=config
    )
    support = int(config["large_domain_reference"]["perturbation_taper_cells"])
    assert (
        reference_boundary_influence_time(
            pad_cells=pad,
            perturbation_support_cells=support,
            dx=dx,
            wave_speed_bound=speed,
        )
        > horizon
    )


def test_float64_metrics_isolate_interior_and_boundary_regions() -> None:
    reference = np.ones((2, 6, 6), dtype=np.float64)
    candidate = reference.copy()
    candidate[:, 0, :] += 1.0e-8
    rows = comparison_metrics(
        candidate,
        reference,
        boundary_band_cells=1,
        absolute_floor=1.0e-12,
    )
    assert all(row["interior_absolute_rms"] == 0.0 for row in rows)
    assert all(row["boundary_absolute_rms"] > 0.0 for row in rows)
    assert all(row["reference_rms"] == 1.0 for row in rows)
    assert all(row["interior_reference_rms"] == 1.0 for row in rows)
    assert all(row["boundary_reference_rms"] == 1.0 for row in rows)
    assert all(row["reference_amplitude"] == 1.0 for row in rows)
    assert all(
        not row[flag]
        for row in rows
        for flag in study.DENOMINATOR_FLOOR_FLAGS
    )
    assert candidate.dtype == np.float64


def test_config_requires_bounded_spawn_single_thread_policy(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert load_config(CONFIG_PATH)["execution"]["max_in_flight"] == 8
    for old, new in (
        ("max_in_flight: 8", "max_in_flight: 0"),
        ("process_start_method: spawn", "process_start_method: fork"),
        ('OMP_NUM_THREADS: "1"', 'OMP_NUM_THREADS: "2"'),
        (
            "boussinesq_padding_control_offsets: [0, 8, 16]",
            "boussinesq_padding_control_offsets: [0, 8, 8]",
        ),
        ("progress_every: 1", "progress_every: 0"),
        (
            "reference_padding_control_fraction: 0.25",
            "reference_padding_control_fraction: 1.0",
        ),
        (
            "perturbation_extension: edge_cosine_taper",
            "perturbation_extension: zero",
        ),
        ("perturbation_taper_cells: 16", "perturbation_taper_cells: 1"),
        (
            "boundary_influence_safety_factor: 1.25",
            "boundary_influence_safety_factor: 1.0",
        ),
        (
            "float64_roundoff_safety_factor: 64.0",
            "float64_roundoff_safety_factor: 0.5",
        ),
    ):
        path = tmp_path / f"bad-{len(list(tmp_path.iterdir()))}.yaml"
        path.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_aggregates_cover_all_requested_dimensions() -> None:
    rows = []
    for case, source, bathy in (("a", "gaussian", "basin"), ("b", "fault", "island")):
        for timestamp in (0.1, 0.2):
            rows.append({
                "qualified_id": case, "solver": "swe_hydrostatic", "candidate": "no_sponge",
                "source_type": source, "bathymetry_type": bathy, "requested_time": timestamp,
                **{name: timestamp for name in (
                    "absolute_rms", "relative_l2", "interior_absolute_rms",
                    "interior_relative_l2", "boundary_absolute_rms", "boundary_relative_l2",
                    "amplitude_absolute_error", "amplitude_relative_error", "phase_correlation_loss")},
            })
    aggregates = _aggregate_metrics(rows)
    assert {row["dimension"] for row in aggregates} == {
        "overall", "source_family", "bathymetry_family", "requested_time",
        "source_x_time", "bathymetry_x_time",
    }
    overall = next(row for row in aggregates if row["dimension"] == "overall")
    assert overall["unique_case_count"] == 2
    assert overall["row_count"] == 4
    time_rows = [row for row in aggregates if row["dimension"] == "requested_time"]
    assert {row["dimension_values"]["requested_time"] for row in time_rows} == {
        0.1,
        0.2,
    }


def test_task_plan_is_deterministic_and_binds_case_identity(tmp_path: Path) -> None:
    case = _authoritative_case(tmp_path)
    kwargs = {
        "study_hash": "a" * 64,
        "config_sha256": "b" * 64,
        "selection_sha256": "c" * 64,
        "code_state_hash": "d" * 64,
        "source_union_hash": "e" * 64,
    }
    first = build_task_plan([case], **kwargs)
    second = build_task_plan([dict(reversed(list(case.items())))], **kwargs)
    assert first == second
    assert [task["ordinal"] for task in first] == [0, 1, 2]
    changed = dict(case)
    changed["source_type"] = "fault"
    assert build_task_plan([changed], **kwargs)[0]["spec_hash"] != first[0]["spec_hash"]


def test_padding_control_uses_reserved_fraction_and_fails_closed() -> None:
    config = _short_config()
    times = np.asarray([0.0005, 0.001], dtype=np.float64)
    enlarged = np.ones((2, 8, 8), dtype=np.float64)
    base = enlarged.copy()
    middle = enlarged.copy()
    rows, summary = padding_control_diagnostics(
        [base, middle, enlarged],
        [0, 2, 4],
        qualified_id="train:fixture",
        times=times,
        config=config,
    )
    assert summary["allowances"] == {
        "relative_l2": 0.0025,
        "interior_relative_l2": 0.00125,
        "amplitude_relative_error": 0.0025,
        "phase_correlation_loss": 0.00125,
    }
    assert summary["adequate"]
    assert summary["absolute_rms_floor_role"] == "denominator_floor_only_not_error_ceiling"
    assert summary["convergence_measure_by_metric"] == {
        "relative_l2": "absolute_rms",
        "interior_relative_l2": "interior_absolute_rms",
        "amplitude_relative_error": None,
        "phase_correlation_loss": None,
    }
    assert all(
        summary["convergence_precision_tolerance_by_metric"][metric] > 0.0
        for metric in ("relative_l2", "interior_relative_l2")
    )
    assert summary["offsets"] == [0, 2, 4]
    assert len(summary["pairs"]) == 2
    assert len(rows) == 4
    assert all(row["adequate"] for row in rows)

    middle[:, 2:6, 2:6] += 0.1
    _rows, failed = padding_control_diagnostics(
        [base, middle, enlarged],
        [0, 2, 4],
        qualified_id="train:fixture",
        times=times,
        config=config,
    )
    assert not failed["adequate"]
    assert not failed["adequate_by_metric"]["relative_l2"]

    first = np.ones((2, 8, 8), dtype=np.float64)
    second = np.full((2, 8, 8), 1.0001, dtype=np.float64)
    third = np.full((2, 8, 8), 1.0003, dtype=np.float64)
    _rows, nonconvergent = padding_control_diagnostics(
        [first, second, third],
        [0, 2, 4],
        qualified_id="train:fixture",
        times=times,
        config=config,
    )
    assert all(nonconvergent["adequate_by_metric"].values())
    assert not nonconvergent["adequate"]
    assert not nonconvergent["nonincreasing_by_metric"]["relative_l2"]

    first = np.ones((2, 8, 8), dtype=np.float64)
    second = np.full((2, 8, 8), 1.00005, dtype=np.float64)
    third = second.copy()
    third[:, 0, 0] += 0.000051
    _rows, nonmonotone_amplitude = padding_control_diagnostics(
        [first, second, third],
        [0, 2, 4],
        qualified_id="train:fixture",
        times=times,
        config=config,
    )
    assert nonmonotone_amplitude["adequate"]
    assert not nonmonotone_amplitude["nonincreasing_by_metric"][
        "amplitude_relative_error"
    ]
    assert not nonmonotone_amplitude["monotonicity_required_by_metric"][
        "amplitude_relative_error"
    ]
    assert all(
        nonmonotone_amplitude["convergence_requirement_met_by_metric"].values()
    )

    first = np.ones((2, 8, 8), dtype=np.float64)
    second = first + 1.0e-5
    third = second + 1.0e-5 + 4.0 * np.spacing(1.0)
    _rows, roundoff_scale_increase = padding_control_diagnostics(
        [first, second, third],
        [0, 2, 4],
        qualified_id="train:fixture",
        times=times,
        config=config,
    )
    previous_absolute = roundoff_scale_increase["pairs"][0][
        "convergence_maxima"
    ]["absolute_rms"]
    final_absolute = roundoff_scale_increase["pairs"][1][
        "convergence_maxima"
    ]["absolute_rms"]
    assert final_absolute > previous_absolute
    assert roundoff_scale_increase["nonincreasing_by_metric"]["relative_l2"]
    assert roundoff_scale_increase["adequate"]


def test_checksum_finalization_is_idempotent_and_detects_corruption(tmp_path: Path) -> None:
    (tmp_path / "STATIC_FREEZE.json").write_text("{}\n", encoding="utf-8")
    execution = tmp_path / "execution"
    execution.mkdir()
    (execution / "result.json").write_text("{}\n", encoding="utf-8")
    (execution / "STUDY_RESULT.json").write_text("{}\n", encoding="utf-8")
    first = verify_artifact_checksums(tmp_path)
    checksum_bytes = (tmp_path / "SHA256SUMS.txt").read_bytes()
    assert verify_artifact_checksums(tmp_path) == first
    assert (tmp_path / "SHA256SUMS.txt").read_bytes() == checksum_bytes
    (execution / "result.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_artifact_checksums(tmp_path)


def test_scientific_digest_excludes_only_operational_fields() -> None:
    base = {"solver": "boussinesq", "rows": [{"relative_l2": 0.1}], "runtime_s": 1.0}
    changed_runtime = {**base, "runtime_s": 99.0, "pid": 123}
    assert scientific_digest(base) == scientific_digest(changed_runtime)
    changed_science = {**base, "rows": [{"relative_l2": 0.2}]}
    assert scientific_digest(base) != scientific_digest(changed_science)


def test_static_freeze_validates_code_plan_and_inputs(tmp_path: Path) -> None:
    config_path, output, _selection = _miniature_static_freeze(tmp_path)
    validated = validate_static_freeze(
        repo_root=tmp_path,
        config_path=config_path,
        output_dir=output,
    )
    assert validated["freeze"]["task_count"] == 36
    assert len(validated["plan"]) == 36
    assert validated["union"].dtype == np.bool_


def test_static_freeze_rejects_code_state_change(tmp_path: Path) -> None:
    config_path, output, _selection = _miniature_static_freeze(tmp_path)
    (tmp_path / "src" / "fixture.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="code state"):
        validate_static_freeze(
            repo_root=tmp_path,
            config_path=config_path,
            output_dir=output,
        )


def test_static_freeze_rejects_source_snapshot_corruption(tmp_path: Path) -> None:
    config_path, output, _selection = _miniature_static_freeze(tmp_path)
    archive = output / "source_snapshot.zip"
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="frozen static file"):
        validate_static_freeze(
            repo_root=tmp_path,
            config_path=config_path,
            output_dir=output,
        )


def test_static_freeze_rejects_source_union_semantic_corruption(
    tmp_path: Path,
) -> None:
    config_path, output, _selection = _miniature_static_freeze(tmp_path)
    union_path = output / "global_source_union.npy"
    union = np.load(union_path, allow_pickle=False)
    union[0, 0] = False
    np.save(union_path, union, allow_pickle=False)
    freeze_path = output / "STATIC_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["static_files"]["global_source_union.npy"] = study._file_record(
        union_path
    )
    freeze["study_hash"] = study._freeze_self_hash(freeze)
    study._write_json(freeze_path, freeze)
    with pytest.raises(RuntimeError, match="global source union"):
        validate_static_freeze(
            repo_root=tmp_path,
            config_path=config_path,
            output_dir=output,
        )


def test_static_freeze_rejects_self_hash_corruption(tmp_path: Path) -> None:
    config_path, output, _selection = _miniature_static_freeze(tmp_path)
    freeze_path = output / "STATIC_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["task_count"] = 35
    study._write_json(freeze_path, freeze)
    with pytest.raises(RuntimeError, match="self-hash"):
        validate_static_freeze(
            repo_root=tmp_path,
            config_path=config_path,
            output_dir=output,
        )


@pytest.fixture(scope="module")
def real_solver_task_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("finite-horizon-real-solvers")
    config = _short_config()
    case = _authoritative_case(root)
    union = np.zeros((8, 8), dtype=bool)
    tasks = _expanded_tasks(case, config, union)
    serial_root = root / "serial"
    parallel_root = root / "parallel"
    serial_provenance: dict[str, object] = {}
    parallel_provenance: dict[str, object] = {}
    serial_progress: list[dict[str, object]] = []
    parallel_progress: list[dict[str, object]] = []
    serial = execute_case_tasks(
        tasks,
        workers=1,
        start_method="spawn",
        max_in_flight=1,
        task_root=serial_root,
        execution_provenance=serial_provenance,
        progress_callback=lambda event: serial_progress.append(dict(event)),
    )
    parallel = execute_case_tasks(
        tasks,
        workers=2,
        start_method="spawn",
        max_in_flight=2,
        task_root=parallel_root,
        execution_provenance=parallel_provenance,
        progress_callback=lambda event: parallel_progress.append(dict(event)),
    )
    return {
        "root": root,
        "config": config,
        "case": case,
        "union": union,
        "tasks": tasks,
        "serial_root": serial_root,
        "parallel_root": parallel_root,
        "serial": serial,
        "parallel": parallel,
        "serial_provenance": serial_provenance,
        "parallel_provenance": parallel_provenance,
        "serial_progress": serial_progress,
        "parallel_progress": parallel_progress,
    }


def test_real_solver_tasks_match_serial_and_spawn(
    real_solver_task_runs: dict[str, object],
) -> None:
    serial = real_solver_task_runs["serial"]
    parallel = real_solver_task_runs["parallel"]
    assert [row["solver"] for row in serial] == list(study.SOLVERS)
    assert [scientific_digest(row) for row in serial] == [
        scientific_digest(row) for row in parallel
    ]
    assert _aggregate_metrics([row for result in serial for row in result["rows"]]) == (
        _aggregate_metrics([row for result in parallel for row in result["rows"]])
    )
    boussinesq = next(row for row in serial if row["solver"] == "boussinesq")
    assert boussinesq["comparison_pad_cells"] > boussinesq["pad_cells"]
    assert len(boussinesq["padding_control_rows"]) == 4
    assert boussinesq["base_reference_health"]["measurement_dtype"] == "float64"
    assert len(boussinesq["padding_reference_health"]) == 3
    assert boussinesq["comparison_reference_health"]["measurement_dtype"] == "float64"
    assert real_solver_task_runs["parallel_provenance"]["peak_in_flight"] == 2
    for key in ("serial_progress", "parallel_progress"):
        events = real_solver_task_runs[key]
        assert events[0]["event"] == "start"
        assert events[-1]["event"] == "complete"
        assert [event["completed"] for event in events[1:-1]] == [1, 2, 3]


def test_task_resume_is_byte_and_timestamp_noop_and_recovers_known_staging(
    real_solver_task_runs: dict[str, object],
) -> None:
    task_root = real_solver_task_runs["serial_root"]
    tasks = real_solver_task_runs["tasks"]
    first_directory = task_root / study._task_directory_name(tasks[0])
    known_staging = task_root / f".{first_directory.name}.staging-123"
    known_staging.mkdir()

    def snapshot() -> dict[str, tuple[bytes, int]]:
        return {
            path.relative_to(task_root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in task_root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    progress: list[dict[str, object]] = []
    resumed = execute_case_tasks(
        tasks,
        workers=1,
        start_method="spawn",
        max_in_flight=1,
        task_root=task_root,
        resume=True,
        progress_callback=lambda event: progress.append(dict(event)),
    )
    assert not known_staging.exists()
    assert snapshot() == before
    assert [scientific_digest(row) for row in resumed] == [
        scientific_digest(row) for row in real_solver_task_runs["serial"]
    ]
    assert [event["event"] for event in progress] == ["start", "complete"]
    assert progress[0]["completed"] == 3
    assert progress[0]["resumed"] == 3


def test_task_corruption_and_extra_artifacts_are_rejected(
    tmp_path: Path, real_solver_task_runs: dict[str, object]
) -> None:
    copied = tmp_path / "tasks"
    shutil.copytree(real_solver_task_runs["serial_root"], copied)
    task = real_solver_task_runs["tasks"][0]
    directory = copied / study._task_directory_name(task)
    (directory / "result.json").write_text('{"corrupt": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        _load_task_artifact(copied, task)

    extra_root = tmp_path / "extra"
    shutil.copytree(real_solver_task_runs["serial_root"], extra_root)
    (extra_root / "unexpected").mkdir()
    with pytest.raises(RuntimeError, match="unexpected task artifact"):
        execute_case_tasks(
            real_solver_task_runs["tasks"],
            workers=1,
            start_method="spawn",
            max_in_flight=1,
            task_root=extra_root,
            resume=True,
        )


def test_failure_preserves_previously_completed_atomic_task(
    tmp_path: Path, real_solver_task_runs: dict[str, object]
) -> None:
    tasks = copy.deepcopy(real_solver_task_runs["tasks"][:2])
    bad_case = dict(tasks[1]["case_record"])
    bad_case["source_cache_path"] = str(tmp_path / "missing-source.npz")
    tasks[1]["case_record"] = bad_case
    task_root = tmp_path / "failure-tasks"
    with pytest.raises(FileNotFoundError):
        execute_case_tasks(
            tasks,
            workers=1,
            start_method="spawn",
            max_in_flight=1,
            task_root=task_root,
        )
    assert (task_root / study._task_directory_name(tasks[0])).is_dir()
    assert not (task_root / study._task_directory_name(tasks[1])).exists()


def test_policy_assessment_is_deterministic(
    real_solver_task_runs: dict[str, object],
) -> None:
    results = real_solver_task_runs["serial"]
    rows = [row for result in results for row in result["rows"]]
    assessment = evaluate_candidate_policies(
        rows,
        results,
        real_solver_task_runs["config"],
        global_source_union=real_solver_task_runs["union"],
    )
    assert set(assessment["candidate_assessments"]) == set(study.SOLVERS)
    assert set(assessment["policies"]) == set(
        real_solver_task_runs["config"]["candidate_policies"]
    )
    assert "finite_horizon_supported" in assessment["policies"]["no_sponge_all"]
    padded = assessment["policies"]["larger_domain_central_crop"]
    assert not padded["production_policy_eligible"]
    assert not padded["finite_horizon_supported"]
