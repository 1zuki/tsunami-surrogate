from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from tsunami_surrogate.data_gen.simulate_dataset import (
    _crop_rollout,
    _downsample_rollout,
    _prepare_buffered_domain,
    _run_fde_rollout,
)

from scripts import run_production_dense_diagnostic as replay


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def case(tmp_path: Path):
    config_path = tmp_path / "dataset.yaml"
    selection_path = tmp_path / "selection.json"
    cache_root = tmp_path / "cache"
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    scenario = "scenario_000001"
    cfg = yaml.safe_load((replay.ROOT / "configs/data/dataset_test.yaml").read_text())
    cfg["dataset"].update(num_samples=1, auto_dt=False, sea_level_offset=0.025)
    cfg["requested_output"].update(
        start=0.035, step=0.035, count=3, horizon=0.105, max_natural_steps=3
    )
    cfg["paired_inputs"].update(
        master_shape=[8, 8], solver_shape=[4, 4], target_shape=[2, 2]
    )
    cfg["paired_inputs"].pop("source_spectral_acceptance")
    cfg["computational_domain"].update(buffer_cells=2, source_taper_cells=2)
    cfg["solver"].update(nx=8, ny=8, dx=1.0, dy=1.0, dt=0.1, sponge_width=2)
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _write_json(
        selection_path,
        {
            "audit_hash": "fixture",
            "dense_validation": {
                "ordered_scenarios": [{"scenario_id": scenario}],
                "count": 1,
                "list_hash": replay.stable_hash_scenario_ids([scenario]),
            },
        },
    )
    dataset, solver_cfgs = replay._resolve_settings(cfg)
    requested_times = dataset.requested_output.requested_times
    bathy = np.linspace(-1.5, -1.25, 16, dtype=np.float32).reshape(4, 4)
    source = np.zeros((4, 4), dtype=np.float32)
    source[1:3, 1:3] = [[0.1, -0.2], [0.3, 0.4]]
    strength = 0.2
    (cache_root / "bathymetry").mkdir(parents=True)
    (cache_root / "sources").mkdir()
    np.savez(
        cache_root / "bathymetry/sample_000001.npz",
        solver_bathymetry=bathy,
        bathymetry_type=["trench"],
    )
    np.savez(
        cache_root / "sources/sample_000001.npz",
        solver_source_field=source,
        source_strength=[strength],
        source_type=["dipole"],
    )
    prepared = _prepare_buffered_domain(
        bathy,
        source,
        strength,
        dataset.sea_level_offset,
        dataset.buffered_domain,
        source_already_tapered=True,
    )
    for solver, solver_cfg in solver_cfgs.items():
        paths = replay._case_artifact_paths(
            1, solver, raw_root=raw_root, cache_root=cache_root
        )
        paths["sample.npz"].parent.mkdir(parents=True)
        rollout = _run_fde_rollout(
            fde_name=solver,
            solver_cfg=solver_cfg,
            dataset=replace(dataset, target_cfl=solver_cfg["cfl"]),
            bathymetry=prepared["solver_bathymetry"],
            eta0=prepared["solver_eta0"],
            h0=prepared["solver_h0"],
            requested_times=requested_times,
            max_natural_steps=dataset.requested_output.max_natural_steps,
        )
        eta = _downsample_rollout(
            _crop_rollout(rollout, prepared["crop"]), (2, 2)
        ).trajectory_eta
        np.savez(
            paths["sample.npz"],
            trajectory_eta=eta.astype(np.float32),
            timestamps=requested_times,
            source_strength=[strength],
        )
        np.savez(paths["provenance.npz"], **rollout.diagnostics)
        lineage = {
            **dataset.paired_inputs.semantics(),
            "solver_bathymetry_sha256": replay._array_hash(bathy),
            "solver_source_sha256": replay._array_hash(source),
        }
        common_metadata = {
            "scenario_id": scenario,
            "sample_index": 1,
            "solver_name": solver,
            "split": "test",
            "contract_hash": dataset.requested_output.contract_hash,
            "resolved_config_hash": "archived-config",
            "code_state_hash": "archived-code",
            "input_lineage": lineage,
        }
        _write_json(
            paths["meta.json"],
            {
                **common_metadata,
                "computational_domain": replay._production_grid(dataset),
                "source_strength": strength,
                "source_type": "dipole",
                "bathymetry_type": "trench",
                "health_summary": {"unavailable": float("nan")},
            },
        )
        _write_json(
            paths["publication.json"],
            {
                **common_metadata,
                "schema_id": replay.PUBLICATION_SCHEMA_ID,
                "files": [
                    {
                        "name": name,
                        "sha256": replay.sha256_file(paths[name]),
                        "size_bytes": paths[name].stat().st_size,
                    }
                    for name in ("sample.npz", "meta.json", "provenance.npz")
                ],
            },
        )
    argv = [
        "--config",
        str(config_path),
        "--selection",
        str(selection_path),
        "--cache-root",
        str(cache_root),
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--workers",
        "1",
    ]
    return argv, output_root, dataset


def test_replay_and_zero_work_resume_preserve_execution(case):
    argv, output_root, _ = case
    replay.main(argv)
    manifest_path = output_root / replay.EXECUTION_MANIFEST_NAME
    checkpoint_path = output_root / replay.CHECKPOINT_NAME
    report_path = output_root / replay.REPORT_NAME
    execution_bytes = manifest_path.read_bytes()
    checkpoint_bytes = checkpoint_path.read_bytes()
    initial_report = replay._decode_json(report_path.read_bytes())
    assert initial_report["requested_times"] == [0.035, 0.07, 0.105]
    assert initial_report["production_grid"]["solver_shape"] == [8, 8]
    assert initial_report["production_grid"]["publication_shape"] == [2, 2]
    assert initial_report["production_grid"]["buffer_cells"] == 2
    assert initial_report["execution_contract"]["sea_level_offset"] == 0.025
    assert initial_report["resumed_row_count"] == 0
    assert initial_report["rows_written"] == 3
    assert {row["solver"] for row in initial_report["rows"]} == set(replay.SOLVERS)
    for row in initial_report["rows"]:
        assert row["requested_count"] == 3
        assert row["requested_horizon"] == 0.105
        assert row["final_natural_timestamp"] == 0.2
        assert row["natural_overshoot"] == pytest.approx(0.095)
        assert row["natural_step_count"] == 2
        assert row["natural_step_count_matches_archive"]
        assert row["natural_dt_max_abs_difference"] == 0
        assert row["replay_max_abs"] == 0
        if row["solver"] == "boussinesq":
            assert row["min_depth_seen"] is None
            assert row["max_speed_seen"] is None
            assert row["requested_horizon_mass_change_estimate"] is None
        else:
            assert row["requested_horizon_mass_change_estimate"] == pytest.approx(
                row["requested_horizon_boundary_mass_change_estimate"]
                + row["requested_horizon_sponge_projection_residual_estimate"]
            )

    replay.main(argv)
    resumed_report = replay._decode_json(report_path.read_bytes())
    assert resumed_report["task_count"] == 0
    assert resumed_report["resumed_row_count"] == 3
    assert resumed_report["rows"] == initial_report["rows"]
    assert (
        resumed_report["execution_fingerprint"]
        == initial_report["execution_fingerprint"]
    )
    assert manifest_path.read_bytes() == execution_bytes
    assert checkpoint_path.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize(
    "mutation", ["config", "selection", "source", "cache", "raw", "provenance"]
)
def test_resume_rejects_changed_execution(case, monkeypatch, mutation):
    argv, output_root, _ = case
    replay.main(argv)
    report_path = output_root / replay.REPORT_NAME
    report_bytes = report_path.read_bytes()
    if mutation == "source":
        source_hashes = replay._source_hashes()
        source_hashes["src/solver/hydrostatic_swe.py"] = "modified"
        monkeypatch.setattr(replay, "_source_hashes", lambda: source_hashes)
    else:
        paths = {
            "config": Path(argv[1]),
            "selection": Path(argv[3]),
            "cache": Path(argv[5]) / "bathymetry/sample_000001.npz",
            "raw": Path(argv[7]) / "hydrostatic/samples/sample_000001/sample.npz",
            "provenance": Path(argv[7])
            / "hydrostatic/samples/sample_000001/provenance.npz",
        }
        path = paths[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="execution manifest mismatch"):
        replay.main(argv)
    assert report_path.read_bytes() == report_bytes


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("legacy", "lack an execution manifest"),
        ("duplicate", "Duplicate checkpoint row"),
        ("extra", "outside the selected solver/scenario set"),
        ("stale", "execution fingerprint mismatch"),
        ("changed_metric", "checksum mismatch"),
    ],
)
def test_resume_rejects_invalid_rows(case, mutation, message):
    argv, output_root, _ = case
    if mutation == "legacy":
        output_root.mkdir()
        (output_root / replay.CHECKPOINT_NAME).write_text(
            '{"solver": "boussinesq", "sample_index": 1}\n'
        )
    else:
        replay.main(argv)
        checkpoint_path = output_root / replay.CHECKPOINT_NAME
        lines = checkpoint_path.read_text().splitlines()
        row = json.loads(lines[0])
        if mutation == "duplicate":
            lines.append(lines[0])
        else:
            if mutation == "extra":
                row.update(scenario_id="scenario_000002", sample_index=2)
                row.pop("row_sha256")
                row["row_sha256"] = replay._fingerprint("checkpoint-row", row)
            elif mutation == "stale":
                row["execution_fingerprint"] = "stale"
            elif mutation == "changed_metric":
                row["natural_step_count"] += 1
            lines[0] = json.dumps(row)
        checkpoint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        replay.main(argv)


def test_partial_resume_fills_exact_coverage(case):
    argv, output_root, _ = case
    replay.main(argv)
    checkpoint_path = output_root / replay.CHECKPOINT_NAME
    first_line = checkpoint_path.read_text().splitlines()[0]
    checkpoint_path.write_text(first_line + "\n", encoding="utf-8")
    replay.main(argv)
    report = replay._decode_json((output_root / replay.REPORT_NAME).read_bytes())
    assert report["resumed_row_count"] == 1
    assert report["task_count"] == 2
    assert report["expected_task_count"] == report["rows_written"] == 3
    assert checkpoint_path.read_text().splitlines()[0] == first_line


def test_replay_enforces_configured_step_cap(case):
    argv, _, dataset = case
    config_path = Path(argv[1])
    config = yaml.safe_load(config_path.read_bytes())
    config["requested_output"]["max_natural_steps"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    # Step cap changes do not alter the archived output schedule/contract.
    assert (
        dataset.requested_output.contract_hash
        == replay._resolve_settings(config)[0].requested_output.contract_hash
    )
    with pytest.raises(SystemExit, match="3 production replay failures"):
        replay.main(argv)
    output_root = Path(argv[9])
    assert not (output_root / replay.REPORT_NAME).exists()
    assert all(
        "Natural-step cap reached" in failure
        for failure in json.loads((output_root / "failures.json").read_text())
    )
