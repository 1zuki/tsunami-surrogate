from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_gen.common_time_v2 import candidate_requested_times, validate_publication
from src.data_gen.simulate_dataset import (
    NATIVE_INPUT_SCHEMA_ID,
    TsunamiDatasetBuilder,
    _block_mean_downsample,
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
    _resolved_solver_cfg_for_fde,
)


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = {
    32: {"solver": 48, "buffer": 8, "taper": 4},
    64: {"solver": 96, "buffer": 16, "taper": 8},
    128: {"solver": 192, "buffer": 32, "taper": 16},
}


def _builder(tmp_path: Path, resolution: int, *, count: int = 1) -> TsunamiDatasetBuilder:
    source = ROOT / f"configs/data/multires/dataset_{resolution}.yaml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = tmp_path / f"res{resolution}"
    cfg["dataset"].update(
        {
            "num_samples": count,
            "num_workers": 1,
            "bathymetry_dir": str(root / "bathymetry"),
            "source_dir": str(root / "sources"),
            "output_dir": str(root / "raw"),
            "manifest_path": str(root / "synthetic/scenario_manifest.jsonl"),
            "copy_configs": False,
        }
    )
    cfg["operations"]["enabled"] = False
    cfg["operations"]["max_in_flight"] = 1
    cfg["paired_inputs"]["inventory_path"] = str(
        root / "synthetic/native_input_inventory.jsonl"
    )
    path = tmp_path / f"dataset_{resolution}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return TsunamiDatasetBuilder(str(path), provenance_config_path=source)


def test_native_configs_resolve_common_time_boundaries_and_external_sponges(
    tmp_path: Path,
) -> None:
    factories = {
        "swe_hydrostatic": _make_hydrostatic_solver_from_cfg,
        "swe_muscl_hr": _make_muscl_solver_from_cfg,
        "boussinesq": _make_boussinesq_solver_from_cfg,
    }
    lineage_hashes = set()
    target_contracts = set()
    for resolution, expected in RESOLUTIONS.items():
        builder = _builder(tmp_path, resolution)
        paired = builder.dataset.paired_inputs
        assert builder.dataset.requested_output.status == "provisional"
        assert (
            builder.dataset.requested_output.execution_scope
            == "preparation-only"
        )
        assert not builder.dataset.requested_output.acknowledged_provisional
        lineage_hashes.add(paired.lineage_hash)
        target_contracts.add(paired.target_contract_hash)
        assert paired.master_shape == (128, 128)
        assert paired.target_shape == (resolution, resolution)
        assert builder.dataset.buffered_domain.buffer_cells == expected["buffer"]
        assert builder.dataset.buffered_domain.source_taper_cells == expected["taper"]
        assert builder.dataset.enabled_fdes == (
            "swe_hydrostatic",
            "swe_muscl_hr",
            "boussinesq",
        )
        assert (builder.solver_cfg["nx"], builder.solver_cfg["ny"]) == (
            expected["solver"],
            expected["solver"],
        )
        np.testing.assert_array_equal(
            builder.dataset.requested_output.requested_times,
            candidate_requested_times(),
        )
        for name, factory in factories.items():
            resolved = _resolved_solver_cfg_for_fde(
                builder.solver_cfg, builder.dataset.solver_profiles, name
            )
            solver = factory(resolved)
            width = expected["buffer"]
            crop = solver.sponge_mask[
                width : width + resolution, width : width + resolution
            ]
            np.testing.assert_array_equal(crop, np.ones((resolution, resolution)))
            expected_boundary = "open" if name == "boussinesq" else "radiation"
            assert solver.boundary_x == solver.boundary_y == expected_boundary
    assert len(lineage_hashes) == 1
    assert len(target_contracts) == 3


def test_native_inputs_are_exact_reductions_of_one_master_scenario(
    tmp_path: Path,
) -> None:
    records = {}
    targets = {}
    masters = {}
    for resolution in RESOLUTIONS:
        builder = _builder(tmp_path, resolution)
        builder._phase_generate_bathymetry([1])
        builder._phase_generate_sources([1])
        checksum = builder._freeze_paired_input_inventory([1])
        assert len(checksum) == 64
        record = json.loads(
            builder.dataset.paired_inputs.inventory_path.read_text(
                encoding="utf-8"
            )
        )
        records[resolution] = record
        with np.load(builder.bathymetry_dir / "sample_000001.npz") as payload:
            targets[resolution] = np.asarray(payload["bathymetry"])
            masters[resolution] = np.asarray(payload["master_bathymetry"])

    assert {record["schema_id"] for record in records.values()} == {
        NATIVE_INPUT_SCHEMA_ID
    }
    assert len({record["lineage_hash"] for record in records.values()}) == 1
    assert len(
        {record["master_input_fingerprint"] for record in records.values()}
    ) == 1
    np.testing.assert_array_equal(masters[32], masters[64])
    np.testing.assert_array_equal(masters[64], masters[128])
    for resolution in RESOLUTIONS:
        np.testing.assert_array_equal(
            targets[resolution],
            _block_mean_downsample(masters[resolution], (resolution, resolution)),
        )


def test_native_cache_and_inventory_reuse_fail_closed(tmp_path: Path) -> None:
    builder = _builder(tmp_path, 32)
    builder._phase_generate_bathymetry([1])
    builder._phase_generate_sources([1])
    builder._freeze_paired_input_inventory([1])

    cache = builder.bathymetry_dir / "sample_000001.npz"
    with np.load(cache, allow_pickle=False) as payload:
        values = {name: np.asarray(payload[name]) for name in payload.files}
    values["bathymetry"] = values["bathymetry"].copy()
    values["bathymetry"][0, 0] += np.float32(1.0)
    np.savez_compressed(cache, **values)
    with pytest.raises(RuntimeError, match="exact master-grid reduction"):
        builder._validate_paired_cache(1)

    values["bathymetry"] = _block_mean_downsample(
        values["master_bathymetry"], (32, 32)
    )
    np.savez_compressed(cache, **values)
    inventory = builder.dataset.paired_inputs.inventory_path
    inventory.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen paired input inventory mismatch"):
        builder._freeze_paired_input_inventory([1])


def test_native_32_end_to_end_freezes_roster_before_publication(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path, 32, count=2)
    builder.run(stop_at=1, acknowledge_provisional=True)

    inventory = builder.dataset.paired_inputs.inventory_path
    records = [
        json.loads(line)
        for line in inventory.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert (builder.bathymetry_dir / "sample_000002.npz").is_file()
    assert (builder.source_dir / "sample_000002.npz").is_file()

    for solver in ("hydrostatic", "muscl_hr", "boussinesq"):
        sample_dir = builder.output_dir / solver / "samples/sample_000001"
        publication = validate_publication(
            sample_dir,
            expected_times=candidate_requested_times(),
        )
        assert publication["input_lineage"]["inventory_sha256"] == (
            builder.dataset.paired_input_inventory_sha256
        )
        with np.load(sample_dir / "sample.npz", allow_pickle=False) as payload:
            assert payload["trajectory_eta"].shape == (50, 32, 32)
            np.testing.assert_array_equal(
                payload["timestamps"], candidate_requested_times()
            )


def test_native_static_pairing_is_multiprocess_safe(tmp_path: Path) -> None:
    builder = _builder(tmp_path, 32, count=2)
    builder.dataset.num_workers = 2
    builder._phase_generate_bathymetry([1, 2])
    builder._phase_generate_sources([1, 2])
    checksum = builder._freeze_paired_input_inventory([1, 2])

    assert len(checksum) == 64
    records = builder.dataset.paired_inputs.inventory_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(records) == 2
