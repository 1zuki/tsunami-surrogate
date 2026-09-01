from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_rebuild_pilot import _generate_inputs, _load_split
from src.data_gen.common_time_v2 import validate_publication
from src.data_gen.simulate_dataset import TsunamiDatasetBuilder


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/dataset.yaml"


def _requested_times() -> np.ndarray:
    times = 8.4 + 8.4 * np.arange(50, dtype=np.float64)
    times[-1] = np.float64(420.0)
    return times


def _builder(tmp_path: Path) -> TsunamiDatasetBuilder:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_root = tmp_path / "rebuild"
    cfg["dataset"].update(
        {
            "num_samples": 1,
            "num_workers": 1,
            "bathymetry_dir": str(data_root / "bathymetry"),
            "source_dir": str(data_root / "sources"),
            "output_dir": str(data_root / "raw"),
            "manifest_path": str(
                data_root / "synthetic/scenario_manifest.jsonl"
            ),
            "copy_configs": False,
        }
    )
    cfg["paired_inputs"]["inventory_path"] = str(
        data_root / "synthetic/native_input_inventory.jsonl"
    )
    cfg["fdes"] = {
        "enabled": ["swe_hydrostatic"],
        "primary": "swe_hydrostatic",
    }
    cfg["operations"].update(
        {"enabled": False, "max_in_flight": 1, "solver_progress": False}
    )
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return TsunamiDatasetBuilder(str(path), provenance_config_path=CONFIG)


def test_rebuild_uses_master_solver_and_publishes_64_grid(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path)

    assert builder.dataset.paired_inputs.solver_input == "solver"
    assert builder.dataset.paired_inputs.source_taper_stage == "master"
    assert builder.dataset.paired_inputs.rough_zero_mean_rms_after_taper
    assert builder.dataset.paired_inputs.master_shape == (384, 384)
    assert builder.dataset.paired_inputs.solver_shape == (128, 128)
    assert builder.dataset.paired_inputs.target_shape == (64, 64)
    assert builder.dataset.max_initial_eta_over_depth == 0.10
    assert (builder.solver_cfg["nx"], builder.solver_cfg["ny"]) == (192, 192)
    assert (builder.solver_cfg["dx"], builder.solver_cfg["dy"]) == (
        18.75,
        18.75,
    )
    assert builder.dataset.requested_output is not None
    assert builder.dataset.requested_output.status == "accepted"
    assert builder.dataset.requested_output.execution_scope == "production"
    assert not builder.dataset.requested_output.acknowledged_provisional
    np.testing.assert_array_equal(
        builder.dataset.requested_output.requested_times,
        _requested_times(),
    )

    builder.run(stop_at=1)

    source_path = builder.source_dir / "sample_000001.npz"
    with np.load(source_path, allow_pickle=False) as payload:
        assert str(payload["source_type"][0]) == "rough"
        raw = np.asarray(payload["raw_master_source_field"], dtype=np.float64)
        effective = np.asarray(payload["master_source_field"], dtype=np.float64)
        solver_source = np.asarray(
            payload["solver_source_field"], dtype=np.float64
        )
        target = np.asarray(payload["source_field"], dtype=np.float64)
        strength = float(payload["source_strength"][0])
        sampled_strength = float(payload["sampled_source_strength"][0])
    assert abs(float(np.mean(effective))) <= 1.0e-9
    np.testing.assert_allclose(
        np.sqrt(np.mean(effective * effective)),
        np.sqrt(np.mean(raw * raw)),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    assert target.shape == (64, 64)
    assert solver_source.shape == (128, 128)
    assert np.count_nonzero(target[[0, -1], :]) == 0
    assert np.count_nonzero(target[:, [0, -1]]) == 0
    assert 0.0 < strength <= sampled_strength
    with np.load(
        builder.bathymetry_dir / "sample_000001.npz",
        allow_pickle=False,
    ) as payload:
        master_bathymetry = np.asarray(
            payload["master_bathymetry"], dtype=np.float64
        )
        solver_bathymetry = np.asarray(
            payload["solver_bathymetry"], dtype=np.float64
        )
    assert float(
        np.max(np.abs(strength * effective) / (-master_bathymetry))
    ) <= 0.10 * (1.0 + 1.0e-6)

    pilot = _generate_inputs(_load_split("train"), 1)
    np.testing.assert_array_equal(
        master_bathymetry.astype(np.float32),
        pilot["bathymetry_master"],
    )
    np.testing.assert_array_equal(
        solver_bathymetry.astype(np.float32),
        pilot["bathymetry_128"],
    )
    np.testing.assert_array_equal(
        effective.astype(np.float32),
        pilot["source_master"],
    )
    np.testing.assert_array_equal(
        solver_source.astype(np.float32),
        pilot["source_128"],
    )
    np.testing.assert_array_equal(
        target.astype(np.float32),
        pilot["source_64"],
    )
    assert strength == pilot["source_strength"]

    sample_dir = (
        builder.output_dir / "hydrostatic/samples/sample_000001"
    )
    publication = validate_publication(
        sample_dir,
        expected_times=_requested_times(),
    )
    assert publication["input_lineage"]["solver_input"] == "solver"
    assert publication["input_lineage"]["solver_shape"] == [128, 128]
    with np.load(sample_dir / "sample.npz", allow_pickle=False) as payload:
        assert payload["bathymetry"].shape == (64, 64)
        assert payload["source_field"].shape == (64, 64)
        assert payload["trajectory_eta"].shape == (50, 64, 64)
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["computational_domain"]["solver_shape"] == [192, 192]
    assert meta["computational_domain"]["publication_shape"] == [64, 64]
