import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.check_temporal_output_sensitivity import LandingSolver
from scripts.plan_reference_sensitivity import select
from scripts.verify_real_bathymetry_configs import verify
from tsunami_surrogate.data_gen.common_time_v2 import parse_requested_output_config


def test_selection_covers_family_pairs_balances_strength_and_ignores_errors():
    rows = []
    for bed in range(5):
        for source in range(6):
            for strength in (0.16, 0.22, 0.29):
                index = 100 + len(rows)
                rows.append(dict(sample_index=index, scenario_id=f"case-{index}",
                                 bathymetry_type=f"b{bed}", source_type=f"s{source}",
                                 source_strength=strength))
    settings = dict(selection_seed=20261125, exclude_indices=[], scenario_count=30)
    roster, _ = select(rows, settings)
    changed = [{**row, "model_error": 1000 - row["sample_index"]} for row in reversed(rows)]
    assert select(changed, settings)[0] == roster
    assert len({(r["bathymetry_type"], r["source_type"]) for r in roster}) == 30
    assert [sum(r["strength_tercile"] == k for r in roster) for k in (1, 2, 3)] == [10, 10, 10]
    settings["exclude_indices"] = [roster[0]["sample_index"]]
    with pytest.raises(ValueError, match="No candidate"):
        select(rows, settings)


def test_output_landing_caps_steps_without_increasing_natural_dt():
    class Solver:
        def suggest_dt(self, *, target_cfl):
            return 0.4

        def step(self, *, dt, auto_dt):
            assert 0 < dt <= 0.4
    solver = LandingSolver(Solver(), [0.3, 0.9])
    steps = []
    while solver.current_time < 0.9:
        dt = solver.suggest_dt(target_cfl=0.1)
        solver.step(dt=dt, auto_dt=False)
        steps.append(solver.current_time)
    np.testing.assert_allclose(steps, [0.3, 0.7, 0.9])


@pytest.fixture
def saved_gebco_suites(tmp_path):
    configs = tmp_path / "configs"
    data = tmp_path / "data"
    configs.mkdir()
    (data / "configs").mkdir(parents=True)
    original = Path("configs/data/real_bathymetry_v2")
    for source in original.glob("*_dataset.yaml"):
        name = source.name.removesuffix("_dataset.yaml")
        settings = yaml.safe_load(source.read_text())
        settings["dataset"]["num_samples"] = 1
        text = yaml.safe_dump(settings)
        public = configs / source.name
        public.write_text(text)
        (data / "configs" / source.name).write_text(text)
        raw = data / "raw" / name
        raw.mkdir(parents=True)
        (raw / "dataset_config.snapshot.yaml").write_text(text)
        field = raw / "hydrostatic/samples/sample_000000"
        field.mkdir(parents=True)
        requested = parse_requested_output_config(settings["requested_output"])
        np.savez_compressed(field / "sample.npz", timestamps=requested.requested_times,
                            contract_hash=np.asarray(requested.contract_hash),
                            trajectory_eta=np.zeros((50, 64, 64)))
        prep = yaml.safe_load((original / (name + "_preprocess.yaml")).read_text())
        prep["raw"]["generation_config"] = str(public)
        (configs / (name + "_preprocess.yaml")).write_text(yaml.safe_dump(prep))
        saved = copy.deepcopy(prep)
        saved["raw"]["generation_config"] = str(data / "configs" / source.name)
        (data / "configs" / (name + "_preprocess.yaml")).write_text(yaml.safe_dump(saved))
    return configs, data


def test_gebco_verification_rejects_changed_generation_settings(saved_gebco_suites):
    configs, data = saved_gebco_suites
    assert verify(configs, data)["status"] == "verified"
    target = next(configs.glob("*_dataset.yaml"))
    settings = yaml.safe_load(target.read_text())
    settings["solver"]["dx"] *= 2
    target.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="Generation settings differ"):
        verify(configs, data)


def test_gebco_verification_rejects_stale_saved_output_times(saved_gebco_suites):
    configs, data = saved_gebco_suites
    field = next((data / "raw").glob("*/hydrostatic/samples/*/sample.npz"))
    with np.load(field) as f:
        arrays = dict(f)
    arrays["timestamps"] = arrays["timestamps"] / 2400
    np.savez_compressed(field, **arrays)
    with pytest.raises(AssertionError):
        verify(configs, data)
