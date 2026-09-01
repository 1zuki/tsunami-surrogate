from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import yaml

from scripts.archive_common_time_stage_c import STAGE_C_SOURCES, archive_stage_c
from src.data_gen.common_time_v2 import (
    CANDIDATE_COUNT,
    CONTRACT_SCHEMA_ID,
    build_candidate_contract,
    candidate_requested_times,
    contract_hash,
    hash_array,
    parse_requested_output_config,
    resolved_config_hash,
    validate_candidate_times,
)


def test_candidate_contract_exact_grid_and_stable_hash() -> None:
    times = candidate_requested_times()
    assert times.dtype == np.float64
    assert times.shape == (CANDIDATE_COUNT,)
    assert times[0] == np.float64(0.0035)
    assert times[-1] == np.float64(0.175)
    np.testing.assert_array_equal(validate_candidate_times(times), times)

    contract = build_candidate_contract()
    reordered = dict(reversed(list(contract.items())))
    assert contract_hash(contract) == contract_hash(reordered)
    changed = copy.deepcopy(contract)
    changed["field"] = "depth"
    assert contract_hash(contract) != contract_hash(changed)


@pytest.mark.parametrize(
    "times",
    [
        np.asarray([], dtype=np.float64),
        np.asarray([0.0035] * 50, dtype=np.float64),
        np.concatenate(([0.0], candidate_requested_times()[1:])),
        np.concatenate((candidate_requested_times()[:-1], [np.nan])),
        candidate_requested_times()[:-1],
    ],
)
def test_candidate_contract_rejects_wrong_grid(times: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_candidate_times(times)


def test_requested_config_yaml_and_unknown_key_rejection() -> None:
    cfg = yaml.safe_load(
        Path("configs/data/dataset.yaml").read_text(encoding="utf-8")
    )
    requested = parse_requested_output_config(cfg["requested_output"])
    assert requested is not None
    assert requested.schema_id == CONTRACT_SCHEMA_ID
    assert requested.requested_times.shape == (50,)
    assert not requested.acknowledged_provisional

    invalid = dict(cfg["requested_output"])
    invalid["typo_key"] = True
    with pytest.raises(ValueError, match="Unknown requested_output keys"):
        parse_requested_output_config(invalid)


def test_requested_config_supports_scaled_physical_time_contract() -> None:
    requested = parse_requested_output_config(
        {
            "enabled": True,
            "status": "provisional",
            "execution_scope": "preparation-only",
            "split": "train",
            "start": 8.4,
            "step": 8.4,
            "count": 50,
            "horizon": 420.0,
            "max_natural_steps": 20000,
            "collect_natural_step_health": True,
            "eta_primary": True,
            "physical_scaling": {
                "horizontal_scale": 2400.0,
                "time_scale": 2400.0,
                "vertical_scale": 1.0,
                "aspect_ratio": 2400.0,
                "length_unit": "m",
                "time_unit": "s",
            },
        }
    )

    assert requested is not None
    assert requested.requested_times[0] == np.float64(8.4)
    assert requested.requested_times[-1] == np.float64(420.0)
    assert requested.contract["physical_scaling"]["aspect_ratio"] == 2400.0
    assert requested.contract_hash != contract_hash()


def test_hashes_bind_array_dtype_shape_and_semantic_config() -> None:
    values32 = np.arange(4, dtype=np.float32).reshape(2, 2)
    values64 = values32.astype(np.float64)
    assert hash_array(values32) != hash_array(values64)
    assert hash_array(values32) != hash_array(values32.reshape(4))

    base = resolved_config_hash(
        solver_name="boussinesq",
        solver_config={"dt": 0.1, "filter_strength": 0.0},
        dataset_semantics={"target_cfl": 0.35},
    )
    changed = resolved_config_hash(
        solver_name="boussinesq",
        solver_config={"dt": 0.1, "filter_strength": 0.01},
        dataset_semantics={"target_cfl": 0.35},
    )
    assert base != changed


def test_stage_c_archive_is_byte_exact_and_refuses_overwrite(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for index, rel in enumerate(STAGE_C_SOURCES):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("decision.json"):
            path.write_text(
                json.dumps({"status": "fail", "index": index}), encoding="utf-8"
            )
        elif rel.endswith(".json"):
            path.write_text(json.dumps({"index": index}), encoding="utf-8")
        else:
            path.write_bytes(f"source-{index}\n".encode())
    before = {rel: (repo / rel).read_bytes() for rel in STAGE_C_SOURCES}
    output = tmp_path / "archive"
    final = archive_stage_c(repo_root=repo, output_root=output)

    for rel, raw in before.items():
        assert (repo / rel).read_bytes() == raw
        assert (final / "payload" / rel).read_bytes() == raw
    assert (final / "manifest.json").is_file()
    assert (final / "SHA256SUMS.txt").is_file()
    with pytest.raises(FileExistsError):
        archive_stage_c(repo_root=repo, output_root=output)
