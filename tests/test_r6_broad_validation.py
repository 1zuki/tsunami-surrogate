from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.r6_broad_validation import (
    R6ValidationError,
    _finalize_output,
    _verify_completed_geoclaw,
    _requested_times,
    _validate_config,
)
from src.evaluation.common_time_v2_level_a import validate_checksums


def _config() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs/eval/r6_broad_validation.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def test_r6_broad_validation_config_freezes_current_geometry_and_times() -> None:
    config = _config()
    _validate_config(config)

    production = config["corrected_production"]
    assert isinstance(production, dict)
    times = _requested_times(production["requested_times"])
    np.testing.assert_array_equal(times, 8.4 + 8.4 * np.arange(50, dtype=np.float64))
    assert production["master_shape"] == [384, 384]
    assert production["solver_input_shape"] == [128, 128]
    assert production["computational_shape"] == [192, 192]
    assert production["publication_shape"] == [64, 64]


def test_r6_broad_validation_rejects_the_historical_geometry() -> None:
    config = deepcopy(_config())
    production = config["corrected_production"]
    assert isinstance(production, dict)
    production["computational_shape"] = [96, 96]

    with pytest.raises(R6ValidationError, match="computational_shape"):
        _validate_config(config)


def test_r6_broad_validation_binds_the_completed_geoclaw_revisions() -> None:
    root = Path(__file__).resolve().parents[1]
    geoclaw = _verify_completed_geoclaw(root, _config())

    assert geoclaw["case_count"] == 3
    assert geoclaw["comparison_count"] == 6
    assert set(geoclaw["external_revisions"]) >= {
        "clawpack_commit",
        "geoclaw_commit",
        "petsc_commit",
    }


def test_r6_broad_validation_checksum_manifest_is_written_after_workspace_cleanup(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".workspace"
    workspace.mkdir()
    (workspace / "transient.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")

    _finalize_output(output_root=tmp_path, workspace=workspace)

    assert not workspace.exists()
    validate_checksums(tmp_path)
    assert ".workspace/" not in (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8")
