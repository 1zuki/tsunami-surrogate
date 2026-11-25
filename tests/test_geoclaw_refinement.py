import numpy as np
import pytest

from scripts.prepare_geoclaw_refinement import _refine_arrays, prepare


def test_refinement_preserves_cell_averages_domain_and_requested_times():
    base = np.arange(16, dtype=np.float64).reshape(4, 4)
    arrays = {
        key: base.copy()
        for key in ("bathymetry", "eta0", "initial_depth", "hu0", "hv0")
    }
    arrays.update(
        output_crop=np.array([1, 3, 1, 3]),
        domain_bounds=np.array([0, 8, 0, 8]),
        requested_times=np.array([8.4, 420.0]),
    )
    result = _refine_arrays(arrays, 2)
    for key in ("bathymetry", "eta0", "initial_depth", "hu0", "hv0"):
        np.testing.assert_array_equal(
            result[key].reshape(4, 2, 4, 2).mean(axis=(1, 3)), base
        )
    np.testing.assert_array_equal(result["output_crop"], [2, 6, 2, 6])
    np.testing.assert_array_equal(result["domain_bounds"], arrays["domain_bounds"])
    np.testing.assert_array_equal(result["requested_times"], arrays["requested_times"])
    np.testing.assert_array_equal(arrays["output_crop"], [1, 3, 1, 3])


def test_refinement_rejects_noninteger_factor():
    with pytest.raises(ValueError):
        _refine_arrays({}, 1.5)


def test_refined_bundle_metadata_matches_execution_arrays(tmp_path):
    import json

    from tsunami_surrogate.data_gen.common_time import stable_hash_payload
    from tsunami_surrogate.evaluation.established_solver_validation import (
        _write_checksums,
    )

    parent = tmp_path / "parent"
    case_dir = parent / "cases" / "one"
    case_dir.mkdir(parents=True)
    arrays = {
        key: np.ones((4, 4))
        for key in ("bathymetry", "eta0", "initial_depth", "hu0", "hv0")
    }
    arrays.update(
        output_crop=np.array([1, 3, 1, 3]),
        domain_bounds=np.array([0, 8, 0, 8]),
        requested_times=np.array([8.4, 420.0]),
        case_hash=np.asarray("old"),
    )
    np.savez_compressed(case_dir / "input.npz", **arrays)
    np.savez_compressed(
        case_dir / "inhouse_swe_hydrostatic.npz",
        eta=np.zeros((2, 1, 1)),
        times=arrays["requested_times"],
        case_hash=np.asarray("old"),
    )
    frozen = {
        "artifact_kind": "corrected-production-geoclaw-swe-frozen-contract",
        "schema_id": "tsunami-surrogate.corrected-production-geoclaw-swe.v1",
        "bundle_hash": "parent",
        "source_config": {"inhouse_references": ["swe_hydrostatic"]},
        "cases": [
            {
                "case_id": "one",
                "case_hash": "old",
                "publication_shape": [1, 1],
                "external_domain": {
                    "shape": [4, 4],
                    "output_crop": [1, 3, 1, 3],
                    "publication_reduction": [2, 2],
                },
            }
        ],
        "external_results": [{"case_id": "one"}],
    }
    (parent / "frozen_contract.json").write_text(json.dumps(frozen))
    _write_checksums(parent)
    output = prepare(parent, tmp_path / "refined", 2)
    metadata = json.loads((output / "frozen_contract.json").read_text())
    case = metadata["cases"][0]
    with np.load(output / "cases/one/input.npz", allow_pickle=False) as payload:
        assert (
            list(payload["bathymetry"].shape)
            == case["external_domain"]["shape"]
            == [8, 8]
        )
        np.testing.assert_array_equal(
            payload["output_crop"], case["external_domain"]["output_crop"]
        )
        assert payload["case_hash"].item() == case["case_hash"]
    preimage = {key: value for key, value in case.items() if key != "case_hash"}
    assert (
        stable_hash_payload(
            artifact_kind="nested-geoclaw-case",
            payload=preimage,
            schema_id=metadata["schema_id"],
        )
        == case["case_hash"]
    )
    requirement = metadata["external_results"][0]
    assert requirement["case_hash"] == case["case_hash"]
    assert requirement["publication_reduction"] == [4, 4]
    with pytest.raises(FileExistsError):
        prepare(parent, output, 2)
