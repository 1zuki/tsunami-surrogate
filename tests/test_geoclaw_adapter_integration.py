from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.geoclaw_adapter import GeoClawEnvironment, run_geoclaw_bundle
from src.evaluation.established_solver_validation import (
    EXTERNAL_RESULT_SCHEMA_ID_V3,
    SCHEMA_ID_V3,
    _write_checksums,
    established_solver_status,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GEOCLAW_INTEGRATION") != "1",
    reason="set RUN_GEOCLAW_INTEGRATION=1 for the installed GeoClaw/PETSc smoke test",
)


def test_installed_geoclaw_swe_and_sgn_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            repo
            / "configs/eval/minimum_established_solver_validation_v3.yaml"
        ).read_text(encoding="utf-8")
    )
    nx, ny = 16, 4
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    eta0 = 1.0e-5 * np.exp(-((x - 0.35) / 0.08) ** 2)[:, None]
    eta0 = np.broadcast_to(eta0, (nx, ny)).copy()
    bathymetry = -np.ones((nx, ny), dtype=np.float64)
    depth = eta0 - bathymetry
    times = np.asarray([0.0035, 0.007], dtype=np.float64)
    case_hash = "integration-case-hash"
    case_id = "integration_periodic_nx16_ny4"
    bundle = tmp_path / "bundle"
    case_dir = bundle / "cases" / case_id
    case_dir.mkdir(parents=True)
    np.savez_compressed(
        case_dir / "input.npz",
        bathymetry=bathymetry,
        eta0=eta0,
        initial_depth=depth,
        hu0=np.zeros_like(depth),
        hv0=np.zeros_like(depth),
        eta_t0=np.zeros_like(depth),
        requested_times=times,
        gauge_indices=np.asarray([[4, 2], [8, 2]], dtype=np.int64),
        case_hash=np.asarray(case_hash),
        output_crop=np.asarray([0, nx, 0, ny], dtype=np.int64),
        domain_bounds=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
    )
    requirements = []
    for comparator_id in ("geoclaw_swe", "geoclaw_sgn"):
        required_keys = [
            "schema_id",
            "case_hash",
            "comparator_id",
            "comparator_version",
            "comparator_commit",
            "clawpack_commit",
            "petsc_commit",
            "adapter_hash",
            "times",
            "actual_times",
            "eta",
            "runtime_seconds",
            "initial_state_max_abs_error",
            "requested_time_max_abs_error",
            "nominal_eta_max_abs_difference",
            "nominal_eta_consistency_floor",
            "solver_health_status",
        ]
        if comparator_id == "geoclaw_sgn":
            required_keys.extend(
                [
                    "ksp_solve_count",
                    "ksp_iteration_max",
                    "ksp_iteration_mean",
                    "ksp_convergence_reasons",
                ]
            )
        requirements.append(
            {
                "case_id": case_id,
                "case_hash": case_hash,
                "comparator_id": comparator_id,
                "comparator_version": "5.14.0",
                "result_schema_id": EXTERNAL_RESULT_SCHEMA_ID_V3,
                "relative_path": f"{case_id}/{comparator_id}.npz",
                "required_npz_keys": required_keys,
                "eta_shape": [times.size, nx, ny],
                "computational_shape": [nx, ny],
                "output_crop": [0, nx, 0, ny],
                "computational_domain_bounds": [0.0, 1.0, 0.0, 1.0],
            }
        )
    frozen = {
        "schema_id": SCHEMA_ID_V3,
        "bundle_hash": "integration-bundle",
        "source_config": config,
        "requested_times": times.tolist(),
        "cases": [
            {
                "case_id": case_id,
                "case_hash": case_hash,
                "boundary": "periodic",
                "nx": nx,
                "ny": ny,
            }
        ],
        "external_results": requirements,
    }
    (bundle / "frozen_contract.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_checksums(bundle)

    external = tmp_path / "external"
    summary = run_geoclaw_bundle(
        bundle_root=bundle,
        external_root=external,
        environment=GeoClawEnvironment(
            claw_root=Path(os.environ["CLAW"]),
            petsc_dir=Path(os.environ["PETSC_DIR"]),
            petsc_arch=os.environ["PETSC_ARCH"],
            python_executable=Path(os.environ["GEOCLAW_PYTHON"]),
        ),
        workers=1,
        progress=print,
    )
    assert summary["executed"] == 2
    status = established_solver_status(bundle_root=bundle, external_root=external)
    assert status["complete"] is True
    assert status["valid"] == 2
    for comparator_id in ("geoclaw_swe", "geoclaw_sgn"):
        with np.load(external / case_id / f"{comparator_id}.npz") as payload:
            assert float(payload["initial_state_max_abs_error"]) <= 5.0e-13
            np.testing.assert_array_equal(payload["times"], times)
            assert np.isfinite(payload["eta"]).all()
            assert str(payload["solver_health_status"]) == "passed"
            if comparator_id == "geoclaw_sgn":
                assert int(payload["ksp_solve_count"]) > 0
                assert all(
                    str(reason).startswith("CONVERGED_")
                    for reason in payload["ksp_convergence_reasons"]
                )
