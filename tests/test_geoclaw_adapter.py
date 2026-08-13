from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.geoclaw_adapter import (
    GeoClawEnvironment,
    _adapter_hash,
    _collect_output,
    _load_case_input,
    _parse_ascii_frame,
    _parse_ksp_health,
    _subprocess_environment,
    _task_boundary,
    _write_state_file,
    validate_geoclaw_environment,
)
from src.evaluation.established_solver_validation import (
    SCHEMA_ID_V3,
    SCHEMA_ID_V4,
    _validate_config,
)


def _write_frame(path: Path, values: np.ndarray) -> None:
    _components, nx, ny = values.shape
    lines = [
        "1 grid_number",
        "1 AMR_level",
        f"{nx} mx",
        f"{ny} my",
        "0.0 xlow",
        "0.0 ylow",
        f"{1.0 / nx:.17e} dx",
        f"{1.0 / ny:.17e} dy",
        "",
    ]
    for j in range(ny):
        for i in range(nx):
            lines.append(" ".join(f"{value:.17e}" for value in values[:, i, j]))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_time(
    path: Path, time_value: float, components: int = 4, grids: int = 1
) -> None:
    path.write_text(
        f"{time_value:.17e} time\n{components} meqn\n{grids} ngrids\n"
        "1 naux\n2 ndim\n2 nghost\nascii format\n",
        encoding="utf-8",
    )


def _patch_lines(
    values: np.ndarray,
    *,
    grid_number: int,
    xlower: float,
    ylower: float,
    dx: float,
    dy: float,
) -> list[str]:
    _components, nx, ny = values.shape
    lines = [
        f"{grid_number} grid_number",
        "1 AMR_level",
        f"{nx} mx",
        f"{ny} my",
        f"{xlower:.17e} xlow",
        f"{ylower:.17e} ylow",
        f"{dx:.17e} dx",
        f"{dy:.17e} dy",
        "",
    ]
    for j in range(ny):
        for i in range(nx):
            lines.append(" ".join(f"{value:.17e}" for value in values[:, i, j]))
        lines.append("")
    return lines


def test_external_execution_policy_is_frozen_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/eval/minimum_established_solver_validation.yaml").read_text(
            encoding="utf-8"
        )
    )
    _validate_config(config)
    changed = json.loads(json.dumps(config))
    changed["external_execution"]["cfl_desired"] = 0.8
    with pytest.raises(ValueError, match="cfl_desired changed"):
        _validate_config(changed)


def test_state_file_preserves_fortran_cell_order_and_boundary_flags(tmp_path: Path) -> None:
    bathymetry = np.asarray([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    arrays = {
        "bathymetry": bathymetry,
        "initial_depth": -bathymetry + 0.1,
        "hu0": np.zeros((2, 2), dtype=np.float64),
        "hv0": np.zeros((2, 2), dtype=np.float64),
        "domain_bounds": np.asarray([0.0, 1.0, 0.0, 1.0]),
    }
    path = tmp_path / "initial_state.dat"
    _write_state_file(path, arrays, periodic=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:4] == [
        "2 2",
        "0.00000000000000000e+00 0.00000000000000000e+00",
        "5.00000000000000000e-01 5.00000000000000000e-01",
        "1 1",
    ]
    assert [float(line.split()[0]) for line in lines[4:]] == [-1.0, -3.0, -2.0, -4.0]


def test_case_loader_separates_nominal_float32_eta_from_natural_state(
    tmp_path: Path,
) -> None:
    shape = (2, 2)
    bathymetry = -np.ones(shape, dtype=np.float64)
    depth = np.ones(shape, dtype=np.float64)
    nominal_eta = np.full(shape, 2.0e-7, dtype=np.float64)
    path = tmp_path / "input.npz"
    np.savez_compressed(
        path,
        bathymetry=bathymetry,
        eta0=nominal_eta,
        initial_depth=depth,
        hu0=np.zeros(shape, dtype=np.float64),
        hv0=np.zeros(shape, dtype=np.float64),
        requested_times=np.asarray([0.1], dtype=np.float64),
        case_hash=np.asarray("case"),
        output_crop=np.asarray([0, 2, 0, 2], dtype=np.int64),
        domain_bounds=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
    )
    arrays = _load_case_input(path)
    np.testing.assert_array_equal(arrays["natural_eta0"], np.zeros(shape))
    assert float(arrays["nominal_eta_max_abs_difference"]) == 2.0e-7
    assert float(arrays["nominal_eta_consistency_floor"]) > 2.0e-7

    np.savez_compressed(
        path,
        bathymetry=bathymetry,
        eta0=np.full(shape, 1.0e-4, dtype=np.float64),
        initial_depth=depth,
        hu0=np.zeros(shape, dtype=np.float64),
        hv0=np.zeros(shape, dtype=np.float64),
        requested_times=np.asarray([0.1], dtype=np.float64),
        case_hash=np.asarray("case"),
        output_crop=np.asarray([0, 2, 0, 2], dtype=np.int64),
        domain_bounds=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="nominal eta is inconsistent"):
        _load_case_input(path)


def test_ascii_frame_parser_restores_x_y_orientation(tmp_path: Path) -> None:
    values = np.zeros((4, 3, 2), dtype=np.float64)
    for i in range(3):
        for j in range(2):
            values[:, i, j] = [i + 10 * j, 1.0, 2.0, 100 + i + 10 * j]
    path = tmp_path / "fort.q0000"
    _write_frame(path, values)
    header, parsed = _parse_ascii_frame(path)
    assert header["nx"] == 3
    assert header["ny"] == 2
    np.testing.assert_array_equal(parsed, values)


def test_output_collection_verifies_initial_state_times_and_crop(tmp_path: Path) -> None:
    nx, ny = 4, 2
    bathymetry = -np.ones((nx, ny), dtype=np.float64)
    eta0 = np.arange(nx * ny, dtype=np.float64).reshape(nx, ny) * 1.0e-6
    depth = eta0 - bathymetry
    arrays = {
        "bathymetry": bathymetry,
        "eta0": eta0,
        "initial_depth": depth,
        "hu0": np.zeros((nx, ny), dtype=np.float64),
        "hv0": np.zeros((nx, ny), dtype=np.float64),
        "requested_times": np.asarray([0.1, 0.2], dtype=np.float64),
        "output_crop": np.asarray([1, 3, 0, 2], dtype=np.int64),
        "domain_bounds": np.asarray([0.0, 1.0, 0.0, 1.0]),
    }
    for frame, time_value in enumerate((0.0, 0.1, 0.2)):
        values = np.zeros((4, nx, ny), dtype=np.float64)
        values[0] = depth
        values[3] = eta0 + time_value
        if frame == 0:
            values[3] = eta0
        _write_frame(tmp_path / f"fort.q{frame:04d}", values)
        _write_time(tmp_path / f"fort.t{frame:04d}", time_value)
    eta, actual_times, diagnostics = _collect_output(
        run_dir=tmp_path,
        arrays=arrays,
        requirement={"eta_shape": [2, 2, 2]},
        tolerance=5.0e-13,
    )
    np.testing.assert_array_equal(actual_times, [0.1, 0.2])
    np.testing.assert_allclose(eta[0], eta0[1:3] + 0.1, rtol=0.0, atol=1.0e-16)
    assert diagnostics["initial_state_max_abs_error"] <= 2.0e-16
    assert diagnostics["requested_time_max_abs_error"] == 0.0


def test_output_collection_stitches_unordered_level_one_patches(
    tmp_path: Path,
) -> None:
    nx, ny = 4, 2
    bathymetry = -np.ones((nx, ny), dtype=np.float64)
    eta0 = np.arange(nx * ny, dtype=np.float64).reshape(nx, ny) * 1.0e-6
    depth = eta0 - bathymetry
    arrays = {
        "bathymetry": bathymetry,
        "eta0": eta0,
        "initial_depth": depth,
        "hu0": np.zeros((nx, ny), dtype=np.float64),
        "hv0": np.zeros((nx, ny), dtype=np.float64),
        "requested_times": np.asarray([0.1], dtype=np.float64),
        "output_crop": np.asarray([0, nx, 0, ny], dtype=np.int64),
        "domain_bounds": np.asarray([0.0, 1.0, 0.0, 1.0]),
    }
    for frame, time_value in enumerate((0.0, 0.1)):
        values = np.zeros((4, nx, ny), dtype=np.float64)
        values[0] = depth
        values[3] = eta0 + time_value
        if frame == 0:
            values[3] = eta0
        lines = [
            *_patch_lines(
                values[:, 2:4, :],
                grid_number=2,
                xlower=0.5,
                ylower=0.0,
                dx=0.25,
                dy=0.5,
            ),
            *_patch_lines(
                values[:, 0:2, :],
                grid_number=1,
                xlower=0.0,
                ylower=0.0,
                dx=0.25,
                dy=0.5,
            ),
        ]
        (tmp_path / f"fort.q{frame:04d}").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        _write_time(tmp_path / f"fort.t{frame:04d}", time_value, grids=2)
    eta, actual_times, diagnostics = _collect_output(
        run_dir=tmp_path,
        arrays=arrays,
        requirement={"eta_shape": [1, nx, ny]},
        tolerance=5.0e-13,
    )
    np.testing.assert_array_equal(actual_times, [0.1])
    np.testing.assert_allclose(eta[0], eta0 + 0.1, rtol=0.0, atol=1.0e-16)
    assert diagnostics["initial_state_max_abs_error"] <= 2.0e-16


def test_adapter_hash_and_boundary_mapping_are_deterministic() -> None:
    first = _adapter_hash(
        execution={"nested": {"b": 2, "a": 1}},
        revisions={"geoclaw_commit": "abc", "petsc_commit": "def"},
    )
    second = _adapter_hash(
        execution={"nested": {"a": 1, "b": 2}},
        revisions={"petsc_commit": "def", "geoclaw_commit": "abc"},
    )
    assert first == second
    hardened_v3 = _adapter_hash(
        execution={"nested": {"a": 1, "b": 2}},
        revisions={"petsc_commit": "def", "geoclaw_commit": "abc"},
        bundle_schema_id=SCHEMA_ID_V3,
    )
    hardened_v4 = _adapter_hash(
        execution={"nested": {"a": 1, "b": 2}},
        revisions={"petsc_commit": "def", "geoclaw_commit": "abc"},
        bundle_schema_id=SCHEMA_ID_V4,
    )
    assert hardened_v4 == hardened_v3
    assert _task_boundary({"boundary": "periodic"}) == "periodic"
    assert (
        _task_boundary(
            {
                "boundary": "radiation",
                "external_domain": {"boundary": "open_extrapolation"},
            }
        )
        == "extrap"
    )


def test_v3_petsc_environment_requests_reason_and_fails_closed(
    tmp_path: Path,
) -> None:
    claw = tmp_path / "claw"
    options = claw / "geoclaw/examples/bouss/petscMPIoptions"
    options.parent.mkdir(parents=True)
    options.write_text("-ksp_type gmres\n", encoding="utf-8")
    petsc = tmp_path / "petsc"
    environment = GeoClawEnvironment(
        claw_root=claw,
        petsc_dir=petsc,
        petsc_arch="arch",
        python_executable=Path("/usr/bin/python"),
    )
    legacy = _subprocess_environment(environment)
    strict = _subprocess_environment(environment, fail_closed_ksp=True)
    assert "-ksp_converged_reason" not in legacy["PETSC_OPTIONS"]
    assert "-ksp_converged_reason" in strict["PETSC_OPTIONS"]
    assert "-ksp_error_if_not_converged" in strict["PETSC_OPTIONS"]
    assert "-on_error_abort" in strict["PETSC_OPTIONS"]


def test_environment_canonicalizes_python_before_subprocess_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    python = repo_root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    target = tmp_path / "system-python"
    target.write_text("", encoding="utf-8")
    python.symlink_to(target)
    monkeypatch.chdir(repo_root)
    environment = GeoClawEnvironment(
        claw_root=Path("claw"),
        petsc_dir=Path("petsc"),
        petsc_arch="arch",
        python_executable=Path(".venv/bin/python"),
    )
    monkeypatch.chdir(tmp_path)
    assert environment.python_executable == python
    assert environment.python_executable != python.resolve()
    assert environment.python_executable.is_file()


def test_environment_validation_rejects_python_without_required_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claw = tmp_path / "claw"
    petsc = tmp_path / "petsc"
    for path in (
        claw / "geoclaw/src/2d/shallow/Makefile.geoclaw",
        claw / "geoclaw/src/2d/bouss/Makefile.bouss",
        claw / "clawutil/src/Makefile.common",
        petsc / "arch/lib/libpetsc.so",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    options = claw / "geoclaw/examples/bouss/petscMPIoptions"
    options.parent.mkdir(parents=True, exist_ok=True)
    options.write_text(
        "-ksp_type gmres\n-ksp_max_it 200\n-ksp_rtol 1.e-9\n",
        encoding="utf-8",
    )
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "src.evaluation.geoclaw_adapter.shutil.which",
        lambda _name: "/bin/true",
    )
    monkeypatch.setattr(
        "src.evaluation.geoclaw_adapter.subprocess.run",
        lambda *args, **kwargs: __import__("subprocess").CompletedProcess(
            args[0],
            1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'numpy'",
        ),
    )
    environment = GeoClawEnvironment(
        claw_root=claw,
        petsc_dir=petsc,
        petsc_arch="arch",
        python_executable=python,
    )
    with pytest.raises(RuntimeError, match="cannot import NumPy"):
        validate_geoclaw_environment(environment)


def test_ksp_health_parser_accepts_only_explicit_convergence(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "Linear solve converged due to CONVERGED_RTOL iterations 7\n"
        "Linear solve converged due to CONVERGED_ATOL iterations 3\n",
        encoding="utf-8",
    )
    health = _parse_ksp_health(run_log)
    assert health["ksp_solve_count"] == 2
    assert health["ksp_iteration_max"] == 7
    assert health["ksp_iteration_mean"] == 5.0
    assert health["ksp_convergence_reasons"] == [
        "CONVERGED_ATOL",
        "CONVERGED_RTOL",
    ]

    run_log.write_text(
        "Linear solve did not converge due to DIVERGED_ITS iterations 200\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        _parse_ksp_health(run_log)

    run_log.write_text("PETSc was called 10 times\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no PETSc convergence records"):
        _parse_ksp_health(run_log)
