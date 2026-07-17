from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data_gen.common_time_v2 import sha256_file, stable_hash_payload
from src.evaluation.common_time_v2_level_a import validate_checksums
from src.evaluation.established_solver_validation import (
    EXTERNAL_RESULT_SCHEMA_ID,
    SCHEMA_ID,
    _load_external_result,
    _read_json,
)


ADAPTER_SCHEMA_ID = "tsunami-surrogate.geoclaw-external-adapter.v1"


@dataclass(frozen=True)
class GeoClawEnvironment:
    claw_root: Path
    petsc_dir: Path
    petsc_arch: str
    python_executable: Path
    mpi_executable: str = "mpiexec"
    mpi_fortran_compiler: str = "mpif90"
    fortran_compiler: str = "gfortran"


FROZEN_STATE_MODULE = r"""
module frozen_state_module
    implicit none
    logical, save :: loaded = .false.
    integer, save :: state_nx, state_ny
    integer, save :: periodic_x, periodic_y
    real(kind=8), save :: state_xlower, state_ylower, state_dx, state_dy
    real(kind=8), allocatable, save :: state_b(:,:), state_h(:,:)
    real(kind=8), allocatable, save :: state_hu(:,:), state_hv(:,:)
contains
    subroutine load_frozen_state()
        integer :: unit, i, j, ios
        if (loaded) return
        open(newunit=unit, file='initial_state.dat', status='old', &
             action='read', form='formatted', iostat=ios)
        if (ios /= 0) error stop 'cannot open initial_state.dat'
        read(unit, *, iostat=ios) state_nx, state_ny
        if (ios /= 0 .or. state_nx <= 0 .or. state_ny <= 0) &
            error stop 'invalid initial-state shape'
        read(unit, *, iostat=ios) state_xlower, state_ylower
        if (ios /= 0) error stop 'invalid initial-state origin'
        read(unit, *, iostat=ios) state_dx, state_dy
        if (ios /= 0 .or. state_dx <= 0.d0 .or. state_dy <= 0.d0) &
            error stop 'invalid initial-state spacing'
        read(unit, *, iostat=ios) periodic_x, periodic_y
        if (ios /= 0) error stop 'invalid initial-state boundary flags'
        allocate(state_b(state_nx,state_ny), state_h(state_nx,state_ny))
        allocate(state_hu(state_nx,state_ny), state_hv(state_nx,state_ny))
        do j = 1, state_ny
            do i = 1, state_nx
                read(unit, *, iostat=ios) state_b(i,j), state_h(i,j), &
                                             state_hu(i,j), state_hv(i,j)
                if (ios /= 0) error stop 'truncated initial-state payload'
            end do
        end do
        close(unit)
        loaded = .true.
    end subroutine load_frozen_state

    subroutine lookup_frozen_state(x, y, b, h, hu, hv)
        real(kind=8), intent(in) :: x, y
        real(kind=8), intent(out) :: b, h, hu, hv
        integer :: i, j
        call load_frozen_state()
        i = floor((x - state_xlower) / state_dx) + 1
        j = floor((y - state_ylower) / state_dy) + 1
        if (periodic_x == 1) then
            i = modulo(i - 1, state_nx) + 1
        else
            i = min(max(i, 1), state_nx)
        end if
        if (periodic_y == 1) then
            j = modulo(j - 1, state_ny) + 1
        else
            j = min(max(j, 1), state_ny)
        end if
        b = state_b(i,j)
        h = state_h(i,j)
        hu = state_hu(i,j)
        hv = state_hv(i,j)
    end subroutine lookup_frozen_state
end module frozen_state_module
""".lstrip()


CUSTOM_SETAUX = r"""
subroutine setaux(mbc,mx,my,xlow,ylow,dx,dy,maux,aux)
    use frozen_state_module, only: load_frozen_state, lookup_frozen_state, &
                                   state_dx, state_dy
    implicit none
    integer, intent(in) :: mbc, mx, my, maux
    real(kind=8), intent(in) :: xlow, ylow, dx, dy
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
    integer :: i, j
    real(kind=8) :: x, y, b, h, hu, hv, scale
    call load_frozen_state()
    scale = max(1.d0, abs(state_dx), abs(state_dy))
    if (abs(dx-state_dx) > 1.d-12*scale .or. &
        abs(dy-state_dy) > 1.d-12*scale) &
        error stop 'GeoClaw patch spacing differs from frozen state'
    aux = 0.d0
    do j = 1-mbc, my+mbc
        y = ylow + (dble(j)-0.5d0)*dy
        do i = 1-mbc, mx+mbc
            x = xlow + (dble(i)-0.5d0)*dx
            call lookup_frozen_state(x,y,b,h,hu,hv)
            aux(1,i,j) = b
        end do
    end do
end subroutine setaux
""".lstrip()


CUSTOM_QINIT = r"""
subroutine qinit(meqn,mbc,mx,my,xlower,ylower,dx,dy,q,maux,aux)
    use frozen_state_module, only: lookup_frozen_state
    implicit none
    integer, intent(in) :: meqn, mbc, mx, my, maux
    real(kind=8), intent(in) :: xlower, ylower, dx, dy
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
    integer :: i, j
    real(kind=8) :: x, y, b, h, hu, hv
    q = 0.d0
    do j = 1, my
        y = ylower + (dble(j)-0.5d0)*dy
        do i = 1, mx
            x = xlower + (dble(i)-0.5d0)*dx
            call lookup_frozen_state(x,y,b,h,hu,hv)
            if (h < 0.d0) error stop 'negative frozen initial depth'
            q(1,i,j) = h
            q(2,i,j) = hu
            q(3,i,j) = hv
        end do
    end do
end subroutine qinit
""".lstrip()


CUSTOM_TOPO_UPDATE = r"""
subroutine topo_update(t)
    implicit none
    real(kind=8), intent(in) :: t
    ! The frozen comparator bathymetry is loaded directly by setaux and is
    ! strictly static.  GeoClaw's default routine expects at least one topo or
    ! dtopo file, so it must not inspect its unallocated topo-file arrays here.
end subroutine topo_update
""".lstrip()


SETRUN_SCRIPT = r'''from __future__ import annotations

import json
import logging.config
from pathlib import Path

# PyClaw's package import eagerly constructs an unused UDP syslog handler.
# The adapter does not use PyClaw logging, and restricted execution environments
# may forbid socket creation even though no log record is ever sent.
logging.config.fileConfig = lambda *args, **kwargs: None

from clawpack.clawutil import data


def main() -> None:
    spec = json.loads(Path("run_spec.json").read_text(encoding="utf-8"))
    execution = spec["execution"]
    clawdata_values = spec["clawdata"]
    rundata = data.ClawRunData("geoclaw", 2)
    clawdata = rundata.clawdata
    clawdata.lower = list(clawdata_values["lower"])
    clawdata.upper = list(clawdata_values["upper"])
    clawdata.num_cells = list(clawdata_values["num_cells"])
    clawdata.num_eqn = 5 if spec["comparator_id"] == "geoclaw_sgn" else 3
    clawdata.num_aux = 1
    clawdata.capa_index = 0
    clawdata.t0 = 0.0
    clawdata.restart = False
    clawdata.output_style = 2
    requested_times = list(clawdata_values["requested_times"])
    clawdata.output_times = [0.0, *requested_times]
    clawdata.output_t0 = bool(execution["output_t0_for_initial_state_verification"])
    clawdata.output_format = execution["output_format"]
    clawdata.output_q_components = "all"
    clawdata.output_aux_components = "none"
    clawdata.output_aux_onlyonce = False
    clawdata.verbosity = 0
    clawdata.dt_variable = True
    clawdata.dt_initial = float(execution["dt_initial"])
    clawdata.dt_max = float(execution["dt_max"])
    clawdata.cfl_desired = float(execution["cfl_desired"])
    clawdata.cfl_max = float(execution["cfl_max"])
    clawdata.steps_max = int(execution["steps_max"])
    clawdata.order = int(execution["spatial_order"])
    clawdata.dimensional_split = execution["dimensional_split"]
    clawdata.transverse_waves = int(execution["transverse_waves"])
    clawdata.num_waves = 3
    clawdata.limiter = [execution["limiter"]] * 3
    clawdata.use_fwaves = bool(execution["use_fwaves"])
    clawdata.source_split = execution["source_split"]
    clawdata.num_ghost = int(execution["num_ghost"])
    boundary = clawdata_values["boundary"]
    clawdata.bc_lower = [boundary, boundary]
    clawdata.bc_upper = [boundary, boundary]
    clawdata.checkpt_style = 0

    rundata.gaugedata.gauges = []
    rundata.regiondata.regions = []
    rundata.fgmax_data.fgmax_grids = []
    rundata.fgout_data.fgout_grids = []

    amrdata = rundata.amrdata
    amrdata.amr_levels_max = int(execution["amr_levels"])
    amrdata.aux_type = ["center"]
    amrdata.flag_richardson = False
    amrdata.flag2refine = False

    geo = rundata.geo_data
    geo.gravity = float(spec["gravity"])
    geo.coordinate_system = int(execution["coordinate_system"])
    geo.coriolis_forcing = bool(execution["coriolis_forcing"])
    geo.sea_level = float(execution["sea_level"])
    geo.dry_tolerance = float(execution["dry_tolerance"])
    geo.friction_forcing = bool(execution["friction_forcing"])
    geo.manning_coefficient = 0.0
    rundata.topo_data.topofiles = []
    rundata.dtopo_data.dtopofiles = []
    rundata.qinit_data.qinit_type = 0
    rundata.qinit_data.qinitfiles = []

    if spec["comparator_id"] == "geoclaw_sgn":
        from clawpack.geoclaw.data import BoussData
        rundata.add_data(BoussData(), "bouss_data")
        sgn = execution["sgn"]
        rundata.bouss_data.bouss_equations = int(sgn["bouss_equations"])
        rundata.bouss_data.bouss_min_level = int(sgn["bouss_min_level"])
        rundata.bouss_data.bouss_max_level = int(sgn["bouss_max_level"])
        rundata.bouss_data.bouss_min_depth = float(sgn["bouss_min_depth"])
        rundata.bouss_data.bouss_solver = int(sgn["bouss_solver"])
        rundata.bouss_data.bouss_tstart = float(sgn["bouss_tstart"])

    rundata.write()


if __name__ == "__main__":
    main()
'''


def _makefile(comparator_id: str) -> str:
    common = """CLAWMAKE = $(CLAW)/clawutil/src/Makefile.common
CLAW_PKG = geoclaw
SETRUN_FILE = setrun.py
OUTDIR = _output
GEOLIB = $(CLAW)/geoclaw/src/2d/shallow
EXCLUDE_SOURCES = $(GEOLIB)/qinit.f90 $(GEOLIB)/setaux.f90 $(GEOLIB)/topo_update.f90
MODULES = ./frozen_state_module.f90
SOURCES = ./qinit.f90 ./setaux.f90 ./topo_update.f90
"""
    if comparator_id == "geoclaw_swe":
        return (
            common
            + """EXE = $(CURDIR)/xgeoclaw_swe
include $(GEOLIB)/Makefile.geoclaw
SOURCES += $(CLAW)/riemann/src/rpn2_geoclaw.f \\
  $(CLAW)/riemann/src/rpt2_geoclaw.f \\
  $(CLAW)/riemann/src/geoclaw_riemann_utils.f
include $(CLAWMAKE)
"""
        )
    if comparator_id != "geoclaw_sgn":
        raise ValueError(f"Unsupported GeoClaw comparator: {comparator_id}")
    return (
        common
        + """EXE = $(CURDIR)/xgeoclaw_sgn
BOUSSLIB = $(CLAW)/geoclaw/src/2d/bouss
AMRLIB = $(CLAW)/amrclaw/src/2d
PETSC_INCLUDE = $(PETSC_DIR)/include $(PETSC_DIR)/$(PETSC_ARCH)/include
INCLUDE += $(BOUSSLIB) $(PETSC_INCLUDE)
PETSC_LFLAGS = $(shell PKG_CONFIG_PATH=$(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig pkg-config --libs-only-L --libs-only-l PETSc)
FFLAGS ?= -O -gno-strict-dwarf -fbounds-check -fopenmp -std=legacy -ffpe-trap='invalid,overflow,zero'
FFLAGS += -DHAVE_PETSC -ffree-line-length-none
LFLAGS += $(PETSC_LFLAGS) -fopenmp
include $(BOUSSLIB)/Makefile.bouss
include $(CLAWMAKE)
"""
    )


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Cannot determine Git revision for {path}")
    return result.stdout.strip()


def validate_geoclaw_environment(environment: GeoClawEnvironment) -> dict[str, str]:
    claw_root = environment.claw_root.resolve()
    petsc_dir = environment.petsc_dir.resolve()
    required = [
        claw_root / "geoclaw/src/2d/shallow/Makefile.geoclaw",
        claw_root / "geoclaw/src/2d/bouss/Makefile.bouss",
        claw_root / "clawutil/src/Makefile.common",
        petsc_dir / environment.petsc_arch / "lib/libpetsc.so",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete GeoClaw/PETSc environment: {missing}")
    for executable in (
        environment.python_executable,
        Path(shutil.which(environment.fortran_compiler) or ""),
        Path(shutil.which(environment.mpi_fortran_compiler) or ""),
        Path(shutil.which(environment.mpi_executable) or ""),
    ):
        if not str(executable) or not executable.is_file():
            raise RuntimeError(f"Missing required executable: {executable}")
    petsc_options = claw_root / "geoclaw/examples/bouss/petscMPIoptions"
    options_text = petsc_options.read_text(encoding="utf-8")
    for required_option in ("-ksp_type gmres", "-ksp_max_it 200", "-ksp_rtol 1.e-9"):
        if required_option not in options_text:
            raise RuntimeError(f"PETSc options missing {required_option!r}")
    return {
        "clawpack_commit": _git_commit(claw_root),
        "geoclaw_commit": _git_commit(claw_root / "geoclaw"),
        "petsc_commit": _git_commit(petsc_dir),
        "petsc_options_sha256": sha256_file(petsc_options),
    }


def _subprocess_environment(environment: GeoClawEnvironment) -> dict[str, str]:
    env = dict(os.environ)
    claw_root = str(environment.claw_root.resolve())
    petsc_dir = str(environment.petsc_dir.resolve())
    petsc_lib = str(Path(petsc_dir) / environment.petsc_arch / "lib")
    env.update(
        {
            "CLAW": claw_root,
            "CLAW_PYTHON": str(environment.python_executable.resolve()),
            "PYTHONPATH": claw_root + os.pathsep + env.get("PYTHONPATH", ""),
            "PETSC_DIR": petsc_dir,
            "PETSC_ARCH": environment.petsc_arch,
            "PETSC_OPTIONS": (
                "-options_file "
                + str(Path(claw_root) / "geoclaw/examples/bouss/petscMPIoptions")
            ),
            "PKG_CONFIG_PATH": (
                str(Path(petsc_lib) / "pkgconfig")
                + os.pathsep
                + env.get("PKG_CONFIG_PATH", "")
            ),
            "LD_LIBRARY_PATH": petsc_lib
            + os.pathsep
            + env.get("LD_LIBRARY_PATH", ""),
            "CLAW_MPIEXEC": environment.mpi_executable,
            "CLAW_MPIFC": environment.mpi_fortran_compiler,
            "FC": environment.fortran_compiler,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _adapter_hash(
    *, execution: Mapping[str, Any], revisions: Mapping[str, str]
) -> str:
    return stable_hash_payload(
        artifact_kind="geoclaw-external-adapter",
        payload={
            "schema_id": ADAPTER_SCHEMA_ID,
            "execution": dict(execution),
            "revisions": dict(revisions),
            "templates": {
                "state_module": FROZEN_STATE_MODULE,
                "setaux": CUSTOM_SETAUX,
                "qinit": CUSTOM_QINIT,
                "topo_update": CUSTOM_TOPO_UPDATE,
                "setrun": SETRUN_SCRIPT,
                "swe_makefile": _makefile("geoclaw_swe"),
                "sgn_makefile": _makefile("geoclaw_sgn"),
            },
        },
        schema_id=ADAPTER_SCHEMA_ID,
    )


def _write_build_sources(build_dir: Path, comparator_id: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "frozen_state_module.f90").write_text(
        FROZEN_STATE_MODULE, encoding="utf-8"
    )
    (build_dir / "setaux.f90").write_text(CUSTOM_SETAUX, encoding="utf-8")
    (build_dir / "qinit.f90").write_text(CUSTOM_QINIT, encoding="utf-8")
    (build_dir / "topo_update.f90").write_text(
        CUSTOM_TOPO_UPDATE, encoding="utf-8"
    )
    (build_dir / "Makefile").write_text(_makefile(comparator_id), encoding="utf-8")
    (build_dir / "setrun.py").write_text("# build-only placeholder\n", encoding="utf-8")


def _build_executable(
    *,
    build_root: Path,
    comparator_id: str,
    adapter_hash: str,
    revisions: Mapping[str, str],
    environment: GeoClawEnvironment,
    progress: Callable[[str], None] | None,
) -> Path:
    build_dir = build_root / comparator_id
    executable = build_dir / (
        "xgeoclaw_sgn" if comparator_id == "geoclaw_sgn" else "xgeoclaw_swe"
    )
    manifest_path = build_dir / "BUILD_MANIFEST.json"
    expected = {
        "schema_id": ADAPTER_SCHEMA_ID,
        "adapter_hash": adapter_hash,
        "comparator_id": comparator_id,
        "revisions": dict(revisions),
    }
    if executable.is_file() and manifest_path.is_file():
        if _read_json(manifest_path) == expected:
            if progress is not None:
                progress(f"[geoclaw-build] reuse {comparator_id}")
            return executable
    if build_dir.exists():
        raise RuntimeError(
            f"Stale GeoClaw build directory requires manual archival/removal: {build_dir}"
        )
    _write_build_sources(build_dir, comparator_id)
    if progress is not None:
        progress(f"[geoclaw-build] compile {comparator_id}")
    log_path = build_dir / "build.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            ["make", "new"],
            cwd=build_dir,
            env=_subprocess_environment(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not executable.is_file():
        raise RuntimeError(f"GeoClaw build failed; inspect {log_path}")
    manifest_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return executable


def _load_case_input(path: Path) -> dict[str, np.ndarray]:
    required = {
        "bathymetry",
        "eta0",
        "initial_depth",
        "hu0",
        "hv0",
        "requested_times",
        "case_hash",
        "output_crop",
        "domain_bounds",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"Frozen GeoClaw input missing keys: {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required}
    shape = arrays["bathymetry"].shape
    for key in ("eta0", "initial_depth", "hu0", "hv0"):
        if arrays[key].shape != shape:
            raise RuntimeError(f"Frozen GeoClaw input {key} shape mismatch")
    if len(shape) != 2 or not all(int(value) > 0 for value in shape):
        raise RuntimeError("Frozen GeoClaw input must be a nonempty 2-D grid")
    expected_eta = arrays["initial_depth"] + arrays["bathymetry"]
    if not np.allclose(expected_eta, arrays["eta0"], rtol=0.0, atol=5.0e-15):
        raise RuntimeError("Frozen GeoClaw depth/bathymetry/eta identity mismatch")
    if not all(np.isfinite(arrays[key]).all() for key in required - {"case_hash"}):
        raise RuntimeError("Frozen GeoClaw input contains nonfinite values")
    return arrays


def _write_state_file(
    path: Path, arrays: Mapping[str, np.ndarray], *, periodic: bool
) -> None:
    bathymetry = np.asarray(arrays["bathymetry"], dtype=np.float64)
    depth = np.asarray(arrays["initial_depth"], dtype=np.float64)
    hu = np.asarray(arrays["hu0"], dtype=np.float64)
    hv = np.asarray(arrays["hv0"], dtype=np.float64)
    bounds = np.asarray(arrays["domain_bounds"], dtype=np.float64)
    nx, ny = bathymetry.shape
    dx = (float(bounds[1]) - float(bounds[0])) / nx
    dy = (float(bounds[3]) - float(bounds[2])) / ny
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{nx} {ny}\n")
        handle.write(f"{bounds[0]:.17e} {bounds[2]:.17e}\n")
        handle.write(f"{dx:.17e} {dy:.17e}\n")
        flag = 1 if periodic else 0
        handle.write(f"{flag} {flag}\n")
        for j in range(ny):
            for i in range(nx):
                handle.write(
                    f"{bathymetry[i,j]:.17e} {depth[i,j]:.17e} "
                    f"{hu[i,j]:.17e} {hv[i,j]:.17e}\n"
                )


def _parse_ascii_frame(path: Path) -> tuple[dict[str, float | int], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) < 9:
        raise RuntimeError(f"Truncated GeoClaw frame: {path}")
    header_values = [nonempty[index].split()[0] for index in range(8)]
    header: dict[str, float | int] = {
        "grid_number": int(header_values[0]),
        "level": int(header_values[1]),
        "nx": int(header_values[2]),
        "ny": int(header_values[3]),
        "xlower": float(header_values[4]),
        "ylower": float(header_values[5]),
        "dx": float(header_values[6]),
        "dy": float(header_values[7]),
    }
    nx = int(header["nx"])
    ny = int(header["ny"])
    data_lines = nonempty[8:]
    if len(data_lines) != nx * ny:
        raise RuntimeError(
            f"GeoClaw frame cell count mismatch: {len(data_lines)} != {nx * ny}"
        )
    first = data_lines[0].split()
    component_count = len(first)
    if component_count not in (4, 6):
        raise RuntimeError(f"Unexpected GeoClaw output component count: {component_count}")
    values = np.empty((component_count, nx, ny), dtype=np.float64)
    cursor = 0
    for j in range(ny):
        for i in range(nx):
            tokens = data_lines[cursor].split()
            cursor += 1
            if len(tokens) != component_count:
                raise RuntimeError(f"Inconsistent GeoClaw component count in {path}")
            values[:, i, j] = [float(token) for token in tokens]
    if not np.isfinite(values).all():
        raise RuntimeError(f"Nonfinite GeoClaw frame: {path}")
    return header, values


def _read_fort_time(path: Path) -> tuple[float, int, int]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"Truncated GeoClaw time header: {path}")
    return float(lines[0].split()[0]), int(lines[1].split()[0]), int(lines[2].split()[0])


def _collect_output(
    *,
    run_dir: Path,
    arrays: Mapping[str, np.ndarray],
    requirement: Mapping[str, Any],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    requested_times = np.asarray(arrays["requested_times"], dtype=np.float64)
    frames: list[tuple[float, Path, int]] = []
    for time_path in sorted(run_dir.glob("fort.t[0-9][0-9][0-9][0-9]")):
        time_value, component_count, grid_count = _read_fort_time(time_path)
        if grid_count != 1:
            raise RuntimeError("GeoClaw adapter requires exactly one output grid")
        frame_number = int(time_path.name[-4:])
        frames.append((time_value, run_dir / f"fort.q{frame_number:04d}", component_count))
    if len(frames) != requested_times.size + 1:
        raise RuntimeError(
            f"GeoClaw output count mismatch: {len(frames)} != {requested_times.size + 1}"
        )
    frames.sort(key=lambda row: row[0])
    initial_time, initial_path, _ = frames[0]
    if abs(initial_time) > tolerance:
        raise RuntimeError("GeoClaw initial verification frame is not at t=0")
    bounds = np.asarray(arrays["domain_bounds"], dtype=np.float64)
    nx, ny = np.asarray(arrays["bathymetry"]).shape
    dx = (float(bounds[1]) - float(bounds[0])) / nx
    dy = (float(bounds[3]) - float(bounds[2])) / ny

    def parse_and_validate(path: Path) -> np.ndarray:
        header, values = _parse_ascii_frame(path)
        expected_header = {
            "nx": nx,
            "ny": ny,
            "xlower": float(bounds[0]),
            "ylower": float(bounds[2]),
            "dx": dx,
            "dy": dy,
        }
        for key, expected in expected_header.items():
            actual = float(header[key])
            if abs(actual - float(expected)) > 5.0e-13 * max(1.0, abs(float(expected))):
                raise RuntimeError(f"GeoClaw frame {key} mismatch: {actual} != {expected}")
        return values

    initial = parse_and_validate(initial_path)
    initial_expected = np.stack(
        [
            np.asarray(arrays["initial_depth"], dtype=np.float64),
            np.asarray(arrays["hu0"], dtype=np.float64),
            np.asarray(arrays["hv0"], dtype=np.float64),
            np.asarray(arrays["eta0"], dtype=np.float64),
        ],
        axis=0,
    )
    initial_actual = initial[[0, 1, 2, -1]]
    initial_max_error = float(np.max(np.abs(initial_actual - initial_expected)))
    if initial_max_error > tolerance:
        raise RuntimeError(
            f"GeoClaw initial-state mapping error {initial_max_error:.3e} > {tolerance:.3e}"
        )
    actual_times = np.asarray([row[0] for row in frames[1:]], dtype=np.float64)
    time_error = float(np.max(np.abs(actual_times - requested_times)))
    if time_error > 5.0e-14:
        raise RuntimeError(f"GeoClaw requested-time error too large: {time_error:.3e}")
    eta_full = np.stack(
        [parse_and_validate(path)[-1] for _time, path, _components in frames[1:]],
        axis=0,
    )
    crop = [int(value) for value in np.asarray(arrays["output_crop"]).tolist()]
    i0, i1, j0, j1 = crop
    eta = eta_full[:, i0:i1, j0:j1]
    if list(eta.shape) != [int(value) for value in requirement["eta_shape"]]:
        raise RuntimeError(f"GeoClaw cropped eta shape mismatch: {eta.shape}")
    return eta, actual_times, {
        "initial_state_max_abs_error": initial_max_error,
        "requested_time_max_abs_error": time_error,
    }


def _task_boundary(case: Mapping[str, Any]) -> str:
    external = case.get("external_domain", {})
    boundary = external.get("boundary", case.get("boundary"))
    if boundary == "periodic":
        return "periodic"
    if boundary in ("open", "open_extrapolation", "radiation"):
        return "extrap"
    raise RuntimeError(f"Unsupported GeoClaw boundary mapping: {boundary!r}")


def _run_task(
    *,
    bundle_root: Path,
    external_root: Path,
    requirement: Mapping[str, Any],
    case: Mapping[str, Any],
    executable: Path,
    environment: GeoClawEnvironment,
    execution: Mapping[str, Any],
    revisions: Mapping[str, str],
    adapter_hash: str,
) -> dict[str, Any]:
    case_id = str(requirement["case_id"])
    comparator_id = str(requirement["comparator_id"])
    arrays = _load_case_input(bundle_root / "cases" / case_id / "input.npz")
    if str(np.asarray(arrays["case_hash"]).reshape(-1)[0]) != str(requirement["case_hash"]):
        raise RuntimeError(f"Frozen GeoClaw case identity mismatch: {case_id}")
    work_base = external_root / ".work" / case_id / comparator_id
    attempt = 1
    while (work_base / f"attempt-{attempt:03d}").exists():
        attempt += 1
    run_dir = work_base / f"attempt-{attempt:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    boundary = _task_boundary(case)
    _write_state_file(run_dir / "initial_state.dat", arrays, periodic=boundary == "periodic")
    (run_dir / "setrun.py").write_text(SETRUN_SCRIPT, encoding="utf-8")
    bounds = np.asarray(arrays["domain_bounds"], dtype=np.float64)
    run_spec = {
        "schema_id": ADAPTER_SCHEMA_ID,
        "bundle_hash": _read_json(bundle_root / "frozen_contract.json")["bundle_hash"],
        "case_id": case_id,
        "case_hash": str(requirement["case_hash"]),
        "comparator_id": comparator_id,
        "execution": dict(execution),
        "gravity": 9.81,
        "clawdata": {
            "lower": [float(bounds[0]), float(bounds[2])],
            "upper": [float(bounds[1]), float(bounds[3])],
            "num_cells": [int(value) for value in np.asarray(arrays["bathymetry"]).shape],
            "requested_times": np.asarray(arrays["requested_times"], dtype=np.float64).tolist(),
            "boundary": boundary,
        },
    }
    (run_dir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env = _subprocess_environment(environment)
    setup_log = run_dir / "setrun.log"
    with setup_log.open("w", encoding="utf-8") as log:
        setup = subprocess.run(
            [str(environment.python_executable), "setrun.py"],
            cwd=run_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if setup.returncode != 0:
        raise RuntimeError(f"GeoClaw setrun failed; inspect {setup_log}")
    command = [str(executable)]
    if comparator_id == "geoclaw_sgn":
        command = [
            environment.mpi_executable,
            "-n",
            str(int(execution["sgn"]["mpi_processes"])),
            str(executable),
        ]
    started = time.monotonic()
    run_log = run_dir / "run.log"
    with run_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    runtime = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"GeoClaw run failed; inspect {run_log}")
    eta, actual_times, diagnostics = _collect_output(
        run_dir=run_dir,
        arrays=arrays,
        requirement=requirement,
        tolerance=float(execution["initial_state_abs_tolerance"]),
    )
    output_path = external_root / str(requirement["relative_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_id=np.asarray(EXTERNAL_RESULT_SCHEMA_ID),
            case_hash=np.asarray(str(requirement["case_hash"])),
            comparator_id=np.asarray(comparator_id),
            comparator_version=np.asarray(str(requirement["comparator_version"])),
            comparator_commit=np.asarray(revisions["geoclaw_commit"]),
            clawpack_commit=np.asarray(revisions["clawpack_commit"]),
            petsc_commit=np.asarray(revisions["petsc_commit"]),
            adapter_hash=np.asarray(adapter_hash),
            times=np.asarray(arrays["requested_times"], dtype=np.float64),
            actual_times=actual_times,
            eta=eta,
            runtime_seconds=np.asarray(runtime, dtype=np.float64),
            initial_state_max_abs_error=np.asarray(
                diagnostics["initial_state_max_abs_error"], dtype=np.float64
            ),
            requested_time_max_abs_error=np.asarray(
                diagnostics["requested_time_max_abs_error"], dtype=np.float64
            ),
        )
    os.replace(temporary, output_path)
    _load_external_result(
        output_path,
        requirement,
        np.asarray(arrays["requested_times"], dtype=np.float64),
    )
    return {
        "case_id": case_id,
        "comparator_id": comparator_id,
        "runtime_seconds": runtime,
        **diagnostics,
        "output_path": str(output_path),
        "run_directory": str(run_dir),
    }


def run_geoclaw_bundle(
    *,
    bundle_root: Path,
    external_root: Path,
    environment: GeoClawEnvironment,
    workers: int = 1,
    resume: bool = False,
    case_ids: Sequence[str] | None = None,
    comparator_ids: Sequence[str] | None = None,
    max_tasks: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    external_root = external_root.resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    validate_checksums(bundle_root)
    frozen = _read_json(bundle_root / "frozen_contract.json")
    if frozen.get("schema_id") != SCHEMA_ID:
        raise RuntimeError("Frozen established-solver bundle schema mismatch")
    execution = frozen["source_config"].get("external_execution")
    if not isinstance(execution, Mapping):
        raise RuntimeError("Frozen bundle predates the complete external execution policy")
    revisions = validate_geoclaw_environment(environment)
    adapter_hash = _adapter_hash(execution=execution, revisions=revisions)
    selected_cases = set(case_ids or [])
    selected_comparators = set(comparator_ids or [])
    requirements = [
        requirement
        for requirement in frozen["external_results"]
        if (not selected_cases or str(requirement["case_id"]) in selected_cases)
        and (
            not selected_comparators
            or str(requirement["comparator_id"]) in selected_comparators
        )
    ]
    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        requirements = requirements[:max_tasks]
    if not requirements:
        raise RuntimeError("No GeoClaw tasks matched the requested selection")
    case_by_id = {str(case["case_id"]): case for case in frozen["cases"]}
    external_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_id": ADAPTER_SCHEMA_ID,
        "bundle_hash": frozen["bundle_hash"],
        "adapter_hash": adapter_hash,
        "revisions": revisions,
        "execution": dict(execution),
    }
    manifest_path = external_root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != run_manifest:
            raise RuntimeError("External result root belongs to a different adapter contract")
    else:
        manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    pending: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    requested_times = np.asarray(frozen["requested_times"], dtype=np.float64)
    for requirement in requirements:
        output_path = external_root / str(requirement["relative_path"])
        if output_path.is_file() and resume:
            _load_external_result(output_path, requirement, requested_times)
            skipped.append(str(requirement["relative_path"]))
        elif output_path.exists():
            raise FileExistsError(f"Refusing to overwrite external result: {output_path}")
        else:
            pending.append(requirement)
    comparator_set = {str(row["comparator_id"]) for row in pending}
    executables = {
        comparator_id: _build_executable(
            build_root=external_root / ".build",
            comparator_id=comparator_id,
            adapter_hash=adapter_hash,
            revisions=revisions,
            environment=environment,
            progress=progress,
        )
        for comparator_id in sorted(comparator_set)
    }
    results: list[dict[str, Any]] = []
    if progress is not None:
        progress(
            f"[geoclaw-run] start completed={len(skipped)} "
            f"pending={len(pending)} total={len(requirements)} workers={workers}"
        )

    def submit_one(requirement: Mapping[str, Any]) -> dict[str, Any]:
        case_id = str(requirement["case_id"])
        comparator_id = str(requirement["comparator_id"])
        return _run_task(
            bundle_root=bundle_root,
            external_root=external_root,
            requirement=requirement,
            case=case_by_id[case_id],
            executable=executables[comparator_id],
            environment=environment,
            execution=execution,
            revisions=revisions,
            adapter_hash=adapter_hash,
        )

    completed_count = len(skipped)
    if workers == 1:
        for requirement in pending:
            result = submit_one(requirement)
            results.append(result)
            completed_count += 1
            if progress is not None:
                progress(
                    f"[geoclaw-run] done {completed_count}/{len(requirements)} "
                    f"{result['case_id']} {result['comparator_id']} "
                    f"runtime={result['runtime_seconds']:.1f}s"
                )
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending) or 1)) as executor:
            future_map = {executor.submit(submit_one, row): row for row in pending}
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                completed_count += 1
                if progress is not None:
                    progress(
                        f"[geoclaw-run] done {completed_count}/{len(requirements)} "
                        f"{result['case_id']} {result['comparator_id']} "
                        f"runtime={result['runtime_seconds']:.1f}s"
                    )
    if progress is not None:
        progress(f"[geoclaw-run] complete {completed_count}/{len(requirements)}")
    return {
        "bundle_hash": frozen["bundle_hash"],
        "adapter_hash": adapter_hash,
        "selected": len(requirements),
        "executed": len(results),
        "skipped": len(skipped),
        "results": sorted(results, key=lambda row: (row["case_id"], row["comparator_id"])),
    }
