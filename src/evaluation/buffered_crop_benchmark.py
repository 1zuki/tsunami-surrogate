from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data_gen.simulate_dataset import (
    BufferedDomainConfig,
    _prepare_buffered_domain,
    _simulate_one_local,
)
from src.evaluation.finite_horizon_boundary_study import comparison_metrics
from src.solver.boussinesq import BoussinesqSolver
from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver
from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver
from src.solver.operator_time import build_sponge_mask


SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")


def cosine_core_window(shape: Sequence[int], taper_cells: int) -> np.ndarray:
    if len(shape) != 2 or min(int(value) for value in shape) <= 1:
        raise ValueError("shape must contain two valid axes")
    if taper_cells < 2 or 2 * taper_cells >= min(int(value) for value in shape):
        raise ValueError("taper_cells must leave a non-empty untapered interior")
    nx, ny = (int(value) for value in shape)
    x_distance = np.minimum(np.arange(nx), np.arange(nx)[::-1])
    y_distance = np.minimum(np.arange(ny), np.arange(ny)[::-1])
    edge_distance = np.minimum(x_distance[:, None], y_distance[None, :])
    coordinate = np.clip(
        edge_distance.astype(np.float64) / float(taper_cells - 1), 0.0, 1.0
    )
    return 0.5 * (1.0 - np.cos(np.pi * coordinate))


def prepare_buffered_case(
    bathymetry: np.ndarray,
    eta0: np.ndarray,
    *,
    buffer_cells: int,
    source_taper_cells: int,
) -> dict[str, Any]:
    bathy = np.asarray(bathymetry, dtype=np.float64)
    eta = np.asarray(eta0, dtype=np.float64)
    if bathy.ndim != 2 or bathy.shape != eta.shape:
        raise ValueError("bathymetry and eta0 must be same-shape 2D arrays")
    if buffer_cells < 0:
        raise ValueError("buffer_cells must be non-negative")

    window = cosine_core_window(bathy.shape, source_taper_cells)
    core_eta = eta * window
    pad = ((buffer_cells, buffer_cells), (buffer_cells, buffer_cells))
    extended_bathy = np.pad(bathy, pad, mode="edge")
    extended_eta = np.zeros_like(extended_bathy)
    crop = (
        slice(buffer_cells, buffer_cells + bathy.shape[0]),
        slice(buffer_cells, buffer_cells + bathy.shape[1]),
    )
    extended_eta[crop] = core_eta
    rest_depth = np.maximum(-extended_bathy, 0.0)
    initial_depth = np.maximum(rest_depth + extended_eta, 0.0)
    if not np.array_equal(extended_bathy[crop], bathy):
        raise RuntimeError("buffer construction changed the core bathymetry")
    if float(np.max(np.abs(core_eta[[0, -1], :]))) != 0.0 or float(
        np.max(np.abs(core_eta[:, [0, -1]]))
    ) != 0.0:
        raise RuntimeError("source taper did not produce an exact-zero crop edge")
    return {
        "bathymetry": extended_bathy,
        "eta0": extended_eta,
        "h0": initial_depth,
        "core_eta0": core_eta,
        "source_window": window,
        "crop": crop,
        "buffer_cells": int(buffer_cells),
    }


def external_sponge_mask(
    shape: Sequence[int], *, buffer_cells: int, min_factor: float = 0.8
) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("shape must contain two axes")
    nx, ny = (int(value) for value in shape)
    if buffer_cells < 0:
        raise ValueError("buffer_cells must be non-negative")
    mask = build_sponge_mask(
        nx=nx,
        ny=ny,
        width=buffer_cells,
        min_factor=min_factor,
        axes="xy",
        profile="cosine",
    )
    if buffer_cells:
        core = mask[
            buffer_cells : nx - buffer_cells,
            buffer_cells : ny - buffer_cells,
        ]
        if not np.array_equal(core, np.ones_like(core)):
            raise RuntimeError("external sponge leaked into the scientific crop")
    return mask


def _make_solver(
    solver_name: str,
    *,
    grid: int,
    dx: float,
    sponge_width: int,
    sponge_min_factor: float,
) -> Any:
    use_sponge = sponge_width > 0
    common = dict(
        nx=grid,
        ny=grid,
        dx=dx,
        dy=dx,
        dt=1.0e-4,
        g=9.81,
        use_sponge=use_sponge,
        sponge_width=sponge_width,
        sponge_min_factor=sponge_min_factor,
        sponge_time_mode=(
            "elapsed_time_consistent" if use_sponge else "legacy_per_step"
        ),
        sponge_reference_dt=0.0035 if use_sponge else None,
        sponge_axes="xy",
        sponge_profile="cosine",
    )
    if solver_name == "swe_hydrostatic":
        return HydrostaticShallowWaterSolver(
            **common,
            cfl=0.45,
            boundary="radiation",
            dry_tolerance=1.0e-6,
            max_velocity=30.0,
        )
    if solver_name == "swe_muscl_hr":
        return MUSCLHRShallowWaterSolver(
            **common,
            cfl=0.45,
            boundary="radiation",
            dry_tolerance=1.0e-6,
            max_velocity=30.0,
            reconstruction_limiter="minmod",
        )
    if solver_name != "boussinesq":
        raise ValueError(f"unsupported solver: {solver_name}")
    return BoussinesqSolver(
        **common,
        cfl=0.35,
        boundary="open",
        alpha=1.0 / 3.0,
        min_depth=1.0e-4,
        depth_scale=1.0,
        mode="linear_variable_depth",
        filter_strength=0.0,
        filter_time_mode="disabled",
        linear_solver_tol=1.0e-10,
        linear_solver_abs_tol=0.0,
        linear_solver_max_iter=max(500, int(math.ceil(500 * grid / 64))),
        cg_failure_mode="strict_v2",
        check_finite=True,
    )


def run_buffered_case(
    record: Mapping[str, Any],
    *,
    solver_name: str,
    total_grid: int,
    core_grid: int = 64,
    source_taper_cells: int = 8,
    sponge_min_factor: float = 0.8,
    sponge_width_cells: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if total_grid < core_grid or (total_grid - core_grid) % 2:
        raise ValueError("total_grid must add an equal integer buffer around the core")
    buffer_cells = (total_grid - core_grid) // 2
    sponge_width = (
        buffer_cells if sponge_width_cells is None else int(sponge_width_cells)
    )
    if sponge_width < 0 or sponge_width > buffer_cells:
        raise ValueError("sponge width must fit entirely inside the exterior buffer")
    # Imported lazily to keep the reusable buffered-domain helpers independent
    # of the Level A module that also consumes this runner.
    from src.evaluation.common_time_v2_level_a import _load_canary_arrays

    bathymetry, source, _strength_array, strength, _arrays = _load_canary_arrays(
        record
    )
    prepared = _prepare_buffered_domain(
        bathymetry,
        source,
        strength,
        0.0,
        BufferedDomainConfig(
            enabled=True,
            buffer_cells=buffer_cells,
            source_taper_cells=source_taper_cells,
            bathymetry_extension="edge",
            output_crop="central",
        ),
    )
    solver = _make_solver(
        solver_name,
        grid=total_grid,
        dx=1.0 / core_grid,
        sponge_width=sponge_width,
        sponge_min_factor=sponge_min_factor,
    )
    mask = external_sponge_mask(
        prepared["solver_bathymetry"].shape,
        buffer_cells=sponge_width,
        min_factor=sponge_min_factor,
    )
    solver.sponge_mask = mask
    solver.reset_operator_diagnostics()
    solver.set_bathymetry(prepared["solver_bathymetry"])
    if solver_name == "boussinesq":
        solver.set_initial_condition(
            prepared["solver_eta0"],
            eta_t0=np.zeros_like(prepared["solver_eta0"]),
        )
        target_cfl = 0.35
    else:
        solver.set_initial_condition(
            prepared["solver_h0"],
            hu0=np.zeros_like(prepared["solver_h0"]),
            hv0=np.zeros_like(prepared["solver_h0"]),
        )
        target_cfl = 0.45

    times = np.arange(1, 51, dtype=np.float64) * 0.0035
    started = time.monotonic()
    states, emitted, dt_history, diagnostics = _simulate_one_local(
        solver,
        n_steps=1,
        save_every=1,
        auto_dt=True,
        target_cfl=target_cfl,
        include_initial_state=False,
        requested_times=times,
        max_natural_steps=20_000,
        collect_natural_step_health=True,
        requested_state_dtype=np.float64,
    )
    runtime_s = time.monotonic() - started
    if not np.array_equal(emitted, times):
        raise RuntimeError("requested timestamps changed")
    eta = (
        states[:, 0]
        if solver_name == "boussinesq"
        else states[:, 0] + prepared["solver_bathymetry"]
    )
    crop = prepared["crop"]
    core_eta = np.asarray(eta[:, crop[0], crop[1]], dtype=np.float64)
    operator = solver.get_operator_diagnostics()
    health = {
        "runtime_s": float(runtime_s),
        "natural_steps": int(dt_history.size),
        "finite": bool(np.isfinite(states).all()),
        "measurement_dtype": str(states.dtype),
        "requested_times_exact": True,
        "max_post_step_cfl": float(
            np.max(np.asarray(diagnostics["post_step_cfl"], dtype=np.float64))
        ),
        "cg_failure_count": int(
            np.sum(np.asarray(diagnostics.get("cg_failed_count", []), dtype=np.int64))
        ),
        "cg_iterations_sum": int(operator.get("cg_iterations_sum", 0)),
        "cg_iterations_max": int(operator.get("cg_iterations_max", 0)),
        "operator": operator,
    }
    row = {
        "qualified_id": str(record["qualified_id"]),
        "input_fingerprint": str(record["input_fingerprint"]),
        "bathymetry_type": str(record["bathymetry_type"]),
        "source_type": str(record["source_type"]),
        "solver": solver_name,
        "core_grid": int(core_grid),
        "total_grid": int(total_grid),
        "buffer_cells": int(buffer_cells),
        "sponge_width_cells": int(sponge_width),
        "cell_multiplier": float((total_grid / core_grid) ** 2),
        "source_taper_cells": int(source_taper_cells),
        "source_edge_max_abs": float(
            max(
                np.max(np.abs(prepared["eta0"][[0, -1], :])),
                np.max(np.abs(prepared["eta0"][:, [0, -1]])),
            )
        ),
        "initial_core_amplitude": float(np.max(np.abs(prepared["eta0"]))),
        "sponge_core_min": float(
            np.min(mask[crop[0], crop[1]])
        ),
        "outer_boundary": "open" if solver_name == "boussinesq" else "radiation",
        "external_sponge": bool(sponge_width > 0),
        "health": health,
    }
    return row, core_eta


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "absolute_rms",
        "relative_l2",
        "interior_absolute_rms",
        "interior_relative_l2",
        "amplitude_relative_error",
        "phase_correlation_loss",
    )
    return {
        key: {
            "maximum": float(max(float(row[key]) for row in rows)),
            "median": float(np.median([float(row[key]) for row in rows])),
            "final": float(rows[-1][key]),
        }
        for key in keys
    }


def load_inventory_records(
    inventory_path: Path, qualified_ids: Sequence[str]
) -> list[dict[str, Any]]:
    wanted = [str(value) for value in qualified_ids]
    records: dict[str, dict[str, Any]] = {}
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qualified_id = str(row["qualified_id"])
            if qualified_id in wanted:
                records[qualified_id] = row
    missing = [value for value in wanted if value not in records]
    if missing:
        raise KeyError(f"inventory is missing selected cases: {missing}")
    return [records[value] for value in wanted]


def run_benchmark(
    records: Sequence[Mapping[str, Any]],
    *,
    total_grids: Sequence[int],
    source_taper_cells: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    grids = sorted({int(value) for value in total_grids})
    if not grids or grids[0] != 64:
        raise ValueError("benchmark grids must include 64 as the timing baseline")
    total = len(records) * len(SOLVERS) * len(grids)
    completed = 0
    run_rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, str, int], np.ndarray] = {}
    started = time.monotonic()
    for record in records:
        for solver_name in SOLVERS:
            for grid in grids:
                label = f"{record['qualified_id']} {solver_name} grid={grid}"
                if progress is not None:
                    progress(f"[{completed + 1}/{total}] starting {label}")
                row, trajectory = run_buffered_case(
                    record,
                    solver_name=solver_name,
                    total_grid=grid,
                    source_taper_cells=source_taper_cells,
                )
                run_rows.append(row)
                trajectories[(str(record["qualified_id"]), solver_name, grid)] = trajectory
                completed += 1
                if progress is not None:
                    progress(
                        f"[{completed}/{total}] finished {label} in "
                        f"{row['health']['runtime_s']:.2f}s"
                    )

    comparisons: list[dict[str, Any]] = []
    reference_grid = grids[-1]
    for record in records:
        qualified_id = str(record["qualified_id"])
        for solver_name in SOLVERS:
            reference = trajectories[(qualified_id, solver_name, reference_grid)]
            for grid in grids[:-1]:
                candidate = trajectories[(qualified_id, solver_name, grid)]
                metrics = comparison_metrics(
                    candidate,
                    reference,
                    boundary_band_cells=12,
                    absolute_floor=1.0e-7,
                )
                comparisons.append(
                    {
                        "qualified_id": qualified_id,
                        "solver": solver_name,
                        "candidate_grid": int(grid),
                        "reference_grid": int(reference_grid),
                        "metrics": _metric_summary(metrics),
                    }
                )

    for row in run_rows:
        baseline = next(
            other
            for other in run_rows
            if other["qualified_id"] == row["qualified_id"]
            and other["solver"] == row["solver"]
            and other["total_grid"] == 64
        )
        row["runtime_multiplier_vs_64"] = float(
            row["health"]["runtime_s"] / baseline["health"]["runtime_s"]
        )
    return {
        "artifact_kind": "buffered-central-crop-feasibility-benchmark",
        "status": "diagnostic_unfrozen_non_decisional",
        "core_grid": 64,
        "total_grids": grids,
        "source_taper_cells": int(source_taper_cells),
        "policy": {
            "source": "fixed 8-cell cosine taper to exact zero at crop edge",
            "bathymetry_extension": "constant edge continuation",
            "sponge": "elapsed-time-consistent cosine, external buffer only",
            "sponge_min_factor": 0.8,
            "swe_outer_boundary": "bathymetry-aware linearized radiation",
            "boussinesq_outer_boundary": "zero-gradient edge padding",
        },
        "case_count": len(records),
        "run_count": len(run_rows),
        "duration_s": float(time.monotonic() - started),
        "platform": {
            "processor": platform.processor(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "runs": run_rows,
        "comparisons": comparisons,
    }


def write_result(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {"result": str(path), "sha256": digest, "checksum": str(checksum_path)}
