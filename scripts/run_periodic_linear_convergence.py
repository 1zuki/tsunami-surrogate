#!/usr/bin/env python
"""Run a periodic nonlinear SWE rollout against a small-amplitude linear reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsunami_surrogate.data_gen.simulate_dataset import (
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
)
from tsunami_surrogate.utils.hashing import sha256_file


FACTORIES = {
    "swe_hydrostatic": _make_hydrostatic_solver_from_cfg,
    "swe_muscl_hr": _make_muscl_solver_from_cfg,
}
DEFAULT_GRIDS = (32, 64, 128, 256, 512)
DEFAULT_CFLS = (0.1125, 0.225, 0.45)
GRAVITY = 9.81
REST_DEPTH = 1.0
DOMAIN_LENGTH = 1.0


def measure_mode(eta: np.ndarray, kx: np.ndarray, phase_exact: float) -> tuple[float, float, float]:
    """Fit A cos(kx - phase) and wrap the phase error into [-pi, pi]."""
    coefficient = 2.0 * np.mean(eta * np.exp(-1j * kx[:, None]))
    phase_num = float(-np.angle(coefficient))
    delta = phase_num - phase_exact
    return float(abs(coefficient)), phase_num, math.atan2(math.sin(delta), math.cos(delta))


def run_wave(
    solver_name: str, grid: int, ny: int, cfl: float, amplitude: float, final_time: float
) -> dict:
    dx = DOMAIN_LENGTH / grid
    dy = DOMAIN_LENGTH / ny
    solver = FACTORIES[solver_name](
        {
            "nx": grid, "ny": ny, "dx": dx, "dy": dy, "dt": 1.0,
            "g": GRAVITY, "cfl": cfl, "boundary": "periodic", "use_sponge": False,
        }
    )
    kx = 2.0 * np.pi * (np.arange(grid) + 0.5) / grid
    wave_speed = math.sqrt(GRAVITY * REST_DEPTH)
    eta0 = np.broadcast_to(amplitude * np.cos(kx[:, None]), (grid, ny))
    bed = np.full((grid, ny), -REST_DEPTH)
    solver.set_bathymetry(bed)
    solver.set_initial_condition(REST_DEPTH + eta0, hu0=wave_speed * eta0, hv0=np.zeros_like(eta0))

    current_time = 0.0
    time_steps = []
    max_cfl_seen = 0.0
    while current_time < final_time:
        dt = min(float(solver.suggest_dt(target_cfl=cfl)), final_time - current_time)
        if not math.isfinite(dt) or dt <= 0 or current_time + dt <= current_time:
            raise RuntimeError(f"Invalid step {dt} at t={current_time}")
        solver.step(dt=dt, auto_dt=False)
        current_time += dt
        time_steps.append(dt)
        step_cfl = float(solver.compute_cfl(dt=dt))
        if not math.isfinite(step_cfl) or not np.isfinite(solver.get_state()).all():
            raise RuntimeError(f"Non-finite state at t={current_time}")
        max_cfl_seen = max(max_cfl_seen, step_cfl)
        if len(time_steps) > 100_000:
            raise RuntimeError("Exceeded 100,000 time steps")

    phase_exact = 2.0 * math.pi * wave_speed * final_time / DOMAIN_LENGTH
    exact = np.broadcast_to(amplitude * np.cos(kx[:, None] - phase_exact), (grid, ny))
    eta = solver.h + bed
    amplitude_num, phase_num, phase_error = measure_mode(eta, kx, phase_exact)
    harmonic = amplitude_num * np.cos(kx[:, None] - phase_num)
    difference = eta - exact
    exact_norm = np.linalg.norm(exact)
    counters = solver.get_operator_diagnostics()
    return {
        "solver_name": solver_name,
        "grid": grid,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "target_cfl": cfl,
        "steps": len(time_steps),
        "dt_min": min(time_steps),
        "dt_max": max(time_steps),
        "dt_mean": final_time / len(time_steps),
        "max_cfl_seen": max_cfl_seen,
        "relative_l2_error": float(np.linalg.norm(difference) / exact_norm),
        "max_abs_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "amplitude_num": amplitude_num,
        "amplitude_rel_error": amplitude_num / amplitude - 1.0,
        "phase_exact": phase_exact,
        "phase_num": phase_num,
        "phase_error": phase_error,
        "nonfundamental_relative_l2": float(np.linalg.norm(eta - harmonic) / exact_norm),
        "operator_health_counts": {
            name: int(value)
            for name, value in counters.items()
            if name in {
                "nan_to_num_replacement_count", "positivity_projection_count",
                "dry_projection_count", "muscl_cell_velocity_clip_count",
                "muscl_face_velocity_clip_count",
            }
        },
    }


def empirical_order(coarse: float, fine: float, scale_ratio: float) -> float | None:
    if min(coarse, fine) <= 0 or scale_ratio <= 1:
        return None
    return math.log(coarse / fine) / math.log(scale_ratio)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grids", nargs="+", type=int, default=DEFAULT_GRIDS)
    parser.add_argument("--cfls", nargs="+", type=float, default=DEFAULT_CFLS)
    parser.add_argument("--ny", default=8, type=int)
    parser.add_argument("--amplitude", default=0.001, type=float)
    parser.add_argument("--final-time", default=0.5, type=float)
    args = parser.parse_args()
    grids = sorted(set(args.grids))
    cfls = sorted(set(args.cfls))
    if (not all(8 <= grid <= 2048 for grid in grids) or not 4 <= args.ny <= 64
        or not all(0.01 <= cfl <= 0.45 for cfl in cfls)
        or not 0 < args.amplitude <= 0.01 or not 0 < args.final_time <= 5):
        parser.error("Expected bounded positive grids, CFLs, amplitude and final time")

    args.output.mkdir(parents=True, exist_ok=False)
    rows = {}
    spatial_orders = []
    cfl_sensitivity = []
    for solver_name in FACTORIES:
        for cfl in cfls:
            for grid in grids:
                row = run_wave(solver_name, grid, args.ny, cfl, args.amplitude, args.final_time)
                rows[solver_name, grid, cfl] = row
                print(
                    f"{solver_name} n={grid} CFL={cfl:g} rel_L2={row['relative_l2_error']:.6g}",
                    flush=True,
                )
            for coarse_grid, fine_grid in zip(grids, grids[1:]):
                coarse = rows[solver_name, coarse_grid, cfl]["relative_l2_error"]
                fine = rows[solver_name, fine_grid, cfl]["relative_l2_error"]
                spatial_orders.append({
                    "solver_name": solver_name, "target_cfl": cfl,
                    "coarse_grid": coarse_grid, "fine_grid": fine_grid,
                    "coarse_l2": coarse, "fine_l2": fine,
                    "observed_order": empirical_order(coarse, fine, fine_grid / coarse_grid),
                })
        for grid in grids:
            for fine_cfl, coarse_cfl in zip(cfls, cfls[1:]):
                fine = rows[solver_name, grid, fine_cfl]
                coarse = rows[solver_name, grid, coarse_cfl]
                cfl_sensitivity.append({
                    "solver_name": solver_name, "grid": grid,
                    "coarse_cfl": coarse_cfl, "fine_cfl": fine_cfl,
                    "coarse_l2": coarse["relative_l2_error"], "fine_l2": fine["relative_l2_error"],
                    "observed_total_error_slope": empirical_order(
                        coarse["relative_l2_error"], fine["relative_l2_error"],
                        coarse["dt_mean"] / fine["dt_mean"],
                    ),
                })

    sources = [Path(__file__).resolve(), ROOT / "src/data_gen/simulate_dataset.py"]
    sources.extend(sorted((ROOT / "src/solver").glob("*.py")))
    flat_rows = list(rows.values())
    payload = {
        "schema_id": "tsunami-surrogate.periodic-linear-wave-convergence.v2",
        "problem": "constant-depth right-going linear SWE mode on a periodic unit domain",
        "interpretation": (
            "Finite-amplitude nonlinear SWE rollouts compared with a linearized reference. "
            "The total error includes spatial and temporal discretization and the physical "
            "difference from linear theory. Fixed-grid CFL slopes measure sensitivity of "
            "total error and cannot identify the temporal order of the scheme. "
            "A nonfundamental residual includes harmonics and numerical shape error."
        ),
        "domain_length": DOMAIN_LENGTH,
        "gravity": GRAVITY,
        "still_water_depth": REST_DEPTH,
        "amplitude": args.amplitude,
        "final_time": args.final_time,
        "grids": grids,
        "ny": args.ny,
        "cfl_values": cfls,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "all_rollouts_finite": True,
        "rows": flat_rows,
        "spatial_orders": spatial_orders,
        "cfl_sensitivity": cfl_sensitivity,
    }
    _write_csv(args.output / "periodic_linear_rows.csv", [
        {**row, "operator_health_counts": json.dumps(row["operator_health_counts"], sort_keys=True)}
        for row in flat_rows
    ])
    _write_csv(args.output / "periodic_linear_spatial_orders.csv", spatial_orders)
    _write_csv(args.output / "periodic_linear_cfl_sensitivity.csv", cfl_sensitivity)
    with (args.output / "periodic_linear_convergence.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
