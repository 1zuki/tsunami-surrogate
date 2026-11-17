#!/usr/bin/env python
"""Legacy normalized boundary-flux diagnostic.

This is a bounded diagnostic for the shallow-water references.  It evaluates the
finite-volume mass flux at the four outer interfaces during an open-boundary,
sponge-enabled rollout.  The difference between observed domain-mass change and
the boundary-flux contribution is reported as a sponge/projection residual; it
is not treated as a conservation error or a physical flux balance.  For the
production 192-cell, 8.4--420 benchmark use
``scripts/run_production_dense_diagnostic.py`` instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tsunami_surrogate.solver.hydrostatic_swe import ShallowWaterSolver
from tsunami_surrogate.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver


SOLVER_NAMES = ("hydrostatic", "muscl_hr")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _make_solver(name: str, *, nx: int, ny: int, dx: float, dy: float):
    kwargs = dict(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=1.0e-3,
        g=9.81,
        cfl=0.45,
        boundary="open",
        use_sponge=True,
        sponge_width=20,
        sponge_min_factor=0.9,
        max_velocity=30.0,
    )
    if name == "hydrostatic":
        return ShallowWaterSolver(**kwargs)
    if name == "muscl_hr":
        return MUSCLHRShallowWaterSolver(
            **kwargs, reconstruction_limiter="minmod"
        )
    raise ValueError(f"unsupported solver: {name}")


def _mass_flux_arrays(solver: Any, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return oriented x/y interface mass flux arrays for one solver state."""
    if state is None:
        h, hu, hv = solver.h, solver.hu, solver.hv
    else:
        h, hu, hv = state
    b = solver.b

    if isinstance(solver, MUSCLHRShallowWaterSolver):
        rec = solver._reconstructed_faces(h, hu, hv, b)
        x_states = solver._reconstructed_x_interface_states(
            rec, h, hu, hv, b
        )
        y_states = solver._reconstructed_y_interface_states(
            rec, h, hu, hv, b
        )
    else:
        x_states = solver._x_interface_states(h, hu, hv, b)
        y_states = solver._y_interface_states(h, hu, hv, b)

    x_flux, _, _ = solver._hydrostatic_interface_fluxes(*x_states, axis="x")
    y_flux, _, _ = solver._hydrostatic_interface_fluxes(*y_states, axis="y")
    return np.asarray(x_flux[0], dtype=float), np.asarray(y_flux[0], dtype=float)


def _flux_metrics(
    x_flux: np.ndarray,
    y_flux: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> dict[str, float]:
    left = x_flux[0, :]
    right = x_flux[-1, :]
    bottom = y_flux[:, 0]
    top = y_flux[:, -1]
    outward_rate = dy * (float(np.sum(right)) - float(np.sum(left)))
    outward_rate += dx * (float(np.sum(top)) - float(np.sum(bottom)))
    max_abs_flux = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        float(np.max(np.abs(bottom))),
        float(np.max(np.abs(top))),
    )
    return {
        "outward_rate": float(outward_rate),
        "max_abs_boundary_mass_flux": float(max_abs_flux),
    }


def _run_case(
    *,
    solver_name: str,
    sample_path: Path,
    final_time: float,
    target_cfl: float,
) -> dict[str, Any]:
    with np.load(sample_path) as sample:
        bathymetry = np.asarray(sample["bathymetry"], dtype=float)
        initial_depth = np.asarray(sample["initial_depth"], dtype=float)
        scenario_id = str(np.asarray(sample["scenario_id"]).item())

    nx, ny = bathymetry.shape
    dx = 1.0 / float(nx)
    dy = 1.0 / float(ny)
    solver = _make_solver(solver_name, nx=nx, ny=ny, dx=dx, dy=dy)
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(
        initial_depth, hu0=np.zeros_like(initial_depth), hv0=np.zeros_like(initial_depth)
    )

    cell_area = dx * dy
    mass0 = float(np.sum(solver.h) * cell_area)
    elapsed = 0.0
    max_abs_flux = 0.0
    max_abs_outward_rate = 0.0
    max_abs_sponge_residual = 0.0
    cumulative_boundary_mass_change = 0.0
    cumulative_sponge_residual = 0.0
    max_cfl = 0.0
    steps = 0

    while elapsed < final_time - 1e-15:
        dt = min(
            float(solver.suggest_dt(target_cfl=target_cfl)),
            float(final_time - elapsed),
        )
        if not math.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"invalid dt={dt} at step {steps}")

        state0 = solver.get_state()
        if solver_name == "muscl_hr":
            x0, y0 = _mass_flux_arrays(solver, state0)
            state1 = solver._euler_step_from_state(state0, dt)
            x1, y1 = _mass_flux_arrays(solver, state1)
            flux0 = _flux_metrics(x0, y0, dx=dx, dy=dy)
            flux1 = _flux_metrics(x1, y1, dx=dx, dy=dy)
            outward_rate = 0.5 * (
                flux0["outward_rate"] + flux1["outward_rate"]
            )
            max_abs_flux = max(
                max_abs_flux,
                flux0["max_abs_boundary_mass_flux"],
                flux1["max_abs_boundary_mass_flux"],
            )
            max_abs_outward_rate = max(
                max_abs_outward_rate,
                abs(flux0["outward_rate"]),
                abs(flux1["outward_rate"]),
            )
        else:
            x_flux, y_flux = _mass_flux_arrays(solver, state0)
            flux = _flux_metrics(x_flux, y_flux, dx=dx, dy=dy)
            outward_rate = flux["outward_rate"]
            max_abs_flux = max(max_abs_flux, flux["max_abs_boundary_mass_flux"])
            max_abs_outward_rate = max(max_abs_outward_rate, abs(outward_rate))

        mass_before = float(np.sum(solver.h) * cell_area)
        flux_mass_change = -dt * outward_rate
        solver.step(dt=dt, auto_dt=False)
        mass_after = float(np.sum(solver.h) * cell_area)
        observed_mass_change = mass_after - mass_before
        sponge_residual = observed_mass_change - flux_mass_change
        cumulative_boundary_mass_change += flux_mass_change
        cumulative_sponge_residual += sponge_residual
        max_abs_sponge_residual = max(max_abs_sponge_residual, abs(sponge_residual))
        max_cfl = max(max_cfl, float(solver.compute_cfl(dt)))
        elapsed += dt
        steps += 1

    final_mass = float(np.sum(solver.h) * cell_area)
    return {
        "scenario_id": scenario_id,
        "solver": solver_name,
        "steps": int(steps),
        "final_time": float(elapsed),
        "mass_initial": mass0,
        "mass_final": final_mass,
        "mass_relative_change": float((final_mass - mass0) / max(abs(mass0), 1e-30)),
        "max_abs_boundary_mass_flux": float(max_abs_flux),
        "max_abs_outward_mass_rate": float(max_abs_outward_rate),
        "cumulative_boundary_mass_change": float(cumulative_boundary_mass_change),
        "cumulative_sponge_projection_residual": float(cumulative_sponge_residual),
        "max_abs_sponge_projection_residual_per_step": float(max_abs_sponge_residual),
        "max_cfl_seen": float(max_cfl),
        "finite_final_state": bool(np.isfinite(solver.get_state()).all()),
        "operator_diagnostics": solver.get_operator_diagnostics(),
    }


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "count": int(values.size),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path("/mnt/Windows/Dataset/current_dense_replay_20261117"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_revision/boundary_flux_current"))
    parser.add_argument("--final-time", type=float, default=0.08)
    parser.add_argument("--target-cfl", type=float, default=0.45)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    selection = json.loads((args.replay_root / "selection.json").read_text())
    scenario_ids = list(selection["dense_validation"]["ordered_scenario_ids"])
    if args.max_scenarios is not None:
        scenario_ids = scenario_ids[: max(0, int(args.max_scenarios))]

    rows: list[dict[str, Any]] = []
    for solver_name in SOLVER_NAMES:
        for scenario_id in scenario_ids:
            sample_path = (
                args.replay_root
                / "hydrostatic"
                / "samples"
                / f"sample_{scenario_id.split('_', 1)[1]}"
                / "sample.npz"
            )
            row = _run_case(
                solver_name=solver_name,
                sample_path=sample_path,
                final_time=float(args.final_time),
                target_cfl=float(args.target_cfl),
            )
            rows.append(row)
            print(
                f"{solver_name} {scenario_id}: max_flux="
                f"{row['max_abs_boundary_mass_flux']:.6e} "
                f"mass_rel={row['mass_relative_change']:.6e}",
                flush=True,
            )

    by_solver: dict[str, Any] = {}
    for solver_name in SOLVER_NAMES:
        subset = [row for row in rows if row["solver"] == solver_name]
        by_solver[solver_name] = {
            "scenario_count": len(subset),
            "finite_final_state_count": int(
                sum(bool(row["finite_final_state"]) for row in subset)
            ),
            "max_abs_boundary_mass_flux": _summarize(
                subset, "max_abs_boundary_mass_flux"
            ),
            "max_abs_outward_mass_rate": _summarize(
                subset, "max_abs_outward_mass_rate"
            ),
            "abs_mass_relative_change": _summarize(
                [{**row, "abs_mass_relative_change": abs(row["mass_relative_change"])} for row in subset],
                "abs_mass_relative_change",
            ),
            "abs_cumulative_sponge_projection_residual": _summarize(
                [
                    {
                        **row,
                        "abs_cumulative_sponge_projection_residual": abs(
                            row["cumulative_sponge_projection_residual"]
                        ),
                    }
                    for row in subset
                ],
                "abs_cumulative_sponge_projection_residual",
            ),
            "max_cfl_seen": _summarize(subset, "max_cfl_seen"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "tsunami-surrogate.boundary-flux-diagnostic.v1",
        "diagnostic": "open_boundary_mass_flux_and_sponge_residual",
        "interpretation": (
            "Boundary mass flux is evaluated at the finite-volume outer interfaces. "
            "The residual is the observed domain-mass change minus the boundary-flux "
            "contribution and therefore includes sponge damping and projections."
        ),
        "config": {
            "replay_root": str(args.replay_root),
            "scenario_count": len(scenario_ids),
            "scenario_ids": scenario_ids,
            "final_time": float(args.final_time),
            "target_cfl": float(args.target_cfl),
            "boundary": "open/radiation",
            "use_sponge": True,
        },
        "by_solver": by_solver,
        "rows": rows,
    }
    (args.output_dir / "boundary_flux_diagnostic.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "boundary_flux_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "scenario_id",
            "solver",
            "steps",
            "final_time",
            "mass_relative_change",
            "max_abs_boundary_mass_flux",
            "max_abs_outward_mass_rate",
            "cumulative_boundary_mass_change",
            "cumulative_sponge_projection_residual",
            "max_abs_sponge_projection_residual_per_step",
            "max_cfl_seen",
            "finite_final_state",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    print(f"saved={args.output_dir / 'boundary_flux_diagnostic.json'}")


if __name__ == "__main__":
    main()
