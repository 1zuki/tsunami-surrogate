#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.hydrostatic_swe import HydrostaticShallowWaterSolver
from src.solver.muscl_hr_swe import MUSCLHRShallowWaterSolver


SOLVERS = {
    "swe_hydrostatic": HydrostaticShallowWaterSolver,
    "swe_muscl_hr": MUSCLHRShallowWaterSolver,
}
SOLVER_LABELS = {
    "swe_hydrostatic": "Hydrostatic",
    "swe_muscl_hr": "MUSCL-HR",
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n, dtype=float) + 0.5) / float(n)
    y = (np.arange(n, dtype=float) + 0.5) / float(n)
    return np.meshgrid(x, y, indexing="ij")


def _gaussian(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, sigma: float, amp: float
) -> np.ndarray:
    return amp * np.exp(
        -0.5 * (((x - cx) ** 2 + (y - cy) ** 2) / max(sigma * sigma, 1e-12))
    )


def _flat_radial_gaussian(n: int) -> dict[str, Any]:
    x, y = _grid(n)
    bathymetry = -np.ones((n, n), dtype=float)
    eta0 = _gaussian(x, y, 0.5, 0.5, 0.07, 0.04)
    eta0 -= float(np.mean(eta0))
    h0 = np.maximum(-bathymetry + eta0, 0.0)
    return {
        "case": "flat_radial_gaussian",
        "bathymetry": bathymetry,
        "h0": h0,
        "boundary": "reflective",
        "description": "Centered radial Gaussian pulse over flat fully wet bathymetry.",
    }


def _smooth_bathymetry_gaussian(n: int) -> dict[str, Any]:
    x, y = _grid(n)
    bathymetry = (
        -1.1
        - 0.18 * _gaussian(x, y, 0.35, 0.55, 0.18, 1.0)
        + 0.08 * _gaussian(x, y, 0.72, 0.30, 0.12, 1.0)
        - 0.05 * (x - 0.5)
    )
    bathymetry = np.clip(bathymetry, -1.45, -0.9)
    eta0 = _gaussian(x, y, 0.42, 0.60, 0.06, 0.035)
    eta0 -= float(np.mean(eta0))
    h0 = np.maximum(-bathymetry + eta0, 0.0)
    return {
        "case": "smooth_bathymetry_gaussian",
        "bathymetry": bathymetry,
        "h0": h0,
        "boundary": "reflective",
        "description": "Gaussian pulse over smooth variable fully wet bathymetry.",
    }


def _wet_bed_dam_break(n: int) -> dict[str, Any]:
    x, _ = _grid(n)
    bathymetry = -np.ones((n, n), dtype=float)
    eta0 = np.where(x < 0.5, 0.05, -0.05)
    h0 = np.maximum(-bathymetry + eta0, 0.0)
    return {
        "case": "wet_bed_dam_break",
        "bathymetry": bathymetry,
        "h0": h0,
        "boundary": "reflective",
        "description": "Two-dimensional wet-bed dam-break with y-invariant initial data.",
    }


CASE_BUILDERS: dict[str, Callable[[int], dict[str, Any]]] = {
    "flat_radial_gaussian": _flat_radial_gaussian,
    "smooth_bathymetry_gaussian": _smooth_bathymetry_gaussian,
    "wet_bed_dam_break": _wet_bed_dam_break,
}


def _make_solver(
    solver_name: str,
    n: int,
    target_cfl: float,
    bathymetry: np.ndarray,
    h0: np.ndarray,
    *,
    boundary: str,
):
    solver_cls = SOLVERS[solver_name]
    solver = solver_cls(
        nx=n,
        ny=n,
        dx=1.0 / float(n),
        dy=1.0 / float(n),
        dt=1e-3,
        g=9.81,
        cfl=float(target_cfl),
        dry_tolerance=1e-6,
        boundary=boundary,
        use_sponge=False,
        max_velocity=30.0,
    )
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))
    return solver


def _step_to_time(solver: Any, final_time: float, target_cfl: float) -> dict[str, Any]:
    elapsed = 0.0
    n_steps = 0
    dt_min = math.inf
    dt_max = -math.inf
    max_cfl = 0.0
    min_h = float(np.min(solver.h))
    max_speed = 0.0
    while elapsed < float(final_time) - 1e-14:
        dt = float(solver.suggest_dt(target_cfl=target_cfl))
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"non-finite/non-positive dt={dt}")
        remaining = float(final_time) - elapsed
        if dt > remaining:
            dt = remaining
        solver.dt = dt
        solver.step(dt=dt, auto_dt=False)
        elapsed += dt
        n_steps += 1
        dt_min = min(dt_min, dt)
        dt_max = max(dt_max, dt)
        max_cfl = max(max_cfl, float(solver.compute_cfl(dt=dt)))
        min_h = min(min_h, float(np.min(solver.h)))
        u, v = solver.compute_velocity()
        max_speed = max(max_speed, float(np.max(np.hypot(u, v))))
    return {
        "end_time": float(elapsed),
        "n_steps": int(n_steps),
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "max_cfl_seen": float(max_cfl),
        "min_h_seen": float(min_h),
        "max_speed_seen": float(max_speed),
    }


def _mass(h: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(np.asarray(h, dtype=float)) * dx * dy)


def _symmetry_metrics(eta: np.ndarray) -> dict[str, float]:
    arr = np.asarray(eta, dtype=float)
    return {
        "reflection_x_max_abs": float(np.max(np.abs(arr - arr[::-1, :]))),
        "reflection_y_max_abs": float(np.max(np.abs(arr - arr[:, ::-1]))),
        "transpose_max_abs": float(np.max(np.abs(arr - arr.T))),
    }


def _y_invariance(eta: np.ndarray) -> float:
    arr = np.asarray(eta, dtype=float)
    return float(np.max(np.abs(arr - arr.mean(axis=1, keepdims=True))))


def _run_case(
    solver_name: str,
    case: Mapping[str, Any],
    n: int,
    final_time: float,
    target_cfl: float,
) -> dict[str, Any]:
    bathymetry = np.asarray(case["bathymetry"], dtype=float)
    h0 = np.asarray(case["h0"], dtype=float)
    dx = dy = 1.0 / float(n)
    solver = _make_solver(
        solver_name,
        n,
        target_cfl,
        bathymetry,
        h0,
        boundary=str(case["boundary"]),
    )
    mass0 = _mass(solver.h, dx, dy)
    eta0 = solver.compute_free_surface().copy()
    step_stats = _step_to_time(solver, final_time, target_cfl)
    eta = np.asarray(solver.compute_free_surface(), dtype=float)
    state = np.asarray(solver.get_state(), dtype=float)
    mass1 = _mass(solver.h, dx, dy)
    mass_rel_change = abs(mass1 - mass0) / max(abs(mass0), 1e-30)

    row: dict[str, Any] = {
        "case": str(case["case"]),
        "solver": SOLVER_LABELS[solver_name],
        "solver_name": solver_name,
        "grid": int(n),
        "target_cfl": float(target_cfl),
        "final_time": float(final_time),
        "boundary": str(case["boundary"]),
        "finite": bool(np.isfinite(state).all() and np.isfinite(eta).all()),
        "mass_relative_change": float(mass_rel_change),
        "max_abs_eta_initial": float(np.max(np.abs(eta0))),
        "max_abs_eta_final": float(np.max(np.abs(eta))),
        "eta_l2_change": float(np.linalg.norm((eta - eta0).ravel())),
        **step_stats,
    }

    if case["case"] == "flat_radial_gaussian":
        row.update(_symmetry_metrics(eta))
        row["pass"] = bool(
            row["finite"]
            and row["min_h_seen"] >= -1e-10
            and row["mass_relative_change"] < 1e-10
            and row["reflection_x_max_abs"] < 5e-4
            and row["reflection_y_max_abs"] < 5e-4
            and row["transpose_max_abs"] < 5e-4
        )
        row["criterion"] = (
            "finite, nonnegative depth, mass drift <1e-10, symmetry errors <5e-4"
        )
    elif case["case"] == "wet_bed_dam_break":
        row["y_invariance_max_abs"] = _y_invariance(eta)
        row["pass"] = bool(
            row["finite"]
            and row["min_h_seen"] >= -1e-10
            and row["mass_relative_change"] < 1e-10
            and row["y_invariance_max_abs"] < 5e-5
        )
        row["criterion"] = (
            "finite, nonnegative depth, mass drift <1e-10, y-invariance error <5e-5"
        )
    else:
        row["pass"] = bool(
            row["finite"]
            and row["min_h_seen"] >= -1e-10
            and row["mass_relative_change"] < 1e-10
        )
        row["criterion"] = "finite, nonnegative depth, mass drift <1e-10"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compact standard SWE sanity-validation cases."
    )
    parser.add_argument("--output-dir", default="results/swe_standard_validation")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--final-time", type=float, default=0.02)
    parser.add_argument("--target-cfl", type=float, default=0.40)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=list(SOLVERS.keys()),
        choices=list(SOLVERS.keys()),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASE_BUILDERS.keys()),
        choices=list(CASE_BUILDERS.keys()),
    )
    args = parser.parse_args()

    if int(args.grid) <= 1:
        raise ValueError("--grid must be greater than 1")
    if float(args.final_time) <= 0.0:
        raise ValueError("--final-time must be positive")
    if float(args.target_cfl) <= 0.0:
        raise ValueError("--target-cfl must be positive")

    started = time.time()
    rows: list[dict[str, Any]] = []
    for case_name in args.cases:
        case = CASE_BUILDERS[case_name](int(args.grid))
        for solver_name in args.solvers:
            print(
                f"{case_name}: {SOLVER_LABELS[solver_name]} grid={args.grid} "
                f"final_time={float(args.final_time):.4g}",
                flush=True,
            )
            try:
                rows.append(
                    _run_case(
                        solver_name,
                        case,
                        int(args.grid),
                        float(args.final_time),
                        float(args.target_cfl),
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "case": case_name,
                        "solver": SOLVER_LABELS[solver_name],
                        "solver_name": solver_name,
                        "grid": int(args.grid),
                        "target_cfl": float(args.target_cfl),
                        "final_time": float(args.final_time),
                        "finite": False,
                        "pass": False,
                        "error": repr(exc),
                    }
                )

    payload = {
        "diagnostic_type": "swe_standard_validation_sanity_cases",
        "grid": int(args.grid),
        "target_cfl": float(args.target_cfl),
        "final_time": float(args.final_time),
        "solvers": list(args.solvers),
        "cases": list(args.cases),
        "notes": [
            "These are compact standard shallow-water sanity cases for the two SWE solvers.",
            "They check finite/nonnegative states, conservation under reflective no-sponge boundaries, radial symmetry for a flat Gaussian pulse, and y-invariance for a wet-bed dam-break.",
            "They are reported as validation diagnostics and do not replace analytic-solution or full convergence studies.",
        ],
        "rows": rows,
        "num_rows": len(rows),
        "num_passed": int(sum(1 for row in rows if bool(row.get("pass", False)))),
        "elapsed_s": float(time.time() - started),
    }

    output_dir = ROOT / args.output_dir
    _write_json(payload, output_dir / "swe_standard_validation.json")
    _write_csv(output_dir / "swe_standard_validation_rows.csv", rows)

    print(f"SWE standard validation -> {output_dir / 'swe_standard_validation.json'}")
    print(
        f"passed={payload['num_passed']}/{payload['num_rows']} elapsed_s={payload['elapsed_s']:.1f}"
    )


if __name__ == "__main__":
    main()
