#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _grid_xy(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n, dtype=float) + 0.5) / float(n)
    y = (np.arange(n, dtype=float) + 0.5) / float(n)
    return np.meshgrid(x, y, indexing="ij")


def _restrict_mean(field: np.ndarray, target_n: int) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.shape != (arr.shape[0], arr.shape[1]) or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square 2-D field, got {arr.shape}")
    source_n = arr.shape[0]
    if source_n == target_n:
        return arr.copy()
    if source_n % target_n != 0:
        raise ValueError(f"cannot restrict {source_n} to {target_n} by integer average")
    factor = source_n // target_n
    return arr.reshape(target_n, factor, target_n, factor).mean(axis=(1, 3))


def _gaussian(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    amp: float,
) -> np.ndarray:
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amp * np.exp(-0.5 * r2 / max(sigma * sigma, 1e-12))


def _make_highres_scenario(
    index: int, n: int, rng: np.random.Generator
) -> dict[str, Any]:
    x, y = _grid_xy(n)
    slope_x = rng.uniform(-0.08, 0.08)
    slope_y = rng.uniform(-0.08, 0.08)
    b = -1.25 - slope_x * (x - 0.5) - slope_y * (y - 0.5)

    for _ in range(3):
        amp = rng.uniform(-0.16, 0.12)
        sigma = rng.uniform(0.06, 0.16)
        cx, cy = rng.uniform(0.18, 0.82, size=2)
        b += _gaussian(x, y, cx, cy, sigma, amp)
    b = np.clip(b, -1.8, -0.85)

    eta = np.zeros_like(b)
    for k in range(rng.integers(1, 4)):
        amp = rng.uniform(0.018, 0.055) * (1.0 if k == 0 else rng.choice([-1.0, 1.0]))
        sigma = rng.uniform(0.035, 0.075)
        cx, cy = rng.uniform(0.25, 0.75, size=2)
        eta += _gaussian(x, y, cx, cy, sigma, amp)

    eta -= float(np.mean(eta))
    eta = np.clip(eta, -0.08, 0.08)
    return {
        "scenario_id": f"conv_{index:04d}",
        "bathymetry_128": b.astype(np.float64),
        "eta0_128": eta.astype(np.float64),
    }


def _scenario_on_grid(
    scenario: Mapping[str, Any], grid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bathymetry = _restrict_mean(
        np.asarray(scenario["bathymetry_128"], dtype=float), grid
    )
    eta0 = _restrict_mean(np.asarray(scenario["eta0_128"], dtype=float), grid)
    h0 = np.maximum(-bathymetry + eta0, 0.0)
    return bathymetry, eta0, h0


def _make_solver(
    solver_name: str,
    grid: int,
    cfl: float,
    bathymetry: np.ndarray,
    h0: np.ndarray,
    *,
    boundary: str,
    use_sponge: bool,
    max_velocity: float,
):
    solver_cls = SOLVERS[solver_name]
    solver = solver_cls(
        nx=int(grid),
        ny=int(grid),
        dx=1.0 / float(grid),
        dy=1.0 / float(grid),
        dt=1e-3,
        g=9.81,
        cfl=float(cfl),
        dry_tolerance=1e-6,
        boundary=boundary,
        use_sponge=use_sponge,
        sponge_width=max(1, int(grid) // 8),
        sponge_min_factor=0.9,
        max_velocity=float(max_velocity),
    )
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))
    return solver


def _run_to_final_time(
    solver_name: str,
    grid: int,
    cfl: float,
    scenario: Mapping[str, Any],
    final_time: float,
    *,
    boundary: str,
    use_sponge: bool,
    max_velocity: float,
) -> dict[str, Any]:
    bathymetry, _, h0 = _scenario_on_grid(scenario, grid)
    solver = _make_solver(
        solver_name,
        grid,
        cfl,
        bathymetry,
        h0,
        boundary=boundary,
        use_sponge=use_sponge,
        max_velocity=max_velocity,
    )
    elapsed = 0.0
    dt_min = math.inf
    dt_max = -math.inf
    n_steps = 0
    max_cfl_seen = 0.0
    min_h_seen = float(np.min(h0))
    max_speed_seen = 0.0
    target = float(final_time)

    while elapsed < target - 1e-14:
        dt = float(solver.suggest_dt(target_cfl=float(cfl)))
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"non-finite/non-positive dt={dt}")
        remaining = target - elapsed
        if dt > remaining:
            dt = remaining
        solver.dt = dt
        solver.step(dt=dt, auto_dt=False)
        elapsed += dt
        n_steps += 1
        dt_min = min(dt_min, dt)
        dt_max = max(dt_max, dt)
        max_cfl_seen = max(max_cfl_seen, float(solver.compute_cfl(dt=dt)))
        min_h_seen = min(min_h_seen, float(np.min(solver.h)))
        u, v = solver.compute_velocity()
        max_speed_seen = max(max_speed_seen, float(np.max(np.hypot(u, v))))

    eta = np.asarray(solver.compute_free_surface(), dtype=np.float64)
    state = np.asarray(solver.get_state(), dtype=np.float64)
    return {
        "ok": bool(np.isfinite(state).all() and np.isfinite(eta).all()),
        "eta": eta,
        "n_steps": int(n_steps),
        "end_time": float(elapsed),
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "max_cfl_seen": float(max_cfl_seen),
        "min_h_seen": float(min_h_seen),
        "max_speed_seen": float(max_speed_seen),
        "max_abs_eta": float(np.max(np.abs(eta))),
    }


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    diff = aa - bb
    denom = max(float(np.linalg.norm(bb.ravel())), 1e-30)
    return {
        "rel_l2": float(np.linalg.norm(diff.ravel()) / denom),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _summary(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _aggregate(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(
        groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])
    ):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["num_rows"] = len(group)
        for metric in metric_names:
            stats = _summary(row.get(metric, float("nan")) for row in group)
            for stat_name, stat_value in stats.items():
                rec[f"{metric}_{stat_name}"] = stat_value
        out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Hydrostatic/MUSCL-HR grid and CFL sensitivity diagnostic on "
            "shared smooth 128-grid scenarios. This is a stability/sensitivity "
            "check, not a formal convergence proof."
        )
    )
    parser.add_argument("--output-dir", default="results/solver_convergence_minimal")
    parser.add_argument("--num-scenarios", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--cfl-values", type=float, nargs="+", default=[0.30, 0.45])
    parser.add_argument("--final-time", type=float, default=0.02)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=list(SOLVERS.keys()),
        choices=list(SOLVERS.keys()),
    )
    parser.add_argument("--boundary", default="reflective")
    parser.add_argument("--use-sponge", action="store_true")
    parser.add_argument("--max-velocity", type=float, default=30.0)
    args = parser.parse_args()

    grids = sorted({int(v) for v in args.grids})
    cfl_values = [float(v) for v in args.cfl_values]
    if grids != [32, 64, 128]:
        raise ValueError(
            "--grids must be exactly 32 64 128 for this minimal diagnostic"
        )
    if int(args.num_scenarios) <= 0:
        raise ValueError("--num-scenarios must be positive")
    if float(args.final_time) <= 0.0:
        raise ValueError("--final-time must be positive")

    started = time.time()
    rng = np.random.default_rng(int(args.seed))
    scenarios = [
        _make_highres_scenario(i, 128, rng) for i in range(int(args.num_scenarios))
    ]

    run_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    cfl_rows: list[dict[str, Any]] = []
    final_eta: dict[tuple[str, str, float, int], np.ndarray] = {}

    total = len(args.solvers) * len(scenarios) * len(cfl_values) * len(grids)
    completed = 0
    for solver_name in args.solvers:
        for scenario in scenarios:
            scenario_id = str(scenario["scenario_id"])
            for cfl in cfl_values:
                for grid in grids:
                    completed += 1
                    print(
                        f"[{completed}/{total}] {SOLVER_LABELS[solver_name]} "
                        f"{scenario_id} grid={grid} cfl={cfl:.2f}",
                        flush=True,
                    )
                    row_base = {
                        "solver": SOLVER_LABELS[solver_name],
                        "solver_name": solver_name,
                        "scenario_id": scenario_id,
                        "grid": int(grid),
                        "target_cfl": float(cfl),
                        "final_time_target": float(args.final_time),
                    }
                    try:
                        result = _run_to_final_time(
                            solver_name,
                            grid,
                            cfl,
                            scenario,
                            float(args.final_time),
                            boundary=str(args.boundary),
                            use_sponge=bool(args.use_sponge),
                            max_velocity=float(args.max_velocity),
                        )
                        final_eta[(solver_name, scenario_id, float(cfl), int(grid))] = (
                            np.asarray(result["eta"], dtype=np.float64)
                        )
                        run_rows.append(
                            {
                                **row_base,
                                **{k: v for k, v in result.items() if k != "eta"},
                            }
                        )
                    except Exception as exc:
                        run_rows.append({**row_base, "ok": False, "error": repr(exc)})

    for solver_name in args.solvers:
        for scenario in scenarios:
            scenario_id = str(scenario["scenario_id"])
            for cfl in cfl_values:
                eta32 = final_eta.get((solver_name, scenario_id, float(cfl), 32))
                eta64 = final_eta.get((solver_name, scenario_id, float(cfl), 64))
                eta128 = final_eta.get((solver_name, scenario_id, float(cfl), 128))
                if eta32 is not None and eta64 is not None:
                    row = {
                        "solver": SOLVER_LABELS[solver_name],
                        "solver_name": solver_name,
                        "scenario_id": scenario_id,
                        "target_cfl": float(cfl),
                        "comparison": "32_vs_restricted_64",
                        "coarse_grid": 32,
                        "fine_grid": 64,
                    }
                    row.update(_metrics(eta32, _restrict_mean(eta64, 32)))
                    grid_rows.append(row)
                if eta64 is not None and eta128 is not None:
                    row = {
                        "solver": SOLVER_LABELS[solver_name],
                        "solver_name": solver_name,
                        "scenario_id": scenario_id,
                        "target_cfl": float(cfl),
                        "comparison": "64_vs_restricted_128",
                        "coarse_grid": 64,
                        "fine_grid": 128,
                    }
                    row.update(_metrics(eta64, _restrict_mean(eta128, 64)))
                    grid_rows.append(row)
                if eta32 is not None and eta128 is not None:
                    row = {
                        "solver": SOLVER_LABELS[solver_name],
                        "solver_name": solver_name,
                        "scenario_id": scenario_id,
                        "target_cfl": float(cfl),
                        "comparison": "32_vs_restricted_128",
                        "coarse_grid": 32,
                        "fine_grid": 128,
                    }
                    row.update(_metrics(eta32, _restrict_mean(eta128, 32)))
                    grid_rows.append(row)

            if len(cfl_values) >= 2:
                low = min(cfl_values)
                high = max(cfl_values)
                for grid in grids:
                    eta_low = final_eta.get(
                        (solver_name, scenario_id, float(low), grid)
                    )
                    eta_high = final_eta.get(
                        (solver_name, scenario_id, float(high), grid)
                    )
                    if eta_low is None or eta_high is None:
                        continue
                    row = {
                        "solver": SOLVER_LABELS[solver_name],
                        "solver_name": solver_name,
                        "scenario_id": scenario_id,
                        "grid": int(grid),
                        "cfl_a": float(low),
                        "cfl_b": float(high),
                    }
                    row.update(_metrics(eta_low, eta_high))
                    cfl_rows.append(row)

    grid_summary = _aggregate(
        grid_rows,
        ("solver", "target_cfl", "comparison"),
        ("rel_l2", "rmse", "mae", "max_abs"),
    )
    cfl_summary = _aggregate(
        cfl_rows,
        ("solver", "grid", "cfl_a", "cfl_b"),
        ("rel_l2", "rmse", "mae", "max_abs"),
    )
    run_summary = _aggregate(
        run_rows,
        ("solver", "grid", "target_cfl"),
        (
            "n_steps",
            "dt_min",
            "dt_max",
            "max_cfl_seen",
            "min_h_seen",
            "max_speed_seen",
            "max_abs_eta",
        ),
    )

    output_dir = ROOT / args.output_dir
    payload = {
        "diagnostic_type": "minimal_solver_grid_time_sensitivity",
        "num_scenarios": int(args.num_scenarios),
        "seed": int(args.seed),
        "solvers": list(args.solvers),
        "grids": grids,
        "cfl_values": cfl_values,
        "final_time": float(args.final_time),
        "boundary": str(args.boundary),
        "use_sponge": bool(args.use_sponge),
        "notes": [
            "Shared physical scenarios are generated at 128x128 and restricted to 64x64/32x32 by cell averaging.",
            "All rollouts are adaptively stepped to the same requested final physical time; the final step is clipped.",
            "Final eta fields are compared after restricting the finer-grid result to the coarser grid.",
            "This compact diagnostic documents stability and grid/time sensitivity; it is not a formal convergence proof.",
        ],
        "run_summary": run_summary,
        "grid_comparison_summary": grid_summary,
        "cfl_sensitivity_summary": cfl_summary,
        "elapsed_s": float(time.time() - started),
    }
    _write_json(payload, output_dir / "solver_convergence_minimal.json")
    _write_csv(output_dir / "run_rows.csv", run_rows)
    _write_csv(output_dir / "grid_comparison_rows.csv", grid_rows)
    _write_csv(output_dir / "grid_comparison_summary.csv", grid_summary)
    _write_csv(output_dir / "cfl_sensitivity_rows.csv", cfl_rows)
    _write_csv(output_dir / "cfl_sensitivity_summary.csv", cfl_summary)

    print(
        f"solver convergence diagnostic -> {output_dir / 'solver_convergence_minimal.json'}"
    )
    print(f"elapsed_s={payload['elapsed_s']:.1f}")


if __name__ == "__main__":
    main()
