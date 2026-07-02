#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ShardedTsunamiDataset
from src.data_gen.simulate_dataset import (
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
)
from src.utils.config import load_config


SOLVERS = ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
SOLVER_DATASETS = {
    "swe_hydrostatic": "hydrostatic",
    "swe_muscl_hr": "muscl_hr",
    "boussinesq": "boussinesq",
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


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def _log(message: str) -> None:
    print(message, flush=True)


def _validate_solver_list(values: Iterable[str], label: str) -> tuple[str, ...]:
    out = tuple(str(v).strip() for v in values if str(v).strip())
    invalid = [v for v in out if v not in SOLVERS]
    if invalid:
        raise ValueError(
            f"{label} contains unsupported solver(s): {invalid}. Supported: {SOLVERS}"
        )
    if not out:
        raise ValueError(f"{label} must include at least one solver")
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _summarize_values(values: Iterable[float | None]) -> dict[str, Any]:
    arr = np.asarray(
        [v for v in values if v is not None and math.isfinite(float(v))], dtype=float
    )
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


def _load_normalization_stats(stats_path: Path) -> dict[str, Any]:
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _denormalize_channel(
    name: str, arr: np.ndarray, stats: Mapping[str, Any]
) -> np.ndarray:
    inputs = stats.get("inputs", {})
    spec = inputs.get(name, {}) if isinstance(inputs, Mapping) else {}
    if not isinstance(spec, Mapping) or "offset" not in spec or "scale" not in spec:
        return np.asarray(arr, dtype=float)
    return np.asarray(arr, dtype=float) * float(spec["scale"]) + float(spec["offset"])


def _choose_indices(num_samples: int, count: int) -> list[int]:
    count = max(0, min(int(count), int(num_samples)))
    if count == 0:
        return []
    return sorted({int(i) for i in np.linspace(0, num_samples - 1, count, dtype=int)})


def _load_physical_cases(
    dataset_path: Path,
    stats_path: Path,
    count: int,
) -> list[dict[str, Any]]:
    dataset = ShardedTsunamiDataset(dataset_path)
    stats = _load_normalization_stats(stats_path)
    cases: list[dict[str, Any]] = []

    for idx in _choose_indices(len(dataset), count):
        item = dataset[idx]
        x = np.asarray(item["x"], dtype=float)
        bathymetry = _denormalize_channel("bathymetry", x[0], stats)
        source = _denormalize_channel("source", x[1], stats)
        initial_depth = _denormalize_channel("initial_depth", x[2], stats)
        cases.append(
            {
                "dataset_index": int(idx),
                "sample_id": str(item.get("sample_id", "")),
                "scenario_id": str(item.get("scenario_id", "")),
                "bathymetry_type": str(item.get("bathymetry_type", "unknown")),
                "source_type": str(item.get("source_type", "unknown")),
                "source_strength": float(item.get("source_strength", float("nan"))),
                "bathymetry": bathymetry,
                "source": source,
                "initial_depth": initial_depth,
            }
        )
    return cases


def _solver_cfg(
    base_cfg: Mapping[str, Any],
    *,
    use_sponge: bool | None = None,
    boundary: str | None = None,
    filter_strength: float | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(dict(base_cfg))
    if use_sponge is not None:
        cfg["use_sponge"] = bool(use_sponge)
    if boundary is not None:
        cfg["boundary"] = str(boundary)
    if filter_strength is not None:
        cfg["filter_strength"] = float(filter_strength)
    return cfg


def _make_solver(solver_name: str, solver_cfg: Mapping[str, Any]):
    if solver_name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(dict(solver_cfg))
    if solver_name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(dict(solver_cfg))
    if solver_name == "boussinesq":
        return _make_boussinesq_solver_from_cfg(dict(solver_cfg))
    raise ValueError(f"Unsupported solver: {solver_name}")


def _setup_dynamic_solver(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    case: Mapping[str, Any],
    sea_level_offset: float,
):
    bathymetry = np.asarray(case["bathymetry"], dtype=float)
    h0 = np.maximum(np.asarray(case["initial_depth"], dtype=float), 0.0)
    solver = _make_solver(solver_name, solver_cfg)
    solver.set_bathymetry(bathymetry)
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))
    else:
        eta0 = h0 + bathymetry - float(sea_level_offset)
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
    return solver


def _setup_lake_rest_solver(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    case: Mapping[str, Any],
    sea_level_offset: float,
):
    bathymetry = np.asarray(case["bathymetry"], dtype=float)
    rest_depth = np.maximum(-bathymetry + float(sea_level_offset), 0.0)
    solver = _make_solver(solver_name, solver_cfg)
    solver.set_bathymetry(bathymetry)
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        solver.set_initial_condition(
            rest_depth, hu0=np.zeros_like(rest_depth), hv0=np.zeros_like(rest_depth)
        )
    else:
        eta0 = np.zeros_like(rest_depth)
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
    return solver


def _eta(solver: Any, solver_name: str) -> np.ndarray:
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        return np.asarray(solver.compute_free_surface(), dtype=float)
    return np.asarray(solver.compute_free_surface(), dtype=float)


def _mass(solver: Any, solver_name: str, cell_area: float) -> float | None:
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        return float(np.sum(np.asarray(solver.h, dtype=float)) * cell_area)
    return None


def _max_velocity_like(solver: Any, solver_name: str) -> float:
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        u, v = solver.compute_velocity()
        return float(max(np.max(np.abs(u)), np.max(np.abs(v))))
    return float(np.max(np.abs(np.asarray(solver.eta_t, dtype=float))))


def _step_solver(solver: Any, target_cfl: float) -> float:
    dt = float(solver.suggest_dt(target_cfl=float(target_cfl)))
    solver.dt = dt
    solver.step(dt=dt, auto_dt=False)
    return dt


def _run_lake_rest(
    cases: list[dict[str, Any]],
    solver_names: Iterable[str],
    solver_cfg: Mapping[str, Any],
    sea_level_offset: float,
    n_steps: int,
    target_cfl: float,
) -> dict[str, Any]:
    by_solver: dict[str, Any] = {}
    cfg = _solver_cfg(solver_cfg, use_sponge=False)

    for solver_name in solver_names:
        rows: list[dict[str, Any]] = []
        for case in cases:
            row: dict[str, Any] = {
                "sample_id": case["sample_id"],
                "scenario_id": case["scenario_id"],
                "bathymetry_type": case["bathymetry_type"],
            }
            try:
                solver = _setup_lake_rest_solver(
                    solver_name, cfg, case, sea_level_offset
                )
                eta0 = _eta(solver, solver_name)
                max_eta_drift = 0.0
                max_velocity_like = 0.0
                dt_min = math.inf
                dt_max = -math.inf
                for _ in range(int(n_steps)):
                    dt = _step_solver(solver, target_cfl)
                    dt_min = min(dt_min, dt)
                    dt_max = max(dt_max, dt)
                    max_eta_drift = max(
                        max_eta_drift,
                        float(np.max(np.abs(_eta(solver, solver_name) - eta0))),
                    )
                    max_velocity_like = max(
                        max_velocity_like, _max_velocity_like(solver, solver_name)
                    )
                state = np.asarray(solver.get_state(), dtype=float)
                row.update(
                    {
                        "ok": bool(np.isfinite(state).all()),
                        "max_eta_drift": float(max_eta_drift),
                        "max_momentum_or_eta_t_drift": float(max_velocity_like),
                        "dt_min": float(dt_min),
                        "dt_max": float(dt_max),
                    }
                )
            except Exception as exc:
                row.update({"ok": False, "error": repr(exc)})
            rows.append(row)
        by_solver[solver_name] = {
            "num_samples": len(rows),
            "finite_pass_fraction": float(
                np.mean([bool(r.get("ok", False)) for r in rows])
            )
            if rows
            else 0.0,
            "max_eta_drift": max(
                (
                    float(r.get("max_eta_drift", float("nan")))
                    for r in rows
                    if r.get("ok")
                ),
                default=float("nan"),
            ),
            "max_momentum_or_eta_t_drift": max(
                (
                    float(r.get("max_momentum_or_eta_t_drift", float("nan")))
                    for r in rows
                    if r.get("ok")
                ),
                default=float("nan"),
            ),
            "rows": rows,
        }
    return {
        "check": "lake_at_rest",
        "n_steps": int(n_steps),
        "target_cfl": float(target_cfl),
        "solver_setup": {
            "use_sponge": False,
            "boundary": solver_cfg.get("boundary", "open"),
        },
        "by_solver": by_solver,
    }


def _run_conservation(
    cases: list[dict[str, Any]],
    solver_names: Iterable[str],
    solver_cfg: Mapping[str, Any],
    sea_level_offset: float,
    n_steps: int,
    target_cfl: float,
) -> dict[str, Any]:
    cell_area = float(solver_cfg["dx"]) * float(solver_cfg["dy"])
    cfg = _solver_cfg(
        solver_cfg, use_sponge=False, boundary="reflective", filter_strength=0.0
    )
    by_solver: dict[str, Any] = {}

    for solver_name in solver_names:
        rows: list[dict[str, Any]] = []
        for case in cases:
            row: dict[str, Any] = {
                "sample_id": case["sample_id"],
                "scenario_id": case["scenario_id"],
                "bathymetry_type": case["bathymetry_type"],
                "source_type": case["source_type"],
            }
            try:
                solver = _setup_dynamic_solver(solver_name, cfg, case, sea_level_offset)
                mass0 = _mass(solver, solver_name, cell_area)
                eta_int0 = float(np.sum(_eta(solver, solver_name)) * cell_area)
                mass_max_rel = 0.0
                eta_max_abs_change = 0.0
                dt_min = math.inf
                dt_max = -math.inf
                for _ in range(int(n_steps)):
                    dt = _step_solver(solver, target_cfl)
                    dt_min = min(dt_min, dt)
                    dt_max = max(dt_max, dt)
                    mass_now = _mass(solver, solver_name, cell_area)
                    if mass0 is not None and mass_now is not None:
                        mass_max_rel = max(
                            mass_max_rel, abs(mass_now - mass0) / max(abs(mass0), 1e-30)
                        )
                    eta_now = float(np.sum(_eta(solver, solver_name)) * cell_area)
                    eta_max_abs_change = max(
                        eta_max_abs_change, abs(eta_now - eta_int0)
                    )
                state = np.asarray(solver.get_state(), dtype=float)
                row.update(
                    {
                        "ok": bool(np.isfinite(state).all()),
                        "mass_relative_change_max": float(mass_max_rel)
                        if mass0 is not None
                        else None,
                        "free_surface_integral_abs_change_max": float(
                            eta_max_abs_change
                        ),
                        "dt_min": float(dt_min),
                        "dt_max": float(dt_max),
                    }
                )
            except Exception as exc:
                row.update({"ok": False, "error": repr(exc)})
            rows.append(row)
        by_solver[solver_name] = {
            "num_samples": len(rows),
            "finite_pass_fraction": float(
                np.mean([bool(r.get("ok", False)) for r in rows])
            )
            if rows
            else 0.0,
            "mass_relative_change_max": max(
                (
                    float(r["mass_relative_change_max"])
                    for r in rows
                    if r.get("ok") and r.get("mass_relative_change_max") is not None
                ),
                default=None,
            ),
            "free_surface_integral_abs_change_max": max(
                (
                    float(r.get("free_surface_integral_abs_change_max", float("nan")))
                    for r in rows
                    if r.get("ok")
                ),
                default=float("nan"),
            ),
            "rows": rows,
        }
    return {
        "check": "conservation_no_sponge_reflective",
        "n_steps": int(n_steps),
        "target_cfl": float(target_cfl),
        "solver_setup": {
            "use_sponge": False,
            "boundary": "reflective",
            "boussinesq_filter_strength": 0.0,
            "note": "No-sponge reflective-boundary diagnostic isolates numerical conservation; the paper dataset uses open boundaries with sponge.",
        },
        "by_solver": by_solver,
    }


def _summarize_dataset_stats(
    processed_root: Path, splits: Iterable[str]
) -> dict[str, Any]:
    by_solver: dict[str, Any] = {}
    for solver_dataset in ("hydrostatic", "muscl_hr", "boussinesq"):
        rows: list[dict[str, Any]] = []
        split_counts: dict[str, int] = {}
        for split in splits:
            path = processed_root / solver_dataset / split / "meta.jsonl"
            split_rows = _read_jsonl(path)
            rows.extend(split_rows)
            split_counts[str(split)] = len(split_rows)
        quality = Counter(str(r.get("quality_status", "unknown")) for r in rows)
        by_solver[solver_dataset] = {
            "num_samples": len(rows),
            "split_counts": split_counts,
            "quality_status_counts": dict(sorted(quality.items())),
            "quality_ok_fraction": float(quality.get("ok", 0) / max(1, len(rows))),
            "nan_count_sum": int(sum(int(r.get("nan_count", 0) or 0) for r in rows)),
            "inf_count_sum": int(sum(int(r.get("inf_count", 0) or 0) for r in rows)),
            "min_h": _summarize_values(_finite_float(r.get("min_h")) for r in rows),
            "max_abs_eta": _summarize_values(
                _finite_float(r.get("max_abs_eta")) for r in rows
            ),
            "max_abs_velocity": _summarize_values(
                _finite_float(r.get("max_abs_velocity")) for r in rows
            ),
            "max_abs_eta_over_depth": _summarize_values(
                _finite_float(r.get("max_abs_eta_over_depth")) for r in rows
            ),
            "dt_min": _summarize_values(_finite_float(r.get("dt_min")) for r in rows),
            "dt_max": _summarize_values(_finite_float(r.get("dt_max")) for r in rows),
        }
    return {"check": "dynamic_finite_state_dataset_statistics", "by_solver": by_solver}


def _summarize_boussinesq_cg(
    processed_root: Path, splits: Iterable[str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split in splits:
        split_rows = _read_jsonl(processed_root / "boussinesq" / split / "meta.jsonl")
        rows.extend(split_rows)
        split_counts[str(split)] = len(split_rows)

    with_diag = [r for r in rows if bool(r.get("has_cg_diagnostics", False))]
    failed_counts = [int(r.get("cg_failed_count", 0) or 0) for r in with_diag]
    converged_fracs = [_finite_float(r.get("cg_converged_fraction")) for r in with_diag]
    iterations = [_finite_float(r.get("max_cg_iterations")) for r in with_diag]
    residuals = [_finite_float(r.get("max_cg_residual_ratio")) for r in with_diag]
    return {
        "check": "boussinesq_cg_convergence_statistics",
        "num_samples": len(rows),
        "split_counts": split_counts,
        "samples_with_cg_diagnostics": len(with_diag),
        "samples_with_cg_failure": int(sum(1 for v in failed_counts if v > 0)),
        "cg_failed_step_count_sum": int(sum(failed_counts)),
        "cg_converged_fraction": _summarize_values(converged_fracs),
        "max_cg_iterations": _summarize_values(iterations),
        "max_cg_residual_ratio": _summarize_values(residuals),
    }


def _run_final_state(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    case: Mapping[str, Any],
    sea_level_offset: float,
    n_steps: int,
    target_cfl: float,
) -> dict[str, Any]:
    solver = _setup_dynamic_solver(solver_name, solver_cfg, case, sea_level_offset)
    dt_min = math.inf
    dt_max = -math.inf
    end_time = 0.0
    for _ in range(int(n_steps)):
        dt = _step_solver(solver, target_cfl)
        dt_min = min(dt_min, dt)
        dt_max = max(dt_max, dt)
        end_time += dt
    state = np.asarray(solver.get_state(), dtype=float)
    eta = _eta(solver, solver_name)
    return {
        "ok": bool(np.isfinite(state).all()),
        "eta": eta,
        "min_h": float(np.min(np.asarray(solver.h, dtype=float)))
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}
        else None,
        "max_abs_eta": float(np.max(np.abs(eta))),
        "max_velocity_or_eta_t": _max_velocity_like(solver, solver_name),
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "end_time": float(end_time),
    }


def _run_to_end_time(
    solver_name: str,
    solver_cfg: Mapping[str, Any],
    case: Mapping[str, Any],
    sea_level_offset: float,
    target_cfl: float,
    end_time: float,
) -> dict[str, Any]:
    solver = _setup_dynamic_solver(solver_name, solver_cfg, case, sea_level_offset)
    dt_min = math.inf
    dt_max = -math.inf
    elapsed = 0.0
    target_end_time = float(end_time)
    while elapsed < target_end_time:
        dt = float(solver.suggest_dt(target_cfl=float(target_cfl)))
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                f"Non-positive or non-finite dt while matching CFL end time: {dt}"
            )
        remaining = target_end_time - elapsed
        if dt > remaining:
            dt = remaining
        solver.dt = dt
        solver.step(dt=dt, auto_dt=False)
        dt_min = min(dt_min, dt)
        dt_max = max(dt_max, dt)
        elapsed += dt

    state = np.asarray(solver.get_state(), dtype=float)
    eta = _eta(solver, solver_name)
    return {
        "ok": bool(np.isfinite(state).all()),
        "eta": eta,
        "min_h": float(np.min(np.asarray(solver.h, dtype=float)))
        if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}
        else None,
        "max_abs_eta": float(np.max(np.abs(eta))),
        "max_velocity_or_eta_t": _max_velocity_like(solver, solver_name),
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "end_time": float(elapsed),
    }


def _run_cfl_sensitivity(
    cases: list[dict[str, Any]],
    solver_names: Iterable[str],
    solver_cfg: Mapping[str, Any],
    sea_level_offset: float,
    n_steps: int,
    cfl_values: Iterable[float],
    baseline_cfl: float,
) -> dict[str, Any]:
    cfg = _solver_cfg(solver_cfg)
    values = [float(v) for v in cfl_values]
    by_solver: dict[str, Any] = {}

    for solver_name in solver_names:
        rows: list[dict[str, Any]] = []
        for case in cases:
            final_by_cfl: dict[float, dict[str, Any]] = {}
            baseline: dict[str, Any] | None = None
            try:
                baseline = _run_final_state(
                    solver_name, cfg, case, sea_level_offset, n_steps, baseline_cfl
                )
            except Exception:
                baseline = None

            for cfl in values:
                row_base = {
                    "sample_id": case["sample_id"],
                    "scenario_id": case["scenario_id"],
                    "bathymetry_type": case["bathymetry_type"],
                    "source_type": case["source_type"],
                    "target_cfl": float(cfl),
                }
                try:
                    if baseline is not None and math.isclose(
                        cfl, baseline_cfl, rel_tol=0.0, abs_tol=1e-12
                    ):
                        result = baseline
                    elif baseline is not None:
                        result = _run_to_end_time(
                            solver_name,
                            cfg,
                            case,
                            sea_level_offset,
                            cfl,
                            float(baseline["end_time"]),
                        )
                    else:
                        raise RuntimeError("baseline CFL rollout failed")
                    final_by_cfl[cfl] = result
                    rows.append(
                        {**row_base, **{k: v for k, v in result.items() if k != "eta"}}
                    )
                except Exception as exc:
                    rows.append({**row_base, "ok": False, "error": repr(exc)})
            if baseline is None or not baseline.get("ok"):
                continue
            eta_ref = np.asarray(baseline["eta"], dtype=float)
            ref_norm = max(float(np.linalg.norm(eta_ref.ravel())), 1e-30)
            for row in rows[-len(values) :]:
                cfl = float(row["target_cfl"])
                result = final_by_cfl.get(cfl)
                if result is None or not result.get("ok"):
                    continue
                eta = np.asarray(result["eta"], dtype=float)
                diff = eta - eta_ref
                row["rel_l2_final_eta_vs_cfl_0_45"] = float(
                    np.linalg.norm(diff.ravel()) / ref_norm
                )
                row["max_abs_final_eta_diff_vs_cfl_0_45"] = float(np.max(np.abs(diff)))

        by_cfl: dict[str, Any] = {}
        for cfl in values:
            cfl_rows = [r for r in rows if float(r["target_cfl"]) == cfl]
            by_cfl[f"{cfl:.2f}"] = {
                "num_samples": len(cfl_rows),
                "finite_pass_fraction": float(
                    np.mean([bool(r.get("ok", False)) for r in cfl_rows])
                )
                if cfl_rows
                else 0.0,
                "max_abs_eta": _summarize_values(
                    _finite_float(r.get("max_abs_eta")) for r in cfl_rows
                ),
                "max_velocity_or_eta_t": _summarize_values(
                    _finite_float(r.get("max_velocity_or_eta_t")) for r in cfl_rows
                ),
                "end_time": _summarize_values(
                    _finite_float(r.get("end_time")) for r in cfl_rows
                ),
                "rel_l2_final_eta_vs_cfl_0_45": _summarize_values(
                    _finite_float(r.get("rel_l2_final_eta_vs_cfl_0_45"))
                    for r in cfl_rows
                ),
                "max_abs_final_eta_diff_vs_cfl_0_45": _summarize_values(
                    _finite_float(r.get("max_abs_final_eta_diff_vs_cfl_0_45"))
                    for r in cfl_rows
                ),
            }
        by_solver[solver_name] = {"by_cfl": by_cfl, "rows": rows}
    return {
        "check": "cfl_sensitivity",
        "n_steps": int(n_steps),
        "cfl_values": values,
        "baseline_cfl": float(baseline_cfl),
        "solver_setup": {
            "use_sponge": bool(solver_cfg.get("use_sponge", True)),
            "boundary": solver_cfg.get("boundary", "open"),
            "note": "Uses default paper boundary/sponge settings and compares final eta against CFL 0.45 at matched physical end time.",
        },
        "by_solver": by_solver,
    }


def _append_table_row(
    rows: list[dict[str, Any]],
    check: str,
    solver: str,
    samples: int,
    metric: str,
    result: Any,
    criterion: str,
) -> None:
    rows.append(
        {
            "check": check,
            "solver": solver,
            "samples": int(samples),
            "metric": metric,
            "result": "" if result is None else str(result),
            "pass_criterion": criterion,
        }
    )


def _write_summary_csv(payload: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    lake = payload.get("lake_at_rest", {}).get("by_solver", {})
    for solver, spec in lake.items():
        _append_table_row(
            rows,
            "Lake at rest",
            solver,
            spec.get("num_samples", 0),
            "max eta drift",
            f"{spec.get('max_eta_drift'):.6e}",
            "< 1e-8 target",
        )
        _append_table_row(
            rows,
            "Lake at rest",
            solver,
            spec.get("num_samples", 0),
            "finite pass rate",
            f"{spec.get('finite_pass_fraction'):.3f}",
            "1.0",
        )

    cons = payload.get("conservation", {}).get("by_solver", {})
    for solver, spec in cons.items():
        _append_table_row(
            rows,
            "Conservation",
            solver,
            spec.get("num_samples", 0),
            "max mass rel. change",
            None
            if spec.get("mass_relative_change_max") is None
            else f"{spec.get('mass_relative_change_max'):.6e}",
            "diagnostic",
        )
        _append_table_row(
            rows,
            "Conservation",
            solver,
            spec.get("num_samples", 0),
            "max eta-integral abs. change",
            f"{spec.get('free_surface_integral_abs_change_max'):.6e}",
            "diagnostic",
        )

    stats = payload.get("dynamic_dataset_statistics", {}).get("by_solver", {})
    for solver, spec in stats.items():
        _append_table_row(
            rows,
            "Dynamic rollout",
            solver,
            spec.get("num_samples", 0),
            "finite pass rate",
            f"{spec.get('quality_ok_fraction'):.3f}",
            "1.0",
        )
        _append_table_row(
            rows,
            "Dynamic rollout",
            solver,
            spec.get("num_samples", 0),
            "nan + inf count",
            int(spec.get("nan_count_sum", 0)) + int(spec.get("inf_count_sum", 0)),
            "0",
        )

    cg = payload.get("boussinesq_cg", {})
    _append_table_row(
        rows,
        "Boussinesq CG",
        "boussinesq",
        cg.get("num_samples", 0),
        "samples with CG failures",
        cg.get("samples_with_cg_failure"),
        "0",
    )
    _append_table_row(
        rows,
        "Boussinesq CG",
        "boussinesq",
        cg.get("num_samples", 0),
        "max iterations",
        cg.get("max_cg_iterations", {}).get("max"),
        "reported",
    )

    cfl = payload.get("cfl_sensitivity", {}).get("by_solver", {})
    for solver, spec in cfl.items():
        for cfl_value, cfl_spec in spec.get("by_cfl", {}).items():
            rel = cfl_spec.get("rel_l2_final_eta_vs_cfl_0_45", {})
            _append_table_row(
                rows,
                f"CFL {cfl_value}",
                solver,
                cfl_spec.get("num_samples", 0),
                "finite pass rate",
                f"{cfl_spec.get('finite_pass_fraction'):.3f}",
                "1.0",
            )
            _append_table_row(
                rows,
                f"CFL {cfl_value}",
                solver,
                cfl_spec.get("num_samples", 0),
                "mean final eta rel-L2 vs CFL 0.45",
                None if rel.get("count", 0) == 0 else f"{rel.get('mean'):.6e}",
                "diagnostic; large values indicate unstable/sensitive timestep regime",
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "check",
                "solver",
                "samples",
                "metric",
                "result",
                "pass_criterion",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 solver-validation diagnostics."
    )
    parser.add_argument("--config", default="configs/data/dataset.yaml")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument(
        "--reference-dataset",
        default="hydrostatic",
        choices=["hydrostatic", "muscl_hr", "boussinesq"],
    )
    parser.add_argument("--output-dir", default="results/solver_validation")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--lake-samples", type=int, default=24)
    parser.add_argument("--conservation-samples", type=int, default=24)
    parser.add_argument("--cfl-samples", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--target-cfl", type=float, default=None)
    parser.add_argument(
        "--cfl-values", type=float, nargs="+", default=[0.30, 0.45, 0.60]
    )
    parser.add_argument(
        "--lake-solvers", nargs="+", default=list(SOLVERS), choices=list(SOLVERS)
    )
    parser.add_argument(
        "--conservation-solvers",
        nargs="+",
        default=list(SOLVERS),
        choices=list(SOLVERS),
    )
    parser.add_argument(
        "--cfl-solvers", nargs="+", default=list(SOLVERS), choices=list(SOLVERS)
    )
    args = parser.parse_args()

    started = time.time()
    lake_solvers = _validate_solver_list(args.lake_solvers, "--lake-solvers")
    conservation_solvers = _validate_solver_list(
        args.conservation_solvers, "--conservation-solvers"
    )
    cfl_solvers = _validate_solver_list(args.cfl_solvers, "--cfl-solvers")
    cfg = load_config(args.config)
    dataset_cfg = dict(cfg.get("dataset", {}))
    solver_cfg = dict(cfg.get("solver", {}))
    if not solver_cfg:
        raise KeyError("Missing solver section in config")

    n_steps = int(
        args.n_steps if args.n_steps is not None else dataset_cfg.get("n_steps", 250)
    )
    target_cfl = float(
        args.target_cfl
        if args.target_cfl is not None
        else dataset_cfg.get("target_cfl", solver_cfg.get("cfl", 0.45))
    )
    sea_level_offset = float(dataset_cfg.get("sea_level_offset", 0.0))
    processed_root = ROOT / args.processed_root
    reference_root = processed_root / args.reference_dataset / "test"
    stats_path = processed_root / args.reference_dataset / "normalization_stats.json"

    case_count = max(
        int(args.lake_samples), int(args.conservation_samples), int(args.cfl_samples)
    )
    cases = _load_physical_cases(reference_root, stats_path, count=case_count)
    lake_cases = cases[: int(args.lake_samples)]
    conservation_cases = cases[: int(args.conservation_samples)]
    cfl_cases = cases[: int(args.cfl_samples)]

    payload: dict[str, Any] = {
        "evaluation_type": "phase3_solver_validation",
        "config_path": str(args.config),
        "processed_root": str(args.processed_root),
        "reference_dataset_for_reruns": args.reference_dataset,
        "n_steps": int(n_steps),
        "target_cfl": float(target_cfl),
        "splits": list(args.splits),
        "started_unix": float(started),
        "notes": [
            "Dynamic dataset statistics and Boussinesq CG summaries are aggregated from processed meta.jsonl files.",
            "Lake-rest, conservation, and CFL diagnostics rerun a deterministic subset using denormalized bathymetry and physical initial_depth inputs.",
        ],
        "rerun_solver_selection": {
            "lake_at_rest": list(lake_solvers),
            "conservation": list(conservation_solvers),
            "cfl_sensitivity": list(cfl_solvers),
        },
    }

    _log(
        f"[1/5] lake-at-rest reruns: solvers={lake_solvers}, samples={len(lake_cases)}, n_steps={n_steps}"
    )
    payload["lake_at_rest"] = _run_lake_rest(
        lake_cases,
        lake_solvers,
        solver_cfg=solver_cfg,
        sea_level_offset=sea_level_offset,
        n_steps=n_steps,
        target_cfl=target_cfl,
    )
    _log(
        f"[2/5] conservation reruns: solvers={conservation_solvers}, samples={len(conservation_cases)}, n_steps={n_steps}"
    )
    payload["conservation"] = _run_conservation(
        conservation_cases,
        conservation_solvers,
        solver_cfg=solver_cfg,
        sea_level_offset=sea_level_offset,
        n_steps=n_steps,
        target_cfl=target_cfl,
    )
    _log(f"[3/5] dynamic finite-state metadata: processed_root={processed_root}")
    payload["dynamic_dataset_statistics"] = _summarize_dataset_stats(
        processed_root, args.splits
    )
    _log("[4/5] Boussinesq CG metadata")
    payload["boussinesq_cg"] = _summarize_boussinesq_cg(processed_root, args.splits)
    _log(
        f"[5/5] CFL sensitivity reruns: solvers={cfl_solvers}, samples={len(cfl_cases)}, n_steps={n_steps}, cfl_values={args.cfl_values}"
    )
    payload["cfl_sensitivity"] = _run_cfl_sensitivity(
        cfl_cases,
        cfl_solvers,
        solver_cfg=solver_cfg,
        sea_level_offset=sea_level_offset,
        n_steps=n_steps,
        cfl_values=args.cfl_values,
        baseline_cfl=0.45,
    )
    payload["elapsed_s"] = float(time.time() - started)

    out_dir = ROOT / args.output_dir
    json_path = out_dir / "phase3_solver_validation.json"
    csv_path = out_dir / "phase3_solver_validation_table.csv"
    _save_json(payload, json_path)
    _write_summary_csv(payload, csv_path)

    print(f"phase3 solver validation -> {json_path}")
    print(f"summary table -> {csv_path}")
    print(f"elapsed_s={payload['elapsed_s']:.1f}")


if __name__ == "__main__":
    main()
