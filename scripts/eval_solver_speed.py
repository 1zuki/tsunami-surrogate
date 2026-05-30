#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen.simulate_dataset import (
    _make_boussinesq_solver_from_cfg,
    _make_hydrostatic_solver_from_cfg,
    _make_muscl_solver_from_cfg,
)
from src.utils.config import load_config
from src.utils.device import hardware_info, resolve_device
from src.utils.io import save_json


def _canonical_solver_name(name: str) -> str:
    raw = str(name).strip().lower()
    if raw == "swe_muscl":
        return "swe_muscl_hr"
    return raw


def _sample_path(root: Path, idx: int) -> Path:
    return root / f"sample_{idx:06d}.npz"


def _discover_sample_ids(bathy_dir: Path, source_dir: Path) -> List[int]:
    patt = re.compile(r"^sample_(\d{6})\.npz$")
    bathy_ids: set[int] = set()
    source_ids: set[int] = set()

    for p in bathy_dir.glob("sample_*.npz"):
        m = patt.match(p.name)
        if m is not None:
            bathy_ids.add(int(m.group(1)))
    for p in source_dir.glob("sample_*.npz"):
        m = patt.match(p.name)
        if m is not None:
            source_ids.add(int(m.group(1)))

    return sorted(bathy_ids & source_ids)


def _load_scenarios(
    sample_ids: Iterable[int],
    bathy_dir: Path,
    source_dir: Path,
    dtype: np.dtype,
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for idx in sample_ids:
        bathy_path = _sample_path(bathy_dir, idx)
        source_path = _sample_path(source_dir, idx)
        if not bathy_path.exists():
            raise FileNotFoundError(bathy_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        with np.load(bathy_path) as z:
            bathymetry = np.asarray(z["bathymetry"], dtype=dtype)
            bathy_type = str(np.asarray(z["bathymetry_type"]).reshape(-1)[0]) if "bathymetry_type" in z else "unknown"
        with np.load(source_path) as z:
            source_field = np.asarray(z["source_field"], dtype=dtype)
            source_type = str(np.asarray(z["source_type"]).reshape(-1)[0]) if "source_type" in z else "unknown"
            strength = float(np.asarray(z["source_strength"]).reshape(-1)[0]) if "source_strength" in z else 1.0

        rows.append(
            {
                "sample_index": int(idx),
                "bathymetry": bathymetry,
                "source_field": source_field,
                "source_strength": strength,
                "bathymetry_type": bathy_type,
                "source_type": source_type,
            }
        )
    elapsed = time.perf_counter() - t0
    return rows, float(elapsed)


def _make_solver(solver_name: str, solver_cfg: Dict[str, Any]):
    if solver_name == "swe_hydrostatic":
        return _make_hydrostatic_solver_from_cfg(solver_cfg)
    if solver_name == "swe_muscl_hr":
        return _make_muscl_solver_from_cfg(solver_cfg)
    if solver_name == "boussinesq":
        return _make_boussinesq_solver_from_cfg(solver_cfg)
    raise ValueError(f"Unsupported solver '{solver_name}'")


def _setup_solver_for_scenario(
    solver_name: str,
    solver_cfg: Dict[str, Any],
    sea_level_offset: float,
    scenario: Dict[str, Any],
):
    bathymetry = np.asarray(scenario["bathymetry"])
    source_field = np.asarray(scenario["source_field"])
    source_strength = float(scenario["source_strength"])

    eta0 = source_strength * source_field
    rest_depth = np.maximum(-bathymetry + float(sea_level_offset), 0.0)
    h0 = np.maximum(rest_depth + eta0, 0.0)

    solver = _make_solver(solver_name, solver_cfg)
    solver.set_bathymetry(bathymetry)
    if solver_name in {"swe_hydrostatic", "swe_muscl_hr"}:
        solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))
    else:
        solver.set_initial_condition(eta0, eta_t0=np.zeros_like(eta0))
    return solver


def _rollout_solver(solver: Any, n_steps: int, auto_dt: bool, target_cfl: float) -> None:
    for _ in range(int(n_steps)):
        if auto_dt:
            dt = float(solver.suggest_dt(target_cfl=float(target_cfl)))
            solver.dt = dt
        else:
            dt = float(solver.dt)
        solver.step(dt=dt, auto_dt=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark NumPy solver rollout speed on cached scenarios.")
    p.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    p.add_argument("--solver", required=True, choices=["swe_hydrostatic", "swe_muscl", "swe_muscl_hr", "boussinesq"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--precision", choices=["float64", "float32"], default="float64")
    p.add_argument("--sample-ids", type=int, nargs="+", default=None, help="Explicit 1-based sample indices.")
    p.add_argument("--max-samples", type=int, default=8, help="Used only when --sample-ids is omitted.")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--auto-dt", type=str, default=None, help="Override auto_dt (true/false).")
    p.add_argument("--target-cfl", type=float, default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--bathymetry-dir", type=str, default=None)
    p.add_argument("--source-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="results/speed/solver_speed.json")
    args = p.parse_args()

    dev = resolve_device(args.device)
    if dev.type != "cpu":
        raise RuntimeError(
            "Current reference solvers are NumPy CPU implementations. "
            "CUDA solver timing is not supported."
        )

    cfg = load_config(args.config)
    ds_cfg = dict(cfg.get("dataset", {}))
    solver_cfg = dict(cfg.get("solver", {}))
    if not solver_cfg:
        raise KeyError("Dataset config missing `solver` section.")

    solver_name = _canonical_solver_name(args.solver)
    if solver_name not in {"swe_hydrostatic", "swe_muscl_hr", "boussinesq"}:
        raise ValueError(f"Unsupported solver: {solver_name}")

    n_steps = int(args.n_steps if args.n_steps is not None else ds_cfg.get("n_steps", 200))
    if n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    target_cfl = float(args.target_cfl if args.target_cfl is not None else ds_cfg.get("target_cfl", 0.45))
    auto_dt_raw = args.auto_dt if args.auto_dt is not None else ds_cfg.get("auto_dt", True)
    if isinstance(auto_dt_raw, bool):
        auto_dt = auto_dt_raw
    else:
        txt = str(auto_dt_raw).strip().lower()
        if txt not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("--auto-dt must be true/false")
        auto_dt = txt in {"true", "1", "yes"}

    sea_level_offset = float(ds_cfg.get("sea_level_offset", 0.0))
    bathy_dir = Path(args.bathymetry_dir or ds_cfg.get("bathymetry_dir", "data/bathymetry"))
    source_dir = Path(args.source_dir or ds_cfg.get("source_dir", "data/sources"))
    if not bathy_dir.exists():
        raise FileNotFoundError(bathy_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    if args.sample_ids:
        sample_ids = sorted({int(x) for x in args.sample_ids})
    else:
        discovered = _discover_sample_ids(bathy_dir, source_dir)
        if not discovered:
            raise RuntimeError("No cached sample ids shared between bathymetry and source caches.")
        sample_ids = discovered[: max(1, int(args.max_samples))]

    requested_precision = str(args.precision)
    dtype = np.float64 if requested_precision == "float64" else np.float32
    scenarios, scenario_load_total_s = _load_scenarios(sample_ids, bathy_dir=bathy_dir, source_dir=source_dir, dtype=dtype)
    if not scenarios:
        raise RuntimeError("No scenarios loaded for solver benchmark.")

    warmup = max(0, int(args.warmup))
    repeats = max(1, int(args.repeats))

    # warmup
    for i in range(warmup):
        s = scenarios[i % len(scenarios)]
        solver = _setup_solver_for_scenario(solver_name, solver_cfg, sea_level_offset=sea_level_offset, scenario=s)
        _rollout_solver(solver, n_steps=n_steps, auto_dt=auto_dt, target_cfl=target_cfl)

    setup_total_s = 0.0
    rollout_total_s = 0.0
    state_dtype = None

    for _ in range(repeats):
        for s in scenarios:
            t0 = time.perf_counter()
            solver = _setup_solver_for_scenario(
                solver_name=solver_name,
                solver_cfg=solver_cfg,
                sea_level_offset=sea_level_offset,
                scenario=s,
            )
            if state_dtype is None:
                try:
                    state_dtype = str(np.asarray(solver.get_state()).dtype)
                except Exception:
                    state_dtype = "unknown"
            t1 = time.perf_counter()
            _rollout_solver(solver, n_steps=n_steps, auto_dt=auto_dt, target_cfl=target_cfl)
            t2 = time.perf_counter()

            setup_total_s += float(t1 - t0)
            rollout_total_s += float(t2 - t1)

    total_timed_s = float(setup_total_s + rollout_total_s)
    num_samples_timed = int(len(scenarios) * repeats)
    per_sample_total = total_timed_s / float(max(1, num_samples_timed))
    per_sample_rollout = rollout_total_s / float(max(1, num_samples_timed))
    per_sample_setup = setup_total_s / float(max(1, num_samples_timed))

    payload: Dict[str, Any] = {
        "evaluation_type": "solver_speed_benchmark",
        "config_path": str(args.config),
        "method": solver_name,
        "solver_name": solver_name,
        "device": "cpu",
        "precision_requested": requested_precision,
        "precision_actual": state_dtype if state_dtype is not None else "unknown",
        "num_scenarios": int(len(scenarios)),
        "sample_ids": [int(s["sample_index"]) for s in scenarios],
        "n_steps": int(n_steps),
        "auto_dt": bool(auto_dt),
        "target_cfl": float(target_cfl),
        "num_warmup": int(warmup),
        "num_repeats": int(repeats),
        "num_samples_timed": int(num_samples_timed),
        "scenario_load_time_total_s": float(scenario_load_total_s),
        "scenario_load_time_per_sample_s": float(scenario_load_total_s / max(1, len(scenarios))),
        "solver_setup_time_total_s": float(setup_total_s),
        "solver_setup_time_per_sample_s": float(per_sample_setup),
        "rollout_time_total_s": float(rollout_total_s),
        "rollout_time_per_sample_s": float(per_sample_rollout),
        "time_total_mean_s": float(total_timed_s),
        "time_per_sample_mean_s": float(per_sample_total),
        "samples_per_second": float(num_samples_timed / max(total_timed_s, 1e-12)),
        "hardware": hardware_info(dev),
        "notes": "speedup denominator should use rollout_time_per_sample_s when comparing against model inference.",
    }

    if requested_precision == "float32" and payload["precision_actual"] != "float32":
        payload["precision_warning"] = (
            "Requested float32, but solver state is float64. "
            "Current NumPy reference solvers use float64 internals."
        )

    output_path = Path(args.output)
    save_json(payload, output_path)
    print(payload)


if __name__ == "__main__":
    main()
