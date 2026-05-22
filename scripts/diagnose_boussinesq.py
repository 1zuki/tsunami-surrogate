#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen.generate_bathymetry import BathymetryGenerator
from src.data_gen.generate_sources import SourceGenerator
from src.solver.boussinesq import BoussinesqSolver


@dataclass
class ScenarioData:
    bathymetry: np.ndarray
    source_field: np.ndarray
    bathymetry_type: str
    source_type: str
    source_strength: float
    sample_seed: int


def _seed_for_sample(run_seed: int, sample_idx: int) -> int:
    return int(run_seed + sample_idx * 10007)


def _parse_strength_range(value: Any) -> Tuple[float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 2:
        raise ValueError("dataset.source_strength_range must be [min, max]")
    lo, hi = float(arr[0]), float(arr[1])
    if lo > hi:
        raise ValueError("dataset.source_strength_range requires min <= max")

    return lo, hi


def _scenario_cache_paths(dataset_cfg: Dict[str, Any], sample_idx: int) -> Tuple[Path, Path]:
    bathy_dir = Path(str(dataset_cfg.get("bathymetry_dir", "data/bathymetry_bouss")))
    source_dir = Path(str(dataset_cfg.get("source_dir", "data/sources_bouss")))

    return (
        bathy_dir / f"sample_{sample_idx:06d}.npz",
        source_dir / f"sample_{sample_idx:06d}.npz",
    )


def _load_scenario_from_cache(dataset_cfg: Dict[str, Any], sample_idx: int) -> ScenarioData | None:
    bathy_path, source_path = _scenario_cache_paths(dataset_cfg, sample_idx)
    if not (bathy_path.exists() and source_path.exists()):
        return None

    with np.load(bathy_path) as bathy_npz:
        bathymetry = np.asarray(bathy_npz["bathymetry"], dtype=np.float32)
        bathymetry_type = str(np.asarray(bathy_npz["bathymetry_type"]).reshape(-1)[0])
        sample_seed = int(np.asarray(bathy_npz.get("sample_seed", np.array([0], dtype=np.int64))).reshape(-1)[0])

    with np.load(source_path) as src_npz:
        source_field = np.asarray(src_npz["source_field"], dtype=np.float32)
        source_type = str(np.asarray(src_npz["source_type"]).reshape(-1)[0])
        source_strength = float(np.asarray(src_npz["source_strength"]).reshape(-1)[0])

    return ScenarioData(
        bathymetry=bathymetry,
        source_field=source_field,
        bathymetry_type=bathymetry_type,
        source_type=source_type,
        source_strength=source_strength,
        sample_seed=sample_seed,
    )


def _generate_scenario(
    dataset_cfg: Dict[str, Any],
    config_cfg: Dict[str, Any],
    sample_idx: int,
) -> ScenarioData:
    bathy_cfg_path = Path(str(config_cfg.get("bathymetry", "configs/data/bathymetry_boussinesq.yaml")))
    source_cfg_path = Path(str(config_cfg.get("source", "configs/data/source_boussinesq.yaml")))
    run_seed_raw = dataset_cfg.get("seed", 42)
    run_seed = int(42 if run_seed_raw is None else run_seed_raw)
    sample_seed = _seed_for_sample(run_seed, sample_idx)

    bathy_generator = BathymetryGenerator(str(bathy_cfg_path))
    bathy_generator.rng = np.random.default_rng([sample_seed, 11])
    bathymetry, bathymetry_type = bathy_generator.generate()

    source_generator = SourceGenerator(str(source_cfg_path))
    source_generator.rng = np.random.default_rng([sample_seed, 23])
    source_field, source_type = source_generator.generate()

    lo, hi = _parse_strength_range(dataset_cfg.get("source_strength_range", [0.01, 0.05]))
    strength_rng = np.random.default_rng([sample_seed, 37])
    source_strength = float(strength_rng.uniform(lo, hi))

    return ScenarioData(
        bathymetry=np.asarray(bathymetry, dtype=np.float32),
        source_field=np.asarray(source_field, dtype=np.float32),
        bathymetry_type=str(bathymetry_type),
        source_type=str(source_type),
        source_strength=source_strength,
        sample_seed=sample_seed,
    )


def _make_solver(solver_cfg: Dict[str, Any]) -> BoussinesqSolver:
    return BoussinesqSolver(
        nx=int(solver_cfg["nx"]),
        ny=int(solver_cfg["ny"]),
        dx=float(solver_cfg["dx"]),
        dy=float(solver_cfg["dy"]),
        dt=float(solver_cfg["dt"]),
        g=float(solver_cfg.get("g", 9.81)),
        cfl=float(solver_cfg.get("cfl", 0.35)),
        alpha=float(solver_cfg.get("alpha", 1.0 / 3.0)),
        min_depth=float(solver_cfg.get("min_depth", 1e-3)),
        sea_level_offset=float(solver_cfg.get("sea_level_offset", 0.0)),
        depth_scale=float(solver_cfg.get("depth_scale", 1.0)),
        boundary=solver_cfg.get("boundary", "open"),
        mode=solver_cfg.get("mode", "linear_variable_depth"),
        use_sponge=solver_cfg.get("use_sponge", None),
        sponge_width=int(solver_cfg.get("sponge_width", 20)),
        sponge_min_factor=float(solver_cfg.get("sponge_min_factor", 0.9)),
        filter_strength=float(solver_cfg.get("filter_strength", 0.0)),
        linear_solver_tol=float(solver_cfg.get("linear_solver_tol", 1e-8)),
        linear_solver_max_iter=int(solver_cfg.get("linear_solver_max_iter", 80)),
        check_finite=bool(solver_cfg.get("check_finite", True)),
    )


def _compute_frame_metrics(
    eta: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
) -> Dict[str, float]:
    eta = np.asarray(eta, dtype=np.float64)
    abs_eta = np.abs(eta)
    max_abs_eta = float(np.max(abs_eta))
    l2_eta = float(np.sqrt(np.sum(eta * eta) * dx * dy))
    center_eta = float(eta[eta.shape[0] // 2, eta.shape[1] // 2])

    weights = eta * eta
    w_sum = float(np.sum(weights))

    if w_sum > 0.0:
        cx = float(np.sum(X * weights) / w_sum)
        cy = float(np.sum(Y * weights) / w_sum)
        energy_radius = float(np.sqrt(np.sum(((X - cx) ** 2 + (Y - cy) ** 2) * weights) / w_sum))
    else:
        cx = float(np.nan)
        cy = float(np.nan)
        energy_radius = float(np.nan)

    argmax_i, argmax_j = np.unravel_index(np.argmax(abs_eta), abs_eta.shape)
    argmax_x = float(x_grid[argmax_i])
    argmax_y = float(y_grid[argmax_j])

    return {
        "max_abs_eta": max_abs_eta,
        "l2_eta": l2_eta,
        "energy_center_x": cx,
        "energy_center_y": cy,
        "energy_radius": energy_radius,
        "center_eta": center_eta,
        "argmax_i": float(argmax_i),
        "argmax_j": float(argmax_j),
        "argmax_x": argmax_x,
        "argmax_y": argmax_y,
    }


def _plot_timeseries(times: np.ndarray, metrics: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    ax = axes.ravel()
    ax[0].plot(times, metrics["max_abs_eta"], lw=1.5)
    ax[0].set_title("max_abs_eta(t)")
    ax[1].plot(times, metrics["l2_eta"], lw=1.5)
    ax[1].set_title("L2_eta(t)")
    ax[2].plot(times, metrics["energy_radius"], lw=1.5)
    ax[2].set_title("energy_radius(t)")
    ax[3].plot(times, metrics["center_eta"], lw=1.5)
    ax[3].set_title("center_eta(t)")
    ax[4].plot(times, metrics["cg_iterations"], lw=1.5)
    ax[4].set_title("CG iterations")
    residual_ratio = metrics["cg_final_residual"] / np.maximum(metrics["cg_initial_residual"], 1e-30)
    ax[5].plot(times, residual_ratio, lw=1.5)
    ax[5].set_title("CG final/initial residual")

    for a in ax:
        a.set_xlabel("time [s]")
        a.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fields(bathymetry: np.ndarray, eta0: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(bathymetry.T, origin="lower", cmap="terrain")
    axes[0].set_title("Bathymetry")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    vmax = float(np.max(np.abs(eta0))) if np.max(np.abs(eta0)) > 0 else 1.0
    im1 = axes[1].imshow(eta0.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Initial Eta (source)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_frames(frames: np.ndarray, times: np.ndarray, out_path: Path, max_frames: int) -> None:
    n = int(frames.shape[0])
    k = max(1, min(max_frames, n))
    idx = np.linspace(0, n - 1, num=k, dtype=int)
    ncols = min(4, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    vmax = float(np.max(np.abs(frames))) if np.max(np.abs(frames)) > 0 else 1.0

    for a in axes_arr:
        a.axis("off")

    for ax, frame_idx in zip(axes_arr, idx.tolist()):
        im = ax.imshow(frames[frame_idx].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"t={times[frame_idx]:.4f}s")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes_arr.tolist(), shrink=0.9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Boussinesq propagation diagnostic with quantitative time-series + frame plots.")
    p.add_argument("--config", type=str, default="configs/data/dataset_boussinesq.yaml")
    p.add_argument("--sample-index", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="results/boussinesq_diagnostic")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--target-cfl", type=float, default=None)
    p.add_argument("--auto-dt", dest="auto_dt", action="store_true")
    p.add_argument("--fixed-dt", dest="auto_dt", action="store_false")
    p.add_argument("--regenerate", action="store_true", help="Ignore cached bathymetry/source and regenerate this scenario.")
    p.add_argument("--max-frames-plot", type=int, default=8)
    p.set_defaults(auto_dt=None)
    args = p.parse_args()

    if args.sample_index < 1:
        raise ValueError("--sample-index must be >= 1")

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_cfg = dict(cfg.get("dataset", {}))
    solver_cfg = dict(cfg.get("solver", {}))
    config_cfg = dict(cfg.get("configs", {}))
    if not solver_cfg:
        raise KeyError("Missing solver section in config.")

    scenario = None
    if not args.regenerate:
        scenario = _load_scenario_from_cache(dataset_cfg, args.sample_index)

    if scenario is None:
        scenario = _generate_scenario(dataset_cfg, config_cfg, args.sample_index)

    solver = _make_solver(solver_cfg)
    eta0 = np.asarray(scenario.source_field * scenario.source_strength, dtype=np.float32)
    eta_t0 = np.zeros_like(eta0, dtype=np.float32)
    solver.set_bathymetry(scenario.bathymetry)
    solver.set_initial_condition(eta0, eta_t0=eta_t0)

    n_steps = int(args.n_steps if args.n_steps is not None else dataset_cfg.get("n_steps", 250))
    save_every = int(args.save_every if args.save_every is not None else dataset_cfg.get("save_every", 5))

    if n_steps <= 0 or save_every <= 0:
        raise ValueError("n_steps and save_every must be positive.")

    auto_dt = bool(args.auto_dt) if args.auto_dt is not None else bool(dataset_cfg.get("auto_dt", True))
    target_cfl = float(args.target_cfl if args.target_cfl is not None else dataset_cfg.get("target_cfl", solver.cfl))

    x_grid = np.arange(solver.nx, dtype=np.float64) * float(solver.dx)
    y_grid = np.arange(solver.ny, dtype=np.float64) * float(solver.dy)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    times: list[float] = []
    max_abs_eta: list[float] = []
    l2_eta: list[float] = []
    energy_center_x: list[float] = []
    energy_center_y: list[float] = []
    energy_radius: list[float] = []
    center_eta: list[float] = []
    argmax_i: list[float] = []
    argmax_j: list[float] = []
    argmax_x: list[float] = []
    argmax_y: list[float] = []
    cg_iterations: list[float] = []
    cg_initial_residual: list[float] = []
    cg_final_residual: list[float] = []
    cg_converged: list[float] = []
    frames: list[np.ndarray] = []

    def record_snapshot(current_time: float) -> None:
        eta = np.asarray(solver.eta, dtype=np.float32)
        frames.append(eta.copy())
        m = _compute_frame_metrics(
            eta=eta,
            x_grid=x_grid,
            y_grid=y_grid,
            X=X,
            Y=Y,
            dx=float(solver.dx),
            dy=float(solver.dy),
        )
        times.append(float(current_time))
        max_abs_eta.append(m["max_abs_eta"])
        l2_eta.append(m["l2_eta"])
        energy_center_x.append(m["energy_center_x"])
        energy_center_y.append(m["energy_center_y"])
        energy_radius.append(m["energy_radius"])
        center_eta.append(m["center_eta"])
        argmax_i.append(m["argmax_i"])
        argmax_j.append(m["argmax_j"])
        argmax_x.append(m["argmax_x"])
        argmax_y.append(m["argmax_y"])
        cg_iterations.append(float(solver.last_cg_iterations))
        cg_initial_residual.append(float(solver.last_cg_initial_residual))
        cg_final_residual.append(float(solver.last_cg_final_residual))
        cg_converged.append(1.0 if solver.last_cg_converged else 0.0)

    current_time = 0.0
    record_snapshot(current_time)

    for step_idx in range(n_steps):
        dt = solver.suggest_dt(target_cfl=target_cfl) if auto_dt else float(solver.dt)
        solver.step(dt=dt, auto_dt=False)
        current_time += float(dt)

        if (step_idx + 1) % save_every == 0:
            record_snapshot(current_time)

    if not frames:
        record_snapshot(current_time)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_arr = np.stack(frames, axis=0).astype(np.float32)
    metrics_np = {
        "time": np.asarray(times, dtype=np.float64),
        "max_abs_eta": np.asarray(max_abs_eta, dtype=np.float64),
        "l2_eta": np.asarray(l2_eta, dtype=np.float64),
        "energy_center_x": np.asarray(energy_center_x, dtype=np.float64),
        "energy_center_y": np.asarray(energy_center_y, dtype=np.float64),
        "energy_radius": np.asarray(energy_radius, dtype=np.float64),
        "center_eta": np.asarray(center_eta, dtype=np.float64),
        "argmax_i": np.asarray(argmax_i, dtype=np.float64),
        "argmax_j": np.asarray(argmax_j, dtype=np.float64),
        "argmax_x": np.asarray(argmax_x, dtype=np.float64),
        "argmax_y": np.asarray(argmax_y, dtype=np.float64),
        "cg_iterations": np.asarray(cg_iterations, dtype=np.float64),
        "cg_initial_residual": np.asarray(cg_initial_residual, dtype=np.float64),
        "cg_final_residual": np.asarray(cg_final_residual, dtype=np.float64),
        "cg_converged": np.asarray(cg_converged, dtype=np.float64),
    }

    np.save(out_dir / "frames_eta.npy", frames_arr)
    np.save(out_dir / "bathymetry.npy", np.asarray(scenario.bathymetry, dtype=np.float32))
    np.save(out_dir / "source_field.npy", np.asarray(scenario.source_field, dtype=np.float32))
    np.save(out_dir / "eta0.npy", np.asarray(eta0, dtype=np.float32))
    np.savez_compressed(out_dir / "timeseries.npz", **metrics_np)

    _plot_fields(np.asarray(scenario.bathymetry), np.asarray(eta0), out_dir / "fields.png")
    _plot_timeseries(metrics_np["time"], metrics_np, out_dir / "timeseries.png")
    _plot_frames(frames_arr, metrics_np["time"], out_dir / "frames_gallery.png", max_frames=int(args.max_frames_plot))

    summary = {
        "config_path": str(cfg_path),
        "sample_index": int(args.sample_index),
        "sample_seed": int(scenario.sample_seed),
        "source_type": scenario.source_type,
        "bathymetry_type": scenario.bathymetry_type,
        "source_strength": float(scenario.source_strength),
        "n_steps": int(n_steps),
        "save_every": int(save_every),
        "auto_dt": bool(auto_dt),
        "target_cfl": float(target_cfl),
        "num_saved_frames": int(frames_arr.shape[0]),
        "final_time": float(metrics_np["time"][-1]),
        "final_max_abs_eta": float(metrics_np["max_abs_eta"][-1]),
        "final_l2_eta": float(metrics_np["l2_eta"][-1]),
        "final_energy_radius": float(metrics_np["energy_radius"][-1]),
        "final_center_eta": float(metrics_np["center_eta"][-1]),
        "final_argmax_ij": [
            int(round(float(metrics_np["argmax_i"][-1]))),
            int(round(float(metrics_np["argmax_j"][-1]))),
        ],
        "last_cg_iterations": float(metrics_np["cg_iterations"][-1]),
        "last_cg_initial_residual": float(metrics_np["cg_initial_residual"][-1]),
        "last_cg_final_residual": float(metrics_np["cg_final_residual"][-1]),
        "all_finite": bool(np.isfinite(frames_arr).all()),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[diagnose_boussinesq] saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
