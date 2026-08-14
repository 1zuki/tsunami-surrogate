#!/usr/bin/env python
"""Appendix-only Boussinesq dispersion and CG diagnostic.

The diagnostic propagates small-amplitude periodic Fourier modes over constant
depth and measures the phase speed by tracking the complex Fourier coefficient
of the initialized mode. The expected curve is the discrete weakly dispersive
trend implemented by ``BoussinesqSolver``:

    omega^2 = g H k_d^2 / (1 + alpha_B H^2 k_d^2),
    k_d = 2 sin(k dx / 2) / dx.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.boussinesq import BoussinesqSolver


DEFAULT_OUTPUT_DIR = Path("results/reviewer_validation/boussinesq_dispersion")
DEFAULT_FIGURE = Path("paper/figures/boussinesq_dispersion.pdf")
DEFAULT_ALPHAS = [0.0, 1.0 / 3.0]
DEFAULT_MODES = [1, 2, 4, 8, 12, 16, 24, 32]


def _expected_frequency(
    *,
    mode: int,
    nx: int,
    dx: float,
    depth: float,
    g: float,
    alpha: float,
) -> dict[str, float]:
    domain_length = float(nx) * float(dx)
    k = 2.0 * math.pi * float(mode) / domain_length
    k_d = 2.0 * math.sin(0.5 * k * dx) / dx
    omega_sq = g * depth * k_d * k_d / (1.0 + alpha * depth * depth * k_d * k_d)
    omega = math.sqrt(max(omega_sq, 0.0))
    return {
        "k": float(k),
        "k_d": float(k_d),
        "omega": float(omega),
        "phase_speed": float(omega / k),
        "omega_sq": float(omega_sq),
    }


def _fourier_coefficient(eta: np.ndarray, basis: np.ndarray) -> complex:
    return complex(np.mean(np.asarray(eta, dtype=float) * basis))


def _phase_fit(
    times: np.ndarray, phases: np.ndarray, fit_start_fraction: float
) -> dict[str, float]:
    if times.size != phases.size or times.size < 3:
        raise ValueError("Need at least three recorded phase samples.")

    start = int(max(0, min(times.size - 3, round(times.size * fit_start_fraction))))
    t = times[start:]
    p = phases[start:]
    slope, intercept = np.polyfit(t, p, 1)
    fitted = slope * t + intercept
    residual = p - fitted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((p - np.mean(p)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return {
        "phase_slope": float(slope),
        "phase_intercept": float(intercept),
        "phase_fit_r2": float(r2),
        "phase_fit_rmse": float(math.sqrt(ss_res / max(float(t.size), 1.0))),
        "fit_start_time": float(t[0]),
        "fit_end_time": float(t[-1]),
        "num_fit_points": int(t.size),
    }


def _run_case(
    *,
    mode: int,
    alpha: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    final_time: float,
    depth: float,
    g: float,
    amplitude: float,
    record_every: int,
    linear_solver_tol: float,
    linear_solver_max_iter: int,
    fit_start_fraction: float,
) -> dict[str, Any]:
    expected = _expected_frequency(
        mode=mode, nx=nx, dx=dx, depth=depth, g=g, alpha=alpha
    )
    k = expected["k"]
    omega = expected["omega"]

    x = np.arange(nx, dtype=float) * dx
    phase = k * x[:, None]
    eta0 = amplitude * np.cos(phase) * np.ones((1, ny), dtype=float)
    eta_t0 = amplitude * omega * np.sin(phase) * np.ones((1, ny), dtype=float)
    basis = np.exp(-1j * k * x)[:, None] * np.ones((1, ny), dtype=complex)

    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=dt,
        g=g,
        alpha=alpha,
        boundary="periodic",
        use_sponge=False,
        mode="linear_constant_depth",
        depth_scale=1.0,
        filter_strength=0.0,
        linear_solver_tol=linear_solver_tol,
        linear_solver_max_iter=linear_solver_max_iter,
    )
    solver.set_bathymetry(-depth * np.ones((nx, ny), dtype=float))
    solver.set_initial_condition(eta0, eta_t0=eta_t0)

    n_steps = int(math.ceil(final_time / dt))
    times: list[float] = []
    phases: list[float] = []
    amplitudes: list[float] = []
    cg_failed_steps = 0
    cg_failed_solves = 0
    cg_step_max_iterations: list[float] = []
    cg_step_max_residual_ratio: list[float] = []

    for step_idx in range(n_steps + 1):
        if step_idx % record_every == 0 or step_idx == n_steps:
            coeff = _fourier_coefficient(solver.eta, basis)
            times.append(float(step_idx * dt))
            phases.append(float(np.angle(coeff)))
            amplitudes.append(float(abs(coeff)))

        if step_idx == n_steps:
            break

        solver.step()
        if alpha > 0.0:
            cg_failed_steps += int(not solver.last_step_cg_converged)
            cg_failed_solves += int(solver.last_step_cg_failed_count)
            cg_step_max_iterations.append(float(solver.last_step_cg_max_iterations))
            cg_step_max_residual_ratio.append(
                float(solver.last_step_cg_max_residual_ratio)
            )

    times_np = np.asarray(times, dtype=float)
    phases_np = np.unwrap(np.asarray(phases, dtype=float))
    fit = _phase_fit(times_np, phases_np, fit_start_fraction=fit_start_fraction)
    measured_omega = -float(fit["phase_slope"])
    measured_phase_speed = measured_omega / k
    expected_phase_speed = float(expected["phase_speed"])

    if alpha == 0.0:
        cg_summary = {
            "cg_path": "direct_non_cg_alpha_zero",
            "linear_solver_tol": float(linear_solver_tol),
            "linear_solver_max_iter": int(linear_solver_max_iter),
            "cg_failed_steps": 0,
            "cg_failed_solves": 0,
            "cg_max_iterations": 0,
            "cg_mean_step_max_iterations": 0.0,
            "cg_max_residual_ratio": None,
            "cg_mean_step_max_residual_ratio": None,
        }
    else:
        residuals = np.asarray(cg_step_max_residual_ratio, dtype=float)
        iterations = np.asarray(cg_step_max_iterations, dtype=float)
        cg_summary = {
            "cg_path": "conjugate_gradient",
            "linear_solver_tol": float(linear_solver_tol),
            "linear_solver_max_iter": int(linear_solver_max_iter),
            "cg_failed_steps": int(cg_failed_steps),
            "cg_failed_solves": int(cg_failed_solves),
            "cg_max_iterations": int(np.max(iterations)) if iterations.size else 0,
            "cg_mean_step_max_iterations": float(np.mean(iterations))
            if iterations.size
            else 0.0,
            "cg_max_residual_ratio": float(np.max(residuals))
            if residuals.size
            else None,
            "cg_mean_step_max_residual_ratio": float(np.mean(residuals))
            if residuals.size
            else None,
        }

    return {
        "mode": int(mode),
        "alpha_B": float(alpha),
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "dy": float(dy),
        "dt": float(dt),
        "final_time": float(n_steps * dt),
        "n_steps": int(n_steps),
        "depth": float(depth),
        "g": float(g),
        "amplitude": float(amplitude),
        "k": float(k),
        "k_d": float(expected["k_d"]),
        "omega_expected": float(expected["omega"]),
        "phase_speed_expected": expected_phase_speed,
        "omega_measured": float(measured_omega),
        "phase_speed_measured": float(measured_phase_speed),
        "phase_speed_abs_error": float(measured_phase_speed - expected_phase_speed),
        "phase_speed_rel_error": float(
            (measured_phase_speed - expected_phase_speed)
            / max(abs(expected_phase_speed), 1e-30)
        ),
        "recorded_points": int(times_np.size),
        "initial_mode_amplitude": float(amplitudes[0]),
        "final_mode_amplitude": float(amplitudes[-1]),
        "mode_amplitude_ratio": float(amplitudes[-1] / max(amplitudes[0], 1e-30)),
        **fit,
        **cg_summary,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "alpha_B",
        "k",
        "k_d",
        "omega_expected",
        "omega_measured",
        "phase_speed_expected",
        "phase_speed_measured",
        "phase_speed_rel_error",
        "cg_path",
        "linear_solver_tol",
        "linear_solver_max_iter",
        "cg_failed_steps",
        "cg_failed_solves",
        "cg_max_iterations",
        "cg_mean_step_max_iterations",
        "cg_max_residual_ratio",
        "cg_mean_step_max_residual_ratio",
        "phase_fit_r2",
        "phase_fit_rmse",
        "mode_amplitude_ratio",
        "n_steps",
        "final_time",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _plot(rows: list[dict[str, Any]], output: Path, png_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    c0 = math.sqrt(float(rows[0]["g"]) * float(rows[0]["depth"]))
    colors = {0.0: "tab:blue", 1.0 / 3.0: "tab:orange"}

    for alpha in sorted({float(r["alpha_B"]) for r in rows}):
        subset = sorted(
            [r for r in rows if float(r["alpha_B"]) == alpha],
            key=lambda r: int(r["mode"]),
        )
        modes = np.asarray([r["mode"] for r in subset], dtype=float)
        measured = np.asarray(
            [r["phase_speed_measured"] / c0 for r in subset], dtype=float
        )
        expected = np.asarray(
            [r["phase_speed_expected"] / c0 for r in subset], dtype=float
        )
        color = colors.get(alpha, None)
        label_alpha = "0" if alpha == 0.0 else f"{alpha:.3g}"
        ax.plot(
            modes,
            expected,
            linestyle="--",
            color=color,
            label=rf"expected $\alpha_B={label_alpha}$",
        )
        ax.scatter(
            modes,
            measured,
            color=color,
            s=38,
            label=rf"measured $\alpha_B={label_alpha}$",
        )

    ax.set_xlabel("Periodic Fourier mode")
    ax.set_ylabel(r"Phase speed / $\sqrt{gH}$")
    ax.set_xticks(sorted({int(r["mode"]) for r in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output, bbox_inches="tight")
    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=int, default=DEFAULT_MODES)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--dt", type=float, default=1.25e-4)
    parser.add_argument("--final-time", type=float, default=1.5)
    parser.add_argument("--amplitude", type=float, default=1.0e-3)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--linear-solver-tol", type=float, default=1.0e-12)
    parser.add_argument("--linear-solver-max-iter", type=int, default=700)
    parser.add_argument("--fit-start-fraction", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--png-output", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a shorter run for quick script checks.",
    )
    args = parser.parse_args()

    modes = [int(m) for m in args.modes]
    alphas = [float(a) for a in args.alphas]
    nx = int(args.nx)
    ny = int(args.ny)
    final_time = float(args.final_time)
    if args.smoke:
        modes = modes[:2]
        nx = min(nx, 64)
        ny = min(ny, 8)
        final_time = min(final_time, 0.25)

    if nx <= 1 or ny <= 1:
        raise ValueError("nx and ny must be greater than 1")
    if any(m <= 0 or m >= nx // 2 for m in modes):
        raise ValueError(f"modes must be in [1, {nx // 2 - 1}] for nx={nx}")

    dx = 1.0 / float(nx)
    dy = 1.0 / float(ny)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for mode in modes:
            row = _run_case(
                mode=mode,
                alpha=alpha,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
                dt=float(args.dt),
                final_time=final_time,
                depth=float(args.depth),
                g=float(args.g),
                amplitude=float(args.amplitude),
                record_every=max(1, int(args.record_every)),
                linear_solver_tol=float(args.linear_solver_tol),
                linear_solver_max_iter=int(args.linear_solver_max_iter),
                fit_start_fraction=float(args.fit_start_fraction),
            )
            rows.append(row)
            print(
                "alpha_B={alpha:.6g} mode={mode} "
                "c_meas={meas:.8f} c_exp={exp:.8f} rel_err={err:.3e} "
                "cg={cg} max_iter={it}".format(
                    alpha=row["alpha_B"],
                    mode=row["mode"],
                    meas=row["phase_speed_measured"],
                    exp=row["phase_speed_expected"],
                    err=row["phase_speed_rel_error"],
                    cg=row["cg_path"],
                    it=row["cg_max_iterations"],
                )
            )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "boussinesq_dispersion.csv"
    json_path = output_dir / "boussinesq_dispersion_summary.json"
    png_output = (
        args.png_output
        if args.png_output is not None
        else args.figure_output.with_suffix(".png")
    )

    _write_csv(rows, csv_path)
    summary = {
        "diagnostic": "constant_depth_periodic_boussinesq_dispersion",
        "interpretation": (
            "Appendix-only implementation check: measured Fourier-mode phase speeds "
            "are compared with the discrete weakly dispersive trend in BoussinesqSolver."
        ),
        "config": {
            "modes": modes,
            "alphas": alphas,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
            "dt": float(args.dt),
            "final_time": final_time,
            "depth": float(args.depth),
            "g": float(args.g),
            "amplitude": float(args.amplitude),
            "linear_solver_tol": float(args.linear_solver_tol),
            "linear_solver_max_iter": int(args.linear_solver_max_iter),
        },
        "max_abs_phase_speed_rel_error": float(
            max(abs(float(r["phase_speed_rel_error"])) for r in rows)
        ),
        "max_cg_failed_steps": int(max(int(r["cg_failed_steps"]) for r in rows)),
        "max_cg_iterations": int(max(int(r["cg_max_iterations"]) for r in rows)),
        "max_cg_residual_ratio": max(
            (
                float(r["cg_max_residual_ratio"])
                for r in rows
                if r["cg_max_residual_ratio"] is not None
            ),
            default=None,
        ),
        "rows": rows,
        "csv_path": str(csv_path),
        "figure_path": str(args.figure_output),
        "png_path": str(png_output),
    }
    _save_json(summary, json_path)
    _plot(
        rows,
        Path(args.figure_output),
        Path(png_output) if png_output is not None else None,
    )

    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_pdf={args.figure_output}")
    print(f"saved_png={png_output}")


if __name__ == "__main__":
    main()
