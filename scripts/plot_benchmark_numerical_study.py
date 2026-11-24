#!/usr/bin/env python
"""Plot completed finite-horizon grid, time-step, and boundary sensitivities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = {
    "swe_hydrostatic": ("Hydrostatic", "#1f77b4"),
    "swe_muscl_hr": ("MUSCL-HR", "#bf5531"),
}


def _selected(
    data: dict, category: str, solver: str, left: str, reference: str, n: int
) -> list[dict]:
    rows = sorted(
        (
            row
            for row in data["comparisons"]
            if row["category"] == category
            and row["left"].endswith(f"__{solver}__{left}")
            and row["reference"].endswith(f"__{solver}__{reference}")
        ),
        key=lambda row: row["case_id"],
    )
    if len(rows) != n or len({row["case_id"] for row in rows}) != n:
        raise ValueError(
            f"Incomplete comparison roster: {solver}/{category}/{left}/{reference}"
        )
    return rows


def _range(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite numerical comparison")
    return {
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "count": int(values.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/full_horizon_numerical_study.pdf"),
    )
    args = parser.parse_args()
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if (
        not data["complete_full_planned_study"]
        or data["rollout_count"] != data["full_scope_rollout_count"]
        or len(data["rollouts"]) != data["rollout_count"]
    ):
        raise ValueError("Manuscript plots require the complete planned study")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), layout="constrained")
    times = np.asarray(data["requested_timestamps"])
    statistics = {"adjacent_grid": {}, "controls": {}, "between_swe_solvers": {}}
    for solver, (label, color) in METHODS.items():
        coarse = _selected(
            data, "spatial_self_difference", solver, "coarse", "baseline", 6
        )
        fine = _selected(data, "spatial_self_difference", solver, "baseline", "fine", 6)
        if [row["case_id"] for row in coarse] != [row["case_id"] for row in fine]:
            raise ValueError("Adjacent-grid cases are not paired")
        a = np.asarray([row["trajectory_rmse"] for row in coarse])
        b = np.asarray([row["trajectory_rmse"] for row in fine])
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError("Adjacent-grid log plot requires positive differences")
        for left, right in zip(a, b):
            axes[0, 0].plot(
                [0, 1], [left, right], color=color, alpha=0.3, linewidth=0.8, marker="."
            )
        axes[0, 0].plot(
            [0, 1],
            [np.median(a), np.median(b)],
            color=color,
            linewidth=2,
            marker="s",
            label=f"{label} median",
        )
        statistics["adjacent_grid"][solver] = {
            "coarse_baseline_rmse": _range(a),
            "baseline_fine_rmse": _range(b),
            "paired_contraction_ratio": _range(a / b),
            "contracting_case_count": int(np.sum(a > b)),
        }
        temporal = np.asarray([row["per_time_rmse"] for row in fine])
        axes[0, 1].plot(times, np.median(temporal, axis=0), color=color, label=label)
        axes[0, 1].fill_between(
            times,
            np.min(temporal, axis=0),
            np.max(temporal, axis=0),
            color=color,
            alpha=0.12,
        )
        statistics["controls"][solver] = {}
        for i, (category, left, right, n) in enumerate(
            (
                ("half_common_cfl", "common_cfl", "half_common_cfl", 6),
                ("sponge_on_off", "baseline", "no_sponge", 3),
                ("radiation_to_open", "no_sponge", "open_no_sponge", 3),
                (
                    "buffer_extent_and_extended_exterior",
                    "no_sponge",
                    "extended_no_sponge",
                    3,
                ),
            )
        ):
            rows = _selected(data, category, solver, left, right, n)
            values = np.asarray([row["trajectory_rmse"] for row in rows])
            x = i + (-0.19 if solver == "swe_hydrostatic" else 0.19)
            axes[1, 1].vlines(
                x, min(values), max(values), color=color,
                linewidth=0.9, alpha=0.55, zorder=1,
            )
            # Deterministic horizontal offsets keep all cases visible.
            offsets = np.linspace(-0.07, 0.07, n)
            offsets[np.isclose(offsets, 0)] = 0.045
            axes[1, 1].scatter(
                x + offsets, values,
                s=20, marker="o", facecolor="white", edgecolor=color,
                linewidth=1, zorder=3,
                label=label if i == 0 else None,
            )
            median = np.median(values)
            axes[1, 1].hlines(
                median, x - 0.055, x + 0.055,
                color=color, linewidth=2, zorder=4,
            )
            statistics["controls"][solver][category] = _range(values)
    for variant, grid in (("coarse", 64), ("baseline", 128), ("fine", 256)):
        rows = sorted(
            (
                row
                for row in data["comparisons"]
                if row["category"] == "between_swe_solvers"
                and row["left"].endswith(f"__{variant}")
            ),
            key=lambda row: row["case_id"],
        )
        if len(rows) != 6:
            raise ValueError("Incomplete between-solver grid roster")
        statistics["between_swe_solvers"][str(grid)] = _range(
            [row["trajectory_rmse"] for row in rows]
        )
    paired = [
        [
            row["trajectory_rmse"]
            for row in sorted(
                (
                    r
                    for r in data["comparisons"]
                    if r["category"] == "between_swe_solvers"
                    and r["left"].endswith(f"__{variant}")
                ),
                key=lambda row: row["case_id"],
            )
        ]
        for variant in ("coarse", "baseline", "fine")
    ]
    for profile in np.asarray(paired).T:
        axes[1, 0].plot(
            [64, 128, 256], profile, color="0.65", linewidth=0.7, marker="."
        )
    axes[1, 0].plot(
        [64, 128, 256],
        np.median(paired, axis=1),
        color="#6b4699",
        marker="s",
        label="Median of six cases",
    )
    axes[0, 0].set(
        title="(a) Adjacent-grid differences",
        ylabel="Trajectory RMSE",
        yscale="log",
        xticks=[0, 1],
        xticklabels=[r"$64\rightarrow128$", r"$128\rightarrow256$"],
        xlabel="Solver core grids",
    )
    axes[0, 1].set(
        title="(b) Difference through the full horizon",
        ylabel="Per-time RMSE",
        xlabel="Elapsed benchmark time",
    )
    axes[1, 0].set(
        title="(c) Difference between SWE references",
        ylabel="Trajectory RMSE",
        xlabel="Solver core grid",
        xticks=[64, 128, 256],
    )
    axes[1, 1].set(
        title="(d) Controlled setting changes",
        ylabel="Trajectory RMSE",
        yscale="log",
        xticks=range(4),
        xticklabels=[
            "Half CFL\n(n=6)",
            "Sponge\n(n=3)",
            "Boundary\n(n=3)",
            "Extent\n(n=3)",
        ],
        xlim=(-0.5, 3.5),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.legend(frameon=False)
    axes[0, 1].legend(frameon=False, title="Lines: medians; shade: range", title_fontsize=8)
    axes[1, 1].legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0, -0.21),
        ncol=2, fontsize=8, borderaxespad=0,
        title="Dots: cases; ticks: medians; lines: range", title_fontsize=7,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.output.with_suffix(".png"), dpi=250)
    plt.close(fig)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Numerical summary changed while plotting")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_result": str(args.input.resolve()),
                "source_sha256": digest,
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "statistics": statistics,
                "interpretation": "Finite-horizon sensitivity on fixed raster inputs under adaptive stepping. Ranges describe tested cases and are not confidence intervals or formal convergence orders.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
