#!/usr/bin/env python
"""Build the sample-scaling (learning-curve) figure and appendix table.

Reads the aggregate written by ``scripts/run_sample_scaling.py`` for the small
training subsets and adds the already-trained full-data FNO as a separately
labelled comparator, then emits:

* ``--main-output`` -- a single-panel rel-L2 vs training-samples curve
  (log x-axis), the compact main-paper learning-curve figure.
* ``--appendix-fig-output`` -- three stacked panels (MAE | RMSE | rel-L2)
  sharing the log x-axis, for the appendix.
* ``--table-output`` -- a booktabs LaTeX table with MAE, RMSE, rel-L2, and
  max-err for every training-set size, matching the ``tab:accuracy`` style.

Metric conventions follow the paper's same-solver accuracy table: MAE, RMSE,
and relative L2 use physical-space values; max-err remains in normalized units.
The six subset runs use seed 42. The full-data comparator uses the ordinary
direct FNO checkpoint with seed 18 and is not connected as a seventh point in
the seed-42 learning curve.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_json, save_json


DEFAULT_RESULTS = "experiments/sample_scaling/sample_scaling_results.json"
DEFAULT_FULL_METRICS = "experiments/fno/eval/metrics.json"
DEFAULT_FULL_N = 10000

DEFAULT_MAIN_OUTPUT = "paper/figures/sample_scaling.pdf"
DEFAULT_APPENDIX_FIG_OUTPUT = "paper/figures/sample_scaling_metrics.pdf"
DEFAULT_TABLE_OUTPUT = "paper/tables/sample_scaling.tex"


def _metric_row(
    n: int,
    metrics: dict[str, Any],
    *,
    seed: int,
    role: str,
) -> dict[str, Any]:
    """Pull the four reported metrics from one eval metrics dict."""
    return {
        "train_samples": int(n),
        "seed": int(seed),
        "role": str(role),
        "mae": float(metrics["mae_physical"]),
        "rmse": float(metrics["rmse_physical"]),
        "rel_l2": float(metrics.get("rel_l2_physical", metrics["rel_l2"])),
        "max_error": float(metrics["max_error"]),
    }


def _collect_points(
    results_path: Path,
    full_metrics_path: Path,
    full_n: int,
    evaluation_run: Path | None,
) -> list[dict[str, Any]]:
    points: dict[int, dict[str, Any]] = {}

    if evaluation_run is not None:
        for metrics_path in sorted(
            (evaluation_run / "sample_scaling").glob("n_*/metrics.json")
        ):
            n = int(metrics_path.parent.name.removeprefix("n_"))
            points[n] = _metric_row(
                n,
                load_json(metrics_path),
                seed=42,
                role="sample_scaling",
            )
        run_full_metrics = evaluation_run / "direct" / "fno" / "metrics.json"
        if run_full_metrics.is_file():
            points[int(full_n)] = _metric_row(
                int(full_n),
                load_json(run_full_metrics),
                seed=18,
                role="full_data_comparator",
            )

    # The sweep aggregate may not exist yet (still running) or be partial; in
    # that case we still produce a figure from whatever points are available,
    # including just the full-data point below.
    if not points and results_path.exists():
        results = load_json(results_path)
        for row in results.get("rows", []):
            if row.get("status") != "ok":
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            n = row.get("train_samples_effective") or row.get("train_samples_requested")
            if n is None:
                continue
            points[int(n)] = _metric_row(
                int(n),
                metrics,
                seed=int(row.get("seed", 42)),
                role="sample_scaling",
            )

    if not points and full_metrics_path.exists():
        full = load_json(full_metrics_path)
        if isinstance(full, dict) and "rel_l2" in full:
            points[int(full_n)] = _metric_row(
                int(full_n),
                full,
                seed=18,
                role="full_data_comparator",
            )

    if not points:
        raise RuntimeError(
            f"No usable sample-scaling points found in {results_path} "
            f"(and no full-data metrics at {full_metrics_path})."
        )

    return [points[n] for n in sorted(points)]


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _split_points(
    points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scaling = [p for p in points if p["role"] == "sample_scaling"]
    comparator = [p for p in points if p["role"] == "full_data_comparator"]
    return scaling, comparator


def _plot_main(points: list[dict[str, Any]], output_path: Path) -> None:
    scaling, comparator = _split_points(points)
    xs = [p["train_samples"] for p in scaling]
    ys = [p["rel_l2"] for p in scaling]

    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    ax.plot(
        xs,
        ys,
        marker="o",
        color="#1f77b4",
        linewidth=1.8,
        markersize=6,
        label="seed 42 scaling runs",
    )
    if comparator:
        ax.scatter(
            [p["train_samples"] for p in comparator],
            [p["rel_l2"] for p in comparator],
            marker="*",
            s=110,
            color="#d62728",
            label="seed 18 full-data comparator",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel(r"test rel-$L_2$")
    ax.set_xticks([p["train_samples"] for p in points])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)
    _save(fig, output_path)


def _plot_appendix(points: list[dict[str, Any]], output_path: Path) -> None:
    scaling, comparator = _split_points(points)
    xs = [p["train_samples"] for p in scaling]
    specs = [
        ("mae", "MAE (phys.)", "#d62728"),
        ("rmse", "RMSE (phys.)", "#2ca02c"),
        ("rel_l2", r"rel-$L_2$", "#1f77b4"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(5.4, 7.6), sharex=True, constrained_layout=True)
    for ax, (key, label, color) in zip(axes, specs):
        ax.plot(
            xs,
            [p[key] for p in scaling],
            marker="o",
            color=color,
            linewidth=1.8,
            markersize=5,
            label="seed 42 scaling runs",
        )
        if comparator:
            ax.scatter(
                [p["train_samples"] for p in comparator],
                [p[key] for p in comparator],
                marker="*",
                s=90,
                color="#9467bd",
                label="seed 18 full-data comparator",
                zorder=3,
            )
        ax.set_ylabel(label)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("training samples")
    axes[-1].set_xticks([p["train_samples"] for p in points])
    axes[-1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[-1].tick_params(axis="x", labelrotation=45)
    axes[0].legend(fontsize=8)
    _save(fig, output_path)


def _format_table(points: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Hydrostatic FNO sample scaling. The six subset runs use seed "
        r"$42$; the $10{,}000$-sample row is the separately trained seed-$18$ "
        r"direct-FNO comparator. MAE, RMSE, and relative $L_2$ are physical-space "
        r"values; max-err remains in normalized units.}",
        r"\label{tab:sample-scaling}",
        r"\begin{tabular}{rrcccc}",
        r"    \toprule",
        r"    Train $N$ & Seed & MAE & RMSE & rel-$L_2$ & max-err \\",
        r"    \midrule",
    ]

    for p in points:
        n_str = f"{p['train_samples']:,}".replace(",", r"{,}")
        lines.append(
            f"    ${n_str}$ & ${p['seed']}$ & ${p['mae']:.5f}$ & "
            f"${p['rmse']:.5f}$ & ${p['rel_l2']:.3f}$ & "
            f"${p['max_error']:.1f}$ \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--full-metrics", default=DEFAULT_FULL_METRICS)
    p.add_argument("--full-n", type=int, default=DEFAULT_FULL_N)
    p.add_argument(
        "--evaluation-run",
        type=Path,
        default=None,
        help=(
            "Validated evaluation-run root containing sample_scaling/n_*/metrics.json "
            "and direct/fno/metrics.json. When supplied, these canonical artifacts "
            "take precedence over the mutable experiment summaries."
        ),
    )
    p.add_argument("--main-output", default=DEFAULT_MAIN_OUTPUT)
    p.add_argument("--appendix-fig-output", default=DEFAULT_APPENDIX_FIG_OUTPUT)
    p.add_argument("--table-output", default=DEFAULT_TABLE_OUTPUT)
    p.add_argument("--points-output", default="experiments/sample_scaling/sample_scaling_points.json")
    args = p.parse_args()

    points = _collect_points(
        Path(args.results),
        Path(args.full_metrics),
        int(args.full_n),
        args.evaluation_run,
    )

    _plot_main(points, Path(args.main_output))
    _plot_appendix(points, Path(args.appendix_fig_output))

    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_format_table(points), encoding="utf-8")

    save_json({"points": points}, Path(args.points_output))

    print(f"points: {[p['train_samples'] for p in points]}")
    for p in points:
        print(f"  N={p['train_samples']:>6}  rel_l2={p['rel_l2']:.4f}  mae={p['mae']:.6f}  rmse={p['rmse']:.6f}")
    print(f"saved_main={args.main_output}")
    print(f"saved_appendix_fig={args.appendix_fig_output}")
    print(f"saved_table={args.table_output}")


if __name__ == "__main__":
    main()
