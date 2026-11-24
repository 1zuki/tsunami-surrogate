#!/usr/bin/env python
"""Draw whole-trajectory coverage and scenario-level error versus retention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures/trajectory_reliability.pdf")
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if (
        data["status"] != "completed_full_splits"
        or data["groups"]["all"]["scenario_count"] != 2500
    ):
        raise ValueError("The manuscript figure requires the complete test split")
    if "after reversing training normalization" not in data.get("risk_metric", ""):
        raise ValueError("Use physical relative errors after reversing target normalization")
    levels = [0.5, 0.8, 0.9, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    plt.rcParams.update({"font.size": 9})
    for key, label, color, marker in (
        ("all", "All test scenarios", "#1f77b4", "o"),
        ("source_strength>=0.285", "High-strength subgroup", "#b63e3e", "s"),
    ):
        group = data["groups"][key]
        axes[0].plot(
            levels,
            [group["simultaneous_coverage"][str(p)] for p in levels],
            marker=marker,
            color=color,
            label=label,
            linewidth=1.6,
            markersize=4,
        )
    axes[0].plot(
        [0.45, 1], [0.45, 1], "--", color="0.45", linewidth=1, label="Nominal coverage"
    )
    axes[0].set(
        xlim=(0.47, 0.98),
        ylim=(0, 1),
        xlabel="Nominal whole-trajectory coverage",
        ylabel="Observed whole-trajectory coverage",
        title="(a) Simultaneous reliability",
    )
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    rows = data["risk_coverage"]
    x = np.asarray([row["retained_fraction"] for row in rows]) * 100
    axes[1].plot(
        x,
        [row["mean_rel_l2"] for row in rows],
        color="#6b4699",
        marker="o",
        linewidth=1.6,
    )
    axes[1].axhline(
        data["groups"]["all"]["mean_rel_l2"], color="0.45", linestyle="--", linewidth=1
    )
    axes[1].set(
        xlim=(20, 105),
        xlabel="Retained test scenarios (%)",
        ylabel=r"Mean per-scenario relative $L_2$",
        title="(b) Retaining lower predicted spread",
    )
    axes[1].set_xticks([25, 50, 75, 100])
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.output.with_suffix(".png"), dpi=250)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_result": str(args.input.resolve()),
                "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "plot_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "interpretation": "Empirical numerical-reference trajectory coverage and retrospective scenario ranking",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
