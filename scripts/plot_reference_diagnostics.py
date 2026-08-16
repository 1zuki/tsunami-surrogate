#!/usr/bin/env python
"""Plot reference-gap scale and cross-reference discrepancy diagnostics.

The figures are deterministic post-hoc views of the checksum-bound artifacts
from a validated evaluation run. They do not rerun a model or numerical solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "evaluation_runs/final-v2-paper-full-r1"
DEFAULT_OUTPUT_DIR = ROOT / "paper/figures"

SOLVER_LABELS = {
    "hydrostatic": "Hydrostatic",
    "muscl_hr": "MUSCL-HR",
    "boussinesq": "Boussinesq",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validated_artifacts(run_root: Path) -> dict[str, str]:
    completion_path = run_root / "completion_manifest.json"
    completion = _load_json(completion_path)
    if completion.get("status") != "validated":
        raise ValueError(f"Evaluation run is not validated: {completion_path}")
    return {
        str(row["path"]): str(row["sha256"])
        for row in completion.get("artifacts", [])
        if isinstance(row, dict) and "path" in row and "sha256" in row
    }


def _load_declared_json(
    run_root: Path,
    relative_path: str,
    declared_hashes: dict[str, str],
) -> tuple[dict[str, Any], str]:
    expected = declared_hashes.get(relative_path)
    if expected is None:
        raise KeyError(
            f"Validated completion manifest does not declare {relative_path}"
        )
    path = run_root / relative_path
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"Validated artifact hash mismatch for {path}: {observed} != {expected}"
        )
    return _load_json(path), observed


def _save_figure(fig: Any, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "tsunami-surrogate",
            "Producer": "Matplotlib",
        },
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path, png_path]


def _solver_gap(
    solver_gap: dict[str, Any],
    solver_a: str,
    solver_b: str,
) -> float:
    matches = [
        row
        for row in solver_gap.get("pairs", [])
        if {str(row["solver_a"]), str(row["solver_b"])}
        == {solver_a, solver_b}
    ]
    if len(matches) != 2:
        raise ValueError(
            f"Expected two directional rows for {solver_a}/{solver_b}, "
            f"found {len(matches)}"
        )
    values = {
        float(row["metrics"]["global_field_rmse"])
        for row in matches
    }
    if len(values) != 1:
        raise ValueError(
            f"Directional global RMSE differs for {solver_a}/{solver_b}: {values}"
        )
    return values.pop()


def _plot_gap_scale(
    solver_gap: dict[str, Any],
    direct_metrics: dict[str, dict[str, Any]],
    output: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    hydro_muscl = _solver_gap(solver_gap, "hydrostatic", "muscl_hr")
    hydro_bouss = _solver_gap(solver_gap, "hydrostatic", "boussinesq")
    muscl_bouss = _solver_gap(solver_gap, "muscl_hr", "boussinesq")

    comparisons = [
        {
            "label": "Hydrostatic--MUSCL-HR\nvs Hydrostatic FNO",
            "solver_gap_rmse": hydro_muscl,
            "surrogate_rmse": float(direct_metrics["fno"]["rmse_physical"]),
        },
        {
            "label": "Hydrostatic--MUSCL-HR\nvs Hydrostatic F-FNO",
            "solver_gap_rmse": hydro_muscl,
            "surrogate_rmse": float(direct_metrics["ffno"]["rmse_physical"]),
        },
        {
            "label": "Hydrostatic--Boussinesq\nvs Boussinesq FNO",
            "solver_gap_rmse": hydro_bouss,
            "surrogate_rmse": float(
                direct_metrics["fno_boussinesq"]["rmse_physical"]
            ),
        },
        {
            "label": "MUSCL-HR--Boussinesq\nvs Boussinesq FNO",
            "solver_gap_rmse": muscl_bouss,
            "surrogate_rmse": float(
                direct_metrics["fno_boussinesq"]["rmse_physical"]
            ),
        },
    ]
    for row in comparisons:
        row["gap_to_surrogate_ratio"] = (
            row["solver_gap_rmse"] / row["surrogate_rmse"]
        )

    y = np.arange(len(comparisons))[::-1]
    surrogate = np.asarray(
        [row["surrogate_rmse"] for row in comparisons],
        dtype=np.float64,
    )
    gap = np.asarray(
        [row["solver_gap_rmse"] for row in comparisons],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7.1, 4.2), constrained_layout=True)
    for y_pos, left, right in zip(y, surrogate, gap, strict=True):
        ax.plot(
            [left, right],
            [y_pos, y_pos],
            color="#7f7f7f",
            linewidth=1.4,
            zorder=1,
        )
    ax.scatter(
        surrogate,
        y,
        marker="o",
        s=55,
        color="#1f77b4",
        label="same-reference surrogate RMSE",
        zorder=3,
    )
    ax.scatter(
        gap,
        y,
        marker="s",
        s=55,
        color="#d62728",
        label="raw solver-gap RMSE",
        zorder=3,
    )
    for y_pos, row in zip(y, comparisons, strict=True):
        ax.annotate(
            f"{row['gap_to_surrogate_ratio']:.1f}x",
            (row["solver_gap_rmse"], y_pos),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("denormalized global field RMSE")
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in comparisons], fontsize=8)
    ax.grid(True, axis="x", which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)
    paths = _save_figure(fig, output)
    return paths, comparisons


def _plot_cross_reference_forest(
    cross_reference: dict[str, Any],
    output: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    preferred_order = [
        ("hydrostatic", "muscl_hr"),
        ("muscl_hr", "hydrostatic"),
        ("hydrostatic", "boussinesq"),
        ("muscl_hr", "boussinesq"),
        ("boussinesq", "hydrostatic"),
        ("boussinesq", "muscl_hr"),
    ]
    by_direction = {
        (str(row["model_solver"]), str(row["benchmark_solver"])): row
        for row in cross_reference.get("directions", [])
    }
    if set(by_direction) != set(preferred_order):
        raise ValueError(
            "Cross-reference artifact does not contain the expected six directions"
        )
    rows = [by_direction[key] for key in preferred_order]

    point = np.asarray(
        [float(row["rho"]["point_estimate"]) for row in rows],
        dtype=np.float64,
    )
    lower = np.asarray(
        [float(row["rho"]["ci_lower"]) for row in rows],
        dtype=np.float64,
    )
    upper = np.asarray(
        [float(row["rho"]["ci_upper"]) for row in rows],
        dtype=np.float64,
    )
    y = np.arange(len(rows))[::-1]

    colors = []
    for lo, hi in zip(lower, upper, strict=True):
        if lo > 1.0:
            colors.append("#d62728")
        elif hi < 1.0:
            colors.append("#1f77b4")
        else:
            colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(7.1, 4.35), constrained_layout=True)
    for y_pos, value, lo, hi, color in zip(
        y,
        point,
        lower,
        upper,
        colors,
        strict=True,
    ):
        ax.errorbar(
            value,
            y_pos,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
            linewidth=1.4,
            zorder=3,
        )
        ax.annotate(
            f"{value:.3f} [{lo:.3f}, {hi:.3f}]",
            (hi, y_pos),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.text(
        1.0,
        len(rows) - 0.35,
        "raw solver-gap scale",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{SOLVER_LABELS[str(row['model_solver'])]} "
            f"$\\rightarrow$ {SOLVER_LABELS[str(row['benchmark_solver'])]}"
            for row in rows
        ],
        fontsize=8,
    )
    padding = 0.035
    ax.set_xlim(float(lower.min() - padding), float(upper.max() + 0.08))
    ax.set_xlabel(
        r"cross-reference discrepancy ratio $\rho$ "
        r"(95\% paired-bootstrap interval)"
    )
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    paths = _save_figure(fig, output)
    return paths, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_root = args.evaluation_run.resolve()
    output_dir = args.output_dir.resolve()
    declared_hashes = _validated_artifacts(run_root)

    source_specs = {
        "solver_gap": "paper_evidence/reference_analysis/solver_gap.json",
        "cross_reference": (
            "paper_evidence/reference_analysis/cross_reference.json"
        ),
        "fno": "direct/fno/metrics.json",
        "ffno": "direct/ffno/metrics.json",
        "fno_muscl_hr": "direct/fno_muscl_hr/metrics.json",
        "fno_boussinesq": "direct/fno_boussinesq/metrics.json",
    }
    loaded: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, str]] = []
    for key, relative_path in source_specs.items():
        payload, digest = _load_declared_json(
            run_root,
            relative_path,
            declared_hashes,
        )
        loaded[key] = payload
        source_rows.append({"path": relative_path, "sha256": digest})

    output_dir.mkdir(parents=True, exist_ok=True)
    gap_paths, gap_comparisons = _plot_gap_scale(
        loaded["solver_gap"],
        {
            key: loaded[key]
            for key in ("fno", "ffno", "fno_muscl_hr", "fno_boussinesq")
        },
        output_dir / "solver_gap_vs_surrogate_error.pdf",
    )
    forest_paths, forest_rows = _plot_cross_reference_forest(
        loaded["cross_reference"],
        output_dir / "cross_reference_ratio_forest.pdf",
    )

    manifest = {
        "schema_id": "tsunami-surrogate.reference-diagnostic-figures.v1",
        "evaluation_run": run_root.relative_to(ROOT).as_posix(),
        "script": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source_artifacts": source_rows,
        "figures": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in [*gap_paths, *forest_paths]
        ],
        "gap_scale_comparisons": gap_comparisons,
        "cross_reference_directions": [
            {
                "model_solver": str(row["model_solver"]),
                "benchmark_solver": str(row["benchmark_solver"]),
                "rho": dict(row["rho"]),
            }
            for row in forest_rows
        ],
        "interpretation": (
            "The gap-scale figure compares selected same-reference surrogate "
            "controls with raw pairwise solver gaps in the same denormalized "
            "RMSE units. The forest plot is benchmark-relative and does not "
            "establish physical superiority."
        ),
    }
    manifest_path = output_dir / "reference_diagnostics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "[reference-diagnostics] "
        f"figures=2 sources={len(source_rows)} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
