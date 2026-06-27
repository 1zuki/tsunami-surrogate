#!/usr/bin/env python
"""Plot representative accepted scenario pairs from the synthetic benchmark.

Three rows, each an accepted (bathymetry, source) scenario pair drawn from the
processed hydrostatic test split, chosen to span the morphology/source families:

    trench      + dipole        (elongated deep channel + uplift/subsidence lobes)
    continental + okada-like    (shelf/slope + fault-like anisotropic deformation)
    seamounts   + multi-gauss   (localized relief + multi-lobe smooth uplift)

Each row is chosen near its source family's median amplitude (not an outlier),
so a shared symmetric source scale stays readable across all three rows.

Left column is de-normalized bathymetry on a shared terrain scale [-10, -0.75];
right column is the de-normalized source template on a shared symmetric scale
[-max_abs, max_abs] (before the per-scenario strength factor is applied), so
uplift and subsidence are directly comparable across families.
"""

from __future__ import annotations

import argparse
import json
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

from src.data.dataset import ShardedTsunamiDataset


DEFAULT_DATA_DIR = "data/processed/hydrostatic/test"
DEFAULT_STATS = "data/processed/hydrostatic/normalization_stats.json"
DEFAULT_OUTPUT = "paper/figures/scenario_gallery.pdf"

BATHY_RANGE = (-10.0, -0.75)

# (bathymetry_family, source_family, human row label). One accepted pair per row.
# Source families are chosen to show clean, comparable geometry: dipole (uplift/
# subsidence lobes), okada-like (anisotropic fault deformation), multi-gauss
# (clustered localized lobes). The high-frequency "rough" family is deliberately
# avoided here -- it is noise-like, shows no readable structure, and its amplitude
# would dominate a shared color scale.
DEFAULT_ROWS = [
    ("trench", "dipole", "Trench / dipole"),
    ("continental", "okada-like", "Continental / Okada-like"),
    ("seamounts", "multi-gauss", "Seamounts / multi-Gaussian"),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _denorm(arr: np.ndarray, stats: dict[str, Any], name: str) -> np.ndarray:
    channel = stats.get("inputs", {}).get(name)
    if not channel:
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32) * float(channel["scale"]) + float(channel["offset"])


def _source_centroid_offset(source: np.ndarray) -> float:
    """Normalized distance of the source's energy centroid from the domain center.

    Weights pixel coordinates by source magnitude, so a source whose mass sits in
    a corner/edge scores high (~1) and a centered one scores low (~0). Used to
    avoid picking an off-center scenario for the illustrative gallery.
    """
    mag = np.abs(np.asarray(source, dtype=np.float64))
    total = float(mag.sum())
    h, w = mag.shape
    if total <= 0.0:
        return 1.0
    ys, xs = np.mgrid[0:h, 0:w]
    cy = float((mag * ys).sum() / total)
    cx = float((mag * xs).sum() / total)
    # Distance from center, normalized by the half-diagonal so it lands in [0, 1].
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    dist = ((cy - center_y) ** 2 + (cx - center_x) ** 2) ** 0.5
    half_diag = ((center_y) ** 2 + (center_x) ** 2) ** 0.5
    return dist / half_diag if half_diag > 0 else 0.0


def _find_pair(
    dataset: ShardedTsunamiDataset,
    bathy_family: str,
    source_family: str,
    forced_index: int | None,
    stats: dict[str, Any],
    max_offset: float = 0.18,
) -> int:
    """Return a dataset index whose sample matches the requested families.

    If forced_index is given it is used directly (after a family sanity check).
    Otherwise, among matching samples that are reasonably centered (source
    centroid offset below max_offset), return the one whose peak amplitude is
    closest to the family-median amplitude -- so the gallery shows a typical
    member of each source family rather than an amplitude outlier, which keeps a
    shared symmetric color scale meaningful across rows.
    """
    if forced_index is not None:
        item = dataset[forced_index]
        if item["bathymetry_type"] != bathy_family or item["source_type"] != source_family:
            raise ValueError(
                f"Forced index {forced_index} is "
                f"({item['bathymetry_type']}, {item['source_type']}), "
                f"expected ({bathy_family}, {source_family})."
            )
        return forced_index

    candidates: list[tuple[int, float, float]] = []  # (idx, offset, max_abs)
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["bathymetry_type"] != bathy_family or item["source_type"] != source_family:
            continue
        source = _denorm(np.asarray(item["x"], dtype=np.float32)[1], stats, "source")
        offset = _source_centroid_offset(source)
        candidates.append((idx, offset, float(np.max(np.abs(source)))))

    if not candidates:
        raise LookupError(f"No ({bathy_family}, {source_family}) pair found in {len(dataset)} samples.")

    # Median peak amplitude over ALL matching samples (the family's typical scale).
    median_amp = float(np.median([amp for _, _, amp in candidates]))

    # Prefer centered candidates; fall back to all if none clear the threshold.
    centered = [c for c in candidates if c[1] <= max_offset]
    pool = centered if centered else candidates

    # Among the pool, pick the sample whose amplitude is closest to the median.
    best_idx, _, _ = min(pool, key=lambda c: abs(c[2] - median_amp))
    return best_idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--stats", default=DEFAULT_STATS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument(
        "--indices",
        type=int,
        nargs=3,
        default=None,
        help="Optional explicit 0-based dataset indices for the three rows "
        "(must match the family pairs in DEFAULT_ROWS).",
    )
    args = p.parse_args()

    dataset = ShardedTsunamiDataset(args.data_dir)
    stats = _load_json(Path(args.stats))

    forced = list(args.indices) if args.indices is not None else [None, None, None]
    rows: list[dict[str, Any]] = []
    for (bathy_family, source_family, label), forced_index in zip(DEFAULT_ROWS, forced):
        idx = _find_pair(dataset, bathy_family, source_family, forced_index, stats)
        item = dataset[idx]
        x = np.asarray(item["x"], dtype=np.float32)
        rows.append(
            {
                "label": label,
                "index": idx,
                "scenario_id": item["scenario_id"],
                "bathymetry": _denorm(x[0], stats, "bathymetry"),
                "source": _denorm(x[1], stats, "source"),
            }
        )

    source_vmax = max(float(np.max(np.abs(r["source"]))) for r in rows)
    if source_vmax <= 0.0:
        source_vmax = 1.0

    n = len(rows)
    panel_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

    # Landscape transpose: bathymetry across the top row, the corresponding
    # source field directly below each one. Much shorter than the portrait stack,
    # so it drops into the text column without pushing body text around.
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n + 1.4, 5.4), constrained_layout=True)
    if n == 1:
        axes = axes[:, None]

    bathy_im = None
    source_im = None
    for col, r in enumerate(rows):
        ax_b = axes[0, col]
        ax_s = axes[1, col]
        bathy_im = ax_b.imshow(
            r["bathymetry"], origin="upper", cmap="terrain",
            vmin=BATHY_RANGE[0], vmax=BATHY_RANGE[1],
        )
        source_im = ax_s.imshow(
            r["source"], origin="upper", cmap="RdBu_r",
            vmin=-source_vmax, vmax=source_vmax,
        )
        for ax in (ax_b, ax_s):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_b.set_title(f"{panel_letters[col]}) {r['label']}", fontsize=10)

    # Row identifiers on the left-most panels.
    axes[0, 0].set_ylabel("Bathymetry", fontsize=10)
    axes[1, 0].set_ylabel("Initial source", fontsize=10)

    # Two shared colorbars on the right, one per row.
    bathy_cbar = fig.colorbar(bathy_im, ax=axes[0, :].tolist(), location="right",
                              fraction=0.046, pad=0.04)
    bathy_cbar.set_label("bed elevation", fontsize=8)
    bathy_cbar.ax.tick_params(labelsize=7)

    source_cbar = fig.colorbar(source_im, ax=axes[1, :].tolist(), location="right",
                               fraction=0.046, pad=0.04)
    source_cbar.set_label("source displacement", fontsize=8)
    source_cbar.ax.tick_params(labelsize=7)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    for r in rows:
        print(f"  row={r['label']:<26} idx={r['index']:>5} {r['scenario_id']}")
    print(f"source_symmetric_vmax={source_vmax:.5f}")
    print(f"saved_pdf={output_path}")
    print(f"saved_png={output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
