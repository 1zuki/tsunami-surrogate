#!/usr/bin/env python
"""Redraw archived marginal-coverage curves with reader-facing suite labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_ensemble_calibration import _plot

LABELS = {
    "validation": "Validation",
    "test": "Test",
    "source_holdout_multi_gauss": "Multi-Gaussian subset",
    "bathymetry_holdout_trench": "Trench subset",
    "source_strength_extreme_high": "High-strength subgroup",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw)
    if set(report["datasets"]) != set(LABELS):
        raise ValueError("Expected the five archived calibration suites")
    if not all(row.get("calibrated_coverage") is not None
               for suite in report["datasets"].values() for row in suite["coverage"]):
        raise ValueError("Archived calibrated coverage is required")
    plot_data = {**report, "calibration_status": "fit_validation_scalar_inflation",
                 "datasets": {LABELS[key]: value for key, value in report["datasets"].items()}}
    _plot(plot_data, args.output, args.output.with_suffix(".png"))
    if args.input.read_bytes() != raw:
        raise RuntimeError("Calibration results changed during plotting")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps({
        "source_result": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": digest(Path(__file__)),
        "plot_implementation_sha256": digest(ROOT / "scripts/eval_ensemble_calibration.py"),
        "labels": LABELS,
        "interpretation": "Unchanged archived marginal-coverage curves; labels identify within-support subgroups.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
