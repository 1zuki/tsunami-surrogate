from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from src.evaluation._common import load_checkpoint_and_model, make_eval_loader, run_inference, save_json
from src.training.metrics import compute_metrics_np


def _collect_meta_array(metas, key: str):
    if not metas:
        return None
    chunks = []
    for meta in metas:
        if key in meta:
            chunks.append(np.asarray(meta[key]))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generalization across bathymetry regimes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config, model, stats, device, state = load_checkpoint_and_model(args.config, args.checkpoint)
    loader = make_eval_loader(config, split=args.split, return_meta=True)
    normalize_targets = bool(config.get("normalization", {}).get("normalize_targets", True))
    overall_metrics, preds, targets, metas, _ = run_inference(model, loader, device, stats, normalize_targets)

    roughness = _collect_meta_array(metas, "roughness")
    mean_depth = _collect_meta_array(metas, "mean_depth")
    source_type = _collect_meta_array(metas, "source_type")
    n_bins = int(config.get("evaluation", {}).get("generalization_bins", 4))

    report = {"overall": overall_metrics, "roughness_bins": [], "depth_bins": [], "source_type": []}

    if roughness is not None:
        edges = np.quantile(roughness, np.linspace(0, 1, n_bins + 1))
        for i in range(n_bins):
            mask = (roughness >= edges[i]) & (roughness <= edges[i + 1] if i == n_bins - 1 else roughness < edges[i + 1])
            if np.any(mask):
                metrics = compute_metrics_np(preds[mask], targets[mask])
                report["roughness_bins"].append({"bin": i, "low": float(edges[i]), "high": float(edges[i + 1]), **metrics})

    if mean_depth is not None:
        edges = np.quantile(mean_depth, np.linspace(0, 1, n_bins + 1))
        for i in range(n_bins):
            mask = (mean_depth >= edges[i]) & (mean_depth <= edges[i + 1] if i == n_bins - 1 else mean_depth < edges[i + 1])
            if np.any(mask):
                metrics = compute_metrics_np(preds[mask], targets[mask])
                report["depth_bins"].append({"bin": i, "low": float(edges[i]), "high": float(edges[i + 1]), **metrics})

    if source_type is not None:
        for st in np.unique(source_type.astype(int)):
            mask = source_type.astype(int) == int(st)
            metrics = compute_metrics_np(preds[mask], targets[mask])
            report["source_type"].append({"source_type": int(st), **metrics})

    out_dir = Path(config.get("paths", {}).get("output_root", "results/default_run")) / f"eval_generalization_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, out_dir / "generalization.json")
    print(report)


if __name__ == "__main__":
    main()
