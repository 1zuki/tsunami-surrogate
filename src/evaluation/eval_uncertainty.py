from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch

from src.evaluation._common import load_checkpoint_and_model, make_eval_loader, save_json
from src.models import build_model
from src.models.uncertainty import ensemble_predict, gaussian_nll, interval_coverage, mc_dropout_predict
from src.utils.visualization import plot_uncertainty


def _prepare_ensemble(config: dict, checkpoint_paths: list[str], device: torch.device):
    models = []
    for ckpt_path in checkpoint_paths:
        state = torch.load(ckpt_path, map_location="cpu")
        model = build_model(config)
        model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictive uncertainty with MC dropout or ensembles.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="mc_dropout", choices=["mc_dropout", "ensemble"])
    parser.add_argument("--ensemble-checkpoints", type=str, nargs="*", default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config, model, stats, device, state = load_checkpoint_and_model(args.config, args.checkpoint)
    loader = make_eval_loader(config, split=args.split, return_meta=True)
    normalize_targets = bool(config.get("normalization", {}).get("normalize_targets", True))
    num_samples = int(config.get("uncertainty", {}).get("num_samples", 20))

    all_nll = []
    all_cov68 = []
    all_cov95 = []
    all_abs_err = []
    all_std = []
    first_example = None

    ensemble_models = None
    if args.method == "ensemble":
        checkpoints = args.ensemble_checkpoints or [str(Path(config.get("paths", {}).get("checkpoint_dir", "results/default/checkpoints")) / "best.pt")]
        ensemble_models = _prepare_ensemble(config, checkpoints, device)

    for batch in loader:
        x, y, meta = batch
        x = x.to(device)
        y = y.to(device)
        if args.method == "mc_dropout":
            mean, std, stacked = mc_dropout_predict(model, x, n_samples=num_samples)
        else:
            mean, std, stacked = ensemble_predict(ensemble_models, x)
        if normalize_targets:
            mean = mean * stats.target_std + stats.target_mean
            std = std * stats.target_std
            y = y * stats.target_std + stats.target_mean
        nll = gaussian_nll(y, mean, std)
        cov68 = interval_coverage(y, mean, std, z_value=1.0)
        cov95 = interval_coverage(y, mean, std, z_value=1.96)
        abs_err = torch.abs(mean - y)

        all_nll.append(float(nll.cpu()))
        all_cov68.append(float(cov68.cpu()))
        all_cov95.append(float(cov95.cpu()))
        all_abs_err.append(float(abs_err.mean().cpu()))
        all_std.append(float(std.mean().cpu()))

        if first_example is None:
            first_example = (
                mean[0].detach().cpu().numpy(),
                std[0].detach().cpu().numpy(),
                y[0].detach().cpu().numpy(),
            )

    report = {
        "method": args.method,
        "nll": float(np.mean(all_nll)),
        "coverage_68": float(np.mean(all_cov68)),
        "coverage_95": float(np.mean(all_cov95)),
        "mean_abs_error": float(np.mean(all_abs_err)),
        "mean_pred_std": float(np.mean(all_std)),
        "error_std_correlation_proxy": float(np.corrcoef(np.asarray(all_abs_err), np.asarray(all_std))[0, 1]) if len(all_abs_err) > 1 else 0.0,
    }

    out_dir = Path(config.get("paths", {}).get("output_root", "results/default_run")) / f"eval_uncertainty_{args.method}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, out_dir / "uncertainty.json")
    if first_example is not None:
        mean, std, truth = first_example
        plot_uncertainty(mean, std, truth, out_dir / "uncertainty_example.png")
    print(report)


if __name__ == "__main__":
    main()
