from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from src.evaluation._common import load_checkpoint_and_model, make_eval_loader, run_inference, save_json
from src.data_gen.dataset import denormalize_inputs
from src.utils.visualization import plot_prediction_vs_truth, plot_time_series_at_points, save_wave_animation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictive accuracy on a tsunami dataset split.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config, model, stats, device, state = load_checkpoint_and_model(args.config, args.checkpoint)
    loader = make_eval_loader(config, split=args.split, return_meta=True)
    normalize_targets = bool(config.get("normalization", {}).get("normalize_targets", True))
    metrics, preds, targets, metas, examples = run_inference(model, loader, device, stats, normalize_targets)

    out_dir = Path(config.get("paths", {}).get("output_root", "results/default_run")) / f"eval_accuracy_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)

    vis_n = min(int(config.get("visualization", {}).get("n_samples", 4)), len(examples))
    for i in range(vis_n):
        x, truth, pred, meta = examples[i]
        sample_x = x[0]
        if bool(config.get("normalization", {}).get("normalize_inputs", True)):
            sample_x = denormalize_inputs(sample_x, stats)
        bathy = sample_x[0]
        disturbance = sample_x[1]
        plot_prediction_vs_truth(bathy, disturbance, truth[0], pred[0], out_dir / f"sample_{i}.png")
        h, w = truth[0].shape[-2], truth[0].shape[-1]
        points = [(h // 4, w // 4), (h // 2, w // 2), (3 * h // 4, 3 * w // 4)]
        plot_time_series_at_points(truth[0], pred[0], points, out_dir / f"sample_{i}_timeseries.png")
        save_wave_animation(pred[0], out_dir / f"sample_{i}_pred.gif", bathymetry=bathy)


if __name__ == "__main__":
    main()
