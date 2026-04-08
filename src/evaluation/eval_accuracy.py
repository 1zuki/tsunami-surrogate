from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation._common import (
    collect_predictions,
    compute_sample_metrics,
    compute_timestep_metrics,
    load_model,
    load_yaml,
    parse_cli,
    prepare_runtime,
    resolve_standardizer,
    save_json,
    build_dataloader,
)
from src.utils.logger import setup_logger
from src.utils.visualization import (
    save_error_histogram,
    save_metric_curves,
    save_rollout_comparison,
)


def main() -> None:
    args = parse_cli("Evaluate surrogate accuracy against held-out simulator trajectories.")
    config = load_yaml(args.config)
    device, seed_value, output_dir = prepare_runtime(config, args)
    logger = setup_logger("eval_accuracy", save_dir=output_dir)
    logger.info("Starting accuracy evaluation")
    logger.info("Device: %s | Seed: %d", device, seed_value)

    dataset, dataloader = build_dataloader(config, dataset_key="dataset")
    logger.info("Loaded dataset with %d samples", len(dataset))

    model, checkpoint_path = load_model(config, device=device, checkpoint_override=args.checkpoint)
    logger.info("Loaded model checkpoint: %s", checkpoint_path if checkpoint_path else "config-only model")

    normalization_cfg = config.get("normalization", {})
    input_standardizer = resolve_standardizer(normalization_cfg.get("input"))
    target_standardizer = resolve_standardizer(normalization_cfg.get("target"))
    output_key = config.get("evaluation", {}).get("output_key")

    pred, target, sample_ids = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        input_standardizer=input_standardizer,
        target_standardizer=target_standardizer,
        output_key=output_key,
    )
    logger.info("Collected predictions with shape %s", pred.shape)

    sample_metrics = compute_sample_metrics(pred, target)
    timestep_metrics = compute_timestep_metrics(pred, target)
    summary = {
        "num_samples": int(pred.shape[0]),
        "num_timesteps": int(pred.shape[1]),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "seed": seed_value,
        "metrics": {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in sample_metrics.items()
        },
        "timestep_metrics": {name: values.tolist() for name, values in timestep_metrics.items()},
        "sample_ids": sample_ids,
    }
    save_json(output_dir / "accuracy_summary.json", summary)
    logger.info("Saved summary JSON")

    np.savez_compressed(
        output_dir / "accuracy_predictions.npz",
        prediction=pred.astype(np.float32),
        target=target.astype(np.float32),
        sample_id=np.asarray(sample_ids),
    )
    logger.info("Saved predictions archive")

    save_metric_curves(
        metric_dict={
            "RMSE": timestep_metrics["rmse"],
            "MAE": timestep_metrics["mae"],
            "Relative L2": timestep_metrics["relative_l2"],
        },
        save_path=output_dir / "accuracy_timestep_curves.png",
        title="Accuracy metrics across rollout horizon",
    )
    save_error_histogram(
        pred - target,
        save_path=output_dir / "accuracy_error_histogram.png",
        title="Pointwise prediction error histogram",
    )

    max_examples = int(config.get("evaluation", {}).get("num_plot_examples", 4))
    example_indices: List[int]
    if pred.shape[0] <= max_examples:
        example_indices = list(range(pred.shape[0]))
    else:
        example_indices = np.linspace(0, pred.shape[0] - 1, max_examples, dtype=int).tolist()

    for example_idx in example_indices:
        save_rollout_comparison(
            target=target[example_idx],
            prediction=pred[example_idx],
            save_path=output_dir / f"example_{example_idx:03d}_rollout.png",
            title=f"Rollout comparison | sample={sample_ids[example_idx]}",
        )

    logger.info("Accuracy evaluation finished successfully")


if __name__ == "__main__":
    main()
