from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from src.evaluation._common import (
    build_dataloader,
    collect_predictions,
    compute_sample_metrics,
    compute_timestep_metrics,
    load_model,
    load_yaml,
    parse_cli,
    prepare_runtime,
    resolve_standardizer,
    save_json,
)
from src.utils.logger import setup_logger
from src.utils.visualization import save_generalization_bar_chart, save_metric_curves


def main() -> None:
    args = parse_cli("Evaluate surrogate generalization across multiple splits or datasets.")
    config = load_yaml(args.config)
    device, seed_value, output_dir = prepare_runtime(config, args)
    logger = setup_logger("eval_generalization", save_dir=output_dir)
    logger.info("Starting generalization evaluation")

    model, checkpoint_path = load_model(config, device=device, checkpoint_override=args.checkpoint)
    logger.info("Loaded model checkpoint: %s", checkpoint_path if checkpoint_path else "config-only model")

    normalization_cfg = config.get("normalization", {})
    input_standardizer = resolve_standardizer(normalization_cfg.get("input"))
    target_standardizer = resolve_standardizer(normalization_cfg.get("target"))
    output_key = config.get("evaluation", {}).get("output_key")

    generalization_cfg = config.get("generalization", {})
    suites = generalization_cfg.get("suites")
    if not suites:
        raise ValueError("Config must contain generalization.suites with at least one dataset suite.")

    suite_results: Dict[str, Dict[str, Any]] = {}
    rmse_by_suite: Dict[str, float] = {}
    rel_l2_by_suite: Dict[str, float] = {}

    for suite in suites:
        label = str(suite.get("label") or suite.get("name") or suite.get("path"))
        suite_config = deepcopy(config)
        suite_dataset_cfg = deepcopy(config.get("dataset", {}))
        for key, value in suite.items():
            if key not in {"label", "name"}:
                suite_dataset_cfg[key] = value
        suite_config["dataset"] = suite_dataset_cfg

        dataset, dataloader = build_dataloader(suite_config, dataset_key="dataset")
        logger.info("Evaluating suite '%s' with %d samples", label, len(dataset))

        pred, target, _ = collect_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            input_standardizer=input_standardizer,
            target_standardizer=target_standardizer,
            output_key=output_key,
        )
        sample_metrics = compute_sample_metrics(pred, target)
        timestep_metrics = compute_timestep_metrics(pred, target)
        summary = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in sample_metrics.items()
        }
        suite_results[label] = {
            "num_samples": int(pred.shape[0]),
            "metrics": summary,
            "timestep_metrics": {name: values.tolist() for name, values in timestep_metrics.items()},
        }
        rmse_by_suite[label] = summary["rmse"]["mean"]
        rel_l2_by_suite[label] = summary["relative_l2"]["mean"]

        save_metric_curves(
            metric_dict={
                "RMSE": timestep_metrics["rmse"],
                "MAE": timestep_metrics["mae"],
                "Relative L2": timestep_metrics["relative_l2"],
            },
            save_path=output_dir / f"{label}_timestep_metrics.png",
            title=f"Per-timestep metrics | {label}",
        )

    summary_payload = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "seed": seed_value,
        "suites": suite_results,
    }
    save_json(output_dir / "generalization_summary.json", summary_payload)
    logger.info("Saved suite summaries")

    save_generalization_bar_chart(
        values=rmse_by_suite,
        save_path=output_dir / "generalization_rmse_bar.png",
        title="Mean RMSE by evaluation suite",
        ylabel="RMSE",
    )
    save_generalization_bar_chart(
        values=rel_l2_by_suite,
        save_path=output_dir / "generalization_relative_l2_bar.png",
        title="Mean relative L2 by evaluation suite",
        ylabel="Relative L2",
    )
    logger.info("Generalization evaluation finished successfully")


if __name__ == "__main__":
    main()
