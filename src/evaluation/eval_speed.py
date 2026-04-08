from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from src.evaluation._common import (
    benchmark_model,
    benchmark_simulator,
    build_dataloader,
    load_model,
    load_yaml,
    parse_cli,
    prepare_runtime,
    resolve_standardizer,
    save_json,
)
from src.utils.logger import setup_logger
from src.utils.visualization import save_generalization_bar_chart


def _extract_representative_batches(dataloader, device: torch.device, limit: int) -> List[torch.Tensor]:
    batches: List[torch.Tensor] = []
    for index, batch in enumerate(dataloader):
        if index >= limit:
            break
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        batches.append(inputs)
    if not batches:
        raise ValueError("Speed evaluation dataset is empty.")
    return batches


def main() -> None:
    args = parse_cli("Benchmark surrogate inference speed against the simulator.")
    config = load_yaml(args.config)
    device, seed_value, output_dir = prepare_runtime(config, args)
    logger = setup_logger("eval_speed", save_dir=output_dir)
    logger.info("Starting speed evaluation")

    dataset, dataloader = build_dataloader(config, dataset_key="dataset")
    logger.info("Loaded dataset with %d samples", len(dataset))

    model, checkpoint_path = load_model(config, device=device, checkpoint_override=args.checkpoint)
    logger.info("Loaded model checkpoint: %s", checkpoint_path if checkpoint_path else "config-only model")

    normalization_cfg = config.get("normalization", {})
    input_standardizer = resolve_standardizer(normalization_cfg.get("input"))
    target_standardizer = resolve_standardizer(normalization_cfg.get("target"))
    output_key = config.get("evaluation", {}).get("output_key")

    speed_cfg = config.get("speed", {})
    num_batches = int(speed_cfg.get("num_batches", 3))
    model_warmup = int(speed_cfg.get("model_warmup", 10))
    model_repeats = int(speed_cfg.get("model_repeats", 50))
    simulator_warmup = int(speed_cfg.get("simulator_warmup", 2))
    simulator_repeats = int(speed_cfg.get("simulator_repeats", 5))

    representative_batches = _extract_representative_batches(dataloader, device=device, limit=num_batches)

    model_runs: List[Dict[str, float]] = []
    for batch_idx, batch_inputs in enumerate(representative_batches):
        stats = benchmark_model(
            model=model,
            batch_inputs=batch_inputs,
            input_standardizer=input_standardizer,
            target_standardizer=target_standardizer,
            output_key=output_key,
            warmup=model_warmup,
            repeats=model_repeats,
        )
        stats["batch_size"] = int(batch_inputs.shape[0])
        stats["batch_index"] = int(batch_idx)
        model_runs.append(stats)
        logger.info("Model batch %d mean latency: %.6f s", batch_idx, stats["mean_seconds"])

    model_mean_latency = float(np.mean([item["mean_seconds"] for item in model_runs]))
    model_mean_throughput = float(
        np.mean([item["batch_size"] / max(item["mean_seconds"], 1e-12) for item in model_runs])
    )

    simulator_target = speed_cfg.get("simulator_callable")
    simulator_summary: Dict[str, Any] = {"available": False}
    if simulator_target:
        simulator_runs: List[Dict[str, float]] = []
        cpu_sample = representative_batches[0][:1].detach().cpu()
        for repeat_idx in range(int(speed_cfg.get("simulator_num_cases", 1))):
            sim_stats = benchmark_simulator(
                simulator_target=simulator_target,
                sample_inputs=cpu_sample,
                simulator_cfg=speed_cfg,
                warmup=simulator_warmup,
                repeats=simulator_repeats,
            )
            sim_stats["case_index"] = repeat_idx
            simulator_runs.append(sim_stats)
        simulator_summary = {
            "available": True,
            "callable": simulator_target,
            "runs": simulator_runs,
            "mean_seconds": float(np.mean([item["mean_seconds"] for item in simulator_runs])),
            "std_seconds": float(np.std([item["mean_seconds"] for item in simulator_runs])),
        }
        logger.info("Simulator mean latency: %.6f s", simulator_summary["mean_seconds"])
    else:
        logger.warning("No speed.simulator_callable found in config. Speedup will be unavailable.")

    results: Dict[str, Any] = {
        "seed": seed_value,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "device": str(device),
        "model": {
            "runs": model_runs,
            "mean_seconds": model_mean_latency,
            "mean_samples_per_second": model_mean_throughput,
        },
        "simulator": simulator_summary,
    }

    if simulator_summary.get("available"):
        speedup = simulator_summary["mean_seconds"] / max(model_mean_latency, 1e-12)
        results["speedup_vs_simulator"] = float(speedup)
        logger.info("Estimated speedup vs simulator: %.2fx", speedup)

    save_json(output_dir / "speed_summary.json", results)

    chart_values = {"Surrogate": model_mean_latency}
    if simulator_summary.get("available"):
        chart_values["Simulator"] = float(simulator_summary["mean_seconds"])
    save_generalization_bar_chart(
        values=chart_values,
        save_path=output_dir / "speed_latency_bar.png",
        title="Mean latency comparison",
        ylabel="Seconds",
    )
    logger.info("Speed evaluation finished successfully")


if __name__ == "__main__":
    main()
