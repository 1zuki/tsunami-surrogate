from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import norm

from src.evaluation._common import (
    build_dataloader,
    collect_predictions,
    load_model,
    load_yaml,
    model_forward,
    parse_cli,
    predict_batch,
    prepare_runtime,
    resolve_standardizer,
    save_json,
)
from src.utils.logger import setup_logger
from src.utils.visualization import (
    save_reliability_curve,
    save_scatter,
    save_uncertainty_panel,
)


def _enable_dropout_only(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


@torch.inference_mode()
def _extract_direct_mean_and_variance(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    input_standardizer,
    target_standardizer,
    output_key: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized_inputs = input_standardizer.normalize(inputs)
    outputs = model(normalized_inputs)

    mean_tensor: Optional[torch.Tensor] = None
    variance_tensor: Optional[torch.Tensor] = None

    if isinstance(outputs, Mapping):
        if output_key and output_key in outputs and isinstance(outputs[output_key], torch.Tensor):
            mean_tensor = outputs[output_key]
        else:
            for candidate in ("mean", "prediction", "pred", "output"):
                if candidate in outputs and isinstance(outputs[candidate], torch.Tensor):
                    mean_tensor = outputs[candidate]
                    break
        if "variance" in outputs:
            variance_tensor = outputs["variance"]
        elif "var" in outputs:
            variance_tensor = outputs["var"]
        elif "std" in outputs:
            variance_tensor = outputs["std"] ** 2
        elif "logvar" in outputs:
            variance_tensor = torch.exp(outputs["logvar"])
    elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        mean_tensor = outputs[0]
        second = outputs[1]
        if not isinstance(mean_tensor, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise TypeError("Direct uncertainty mode expects tensor outputs.")
        if second.min().item() < 0:
            variance_tensor = torch.exp(second)
        else:
            variance_tensor = second
    else:
        raise TypeError(
            "Direct uncertainty mode expects model outputs as dict or tuple containing mean and variance information."
        )

    if mean_tensor is None or variance_tensor is None:
        raise ValueError("Could not extract mean and variance from model output.")

    mean_tensor = target_standardizer.denormalize(mean_tensor)
    if target_standardizer.is_active():
        scale = target_standardizer.std.to(device=variance_tensor.device, dtype=variance_tensor.dtype)
        variance_tensor = variance_tensor * (scale ** 2)

    variance_tensor = variance_tensor.clamp_min(1e-12)
    if mean_tensor.ndim == 5 and mean_tensor.shape[2] == 1:
        mean_tensor = mean_tensor[:, :, 0, ...]
        variance_tensor = variance_tensor[:, :, 0, ...]
    return mean_tensor, variance_tensor


@torch.inference_mode()
def _predict_mc_dropout(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    input_standardizer,
    target_standardizer,
    output_key: Optional[str],
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _enable_dropout_only(model)
    draws: List[torch.Tensor] = []
    for _ in range(num_samples):
        pred = predict_batch(
            model=model,
            batch_inputs=inputs,
            input_standardizer=input_standardizer,
            target_standardizer=target_standardizer,
            output_key=output_key,
        )
        if pred.ndim == 5 and pred.shape[2] == 1:
            pred = pred[:, :, 0, ...]
        draws.append(pred)
    stacked = torch.stack(draws, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False).clamp_min(1e-12)


@torch.inference_mode()
def _predict_ensemble(
    models: Sequence[torch.nn.Module],
    inputs: torch.Tensor,
    input_standardizer,
    target_standardizer,
    output_key: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    draws: List[torch.Tensor] = []
    for model in models:
        pred = predict_batch(
            model=model,
            batch_inputs=inputs,
            input_standardizer=input_standardizer,
            target_standardizer=target_standardizer,
            output_key=output_key,
        )
        if pred.ndim == 5 and pred.shape[2] == 1:
            pred = pred[:, :, 0, ...]
        draws.append(pred)
    stacked = torch.stack(draws, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False).clamp_min(1e-12)


def _compute_uncertainty_metrics(mean: np.ndarray, variance: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    error = mean - target
    std = np.sqrt(np.clip(variance, 1e-12, None))
    abs_error = np.abs(error)
    nll = 0.5 * (np.log(2.0 * math.pi * variance) + (error ** 2) / np.clip(variance, 1e-12, None))

    flattened_std = std.reshape(-1)
    flattened_abs_error = abs_error.reshape(-1)
    if flattened_std.size > 1:
        corr = float(np.corrcoef(flattened_std, flattened_abs_error)[0, 1])
    else:
        corr = 0.0

    nominal_coverages = np.asarray([0.50, 0.68, 0.80, 0.90, 0.95], dtype=np.float64)
    observed_coverages: List[float] = []
    for coverage in nominal_coverages:
        z = float(norm.ppf(0.5 + coverage / 2.0))
        inside = np.abs(error) <= z * std
        observed_coverages.append(float(np.mean(inside)))

    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(abs_error)),
        "nll": float(np.mean(nll)),
        "mean_predictive_std": float(np.mean(std)),
        "sharpness": float(np.mean(std)),
        "error_uncertainty_correlation": corr,
        "nominal_coverages": nominal_coverages.tolist(),
        "observed_coverages": observed_coverages,
    }


def main() -> None:
    args = parse_cli("Evaluate predictive uncertainty via MC dropout, ensembles, or direct probabilistic outputs.")
    config = load_yaml(args.config)
    device, seed_value, output_dir = prepare_runtime(config, args)
    logger = setup_logger("eval_uncertainty", save_dir=output_dir)
    logger.info("Starting uncertainty evaluation")

    dataset, dataloader = build_dataloader(config, dataset_key="dataset")
    logger.info("Loaded dataset with %d samples", len(dataset))

    normalization_cfg = config.get("normalization", {})
    input_standardizer = resolve_standardizer(normalization_cfg.get("input"))
    target_standardizer = resolve_standardizer(normalization_cfg.get("target"))
    output_key = config.get("evaluation", {}).get("output_key")

    uncertainty_cfg = config.get("uncertainty", {})
    mode = str(uncertainty_cfg.get("mode", "mc_dropout")).lower()
    num_samples = int(uncertainty_cfg.get("num_samples", 20))

    models: List[torch.nn.Module] = []
    checkpoint_paths: List[str] = []

    if mode == "ensemble":
        checkpoints = uncertainty_cfg.get("checkpoints")
        if not checkpoints:
            raise ValueError("uncertainty.checkpoints must be provided for ensemble mode.")
        for checkpoint in checkpoints:
            model, checkpoint_path = load_model(config, device=device, checkpoint_override=checkpoint)
            models.append(model)
            checkpoint_paths.append(str(checkpoint_path) if checkpoint_path else str(checkpoint))
        logger.info("Loaded %d ensemble members", len(models))
    else:
        model, checkpoint_path = load_model(config, device=device, checkpoint_override=args.checkpoint)
        models = [model]
        checkpoint_paths = [str(checkpoint_path) if checkpoint_path else "config-only model"]
        logger.info("Loaded checkpoint: %s", checkpoint_paths[0])

    predictive_means: List[np.ndarray] = []
    predictive_vars: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    sample_ids: List[str] = []

    for batch in dataloader:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = target_standardizer.denormalize(target)

        if mode == "mc_dropout":
            mean_t, var_t = _predict_mc_dropout(
                model=models[0],
                inputs=inputs,
                input_standardizer=input_standardizer,
                target_standardizer=target_standardizer,
                output_key=output_key,
                num_samples=num_samples,
            )
        elif mode == "ensemble":
            mean_t, var_t = _predict_ensemble(
                models=models,
                inputs=inputs,
                input_standardizer=input_standardizer,
                target_standardizer=target_standardizer,
                output_key=output_key,
            )
        elif mode == "direct":
            mean_t, var_t = _extract_direct_mean_and_variance(
                model=models[0],
                inputs=inputs,
                input_standardizer=input_standardizer,
                target_standardizer=target_standardizer,
                output_key=output_key,
            )
        else:
            raise ValueError("uncertainty.mode must be one of: mc_dropout, ensemble, direct")

        if target.ndim == 5 and target.shape[2] == 1:
            target = target[:, :, 0, ...]
        predictive_means.append(mean_t.detach().cpu().numpy())
        predictive_vars.append(var_t.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        sample_ids.extend(list(batch["sample_id"]))

    mean_arr = np.concatenate(predictive_means, axis=0)
    var_arr = np.concatenate(predictive_vars, axis=0)
    target_arr = np.concatenate(targets, axis=0)

    metrics = _compute_uncertainty_metrics(mean_arr, var_arr, target_arr)
    payload = {
        "seed": seed_value,
        "mode": mode,
        "checkpoints": checkpoint_paths,
        "num_samples": int(mean_arr.shape[0]),
        "num_timesteps": int(mean_arr.shape[1]),
        "metrics": metrics,
        "sample_ids": sample_ids,
    }
    save_json(output_dir / "uncertainty_summary.json", payload)
    np.savez_compressed(
        output_dir / "uncertainty_predictions.npz",
        mean=mean_arr.astype(np.float32),
        variance=var_arr.astype(np.float32),
        target=target_arr.astype(np.float32),
        sample_id=np.asarray(sample_ids),
    )

    save_reliability_curve(
        nominal_coverages=metrics["nominal_coverages"],
        observed_coverages=metrics["observed_coverages"],
        save_path=output_dir / "uncertainty_reliability.png",
        title=f"Reliability diagram ({mode})",
    )

    std_arr = np.sqrt(np.clip(var_arr, 1e-12, None))
    save_scatter(
        x=std_arr,
        y=np.abs(mean_arr - target_arr),
        save_path=output_dir / "uncertainty_error_vs_std.png",
        title="Absolute error vs predictive std",
        xlabel="Predictive std",
        ylabel="Absolute error",
    )

    example_index = 0
    timestep_index = int(uncertainty_cfg.get("visualization_timestep", mean_arr.shape[1] - 1))
    timestep_index = max(0, min(timestep_index, mean_arr.shape[1] - 1))
    save_uncertainty_panel(
        mean_field=mean_arr[example_index, timestep_index],
        std_field=std_arr[example_index, timestep_index],
        target_field=target_arr[example_index, timestep_index],
        save_path=output_dir / "uncertainty_example_panel.png",
        title=f"Predictive uncertainty | sample={sample_ids[example_index]} | t={timestep_index}",
    )

    logger.info("Uncertainty evaluation finished successfully")


if __name__ == "__main__":
    main()
