#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.benchmark import benchmark_inference
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import hardware_info, resolve_device
from src.utils.io import save_json
from src.utils.model_io import validate_model_io_channels
from src.utils.precision import (
    cast_model_for_precision,
    configure_torch_precision,
    parse_optional_bool,
    tf32_backend_state,
)


def _dataset_num_samples(loader: Any) -> int:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return -1
    try:
        return int(len(ds))
    except Exception:
        return -1


def _default_output(eval_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    return Path(output_dir) / "speed.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark model inference runtime on a processed test split.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--dataset-path", type=str, default=None, help="Override eval dataset path/directory.")
    p.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    p.add_argument("--precision", choices=["fp32"], default=None)
    p.add_argument(
        "--allow-tf32",
        type=str,
        default=None,
        help="Set CUDA TF32 matmul/cuDNN behavior (true/false).",
    )
    p.add_argument("--warmup", type=int, default=None, help="Override warmup batch count.")
    p.add_argument("--repeats", type=int, default=None, help="Override timed repeat count.")
    p.add_argument("--max-batches", type=int, default=None, help="Max unique batches to cache from test loader.")
    p.add_argument("--method-name", type=str, default=None, help="Optional explicit method label for reports.")
    p.add_argument("--output", type=str, default=None, help="Output JSON path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    speed_cfg = cfg.get("speed", {})
    data_cfg = dict(cfg.get("data", {}))
    dataset_cfg = cfg.get("dataset", {})

    if args.device is not None:
        cfg["device"] = args.device

    if not data_cfg and isinstance(dataset_cfg, dict):
        dataset_path = dataset_cfg.get("path")
        if dataset_path:
            data_cfg["test_path"] = dataset_path
        if "batch_size" in dataset_cfg:
            data_cfg["batch_size"] = dataset_cfg["batch_size"]

    dataset_path_override = args.dataset_path or eval_cfg.get("dataset_path")
    if dataset_path_override:
        data_cfg["test_path"] = str(dataset_path_override)
    if args.batch_size is not None:
        data_cfg["batch_size"] = int(args.batch_size)
    elif "batch_size" in eval_cfg:
        data_cfg["batch_size"] = eval_cfg["batch_size"]
    cfg["data"] = data_cfg

    if "test_path" not in data_cfg and "path" not in data_cfg:
        raise KeyError(
            "No dataset path configured. Set eval.dataset_path, data.test_path, data.path, or --dataset-path."
        )

    device = resolve_device(cfg.get("device", "auto"))
    precision_name = args.precision or str(speed_cfg.get("precision", eval_cfg.get("precision", "fp32")))
    allow_tf32 = parse_optional_bool(args.allow_tf32)
    if allow_tf32 is None and "allow_tf32" in speed_cfg:
        allow_tf32 = parse_optional_bool(speed_cfg.get("allow_tf32"))
    precision_cfg = configure_torch_precision(device, precision=precision_name, allow_tf32=allow_tf32)
    tf32_matmul_actual, tf32_cudnn_actual = tf32_backend_state(device)

    warmup = (
        int(args.warmup)
        if args.warmup is not None
        else int(speed_cfg.get("model_warmup", eval_cfg.get("warmup_steps", 5)))
    )
    repeats = (
        int(args.repeats)
        if args.repeats is not None
        else int(speed_cfg.get("model_repeats", eval_cfg.get("timed_steps", 20)))
    )
    max_batches = int(args.max_batches) if args.max_batches is not None else None

    loaders = create_dataloaders(cfg)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise KeyError("No test dataloader found. Set eval.dataset_path or data.test_path.")
    validate_model_io_channels(cfg, loaders, preferred_splits=("test", "val", "train"))

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    model = cast_model_for_precision(model, precision_cfg)

    metrics = benchmark_inference(
        model=model,
        loader=test_loader,
        device=device,
        warmup_steps=warmup,
        timed_steps=repeats,
        max_batches=max_batches,
        precision_cfg=precision_cfg,
    )

    payload: Dict[str, Any] = {
        "evaluation_type": "model_speed_benchmark",
        "method": args.method_name or str(cfg.get("model", {}).get("name", "model")),
        "model_name": str(cfg.get("model", {}).get("name", "unknown")),
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(data_cfg.get("test_path", data_cfg.get("path", ""))),
        "device": str(device),
        "precision": str(precision_cfg.name),
        "precision_label": str(precision_cfg.label),
        "allow_tf32_requested": bool(allow_tf32) if allow_tf32 is not None else None,
        "allow_tf32": bool(allow_tf32) if allow_tf32 is not None else None,
        "tf32_matmul_actual": tf32_matmul_actual,
        "tf32_cudnn_actual": tf32_cudnn_actual,
        "batch_size": int(data_cfg.get("batch_size", -1)),
        "num_warmup": int(metrics.get("num_warmup", warmup)),
        "num_repeats": int(metrics.get("num_repeats", repeats)),
        "num_samples_dataset": int(_dataset_num_samples(test_loader)),
        "num_batches_timed": int(metrics.get("num_batches_timed", 0)),
        "num_samples_timed": int(metrics.get("num_samples_timed", 0)),
        "time_total_mean_s": float(metrics.get("time_total_s", 0.0)),
        "time_per_batch_mean_s": float(metrics.get("time_per_batch_mean_s", 0.0)),
        "time_per_sample_mean_s": float(metrics.get("time_per_sample_mean_s", 0.0)),
        "samples_per_second": float(metrics.get("samples_per_second", 0.0)),
        "hardware": hardware_info(device),
        "checkpoint_compatibility": checkpoint.get("compatibility", {}),
    }

    output_path = Path(args.output) if args.output else _default_output(eval_cfg, cfg)
    save_json(payload, output_path)
    print(payload)


if __name__ == "__main__":
    main()
