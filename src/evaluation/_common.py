from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import make_worker_init_fn, seed_everything  # noqa: E402

TensorOrArray = Union[torch.Tensor, np.ndarray]


@dataclass
class Standardizer:
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

    def is_active(self) -> bool:
        return self.mean is not None and self.std is not None

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_active():
            return tensor
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std.clamp_min(1e-8)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_active():
            return tensor
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return tensor * std + mean


class SequenceDataset(Dataset):
    def __init__(self, dataset_cfg: Mapping[str, Any]):
        self.cfg = dict(dataset_cfg)
        self.inputs_key = self.cfg.get("inputs_key", self.cfg.get("input_key", "inputs"))
        self.targets_key = self.cfg.get("targets_key", self.cfg.get("target_key", "targets"))
        self.ids_key = self.cfg.get("ids_key")
        self.channel_last_inputs = bool(self.cfg.get("channel_last_inputs", False))
        self.channel_last_targets = bool(self.cfg.get("channel_last_targets", False))
        self.target_squeeze_last = bool(self.cfg.get("target_squeeze_last", True))
        self._mode = "unknown"
        self._len = 0
        self._sample_files: List[Path] = []
        self._inputs: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._ids: Optional[np.ndarray] = None
        self._init_storage()

    def _init_storage(self) -> None:
        path_value = self.cfg.get("path")
        inputs_path_value = self.cfg.get("inputs_path")
        targets_path_value = self.cfg.get("targets_path")

        if path_value:
            path = Path(path_value)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {path}")
            if path.is_dir():
                pattern = self.cfg.get("file_pattern", "*.npz")
                self._sample_files = sorted(path.glob(pattern))
                if not self._sample_files:
                    raise FileNotFoundError(f"No sample files matched {pattern} in {path}")
                self._mode = "directory"
                self._len = len(self._sample_files)
                return
            suffix = path.suffix.lower()
            if suffix == ".npz":
                blob = np.load(path, allow_pickle=True)
                self._inputs = np.asarray(blob[self.inputs_key])
                self._targets = np.asarray(blob[self.targets_key])
                if self.ids_key and self.ids_key in blob:
                    self._ids = np.asarray(blob[self.ids_key])
                self._mode = "archive"
                self._len = int(self._inputs.shape[0])
                return
            raise ValueError(f"Unsupported dataset file format: {path.suffix}")

        if inputs_path_value and targets_path_value:
            inputs_path = Path(inputs_path_value)
            targets_path = Path(targets_path_value)
            if not inputs_path.exists() or not targets_path.exists():
                raise FileNotFoundError("inputs_path and targets_path must both exist")
            self._inputs = np.load(inputs_path, allow_pickle=True)
            self._targets = np.load(targets_path, allow_pickle=True)
            self._mode = "separate_arrays"
            self._len = int(self._inputs.shape[0])
            return

        raise ValueError("Dataset config must define either 'path' or both 'inputs_path' and 'targets_path'.")

    def __len__(self) -> int:
        return self._len

    def _format_inputs(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and self.channel_last_inputs:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim != 3:
            raise ValueError(f"Expected input with 2 or 3 dims per sample, got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def _format_targets(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            if self.channel_last_targets:
                arr = np.moveaxis(arr, -1, 0)
            else:
                pass
        elif arr.ndim == 4:
            if self.channel_last_targets:
                arr = np.moveaxis(arr, -1, 1)
            if arr.shape[1] == 1 and self.target_squeeze_last:
                arr = arr[:, 0, ...]
            elif arr.shape[-1] == 1 and self.target_squeeze_last:
                arr = arr[..., 0]
        else:
            raise ValueError(f"Expected target with 2, 3, or 4 dims per sample, got shape {arr.shape}")

        if arr.ndim == 3:
            return arr.astype(np.float32, copy=False)
        raise ValueError(
            "Targets must resolve to [T,H,W] after formatting. "
            f"Received final target shape {arr.shape}."
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._mode == "directory":
            sample_path = self._sample_files[index]
            blob = np.load(sample_path, allow_pickle=True)
            inputs = self._format_inputs(blob[self.inputs_key])
            targets = self._format_targets(blob[self.targets_key])
            sample_id = sample_path.stem
            if self.ids_key and self.ids_key in blob:
                raw_id = blob[self.ids_key]
                if np.asarray(raw_id).shape == ():
                    sample_id = str(np.asarray(raw_id).item())
                else:
                    sample_id = str(raw_id)
        else:
            assert self._inputs is not None and self._targets is not None
            inputs = self._format_inputs(self._inputs[index])
            targets = self._format_targets(self._targets[index])
            if self._ids is not None:
                sample_id = str(np.asarray(self._ids[index]).item())
            else:
                sample_id = f"sample_{index:06d}"

        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "sample_id": sample_id,
        }


def resolve_project_root() -> Path:
    return PROJECT_ROOT


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping/dictionary.")
    return data


def save_json(path: Union[str, Path], data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def parse_cli(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Overrides config.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Overrides config.")
    parser.add_argument("--device", default=None, help="Optional device string, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    return parser.parse_args()


def ensure_dir(path: Union[str, Path]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def import_from_string(target: str) -> Any:
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_device(config: Mapping[str, Any], override: Optional[str] = None) -> torch.device:
    if override:
        return torch.device(override)
    device_value = config.get("runtime", {}).get("device") or config.get("device")
    if device_value:
        return torch.device(device_value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(blob: Any) -> Tuple[Optional[Mapping[str, torch.Tensor]], Any]:
    if isinstance(blob, torch.nn.Module):
        return None, blob
    if isinstance(blob, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "network"):
            if key in blob and isinstance(blob[key], Mapping):
                return blob[key], None
        if all(isinstance(value, torch.Tensor) for value in blob.values()):
            return blob, None
    return None, None


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def _instantiate_model(model_cfg: Mapping[str, Any]) -> torch.nn.Module:
    if "callable" in model_cfg:
        builder = import_from_string(str(model_cfg["callable"]))
        kwargs = dict(model_cfg.get("kwargs", {}))
        model = builder(**kwargs)
    else:
        module_name = model_cfg.get("module")
        class_name = model_cfg.get("class_name")
        if not module_name or not class_name:
            raise ValueError(
                "Model config must define either 'callable' or both 'module' and 'class_name'."
            )
        module = importlib.import_module(str(module_name))
        model_cls = getattr(module, str(class_name))
        kwargs = dict(model_cfg.get("kwargs", {}))
        model = model_cls(**kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Instantiated model is not a torch.nn.Module")
    return model


def load_model(
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_override: Optional[Union[str, Path]] = None,
) -> Tuple[torch.nn.Module, Optional[Path]]:
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})
    checkpoint_path = checkpoint_override or eval_cfg.get("checkpoint") or model_cfg.get("checkpoint")

    if checkpoint_path is None and not model_cfg:
        raise ValueError("No model specification found in config.")

    checkpoint_blob: Any = None
    loaded_module: Optional[torch.nn.Module] = None
    checkpoint_path_obj: Optional[Path] = None
    if checkpoint_path is not None:
        checkpoint_path_obj = Path(checkpoint_path)
        checkpoint_blob = torch.load(checkpoint_path_obj, map_location=device)
        _, maybe_module = _extract_state_dict(checkpoint_blob)
        if maybe_module is not None:
            loaded_module = maybe_module.to(device)

    if loaded_module is not None:
        model = loaded_module
    else:
        model = _instantiate_model(model_cfg)
        if checkpoint_blob is not None:
            state_dict, _ = _extract_state_dict(checkpoint_blob)
            if state_dict is None:
                raise ValueError(f"Could not find a state_dict inside checkpoint: {checkpoint_path_obj}")
            incompatible = model.load_state_dict(_strip_module_prefix(state_dict), strict=bool(model_cfg.get("strict_load", True)))
            if isinstance(incompatible, tuple):
                pass

    model.to(device)
    model.eval()
    return model, checkpoint_path_obj


def resolve_standardizer(stats_cfg: Optional[Mapping[str, Any]]) -> Standardizer:
    if not stats_cfg:
        return Standardizer()
    mean = stats_cfg.get("mean")
    std = stats_cfg.get("std")
    if "path" in stats_cfg:
        blob = np.load(Path(stats_cfg["path"]), allow_pickle=True)
        mean_key = stats_cfg.get("mean_key", "mean")
        std_key = stats_cfg.get("std_key", "std")
        mean = blob[mean_key]
        std = blob[std_key]
    if mean is None or std is None:
        return Standardizer()
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32)
    return Standardizer(mean_tensor, std_tensor)


def build_dataloader(config: Mapping[str, Any], dataset_key: str = "dataset") -> Tuple[SequenceDataset, DataLoader]:
    dataset_cfg = config.get(dataset_key, {})
    if not dataset_cfg:
        raise ValueError(f"Missing dataset config under key '{dataset_key}'.")
    dataset = SequenceDataset(dataset_cfg)
    seed_value = int(config.get("runtime", {}).get("seed", 42))
    batch_size = int(dataset_cfg.get("batch_size", 1))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", torch.cuda.is_available()))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=make_worker_init_fn(seed_value),
    )
    return dataset, dataloader


def seed_from_config(config: Mapping[str, Any], override: Optional[int] = None) -> int:
    seed_value = int(override if override is not None else config.get("runtime", {}).get("seed", 42))
    seed_everything(seed_value)
    return seed_value


@torch.inference_mode()
def model_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    output_key: Optional[str] = None,
) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, Mapping):
        if output_key and output_key in outputs:
            value = outputs[output_key]
        else:
            for candidate in ("prediction", "pred", "output", "outputs", "mean"):
                if candidate in outputs:
                    value = outputs[candidate]
                    break
            else:
                raise KeyError("Could not infer prediction tensor from dict model output.")
        if not isinstance(value, torch.Tensor):
            raise TypeError("Selected model output is not a tensor.")
        return value
    if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    batch_inputs: torch.Tensor,
    input_standardizer: Standardizer,
    target_standardizer: Standardizer,
    output_key: Optional[str] = None,
) -> torch.Tensor:
    normalized_inputs = input_standardizer.normalize(batch_inputs)
    predictions = model_forward(model, normalized_inputs, output_key=output_key)
    predictions = target_standardizer.denormalize(predictions)
    return predictions


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    input_standardizer: Standardizer,
    target_standardizer: Standardizer,
    output_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    sample_ids: List[str] = []
    for batch in dataloader:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = target_standardizer.denormalize(target)
        pred = predict_batch(
            model=model,
            batch_inputs=inputs,
            input_standardizer=input_standardizer,
            target_standardizer=target_standardizer,
            output_key=output_key,
        )
        if pred.ndim == 5 and pred.shape[2] == 1:
            pred = pred[:, :, 0, ...]
        if pred.ndim == 4 and target.ndim == 4 and pred.shape != target.shape:
            raise ValueError(f"Prediction shape {tuple(pred.shape)} does not match target shape {tuple(target.shape)}")
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        sample_ids.extend(list(batch["sample_id"]))
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), sample_ids


def flatten_per_sample(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1)


def compute_sample_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    error = pred - target
    mse = np.mean(error ** 2, axis=tuple(range(1, error.ndim)))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error), axis=tuple(range(1, error.ndim)))
    max_abs = np.max(np.abs(error), axis=tuple(range(1, error.ndim)))

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)
    diff_flat = pred_flat - target_flat
    denom = np.linalg.norm(target_flat, axis=1)
    rel_l2 = np.linalg.norm(diff_flat, axis=1) / np.clip(denom, 1e-12, None)

    target_mean = np.mean(target_flat, axis=1, keepdims=True)
    ss_res = np.sum((target_flat - pred_flat) ** 2, axis=1)
    ss_tot = np.sum((target_flat - target_mean) ** 2, axis=1)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "relative_l2": rel_l2,
        "r2": r2,
    }


def summarize_metrics(sample_metrics: Mapping[str, np.ndarray]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name, values in sample_metrics.items():
        summary[name] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
        summary[f"{name}_median"] = float(np.median(values))
    return summary


def compute_timestep_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} must match target shape {target.shape}")
    if pred.ndim != 4:
        raise ValueError("Expected [N,T,H,W] arrays for timestep metrics.")
    error = pred - target
    mse_t = np.mean(error ** 2, axis=(0, 2, 3))
    rmse_t = np.sqrt(mse_t)
    mae_t = np.mean(np.abs(error), axis=(0, 2, 3))
    target_norm = np.linalg.norm(target, axis=(2, 3))
    error_norm = np.linalg.norm(error, axis=(2, 3))
    rel_l2_t = np.mean(error_norm / np.clip(target_norm, 1e-12, None), axis=0)
    return {
        "mse": mse_t,
        "rmse": rmse_t,
        "mae": mae_t,
        "relative_l2": rel_l2_t,
    }


def benchmark_callable(
    fn: Callable[[], Any],
    warmup: int = 10,
    repeats: int = 50,
    synchronize_cuda: bool = True,
) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        if synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_seconds": float(arr.mean()),
        "std_seconds": float(arr.std()),
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "median_seconds": float(np.median(arr)),
        "repeats": int(arr.size),
    }


@torch.inference_mode()
def benchmark_model(
    model: torch.nn.Module,
    batch_inputs: torch.Tensor,
    input_standardizer: Standardizer,
    target_standardizer: Standardizer,
    output_key: Optional[str] = None,
    warmup: int = 10,
    repeats: int = 50,
) -> Dict[str, float]:
    batch_inputs = batch_inputs.to(next(model.parameters()).device if any(True for _ in model.parameters()) else batch_inputs.device)

    def _run() -> torch.Tensor:
        return predict_batch(
            model=model,
            batch_inputs=batch_inputs,
            input_standardizer=input_standardizer,
            target_standardizer=target_standardizer,
            output_key=output_key,
        )

    return benchmark_callable(_run, warmup=warmup, repeats=repeats)


@torch.inference_mode()
def call_simulator(sim_fn: Callable[..., Any], sample_inputs: torch.Tensor, simulator_cfg: Mapping[str, Any]) -> Any:
    kwargs = dict(simulator_cfg.get("kwargs", {}))
    return sim_fn(sample_inputs, **kwargs)


@torch.inference_mode()
def benchmark_simulator(
    simulator_target: str,
    sample_inputs: torch.Tensor,
    simulator_cfg: Mapping[str, Any],
    warmup: int = 3,
    repeats: int = 10,
) -> Dict[str, float]:
    sim_fn = import_from_string(simulator_target)
    tensor_arg = sample_inputs.detach().cpu()

    def _run() -> Any:
        return call_simulator(sim_fn, tensor_arg, simulator_cfg)

    return benchmark_callable(_run, warmup=warmup, repeats=repeats, synchronize_cuda=False)


def pick_example_indices(num_samples: int, max_examples: int) -> List[int]:
    if num_samples <= 0:
        return []
    if num_samples <= max_examples:
        return list(range(num_samples))
    return np.linspace(0, num_samples - 1, max_examples, dtype=int).tolist()


def mean_dicts(values: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in values for key in item})
    merged: Dict[str, float] = {}
    for key in keys:
        series = [float(item[key]) for item in values if key in item]
        if series:
            merged[key] = float(np.mean(series))
    return merged


def namespace_to_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "device": args.device,
        "seed": args.seed,
    }


def prepare_runtime(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[torch.device, int, Path]:
    device = get_device(config, override=args.device)
    seed_value = seed_from_config(config, override=args.seed)
    default_output = config.get("evaluation", {}).get("output_dir", "results/evaluation")
    output_dir = ensure_dir(args.output_dir or default_output)
    return device, seed_value, output_dir
