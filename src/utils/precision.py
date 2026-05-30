from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    model_dtype: torch.dtype
    input_dtype: torch.dtype
    label: str


def parse_optional_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'. Use true/false.")


def _normalize_precision_name(precision: str) -> str:
    p = str(precision).strip().lower()
    aliases = {
        "float32": "fp32",
    }
    return aliases.get(p, p)


def tf32_backend_state(device: torch.device) -> tuple[bool | None, bool | None]:
    if device.type != "cuda":
        return None, None
    return bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32)


def tf32_precision_label(device: torch.device) -> str:
    matmul_tf32, cudnn_tf32 = tf32_backend_state(device)
    if matmul_tf32 is None or cudnn_tf32 is None:
        return "fp32"
    if matmul_tf32 and cudnn_tf32:
        return "fp32/tf32"
    if cudnn_tf32 and not matmul_tf32:
        return "fp32/cudnn_tf32"
    if matmul_tf32 and not cudnn_tf32:
        return "fp32/matmul_tf32"
    return "fp32"


def configure_torch_precision(
    device: torch.device,
    precision: str = "fp32",
    allow_tf32: bool | None = None,
) -> PrecisionConfig:
    p = _normalize_precision_name(precision)
    if p != "fp32":
        raise ValueError("Only fp32 precision is supported in this benchmark pipeline.")

    if allow_tf32 is not None and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    label = tf32_precision_label(device)
    return PrecisionConfig(
        name="fp32",
        model_dtype=torch.float32,
        input_dtype=torch.float32,
        label=label,
    )


def cast_model_for_precision(model: torch.nn.Module, precision_cfg: PrecisionConfig) -> torch.nn.Module:
    _ = precision_cfg
    return model.float()


def autocast_context(device: torch.device, precision_cfg: PrecisionConfig) -> ContextManager:
    _ = device
    _ = precision_cfg
    return nullcontext()
