from __future__ import annotations

import os
import platform
from typing import Any, Dict

import torch


def resolve_device(device: str = "auto") -> torch.device:
    requested = str(device).strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device(requested)

    if requested.startswith("cpu"):
        return torch.device("cpu")

    # keep explicit for now; add MPS or other backends only when needed.
    raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda")


def _cpu_name() -> str:
    cpu = platform.processor().strip()
    if cpu:
        return cpu

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                        if name:
                            return name
    except Exception:
        pass

    uname_cpu = platform.uname().processor.strip()
    if uname_cpu:
        return uname_cpu

    return "unknown"


def hardware_info(device: torch.device) -> Dict[str, Any]:
    selected = str(device)
    info: Dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "gpu_name": None,
        "cpu": _cpu_name(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": int(os.getpid()),
        "selected_device": selected,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_index"] = int(idx)
    return info
