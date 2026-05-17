from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch


def resolve_dataset_npz(path: str | Path) -> Path:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if dataset_path.is_dir():
        candidate = dataset_path / "eval_dataset.npz"
        if candidate.exists():
            return candidate

        npz_candidates = sorted(dataset_path.glob("*.npz"))
        if not npz_candidates:
            raise FileNotFoundError(f"No .npz found in directory: {dataset_path}")

        return npz_candidates[0]

    return dataset_path


def _read_normalized_targets_flag(npz_path: Path) -> Optional[bool]:
    manifest_path = npz_path.with_name("eval_manifest.json")
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None

    value = manifest.get("normalized_targets")
    if isinstance(value, bool):
        return value

    return None


def load_target_denorm(path: str | Path) -> Optional[Tuple[float, float]]:
    npz_path = resolve_dataset_npz(path)
    normalized_flag = _read_normalized_targets_flag(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        if "target_mean" not in data or "target_std" not in data:
            return None

        offset_arr = np.asarray(data["target_mean"]).reshape(-1)
        scale_arr = np.asarray(data["target_std"]).reshape(-1)
        if offset_arr.size == 0 or scale_arr.size == 0:
            return None

        offset = float(offset_arr[0])
        scale = float(scale_arr[0])

    if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
        return None
    if normalized_flag is False:
        return None
    if normalized_flag is None and abs(offset) <= 1e-12 and abs(scale - 1.0) <= 1e-12:
        return None

    return offset, scale


def apply_target_denorm(tensor: torch.Tensor, denorm: Optional[Tuple[float, float]]) -> torch.Tensor:
    if denorm is None:
        return tensor
    offset, scale = denorm

    return tensor * float(scale) + float(offset)


def resolve_eval_dataset_path(cfg: dict[str, Any], split: str = "test") -> Optional[Path]:
    eval_cfg = cfg.get("eval", {})
    if split == "test":
        eval_dataset_path = eval_cfg.get("dataset_path")
        if eval_dataset_path:
            return Path(eval_dataset_path)

    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    split_key = f"{split}_path"
    split_path = data_cfg.get(split_key)

    if split_path:
        return Path(split_path)

    data_path = data_cfg.get("path")
    if data_path:
        return Path(data_path)

    return None
