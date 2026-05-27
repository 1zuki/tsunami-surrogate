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


def _read_target_offset_scale(npz_path: Path) -> Optional[Tuple[float, float]]:
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

    return offset, scale


def load_target_denorm(path: str | Path) -> Optional[Tuple[float, float]]:
    npz_path = resolve_dataset_npz(path)
    normalized_flag = _read_normalized_targets_flag(npz_path)
    stats = _read_target_offset_scale(npz_path)
    if stats is None:
        return None
    offset, scale = stats
    if normalized_flag is False:
        return None
    if normalized_flag is None and abs(offset) <= 1e-12 and abs(scale - 1.0) <= 1e-12:
        return None

    return offset, scale


def target_signature(path: str | Path) -> dict[str, Any]:
    npz_path = resolve_dataset_npz(path)
    normalized_flag = _read_normalized_targets_flag(npz_path)
    denorm = load_target_denorm(npz_path)

    if denorm is not None:
        offset, scale = denorm
        normalized_targets = True if normalized_flag is None else bool(normalized_flag)
        return {
            "dataset_path": str(npz_path),
            "normalized_targets": normalized_targets,
            "target_offset": float(offset),
            "target_scale": float(scale),
        }

    return {
        "dataset_path": str(npz_path),
        "normalized_targets": False if normalized_flag is None else bool(normalized_flag),
        "target_offset": None,
        "target_scale": None,
    }


def signatures_match(reference: dict[str, Any], candidate: dict[str, Any], tol: float = 1e-6) -> bool:
    ref_norm = bool(reference.get("normalized_targets", False))
    cand_norm = bool(candidate.get("normalized_targets", False))
    if ref_norm != cand_norm:
        return False

    ref_off = reference.get("target_offset")
    ref_scale = reference.get("target_scale")
    cand_off = candidate.get("target_offset")
    cand_scale = candidate.get("target_scale")

    if ref_off is None or ref_scale is None or cand_off is None or cand_scale is None:
        return (not ref_norm) and (not cand_norm)

    return bool(
        abs(float(ref_off) - float(cand_off)) <= tol
        and abs(float(ref_scale) - float(cand_scale)) <= tol
    )


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
