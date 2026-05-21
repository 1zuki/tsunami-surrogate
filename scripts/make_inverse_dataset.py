#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import save_json


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _build_observation(targets: np.ndarray, mode: str) -> np.ndarray:
    if targets.ndim == 4:
        # [N, T, H, W]
        if mode == "final_eta":
            return targets[:, -1]
        if mode == "mean_eta":
            return np.mean(targets, axis=1)
        if mode == "max_abs_eta":
            return np.max(np.abs(targets), axis=1)
        raise ValueError(f"Unsupported observation mode for 4D targets: {mode}")

    if targets.ndim == 3:
        # [N, H, W]
        return targets

    raise ValueError(f"Expected targets ndim 3 or 4, got shape={targets.shape}")


def _split_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    in_cfg = dict(cfg.get("forward_data", {}))
    mapping = {
        "train": in_cfg.get("train_path"),
        "val": in_cfg.get("val_path"),
        "test": in_cfg.get("test_path"),
    }
    out: Dict[str, Path] = {}
    for split, path in mapping.items():
        if path:
            out[split] = Path(str(path))
    if not out:
        raise KeyError("forward_data must provide at least one of: train_path, val_path, test_path")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Create an inverse-problem scaffold dataset from forward processed eval archives.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    split_paths = _split_paths(cfg)
    output_root = Path(str(cfg.get("output_root", "data/processed_inverse/scaffold")))
    source_channel = int(cfg.get("source_channel_index", 1))
    bathymetry_channel = int(cfg.get("bathymetry_channel_index", 0))
    initial_depth_channel = int(cfg.get("initial_depth_channel_index", 2))

    obs_cfg = dict(cfg.get("observation", {}))
    obs_mode = str(obs_cfg.get("mode", "final_eta"))
    include_bathymetry = bool(obs_cfg.get("include_bathymetry", True))
    include_initial_depth = bool(obs_cfg.get("include_initial_depth", False))
    noise_std = float(obs_cfg.get("noise_std", 0.0))
    rng_seed = int(obs_cfg.get("seed", 42))
    rng = np.random.default_rng(rng_seed)

    for split, in_path in split_paths.items():
        if not in_path.exists():
            raise FileNotFoundError(in_path)

        payload = _load_npz(in_path)
        if "inputs" not in payload or "targets" not in payload:
            raise KeyError(f"{in_path} must contain inputs and targets")

        x = np.asarray(payload["inputs"], dtype=np.float32)
        y = np.asarray(payload["targets"], dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected inputs shape [N,C,H,W], got {x.shape}")
        n, c, _, _ = x.shape
        if source_channel < 0 or source_channel >= c:
            raise ValueError(f"source_channel_index {source_channel} out of range for inputs with C={c}")

        obs = _build_observation(y, obs_mode).astype(np.float32)
        if noise_std > 0:
            obs = obs + rng.normal(scale=noise_std, size=obs.shape).astype(np.float32)

        inverse_inputs = [obs[:, None, :, :]]
        if include_bathymetry:
            if bathymetry_channel < 0 or bathymetry_channel >= c:
                raise ValueError(f"bathymetry_channel_index {bathymetry_channel} out of range for inputs with C={c}")
            inverse_inputs.append(x[:, bathymetry_channel : bathymetry_channel + 1])
        if include_initial_depth:
            if initial_depth_channel < 0 or initial_depth_channel >= c:
                raise ValueError(
                    f"initial_depth_channel_index {initial_depth_channel} out of range for inputs with C={c}"
                )
            inverse_inputs.append(x[:, initial_depth_channel : initial_depth_channel + 1])

        x_inv = np.concatenate(inverse_inputs, axis=1).astype(np.float32)
        y_inv = x[:, source_channel].astype(np.float32)

        out_dir = output_root / split
        out_npz = out_dir / "eval_dataset.npz"
        out_manifest = out_dir / "eval_manifest.json"

        if out_npz.exists() and not args.overwrite:
            raise FileExistsError(f"{out_npz} already exists. Use --overwrite to replace.")
        out_dir.mkdir(parents=True, exist_ok=True)

        export: Dict[str, np.ndarray] = {
            "inputs": x_inv,
            "targets": y_inv,
        }
        for key in ("sample_id", "source_type", "bathymetry_type", "source_strength", "scenario_id", "solver_name"):
            if key in payload:
                arr = np.asarray(payload[key])
                if arr.shape and arr.shape[0] == n:
                    export[key] = arr

        np.savez_compressed(out_npz, **export)

        manifest = {
            "split": split,
            "forward_input_path": str(in_path),
            "inverse_output_path": str(out_npz),
            "num_samples": int(n),
            "observation_mode": obs_mode,
            "noise_std": noise_std,
            "source_channel_index": source_channel,
            "include_bathymetry": include_bathymetry,
            "include_initial_depth": include_initial_depth,
            "inverse_inputs_shape": list(map(int, x_inv.shape)),
            "inverse_targets_shape": list(map(int, y_inv.shape)),
        }
        save_json(manifest, out_manifest)
        print(
            f"[inverse] split={split:<5} n={n:5d} x_shape={tuple(x_inv.shape)} "
            f"y_shape={tuple(y_inv.shape)} -> {out_npz}"
        )

    print(f"[inverse] done. output_root={output_root}")


if __name__ == "__main__":
    main()
