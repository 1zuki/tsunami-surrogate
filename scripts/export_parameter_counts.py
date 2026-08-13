#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models import build_model
from src.utils.config import load_config


DEFAULT_MODELS = [
    ("fno", "single_pass", "configs/model/fno.yaml", "experiments/fno/best.pt"),
    ("ffno", "single_pass", "configs/model/ffno.yaml", "experiments/ffno/best.pt"),
    ("cnn", "single_pass", "configs/model/cnn.yaml", "experiments/cnn/best.pt"),
    ("unet", "single_pass", "configs/model/unet.yaml", "experiments/unet/best.pt"),
    (
        "convlstm",
        "single_pass_baseline",
        "configs/model/convlstm.yaml",
        "experiments/convlstm/best.pt",
    ),
    (
        "fno_modes8",
        "single_pass_ablation",
        "configs/model/fno_modes8.yaml",
        "experiments/fno_modes8/best.pt",
    ),
    (
        "fno_modes20",
        "single_pass_ablation",
        "configs/model/fno_modes20.yaml",
        "experiments/fno_modes20/best.pt",
    ),
    (
        "ufno",
        "single_pass_baseline",
        "configs/model/ufno.yaml",
        "experiments/ufno/best.pt",
    ),
    (
        "wno",
        "single_pass_baseline",
        "configs/model/wno.yaml",
        "experiments/wno/best.pt",
    ),
    (
        "fno_muscl_hr",
        "target_solver",
        "configs/model/fno_muscl_hr.yaml",
        "experiments/fno_muscl_hr/fno_muscl_hr_seed_18/best.pt",
    ),
    (
        "fno_boussinesq",
        "target_solver",
        "configs/model/fno_boussinesq.yaml",
        "experiments/fno_boussinesq/best.pt",
    ),
    (
        "fno_window5_hydrostatic",
        "seeded_window",
        "configs/model/fno_window5_hydrostatic.yaml",
        "experiments/fno_window5_hydrostatic/best.pt",
    ),
    (
        "ffno_window5_hydrostatic",
        "seeded_window",
        "configs/model/ffno_window5_hydrostatic.yaml",
        "experiments/ffno_window5_hydrostatic/best.pt",
    ),
]


def _load_checkpoint_config(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return payload["config"]
    return None


def _count_params(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in parameters))


def _model_row(
    name: str, group: str, config_path: str, checkpoint_path: str
) -> Dict[str, Any]:
    cfg_path = ROOT / config_path
    ckpt_path = ROOT / checkpoint_path
    cfg = _load_checkpoint_config(ckpt_path)
    config_source = "checkpoint" if cfg is not None else "yaml"
    if cfg is None:
        cfg = load_config(cfg_path)

    model = build_model(cfg)
    trainable = _count_params(p for p in model.parameters() if p.requires_grad)
    total = _count_params(model.parameters())
    non_trainable = total - trainable
    model_cfg = cfg.get("model", cfg)
    architecture = model_cfg.get("name")
    uses_fourier_modes = architecture in {"fno2d", "ffno2d", "ufno2d"}
    uses_depth = architecture in {"fno2d", "ffno2d", "ufno2d", "wno2d"}

    return {
        "model": name,
        "group": group,
        "architecture": architecture,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": ckpt_path.is_file(),
        "config_source": config_source,
        "in_channels": model_cfg.get("in_channels"),
        "out_channels": model_cfg.get("out_channels"),
        "modes1": model_cfg.get("modes1") if uses_fourier_modes else None,
        "modes2": model_cfg.get("modes2") if uses_fourier_modes else None,
        "width": model_cfg.get("width"),
        "depth": model_cfg.get("depth") if uses_depth else None,
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params_millions": total / 1_000_000.0,
        "trainable_params_millions": trainable / 1_000_000.0,
        "fp32_parameter_memory_mb": total * 4.0 / 1_000_000.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export parameter counts for paper-facing models."
    )
    p.add_argument("--output", default="results/parameter_counts.json")
    p.add_argument("--csv-output", default="results/parameter_counts.csv")
    args = p.parse_args()

    rows = [_model_row(*item) for item in DEFAULT_MODELS]
    missing_checkpoints = [
        row["checkpoint_path"] for row in rows if not row["checkpoint_exists"]
    ]
    out = {
        "evaluation_type": "parameter_counts",
        "notes": (
            "Counts are computed from the checkpoint-stored config when available, "
            "with the YAML config as fallback. Parameter memory assumes fp32 weights."
        ),
        "rows": rows,
        "missing_checkpoints": missing_checkpoints,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_path = ROOT / args.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "group",
        "architecture",
        "config_path",
        "checkpoint_path",
        "checkpoint_exists",
        "config_source",
        "in_channels",
        "out_channels",
        "modes1",
        "modes2",
        "width",
        "depth",
        "total_params",
        "trainable_params",
        "non_trainable_params",
        "total_params_millions",
        "trainable_params_millions",
        "fp32_parameter_memory_mb",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"parameter count rows={len(rows)} missing_checkpoints={len(missing_checkpoints)} -> {output_path}"
    )
    print(f"csv -> {csv_path}")


if __name__ == "__main__":
    main()
