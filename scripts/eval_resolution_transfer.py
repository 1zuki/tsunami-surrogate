#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.multires_dataset import MultiResolutionDataset
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.training.metrics import MetricAccumulator
from src.evaluation.target_scaling import load_target_denorm
from src.utils.io import save_json
from src.utils.seed import seed_everything
import torch


def _model_output(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _validate_resolution_transfer_channels(cfg, loader, resolutions) -> None:
    model_cfg = cfg.get("model", cfg)
    expected_in = int(model_cfg.get("in_channels", 0))
    expected_out = int(model_cfg.get("out_channels", 0))
    batch = next(iter(loader))
    ref_res = int(resolutions[0])
    x_key = f"x_{ref_res}"
    y_key = f"y_{ref_res}"
    if x_key not in batch or y_key not in batch:
        raise KeyError(f"Expected keys '{x_key}' and '{y_key}' in resolution-transfer batch.")
    actual_in = int(batch[x_key].shape[1])
    actual_out = int(batch[y_key].shape[1])
    if expected_in and expected_in != actual_in:
        raise ValueError(
            f"model.in_channels ({expected_in}) does not match dataset x channels ({actual_in}) at resolution {ref_res}."
        )
    if expected_out and expected_out != actual_out:
        raise ValueError(
            f"model.out_channels ({expected_out}) does not match dataset y channels ({actual_out}) at resolution {ref_res}."
        )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--output", default=None)

    args = p.parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get('device', 'auto'))
    resolutions = cfg.get('resolution_transfer', {}).get('eval_resolutions', [32, 64])
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    
    if not data_cfg and isinstance(cfg.get("dataset", {}), dict):
        data_cfg = {"path": cfg.get("dataset", {}).get("path"), "batch_size": cfg.get("dataset", {}).get("batch_size", 8)}
    
    data_path = Path(eval_cfg.get("dataset_path", data_cfg.get("path", "")))
    if not str(data_path):
        raise KeyError(
            "Resolution-transfer dataset path is missing. Set `eval.dataset_path` (preferred) "
            "or `data.path` in the config."
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Resolution-transfer dataset not found: {data_path}. "
            "Prepare the dedicated cross-resolution data first (e.g., train_32/test_64)."
        )
    ds = MultiResolutionDataset(data_path, resolutions)
    loader = DataLoader(ds, batch_size=eval_cfg.get('batch_size', data_cfg.get('batch_size', 8)))
    _validate_resolution_transfer_channels(cfg, loader, resolutions)
    model = build_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    target_denorm = load_target_denorm(data_path) if bool(eval_cfg.get("report_physical_metrics", True)) else None
    rows = {}

    for res in resolutions:
        metrics_acc = MetricAccumulator()
        metrics_physical_acc = MetricAccumulator()
        n = 0

        for batch in loader:
            x = batch[f'x_{res}'].to(device)
            y = batch[f'y_{res}'].to(device)
            pred = _model_output(model, x)
            metrics_acc.update(pred, y)

            if target_denorm is not None:
                offset, scale = target_denorm
                pred_physical = pred * float(scale) + float(offset)
                y_physical = y * float(scale) + float(offset)
                metrics_physical_acc.update(pred_physical, y_physical)

            n += x.size(0)

        rows[str(res)] = metrics_acc.compute()
        rows[str(res)]["num_samples"] = int(n)
        if target_denorm is not None:
            rows[str(res)].update({f"{k}_physical": v for k, v in metrics_physical_acc.compute().items()})
            rows[str(res)]["target_offset"] = float(target_denorm[0])
            rows[str(res)]["target_scale"] = float(target_denorm[1])

    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    summary = {
        "evaluation_type": "proxy_resolution_transfer",
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "dataset_path": str(data_path),
        "dataset_num_samples": int(len(ds)),
        "eval_resolutions": [int(r) for r in resolutions],
        "rows": rows,
    }
    print(summary)
    output_path = (
        Path(args.output)
        if args.output
        else Path(output_dir) / "resolution_transfer_proxy.json"
    )
    save_json(summary, output_path)


if __name__ == '__main__':
    main()
