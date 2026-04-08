from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.data_gen.dataset import DatasetStats, TsunamiDataset, compute_stats, denormalize_inputs
from src.models import build_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import CompositeLoss
from src.training.metrics import average_metric_dicts, compute_metrics_torch
from src.utils.config import load_config, maybe_resolve_device
from src.utils.logger import ExperimentLogger
from src.utils.seed import make_generator, seed_worker, set_seed
from src.utils.visualization import plot_prediction_vs_truth


def _get_stats(config: dict) -> DatasetStats:
    paths = config.get("paths", {})
    stats_path = Path(paths.get("stats_file", "data/synthetic/default/stats.yaml"))
    if stats_path.exists():
        return DatasetStats.load(stats_path)
    train_file = paths.get("train_file")
    stats = compute_stats(train_file, input_keys=["bathymetry", "disturbance"], target_key="wave")
    stats.save(stats_path)
    return stats


def _make_loader(file_path: str, stats: DatasetStats, config: dict, train: bool) -> DataLoader:
    dataset = TsunamiDataset(
        file_path=file_path,
        input_keys=["bathymetry", "disturbance"],
        target_key="wave",
        stats=stats,
        normalize_input=bool(config.get("normalization", {}).get("normalize_inputs", True)),
        normalize_target=bool(config.get("normalization", {}).get("normalize_targets", True)),
        augment=config.get("augmentation", {}) if train else None,
        return_meta=False,
    )
    trn = config.get("training", {})
    return DataLoader(
        dataset,
        batch_size=int(trn.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(trn.get("num_workers", 0)),
        pin_memory=bool(trn.get("pin_memory", False)),
        worker_init_fn=seed_worker if int(trn.get("num_workers", 0)) > 0 else None,
        generator=make_generator(int(config.get("project", {}).get("seed", 42))),
    )


def _denorm_if_needed(tensor: torch.Tensor, stats: DatasetStats, normalize_targets: bool) -> torch.Tensor:
    if not normalize_targets:
        return tensor
    return tensor * stats.target_std + stats.target_mean


def run_epoch(model, loader, criterion, device, stats, normalize_targets: bool, optimizer=None, scaler=None, grad_clip: float | None = None, mixed_precision: bool = False):
    training = optimizer is not None
    model.train(training)
    loss_logs = []
    metric_logs = []
    first_batch = None

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = mixed_precision and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            pred = model(x)
            loss, loss_log = criterion(pred, y)

        if training:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            pred_eval = _denorm_if_needed(pred, stats, normalize_targets)
            y_eval = _denorm_if_needed(y, stats, normalize_targets)
            metric_logs.append(compute_metrics_torch(pred_eval, y_eval))
            loss_logs.append(loss_log)
            if first_batch is None:
                first_batch = (x.detach().cpu(), y_eval.detach().cpu(), pred_eval.detach().cpu())

    if not loss_logs or not metric_logs:
        return {}, first_batch
    losses = average_metric_dicts(loss_logs)
    metrics = average_metric_dicts(metric_logs)
    metrics = {**losses, **{f"metric_{k}": v for k, v in metrics.items()}}
    return metrics, first_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tsunami surrogate model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = maybe_resolve_device(str(config.get("project", {}).get("device", "auto")))
    device = torch.device(device_name)
    set_seed(int(config.get("project", {}).get("seed", 42)), deterministic=bool(config.get("project", {}).get("deterministic", False)))

    paths = config.get("paths", {})
    logger = ExperimentLogger(paths.get("log_dir", "results/default/logs"), name=config.get("project", {}).get("name", "train"))
    logger.save_config(config)
    stats = _get_stats(config)

    train_loader = _make_loader(paths.get("train_file"), stats, config, train=True)
    val_loader = _make_loader(paths.get("val_file"), stats, config, train=False)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("lr", 1e-3)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-6)),
    )
    scheduler_cfg = config.get("training", {}).get("scheduler", {})
    scheduler_name = str(scheduler_cfg.get("name", "plateau")).lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 4)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.get("training", {}).get("epochs", 40)))
    else:
        scheduler = None

    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    criterion = CompositeLoss(config.get("loss", {}))
    checkpoint = ModelCheckpoint(paths.get("checkpoint_dir", "results/default/checkpoints"), monitor="val_rmse", mode="min")
    early_stopping = EarlyStopping(
        patience=int(config.get("training", {}).get("early_stopping_patience", 10)),
        min_delta=float(config.get("training", {}).get("early_stopping_min_delta", 0.0)),
        mode="min",
    )
    use_amp = bool(config.get("training", {}).get("mixed_precision", False))
    try:
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    best_visual_saved = False
    normalize_targets = bool(config.get("normalization", {}).get("normalize_targets", True))
    normalize_inputs_flag = bool(config.get("normalization", {}).get("normalize_inputs", True))

    for epoch in range(start_epoch, int(config.get("training", {}).get("epochs", 40)) + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            stats,
            normalize_targets,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=float(config.get("training", {}).get("grad_clip", 1.0)),
            mixed_precision=use_amp,
        )
        val_metrics, first_batch = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            stats,
            normalize_targets,
            optimizer=None,
            scaler=None,
            grad_clip=None,
            mixed_precision=use_amp,
        )

        logger.log_metrics(epoch, "train", train_metrics)
        if val_metrics:
            logger.log_metrics(epoch, "val", val_metrics)
            val_rmse = float(val_metrics.get("metric_rmse", train_metrics.get("metric_rmse", train_metrics.get("rmse", 0.0))))
        else:
            logger.info("Validation split is empty; falling back to training RMSE for checkpointing.")
            val_rmse = float(train_metrics.get("metric_rmse", train_metrics.get("rmse", 0.0)))
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "stats": stats.to_dict(),
            "val_metrics": val_metrics,
        }
        saved = checkpoint.save(state, score=val_rmse)
        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_rmse)
            else:
                scheduler.step()

        if saved["best"] is not None and first_batch is not None:
            x0, y0, p0 = first_batch
            x0_np = x0[0].numpy()
            if normalize_inputs_flag:
                x0_np = denormalize_inputs(x0_np, stats)
            figure_dir = Path(paths.get("figure_dir", "results/default/figures"))
            figure_dir.mkdir(parents=True, exist_ok=True)
            plot_prediction_vs_truth(
                bathymetry=x0_np[0],
                disturbance=x0_np[1],
                truth=y0[0].numpy(),
                pred=p0[0].numpy(),
                path=figure_dir / f"best_epoch_{epoch}.png",
            )
            best_visual_saved = True

        if early_stopping.step(val_rmse):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training finished.")
    if not best_visual_saved:
        logger.info("No best-epoch visualization was saved.")


if __name__ == "__main__":
    main()
