from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import torch
from torch import optim

from .losses import build_loss
from .engine import train_one_epoch, evaluate_epoch
from .callbacks import EarlyStopping
from .checkpointing import save_checkpoint
from src.utils.io import save_json


class Trainer:
    def __init__(self, model, loaders, cfg: Dict[str, Any], device):
        self.model = model.to(device)
        self.loaders = loaders
        self.cfg = cfg
        self.device = device

        self.output_dir = Path(cfg.get("output_dir", "experiments/default"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg.get("train", {})

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        )

        self.loss_fn = build_loss(train_cfg.get("loss", "mse"), train_cfg=train_cfg)
        scheduler_name = str(train_cfg.get("scheduler", "none")).lower()
        if scheduler_name == "cosine":
            t_max = max(1, int(train_cfg.get("epochs", 5)))
            min_lr = float(train_cfg.get("min_lr", 1e-5))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=min_lr)
        else:
            self.scheduler = None

        early_cfg = train_cfg.get("early_stopping", {})
        early_mode = str(early_cfg.get("mode", "min")).strip().lower()
        if early_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported early_stopping.mode: {early_mode}. Use 'min' or 'max'.")
        self.early = EarlyStopping(early_cfg.get("patience", 10), early_mode)
        checkpoint_mode = str(train_cfg.get("checkpoint_mode", early_mode)).strip().lower()
        if checkpoint_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported checkpoint_mode: {checkpoint_mode}. Use 'min' or 'max'.")
        self.checkpoint_mode = checkpoint_mode

    def fit(self):
        train_cfg = self.cfg.get("train", {})
        epochs = int(train_cfg.get("epochs", 5))
        grad_clip = train_cfg.get("grad_clip", None)
        best_value = float("inf") if self.checkpoint_mode == "min" else -float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(self.model, self.loaders["train"], self.optimizer, self.loss_fn, self.device, grad_clip)
            val_metrics = evaluate_epoch(self.model, self.loaders["val"], self.loss_fn, self.device) if "val" in self.loaders else {}

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            row["lr"] = float(self.optimizer.param_groups[0]["lr"])
            history.append(row)
            metric_name = train_cfg.get("checkpoint_metric", "val_rel_l2")
            value = row.get(metric_name, row.get("val_loss", row.get("train_loss")))

            if value is not None and (
                (self.checkpoint_mode == "min" and value < best_value)
                or (self.checkpoint_mode == "max" and value > best_value)
            ):
                best_value = value
                save_checkpoint(self.output_dir / "best.pt", self.model, self.optimizer, epoch, row, self.cfg)

            save_json(history, self.output_dir / "history.json")
            print(row)

            if value is not None and self.early.step(float(value)):
                print(f"Early stopping at epoch {epoch}")
                break

            if self.scheduler is not None:
                self.scheduler.step()

        save_checkpoint(self.checkpoint_dir / "last.pt", self.model, self.optimizer, epoch, history[-1], self.cfg)

        return history
