from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import torch
from torch import optim

from .losses import build_loss
from .engine import train_one_epoch, evaluate_epoch
from .callbacks import EarlyStopping
from .checkpointing import save_checkpoint, load_checkpoint
from src.utils.io import save_json, load_json


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
        self.early = EarlyStopping(
            early_cfg.get("patience", 10),
            early_mode,
            min_delta=float(early_cfg.get("min_delta", 0.0)),
        )
        checkpoint_mode = str(train_cfg.get("checkpoint_mode", early_mode)).strip().lower()
        if checkpoint_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported checkpoint_mode: {checkpoint_mode}. Use 'min' or 'max'.")
        self.checkpoint_mode = checkpoint_mode
        self.checkpoint_min_delta = float(
            train_cfg.get("checkpoint_min_delta", early_cfg.get("min_delta", 0.0))
        )

    def _resume_from(self, resume_path: Path):
        ckpt = load_checkpoint(resume_path, self.model, self.optimizer, self.scheduler, map_location=self.device)
        state = ckpt.get("trainer_state") or {}

        start_epoch = int(state.get("epoch", ckpt.get("epoch", 0))) + 1
        best_value = state.get("best_value")
        if best_value is None:
            best_value = float("inf") if self.checkpoint_mode == "min" else -float("inf")

        self.early.best = state.get("early_best")
        self.early.count = int(state.get("early_count", 0))

        history_path = self.output_dir / "history.json"
        history = load_json(history_path) if history_path.exists() else []
        history = [r for r in history if int(r.get("epoch", 0)) < start_epoch]

        print(f"[train] resuming from {resume_path} at epoch {start_epoch} (best {self.checkpoint_mode}={best_value})")
        return start_epoch, float(best_value), history

    def fit(self, resume_path=None):
        train_cfg = self.cfg.get("train", {})
        epochs = int(train_cfg.get("epochs", 5))
        grad_clip = train_cfg.get("grad_clip", None)

        if resume_path is not None:
            start_epoch, best_value, history = self._resume_from(Path(resume_path))
        else:
            start_epoch = 1
            best_value = float("inf") if self.checkpoint_mode == "min" else -float("inf")
            history = []

        epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = train_one_epoch(self.model, self.loaders["train"], self.optimizer, self.loss_fn, self.device, grad_clip)
            val_metrics = evaluate_epoch(self.model, self.loaders["val"], self.loss_fn, self.device) if "val" in self.loaders else {}

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            row["lr"] = float(self.optimizer.param_groups[0]["lr"])
            history.append(row)
            metric_name = train_cfg.get("checkpoint_metric", "val_rel_l2")
            value = row.get(metric_name, row.get("val_loss", row.get("train_loss")))

            if value is not None and self._is_checkpoint_improved(float(value), float(best_value)):
                best_value = value
                save_checkpoint(self.output_dir / "best.pt", self.model, self.optimizer, epoch, row, self.cfg,
                                scheduler=self.scheduler, trainer_state=self._trainer_state(epoch, best_value))

            save_json(history, self.output_dir / "history.json")
            print(row)

            stop = value is not None and self.early.step(float(value))

            if self.scheduler is not None and not stop:
                self.scheduler.step()

            save_checkpoint(self.checkpoint_dir / "last.pt", self.model, self.optimizer, epoch, history[-1], self.cfg,
                            scheduler=self.scheduler, trainer_state=self._trainer_state(epoch, best_value))

            if stop:
                print(f"Early stopping at epoch {epoch}")
                break

        return history

    def _trainer_state(self, epoch, best_value):
        return {
            "epoch": int(epoch),
            "best_value": float(best_value),
            "early_best": self.early.best,
            "early_count": int(self.early.count),
        }

    def _is_checkpoint_improved(self, value: float, best_value: float) -> bool:
        if self.checkpoint_mode == "min":
            return value < best_value - self.checkpoint_min_delta
        return value > best_value + self.checkpoint_min_delta
