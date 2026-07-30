from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import random
import numpy as np
import torch
from torch import optim

from .losses import build_loss
from .engine import train_one_epoch, evaluate_epoch
from .callbacks import EarlyStopping
from .checkpointing import save_checkpoint, load_checkpoint
from src.utils.io import save_json, load_json


def _capture_rng_state() -> Dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(bit_generator),
            "keys": keys.astype(np.uint32, copy=False).tolist(),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    elif numpy_state is not None:
        # Compatibility with any in-memory state captured before the
        # weights-only-safe representation above was introduced.
        np.random.set_state(numpy_state)
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sampler_objects(loader: Any) -> Dict[str, Any]:
    samplers: Dict[str, Any] = {}
    seen: set[int] = set()
    for name in ("batch_sampler", "sampler"):
        sampler = getattr(loader, name, None)
        if sampler is None or id(sampler) in seen:
            continue
        seen.add(id(sampler))
        samplers[name] = sampler
    return samplers


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _loader_contract(loader: Any) -> Dict[str, Any] | None:
    required = (
        "dataset",
        "batch_size",
        "drop_last",
        "num_workers",
        "persistent_workers",
        "sampler",
        "batch_sampler",
    )
    if any(not hasattr(loader, name) for name in required):
        return None
    return {
        "dataset_type": _type_name(loader.dataset),
        "dataset_length": int(len(loader.dataset)),
        "batch_count": int(len(loader)),
        "batch_size": (
            None if loader.batch_size is None else int(loader.batch_size)
        ),
        "drop_last": bool(loader.drop_last),
        "num_workers": int(loader.num_workers),
        "persistent_workers": bool(loader.persistent_workers),
        "sampler_type": _type_name(loader.sampler),
        "batch_sampler_type": _type_name(loader.batch_sampler),
    }


def _capture_loader_state(loader: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {"samplers": {}}
    contract = _loader_contract(loader)
    if contract is not None:
        state["contract"] = contract
    for name, sampler in _sampler_objects(loader).items():
        if hasattr(sampler, "state_dict"):
            state["samplers"][name] = sampler.state_dict()

    generator = getattr(loader, "generator", None)
    if isinstance(generator, torch.Generator):
        state["generator_state"] = generator.get_state()
    return state


def _restore_loader_state(loader: Any, state: Dict[str, Any] | None) -> None:
    if not state:
        return
    expected_contract = state.get("contract")
    if isinstance(expected_contract, Mapping):
        observed_contract = _loader_contract(loader)
        if observed_contract is None or dict(expected_contract) != observed_contract:
            raise ValueError(
                "Training DataLoader contract mismatch during resume. "
                f"checkpoint={dict(expected_contract)}, runtime={observed_contract}"
            )

    samplers = _sampler_objects(loader)
    for name, sampler_state in dict(state.get("samplers", {})).items():
        sampler = samplers.get(name)
        if sampler is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(sampler_state)

    generator = getattr(loader, "generator", None)
    generator_state = state.get("generator_state")
    if isinstance(generator, torch.Generator) and generator_state is not None:
        generator.set_state(generator_state)


def _set_loader_epoch(loader: Any, epoch_index: int) -> None:
    for sampler in _sampler_objects(loader).values():
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch_index))


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
        ckpt = load_checkpoint(
            resume_path,
            self.model,
            self.optimizer,
            self.scheduler,
            map_location=self.device,
            validate_training_data=True,
        )
        state = ckpt.get("trainer_state") or {}

        start_epoch = int(state.get("epoch", ckpt.get("epoch", 0))) + 1
        best_value = state.get("best_value")
        if best_value is None:
            best_value = float("inf") if self.checkpoint_mode == "min" else -float("inf")

        self.early.best = state.get("early_best")
        self.early.count = int(state.get("early_count", 0))
        _restore_loader_state(self.loaders["train"], state.get("train_loader_state"))
        _restore_rng_state(state.get("rng_state"))

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
            _set_loader_epoch(self.loaders["train"], epoch - 1)
            train_metrics = train_one_epoch(self.model, self.loaders["train"], self.optimizer, self.loss_fn, self.device, grad_clip)
            val_metrics = evaluate_epoch(self.model, self.loaders["val"], self.loss_fn, self.device) if "val" in self.loaders else {}

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            row["lr"] = float(self.optimizer.param_groups[0]["lr"])
            history.append(row)
            metric_name = train_cfg.get("checkpoint_metric", "val_rel_l2")
            value = row.get(metric_name, row.get("val_loss", row.get("train_loss")))

            improved = value is not None and self._is_checkpoint_improved(
                float(value), float(best_value)
            )
            if improved:
                best_value = float(value)

            save_json(history, self.output_dir / "history.json")
            print(row)

            stop = value is not None and self.early.step(float(value))

            if self.scheduler is not None and not stop:
                self.scheduler.step()

            trainer_state = self._trainer_state(epoch, best_value)
            if improved:
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    epoch,
                    row,
                    self.cfg,
                    scheduler=self.scheduler,
                    trainer_state=trainer_state,
                )

            save_checkpoint(self.checkpoint_dir / "last.pt", self.model, self.optimizer, epoch, history[-1], self.cfg,
                            scheduler=self.scheduler, trainer_state=trainer_state)

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
            "rng_state": _capture_rng_state(),
            "train_loader_state": _capture_loader_state(self.loaders["train"]),
        }

    def _is_checkpoint_improved(self, value: float, best_value: float) -> bool:
        if self.checkpoint_mode == "min":
            return value < best_value - self.checkpoint_min_delta
        return value > best_value + self.checkpoint_min_delta
