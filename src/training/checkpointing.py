from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, cfg, scheduler=None, trainer_state=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'trainer_state': trainer_state,
        'epoch': epoch,
        'metrics': metrics,
        'config': cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])

    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if scheduler is not None and ckpt.get('scheduler_state') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])

    return ckpt
