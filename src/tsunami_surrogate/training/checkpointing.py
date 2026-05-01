from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'metrics': metrics,
        'config': cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt
