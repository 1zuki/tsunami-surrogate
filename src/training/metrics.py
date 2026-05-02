from __future__ import annotations

import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def rel_l2(pred, target, eps=1e-8):
    return torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + eps)


def max_error(pred, target):
    return torch.max(torch.abs(pred - target))


def compute_metrics(pred, target):
    return {
        'mae': float(mae(pred, target).detach().cpu()),
        'rmse': float(rmse(pred, target).detach().cpu()),
        'rel_l2': float(rel_l2(pred, target).detach().cpu()),
        'max_error': float(max_error(pred, target).detach().cpu()),
    }
