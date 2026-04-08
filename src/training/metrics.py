from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def compute_metrics_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    diff = pred - target
    mse = torch.mean(diff**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(diff))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / (
        torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    )
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    corr = torch.sum(pred_centered * target_centered, dim=1) / (
        torch.sqrt(torch.sum(pred_centered**2, dim=1) + eps) * torch.sqrt(torch.sum(target_centered**2, dim=1) + eps)
    )
    p_mass = pred.sum(dim=(-1, -2, -3))
    t_mass = target.sum(dim=(-1, -2, -3))
    mass_error = torch.mean(torch.abs(p_mass - t_mass) / (torch.abs(t_mass) + eps))
    peak_pred = torch.amax(torch.abs(pred), dim=(-1, -2, -3))
    peak_true = torch.amax(torch.abs(target), dim=(-1, -2, -3))
    peak_error = torch.mean(torch.abs(peak_pred - peak_true) / (peak_true + eps))
    return {
        "mse": float(mse.detach()),
        "rmse": float(rmse.detach()),
        "mae": float(mae.detach()),
        "rel_l2": float(rel_l2.mean().detach()),
        "corr": float(corr.mean().detach()),
        "mass_error": float(mass_error.detach()),
        "peak_error": float(peak_error.detach()),
    }


def compute_metrics_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    diff = pred - target
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    rel = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1) / (np.linalg.norm(target_flat, axis=1) + eps)
    pred_centered = pred_flat - pred_flat.mean(axis=1, keepdims=True)
    target_centered = target_flat - target_flat.mean(axis=1, keepdims=True)
    corr = np.sum(pred_centered * target_centered, axis=1) / (
        np.sqrt(np.sum(pred_centered**2, axis=1) + eps) * np.sqrt(np.sum(target_centered**2, axis=1) + eps)
    )
    p_mass = pred.sum(axis=(-1, -2, -3))
    t_mass = target.sum(axis=(-1, -2, -3))
    mass_error = float(np.mean(np.abs(p_mass - t_mass) / (np.abs(t_mass) + eps)))
    peak_pred = np.max(np.abs(pred), axis=(-1, -2, -3))
    peak_true = np.max(np.abs(target), axis=(-1, -2, -3))
    peak_error = float(np.mean(np.abs(peak_pred - peak_true) / (peak_true + eps)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": float(np.mean(rel)),
        "corr": float(np.mean(corr)),
        "mass_error": mass_error,
        "peak_error": peak_error,
    }


def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}
