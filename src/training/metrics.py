from __future__ import annotations

import torch


class MetricAccumulator:
    """Dataset-level aggregate metrics accumulated from raw error tensors."""
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0
        self.sum_target_sq = 0.0
        self.n_elements = 0
        self.max_abs_err = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = (pred - target).detach()
        if diff.numel() == 0:
            return

        abs_err = torch.abs(diff)
        target_detached = target.detach()

        self.sum_abs_err += float(abs_err.sum().cpu())
        self.sum_sq_err += float((diff * diff).sum().cpu())
        self.sum_target_sq += float((target_detached * target_detached).sum().cpu())
        self.n_elements += int(diff.numel())
        self.max_abs_err = max(self.max_abs_err, float(abs_err.max().cpu()))

    def compute(self) -> dict[str, float]:
        n = max(1, int(self.n_elements))
        return {
            "mae": self.sum_abs_err / float(n),
            "rmse": (self.sum_sq_err / float(n)) ** 0.5,
            # This eps is added to the final dataset norm, not each batch norm.
            "rel_l2": (self.sum_sq_err ** 0.5) / ((self.sum_target_sq ** 0.5) + self.eps),
            "max_error": self.max_abs_err,
        }


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
