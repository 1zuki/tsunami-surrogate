from __future__ import annotations

import math

import torch


def error_uncertainty_correlation(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> float:
    std = torch.sqrt(torch.clamp(variance, min=1e-12)).reshape(-1).float()
    err = torch.abs(target - mean).reshape(-1).float()

    if std.numel() < 2:
        return 0.0

    std = std - std.mean()
    err = err - err.mean()
    denom = torch.sqrt((std * std).sum()) * torch.sqrt((err * err).sum())

    if float(denom) < 1e-12:
        return 0.0

    corr = (std * err).sum() / denom

    return float(corr.item())


class ErrorUncertaintyCorrelationAccumulator:
    """Stream a single Pearson correlation over all evaluated elements."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_uncertainty = 0.0
        self.sum_error = 0.0
        self.sum_uncertainty_sq = 0.0
        self.sum_error_sq = 0.0
        self.sum_product = 0.0

    def update(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        uncertainty = torch.sqrt(
            torch.clamp(variance, min=1.0e-12)
        ).reshape(-1).to(torch.float64)
        error = torch.abs(target - mean).reshape(-1).to(torch.float64)
        if uncertainty.numel() != error.numel():
            raise ValueError("Uncertainty and error element counts differ")
        if not bool(torch.isfinite(uncertainty).all().item()):
            raise FloatingPointError("Nonfinite uncertainty encountered")
        if not bool(torch.isfinite(error).all().item()):
            raise FloatingPointError("Nonfinite absolute error encountered")

        self.count += int(uncertainty.numel())
        self.sum_uncertainty += float(uncertainty.sum().cpu())
        self.sum_error += float(error.sum().cpu())
        self.sum_uncertainty_sq += float((uncertainty * uncertainty).sum().cpu())
        self.sum_error_sq += float((error * error).sum().cpu())
        self.sum_product += float((uncertainty * error).sum().cpu())

    def compute(self) -> float:
        if self.count < 2:
            return 0.0
        count = float(self.count)
        covariance = self.sum_product - (
            self.sum_uncertainty * self.sum_error / count
        )
        uncertainty_var = self.sum_uncertainty_sq - (
            self.sum_uncertainty * self.sum_uncertainty / count
        )
        error_var = self.sum_error_sq - (
            self.sum_error * self.sum_error / count
        )
        denominator = math.sqrt(max(uncertainty_var, 0.0)) * math.sqrt(
            max(error_var, 0.0)
        )
        if denominator <= 1.0e-12:
            return 0.0
        correlation = covariance / denominator
        if not math.isfinite(correlation):
            raise FloatingPointError(
                f"Nonfinite error-uncertainty correlation: {correlation!r}"
            )
        return float(max(-1.0, min(1.0, correlation)))
