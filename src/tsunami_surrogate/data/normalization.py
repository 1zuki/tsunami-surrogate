from __future__ import annotations

import torch


class ChannelNormalizer:
    """Per-channel standardization for [N,C,H,W] tensors."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> 'ChannelNormalizer':
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)
        self.std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError('Normalizer must be fitted before transform.')
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError('Normalizer must be fitted before inverse.')
        return x * self.std.to(x.device) + self.mean.to(x.device)
