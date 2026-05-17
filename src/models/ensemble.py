from __future__ import annotations

from typing import Iterable, List
import torch
from torch import nn


class EnsemblePredictor(nn.Module):
    """aggregates predictions from multiple trained models"""

    def __init__(self, members: Iterable[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(list(members))
        n = len(self.members)
        if n < 2:
            raise ValueError(
                f"EnsemblePredictor requires at least 2 members for meaningful predictive variance, got {n}."
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        preds: List[torch.Tensor] = []
 
        for model in self.members:
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]

            preds.append(out)

        stack = torch.stack(preds, dim=0)

        return {
            'members': stack,
            'mean': stack.mean(dim=0),
            'variance': stack.var(dim=0, unbiased=False),
            'std': stack.std(dim=0, unbiased=False),
        }
