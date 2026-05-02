from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F


def resize_field(x: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear') -> torch.Tensor:
    """Resize a field tensor [B,C,H,W] or [C,H,W] to a new spatial size."""
    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    if x.dim() != 4:
        raise ValueError(f'Expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}')
    align = False if mode in {'bilinear', 'bicubic'} else None
    out = F.interpolate(x, size=size, mode=mode, align_corners=align)
    return out.squeeze(0) if squeeze else out


def restrict_fine_to_coarse(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Average-pool a fine-grid field to a coarse grid."""
    if factor <= 0:
        raise ValueError('factor must be positive')
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)


def project_coarse_to_fine(x: torch.Tensor, factor: int = 2, mode: str = 'bilinear') -> torch.Tensor:
    h, w = x.shape[-2:]
    return resize_field(x, (h * factor, w * factor), mode=mode)
