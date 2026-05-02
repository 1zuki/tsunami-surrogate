from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _to_2d(t: torch.Tensor):
    if t.dim() == 4:
        return t[0, 0].detach().cpu().numpy()
    if t.dim() == 3:
        return t[0].detach().cpu().numpy()
    if t.dim() == 2:
        return t.detach().cpu().numpy()
    raise ValueError(f"Unsupported tensor shape for visualization: {tuple(t.shape)}")


def save_prediction_triplet(x: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x2d = _to_2d(x)
    p2d = _to_2d(pred)
    y2d = _to_2d(target)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [(x2d, "Input"), (p2d, "Prediction"), (y2d, "Target")]
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

