from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch


def save_prediction_triplet(x: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = x[0, 0].detach().cpu()
    pred0 = pred[0, 0].detach().cpu()
    target0 = target[0, 0].detach().cpu()
    err = torch.abs(pred0 - target0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, arr, title in zip(axes, [source, pred0, target0, err], ['source', 'prediction', 'target', 'abs error']):
        im = ax.imshow(arr)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
