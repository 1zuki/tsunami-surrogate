from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _ensure_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _select_frames(num_frames: int, max_frames: int = 5) -> List[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, max_frames, dtype=int).tolist()


def save_metric_curves(
    metric_dict: Mapping[str, Sequence[float]],
    save_path: Union[str, Path],
    title: str = "Metric curves over rollout horizon",
    xlabel: str = "Timestep",
) -> None:
    save_path = _ensure_path(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = None
    for name, values in metric_dict.items():
        values_arr = np.asarray(values, dtype=np.float64)
        if x is None:
            x = np.arange(values_arr.size)
        ax.plot(x, values_arr, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_error_histogram(
    errors: ArrayLike,
    save_path: Union[str, Path],
    title: str = "Error histogram",
    bins: int = 40,
) -> None:
    save_path = _ensure_path(save_path)
    arr = _to_numpy(errors).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arr, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_generalization_bar_chart(
    values: Mapping[str, float],
    save_path: Union[str, Path],
    title: str,
    ylabel: str,
) -> None:
    save_path = _ensure_path(save_path)
    labels = list(values.keys())
    heights = [float(values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels)), 5))
    bars = ax.bar(np.arange(len(labels)), heights)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, height in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_spatial_map(
    field: ArrayLike,
    save_path: Union[str, Path],
    title: str,
    cmap: str = "viridis",
    colorbar_label: str = "Value",
) -> None:
    save_path = _ensure_path(save_path)
    arr = _to_numpy(field)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape {arr.shape}")
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(arr, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_rollout_comparison(
    target: ArrayLike,
    prediction: ArrayLike,
    save_path: Union[str, Path],
    title: str = "Rollout comparison",
    frames: Optional[Sequence[int]] = None,
) -> None:
    save_path = _ensure_path(save_path)
    target_arr = _to_numpy(target)
    pred_arr = _to_numpy(prediction)

    if target_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Rollout comparison expects [T,H,W] arrays.")
    if target_arr.shape != pred_arr.shape:
        raise ValueError(f"Target and prediction shapes must match. Got {target_arr.shape} and {pred_arr.shape}")

    num_frames = target_arr.shape[0]
    selected = list(frames) if frames is not None else _select_frames(num_frames)
    ncols = len(selected)
    fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 9), squeeze=False)
    shared_vmin = float(min(target_arr.min(), pred_arr.min()))
    shared_vmax = float(max(target_arr.max(), pred_arr.max()))
    error_arr = np.abs(pred_arr - target_arr)
    error_vmax = float(error_arr.max()) if error_arr.max() > 0 else 1.0

    for col, t in enumerate(selected):
        ax_target = axes[0, col]
        ax_pred = axes[1, col]
        ax_err = axes[2, col]
        im0 = ax_target.imshow(target_arr[t], origin="lower", vmin=shared_vmin, vmax=shared_vmax)
        im1 = ax_pred.imshow(pred_arr[t], origin="lower", vmin=shared_vmin, vmax=shared_vmax)
        im2 = ax_err.imshow(error_arr[t], origin="lower", vmin=0.0, vmax=error_vmax, cmap="magma")

        ax_target.set_title(f"Target t={t}")
        ax_pred.set_title(f"Prediction t={t}")
        ax_err.set_title(f"Absolute error t={t}")
        for ax in (ax_target, ax_pred, ax_err):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title)
    fig.colorbar(im0, ax=axes[0, :].ravel().tolist(), shrink=0.65)
    fig.colorbar(im1, ax=axes[1, :].ravel().tolist(), shrink=0.65)
    fig.colorbar(im2, ax=axes[2, :].ravel().tolist(), shrink=0.65)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_uncertainty_panel(
    mean_field: ArrayLike,
    std_field: ArrayLike,
    target_field: Optional[ArrayLike],
    save_path: Union[str, Path],
    title: str = "Uncertainty visualization",
) -> None:
    save_path = _ensure_path(save_path)
    mean_arr = _to_numpy(mean_field)
    std_arr = _to_numpy(std_field)
    target_arr = _to_numpy(target_field) if target_field is not None else None

    num_panels = 4 if target_arr is not None else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 4.5), squeeze=False)
    axes = axes[0]

    vmin = float(mean_arr.min())
    vmax = float(mean_arr.max())
    im0 = axes[0].imshow(mean_arr, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Predictive mean")
    im1 = axes[1].imshow(std_arr, origin="lower", cmap="magma")
    axes[1].set_title("Predictive std")

    if target_arr is not None:
        im2 = axes[2].imshow(target_arr, origin="lower", vmin=vmin, vmax=vmax)
        axes[2].set_title("Target")
        abs_err = np.abs(mean_arr - target_arr)
        im3 = axes[3].imshow(abs_err, origin="lower", cmap="magma")
        axes[3].set_title("Absolute error")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)
        fig.colorbar(im3, ax=axes[3], shrink=0.8)

    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_reliability_curve(
    nominal_coverages: Sequence[float],
    observed_coverages: Sequence[float],
    save_path: Union[str, Path],
    title: str = "Uncertainty calibration",
) -> None:
    save_path = _ensure_path(save_path)
    nominal = np.asarray(nominal_coverages, dtype=np.float64)
    observed = np.asarray(observed_coverages, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(nominal, observed, marker="o", label="Observed")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_scatter(
    x: ArrayLike,
    y: ArrayLike,
    save_path: Union[str, Path],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    save_path = _ensure_path(save_path)
    x_arr = _to_numpy(x).reshape(-1)
    y_arr = _to_numpy(y).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_arr, y_arr, alpha=0.35, s=12)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
