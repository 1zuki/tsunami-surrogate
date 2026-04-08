from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot_fields(fields: dict[str, np.ndarray], path: str | Path, title: str | None = None, cmap: str = "viridis", dpi: int = 140) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, fields.items()):
        arr = _to_numpy(arr)
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_vs_truth(bathymetry, disturbance, truth, pred, path: str | Path, timesteps: Sequence[int] | None = None, cmap_wave: str = "RdBu_r", cmap_bathy: str = "viridis", dpi: int = 140) -> None:
    bathymetry = _to_numpy(bathymetry)
    disturbance = _to_numpy(disturbance)
    truth = _to_numpy(truth)
    pred = _to_numpy(pred)
    n_time = truth.shape[0]
    if timesteps is None:
        timesteps = np.linspace(0, n_time - 1, num=min(4, n_time), dtype=int).tolist()

    rows = 2 + len(timesteps)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.2 * rows), constrained_layout=True)

    im0 = axes[0, 0].imshow(bathymetry, cmap=cmap_bathy, origin="lower")
    axes[0, 0].set_title("Bathymetry")
    axes[0, 1].imshow(disturbance, cmap=cmap_wave, origin="lower")
    axes[0, 1].set_title("Initial disturbance")
    axes[0, 2].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    abs_err = np.abs(pred - truth)
    axes[1, 0].imshow(np.max(np.abs(truth), axis=0), cmap=cmap_wave, origin="lower")
    axes[1, 0].set_title("Max |truth|")
    axes[1, 1].imshow(np.max(np.abs(pred), axis=0), cmap=cmap_wave, origin="lower")
    axes[1, 1].set_title("Max |pred|")
    im_err = axes[1, 2].imshow(np.mean(abs_err, axis=0), cmap="magma", origin="lower")
    axes[1, 2].set_title("Mean absolute error")
    fig.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for row, t in enumerate(timesteps, start=2):
        v = float(max(np.max(np.abs(truth[t])), np.max(np.abs(pred[t])), 1e-6))
        im_a = axes[row, 0].imshow(truth[t], cmap=cmap_wave, origin="lower", vmin=-v, vmax=v)
        im_b = axes[row, 1].imshow(pred[t], cmap=cmap_wave, origin="lower", vmin=-v, vmax=v)
        im_c = axes[row, 2].imshow(pred[t] - truth[t], cmap=cmap_wave, origin="lower")
        axes[row, 0].set_title(f"Truth t={t}")
        axes[row, 1].set_title(f"Pred t={t}")
        axes[row, 2].set_title(f"Error t={t}")
        fig.colorbar(im_a, ax=axes[row, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im_b, ax=axes[row, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im_c, ax=axes[row, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_time_series_at_points(truth, pred, points: Iterable[tuple[int, int]], path: str | Path, dpi: int = 140) -> None:
    truth = _to_numpy(truth)
    pred = _to_numpy(pred)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    t = np.arange(truth.shape[0])
    for idx, (y, x) in enumerate(points):
        ax.plot(t, truth[:, y, x], label=f"truth p{idx}")
        ax.plot(t, pred[:, y, x], linestyle="--", label=f"pred p{idx}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Wave height")
    ax.legend(ncol=2, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(mean, std, truth, path: str | Path, timestep: int | None = None, dpi: int = 140) -> None:
    mean = _to_numpy(mean)
    std = _to_numpy(std)
    truth = _to_numpy(truth)
    if timestep is None:
        timestep = mean.shape[0] // 2
    err = np.abs(mean[timestep] - truth[timestep])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fields = [truth[timestep], mean[timestep], std[timestep], err]
    titles = ["Truth", "Predictive mean", "Predictive std", "Absolute error"]
    cmaps = ["RdBu_r", "RdBu_r", "magma", "magma"]
    for ax, arr, title, cmap in zip(axes, fields, titles, cmaps):
        im = ax.imshow(arr, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_wave_animation(field_seq, path: str | Path, bathymetry=None, interval: int = 200, cmap: str = "RdBu_r", dpi: int = 120) -> None:
    field_seq = _to_numpy(field_seq)
    bathymetry = None if bathymetry is None else _to_numpy(bathymetry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vmax = float(np.max(np.abs(field_seq)) + 1e-8)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(field_seq[0], origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, animated=True)
    if bathymetry is not None:
        ax.contour(bathymetry, levels=8, colors="k", linewidths=0.3, alpha=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Wave evolution")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame: int):
        im.set_data(field_seq[frame])
        ax.set_title(f"Wave evolution t={frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=field_seq.shape[0], interval=interval, blit=True)
    try:
        ani.save(path, writer=animation.PillowWriter(fps=max(1, int(1000 / max(interval, 1)))))
    except Exception:
        png_fallback = path.with_suffix(".png")
        fig.savefig(png_fallback, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
