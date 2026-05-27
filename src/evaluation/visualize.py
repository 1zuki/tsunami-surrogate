from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from src.evaluation.calibration import interval_calibration
from src.evaluation.target_scaling import load_target_denorm
from src.evaluation.uncertainty import error_uncertainty_correlation
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device


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


def _as_tchw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # [T, C, H, W] or [T, H, W, C], use first channel
        if arr.shape[1] <= 4:
            return arr[:, 0, :, :]
        if arr.shape[-1] <= 4:
            return arr[:, :, :, 0]

    raise ValueError(f"Unsupported rollout shape: {arr.shape}")


def _model_mean_and_variance(
    model: torch.nn.Module,
    x: torch.Tensor,
    mc_samples: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = model(x)
    mean: torch.Tensor
    variance: Optional[torch.Tensor] = None

    if isinstance(out, tuple):
        mean = out[0]
        if len(out) > 1:
            log_var = out[1]
            variance = torch.exp(log_var).clamp_min(1e-12)
    elif isinstance(out, dict):
        mean = out.get("mean", next(iter(out.values())))
        if "variance" in out:
            variance = out["variance"].clamp_min(1e-12)
        elif "log_variance" in out:
            variance = torch.exp(out["log_variance"]).clamp_min(1e-12)
    else:
        mean = out

    if mc_samples > 1:
        prev_mode = model.training
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                sample_out = model(x)
                if isinstance(sample_out, tuple):
                    sample_mean = sample_out[0]
                elif isinstance(sample_out, dict):
                    sample_mean = sample_out.get("mean", next(iter(sample_out.values())))
                else:
                    sample_mean = sample_out
                preds.append(sample_mean)
        pred_stack = torch.stack(preds, dim=0)
        epi_var = pred_stack.var(dim=0, unbiased=False).clamp_min(1e-12)
        model.train(prev_mode)
        if variance is None:
            variance = epi_var
        else:
            variance = variance + epi_var

    return mean, variance


def _load_processed_eval_dataset(processed_path: str | Path) -> Dict[str, np.ndarray]:
    processed_path = Path(processed_path)
    npz_path = processed_path
    if processed_path.is_dir():
        candidate = processed_path / "eval_dataset.npz"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing eval dataset archive: {candidate}")
        npz_path = candidate

    with np.load(npz_path, allow_pickle=True) as data:
        if "inputs" not in data or "targets" not in data:
            raise KeyError(f"{npz_path} must contain 'inputs' and 'targets'")
        out = {
            "inputs": np.asarray(data["inputs"], dtype=np.float32),
            "targets": np.asarray(data["targets"], dtype=np.float32),
        }
        if "sample_id" in data:
            out["sample_id"] = np.asarray(data["sample_id"])
        else:
            out["sample_id"] = np.asarray([f"sample_{i:06d}" for i in range(out["inputs"].shape[0])], dtype=object)
    manifest_path = npz_path.with_name("eval_manifest.json")
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            input_order = manifest.get("input_order")
            if isinstance(input_order, list) and all(isinstance(x, str) for x in input_order):
                out["input_order"] = np.asarray(input_order, dtype=object)
        except Exception:
            pass
    return out


def _normalize_sample_ids(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    out = []
    for x in flat:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def _pick_sample_index(sample_ids: np.ndarray, sample_id: Optional[str], sample_index: int) -> int:
    if sample_id is None:
        idx = int(sample_index)
        if idx < 0 or idx >= len(sample_ids):
            raise IndexError(f"sample_index out of range: {idx} not in [0, {len(sample_ids) - 1}]")
        return idx

    matches = np.where(sample_ids == sample_id)[0]
    if len(matches) == 0:
        raise KeyError(f"sample_id '{sample_id}' not found in processed dataset.")
    return int(matches[0])


def _load_raw_sample_bathymetry_and_timestamps(raw_dir: Path, sample_id: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    sample_npz = raw_dir / sample_id / "sample.npz"
    if not sample_npz.exists():
        return None, None

    with np.load(sample_npz) as data:
        bathymetry = np.asarray(data["bathymetry"], dtype=np.float32) if "bathymetry" in data else None
        timestamps = np.asarray(data["timestamps"], dtype=np.float32) if "timestamps" in data else None
    return bathymetry, timestamps


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    err = pred - target
    abs_err = np.abs(err)
    sq_err = err ** 2

    frame_mae = abs_err.mean(axis=(1, 2))
    frame_rmse = np.sqrt(sq_err.mean(axis=(1, 2)))
    frame_rel_l2 = np.linalg.norm(err.reshape(err.shape[0], -1), axis=1) / (
        np.linalg.norm(target.reshape(target.shape[0], -1), axis=1) + 1e-8
    )

    return {
        "global_mae": float(abs_err.mean()),
        "global_rmse": float(np.sqrt(sq_err.mean())),
        "global_rel_l2": float(np.linalg.norm(err.ravel()) / (np.linalg.norm(target.ravel()) + 1e-8)),
        "global_max_error": float(abs_err.max()),
        "frame_mae": frame_mae,
        "frame_rmse": frame_rmse,
        "frame_rel_l2": frame_rel_l2,
    }


@dataclass
class VisualRollout:
    sample_id: str
    bathymetry: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    uncertainty_std: Optional[np.ndarray]
    timestamps: Optional[np.ndarray]
    metrics: Dict[str, Any]
    uncertainty_metrics: Dict[str, float]
    target_denorm: Optional[tuple[float, float]]
    used_raw_bathymetry: bool
    notes: tuple[str, ...]


def prepare_visual_rollout(
    config_path: str | Path,
    checkpoint_path: str | Path,
    processed_path: str | Path,
    raw_dir: str | Path = "data/raw/samples",
    sample_id: Optional[str] = None,
    sample_index: int = 0,
    mc_samples: int = 0,
    device: str = "auto",
) -> VisualRollout:
    cfg = load_config(config_path)
    model = build_model(cfg)
    dev = resolve_device(device if device != "auto" else cfg.get("device", "auto"))
    load_checkpoint(checkpoint_path, model, map_location=dev)
    model = model.to(dev).eval()

    processed = _load_processed_eval_dataset(processed_path)
    sample_ids = _normalize_sample_ids(processed["sample_id"])
    idx = _pick_sample_index(sample_ids, sample_id=sample_id, sample_index=sample_index)
    sid = str(sample_ids[idx])

    x_np = np.asarray(processed["inputs"][idx], dtype=np.float32)
    y_np = _as_tchw(processed["targets"][idx])
    x = torch.from_numpy(x_np).unsqueeze(0).to(dev)

    with torch.no_grad():
        pred_t, var_t = _model_mean_and_variance(model, x, mc_samples=mc_samples)
    pred_np = _as_tchw(pred_t.squeeze(0).detach().cpu().numpy())
    var_np = _as_tchw(var_t.squeeze(0).detach().cpu().numpy()) if var_t is not None else None

    t = min(pred_np.shape[0], y_np.shape[0])
    pred_np = pred_np[:t]
    y_np = y_np[:t]
    if var_np is not None:
        var_np = var_np[:t]

    target_denorm = load_target_denorm(processed_path)
    if target_denorm is not None:
        offset, scale = target_denorm
        pred_np = pred_np * float(scale) + float(offset)
        y_np = y_np * float(scale) + float(offset)
        if var_np is not None:
            var_np = var_np * float(scale) ** 2

    bathy_raw, ts_raw = _load_raw_sample_bathymetry_and_timestamps(Path(raw_dir), sid)
    notes: list[str] = []
    if bathy_raw is None:
        input_order_values = processed.get("input_order")
        bathy_idx = 0
        if input_order_values is not None:
            order = [str(v) for v in np.asarray(input_order_values).reshape(-1).tolist()]
            if "bathymetry" in order:
                bathy_idx = int(order.index("bathymetry"))
            else:
                notes.append(
                    "Processed input_order has no 'bathymetry' entry; falling back to channel 0 for bathymetry visualization."
                )
        bathymetry = np.asarray(x_np[bathy_idx], dtype=np.float32)
        note = (
            f"Raw bathymetry was not found for this sample, so processed input channel {bathy_idx} is used instead. "
            "This can be normalized/scaled and may not be physical depth units."
        )
        warnings.warn(note, RuntimeWarning)
        notes.append(note)
        used_raw_bathymetry = False
    else:
        bathymetry = bathy_raw
        used_raw_bathymetry = True

    timestamps = None
    if ts_raw is not None:
        # targets are typically forecast frames from t1..tT
        if ts_raw.shape[0] >= t + 1:
            timestamps = ts_raw[1 : t + 1]
        else:
            timestamps = ts_raw[:t]

    metrics = _compute_metrics(pred_np, y_np)
    uncertainty_metrics: Dict[str, float] = {}
    uncertainty_std = np.sqrt(np.clip(var_np, 1e-12, None)) if var_np is not None else None

    if var_np is not None:
        mean_t = torch.from_numpy(pred_np)
        var_t_cpu = torch.from_numpy(var_np)
        target_t = torch.from_numpy(y_np)
        uncertainty_metrics["error_uncertainty_corr"] = error_uncertainty_correlation(mean_t, var_t_cpu, target_t)
        calib = interval_calibration(mean_t, var_t_cpu, target_t, levels=[0.8, 0.9, 0.95])
        uncertainty_metrics.update(calib)
        uncertainty_metrics["std_mean"] = float(uncertainty_std.mean())
        uncertainty_metrics["std_max"] = float(uncertainty_std.max())

    return VisualRollout(
        sample_id=sid,
        bathymetry=bathymetry,
        target=y_np,
        prediction=pred_np,
        uncertainty_std=uncertainty_std,
        timestamps=timestamps,
        metrics=metrics,
        uncertainty_metrics=uncertainty_metrics,
        target_denorm=target_denorm,
        used_raw_bathymetry=used_raw_bathymetry,
        notes=tuple(notes),
    )


class RolloutFigure:
    def __init__(
        self,
        rollout: VisualRollout,
        interval_ms: int = 120,
        repeat: bool = False,
        elev: float = 35.0,
        azim: float = -60.0,
        wave_scale: Optional[float] = None,
        wave_3d_mode: str = "eta",
    ) -> None:
        self.rollout = rollout
        self.interval_ms = int(interval_ms)
        self.repeat = bool(repeat)
        self.elev = float(elev)
        self.azim = float(azim)
        
        mode = str(wave_3d_mode).strip().lower()
        if mode not in {"eta", "overlay"}:
            raise ValueError("wave_3d_mode must be one of: eta, overlay")
        
        self.wave_3d_mode = mode

        self.t = int(self.rollout.target.shape[0])
        self.vmin = float(min(self.rollout.target.min(), self.rollout.prediction.min()))
        self.vmax = float(max(self.rollout.target.max(), self.rollout.prediction.max()))
        if np.isclose(self.vmin, self.vmax):
            self.vmax = self.vmin + 1e-6
        self.err_max = float(np.abs(self.rollout.prediction - self.rollout.target).max())
        if self.err_max <= 0:
            self.err_max = 1e-6
        self.unc_max = None
        if self.rollout.uncertainty_std is not None:
            self.unc_max = float(self.rollout.uncertainty_std.max())
            if self.unc_max <= 0:
                self.unc_max = 1e-6

        eta_peak = float(
            max(
                np.max(np.abs(self.rollout.target)),
                np.max(np.abs(self.rollout.prediction)),
            )
        )
        bathy_range = float(np.max(self.rollout.bathymetry) - np.min(self.rollout.bathymetry))
        if wave_scale is None:
            if eta_peak > 0 and bathy_range > 0:
                # auto-scale eta for 3D readability while preserving sign
                self.wave_scale = 0.2 * bathy_range / eta_peak
            else:
                self.wave_scale = 1.0
        else:
            self.wave_scale = float(wave_scale)

        self.fig = plt.figure(figsize=(21, 10))
        gs = self.fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.05], wspace=0.24, hspace=0.24)
        self.ax_bathy_2d = self.fig.add_subplot(gs[0, 0])
        self.ax_bathy_3d = self.fig.add_subplot(gs[1, 0], projection="3d")
        self.ax_true_2d = self.fig.add_subplot(gs[0, 1])
        self.ax_true_3d = self.fig.add_subplot(gs[1, 1], projection="3d")
        self.ax_pred_2d = self.fig.add_subplot(gs[0, 2])
        self.ax_pred_3d = self.fig.add_subplot(gs[1, 2], projection="3d")
        self.ax_unc_2d = self.fig.add_subplot(gs[0, 3])
        self.ax_text = self.fig.add_subplot(gs[1, 3])

        self.ax_text.axis("off")
        self.metrics_text = self.ax_text.text(
            0.02, 0.98, "", va="top", ha="left", fontsize=10, family="monospace", transform=self.ax_text.transAxes
        )

        self.im_true = None
        self.im_pred = None
        self.im_unc = None
        self._mesh_x, self._mesh_y = self._meshgrid(self.rollout.bathymetry.shape)
        self._init_static_panels()

    @staticmethod
    def _meshgrid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = shape
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        return np.meshgrid(x, y)

    def _plot_wave_surface(self, ax, wave: np.ndarray, title: str) -> None:
        ax.cla()
        if self.wave_3d_mode == "overlay":
            base = self.rollout.bathymetry
            surface = base + self.wave_scale * wave
            ax.plot_surface(
                self._mesh_x,
                self._mesh_y,
                base,
                cmap="terrain",
                linewidth=0,
                antialiased=False,
                alpha=0.90,
            )
            ax.plot_surface(
                self._mesh_x,
                self._mesh_y,
                surface,
                cmap="RdBu_r",
                linewidth=0,
                antialiased=False,
                alpha=0.70,
            )
            zlabel = "elevation"
        else:
            eta_surface = self.wave_scale * wave
            ax.plot_surface(
                self._mesh_x,
                self._mesh_y,
                eta_surface,
                cmap="RdBu_r",
                linewidth=0,
                antialiased=False,
                alpha=0.95,
            )
            zlabel = "eta"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(zlabel)
        ax.view_init(elev=self.elev, azim=self.azim)

    def _init_static_panels(self) -> None:
        im_bathy = self.ax_bathy_2d.imshow(self.rollout.bathymetry, cmap="terrain", origin="lower")
        self.ax_bathy_2d.set_title("Bathymetry (2D)")
        self.ax_bathy_2d.set_xticks([])
        self.ax_bathy_2d.set_yticks([])
        self.fig.colorbar(im_bathy, ax=self.ax_bathy_2d, fraction=0.046, pad=0.04)

        self.ax_bathy_3d.plot_surface(
            self._mesh_x, self._mesh_y, self.rollout.bathymetry, cmap="terrain", linewidth=0, antialiased=False
        )
        self.ax_bathy_3d.set_title("Bathymetry (3D)")
        self.ax_bathy_3d.set_xlabel("x")
        self.ax_bathy_3d.set_ylabel("y")
        self.ax_bathy_3d.set_zlabel("depth")
        self.ax_bathy_3d.view_init(elev=self.elev, azim=self.azim)

    def _update_text_panel(self, frame_idx: int) -> None:
        m = self.rollout.metrics
        units = "physical" if self.rollout.target_denorm is not None else "normalized"
        lines = [
            f"sample_id: {self.rollout.sample_id}",
            f"frame: {frame_idx + 1}/{self.t}",
            f"target units: {units}",
            f"3D mode: {self.wave_3d_mode} (wave_scale={self.wave_scale:.4g})",
        ]
        if self.rollout.timestamps is not None and frame_idx < len(self.rollout.timestamps):
            lines.append(f"time: {self.rollout.timestamps[frame_idx]:.5f}")
        lines += [
            "",
            "Global Metrics",
            f"MAE     : {m['global_mae']:.6f}",
            f"RMSE    : {m['global_rmse']:.6f}",
            f"Rel L2  : {m['global_rel_l2']:.6f}",
            f"Max Err : {m['global_max_error']:.6f}",
            "",
            "Frame Metrics",
            f"MAE     : {float(m['frame_mae'][frame_idx]):.6f}",
            f"RMSE    : {float(m['frame_rmse'][frame_idx]):.6f}",
            f"Rel L2  : {float(m['frame_rel_l2'][frame_idx]):.6f}",
        ]
        if self.rollout.uncertainty_std is not None:
            u = self.rollout.uncertainty_metrics
            lines += [
                "",
                "Uncertainty",
                f"std mean: {u.get('std_mean', float('nan')):.6f}",
                f"std max : {u.get('std_max', float('nan')):.6f}",
                f"err-unc corr: {u.get('error_uncertainty_corr', float('nan')):.6f}",
                f"cov80   : {u.get('coverage_80', float('nan')):.6f}",
                f"cov90   : {u.get('coverage_90', float('nan')):.6f}",
                f"cov95   : {u.get('coverage_95', float('nan')):.6f}",
            ]
        else:
            lines += ["", "Uncertainty", "not available (deterministic output)"]
        if self.rollout.notes:
            lines += ["", "Notes"]
            lines += [f"- {msg}" for msg in self.rollout.notes]
        self.metrics_text.set_text("\n".join(lines))

    def update(self, frame_idx: int):
        true_frame = self.rollout.target[frame_idx]
        pred_frame = self.rollout.prediction[frame_idx]

        if self.im_true is None:
            self.im_true = self.ax_true_2d.imshow(true_frame, origin="lower", cmap="RdBu_r", vmin=self.vmin, vmax=self.vmax)
            self.ax_true_2d.set_title("True Wave (2D heatmap)")
            self.ax_true_2d.set_xticks([])
            self.ax_true_2d.set_yticks([])
            self.fig.colorbar(self.im_true, ax=self.ax_true_2d, fraction=0.046, pad=0.04)
        else:
            self.im_true.set_data(true_frame)

        if self.im_pred is None:
            self.im_pred = self.ax_pred_2d.imshow(pred_frame, origin="lower", cmap="RdBu_r", vmin=self.vmin, vmax=self.vmax)
            self.ax_pred_2d.set_title("Model Prediction (2D heatmap)")
            self.ax_pred_2d.set_xticks([])
            self.ax_pred_2d.set_yticks([])
            self.fig.colorbar(self.im_pred, ax=self.ax_pred_2d, fraction=0.046, pad=0.04)
        else:
            self.im_pred.set_data(pred_frame)

        if self.rollout.uncertainty_std is not None:
            unc_frame = self.rollout.uncertainty_std[frame_idx]
            title = "Predictive Std (uncertainty)"
            vmin, vmax = 0.0, self.unc_max
            cmap = "magma"
        else:
            unc_frame = np.abs(pred_frame - true_frame)
            title = "Absolute Error (uncertainty unavailable)"
            vmin, vmax = 0.0, self.err_max
            cmap = "magma"

        if self.im_unc is None:
            self.im_unc = self.ax_unc_2d.imshow(unc_frame, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            self.ax_unc_2d.set_title(title)
            self.ax_unc_2d.set_xticks([])
            self.ax_unc_2d.set_yticks([])
            self.fig.colorbar(self.im_unc, ax=self.ax_unc_2d, fraction=0.046, pad=0.04)
        else:
            self.im_unc.set_data(unc_frame)
            self.ax_unc_2d.set_title(title)

        if self.wave_3d_mode == "overlay":
            true_title = "True Wave on Bathymetry (3D)"
            pred_title = "Predicted Wave on Bathymetry (3D)"
        else:
            true_title = "True Wave Eta Surface (3D)"
            pred_title = "Predicted Wave Eta Surface (3D)"
        self._plot_wave_surface(self.ax_true_3d, true_frame, true_title)
        self._plot_wave_surface(self.ax_pred_3d, pred_frame, pred_title)
        self._update_text_panel(frame_idx)

        time_label = ""
        if self.rollout.timestamps is not None and frame_idx < len(self.rollout.timestamps):
            time_label = f" | t={self.rollout.timestamps[frame_idx]:.5f}"
        self.fig.suptitle(f"Tsunami Sample Explorer: {self.rollout.sample_id} | frame {frame_idx + 1}/{self.t}{time_label}", fontsize=13)
        return []

    def animate(self, max_frames: Optional[int] = None) -> FuncAnimation:
        n_frames = self.t if max_frames is None else min(self.t, int(max_frames))
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=range(n_frames),
            interval=self.interval_ms,
            blit=False,
            repeat=self.repeat,
            cache_frame_data=False,
        )
        return ani


def run_visualization(
    config_path: str | Path,
    checkpoint_path: str | Path,
    processed_path: str | Path,
    raw_dir: str | Path = "data/raw/samples",
    sample_id: Optional[str] = None,
    sample_index: int = 0,
    mc_samples: int = 0,
    device: str = "auto",
    interval_ms: int = 120,
    repeat: bool = False,
    elev: float = 35.0,
    azim: float = -60.0,
    wave_scale: Optional[float] = None,
    wave_3d_mode: str = "eta",
    max_frames: Optional[int] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    rollout = prepare_visual_rollout(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        processed_path=processed_path,
        raw_dir=raw_dir,
        sample_id=sample_id,
        sample_index=sample_index,
        mc_samples=mc_samples,
        device=device,
    )
    viz = RolloutFigure(
        rollout,
        interval_ms=interval_ms,
        repeat=repeat,
        elev=elev,
        azim=azim,
        wave_scale=wave_scale,
        wave_3d_mode=wave_3d_mode,
    )
    ani = viz.animate(max_frames=max_frames)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".gif":
            ani.save(save_path, writer="pillow")
        else:
            ani.save(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
