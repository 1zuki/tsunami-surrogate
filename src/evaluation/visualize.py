from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.widgets import Button

from src.evaluation.calibration import interval_calibration
from src.evaluation.target_scaling import load_target_denorm, resolve_eval_dataset_path
from src.evaluation.uncertainty import error_uncertainty_correlation
from src.evaluation.window_rollout import rollout_trajectory
from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import resolve_device


_SAMPLE_METADATA_KEYS = (
    "scenario_id",
    "solver_name",
    "source_id",
    "source_type",
    "bathymetry_type",
    "source_strength",
)


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


def _read_input_order_manifest(manifest_path: Path) -> Optional[np.ndarray]:
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        input_order = manifest.get("input_order")
        if isinstance(input_order, list) and all(isinstance(x, str) for x in input_order):
            return np.asarray(input_order, dtype=object)
    except Exception:
        return None

    return None


def _copy_selected_metadata(
    data: Any,
    out: Dict[str, np.ndarray],
    index: int,
) -> None:
    for key in _SAMPLE_METADATA_KEYS:
        if key not in data:
            continue
        values = np.asarray(data[key])
        if values.ndim == 0:
            out[key] = values.reshape(1)
        else:
            out[key] = values[index : index + 1]


def _load_processed_eval_dataset(
    processed_path: str | Path,
    sample_id: Optional[str] = None,
    sample_index: int = 0,
) -> Dict[str, np.ndarray]:
    processed_path = Path(processed_path)
    if (
        processed_path.name == "eval_dataset.npz"
        and (processed_path.parent / "shards_manifest.json").is_file()
    ):
        processed_path = processed_path.parent
    npz_path = processed_path
    if processed_path.is_dir():
        shard_manifest = processed_path / "shards_manifest.json"
        if shard_manifest.exists():
            with shard_manifest.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            shards = list(manifest.get("shards", []))
            if not shards:
                raise FileNotFoundError(f"No shards found in directory: {processed_path}")

            if sample_id is None:
                global_idx = int(sample_index)
                if global_idx < 0:
                    global_idx += int(manifest.get("num_samples", 0))
                if global_idx < 0:
                    raise IndexError(f"sample_index out of range: {int(sample_index)}")

                offset = 0
                selected = None
                for shard in shards:
                    count = int(shard.get("num_samples", 0))
                    if global_idx < offset + count:
                        selected = (shard, global_idx - offset)
                        break
                    offset += count
                if selected is None:
                    raise IndexError(f"sample_index out of range: {int(sample_index)} not in [0, {offset - 1}]")
            else:
                selected = None
                for shard in shards:
                    shard_file = shard.get("file")
                    if not shard_file:
                        raise KeyError(f"Shard entry in {shard_manifest} is missing a file path.")
                    with np.load(processed_path / str(shard_file), allow_pickle=True) as data:
                        ids = _normalize_sample_ids(data["sample_id"])
                        matches = np.where(ids == sample_id)[0]
                        if len(matches) > 0:
                            selected = (shard, int(matches[0]))
                            break
                if selected is None:
                    raise KeyError(f"sample_id '{sample_id}' not found in processed shard dataset.")

            shard, local_idx = selected
            shard_file = shard.get("file")
            if not shard_file:
                raise KeyError(f"Selected shard in {shard_manifest} is missing a file path.")
            shard_path = processed_path / str(shard_file)

            with np.load(shard_path, allow_pickle=True) as data:
                if "inputs" not in data or "targets" not in data:
                    raise KeyError(f"{shard_path} must contain 'inputs' and 'targets'")
                ids = _normalize_sample_ids(data["sample_id"]) if "sample_id" in data else None
                out = {
                    "inputs": np.asarray(data["inputs"][local_idx : local_idx + 1], dtype=np.float32),
                    "targets": np.asarray(data["targets"][local_idx : local_idx + 1], dtype=np.float32),
                }
                if ids is not None:
                    out["sample_id"] = np.asarray([ids[local_idx]], dtype=object)
                else:
                    out["sample_id"] = np.asarray([f"sample_{local_idx:06d}"], dtype=object)
                if "input_order" in data:
                    out["input_order"] = np.asarray(data["input_order"], dtype=object)
                _copy_selected_metadata(data, out, local_idx)

            if "input_order" not in out:
                input_order = _read_input_order_manifest(processed_path / "eval_manifest.json")
                if input_order is not None:
                    out["input_order"] = input_order

            return out

        candidate = processed_path / "eval_dataset.npz"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing eval dataset archive: {candidate}")
        npz_path = candidate

    with np.load(npz_path, allow_pickle=True) as data:
        if "inputs" not in data or "targets" not in data:
            raise KeyError(f"{npz_path} must contain 'inputs' and 'targets'")
        if "sample_id" in data:
            sample_ids = _normalize_sample_ids(data["sample_id"])
        else:
            sample_ids = np.asarray([f"sample_{i:06d}" for i in range(data["inputs"].shape[0])], dtype=object)
        idx = _pick_sample_index(sample_ids, sample_id=sample_id, sample_index=sample_index)
        out = {
            "inputs": np.asarray(data["inputs"][idx : idx + 1], dtype=np.float32),
            "targets": np.asarray(data["targets"][idx : idx + 1], dtype=np.float32),
            "sample_id": np.asarray([sample_ids[idx]], dtype=object),
        }
        if "input_order" in data:
            out["input_order"] = np.asarray(data["input_order"], dtype=object)
        _copy_selected_metadata(data, out, idx)
    manifest_path = npz_path.with_name("eval_manifest.json")
    if "input_order" not in out:
        input_order = _read_input_order_manifest(manifest_path)
        if input_order is not None:
            out["input_order"] = input_order
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


def _zero_based_sample_index(sample_index: int) -> int:
    sample_number = int(sample_index)
    if sample_number < 1:
        raise ValueError(
            "sample_index must be >= 1 for user-facing visualization "
            "(1 selects sample_000001)"
        )
    return sample_number - 1


def _load_raw_sample_bathymetry_and_timestamps(
    raw_dir: Optional[Path],
    sample_id: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw_dir is None:
        return None, None

    sample_npz = raw_dir / sample_id / "sample.npz"
    if not sample_npz.exists():
        return None, None

    with np.load(sample_npz) as data:
        bathymetry = np.asarray(data["bathymetry"], dtype=np.float32) if "bathymetry" in data else None
        timestamps = np.asarray(data["timestamps"], dtype=np.float32) if "timestamps" in data else None
    return bathymetry, timestamps


def _metadata_text(processed: Dict[str, np.ndarray], key: str) -> str:
    values = processed.get(key)
    if values is None:
        return ""
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return ""
    value = flat[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _solver_directory_name(solver_name: str, processed_path: str | Path) -> str:
    normalized = str(solver_name).strip().lower()
    mapping = {
        "swe_hydrostatic": "hydrostatic",
        "hydrostatic": "hydrostatic",
        "swe_muscl_hr": "muscl_hr",
        "muscl_hr": "muscl_hr",
        "boussinesq": "boussinesq",
    }
    if normalized in mapping:
        return mapping[normalized]

    parts = {part.lower() for part in Path(processed_path).parts}
    for candidate in ("hydrostatic", "muscl_hr", "boussinesq"):
        if candidate in parts:
            return candidate
    return ""


def _infer_raw_dir(
    processed_path: str | Path,
    solver_name: str,
) -> Optional[Path]:
    path = Path(processed_path).expanduser().resolve()
    split_dir = path.parent if path.suffix.lower() == ".npz" else path
    split = split_dir.name
    solver_dir = _solver_directory_name(solver_name, processed_path)
    if split not in {"train", "val", "eval", "test"} or not solver_dir:
        return None
    raw_split = "eval" if split == "val" else split

    data_root = next(
        (candidate for candidate in (split_dir, *split_dir.parents) if candidate.name == "data"),
        None,
    )
    if data_root is None:
        return None

    candidate = data_root / raw_split / "raw" / solver_dir / "samples"
    return candidate if candidate.is_dir() else None


def _resolve_raw_dir(
    raw_dir: str | Path | None,
    processed_path: str | Path,
    solver_name: str,
) -> Optional[Path]:
    if raw_dir is None or str(raw_dir).strip().lower() == "auto":
        return _infer_raw_dir(processed_path, solver_name)
    return Path(raw_dir).expanduser()


def _normalization_stats_path(processed_path: str | Path) -> Optional[Path]:
    path = Path(processed_path).expanduser()
    split_dir = path.parent if path.suffix.lower() == ".npz" else path
    candidates = (
        split_dir / "normalization_stats.json",
        split_dir.parent / "normalization_stats.json",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _denormalize_input_channel(
    values: np.ndarray,
    processed_path: str | Path,
    channel_name: str,
) -> tuple[np.ndarray, bool]:
    stats_path = _normalization_stats_path(processed_path)
    if stats_path is None:
        return np.asarray(values, dtype=np.float32), False
    try:
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        channel = stats.get("inputs", {}).get(channel_name)
        if not isinstance(channel, dict):
            return np.asarray(values, dtype=np.float32), False
        offset = float(channel["offset"])
        scale = float(channel["scale"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return np.asarray(values, dtype=np.float32), False
    if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0.0:
        return np.asarray(values, dtype=np.float32), False
    denormalized = np.asarray(values, dtype=np.float32) * scale + offset
    return np.asarray(denormalized, dtype=np.float32), True


def _compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    global_start: int = 0,
) -> Dict[str, Any]:
    err = pred - target
    abs_err = np.abs(err)
    sq_err = err ** 2

    frame_mae = abs_err.mean(axis=(1, 2))
    frame_rmse = np.sqrt(sq_err.mean(axis=(1, 2)))
    frame_rel_l2 = np.linalg.norm(err.reshape(err.shape[0], -1), axis=1) / (
        np.linalg.norm(target.reshape(target.shape[0], -1), axis=1) + 1e-8
    )

    start = int(global_start)
    if start < 0 or start >= pred.shape[0]:
        raise ValueError(
            f"global_start must be in [0, {pred.shape[0] - 1}], got {start}"
        )
    global_err = err[start:]
    global_target = target[start:]
    global_abs_err = np.abs(global_err)
    global_sq_err = global_err ** 2

    return {
        "global_mae": float(global_abs_err.mean()),
        "global_rmse": float(np.sqrt(global_sq_err.mean())),
        "global_rel_l2": float(
            np.linalg.norm(global_err.ravel())
            / (np.linalg.norm(global_target.ravel()) + 1e-8)
        ),
        "global_max_error": float(global_abs_err.max()),
        "frame_mae": frame_mae,
        "frame_rmse": frame_rmse,
        "frame_rel_l2": frame_rel_l2,
    }


@dataclass
class VisualRollout:
    sample_id: str
    reference_name: str
    prediction_mode: str
    seeded_frames: int
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
    processed_path: str | Path | None = "auto",
    raw_dir: str | Path | None = "auto",
    sample_id: Optional[str] = None,
    sample_index: int = 0,
    mc_samples: int = 0,
    device: str = "auto",
) -> VisualRollout:
    cfg = load_config(config_path)
    if processed_path is None or str(processed_path).strip().lower() == "auto":
        resolved_processed_path = resolve_eval_dataset_path(cfg, split="test")
        if resolved_processed_path is None:
            raise ValueError(
                "Could not infer the test dataset from the model config; pass "
                "--processed-path explicitly."
            )
        processed_path = resolved_processed_path
    model = build_model(cfg)
    dev = resolve_device(device if device != "auto" else cfg.get("device", "auto"))
    load_checkpoint(checkpoint_path, model, map_location=dev)
    model = model.to(dev).eval()

    processed = _load_processed_eval_dataset(
        processed_path,
        sample_id=sample_id,
        sample_index=sample_index,
    )
    sample_ids = _normalize_sample_ids(processed["sample_id"])
    idx = 0
    sid = str(sample_ids[idx])

    x_np = np.asarray(processed["inputs"][idx], dtype=np.float32)
    y_np = _as_tchw(processed["targets"][idx])
    x = torch.from_numpy(x_np).unsqueeze(0).to(dev)
    y = torch.from_numpy(y_np).unsqueeze(0).to(dev)

    data_cfg = dict(cfg.get("data", {}))
    windowed = bool(data_cfg.get("windowed", False))
    seeded_frames = 0
    if windowed:
        if mc_samples > 1:
            raise ValueError(
                "MC sampling is not supported for autoregressive windowed visualization"
            )
        K = int(data_cfg.get("window_K", 5))
        include_source = bool(data_cfg.get("window_include_source", True))
        use_prev = bool(data_cfg.get("window_prev", True))
        with torch.no_grad():
            pred_tail = rollout_trajectory(
                model,
                x,
                y[:, 0],
                int(y.shape[1]),
                K,
                include_source,
                use_prev,
                dev,
            )
        pred_t = torch.cat([y[:, :1], pred_tail], dim=1)
        var_t = None
        prediction_mode = f"seeded-window K={K}"
        seeded_frames = 1
    else:
        with torch.no_grad():
            pred_t, var_t = _model_mean_and_variance(
                model,
                x,
                mc_samples=mc_samples,
            )
        prediction_mode = "direct"

    pred_np = _as_tchw(pred_t.squeeze(0).detach().cpu().numpy())
    var_np = _as_tchw(var_t.squeeze(0).detach().cpu().numpy()) if var_t is not None else None

    if pred_np.shape != y_np.shape:
        raise ValueError(
            "Prediction shape does not match the selected target trajectory: "
            f"prediction={pred_np.shape}, target={y_np.shape}"
        )
    t = int(y_np.shape[0])
    if var_np is not None:
        if var_np.shape != y_np.shape:
            raise ValueError(
                "Predictive variance shape does not match the selected target "
                f"trajectory: variance={var_np.shape}, target={y_np.shape}"
            )

    target_denorm = load_target_denorm(processed_path)
    if target_denorm is not None:
        offset, scale = target_denorm
        pred_np = pred_np * float(scale) + float(offset)
        y_np = y_np * float(scale) + float(offset)
        if var_np is not None:
            var_np = var_np * float(scale) ** 2

    solver_name = _metadata_text(processed, "solver_name")
    solver_dir = _solver_directory_name(solver_name, processed_path)
    reference_name = solver_name or solver_dir or "unknown"
    resolved_raw_dir = _resolve_raw_dir(raw_dir, processed_path, solver_name)
    bathy_raw, ts_raw = _load_raw_sample_bathymetry_and_timestamps(
        resolved_raw_dir,
        sid,
    )
    notes: list[str] = []
    input_order_values = processed.get("input_order")
    bathy_idx = 0
    if input_order_values is not None:
        order = [
            str(value)
            for value in np.asarray(input_order_values).reshape(-1).tolist()
        ]
        if "bathymetry" in order:
            bathy_idx = int(order.index("bathymetry"))
        else:
            notes.append(
                "Processed input_order has no 'bathymetry' entry; channel 0 is "
                "used for bathymetry visualization."
            )
    processed_bathymetry, input_denormalized = _denormalize_input_channel(
        x_np[bathy_idx],
        processed_path,
        "bathymetry",
    )

    if bathy_raw is not None and bathy_raw.shape == y_np.shape[1:]:
        bathymetry = bathy_raw
        used_raw_bathymetry = True
    else:
        bathymetry = processed_bathymetry
        used_raw_bathymetry = False
        if bathy_raw is not None:
            notes.append(
                "The raw bathymetry shape did not match the processed target grid; "
                "the processed bathymetry channel was used instead."
            )
        if input_denormalized:
            note = (
                f"Raw bathymetry was not found for {sid}; the processed bathymetry "
                "channel was de-normalized with the dataset statistics."
            )
        else:
            note = (
                f"Raw bathymetry was not found for {sid}; processed input channel "
                f"{bathy_idx} is shown in normalized units."
            )
        note = (
            note
            if resolved_raw_dir is not None
            else f"{note} Automatic raw-path resolution was unavailable."
        )
        warnings.warn(note, RuntimeWarning)
        notes.append(note)

    if windowed:
        notes.append(
            "Frame 1 is given; global metrics score only frames 2..T."
        )

    timestamps = None
    if ts_raw is not None:
        if ts_raw.shape[0] == t:
            timestamps = ts_raw
        elif ts_raw.shape[0] >= t + 1:
            timestamps = ts_raw[1 : t + 1]
        else:
            timestamps = ts_raw[:t]

    metrics = _compute_metrics(
        pred_np,
        y_np,
        global_start=seeded_frames,
    )
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
        reference_name=reference_name,
        prediction_mode=prediction_mode,
        seeded_frames=seeded_frames,
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
        eta_limit: Optional[float] = None,
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
        observed_eta_limit = float(
            max(
                np.max(np.abs(self.rollout.target)),
                np.max(np.abs(self.rollout.prediction)),
            )
        )
        if eta_limit is None:
            self.eta_limit = observed_eta_limit
        else:
            self.eta_limit = float(eta_limit)
        if not np.isfinite(self.eta_limit) or self.eta_limit <= 0.0:
            raise ValueError("eta_limit must be a positive finite value")
        self.vmin = -self.eta_limit
        self.vmax = self.eta_limit
        self.wave_norm = Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)
        self.err_max = float(np.abs(self.rollout.prediction - self.rollout.target).max())
        if self.err_max <= 0:
            self.err_max = 1e-6
        self.unc_max = None
        if self.rollout.uncertainty_std is not None:
            self.unc_max = float(self.rollout.uncertainty_std.max())
            if self.unc_max <= 0:
                self.unc_max = 1e-6

        eta_peak = observed_eta_limit
        bathy_range = float(np.max(self.rollout.bathymetry) - np.min(self.rollout.bathymetry))
        if wave_scale is None:
            if eta_peak > 0 and bathy_range > 0:
                # auto-scale eta for 3D readability while preserving sign
                self.wave_scale = 0.2 * bathy_range / eta_peak
            else:
                self.wave_scale = 1.0
        else:
            self.wave_scale = float(wave_scale)
        if not np.isfinite(self.wave_scale):
            raise ValueError("wave_scale must be finite")

        self.bathy_min = float(np.min(self.rollout.bathymetry))
        self.bathy_max = float(np.max(self.rollout.bathymetry))
        if np.isclose(self.bathy_min, self.bathy_max):
            self.bathy_max = self.bathy_min + 1e-6
        scaled_eta_limit = max(abs(self.wave_scale) * self.eta_limit, 1e-9)
        self.eta_zlim = (-scaled_eta_limit, scaled_eta_limit)
        self.overlay_zlim = (
            self.bathy_min - scaled_eta_limit,
            self.bathy_max + scaled_eta_limit,
        )

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
        wave = np.asarray(wave, dtype=np.float32)
        if self.wave_3d_mode == "overlay":
            base = self.rollout.bathymetry
            surface = base + self.wave_scale * wave
            ax.plot_surface(
                self._mesh_x,
                self._mesh_y,
                base,
                cmap="terrain",
                vmin=self.bathy_min,
                vmax=self.bathy_max,
                linewidth=0,
                antialiased=False,
                alpha=0.90,
            )
            wave_colors = plt.get_cmap("RdBu_r")(self.wave_norm(wave))
            ax.plot_surface(
                self._mesh_x,
                self._mesh_y,
                surface,
                facecolors=wave_colors,
                linewidth=0,
                antialiased=False,
                alpha=0.70,
                shade=False,
            )
            zlabel = "elevation"
            ax.set_zlim(*self.overlay_zlim)
        else:
            eta_surface = self.wave_scale * wave
            ax.plot_surface(
                self._mesh_x,
                self._mesh_y,
                eta_surface,
                cmap="RdBu_r",
                vmin=self.eta_zlim[0],
                vmax=self.eta_zlim[1],
                linewidth=0,
                antialiased=False,
                alpha=0.95,
            )
            zlabel = "eta"
            ax.set_zlim(*self.eta_zlim)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(zlabel)
        ax.set_xlim(float(self._mesh_x.min()), float(self._mesh_x.max()))
        ax.set_ylim(float(self._mesh_y.min()), float(self._mesh_y.max()))
        ax.view_init(elev=self.elev, azim=self.azim)

    def _init_static_panels(self) -> None:
        im_bathy = self.ax_bathy_2d.imshow(self.rollout.bathymetry, cmap="terrain", origin="lower")
        self.ax_bathy_2d.set_title("Bathymetry (2D)")
        self.ax_bathy_2d.set_xticks([])
        self.ax_bathy_2d.set_yticks([])
        self.fig.colorbar(im_bathy, ax=self.ax_bathy_2d, fraction=0.046, pad=0.04)

        self.ax_bathy_3d.plot_surface(
            self._mesh_x,
            self._mesh_y,
            self.rollout.bathymetry,
            cmap="terrain",
            vmin=self.bathy_min,
            vmax=self.bathy_max,
            linewidth=0,
            antialiased=False,
        )
        self.ax_bathy_3d.set_title("Bathymetry (3D)")
        self.ax_bathy_3d.set_xlabel("x")
        self.ax_bathy_3d.set_ylabel("y")
        self.ax_bathy_3d.set_zlabel("bathymetry")
        self.ax_bathy_3d.set_zlim(self.bathy_min, self.bathy_max)
        self.ax_bathy_3d.view_init(elev=self.elev, azim=self.azim)

    def _update_text_panel(self, frame_idx: int) -> None:
        m = self.rollout.metrics
        units = "physical" if self.rollout.target_denorm is not None else "normalized"
        lines = [
            f"sample_id: {self.rollout.sample_id}",
            f"reference: {self.rollout.reference_name}",
            f"prediction: {self.rollout.prediction_mode}",
            f"frame: {frame_idx + 1}/{self.t}",
            f"target units: {units}",
            "displayed field: absolute eta",
            f"3D mode: {self.wave_3d_mode} (wave_scale={self.wave_scale:.4g})",
            f"fixed eta range: [{self.vmin:.5g}, {self.vmax:.5g}]",
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
            self.ax_true_2d.set_title("True Surface Elevation Eta (2D)")
            self.ax_true_2d.set_xticks([])
            self.ax_true_2d.set_yticks([])
            self.fig.colorbar(self.im_true, ax=self.ax_true_2d, fraction=0.046, pad=0.04)
        else:
            self.im_true.set_data(true_frame)

        if self.im_pred is None:
            self.im_pred = self.ax_pred_2d.imshow(pred_frame, origin="lower", cmap="RdBu_r", vmin=self.vmin, vmax=self.vmax)
            self.ax_pred_2d.set_title("Predicted Surface Elevation Eta (2D)")
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
            true_title = "True Absolute Eta on Bathymetry (3D)"
            pred_title = "Predicted Absolute Eta on Bathymetry (3D)"
        else:
            true_title = "True Absolute Eta Surface (3D)"
            pred_title = "Predicted Absolute Eta Surface (3D)"
        self._plot_wave_surface(self.ax_true_3d, true_frame, true_title)
        self._plot_wave_surface(self.ax_pred_3d, pred_frame, pred_title)
        self._update_text_panel(frame_idx)

        time_label = ""
        if self.rollout.timestamps is not None and frame_idx < len(self.rollout.timestamps):
            time_label = f" | t={self.rollout.timestamps[frame_idx]:.5f}"
        seed_label = " | given seed" if frame_idx < self.rollout.seeded_frames else ""
        self.fig.suptitle(
            f"Tsunami Sample Explorer: {self.rollout.sample_id} | "
            f"{self.rollout.reference_name} | frame {frame_idx + 1}/{self.t}"
            f"{time_label}{seed_label}",
            fontsize=13,
        )
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

    def cache_frames(
        self,
        max_frames: Optional[int] = None,
        dpi: int = 80,
    ) -> tuple[bytes, ...]:
        n_frames = self.t if max_frames is None else min(self.t, int(max_frames))
        if n_frames < 1:
            raise ValueError("At least one frame must be cached")
        dpi = int(dpi)
        if dpi < 40:
            raise ValueError("cache dpi must be at least 40")

        cached: list[bytes] = []
        progress_every = max(1, n_frames // 10)
        for frame_idx in range(n_frames):
            self.update(frame_idx)
            self.fig.canvas.draw()
            buffer = BytesIO()
            self.fig.savefig(
                buffer,
                format="png",
                dpi=dpi,
                facecolor=self.fig.get_facecolor(),
            )
            cached.append(buffer.getvalue())
            if (
                frame_idx == 0
                or frame_idx + 1 == n_frames
                or (frame_idx + 1) % progress_every == 0
            ):
                print(
                    f"[visualize] cached frame {frame_idx + 1}/{n_frames}",
                    flush=True,
                )
        return tuple(cached)


class CachedRolloutPlayer:
    def __init__(
        self,
        frames: tuple[bytes, ...],
        interval_ms: int,
        repeat: bool,
        controls: bool,
    ) -> None:
        if not frames:
            raise ValueError("CachedRolloutPlayer requires at least one frame")

        self.frames = frames
        self.interval_ms = int(interval_ms)
        self.repeat = bool(repeat)
        self.controls = bool(controls)
        self.index = 0
        self.playing = False

        first = self._decode_frame(0)
        height, width = first.shape[:2]
        if self.controls:
            display_width = min(18.0, width / 100.0)
            display_height = display_width * height / width
            if display_height > 9.5:
                display_height = 9.5
                display_width = display_height * width / height
            figure_height = display_height + 0.8
        else:
            display_width = width / 100.0
            display_height = height / 100.0
            figure_height = display_height

        self.fig = plt.figure(
            figsize=(display_width, figure_height),
            dpi=100,
        )
        if self.controls:
            image_bottom = 0.09
            self.ax = self.fig.add_axes([0.0, image_bottom, 1.0, 1.0 - image_bottom])
        else:
            self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.axis("off")
        self.image = self.ax.imshow(first)

        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._timer_step)
        self.button_back: Optional[Button] = None
        self.button_play: Optional[Button] = None
        self.button_step: Optional[Button] = None
        self.status_text = None

        if self.controls:
            self.button_back = Button(
                self.fig.add_axes([0.32, 0.015, 0.10, 0.045]),
                "Back",
            )
            self.button_play = Button(
                self.fig.add_axes([0.45, 0.015, 0.10, 0.045]),
                "Pause",
            )
            self.button_step = Button(
                self.fig.add_axes([0.58, 0.015, 0.10, 0.045]),
                "Step",
            )
            self.button_back.on_clicked(self._back)
            self.button_play.on_clicked(self._toggle_play)
            self.button_step.on_clicked(self._step)
            self.status_text = self.fig.text(
                0.02,
                0.037,
                "",
                ha="left",
                va="center",
                family="monospace",
                fontsize=9,
            )
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            self.fig.canvas.mpl_connect("close_event", self._on_close)
            self._update_status()

    @lru_cache(maxsize=8)
    def _decode_frame(self, index: int) -> np.ndarray:
        return np.asarray(
            plt.imread(BytesIO(self.frames[int(index)]), format="png")
        )

    def _update_status(self) -> None:
        if self.status_text is not None:
            state = "playing" if self.playing else "paused"
            self.status_text.set_text(
                f"frame {self.index + 1}/{len(self.frames)} | {state} | "
                "keys: left, right, space"
            )
        if self.button_play is not None:
            self.button_play.label.set_text("Pause" if self.playing else "Play")

    def _show_index(self, index: int) -> None:
        self.index = max(0, min(int(index), len(self.frames) - 1))
        self.image.set_data(self._decode_frame(self.index))
        self._update_status()
        self.fig.canvas.draw_idle()

    def _set_playing(self, playing: bool) -> None:
        self.playing = bool(playing)
        if self.playing:
            if self.index >= len(self.frames) - 1 and not self.repeat:
                self.index = 0
            self.timer.start()
        else:
            self.timer.stop()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _timer_step(self) -> bool:
        next_index = self.index + 1
        if next_index >= len(self.frames):
            if self.repeat:
                next_index = 0
            else:
                self._set_playing(False)
                return True
        self._show_index(next_index)
        return True

    def _back(self, _event=None) -> None:
        self._set_playing(False)
        self._show_index(self.index - 1)

    def _step(self, _event=None) -> None:
        self._set_playing(False)
        self._show_index(self.index + 1)

    def _toggle_play(self, _event=None) -> None:
        self._set_playing(not self.playing)

    def _on_key(self, event) -> None:
        if event.key == "left":
            self._back()
        elif event.key == "right":
            self._step()
        elif event.key in {" ", "space"}:
            self._toggle_play()

    def _on_close(self, _event=None) -> None:
        self.timer.stop()

    def animate(self) -> FuncAnimation:
        def update(index: int):
            self.image.set_data(self._decode_frame(index))
            return [self.image]

        return FuncAnimation(
            self.fig,
            update,
            frames=range(len(self.frames)),
            interval=self.interval_ms,
            blit=True,
            repeat=self.repeat,
            cache_frame_data=True,
        )

    def show(self) -> None:
        self._set_playing(True)
        plt.show()


def run_visualization(
    config_path: str | Path,
    checkpoint_path: str | Path,
    processed_path: str | Path | None = "auto",
    raw_dir: str | Path | None = "auto",
    sample_id: Optional[str] = None,
    sample_index: int = 1,
    mc_samples: int = 0,
    device: str = "auto",
    interval_ms: int = 120,
    repeat: bool = False,
    elev: float = 35.0,
    azim: float = -60.0,
    wave_scale: Optional[float] = None,
    wave_3d_mode: str = "eta",
    eta_limit: Optional[float] = None,
    max_frames: Optional[int] = None,
    cache_dpi: int = 80,
    save_path: Optional[str | Path] = None,
) -> None:
    dataset_sample_index = (
        int(sample_index)
        if sample_id is not None
        else _zero_based_sample_index(sample_index)
    )
    rollout = prepare_visual_rollout(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        processed_path=processed_path,
        raw_dir=raw_dir,
        sample_id=sample_id,
        sample_index=dataset_sample_index,
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
        eta_limit=eta_limit,
    )
    cached_frames = viz.cache_frames(
        max_frames=max_frames,
        dpi=cache_dpi,
    )
    plt.close(viz.fig)

    if save_path:
        player = CachedRolloutPlayer(
            cached_frames,
            interval_ms=interval_ms,
            repeat=repeat,
            controls=False,
        )
        ani = player.animate()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".gif":
            ani.save(save_path, writer="pillow")
        else:
            ani.save(save_path)
        plt.close(player.fig)
        print(f"Saved visualization to {save_path}")
    else:
        player = CachedRolloutPlayer(
            cached_frames,
            interval_ms=interval_ms,
            repeat=repeat,
            controls=True,
        )
        player.show()
