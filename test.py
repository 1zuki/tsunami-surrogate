from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import src.data_gen.simulate_dataset as sd

@dataclass
class SampleView:
    sample_id: str
    bathymetry: np.ndarray          # [H, W]
    wave_frames: np.ndarray         # [T, H, W] free-surface elevation (eta)
    timestamps: Optional[np.ndarray] = None
    meta: Optional[dict] = None


class BaseDatasetAdapter:
    def __len__(self) -> int:
        raise NotImplementedError

    def get_sample(self, index: int) -> SampleView:
        raise NotImplementedError


class RawDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, raw_samples_dir: Path, start_idx: int = 1, end_idx: Optional[int] = None) -> None:
        if not raw_samples_dir.exists():
            raise FileNotFoundError(f"Raw samples dir not found: {raw_samples_dir}")

        dirs = sorted(p for p in raw_samples_dir.glob("sample_*") if p.is_dir())
        if not dirs:
            raise FileNotFoundError(f"No sample_* directories found under: {raw_samples_dir}")

        start = max(1, start_idx)
        stop = end_idx if end_idx is not None else len(dirs)
        self.sample_dirs = dirs[start - 1: stop]

        if not self.sample_dirs:
            raise ValueError("Requested raw sample range is empty.")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def get_sample(self, index: int) -> SampleView:
        sample_dir = self.sample_dirs[index]
        npz_path = sample_dir / "sample.npz"
        meta_path = sample_dir / "meta.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing raw sample file: {npz_path}")

        with np.load(npz_path) as data:
            bathymetry = np.asarray(data["bathymetry"], dtype=np.float32)
            trajectory = np.asarray(data["trajectory"], dtype=np.float32)
            timestamps = np.asarray(data["timestamps"], dtype=np.float32) if "timestamps" in data else None

        if trajectory.ndim == 4:
            # expected raw layout: [T, C, H, W], where C0 is water depth h
            depth_frames = trajectory[:, 0]
        elif trajectory.ndim == 3:
            depth_frames = trajectory
        else:
            raise ValueError(f"Unsupported raw trajectory shape: {trajectory.shape}")

        # free-surface elevation eta = h + b; rest state is ~0 when sea level offset is 0
        wave_frames = depth_frames + bathymetry[None, ...]

        meta = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        return SampleView(
            sample_id=sample_dir.name,
            bathymetry=bathymetry,
            wave_frames=wave_frames,
            timestamps=timestamps,
            meta=meta,
        )


class ProcessedDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, processed_split_dir: Path, start_idx: int = 1, end_idx: Optional[int] = None) -> None:
        if not processed_split_dir.exists():
            raise FileNotFoundError(f"Processed split dir not found: {processed_split_dir}")

        bathy_path = processed_split_dir / "X_bathymetry.npz"
        target_path = processed_split_dir / "Y.npy"
        meta_path = processed_split_dir / "meta.jsonl"

        if not bathy_path.exists():
            raise FileNotFoundError(f"Missing processed bathymetry file: {bathy_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing processed target file: {target_path}")

        self.bathymetry = self._load_npz_array(bathy_path).astype(np.float32)
        self.targets = np.load(target_path)
        self.meta = self._load_meta(meta_path)

        total = min(len(self.bathymetry), len(self.targets))
        start = max(1, start_idx)
        stop = end_idx if end_idx is not None else total
        self.indices = list(range(start - 1, min(stop, total)))

        if not self.indices:
            raise ValueError("Requested processed sample range is empty.")

    @staticmethod
    def _load_npz_array(path: Path) -> np.ndarray:
        with np.load(path) as data:
            if "data" in data.files:
                return np.asarray(data["data"])
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            raise ValueError(f"Could not infer array key from {path}; keys={data.files}")

    @staticmethod
    def _load_meta(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.indices)

    def _coerce_processed_target(self, target: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        h, w = bathymetry.shape

        if target.ndim == 2 and target.shape == (h, w):
            return target[None, ...]

        if target.ndim == 3:
            # [C, H, W]  -> next-step state, usually C=3 and C0 is water depth
            if target.shape[1:] == (h, w) and target.shape[0] <= 4:
                return (target[0] + bathymetry)[None, ...]
            # [T, H, W] -> already frame-first
            if target.shape[1:] == (h, w):
                return target

        if target.ndim == 4:
            # [T, C, H, W] -> trajectory of states, use channel 0 (depth)
            if target.shape[-2:] == (h, w) and target.shape[1] <= 4:
                return target[:, 0] + bathymetry[None, ...]
            # [T, H, W, C] -> channel-last trajectory
            if target.shape[1:3] == (h, w) and target.shape[-1] <= 4:
                return target[..., 0]

        raise ValueError(
            f"Unsupported processed target shape {target.shape}. Expected [H,W], [C,H,W], [T,H,W], or [T,C,H,W]."
        )

    def get_sample(self, index: int) -> SampleView:
        global_idx = self.indices[index]
        bathymetry = np.asarray(self.bathymetry[global_idx], dtype=np.float32)
        target = np.asarray(self.targets[global_idx], dtype=np.float32)
        wave_frames = self._coerce_processed_target(target, bathymetry)
        meta = self.meta[global_idx] if global_idx < len(self.meta) else None

        return SampleView(
            sample_id=f"sample_{global_idx + 1:06d}",
            bathymetry=bathymetry,
            wave_frames=wave_frames,
            timestamps=None,
            meta=meta,
        )


class DatasetVisualizer:
    def __init__(
        self,
        adapter: BaseDatasetAdapter,
        steps_per_sample: Optional[int],
        interval_ms: int,
        repeat: bool,
        elev: float,
        azim: float,
    ) -> None:
        self.adapter = adapter
        self.steps_per_sample = steps_per_sample
        self.interval_ms = interval_ms
        self.repeat = repeat
        self.elev = elev
        self.azim = azim

        self.current_sample_idx: Optional[int] = None
        self.current_sample: Optional[SampleView] = None

        self.fig = plt.figure(figsize=(15, 10))
        self.ax_bathy_2d = self.fig.add_subplot(2, 2, 1)
        self.ax_bathy_3d = self.fig.add_subplot(2, 2, 2, projection="3d")
        self.ax_wave_2d = self.fig.add_subplot(2, 2, 3)
        self.ax_combo_3d = self.fig.add_subplot(2, 2, 4, projection="3d")

        self.im_bathy = None
        self.im_wave = None
        self._xy_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        self._init_axes()

    def _init_axes(self) -> None:
        self.ax_bathy_2d.set_title("Bathymetry (2D heatmap)")
        self.ax_bathy_2d.set_xlabel("x")
        self.ax_bathy_2d.set_ylabel("y")

        self.ax_wave_2d.set_title("Wave elevation eta (2D heatmap)")
        self.ax_wave_2d.set_xlabel("x")
        self.ax_wave_2d.set_ylabel("y")

        self.ax_bathy_3d.set_title("Bathymetry (3D surface)")
        self.ax_combo_3d.set_title("Wave on bathymetry (3D overlay)")

    def _frame_sequence(self) -> Iterable[tuple[int, int]]:
        while True:
            for sample_idx in range(len(self.adapter)):
                sample = self.adapter.get_sample(sample_idx)
                n_steps = sample.wave_frames.shape[0]
                if self.steps_per_sample is not None:
                    n_steps = min(n_steps, self.steps_per_sample)
                for local_step in range(n_steps):
                    yield sample_idx, local_step
            if not self.repeat:
                break

    @staticmethod
    def _compute_xy(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = shape
        x = np.arange(w)
        y = np.arange(h)
        return np.meshgrid(x, y)

    def _get_xy(self, bathymetry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = bathymetry.shape
        if key not in self._xy_cache:
            self._xy_cache[key] = self._compute_xy(key)
        return self._xy_cache[key]

    def _ensure_current_sample(self, sample_idx: int) -> None:
        if self.current_sample_idx == sample_idx and self.current_sample is not None:
            return
        self.current_sample_idx = sample_idx
        self.current_sample = self.adapter.get_sample(sample_idx)

    def _plot_bathy_3d(self, bathymetry: np.ndarray) -> None:
        ax = self.ax_bathy_3d
        ax.cla()
        X, Y = self._get_xy(bathymetry)
        ax.plot_surface(X, Y, bathymetry, cmap="terrain", linewidth=0, antialiased=False)
        ax.set_title("Bathymetry (3D surface)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.view_init(elev=self.elev, azim=self.azim)

    def _plot_combo_3d(self, bathymetry: np.ndarray, wave: np.ndarray) -> None:
        ax = self.ax_combo_3d
        ax.cla()
        X, Y = self._get_xy(bathymetry)
        ax.plot_surface(X, Y, bathymetry, cmap="terrain", linewidth=0, antialiased=False, alpha=0.95)
        ax.plot_surface(X, Y, wave, cmap="Blues", linewidth=0, antialiased=False, alpha=0.65)
        ax.set_title("Wave on bathymetry (3D overlay)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("elevation")
        ax.view_init(elev=self.elev, azim=self.azim)

    def _update_2d_images(self, bathymetry: np.ndarray, wave: np.ndarray):
        if self.im_bathy is None:
            self.im_bathy = self.ax_bathy_2d.imshow(bathymetry, cmap="terrain", origin="lower", aspect="auto")
            self.fig.colorbar(self.im_bathy, ax=self.ax_bathy_2d, fraction=0.046, pad=0.04)
        else:
            self.im_bathy.set_data(bathymetry)
            self.im_bathy.set_clim(vmin=float(bathymetry.min()), vmax=float(bathymetry.max()))

        if self.im_wave is None:
            vmax = float(np.max(np.abs(wave)))
            self.im_wave = self.ax_wave_2d.imshow(
                wave,
                cmap="RdBu_r",
                origin="lower",
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )
            self.fig.colorbar(self.im_wave, ax=self.ax_wave_2d, fraction=0.046, pad=0.04)
        else:
            vmax = max(float(np.max(np.abs(wave))), 1e-8)
            self.im_wave.set_data(wave)
            self.im_wave.set_clim(vmin=-vmax, vmax=vmax)

    def update(self, frame_spec: tuple[int, int]):
        sample_idx, local_step = frame_spec
        self._ensure_current_sample(sample_idx)
        assert self.current_sample is not None
        sample = self.current_sample

        bathymetry = sample.bathymetry
        wave = sample.wave_frames[local_step]

        self._update_2d_images(bathymetry, wave)
        self._plot_bathy_3d(bathymetry)
        self._plot_combo_3d(bathymetry, wave)

        time_label = ""
        if sample.timestamps is not None and local_step < len(sample.timestamps):
            time_label = f", t={sample.timestamps[local_step]:.4f}"

        self.fig.suptitle(
            f"{sample.sample_id} | step {local_step + 1}/{sample.wave_frames.shape[0]}{time_label}",
            fontsize=14,
        )

        return []

    def animate(self) -> FuncAnimation:
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self._frame_sequence(),
            interval=self.interval_ms,
            blit=False,
            repeat=self.repeat,
            cache_frame_data=False,
        )
        return ani


def build_adapter(args: argparse.Namespace) -> BaseDatasetAdapter:
    if args.mode == "raw":
        raw_dir = Path(args.raw_dir)
        return RawDatasetAdapter(raw_dir, start_idx=args.start, end_idx=args.end)

    if args.mode == "processed":
        processed_dir = Path(args.processed_dir)
        return ProcessedDatasetAdapter(processed_dir, start_idx=args.start, end_idx=args.end)

    # auto mode
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    if raw_dir.exists() and any(raw_dir.glob("sample_*")):
        return RawDatasetAdapter(raw_dir, start_idx=args.start, end_idx=args.end)
    if processed_dir.exists() and (processed_dir / "X_bathymetry.npz").exists() and (processed_dir / "Y.npy").exists():
        return ProcessedDatasetAdapter(processed_dir, start_idx=args.start, end_idx=args.end)

    raise FileNotFoundError(
        "Could not auto-detect dataset layout. Pass --mode raw with --raw-dir, or --mode processed with --processed-dir."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cycle through tsunami samples and visualize bathymetry + waves in one 4-panel window."
    )
    parser.add_argument("--mode", choices=["auto", "raw", "processed"], default="auto")
    parser.add_argument("--raw-dir", type=str, default="data/raw/samples",
                        help="Path to raw sample folders (sample_000001/sample.npz, ...)")
    parser.add_argument("--processed-dir", type=str, default="data/processed/test",
                        help="Path to processed split folder (X_bathymetry.npz, Y.npy, ...)")
    parser.add_argument("--start", type=int, default=1, help="1-based first sample index to show")
    parser.add_argument("--end", type=int, default=100, help="1-based last sample index to show")
    parser.add_argument("--steps-per-sample", type=int, default=20,
                        help="How many frames to render before switching to the next sample. Use -1 for all.")
    parser.add_argument("--interval", type=int, default=150, help="Animation interval in milliseconds")
    parser.add_argument("--repeat", action="store_true", help="Loop forever instead of stopping at the last sample")
    parser.add_argument("--elev", type=float, default=35.0, help="3D camera elevation")
    parser.add_argument("--azim", type=float, default=-60.0, help="3D camera azimuth")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional output path (.gif or .mp4). If omitted, opens an interactive window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps_per_sample is not None and args.steps_per_sample < 0:
        args.steps_per_sample = None

    adapter = build_adapter(args)
    visualizer = DatasetVisualizer(
        adapter=adapter,
        steps_per_sample=args.steps_per_sample,
        interval_ms=args.interval,
        repeat=args.repeat,
        elev=args.elev,
        azim=args.azim,
    )
    ani = visualizer.animate()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".gif":
            ani.save(out_path, writer="pillow")
        else:
            ani.save(out_path)
        print(f"Saved animation to {out_path}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Generating dataset.")
    print("Finished, running visualization")
    sd.main()
    main()
