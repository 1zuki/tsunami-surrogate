from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class WindowedTrajectoryDataset(Dataset):
    """Expose non-overlapping K-frame windows over an eta trajectory for windowed FNO training.

    Wraps a base tsunami dataset whose items provide:
      x: [3, H, W] = [bathymetry, source, initial_depth]   (initial_depth is still-water DEPTH, not eta)
      y: [T, H, W] = eta trajectory frames (T=50)

    Index convention (eta frame numbering = base y indexing, 0..T-1):
      A window with start s predicts y[s+1 : s+1+K] from the state at frame s.
      State channels: [bathymetry, source, eta_s, eta_{s-1}].
      For s == 0 there is no previous frame, so eta_{-1} := eta_0 (== y[0]).
    Frame y[0] is treated as given (the rollout seed); the model never predicts it.

    Window starts are non-overlapping: s in {0, K, 2K, ...} with s+1+K <= T (last window clamped).
    Each base sample with T=50, K=5 yields starts {0,5,...,45} -> 10 windows, targets covering y[1..50].
    """

    def __init__(
        self,
        base_dataset: Dataset,
        K: int = 5,
        prev: bool = True,
        include_source: bool = True,
    ) -> None:
        self.base = base_dataset
        self.K = int(K)
        self.prev = bool(prev)
        self.include_source = bool(include_source)
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")

        # Determine T from the first base item.
        first = self.base[0]
        y0 = first["y"]
        self.T = int(y0.shape[0])
        if self.T < self.K + 1:
            raise ValueError(f"trajectory T={self.T} too short for window K={self.K}")

        # Non-overlapping starts; clamp the final start so the target window stays in range.
        starts: List[int] = []
        s = 0
        while s + 1 < self.T:
            start = min(s, self.T - 1 - self.K)
            if start < 0:
                start = 0
            if not starts or starts[-1] != start:
                starts.append(start)
            s += self.K
        self.starts = starts
        self.windows_per_sample = len(self.starts)

    def __len__(self) -> int:
        return len(self.base) * self.windows_per_sample

    def _decode(self, idx: int) -> Tuple[int, int]:
        base_idx = idx // self.windows_per_sample
        win_idx = idx % self.windows_per_sample
        return base_idx, win_idx

    def base_index_for_window(self, idx: int) -> int:
        """Return the underlying base-dataset index for a window index.

        Used by the windowed batch sampler to group windows that share a base
        sample (hence the same shard) into the same batch, keeping shard access
        local and avoiding cache thrashing under shuffling.
        """
        return int(idx) // self.windows_per_sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx, win_idx = self._decode(idx)
        item = self.base[base_idx]
        x = item["x"]
        y = item["y"]
        start = self.starts[win_idx]

        bathy = x[0]
        source = x[1]
        eta_t = y[start]
        eta_prev = y[start - 1] if (self.prev and start >= 1) else eta_t

        chans = [bathy, source] if self.include_source else [bathy]
        chans.append(eta_t)
        if self.prev:
            chans.append(eta_prev)
        win_x = torch.stack(chans, dim=0)

        end = start + 1 + self.K
        win_y = y[start + 1 : end]
        # Clamp guarantees end <= T, so win_y has exactly K frames.

        return {
            "x": win_x,
            "y": win_y,
            "sample_id": item.get("sample_id", ""),
            "source_id": item.get("source_id", ""),
            "source_type": item.get("source_type", ""),
            "bathymetry_type": item.get("bathymetry_type", ""),
            "source_strength": item.get("source_strength", 0.0),
            "scenario_id": item.get("scenario_id", ""),
            "solver_name": item.get("solver_name", ""),
        }
