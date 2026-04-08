from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


@dataclass
class ModelCheckpoint:
    dirpath: str | Path
    monitor: str = "val_rmse"
    mode: str = "min"
    save_last: bool = True
    best_score: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.dirpath = Path(self.dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def save(self, state: Dict[str, Any], score: float) -> dict[str, Path | None]:
        saved = {"best": None, "last": None}
        if self._is_better(score):
            self.best_score = score
            best_path = self.dirpath / "best.pt"
            torch.save(state, best_path)
            saved["best"] = best_path
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            torch.save(state, last_path)
            saved["last"] = last_path
        return saved
