from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from src.utils.config import save_yaml


def configure_logger(log_dir: str | Path, name: str = "tsunami", level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


@dataclass
class ExperimentLogger:
    log_dir: str | Path
    name: str = "experiment"
    logger: logging.Logger = field(init=False)
    metrics_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logger(self.log_dir, self.name)
        self.metrics_path = self.log_dir / "metrics.jsonl"

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def save_config(self, config: Dict[str, Any], filename: str = "config_resolved.yaml") -> None:
        save_yaml(config, self.log_dir / filename)

    def log_metrics(self, step: int, split: str, metrics: Dict[str, Any]) -> None:
        payload = {"step": int(step), "split": split}
        payload.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        formatted = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        self.info(f"[{split}] step={step} {formatted}")
