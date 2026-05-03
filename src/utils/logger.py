from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


def setup_logger(
    name: str,
    save_dir: Optional[Union[str, Path]] = None,
    filename: str = "run.log",
    level: Union[int, str] = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(save_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def save_json(path: Union[str, Path], data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def save_yaml(path: Union[str, Path], data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def append_jsonl(path: Union[str, Path], data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(data), sort_keys=True) + "\n")


class ExperimentLogger:
    def __init__(self, root_dir: Union[str, Path], run_name: str = "run") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = Path(root_dir) / f"{run_name}_{timestamp}"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(run_name, save_dir=self.root_dir)
        self.metrics_path = self.root_dir / "metrics.jsonl"

    def log(self, message: str) -> None:
        self.logger.info(message)

    def log_metrics(self, step: int, metrics: Mapping[str, Any], prefix: str = "") -> None:
        payload: Dict[str, Any] = {"step": int(step)}

        for key, value in metrics.items():
            payload[f"{prefix}{key}"] = value

        append_jsonl(self.metrics_path, payload)

    def save_config(self, config: Mapping[str, Any], filename: str = "config.yaml") -> None:
        save_yaml(self.root_dir / filename, config)
