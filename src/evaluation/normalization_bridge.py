from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.evaluation.target_scaling import resolve_dataset_npz


@dataclass(frozen=True)
class StandardizationSpec:
    path: Path
    inputs: dict[str, tuple[float, float]]
    target: tuple[float, float]
    target_variable: str


def _finite_offset_scale(spec: Mapping[str, Any], label: str) -> tuple[float, float]:
    offset = float(spec.get("offset", 0.0))
    scale = float(spec.get("scale", 1.0))
    if not math.isfinite(offset) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Invalid standardization for {label}: offset={offset}, scale={scale}"
        )
    return offset, scale


def load_standardization_spec(path: str | Path) -> StandardizationSpec:
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    with stats_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected object JSON in {stats_path}")
    if str(payload.get("method", "")).lower() != "standardize":
        raise ValueError(
            f"Normalization bridge requires method=standardize in {stats_path}"
        )

    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError(f"Missing inputs mapping in {stats_path}")
    inputs: dict[str, tuple[float, float]] = {}
    for name, raw_spec in raw_inputs.items():
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"Invalid input stats for {name!r} in {stats_path}")
        inputs[str(name)] = _finite_offset_scale(
            raw_spec, f"input {name!r} in {stats_path}"
        )

    raw_target = payload.get("targets")
    if not isinstance(raw_target, Mapping) or not bool(raw_target.get("enabled", True)):
        raise ValueError(
            f"Normalization bridge requires enabled target stats in {stats_path}"
        )
    target = _finite_offset_scale(raw_target, f"target in {stats_path}")
    target_variable = str(raw_target.get("variable", ""))
    if not target_variable:
        raise ValueError(f"Missing target variable in {stats_path}")

    return StandardizationSpec(
        path=stats_path,
        inputs=inputs,
        target=target,
        target_variable=target_variable,
    )


def load_input_order(dataset_path: str | Path) -> list[str]:
    npz_path = resolve_dataset_npz(dataset_path)
    with np.load(npz_path, allow_pickle=True) as data:
        if "input_order" not in data:
            raise KeyError(f"Missing input_order in {npz_path}")
        values = np.asarray(data["input_order"]).reshape(-1).tolist()

    order = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]
    if not order or any(not name for name in order) or len(set(order)) != len(order):
        raise ValueError(f"Invalid input_order in {npz_path}: {order}")
    return order


class EvaluationNormalizationBridge:
    """Translate an evaluation batch into a checkpoint's training normalization."""

    def __init__(
        self,
        dataset_stats: StandardizationSpec,
        model_stats: StandardizationSpec,
        input_order: list[str],
    ) -> None:
        if dataset_stats.target_variable != model_stats.target_variable:
            raise ValueError(
                "Target variable mismatch: "
                f"{dataset_stats.target_variable!r} != {model_stats.target_variable!r}"
            )
        known_channels = set(input_order)
        stats_channels = set(dataset_stats.inputs) | set(model_stats.inputs)
        unknown = sorted(stats_channels - known_channels)
        if unknown:
            raise ValueError(
                f"Normalization stats contain channels absent from input_order: {unknown}"
            )

        self.dataset_stats = dataset_stats
        self.model_stats = model_stats
        self.input_order = list(input_order)

    @property
    def model_target_denorm(self) -> tuple[float, float]:
        return self.model_stats.target

    @staticmethod
    def _rebase(
        tensor: torch.Tensor,
        source: tuple[float, float] | None,
        destination: tuple[float, float] | None,
    ) -> torch.Tensor:
        physical = tensor
        if source is not None:
            physical = physical * source[1] + source[0]
        if destination is not None:
            return (physical - destination[0]) / destination[1]
        return physical

    def transform(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim < 2 or int(x.shape[1]) != len(self.input_order):
            raise ValueError(
                "Input channel mismatch for normalization bridge: "
                f"tensor has {x.shape[1] if x.ndim >= 2 else 'no'} channels, "
                f"input_order has {len(self.input_order)}"
            )

        channels = []
        for index, name in enumerate(self.input_order):
            channels.append(
                self._rebase(
                    x[:, index],
                    self.dataset_stats.inputs.get(name),
                    self.model_stats.inputs.get(name),
                )
            )
        x_model = torch.stack(channels, dim=1)
        y_model = self._rebase(
            y,
            self.dataset_stats.target,
            self.model_stats.target,
        )
        return x_model, y_model

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "dataset_stats_path": str(self.dataset_stats.path),
            "model_stats_path": str(self.model_stats.path),
            "input_order": list(self.input_order),
            "target_variable": self.model_stats.target_variable,
            "comparison_normalization": "model_training_stats",
            "physical_metrics_denorm": "model_training_target_stats",
        }


def load_evaluation_normalization_bridge(
    dataset_path: str | Path,
    dataset_stats_path: str | Path,
    model_stats_path: str | Path,
) -> EvaluationNormalizationBridge:
    return EvaluationNormalizationBridge(
        dataset_stats=load_standardization_spec(dataset_stats_path),
        model_stats=load_standardization_spec(model_stats_path),
        input_order=load_input_order(dataset_path),
    )
