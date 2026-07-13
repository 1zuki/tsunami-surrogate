from __future__ import annotations

from typing import Literal

import numpy as np


SpongeTimeMode = Literal["legacy_per_step", "elapsed_time_consistent"]
FilterTimeMode = Literal[
    "legacy_per_step",
    "disabled",
    "elapsed_time_consistent",
]
CGFailureMode = Literal["legacy_posthoc", "strict_v2"]


def validate_sponge_time_mode(mode: str, reference_dt: float | None) -> SpongeTimeMode:
    normalized = str(mode).strip().lower()
    if normalized not in {"legacy_per_step", "elapsed_time_consistent"}:
        raise ValueError(
            "sponge_time_mode must be legacy_per_step or elapsed_time_consistent"
        )
    if normalized == "elapsed_time_consistent":
        if reference_dt is None or not np.isfinite(reference_dt) or reference_dt <= 0:
            raise ValueError(
                "elapsed_time_consistent sponge requires a positive sponge_reference_dt"
            )
    return normalized  # type: ignore[return-value]


def sponge_factor(
    reference_mask: np.ndarray,
    *,
    dt: float,
    mode: SpongeTimeMode,
    reference_dt: float | None,
) -> np.ndarray:
    mask = np.asarray(reference_mask, dtype=float)
    if mode == "legacy_per_step":
        return mask
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if reference_dt is None or not np.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("sponge_reference_dt must be positive and finite")
    return np.power(mask, float(dt) / float(reference_dt))


def validate_filter_time_mode(mode: str, reference_dt: float | None) -> FilterTimeMode:
    normalized = str(mode).strip().lower()
    allowed = {"legacy_per_step", "disabled", "elapsed_time_consistent"}
    if normalized not in allowed:
        raise ValueError(
            "filter_time_mode must be legacy_per_step, disabled, or "
            "elapsed_time_consistent"
        )
    if normalized == "elapsed_time_consistent":
        if reference_dt is None or not np.isfinite(reference_dt) or reference_dt <= 0:
            raise ValueError(
                "elapsed_time_consistent filter requires a positive filter_reference_dt"
            )
    return normalized  # type: ignore[return-value]


def filter_coefficient(
    reference_strength: float,
    *,
    dt: float,
    mode: FilterTimeMode,
    reference_dt: float | None,
) -> float:
    strength = float(reference_strength)
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("filter_strength must be finite and non-negative")
    if mode == "disabled":
        return 0.0
    if mode == "legacy_per_step":
        return min(strength, 0.25)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if reference_dt is None or not np.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("filter_reference_dt must be positive and finite")
    coefficient = strength * float(dt) / float(reference_dt)
    if coefficient < 0.0 or coefficient > 0.25:
        raise ValueError(
            f"elapsed-time filter coefficient must lie in [0, 0.25]; got {coefficient}"
        )
    return coefficient


def validate_cg_failure_mode(mode: str) -> CGFailureMode:
    normalized = str(mode).strip().lower()
    if normalized not in {"legacy_posthoc", "strict_v2"}:
        raise ValueError("cg_failure_mode must be legacy_posthoc or strict_v2")
    return normalized  # type: ignore[return-value]
