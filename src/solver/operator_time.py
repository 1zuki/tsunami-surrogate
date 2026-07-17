from __future__ import annotations

from typing import Literal

import numpy as np


SpongeTimeMode = Literal["legacy_per_step", "elapsed_time_consistent"]
SpongeProfile = Literal["quadratic", "cosine"]
FilterTimeMode = Literal[
    "legacy_per_step",
    "disabled",
    "elapsed_time_consistent",
]
CGFailureMode = Literal["legacy_posthoc", "strict_v2"]


def validate_sponge_profile(profile: str) -> SpongeProfile:
    normalized = str(profile).strip().lower()
    if normalized not in {"quadratic", "cosine"}:
        raise ValueError("sponge_profile must be quadratic or cosine")
    return normalized  # type: ignore[return-value]


def build_sponge_mask(
    *,
    nx: int,
    ny: int,
    width: int,
    min_factor: float,
    axes: str,
    profile: SpongeProfile,
) -> np.ndarray:
    """Build the reference mask without assigning timestep semantics."""
    mask = np.ones((int(nx), int(ny)), dtype=float)
    effective_width = int(max(0, width))
    if effective_width == 0:
        return mask
    max_width = (
        max(1, min(int(nx), int(ny)) // 2)
        if axes == "xy"
        else max(1, int(nx) // 2)
    )
    effective_width = min(effective_width, max_width)
    for distance in range(effective_width):
        coordinate = (effective_width - distance) / effective_width
        if profile == "quadratic":
            weight = coordinate * coordinate
        else:
            weight = 0.5 * (1.0 - np.cos(np.pi * coordinate))
        value = 1.0 - (1.0 - float(min_factor)) * weight
        mask[distance, :] = np.minimum(mask[distance, :], value)
        mask[-(distance + 1), :] = np.minimum(mask[-(distance + 1), :], value)
        if axes == "xy":
            mask[:, distance] = np.minimum(mask[:, distance], value)
            mask[:, -(distance + 1)] = np.minimum(mask[:, -(distance + 1)], value)
    return mask


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
