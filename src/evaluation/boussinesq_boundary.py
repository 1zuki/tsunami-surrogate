"""Evaluation-only spectral tools for Boussinesq boundary verification.

The production solver evolves ``[eta, eta_t]`` with the semi-discrete model

    (I - alpha div(H^2 grad)) eta_tt = g div(H grad(eta)).

This module mirrors the constant-depth face-flux symbol used by that solver.
It deliberately contains no production boundary implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


Direction = Literal["left", "right"]


@dataclass(frozen=True)
class SpectralPacketSpec:
    length: float = 32.0
    dx: float = 0.125
    ny: int = 4
    dy: float = 0.25
    center: float = 16.0
    carrier_wavenumber: float = 1.5
    spectral_width: float = 0.3
    amplitude: float = 1.0e-5
    direction: Direction = "left"
    depth: float = 1.0
    gravity: float = 9.81
    alpha: float = 1.0 / 3.0
    spectral_energy_tail: float = 1.0e-6
    spatial_energy_tail: float = 1.0e-8
    reference_length: float = 512.0

    @property
    def nx(self) -> int:
        cells = self.length / self.dx
        rounded = int(round(cells))
        if not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("packet length must be an integer multiple of dx")
        return rounded


def validate_packet_spec(spec: SpectralPacketSpec) -> None:
    if spec.length <= 0.0 or spec.dx <= 0.0 or spec.dy <= 0.0:
        raise ValueError("packet lengths and spacings must be positive")
    if spec.ny <= 1:
        raise ValueError("packet ny must be greater than one")
    if not (0.0 < spec.center < spec.length):
        raise ValueError("packet center must lie inside the finite domain")
    if spec.carrier_wavenumber <= 0.0 or spec.spectral_width <= 0.0:
        raise ValueError("packet spectral parameters must be positive")
    if spec.amplitude <= 0.0 or spec.depth <= 0.0:
        raise ValueError("packet amplitude and depth must be positive")
    if spec.gravity <= 0.0 or spec.alpha < 0.0:
        raise ValueError("packet gravity must be positive and alpha non-negative")
    if spec.direction not in ("left", "right"):
        raise ValueError("packet direction must be left or right")
    for name, value in (
        ("spectral_energy_tail", spec.spectral_energy_tail),
        ("spatial_energy_tail", spec.spatial_energy_tail),
    ):
        if not (0.0 < value < 1.0):
            raise ValueError(f"{name} must lie in (0, 1)")
    if spec.reference_length <= spec.length:
        raise ValueError("reference domain must be longer than the finite domain")
    _ = spec.nx
    reference_cells = spec.reference_length / spec.dx
    if not math.isclose(
        reference_cells,
        round(reference_cells),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("reference length must be an integer multiple of dx")


def discrete_wavenumber(wavenumber: np.ndarray, *, dx: float) -> np.ndarray:
    values = np.asarray(wavenumber, dtype=np.float64)
    return 2.0 * np.sin(0.5 * values * dx) / dx


def discrete_dispersion(
    wavenumber: np.ndarray,
    *,
    dx: float,
    depth: float,
    gravity: float = 9.81,
    alpha: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-negative omega, signed phase velocity, and group velocity."""
    if dx <= 0.0 or depth <= 0.0 or gravity <= 0.0 or alpha < 0.0:
        raise ValueError("invalid discrete-dispersion parameters")
    k = np.asarray(wavenumber, dtype=np.float64)
    kd = discrete_wavenumber(k, dx=dx)
    denominator = 1.0 + alpha * depth * depth * kd * kd
    omega = math.sqrt(gravity * depth) * np.abs(kd) / np.sqrt(denominator)
    phase = np.zeros_like(omega)
    nonzero = k != 0.0
    phase[nonzero] = omega[nonzero] / k[nonzero]
    group = (
        math.sqrt(gravity * depth)
        * np.sign(kd)
        * np.cos(0.5 * k * dx)
        / denominator**1.5
    )
    group[~nonzero] = math.sqrt(gravity * depth)
    return omega, phase, group


def _signed_omega(
    nx: int,
    *,
    dx: float,
    depth: float,
    gravity: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    kd = discrete_wavenumber(k, dx=dx)
    omega, _, group = discrete_dispersion(
        k,
        dx=dx,
        depth=depth,
        gravity=gravity,
        alpha=alpha,
    )
    return k, np.sign(kd) * omega, group


def directional_rate(
    eta: np.ndarray,
    *,
    dx: float,
    depth: float,
    direction: Direction,
    gravity: float = 9.81,
    alpha: float = 1.0 / 3.0,
) -> np.ndarray:
    values = np.asarray(eta, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("eta must have shape [x, y]")
    _, signed_omega, _ = _signed_omega(
        values.shape[0],
        dx=dx,
        depth=depth,
        gravity=gravity,
        alpha=alpha,
    )
    direction_sign = 1.0 if direction == "left" else -1.0
    rate_hat = direction_sign * 1j * signed_omega[:, None] * np.fft.fft(values, axis=0)
    return np.fft.ifft(rate_hat, axis=0).real


def directional_states(
    states: np.ndarray,
    *,
    dx: float,
    depth: float,
    gravity: float = 9.81,
    alpha: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[time, 2, x, y]`` states into right- and left-going states."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("states must have shape [time, 2, x, y]")
    _, signed_omega, _ = _signed_omega(
        values.shape[2],
        dx=dx,
        depth=depth,
        gravity=gravity,
        alpha=alpha,
    )
    eta_hat = np.fft.fft(values[:, 0], axis=1)
    rate_hat = np.fft.fft(values[:, 1], axis=1)
    inverse = np.zeros_like(signed_omega)
    nonzero = signed_omega != 0.0
    inverse[nonzero] = 1.0 / signed_omega[nonzero]
    scaled_rate = 1j * rate_hat * inverse[None, :, None]
    right_eta_hat = 0.5 * (eta_hat + scaled_rate)
    left_eta_hat = 0.5 * (eta_hat - scaled_rate)
    right_eta_hat[:, ~nonzero, :] = 0.0
    left_eta_hat[:, ~nonzero, :] = 0.0
    right_rate_hat = -1j * signed_omega[None, :, None] * right_eta_hat
    left_rate_hat = 1j * signed_omega[None, :, None] * left_eta_hat

    def _state(eta_branch: np.ndarray, rate_branch: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.fft.ifft(eta_branch, axis=1).real,
                np.fft.ifft(rate_branch, axis=1).real,
            ],
            axis=1,
        )

    return _state(right_eta_hat, right_rate_hat), _state(left_eta_hat, left_rate_hat)


def _periodic_forward_difference(
    values: np.ndarray, spacing: float, axis: int
) -> np.ndarray:
    return (np.roll(values, -1, axis=axis) - values) / spacing


def energy_density(
    state: np.ndarray,
    *,
    dx: float,
    dy: float,
    depth: float,
    gravity: float = 9.81,
    alpha: float = 1.0 / 3.0,
) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.shape[0] != 2 or values.ndim != 3:
        raise ValueError("state must have shape [2, x, y]")
    eta, rate = values
    eta_x = _periodic_forward_difference(eta, dx, axis=0)
    eta_y = _periodic_forward_difference(eta, dy, axis=1)
    rate_x = _periodic_forward_difference(rate, dx, axis=0)
    rate_y = _periodic_forward_difference(rate, dy, axis=1)
    return 0.5 * (
        rate * rate
        + alpha * depth * depth * (rate_x * rate_x + rate_y * rate_y)
        + gravity * depth * (eta_x * eta_x + eta_y * eta_y)
    )


def discrete_energy(
    state: np.ndarray,
    *,
    dx: float,
    dy: float,
    depth: float,
    gravity: float = 9.81,
    alpha: float = 1.0 / 3.0,
) -> float:
    density = energy_density(
        state,
        dx=dx,
        dy=dy,
        depth=depth,
        gravity=gravity,
        alpha=alpha,
    )
    return float(math.fsum(density.ravel()) * dx * dy)


def _spectral_envelope(k: np.ndarray, spec: SpectralPacketSpec) -> np.ndarray:
    envelope = np.exp(
        -0.5 * ((np.abs(k) - spec.carrier_wavenumber) / spec.spectral_width) ** 2
    )
    envelope[k == 0.0] = 0.0
    return envelope


def build_reference_packet(
    spec: SpectralPacketSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | list[float]]]:
    """Build the packet on a padded periodic domain and return its finite crop."""
    validate_packet_spec(spec)
    reference_nx = int(round(spec.reference_length / spec.dx))
    reference_left = -0.5 * (spec.reference_length - spec.length)
    x = reference_left + np.arange(reference_nx, dtype=np.float64) * spec.dx
    k, signed_omega, group = _signed_omega(
        reference_nx,
        dx=spec.dx,
        depth=spec.depth,
        gravity=spec.gravity,
        alpha=spec.alpha,
    )
    eta_hat = _spectral_envelope(k, spec).astype(np.complex128)
    center_index_coordinate = spec.center - reference_left
    eta_hat *= np.exp(-1j * k * center_index_coordinate)
    eta = np.fft.ifft(eta_hat).real
    eta *= spec.amplitude / max(float(np.max(np.abs(eta))), 1.0e-30)
    eta = eta[:, None] * np.ones((1, spec.ny), dtype=np.float64)
    rate = directional_rate(
        eta,
        dx=spec.dx,
        depth=spec.depth,
        direction=spec.direction,
        gravity=spec.gravity,
        alpha=spec.alpha,
    )
    reference_state = np.stack([eta, rate], axis=0)
    crop_start = int(round((0.0 - reference_left) / spec.dx))
    crop_stop = crop_start + spec.nx
    finite_state = reference_state[:, crop_start:crop_stop].copy()

    positive = k > 0.0
    modal_energy = np.zeros_like(k)
    kd = discrete_wavenumber(k, dx=spec.dx)
    modal_energy[positive] = (
        spec.gravity
        * spec.depth
        * kd[positive] ** 2
        * np.abs(np.fft.fft(eta[:, 0])[positive]) ** 2
    )
    positive_indices = np.flatnonzero(positive)
    weights = modal_energy[positive_indices]
    weights /= max(float(math.fsum(weights)), 1.0e-300)
    cumulative = np.cumsum(weights)
    lower_q = 0.5 * spec.spectral_energy_tail
    upper_q = 1.0 - lower_q
    lower_position = min(int(np.searchsorted(cumulative, lower_q)), weights.size - 1)
    upper_position = min(int(np.searchsorted(cumulative, upper_q)), weights.size - 1)
    support_indices = positive_indices[lower_position : upper_position + 1]
    support_group = np.abs(group[support_indices])
    if support_indices.size == 0 or np.any(support_group <= 0.0):
        raise ValueError("packet has no propagating significant spectral support")

    column_density = np.sum(
        energy_density(
            finite_state,
            dx=spec.dx,
            dy=spec.dy,
            depth=spec.depth,
            gravity=spec.gravity,
            alpha=spec.alpha,
        ),
        axis=1,
    )
    order = np.argsort(np.abs(np.arange(spec.nx) * spec.dx - spec.center))
    cumulative_space = np.cumsum(column_density[order]) / max(
        float(math.fsum(column_density)), 1.0e-300
    )
    count = min(
        int(np.searchsorted(cumulative_space, 1.0 - spec.spatial_energy_tail)) + 1,
        spec.nx,
    )
    spatial_indices = np.sort(order[:count])
    support_left = float(spatial_indices[0] * spec.dx)
    support_right = float((spatial_indices[-1] + 1) * spec.dx)
    reference_distance = min(
        spec.center - reference_left,
        reference_left + spec.reference_length - spec.center,
    )
    metadata: dict[str, float | int | list[float]] = {
        "reference_nx": reference_nx,
        "reference_left": reference_left,
        "crop_start": crop_start,
        "crop_stop": crop_stop,
        "significant_k_min": float(k[support_indices[0]]),
        "significant_k_max": float(k[support_indices[-1]]),
        "group_velocity_min": float(np.min(support_group)),
        "group_velocity_max": float(np.max(support_group)),
        "spatial_support_left": support_left,
        "spatial_support_right": support_right,
        "reference_distance": float(reference_distance),
    }
    return finite_state, reference_state, metadata


def packet_timing(
    spec: SpectralPacketSpec,
    metadata: dict[str, float | int | list[float]],
    *,
    production_horizon: float = 0.175,
    prearrival_count: int = 4,
    postexit_count: int = 5,
) -> dict[str, float | bool | list[float]]:
    if prearrival_count < 2 or postexit_count < 2:
        raise ValueError("packet timing requires at least two pre/post samples")
    vmin = float(metadata["group_velocity_min"])
    vmax = float(metadata["group_velocity_max"])
    left = float(metadata["spatial_support_left"])
    right = float(metadata["spatial_support_right"])
    if spec.direction == "left":
        leading_distance, trailing_distance = left, right
    else:
        leading_distance = spec.length - right
        trailing_distance = spec.length - left
    arrival = leading_distance / vmax
    exit_time = trailing_distance / vmin
    support_width = right - left
    observation_end = exit_time + support_width / vmax
    prearrival = np.linspace(
        max(production_horizon, 0.25 * arrival), 0.9 * arrival, prearrival_count
    )
    postexit = np.linspace(exit_time, observation_end, postexit_count)
    production = np.arange(1, 51, dtype=np.float64) * (production_horizon / 50.0)
    requested = np.unique(np.concatenate([production, prearrival, postexit]))
    reference_safe = (
        float(metadata["reference_distance"]) > vmax * observation_end + support_width
    )
    return {
        "leading_edge_arrival_time": float(arrival),
        "trailing_edge_exit_time": float(exit_time),
        "observation_end_time": float(observation_end),
        "prearrival_times": prearrival.tolist(),
        "postexit_times": postexit.tolist(),
        "production_times": production.tolist(),
        "requested_times": requested.tolist(),
        "reference_safe": bool(reference_safe),
    }


def evolve_reference(
    reference_state: np.ndarray,
    times: np.ndarray,
    *,
    spec: SpectralPacketSpec,
) -> np.ndarray:
    values = np.asarray(reference_state, dtype=np.float64)
    queries = np.asarray(times, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("reference_state must have shape [2, x, y]")
    if queries.ndim != 1 or np.any(queries < 0.0):
        raise ValueError("reference times must be a non-negative 1-D array")
    _, signed_omega, _ = _signed_omega(
        values.shape[1],
        dx=spec.dx,
        depth=spec.depth,
        gravity=spec.gravity,
        alpha=spec.alpha,
    )
    eta_hat0 = np.fft.fft(values[0], axis=0)
    sign = 1.0 if spec.direction == "left" else -1.0
    phase = np.exp(sign * 1j * queries[:, None] * signed_omega[None, :])
    eta_hat = phase[:, :, None] * eta_hat0[None, :, :]
    rate_hat = sign * 1j * signed_omega[None, :, None] * eta_hat
    return np.stack(
        [
            np.fft.ifft(eta_hat, axis=1).real,
            np.fft.ifft(rate_hat, axis=1).real,
        ],
        axis=1,
    )


def cosine_taper(nx: int, edge_cells: int) -> np.ndarray:
    if nx <= 1 or edge_cells < 0 or 2 * edge_cells >= nx:
        raise ValueError("invalid cosine taper shape")
    taper = np.ones(nx, dtype=np.float64)
    if edge_cells == 0:
        return taper
    phase = np.arange(1, edge_cells + 1, dtype=np.float64) / (edge_cells + 1)
    ramp = 0.5 * (1.0 - np.cos(math.pi * phase))
    taper[:edge_cells] = ramp
    taper[-edge_cells:] = ramp[::-1]
    return taper
