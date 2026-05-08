from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np

BoundaryMode = Literal["open", "reflective", "periodic"]

def validate_boundary_mode(mode: BoundaryMode, name: str) -> None:
    if mode not in ("open", "reflective", "periodic"):
        raise ValueError(f"{name} must be one of: open, reflective, periodic")

def resolve_boundary_modes(
        boundary: Union[BoundaryMode, Tuple[BoundaryMode, BoundaryMode]]
    ) -> tuple[BoundaryMode, BoundaryMode]:
    
    if isinstance(boundary, tuple):
        if len(boundary) != 2:
            raise ValueError("boundary tuple must have 2 values: (boundary_x, boundary_y)")

        boundary_x, boundary_y = boundary

    else:
        boundary_x = boundary
        boundary_y = boundary

    validate_boundary_mode(boundary_x, "boundary_x")
    validate_boundary_mode(boundary_y, "boundary_y")

    return boundary_x, boundary_y

def pad_mode(mode: BoundaryMode) -> str:
    return "wrap" if mode == "periodic" else "edge"

def pad_scalar_field(A: np.ndarray, boundary_x: BoundaryMode, boundary_y: BoundaryMode) -> np.ndarray:
    if A.ndim != 2:
        raise ValueError(f"scalar field must be 2D, got shape {A.shape}")

    if boundary_x == boundary_y:
        return np.pad(A, pad_width=1, mode=pad_mode(boundary_x))

    padded = np.pad(A, pad_width=((1, 1), (0, 0)), mode=pad_mode(boundary_x))
    padded = np.pad(padded, pad_width=((0, 0), (1, 1)), mode=pad_mode(boundary_y))

    return padded

def pad_state_with_reflective_momentum(
    U: np.ndarray,
    boundary_x: BoundaryMode,
    boundary_y: BoundaryMode,
    normal_momentum_x_channel: int = 1,
    normal_momentum_y_channel: int = 2,
) -> np.ndarray:
    if U.ndim != 3:
        raise ValueError(f"stacked state must be 3D [C, H, W], got shape {U.shape}")

    if boundary_x == boundary_y:
        padded = np.pad(U, pad_width=((0, 0), (1, 1), (1, 1)), mode=pad_mode(boundary_x))

        if boundary_x == "reflective":
            padded[normal_momentum_x_channel, 0, :] *= -1
            padded[normal_momentum_x_channel, -1, :] *= -1
            padded[normal_momentum_y_channel, :, 0] *= -1
            padded[normal_momentum_y_channel, :, -1] *= -1

        return padded

    padded = np.pad(U, pad_width=((0, 0), (1, 1), (0, 0)), mode=pad_mode(boundary_x))

    if boundary_x == "reflective":
        padded[normal_momentum_x_channel, 0, :] *= -1
        padded[normal_momentum_x_channel, -1, :] *= -1

    padded = np.pad(padded, pad_width=((0, 0), (0, 0), (1, 1)), mode=pad_mode(boundary_y))

    if boundary_y == "reflective":
        padded[normal_momentum_y_channel, :, 0] *= -1
        padded[normal_momentum_y_channel, :, -1] *= -1

    return padded

def boundary_state_x(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    b: np.ndarray,
    j: int,
    side: Literal["left", "right"],
    boundary_x: BoundaryMode,
) -> tuple[float, float, float, float]:
    nx = h.shape[0]

    if side == "left":
        inside_i = 0
        periodic_i = nx - 1
    elif side == "right":
        inside_i = nx - 1
        periodic_i = 0
    else:
        raise ValueError("side must be 'left' or 'right'")

    if boundary_x == "periodic":
        return h[periodic_i, j], hu[periodic_i, j], hv[periodic_i, j], b[periodic_i, j]

    h_g = h[inside_i, j]
    hu_g = hu[inside_i, j]
    hv_g = hv[inside_i, j]
    b_g = b[inside_i, j]

    if boundary_x == "reflective":
        hu_g = -hu_g

    return h_g, hu_g, hv_g, b_g

def boundary_state_y(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    b: np.ndarray,
    i: int,
    side: Literal["bottom", "top"],
    boundary_y: BoundaryMode,
) -> tuple[float, float, float, float]:
    ny = h.shape[1]

    if side == "bottom":
        inside_j = 0
        periodic_j = ny - 1

    elif side == "top":
        inside_j = ny - 1
        periodic_j = 0
    else:
        raise ValueError("side must be 'bottom' or 'top'")

    if boundary_y == "periodic":
        return h[i, periodic_j], hu[i, periodic_j], hv[i, periodic_j], b[i, periodic_j]

    h_g = h[i, inside_j]
    hu_g = hu[i, inside_j]
    hv_g = hv[i, inside_j]
    b_g = b[i, inside_j]

    if boundary_y == "reflective":
        hv_g = -hv_g

    return h_g, hu_g, hv_g, b_g
