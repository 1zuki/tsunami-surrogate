from __future__ import annotations
from typing import Any, Literal, Mapping, NamedTuple, Optional, Tuple, Union
import numpy as np

try:
    from src.solver.boundary_conditions import (
        BoundaryMode,
        boundary_state_x,
        boundary_state_y,
        pad_scalar_field,
        pad_state_with_reflective_momentum,
        resolve_boundary_modes,
    )
except ImportError:
    from boundary_conditions import (
        BoundaryMode,
        boundary_state_x,
        boundary_state_y,
        pad_scalar_field,
        pad_state_with_reflective_momentum,
        resolve_boundary_modes,
    )

try:
    from src.solver.operator_time import sponge_factor, validate_sponge_time_mode
except ImportError:
    from operator_time import sponge_factor, validate_sponge_time_mode

class SolverInfo(NamedTuple):
    """ metadata bundle for logging / experiment tracking"""
    nx: int
    ny: int
    dx: float
    dy: float
    dt: float
    g: float
    cfl: float
    dry_tolerance: float
    boundary_x: BoundaryMode
    boundary_y: BoundaryMode

class ShallowWaterSolver:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float, g: float = 9.81, cfl: float = 0.45,
                 dry_tolerance: float = 1e-6, boundary: Union[BoundaryMode, Tuple[BoundaryMode, BoundaryMode]] = "open",
                 use_sponge: bool = True, sponge_width = 20, sponge_min_factor: float = 0.9,
                 eps: float = 1e-9, max_velocity: float = 50.0,
                 sponge_time_mode: str = "legacy_per_step",
                 sponge_reference_dt: float | None = None) -> None:
        if nx <= 1 or ny <= 1:
            raise ValueError("nx and ny must be greater than 1")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if g <= 0:
            raise ValueError("g must be positive")
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        if dry_tolerance <= 0:
            raise ValueError("dry_tolerance must be positive")
        if sponge_width < 0:
            raise ValueError("sponge_width must be non-negative")
        if not (0.0 < sponge_min_factor <= 1.0):
            raise ValueError("sponge_min_factor must be in (0, 1]")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be positive")

        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.g = float(g)
        self.cfl = float(cfl)
        self.dry_tolerance = float(dry_tolerance)
        self.eps = float(eps)
        self.max_velocity = float(max_velocity)

        self.boundary_x, self.boundary_y = resolve_boundary_modes(boundary)

        self.h = np.zeros((self.nx, self.ny), dtype=float)
        self.hu = np.zeros((self.nx, self.ny), dtype=float)
        self.hv = np.zeros((self.nx, self.ny), dtype=float)
        self.b = np.zeros((self.nx, self.ny), dtype=float)

        self.use_sponge = bool(use_sponge)
        self.sponge_width = int(sponge_width)
        self.sponge_min_factor = float(sponge_min_factor)
        self.sponge_time_mode = validate_sponge_time_mode(
            sponge_time_mode, sponge_reference_dt
        )
        self.sponge_reference_dt = (
            None if sponge_reference_dt is None else float(sponge_reference_dt)
        )
        self.operator_diagnostics: dict[str, float | int | bool | str | None] = {}

        self._db_dx: Optional[np.ndarray] = None
        self._db_dy: Optional[np.ndarray] = None

        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)
        if self.use_sponge:
            self._init_sponge_layer(width=self.sponge_width, min_factor=self.sponge_min_factor)
        self.reset_operator_diagnostics()


    def _check_shape(self, arr: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)

        if arr.shape != (self.nx, self.ny):
            raise ValueError(f"{name} shape must be {(self.nx, self.ny)}, got {arr.shape}")

        return arr

    # settets
    def set_initial_condition(self, h0: np.ndarray, hu0: Optional[np.ndarray] = None, hv0: Optional[np.ndarray] = None) -> None:
        """ initial water depth (and momentum) """
        self.h = self._check_shape(h0, "h0").copy()
        self.h = np.maximum(self.h, 0.0)

        self.hu = np.zeros_like(self.h)
        self.hv = np.zeros_like(self.h)

        if hu0 is not None:
            self.hu = self._check_shape(hu0, "hu0").copy()

        if hv0 is not None:
            self.hv = self._check_shape(hv0, "hv0").copy()

        self._zero_momentum_in_dry_cells()

    def set_bathymetry(self, b: np.ndarray) -> None:
        self.b = self._check_shape(b, "b").copy()
        self._db_dx = None
        self._db_dy = None

    def _bathymetry_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        if self._db_dx is None or self._db_dy is None:
            if self.nx >= 3 and self.ny >= 3:
                self._db_dx, self._db_dy = np.gradient(self.b, self.dx, self.dy, edge_order=2)
            else:
                self._db_dx, self._db_dy = np.gradient(self.b, self.dx, self.dy)
        return self._db_dx, self._db_dy

    # state utils
    def get_state(self) -> np.ndarray:
        """ return state as [h, hu, hv] """
        return np.stack([self.h, self.hu, self.hv], axis=0)

    def set_state(self, U: np.ndarray) -> None:
        """ replace the state from a stacked array """
        U = np.asarray(U, dtype=float)

        if U.shape != (3, self.nx, self.ny):
            raise ValueError(f"U shape must be {(3, self.nx, self.ny)}, got {U.shape}")

        self.h = np.maximum(U[0].copy(), 0.0)
        self.hu = U[1].copy()
        self.hv = U[2].copy()
        self._zero_momentum_in_dry_cells()

    def compute_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """ return velocity (u, v) """
        h_safe = np.maximum(self.h, self.dry_tolerance)
        wet = self.h > self.dry_tolerance

        u = np.zeros_like(self.h)
        v = np.zeros_like(self.h)
        u[wet] = self.hu[wet] / h_safe[wet]
        v[wet] = self.hv[wet] / h_safe[wet]

        return u, v

    def compute_free_surface(self) -> np.ndarray:
        """ return the free-surface elevation eta = h + b """
        return self.h + self.b

    # flux func
    def _primitive(self, h: float, hu: float, hv: float) -> tuple[float, float]:
        """Return velocities (u, v) with dry-cell protection."""
        if h <= self.dry_tolerance:
            return 0.0, 0.0
        return hu / h, hv / h


    def _flux_x(self, h: float, hu: float, hv: float) -> np.ndarray:
        """Physical x-flux for the shallow-water system."""
        if h <= self.dry_tolerance:
            return np.zeros(3, dtype=float)

        u, v = self._primitive(h, hu, hv)
        return np.array(
            [
                hu,
                hu * u + 0.5 * self.g * h * h,
                hu * v,
            ],
            dtype=float,
        )


    def _flux_y(self, h: float, hu: float, hv: float) -> np.ndarray:
        """Physical y-flux for the shallow-water system."""
        if h <= self.dry_tolerance:
            return np.zeros(3, dtype=float)

        u, v = self._primitive(h, hu, hv)
        return np.array(
            [
                hv,
                hu * v,
                hv * v + 0.5 * self.g * h * h,
            ],
            dtype=float,
        )


    def _rusanov_x(self, qL: np.ndarray, qR: np.ndarray) -> np.ndarray:
        """Rusanov flux in x-direction."""
        hL, huL, hvL = qL
        hR, huR, hvR = qR

        fL = self._flux_x(hL, huL, hvL)
        fR = self._flux_x(hR, huR, hvR)

        uL, _ = self._primitive(hL, huL, hvL)
        uR, _ = self._primitive(hR, huR, hvR)

        cL = np.sqrt(self.g * max(hL, 0.0))
        cR = np.sqrt(self.g * max(hR, 0.0))
        smax = max(abs(uL) + cL, abs(uR) + cR)

        return 0.5 * (fL + fR) - 0.5 * smax * (qR - qL)


    def _rusanov_y(self, qL: np.ndarray, qR: np.ndarray) -> np.ndarray:
        """Rusanov flux in y-direction."""
        hL, huL, hvL = qL
        hR, huR, hvR = qR

        fL = self._flux_y(hL, huL, hvL)
        fR = self._flux_y(hR, huR, hvR)

        _, vL = self._primitive(hL, huL, hvL)
        _, vR = self._primitive(hR, huR, hvR)

        cL = np.sqrt(self.g * max(hL, 0.0))
        cR = np.sqrt(self.g * max(hR, 0.0))
        smax = max(abs(vL) + cL, abs(vR) + cR)

        return 0.5 * (fL + fR) - 0.5 * smax * (qR - qL)

    # source term
    def compute_source(self, h: Optional[np.ndarray] = None) -> np.ndarray:
        """ bathymetry source term
        For the standard shallow-water equations with bottom topography:
            S = [0, -g h db/dx, -g h db/dy]^T
        """
        if h is None:
            h = self.h
        else:
            h = self._check_shape(h, "h")

        db_dx, db_dy = self._bathymetry_gradients()

        zero = np.zeros_like(h)
        return np.stack([zero,
                        -self.g * h * db_dx,
                        -self.g * h * db_dy],
                        axis=0)

    """ hydrostatic reconstruction + rusanov flux for x/y-faces """
    def _hydro_face_x(self, hL: float, huL: float, hvL: float, bL: float,
                      hR: float, huR: float, hvR: float, bR: float, use_left_correction: bool) -> np.ndarray:
        z = max(bL, bR)

        hLr = max(0.0, hL + bL - z)
        hRr = max(0.0, hR + bR - z)

        uL, vL = self._primitive(hL, huL, hvL)
        uR, vR = self._primitive(hR, huR, hvR)

        qL = np.array([hLr, hLr * uL, hLr * vL], dtype=float)
        qR = np.array([hRr, hRr * uR, hRr * vR], dtype=float)

        base = self._rusanov_x(qL, qR)

        if use_left_correction:
            corr = np.array([0.0, 0.5 * self.g * (hL * hL - hLr * hLr), 0.0], dtype=float)
        else:
            corr = np.array([0.0, 0.5 * self.g * (hR * hR - hRr * hRr), 0.0], dtype=float)

        return base + corr


    def _hydro_face_y(self, hL: float, huL: float, hvL: float, bL: float,
                      hR: float, huR: float, hvR: float, bR: float, use_left_correction: bool) -> np.ndarray:

        z = max(bL, bR)

        hLr = max(0.0, hL + bL - z)
        hRr = max(0.0, hR + bR - z)

        uL, vL = self._primitive(hL, huL, hvL)
        uR, vR = self._primitive(hR, huR, hvR)

        qL = np.array([hLr, hLr * uL, hLr * vL], dtype=float)
        qR = np.array([hRr, hRr * uR, hRr * vR], dtype=float)

        base = self._rusanov_y(qL, qR)

        if use_left_correction:
            corr = np.array([0.0, 0.0, 0.5 * self.g * (hL * hL - hLr * hLr)], dtype=float)
        else:
            corr = np.array([0.0, 0.0, 0.5 * self.g * (hR * hR - hRr * hRr)], dtype=float)

        return base + corr

    # limiters / reconstruction
    @staticmethod
    def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """minmod limiter """
        same_sign = np.sign(a) == np.sign(b)
        return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def _slope(self, U: np.ndarray, axis: Literal["x", "y"]) -> np.ndarray:
        """ limited slope on a padded state array """
        slope = np.zeros_like(U)

        if axis == "x":
            forward = U[:, 2:, :] - U[:, 1:-1, :]
            backward = U[:, 1:-1, :] - U[:, :-2, :]
            slope[:, 1:-1, :] = self.minmod(forward, backward)

        elif axis == "y":
            forward = U[:, :, 2:] - U[:, :, 1:-1]
            backward = U[:, :, 1:-1] - U[:, :, :-2]
            slope[:, :, 1:-1] = self.minmod(forward, backward)

        else:
            raise ValueError("axis must be 'x' or 'y'")

        return slope
    
    # sponge
    def _init_sponge_layer(self, width: int = 20, min_factor: float = 0.90) -> None:
        """ 
        initializes a multiplicative mask to damp waves near the boundaries.
        factor = 1.0 in the center, dropping smoothly to min_factor at the edge
        """
        width = int(max(0, width))
        min_factor = float(min_factor)

        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)

        if width == 0:
            return

        # avoid overextending the sponge on very small grids
        max_width = max(1, min(self.nx, self.ny) // 2)
        width = min(width, max_width)

        for d in range(width):
            t = (width - d) / width
            val = 1.0 - (1.0 - min_factor) * (t * t)

            self.sponge_mask[d, :] = np.minimum(self.sponge_mask[d, :], val)
            self.sponge_mask[-(d + 1), :] = np.minimum(self.sponge_mask[-(d + 1), :], val)
            self.sponge_mask[:, d] = np.minimum(self.sponge_mask[:, d], val)
            self.sponge_mask[:, -(d + 1)] = np.minimum(self.sponge_mask[:, -(d + 1)], val)

    def reset_operator_diagnostics(self) -> None:
        if self.sponge_reference_dt is None:
            reference_rate_min = None
            reference_rate_max = None
        else:
            reference_rates = -np.log(self.sponge_mask) / self.sponge_reference_dt
            reference_rate_min = float(np.min(reference_rates))
            reference_rate_max = float(np.max(reference_rates))
        self.operator_diagnostics = {
            "sponge_time_mode": self.sponge_time_mode,
            "sponge_reference_dt": self.sponge_reference_dt,
            "sponge_reference_decay_rate_min": reference_rate_min,
            "sponge_reference_decay_rate_max": reference_rate_max,
            "sponge_applications": 0,
            "sponge_elapsed_time": 0.0,
            "sponge_accumulated_exponent": 0.0,
            "sponge_effective_factor_min": 1.0,
            "sponge_effective_factor_max": 1.0,
            "positivity_projection_count": 0,
            "dry_projection_count": 0,
            "nan_to_num_replacement_count": 0,
            "nan_to_num_replacement_occurred": False,
        }

    def get_operator_diagnostics(self) -> dict[str, float | int | bool | str | None]:
        return dict(self.operator_diagnostics)

    def _nan_to_num_with_diagnostics(self, values: np.ndarray) -> np.ndarray:
        replacements = int(np.count_nonzero(~np.isfinite(values)))
        self.operator_diagnostics["nan_to_num_replacement_count"] = int(
            self.operator_diagnostics["nan_to_num_replacement_count"]
        ) + replacements
        self.operator_diagnostics["nan_to_num_replacement_occurred"] = bool(
            self.operator_diagnostics["nan_to_num_replacement_occurred"]
        ) or replacements > 0
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def apply_sponge_layer(self, dt: float | None = None) -> None:
        """Gently damp momentum and wave elevation inside the sponge zone."""
        if not self.use_sponge:
            return
        if not hasattr(self, "sponge_mask"):
            return
        if dt is None:
            dt = self.dt
        factor = sponge_factor(
            self.sponge_mask,
            dt=float(dt),
            mode=self.sponge_time_mode,
            reference_dt=self.sponge_reference_dt,
        )
        self.hu *= factor
        self.hv *= factor

        h_rest = np.maximum(-self.b, 0.0)
        elevation = self.h - h_rest
        self.h = h_rest + (elevation * factor)
        exponent = (
            1.0
            if self.sponge_time_mode == "legacy_per_step"
            else float(dt) / float(self.sponge_reference_dt)
        )
        self.operator_diagnostics["sponge_applications"] = int(
            self.operator_diagnostics["sponge_applications"]
        ) + 1
        self.operator_diagnostics["sponge_elapsed_time"] = float(
            self.operator_diagnostics["sponge_elapsed_time"]
        ) + float(dt)
        self.operator_diagnostics["sponge_accumulated_exponent"] = float(
            self.operator_diagnostics["sponge_accumulated_exponent"]
        ) + exponent
        self.operator_diagnostics["sponge_effective_factor_min"] = min(
            float(self.operator_diagnostics["sponge_effective_factor_min"]),
            float(np.min(factor)),
        )
        self.operator_diagnostics["sponge_effective_factor_max"] = max(
            float(self.operator_diagnostics["sponge_effective_factor_max"]),
            float(np.max(factor)),
        )

    def _pad_state(self, U: np.ndarray) -> np.ndarray:
        """ pad a stacked state array with one ghost cell on every side """
        if U.shape != (3, self.nx, self.ny):
            raise ValueError(f"U shape must be {(3, self.nx, self.ny)}, got {U.shape}")

        return pad_state_with_reflective_momentum(U, self.boundary_x, self.boundary_y)

    def _pad_scalar(self, A: np.ndarray) -> np.ndarray:
        """ pad a scalar field with one ghost cell on every side """
        A = self._check_shape(A, "scalar field")

        return pad_scalar_field(A, self.boundary_x, self.boundary_y)

    # fluxes
    def _stabilized_conserved(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Bound momentum by local depth to prevent hu^2/h overflows in MUSCL fluxes.
        """
        h_raw = U[0]
        h = np.maximum(h_raw, 0.0)
        wet = h > self.dry_tolerance
        h_safe = np.maximum(h, self.dry_tolerance)

        hu = np.where(wet, U[1], 0.0)
        hv = np.where(wet, U[2], 0.0)

        momentum_cap = self.max_velocity * h_safe
        hu = np.clip(hu, -momentum_cap, momentum_cap)
        hv = np.clip(hv, -momentum_cap, momentum_cap)

        U_stable = np.stack([h, hu, hv], axis=0)
        return U_stable, h_safe

    def flux_x_from_U(self, U: np.ndarray) -> np.ndarray:
        U_stable, h_safe = self._stabilized_conserved(U)
        h = U_stable[0]
        hu = U_stable[1]
        hv = U_stable[2]

        return np.stack(
            [
                hu,
                hu ** 2 / h_safe + 0.5 * self.g * h ** 2,
                hu * hv / h_safe,
            ],
            axis=0,
        )


    def flux_y_from_U(self, U: np.ndarray) -> np.ndarray:
        U_stable, h_safe = self._stabilized_conserved(U)
        h = U_stable[0]
        hu = U_stable[1]
        hv = U_stable[2]

        return np.stack(
            [
                hv,
                hu * hv / h_safe,
                hv ** 2 / h_safe + 0.5 * self.g * h ** 2,
            ],
            axis=0,
        )

    def _rusanov_flux(self, U_L: np.ndarray, U_R: np.ndarray, axis: Literal["x", "y"]) -> np.ndarray:
        """ compute the rusanov flux at all interfaces """
        U_Ls, h_L = self._stabilized_conserved(U_L)
        U_Rs, h_R = self._stabilized_conserved(U_R)

        hu_L = U_Ls[1]
        hu_R = U_Rs[1]
        hv_L = U_Ls[2]
        hv_R = U_Rs[2]

        if axis == "x":
            F_L = self.flux_x_from_U(U_Ls)
            F_R = self.flux_x_from_U(U_Rs)

            u_L = hu_L / h_L
            u_R = hu_R / h_R
            c_L = np.sqrt(self.g * h_L)
            c_R = np.sqrt(self.g * h_R)

            wave_speed = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)
        
        elif axis == "y":
            F_L = self.flux_y_from_U(U_Ls)
            F_R = self.flux_y_from_U(U_Rs)

            v_L = hv_L / h_L
            v_R = hv_R / h_R
            c_L = np.sqrt(self.g * h_L)
            c_R = np.sqrt(self.g * h_R)

            wave_speed = np.maximum(np.abs(v_L) + c_L, np.abs(v_R) + c_R)
        
        else:
            raise ValueError("axis must be 'x' or 'y'")

        wave_speed = wave_speed[None, :, :]

        flux = 0.5 * (F_L + F_R) - 0.5 * wave_speed * (U_Rs - U_Ls)
        return self._nan_to_num_with_diagnostics(flux)

    def _flux_divergence_x(self, U: np.ndarray) -> np.ndarray:
        """ return dF/dx on the physical domain """
        U_pad = self._pad_state(U)
        slope_x = self._slope(U_pad, axis="x")

        U_L = U_pad[:, :-1, :] + 0.5 * slope_x[:, :-1, :]
        U_R = U_pad[:, 1:, :] - 0.5 * slope_x[:, 1:, :]

        F_half = self._rusanov_flux(U_L, U_R, axis="x")
        div = (F_half[:, 1 : self.nx + 1, 1:-1] - F_half[:, 0:self.nx, 1:-1]) / self.dx
        return self._nan_to_num_with_diagnostics(div)

    def _flux_divergence_y(self, U: np.ndarray) -> np.ndarray:
        """ return dG/dy on the physical domain """
        U_pad = self._pad_state(U)
        slope_y = self._slope(U_pad, axis="y")

        U_L = U_pad[:, :, :-1] + 0.5 * slope_y[:, :, :-1]
        U_R = U_pad[:, :, 1:] - 0.5 * slope_y[:, :, 1:]

        G_half = self._rusanov_flux(U_L, U_R, axis="y")
        div = (G_half[:, 1:-1, 1 : self.ny + 1] - G_half[:, 1:-1, 0:self.ny]) / self.dy
        return self._nan_to_num_with_diagnostics(div)

    # time step
    def _zero_momentum_in_dry_cells(self) -> None:
        dry = self.h <= self.dry_tolerance
        self.hu[dry] = 0.0
        self.hv[dry] = 0.0

    def compute_cfl(self, dt: Optional[float] = None) -> float:
        """ return the current cfl number for a candidate time step """
        if dt is None:
            dt = self.dt

        u, v = self.compute_velocity()
        wave_speed = np.sqrt(self.g * np.maximum(self.h, self.dry_tolerance))

        cfl_x = np.max((np.abs(u) + wave_speed) * dt / self.dx)
        cfl_y = np.max((np.abs(v) + wave_speed) * dt / self.dy)
        
        return float(max(cfl_x, cfl_y))

    def suggest_dt(self, target_cfl: Optional[float] = None) -> float:
        """ suggest a stable time step based on the current state """
        if target_cfl is None:
            target_cfl = self.cfl

        u, v = self.compute_velocity()
        wave_speed = np.sqrt(self.g * np.maximum(self.h, self.dry_tolerance))

        speed_x = np.max(np.abs(u) + wave_speed)
        speed_y = np.max(np.abs(v) + wave_speed)

        denom = max(speed_x / self.dx, speed_y / self.dy)
        
        if denom <= 0.0:
            return self.dt

        return float(target_cfl / denom)

    def adjust_dt(self, target_cfl: Optional[float] = None) -> float:
        """ update self.dt to a CFL-safe value and return """
        self.dt = self.suggest_dt(target_cfl=target_cfl)
        
        return self.dt

    def _boundary_state_x (self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray, 
                           j: int, side: Literal["left", "right"]) -> tuple[float, float, float, float]:
        return boundary_state_x(h, hu, hv, b, j, side, self.boundary_x)

    def _boundary_state_y(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray,
                          i: int, side: Literal["bottom", "top"]) -> tuple[float, float, float, float]:
        return boundary_state_y(h, hu, hv, b, i, side, self.boundary_y)

    def update(self, dt: float) -> None:
        """ advance the solution by one explicit finite-volume step """
        if dt <= 0:
            raise ValueError("dt must be positive")

        h_old = self.h.copy()
        hu_old = self.hu.copy()
        hv_old = self.hv.copy()
        b = self.b

        h_new = h_old.copy()
        hu_new = hu_old.copy()
        hv_new = hv_old.copy()

        for i in range(self.nx):
            for j in range(self.ny):
                hC = h_old[i, j]
                huC = hu_old[i, j]
                hvC = hv_old[i, j]
                bC = b[i, j]

                # x-direction fluxes
                if i == 0:
                    hL, huL, hvL, bL = self._boundary_state_x(h_old, hu_old, hv_old, b, j, side="left")
                else:
                    hL = h_old[i - 1, j]
                    huL = hu_old[i - 1, j]
                    hvL = hv_old[i - 1, j]
                    bL = b[i - 1, j]

                FxL = self._hydro_face_x (hL, huL, hvL, bL, hC, huC, hvC, bC, use_left_correction=False)

                if i == self.nx - 1:
                    hR, huR, hvR, bR = self._boundary_state_x(h_old, hu_old, hv_old, b, j, side="right")
                else:
                    hR = h_old[i + 1, j]
                    huR = hu_old[i + 1, j]
                    hvR = hv_old[i + 1, j]
                    bR = b[i + 1, j]

                FxR = self._hydro_face_x (hC, huC, hvC, bC, hR, huR, hvR, bR, use_left_correction=True)

                # y-direction fluxes
                if j == 0:
                    hB, huB, hvB, bB = self._boundary_state_y(h_old, hu_old, hv_old, b, i, side="bottom")
                else:
                    hB = h_old[i, j - 1]
                    huB = hu_old[i, j - 1]
                    hvB = hv_old[i, j - 1]
                    bB = b[i, j - 1]

                FyB = self._hydro_face_y (hB, huB, hvB, bB, hC, huC, hvC, bC, use_left_correction=False)

                if j == self.ny - 1:
                    hT, huT, hvT, bT = self._boundary_state_y(h_old, hu_old, hv_old, b, i, side="top")
                else:
                    hT = h_old[i, j + 1]
                    huT = hu_old[i, j + 1]
                    hvT = hv_old[i, j + 1]
                    bT = b[i, j + 1]

                FyT = self._hydro_face_y (hC, huC, hvC, bC, hT, huT, hvT, bT, use_left_correction=True)

                # finite-volume update
                h_new[i, j] = (h_old[i, j] - (dt / self.dx) * (FxR[0] - FxL[0]) - (dt / self.dy) * (FyT[0] - FyB[0]))

                hu_new[i, j] = (hu_old[i, j] - (dt / self.dx) * (FxR[1] - FxL[1]) - (dt / self.dy) * (FyT[1] - FyB[1]))

                hv_new[i, j] = (hv_old[i, j] - (dt / self.dx) * (FxR[2] - FxL[2]) - (dt / self.dy) * (FyT[2] - FyB[2]))

        # dry-cell safety
        self.operator_diagnostics["positivity_projection_count"] = int(
            self.operator_diagnostics["positivity_projection_count"]
        ) + int(np.count_nonzero(h_new < 0.0))
        h_new = np.maximum(h_new, 0.0)
        dry = h_new <= self.dry_tolerance
        self.operator_diagnostics["dry_projection_count"] = int(
            self.operator_diagnostics["dry_projection_count"]
        ) + int(np.count_nonzero(dry))
        hu_new[dry] = 0.0
        hv_new[dry] = 0.0

        self.h = h_new
        self.hu = hu_new
        self.hv = hv_new

    def apply_boundary_conditions(self) -> None:
        """
        this is kept for API compatibility

        boundary conditions are handled during update() through per-face ghost
        states. this method keeps the dry-cell cleanup contract.
        """
        self._zero_momentum_in_dry_cells()

    def step(self, dt: Optional[float] = None, auto_dt: bool = False) -> None:
        """ one simulation step """
        if auto_dt:
            dt = self.suggest_dt()
            self.dt = dt

        if dt is None:
            dt = self.dt       

        self.update(dt=dt)
        self.apply_boundary_conditions()
        self.apply_sponge_layer(dt=dt)

    def run(self, n_steps: int, record_every: int = 1, auto_dt: bool = False, return_history: bool = False) -> Optional[list[np.ndarray]]:
        """
        run the simulation for n_steps

        n_steps: number of explicit steps 
        record_every: store every k-th frame when return_history=True
        auto_dt: recompute dt from CFL before each step
        return_history: if True, return a list of snapshots as stacked arrays
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        if record_every <= 0:
            raise ValueError("record_every must be positive")

        history: list[np.ndarray] = []

        if return_history:
            history.append(self.get_state().copy())

        for step_idx in range(n_steps):
            self.step(auto_dt=auto_dt)
            
            if return_history and ((step_idx + 1) % record_every == 0):
                history.append(self.get_state().copy())

        return history if return_history else None

    # helper
    def info(self) -> SolverInfo:
        """ return desc of the solver configuration """
        return SolverInfo(nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, dt=self.dt, g=self.g, cfl=self.cfl,
                          dry_tolerance=self.dry_tolerance, boundary_x=self.boundary_x, boundary_y=self.boundary_y,)


HydrostaticShallowWaterSolver = ShallowWaterSolver

def _to_sample_array(sample_inputs: Any) -> np.ndarray:
    if hasattr(sample_inputs, "detach"):
        sample_inputs = sample_inputs.detach().cpu().numpy()

    arr = np.asarray(sample_inputs, dtype=float)
    if arr.ndim == 4:
        # [B,C,H,W] -> first sample
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"sample_inputs must have shape [C,H,W] or [B,C,H,W], got {arr.shape}")

    return arr

def simulate_rollout(sample_inputs: Any, **kwargs: Any) -> np.ndarray:
    """
    Run a shallow-water rollout from an evaluation input sample.

    Expected input channel layout (overridable via channel_map):
    - bathymetry: channel 0
    - source: channel 1
    - initial_depth: channel 2
    - initial_surface: channel 3

    Returns:
        np.ndarray with shape [T, H, W] for output_field in {eta, depth}
        or [T, 3, H, W] for output_field == state.
    """
    channels = _to_sample_array(sample_inputs)
    _, nx, ny = channels.shape

    channel_map_cfg = kwargs.get("channel_map", {})

    if not isinstance(channel_map_cfg, Mapping):
        raise ValueError("channel_map must be a mapping")

    def _idx(name: str, default: int) -> Optional[int]:
        value = channel_map_cfg.get(name, default)
        if value is None:
            return None
        idx = int(value)
        if idx < 0 or idx >= channels.shape[0]:
            return None

        return idx

    idx_bathy = _idx("bathymetry", 0)
    idx_source = _idx("source", 1)
    idx_h0 = _idx("initial_depth", 2)
    idx_eta0 = _idx("initial_surface", 3)

    bathymetry = channels[idx_bathy] if idx_bathy is not None else np.zeros((nx, ny), dtype=float)
    source_field = channels[idx_source] if idx_source is not None else np.zeros((nx, ny), dtype=float)
    sea_level_offset = float(kwargs.get("sea_level_offset", 0.0))
    source_scale = float(kwargs.get("source_scale", 1.0))

    if idx_h0 is not None:
        h0 = np.maximum(channels[idx_h0], 0.0)
    elif idx_eta0 is not None:
        h0 = np.maximum(channels[idx_eta0] - bathymetry, 0.0)
    else:
        rest_depth = np.maximum(-bathymetry + sea_level_offset, 0.0)
        h0 = np.maximum(rest_depth + source_scale * source_field, 0.0)

    solver = ShallowWaterSolver(
        nx=nx,
        ny=ny,
        dx=float(kwargs.get("dx", 1.0 / max(nx, 1))),
        dy=float(kwargs.get("dy", 1.0 / max(ny, 1))),
        dt=float(kwargs.get("dt", 1e-3)),
        g=float(kwargs.get("g", 9.81)),
        cfl=float(kwargs.get("cfl", 0.45)),
        dry_tolerance=float(kwargs.get("dry_tolerance", 1e-6)),
        boundary=kwargs.get("boundary", "open"),
        use_sponge=bool(kwargs.get("use_sponge", True)),
        sponge_width=int(kwargs.get("sponge_width", 20)),
        sponge_min_factor=float(kwargs.get("sponge_min_factor", 0.9)),
        max_velocity=float(kwargs.get("max_velocity", 50.0)),
    )

    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(h0, hu0=np.zeros_like(h0), hv0=np.zeros_like(h0))

    n_steps = int(kwargs.get("n_steps", 200))
    record_every = int(kwargs.get("record_every", 1))

    if record_every <= 0:
        raise ValueError("record_every must be positive")

    auto_dt = bool(kwargs.get("auto_dt", True))
    target_cfl = float(kwargs.get("target_cfl", solver.cfl))
    include_initial_state = bool(kwargs.get("include_initial_state", True))
    output_field = str(kwargs.get("output_field", "eta")).strip().lower()

    if output_field not in ("eta", "depth", "state"):
        raise ValueError("output_field must be one of: eta, depth, state")

    def _snapshot() -> np.ndarray:
        if output_field == "eta":
            return solver.compute_free_surface().copy()
        if output_field == "depth":
            return solver.h.copy()

        return solver.get_state().copy()

    frames: list[np.ndarray] = []
    if include_initial_state:
        frames.append(_snapshot())

    for step_idx in range(max(0, n_steps)):
        if auto_dt:
            dt = solver.suggest_dt(target_cfl=target_cfl)
        else:
            dt = solver.dt

        solver.step(dt=dt, auto_dt=False)
        if (step_idx + 1) % record_every == 0:
            frames.append(_snapshot())

    if not frames:
        frames.append(_snapshot())

    return np.stack(frames, axis=0).astype(np.float32)

class SimpleWaterSolver:
    """
    toy wave-like solver for framework testing and/or training (to compare with above more accurate FDE).

    faster and smoother -> easier to learn    
    """
    def simple_shallow_water_solver(source: np.ndarray, bathymetry: np.ndarray, steps: int = 30, dt: float = 0.15) -> np.ndarray:
        def laplacian(u: np.ndarray) -> np.ndarray:
            return (
                np.roll(u, 1, axis=-2) + np.roll(u, -1, axis=-2) +
                np.roll(u, 1, axis=-1) + np.roll(u, -1, axis=-1) - 4 * u
            )
        
        def reflective_boundary(field: np.ndarray) -> np.ndarray:
            out = field.copy()
            out[..., 0, :] = out[..., 1, :]
            out[..., -1, :] = out[..., -2, :]
            out[..., :, 0] = out[..., :, 1]
            out[..., :, -1] = out[..., :, -2]

            return out

        def sponge_mask(h: int, w: int, width: int = 4) -> np.ndarray:
            mask = np.ones((h, w), dtype=np.float32)
            
            for i in range(width):
                val = (i + 1) / (width + 1)
                mask[i, :] *= val
                mask[-i-1, :] *= val
                mask[:, i] *= val
                mask[:, -i-1] *= val
                
            return mask

        eta_prev = source.astype(np.float32)
        eta = source.astype(np.float32)
        h, w = source.shape
        damp = sponge_mask(h, w, width=max(2, h // 16))
        depth_speed = np.clip(np.abs(bathymetry) / (np.abs(bathymetry).max() + 1e-6), 0.1, 1.0)
        c2 = 0.18 + 0.82 * depth_speed

        for _ in range(steps):
            nxt = 2 * eta - eta_prev + (dt ** 2) * c2 * laplacian(eta)
            nxt = reflective_boundary(nxt[None, ...])[0]
            nxt *= damp
            eta_prev, eta = eta, nxt.astype(np.float32)

        return eta.astype(np.float32)

"""
Reference notes:

[1] LeVeque, R. J. (2002)
Finite volume methods for hyperbolic problems.
https://doi.org/10.1017/CBO9780511791253

[2] Delis A. I., Katsaounis Th. (2004)
Numerical solution of the two-dimensional shallow water equations by the application of relaxation methods.
DOI:10.1016/j.apm.2004.11.001

[3] Eleuterio F. T, & Tokareva S. A. (2026)
Rusanov-type schemes for hyperbolic equations: Wave-speed estimates, monotonicity and stability
https://doi.org/10.48550/arXiv.2412.03522
"""
