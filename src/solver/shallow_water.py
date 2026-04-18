from __future__ import annotations
from typing import Any, Literal, Mapping, NamedTuple, Optional, Tuple, Union
import numpy as np

BoundaryMode = Literal["open", "reflective", "periodic"]

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
                 use_sponge: bool = True, sponge_width = 20, sponge_min_factor: float = 0.9) -> None:
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

        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.g = float(g)
        self.cfl = float(cfl)
        self.dry_tolerance = float(dry_tolerance)

        if isinstance(boundary, tuple):
            self.boundary_x, self.boundary_y = boundary

        else:
            self.boundary_x = boundary
            self.boundary_y = boundary

        self._validate_boundary(self.boundary_x, "boundary_x")
        self._validate_boundary(self.boundary_y, "boundary_y")

        self.h = np.zeros((self.nx, self.ny), dtype=float)
        self.hu = np.zeros((self.nx, self.ny), dtype=float)
        self.hv = np.zeros((self.nx, self.ny), dtype=float)
        self.b = np.zeros((self.nx, self.ny), dtype=float)

        self.use_sponge = bool(use_sponge)
        self.sponge_width = int(sponge_width)
        self.sponge_min_factor = float(sponge_min_factor)

        self._db_dx: Optional[np.ndarray] = None
        self._db_dy: Optional[np.ndarray] = None

        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)
        if self.use_sponge:
            self._init_sponge_layer(width=self.sponge_width, min_factor=self.sponge_min_factor)


    # validation
    @staticmethod
    def _validate_boundary(mode: BoundaryMode, name: str) -> None:
        if mode not in ("open", "reflective", "periodic"):
            raise ValueError(f"{name} must be one of: open, reflective, periodic")

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

    def apply_sponge_layer(self) -> None:
        """ gently dampens momentum and wave elevation inside the sponge zone """
        if not self.use_sponge:
            return
        
        if not hasattr(self, 'sponge_mask'):
            return

        self.hu *= self.sponge_mask
        self.hv *= self.sponge_mask

        h_rest = np.maximum(-self.b, 0.0)
        elevation = self.h - h_rest
        self.h = h_rest + (elevation * self.sponge_mask)

    # boundary padding
    @staticmethod
    def _pad_mode(mode: BoundaryMode) -> str:
        return "wrap" if mode == "periodic" else "edge"

    def _pad_state(self, U: np.ndarray) -> np.ndarray:
        """ pad a stacked state array with one ghost cell on every side """
        if U.shape != (3, self.nx, self.ny):
            raise ValueError(f"U shape must be {(3, self.nx, self.ny)}, got {U.shape}")

        padded = np.pad(U,
                        pad_width=((0, 0), (1, 1), (1, 1)),
                        mode=self._pad_mode(self.boundary_x if self.boundary_x == self.boundary_y else "open"))

        # if x and y use different boundary modes -> repad from the original state
        # axis-by-axis so that the two modes are respected independently
        if self.boundary_x != self.boundary_y:
            padded = np.pad(U, pad_width=((0, 0), (1, 1), (0, 0)), mode=self._pad_mode(self.boundary_x))
            
            if self.boundary_x == "reflective":
                padded[1, 0, :] *= -1
                padded[1, -1, :] *= -1

            padded = np.pad(padded, pad_width=((0, 0), (0, 0), (1, 1)), mode=self._pad_mode(self.boundary_y))
            
            if self.boundary_y == "reflective":
                padded[2, :, 0] *= -1
                padded[2, :, -1] *= -1

            return padded

        if self.boundary_x == "reflective":
            padded[1, 0, :] *= -1
            padded[1, -1, :] *= -1
            padded[2, :, 0] *= -1
            padded[2, :, -1] *= -1

        return padded

    def _pad_scalar(self, A: np.ndarray) -> np.ndarray:
        """ pad a scalar field with one ghost cell on every side """
        A = self._check_shape(A, "scalar field")
        if self.boundary_x == self.boundary_y:
            return np.pad(A, pad_width=1, mode=self._pad_mode(self.boundary_x))

        padded = np.pad(A, pad_width=((1, 1), (0, 0)), mode=self._pad_mode(self.boundary_x))
        padded = np.pad(padded, pad_width=((0, 0), (1, 1)), mode=self._pad_mode(self.boundary_y))

        return padded

    # fluxes
    def _rusanov_flux(self, U_L: np.ndarray, U_R: np.ndarray, axis: Literal["x", "y"]) -> np.ndarray:
        """ compute the rusanov flux at all interfaces """
        h_L = np.maximum(U_L[0], self.dry_tolerance)
        h_R = np.maximum(U_R[0], self.dry_tolerance)

        hu_L = U_L[1]
        hu_R = U_R[1]
        hv_L = U_L[2]
        hv_R = U_R[2]

        if axis == "x":
            F_L = self.flux_x_from_U(U_L)
            F_R = self.flux_x_from_U(U_R)

            u_L = hu_L / h_L
            u_R = hu_R / h_R
            c_L = np.sqrt(self.g * h_L)
            c_R = np.sqrt(self.g * h_R)

            wave_speed = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)
        
        elif axis == "y":
            F_L = self.flux_y_from_U(U_L)
            F_R = self.flux_y_from_U(U_R)

            v_L = hv_L / h_L
            v_R = hv_R / h_R
            c_L = np.sqrt(self.g * h_L)
            c_R = np.sqrt(self.g * h_R)

            wave_speed = np.maximum(np.abs(v_L) + c_L, np.abs(v_R) + c_R)
        
        else:
            raise ValueError("axis must be 'x' or 'y'")

        wave_speed = wave_speed[None, :, :]
        
        return 0.5 * (F_L + F_R) - 0.5 * wave_speed * (U_R - U_L)

    def _flux_divergence_x(self, U: np.ndarray) -> np.ndarray:
        """ return dF/dx on the physical domain """
        U_pad = self._pad_state(U)
        slope_x = self._slope(U_pad, axis="x")

        U_L = U_pad[:, :-1, :] + 0.5 * slope_x[:, :-1, :]
        U_R = U_pad[:, 1:, :] - 0.5 * slope_x[:, 1:, :]

        F_half = self._rusanov_flux(U_L, U_R, axis="x")
        
        return (F_half[:, 1 : self.nx + 1, 1:-1] - F_half[:, 0:self.nx, 1:-1]) / self.dx

    def _flux_divergence_y(self, U: np.ndarray) -> np.ndarray:
        """ return dG/dy on the physical domain """
        U_pad = self._pad_state(U)
        slope_y = self._slope(U_pad, axis="y")

        U_L = U_pad[:, :, :-1] + 0.5 * slope_y[:, :, :-1]
        U_R = U_pad[:, :, 1:] - 0.5 * slope_y[:, :, 1:]

        G_half = self._rusanov_flux(U_L, U_R, axis="y")
        
        return (G_half[:, 1:-1, 1 : self.ny + 1] - G_half[:, 1:-1, 0:self.ny]) / self.dy

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
        if side == "left":
            inside_i = 0
            periodic_i = self.nx - 1
        elif side == "right":
            inside_i = self.nx - 1
            periodic_i = 0
        else:
            raise ValueError("side must be 'left' or 'right'")

        if self.boundary_x == "periodic":
            return h[periodic_i, j], hu[periodic_i, j], hv[periodic_i, j], b[periodic_i, j]

        h_g = h[inside_i, j]
        hu_g = hu[inside_i, j]
        hv_g = hv[inside_i, j]
        b_g = b[inside_i, j]

        if self.boundary_x == "reflective":
            hu_g = -hu_g

        return h_g, hu_g, hv_g, b_g

    def _boundary_state_y(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray,
                          i: int, side: Literal["bottom", "top"]) -> tuple[float, float, float, float]:

        if side == "bottom":
            inside_j = 0
            periodic_j = self.ny - 1
        elif side == "top":
            inside_j = self.ny - 1
            periodic_j = 0
        else:
            raise ValueError("side must be 'bottom' or 'top'")

        if self.boundary_y == "periodic":
            return h[i, periodic_j], hu[i, periodic_j], hv[i, periodic_j], b[i, periodic_j]

        h_g = h[i, inside_j]
        hu_g = hu[i, inside_j]
        hv_g = hv[i, inside_j]
        b_g = b[i, inside_j]

        if self.boundary_y == "reflective":
            hv_g = -hv_g

        return h_g, hu_g, hv_g, b_g

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
        h_new = np.maximum(h_new, 0.0)
        dry = h_new <= self.dry_tolerance
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
        self.apply_sponge_layer()

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

"""
Reference notes:

[1] LeVeque, R. J. (2002)
Finite volume methods for hyperbolic problems.

[2] Delis A. I., Katsaounis T. D., and Mitsotakis D. (2005)
Numerical solution of the two-dimensional shallow water equations by the application of relaxation methods.

[3] Eleuterio F. T, & Tokareva S. A. (2026)
Rusanov-type schemes for hyperbolic equations: Wave-speed estimates, monotonicity and stability
"""
