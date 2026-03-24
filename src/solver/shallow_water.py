from __future__ import annotations
from typing import Literal, NamedTuple, Optional, Tuple, Union
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
                 dry_tolerance: float = 1e-6, boundary: Union[BoundaryMode, Tuple[BoundaryMode, BoundaryMode]] = "open") -> None:
        if nx <= 1 or ny <= 1:
            raise ValueError("nx and ny must be greater than 1.")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if g <= 0:
            raise ValueError("g must be positive.")
        if cfl <= 0:
            raise ValueError("cfl must be positive.")
        if dry_tolerance <= 0:
            raise ValueError("dry_tolerance must be positive.")

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

        self._init_sponge_layer(width = 20, min_factor = 0.9)

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
    def flux_x_from_U(self, U: np.ndarray) -> np.ndarray:
        """ physical x-flux evaluated from a stacked state array """
        h = np.maximum(U[0], self.dry_tolerance)
        hu = U[1]
        hv = U[2]

        return np.stack([hu,
                        hu * hu / h + 0.5 * self.g * h * h,
                        hu * hv / h],
                        axis=0)

    def flux_y_from_U(self, U: np.ndarray) -> np.ndarray:
        """ physical y-flux evaluated from a stacked state array """
        h = np.maximum(U[0], self.dry_tolerance)
        hu = U[1]
        hv = U[2]

        return np.stack([hv,
                        hu * hv / h,
                        hv * hv / h + 0.5 * self.g * h * h],
                        axis=0)

    def compute_flux_x(self) -> np.ndarray:
        """ cell-centered x-flux of the current state """
        return self.flux_x_from_U(self.get_state())

    def compute_flux_y(self) -> np.ndarray:
        """ cell-centered y-flux of the current state """
        return self.flux_y_from_U(self.get_state())

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

        if self.nx >= 3 and self.ny >= 3:
            db_dx, db_dy = np.gradient(self.b, self.dx, self.dy, edge_order=2)

        else:
            db_dx, db_dy = np.gradient(self.b, self.dx, self.dy)

        zero = np.zeros_like(h)
        return np.stack([zero,
                        -self.g * h * db_dx,
                        -self.g * h * db_dy],
                        axis=0)

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
        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)
        
        for d in range(width):
            decay = (1.0 - min_factor) * ((width - d) / width)**2
            val = 1.0 - decay
            
            self.sponge_mask[d, :] = np.minimum(self.sponge_mask[d, :], val)
            self.sponge_mask[-(d+1), :] = np.minimum(self.sponge_mask[-(d+1), :], val)
            self.sponge_mask[:, d] = np.minimum(self.sponge_mask[:, d], val)
            self.sponge_mask[:, -(d+1)] = np.minimum(self.sponge_mask[:, -(d+1)], val)

    def apply_sponge_layer(self) -> None:
        """ gently dampens momentum and wave elevation inside the sponge zone """
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

    def update(self, dt: Optional[float] = None) -> None:
        """ advance the solution by one explicit finite-volume step """
        if dt is None:
            dt = self.dt
        
        if dt <= 0:
            raise ValueError("dt must be positive")

        U = self.get_state()
        divergence = self._flux_divergence_x(U) + self._flux_divergence_y(U)
        source = self.compute_source(self.h)

        U_new = U - dt * divergence + dt * source

        self.h = np.maximum(U_new[0], 0.0)
        self.hu = U_new[1]
        self.hv = U_new[2]
        self._zero_momentum_in_dry_cells()

    def apply_boundary_conditions(self) -> None:
        """
        this is kept for API compatibility

        the actual numerical boundaries are enforced through ghost-cell padding
        inside the flux computation, so this method only ensures dry-cell cleanup
        """
        self._zero_momentum_in_dry_cells()

    def step(self, dt: Optional[float] = None, auto_dt: bool = False) -> None:
        """ one simulation step """
        if auto_dt:
            dt = self.suggest_dt()

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

        history = []

        for step_idx in range(n_steps):
            self.step(auto_dt=auto_dt)
            
            if return_history and (step_idx % record_every == 0):
                history.append(self.get_state().copy())

        return history if return_history else None

    # helper
    def info(self) -> SolverInfo:
        """ return desc of the solver configuration """
        return SolverInfo(nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, dt=self.dt, g=self.g, cfl=self.cfl,
                          dry_tolerance=self.dry_tolerance, boundary_x=self.boundary_x, boundary_y=self.boundary_y,)

"""
Reference notes:
- Shallow-water equations: conservative 2D form with bathymetry source term.
- MUSCL reconstruction with minmod limiter.
- Rusanov flux for robust shock-capturing.
"""
