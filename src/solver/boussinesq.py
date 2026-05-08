from __future__ import annotations

from typing import Literal, NamedTuple, Optional, Tuple, Union

import numpy as np

BoundaryMode = Literal["open", "reflective", "periodic"]
BoussinesqMode = Literal[
    "linear_constant_depth",
    "linear_variable_depth",
    "weakly_nonlinear_planned",
]


class BoussinesqInfo(NamedTuple):
    """Metadata bundle for logging / experiment tracking."""

    nx: int
    ny: int
    dx: float
    dy: float
    dt: float
    g: float
    cfl: float
    alpha: float
    min_depth: float
    sea_level_offset: float
    boundary_x: BoundaryMode
    boundary_y: BoundaryMode
    mode: BoussinesqMode


class BoussinesqSolver:
    """
    Elevation-only weakly dispersive Boussinesq-type solver.

    First-version model:
        (I - alpha * H^2 * Laplacian) eta_tt = g * div(H * grad(eta))

    State:
        eta   : free-surface elevation above still water level, shape [nx, ny]
        eta_t : time derivative of eta, shape [nx, ny]
        b     : bathymetry/topography. Same convention as shallow_water.py:
                b < 0 underwater, H = max(-b + sea_level_offset, min_depth)

    This solver is intended as a comparison/reference option, not as the
    primary wet/dry inundation solver.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dt: float,
        g: float = 9.81,
        cfl: float = 0.35,
        alpha: float = 1.0 / 3.0,
        min_depth: float = 1e-3,
        sea_level_offset: float = 0.0,
        boundary: Union[BoundaryMode, Tuple[BoundaryMode, BoundaryMode]] = "open",
        mode: BoussinesqMode = "linear_variable_depth",
        use_sponge: bool = True,
        sponge_width: int = 20,
        sponge_min_factor: float = 0.9,
        filter_strength: float = 0.0,
        linear_solver_tol: float = 1e-8,
        linear_solver_max_iter: int = 80,
    ) -> None:
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
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        if sponge_width < 0:
            raise ValueError("sponge_width must be non-negative")
        if not (0.0 < sponge_min_factor <= 1.0):
            raise ValueError("sponge_min_factor must be in (0, 1]")
        if filter_strength < 0:
            raise ValueError("filter_strength must be non-negative")
        if linear_solver_tol <= 0:
            raise ValueError("linear_solver_tol must be positive")
        if linear_solver_max_iter <= 0:
            raise ValueError("linear_solver_max_iter must be positive")

        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.g = float(g)
        self.cfl = float(cfl)
        self.alpha = float(alpha)
        self.min_depth = float(min_depth)
        self.sea_level_offset = float(sea_level_offset)
        self.mode = mode
        self.filter_strength = float(filter_strength)
        self.linear_solver_tol = float(linear_solver_tol)
        self.linear_solver_max_iter = int(linear_solver_max_iter)

        if isinstance(boundary, tuple):
            self.boundary_x, self.boundary_y = boundary
        else:
            self.boundary_x = boundary
            self.boundary_y = boundary

        self._validate_boundary(self.boundary_x, "boundary_x")
        self._validate_boundary(self.boundary_y, "boundary_y")
        self._validate_mode(self.mode)

        self.eta = np.zeros((self.nx, self.ny), dtype=float)
        self.eta_t = np.zeros((self.nx, self.ny), dtype=float)
        self.b = np.zeros((self.nx, self.ny), dtype=float)
        self.H = np.ones((self.nx, self.ny), dtype=float)

        self.use_sponge = bool(use_sponge)
        self.sponge_width = int(sponge_width)
        self.sponge_min_factor = float(sponge_min_factor)
        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)
        if self.use_sponge:
            self._init_sponge_layer(width=self.sponge_width, min_factor=self.sponge_min_factor)

    @staticmethod
    def _validate_boundary(mode: BoundaryMode, name: str) -> None:
        if mode not in ("open", "reflective", "periodic"):
            raise ValueError(f"{name} must be one of: open, reflective, periodic")

    @staticmethod
    def _validate_mode(mode: BoussinesqMode) -> None:
        if mode not in (
            "linear_constant_depth",
            "linear_variable_depth",
            "weakly_nonlinear_planned",
        ):
            raise ValueError(
                "mode must be one of: linear_constant_depth, "
                "linear_variable_depth, weakly_nonlinear_planned"
            )
        if mode == "weakly_nonlinear_planned":
            raise NotImplementedError("weakly_nonlinear_planned is reserved for future work")

    def _check_shape(self, arr: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (self.nx, self.ny):
            raise ValueError(f"{name} shape must be {(self.nx, self.ny)}, got {arr.shape}")
        return arr

    def set_bathymetry(self, b: np.ndarray) -> None:
        """
        Store bathymetry and precompute still-water depth H.

        Sign convention inherited from shallow_water.py:
            b < 0 underwater
            H = max(-b + sea_level_offset, min_depth)
        """
        self.b = self._check_shape(b, "b").copy()
        H = np.maximum(-self.b + self.sea_level_offset, self.min_depth)

        if self.mode == "linear_constant_depth":
            H0 = float(np.mean(H))
            self.H = np.full_like(H, max(H0, self.min_depth))
        else:
            self.H = H

    def set_initial_condition(
        self,
        eta0: np.ndarray,
        eta_t0: Optional[np.ndarray] = None,
    ) -> None:
        """Set initial free-surface elevation and optional eta_t."""
        self.eta = self._check_shape(eta0, "eta0").copy()
        self.eta_t = np.zeros_like(self.eta)
        if eta_t0 is not None:
            self.eta_t = self._check_shape(eta_t0, "eta_t0").copy()

    def get_state(self) -> np.ndarray:
        """Return stacked Boussinesq state as [eta, eta_t]."""
        return np.stack([self.eta, self.eta_t], axis=0)

    def set_state(self, U: np.ndarray) -> None:
        """Replace state from a [2, nx, ny] stacked array."""
        U = np.asarray(U, dtype=float)
        if U.shape != (2, self.nx, self.ny):
            raise ValueError(f"U shape must be {(2, self.nx, self.ny)}, got {U.shape}")
        self.eta = U[0].copy()
        self.eta_t = U[1].copy()

    def compute_free_surface(self) -> np.ndarray:
        """Compatibility helper: Boussinesq state already stores eta."""
        return self.eta.copy()

    @staticmethod
    def _pad_mode(mode: BoundaryMode) -> str:
        return "wrap" if mode == "periodic" else "edge"

    def _pad_scalar(self, A: np.ndarray) -> np.ndarray:
        A = self._check_shape(A, "scalar field")

        if self.boundary_x == self.boundary_y:
            return np.pad(A, pad_width=1, mode=self._pad_mode(self.boundary_x))

        padded = np.pad(A, pad_width=((1, 1), (0, 0)), mode=self._pad_mode(self.boundary_x))
        padded = np.pad(padded, pad_width=((0, 0), (1, 1)), mode=self._pad_mode(self.boundary_y))
        return padded

    def gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centered finite-difference gradient with configured scalar boundaries."""
        A = self._pad_scalar(field)
        d_dx = (A[2:, 1:-1] - A[:-2, 1:-1]) / (2.0 * self.dx)
        d_dy = (A[1:-1, 2:] - A[1:-1, :-2]) / (2.0 * self.dy)
        return d_dx, d_dy

    def divergence(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Centered finite-difference divergence of a vector field."""
        fx_pad = self._pad_scalar(fx)
        fy_pad = self._pad_scalar(fy)
        dfx_dx = (fx_pad[2:, 1:-1] - fx_pad[:-2, 1:-1]) / (2.0 * self.dx)
        dfy_dy = (fy_pad[1:-1, 2:] - fy_pad[1:-1, :-2]) / (2.0 * self.dy)
        return dfx_dx + dfy_dy

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Second-order finite-difference Laplacian."""
        A = self._pad_scalar(field)
        center = A[1:-1, 1:-1]
        d2x = (A[2:, 1:-1] - 2.0 * center + A[:-2, 1:-1]) / (self.dx * self.dx)
        d2y = (A[1:-1, 2:] - 2.0 * center + A[1:-1, :-2]) / (self.dy * self.dy)
        return d2x + d2y

    def rhs(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute g * div(H * grad(eta))."""
        if eta is None:
            eta = self.eta
        else:
            eta = self._check_shape(eta, "eta")

        eta_x, eta_y = self.gradient(eta)
        return self.g * self.divergence(self.H * eta_x, self.H * eta_y)

    def apply_mass_operator(self, a: np.ndarray) -> np.ndarray:
        """Apply M(a) = a - alpha * div(H^2 * grad(a))."""
        a = self._check_shape(a, "a")
        if self.alpha == 0.0:
            return a.copy()
        a_x, a_y = self.gradient(a)
        h2 = self.H * self.H
        dispersive = self.divergence(h2 * a_x, h2 * a_y)
        return a - self.alpha * dispersive

    def solve_acceleration(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve M(a) = rhs(eta) using a small matrix-free conjugate-gradient loop."""
        b = self.rhs(eta)

        if self.alpha == 0.0:
            return b

        x = np.zeros_like(b)
        r = b - self.apply_mass_operator(x)
        p = r.copy()
        rs_old = float(np.sum(r * r))

        if rs_old <= self.linear_solver_tol * self.linear_solver_tol:
            return x

        eps = 1e-30
        for _ in range(self.linear_solver_max_iter):
            Ap = self.apply_mass_operator(p)
            denom = float(np.sum(p * Ap))
            if abs(denom) <= eps:
                break

            alpha_cg = rs_old / denom
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap

            rs_new = float(np.sum(r * r))
            if rs_new <= self.linear_solver_tol * self.linear_solver_tol:
                break

            beta = rs_new / max(rs_old, eps)
            p = r + beta * p
            rs_old = rs_new

        return x

    def _init_sponge_layer(self, width: int = 20, min_factor: float = 0.9) -> None:
        width = int(max(0, width))
        min_factor = float(min_factor)
        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)

        if width == 0:
            return

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
        """Damp eta and eta_t near boundaries."""
        if not self.use_sponge:
            return
        self.eta *= self.sponge_mask
        self.eta_t *= self.sponge_mask

    def apply_filter(self) -> None:
        """Optional light Laplacian smoothing for high-frequency cleanup."""
        if self.filter_strength <= 0.0:
            return

        strength = min(self.filter_strength, 0.25)
        self.eta = self.eta + strength * min(self.dx, self.dy) ** 2 * self.laplacian(self.eta)
        self.eta_t = self.eta_t + strength * min(self.dx, self.dy) ** 2 * self.laplacian(self.eta_t)

    def suggest_dt(self, target_cfl: Optional[float] = None) -> float:
        """Suggest a conservative explicit step based on shallow long-wave speed."""
        if target_cfl is None:
            target_cfl = self.cfl

        cmax = float(np.sqrt(self.g * np.max(self.H)))
        denom = cmax * (1.0 / self.dx + 1.0 / self.dy)
        if denom <= 0.0:
            return self.dt
        return float(target_cfl / denom)

    def compute_cfl(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.dt
        cmax = float(np.sqrt(self.g * np.max(self.H)))
        return float(dt * cmax * (1.0 / self.dx + 1.0 / self.dy))

    def adjust_dt(self, target_cfl: Optional[float] = None) -> float:
        self.dt = self.suggest_dt(target_cfl=target_cfl)
        return self.dt

    def step(self, dt: Optional[float] = None, auto_dt: bool = False) -> None:
        """Advance one velocity-Verlet step."""
        if auto_dt:
            dt = self.suggest_dt()
            self.dt = dt

        if dt is None:
            dt = self.dt
        if dt <= 0:
            raise ValueError("dt must be positive")

        a0 = self.solve_acceleration(self.eta)
        eta_next = self.eta + dt * self.eta_t + 0.5 * dt * dt * a0
        a1 = self.solve_acceleration(eta_next)
        eta_t_next = self.eta_t + 0.5 * dt * (a0 + a1)

        self.eta = eta_next
        self.eta_t = eta_t_next
        self.apply_filter()
        self.apply_sponge_layer()

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        auto_dt: bool = False,
        return_history: bool = False,
    ) -> Optional[list[np.ndarray]]:
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

    def info(self) -> BoussinesqInfo:
        return BoussinesqInfo(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy,
            dt=self.dt,
            g=self.g,
            cfl=self.cfl,
            alpha=self.alpha,
            min_depth=self.min_depth,
            sea_level_offset=self.sea_level_offset,
            boundary_x=self.boundary_x,
            boundary_y=self.boundary_y,
            mode=self.mode,
        )
