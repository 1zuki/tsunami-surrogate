from __future__ import annotations

from typing import Any, Literal, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np

from src.solver.boundary_conditions import (
    BoundaryMode,
    pad_scalar_field,
    resolve_boundary_modes,
)

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
    depth_scale: float
    boundary_x: BoundaryMode
    boundary_y: BoundaryMode
    mode: BoussinesqMode
    check_finite: bool


class BoussinesqSolver:
    """
    Elevation-only weakly dispersive Boussinesq-type solver.

    First-version model:
        (I - alpha * div(H^2 * grad)) eta_tt = g * div(H * grad(eta))

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
        depth_scale: float = 1.0,
        boundary: Union[BoundaryMode, Tuple[BoundaryMode, BoundaryMode]] = "open",
        mode: BoussinesqMode = "linear_variable_depth",
        use_sponge: Optional[bool] = None,
        sponge_width: int = 20,
        sponge_min_factor: float = 0.9,
        filter_strength: float = 0.0,
        linear_solver_tol: float = 1e-8,
        linear_solver_max_iter: int = 80,
        check_finite: bool = True,
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
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
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
        self.depth_scale = float(depth_scale)
        self.mode = mode
        self.filter_strength = float(filter_strength)
        self.linear_solver_tol = float(linear_solver_tol)
        self.linear_solver_max_iter = int(linear_solver_max_iter)
        self.check_finite = bool(check_finite)

        self.boundary_x, self.boundary_y = resolve_boundary_modes(boundary)
        self._validate_mode(self.mode)

        self.eta = np.zeros((self.nx, self.ny), dtype=float)
        self.eta_t = np.zeros((self.nx, self.ny), dtype=float)
        self.b = np.zeros((self.nx, self.ny), dtype=float)
        self.H = np.ones((self.nx, self.ny), dtype=float)
        self._mass_diag_inv = np.ones((self.nx, self.ny), dtype=float)

        if use_sponge is None:
            self.use_sponge = "periodic" not in (self.boundary_x, self.boundary_y)
        else:
            self.use_sponge = bool(use_sponge)
        self.sponge_width = int(sponge_width)
        self.sponge_min_factor = float(sponge_min_factor)
        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)
        self.last_cg_iterations = 0
        self.last_cg_initial_residual = 0.0
        self.last_cg_final_residual = 0.0
        self.last_cg_converged = True
        self.last_step_cg_converged = True
        self.last_step_cg_failed_count = 0
        self.last_step_cg_max_iterations = 0
        self.last_step_cg_max_residual_ratio = 0.0
        if self.use_sponge:
            self._init_sponge_layer(width=self.sponge_width, min_factor=self.sponge_min_factor)

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
        H = np.maximum((-self.b + self.sea_level_offset) * self.depth_scale, self.min_depth)

        if self.mode == "linear_constant_depth":
            H0 = float(np.mean(H))
            self.H = np.full_like(H, max(H0, self.min_depth))
        else:
            self.H = H
        self._update_mass_preconditioner()

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

    def _pad_scalar(self, A: np.ndarray) -> np.ndarray:
        A = self._check_shape(A, "scalar field")

        return pad_scalar_field(A, self.boundary_x, self.boundary_y)

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

    def _flux_divergence(self, field: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
        """Compute div(coefficient * grad(field)) from face-centered fluxes."""
        field = self._check_shape(field, "field")
        coefficient = self._check_shape(coefficient, "coefficient")

        A = self._pad_scalar(field)
        C = self._pad_scalar(coefficient)

        coeff_x = 0.5 * (C[1:, 1:-1] + C[:-1, 1:-1])
        q_x = coeff_x * (A[1:, 1:-1] - A[:-1, 1:-1]) / self.dx

        coeff_y = 0.5 * (C[1:-1, 1:] + C[1:-1, :-1])
        q_y = coeff_y * (A[1:-1, 1:] - A[1:-1, :-1]) / self.dy

        return (q_x[1:, :] - q_x[:-1, :]) / self.dx + (q_y[:, 1:] - q_y[:, :-1]) / self.dy

    def rhs(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute g * div(H * grad(eta))."""
        if eta is None:
            eta = self.eta
        else:
            eta = self._check_shape(eta, "eta")

        return self.g * self._flux_divergence(eta, self.H)

    def apply_mass_operator(self, a: np.ndarray) -> np.ndarray:
        """Apply M(a) = a - alpha * div(H^2 * grad(a))."""
        a = self._check_shape(a, "a")
        if self.alpha == 0.0:
            return a.copy()
        dispersive = self._flux_divergence(a, self.H * self.H)
        return a - self.alpha * dispersive

    def _update_mass_preconditioner(self) -> None:
        """Build a diagonal Jacobi preconditioner for the dispersive mass solve."""
        if self.alpha == 0.0:
            self._mass_diag_inv = np.ones_like(self.H)
            return

        H2 = self.H * self.H
        C = self._pad_scalar(H2)
        coeff_x = 0.5 * (C[1:, 1:-1] + C[:-1, 1:-1])
        coeff_y = 0.5 * (C[1:-1, 1:] + C[1:-1, :-1])
        diag = np.ones_like(self.H)
        diag += self.alpha * (
            (coeff_x[1:, :] + coeff_x[:-1, :]) / (self.dx * self.dx)
            + (coeff_y[:, 1:] + coeff_y[:, :-1]) / (self.dy * self.dy)
        )
        self._mass_diag_inv = 1.0 / np.maximum(diag, 1e-30)

    def solve_acceleration(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve M(a) = rhs(eta) using a small matrix-free conjugate-gradient loop."""
        b = self.rhs(eta)

        if self.alpha == 0.0:
            self.last_cg_iterations = 0
            self.last_cg_initial_residual = 0.0
            self.last_cg_final_residual = 0.0
            self.last_cg_converged = True
            return b

        x = np.zeros_like(b)
        r = b - self.apply_mass_operator(x)
        z = self._mass_diag_inv * r
        p = z.copy()
        rs_old = float(np.sum(r * r))
        rz_old = float(np.sum(r * z))
        rs0 = rs_old
        residual0 = float(np.sqrt(rs0))
        threshold = (self.linear_solver_tol * residual0) ** 2
        self.last_cg_iterations = 0
        self.last_cg_initial_residual = residual0
        self.last_cg_final_residual = residual0
        self.last_cg_converged = rs0 == 0.0

        if rs0 == 0.0:
            return x

        eps = 1e-30
        for iteration in range(1, self.linear_solver_max_iter + 1):
            Ap = self.apply_mass_operator(p)
            denom = float(np.sum(p * Ap))
            if abs(denom) <= eps or abs(rz_old) <= eps:
                self.last_cg_iterations = iteration - 1
                self.last_cg_final_residual = float(np.sqrt(rs_old))
                self.last_cg_converged = False
                break

            alpha_cg = rz_old / denom
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap

            rs_new = float(np.sum(r * r))
            self.last_cg_iterations = iteration
            self.last_cg_final_residual = float(np.sqrt(rs_new))
            if rs_new <= threshold:
                self.last_cg_converged = True
                break

            z = self._mass_diag_inv * r
            rz_new = float(np.sum(r * z))
            beta = rz_new / max(rz_old, eps)
            p = z + beta * p
            rs_old = rs_new
            rz_old = rz_new
        else:
            self.last_cg_converged = False

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
        cg0_converged = bool(self.last_cg_converged)
        cg0_iterations = int(self.last_cg_iterations)
        cg0_ratio = self.last_cg_final_residual / max(self.last_cg_initial_residual, 1e-30)
        eta_next = self.eta + dt * self.eta_t + 0.5 * dt * dt * a0
        a1 = self.solve_acceleration(eta_next)
        cg1_converged = bool(self.last_cg_converged)
        cg1_iterations = int(self.last_cg_iterations)
        cg1_ratio = self.last_cg_final_residual / max(self.last_cg_initial_residual, 1e-30)
        eta_t_next = self.eta_t + 0.5 * dt * (a0 + a1)

        self.last_step_cg_converged = cg0_converged and cg1_converged
        self.last_step_cg_failed_count = int(not cg0_converged) + int(not cg1_converged)
        self.last_step_cg_max_iterations = max(cg0_iterations, cg1_iterations)
        self.last_step_cg_max_residual_ratio = float(max(cg0_ratio, cg1_ratio))

        self.eta = eta_next
        self.eta_t = eta_t_next
        self.apply_filter()
        self.apply_sponge_layer()

        if self.check_finite:
            state = self.get_state()
            if not np.isfinite(state).all():
                raise FloatingPointError(
                    "Non-finite Boussinesq state detected after step. "
                    f"eta_range=({float(np.nanmin(self.eta)):.6e}, {float(np.nanmax(self.eta)):.6e}), "
                    f"eta_t_range=({float(np.nanmin(self.eta_t)):.6e}, {float(np.nanmax(self.eta_t)):.6e}), "
                    f"dt={float(dt):.6e}, alpha={self.alpha:.6e}, depth_scale={self.depth_scale:.6e}, "
                    f"mode={self.mode}"
                )

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
            depth_scale=self.depth_scale,
            boundary_x=self.boundary_x,
            boundary_y=self.boundary_y,
            mode=self.mode,
            check_finite=self.check_finite,
        )



def _to_sample_array(sample_inputs: Any) -> np.ndarray:
    if hasattr(sample_inputs, "detach"):
        sample_inputs = sample_inputs.detach().cpu().numpy()

    arr = np.asarray(sample_inputs, dtype=float)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"sample_inputs must have shape [C,H,W] or [B,C,H,W], got {arr.shape}")

    return arr


def simulate_rollout(sample_inputs: Any, **kwargs: Any) -> np.ndarray:
    """
    Run a Boussinesq rollout from an evaluation input sample.

    Expected input channel layout (overridable via channel_map):
    - bathymetry: channel 0
    - source: channel 1
    - initial_depth: channel 2
    - initial_surface: optional channel, default None
    - initial_surface_t: optional channel, default None

    For Boussinesq, eta0 defaults to source-based initialization
    (eta0 = source_scale * source_field) unless overridden.

    Priority:
    1) eta0 keyword override
    2) initial_surface channel (absolute surface -> eta via sea_level_offset)
    3) source channel
    4) initial_depth channel (depth -> eta via bathymetry and sea_level_offset)
    5) zeros
    """
    channels = _to_sample_array(sample_inputs)
    _, nx, ny = channels.shape

    channel_map_cfg = kwargs.get("channel_map", {})
    if not isinstance(channel_map_cfg, Mapping):
        raise ValueError("channel_map must be a mapping")

    def _idx(name: str, default: Optional[int]) -> Optional[int]:
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
    idx_eta0 = _idx("initial_surface", None)
    idx_eta_t0 = _idx("initial_surface_t", None)

    sea_level_offset = float(kwargs.get("sea_level_offset", 0.0))
    default_depth = float(kwargs.get("default_depth", 1.0))
    source_scale = float(kwargs.get("source_scale", 1.0))

    if idx_bathy is not None:
        bathymetry = channels[idx_bathy]
    else:
        bathymetry = -max(default_depth, 0.0) * np.ones((nx, ny), dtype=float)

    source_field = channels[idx_source] if idx_source is not None else None

    eta0_override = kwargs.get("eta0", None)
    if eta0_override is not None:
        eta0 = np.asarray(eta0_override, dtype=float)
        if eta0.shape != (nx, ny):
            raise ValueError(f"eta0 shape must be {(nx, ny)}, got {eta0.shape}")
    elif idx_eta0 is not None:
        # Convert absolute free surface to disturbance above still-water level.
        eta0 = channels[idx_eta0] - sea_level_offset
    elif source_field is not None:
        eta0 = source_scale * source_field
    elif idx_h0 is not None:
        # Convert depth to disturbance: eta = h + b - sea_level_offset.
        eta0 = channels[idx_h0] + bathymetry - sea_level_offset
    else:
        eta0 = np.zeros((nx, ny), dtype=float)

    eta_t0_override = kwargs.get("eta_t0", None)
    if eta_t0_override is not None:
        eta_t0 = np.asarray(eta_t0_override, dtype=float)
        if eta_t0.shape != (nx, ny):
            raise ValueError(f"eta_t0 shape must be {(nx, ny)}, got {eta_t0.shape}")
    elif idx_eta_t0 is not None:
        eta_t0 = channels[idx_eta_t0].copy()
    else:
        eta_t0 = np.zeros_like(eta0)

    use_sponge = kwargs.get("use_sponge", None)
    solver = BoussinesqSolver(
        nx=nx,
        ny=ny,
        dx=float(kwargs.get("dx", 1.0 / max(nx, 1))),
        dy=float(kwargs.get("dy", 1.0 / max(ny, 1))),
        dt=float(kwargs.get("dt", 1e-3)),
        g=float(kwargs.get("g", 9.81)),
        cfl=float(kwargs.get("cfl", 0.35)),
        alpha=float(kwargs.get("alpha", 1.0 / 3.0)),
        min_depth=float(kwargs.get("min_depth", 1e-3)),
        sea_level_offset=sea_level_offset,
        depth_scale=float(kwargs.get("depth_scale", 1.0)),
        boundary=kwargs.get("boundary", "open"),
        mode=kwargs.get("mode", "linear_variable_depth"),
        use_sponge=bool(use_sponge) if use_sponge is not None else None,
        sponge_width=int(kwargs.get("sponge_width", 20)),
        sponge_min_factor=float(kwargs.get("sponge_min_factor", 0.9)),
        filter_strength=float(kwargs.get("filter_strength", 0.0)),
        linear_solver_tol=float(kwargs.get("linear_solver_tol", 1e-8)),
        linear_solver_max_iter=int(kwargs.get("linear_solver_max_iter", 80)),
        check_finite=bool(kwargs.get("check_finite", True)),
    )
    solver.set_bathymetry(bathymetry)
    solver.set_initial_condition(eta0, eta_t0=eta_t0)

    n_steps = int(kwargs.get("n_steps", 200))
    record_every = int(kwargs.get("record_every", 1))
    if record_every <= 0:
        raise ValueError("record_every must be positive")

    auto_dt = bool(kwargs.get("auto_dt", True))
    target_cfl = float(kwargs.get("target_cfl", solver.cfl))
    include_initial_state = bool(kwargs.get("include_initial_state", True))
    output_field = str(kwargs.get("output_field", "eta")).strip().lower()

    if output_field not in ("eta", "depth", "eta_t", "state"):
        raise ValueError("output_field must be one of: eta, depth, eta_t, state")

    def _snapshot() -> np.ndarray:
        if output_field == "eta":
            return solver.compute_free_surface().copy()
        if output_field == "depth":
            return solver.H + solver.eta
        if output_field == "eta_t":
            return solver.eta_t.copy()

        return solver.get_state().copy()

    frames: list[np.ndarray] = []
    if include_initial_state:
        frames.append(_snapshot())

    for step_idx in range(max(0, n_steps)):
        dt = solver.suggest_dt(target_cfl=target_cfl) if auto_dt else solver.dt
        solver.step(dt=dt, auto_dt=False)

        if (step_idx + 1) % record_every == 0:
            frames.append(_snapshot())

    if not frames:
        frames.append(_snapshot())

    return np.stack(frames, axis=0).astype(np.float32)
