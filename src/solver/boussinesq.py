from __future__ import annotations

from typing import Any, Literal, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np

from src.solver.boundary_conditions import (
    BoundaryMode,
    pad_scalar_field,
    resolve_boundary_modes,
)
from src.solver.operator_time import (
    build_sponge_mask,
    filter_coefficient,
    sponge_factor,
    validate_cg_failure_mode,
    validate_filter_time_mode,
    validate_sponge_profile,
    validate_sponge_time_mode,
)

BoussinesqMode = Literal[
    "linear_constant_depth",
    "linear_variable_depth",
    "weakly_nonlinear_planned",
]
LinearSolverPreconditioner = Literal["jacobi", "sparse_lu"]


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
        linear_solver_abs_tol: float = 0.0,
        linear_solver_max_iter: int = 80,
        linear_solver_preconditioner: LinearSolverPreconditioner = "jacobi",
        check_finite: bool = True,
        sponge_time_mode: str = "legacy_per_step",
        sponge_reference_dt: float | None = None,
        filter_time_mode: str = "legacy_per_step",
        filter_reference_dt: float | None = None,
        cg_failure_mode: str = "legacy_posthoc",
        sponge_axes: str = "xy",
        sponge_profile: str = "quadratic",
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
        if sponge_axes not in ("xy", "x"):
            raise ValueError("sponge_axes must be 'xy' or 'x'")
        if filter_strength < 0:
            raise ValueError("filter_strength must be non-negative")
        if linear_solver_tol <= 0:
            raise ValueError("linear_solver_tol must be positive")
        if linear_solver_abs_tol < 0:
            raise ValueError("linear_solver_abs_tol must be non-negative")
        if linear_solver_max_iter <= 0:
            raise ValueError("linear_solver_max_iter must be positive")
        if linear_solver_preconditioner not in ("jacobi", "sparse_lu"):
            raise ValueError(
                "linear_solver_preconditioner must be 'jacobi' or 'sparse_lu'"
            )

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
        self.linear_solver_abs_tol = float(linear_solver_abs_tol)
        self.linear_solver_max_iter = int(linear_solver_max_iter)
        self.linear_solver_preconditioner = linear_solver_preconditioner
        self.check_finite = bool(check_finite)
        self.sponge_time_mode = validate_sponge_time_mode(
            sponge_time_mode, sponge_reference_dt
        )
        self.sponge_reference_dt = (
            None if sponge_reference_dt is None else float(sponge_reference_dt)
        )
        self.filter_time_mode = validate_filter_time_mode(
            filter_time_mode, filter_reference_dt
        )
        self.filter_reference_dt = (
            None if filter_reference_dt is None else float(filter_reference_dt)
        )
        self.cg_failure_mode = validate_cg_failure_mode(cg_failure_mode)

        self.boundary_x, self.boundary_y = resolve_boundary_modes(boundary)
        if "radiation" in (self.boundary_x, self.boundary_y):
            raise ValueError(
                "radiation boundary is currently implemented only for SWE solvers"
            )
        self._validate_mode(self.mode)

        self.eta = np.zeros((self.nx, self.ny), dtype=float)
        self.eta_t = np.zeros((self.nx, self.ny), dtype=float)
        self.b = np.zeros((self.nx, self.ny), dtype=float)
        self.H = np.ones((self.nx, self.ny), dtype=float)
        self._depth_coeff_x, self._depth_coeff_y = self._face_coefficients(self.H)
        self._mass_coeff_x, self._mass_coeff_y = self._face_coefficients(
            self.H * self.H
        )
        self._mass_diag_inv = np.ones((self.nx, self.ny), dtype=float)
        self._mass_sparse_factor: Any | None = None
        self._mass_sparse_nnz = 0

        if use_sponge is None:
            self.use_sponge = "periodic" not in (self.boundary_x, self.boundary_y)
        else:
            self.use_sponge = bool(use_sponge)
        self.sponge_width = int(sponge_width)
        self.sponge_min_factor = float(sponge_min_factor)
        self.sponge_axes = str(sponge_axes)
        self.sponge_profile = validate_sponge_profile(sponge_profile)
        self.sponge_mask = np.ones((self.nx, self.ny), dtype=float)
        self.last_cg_iterations = 0
        self.last_cg_initial_residual = 0.0
        self.last_cg_final_residual = 0.0
        self.last_cg_converged = True
        self.last_step_cg_converged = True
        self.last_step_cg_failed_count = 0
        self.last_step_cg_max_iterations = 0
        self.last_step_cg_max_residual_ratio = 0.0
        self.last_step_cg_solve_converged = (True, True)
        self.last_step_cg_solve_iterations = (0, 0)
        self.last_step_cg_solve_initial_residual = (0.0, 0.0)
        self.last_step_cg_solve_final_residual = (0.0, 0.0)
        self.last_step_cg_solve_residual_ratio = (0.0, 0.0)
        self.operator_diagnostics: dict[str, float | int | bool | str | None] = {}
        if self.use_sponge:
            self._init_sponge_layer(width=self.sponge_width, min_factor=self.sponge_min_factor)
        self.reset_operator_diagnostics()

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
        self._update_operator_coefficients()
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

    def _face_coefficients(
        self, coefficient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average a cell-centered coefficient onto x and y faces."""
        coefficient = self._check_shape(coefficient, "coefficient")
        C = self._pad_scalar(coefficient)
        coeff_x = 0.5 * (C[1:, 1:-1] + C[:-1, 1:-1])
        coeff_y = 0.5 * (C[1:-1, 1:] + C[1:-1, :-1])
        return coeff_x, coeff_y

    def _flux_divergence_from_faces(
        self,
        field: np.ndarray,
        coeff_x: np.ndarray,
        coeff_y: np.ndarray,
    ) -> np.ndarray:
        """Apply a divergence using precomputed face coefficients."""
        field = self._check_shape(field, "field")
        difference_x = np.empty((self.nx + 1, self.ny), dtype=float)
        difference_x[1:-1, :] = field[1:, :] - field[:-1, :]
        if self.boundary_x == "periodic":
            seam = field[0, :] - field[-1, :]
            difference_x[0, :] = seam
            difference_x[-1, :] = seam
        else:
            difference_x[0, :] = 0.0
            difference_x[-1, :] = 0.0

        difference_y = np.empty((self.nx, self.ny + 1), dtype=float)
        difference_y[:, 1:-1] = field[:, 1:] - field[:, :-1]
        if self.boundary_y == "periodic":
            seam = field[:, 0] - field[:, -1]
            difference_y[:, 0] = seam
            difference_y[:, -1] = seam
        else:
            difference_y[:, 0] = 0.0
            difference_y[:, -1] = 0.0

        q_x = coeff_x * difference_x / self.dx
        q_y = coeff_y * difference_y / self.dy
        return (q_x[1:, :] - q_x[:-1, :]) / self.dx + (q_y[:, 1:] - q_y[:, :-1]) / self.dy

    def _flux_divergence(self, field: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
        """Compute div(coefficient * grad(field)) from face-centered fluxes."""
        coeff_x, coeff_y = self._face_coefficients(coefficient)
        return self._flux_divergence_from_faces(field, coeff_x, coeff_y)

    def _update_operator_coefficients(self) -> None:
        self._depth_coeff_x, self._depth_coeff_y = self._face_coefficients(self.H)
        self._mass_coeff_x, self._mass_coeff_y = self._face_coefficients(
            self.H * self.H
        )
        self._mass_sparse_factor = None
        self._mass_sparse_nnz = 0

    def rhs(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute g * div(H * grad(eta))."""
        if eta is None:
            eta = self.eta
        else:
            eta = self._check_shape(eta, "eta")

        return self.g * self._flux_divergence_from_faces(
            eta, self._depth_coeff_x, self._depth_coeff_y
        )

    def apply_mass_operator(self, a: np.ndarray) -> np.ndarray:
        """Apply M(a) = a - alpha * div(H^2 * grad(a))."""
        a = self._check_shape(a, "a")
        if self.alpha == 0.0:
            return a.copy()
        dispersive = self._flux_divergence_from_faces(
            a, self._mass_coeff_x, self._mass_coeff_y
        )
        return a - self.alpha * dispersive

    def _update_mass_preconditioner(self) -> None:
        """Build a diagonal Jacobi preconditioner for the dispersive mass solve."""
        if self.alpha == 0.0:
            self._mass_diag_inv = np.ones_like(self.H)
            return

        diag = np.ones_like(self.H)
        diag += self.alpha * (
            (self._mass_coeff_x[1:, :] + self._mass_coeff_x[:-1, :])
            / (self.dx * self.dx)
            + (self._mass_coeff_y[:, 1:] + self._mass_coeff_y[:, :-1])
            / (self.dy * self.dy)
        )
        self._mass_diag_inv = 1.0 / np.maximum(diag, 1e-30)

    def _build_mass_sparse_factor(self) -> Any:
        """Factor the fixed dispersive mass matrix for CG preconditioning."""
        try:
            from scipy import sparse
            from scipy.sparse.linalg import splu
        except ImportError as exc:
            raise RuntimeError(
                "linear_solver_preconditioner='sparse_lu' requires SciPy"
            ) from exc

        cell_ids = np.arange(self.nx * self.ny, dtype=np.int64).reshape(
            self.nx, self.ny
        )
        diagonal = np.ones(self.nx * self.ny, dtype=np.float64)
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []

        def add_edges(
            first: np.ndarray,
            second: np.ndarray,
            weights: np.ndarray,
        ) -> None:
            first_flat = np.asarray(first, dtype=np.int64).reshape(-1)
            second_flat = np.asarray(second, dtype=np.int64).reshape(-1)
            weight_flat = np.asarray(weights, dtype=np.float64).reshape(-1)
            np.add.at(diagonal, first_flat, weight_flat)
            np.add.at(diagonal, second_flat, weight_flat)
            row_parts.extend((first_flat, second_flat))
            column_parts.extend((second_flat, first_flat))
            value_parts.extend((-weight_flat, -weight_flat))

        add_edges(
            cell_ids[:-1, :],
            cell_ids[1:, :],
            self.alpha * self._mass_coeff_x[1:self.nx, :] / (self.dx * self.dx),
        )
        if self.boundary_x == "periodic":
            add_edges(
                cell_ids[0, :],
                cell_ids[-1, :],
                self.alpha * self._mass_coeff_x[0, :] / (self.dx * self.dx),
            )

        add_edges(
            cell_ids[:, :-1],
            cell_ids[:, 1:],
            self.alpha * self._mass_coeff_y[:, 1:self.ny] / (self.dy * self.dy),
        )
        if self.boundary_y == "periodic":
            add_edges(
                cell_ids[:, 0],
                cell_ids[:, -1],
                self.alpha * self._mass_coeff_y[:, 0] / (self.dy * self.dy),
            )

        flat_ids = cell_ids.reshape(-1)
        rows = np.concatenate((flat_ids, *row_parts))
        columns = np.concatenate((flat_ids, *column_parts))
        values = np.concatenate((diagonal, *value_parts))
        matrix = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(flat_ids.size, flat_ids.size),
        ).tocsc()
        self._mass_sparse_nnz = int(matrix.nnz)
        self._mass_sparse_factor = splu(matrix)
        self.operator_diagnostics["linear_solver_factorization_count"] = int(
            self.operator_diagnostics["linear_solver_factorization_count"]
        ) + 1
        self.operator_diagnostics["linear_solver_factorization_nnz"] = int(
            matrix.nnz
        )
        return self._mass_sparse_factor

    def _apply_linear_preconditioner(self, residual: np.ndarray) -> np.ndarray:
        if self.linear_solver_preconditioner == "jacobi":
            return self._mass_diag_inv * residual
        factor = self._mass_sparse_factor
        if factor is None:
            factor = self._build_mass_sparse_factor()
        return np.asarray(
            factor.solve(np.asarray(residual, dtype=np.float64).reshape(-1)),
            dtype=np.float64,
        ).reshape(self.nx, self.ny)

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
        z = self._apply_linear_preconditioner(r)
        p = z.copy()
        rs_old = float(np.sum(r * r))
        rz_old = float(np.sum(r * z))
        rs0 = rs_old
        residual0 = float(np.sqrt(rs0))
        threshold = max(
            (self.linear_solver_tol * residual0) ** 2,
            self.linear_solver_abs_tol**2,
        )
        self.last_cg_iterations = 0
        self.last_cg_initial_residual = residual0
        self.last_cg_final_residual = residual0
        self.last_cg_converged = residual0 <= self.linear_solver_abs_tol

        if rs0 == 0.0 or residual0 <= self.linear_solver_abs_tol:
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

            z = self._apply_linear_preconditioner(r)
            rz_new = float(np.sum(r * z))
            beta = rz_new / max(rz_old, eps)
            p = z + beta * p
            rs_old = rs_new
            rz_old = rz_new
        else:
            self.last_cg_converged = False

        return x

    def _init_sponge_layer(self, width: int = 20, min_factor: float = 0.9) -> None:
        self.sponge_mask = build_sponge_mask(
            nx=self.nx,
            ny=self.ny,
            width=width,
            min_factor=min_factor,
            axes=self.sponge_axes,
            profile=self.sponge_profile,
        )

    def reset_operator_diagnostics(self) -> None:
        if self.sponge_reference_dt is None:
            reference_rate_min = None
            reference_rate_max = None
        else:
            reference_rates = -np.log(self.sponge_mask) / self.sponge_reference_dt
            reference_rate_min = float(np.min(reference_rates))
            reference_rate_max = float(np.max(reference_rates))
        filter_reference_rate = (
            None
            if self.filter_reference_dt is None
            else self.filter_strength / self.filter_reference_dt
        )
        self.operator_diagnostics = {
            "sponge_time_mode": self.sponge_time_mode,
            "sponge_axes": self.sponge_axes,
            "sponge_profile": self.sponge_profile,
            "sponge_reference_dt": self.sponge_reference_dt,
            "sponge_reference_decay_rate_min": reference_rate_min,
            "sponge_reference_decay_rate_max": reference_rate_max,
            "sponge_applications": 0,
            "sponge_elapsed_time": 0.0,
            "sponge_accumulated_exponent": 0.0,
            "sponge_effective_factor_min": 1.0,
            "sponge_effective_factor_max": 1.0,
            "filter_time_mode": self.filter_time_mode,
            "filter_reference_dt": self.filter_reference_dt,
            "filter_reference_coefficient_rate": filter_reference_rate,
            "filter_applications": 0,
            "filter_effective_coefficient_last": 0.0,
            "filter_effective_coefficient_max": 0.0,
            "cg_failure_mode": self.cg_failure_mode,
            "cg_absolute_residual_tolerance": self.linear_solver_abs_tol,
            "linear_solver_preconditioner": self.linear_solver_preconditioner,
            "linear_solver_factorization_count": 0,
            "linear_solver_factorization_nnz": int(self._mass_sparse_nnz),
            "cg_solve_count": 0,
            "cg_failure_count": 0,
            "cg_iterations_sum": 0,
            "cg_iterations_max": 0,
            "cg_initial_residual_min": None,
            "cg_initial_residual_max": 0.0,
            "cg_final_residual_min": None,
            "cg_final_residual_max": 0.0,
            "cg_residual_ratio_max": 0.0,
            "nan_to_num_replacement_count": 0,
            "nan_to_num_replacement_occurred": False,
        }

    def get_operator_diagnostics(self) -> dict[str, float | int | bool | str | None]:
        return dict(self.operator_diagnostics)

    def _record_cg_diagnostic(self) -> None:
        ratio = float(self.last_cg_final_residual) / max(
            float(self.last_cg_initial_residual), 1e-30
        )
        self.operator_diagnostics["cg_solve_count"] = int(
            self.operator_diagnostics["cg_solve_count"]
        ) + 1
        self.operator_diagnostics["cg_failure_count"] = int(
            self.operator_diagnostics["cg_failure_count"]
        ) + int(not self.last_cg_converged)
        self.operator_diagnostics["cg_iterations_sum"] = int(
            self.operator_diagnostics["cg_iterations_sum"]
        ) + int(self.last_cg_iterations)
        self.operator_diagnostics["cg_iterations_max"] = max(
            int(self.operator_diagnostics["cg_iterations_max"]),
            int(self.last_cg_iterations),
        )
        initial_residual = float(self.last_cg_initial_residual)
        final_residual = float(self.last_cg_final_residual)
        initial_min = self.operator_diagnostics["cg_initial_residual_min"]
        final_min = self.operator_diagnostics["cg_final_residual_min"]
        self.operator_diagnostics["cg_initial_residual_min"] = (
            initial_residual
            if initial_min is None
            else min(float(initial_min), initial_residual)
        )
        self.operator_diagnostics["cg_initial_residual_max"] = max(
            float(self.operator_diagnostics["cg_initial_residual_max"]),
            initial_residual,
        )
        self.operator_diagnostics["cg_final_residual_min"] = (
            final_residual
            if final_min is None
            else min(float(final_min), final_residual)
        )
        self.operator_diagnostics["cg_final_residual_max"] = max(
            float(self.operator_diagnostics["cg_final_residual_max"]),
            final_residual,
        )
        self.operator_diagnostics["cg_residual_ratio_max"] = max(
            float(self.operator_diagnostics["cg_residual_ratio_max"]), ratio
        )

    def apply_sponge_layer(self, dt: float | None = None) -> None:
        """Damp eta and eta_t near boundaries."""
        if not self.use_sponge:
            return
        if dt is None:
            dt = self.dt
        factor = sponge_factor(
            self.sponge_mask,
            dt=float(dt),
            mode=self.sponge_time_mode,
            reference_dt=self.sponge_reference_dt,
        )
        self.eta *= factor
        self.eta_t *= factor
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

    def apply_filter(self, dt: float | None = None) -> None:
        """Optional Laplacian smoothing with explicit time semantics."""
        if dt is None:
            dt = self.dt
        strength = filter_coefficient(
            self.filter_strength,
            dt=float(dt),
            mode=self.filter_time_mode,
            reference_dt=self.filter_reference_dt,
        )
        if strength <= 0.0:
            return

        self.eta = self.eta + strength * min(self.dx, self.dy) ** 2 * self.laplacian(self.eta)
        self.eta_t = self.eta_t + strength * min(self.dx, self.dy) ** 2 * self.laplacian(self.eta_t)
        self.operator_diagnostics["filter_applications"] = int(
            self.operator_diagnostics["filter_applications"]
        ) + 1
        self.operator_diagnostics["filter_effective_coefficient_last"] = strength
        self.operator_diagnostics["filter_effective_coefficient_max"] = max(
            float(self.operator_diagnostics["filter_effective_coefficient_max"]),
            strength,
        )

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
        self._record_cg_diagnostic()
        cg0_converged = bool(self.last_cg_converged)
        cg0_iterations = int(self.last_cg_iterations)
        cg0_initial_residual = float(self.last_cg_initial_residual)
        cg0_final_residual = float(self.last_cg_final_residual)
        cg0_ratio = cg0_final_residual / max(cg0_initial_residual, 1e-30)
        if self.cg_failure_mode == "strict_v2" and not cg0_converged:
            raise RuntimeError("Boussinesq CG solve 0 failed in strict_v2 mode")
        eta_next = self.eta + dt * self.eta_t + 0.5 * dt * dt * a0
        a1 = self.solve_acceleration(eta_next)
        self._record_cg_diagnostic()
        cg1_converged = bool(self.last_cg_converged)
        cg1_iterations = int(self.last_cg_iterations)
        cg1_initial_residual = float(self.last_cg_initial_residual)
        cg1_final_residual = float(self.last_cg_final_residual)
        cg1_ratio = cg1_final_residual / max(cg1_initial_residual, 1e-30)
        if self.cg_failure_mode == "strict_v2" and not cg1_converged:
            raise RuntimeError("Boussinesq CG solve 1 failed in strict_v2 mode")
        eta_t_next = self.eta_t + 0.5 * dt * (a0 + a1)

        self.last_step_cg_converged = cg0_converged and cg1_converged
        self.last_step_cg_failed_count = int(not cg0_converged) + int(not cg1_converged)
        self.last_step_cg_max_iterations = max(cg0_iterations, cg1_iterations)
        self.last_step_cg_max_residual_ratio = float(max(cg0_ratio, cg1_ratio))
        self.last_step_cg_solve_converged = (cg0_converged, cg1_converged)
        self.last_step_cg_solve_iterations = (cg0_iterations, cg1_iterations)
        self.last_step_cg_solve_initial_residual = (
            cg0_initial_residual,
            cg1_initial_residual,
        )
        self.last_step_cg_solve_final_residual = (
            cg0_final_residual,
            cg1_final_residual,
        )
        self.last_step_cg_solve_residual_ratio = (float(cg0_ratio), float(cg1_ratio))

        self.eta = eta_next
        self.eta_t = eta_t_next
        self.apply_filter(dt=dt)
        self.apply_sponge_layer(dt=dt)

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
        sponge_axes=str(kwargs.get("sponge_axes", "xy")),
        sponge_profile=str(kwargs.get("sponge_profile", "quadratic")),
        filter_strength=float(kwargs.get("filter_strength", 0.0)),
        linear_solver_tol=float(kwargs.get("linear_solver_tol", 1e-8)),
        linear_solver_abs_tol=float(kwargs.get("linear_solver_abs_tol", 0.0)),
        linear_solver_max_iter=int(kwargs.get("linear_solver_max_iter", 80)),
        linear_solver_preconditioner=str(
            kwargs.get("linear_solver_preconditioner", "jacobi")
        ),
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
