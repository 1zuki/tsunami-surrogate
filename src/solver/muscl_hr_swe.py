from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import numpy as np

try:
    from src.solver.hydrostatic_swe import ShallowWaterSolver
    from src.solver.boundary_conditions import (
        radiation_boundary_state_x,
        radiation_boundary_state_y,
    )
except ImportError:
    from hydrostatic_swe import ShallowWaterSolver
    from boundary_conditions import (
        radiation_boundary_state_x,
        radiation_boundary_state_y,
    )


class MUSCLHRShallowWaterSolver(ShallowWaterSolver):
    """
    Second-order MUSCL-HR shallow-water solver

    Uses MUSCL reconstruction for h, eta, u, v, then applies hydrostatic
    reconstruction at interfaces for a better bathymetry-balanced update
    """

    def __init__(
        self,
        *args: Any,
        reconstruction_limiter: Literal["minmod", "unlimited"] = "minmod",
        **kwargs: Any,
    ) -> None:
        if reconstruction_limiter not in ("minmod", "unlimited"):
            raise ValueError(
                "reconstruction_limiter must be 'minmod' or 'unlimited'"
            )
        self.reconstruction_limiter = reconstruction_limiter
        super().__init__(*args, **kwargs)

    def reset_operator_diagnostics(self) -> None:
        super().reset_operator_diagnostics()
        self.operator_diagnostics.update(
            {
                "muscl_reconstruction_limiter": self.reconstruction_limiter,
                "muscl_cell_velocity_clip_count": 0,
                "muscl_face_velocity_clip_count": 0,
                "muscl_limiter_total_count": 0,
                "muscl_limiter_zeroed_count": 0,
                "muscl_limiter_limited_count": 0,
                "muscl_limiter_x_seam_total_count": 0,
                "muscl_limiter_x_seam_zeroed_count": 0,
                "muscl_limiter_x_seam_limited_count": 0,
                "muscl_limiter_y_seam_total_count": 0,
                "muscl_limiter_y_seam_zeroed_count": 0,
                "muscl_limiter_y_seam_limited_count": 0,
            }
        )

    def _minmod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        same = np.sign(a) == np.sign(b)
        result = np.where(
            same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0
        )
        self.operator_diagnostics["muscl_limiter_total_count"] = int(
            self.operator_diagnostics["muscl_limiter_total_count"]
        ) + int(result.size)
        self.operator_diagnostics["muscl_limiter_zeroed_count"] = int(
            self.operator_diagnostics["muscl_limiter_zeroed_count"]
        ) + int(np.count_nonzero(result == 0.0))
        unlimited = 0.5 * (a + b)
        self.operator_diagnostics["muscl_limiter_limited_count"] = int(
            self.operator_diagnostics["muscl_limiter_limited_count"]
        ) + int(np.count_nonzero(result != unlimited))
        return result

    def _limited_slope(
        self,
        forward: np.ndarray,
        backward: np.ndarray,
        *,
        axis: Literal["x", "y"],
        periodic: bool,
    ) -> np.ndarray:
        if self.reconstruction_limiter == "unlimited":
            return 0.5 * (forward + backward)
        result = self._minmod(forward, backward)
        if periodic:
            unlimited = 0.5 * (forward + backward)
            seam = result[[0, -1], :] if axis == "x" else result[:, [0, -1]]
            seam_unlimited = (
                unlimited[[0, -1], :]
                if axis == "x"
                else unlimited[:, [0, -1]]
            )
            prefix = f"muscl_limiter_{axis}_seam"
            self.operator_diagnostics[f"{prefix}_total_count"] = int(
                self.operator_diagnostics[f"{prefix}_total_count"]
            ) + int(seam.size)
            self.operator_diagnostics[f"{prefix}_zeroed_count"] = int(
                self.operator_diagnostics[f"{prefix}_zeroed_count"]
            ) + int(np.count_nonzero(seam == 0.0))
            self.operator_diagnostics[f"{prefix}_limited_count"] = int(
                self.operator_diagnostics[f"{prefix}_limited_count"]
            ) + int(np.count_nonzero(seam != seam_unlimited))
        return result

    def _slope_x(self, field: np.ndarray) -> np.ndarray:
        if self.boundary_x == "periodic":
            forward = np.roll(field, -1, axis=0) - field
            backward = field - np.roll(field, 1, axis=0)
            return self._limited_slope(
                forward, backward, axis="x", periodic=True
            )

        slope = np.zeros_like(field)
        fwd = field[2:, :] - field[1:-1, :]
        bwd = field[1:-1, :] - field[:-2, :]
        slope[1:-1, :] = self._limited_slope(
            fwd, bwd, axis="x", periodic=False
        )

        return slope

    def _slope_y(self, field: np.ndarray) -> np.ndarray:
        if self.boundary_y == "periodic":
            forward = np.roll(field, -1, axis=1) - field
            backward = field - np.roll(field, 1, axis=1)
            return self._limited_slope(
                forward, backward, axis="y", periodic=True
            )

        slope = np.zeros_like(field)
        fwd = field[:, 2:] - field[:, 1:-1]
        bwd = field[:, 1:-1] - field[:, :-2]
        slope[:, 1:-1] = self._limited_slope(
            fwd, bwd, axis="y", periodic=False
        )

        return slope

    def _reconstruct(self, field: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        west = field - 0.5 * sx
        east = field + 0.5 * sx
        south = field - 0.5 * sy
        north = field + 0.5 * sy

        return west, east, south, north

    def _reconstructed_faces(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        hu = np.asarray(hu, dtype=float)
        hv = np.asarray(hv, dtype=float)
        b = np.asarray(b, dtype=float)

        h_safe = np.maximum(h, self.dry_tolerance)
        wet = h > self.dry_tolerance

        u = np.zeros_like(h)
        v = np.zeros_like(h)
        u[wet] = hu[wet] / h_safe[wet]
        v[wet] = hv[wet] / h_safe[wet]
        self.operator_diagnostics["muscl_cell_velocity_clip_count"] = int(
            self.operator_diagnostics["muscl_cell_velocity_clip_count"]
        ) + int(
            np.count_nonzero(np.abs(u) > self.max_velocity)
            + np.count_nonzero(np.abs(v) > self.max_velocity)
        )
        u = np.clip(u, -self.max_velocity, self.max_velocity)
        v = np.clip(v, -self.max_velocity, self.max_velocity)
        eta = h + b

        sx_h, sy_h = self._slope_x(h), self._slope_y(h)
        sx_eta, sy_eta = self._slope_x(eta), self._slope_y(eta)
        sx_u, sy_u = self._slope_x(u), self._slope_y(u)
        sx_v, sy_v = self._slope_x(v), self._slope_y(v)

        h_w, h_e, h_s, h_n = self._reconstruct(h, sx_h, sy_h)
        eta_w, eta_e, eta_s, eta_n = self._reconstruct(eta, sx_eta, sy_eta)
        u_w, u_e, u_s, u_n = self._reconstruct(u, sx_u, sy_u)
        v_w, v_e, v_s, v_n = self._reconstruct(v, sx_v, sy_v)

        h_w = np.maximum(h_w, 0.0)
        h_e = np.maximum(h_e, 0.0)
        h_s = np.maximum(h_s, 0.0)
        h_n = np.maximum(h_n, 0.0)

        b_w = eta_w - h_w
        b_e = eta_e - h_e
        b_s = eta_s - h_s
        b_n = eta_n - h_n

        face_values = (u_w, u_e, u_s, u_n, v_w, v_e, v_s, v_n)
        self.operator_diagnostics["muscl_face_velocity_clip_count"] = int(
            self.operator_diagnostics["muscl_face_velocity_clip_count"]
        ) + int(
            sum(np.count_nonzero(np.abs(value) > self.max_velocity) for value in face_values)
        )
        u_w = np.clip(u_w, -self.max_velocity, self.max_velocity)
        u_e = np.clip(u_e, -self.max_velocity, self.max_velocity)
        u_s = np.clip(u_s, -self.max_velocity, self.max_velocity)
        u_n = np.clip(u_n, -self.max_velocity, self.max_velocity)

        v_w = np.clip(v_w, -self.max_velocity, self.max_velocity)
        v_e = np.clip(v_e, -self.max_velocity, self.max_velocity)
        v_s = np.clip(v_s, -self.max_velocity, self.max_velocity)
        v_n = np.clip(v_n, -self.max_velocity, self.max_velocity)

        return {
            "h_w": h_w, "h_e": h_e, "h_s": h_s, "h_n": h_n,
            "b_w": b_w, "b_e": b_e, "b_s": b_s, "b_n": b_n,
            "hu_w": h_w * u_w, "hu_e": h_e * u_e, "hu_s": h_s * u_s, "hu_n": h_n * u_n,
            "hv_w": h_w * v_w, "hv_e": h_e * v_e, "hv_s": h_s * v_s, "hv_n": h_n * v_n,
        }

    def _reconstructed_x_interface_states(
        self,
        rec: Mapping[str, np.ndarray],
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return reconstructed left/right states at all x interfaces."""
        shape = (self.nx + 1, self.ny)
        left = [np.empty(shape, dtype=float) for _ in range(4)]
        right = [np.empty(shape, dtype=float) for _ in range(4)]
        east = (rec["h_e"], rec["hu_e"], rec["hv_e"], rec["b_e"])
        west = (rec["h_w"], rec["hu_w"], rec["hv_w"], rec["b_w"])

        for target, values in zip(left, east):
            target[1:, :] = values
        for target, values in zip(right, west):
            target[:-1, :] = values

        for j in range(self.ny):
            if self.boundary_x == "periodic":
                left_boundary = tuple(values[-1, j] for values in east)
                right_boundary = tuple(values[0, j] for values in west)
            elif self.boundary_x == "radiation":
                left_boundary = radiation_boundary_state_x(
                    *(values[0, j] for values in west),
                    side="left",
                    g=self.g,
                    dry_tolerance=self.dry_tolerance,
                )
                right_boundary = radiation_boundary_state_x(
                    *(values[-1, j] for values in east),
                    side="right",
                    g=self.g,
                    dry_tolerance=self.dry_tolerance,
                )
            else:
                left_boundary = self._boundary_state_x(
                    h, hu, hv, b, j, side="left"
                )
                right_boundary = self._boundary_state_x(
                    h, hu, hv, b, j, side="right"
                )
            for target, value in zip(left, left_boundary):
                target[0, j] = value
            for target, value in zip(right, right_boundary):
                target[-1, j] = value

        return (*left, *right)

    def _reconstructed_y_interface_states(
        self,
        rec: Mapping[str, np.ndarray],
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return reconstructed bottom/top states at all y interfaces."""
        shape = (self.nx, self.ny + 1)
        bottom = [np.empty(shape, dtype=float) for _ in range(4)]
        top = [np.empty(shape, dtype=float) for _ in range(4)]
        north = (rec["h_n"], rec["hu_n"], rec["hv_n"], rec["b_n"])
        south = (rec["h_s"], rec["hu_s"], rec["hv_s"], rec["b_s"])

        for target, values in zip(bottom, north):
            target[:, 1:] = values
        for target, values in zip(top, south):
            target[:, :-1] = values

        for i in range(self.nx):
            if self.boundary_y == "periodic":
                bottom_boundary = tuple(values[i, -1] for values in north)
                top_boundary = tuple(values[i, 0] for values in south)
            elif self.boundary_y == "radiation":
                bottom_boundary = radiation_boundary_state_y(
                    *(values[i, 0] for values in south),
                    side="bottom",
                    g=self.g,
                    dry_tolerance=self.dry_tolerance,
                )
                top_boundary = radiation_boundary_state_y(
                    *(values[i, -1] for values in north),
                    side="top",
                    g=self.g,
                    dry_tolerance=self.dry_tolerance,
                )
            else:
                bottom_boundary = self._boundary_state_y(
                    h, hu, hv, b, i, side="bottom"
                )
                top_boundary = self._boundary_state_y(
                    h, hu, hv, b, i, side="top"
                )
            for target, value in zip(bottom, bottom_boundary):
                target[i, 0] = value
            for target, value in zip(top, top_boundary):
                target[i, -1] = value

        return (*bottom, *top)

    def _euler_step_from_state(self, U: np.ndarray, dt: float) -> np.ndarray:
        h_old = np.maximum(U[0], 0.0)
        hu_old = U[1]
        hv_old = U[2]
        b = self.b

        rec = self._reconstructed_faces(h_old, hu_old, hv_old, b)

        x_states = self._reconstructed_x_interface_states(
            rec, h_old, hu_old, hv_old, b
        )
        x_flux, x_left_correction, x_right_correction = (
            self._hydrostatic_interface_fluxes(*x_states, axis="x")
        )
        flux_x_left = x_flux[:, :-1, :].copy()
        flux_x_right = x_flux[:, 1:, :].copy()
        flux_x_left[1] += x_right_correction[:-1, :]
        flux_x_right[1] += x_left_correction[1:, :]

        y_states = self._reconstructed_y_interface_states(
            rec, h_old, hu_old, hv_old, b
        )
        y_flux, y_bottom_correction, y_top_correction = (
            self._hydrostatic_interface_fluxes(*y_states, axis="y")
        )
        flux_y_bottom = y_flux[:, :, :-1].copy()
        flux_y_top = y_flux[:, :, 1:].copy()
        flux_y_bottom[2] += y_top_correction[:, :-1]
        flux_y_top[2] += y_bottom_correction[:, 1:]

        h_new = (
            h_old
            - (dt / self.dx) * (flux_x_right[0] - flux_x_left[0])
            - (dt / self.dy) * (flux_y_top[0] - flux_y_bottom[0])
        )
        hu_new = (
            hu_old
            - (dt / self.dx) * (flux_x_right[1] - flux_x_left[1])
            - (dt / self.dy) * (flux_y_top[1] - flux_y_bottom[1])
        )
        hv_new = (
            hv_old
            - (dt / self.dx) * (flux_x_right[2] - flux_x_left[2])
            - (dt / self.dy) * (flux_y_top[2] - flux_y_bottom[2])
        )

        # Keep the well-balanced bathymetry source correction unchanged.
        source_hu = -self.g * 0.5 * (rec["h_e"] + rec["h_w"]) * (
            rec["b_e"] - rec["b_w"]
        ) / self.dx
        source_hv = -self.g * 0.5 * (rec["h_n"] + rec["h_s"]) * (
            rec["b_n"] - rec["b_s"]
        ) / self.dy
        hu_new += dt * source_hu
        hv_new += dt * source_hv

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

        U_new = np.stack([h_new, hu_new, hv_new], axis=0)
        return self._nan_to_num_with_diagnostics(U_new)

    def update(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")

        U0 = self.get_state()
        U1 = self._euler_step_from_state(U0, dt)
        U2 = self._euler_step_from_state(U1, dt)

        U_new = 0.5 * (U0 + U2)
        U_new = self._nan_to_num_with_diagnostics(U_new)
        self.operator_diagnostics["positivity_projection_count"] = int(
            self.operator_diagnostics["positivity_projection_count"]
        ) + int(np.count_nonzero(U_new[0] < 0.0))
        U_new[0] = np.maximum(U_new[0], 0.0)

        self.set_state(U_new)


def _to_sample_array(sample_inputs: Any) -> np.ndarray:
    if hasattr(sample_inputs, "detach"):
        sample_inputs = sample_inputs.detach().cpu().numpy()

    arr = np.asarray(sample_inputs, dtype=float)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(
            f"sample_inputs must have shape [C,H,W] or [B,C,H,W], got {arr.shape}"
        )

    return arr


def simulate_rollout(sample_inputs: Any, **kwargs: Any) -> np.ndarray:
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

    solver = MUSCLHRShallowWaterSolver(
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
        sponge_axes=str(kwargs.get("sponge_axes", "xy")),
        max_velocity=float(kwargs.get("max_velocity", 50.0)),
        reconstruction_limiter=str(kwargs.get("reconstruction_limiter", "minmod")),
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
        dt = solver.suggest_dt(target_cfl=target_cfl) if auto_dt else solver.dt
        solver.step(dt=dt, auto_dt=False)

        if (step_idx + 1) % record_every == 0:
            frames.append(_snapshot())

    if not frames:
        frames.append(_snapshot())

    return np.stack(frames, axis=0).astype(np.float32)

"""
Clain, S., C.Reis, R.Costa, J.Figueiredo, M. A.Baptista, and J. M.Miranda (2016), Second-order finite volume with hydrostatic reconstruction for tsunami simulation, J. Adv. Model. Earth Syst., 8, 1691–1713, doi:10.1002/2015MS000603. 

Jingming Hou, Qiuhua Liang, Hongbin Zhang, Reinhard Hinkelmann,
An efficient unstructured MUSCL scheme for solving the 2D shallow water equations,
Environmental Modelling & Software,
Volume 66,
2015,
Pages 131-152,
ISSN 1364-8152,
https://doi.org/10.1016/j.envsoft.2014.12.007.

"""
