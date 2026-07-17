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

    def _euler_step_from_state(self, U: np.ndarray, dt: float) -> np.ndarray:
        h_old = np.maximum(U[0], 0.0)
        hu_old = U[1]
        hv_old = U[2]
        b = self.b

        rec = self._reconstructed_faces(h_old, hu_old, hv_old, b)

        h_new = h_old.copy()
        hu_new = hu_old.copy()
        hv_new = hv_old.copy()

        for i in range(self.nx):
            for j in range(self.ny):
                # left x-face
                hC, huC, hvC, bC = rec["h_w"][i, j], rec["hu_w"][i, j], rec["hv_w"][i, j], rec["b_w"][i, j]

                if i == 0:
                    if self.boundary_x == "periodic":
                        hL, huL, hvL, bL = rec["h_e"][-1, j], rec["hu_e"][-1, j], rec["hv_e"][-1, j], rec["b_e"][-1, j]
                    elif self.boundary_x == "radiation":
                        hL, huL, hvL, bL = radiation_boundary_state_x(
                            hC,
                            huC,
                            hvC,
                            bC,
                            side="left",
                            g=self.g,
                            dry_tolerance=self.dry_tolerance,
                        )
                    else:
                        hL, huL, hvL, bL = self._boundary_state_x(h_old, hu_old, hv_old, b, j, side="left")
                else:
                    hL, huL, hvL, bL = rec["h_e"][i - 1, j], rec["hu_e"][i - 1, j], rec["hv_e"][i - 1, j], rec["b_e"][i - 1, j]

                FxL = self._hydro_face_x(hL, huL, hvL, bL, hC, huC, hvC, bC, use_left_correction=False)

                # right x-face
                hC, huC, hvC, bC = rec["h_e"][i, j], rec["hu_e"][i, j], rec["hv_e"][i, j], rec["b_e"][i, j]

                if i == self.nx - 1:
                    if self.boundary_x == "periodic":
                        hR, huR, hvR, bR = rec["h_w"][0, j], rec["hu_w"][0, j], rec["hv_w"][0, j], rec["b_w"][0, j]
                    elif self.boundary_x == "radiation":
                        hR, huR, hvR, bR = radiation_boundary_state_x(
                            hC,
                            huC,
                            hvC,
                            bC,
                            side="right",
                            g=self.g,
                            dry_tolerance=self.dry_tolerance,
                        )
                    else:
                        hR, huR, hvR, bR = self._boundary_state_x(h_old, hu_old, hv_old, b, j, side="right")
                else:
                    hR, huR, hvR, bR = rec["h_w"][i + 1, j], rec["hu_w"][i + 1, j], rec["hv_w"][i + 1, j], rec["b_w"][i + 1, j]

                FxR = self._hydro_face_x(hC, huC, hvC, bC, hR, huR, hvR, bR, use_left_correction=True)

                # bottom y-face
                hC, huC, hvC, bC = rec["h_s"][i, j], rec["hu_s"][i, j], rec["hv_s"][i, j], rec["b_s"][i, j]

                if j == 0:
                    if self.boundary_y == "periodic":
                        hB, huB, hvB, bB = rec["h_n"][i, -1], rec["hu_n"][i, -1], rec["hv_n"][i, -1], rec["b_n"][i, -1]
                    elif self.boundary_y == "radiation":
                        hB, huB, hvB, bB = radiation_boundary_state_y(
                            hC,
                            huC,
                            hvC,
                            bC,
                            side="bottom",
                            g=self.g,
                            dry_tolerance=self.dry_tolerance,
                        )
                    else:
                        hB, huB, hvB, bB = self._boundary_state_y(h_old, hu_old, hv_old, b, i, side="bottom")
                else:
                    hB, huB, hvB, bB = rec["h_n"][i, j - 1], rec["hu_n"][i, j - 1], rec["hv_n"][i, j - 1], rec["b_n"][i, j - 1]

                FyB = self._hydro_face_y(hB, huB, hvB, bB, hC, huC, hvC, bC, use_left_correction=False)

                # top y-face
                hC, huC, hvC, bC = rec["h_n"][i, j], rec["hu_n"][i, j], rec["hv_n"][i, j], rec["b_n"][i, j]

                if j == self.ny - 1:
                    if self.boundary_y == "periodic":
                        hT, huT, hvT, bT = rec["h_s"][i, 0], rec["hu_s"][i, 0], rec["hv_s"][i, 0], rec["b_s"][i, 0]
                    elif self.boundary_y == "radiation":
                        hT, huT, hvT, bT = radiation_boundary_state_y(
                            hC,
                            huC,
                            hvC,
                            bC,
                            side="top",
                            g=self.g,
                            dry_tolerance=self.dry_tolerance,
                        )
                    else:
                        hT, huT, hvT, bT = self._boundary_state_y(h_old, hu_old, hv_old, b, i, side="top")
                else:
                    hT, huT, hvT, bT = rec["h_s"][i, j + 1], rec["hu_s"][i, j + 1], rec["hv_s"][i, j + 1], rec["b_s"][i, j + 1]

                FyT = self._hydro_face_y(hC, huC, hvC, bC, hT, huT, hvT, bT, use_left_correction=True)

                h_new[i, j] = h_old[i, j] - (dt / self.dx) * (FxR[0] - FxL[0]) - (dt / self.dy) * (FyT[0] - FyB[0])
                hu_new[i, j] = hu_old[i, j] - (dt / self.dx) * (FxR[1] - FxL[1]) - (dt / self.dy) * (FyT[1] - FyB[1])
                hv_new[i, j] = hv_old[i, j] - (dt / self.dx) * (FxR[2] - FxL[2]) - (dt / self.dy) * (FyT[2] - FyB[2])

                """
                IMPORTANT: DO NOT REMOVE ANY OF THOSE LINE BELOW
                THIS IS IMPORTANT FOR LAKE-AT-REST STABILITY

                TESTED FOR 500 STEPS AT DIFF GRIDS, BATHYMETRY, BOUNDARIES AND THE DRIFT IS MINIMAL ~2e-16 -> 4e-16
                """
                S_hu = -self.g * 0.5 * (rec["h_e"][i, j] + rec["h_w"][i, j]) * (
                    rec["b_e"][i, j] - rec["b_w"][i, j]
                ) / self.dx

                S_hv = -self.g * 0.5 * (rec["h_n"][i, j] + rec["h_s"][i, j]) * (
                    rec["b_n"][i, j] - rec["b_s"][i, j]
                ) / self.dy

                hu_new[i, j] += dt * S_hu
                hv_new[i, j] += dt * S_hv

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
