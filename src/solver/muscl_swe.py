from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

try:
    from src.solver.hydrostatic_swe import ShallowWaterSolver
except ImportError:
    from hydrostatic_swe import ShallowWaterSolver


class MUSCLShallowWaterSolver(ShallowWaterSolver):
    """
    MUSCL finite-volume SWE solver.

    Reuses the shared SWE utilities from the hydrostatic solver module and
    overrides only the time update to use vectorized MUSCL-Rusanov fluxes.
    """

    def update(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")

        U = self.get_state()
        flux_div_x = self._flux_divergence_x(U)
        flux_div_y = self._flux_divergence_y(U)
        source = self.compute_source(h=U[0])

        U_new = U - dt * (flux_div_x + flux_div_y) + dt * source
        U_new = np.nan_to_num(U_new, nan=0.0, posinf=0.0, neginf=0.0)
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

    bathymetry = (
        channels[idx_bathy]
        if idx_bathy is not None
        else np.zeros((nx, ny), dtype=float)
    )
    source_field = (
        channels[idx_source]
        if idx_source is not None
        else np.zeros((nx, ny), dtype=float)
    )
    sea_level_offset = float(kwargs.get("sea_level_offset", 0.0))
    source_scale = float(kwargs.get("source_scale", 1.0))

    if idx_h0 is not None:
        h0 = np.maximum(channels[idx_h0], 0.0)
    elif idx_eta0 is not None:
        h0 = np.maximum(channels[idx_eta0] - bathymetry, 0.0)
    else:
        rest_depth = np.maximum(-bathymetry + sea_level_offset, 0.0)
        h0 = np.maximum(rest_depth + source_scale * source_field, 0.0)

    solver = MUSCLShallowWaterSolver(
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
