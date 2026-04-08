from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.solver.boundary_problems import pad_state, sponge_mask


@dataclass
class ShallowWaterSolver:
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 0.05
    gravity: float = 1.0
    friction: float = 0.002
    min_depth: float = 0.15
    boundary: str = "transmissive"
    cfl: float = 0.35
    adaptive_dt: bool = True
    sponge_width: int = 0
    sponge_strength: float = 0.12

    @classmethod
    def from_config(cls, config: dict) -> "ShallowWaterSolver":
        sim = config.get("simulation", {})
        return cls(
            dx=float(sim.get("dx", 1.0)),
            dy=float(sim.get("dy", 1.0)),
            dt=float(sim.get("dt", 0.05)),
            gravity=float(sim.get("gravity", 1.0)),
            friction=float(sim.get("friction", 0.002)),
            min_depth=float(sim.get("min_depth", 0.15)),
            boundary=str(sim.get("boundary", "transmissive")),
            cfl=float(sim.get("cfl", 0.35)),
            adaptive_dt=bool(sim.get("adaptive_dt", True)),
            sponge_width=int(sim.get("sponge_width", 0)),
            sponge_strength=float(sim.get("sponge_strength", 0.12)),
        )

    def initial_state(self, bathymetry: np.ndarray, eta0: np.ndarray) -> np.ndarray:
        h0 = np.maximum(eta0 - bathymetry, self.min_depth)
        qx0 = np.zeros_like(h0, dtype=np.float32)
        qy0 = np.zeros_like(h0, dtype=np.float32)
        return np.stack([h0, qx0, qy0], axis=0).astype(np.float32)

    def state_to_eta(self, state: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
        return (state[0] + bathymetry).astype(np.float32)

    def _primitive(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.maximum(U[0], self.min_depth)
        u = U[1] / h
        v = U[2] / h
        return h, u, v

    def _flux_x(self, U: np.ndarray) -> np.ndarray:
        h, u, v = self._primitive(U)
        return np.stack(
            [
                U[1],
                U[1] * u + 0.5 * self.gravity * h**2,
                U[1] * v,
            ],
            axis=0,
        ).astype(np.float32)

    def _flux_y(self, U: np.ndarray) -> np.ndarray:
        h, u, v = self._primitive(U)
        return np.stack(
            [
                U[2],
                U[2] * u,
                U[2] * v + 0.5 * self.gravity * h**2,
            ],
            axis=0,
        ).astype(np.float32)

    def _rusanov_flux_x(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        hL, uL, _ = self._primitive(left)
        hR, uR, _ = self._primitive(right)
        smax = np.maximum(np.abs(uL) + np.sqrt(self.gravity * hL), np.abs(uR) + np.sqrt(self.gravity * hR))
        return 0.5 * (self._flux_x(left) + self._flux_x(right)) - 0.5 * smax[None, ...] * (right - left)

    def _rusanov_flux_y(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        hL, _, vL = self._primitive(lower)
        hR, _, vR = self._primitive(upper)
        smax = np.maximum(np.abs(vL) + np.sqrt(self.gravity * hL), np.abs(vR) + np.sqrt(self.gravity * hR))
        return 0.5 * (self._flux_y(lower) + self._flux_y(upper)) - 0.5 * smax[None, ...] * (upper - lower)

    def _max_stable_dt(self, U: np.ndarray) -> float:
        h, u, v = self._primitive(U)
        wave_speed = np.max(np.abs(u) + np.sqrt(self.gravity * h))
        wave_speed = max(wave_speed, np.max(np.abs(v) + np.sqrt(self.gravity * h)), 1e-6)
        dx_min = min(self.dx, self.dy)
        return self.cfl * dx_min / wave_speed

    def _source_terms(self, U: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
        h = np.maximum(U[0], self.min_depth)
        dzdy, dzdx = np.gradient(bathymetry, self.dy, self.dx)
        qx = U[1]
        qy = U[2]
        src = np.zeros_like(U, dtype=np.float32)
        src[1] = -self.gravity * h * dzdx - self.friction * qx
        src[2] = -self.gravity * h * dzdy - self.friction * qy
        return src

    def _postprocess_state(self, U: np.ndarray) -> np.ndarray:
        U = U.astype(np.float32)
        wet_mask = U[0] > self.min_depth
        U[0] = np.maximum(U[0], self.min_depth)
        U[1] *= wet_mask
        U[2] *= wet_mask
        return U

    def rhs(self, U: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
        Up = pad_state(U, self.boundary)

        fx = self._rusanov_flux_x(Up[:, 1:-1, :-1], Up[:, 1:-1, 1:])
        fy = self._rusanov_flux_y(Up[:, :-1, 1:-1], Up[:, 1:, 1:-1])

        div_x = (fx[:, :, 1:] - fx[:, :, :-1]) / self.dx
        div_y = (fy[:, 1:, :] - fy[:, :-1, :]) / self.dy

        rhs = -(div_x + div_y) + self._source_terms(U, bathymetry)

        if self.sponge_width > 0:
            mask = sponge_mask(U.shape[1], U.shape[2], self.sponge_width, self.sponge_strength)
            rhs[1] *= mask
            rhs[2] *= mask
        return rhs.astype(np.float32)

    def step(self, U: np.ndarray, bathymetry: np.ndarray, dt: float) -> np.ndarray:
        # SSP RK2 / Heun update.
        k1 = self.rhs(U, bathymetry)
        U1 = self._postprocess_state(U + dt * k1)
        k2 = self.rhs(U1, bathymetry)
        U2 = self._postprocess_state(0.5 * U + 0.5 * (U1 + dt * k2))
        return U2

    def simulate(self, bathymetry: np.ndarray, eta0: np.ndarray, nt: int, save_every: int = 1, return_state: bool = False) -> np.ndarray:
        U = self.initial_state(bathymetry, eta0)
        outputs = []
        if save_every <= 0:
            save_every = 1

        for step in range(nt):
            dt = min(self.dt, self._max_stable_dt(U)) if self.adaptive_dt else self.dt
            U = self.step(U, bathymetry, dt)
            if step % save_every == 0:
                outputs.append(U.copy() if return_state else self.state_to_eta(U, bathymetry))

        result = np.stack(outputs, axis=0).astype(np.float32)
        return result

    def benchmark(self, bathymetry: np.ndarray, eta0: np.ndarray, nt: int, repeats: int = 1) -> Dict[str, float]:
        import time
        durations = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = self.simulate(bathymetry, eta0, nt=nt)
            durations.append(time.perf_counter() - t0)
        durations = np.asarray(durations, dtype=np.float64)
        return {
            "mean_seconds": float(np.mean(durations)),
            "std_seconds": float(np.std(durations)),
            "min_seconds": float(np.min(durations)),
            "max_seconds": float(np.max(durations)),
        }
