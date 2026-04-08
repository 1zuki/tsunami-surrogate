from __future__ import annotations

import numpy as np

from src.solver.shallow_water import ShallowWaterSolver


class WeaklyDispersiveSolver(ShallowWaterSolver):
    """A lightweight weakly dispersive extension of the shallow-water solver.

    This is not a full high-fidelity Boussinesq implementation. It adds a small dispersive
    correction through gradients of the surface-elevation Laplacian, which is useful for
    research experiments and multi-fidelity surrogate studies.
    """

    def __init__(self, dispersion_coeff: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.dispersion_coeff = float(dispersion_coeff)

    @classmethod
    def from_config(cls, config: dict) -> "WeaklyDispersiveSolver":
        sim = config.get("simulation", {})
        return cls(
            dispersion_coeff=float(sim.get("dispersion_coeff", 0.01)),
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

    def _dispersion_term(self, U: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
        eta = self.state_to_eta(U, bathymetry)
        d_eta_dy, d_eta_dx = np.gradient(eta, self.dy, self.dx)
        d2x = np.gradient(d_eta_dx, self.dx, axis=1)
        d2y = np.gradient(d_eta_dy, self.dy, axis=0)
        lap = d2x + d2y
        dlap_dy, dlap_dx = np.gradient(lap, self.dy, self.dx)
        h = np.maximum(U[0], self.min_depth)
        correction = np.zeros_like(U, dtype=np.float32)
        correction[1] = self.dispersion_coeff * h * dlap_dx
        correction[2] = self.dispersion_coeff * h * dlap_dy
        return correction

    def rhs(self, U: np.ndarray, bathymetry: np.ndarray) -> np.ndarray:
        rhs = super().rhs(U, bathymetry)
        rhs += self._dispersion_term(U, bathymetry)
        return rhs.astype(np.float32)
