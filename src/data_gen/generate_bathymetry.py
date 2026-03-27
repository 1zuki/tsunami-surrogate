import numpy as np
import yaml
from typing import Literal, Optional
from scipy.ndimage import gaussian_filter

Type = Literal["trench", "continental", "seamounts", "canyon", "island"]
VALID_TYPES = ("trench", "continental", "seamounts", "canyon", "island")

Base = Literal["slope", "flat", "basin"]
VALID_BASE = ("slope", "flat", "basin")

class BathymetryGenerator:
    def __init__(self, config: str) -> None:
        cfg = {}

        try:
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"could not find {config}, is the path correct")
        
        if cfg is None:
            raise ValueError("yaml config is empty/invalid")

        self.nx = int(cfg["nx"])
        self.ny = int(cfg["ny"])

        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("nx and ny must be greater than 1")

        self.seed = cfg.get("seed", None)

        if self.seed is not None:
            self.seed = abs(int(self.seed))

        self.rng = np.random.default_rng(self.seed)

        self.x = np.linspace(0, 1, self.nx)
        self.y = np.linspace(0, 1, self.ny)

        self.x, self.y = np.meshgrid(self.x, self.y, indexing="ij")

        self.b_type = VALID_TYPES
        
        # small helper
        def _parse_array_int(host: str, key: str, default: Optional[list[int]] = None) -> np.ndarray:
            section = cfg.get(host, {})
            value = section.get(key, default)

            if value is None:
                raise KeyError(f"missing config key: {key}")
            
            arr = np.array(value, dtype=int)

            if arr.size != 2:
                raise ValueError(f"{key} must have 2 values [min, max]")
            
            if arr[0] > arr[1]:
                raise ValueError(f"{key} must have min <= max")

            return arr
        
        def _parse_array_float(host: str, key: str, default: Optional[list[float]]) -> np.ndarray:
            section = cfg.get(host, {})
            value = section.get(key, default)

            if value is None:
                raise KeyError(f"missing config key: {key}")
            
            arr = np.array(value, dtype=float)

            if arr.size!= 2:
                raise ValueError(f"{key} must have 2 values [min, max]")
            
            if arr[0] > arr[1]:
                raise ValueError(f"{key} must have min <= max")
            
            return arr
        
        # base
        self.slope_range = _parse_array_float("base", "slope_range", [0.0, 0.15])
        self.base_kind = cfg.get("base").get("kind", "slope") # slope / flat / basin

        if self.base_kind not in VALID_BASE:
            raise ValueError("Base must be slope/flat/basin")

        # gaussians
        self.enabled_g = bool(cfg.get("gaussian").get("enabled", True))
        self.range_g = _parse_array_int("gaussian", "range", [1, 4])
        self.amp_range_g = _parse_array_float("gaussian", "amp_range", [-0.4, 0.4])
        self.sigma_range_g = _parse_array_float("gaussian", "sigma_range", [0.01, 0.08])

        # ridges
        self.enabled_r = bool(cfg.get("ridges").get("enabled", True))
        self.range_r = _parse_array_int("ridges", "range", [0, 3])
        self.amp_range_r = _parse_array_float("ridges", "amp_range", [-0.3, 0.3])
        self.len_scale_r = _parse_array_float("ridges", "len_scale", [0.02, 0.15])

        # noise
        self.enabled_n = bool(cfg.get("noise").get("enabled", True))
        self.scale_range_n = _parse_array_float("noise", "scale_range", [0.01, 0.08])
        self.smoothing_sigma_n = _parse_array_float("noise", "smoothing_sigma", [1.5, 4.5])

        # nomalization
        self.depth_min = float(cfg.get("normalization").get("depth_min", -5.0))
        self.depth_max = float(cfg.get("normalization").get("depth_max", 0.0))

        # extra config for terrain control
        self.warp_scale = float(cfg.get("terrain").get("warp_scale", 0.08))
        self.warp_sigma = float(cfg.get("terrain").get("warp_sigma", 3.0))
        self.bias_strength = float(cfg.get("terrain").get("bias_strength", 1.0))

    def terrain_type(self) -> Type:
        return self.rng.choice(self.b_type)

    def generate(self) -> tuple[np.ndarray, Type]:
        t_type = self.terrain_type()
        terrain = self.generate_base()
        
        # warping -> avoid axis aligned struct
        warp_x = gaussian_filter(self.rng.standard_normal((self.nx, self.ny)), sigma=self.warp_sigma) * self.warp_scale
        warp_y = gaussian_filter(self.rng.standard_normal((self.nx, self.ny)), sigma=self.warp_sigma) * self.warp_scale

        w_x = np.clip(self.x + warp_x, 0.0, 1.0)
        w_y = np.clip(self.y + warp_y, 0.0, 1.0)

        terrain = self.apply_bias(terrain, t_type, w_x, w_y)

        if self.enabled_g:
            terrain = self.add_gaussians(terrain, t_type)

        if self.enabled_r:
            terrain = self.add_ridges(terrain, t_type)

        if self.enabled_n:
            terrain = self.add_noise(terrain)

        terrain = self.normalize(terrain)
        terrain = np.clip(terrain, self.depth_min, self.depth_max)

        return terrain, t_type

    def generate_base(self) -> np.ndarray:
        if self.base_kind == "flat":
            base = np.zeros((self.nx, self.ny), dtype=float)

        elif self.base_kind == "basin":
            cx = 0.5 + self.rng.uniform(-0.08, 0.08)
            cy = 0.5 + self.rng.uniform(-0.08, 0.08)

            r2 = (self.x - cx) ** 2 + (self.y - cy) ** 2
            base = 0.12 * np.exp(-r2 / 0.025)

        else: # slope
            slope_x = self.rng.uniform(self.slope_range[0], self.slope_range[1])
            slope_y = self.rng.uniform(self.slope_range[0], self.slope_range[1])

            angle = self.rng.uniform(0, 2 * np.pi)

            x_r = (self.x - 0.5) * np.cos(angle) + (self.y - 0.5) * np.sin(angle)
            y_r = -(self.x - 0.5) * np.sin(angle) + (self.y - 0.5) * np.cos(angle)

            base = slope_x * x_r + slope_y * y_r

        return base

    def apply_bias(self, terrain: np.ndarray, t_type: Type, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """ type specific features -> add bias"""
        bias_scale = (self.depth_max - self.depth_min) * 0.5 * self.bias_strength

        if t_type == "trench":
            terrain += self._add_trench_system(X, Y, bias_scale)

        elif t_type == "continental":
            terrain += self._add_continental_shelf(X, Y, bias_scale)

        elif t_type == "seamounts":
            terrain += self._add_seamount_field(X, Y, bias_scale)
        
        elif t_type == "canyon":
            terrain += self._add_canyon_system(X, Y, bias_scale)

        else: # island
            terrain += self._add_island_cluster(X, Y, bias_scale)

        return terrain
    
    def _add_trench_system(self, X: np.ndarray, Y: np.ndarray, scale: float) -> np.ndarray:
        trench = np.zeros_like(X)
        n_trenches = self.rng.integers(1, 3)

        for _ in range(n_trenches):
            x0, y0 = self.rng.uniform(0.2, 0.8, size=2)

            angle = self.rng.uniform(0, np.pi)
            length = self.rng.uniform(0.25, 0.7)
            width = self.rng.uniform(0.01, 0.05)
            depth = self.rng.uniform(0.3, 0.9) * scale

            Xc = X - x0
            Yc = Y - y0
            u = Xc * np.cos(angle) + Yc * np.sin(angle)
            v = -Xc * np.sin(angle) + Yc * np.cos(angle)

            # finite envelope along the trench length
            envelope_u = np.exp(-(u ** 2) / (2 * (length * 0.5) ** 2))

            # slightly curved centerline
            curve = 0.03 * np.sin(6 * np.pi * u / max(length, 1e-6))
            trench += -depth * envelope_u * np.exp(-((v - curve) ** 2) / (2 * width ** 2))

        return trench
    

    def _add_canyon_system(self, X: np.ndarray, Y: np.ndarray, scale: float) -> np.ndarray:
        canyon = np.zeros_like(X)

        n_canyons = self.rng.integers(1, 4)
        for _ in range(n_canyons):
            x0, y0 = self.rng.uniform(0.15, 0.85, size=2)

            angle = self.rng.uniform(0, 2 * np.pi)
            length = self.rng.uniform(0.25, 0.8)
            width = self.rng.uniform(0.005, 0.03)
            depth = self.rng.uniform(0.15, 0.75) * scale

            Xc = X - x0
            Yc = Y - y0
            u = Xc * np.cos(angle) + Yc * np.sin(angle)
            v = -Xc * np.sin(angle) + Yc * np.cos(angle)

            envelope_u = np.exp(-(u ** 2) / (2 * (length * 0.45) ** 2))
            centerline = 0.02 * np.sin(4 * np.pi * u / max(length, 1e-6))

            canyon += -depth * envelope_u * np.exp(-((v - centerline) ** 2) / (2 * width ** 2))

        return canyon

    def _add_island_cluster(self, X: np.ndarray, Y: np.ndarray, scale: float) -> np.ndarray:
        island = np.zeros_like(X)

        # main island
        x0, y0 = self.rng.uniform(0.25, 0.75, size=2)
        main_amp = self.rng.uniform(0.6, 1.0) * scale
        main_sigma = self.rng.uniform(0.05, 0.12)
        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        island += main_amp * np.exp(-r2 / (2 * main_sigma ** 2))

        # satellites / islets
        n_islets = self.rng.integers(2, 7)
        for _ in range(n_islets):
            dx, dy = self.rng.normal(0.0, 0.08, size=2)
            amp = self.rng.uniform(0.1, 0.4) * scale
            sigma = self.rng.uniform(0.015, 0.06)

            cx = np.clip(x0 + dx, 0.0, 1.0)
            cy = np.clip(y0 + dy, 0.0, 1.0)

            r2i = (X - cx) ** 2 + (Y - cy) ** 2
            island += amp * np.exp(-r2i / (2 * sigma ** 2))

        # avoid symmetry
        island += 0.12 * scale * np.exp(-(((X - x0) * 1.8) ** 2 + ((Y - y0) * 0.8) ** 2) / (2 * 0.08 ** 2))

        return island

    def _add_seamount_field(self, X: np.ndarray, Y: np.ndarray, scale: float) -> np.ndarray:
        field = np.zeros_like(X)

        n_peaks = self.rng.integers(8, 20)
        for _ in range(n_peaks):
            cx, cy = self.rng.uniform(0.05, 0.95, size=2)
            amp = self.rng.uniform(0.15, 0.7) * scale
            sigma = self.rng.uniform(0.01, 0.05)
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            field += amp * np.exp(-r2 / (2 * sigma ** 2))

        return field

    def _add_continental_shelf(self, X: np.ndarray, Y: np.ndarray, scale: float) -> np.ndarray:
        shelf = np.zeros_like(X)

        # broad slope
        angle = self.rng.uniform(0, 2 * np.pi)
        u = (X - 0.5) * np.cos(angle) + (Y - 0.5) * np.sin(angle)
        slope_strength = self.rng.uniform(0.2, 0.6) * scale
        shelf += slope_strength * (1.0 / (1.0 + np.exp(-12 * (u - 0.15))))

        # shelf break
        break_pos = self.rng.uniform(0.15, 0.45)
        shelf += 0.25 * scale * np.exp(-((u - break_pos) ** 2) / (2 * 0.02 ** 2))

        # gentle ridges
        for _ in range(self.rng.integers(1, 4)):
            cx, cy = self.rng.uniform(0.1, 0.9, size=2)
            amp = self.rng.uniform(0.05, 0.2) * scale
            sigma = self.rng.uniform(0.03, 0.08)
            shelf += amp * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)))

        return shelf

    def add_gaussians(self, terrain: np.ndarray, t_type: Type) -> np.ndarray:
        n_gaussians = self.rng.integers(self.range_g[0], self.range_g[1] + 1)

        for _ in range(n_gaussians):
            # amp bias for types
            if t_type in ("trench", "canyon"):
                amp = self.rng.uniform(self.amp_range_g[0], 0.0)

            elif t_type in ("island", "seamounts", "continental"):
                amp = self.rng.uniform(0.0, self.amp_range_g[1])

            else:
                amp = self.rng.uniform(self.amp_range_g[0], self.amp_range_g[1])

            sigma_x = self.rng.uniform(self.sigma_range_g[0], self.sigma_range_g[1])
            sigma_y = self.rng.uniform(self.sigma_range_g[0], self.sigma_range_g[1])

            x0 = self.rng.uniform(0.0, 1.0)
            y0 = self.rng.uniform(0.0, 1.0)

            gaussian = amp * np.exp(-(((self.x - x0) ** 2) / (2 * sigma_x ** 2)
                                      + ((self.y - y0) ** 2) / (2 * sigma_y ** 2)))
            terrain += gaussian

        return terrain
    
    def add_ridges(self, terrain: np.ndarray, t_type: Type) -> np.ndarray:
        n_ridges = self.rng.integers(self.range_r[0], self.range_r[1] + 1)

        for _ in range(n_ridges):
            amp = self.rng.uniform(self.amp_range_r[0], self.amp_range_r[1])
            len_scale = self.rng.uniform(self.len_scale_r[0], self.len_scale_r[1])

            x0 = self.rng.uniform(0.0, 1.0)
            y0 = self.rng.uniform(0.0, 1.0)
            angle = self.rng.uniform(0, 2 * np.pi)

            X_rot = (self.x - x0) * np.cos(angle) + (self.y - y0) * np.sin(angle)

            # type-specific ridge bias
            if t_type in ("canyon", "trench"):
                ridge_amp = -abs(amp)
            elif t_type in ("island", "seamounts"):
                ridge_amp = abs(amp)
            else:
                ridge_amp = amp

            ridge = ridge_amp * np.exp(-(X_rot ** 2) / (2 * len_scale ** 2))
            terrain += ridge

        return terrain
    
    def add_noise(self, terrain: np.ndarray) -> np.ndarray:
        # low freq roughness
        scale = self.rng.uniform(self.scale_range_n[0], self.scale_range_n[1])
        sigma_val = self.rng.uniform(self.smoothing_sigma_n[0], self.smoothing_sigma_n[1])

        noise = scale * self.rng.standard_normal(terrain.shape)
        rough = gaussian_filter(noise, sigma=sigma_val)
        terrain = terrain + rough

        # smaller scale layer to make terrain less smooth
        micro_scale = 0.35 * scale
        micro_noise = micro_scale * self.rng.standard_normal(terrain.shape)
        micro_rough = gaussian_filter(micro_noise, sigma=max(0.5, 0.5 * sigma_val))

        terrain = terrain + micro_rough

        return terrain
    
    def normalize(self, terrain: np.ndarray) -> np.ndarray:
        """ map into [depth_min, depth_max], preverse shape"""
        tmin = np.min(terrain)
        tmax = np.max(terrain)

        if np.isclose(tmax, tmin):
            return np.full_like(terrain, self.depth_min)

        terrain = (terrain - tmin) / (tmax - tmin)
        terrain = terrain * (self.depth_max - self.depth_min) + self.depth_min
        return terrain

if __name__ == "__main__":
    generator = BathymetryGenerator("configs/bathymetry-gen-test.yaml")

    for i in range(10):
        bathymetry, t_type = generator.generate()
        np.save(f"data/raw/bathymetry_{i + 1}.npy", bathymetry)
        np.save(f"data/raw/type_{i + 1}.npy", np.array(t_type))

"""
References:

[1] LeVeque, R. J. (2002)
Finite Volume Methods for Hyperbolic Problems

[2] Titov, V. V., & Synolakis, C. E. (1998)
Numerical modeling of tidal wave runup

[3] Musgrave, F. K., Kolb, C. E., & Mace, R. S. (1989)
The synthesis and rendering of eroded fractal terrains

[4] Ebert, D. S., et al. (2003)
Texturing and Modeling: A Procedural Approach

[5] Rasmussen, C. E., & Williams, C. K. I. (2006)
Gaussian Processes for Machine Learning

[6] Sandwell, D. T., et al. (2014)
Marine gravity and bathymetry

[7] Toro, E. F. (2009)
Riemann Solvers and Numerical Methods for Fluid Dynamics
"""