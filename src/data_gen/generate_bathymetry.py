import numpy as np
import yaml
from typing import Literal
from scipy.ndimage import gaussian_filter

Type = Literal["trench", "continental", "seamounts", "canyon", "island"]
VALID_TYPES = ("trench", "continental", "seamounts", "canyon", "island")

class BathymetryGenerator:
    def __init__(self, config: str) -> None:
        cfg = {}

        try:
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

        except FileNotFoundError:
            raise FileNotFoundError(f"could not find {config}, is the path correct")

        nx = int(cfg["nx"])
        ny = int(cfg["ny"])

        self.seed = cfg.get("seed", None)

        if self.seed is not None:
            self.seed = abs(int(self.seed))

        self.rng = np.random.default_rng(self.seed)

        if nx <= 1 or ny <= 1:
            raise ValueError("nx and ny most be greater than 1")
        
        self.nx = nx
        self.ny = ny

        self.x = np.linspace(0, 1, self.nx)
        self.y = np.linspace(0, 1, self.ny)

        self.x, self.y = np.meshgrid(self.x, self.y, indexing="ij")

        self.b_type = VALID_TYPES
        
        # small helper
        def _parse_array_int(key: str) -> np.ndarray:
            return np.array(cfg[key], dtype=int)
        
        def _parse_array_float(key: str) -> np.ndarray:
            return np.array(cfg[key], dtype=float)

        # base
        self.slope_range = _parse_array_float("slope_range")

        # gaussians
        self.enabled_g = bool(cfg["enabled_g"])
        self.range_g = _parse_array_int("range_g")
        self.amp_range_g = _parse_array_float("amp_range_g")
        self.sigma_range_g = _parse_array_float("sigma_range_g")

        # ridges
        self.enabled_r = bool(cfg["enabled_r"])
        self.range_r = _parse_array_int("range_r")
        self.amp_range_r = _parse_array_float("amp_range_r")
        self.len_scale_r = _parse_array_float("len_scale_r")

        # noise
        self.enabled_n = bool(cfg["enabled_n"])
        self.scale_range_n = _parse_array_float("scale_range_n")
        self.smoothing_sigma_n = _parse_array_float("smoothing_sigma_n")

        # nomalization
        self.depth_min = float(cfg["depth_min"])
        self.depth_max = float(cfg["depth_max"])

    def terrain_type(self) -> Type:
        return self.rng.choice(self.b_type)

    def generate(self) -> tuple[np.ndarray, Type]:
        t_type = self.terrain_type()
        terrain = self.generate_base()
        terrain = self.apply_bias(terrain, t_type)

        if self.enabled_g:
            terrain = self.add_gaussians(terrain)
        
        if self.enabled_r:
            terrain = self.add_ridges(terrain)
        
        if self.enabled_n:
            terrain = self.add_noise(terrain)

        terrain = self.normalize(terrain)

        return terrain, t_type

    def generate_base(self) -> np.ndarray:
        # simple linear slope from left (shallow) to right (deep)
        slope_x = self.rng.uniform(self.slope_range[0], self.slope_range[1])
        slope_y = self.rng.uniform(self.slope_range[0], self.slope_range[1])
        base = self.x * slope_x + self.y * slope_y

        return base
    
    def apply_bias(self, terrain: np.ndarray, t_type: Type) -> np.ndarray:
        bias_scale = (self.depth_max - self.depth_min) * 0.5

        if t_type == "trench":
            bias = -0.5 * bias_scale * np.exp(-((self.x - 0.5)**2 + (self.y - 0.5)**2) / 0.02)

        elif t_type == "continental":
            bias = 0.5 * bias_scale * np.exp(-((self.x - 0.2)**2 + (self.y - 0.2)**2) / 0.02)

        elif t_type == "seamounts":
            bias = 0.3 * bias_scale * np.exp(-((self.x - 0.7)**2 + (self.y - 0.7)**2) / 0.01)

        elif t_type == "canyon":
            bias = -0.3 * bias_scale * np.exp(-((self.x - 0.3)**2 + (self.y - 0.8)**2) / 0.01)

        elif t_type == "island":
            bias = 0.4 * bias_scale * np.exp(-((self.x - 0.8)**2 + (self.y - 0.3)**2) / 0.01)

        else:
            bias = np.zeros_like(terrain)

        return terrain + bias
    
    def add_gaussians(self, terrain: np.ndarray) -> np.ndarray:
        n_gaussians = self.rng.integers(self.range_g[0], self.range_g[1] + 1)
        
        for _ in range(n_gaussians):
            amp = self.rng.uniform(self.amp_range_g[0], self.amp_range_g[1])
            
            sigma_x = self.rng.uniform(self.sigma_range_g[0], self.sigma_range_g[1])
            sigma_y = self.rng.uniform(self.sigma_range_g[0], self.sigma_range_g[1])
            
            x0 = self.rng.uniform(0, 1)
            y0 = self.rng.uniform(0, 1)
            
            gaussian = amp * np.exp(-(((self.x - x0)**2) / (2 * sigma_x**2) + ((self.y - y0)**2) / (2 * sigma_y**2)))
            terrain += gaussian

        return terrain
    
    def add_ridges(self, terrain: np.ndarray) -> np.ndarray:
        n_ridges = self.rng.integers(self.range_r[0], self.range_r[1] + 1)

        for _ in range(n_ridges):
            amp = self.rng.uniform(self.amp_range_r[0], self.amp_range_r[1])
            len_scale = self.rng.uniform(self.len_scale_r[0], self.len_scale_r[1])

            x0 = self.rng.uniform(0, 1)
            y0 = self.rng.uniform(0, 1)

            angle = self.rng.uniform(0, 2 * np.pi)

            X_rot = (self.x - x0) * np.cos(angle) + (self.y - y0) * np.sin(angle)
            
            ridge = amp * np.exp(-(X_rot**2) / (2 * len_scale**2))
            terrain += ridge

        return terrain
    
    def add_noise(self, terrain: np.ndarray) -> np.ndarray:
        scale = self.rng.uniform(self.scale_range_n[0], self.scale_range_n[1])
        noise = scale * self.rng.standard_normal(terrain.shape)

        sigma_val = self.rng.uniform(self.smoothing_sigma_n[0], self.smoothing_sigma_n[1])
        smoothed_noise = gaussian_filter(noise, sigma=sigma_val)

        return terrain + smoothed_noise
    
    def normalize(self, terrain: np.ndarray) -> np.ndarray:
        denom = terrain.max() - terrain.min()

        if denom == 0:
            return np.full_like(terrain, self.depth_min)
        
        terrain = (terrain - terrain.min()) / (denom)
        terrain = terrain * (self.depth_max - self.depth_min) + self.depth_min

        return terrain

if __name__ == "__main__":
    generator = BathymetryGenerator("configs/test-config.yaml")

    for i in range(3):
        bathymetry, t_type = generator.generate()
        np.save(f"data/raw/bathymetry_{i}.npy", bathymetry)
        np.save(f"data/raw/type_{i}.npy", np.array(t_type), dtype="U16")