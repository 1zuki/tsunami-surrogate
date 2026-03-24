import numpy as np
from typing import Literal

Type = Literal["trench", "continental", "seamounts", "canyon", "island"]
VALID_TYPES = ("trench", "continental", "seamounts", "canyon", "island")

class BathymetryGenerator:
    def __init__(self, config: str) -> None:
        cfg = {}

        try:
            with open(config, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(":", 1)
                        
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            cfg[key] = value

        except FileNotFoundError:
            raise FileNotFoundError(f"could not find {config}, is the path correct")

        nx = int(cfg["nx"])
        ny = int(cfg["ny"])

        if cfg["seed"] == "null":
            self.seed = None
        else:
            self.seed = abs(int(cfg["seed"]))

        self.rng = np.random.default_rng(self.seed)

        if nx <= 1 or ny <= 1:
            raise ValueError("nx and ny most be greater than 1")
        
        self.nx = nx
        self.ny = ny

        self.b_type = VALID_TYPES
        
        # small helper
        def _parse_array_int(key: str) -> np.ndarray:
            return np.array([int(x) for x in cfg[key].split(",")], dtype=int)
        
        def _parse_array_float(key: str) -> np.ndarray:
            return np.array([float(x) for x in cfg[key].split(",")], dtype=float)

        # base
        self.slope_range = _parse_array_float("slope_range")

        # gaussians
        self.enabled_g = cfg["enabled_g"].lower() == "true"
        self.range_g = _parse_array_int("range_g")
        self.amp_range_g = _parse_array_float("amp_range_g")
        self.sigma_range_g = _parse_array_float("sigma_range_g")

        # ridges
        self.enabled_r = cfg["enabled_r"].lower() == "true"
        self.range_r = _parse_array_int("range_r")
        self.amp_range_r = _parse_array_float("amp_range_r")
        self.len_scale_r = _parse_array_float("len_scale_r")

        # noise
        self.enabled_n = cfg["enabled_n"].lower() == "true"
        self.scale_range_n = _parse_array_float("scale_range_n")
        self.smoothing_sigma_n = _parse_array_float("smoothing_sigma_n")

        # nomalization
        self.depth_min = float(cfg["depth_min"])
        self.depth_max = float(cfg["depth_max"])

    def terrain_type(self) -> Type:
        return self.rng.choice(self.b_type)

    def generate(self):
        # 3 terrian type per config
        for _ in range(3):
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

            yield terrain
            
    def generate_base(self) -> np.ndarray:
        # simple linear slope from left (shallow) to right (deep)
        x = np.linspace(0, 1, self.nx)
        slope = self.rng.uniform(self.slope_range[0], self.slope_range[1])
        base = np.outer(x * slope, np.ones(self.ny))
        return base
    
    def apply_bias(self, terrain: np.ndarray, t_type: Type) -> np.ndarray:
        if t_type == "trench":
            bias = -50 * np.exp(-((np.linspace(0, 1, self.nx) - 0.5)**2) / 0.02)

        elif t_type == "continental":
            bias = 20 * np.exp(-((np.linspace(0, 1, self.nx) - 0.2)**2) / 0.01)

        elif t_type == "seamounts":
            bias = 10 * np.exp(-((np.linspace(0, 1, self.nx) - 0.7)**2) / 0.01)

        elif t_type == "canyon":
            bias = -30 * np.exp(-((np.linspace(0, 1, self.nx) - 0.3)**2) / 0.005)

        elif t_type == "island":
            bias = 15 * np.exp(-((np.linspace(0, 1, self.nx) - 0.8)**2) / 0.02)

        else:
            bias = np.zeros(self.nx)

        return terrain + np.outer(bias, np.ones(self.ny))
    
    def add_gaussians(self, terrain: np.ndarray) -> np.ndarray:
        n_gaussians = self.rng.integers(self.range_g[0], self.range_g[1] + 1)
        
        for _ in range(n_gaussians):
            amp = self.rng.uniform(self.amp_range_g[0], self.amp_range_g[1])
            
            sigma_x = self.rng.uniform(self.sigma_range_g[0], self.sigma_range_g[1])
            sigma_y = self.rng.uniform(self.sigma_range_g[0], self.sigma_range_g[1])
            
            x0 = self.rng.uniform(0, 1)
            y0 = self.rng.uniform(0, 1)

            x = np.linspace(0, 1, self.nx)
            y = np.linspace(0, 1, self.ny)
            
            X, Y = np.meshgrid(x, y, indexing="ij")
            
            gaussian = amp * np.exp(-(((X - x0)**2) / (2 * sigma_x**2) + ((Y - y0)**2) / (2 * sigma_y**2)))
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

            x = np.linspace(0, 1, self.nx)
            y = np.linspace(0, 1, self.ny)
            
            X, Y = np.meshgrid(x, y, indexing="ij")
            X_rot = (X - x0) * np.cos(angle) + (Y - y0) * np.sin(angle)
            
            ridge = amp * np.exp(-(X_rot**2) / (2 * len_scale**2))
            terrain += ridge
        
        return terrain
    
    def add_noise(self, terrain: np.ndarray) -> np.ndarray:
        scale = self.rng.uniform(self.scale_range_n[0], self.scale_range_n[1])
        noise = scale * self.rng.standard_normal(terrain.shape)

        from scipy.ndimage import gaussian_filter

        sigma_val = self.rng.uniform(self.smoothing_sigma_n[0], self.smoothing_sigma_n[1])
        smoothed_noise = gaussian_filter(noise, sigma=sigma_val)

        return terrain + smoothed_noise
    
    def normalize(self, terrain: np.ndarray) -> np.ndarray:
        terrain = np.clip(terrain, self.depth_min, self.depth_max)
        return terrain

if __name__ == "__main__":
    generator = BathymetryGenerator("configs/config.txt")
    for i, bathymetry in enumerate(generator.generate()):
        np.save(f"data/raw/bathymetry_{i}.npy", bathymetry)