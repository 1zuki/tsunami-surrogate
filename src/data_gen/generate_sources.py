import numpy as np
import yaml
from typing import Literal, Optional

Type = Literal["gaussian", "multi-gauss", "okada-like", "dipole", "fault", "rough"]
VALID_TYPES = ("gaussian", "multi-gauss", "okada-like", "dipole", "fault", "rough")

class SourceGenerator:
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

        self.s_type = cfg.get("source_type", VALID_TYPES)

        def _parse_array_int(host: str, key: str, default: Optional[list[int, int]] = None) -> np.ndarray:
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
        
        def _parse_array_float(host: str, key: str, default: Optional[list[float, float]] = None) -> np.ndarray:
            section = cfg.get(host, {})
            value = section.get(key, default)

            if value is None:
                raise KeyError(f"missing config key: {key}")
            
            arr = np.array(value, dtype=float)

            if arr.size != 2:
                raise ValueError(f"{key} must have 2 values [min, max]")
            
            if arr[0] > arr[1]:
                raise ValueError(f"{key} must have min <= max")

            return arr
        
        # gaussian
        self.enabled_g = bool(cfg.get("gaussian").get("enabled", True))
        self.amp_range_g = _parse_array_float("gaussian", "amp_range", [0.2, 2.0])
        self.sigma_range_g = _parse_array_float("gaussian", "sigma_range", [0.01, 0.08])
        self.num_range_g = _parse_array_int("gaussian", "num_range", [1, 3])

        # multi gaussian
        self.enabled_mg = bool(cfg.get("multi").get("enabled", True))
        self.num_sources = _parse_array_int("multi", "num_sources", [1, 3])

        # dipole
        self.enabled_d = bool(cfg.get("dipole").get("enabled", True))
        self.amp_range_d = _parse_array_float("dipole", "amp_range", [0.5, 2.5])
        self.sigma_range_d = _parse_array_float("dipole", "sigma_range", [0.02, 0.08])
        self.sep_range_d = _parse_array_float("dipole", "separation_range", [0.05, 0.15])
        self.angle_range_d = _parse_array_float("dipole", "angle_range", [0.0, 3.14])

        # fault
        self.enabled_f = bool(cfg.get("fault").get("enabled", True))
        self.amp_range_f = _parse_array_float("fault", "amp_range", [0.5, 2.0])
        self.len_range_f = _parse_array_float("fault", "length_range", [0.2, 0.6])
        self.width_range_f = _parse_array_float("fault", "width_range", [0.02, 0.08])
        self.angle_range_f = _parse_array_float("fault", "angle_range", [0.0, 3.14])
        self.smoothing_sigma_f = _parse_array_float("fault", "smoothing_sigma", [0.01, 0.03])

        # rough
        self.enabled_r = bool(cfg.get("rough").get("enabled", True))
        self.amp_range_r = _parse_array_float("rough", "amp_range", [0.5, 2.0])
        self.smoothing_sigma = _parse_array_float("rough", "smoothing_sigma", [1.0, 3.0])

        # okada-like
        self.enabled_o = bool(cfg.get("okada").get("enabled", True))
        self.len_range_o = _parse_array_float("okada", "length_range", [0.1, 0.4])
        self.width_range_o = _parse_array_float("okada", "width_range", [0.05, 0.2])
        self.slip_range_o = _parse_array_float("okada", "slip_range", [0.5, 2.0])
        self.angle_range_o = _parse_array_float("okada", "angle_range", [0.0, 3.14])

        # noise
        self.enabled_n = bool(cfg.get("noise").get("enabled", True))
        self.scale_range_n = _parse_array_float("noise", "scale_range", [0.01, 0.05])
        self.smoothing_sigma_n = _parse_array_float("noise", "smoothing_sigma", [1.0, 3.0])

        # normalization
        self.height_scale = _parse_array_float("normalization", "height_scale", [-1.0, 1.0])

    def source_type(self) -> Type:
        available = {}

        if self.enabled_g:
            available.append("gaussian")
        
        if self.enabled_mg:
            available.append("multi-gauss")

        if self.enabled_d:
            available.append("dipole")
        
        if self.enabled_f:
            available.append("fault")

        if self.enabled_r:
            available.append("rough")
        
        if self.enabled_o:
            available.append("okada-like")

        return self.rng.choice(available)
    
    def generate(self) -> tuple[np.ndarray, str]:
        """ type -> gen -> warp -> normalize -> return """
        s_type = self.source_type()

        if s_type == "gaussian":
            pass

        elif s_type == "multi-gauss":
            pass

        elif s_type == "dipole":
            pass

        elif s_type == "fault":
            pass

        elif s_type == "rough":
            pass

        else: # okada-like
            pass

    # helper
    def _gaussian_2d(self, x0, y0, sigma_x, sigma_y, amp):
        pass

    def _rotate(self, X, Y, x0, y0, angle):
        pass

    def _sample(self, range_array):
        pass

    # generator
    def _gen_gaussian(self) -> np.ndarray:
        pass

    def _gen_multi_gaussian(self):
        pass

    def _gen_dipole(self):
        pass

    def _gen_fault(self):
        pass

    def _gen_okada_like(self):
        pass

    def _gen_rough(self):
        pass

    def add_noise(self, h):
        pass

    def normalize(self, h):
        pass

    def build_output(self, h):
        pass