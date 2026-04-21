import numpy as np
import yaml
from typing import Literal, Optional, Tuple
from scipy.ndimage import gaussian_filter

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

        requested_types = cfg.get("source_type", list(VALID_TYPES))
        self.s_type = self._parse_source_type(requested_types)

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

        def _parse_array_float_direct(value: object, key: str) -> np.ndarray:
            arr = np.array(value, dtype=float)
            if arr.size != 2:
                raise ValueError(f"{key} must have 2 values [min, max]")
            if arr[0] > arr[1]:
                raise ValueError(f"{key} must have min <= max")
            return arr
        
        # gaussian
        self.enabled_g = bool(cfg.get("gaussian", {}).get("enabled", True))
        self.amp_range_g = _parse_array_float("gaussian", "amp_range", [0.2, 2.0])
        self.sigma_range_g = _parse_array_float("gaussian", "sigma_range", [0.01, 0.08])
        self.num_range_g = _parse_array_int("gaussian", "num_range", [1, 3])

        # multi gaussian
        self.enabled_mg = bool(cfg.get("multi", {}).get("enabled", True))
        self.num_sources = _parse_array_int("multi", "num_sources", [1, 3])

        # dipole
        self.enabled_d = bool(cfg.get("dipole", {}).get("enabled", True))
        self.amp_range_d = _parse_array_float("dipole", "amp_range", [0.5, 2.5])
        self.sigma_range_d = _parse_array_float("dipole", "sigma_range", [0.02, 0.08])
        self.sep_range_d = _parse_array_float("dipole", "separation_range", [0.05, 0.15])
        self.angle_range_d = _parse_array_float("dipole", "angle_range", [0.0, 3.14])

        # fault
        self.enabled_f = bool(cfg.get("fault", {}).get("enabled", True))
        self.amp_range_f = _parse_array_float("fault", "amp_range", [0.5, 2.0])
        self.len_range_f = _parse_array_float("fault", "length_range", [0.2, 0.6])
        self.width_range_f = _parse_array_float("fault", "width_range", [0.02, 0.08])
        self.angle_range_f = _parse_array_float("fault", "angle_range", [0.0, 3.14])
        self.smoothing_sigma_f = _parse_array_float("fault", "smoothing_sigma", [0.01, 0.03])

        # rough
        self.enabled_r = bool(cfg.get("rough", {}).get("enabled", True))
        self.amp_range_r = _parse_array_float("rough", "amp_range", [0.5, 2.0])
        self.smoothing_sigma_r = _parse_array_float("rough", "smoothing_sigma", [1.0, 3.0])

        # okada-like
        self.enabled_o = bool(cfg.get("okada", {}).get("enabled", True))
        self.len_range_o = _parse_array_float("okada", "length_range", [0.1, 0.4])
        self.width_range_o = _parse_array_float("okada", "width_range", [0.05, 0.2])
        self.slip_range_o = _parse_array_float("okada", "slip_range", [0.5, 2.0])
        self.angle_range_o = _parse_array_float("okada", "angle_range", [0.0, 3.14])
        self.dip_range_o = _parse_array_float("okada", "dip_range", [0.0, 0.5])
        self.depth_range_o = _parse_array_float("okada", "depth_range", [0.0, 0.2])
        self.smoothing_sigma_o = _parse_array_float("okada", "smoothing_sigma", [0.01, 0.03])

        # noise
        self.enabled_n = bool(cfg.get("noise", {}).get("enabled", True))
        self.scale_range_n = _parse_array_float("noise", "scale_range", [0.01, 0.05])
        self.smoothing_sigma_n = _parse_array_float("noise", "smoothing_sigma", [1.0, 3.0])

        # normalization / output shaping
        norm_cfg = cfg.get("normalization", {})
        legacy_height_scale = cfg.get("height_scale", None)
        if "height_scale" in norm_cfg:
            self.height_scale = _parse_array_float("normalization", "height_scale", [-1.0, 1.0])
        elif legacy_height_scale is not None:
            self.height_scale = _parse_array_float_direct(legacy_height_scale, "height_scale")
        else:
            self.height_scale = np.array([-1.0, 1.0], dtype=float)

        self.normalize_mode = str(norm_cfg.get("mode", "none")).strip().lower()
        if self.normalize_mode not in ("none", "per_sample"):
            raise ValueError("normalization.mode must be one of: none, per_sample")

        self.clip_output = bool(norm_cfg.get("clip_output", self.normalize_mode == "per_sample"))

    @staticmethod
    def _parse_source_type(source_type: object) -> tuple[str, ...]:
        if isinstance(source_type, str):
            requested = [source_type]

        elif isinstance(source_type, (list, tuple)):
            requested = [str(s) for s in source_type]

        else:
            raise ValueError("source_type must be a string or a list of strings")

        if not requested:
            raise ValueError("source_type cannot be empty")

        invalid = [s for s in requested if s not in VALID_TYPES]
        if invalid:
            raise ValueError(f"unknown source type(s): {invalid}; valid: {VALID_TYPES}")

        # preserve order while removing duplicates
        return tuple(dict.fromkeys(requested))

    def source_type(self) -> Type:
        enabled = []

        if self.enabled_g:
            enabled.append("gaussian")
        
        if self.enabled_mg:
            enabled.append("multi-gauss")

        if self.enabled_d:
            enabled.append("dipole")
        
        if self.enabled_f:
            enabled.append("fault")

        if self.enabled_r:
            enabled.append("rough")
        
        if self.enabled_o:
            enabled.append("okada-like")

        available = [s for s in self.s_type if s in enabled]

        if not available:
            raise ValueError("no source types are available (check source_type and enabled flags)")

        return self.rng.choice(available)
    
    def generate(self) -> tuple[np.ndarray, str]:
        """ type -> gen -> warp -> normalize -> return """
        s_type = self.source_type()

        if s_type == "gaussian":
            src = self._gen_gaussian()

        elif s_type == "multi-gauss":
            src = self._gen_multi_gaussian()

        elif s_type == "dipole":
            src = self._gen_dipole()

        elif s_type == "fault":
            src = self._gen_fault()

        elif s_type == "rough":
            src = self._gen_rough()

        else: # okada-like
            src = self._gen_okada_like()

        src = self.build_output(src)

        return src, s_type

    # helper
    def _gaussian_2d(self, x0: float, y0: float, sigma_x: float, sigma_y: float, amp: float) -> np.ndarray:
        x = (self.x - x0) ** 2 / (2 * sigma_x ** 2)
        y = (self.y - y0) ** 2 / (2 * sigma_y ** 2)

        return amp * np.exp(-(x + y))

    def _rotate(self, X: np.ndarray, Y: np.ndarray, x0: float, y0: float, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """ rotate coordinates (X, Y) about centre (x0, y0) in radians """
        X_c = X - x0
        Y_c = Y - y0

        X_r = X_c * np.cos(angle) + Y_c * np.sin(angle)
        Y_r = -X_c * np.sin(angle) + Y_c * np.cos(angle)

        return X_r + x0, Y_r + y0

    def _sample(self, range_array: np.ndarray) -> np.ndarray:
        return self.rng.uniform(range_array[0], range_array[1])

    # generator
    def _gen_gaussian(self) -> np.ndarray:
        n_gaussian = self.rng.integers(self.num_range_g[0], self.num_range_g[1] + 1)
        src = np.zeros((self.nx, self.ny))

        for _ in range(n_gaussian):
            x0 = self.rng.uniform(0.0, 1.0)
            y0 = self.rng.uniform(0.0, 1.0)

            sigma_y = self._sample(self.sigma_range_g)
            sigma_x = self._sample(self.sigma_range_g)

            amp = self._sample(self.amp_range_g)

            src += self._gaussian_2d(x0, y0, sigma_x, sigma_y, amp)

        return src

    def _gen_multi_gaussian(self) -> np.ndarray:
        n_multi_gauss = self.rng.integers(self.num_sources[0], self.num_sources[1] + 1)
        field = np.zeros((self.nx, self.ny))

        for _ in range(n_multi_gauss):
            field += self._gen_gaussian()

        return field

    def _gen_dipole(self) -> np.ndarray:
        x_c = self.rng.uniform(0.0, 1.0)
        y_c = self.rng.uniform(0.0, 1.0)

        sep = self._sample(self.sep_range_d)
        angle = self._sample(self.angle_range_d)
        sigma_x = self._sample(self.sigma_range_d)
        sigma_y = self._sample(self.sigma_range_d)
        amp = self._sample(self.amp_range_d)

        dx = (sep / 2.0) * np.cos(angle)
        dy = (sep / 2.0) * np.sin(angle)

        x_1 = x_c + dx
        y_1 = y_c + dy
        x_2 = x_c - dx
        y_2 = y_c - dy

        gauss_1 = self._gaussian_2d(x_1, y_1, sigma_x, sigma_y, amp)
        gauss_2 = self._gaussian_2d(x_2, y_2, sigma_x, sigma_y, - amp)

        dipole_field = gauss_1 + gauss_2

        return dipole_field

    def _gen_fault(self) -> np.ndarray:
        x_c = self.rng.uniform(0.0, 1.0)
        y_c = self.rng.uniform(0.0, 1.0)

        amp = self._sample(self.amp_range_f)
        length = self._sample(self.len_range_f)
        width = self._sample(self.width_range_f)
        angle = self._sample(self.angle_range_f)
        smoothing = self._sample(self.smoothing_sigma_f)

        x_rot, y_rot = self._rotate(self.x, self.y, x_c, y_c, angle)

        u = x_rot - x_c
        v = y_rot - y_c

        ridge = amp * np.exp(-(u ** 2) / (2 * (length / 2.0) ** 2) - (v ** 2) / (2 * (width / 2.0) ** 2))

        if smoothing > 0:
            ridge = gaussian_filter(ridge, sigma=smoothing)

        return ridge

    @staticmethod
    def _alpha(nu: float) -> float:
        return (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    
    def _gen_okada_like(self) -> np.ndarray:
        x_c = self.rng.uniform(0.0, 1.0)
        y_c = self.rng.uniform(0.0, 1.0)

        slip = self._sample(self.slip_range_o)
        length = self._sample(self.len_range_o)
        width = self._sample(self.width_range_o)
        angle = self._sample(self.angle_range_o)
        dip = self._sample(self.dip_range_o)
        depth = self._sample(self.depth_range_o)
        smoothing = self._sample(self.smoothing_sigma_o)

        nu = 0.25
        alpha = self._alpha(nu)

        X = self.x - x_c
        Y = self.y - y_c

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        X_a = cos_a * X + sin_a * Y
        Y_a = -sin_a * X + cos_a * Y
        
        cos_d = np.cos(dip)
        sin_d = np.sin(dip)

        p = Y_a * cos_d + depth * sin_d
        q = depth * cos_d - Y_a * sin_d
        u = X_a
        v = p

        def _corner_contribution(xi: float, eta: float) -> np.ndarray:
            u_c = u - xi
            v_c = v - eta

            R = np.sqrt(u_c ** 2 + v_c ** 2 + q ** 2)

            term1 = -alpha * slip * np.arctan2(u_c * q, (v_c + R) * R)
            term2 = (1.0 - alpha) * slip * np.log(v_c + R + 1e-12)
            term3 = -alpha * slip * np.log(R + q + 1e-12)
            return term1 + term2 + term3

        u00 = _corner_contribution(0.0, 0.0)
        uL0 = _corner_contribution(length, 0.0)
        u0W = _corner_contribution(0.0, width)
        uLW = _corner_contribution(length, width)

        uz = u00 - uL0 - u0W + uLW

        if smoothing > 0:
            uz = gaussian_filter(uz, sigma=smoothing)

        return uz

    def _gen_rough(self) -> np.ndarray:
        amp = self._sample(self.amp_range_r)
        sigma_big = self._sample(self.smoothing_sigma_r)

        noise_big = self.rng.standard_normal((self.nx, self.ny))
        coarse = amp * gaussian_filter(noise_big, sigma=sigma_big)

        sigma_small = max(0.2, sigma_big / 4.0)
        micro_amp = 0.3 * amp

        noise_small = self.rng.standard_normal((self.nx, self.ny))
        micro = micro_amp * gaussian_filter(noise_small, sigma=sigma_small)

        rough_field = coarse + micro

        return rough_field

    def add_noise(self, h: np.ndarray) -> np.ndarray:
        scale = self._sample(self.scale_range_n)
        sigma = self._sample(self.smoothing_sigma_n)

        noise = scale * self.rng.standard_normal(h.shape)

        return h + gaussian_filter(noise, sigma)

    def normalize(self, h: np.ndarray) -> np.ndarray:
        h_min, h_max = h.min(), h.max()

        if np.isclose(h_max, h_min):
            return np.full_like(h, self.height_scale[0])

        norm = (h - h_min) / (h_max - h_min)

        return norm * (self.height_scale[1] - self.height_scale[0]) + self.height_scale[0]

    def build_output(self, h: np.ndarray) -> np.ndarray:
        if self.enabled_n:
            h = self.add_noise(h)

        if self.normalize_mode == "per_sample":
            h = self.normalize(h)

        if self.clip_output:
            h = np.clip(h, self.height_scale[0], self.height_scale[1])

        return h
    
"""
Reference notes:

[1] LeVeque, R. J. (2002)
Finite Volume Methods for Hyperbolic Problems
https://doi.org/10.1017/CBO9780511791253


[2] Scivier, A., & Nissen-Meyer, T., & Koelemeijer, P., & Baydin, A. G. (2024)
Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking
https://doi.org/10.48550/arXiv.2412.03299

[3] Okada, Y. (1985)
Surface deformation due to shear and tensile faults in a half-space
https://doi.org/10.1785/BSSA0750041135
"""
