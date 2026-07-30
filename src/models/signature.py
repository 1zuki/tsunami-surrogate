from __future__ import annotations

from typing import Any, Dict, Mapping


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "fno2d": {
        "in_channels": 3,
        "out_channels": 1,
        "modes1": 12,
        "modes2": 12,
        "width": 32,
        "depth": 4,
        "padding": 6,
        "use_grid": True,
    },
    "ffno2d": {
        "in_channels": 3,
        "out_channels": 1,
        "modes1": 12,
        "modes2": 12,
        "width": 32,
        "depth": 4,
        "padding": 6,
        "use_grid": True,
    },
    "wno2d": {
        "in_channels": 3,
        "out_channels": 1,
        "width": 32,
        "depth": 4,
        "padding": 6,
        "use_grid": True,
        "wavelet_kernel_size": 3,
    },
    "ufno2d": {
        "in_channels": 3,
        "out_channels": 1,
        "modes1": 12,
        "modes2": 12,
        "width": 32,
        "depth": 4,
        "padding": 6,
        "use_grid": True,
    },
    "fno2d_probabilistic": {
        "in_channels": 3,
        "out_channels": 1,
        "modes1": 12,
        "modes2": 12,
        "width": 32,
        "depth": 4,
        "padding": 6,
        "use_grid": True,
    },
    "cnn": {
        "in_channels": 3,
        "out_channels": 1,
        "width": 32,
    },
    "unet": {
        "in_channels": 3,
        "out_channels": 1,
        "width": 32,
    },
    "convlstm": {
        "in_channels": 3,
        "out_channels": 1,
        "hidden_channels": 48,
        "num_layers": 2,
        "kernel_size": 3,
        "context_channels": None,
        "use_feedback": True,
    },
}


def model_config_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw_model_cfg = cfg.get("model", cfg)
    if not isinstance(raw_model_cfg, Mapping):
        raise TypeError("model configuration must be a mapping")

    name = str(raw_model_cfg.get("name", "fno2d"))
    defaults = MODEL_DEFAULTS.get(name)
    if defaults is None:
        return {"name": name}

    signature: Dict[str, Any] = {"name": name}
    for key, default in defaults.items():
        if name == "convlstm" and key == "hidden_channels":
            value = raw_model_cfg.get(
                "hidden_channels", raw_model_cfg.get("width", default)
            )
        else:
            value = raw_model_cfg.get(key, default)
        signature[key] = value
    return signature
