from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively update a nested mapping."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping at the root: {path}")
    return data


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    config = load_yaml(path)
    base_ref = config.pop("base_config", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)
        config = deep_update(base_cfg, config)
    return config


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def get_nested(config: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(config: MutableMapping[str, Any], keys: Iterable[str], value: Any) -> None:
    keys = list(keys)
    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def clone_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(config))


def maybe_resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
