from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """ recursively merge two dictionaries """
    out = dict(base)

    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> Dict[str, Any]:
    """ load a YAML config,sSupports an optional `defaults` list of YAML files """
    cfg = load_yaml(path)
    defaults = cfg.pop('defaults', []) or []
    merged: Dict[str, Any] = {}

    for default in defaults:
        default_path = Path(path).parent / default
        merged = deep_update(merged, load_config(default_path))

    return deep_update(merged, cfg)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
