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
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    return cfg


def load_config(path: str | Path) -> Dict[str, Any]:
    """Merge defaults relative to their config file; keep data paths unchanged."""
    return _load_config(Path(path), ())


def _load_config(path: Path, parents: tuple[Path, ...]) -> Dict[str, Any]:
    resolved = path.resolve()
    if resolved in parents:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    cfg = load_yaml(path)
    defaults = cfg.pop('defaults', [])
    if not isinstance(defaults, list) or any(not isinstance(p, str) for p in defaults):
        raise ValueError(f"defaults must be a list of configuration paths in {path}")
    merged: Dict[str, Any] = {}
    for default in defaults:
        merged = deep_update(
            merged, _load_config(path.parent / default, (*parents, resolved))
        )
    return deep_update(merged, cfg)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
