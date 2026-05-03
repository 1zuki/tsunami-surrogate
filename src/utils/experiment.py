from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from .io import ensure_dir, save_json, get_git_commit
from .config import save_config


def init_run(output_dir: str | Path, cfg: Dict[str, Any]) -> Path:
    out = ensure_dir(output_dir)
    ensure_dir(out / 'checkpoints')
    ensure_dir(out / 'figures')
    ensure_dir(out / 'tables')
    save_config(cfg, out / 'config_resolved.yaml')

    save_json({
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_commit(),
        'seed': cfg.get('seed'),
        'output_dir': str(out),
    }, out / 'run_metadata.json')

    return out
