from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from .io import ensure_dir, save_json, get_git_commit
from .config import save_config


RUN_ARTIFACT_PATHS = (
    Path("config_resolved.yaml"),
    Path("run_metadata.json"),
    Path("history.json"),
    Path("best.pt"),
    Path("checkpoints") / "last.pt",
)


def occupied_run_artifacts(output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    return [out / relative for relative in RUN_ARTIFACT_PATHS if (out / relative).exists()]


def init_run(output_dir: str | Path, cfg: Dict[str, Any], fresh: bool = True) -> Path:
    out = Path(output_dir)
    if not fresh:
        if not out.is_dir():
            raise FileNotFoundError(
                f"Cannot resume missing run directory: {out}"
            )
        return out

    if fresh:
        existing = occupied_run_artifacts(out)
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing run artifacts: "
                + ", ".join(str(path) for path in existing)
            )

    out = ensure_dir(out)
    ensure_dir(out / "checkpoints")
    ensure_dir(out / "figures")
    ensure_dir(out / "tables")

    save_config(cfg, out / "config_resolved.yaml")

    save_json(
        {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "git_commit": get_git_commit(),
            "seed": cfg.get("seed"),
            "output_dir": str(out),
        },
        out / "run_metadata.json",
    )

    return out
