#!/usr/bin/env python
"""Read completed training histories and optimizer step counters without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for model in ("fno", "ffno", "ufno", "convlstm"):
        for seed in (18, 36, 67):
            root = Path(f"experiments/{model}/seed_{seed}")
            paths = {
                "config": root / "config_resolved.yaml",
                "history": root / "history.json",
                "last_checkpoint": root / "checkpoints/last.pt",
            }
            hashes = {name: _digest(path) for name, path in paths.items()}
            config = yaml.safe_load(paths["config"].read_text())
            history = json.loads(paths["history"].read_text())
            payload = torch.load(
                paths["last_checkpoint"],
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            epoch = int(payload["epoch"])
            if (
                config["seed"] != seed
                or payload["config"]["seed"] != seed
                or [row["epoch"] for row in history] != list(range(1, epoch + 1))
            ):
                raise ValueError(f"Training history or seed mismatch: {root}")
            steps = sorted(
                {
                    int(state["step"].item())
                    for state in payload["optimizer_state"]["state"].values()
                }
            )
            if len(steps) != 1 or steps[0] <= 0 or steps[0] % epoch:
                raise ValueError(
                    f"Optimizer counters are not uniform complete epochs: {root}"
                )
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "batch_size": config["data"]["batch_size"],
                    "completed_epochs": epoch,
                    "optimizer_updates": steps[0],
                    "updates_per_epoch": steps[0] // epoch,
                    "source_files": {
                        str(path.resolve()): hashes[name]
                        for name, path in paths.items()
                    },
                }
            )
            if any(_digest(path) != hashes[name] for name, path in paths.items()):
                raise RuntimeError("Training evidence changed during inspection")
            print(model, seed, epoch, steps[0], flush=True)
    result = {
        "scope": "Four leading Hydrostatic models, three completed training seeds each",
        "interpretation": "Optimizer-update counts are observed in final training states. They do not measure matched hardware cost, floating-point work, or controlled wall-clock time.",
        "script_sha256": _digest(Path(__file__)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
