#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_RESUME_ARTIFACTS = (
    Path("config_resolved.yaml"),
    Path("run_metadata.json"),
    Path("history.json"),
    Path("best.pt"),
    Path("checkpoints") / "last.pt",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"unreadable checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _checkpoint_summary(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    cfg = payload.get("config")
    if not isinstance(cfg, Mapping):
        raise ValueError(f"checkpoint config is missing: {path}")
    train_cfg = cfg.get("train")
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}
    early_cfg = train_cfg.get("early_stopping")
    if not isinstance(early_cfg, Mapping):
        early_cfg = {}
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, Mapping):
        raise ValueError(f"checkpoint trainer_state is missing: {path}")
    return {
        "seed": int(cfg.get("seed", -1)),
        "epoch": int(trainer_state.get("epoch", payload.get("epoch", -1))),
        "epochs": int(train_cfg.get("epochs", -1)),
        "early_count": int(trainer_state.get("early_count", -1)),
        "patience": int(early_cfg.get("patience", -1)),
        "metrics": payload.get("metrics"),
    }


def classify_run(
    output_dir: str | Path,
    expected_seed: int,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    present = [path for path in REQUIRED_RESUME_ARTIFACTS if (output / path).exists()]
    if not output.exists() or not present:
        return {"action": "fresh", "reason": "no run artifacts"}
    if len(present) != len(REQUIRED_RESUME_ARTIFACTS):
        missing = [
            path.as_posix()
            for path in REQUIRED_RESUME_ARTIFACTS
            if not (output / path).exists()
        ]
        raise ValueError(
            "partial run artifacts; refusing to overwrite or guess: "
            + ", ".join(missing)
        )

    history = _read_json(output / "history.json")
    if not isinstance(history, list) or not history:
        raise ValueError("history.json must be a non-empty list")
    epochs = [
        int(row.get("epoch", -1))
        for row in history
        if isinstance(row, Mapping)
    ]
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError("history.json epochs must be contiguous from 1")

    last_path = output / "checkpoints" / "last.pt"
    best_path = output / "best.pt"
    last_payload = _checkpoint_payload(last_path)
    best_payload = _checkpoint_payload(best_path)
    last = _checkpoint_summary(last_payload, last_path)
    best = _checkpoint_summary(best_payload, best_path)
    if last["seed"] != int(expected_seed) or best["seed"] != int(expected_seed):
        raise ValueError(
            f"checkpoint seed mismatch: expected {expected_seed}, "
            f"last={last['seed']}, best={best['seed']}"
        )
    if last["epoch"] != len(history):
        raise ValueError("last.pt epoch does not match history.json")
    if history[-1] != last["metrics"]:
        raise ValueError("last.pt metrics do not match the final history row")
    if best["epoch"] < 1 or best["epoch"] > last["epoch"]:
        raise ValueError("best.pt epoch is outside the completed history")
    if config_path is not None:
        generated = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(generated, Mapping):
            raise ValueError("generated config must be a YAML mapping")
        if dict(last_payload.get("config", {})) != dict(generated):
            raise ValueError("last.pt config does not match the generated config")
        if dict(best_payload.get("config", {})) != dict(generated):
            raise ValueError("best.pt config does not match the generated config")

    horizon_complete = last["epochs"] > 0 and last["epoch"] >= last["epochs"]
    early_complete = (
        last["patience"] > 0 and last["early_count"] >= last["patience"]
    )
    if horizon_complete or early_complete:
        return {
            "action": "skip",
            "reason": "training already complete",
            "last_epoch": last["epoch"],
        }
    if last["epochs"] <= 0 or last["epoch"] >= last["epochs"]:
        raise ValueError("run is neither complete nor safely resumable")
    if last["patience"] > 0 and last["early_count"] >= last["patience"]:
        raise ValueError("run already reached terminal early stopping")
    return {
        "action": "resume",
        "reason": "complete resumable artifact set",
        "checkpoint": (output / "checkpoints" / "last.pt").as_posix(),
        "last_epoch": last["epoch"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a cluster run as fresh, resumable, complete, or invalid."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        result = classify_run(args.output_dir, args.expected_seed, args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error\t{exc}")
        raise SystemExit(2) from exc
    fields = [result["action"], result["reason"]]
    if result.get("checkpoint"):
        fields.append(str(result["checkpoint"]))
    print("\t".join(fields))


if __name__ == "__main__":
    main()
