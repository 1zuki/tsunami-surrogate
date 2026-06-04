#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, save_config
from src.utils.io import get_git_commit, load_json, save_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_samples(values: list[str]) -> list[int]:
    samples: list[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            count = int(part)
            if count <= 0:
                raise ValueError(f"Sample counts must be positive, got {count}.")
            samples.append(count)

    if not samples:
        raise ValueError("At least one sample count is required.")
    return samples


def _command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sample-scaling] $ {_command_text(cmd)}", flush=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {_command_text(cmd)}\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        return_code = proc.wait()
        log.write(f"\n[exit_code] {return_code}\n")
        log.flush()

    return return_code


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception as exc:
        return {"load_error": str(exc)}


def _history_summary(history_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    history = _load_json_if_exists(history_path)
    if not isinstance(history, list) or not history:
        return {}

    train_cfg = cfg.get("train", {})
    metric_name = str(train_cfg.get("checkpoint_metric", "val_rel_l2"))
    early_cfg = train_cfg.get("early_stopping", {})
    mode = str(train_cfg.get("checkpoint_mode", early_cfg.get("mode", "min"))).lower()
    if mode not in {"min", "max"}:
        mode = "min"

    rows = [row for row in history if isinstance(row, dict)]
    numeric_rows = [row for row in rows if isinstance(row.get(metric_name), (int, float))]
    summary: dict[str, Any] = {
        "last_epoch": int(rows[-1].get("epoch", len(rows))) if rows else len(history),
        "checkpoint_metric": metric_name,
        "checkpoint_mode": mode,
    }

    if numeric_rows:
        best_row = min(numeric_rows, key=lambda row: float(row[metric_name])) if mode == "min" else max(numeric_rows, key=lambda row: float(row[metric_name]))
        summary["best_epoch"] = int(best_row.get("epoch", -1))
        summary["best_metric_value"] = float(best_row[metric_name])

    return summary


def _flatten_for_csv(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        out_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_for_csv(value, out_key))
        elif isinstance(value, (list, tuple)):
            flat[out_key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[out_key] = value
    return flat


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_for_csv(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _write_results(payload: dict[str, Any], rows: list[dict[str, Any]], output_root: Path) -> None:
    payload = dict(payload)
    payload["updated_at"] = _utc_now()
    payload["rows"] = rows
    save_json(payload, output_root / "sample_scaling_results.json")
    _write_csv(rows, output_root / "sample_scaling_results.csv")


def _build_config(base_config: str, output_root: Path, n_samples: int, device: str | None, val_samples: int | None, test_samples: int | None) -> tuple[dict[str, Any], Path, Path]:
    cfg = load_config(base_config)
    run_dir = output_root / f"n_{n_samples:06d}"
    cfg["output_dir"] = str(run_dir)

    data_cfg = dict(cfg.get("data", {}))
    data_cfg["n_samples"] = int(n_samples)
    if val_samples is not None:
        data_cfg["val_samples"] = int(val_samples)
    if test_samples is not None:
        data_cfg["test_samples"] = int(test_samples)
    cfg["data"] = data_cfg

    eval_cfg = dict(cfg.get("eval", {}))
    eval_cfg["output_dir"] = str(run_dir / "eval")
    cfg["eval"] = eval_cfg

    if device is not None:
        cfg["device"] = device

    cfg["sample_scaling"] = {
        "base_config": str(base_config),
        "requested_train_samples": int(n_samples),
    }

    config_dir = output_root / "configs"
    config_path = config_dir / f"{Path(base_config).stem}_n_{n_samples:06d}.yaml"
    save_config(cfg, config_path)
    return cfg, run_dir, config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate one model config across multiple training-set sizes.")
    parser.add_argument("--config", default="configs/model/fno.yaml", help="Base model YAML config.")
    parser.add_argument("--samples", nargs="+", default=["8,16,32,64"], help="Training sample counts, e.g. '8,16,32' or 8 16 32.")
    parser.add_argument("--output-root", default="experiments/sample_scaling", help="Directory for generated configs, logs, and aggregate results.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Optional device override saved into generated configs.")
    parser.add_argument("--val-samples", type=int, default=None, help="Optional validation subset size for faster sweeps.")
    parser.add_argument("--test-samples", type=int, default=None, help="Optional test subset size for faster sweeps.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for train/eval subprocesses.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate existing best.pt checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training.")
    parser.add_argument("--dry-run", action="store_true", help="Write generated configs/results without launching train/eval.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep sweeping after a failed train/eval command.")
    args = parser.parse_args()

    samples = _parse_samples(args.samples)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": _utc_now(),
        "git_commit": get_git_commit(),
        "base_config": str(args.config),
        "samples": samples,
        "output_root": str(output_root),
        "device": args.device,
        "dry_run": bool(args.dry_run),
    }
    rows: list[dict[str, Any]] = []

    for n_samples in samples:
        cfg, run_dir, config_path = _build_config(
            args.config,
            output_root,
            n_samples,
            args.device,
            args.val_samples,
            args.test_samples,
        )
        log_path = run_dir / "console.log"
        checkpoint_path = run_dir / "best.pt"
        eval_metrics_path = run_dir / "eval" / "metrics.json"
        row: dict[str, Any] = {
            "train_samples_requested": int(n_samples),
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "console_log": str(log_path),
            "checkpoint_path": str(checkpoint_path),
            "status": "pending",
        }

        train_cmd = [args.python, "scripts/train.py", "--config", str(config_path)]
        eval_cmd = [args.python, "scripts/eval_accuracy.py", "--config", str(config_path), "--checkpoint", str(checkpoint_path)]
        row["train_command"] = _command_text(train_cmd)
        row["eval_command"] = _command_text(eval_cmd)

        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            _write_results(payload, rows, output_root)
            continue

        try:
            if not args.skip_train:
                code = _run_logged(train_cmd, log_path)
                if code != 0:
                    row["status"] = "train_failed"
                    row["train_exit_code"] = code
                    rows.append(row)
                    _write_results(payload, rows, output_root)
                    if args.continue_on_error:
                        continue
                    raise SystemExit(code)

            split_info = _load_json_if_exists(run_dir / "split_sizes.json")
            if isinstance(split_info, dict):
                row["split_sizes"] = split_info.get("split_sizes", {})
                row["data_limits"] = split_info.get("data_limits", {})
                row["train_samples_effective"] = row["split_sizes"].get("train")

            row["history_summary"] = _history_summary(run_dir / "history.json", cfg)

            if not args.skip_eval:
                code = _run_logged(eval_cmd, log_path)
                if code != 0:
                    row["status"] = "eval_failed"
                    row["eval_exit_code"] = code
                    rows.append(row)
                    _write_results(payload, rows, output_root)
                    if args.continue_on_error:
                        continue
                    raise SystemExit(code)

                metrics = _load_json_if_exists(eval_metrics_path)
                if isinstance(metrics, dict):
                    row["metrics"] = metrics

            row["status"] = "ok"
            rows.append(row)
            _write_results(payload, rows, output_root)
        except KeyboardInterrupt:
            row["status"] = "interrupted"
            rows.append(row)
            _write_results(payload, rows, output_root)
            raise

    _write_results(payload, rows, output_root)
    print(f"[sample-scaling] wrote {output_root / 'sample_scaling_results.json'}")
    print(f"[sample-scaling] wrote {output_root / 'sample_scaling_results.csv'}")


if __name__ == "__main__":
    main()
