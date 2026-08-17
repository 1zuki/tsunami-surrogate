#!/usr/bin/env python
"""Train two independent ConvLSTM seeds concurrently on Kaggle GPUs.

The launcher assigns one seed to each physical GPU instead of using
data-parallel training. It keeps the repository's canonical relative data and
output paths so downloaded checkpoints can be resumed locally after extraction
at the repository root.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATA_ROOT = Path("data/processed/hydrostatic")
DEFAULT_KAGGLE_DATA_ROOT = Path("/kaggle/input/tsunami-surrogate-hydrostatic")
DEFAULT_OUTPUT_ROOT = Path("experiments/multiseed_v2/convlstm_hydrostatic")
EXPECTED_SPLIT_COUNTS = {"train": 10_000, "val": 1_000, "test": 2_500}
EXPECTED_INPUT_ORDER = ["bathymetry", "source", "initial_depth"]
RUN_ARTIFACTS = (
    Path("config_resolved.yaml"),
    Path("run_metadata.json"),
    Path("history.json"),
    Path("best.pt"),
    Path("checkpoints") / "last.pt",
)
RESUME_ARTIFACTS = (
    Path("history.json"),
    Path("best.pt"),
    Path("checkpoints") / "last.pt",
)


@dataclass(frozen=True)
class WorkerSpec:
    repo_root: str
    output_root: str
    seed: int
    physical_gpu: int
    batch_size: int
    num_workers: int
    resume_mode: str
    log_path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_dataset_root(
    root: Path,
    expected_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
) -> dict[str, Any]:
    """Validate the cheap, manifest-level processed-data contract."""

    root = root.expanduser().resolve()
    stats_path = root / "normalization_stats.json"
    stats = _read_json_object(stats_path)
    if not isinstance(stats.get("inputs"), dict) or not isinstance(
        stats.get("targets"), dict
    ):
        raise ValueError(
            f"Normalization statistics lack inputs/targets sections: {stats_path}"
        )

    split_summaries: dict[str, Any] = {}
    for split, expected_count in expected_counts.items():
        split_root = root / split
        manifest_path = split_root / "shards_manifest.json"
        manifest = _read_json_object(manifest_path)

        if not bool(manifest.get("sharded", False)):
            raise ValueError(f"Expected a sharded processed split: {manifest_path}")
        if str(manifest.get("split", "")) != split:
            raise ValueError(
                f"Processed split label mismatch in {manifest_path}: "
                f"{manifest.get('split')!r}"
            )
        if list(manifest.get("input_order", [])) != EXPECTED_INPUT_ORDER:
            raise ValueError(
                f"Processed input order mismatch in {manifest_path}: "
                f"{manifest.get('input_order')!r}"
            )
        if str(manifest.get("target_variable", "")) != "eta":
            raise ValueError(f"Processed target variable must be eta: {manifest_path}")
        if not bool(manifest.get("normalized_targets", False)):
            raise ValueError(
                f"Processed targets must already be normalized: {manifest_path}"
            )

        declared_count = int(manifest.get("num_samples", -1))
        if declared_count != int(expected_count):
            raise ValueError(
                f"Unexpected {split} sample count in {manifest_path}: "
                f"expected={expected_count} observed={declared_count}"
            )

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Processed split contains no shards: {manifest_path}")

        observed_count = 0
        for index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise ValueError(f"Invalid shard entry {index}: {manifest_path}")
            relative = Path(str(shard.get("file", "")))
            shard_path = split_root / relative
            if not relative.as_posix() or not _is_within(shard_path, split_root):
                raise ValueError(f"Unsafe shard path at index {index}: {manifest_path}")
            if not shard_path.is_file() or shard_path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"Missing or empty processed shard: {shard_path}"
                )

            shard_count = int(shard.get("num_samples", -1))
            inputs_shape = shard.get("inputs_shape")
            targets_shape = shard.get("targets_shape")
            if not isinstance(inputs_shape, list) or inputs_shape != [
                shard_count,
                3,
                64,
                64,
            ]:
                raise ValueError(
                    f"Unexpected input shape for shard {index}: {inputs_shape!r}"
                )
            if not isinstance(targets_shape, list) or targets_shape != [
                shard_count,
                50,
                64,
                64,
            ]:
                raise ValueError(
                    f"Unexpected target shape for shard {index}: {targets_shape!r}"
                )
            observed_count += shard_count

        if observed_count != declared_count:
            raise ValueError(
                f"Shard counts do not sum to the declared {split} count: "
                f"declared={declared_count} observed={observed_count}"
            )

        split_summaries[split] = {
            "num_samples": declared_count,
            "num_shards": len(shards),
            "manifest_version": manifest.get("version"),
        }

    return {
        "root": str(root),
        "normalization_stats": str(stats_path),
        "splits": split_summaries,
    }


def _looks_like_dataset_root(path: Path) -> bool:
    return (path / "normalization_stats.json").is_file() and all(
        (path / split / "shards_manifest.json").is_file()
        for split in EXPECTED_SPLIT_COUNTS
    )


def resolve_dataset_root(requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if _looks_like_dataset_root(DEFAULT_KAGGLE_DATA_ROOT):
        return DEFAULT_KAGGLE_DATA_ROOT.resolve()

    kaggle_input = Path("/kaggle/input")
    candidates = (
        sorted(
            path.resolve()
            for path in kaggle_input.iterdir()
            if path.is_dir() and _looks_like_dataset_root(path)
        )
        if kaggle_input.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple compatible Kaggle datasets were found; pass --data-root "
            f"explicitly: {[str(path) for path in candidates]}"
        )

    local = ROOT / CANONICAL_DATA_ROOT
    if _looks_like_dataset_root(local):
        return local.resolve()
    raise FileNotFoundError(
        "Could not discover the processed Hydrostatic dataset. Attach the "
        "Kaggle dataset or pass --data-root."
    )


def ensure_canonical_data_location(data_root: Path) -> Path:
    """Expose the Kaggle input through the repository's canonical data path."""

    canonical = ROOT / CANONICAL_DATA_ROOT
    data_root = data_root.resolve()

    if canonical.is_symlink():
        if canonical.resolve() != data_root:
            raise RuntimeError(
                f"Canonical data link points elsewhere: {canonical} -> "
                f"{canonical.resolve()}"
            )
        return canonical

    if canonical.exists():
        if canonical.resolve() != data_root:
            raise RuntimeError(
                "Canonical processed-data path already exists and differs from "
                f"--data-root: canonical={canonical.resolve()} data={data_root}"
            )
        return canonical

    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(data_root, target_is_directory=True)
    return canonical


def _validate_relative_output_root(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(
            "--output-root must be repository-relative so checkpoints remain "
            "portable between Kaggle and the local checkout"
        )
    resolved = (ROOT / path).resolve()
    if not _is_within(resolved, ROOT):
        raise ValueError("--output-root must remain inside the repository")
    return path


def build_runtime_config(
    seed: int,
    output_root: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], Path]:
    sys.path.insert(0, str(ROOT))
    from src.utils.config import load_config

    cfg = load_config(ROOT / "configs/model/convlstm.yaml")
    cfg.pop("seeds", None)
    cfg["seed"] = int(seed)
    cfg["device"] = "cuda"

    run_dir = output_root / f"{output_root.name}_seed_{int(seed)}"
    cfg["output_dir"] = run_dir.as_posix()

    data_cfg = dict(cfg.get("data", {}))
    data_cfg.update(
        {
            "train_path": (
                CANONICAL_DATA_ROOT / "train" / "eval_dataset.npz"
            ).as_posix(),
            "val_path": (CANONICAL_DATA_ROOT / "val" / "eval_dataset.npz").as_posix(),
            "test_path": (CANONICAL_DATA_ROOT / "test" / "eval_dataset.npz").as_posix(),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
        }
    )
    cfg["data"] = data_cfg

    eval_cfg = dict(cfg.get("eval", {}))
    eval_cfg["dataset_path"] = (
        CANONICAL_DATA_ROOT / "test" / "eval_dataset.npz"
    ).as_posix()
    eval_cfg["output_dir"] = (run_dir / "eval").as_posix()
    cfg["eval"] = eval_cfg
    return cfg, run_dir


def classify_run(run_dir: Path, resume_mode: str) -> str:
    status_path = run_dir / "kaggle_run_status.json"
    status = _read_json_object(status_path) if status_path.is_file() else {}
    resume_ready = all((run_dir / path).is_file() for path in RESUME_ARTIFACTS)
    occupied = [path for path in RUN_ARTIFACTS if (run_dir / path).exists()]

    if status.get("state") == "completed":
        if not resume_ready:
            raise RuntimeError(
                f"Run is marked completed but artifacts are incomplete: {run_dir}"
            )
        return "skip"

    if resume_mode == "never":
        if occupied:
            raise RuntimeError(
                f"Fresh-only mode found existing run artifacts: {run_dir}"
            )
        return "fresh"

    if resume_mode == "require":
        if not resume_ready:
            raise RuntimeError(f"Resume artifacts are incomplete: {run_dir}")
        return "resume"

    if resume_ready:
        return "resume"
    if not occupied:
        return "fresh"
    raise RuntimeError(
        "Run contains partial startup artifacts but no safe resume checkpoint: "
        f"{run_dir}. Preserve it and inspect before restarting."
    )


def _checkpoint_is_terminal(last_path: Path, cfg: Mapping[str, Any]) -> str | None:
    import torch

    payload = torch.load(last_path, map_location="cpu")
    state = payload.get("trainer_state")
    if not isinstance(state, Mapping):
        return None

    train_cfg = cfg.get("train", {})
    early_cfg = train_cfg.get("early_stopping", {})
    epoch = int(state.get("epoch", payload.get("epoch", 0)))
    if epoch >= int(train_cfg.get("epochs", 5)):
        return "epoch_horizon"
    if int(state.get("early_count", 0)) >= int(early_cfg.get("patience", 10)):
        return "early_stopping"
    return None


def _worker_entry(spec: WorkerSpec) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.physical_gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    repo_root = Path(spec.repo_root)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    output_root = Path(spec.output_root)
    cfg, run_dir = build_runtime_config(
        spec.seed,
        output_root,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
    )
    status_path = run_dir / "kaggle_run_status.json"
    log_path = Path(spec.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                import torch

                from scripts.train import train_one
                from src.utils.device import resolve_device

                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError(
                        "Each worker must see exactly one CUDA GPU after "
                        f"CUDA_VISIBLE_DEVICES={spec.physical_gpu}; "
                        f"observed={torch.cuda.device_count()}"
                    )
                torch.cuda.set_device(0)
                device = resolve_device("cuda")
                action = classify_run(run_dir, spec.resume_mode)

                print(
                    f"[kaggle-convlstm] seed={spec.seed} "
                    f"physical_gpu={spec.physical_gpu} "
                    f"visible_gpu={torch.cuda.get_device_name(0)} action={action}",
                    flush=True,
                )
                if action == "skip":
                    return

                resume_path = (
                    run_dir / "checkpoints" / "last.pt" if action == "resume" else None
                )
                if resume_path is not None:
                    terminal_reason = _checkpoint_is_terminal(resume_path, cfg)
                    if terminal_reason is not None:
                        _write_json_atomic(
                            {
                                "state": "completed",
                                "completed_at": _utc_now(),
                                "completion_recovered_from_checkpoint": True,
                                "terminal_reason": terminal_reason,
                                "seed": spec.seed,
                                "physical_gpu": spec.physical_gpu,
                            },
                            status_path,
                        )
                        print(
                            "[kaggle-convlstm] checkpoint is already terminal; "
                            f"marked complete ({terminal_reason})",
                            flush=True,
                        )
                        return

                run_dir.mkdir(parents=True, exist_ok=True)
                _write_json_atomic(
                    {
                        "state": "running",
                        "started_or_resumed_at": _utc_now(),
                        "action": action,
                        "seed": spec.seed,
                        "physical_gpu": spec.physical_gpu,
                        "visible_gpu": torch.cuda.get_device_name(0),
                        "torch_version": str(torch.__version__),
                        "cuda_version": str(torch.version.cuda),
                        "cudnn_version": torch.backends.cudnn.version(),
                        "batch_size": spec.batch_size,
                        "num_workers": spec.num_workers,
                    },
                    status_path,
                )
                train_one(
                    cfg,
                    device,
                    resume_path=str(resume_path) if resume_path is not None else None,
                )

                history = _read_json_object(status_path)
                history_path = run_dir / "history.json"
                rows = json.loads(history_path.read_text(encoding="utf-8"))
                final_epoch = (
                    int(rows[-1]["epoch"]) if isinstance(rows, list) and rows else None
                )
                history.update(
                    {
                        "state": "completed",
                        "completed_at": _utc_now(),
                        "final_epoch": final_epoch,
                    }
                )
                _write_json_atomic(history, status_path)
                print(
                    f"[kaggle-convlstm] completed seed={spec.seed} "
                    f"final_epoch={final_epoch}",
                    flush=True,
                )
            except BaseException as exc:
                failure = {
                    "state": "failed",
                    "failed_at": _utc_now(),
                    "seed": spec.seed,
                    "physical_gpu": spec.physical_gpu,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _write_json_atomic(failure, status_path)
                traceback.print_exc()
                raise


def query_nvidia_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi could not enumerate Kaggle GPUs") from exc

    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mib": int(parts[2]),
            }
        )
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def archive_outputs(output_root: Path, archive_path: Path) -> Path:
    source = ROOT / output_root
    if not source.is_dir():
        raise FileNotFoundError(f"Training output does not exist: {source}")

    archive_path = archive_path.expanduser()
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging = archive_path.with_name(f".{archive_path.name}.tmp")
    if staging.exists():
        staging.unlink()
    with tarfile.open(staging, "w:gz") as archive:
        archive.add(source, arcname=output_root.as_posix())
    os.replace(staging, archive_path)
    return archive_path


def _last_log_line(path: Path) -> str:
    if not path.is_file():
        return "log pending"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"log unavailable: {exc}"
    return next((line for line in reversed(lines) if line.strip()), "log empty")


def _default_archive_path() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "convlstm_multiseed_v2.tar.gz"
    return Path("results/multiseed_v2/convlstm_multiseed_v2.tar.gz")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Processed Hydrostatic root containing normalization_stats.json "
            "and train/val/test. Auto-discovers the Kaggle input by default."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[36, 67])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Repository-relative output root.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        choices=("auto", "never", "require"),
        default="auto",
    )
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Create the downloadable archive without launching training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/configuration without launching GPU workers.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_root = _validate_relative_output_root(args.output_root)
    archive_path = args.archive or _default_archive_path()

    if args.package_only:
        archived = archive_outputs(output_root, archive_path)
        print(f"[kaggle-convlstm] archive={archived}")
        return 0

    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be unique")
    if len(args.gpus) != len(args.seeds):
        raise ValueError("--gpus must contain exactly one GPU for each seed")
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError("Each seed requires a distinct physical GPU")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")

    data_root = resolve_dataset_root(args.data_root)
    dataset_summary = validate_dataset_root(data_root)
    canonical = ensure_canonical_data_location(data_root)
    gpu_rows = query_nvidia_gpus()
    available = {int(row["index"]) for row in gpu_rows}
    missing = sorted(set(args.gpus) - available)
    if missing:
        raise RuntimeError(
            f"Requested GPU indices are unavailable: {missing}; detected={gpu_rows}"
        )

    (ROOT / output_root).mkdir(parents=True, exist_ok=True)
    launcher_manifest = {
        "schema_id": "tsunami-surrogate.kaggle-convlstm-launch.v1",
        "created_at": _utc_now(),
        "repo_root": str(ROOT),
        "git_commit": _git_commit(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset_summary,
        "canonical_data_location": str(canonical),
        "seeds": list(args.seeds),
        "physical_gpus": list(args.gpus),
        "detected_gpus": gpu_rows,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resume_mode": args.resume,
        "output_root": output_root.as_posix(),
        "archive": str(archive_path),
    }
    _write_json_atomic(
        launcher_manifest,
        ROOT / output_root / "kaggle_launcher_manifest.json",
    )

    print(f"[kaggle-convlstm] dataset={data_root} splits={dataset_summary['splits']}")
    print(f"[kaggle-convlstm] gpus={gpu_rows}")
    print(
        "[kaggle-convlstm] assignments="
        + ", ".join(
            f"seed {seed} -> GPU {gpu}" for seed, gpu in zip(args.seeds, args.gpus)
        )
    )
    if args.dry_run:
        print("[kaggle-convlstm] dry-run passed; no training launched")
        return 0

    specs = [
        WorkerSpec(
            repo_root=str(ROOT),
            output_root=output_root.as_posix(),
            seed=int(seed),
            physical_gpu=int(gpu),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            resume_mode=str(args.resume),
            log_path=str(ROOT / output_root / "logs" / f"seed_{int(seed)}.log"),
        )
        for seed, gpu in zip(args.seeds, args.gpus)
    ]
    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=_worker_entry,
            args=(spec,),
            name=f"convlstm-seed-{spec.seed}",
        )
        for spec in specs
    ]

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[kaggle-convlstm] received signal {signum}; stopping workers",
            flush=True,
        )

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for worker, spec in zip(workers, specs):
            worker.start()
            print(
                f"[kaggle-convlstm] launched seed={spec.seed} "
                f"gpu={spec.physical_gpu} pid={worker.pid} log={spec.log_path}",
                flush=True,
            )

        last_heartbeat = 0.0
        while any(worker.is_alive() for worker in workers):
            if stop_requested:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                break
            now = time.monotonic()
            if now - last_heartbeat >= 60.0:
                for worker, spec in zip(workers, specs):
                    state = "running" if worker.is_alive() else str(worker.exitcode)
                    latest = _last_log_line(Path(spec.log_path))
                    print(
                        f"[kaggle-convlstm] heartbeat {worker.name}:{state} "
                        f"latest={latest}",
                        flush=True,
                    )
                last_heartbeat = now
            for worker in workers:
                worker.join(timeout=0.5)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    archived = archive_outputs(output_root, archive_path)
    exit_codes = {worker.name: worker.exitcode for worker in workers}
    print(f"[kaggle-convlstm] exit_codes={exit_codes}")
    print(f"[kaggle-convlstm] archive={archived}")
    if stop_requested:
        return 130
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
