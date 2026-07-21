#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import argparse
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen.simulate_dataset import TsunamiDatasetBuilder


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _configure_threads(workers: int) -> None:
    if workers <= 1:
        return
    invalid = {
        key: value
        for key in THREAD_ENV_KEYS
        if (value := os.environ.get(key)) not in (None, "1")
    }
    if invalid:
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(invalid.items())
        )
        raise SystemExit(
            "Multiprocess generation requires single-thread numerical backends; "
            f"found {details}"
        )
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw tsunami dataset via full simulator pipeline.")
    parser.add_argument("--config", type=str, default="configs/data/dataset.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--split", choices=("train", "eval", "test"), default=None)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument(
        "--solver-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print start/completion lines for each solver rollout.",
    )
    parser.add_argument("--cloud-provider", type=str, default=None)
    parser.add_argument("--cloud-zone", type=str, default=None)
    parser.add_argument("--machine-type", type=str, default=None)
    parser.add_argument("--storage-class", type=str, default=None)
    parser.add_argument("--hourly-cost-usd", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--bathymetry-dir", type=str, default=None)
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--continue", dest="continue_from_last", action="store_true")
    parser.add_argument("--start-at", type=int, default=None)
    parser.add_argument("--stop-at", type=int, default=None)
    parser.add_argument("--allow-override", action="store_true")
    parser.add_argument("--rebuild-manifests", action="store_true")
    parser.add_argument("--acknowledge-provisional", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("dataset", {})
    ds = cfg["dataset"]
    if args.num_samples is not None:
        ds["num_samples"] = int(args.num_samples)
    if args.n_steps is not None:
        ds["n_steps"] = int(args.n_steps)
    if args.save_every is not None:
        ds["save_every"] = int(args.save_every)
    if args.num_workers is not None:
        ds["num_workers"] = int(args.num_workers)
    if args.split is not None:
        cfg.setdefault("requested_output", {})["split"] = args.split
    if args.output_dir is not None:
        ds["output_dir"] = str(args.output_dir)
    if args.bathymetry_dir is not None:
        ds["bathymetry_dir"] = str(args.bathymetry_dir)
    if args.source_dir is not None:
        ds["source_dir"] = str(args.source_dir)
    if args.manifest_path is not None:
        ds["manifest_path"] = str(args.manifest_path)
    cfg.setdefault("operations", {})
    operations = cfg["operations"]
    if args.max_in_flight is not None:
        operations["max_in_flight"] = int(args.max_in_flight)
    if args.progress_every is not None:
        operations["progress_every"] = int(args.progress_every)
    if args.solver_progress is not None:
        operations["solver_progress"] = bool(args.solver_progress)
    for key in (
        "cloud_provider",
        "cloud_zone",
        "machine_type",
        "storage_class",
    ):
        value = getattr(args, key)
        if value is not None:
            operations[key] = str(value)
    if args.hourly_cost_usd is not None:
        operations["hourly_cost_usd"] = float(args.hourly_cost_usd)

    _configure_threads(int(ds.get("num_workers", 1)))

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False)
        tmp_cfg = Path(tf.name)

    try:
        builder = TsunamiDatasetBuilder(
            str(tmp_cfg), provenance_config_path=cfg_path
        )
        builder.run(
            continue_from_last=bool(args.continue_from_last),
            start_at=args.start_at,
            stop_at=args.stop_at,
            allow_override=bool(args.allow_override),
            rebuild_manifests=bool(args.rebuild_manifests),
            acknowledge_provisional=bool(args.acknowledge_provisional),
        )
        print(f"Dataset generation complete using {cfg_path}")
    finally:
        try:
            tmp_cfg.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
