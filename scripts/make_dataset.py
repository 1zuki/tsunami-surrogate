#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.simulate_dataset import TsunamiDatasetBuilder  # noqa: E402


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
DEFAULT_CONFIG = ROOT / "configs/data/dataset.yaml"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate common-time bathymetry, sources, and all solver rollouts "
            "in one pass."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument(
        "--solver-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cloud-provider")
    parser.add_argument("--cloud-zone")
    parser.add_argument("--machine-type")
    parser.add_argument("--storage-class")
    parser.add_argument("--hourly-cost-usd", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--bathymetry-dir")
    parser.add_argument("--source-dir")
    parser.add_argument("--manifest-path")
    parser.add_argument("--paired-inventory-path")
    parser.add_argument("--continue", dest="continue_from_last", action="store_true")
    parser.add_argument("--start-at", type=int)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--allow-override", action="store_true")
    parser.add_argument("--rebuild-manifests", action="store_true")
    parser.add_argument("--acknowledge-provisional", action="store_true")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return raw


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    dataset = cfg.setdefault("dataset", {})
    for argument in ("num_samples", "n_steps", "save_every", "num_workers"):
        value = getattr(args, argument)
        if value is not None:
            dataset[argument] = int(value)
    for argument in (
        "output_dir",
        "bathymetry_dir",
        "source_dir",
        "manifest_path",
    ):
        value = getattr(args, argument)
        if value is not None:
            dataset[argument] = str(value)
    if args.paired_inventory_path is not None:
        paired = cfg.get("paired_inputs")
        if not isinstance(paired, dict) or not bool(paired.get("enabled", False)):
            raise SystemExit(
                "--paired-inventory-path requires paired_inputs.enabled=true"
            )
        paired["inventory_path"] = str(args.paired_inventory_path)

    operations = cfg.setdefault("operations", {})
    if args.max_in_flight is not None:
        operations["max_in_flight"] = int(args.max_in_flight)
    if args.progress_every is not None:
        operations["progress_every"] = int(args.progress_every)
    if args.solver_progress is not None:
        operations["solver_progress"] = bool(args.solver_progress)
    for key in ("cloud_provider", "cloud_zone", "machine_type", "storage_class"):
        value = getattr(args, key)
        if value is not None:
            operations[key] = str(value)
    if args.hourly_cost_usd is not None:
        operations["hourly_cost_usd"] = float(args.hourly_cost_usd)


def main() -> None:
    args = _build_parser().parse_args()
    config_path = args.config.resolve()
    cfg = _load_config(config_path)
    _apply_overrides(cfg, args)

    dataset = cfg["dataset"]
    workers = int(dataset.get("num_workers", 1))
    _configure_threads(workers)
    output_dir = Path(str(dataset["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        prefix=".resolved-dataset-",
        suffix=".yaml",
        dir=output_dir,
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temporary_config = Path(handle.name)
    try:
        builder = TsunamiDatasetBuilder(
            str(temporary_config), provenance_config_path=config_path
        )
        builder.run(
            continue_from_last=bool(args.continue_from_last),
            start_at=args.start_at,
            stop_at=args.stop_at,
            allow_override=bool(args.allow_override),
            rebuild_manifests=bool(args.rebuild_manifests),
            acknowledge_provisional=bool(args.acknowledge_provisional),
        )
        print(f"Dataset generation complete using {config_path}")
    finally:
        temporary_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
