#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.buffered_crop_benchmark import (
    load_inventory_records,
    run_benchmark,
    write_result,
)


DEFAULT_INVENTORY = Path(
    "artifacts/common_time_v2/h0/"
    "830f219cee525d08adb3567c1b135da2ae25572d9f246477ca5f7687f07ecb6b/"
    "h0_input_inventory.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark source-compatible buffered grids and a central 64x64 crop."
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=["train:scenario_000001"],
    )
    parser.add_argument("--grids", nargs="+", type=int, default=[64, 96, 128])
    parser.add_argument("--source-taper-cells", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/common_time_v2/buffered_crop_benchmark_v1/result.json"
        ),
    )
    args = parser.parse_args()

    records = load_inventory_records(args.inventory, args.case_ids)
    payload = run_benchmark(
        records,
        total_grids=args.grids,
        source_taper_cells=args.source_taper_cells,
        progress=lambda message: print(message, flush=True),
    )
    written = write_result(args.output, payload)
    print(
        f"completed {payload['run_count']} runs in {payload['duration_s']:.2f}s; "
        f"result={written['result']} sha256={written['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
