#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen.operational_timing import (  # noqa: E402
    summarize_generation_timings,
    write_generation_timing_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate common-time-v2 generation timing sidecars."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write operational_timing_summary.json below the output directory.",
    )
    args = parser.parse_args()
    if args.write:
        path = write_generation_timing_summary(args.output_dir)
        print(path)
        return
    print(
        json.dumps(
            summarize_generation_timings(args.output_dir), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
