#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.common_time_v2_level_a import preregister_level_a


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister immutable common-time-v2 Level A contract"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/eval/common_time_v2_level_a.yaml")
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    path = preregister_level_a(
        repo_root=ROOT, config_path=args.config, output_root=args.output_root
    )
    print(path)


if __name__ == "__main__":
    main()
