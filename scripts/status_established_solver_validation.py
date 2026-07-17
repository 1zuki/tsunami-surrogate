#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.established_solver_validation import established_solver_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report canonical GeoClaw result progress for a frozen bundle."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            established_solver_status(
                bundle_root=args.bundle_root,
                external_root=args.external_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
