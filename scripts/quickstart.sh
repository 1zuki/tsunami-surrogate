#!/usr/bin/env bash
set -euo pipefail
python scripts/make_dataset.py --config configs/data/dataset.yaml --stop-at 1
