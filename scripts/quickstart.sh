#!/usr/bin/env bash
set -euo pipefail
python scripts/make_toy_data.py --num-samples 4 --resolution 8 --out data/processed/toy_8_smoke.npz
python scripts/train.py --config configs/experiments/exp000_smoke.yaml
python scripts/evaluate.py --config configs/experiments/exp000_smoke.yaml --checkpoint experiments/exp000_smoke/best.pt
