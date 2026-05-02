#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from src.data.dataset import make_toy_dataset, save_npz


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--resolution', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--solver', default='shallow_water', choices=['shallow_water', 'boussinesq'])
    p.add_argument('--out', default='data/processed/toy_32.npz')
    args = p.parse_args()
    x, y, metadata = make_toy_dataset(args.num_samples, args.resolution, seed=args.seed)
    save_npz(args.out, x, y, metadata)
    print(f'Saved {args.out} with x={x.shape}, y={y.shape}')


if __name__ == '__main__':
    main()
