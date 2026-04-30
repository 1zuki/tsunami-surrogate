#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from tsunami_surrogate.data_gen.simulate_dataset import simulate_dataset, save_npz


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--resolution', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--solver', default='shallow_water', choices=['shallow_water', 'boussinesq'])
    p.add_argument('--out', default='data/processed/toy_32.npz')
    args = p.parse_args()
    x, y, metadata = simulate_dataset(args.num_samples, args.resolution, seed=args.seed, solver=args.solver)
    save_npz(args.out, x, y, metadata)
    print(f'Saved {args.out} with x={x.shape}, y={y.shape}')


if __name__ == '__main__':
    main()
