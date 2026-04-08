from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import zoom

from src.data_gen.dataset import compute_stats, load_npz_dict


def resize_last_two_dims(array: np.ndarray, new_hw: tuple[int, int], order: int = 1) -> np.ndarray:
    new_h, new_w = new_hw
    if array.ndim < 2:
        return array
    old_h, old_w = array.shape[-2], array.shape[-1]
    factors = [1.0] * array.ndim
    factors[-2] = new_h / old_h
    factors[-1] = new_w / old_w
    return zoom(array, zoom=factors, order=order).astype(np.float32)


def resize_npz(input_path: str | Path, output_path: str | Path, new_hw: tuple[int, int]) -> None:
    arrays = load_npz_dict(input_path)
    resized: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if value.ndim >= 3:
            resized[key] = resize_last_two_dims(value, new_hw, order=1)
        else:
            resized[key] = value.astype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **resized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess tsunami datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_p = subparsers.add_parser("stats", help="Compute normalization stats for an NPZ dataset.")
    stats_p.add_argument("--input", type=str, required=True)
    stats_p.add_argument("--output", type=str, required=True)

    resize_p = subparsers.add_parser("resize", help="Resize the spatial grid of an NPZ dataset.")
    resize_p.add_argument("--input", type=str, required=True)
    resize_p.add_argument("--output", type=str, required=True)
    resize_p.add_argument("--height", type=int, required=True)
    resize_p.add_argument("--width", type=int, required=True)

    args = parser.parse_args()

    if args.command == "stats":
        stats = compute_stats(args.input, input_keys=["bathymetry", "disturbance"], target_key="wave")
        stats.save(args.output)
        print(f"Saved stats to {args.output}")
    elif args.command == "resize":
        resize_npz(args.input, args.output, (args.height, args.width))
        print(f"Saved resized dataset to {args.output}")


if __name__ == "__main__":
    main()
