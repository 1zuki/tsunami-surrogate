#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REGIONS: dict[tuple[float, float, float, float], tuple[str, str, str]] = {
    (95.0, 110.0, -20.0, -5.0): (
        "sumatra_java_trench",
        "Sumatra-Java trench",
        "Sunda subduction margin; strong trench/shelf-slope structure",
    ),
    (-155.0, -140.0, 45.0, 60.0): (
        "alaska_aleutian_margin",
        "Alaska-Aleutian margin",
        "subduction margin with complex shelf and trench morphology",
    ),
    (-135.0, -123.0, 40.0, 52.0): (
        "cascadia_shelf_slope",
        "Cascadia shelf-slope",
        "broad continental shelf/slope and Cascadia subduction setting",
    ),
    (25.0, 30.0, 32.0, 37.0): (
        "hellenic_arc_aegean",
        "Hellenic Arc / Aegean",
        "complex semi-enclosed basin and island-arc morphology",
    ),
    (-80.0, -65.0, 20.0, 35.0): (
        "florida_bahamas_shelf",
        "Florida-Bahamas shelf",
        "shallow carbonate platform / broad shelf control case",
    ),
    (-70.0, -60.0, 10.0, 20.0): (
        "caribbean_island_arc",
        "Caribbean island arc",
        "island-arc and basin morphology",
    ),
    (125.0, 135.0, 5.0, 15.0): (
        "philippines_east_trench",
        "Philippines east trench",
        "Philippine Trench / island-arc margin",
    ),
    (146.0, 160.0, 36.0, 50.0): (
        "kuril_japan_trench",
        "Kuril-Japan trench",
        "trench/island-arc morphology northeast of Japan",
    ),
    (141.0, 147.0, 34.0, 40.0): (
        "japan_trench_tohoku_offshore",
        "Japan Trench / Tohoku offshore",
        "classic Japan Trench margin east of Honshu",
    ),
    (138.0, 144.0, 30.0, 36.0): (
        "nankai_trough_southwest_japan",
        "Nankai Trough / southwest Japan",
        "Nankai/southwest Japan margin and trench-slope morphology",
    ),
}

ORDERED_KEYS = [
    (95.0, 110.0, -20.0, -5.0),
    (-155.0, -140.0, 45.0, 60.0),
    (-135.0, -123.0, 40.0, 52.0),
    (25.0, 30.0, 32.0, 37.0),
    (-80.0, -65.0, 20.0, 35.0),
    (-70.0, -60.0, 10.0, 20.0),
    (125.0, 135.0, 5.0, 15.0),
    (146.0, 160.0, 36.0, 50.0),
    (141.0, 147.0, 34.0, 40.0),
    (138.0, 144.0, 30.0, 36.0),
]

FILENAME_RE = re.compile(
    r"^gebco_2026_n(?P<n>-?\d+(?:\.\d+)?)_s(?P<s>-?\d+(?:\.\d+)?)_w(?P<w>-?\d+(?:\.\d+)?)_e(?P<e>-?\d+(?:\.\d+)?)\.nc$"
)


def _bounds_from_path(path: Path) -> tuple[float, float, float, float] | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None
    west = float(match.group("w"))
    east = float(match.group("e"))
    south = float(match.group("s"))
    north = float(match.group("n"))
    return (west, east, south, north)


def _rescale_to_range(arr: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    in_min = float(np.nanmin(arr))
    in_max = float(np.nanmax(arr))
    if not np.isfinite(in_min) or not np.isfinite(in_max) or in_max <= in_min:
        raise ValueError(f"Invalid bathymetry range [{in_min}, {in_max}]")
    scaled = (arr - in_min) / (in_max - in_min)
    return (out_min + scaled * (out_max - out_min)).astype(np.float32)


def _read_gebco(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(str(path), "r", mmap=False) as dataset:
        if "elevation" not in dataset.variables:
            raise KeyError(f"{path} has no 'elevation' variable")
        elevation = np.asarray(dataset.variables["elevation"].data, dtype=np.float32)
        lat = np.asarray(dataset.variables["lat"].data, dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"].data, dtype=np.float64)
    return elevation, lat, lon


def _resize(arr: np.ndarray, resolution: int) -> np.ndarray:
    factors = (
        float(resolution) / float(arr.shape[0]),
        float(resolution) / float(arr.shape[1]),
    )
    return zoom(arr, factors, order=1).astype(np.float32)


def _collect_files(
    input_dir: Path,
) -> list[tuple[tuple[float, float, float, float], Path]]:
    by_key: dict[tuple[float, float, float, float], Path] = {}
    for path in sorted(input_dir.glob("gebco_2026_*.nc")):
        if "_tid_" in path.name:
            continue
        key = _bounds_from_path(path)
        if key is None:
            continue
        by_key[key] = path

    missing = [key for key in ORDERED_KEYS if key not in by_key]
    if missing:
        raise FileNotFoundError(f"Missing GEBCO crops for bounds: {missing}")
    return [(key, by_key[key]) for key in ORDERED_KEYS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GEBCO NetCDF crops into real-bathymetry sample_*.npz files."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--depth-min", type=float, default=-10.0)
    parser.add_argument("--depth-max", type=float, default=-0.75)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if (
        output_dir.exists()
        and any(output_dir.glob("sample_*.npz"))
        and not args.allow_overwrite
    ):
        raise FileExistsError(
            f"{output_dir} already has sample_*.npz files; pass --allow-overwrite to replace them"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, (bounds, path) in enumerate(_collect_files(input_dir), start=1):
        slug, label, note = REGIONS[bounds]
        elevation, lat, lon = _read_gebco(path)
        resized = _resize(elevation, int(args.resolution))
        fully_wet = _rescale_to_range(
            resized, float(args.depth_min), float(args.depth_max)
        )
        out_path = output_dir / f"sample_{idx:06d}.npz"
        np.savez_compressed(
            out_path,
            bathymetry=fully_wet,
            bathymetry_type=np.asarray([f"gebco_{slug}_fully_wet_scaled"], dtype="U96"),
            sample_seed=np.asarray([idx], dtype=np.int64),
            region_label=np.asarray([label], dtype="U96"),
            morphology_note=np.asarray([note], dtype="U256"),
            source_file=np.asarray([str(path)], dtype="U256"),
            bounds_wesn=np.asarray(bounds, dtype=np.float32),
            original_shape=np.asarray(elevation.shape, dtype=np.int32),
            lat_range=np.asarray(
                [float(np.nanmin(lat)), float(np.nanmax(lat))], dtype=np.float32
            ),
            lon_range=np.asarray(
                [float(np.nanmin(lon)), float(np.nanmax(lon))], dtype=np.float32
            ),
            original_elevation_minmax=np.asarray(
                [float(np.nanmin(elevation)), float(np.nanmax(elevation))],
                dtype=np.float32,
            ),
            derivation=np.asarray(
                [
                    f"bilinear_resize_to_{args.resolution}_then_linear_rescale_to_fully_wet_{args.depth_min}_{args.depth_max}"
                ],
                dtype="U128",
            ),
        )
        rows.append(
            {
                "sample": out_path.name,
                "label": label,
                "bounds_wesn": bounds,
                "source_file": str(path),
                "original_min": float(np.nanmin(elevation)),
                "original_max": float(np.nanmax(elevation)),
                "processed_min": float(np.nanmin(fully_wet)),
                "processed_max": float(np.nanmax(fully_wet)),
            }
        )
        print(
            f"{out_path}: {label} raw=[{rows[-1]['original_min']:.1f},{rows[-1]['original_max']:.1f}] processed=[{rows[-1]['processed_min']:.2f},{rows[-1]['processed_max']:.2f}]"
        )

    np.savez_compressed(
        output_dir / "suite_metadata.npz", rows=np.asarray(rows, dtype=object)
    )
    print(f"wrote {len(rows)} GEBCO morphology samples -> {output_dir}")


if __name__ == "__main__":
    main()
