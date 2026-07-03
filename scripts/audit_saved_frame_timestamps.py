#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")
DEFAULT_SPLITS = ("train", "val", "test")
SOLVER_LABELS = {
    "hydrostatic": "Hydrostatic",
    "muscl_hr": "MUSCL-HR",
    "boussinesq": "Boussinesq",
    "swe_hydrostatic": "Hydrostatic",
    "swe_muscl_hr": "MUSCL-HR",
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _scenario_id(row: dict[str, Any]) -> str:
    raw = row.get("scenario_id")
    if raw is not None and str(raw).strip():
        return str(raw)
    if "sample_index" in row:
        return f"scenario_{int(row['sample_index']):06d}"
    sample_dir = str(row.get("sample_dir", ""))
    name = Path(sample_dir).name
    if name.startswith("sample_") and name.split("sample_", 1)[-1].isdigit():
        return f"scenario_{int(name.split('sample_', 1)[-1]):06d}"
    return "scenario_unknown"


def _scenario_key(split: str, scenario_id: str) -> str:
    return f"{split}:{scenario_id}"


def _display_solver(raw: str) -> str:
    key = str(raw).strip()
    return SOLVER_LABELS.get(key, key)


def _candidate_sample_dirs(sample_dir: str) -> Iterable[Path]:
    raw = Path(str(sample_dir))
    yield raw

    text = str(sample_dir)
    marker = "/data/"
    if marker in text:
        yield ROOT / ("data/" + text.split(marker, 1)[1])

    repo_name = ROOT.name
    repo_marker = f"/{repo_name}/"
    if repo_marker in text:
        rel = text.split(repo_marker, 1)[1]
        yield ROOT / rel


def _resolve_sample_dir(sample_dir: str) -> Path | None:
    seen: set[str] = set()
    for candidate in _candidate_sample_dirs(sample_dir):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate
    return None


def _load_timestamps(sample_dir: Path) -> np.ndarray:
    for filename in ("rollout.npz", "sample.npz"):
        path = sample_dir / filename
        if not path.is_file():
            continue
        with np.load(path) as z:
            if "timestamps" in z:
                return np.asarray(z["timestamps"], dtype=np.float64).reshape(-1)
    raise KeyError(f"timestamps not found in {sample_dir}/rollout.npz or sample.npz")


def _summary(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit saved-frame timestamps by reading processed meta.jsonl rows and "
            "following sample_dir back to raw rollout/sample npz files."
        )
    )
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--output-dir", default="results/timestamp_audit")
    parser.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--frame-index", type=int, default=50)
    parser.add_argument(
        "--max-rows-per-solver",
        type=int,
        default=None,
        help="Optional debug limit after concatenating requested splits.",
    )
    args = parser.parse_args()

    frame_index = int(args.frame_index)
    if frame_index < 0:
        raise ValueError("--frame-index must be non-negative")

    processed_root = ROOT / args.processed_root
    output_dir = ROOT / args.output_dir
    sample_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for solver_dir in args.solvers:
        solver_dir = str(solver_dir)
        raw_rows: list[tuple[str, dict[str, Any]]] = []
        for split in args.splits:
            raw_rows.extend(
                (str(split), row)
                for row in _read_jsonl(
                    processed_root / solver_dir / str(split) / "meta.jsonl"
                )
            )
        if args.max_rows_per_solver is not None:
            raw_rows = raw_rows[: int(args.max_rows_per_solver)]

        for split, row in raw_rows:
            scenario_id = _scenario_id(row)
            scenario_key = _scenario_key(split, scenario_id)
            solver_name = str(row.get("solver_name", row.get("fde_name", solver_dir)))
            display_solver = _display_solver(solver_name)
            sample_dir_raw = str(row.get("sample_dir", ""))
            resolved = _resolve_sample_dir(sample_dir_raw)
            if resolved is None:
                missing_rows.append(
                    {
                        "solver": display_solver,
                        "solver_dir": solver_dir,
                        "split": split,
                        "scenario_id": scenario_id,
                        "scenario_key": scenario_key,
                        "sample_dir": sample_dir_raw,
                        "error": "sample_dir_not_found",
                    }
                )
                continue
            try:
                timestamps = _load_timestamps(resolved)
                if timestamps.size <= frame_index:
                    raise IndexError(
                        f"timestamps length {timestamps.size} <= frame index {frame_index}"
                    )
                timestamp = float(timestamps[frame_index])
            except Exception as exc:
                missing_rows.append(
                    {
                        "solver": display_solver,
                        "solver_dir": solver_dir,
                        "split": split,
                        "scenario_id": scenario_id,
                        "scenario_key": scenario_key,
                        "sample_dir": str(resolved),
                        "error": repr(exc),
                    }
                )
                continue

            rec = {
                "solver": display_solver,
                "solver_dir": solver_dir,
                "solver_name": solver_name,
                "split": split,
                "scenario_id": scenario_id,
                "scenario_key": scenario_key,
                "sample_index": row.get("sample_index"),
                "sample_dir": str(resolved),
                "num_timestamps": int(timestamps.size),
                f"timestamp_{frame_index}": timestamp,
                "timestamp_final": float(timestamps[-1]),
            }
            sample_rows.append(rec)
            by_scenario[scenario_key][display_solver] = rec

    if not sample_rows:
        raise RuntimeError("No timestamp rows could be loaded")

    timestamp_key = f"timestamp_{frame_index}"
    solver_summaries: list[dict[str, Any]] = []
    for solver in sorted({str(r["solver"]) for r in sample_rows}):
        rows = [r for r in sample_rows if r["solver"] == solver]
        summary = _summary(float(r[timestamp_key]) for r in rows)
        solver_summaries.append(
            {
                "solver": solver,
                "num_rows": len(rows),
                "num_missing": sum(1 for r in missing_rows if r["solver"] == solver),
                **{f"frame{frame_index}_{k}": v for k, v in summary.items()},
            }
        )

    desired_order = [
        _display_solver(s) for s in ("swe_hydrostatic", "swe_muscl_hr", "boussinesq")
    ]
    solvers_present = [
        s for s in desired_order if s in {r["solver"] for r in sample_rows}
    ]
    for solver in sorted({r["solver"] for r in sample_rows}):
        if solver not in solvers_present:
            solvers_present.append(solver)

    pair_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for i, solver_a in enumerate(solvers_present):
        for solver_b in solvers_present[i + 1 :]:
            rows_for_pair: list[dict[str, Any]] = []
            for scenario_key, recs in by_scenario.items():
                if solver_a not in recs or solver_b not in recs:
                    continue
                ta = float(recs[solver_a][timestamp_key])
                tb = float(recs[solver_b][timestamp_key])
                rows_for_pair.append(
                    {
                        "solver_a": solver_a,
                        "solver_b": solver_b,
                        "split": str(recs[solver_a].get("split", "")),
                        "scenario_id": str(
                            recs[solver_a].get("scenario_id", scenario_key)
                        ),
                        "scenario_key": scenario_key,
                        "timestamp_a": ta,
                        "timestamp_b": tb,
                        "signed_gap_a_minus_b": ta - tb,
                        "abs_gap": abs(ta - tb),
                    }
                )
            pair_rows.extend(rows_for_pair)
            gap_summary = _summary(r["abs_gap"] for r in rows_for_pair)
            signed_summary = _summary(r["signed_gap_a_minus_b"] for r in rows_for_pair)
            pair_summaries.append(
                {
                    "solver_a": solver_a,
                    "solver_b": solver_b,
                    "num_shared_scenarios": len(rows_for_pair),
                    **{f"abs_gap_{k}": v for k, v in gap_summary.items()},
                    **{f"signed_gap_{k}": v for k, v in signed_summary.items()},
                }
            )

    payload = {
        "audit_type": "saved_frame_timestamp_audit",
        "processed_root": str(args.processed_root),
        "splits": list(args.splits),
        "solvers": list(args.solvers),
        "frame_index": frame_index,
        "frame_index_note": (
            "With include_initial_state=true and save_every=5, frame index 50 is "
            "the stored solver-step-250 snapshot, not an interpolated common-time value."
        ),
        "num_loaded_rows": len(sample_rows),
        "num_missing_rows": len(missing_rows),
        "solver_summaries": solver_summaries,
        "pairwise_frame_timestamp_gap_summaries": pair_summaries,
        "missing_rows_preview": missing_rows[:20],
        "output_files": {
            "json": str(output_dir / "timestamp_audit.json"),
            "frame50_by_solver_csv": str(
                output_dir / f"frame{frame_index}_by_solver.csv"
            ),
            "pairwise_gaps_csv": str(
                output_dir / f"frame{frame_index}_pairwise_gaps.csv"
            ),
            "pairwise_summary_csv": str(
                output_dir / f"frame{frame_index}_pairwise_gap_summary.csv"
            ),
            "missing_csv": str(output_dir / "missing_timestamp_rows.csv"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "timestamp_audit.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    _write_csv(
        output_dir / f"frame{frame_index}_by_solver.csv",
        solver_summaries,
        sorted({k for row in solver_summaries for k in row.keys()}),
    )
    _write_csv(
        output_dir / f"frame{frame_index}_pairwise_gap_summary.csv",
        pair_summaries,
        sorted({k for row in pair_summaries for k in row.keys()}),
    )
    _write_csv(
        output_dir / f"frame{frame_index}_pairwise_gaps.csv",
        pair_rows,
        [
            "solver_a",
            "solver_b",
            "split",
            "scenario_id",
            "scenario_key",
            "timestamp_a",
            "timestamp_b",
            "signed_gap_a_minus_b",
            "abs_gap",
        ],
    )
    _write_csv(
        output_dir / "missing_timestamp_rows.csv",
        missing_rows,
        [
            "solver",
            "solver_dir",
            "split",
            "scenario_id",
            "scenario_key",
            "sample_dir",
            "error",
        ],
    )

    print(f"timestamp audit -> {output_dir / 'timestamp_audit.json'}")
    for row in solver_summaries:
        print(
            f"{row['solver']}: n={row['num_rows']} "
            f"frame{frame_index}_mean={row.get(f'frame{frame_index}_mean'):.8g} "
            f"min={row.get(f'frame{frame_index}_min'):.8g} "
            f"max={row.get(f'frame{frame_index}_max'):.8g}"
        )
    for row in pair_summaries:
        print(
            f"{row['solver_a']} vs {row['solver_b']}: "
            f"n={row['num_shared_scenarios']} "
            f"mean_abs_gap={row.get('abs_gap_mean'):.8g} "
            f"max_abs_gap={row.get('abs_gap_max'):.8g}"
        )


if __name__ == "__main__":
    main()
