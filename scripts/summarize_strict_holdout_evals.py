#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import save_json


HOLDOUTS = [
    {
        "label": "bathymetry_trench",
        "family": "bathymetry",
        "config_heldout": "configs/model/fno_holdout_bathymetry_trench.yaml",
        "config_id": "configs/model/fno_holdout_bathymetry_trench_eval_id.yaml",
        "config_full": "configs/model/fno_full_on_holdout_bathymetry_trench.yaml",
        "checkpoint": "experiments/fno_holdout/bathymetry_trench/best.pt",
        "full_checkpoint": "experiments/fno/best.pt",
        "manifest": "data/processed_strict_holdout/hydrostatic/bathymetry_holdout_trench/holdout_manifest.json",
    },
    {
        "label": "bathymetry_continental",
        "family": "bathymetry",
        "config_heldout": "configs/model/fno_holdout_bathymetry_continental.yaml",
        "config_id": "configs/model/fno_holdout_bathymetry_continental_eval_id.yaml",
        "config_full": "configs/model/fno_full_on_holdout_bathymetry_continental.yaml",
        "checkpoint": "experiments/fno_holdout/bathymetry_continental/best.pt",
        "full_checkpoint": "experiments/fno/best.pt",
        "manifest": "data/processed_strict_holdout/hydrostatic/bathymetry_holdout_continental/holdout_manifest.json",
    },
    {
        "label": "source_rough",
        "family": "source",
        "config_heldout": "configs/model/fno_holdout_source_rough.yaml",
        "config_id": "configs/model/fno_holdout_source_rough_eval_id.yaml",
        "config_full": "configs/model/fno_full_on_holdout_source_rough.yaml",
        "checkpoint": "experiments/fno_holdout/source_rough/best.pt",
        "full_checkpoint": "experiments/fno/best.pt",
        "manifest": "data/processed_strict_holdout/hydrostatic/source_holdout_rough/holdout_manifest.json",
    },
    {
        "label": "source_okada_like",
        "family": "source",
        "config_heldout": "configs/model/fno_holdout_source_okada_like.yaml",
        "config_id": "configs/model/fno_holdout_source_okada_like_eval_id.yaml",
        "config_full": "configs/model/fno_full_on_holdout_source_okada_like.yaml",
        "checkpoint": "experiments/fno_holdout/source_okada_like/best.pt",
        "full_checkpoint": "experiments/fno/best.pt",
        "manifest": "data/processed_strict_holdout/hydrostatic/source_holdout_okada_like/holdout_manifest.json",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object JSON in {path}")
    return data


def _eval_output_dir(config_path: Path) -> Path:
    cfg = load_config(config_path)
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    path = Path(output_dir)
    return path if path.is_absolute() else ROOT / path


def _split_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in manifest.get("splits", []):
        if not isinstance(split, dict):
            continue
        kind = split.get("kind", split.get("split"))
        if kind is None:
            continue
        counts[str(kind)] = int(split.get("num_samples", 0))
    return counts


def _metric(metrics: dict[str, Any] | None, key: str) -> float | None:
    if metrics is None or key not in metrics:
        return None
    value = metrics[key]
    if value is None:
        return None
    return float(value)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return float(num / den)


def _physics_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = _read_json(path)
    diagnostics = data.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return None
    spectral = diagnostics.get("spectral_band_rel_l2", {})
    if not isinstance(spectral, dict):
        spectral = {}
    return {
        "num_samples_seen": data.get("num_samples_seen"),
        "evaluation_mode": data.get("evaluation_mode"),
        "target_units": data.get("target_units"),
        "free_surface_integral_rel_l2": (
            diagnostics.get("free_surface_integral", {}).get("rel_l2")
            if isinstance(diagnostics.get("free_surface_integral"), dict)
            else None
        ),
        "mass_proxy_integral_rel_l2": (
            diagnostics.get("mass_proxy_integral", {}).get("rel_l2")
            if isinstance(diagnostics.get("mass_proxy_integral"), dict)
            else None
        ),
        "spectral_band_rel_l2": {
            k: v.get("rel_l2")
            for k, v in spectral.items()
            if isinstance(v, dict) and "rel_l2" in v
        },
        "sample_rel_l2_summary": diagnostics.get("sample_rel_l2_summary"),
    }


def _perframe_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = _read_json(path)
    frames = data.get("per_frame", [])
    phys_frames = data.get("per_frame_physical", [])
    out: dict[str, Any] = {
        "num_samples": data.get("num_samples"),
        "num_frames": data.get("num_frames"),
    }
    if isinstance(frames, list) and frames:
        out["final_frame"] = frames[-1]
    if isinstance(phys_frames, list) and phys_frames:
        out["final_frame_physical"] = phys_frames[-1]
    return out


def _build_row(
    spec: dict[str, str], *, allow_missing: bool
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    heldout_dir = _eval_output_dir(ROOT / spec["config_heldout"])
    id_dir = _eval_output_dir(ROOT / spec["config_id"])
    full_dir = _eval_output_dir(ROOT / spec["config_full"])
    heldout_metrics_path = heldout_dir / "metrics.json"
    id_metrics_path = id_dir / "metrics.json"
    full_metrics_path = full_dir / "metrics.json"

    heldout_metrics = (
        _read_json(heldout_metrics_path) if heldout_metrics_path.is_file() else None
    )
    id_metrics = _read_json(id_metrics_path) if id_metrics_path.is_file() else None
    full_metrics = (
        _read_json(full_metrics_path) if full_metrics_path.is_file() else None
    )
    for path, data in [
        (heldout_metrics_path, heldout_metrics),
        (id_metrics_path, id_metrics),
        (full_metrics_path, full_metrics),
    ]:
        if data is None:
            missing.append(str(path))

    if missing and not allow_missing:
        return {}, missing

    manifest_path = ROOT / spec["manifest"]
    manifest = _read_json(manifest_path)
    counts = _split_counts(manifest)
    sanity = manifest.get("sanity_checks", {})
    if not isinstance(sanity, dict):
        sanity = {}

    id_rel_l2 = _metric(id_metrics, "rel_l2")
    heldout_rel_l2 = _metric(heldout_metrics, "rel_l2")
    full_rel_l2 = _metric(full_metrics, "rel_l2")
    id_rel_l2_phys = _metric(id_metrics, "rel_l2_physical")
    heldout_rel_l2_phys = _metric(heldout_metrics, "rel_l2_physical")
    full_rel_l2_phys = _metric(full_metrics, "rel_l2_physical")

    return {
        "label": spec["label"],
        "family": spec["family"],
        "holdout_key": manifest.get("holdout_key"),
        "holdout_value": manifest.get("holdout_value"),
        "checkpoint": spec["checkpoint"],
        "full_checkpoint": spec["full_checkpoint"],
        "config_id": spec["config_id"],
        "config_heldout": spec["config_heldout"],
        "config_full": spec["config_full"],
        "manifest": spec["manifest"],
        "normalization_policy": manifest.get("normalization_policy"),
        "normalization_from_train_only": sanity.get("normalization_from_train_only"),
        "sanity_passed": sanity.get("passed"),
        "counts": {
            "train": counts.get("train"),
            "val": counts.get("val"),
            "test_id": counts.get("test_id"),
            "test_heldout": counts.get("test_heldout"),
        },
        "id": {
            "output_dir": str(id_dir),
            "metrics": id_metrics,
            "perframe": _perframe_summary(id_dir / "perframe.json"),
            "physics": _physics_summary(id_dir / "physics_diagnostics.json"),
        },
        "heldout": {
            "output_dir": str(heldout_dir),
            "metrics": heldout_metrics,
            "perframe": _perframe_summary(heldout_dir / "perframe.json"),
            "physics": _physics_summary(heldout_dir / "physics_diagnostics.json"),
        },
        "full_model_on_heldout": {
            "output_dir": str(full_dir),
            "metrics": full_metrics,
        },
        "degradation": {
            "rel_l2_ratio_heldout_over_id": _ratio(heldout_rel_l2, id_rel_l2),
            "rel_l2_delta_heldout_minus_id": (
                None
                if heldout_rel_l2 is None or id_rel_l2 is None
                else heldout_rel_l2 - id_rel_l2
            ),
            "rel_l2_ratio_heldout_over_full": _ratio(heldout_rel_l2, full_rel_l2),
            "rel_l2_delta_heldout_minus_full": (
                None
                if heldout_rel_l2 is None or full_rel_l2 is None
                else heldout_rel_l2 - full_rel_l2
            ),
            "rel_l2_physical_ratio_heldout_over_id": _ratio(
                heldout_rel_l2_phys, id_rel_l2_phys
            ),
            "rel_l2_physical_delta_heldout_minus_id": (
                None
                if heldout_rel_l2_phys is None or id_rel_l2_phys is None
                else heldout_rel_l2_phys - id_rel_l2_phys
            ),
            "rel_l2_physical_ratio_heldout_over_full": _ratio(
                heldout_rel_l2_phys, full_rel_l2_phys
            ),
            "rel_l2_physical_delta_heldout_minus_full": (
                None
                if heldout_rel_l2_phys is None or full_rel_l2_phys is None
                else heldout_rel_l2_phys - full_rel_l2_phys
            ),
        },
    }, missing


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    id_metrics = row["id"].get("metrics") or {}
    heldout_metrics = row["heldout"].get("metrics") or {}
    full_metrics = row["full_model_on_heldout"].get("metrics") or {}
    counts = row.get("counts", {})
    degradation = row.get("degradation", {})
    id_physics = row["id"].get("physics") or {}
    heldout_physics = row["heldout"].get("physics") or {}
    return {
        "label": row.get("label"),
        "family": row.get("family"),
        "holdout_key": row.get("holdout_key"),
        "holdout_value": row.get("holdout_value"),
        "train_n": counts.get("train"),
        "val_n": counts.get("val"),
        "test_id_n": counts.get("test_id"),
        "test_heldout_n": counts.get("test_heldout"),
        "sanity_passed": row.get("sanity_passed"),
        "normalization_from_train_only": row.get("normalization_from_train_only"),
        "id_rel_l2": id_metrics.get("rel_l2"),
        "heldout_rel_l2": heldout_metrics.get("rel_l2"),
        "full_model_heldout_rel_l2": full_metrics.get("rel_l2"),
        "rel_l2_ratio_heldout_over_id": degradation.get("rel_l2_ratio_heldout_over_id"),
        "rel_l2_delta_heldout_minus_id": degradation.get(
            "rel_l2_delta_heldout_minus_id"
        ),
        "rel_l2_ratio_heldout_over_full": degradation.get(
            "rel_l2_ratio_heldout_over_full"
        ),
        "rel_l2_delta_heldout_minus_full": degradation.get(
            "rel_l2_delta_heldout_minus_full"
        ),
        "id_rel_l2_physical": id_metrics.get("rel_l2_physical"),
        "heldout_rel_l2_physical": heldout_metrics.get("rel_l2_physical"),
        "full_model_heldout_rel_l2_physical": full_metrics.get("rel_l2_physical"),
        "rel_l2_physical_ratio_heldout_over_id": degradation.get(
            "rel_l2_physical_ratio_heldout_over_id"
        ),
        "rel_l2_physical_delta_heldout_minus_id": degradation.get(
            "rel_l2_physical_delta_heldout_minus_id"
        ),
        "rel_l2_physical_ratio_heldout_over_full": degradation.get(
            "rel_l2_physical_ratio_heldout_over_full"
        ),
        "rel_l2_physical_delta_heldout_minus_full": degradation.get(
            "rel_l2_physical_delta_heldout_minus_full"
        ),
        "id_rmse": id_metrics.get("rmse"),
        "heldout_rmse": heldout_metrics.get("rmse"),
        "full_model_heldout_rmse": full_metrics.get("rmse"),
        "id_mae": id_metrics.get("mae"),
        "heldout_mae": heldout_metrics.get("mae"),
        "full_model_heldout_mae": full_metrics.get("mae"),
        "id_num_samples": id_metrics.get("num_samples"),
        "heldout_num_samples": heldout_metrics.get("num_samples"),
        "full_model_heldout_num_samples": full_metrics.get("num_samples"),
        "id_eta_integral_rel_l2": id_physics.get("free_surface_integral_rel_l2"),
        "heldout_eta_integral_rel_l2": heldout_physics.get(
            "free_surface_integral_rel_l2"
        ),
        "id_mass_proxy_rel_l2": id_physics.get("mass_proxy_integral_rel_l2"),
        "heldout_mass_proxy_rel_l2": heldout_physics.get("mass_proxy_integral_rel_l2"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [_csv_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not flat_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize strict held-out family FNO evals."
    )
    p.add_argument(
        "--output", default="results/strict_holdout/strict_holdout_summary.json"
    )
    p.add_argument(
        "--csv-output", default="results/strict_holdout/strict_holdout_summary.csv"
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write partial summary instead of failing when metrics.json files are missing.",
    )
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in HOLDOUTS:
        row, row_missing = _build_row(spec, allow_missing=bool(args.allow_missing))
        missing.extend(row_missing)
        if row:
            rows.append(row)

    if missing and not args.allow_missing:
        missing_text = "\n".join(f"  - {p}" for p in missing)
        print(
            "Missing strict-holdout eval outputs. Run scripts/run_strict_holdout_evals.sh first:\n"
            f"{missing_text}",
            file=sys.stderr,
        )
        sys.exit(1)

    output = {
        "summary_type": "strict_holdout_fno",
        "num_holdouts": len(rows),
        "expected_holdouts": len(HOLDOUTS),
        "missing_outputs": missing,
        "rows": rows,
    }
    save_json(output, ROOT / args.output)
    _write_csv(ROOT / args.csv_output, rows)
    print(
        f"[strict-holdout] wrote {args.output} and {args.csv_output} ({len(rows)} rows)"
    )
    if missing:
        print(f"[strict-holdout] missing {len(missing)} expected metrics files")


if __name__ == "__main__":
    main()
