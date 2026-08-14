#!/usr/bin/env python
"""Plot selected high-error appendix failure cases.

The figure is intentionally a failure-case panel, not an average-case or
representative-results figure. Each row uses a fixed intended dataset,
checkpoint, and sample ID. Missing inputs fail with the expected path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config


DEFAULT_OUTPUT_DIR = Path("results/reviewer_validation/failure_cases")
DEFAULT_FIGURE = Path("paper/figures/appendix_failure_cases.pdf")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    data_path: Path
    stats_path: Path
    config_path: Path
    checkpoint_path: Path
    sample_id: str
    loader: str
    expected_rel_l2: float
    note: str
    crop_label: str = ""


DEFAULT_CASES = [
    CaseSpec(
        key="rough_source_hard_case",
        label="Rough-source hard case",
        data_path=Path("data/processed/hydrostatic/test"),
        stats_path=Path("data/processed/hydrostatic/normalization_stats.json"),
        config_path=Path("configs/model/fno.yaml"),
        checkpoint_path=Path("experiments/fno/best.pt"),
        sample_id="sample_001849",
        loader="sharded",
        expected_rel_l2=0.751,
        note=(
            "Highest-error rough-source case in the canonical R1 ordinary-test "
            "per-sample diagnostics."
        ),
    ),
    CaseSpec(
        key="strict_rough_holdout_failure",
        label="Strict rough-source holdout",
        data_path=Path(
            "data/processed_strict_holdout/hydrostatic/source_holdout_rough/test_heldout"
        ),
        stats_path=Path(
            "data/processed_strict_holdout/hydrostatic/source_holdout_rough/normalization_stats.json"
        ),
        config_path=Path("configs/model/fno_holdout_source_rough.yaml"),
        checkpoint_path=Path("experiments/fno_holdout/source_rough/best.pt"),
        sample_id="sample_002127",
        loader="sharded",
        expected_rel_l2=1.095,
        note=(
            "Highest-error case in the canonical R1 family-strict rough-source "
            "per-sample diagnostics."
        ),
    ),
    CaseSpec(
        key="wet_dry_stress_failure",
        label="Wet-dry stress failure",
        data_path=Path(
            "data/processed_real_bathymetry_v2/appendix_coastline_stress/hydrostatic/test"
        ),
        stats_path=Path(
            "data/processed_real_bathymetry_v2/appendix_coastline_stress/hydrostatic/normalization_stats.json"
        ),
        config_path=Path("configs/model/fno.yaml"),
        checkpoint_path=Path("experiments/fno/best.pt"),
        sample_id="sample_000001",
        loader="flat_npy",
        expected_rel_l2=0.985,
        note=(
            "Selected coastline wet-dry stress case from the accepted v2 suite "
            "with the suite-specific 5.01 eta ceiling."
        ),
    ),
]


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected {label}: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing expected {label}: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    _require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_meta_jsonl(data_path: Path) -> dict[str, dict[str, Any]]:
    meta_path = data_path / "meta.jsonl"
    if not meta_path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_dir = str(row.get("sample_dir", ""))
            sample_id = Path(sample_dir).name if sample_dir else ""
            if not sample_id and row.get("sample_index") is not None:
                sample_id = f"sample_{int(row['sample_index']):06d}"
            if sample_id:
                out[sample_id] = row
    return out


def _load_sharded_sample(
    data_path: Path, sample_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _require_dir(data_path, "sharded dataset directory")
    manifest_path = data_path / "shards_manifest.json"
    _require_file(manifest_path, "shards manifest")
    manifest = _load_json(manifest_path)
    for shard in manifest.get("shards", []):
        shard_file = str(shard.get("file", ""))
        if not shard_file:
            continue
        shard_path = data_path / shard_file
        _require_file(shard_path, "dataset shard")
        with np.load(shard_path, allow_pickle=True) as data:
            sample_ids = [str(v) for v in data["sample_id"]]
            if sample_id not in sample_ids:
                continue
            local_idx = sample_ids.index(sample_id)
            inputs = data["inputs"][local_idx].astype(np.float32)
            targets = data["targets"][local_idx].astype(np.float32)
            meta = {
                "sample_id": sample_id,
                "scenario_id": str(data["scenario_id"][local_idx])
                if "scenario_id" in data
                else "",
                "source_type": str(data["source_type"][local_idx])
                if "source_type" in data
                else "",
                "bathymetry_type": str(data["bathymetry_type"][local_idx])
                if "bathymetry_type" in data
                else "",
                "source_strength": float(data["source_strength"][local_idx])
                if "source_strength" in data
                else float("nan"),
                "shard_path": str(shard_path),
                "local_index": int(local_idx),
            }
            return inputs, targets, meta
    raise FileNotFoundError(
        f"Sample {sample_id} not found under expected sharded dataset: {data_path}"
    )


def _load_flat_npy_sample(
    data_path: Path, sample_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _require_dir(data_path, "flat dataset directory")
    inputs_path = data_path / "inputs.npy"
    targets_path = data_path / "targets.npy"
    sample_ids_path = data_path / "sample_id.npy"
    _require_file(inputs_path, "inputs array")
    _require_file(targets_path, "targets array")
    _require_file(sample_ids_path, "sample_id array")

    sample_ids = [str(v) for v in np.load(sample_ids_path, allow_pickle=True)]
    if sample_id not in sample_ids:
        raise FileNotFoundError(
            f"Sample {sample_id} not found under expected flat dataset: {data_path}"
        )
    idx = sample_ids.index(sample_id)
    inputs = np.load(inputs_path).astype(np.float32)[idx]
    targets = np.load(targets_path).astype(np.float32)[idx]
    meta = {
        "sample_id": sample_id,
        "scenario_id": "",
        "source_type": "",
        "bathymetry_type": "",
        "source_strength": float("nan"),
        "array_index": int(idx),
    }
    meta.update(_load_meta_jsonl(data_path).get(sample_id, {}))
    if not meta.get("sample_id"):
        meta["sample_id"] = sample_id
    return inputs, targets, meta


def _denorm_input(arr: np.ndarray, stats: dict[str, Any], name: str) -> np.ndarray:
    channel = stats.get("inputs", {}).get(name)
    if not channel:
        return arr
    return arr * float(channel["scale"]) + float(channel["offset"])


def _denorm_target(arr: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    target = stats["targets"]
    return arr * float(target["scale"]) + float(target["offset"])


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    return float(
        np.linalg.norm((pred64 - target64).ravel())
        / (np.linalg.norm(target64.ravel()) + 1e-12)
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _model_output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out.get("mean", next(iter(out.values())))
    return out


def _load_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    _require_file(config_path, "model config")
    _require_file(checkpoint_path, "model checkpoint")
    cfg = load_config(config_path)
    cfg["device"] = str(device)
    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model


def _compute_case(
    case: CaseSpec,
    model_cache: dict[tuple[Path, Path], torch.nn.Module],
    device: torch.device,
    frame_index: int,
) -> dict[str, Any]:
    _require_file(case.stats_path, "normalization stats")
    stats = _load_json(case.stats_path)
    if case.loader == "sharded":
        inputs, targets, meta = _load_sharded_sample(case.data_path, case.sample_id)
    elif case.loader == "flat_npy":
        inputs, targets, meta = _load_flat_npy_sample(case.data_path, case.sample_id)
    else:
        raise ValueError(f"Unsupported case loader: {case.loader}")

    key = (case.config_path, case.checkpoint_path)
    if key not in model_cache:
        model_cache[key] = _load_model(case.config_path, case.checkpoint_path, device)
    model = model_cache[key]

    with torch.no_grad():
        x = torch.from_numpy(inputs[np.newaxis]).to(device)
        pred = _model_output(model, x)[0].detach().cpu().numpy()
    if pred.shape != targets.shape:
        raise ValueError(
            f"{case.key}: prediction shape {pred.shape} does not match target shape {targets.shape}"
        )
    if frame_index < 0 or frame_index >= targets.shape[0]:
        raise ValueError(
            f"{case.key}: frame {frame_index} outside [0, {targets.shape[0] - 1}]"
        )

    bathymetry = _denorm_input(inputs[0], stats, "bathymetry")
    source = _denorm_input(inputs[1], stats, "source")
    target_phys = _denorm_target(targets, stats)
    pred_phys = _denorm_target(pred, stats)
    error_frame = np.abs(pred_phys[frame_index] - target_phys[frame_index])
    rel_l2_normalized = _rel_l2(pred, targets)
    frame_rel_l2_normalized = _rel_l2(pred[frame_index], targets[frame_index])

    quality_violations = meta.get("quality_violations", [])
    if isinstance(quality_violations, str):
        quality_violations = [quality_violations]

    return {
        "key": case.key,
        "label": case.label,
        "note": case.note,
        "crop_label": case.crop_label,
        "data_path": str(case.data_path),
        "stats_path": str(case.stats_path),
        "config_path": str(case.config_path),
        "checkpoint_path": str(case.checkpoint_path),
        "sample_id": case.sample_id,
        "expected_rel_l2": float(case.expected_rel_l2),
        "computed_rel_l2": _rel_l2(pred_phys, target_phys),
        "computed_rel_l2_normalized": rel_l2_normalized,
        "frame_index": int(frame_index),
        "frame_rel_l2": _rel_l2(pred_phys[frame_index], target_phys[frame_index]),
        "frame_rel_l2_normalized": frame_rel_l2_normalized,
        "max_abs_error_frame": float(np.nanmax(error_frame)),
        "max_abs_target": float(np.nanmax(np.abs(target_phys))),
        "max_abs_prediction": float(np.nanmax(np.abs(pred_phys))),
        "meta": {
            "sample_id": str(meta.get("sample_id", case.sample_id)),
            "scenario_id": str(meta.get("scenario_id", "")),
            "source_type": str(meta.get("source_type", "")),
            "bathymetry_type": str(meta.get("bathymetry_type", "")),
            "source_strength": _finite_or_none(meta.get("source_strength")),
            "quality_status": str(meta.get("quality_status", "")),
            "quality_violations": quality_violations,
            "max_abs_eta": _finite_or_none(meta.get("max_abs_eta")),
        },
        "arrays": {
            "bathymetry": bathymetry,
            "source": source,
            "reference_frame": target_phys[frame_index],
            "prediction_frame": pred_phys[frame_index],
            "absolute_error_frame": error_frame,
        },
    }


def _positive_vmax(*arrays: np.ndarray) -> float:
    vmax = max(float(np.nanmax(np.abs(arr))) for arr in arrays)
    return vmax if vmax > 0.0 else 1.0


def _source_norm(source: np.ndarray) -> TwoSlopeNorm | None:
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    if vmin < 0.0 < vmax:
        lim = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    return None


def _draw_panel(
    fig, ax, array: np.ndarray, title: str, cmap: str, **kwargs: Any
) -> None:
    im = ax.imshow(array, origin="upper", cmap=cmap, **kwargs)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=5)


def _plot(rows: list[dict[str, Any]], output: Path, png_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(13.2, 2.45 * n_rows), constrained_layout=True
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Bathymetry",
        "Source",
        "Reference late $\\eta$",
        "FNO prediction",
        "$|$error$|$",
    ]
    for r, row in enumerate(rows):
        arrays = row["arrays"]
        ref = arrays["reference_frame"]
        pred = arrays["prediction_frame"]
        err = arrays["absolute_error_frame"]
        eta_vmax = _positive_vmax(ref, pred)
        eta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-eta_vmax, vmax=eta_vmax)
        err_vmax = float(np.nanmax(err)) if float(np.nanmax(err)) > 0.0 else 1.0
        source_norm = _source_norm(arrays["source"])
        panels = [
            (arrays["bathymetry"], "terrain", {}),
            (
                arrays["source"],
                "RdBu_r" if source_norm is not None else "viridis",
                {"norm": source_norm} if source_norm is not None else {},
            ),
            (ref, "RdBu_r", {"norm": eta_norm}),
            (pred, "RdBu_r", {"norm": eta_norm}),
            (err, "magma", {"vmin": 0.0, "vmax": err_vmax}),
        ]
        for c, (array, cmap, kwargs) in enumerate(panels):
            title = col_titles[c] if r == 0 else ""
            _draw_panel(fig, axes[r, c], array, title, cmap, **kwargs)

        meta = row["meta"]
        crop = f"\n{row['crop_label']}" if row.get("crop_label") else ""
        axes[r, 0].set_ylabel(
            f"{row['label']}{crop}\n{row['sample_id']}  rel-$L_2$={row['computed_rel_l2']:.3f}",
            fontsize=8,
            rotation=90,
            labelpad=10,
        )
        axes[r, 1].set_xlabel(
            f"{meta.get('source_type', '')} / {meta.get('bathymetry_type', '')}",
            fontsize=6,
        )

    fig.savefig(output, bbox_inches="tight")
    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k != "arrays"}
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "label",
        "sample_id",
        "data_path",
        "checkpoint_path",
        "frame_index",
        "computed_rel_l2",
        "computed_rel_l2_normalized",
        "expected_rel_l2",
        "frame_rel_l2",
        "frame_rel_l2_normalized",
        "max_abs_error_frame",
        "max_abs_target",
        "max_abs_prediction",
        "quality_status",
        "quality_violations",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = row.get("meta", {})
            writer.writerow(
                {
                    **{k: row.get(k, "") for k in fields},
                    "quality_status": meta.get("quality_status", ""),
                    "quality_violations": "; ".join(
                        str(v) for v in meta.get("quality_violations", [])
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--png-output", type=Path, default=None)
    parser.add_argument("--frame-index", type=int, default=49)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.key for case in DEFAULT_CASES],
        help="Restrict to one or more default cases. Repeat for multiple rows.",
    )
    args = parser.parse_args()

    selected = DEFAULT_CASES
    if args.case:
        requested = set(args.case)
        selected = [case for case in DEFAULT_CASES if case.key in requested]
    device = torch.device(args.device)
    model_cache: dict[tuple[Path, Path], torch.nn.Module] = {}
    rows = [
        _compute_case(case, model_cache, device, int(args.frame_index))
        for case in selected
    ]

    output_dir = Path(args.output_dir)
    json_path = output_dir / "failure_cases.json"
    csv_path = output_dir / "failure_cases.csv"
    png_output = (
        args.png_output
        if args.png_output is not None
        else args.figure_output.with_suffix(".png")
    )
    _plot(
        rows,
        Path(args.figure_output),
        Path(png_output) if png_output is not None else None,
    )
    _write_csv(rows, csv_path)
    payload = {
        "diagnostic": "selected_high_error_failure_cases",
        "interpretation": (
            "Rows are selected high-error examples requested for appendix reviewer validation; "
            "they are not averages or representative means."
        ),
        "frame_index": int(args.frame_index),
        "rows": [_json_safe(row) for row in rows],
        "figure_path": str(args.figure_output),
        "png_path": str(png_output),
        "csv_path": str(csv_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    for row in rows:
        print(
            f"{row['key']}: sample={row['sample_id']} "
            f"rel_l2={row['computed_rel_l2']:.6f} "
            f"rel_l2_norm={row['computed_rel_l2_normalized']:.6f} "
            f"frame_rel_l2={row['frame_rel_l2']:.6f}"
        )
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_pdf={args.figure_output}")
    print(f"saved_png={png_output}")


if __name__ == "__main__":
    main()
