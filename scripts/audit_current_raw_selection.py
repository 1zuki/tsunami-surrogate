#!/usr/bin/env python
"""Audit the mounted common-time-v2 test bundle and build selection evidence.

This audit is separate from the historical stride-5 audit.  The mounted bundle
stores requested-output publications in ``sample.npz`` and keeps the 128-cell
solver inputs in the paired bathymetry/source caches.  Use ``--raw-root`` and
``--cache-root`` when the data is not mounted at the historical Windows path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test/raw")
DEFAULT_CACHE_ROOT = Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test")
SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")
SOLVER_NAMES = {
    "hydrostatic": "swe_hydrostatic",
    "muscl_hr": "swe_muscl_hr",
    "boussinesq": "boussinesq",
}
COMMON_FIELDS = (
    "bathymetry",
    "source_field",
    "eta0",
    "rest_depth",
    "initial_depth",
    "free_surface0",
)
EXPECTED_TIMES = np.asarray([8.4 * index for index in range(1, 51)], dtype=np.float64)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sample_dir(solver: str, index: int, *, raw_root: Path) -> Path:
    return raw_root / solver / "samples" / f"sample_{index:06d}"


def _cache_path(kind: str, index: int, *, cache_root: Path) -> Path:
    return cache_root / kind / f"sample_{index:06d}.npz"


def _audit_sample(
    solver: str,
    index: int,
    manifest_row: dict[str, Any],
    *,
    raw_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    sample_dir = _sample_dir(solver, index, raw_root=raw_root)
    issues: list[str] = []
    meta_path = sample_dir / "meta.json"
    sample_path = sample_dir / "sample.npz"
    publication_path = sample_dir / "publication.json"
    if not sample_path.is_file() or not meta_path.is_file():
        return {}, [f"missing_publication:{solver}:{index:06d}"]

    meta = _load_json(meta_path)
    publication = _load_json(publication_path) if publication_path.is_file() else {}
    with np.load(sample_path, allow_pickle=False) as payload:
        arrays = {field: np.asarray(payload[field]) for field in COMMON_FIELDS}
        trajectory = np.asarray(payload["trajectory_eta"], dtype=np.float64)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        scenario_id = str(np.asarray(payload["scenario_id"]).reshape(-1)[0])
        solver_name = str(np.asarray(payload["solver_name"]).reshape(-1)[0])

    expected_id = str(manifest_row["scenario_id"])
    if scenario_id != expected_id:
        issues.append(f"scenario_id:{solver}:{index:06d}:{scenario_id}")
    if solver_name != SOLVER_NAMES[solver]:
        issues.append(f"solver_name:{solver}:{index:06d}:{solver_name}")
    if timestamps.shape != EXPECTED_TIMES.shape or not np.allclose(
        timestamps, EXPECTED_TIMES, atol=1.0e-12, rtol=0.0
    ):
        issues.append(f"timestamps:{solver}:{index:06d}")
    if trajectory.shape != (50, 64, 64) or not np.isfinite(trajectory).all():
        issues.append(f"trajectory:{solver}:{index:06d}")
    for field, array in arrays.items():
        if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            issues.append(f"nonfinite_{field}:{solver}:{index:06d}")
    if meta.get("quality_status") != "ok" or meta.get("quality_violations"):
        issues.append(f"quality:{solver}:{index:06d}")
    if publication.get("quality_status") not in (None, "ok"):
        issues.append(f"publication_quality:{solver}:{index:06d}")
    if meta.get("scenario_id") != expected_id:
        issues.append(f"meta_scenario_id:{solver}:{index:06d}")

    return (
        {
            "solver": solver,
            "sample_index": int(index),
            "scenario_id": expected_id,
            "bathymetry_type": str(meta.get("bathymetry_type", manifest_row["bathymetry_type"])),
            "source_type": str(meta.get("source_type", manifest_row["source_type"])),
            "source_strength": float(meta.get("source_strength", manifest_row["source_strength"])),
            "input_hashes": {field: _array_hash(arrays[field]) for field in COMMON_FIELDS},
            "trajectory_hash": _array_hash(trajectory),
            "input_fingerprint": str(meta.get("input_fingerprint", "")),
            "code_state_hash": str(meta.get("code_state_hash", "")),
            "contract_hash": str(meta.get("contract_hash", "")),
            "resolved_config_hash": str(meta.get("resolved_config_hash", "")),
            "sample_dir": str(sample_dir),
        },
        issues,
    )


def run_audit(
    output: Path,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    raw_root = raw_root.resolve()
    cache_root = cache_root.resolve()
    output = output.resolve()
    manifest_path = cache_root / "synthetic" / "scenario_manifest.jsonl"
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest_rows) != 2500:
        raise RuntimeError(f"Expected 2500 scenario rows, found {len(manifest_rows)}")

    issues: list[str] = []
    eligible: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for position, manifest_row in enumerate(manifest_rows, 1):
        index = int(manifest_row["sample_index"])
        scenario_id = str(manifest_row["scenario_id"])
        per_solver: dict[str, dict[str, Any]] = {}
        for solver in SOLVERS:
            record, sample_issues = _audit_sample(
                solver,
                index,
                manifest_row,
                raw_root=raw_root,
            )
            per_solver[solver] = record
            issues.extend(sample_issues)
        if any(not record for record in per_solver.values()):
            continue

        reference = per_solver[SOLVERS[0]]
        for solver in SOLVERS[1:]:
            other = per_solver[solver]
            for field in COMMON_FIELDS:
                if other["input_hashes"][field] != reference["input_hashes"][field]:
                    issues.append(f"shared_input:{scenario_id}:{field}:{solver}")
            for field in ("bathymetry_type", "source_type", "source_strength"):
                if field == "source_strength":
                    equal = abs(float(other[field]) - float(reference[field])) <= 1.0e-12
                else:
                    equal = other[field] == reference[field]
                if not equal:
                    issues.append(f"shared_metadata:{scenario_id}:{field}:{solver}")

        if not any(item.startswith(f"shared_input:{scenario_id}:") for item in issues):
            eligible.append(
                {
                    "scenario_id": scenario_id,
                    "bathymetry_type": reference["bathymetry_type"],
                    "source_type": reference["source_type"],
                    "source_strength": reference["source_strength"],
                }
            )
        sample_records.append(
            {
                "scenario_id": scenario_id,
                "sample_index": index,
                "solvers": per_solver,
            }
        )
        if position % 250 == 0:
            print(f"audited={position}/2500", flush=True)

    config_path = ROOT / "configs/data/dataset_test.yaml"
    code_hashes = sorted(
        {
            record["code_state_hash"]
            for sample in sample_records
            for record in sample["solvers"].values()
            if record.get("code_state_hash")
        }
    )
    contract_hashes = sorted(
        {
            record["contract_hash"]
            for sample in sample_records
            for record in sample["solvers"].values()
            if record.get("contract_hash")
        }
    )
    payload: dict[str, Any] = {
        "schema_id": "tsunami-surrogate.alignment.v1",
        "artifact_kind": "paired-reference-audit",
        "status": "pass" if not issues else "fail",
        "created_at_utc": "2026-11-17T00:00:00+00:00",
        "config": {
            "audit": {
                "expected_scenario_count": 2500,
                "raw_test_solver_roots": {
                    solver: str(raw_root / solver / "samples") for solver in SOLVERS
                },
                "scenario_manifest_path": str(manifest_path),
                "generation_config_path": str(config_path),
                "generation_config_sha256": _sha256_file(config_path),
                "common_time_grid": EXPECTED_TIMES.tolist(),
                "raw_contract_hashes": contract_hashes,
                "raw_code_state_hashes": code_hashes,
            }
        },
        "counts": {
            "manifest_scenarios": len(manifest_rows),
            "complete_paired_scenarios": len(sample_records),
            "eligible_scenarios": len(eligible),
            "issue_count": len(issues),
        },
        "issues": issues[:1000],
        "eligible_scenarios": eligible,
        "sample_records_hash": _stable_hash(sample_records),
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "raw_root": str(raw_root),
            "cache_root": str(cache_root),
        },
    }
    payload["audit_hash"] = _stable_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={payload['status']} eligible={len(eligible)} issues={len(issues)} output={output}",
        flush=True,
    )
    if issues:
        raise SystemExit(1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/common_time_validation/audit/current_raw_selection.json"),
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()
    run_audit(args.output, raw_root=args.raw_root, cache_root=args.cache_root)


if __name__ == "__main__":
    main()
