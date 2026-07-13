#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.archive_common_time_stage_c import STAGE_C_SOURCES
from src.data_gen.common_time_v2 import (
    CONTRACT_SCHEMA_ID,
    authoritative_input_fingerprint,
    build_candidate_contract,
    code_state,
    contract_hash,
    hash_array,
    sha256_file,
    split_qualified_identity,
    stable_hash_payload,
)


H0_SCHEMA_ID = "tsunami-surrogate.common-time-v2.h0.v1"
SOLVERS = ("hydrostatic", "muscl_hr", "boussinesq")
SWE_SOLVERS = ("hydrostatic", "muscl_hr")
STATIC_FIELDS = (
    "bathymetry",
    "source_field",
    "rest_depth",
    "eta0",
    "initial_depth",
    "free_surface0",
)
DEFAULT_SPLITS = {"train": 10000, "eval": 1000, "test": 2500}
HORIZON = 0.175


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, Mapping):
                raise TypeError(f"Expected object at {path}:{line_number}")
            yield dict(raw)


def _scalar(payload: Mapping[str, np.ndarray], key: str) -> Any:
    values = np.asarray(payload[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{key} must contain exactly one scalar")
    return values[0].item()


def _sample_path(split_root: Path, solver: str, sample_index: int) -> Path:
    return (
        split_root
        / "raw"
        / solver
        / "samples"
        / f"sample_{sample_index:06d}"
        / "sample.npz"
    )


def _cache_path(split_root: Path, kind: str, sample_index: int) -> Path:
    return split_root / kind / f"sample_{sample_index:06d}.npz"


def _array_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and bool(np.array_equal(left, right, equal_nan=False))
    )


def _issue(
    issues: list[dict[str, Any]],
    *,
    split: str,
    scenario_id: str,
    code: str,
    message: str,
    blocking: bool = True,
) -> None:
    issues.append(
        {
            "split": split,
            "scenario_id": scenario_id,
            "qualified_id": f"{split}:{scenario_id}",
            "code": code,
            "message": message,
            "blocking": bool(blocking),
        }
    )


def _snapshot_reconciliation(
    split: str, split_root: Path, materialized_count: int
) -> dict[str, Any]:
    snapshots = sorted(split_root.rglob("dataset_config.snapshot.yaml"))
    advertised: list[dict[str, Any]] = []
    for path in snapshots:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            dataset = raw.get("dataset", {}) if isinstance(raw, Mapping) else {}
            value = dataset.get("num_samples") if isinstance(dataset, Mapping) else None
        except Exception as exc:
            advertised.append({"path": str(path), "error": str(exc)})
            continue
        advertised.append({"path": str(path), "advertised_num_samples": value})
    return {
        "split": split,
        "materialized_count": int(materialized_count),
        "snapshots": advertised,
        "mismatches": [
            item
            for item in advertised
            if item.get("advertised_num_samples") is not None
            and int(item["advertised_num_samples"]) != int(materialized_count)
        ],
        "authority": "materialized-manifest-cache-and-three-raw-reference-identity",
    }


def _verify_stage_c(
    repo_root: Path, archive_dir: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    source_records = []
    for rel in STAGE_C_SOURCES:
        path = repo_root / rel
        if not path.is_file():
            issues.append(
                {"code": "stage_c_source_missing", "path": str(path), "blocking": True}
            )
            continue
        source_records.append(
            {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    decision_path = (
        repo_root
        / "results/common_time_validation/dense_reference_validation/dense_validation/decision.json"
    )
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("status") != "fail":
            issues.append(
                {"code": "stage_c_negative_decision_missing", "blocking": True}
            )
    if archive_dir is None:
        issues.append({"code": "stage_c_archive_not_supplied", "blocking": True})
        return {"sources": source_records, "archive_verified": False}, issues
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.is_file():
        issues.append({"code": "stage_c_archive_manifest_missing", "blocking": True})
        return {"sources": source_records, "archive_verified": False}, issues
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_id") != (
        "tsunami-surrogate.common-time-v2.stage-c-archive.v1"
    ):
        issues.append({"code": "stage_c_archive_schema_mismatch", "blocking": True})
    if not bool(manifest.get("negative_scientific_decision", False)):
        issues.append(
            {"code": "stage_c_archive_negative_flag_missing", "blocking": True}
        )
    archived_items = manifest.get("files", [])
    expected_paths = set(STAGE_C_SOURCES)
    observed_paths = {
        str(item.get("source_relative_path", "")) for item in archived_items
    }
    if observed_paths != expected_paths or len(archived_items) != len(expected_paths):
        issues.append({"code": "stage_c_archive_file_set_mismatch", "blocking": True})
    bundle_payload = {
        "schema_id": manifest.get("schema_id"),
        "artifact_kind": manifest.get("artifact_kind"),
        "files": archived_items,
        "embedded_provenance": manifest.get("embedded_provenance", {}),
    }
    expected_bundle_hash = stable_hash_payload(
        artifact_kind="stage-c-archive-bundle",
        payload=bundle_payload,
        schema_id="tsunami-surrogate.common-time-v2.stage-c-archive.v1",
    )
    if (
        manifest.get("bundle_hash") != expected_bundle_hash
        or archive_dir.resolve().name != expected_bundle_hash
    ):
        issues.append(
            {"code": "stage_c_archive_bundle_hash_mismatch", "blocking": True}
        )
    archived = {item["source_relative_path"]: item for item in archived_items}
    for source in source_records:
        item = archived.get(source["relative_path"])
        if item is None or item.get("sha256") != source["sha256"]:
            issues.append(
                {
                    "code": "stage_c_archive_source_mismatch",
                    "path": source["relative_path"],
                    "blocking": True,
                }
            )
            continue
        archive_relative = str(item.get("archive_relative_path", ""))
        expected_relative = f"payload/{source['relative_path']}"
        if archive_relative != expected_relative:
            issues.append(
                {
                    "code": "stage_c_archive_path_mismatch",
                    "path": archive_relative,
                    "blocking": True,
                }
            )
            continue
        copied = (archive_dir / archive_relative).resolve()
        try:
            copied.relative_to(archive_dir.resolve())
        except ValueError:
            issues.append(
                {
                    "code": "stage_c_archive_path_escape",
                    "path": str(copied),
                    "blocking": True,
                }
            )
            continue
        if not copied.is_file() or sha256_file(copied) != source["sha256"]:
            issues.append(
                {
                    "code": "stage_c_archive_payload_mismatch",
                    "path": str(copied),
                    "blocking": True,
                }
            )
    return {
        "sources": source_records,
        "archive_dir": str(archive_dir),
        "archive_verified": not issues,
        "archive_manifest_hash": sha256_file(manifest_path),
    }, issues


def audit_h0(
    *,
    split_roots: Mapping[str, Path],
    expected_counts: Mapping[str, int],
    output_root: Path,
    repo_root: Path = ROOT,
    stage_c_archive: Path | None = None,
) -> Path:
    candidate = build_candidate_contract()
    candidate_hash = contract_hash(candidate)
    issues: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    endpoint_values: dict[str, dict[str, list[float]]] = {
        split: {solver: [] for solver in SOLVERS} for split in split_roots
    }
    split_summaries: dict[str, Any] = {}
    snapshot_records: list[dict[str, Any]] = []

    for split, root in split_roots.items():
        root = root.resolve()
        manifest_path = root / "synthetic/scenario_manifest.jsonl"
        if not manifest_path.is_file():
            _issue(
                issues,
                split=split,
                scenario_id="*",
                code="manifest_missing",
                message=str(manifest_path),
            )
            split_summaries[split] = {
                "materialized_count": 0,
                "expected_count": expected_counts[split],
            }
            continue
        rows = list(_read_jsonl(manifest_path))
        expected = int(expected_counts[split])
        if len(rows) != expected:
            _issue(
                issues,
                split=split,
                scenario_id="*",
                code="materialized_count_mismatch",
                message=f"manifest={len(rows)} expected={expected}",
            )
        ids = [str(row.get("scenario_id", "")) for row in rows]
        duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
        for scenario_id in duplicate_ids:
            _issue(
                issues,
                split=split,
                scenario_id=scenario_id,
                code="duplicate_scenario_id",
                message="duplicate within split",
            )
        snapshot_records.append(_snapshot_reconciliation(split, root, len(rows)))

        for row in rows:
            scenario_id = str(row.get("scenario_id", ""))
            try:
                sample_index = int(row["sample_index"])
                identity = split_qualified_identity(split, scenario_id)
            except Exception as exc:
                _issue(
                    issues,
                    split=split,
                    scenario_id=scenario_id or "*",
                    code="invalid_identity",
                    message=str(exc),
                )
                continue
            bathy_type = str(row.get("bathymetry_type", ""))
            source_type = str(row.get("source_type", ""))
            if not bathy_type or not source_type:
                _issue(
                    issues,
                    split=split,
                    scenario_id=scenario_id,
                    code="family_metadata_missing",
                    message="bathymetry_type/source_type required",
                )

            bathy_path = _cache_path(root, "bathymetry", sample_index)
            source_path = _cache_path(root, "sources", sample_index)
            paths = [
                bathy_path,
                source_path,
                *(_sample_path(root, solver, sample_index) for solver in SOLVERS),
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                _issue(
                    issues,
                    split=split,
                    scenario_id=scenario_id,
                    code="materialized_file_missing",
                    message="; ".join(missing),
                )
                continue

            try:
                with np.load(bathy_path, allow_pickle=False) as payload:
                    cache_bathy = np.asarray(payload["bathymetry"])
                    cache_bathy_type = str(_scalar(payload, "bathymetry_type"))
                with np.load(source_path, allow_pickle=False) as payload:
                    cache_source = np.asarray(payload["source_field"])
                    cache_source_type = str(_scalar(payload, "source_type"))
                    cache_strength_array = np.asarray(payload["source_strength"])
                    cache_strength = float(_scalar(payload, "source_strength"))
                if cache_bathy_type != bathy_type or cache_source_type != source_type:
                    _issue(
                        issues,
                        split=split,
                        scenario_id=scenario_id,
                        code="cache_family_mismatch",
                        message=f"manifest=({bathy_type},{source_type}) cache=({cache_bathy_type},{cache_source_type})",
                    )
                if (
                    not np.isfinite(cache_bathy).all()
                    or not np.isfinite(cache_source).all()
                    or not np.isfinite(cache_strength)
                ):
                    _issue(
                        issues,
                        split=split,
                        scenario_id=scenario_id,
                        code="nonfinite_authoritative_input",
                        message="cache contains nonfinite values",
                    )

                static_by_solver: dict[str, dict[str, np.ndarray]] = {}
                sample_meta: dict[str, Any] = {}
                for solver in SOLVERS:
                    path = _sample_path(root, solver, sample_index)
                    with np.load(path, allow_pickle=False) as payload:
                        missing_fields = [
                            field for field in STATIC_FIELDS if field not in payload
                        ]
                        if missing_fields:
                            raise KeyError(f"{solver} missing {missing_fields}")
                        static_by_solver[solver] = {
                            field: np.asarray(payload[field]) for field in STATIC_FIELDS
                        }
                        stored_id = str(_scalar(payload, "scenario_id"))
                        stored_solver = str(_scalar(payload, "solver_name"))
                        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
                    if stored_id != scenario_id:
                        _issue(
                            issues,
                            split=split,
                            scenario_id=scenario_id,
                            code="sample_identity_mismatch",
                            message=f"{solver} stored {stored_id}",
                        )
                    expected_solver_name = {
                        "hydrostatic": "swe_hydrostatic",
                        "muscl_hr": "swe_muscl_hr",
                        "boussinesq": "boussinesq",
                    }[solver]
                    if stored_solver != expected_solver_name:
                        _issue(
                            issues,
                            split=split,
                            scenario_id=scenario_id,
                            code="sample_solver_mismatch",
                            message=f"{solver} stored {stored_solver}",
                        )
                    if (
                        timestamps.size == 0
                        or not np.isfinite(timestamps).all()
                        or (timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0))
                    ):
                        _issue(
                            issues,
                            split=split,
                            scenario_id=scenario_id,
                            code="invalid_legacy_timestamps",
                            message=solver,
                        )
                    else:
                        endpoint_values[split][solver].append(float(timestamps[-1]))

                reference = static_by_solver["hydrostatic"]
                for solver, fields in static_by_solver.items():
                    for field in STATIC_FIELDS:
                        if not _array_equal_exact(reference[field], fields[field]):
                            _issue(
                                issues,
                                split=split,
                                scenario_id=scenario_id,
                                code="cross_reference_static_input_mismatch",
                                message=f"field={field} solver={solver}",
                            )
                if not _array_equal_exact(reference["bathymetry"], cache_bathy):
                    _issue(
                        issues,
                        split=split,
                        scenario_id=scenario_id,
                        code="bathymetry_cache_raw_mismatch",
                        message="hydrostatic raw differs from cache",
                    )
                if not _array_equal_exact(reference["source_field"], cache_source):
                    _issue(
                        issues,
                        split=split,
                        scenario_id=scenario_id,
                        code="source_cache_raw_mismatch",
                        message="hydrostatic raw differs from cache",
                    )

                eta_expected = np.asarray(
                    cache_strength * cache_source, dtype=reference["eta0"].dtype
                )
                h_expected = np.maximum(reference["rest_depth"] + eta_expected, 0.0)
                surface_expected = h_expected + reference["bathymetry"]
                for field, expected_values in (
                    ("eta0", eta_expected),
                    ("initial_depth", h_expected),
                    ("free_surface0", surface_expected),
                ):
                    if not np.array_equal(reference[field], expected_values):
                        _issue(
                            issues,
                            split=split,
                            scenario_id=scenario_id,
                            code="initial_condition_reconstruction_mismatch",
                            message=field,
                        )
                row_strength = float(row.get("source_strength", np.nan))
                if not np.isfinite(row_strength) or np.float32(
                    row_strength
                ) != np.float32(cache_strength):
                    _issue(
                        issues,
                        split=split,
                        scenario_id=scenario_id,
                        code="source_strength_mismatch",
                        message=f"manifest={row_strength} cache={cache_strength}",
                    )

                array_hashes = {
                    field: hash_array(reference[field]) for field in STATIC_FIELDS
                }
                fingerprint = authoritative_input_fingerprint(
                    split=split,
                    sample_index=sample_index,
                    scenario_id=scenario_id,
                    bathymetry_type=bathy_type,
                    source_type=source_type,
                    source_strength=cache_strength_array,
                    arrays={field: reference[field] for field in STATIC_FIELDS},
                )
                inventory_records.append(
                    {
                        **identity,
                        "sample_index": sample_index,
                        "bathymetry_type": bathy_type,
                        "source_type": source_type,
                        "source_strength": cache_strength,
                        "input_fingerprint": fingerprint,
                        "array_hashes": array_hashes,
                        "bathymetry_cache_path": str(bathy_path),
                        "source_cache_path": str(source_path),
                        "raw_sample_paths": {
                            solver: str(_sample_path(root, solver, sample_index))
                            for solver in SOLVERS
                        },
                    }
                )
            except Exception as exc:
                _issue(
                    issues,
                    split=split,
                    scenario_id=scenario_id,
                    code="scenario_audit_exception",
                    message=f"{type(exc).__name__}: {exc}",
                )

        split_summaries[split] = {
            "expected_count": expected,
            "manifest_count": len(rows),
            "audited_inventory_count": sum(
                1 for item in inventory_records if item["split"] == split
            ),
            "ordered_qualified_id_hash": stable_hash_payload(
                artifact_kind="ordered-split-qualified-identities",
                schema_id=H0_SCHEMA_ID,
                payload=[f"{split}:{scenario_id}" for scenario_id in ids],
            ),
            "duplicate_ids": duplicate_ids,
        }

    endpoint_summary: dict[str, Any] = {}
    for split, solver_map in endpoint_values.items():
        endpoint_summary[split] = {}
        for solver, values in solver_map.items():
            array = np.asarray(values, dtype=np.float64)
            summary = {
                "count": int(array.size),
                "min": float(np.min(array)) if array.size else None,
                "max": float(np.max(array)) if array.size else None,
                "quantiles": (
                    {str(q): float(np.quantile(array, q)) for q in (0.05, 0.5, 0.95)}
                    if array.size
                    else {}
                ),
                "count_below_candidate_horizon": int(np.count_nonzero(array < HORIZON)),
            }
            endpoint_summary[split][solver] = summary
            if solver in SWE_SOLVERS and summary["count_below_candidate_horizon"]:
                _issue(
                    issues,
                    split=split,
                    scenario_id="*",
                    code="swe_endpoint_below_candidate_horizon",
                    message=f"{solver}: {summary['count_below_candidate_horizon']}",
                )

    stage_c, stage_c_issues = _verify_stage_c(repo_root, stage_c_archive)
    for item in stage_c_issues:
        issues.append(
            {
                "split": "stage_c",
                "scenario_id": "*",
                "qualified_id": "stage_c:*",
                "message": item.get("path", item["code"]),
                **item,
            }
        )
    blocking = [item for item in issues if item.get("blocking", True)]
    decision = {
        "schema_id": H0_SCHEMA_ID,
        "artifact_kind": "common-time-v2-h0-decision",
        "audit_completed": True,
        "audit_passed": not blocking,
        "blocking_issue_count": len(blocking),
        "candidate_horizon": HORIZON,
        "candidate_horizon_precommitted": True,
        "three_reference_contract_accepted": False,
        "boussinesq_long_horizon_status": "unproven",
    }
    audit_code_state = code_state(repo_root)
    summary = {
        "schema_id": H0_SCHEMA_ID,
        "artifact_kind": "common-time-v2-h0-summary",
        "candidate_contract_hash": candidate_hash,
        "audit_code_state": audit_code_state,
        "split_summaries": split_summaries,
        "inventory_count": len(inventory_records),
        "issue_count": len(issues),
        "blocking_issue_count": len(blocking),
        "stage_c": stage_c,
        "decision": decision,
    }
    complete_evidence = {
        "candidate_contract_hash": candidate_hash,
        "summary": summary,
        "decision": decision,
        "inventory_records": inventory_records,
        "issues": issues,
        "endpoint_summary": endpoint_summary,
        "snapshot_reconciliation": snapshot_records,
        "stage_c": stage_c,
    }
    h0_hash = stable_hash_payload(
        artifact_kind="h0-complete-evidence",
        schema_id=H0_SCHEMA_ID,
        payload=complete_evidence,
    )
    final = output_root.resolve() / h0_hash
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite H0 artifact: {final}")
    staging = output_root.resolve() / f".{h0_hash}.staging"
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite H0 staging root: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        files: dict[str, Any] = {
            "h0_summary.json": summary,
            "h0_decision.json": decision,
            "h0_endpoint_summary.json": endpoint_summary,
            "h0_snapshot_reconciliation.json": snapshot_records,
            "candidate_contract.json": candidate,
        }
        for name, payload in files.items():
            (staging / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        with (staging / "h0_input_inventory.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for item in inventory_records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        with (staging / "h0_issues.jsonl").open("w", encoding="utf-8") as handle:
            for item in issues:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        (staging / "REPORT.md").write_text(
            "# Common-time-v2 H0 authoritative-input audit\n\n"
            f"- Audit completed: yes\n- Audit passed: {'yes' if decision['audit_passed'] else 'no'}\n"
            f"- Materialized records audited: {len(inventory_records)}\n"
            f"- Blocking issues: {len(blocking)}\n"
            "- Candidate horizon: 0.1750 elapsed benchmark-time units\n"
            "- Three-reference contract accepted: no\n"
            "- Boussinesq production long-horizon status: unproven\n",
            encoding="utf-8",
        )
        content = sorted(path.name for path in staging.iterdir() if path.is_file())
        (staging / "ARCHIVE_CONTENTS.txt").write_text(
            "\n".join(content) + "\n", encoding="utf-8"
        )
        checksum_paths = sorted(path for path in staging.iterdir() if path.is_file())
        with (staging / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
            for path in checksum_paths:
                handle.write(f"{sha256_file(path)}  {path.name}\n")
        output_root.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _mapping_arg(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, sep, path = value.partition("=")
        if not sep:
            raise ValueError(f"Expected SPLIT=PATH, got {value!r}")
        result[key] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal common-time-v2 H0 audit")
    parser.add_argument(
        "--split-root",
        action="append",
        default=[],
        help="Override split root as train=PATH, eval=PATH, or test=PATH",
    )
    parser.add_argument(
        "--expected-count",
        action="append",
        default=[],
        help="Override expected count as train=N, eval=N, or test=N",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/common_time_v2/h0",
    )
    parser.add_argument("--stage-c-archive", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "train": Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/train"),
        "eval": Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/eval"),
        "test": Path("/mnt/Windows/Users/Izu/tsunami-surrogate/data/test"),
    }
    roots.update(_mapping_arg(args.split_root))
    counts = dict(DEFAULT_SPLITS)
    for key, path in _mapping_arg(args.expected_count).items():
        counts[key] = int(str(path))
    final = audit_h0(
        split_roots=roots,
        expected_counts=counts,
        output_root=args.output_root,
        stage_c_archive=args.stage_c_archive,
    )
    print(final)


if __name__ == "__main__":
    main()
