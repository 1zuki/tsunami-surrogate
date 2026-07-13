#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.common_time_v2 import (
    CONTRACT_SCHEMA_ID,
    code_state,
    sha256_file,
    stable_hash_payload,
)


STAGE_C_SOURCES = (
    "results/common_time_validation/audit/paired_reference_audit.json",
    "results/common_time_validation/dense_reference_validation/smoke/summary.json",
    "results/common_time_validation/dense_reference_validation/smoke/decision.json",
    "results/common_time_validation/dense_reference_validation/smoke/scenario_metrics.jsonl",
    "results/common_time_validation/dense_reference_validation/dense_validation/summary.json",
    "results/common_time_validation/dense_reference_validation/dense_validation/decision.json",
    "results/common_time_validation/dense_reference_validation/dense_validation/scenario_metrics.jsonl",
    "configs/eval/common_time_alignment.yaml",
    "configs/eval/dense_reference_validation.yaml",
    "configs/eval/common_time_validation_scenarios.json",
)
ARCHIVE_SCHEMA_ID = "tsunami-surrogate.common-time-v2.stage-c-archive.v1"


def _embedded_provenance(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rel in STAGE_C_SOURCES:
        if not rel.endswith(".json"):
            continue
        path = repo_root / rel
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        for key in (
            "git_commit",
            "commit",
            "code_commit",
            "schema_id",
            "artifact_kind",
        ):
            if key in payload:
                out[f"{rel}:{key}"] = payload[key]
    return out


def archive_stage_c(*, repo_root: Path = ROOT, output_root: Path | None = None) -> Path:
    repo_root = repo_root.resolve()
    output_root = (
        repo_root / "artifacts/common_time_v2/stage_c_legacy_stride5_negative"
        if output_root is None
        else output_root.resolve()
    )
    records: list[dict[str, Any]] = []
    for rel in STAGE_C_SOURCES:
        source = repo_root / rel
        if not source.is_file():
            raise FileNotFoundError(f"Missing Stage C source: {source}")
        records.append(
            {
                "source_relative_path": rel,
                "archive_relative_path": f"payload/{rel}",
                "size_bytes": int(source.stat().st_size),
                "sha256": sha256_file(source),
            }
        )

    bundle_payload = {
        "schema_id": ARCHIVE_SCHEMA_ID,
        "artifact_kind": "stage-c-legacy-stride5-negative-source-inventory",
        "files": records,
        "embedded_provenance": _embedded_provenance(repo_root),
    }
    bundle_hash = stable_hash_payload(
        artifact_kind="stage-c-archive-bundle",
        payload=bundle_payload,
        schema_id=ARCHIVE_SCHEMA_ID,
    )
    final = output_root / bundle_hash
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite Stage C archive: {final}")
    staging = output_root / f".{bundle_hash}.staging"
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite Stage C staging root: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for record in records:
            source = repo_root / record["source_relative_path"]
            destination = staging / record["archive_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256_file(destination) != record["sha256"]:
                raise RuntimeError(f"Copied Stage C hash mismatch: {destination}")
            if int(destination.stat().st_size) != record["size_bytes"]:
                raise RuntimeError(f"Copied Stage C size mismatch: {destination}")

        for record in records:
            source = repo_root / record["source_relative_path"]
            if sha256_file(source) != record["sha256"]:
                raise RuntimeError(f"Stage C source changed during archival: {source}")

        manifest = {
            **bundle_payload,
            "bundle_hash": bundle_hash,
            "archive_code_state": code_state(repo_root),
            "source_files_unchanged": True,
            "negative_scientific_decision": True,
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        with (staging / "README.md").open("w", encoding="utf-8") as handle:
            handle.write(
                "# Stage C legacy stride-5 negative result\n\n"
                "This bundle preserves the completed Stage C evidence byte-for-byte. "
                "The implementation/replay checks passed, but every preregistered scientific "
                "interpolation gate failed. Do not reinterpret or relax those gates.\n"
            )
        content_paths = sorted(record["archive_relative_path"] for record in records)
        content_paths.extend(["manifest.json", "README.md"])
        with (staging / "ARCHIVE_CONTENTS.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(content_paths) + "\n")
        checksum_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        with (staging / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
            for path in checksum_paths:
                handle.write(f"{sha256_file(path)}  {path.relative_to(staging)}\n")
        output_root.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive immutable Stage C evidence")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    print(archive_stage_c(repo_root=args.repo_root, output_root=args.output_root))


if __name__ == "__main__":
    main()
