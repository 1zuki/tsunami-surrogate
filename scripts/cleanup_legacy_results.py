#!/usr/bin/env python
"""Archive and optionally remove allowlisted legacy results after a validated run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
COMPLETION_SCHEMA = "tsunami-surrogate.evaluation-run-completion.v1"
ARCHIVE_SCHEMA = "tsunami-surrogate.legacy-results-archive.v1"

REPLACEMENT_PATTERNS = {
    "consolidated": ("all_results.json",),
    "dataset_summary": ("dataset_summary.json",),
    "parameter_counts": ("parameter_counts.json", "parameter_counts.csv"),
    "direct_accuracy": ("accuracy_*.json",),
    "direct_perframe": ("perframe_*.json",),
    "direct_physics": ("physics_diagnostics_*.json",),
    "conditional_window_accuracy": ("window_rollout_*.json",),
    "conditional_window_perframe": ("window_rollout_perframe_*.json",),
    "strict_holdout_summary": (
        "strict_holdout_summary.json",
        "strict_holdout_summary.csv",
    ),
    "real_bathymetry_direct": ("real_bathymetry_*.json",),
    "real_bathymetry_conditional_window": ("window5_real_bathymetry_*.json",),
    "uncertainty": ("uncertainty_hydrostatic_indist_m7.json",),
    "model_speed": ("speed_*.json",),
    "solver_speed": ("solver_speed_*.json",),
    "speed_table": ("speed_table.json", "speed_table.csv"),
}


class CleanupError(RuntimeError):
    """Raised when cleanup authorization or containment validation fails."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupError(f"Unreadable JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise CleanupError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_completion_manifest(path: Path) -> dict[str, Any]:
    completion = _read_object(path)
    if completion.get("schema_id") != COMPLETION_SCHEMA:
        raise CleanupError("Replacement-run completion schema mismatch")
    if completion.get("status") != "validated":
        raise CleanupError("Replacement run is not validated")
    run_root = path.parent.resolve()
    expected_root = (ROOT / "evaluation_runs").resolve()
    if run_root == expected_root or expected_root not in run_root.parents:
        raise CleanupError(
            "Completion manifest must belong to evaluation_runs/<run-id>/"
        )
    if run_root.name.endswith(".staging"):
        raise CleanupError("Staging runs cannot authorize legacy cleanup")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CleanupError("Completion manifest has no artifact inventory")
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise CleanupError("Invalid completion artifact row")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise CleanupError(f"Unsafe completion artifact path: {relative}")
        artifact = (run_root / relative).resolve()
        if artifact == run_root or run_root not in artifact.parents:
            raise CleanupError(f"Completion artifact escapes run root: {relative}")
        if not artifact.is_file():
            raise CleanupError(f"Replacement artifact is missing: {artifact}")
        if int(artifact.stat().st_size) != int(item.get("size_bytes", -1)):
            raise CleanupError(f"Replacement artifact size mismatch: {artifact}")
        if _sha256(artifact) != str(item.get("sha256", "")):
            raise CleanupError(f"Replacement artifact hash mismatch: {artifact}")
    run_manifest_relative = Path(str(completion.get("run_manifest", "")))
    if (
        run_manifest_relative.is_absolute()
        or ".." in run_manifest_relative.parts
        or not run_manifest_relative.parts
    ):
        raise CleanupError("Replacement run manifest path is unsafe")
    run_manifest = (run_root / run_manifest_relative).resolve()
    if run_manifest == run_root or run_root not in run_manifest.parents:
        raise CleanupError("Replacement run manifest escapes run root")
    if not run_manifest.is_file() or _sha256(run_manifest) != str(
        completion.get("run_manifest_sha256", "")
    ):
        raise CleanupError("Replacement run manifest is missing or changed")
    manifest = _read_object(run_manifest)
    if manifest.get("schema_id") != "tsunami-surrogate.evaluation-run-manifest.v1":
        raise CleanupError("Replacement run manifest schema mismatch")
    completion["_validated_run_manifest"] = manifest
    return completion


def replacement_patterns(completion: Mapping[str, Any]) -> tuple[str, ...]:
    manifest = completion.get("_validated_run_manifest")
    if not isinstance(manifest, Mapping):
        raise CleanupError("Validated replacement run manifest is unavailable")
    groups = {
        str(row.get("group"))
        for row in manifest.get("cells", [])
        if isinstance(row, Mapping)
    }
    groups.add("consolidated")
    patterns = {
        pattern for group in groups for pattern in REPLACEMENT_PATTERNS.get(group, ())
    }
    return tuple(sorted(patterns))


def collect_legacy_files(
    results_root: Path,
    *,
    patterns: tuple[str, ...],
) -> list[Path]:
    resolved_root = results_root.resolve()
    expected_root = (ROOT / "results").resolve()
    if resolved_root != expected_root:
        raise CleanupError("Legacy cleanup is restricted to the repository results/")
    files: set[Path] = set()
    for pattern in patterns:
        for path in results_root.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            resolved = path.resolve()
            if resolved_root not in resolved.parents:
                raise CleanupError(f"Legacy result escapes results/: {path}")
            files.add(path)
    return sorted(files)


def archive_legacy_files(
    *,
    files: list[Path],
    archive_path: Path,
    replacement: Mapping[str, Any],
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    archive_root = (ROOT / "results_archive").resolve()
    if archive_path == archive_root or archive_root not in archive_path.parents:
        raise CleanupError("Archive must be a child of results_archive/")
    if archive_path.exists():
        raise CleanupError(f"Archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging = archive_path.with_name(f".{archive_path.name}.tmp-{os.getpid()}")
    with tarfile.open(staging, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT))
    os.replace(staging, archive_path)
    return {
        "schema_id": ARCHIVE_SCHEMA,
        "replacement_run_id": replacement.get("run_id"),
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": _sha256(archive_path),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the checksummed archive and remove only allowlisted files.",
    )
    args = parser.parse_args()

    completion_path = Path(args.completion_manifest)
    if not completion_path.is_absolute():
        completion_path = ROOT / completion_path
    completion = validate_completion_manifest(completion_path.resolve())
    patterns = replacement_patterns(completion)
    files = collect_legacy_files(ROOT / "results", patterns=patterns)

    print(
        f"[legacy-results] replacement={completion.get('run_id')} "
        f"replacement_patterns={len(patterns)} files={len(files)}"
    )
    for path in files:
        print(f"  {path.relative_to(ROOT)}")
    if not args.execute:
        print("[legacy-results] dry-run only; nothing archived or removed")
        return

    archive_path = Path(args.archive)
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    archive_manifest = archive_legacy_files(
        files=files,
        archive_path=archive_path,
        replacement=completion,
    )
    manifest_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    staging = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(archive_manifest, handle, indent=2, sort_keys=True)
    os.replace(staging, manifest_path)

    for path in files:
        path.unlink()
    print(
        f"[legacy-results] archived={archive_path.relative_to(ROOT)} "
        f"removed={len(files)}"
    )


if __name__ == "__main__":
    main()
