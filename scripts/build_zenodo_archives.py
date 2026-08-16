#!/usr/bin/env python
"""Build checksum-bound Zenodo archives for the common-time-v2 paper release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "evaluation_runs/final-v2-paper-full-r1"
DEFAULT_RELEASE_ROOT = ROOT / "release/common-time-v2-zenodo"
DEFAULT_PROJECT_PYTHON = ROOT / ".venv/bin/python"
RELEASE_VERSION = "2.0.0"
DEFAULT_PREPARED_DATE = "2026-08-15"


@dataclass(frozen=True)
class ArchiveSpec:
    relative_output: str
    sources: tuple[Path, ...]
    purpose: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _iter_files(sources: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for source in sources:
        source = source.resolve()
        candidates = [source] if source.is_file() else source.rglob("*")
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def _source_summary(sources: tuple[Path, ...]) -> tuple[int, int]:
    files = list(_iter_files(sources))
    return len(files), sum(path.stat().st_size for path in files)


def _source_snapshot(
    sources: tuple[Path, ...],
) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in _iter_files(sources):
        stat = path.stat()
        snapshot[path] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _validate_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_manifest_path = run_root / "run_manifest.json"
    completion_path = run_root / "completion_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    completion = _load_json(completion_path)

    if completion.get("status") != "validated":
        raise ValueError(f"Evaluation run is not validated: {completion_path}")
    if completion.get("run_id") != run_manifest.get("run_id"):
        raise ValueError("Evaluation run/completion IDs differ")
    if completion.get("run_manifest_sha256") != _sha256(run_manifest_path):
        raise ValueError("Evaluation run manifest hash mismatch")

    for row in completion.get("artifacts", []):
        relative = str(row["path"])
        path = run_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation artifact: {path}")
        expected_size = row.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise ValueError(f"Evaluation artifact size mismatch: {path}")
        if _sha256(path) != str(row["sha256"]):
            raise ValueError(f"Evaluation artifact hash mismatch: {path}")

    return run_manifest, completion


def _checkpoint_hashes(run_manifest: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for cell in run_manifest.get("cells", []):
        checkpoint = cell.get("checkpoint")
        checkpoint_hash = cell.get("checkpoint_sha256")
        if checkpoint and checkpoint_hash:
            previous = hashes.setdefault(str(checkpoint), str(checkpoint_hash))
            if previous != str(checkpoint_hash):
                raise ValueError(f"Conflicting checkpoint hashes: {checkpoint}")

        checkpoints = cell.get("checkpoints", [])
        checkpoint_hashes = cell.get("checkpoint_sha256s", [])
        if checkpoints or checkpoint_hashes:
            if len(checkpoints) != len(checkpoint_hashes):
                raise ValueError(f"Checkpoint/hash count mismatch: {cell.get('id')}")
            for path, digest in zip(checkpoints, checkpoint_hashes, strict=True):
                previous = hashes.setdefault(str(path), str(digest))
                if previous != str(digest):
                    raise ValueError(f"Conflicting checkpoint hashes: {path}")
    return hashes


def _selected_model_files(run_manifest: dict[str, Any]) -> tuple[Path, ...]:
    checkpoint_hashes = _checkpoint_hashes(run_manifest)
    selected: list[Path] = []
    for relative, expected_hash in sorted(checkpoint_hashes.items()):
        checkpoint = ROOT / relative
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
        if _sha256(checkpoint) != expected_hash:
            raise ValueError(f"Selected checkpoint hash mismatch: {checkpoint}")
        selected.append(checkpoint)
        for name in ("config_resolved.yaml", "history.json"):
            companion = checkpoint.parent / name
            if not companion.is_file():
                raise FileNotFoundError(f"Missing checkpoint companion: {companion}")
            selected.append(companion)
    return tuple(selected)


def _strict_holdout_eval_sources() -> tuple[Path, ...]:
    root = ROOT / "data/processed_strict_holdout/hydrostatic"
    sources: list[Path] = [root / "strict_holdout_index.json"]
    families = (
        "bathymetry_holdout_continental",
        "bathymetry_holdout_trench",
        "source_holdout_okada_like",
        "source_holdout_rough",
    )
    for family in families:
        family_root = root / family
        sources.extend(
            (
                family_root / "holdout_manifest.json",
                family_root / "normalization_stats.json",
                family_root / "test_id",
                family_root / "test_heldout",
            )
        )
    return tuple(sources)


def _native_muscl_eval_sources() -> tuple[Path, ...]:
    sources: list[Path] = []
    for resolution in (32, 64, 128):
        root = ROOT / f"data/processed_res{resolution}/muscl_hr"
        sources.extend((root / "normalization_stats.json", root / "test"))
    return tuple(sources)


def _paper_sources() -> tuple[Path, ...]:
    return (
        ROOT / "paper/main.tex",
        ROOT / "paper/sections",
        ROOT / "paper/figures",
        ROOT / "paper/cas-common.sty",
        ROOT / "paper/cas-model2-names.bst",
        ROOT / "paper/cas-sc.cls",
        ROOT / "paper/build/main.pdf",
    )


def _assert_sources(specs: Iterable[ArchiveSpec]) -> None:
    for spec in specs:
        for source in spec.sources:
            resolved = source.resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f"Archive source escapes repository: {source}") from exc
            if not source.exists():
                raise FileNotFoundError(f"Missing archive source: {source}")


def _reproduction_specs(
    run_root: Path, run_manifest: dict[str, Any]
) -> tuple[ArchiveSpec, ...]:
    return (
        ArchiveSpec(
            "main_processed/hydrostatic_processed.tar.zst",
            (ROOT / "data/processed/hydrostatic",),
            "Full Hydrostatic train/validation/test processed dataset.",
        ),
        ArchiveSpec(
            "main_processed/muscl_hr_processed.tar.zst",
            (ROOT / "data/processed/muscl_hr",),
            "Full MUSCL-HR train/validation/test processed dataset.",
        ),
        ArchiveSpec(
            "main_processed/boussinesq_processed.tar.zst",
            (ROOT / "data/processed/boussinesq",),
            "Full Boussinesq train/validation/test processed dataset.",
        ),
        ArchiveSpec(
            "supplementary/strict_holdout_evaluation.tar.zst",
            _strict_holdout_eval_sources(),
            "Strict-family ID and held-out evaluation subsets plus normalization.",
        ),
        ArchiveSpec(
            "supplementary/native_muscl_evaluation.tar.zst",
            _native_muscl_eval_sources(),
            "Native 32/64/128 MUSCL-HR test sets used by the paper matrix.",
        ),
        ArchiveSpec(
            "supplementary/real_bathymetry_v2.tar.zst",
            (
                ROOT / "data/real_bathymetry_raw",
                ROOT / "data/real_bathymetry_inputs_v2",
                ROOT / "data/real_bathymetry_v2",
                ROOT / "data/processed_real_bathymetry_v2",
            ),
            "GEBCO-derived inputs, common-time labels, and processed transfer suites.",
        ),
        ArchiveSpec(
            "models/selected_checkpoints.tar.zst",
            _selected_model_files(run_manifest),
            "All 33 selected checkpoints with resolved configs and histories.",
        ),
        ArchiveSpec(
            "results/final_paper_evaluation.tar.zst",
            (
                run_root,
                ROOT / "scripts/summarize_direct_model_statistics.py",
                ROOT / "scripts/plot_reference_diagnostics.py",
                ROOT / "scripts/summarize_geoclaw_discrepancy.py",
                ROOT
                / "artifacts/common_time_v2/level_b_minimum/"
                / "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
                / "frozen_contract.json",
                ROOT
                / "artifacts/common_time_v2/level_b_minimum_evaluation/"
                / "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
                / "comparison_rows.json",
                ROOT
                / "artifacts/common_time_v2/level_b_minimum_external/"
                / "3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460"
                / "RUN_MANIFEST.json",
            ),
            (
                "Validated final evaluation, numerical evidence, statistics "
                "helper, and reference-diagnostic plotting scripts."
            ),
        ),
        ArchiveSpec(
            "paper/paper_snapshot.tar.zst",
            _paper_sources(),
            "Elsevier manuscript sources, figures, template files, and compiled PDF.",
        ),
    )


def _raw_specs() -> tuple[ArchiveSpec, ...]:
    return (
        ArchiveSpec(
            "raw/train_common_time_v2.tar.zst",
            (ROOT / "data/train",),
            "10,000 scenarios and 30,000 solver publications.",
        ),
        ArchiveSpec(
            "raw/validation_common_time_v2.tar.zst",
            (ROOT / "data/eval",),
            "1,000 scenarios and 3,000 solver publications.",
        ),
        ArchiveSpec(
            "raw/test_common_time_v2.tar.zst",
            (ROOT / "data/test",),
            "2,500 scenarios and 7,500 solver publications.",
        ),
    )


def _validate_raw_inputs() -> dict[str, Any]:
    expected = {
        "train": (
            ROOT / "data/train/raw/operational_shards/train_000001_010000.json",
            30000,
        ),
        "eval": (
            ROOT / "data/eval/raw/operational_shards/eval_000001_001000.json",
            3000,
        ),
        "test": (
            ROOT / "data/test/raw/operational_shards/test_000001_002500.json",
            7500,
        ),
    }
    records: dict[str, Any] = {}
    contract_hashes: set[str] = set()
    code_state_hashes: set[str] = set()
    for split, (path, expected_publications) in expected.items():
        payload = _load_json(path)
        if payload.get("complete") is not True:
            raise ValueError(f"Raw operational shard is incomplete: {path}")
        if payload.get("split") != split:
            raise ValueError(f"Raw operational shard split mismatch: {path}")
        publications = payload.get("publications", [])
        if len(publications) != expected_publications:
            raise ValueError(
                f"Raw publication count mismatch for {split}: "
                f"{len(publications)} != {expected_publications}"
            )
        identities = [str(row["qualified_id"]) for row in publications]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate raw publication identities: {path}")
        if any(not identity.startswith(f"{split}:") for identity in identities):
            raise ValueError(f"Wrong split-qualified raw identity: {path}")
        if any(len(str(row["publication_hash"])) != 64 for row in publications):
            raise ValueError(f"Malformed raw publication hash: {path}")
        contract_hashes.add(str(payload["contract_hash"]))
        code_state_hashes.add(str(payload["code_state_hash"]))
        records[split] = {
            "operational_shard": _relative(path),
            "publications": len(publications),
            "contract_hash": str(payload["contract_hash"]),
            "code_state_hash": str(payload["code_state_hash"]),
        }
    if len(contract_hashes) != 1 or len(code_state_hashes) != 1:
        raise ValueError("Raw splits do not share one contract/code state")
    return records


def _create_archive(
    spec: ArchiveSpec,
    destination_root: Path,
    *,
    zstd_level: int,
    threads: int,
) -> dict[str, Any]:
    destination = destination_root / spec.relative_output
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    source_snapshot = _source_snapshot(spec.sources)

    relative_sources = [_relative(path) for path in spec.sources]
    tar_command = [
        "tar",
        "--create",
        "--file=-",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--pax-option=delete=atime,delete=ctime",
        "--directory",
        str(ROOT),
        *relative_sources,
    ]
    zstd_command = [
        "zstd",
        f"-{zstd_level}",
        f"-T{threads}",
        "--quiet",
        "--force",
        "-o",
        str(temporary),
    ]

    try:
        with subprocess.Popen(tar_command, stdout=subprocess.PIPE) as tar_process:
            if tar_process.stdout is None:
                raise RuntimeError("Failed to open tar output pipe")
            zstd_result = subprocess.run(
                zstd_command,
                stdin=tar_process.stdout,
                check=False,
            )
            tar_process.stdout.close()
            tar_returncode = tar_process.wait()
        if tar_returncode != 0 or zstd_result.returncode != 0:
            raise RuntimeError(
                f"Archive command failed: tar={tar_returncode} "
                f"zstd={zstd_result.returncode}"
            )
        if _source_snapshot(spec.sources) != source_snapshot:
            raise RuntimeError(
                f"Archive sources changed during build: {spec.relative_output}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    source_files = len(source_snapshot)
    source_bytes = sum(size for size, _ in source_snapshot.values())
    return {
        "archive": destination.relative_to(destination_root).as_posix(),
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": _sha256(destination),
        "purpose": spec.purpose,
        "source_bytes": source_bytes,
        "source_files": source_files,
        "sources": relative_sources,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_release_files(
    destination: Path,
    *,
    profile: str,
    run_manifest: dict[str, Any] | None,
    profile_state: dict[str, Any] | None,
    records: list[dict[str, Any]],
    standalone_files: list[dict[str, Any]],
    prepared: str,
) -> None:
    total_archive_bytes = sum(int(row["archive_bytes"]) for row in records)
    total_source_bytes = sum(int(row["source_bytes"]) for row in records)
    run_id = None if run_manifest is None else str(run_manifest["run_id"])
    code_state = None if run_manifest is None else run_manifest.get("code_state")

    manifest = {
        "schema_id": "tsunami-surrogate.zenodo-release-manifest.v1",
        "profile": profile,
        "release_version": RELEASE_VERSION,
        "prepared_date": prepared,
        "evaluation_run": run_id,
        "evaluation_code_state": code_state,
        "profile_state": profile_state,
        "archives": records,
        "standalone_files": standalone_files,
        "total_archive_bytes": total_archive_bytes,
        "total_source_bytes": total_source_bytes,
    }
    _write_json(destination / "RELEASE_MANIFEST.json", manifest)

    with (destination / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for row in sorted(records, key=lambda item: str(item["archive"])):
            handle.write(f"{row['archive_sha256']}  {row['archive']}\n")
        for row in sorted(
            standalone_files, key=lambda item: str(item["path"])
        ):
            handle.write(f"{row['sha256']}  {row['path']}\n")

    with (destination / "ARCHIVE_CONTENTS.tsv").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(
            "archive\tarchive_bytes\tsource_bytes\tsource_files\tpurpose\n"
        )
        for row in sorted(records, key=lambda item: str(item["archive"])):
            handle.write(
                f"{row['archive']}\t{row['archive_bytes']}\t"
                f"{row['source_bytes']}\t{row['source_files']}\t"
                f"{row['purpose']}\n"
            )

    if profile == "reproduction":
        title = "Tsunami-Surrogate Common-Time V2 Reproduction Package"
        description = (
            "Processed benchmark datasets, selected checkpoints, validated "
            "evaluation artifacts, numerical evidence, and manuscript snapshot "
            "for the common-time-v2 multi-reference tsunami-surrogate benchmark."
        )
        scope = """This package is intended as a new version of the existing
processed-data Zenodo record. It is sufficient to rerun every reported
evaluation and figure. The three main processed datasets include complete
train/validation/test splits. Strict-holdout and native-resolution archives
include the evaluation subsets required for the reported auxiliary analyses;
their selected checkpoints, resolved configurations, and training histories
are included, but their full auxiliary train/validation arrays are not."""
        extraction = """Extract archives into a fresh clone's repository root:

```bash
sha256sum -c SHA256SUMS.txt
tar --use-compress-program=unzstd -xf main_processed/hydrostatic_processed.tar.zst -C /path/to/tsunami-surrogate
tar --use-compress-program=unzstd -xf models/selected_checkpoints.tar.zst -C /path/to/tsunami-surrogate
tar --use-compress-program=unzstd -xf results/final_paper_evaluation.tar.zst -C /path/to/tsunami-surrogate
```

Repeat for the remaining archives needed by the analysis."""
        provenance = f"""- Repository: https://github.com/1zuki/tsunami-surrogate
- Validated evaluation run: {run_id}
- Evaluation code state: `{json.dumps(code_state, sort_keys=True)}`
- Common requested times: `0.0035, 0.0070, ..., 0.1750`
- Numerical computation: 96x96 with a central 64x64 publication crop

The main processed containers carry common-time-v2 payloads in an older
manifest envelope. The final evaluation preflight found no legacy saved-step
payloads, but checkpoint-to-training-data identity is manifest-bound rather
than independently bound to every shard content hash. This limitation is
disclosed in the manuscript and retained evaluation report."""
        gebco = """## GEBCO attribution

The real-bathymetry transfer suite contains derived GEBCO_2026 material.
GEBCO must be acknowledged and cited. The derived crops are rescaled research
inputs and must not be used for navigation."""
    else:
        title = "Tsunami-Surrogate Common-Time V2 Raw Numerical Publications"
        description = (
            "Eta-primary common-time raw train, validation, and test numerical "
            "publications for Hydrostatic, MUSCL-HR, and Boussinesq references."
        )
        scope = """This is the optional raw companion record. It contains all
13,500 shared scenarios and 40,500 solver publications. Each solver publication
stores 50 requested-time surface-elevation frames plus requested-time,
adjacent-step interpolation, health, contract, and checksum provenance. Full
natural-step states are intentionally not published."""
        extraction = """Extract archives into a fresh clone's repository root:

```bash
sha256sum -c SHA256SUMS.txt
tar --use-compress-program=unzstd -xf raw/train_common_time_v2.tar.zst -C /path/to/tsunami-surrogate
tar --use-compress-program=unzstd -xf raw/validation_common_time_v2.tar.zst -C /path/to/tsunami-surrogate
tar --use-compress-program=unzstd -xf raw/test_common_time_v2.tar.zst -C /path/to/tsunami-surrogate
```"""
        if profile_state is None:
            raise ValueError("Raw release metadata requires generation state")
        states = list(profile_state.values())
        contract_hash = states[0]["contract_hash"]
        code_state_hash = states[0]["code_state_hash"]
        split_rows = "\n".join(
            f"- {split}: {row['publications']:,} solver publications"
            for split, row in profile_state.items()
        )
        provenance = f"""- Repository: https://github.com/1zuki/tsunami-surrogate
- Frozen generation contract: `{contract_hash}`
- Frozen generation code-state hash: `{code_state_hash}`
- Common requested times: `0.0035, 0.0070, ..., 0.1750`
- Numerical computation: 96x96 with a central 64x64 publication crop

The release builder confirmed that all three operational shard manifests are
complete, contain unique split-qualified publication identities, and share the
same frozen contract and code state:

{split_rows}

The top-level archive hashes verify transfer integrity. The raw publications
also retain their original per-publication hashes and provenance."""
        gebco = ""

    readme = f"""# {title}

Version: {RELEASE_VERSION}

Prepared: {prepared}

{description}

This is a controlled synthetic, finite-horizon research benchmark. It is not
an operational tsunami-warning, navigation, inundation, run-up, or
site-specific hazard product.

## Scope

{scope}

## Integrity

```bash
sha256sum -c SHA256SUMS.txt
```

`RELEASE_MANIFEST.json` records archive hashes, byte sizes, source file counts,
source paths, the validated evaluation run, and the evaluation code state.

## Extraction

{extraction}

## Code and provenance

{provenance}

{gebco}

## License before publication

The repository code license does not automatically choose the data license.
Select and record the intended dataset license in Zenodo before publishing this
version. Do not upload this draft without making that explicit choice.

## Citation

Use the DOI assigned to the published Zenodo version. Do not cite the local
draft directory as an immutable public release.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    metadata = {
        "title": title,
        "upload_type": "dataset",
        "description": description,
        "creators": [
            {
                "name": "Nguyen, Tho Binh An",
                "affiliation": (
                    "VNUHCM - University of Information Technology"
                ),
            },
            {
                "name": "Le, Minh Nhut Tan",
                "affiliation": (
                    "VNUHCM - University of Information Technology"
                ),
            },
            {
                "name": "Mai, Tien Dung",
                "affiliation": (
                    "VNUHCM - University of Information Technology"
                ),
            },
        ],
        "keywords": [
            "tsunami surrogate modeling",
            "scientific machine learning",
            "neural operators",
            "shallow-water equations",
            "benchmark dataset",
            "common-time evaluation",
        ],
        "version": RELEASE_VERSION,
        "publication_date": prepared,
        "access_right": "open",
        "related_identifiers": [
            {
                "identifier": "https://github.com/1zuki/tsunami-surrogate",
                "relation": "isSupplementedBy",
                "scheme": "url",
            }
        ],
        "notes": (
            "Select the dataset license and link the raw/reproduction companion "
            "record after reserving both DOIs."
        ),
    }
    _write_json(destination / "ZENODO_METADATA_TEMPLATE.json", metadata)

    checklist = """# Manual Zenodo upload checklist

1. Verify `sha256sum -c SHA256SUMS.txt`.
2. Open the existing processed-data record and choose **New version** for the
   reproduction package. Use a separate dataset record for the raw package.
3. Reserve both DOIs before publication and link the two records as companion
   datasets.
4. Copy and review `ZENODO_METADATA_TEMPLATE.json`.
5. Choose the dataset license explicitly; do not assume the code license.
6. If uploading after the prepared date, update `publication_date`.
7. Upload README, manifests, checksums, and every archive.
8. Confirm the displayed total size and every filename before publishing.
9. Publish the raw record first, then add its DOI to the reproduction record.
10. Publish the reproduction version and replace the manuscript's pending
    data-availability text with the final persistent identifier.
11. Download one archive from Zenodo and re-run its SHA-256 check as an
    independent post-upload smoke test.
"""
    (destination / "UPLOAD_CHECKLIST.md").write_text(checklist, encoding="utf-8")


def _build_profile(
    profile: str,
    destination: Path,
    *,
    run_root: Path,
    zstd_level: int,
    threads: int,
    dry_run: bool,
    prepared: str,
    project_python: Path,
) -> None:
    run_manifest: dict[str, Any] | None = None
    profile_state: dict[str, Any] | None = None
    if profile == "reproduction":
        run_manifest, _ = _validate_run(run_root)
        specs = _reproduction_specs(run_root, run_manifest)
    elif profile == "raw":
        profile_state = _validate_raw_inputs()
        specs = _raw_specs()
    else:
        raise ValueError(f"Unknown profile: {profile}")

    _assert_sources(specs)
    destination.mkdir(parents=True, exist_ok=True)
    existing_files = [
        path for path in destination.rglob("*") if path.is_file()
    ]
    if existing_files:
        raise FileExistsError(
            f"Release destination contains files: {destination}. "
            "Use a new destination for a new immutable build."
        )

    estimates = []
    for spec in specs:
        files, size = _source_summary(spec.sources)
        estimates.append((spec, files, size))
        print(
            f"[zenodo] {profile} {spec.relative_output} "
            f"files={files} source_bytes={size}"
        )
    if dry_run:
        print(f"[zenodo] dry-run only; no files written under {destination}")
        destination.rmdir()
        return

    standalone_files: list[dict[str, Any]] = []
    if profile == "reproduction":
        statistics_path = destination / "direct_model_statistics.json"
        subprocess.run(
            [
                str(project_python),
                str(ROOT / "scripts/summarize_direct_model_statistics.py"),
                "--evaluation-run",
                _relative(run_root),
                "--output",
                str(statistics_path),
            ],
            cwd=ROOT,
            check=True,
        )
        standalone_files.append(
            {
                "path": statistics_path.relative_to(destination).as_posix(),
                "bytes": statistics_path.stat().st_size,
                "sha256": _sha256(statistics_path),
                "purpose": (
                    "Paired per-scenario direct-model bootstrap statistics "
                    "reported in the manuscript."
                ),
            }
        )

    records = []
    for spec, _, _ in estimates:
        print(f"[zenodo] building {profile}/{spec.relative_output}", flush=True)
        record = _create_archive(
            spec,
            destination,
            zstd_level=zstd_level,
            threads=threads,
        )
        records.append(record)
        print(
            f"[zenodo] built {record['archive']} "
            f"bytes={record['archive_bytes']} "
            f"sha256={record['archive_sha256']}",
            flush=True,
        )

    _write_release_files(
        destination,
        profile=profile,
        run_manifest=run_manifest,
        profile_state=profile_state,
        records=records,
        standalone_files=standalone_files,
        prepared=prepared,
    )
    print(
        f"[zenodo] complete profile={profile} destination={destination}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("reproduction", "raw", "all"),
        default="reproduction",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
    )
    parser.add_argument("--zstd-level", type=int, default=1)
    parser.add_argument(
        "--project-python",
        type=Path,
        default=DEFAULT_PROJECT_PYTHON,
        help="Python environment used for NumPy-based release summaries.",
    )
    parser.add_argument(
        "--prepared-date",
        default=DEFAULT_PREPARED_DATE,
        help="ISO date written into release metadata.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="zstd thread count; 0 uses all detected CPU cores.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.zstd_level <= 19:
        raise ValueError("--zstd-level must be between 1 and 19")
    if args.threads < 0:
        raise ValueError("--threads must be nonnegative")
    date.fromisoformat(args.prepared_date)
    if not args.project_python.is_file():
        raise FileNotFoundError(
            f"Project Python is missing: {args.project_python}"
        )

    release_root = args.release_root.resolve()
    profiles = (
        ("reproduction", "reproduction"),
        ("raw", "raw"),
    )
    if args.profile != "all":
        profiles = tuple(row for row in profiles if row[0] == args.profile)

    for profile, folder in profiles:
        _build_profile(
            profile,
            release_root / folder,
            run_root=args.run_root.resolve(),
            zstd_level=args.zstd_level,
            threads=args.threads,
            dry_run=bool(args.dry_run),
            prepared=str(args.prepared_date),
            project_python=args.project_python.absolute(),
        )


if __name__ == "__main__":
    main()
