from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.audit_common_time_v2_h0 import audit_h0, audit_h0_regression
from scripts.archive_common_time_stage_c import STAGE_C_SOURCES, archive_stage_c
from src.data_gen.common_time_v2 import (
    ETA_SAMPLE_SCHEMA_ID,
    PUBLICATION_SCHEMA_ID,
    candidate_requested_times,
    sha256_file,
)


def _write_split(root: Path, *, split: str, endpoint: float = 0.18) -> None:
    split_root = root / split
    (split_root / "synthetic").mkdir(parents=True)
    (split_root / "bathymetry").mkdir(parents=True)
    (split_root / "sources").mkdir(parents=True)
    bathymetry = np.asarray([[-1.0, -1.2], [-0.8, -1.1]], dtype=np.float32)
    source = np.asarray([[0.1, -0.1], [0.05, -0.05]], dtype=np.float32)
    strength = np.float32(0.5)
    rest = np.maximum(-bathymetry, 0.0).astype(np.float32)
    eta0 = (strength * source).astype(np.float32)
    h0 = np.maximum(rest + eta0, 0.0).astype(np.float32)
    surface = (h0 + bathymetry).astype(np.float32)
    np.savez_compressed(
        split_root / "bathymetry/sample_000001.npz",
        bathymetry=bathymetry,
        bathymetry_type=np.asarray(["slope"], dtype="U64"),
        sample_seed=np.asarray([1], dtype=np.int64),
    )
    np.savez_compressed(
        split_root / "sources/sample_000001.npz",
        source_field=source,
        source_type=np.asarray(["gaussian"], dtype="U64"),
        source_strength=np.asarray([strength], dtype=np.float32),
        sample_seed=np.asarray([1], dtype=np.int64),
    )
    row = {
        "sample_index": 1,
        "scenario_id": "scenario_000001",
        "bathymetry_type": "slope",
        "source_type": "gaussian",
        "source_strength": float(strength),
    }
    (split_root / "synthetic/scenario_manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    for solver, channels in (("hydrostatic", 3), ("muscl_hr", 3), ("boussinesq", 2)):
        sample_dir = split_root / "raw" / solver / "samples/sample_000001"
        sample_dir.mkdir(parents=True)
        trajectory = np.zeros((2, channels, 2, 2), dtype=np.float32)
        np.savez_compressed(
            sample_dir / "sample.npz",
            bathymetry=bathymetry,
            source_field=source,
            rest_depth=rest,
            eta0=eta0,
            initial_depth=h0,
            free_surface0=surface,
            trajectory=trajectory,
            trajectory_eta=trajectory[:, 0],
            timestamps=np.asarray([0.0, endpoint], dtype=np.float32),
            dt_history=np.asarray([0.0, endpoint], dtype=np.float32),
            solver_name=np.asarray(
                [
                    {
                        "hydrostatic": "swe_hydrostatic",
                        "muscl_hr": "swe_muscl_hr",
                        "boussinesq": "boussinesq",
                    }[solver]
                ],
                dtype="U64",
            ),
            scenario_id=np.asarray(["scenario_000001"], dtype="U64"),
        )
    (split_root / "raw/dataset_config.snapshot.yaml").parent.mkdir(
        parents=True, exist_ok=True
    )
    advertised = 10 if split == "eval" else 1
    (split_root / "raw/dataset_config.snapshot.yaml").write_text(
        f"dataset:\n  num_samples: {advertised}\n", encoding="utf-8"
    )


def _stage_c_repo(root: Path) -> Path:
    repo = root / "repo"
    for index, rel in enumerate(STAGE_C_SOURCES):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("dense_validation/decision.json"):
            path.write_text(json.dumps({"status": "fail"}), encoding="utf-8")
        elif rel.endswith(".json"):
            path.write_text(json.dumps({"index": index}), encoding="utf-8")
        else:
            path.write_text(f"value: {index}\n", encoding="utf-8")
    return repo


def _replace_with_v2_publications(
    data: Path,
    *,
    accepted_h0_root: Path,
    contract_hash: str,
    corrupt_authoritative_binding: bool = False,
) -> None:
    accepted_inventory_sha256 = sha256_file(
        accepted_h0_root / "h0_input_inventory.jsonl"
    )
    accepted_rows = {
        row["qualified_id"]: row
        for row in (
            json.loads(line)
            for line in (
                accepted_h0_root / "h0_input_inventory.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    times = candidate_requested_times()
    solver_names = {
        "hydrostatic": "swe_hydrostatic",
        "muscl_hr": "swe_muscl_hr",
        "boussinesq": "boussinesq",
    }
    for split in ("train", "eval", "test"):
        split_root = data / split
        manifest = json.loads(
            (split_root / "synthetic/scenario_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        accepted = accepted_rows[f"{split}:{manifest['scenario_id']}"]
        with np.load(
            split_root / "bathymetry/sample_000001.npz",
            allow_pickle=False,
        ) as payload:
            bathymetry = np.asarray(payload["bathymetry"])
        with np.load(
            split_root / "sources/sample_000001.npz",
            allow_pickle=False,
        ) as payload:
            source = np.asarray(payload["source_field"])
            strength = float(np.asarray(payload["source_strength"]).reshape(-1)[0])
        effective_source = source.copy()
        effective_source[[0, -1], :] = 0.0
        effective_source[:, [0, -1]] = 0.0
        rest_depth = np.maximum(-bathymetry, 0.0).astype(np.float32)
        eta0 = np.asarray(strength * effective_source, dtype=np.float32)
        initial_depth = np.maximum(rest_depth + eta0, 0.0).astype(np.float32)
        free_surface0 = np.asarray(
            initial_depth + bathymetry,
            dtype=np.float32,
        )
        for solver, solver_name in solver_names.items():
            sample_dir = (
                split_root / "raw" / solver / "samples/sample_000001"
            )
            np.savez_compressed(
                sample_dir / "sample.npz",
                bathymetry=bathymetry,
                source_field=effective_source,
                source_strength=np.asarray([strength], dtype=np.float64),
                rest_depth=rest_depth,
                eta0=eta0,
                initial_depth=initial_depth,
                free_surface0=free_surface0,
                trajectory_eta=np.zeros(
                    (times.size, *bathymetry.shape),
                    dtype=np.float32,
                ),
                timestamps=times,
                solver_name=np.asarray([solver_name], dtype="U64"),
                scenario_id=np.asarray([manifest["scenario_id"]], dtype="U64"),
                split=np.asarray([split], dtype="U16"),
                schema_id=np.asarray([ETA_SAMPLE_SCHEMA_ID], dtype="U96"),
                contract_hash=np.asarray([contract_hash], dtype="U64"),
            )
            np.savez_compressed(
                sample_dir / "provenance.npz",
                natural_steps=np.asarray([1], dtype=np.int64),
            )
            (sample_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "schema_id": ETA_SAMPLE_SCHEMA_ID,
                        "split": split,
                        "scenario_id": manifest["scenario_id"],
                        "sample_index": 1,
                        "solver_name": solver_name,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            payload_files = []
            for name in ("meta.json", "provenance.npz", "sample.npz"):
                path = sample_dir / name
                payload_files.append(
                    {
                        "name": name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            authoritative_fingerprint = str(accepted["input_fingerprint"])
            if corrupt_authoritative_binding and split == "train" and solver == "hydrostatic":
                authoritative_fingerprint = "wrong-authoritative-fingerprint"
            publication = {
                "schema_id": PUBLICATION_SCHEMA_ID,
                "artifact_kind": "requested-output-publication",
                "split": split,
                "qualified_id": f"{split}:{manifest['scenario_id']}",
                "scenario_id": manifest["scenario_id"],
                "sample_index": 1,
                "solver_name": solver_name,
                "input_fingerprint": f"effective-{split}",
                "authoritative_input_fingerprint": authoritative_fingerprint,
                "authoritative_inventory_sha256": accepted_inventory_sha256,
                "h0_contract_hash": accepted_h0_root.name,
                "contract_hash": contract_hash,
                "resolved_config_hash": f"config-{solver}",
                "code_state_hash": "generation-code-state",
                "quality_status": "ok",
                "files": payload_files,
            }
            (sample_dir / "publication.json").write_text(
                json.dumps(publication, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def test_h0_passes_split_qualified_repeated_ids_and_discloses_snapshot(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    for split in ("train", "eval", "test"):
        _write_split(data, split=split)
    repo = _stage_c_repo(tmp_path)
    archive = archive_stage_c(repo_root=repo, output_root=tmp_path / "stage-c")
    final = audit_h0(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        output_root=tmp_path / "h0",
        repo_root=repo,
        stage_c_archive=archive,
    )
    decision = json.loads((final / "h0_decision.json").read_text())
    reconciliation = json.loads((final / "h0_snapshot_reconciliation.json").read_text())
    assert decision["audit_completed"]
    assert decision["audit_passed"]
    assert not decision["three_reference_contract_accepted"]
    eval_record = next(item for item in reconciliation if item["split"] == "eval")
    assert eval_record["mismatches"]


def test_h0_preserves_failed_horizon_decision(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for split in ("train", "eval", "test"):
        _write_split(data, split=split, endpoint=0.17 if split == "train" else 0.18)
    repo = _stage_c_repo(tmp_path)
    archive = archive_stage_c(repo_root=repo, output_root=tmp_path / "stage-c")
    final = audit_h0(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        output_root=tmp_path / "h0",
        repo_root=repo,
        stage_c_archive=archive,
    )
    decision = json.loads((final / "h0_decision.json").read_text())
    issues = (final / "h0_issues.jsonl").read_text()
    assert decision["audit_completed"]
    assert not decision["audit_passed"]
    assert "swe_endpoint_below_candidate_horizon" in issues


def test_h0_rejects_dtype_mismatch(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for split in ("train", "eval", "test"):
        _write_split(data, split=split)
    path = data / "train/raw/muscl_hr/samples/sample_000001/sample.npz"
    with np.load(path, allow_pickle=False) as payload:
        changed = {key: np.asarray(payload[key]) for key in payload.files}
    changed["bathymetry"] = changed["bathymetry"].astype(np.float64)
    np.savez_compressed(path, **changed)
    repo = _stage_c_repo(tmp_path)
    archive = archive_stage_c(repo_root=repo, output_root=tmp_path / "stage-c")
    final = audit_h0(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        output_root=tmp_path / "h0",
        repo_root=repo,
        stage_c_archive=archive,
    )
    assert not json.loads((final / "h0_decision.json").read_text())["audit_passed"]
    assert (
        "cross_reference_static_input_mismatch"
        in (final / "h0_issues.jsonl").read_text()
    )


def test_h0_regression_accepts_v2_tapered_publications(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    for split in ("train", "eval", "test"):
        _write_split(data, split=split)
    repo = _stage_c_repo(tmp_path)
    archive = archive_stage_c(repo_root=repo, output_root=tmp_path / "stage-c")
    accepted_h0 = audit_h0(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        output_root=tmp_path / "accepted-h0",
        repo_root=repo,
        stage_c_archive=archive,
    )
    contract_hash = "publication-contract"
    _replace_with_v2_publications(
        data,
        accepted_h0_root=accepted_h0,
        contract_hash=contract_hash,
    )

    regression = audit_h0_regression(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        accepted_h0_root=accepted_h0,
        expected_publication_contract_hash=contract_hash,
        output_root=tmp_path / "regression-h0",
        repo_root=repo,
    )

    decision = json.loads((regression / "h0_decision.json").read_text())
    summary = json.loads((regression / "h0_summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (
            regression / "h0_input_inventory.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert decision["audit_passed"]
    assert summary["inventory_count"] == 3
    assert sum(
        row["validated_publications"]
        for row in summary["publication_summaries"].values()
    ) == 9
    assert all(Path(row["source_cache_path"]).is_file() for row in rows)


def test_h0_regression_rejects_wrong_authoritative_publication_binding(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    for split in ("train", "eval", "test"):
        _write_split(data, split=split)
    repo = _stage_c_repo(tmp_path)
    archive = archive_stage_c(repo_root=repo, output_root=tmp_path / "stage-c")
    accepted_h0 = audit_h0(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        output_root=tmp_path / "accepted-h0",
        repo_root=repo,
        stage_c_archive=archive,
    )
    contract_hash = "publication-contract"
    _replace_with_v2_publications(
        data,
        accepted_h0_root=accepted_h0,
        contract_hash=contract_hash,
        corrupt_authoritative_binding=True,
    )

    regression = audit_h0_regression(
        split_roots={split: data / split for split in ("train", "eval", "test")},
        expected_counts={split: 1 for split in ("train", "eval", "test")},
        accepted_h0_root=accepted_h0,
        expected_publication_contract_hash=contract_hash,
        output_root=tmp_path / "regression-h0",
        repo_root=repo,
    )

    decision = json.loads((regression / "h0_decision.json").read_text())
    issues = (regression / "h0_issues.jsonl").read_text(encoding="utf-8")
    assert not decision["audit_passed"]
    assert "authoritative_input_fingerprint mismatch" in issues
