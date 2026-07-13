from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.audit_common_time_v2_h0 import audit_h0
from scripts.archive_common_time_stage_c import STAGE_C_SOURCES, archive_stage_c


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
