from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import torch

from scripts import build_core_checkpoint_release as release


def _fixture_payload(root: Path):
    model = torch.nn.Linear(2, 1)
    original = root / "experiments/toy/seed_18/best.pt"
    original.parent.mkdir(parents=True)
    config_path = original.with_name("config_resolved.yaml")
    config_path.write_text("seed: 18\n")
    torch.save(
        {
            "config": {"seed": 18},
            "epoch": 7,
            "model_state": model.state_dict(),
            "model_signature": {},
            "metrics": {},
            "data_provenance": {},
            "training_contract": {},
            "optimizer_state": {"private_tensor": torch.ones(512)},
            "scheduler_state": {"state": 1},
            "trainer_state": {"state": 2},
        },
        original,
    )
    name = original.relative_to(root)
    checkpoint = release.Checkpoint(name, 18, 7, release.sha256_file(original))
    files = {config_path.relative_to(root): None}
    return model, checkpoint, files


def test_release_archives_inference_only_and_verifies_payload(tmp_path, monkeypatch):
    model, checkpoint, files = _fixture_payload(tmp_path)
    metadata = {"schema_id": release.SCHEMA}
    monkeypatch.setattr(release, "dataset_inventory", lambda *_: ({}, []))
    first, second = (tmp_path / "first.tar", tmp_path / "second.tar")
    for output in (first, second):
        release.build(
            tmp_path, output, metadata, {checkpoint.path: checkpoint}, files, {}
        )
    assert release.sha256_file(first) == release.sha256_file(second)
    with pytest.raises(FileExistsError):
        release.build(
            tmp_path, first, metadata, {checkpoint.path: checkpoint}, files, {}
        )

    extracted = tmp_path / "restored"
    with tarfile.open(first) as archive:
        assert all(
            member.isfile() and member.mtime == 0 for member in archive.getmembers()
        )
        archive.extractall(extracted, filter="data")

    manifest = release.verify(extracted)
    packaged = manifest["checkpoints"][0]
    assert packaged["source_checkpoint_sha256"] == checkpoint.expected_sha256
    assert packaged["training_seed"] == 18
    weights_path = extracted / checkpoint.path
    payload = torch.load(weights_path, weights_only=True)
    assert payload["epoch"] == 7
    assert not {"optimizer_state", "scheduler_state", "trainer_state"} & payload.keys()
    replay_model = torch.nn.Linear(2, 1)
    replay_model.load_state_dict(payload["model_state"])
    inputs = torch.ones((1, 2))
    torch.testing.assert_close(replay_model(inputs), model(inputs))

    weights_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.verify(extracted)


def test_release_refuses_changed_source(tmp_path, monkeypatch):
    _, checkpoint, files = _fixture_payload(tmp_path)
    monkeypatch.setattr(release, "dataset_inventory", lambda *_: ({}, []))
    output = tmp_path / "release.tar"
    original = tmp_path / checkpoint.path
    with original.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.build(tmp_path, output, {}, {checkpoint.path: checkpoint}, files, {})
    assert not output.exists()


@pytest.mark.parametrize(
    "unsafe_path", ["../escape", "/absolute", "nested/../../../escape"]
)
def test_release_rejects_paths_outside_bundle(tmp_path, unsafe_path):
    with pytest.raises(ValueError, match="Unsafe release path"):
        release.safe_path(tmp_path, unsafe_path)


def test_release_rejects_symlinks(tmp_path):
    target = tmp_path / "dataset"
    target.mkdir()
    (tmp_path / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="Unsafe release path"):
        release.safe_path(tmp_path, "alias/shards_manifest.json")


def test_replay_checks_dataset_before_launching_evaluator(tmp_path, monkeypatch):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({'device': sys.argv[3]}))\n"
    )
    dataset = tmp_path / "test-shard.npz"
    dataset.write_bytes(b"expected shard payload")
    monkeypatch.setattr(
        release,
        "verify",
        lambda _: {
            "external_processed_data": {dataset.name: release.sha256_file(dataset)},
            "recipes": {
                "direct": [
                    {"script": evaluator.name, "arguments": ["{output}/test.json"]}
                ]
            },
        },
    )
    output = tmp_path / "replay"
    release.replay(tmp_path, output, "cpu", "direct")
    assert json.loads((output / "test.json").read_text()) == {"device": "cpu"}

    frozen_digest = release.sha256_file(dataset)
    dataset.write_bytes(b"different shard")
    monkeypatch.setattr(
        release,
        "verify",
        lambda _: {"external_processed_data": {dataset.name: frozen_digest}},
    )
    wrong_data_output = tmp_path / "wrong-data"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.replay(tmp_path, wrong_data_output, "cpu", "direct")
    assert not wrong_data_output.exists()


def _trajectory_fixture(source_dir: Path):
    def save(name, payload):
        path = source_dir / name
        path.write_text(json.dumps(payload))
        return release.sha256_file(path)

    inference_source = source_dir / release.TRAJECTORY_SOURCE
    inference_source.write_text("inference")
    summary_source = source_dir / release.TRAJECTORY_SUMMARY_SOURCE
    summary_source.write_text("summary")
    checkpoints = {}
    for seed in release.ENSEMBLE_SEEDS:
        name = Path(f"experiments/ensemble/member_{seed}/best.pt")
        checkpoints[name] = release.Checkpoint(name, seed, 1, f"{seed:064x}")

    raw = {
        "status": "completed_full_splits",
        "calibration_fit_split": "validation",
        "thresholds": {"0.9": 5.0},
        "records": {},
        "provenance": {
            "script_sha256": release.sha256_file(inference_source),
            "checkpoints": {
                f"/historical/checkout/{path}": row.expected_sha256
                for path, row in checkpoints.items()
            },
        },
    }
    lineage = {
        "status": "passed",
        "splits": {},
        "split_pair_intersection_counts": {
            name: {"master_input_hash_pair": 0}
            for name in ("train__validation", "train__test", "validation__test")
        },
    }
    for name, directory, count in (("validation", "val", 1000), ("test", "test", 2500)):
        path = Path("/historical/checkout/data/processed/hydrostatic") / directory
        raw["records"][name] = [
            {"scenario_id": str(index), "score": 4.0, "rel_l2": 1.0}
            for index in range(count)
        ]
        raw["provenance"][name] = {
            str(path / filename): f"{index:064x}"
            for index, filename in enumerate(
                ("eval_manifest.json", "shards_manifest.json", "meta.jsonl"), 1
            )
        }
        lineage["splits"][name] = {
            "metadata_sha256": raw["provenance"][name][str(path / "meta.jsonl")],
            "rows": count,
        }
    raw_sha = save(release.TRAJECTORY_RAW, raw)
    save(release.TRAJECTORY_LINEAGE, lineage)
    summary = {
        **raw,
        "records": {
            name: [
                {**row, "rel_l2_normalized": row["rel_l2"], "rel_l2": 0.9}
                for row in rows
            ]
            for name, rows in raw["records"].items()
        },
        "groups": {"source_strength>=0.285": {"scenario_count": 177}},
        "postprocessing": {
            "inference_result_sha256": raw_sha,
            "script_sha256": release.sha256_file(summary_source),
            "score_and_thresholds_changed": False,
        },
    }
    save(release.TRAJECTORY_RESULT, summary)
    return checkpoints, summary


def test_release_binds_trajectory_summary_to_external_evidence(tmp_path, monkeypatch):
    source_dir = tmp_path / "historical"
    source_dir.mkdir()
    checkpoints, summary = _trajectory_fixture(source_dir)
    evidence, data = release.trajectory_evidence(source_dir, checkpoints)
    assert len(evidence) == 5
    assert data[Path("data/processed/hydrostatic/val/meta.jsonl")] == f"{3:064x}"
    report = release.TRAJECTORY_DIR / release.TRAJECTORY_RESULT

    monkeypatch.setattr(release, "dataset_inventory", lambda *_: ({}, []))
    archive_path = tmp_path / "release.tar"
    with pytest.raises(ValueError, match="Replay data differ"):
        release.build(
            tmp_path,
            archive_path,
            {},
            {},
            {},
            {},
            evidence=evidence,
            expected_data=data,
        )
    assert not archive_path.exists()

    monkeypatch.setattr(
        release,
        "dataset_inventory",
        lambda *_: ({str(name): sha256 for name, sha256 in data.items()}, []),
    )
    release.build(
        tmp_path,
        archive_path,
        {"schema_id": release.SCHEMA},
        {},
        {},
        {},
        evidence=evidence,
        expected_data=data,
    )
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
        assert str(report) in names
        assert all(not name.startswith("/") for name in names)
        archived = archive.extractfile(str(report))
        assert json.load(archived) == summary


def test_release_rejects_changed_trajectory_scores(tmp_path):
    checkpoints, summary = _trajectory_fixture(tmp_path)
    summary["records"]["test"][0]["score"] += 1.0
    (tmp_path / release.TRAJECTORY_RESULT).write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="changed the test scenario scores"):
        release.trajectory_evidence(tmp_path, checkpoints)


def _reference_resolution_proof(root):
    sources = root / "scripts"
    sources.mkdir()
    preflight_script = sources / "eval_suite_preflight.py"
    preflight_script.write_text("completion check")
    rows = []
    checkpoints = {}
    for model in release.SEED_MODELS:
        for seed in release.SEEDS:
            name = Path(f"experiments/{model}/seed_{seed}/best.pt")
            path = root / name
            path.parent.mkdir(parents=True)
            torch.save({"config": {"seed": seed}, "epoch": 5}, path)
            checkpoints[name] = release.Checkpoint(name, seed, 5, None)
            rows.append(
                {
                    "checkpoint": name.as_posix(),
                    "seed": seed,
                    "best_epoch": 5,
                    "completion": "horizon",
                    "configured_epochs": 10,
                    "last_epoch": 10,
                    "early_count": 0,
                    "patience": 0,
                    "training_data_identity_strength": "content_bound",
                    "best_checkpoint_sha256": release.digest(path),
                }
            )
    proof = root / "proof.json"
    payload = {
        "status": "passed",
        "source_files": {str(preflight_script): release.digest(preflight_script)},
        "checkpoints": rows,
    }
    proof.write_text(json.dumps(payload))
    return proof, payload, checkpoints


def test_reference_resolution_proof_binds_original_checkpoint_sha(tmp_path):
    proof, payload, checkpoints = _reference_resolution_proof(tmp_path)
    sha, bound, sources = release.reference_resolution_preflight(
        tmp_path, proof, checkpoints
    )
    assert sha == release.digest(proof)
    assert len(bound) == 12
    assert [
        bound[Path(row["checkpoint"])].proof_sha256 for row in payload["checkpoints"]
    ] == [row["best_checkpoint_sha256"] for row in payload["checkpoints"]]
    assert all(entry.expected_sha256 is None for entry in bound.values())
    assert list(sources) == [Path("scripts/eval_suite_preflight.py")]
    row = payload["checkpoints"][0]
    row["best_checkpoint_sha256"] = "0" * 64
    proof.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.reference_resolution_preflight(tmp_path, proof, checkpoints)


def test_reference_resolution_proof_requires_exact_roster_and_source_hash(tmp_path):
    proof, payload, checkpoints = _reference_resolution_proof(tmp_path)
    repeated = payload["checkpoints"].pop()
    payload["checkpoints"].append(payload["checkpoints"][0])
    proof.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="twelve prescribed checkpoints"):
        release.reference_resolution_preflight(tmp_path, proof, checkpoints)
    payload["checkpoints"][-1] = repeated
    proof.write_text(json.dumps(payload))
    next(iter((tmp_path / "scripts").glob("*.py"))).write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.reference_resolution_preflight(tmp_path, proof, checkpoints)
