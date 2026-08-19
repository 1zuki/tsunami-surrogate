from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from scripts import eval_suite_preflight
from src.utils.hashing import sha256_file


def test_deep_payload_audit_runs_jobs_in_parallel(
    monkeypatch,
) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    audited: list[int] = []
    progress: list[str] = []

    def fake_validate(**kwargs: Any) -> None:
        nonlocal active, max_active
        assert kwargs["deep_payload_audit"] is True
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            audited.append(int(kwargs["job_id"]))
            active -= 1

    monkeypatch.setattr(
        eval_suite_preflight,
        "_validate_raw_timestamp_payload",
        fake_validate,
    )
    jobs = [{"job_id": index} for index in range(8)]
    eval_suite_preflight._run_payload_audits(
        jobs,
        deep_payload_audit=True,
        workers=4,
        label="fixture",
        progress_callback=progress.append,
    )

    assert sorted(audited) == list(range(8))
    assert max_active > 1
    assert progress[0] == "[payload-audit] start fixture samples=8 workers=4"
    assert progress[-1] == "[payload-audit] complete fixture samples=8"


def test_parallel_payload_audit_propagates_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_validate(**kwargs: Any) -> None:
        if kwargs["job_id"] == 2:
            raise eval_suite_preflight.PreflightError("corrupt payload")

    monkeypatch.setattr(
        eval_suite_preflight,
        "_validate_raw_timestamp_payload",
        fake_validate,
    )
    with pytest.raises(eval_suite_preflight.PreflightError, match="corrupt payload"):
        eval_suite_preflight._run_payload_audits(
            [{"job_id": index} for index in range(4)],
            deep_payload_audit=True,
            workers=4,
            label="fixture",
            progress_callback=None,
        )


def _write_manual_completion_fixture(run_dir: Path) -> dict[str, Path]:
    run_dir.mkdir()
    best_path = run_dir / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    last_path.parent.mkdir()
    best_path.write_bytes(b"best checkpoint")
    last_path.write_bytes(b"last checkpoint")
    evaluation_path = run_dir / "manual_stop_test_metrics.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "num_samples": 2500,
                "training_seeds": [18, 36, 67],
                "members": [
                    {"training_seed": 18, "checkpoint_epoch": 91},
                    {"training_seed": 36, "checkpoint_epoch": 73},
                    {"training_seed": 67, "checkpoint_epoch": 118},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "completion": "manual_resource_stop",
                "manual_completion_record": "manual_completion.json",
            }
        ),
        encoding="utf-8",
    )
    record = {
        "schema_id": eval_suite_preflight.MANUAL_COMPLETION_SCHEMA_ID,
        "state": "completed",
        "completion": "manual_resource_stop",
        "seed": 67,
        "last_epoch": 124,
        "best_epoch": 118,
        "checkpoint_selection_basis": "validation_metric_only",
        "best_checkpoint_frozen_before_test_evaluation": True,
        "best_checkpoint_modified_after_snapshot": False,
        "continued_training_after_test_result": False,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "evaluation_artifact": evaluation_path.name,
        "evaluation_artifact_sha256": sha256_file(evaluation_path),
        "evaluation_num_samples": 2500,
        "evaluation_training_seeds": [18, 36, 67],
        "reason": "resource stop after a locked one-shot evaluation",
    }
    (run_dir / "manual_completion.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    return {
        "best": best_path,
        "last": last_path,
        "evaluation": evaluation_path,
    }


def test_manual_training_completion_is_hash_bound(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    paths = _write_manual_completion_fixture(run_dir)

    summary = eval_suite_preflight._validate_manual_training_completion(
        run_dir,
        seed=67,
        last_epoch=124,
        best_epoch=118,
        best_path=paths["best"],
        last_path=paths["last"],
    )

    assert summary is not None
    assert summary["evaluation"].endswith("manual_stop_test_metrics.json")


def test_manual_training_completion_rejects_changed_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    paths = _write_manual_completion_fixture(run_dir)
    paths["best"].write_bytes(b"changed checkpoint")

    with pytest.raises(
        eval_suite_preflight.PreflightError,
        match="best-checkpoint hash mismatch",
    ):
        eval_suite_preflight._validate_manual_training_completion(
            run_dir,
            seed=67,
            last_epoch=124,
            best_epoch=118,
            best_path=paths["best"],
            last_path=paths["last"],
        )
