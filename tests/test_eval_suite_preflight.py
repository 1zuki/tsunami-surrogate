from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from scripts import eval_suite_preflight


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
