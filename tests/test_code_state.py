from __future__ import annotations

import os
from pathlib import Path

from src.data_gen.common_time_v2 import code_state


def test_code_state_ignores_ignored_runtime_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_path = (
        root
        / "configs"
        / "cluster"
        / "generated"
        / f".code-state-test-{os.getpid()}.yaml"
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime_path.write_text("runtime: first\n", encoding="utf-8")
        before = code_state(root)
        runtime_path.write_text("runtime: second\n", encoding="utf-8")
        after = code_state(root)
    finally:
        runtime_path.unlink(missing_ok=True)

    assert before["source_inventory_hash"] == after["source_inventory_hash"]
    assert before["source_file_count"] == after["source_file_count"]
