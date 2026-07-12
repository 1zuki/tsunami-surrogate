from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def default_progress_every_for_suite(suite_name: str) -> int:
    normalized = str(suite_name).strip().lower()
    if normalized == "smoke":
        return 1
    if normalized == "dense_validation":
        return 10
    if normalized == "full":
        return 25
    return 25


def resolve_progress_every(
    suite_name: str,
    progress_every: int | None,
) -> int:
    if progress_every is None:
        return default_progress_every_for_suite(suite_name)
    resolved = int(progress_every)
    if resolved <= 0:
        raise ValueError("progress_every must be a positive integer")
    return resolved


@dataclass
class ScenarioProgressLogger:
    label: str
    progress_every: int
    quiet: bool = False
    emit: Callable[[str], None] = print

    def __call__(
        self,
        completed: int,
        total: int,
        scenario_id: str | None = None,
    ) -> None:
        if self.quiet:
            return
        completed_count = int(completed)
        total_count = int(total)
        if completed_count <= 0 or total_count <= 0:
            return
        if (
            completed_count != total_count
            and completed_count % int(self.progress_every) != 0
        ):
            return
        suffix = f" last_scenario={scenario_id}" if scenario_id else ""
        self.emit(
            f"[{self.label}] progress completed={completed_count}/{total_count}{suffix}"
        )
