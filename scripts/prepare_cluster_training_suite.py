#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import deep_update, load_config, load_yaml, save_config
from src.utils.io import get_git_commit


DEFAULT_MANIFEST = Path("configs/cluster/legacy_dev_suite.yaml")
DEFAULT_GENERATED_ROOT = Path("configs/cluster/generated")
ARRAY_SCRIPT = Path("slurm/train_suite_array.slurm")
REQUIRED_TRACKED_FILES = (
    Path("configs/cluster/legacy_dev_suite.yaml"),
    Path("scripts/prepare_cluster_training_suite.py"),
    Path("slurm/train_suite_array.slurm"),
)
MAX_ACCOUNT_MPS = 20
MAX_ACCOUNT_CPUS = 32
MAX_ACCOUNT_CONCURRENT_JOBS = 5
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class PreparedRun:
    label: str
    config_path: str
    output_dir: str
    base_config: str
    seed: int


@dataclass(frozen=True)
class PreparedSuite:
    suite_id: str
    classification: str
    generated_dir: str
    runs_file: str
    runs: tuple[PreparedRun, ...]
    disabled_entries: tuple[dict[str, str], ...]
    max_concurrent: int


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _require_slug(value: Any, name: str) -> str:
    slug = str(value).strip()
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"{name} must match {_SLUG_RE.pattern!r}, got {slug!r}"
        )
    return slug


def _require_seed_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise ValueError(f"{name} must be a non-empty list of integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return [int(seed) for seed in value]


def _path_exists_for_training(root: Path, raw_path: str) -> bool:
    path = root / raw_path
    if path.exists():
        return True
    if path.name == "eval_dataset.npz":
        return (path.parent / "shards_manifest.json").is_file()
    return False


def _validate_resources(resources: dict[str, Any]) -> int:
    max_concurrent = int(resources.get("max_concurrent", 0))
    mps_per_job = int(resources.get("mps_per_job", 0))
    cpus_per_job = int(resources.get("cpus_per_job", 0))
    time_limit = str(resources.get("time_limit", ""))

    if not 1 <= max_concurrent <= MAX_ACCOUNT_CONCURRENT_JOBS:
        raise ValueError(
            f"resources.max_concurrent must be in [1, "
            f"{MAX_ACCOUNT_CONCURRENT_JOBS}]"
        )
    if mps_per_job <= 0 or max_concurrent * mps_per_job > MAX_ACCOUNT_MPS:
        raise ValueError("suite MPS request exceeds the account limit")
    if cpus_per_job <= 0 or max_concurrent * cpus_per_job > MAX_ACCOUNT_CPUS:
        raise ValueError("suite CPU request exceeds the account limit")
    if time_limit != "72:00:00":
        raise ValueError(
            "resources.time_limit must remain at the documented 72-hour cap"
        )
    return max_concurrent


def _relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _tracked_worktree_is_clean(root: Path) -> bool:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=root,
        check=False,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=root,
        check=False,
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def _required_suite_files_are_tracked(root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            *(path.as_posix() for path in REQUIRED_TRACKED_FILES),
        ],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def prepare_suite(
    manifest_path: Path,
    generated_root: Path,
    *,
    root: Path = ROOT,
    check_data: bool = True,
) -> PreparedSuite:
    manifest_abs = manifest_path if manifest_path.is_absolute() else root / manifest_path
    manifest = _require_mapping(load_yaml(manifest_abs), "manifest")

    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported suite schema_version")

    suite_id = _require_slug(manifest.get("suite_id"), "suite_id")
    classification = str(manifest.get("classification", "")).strip()
    if classification != "legacy_dev_only":
        raise ValueError(
            "this submitter only accepts manifests classified legacy_dev_only"
        )

    resources = _require_mapping(manifest.get("resources"), "resources")
    max_concurrent = _validate_resources(resources)

    required_paths = manifest.get("required_paths", [])
    if not isinstance(required_paths, list):
        raise ValueError("required_paths must be a list")
    if check_data:
        missing = [
            str(path)
            for path in required_paths
            if not _path_exists_for_training(root, str(path))
        ]
        if missing:
            raise FileNotFoundError(
                "required training data is missing: " + ", ".join(missing)
            )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")

    generated_dir = generated_root / suite_id
    generated_abs = generated_dir if generated_dir.is_absolute() else root / generated_dir
    generated_abs.mkdir(parents=True, exist_ok=True)

    runs: list[PreparedRun] = []
    disabled_entries: list[dict[str, str]] = []
    labels: set[str] = set()
    output_dirs: set[str] = set()

    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"entries[{index}]")
        name = _require_slug(entry.get("name"), f"entries[{index}].name")
        base_config = Path(str(entry.get("config", "")))
        base_abs = base_config if base_config.is_absolute() else root / base_config
        if not base_abs.is_file():
            raise FileNotFoundError(f"missing base config: {base_config}")

        seeds = _require_seed_list(entry.get("seeds"), f"entries[{index}].seeds")
        if not bool(entry.get("enabled", True)):
            reason = str(entry.get("blocked_reason", "")).strip()
            if not reason:
                raise ValueError(f"disabled entry {name!r} needs blocked_reason")
            disabled_entries.append({"name": name, "reason": reason})
            continue

        overrides = _require_mapping(entry.get("overrides", {}), "overrides")
        for seed in seeds:
            label = f"{name}_seed_{seed}"
            if label in labels:
                raise ValueError(f"duplicate run label: {label}")
            labels.add(label)

            cfg = deepcopy(load_config(base_abs))
            cfg = deep_update(cfg, overrides)
            cfg.pop("seeds", None)
            cfg["seed"] = int(seed)
            cfg["device"] = "cuda"

            output_dir = (
                Path("experiments")
                / "slurm"
                / suite_id
                / name
                / f"seed_{seed}"
            )
            output_text = output_dir.as_posix()
            if output_text in output_dirs:
                raise ValueError(f"duplicate output directory: {output_text}")
            output_dirs.add(output_text)
            cfg["output_dir"] = output_text

            eval_cfg = dict(cfg.get("eval", {}))
            eval_cfg["output_dir"] = (output_dir / "eval").as_posix()
            cfg["eval"] = eval_cfg
            cfg["cluster_suite"] = {
                "suite_id": suite_id,
                "classification": classification,
                "run_label": label,
                "base_config": _relative_to_root(base_abs, root),
                "seed": int(seed),
                "source_git_commit": get_git_commit(),
            }

            config_abs = generated_abs / f"{label}.yaml"
            save_config(cfg, config_abs)
            runs.append(
                PreparedRun(
                    label=label,
                    config_path=_relative_to_root(config_abs, root),
                    output_dir=output_text,
                    base_config=_relative_to_root(base_abs, root),
                    seed=int(seed),
                )
            )

    if not runs:
        raise ValueError("suite has no enabled runs")

    runs_file_abs = generated_abs / "runs.tsv"
    runs_file_abs.write_text(
        "".join(
            f"{run.label}\t{run.config_path}\t{run.output_dir}\n"
            for run in runs
        ),
        encoding="utf-8",
    )
    summary_path = generated_abs / "suite_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "classification": classification,
                "run_count": len(runs),
                "max_concurrent": max_concurrent,
                "runs": [asdict(run) for run in runs],
                "disabled_entries": disabled_entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return PreparedSuite(
        suite_id=suite_id,
        classification=classification,
        generated_dir=_relative_to_root(generated_abs, root),
        runs_file=_relative_to_root(runs_file_abs, root),
        runs=tuple(runs),
        disabled_entries=tuple(disabled_entries),
        max_concurrent=max_concurrent,
    )


def build_sbatch_command(
    suite: PreparedSuite,
    *,
    afterok: int,
    max_concurrent: int | None = None,
    root: Path = ROOT,
) -> list[str]:
    array_script = root / ARRAY_SCRIPT
    if not array_script.is_file():
        raise FileNotFoundError(f"missing Slurm array script: {ARRAY_SCRIPT}")
    if afterok <= 0:
        raise ValueError("afterok must be a positive smoke-job ID")

    concurrency = (
        suite.max_concurrent
        if max_concurrent is None
        else int(max_concurrent)
    )
    if not 1 <= concurrency <= suite.max_concurrent:
        raise ValueError(
            "max_concurrent must be between 1 and the manifest limit "
            f"({suite.max_concurrent})"
        )

    last_index = len(suite.runs) - 1
    return [
        "sbatch",
        f"--array=0-{last_index}%{concurrency}",
        f"--dependency=afterok:{afterok}",
        (
            "--export=ALL,"
            f"RUNS_FILE={suite.runs_file},"
            f"SUITE_ID={suite.suite_id},"
            "GPU_HELPER_VERIFIED=1"
        ),
        ARRAY_SCRIPT.as_posix(),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one-seed-per-task configs for the guarded legacy/dev "
            "cluster training suite."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Prepare configs before data is present; forbidden with --submit.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Queue every prepared run as one Slurm array. Tasks above the "
            "concurrency limit remain pending. Default behavior is prepare-only."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Maximum array tasks allowed to run simultaneously. Defaults to "
            "the manifest limit and cannot exceed it."
        ),
    )
    parser.add_argument(
        "--afterok",
        type=int,
        help="Successful smoke-job ID used as an afterok dependency.",
    )
    parser.add_argument(
        "--acknowledge-legacy-dev",
        action="store_true",
        help="Acknowledge that this suite is not common-time-v2 final evidence.",
    )
    parser.add_argument(
        "--gpu-helper-verified",
        action="store_true",
        help="Confirm gpu_check.sh was inspected and validated on a compute node.",
    )
    args = parser.parse_args()

    if args.submit and args.skip_data_check:
        parser.error("--skip-data-check cannot be combined with --submit")

    try:
        suite = prepare_suite(
            args.manifest,
            args.generated_root,
            check_data=not args.skip_data_check,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"[cluster-suite] prepared {len(suite.runs)} runs "
        f"under {suite.generated_dir}"
    )
    for entry in suite.disabled_entries:
        print(
            f"[cluster-suite] blocked {entry['name']}: {entry['reason']}"
        )

    if not args.submit:
        print("[cluster-suite] prepare-only; no jobs submitted")
        return

    if not args.acknowledge_legacy_dev:
        parser.error("--submit requires --acknowledge-legacy-dev")
    if not args.gpu_helper_verified:
        parser.error("--submit requires --gpu-helper-verified")
    if args.afterok is None:
        parser.error("--submit requires --afterok SMOKE_JOB_ID")
    if not _tracked_worktree_is_clean(ROOT):
        parser.error(
            "--submit requires a clean tracked worktree; commit or restore "
            "tracked changes first"
        )
    if not _required_suite_files_are_tracked(ROOT):
        parser.error(
            "--submit requires the suite manifest, submitter, and array script "
            "to be committed"
        )

    try:
        cmd = build_sbatch_command(
            suite,
            afterok=args.afterok,
            max_concurrent=args.max_concurrent,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    concurrency = (
        suite.max_concurrent
        if args.max_concurrent is None
        else args.max_concurrent
    )
    print(
        f"[cluster-suite] queueing all {len(suite.runs)} tasks; "
        f"at most {concurrency} may run concurrently"
    )
    print("[cluster-suite] submitting:", " ".join(cmd))
    (ROOT / "logs" / "slurm").mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
