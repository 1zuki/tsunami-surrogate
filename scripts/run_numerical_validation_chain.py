#!/usr/bin/env python
"""Run a fresh isolated common-time-v2 numerical-validation regression chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_common_time_v2_h0 import audit_h0
from src.data_gen.common_time_v2 import code_state
from src.evaluation.common_time_v2_h1 import (
    execute_h1_contract,
    freeze_h1_contract,
)
from src.evaluation.common_time_v2_h2 import (
    execute_h2_contract,
    freeze_h2_contract,
)
from src.evaluation.common_time_v2_level_a import (
    execute_level_a,
    preregister_level_a,
)
from src.evaluation.established_solver_validation import (
    evaluate_minimum_established_solver_validation,
    prepare_minimum_established_solver_validation,
)
from src.evaluation.geoclaw_adapter import (
    GeoClawEnvironment,
    run_geoclaw_bundle,
    validate_geoclaw_environment,
)


STAGE_C_ARCHIVE = ROOT / (
    "artifacts/common_time_v2/stage_c_legacy_stride5_negative/"
    "1db46ddc9f6d2547ff01e74176c94d82fe4d0d962320a31b1510092d3be60ca6"
)
PRIOR_H2 = ROOT / (
    "artifacts/common_time_v2/h2/"
    "b0a91373ea8dc6ba4304a2b2d319cbeb551d5e211279b8fb799228b811058be9"
)
SWE_DIAGNOSTIC = ROOT / (
    "artifacts/common_time_v2/h2_diagnostics/swe_cfl_refinement_v1/"
    "1a29fea6ee28afb528e844abb55a6988249c09ed507e655e5aad1f657e2da138"
)
HYDRO_CONTINUATION = ROOT / (
    "artifacts/common_time_v2/h2_diagnostics/hydro_cfl_continuation_v1/"
    "55c1766cefa2300299ab6067bda09b2ddfeac51be16c00a68619aa058ef02272"
)
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_object(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(staging, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


def _require_path(path: Path, *, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(path)
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(path)


def validate_prerequisites(
    *,
    output_root: Path | None,
    claw_root: Path,
    petsc_dir: Path,
    petsc_arch: str,
    geoclaw_python: Path,
) -> dict[str, Any]:
    state = code_state(ROOT)
    if state["dirty"]:
        raise RuntimeError(
            "Fresh numerical validation requires a clean committed scientific "
            "code/config tree (src/, scripts/, configs/, and dependency files)"
        )
    if output_root is not None and output_root.exists():
        raise FileExistsError(
            f"Refusing to reuse numerical-validation output root: {output_root}"
        )
    for path, kind in (
        (ROOT / "data/train", "dir"),
        (ROOT / "data/eval", "dir"),
        (ROOT / "data/test", "dir"),
        (STAGE_C_ARCHIVE / "manifest.json", "file"),
        (PRIOR_H2 / "execution/result.json", "file"),
        (SWE_DIAGNOSTIC / "execution/result.json", "file"),
        (HYDRO_CONTINUATION / "execution/result.json", "file"),
        (ROOT / "configs/eval/common_time_v2_level_a.yaml", "file"),
        (ROOT / "configs/eval/minimum_established_solver_validation_v4.yaml", "file"),
        (ROOT / "configs/eval/common_time_v2_h1.yaml", "file"),
        (ROOT / "configs/eval/common_time_v2_h2_v2.yaml", "file"),
    ):
        _require_path(path, kind=kind)
    for executable in ("tar", "zstd"):
        if shutil.which(executable) is None:
            raise RuntimeError(
                f"Numerical-validation archive tool is missing: {executable}"
            )
    environment = GeoClawEnvironment(
        claw_root=claw_root,
        petsc_dir=petsc_dir,
        petsc_arch=petsc_arch,
        python_executable=geoclaw_python,
    )
    external_revisions = validate_geoclaw_environment(environment)
    return {
        "status": "ready",
        "code_state": state,
        "stage_c_archive": str(STAGE_C_ARCHIVE.relative_to(ROOT)),
        "external_revisions": external_revisions,
        "claw_root": str(claw_root.resolve()),
        "petsc_dir": str(petsc_dir.resolve()),
        "petsc_arch": petsc_arch,
        "geoclaw_python": str(geoclaw_python.resolve()),
    }


def _require_decision(
    path: Path,
    *,
    key: str,
    expected: Any,
) -> dict[str, Any]:
    payload = _read_object(path)
    if payload.get(key) != expected:
        raise RuntimeError(
            f"Numerical-validation decision mismatch at {path}: "
            f"{payload.get(key)!r} != {expected!r}"
        )
    return payload


def _archive_workspace(workspace: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "--zstd",
            "-cf",
            str(archive_path),
            "-C",
            str(workspace.parent),
            workspace.name,
        ],
        check=True,
    )


def _event_progress(stage: str):
    def report(event: Mapping[str, Any]) -> None:
        kind = str(event.get("event", "progress"))
        completed = event.get("completed")
        total = event.get("total")
        decision = event.get("decision")
        if completed is not None and total is not None:
            print(
                f"[numerical-chain:{stage}] {kind} {completed}/{total}",
                flush=True,
            )
        elif decision is not None:
            print(
                f"[numerical-chain:{stage}] {kind} decision={decision}",
                flush=True,
            )
        else:
            print(f"[numerical-chain:{stage}] {kind}", flush=True)

    return report


def run_chain(
    *,
    output_root: Path,
    workers: int,
    geoclaw_workers: int,
    claw_root: Path,
    petsc_dir: Path,
    petsc_arch: str,
    geoclaw_python: Path,
) -> Path:
    preflight = validate_prerequisites(
        output_root=output_root,
        claw_root=claw_root,
        petsc_dir=petsc_dir,
        petsc_arch=petsc_arch,
        geoclaw_python=geoclaw_python,
    )
    for key in THREAD_ENV_KEYS:
        os.environ[key] = "1"

    output_root.mkdir(parents=True, exist_ok=False)
    workspace = output_root / ".chain-workspace"
    workspace.mkdir()
    configs_root = workspace / "configs"
    artifacts_root = workspace / "artifacts"

    print("[numerical-chain] H0 authoritative-input audit", flush=True)
    h0_root = audit_h0(
        split_roots={
            "train": ROOT / "data/train",
            "eval": ROOT / "data/eval",
            "test": ROOT / "data/test",
        },
        expected_counts={"train": 10_000, "eval": 1_000, "test": 2_500},
        output_root=artifacts_root / "h0",
        stage_c_archive=STAGE_C_ARCHIVE,
    )
    h0_decision = _require_decision(
        h0_root / "h0_decision.json",
        key="audit_passed",
        expected=True,
    )

    level_a_config = _load_yaml(ROOT / "configs/eval/common_time_v2_level_a.yaml")
    level_a_config["canaries"]["h0_inventory"] = str(
        h0_root / "h0_input_inventory.jsonl"
    )
    level_a_config_path = configs_root / "common_time_v2_level_a.yaml"
    _write_yaml(level_a_config_path, level_a_config)
    print("[numerical-chain] Level A preregistration and execution", flush=True)
    level_a_root = preregister_level_a(
        repo_root=ROOT,
        config_path=level_a_config_path,
        output_root=artifacts_root / "level_a",
        h0_root=h0_root,
    )
    execute_level_a(
        repo_root=ROOT,
        contract_root=level_a_root,
        workers=workers,
        max_in_flight=workers,
        resume=False,
        progress_callback=_event_progress("level-a"),
    )
    level_a_decision = _require_decision(
        level_a_root / "execution/decision.json",
        key="decision",
        expected="pass_to_H1",
    )

    print("[numerical-chain] Minimum established-solver bundle", flush=True)
    level_b_root = prepare_minimum_established_solver_validation(
        repo_root=ROOT,
        config_path=ROOT / "configs/eval/minimum_established_solver_validation_v4.yaml",
        level_a_root=level_a_root,
        output_root=artifacts_root / "level_b_minimum",
        workers=workers,
        progress=lambda message: print(message, flush=True),
    )
    external_root = artifacts_root / "level_b_minimum_external" / level_b_root.name
    run_geoclaw_bundle(
        bundle_root=level_b_root,
        external_root=external_root,
        environment=GeoClawEnvironment(
            claw_root=claw_root,
            petsc_dir=petsc_dir,
            petsc_arch=petsc_arch,
            python_executable=geoclaw_python,
        ),
        workers=geoclaw_workers,
        resume=False,
        progress=lambda message: print(message, flush=True),
    )
    level_b_evaluation_root = (
        artifacts_root / "level_b_minimum_evaluation" / level_b_root.name
    )
    evaluate_minimum_established_solver_validation(
        bundle_root=level_b_root,
        external_root=external_root,
        output_root=level_b_evaluation_root,
        progress=lambda message: print(message, flush=True),
    )
    level_b_decision = _require_decision(
        level_b_evaluation_root / "decision.json",
        key="decision",
        expected="pass_to_H1",
    )

    print("[numerical-chain] H1 implementation/health smoke", flush=True)
    h1_config = _load_yaml(ROOT / "configs/eval/common_time_v2_h1.yaml")
    h1_config["prerequisites"].update(
        {
            "h0_contract_hash": h0_root.name,
            "level_a_contract_hash": level_a_root.name,
            "level_b_bundle_hash": level_b_root.name,
        }
    )
    h1_config_path = configs_root / "common_time_v2_h1.yaml"
    _write_yaml(h1_config_path, h1_config)
    h1_root = freeze_h1_contract(
        repo_root=ROOT,
        config_path=h1_config_path,
        h0_root=h0_root,
        level_a_root=level_a_root,
        level_b_bundle_root=level_b_root,
        level_b_evaluation_root=level_b_evaluation_root,
        output_base=artifacts_root / "h1",
    )
    execute_h1_contract(
        repo_root=ROOT,
        contract_root=h1_root,
        workers=workers,
        max_in_flight=workers,
        resume=False,
        progress=_event_progress("h1"),
    )
    h1_decision = _require_decision(
        h1_root / "execution/decision.json",
        key="decision",
        expected="pass_to_H2",
    )

    print("[numerical-chain] H2 paired-CFL regression", flush=True)
    h2_config = _load_yaml(ROOT / "configs/eval/common_time_v2_h2_v2.yaml")
    h2_config["prerequisites"].update(
        {
            "h0_contract_hash": h0_root.name,
            "level_a_contract_hash": level_a_root.name,
            "level_b_bundle_hash": level_b_root.name,
            "h1_contract_hash": h1_root.name,
        }
    )
    h2_config["selection"]["exclude_h1_contract_hash"] = h1_root.name
    h2_config_path = configs_root / "common_time_v2_h2_v2.yaml"
    _write_yaml(h2_config_path, h2_config)
    h2_root = freeze_h2_contract(
        repo_root=ROOT,
        config_path=h2_config_path,
        h0_root=h0_root,
        level_a_root=level_a_root,
        level_b_bundle_root=level_b_root,
        level_b_evaluation_root=level_b_evaluation_root,
        h1_root=h1_root,
        output_base=artifacts_root / "h2",
        prior_h2_root=PRIOR_H2,
        swe_diagnostic_root=SWE_DIAGNOSTIC,
        hydro_continuation_root=HYDRO_CONTINUATION,
    )
    execute_h2_contract(
        repo_root=ROOT,
        contract_root=h2_root,
        workers=workers,
        max_in_flight=workers,
        resume=False,
        progress=_event_progress("h2"),
    )
    h2_decision = _require_decision(
        h2_root / "execution/decision.json",
        key="decision",
        expected="pass_to_common_time_v2_contract_freeze",
    )

    print("[numerical-chain] Archiving checksum-bound evidence", flush=True)
    archive_path = output_root / "chain.tar.zst"
    _archive_workspace(workspace, archive_path)
    archive_sha256 = _sha256(archive_path)
    summary = {
        "evaluation_type": "v2_numerical_validation_regression_chain",
        "status": "passed",
        "interpretation": (
            "Fresh regression replay under the current committed source and "
            "the frozen reviewed thresholds. It supplements but does not "
            "rewrite the accepted historical production-validation artifacts."
        ),
        "code_state": preflight["code_state"],
        "external_revisions": preflight["external_revisions"],
        "stages": [
            {
                "id": "h0",
                "artifact_hash": h0_root.name,
                "decision": bool(h0_decision["audit_passed"]),
            },
            {
                "id": "level_a",
                "artifact_hash": level_a_root.name,
                "decision": level_a_decision["decision"],
            },
            {
                "id": "minimum_established_solver",
                "artifact_hash": level_b_root.name,
                "decision": level_b_decision["decision"],
            },
            {
                "id": "h1",
                "artifact_hash": h1_root.name,
                "decision": h1_decision["decision"],
            },
            {
                "id": "h2_v2",
                "artifact_hash": h2_root.name,
                "decision": h2_decision["decision"],
            },
        ],
        "archive_path": (
            Path(output_root.name) / archive_path.relative_to(output_root)
        ).as_posix(),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": int(archive_path.stat().st_size),
    }
    summary_path = output_root / "summary.json"
    _write_object(summary_path, summary)
    shutil.rmtree(workspace)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--geoclaw-workers", type=int, default=4)
    parser.add_argument(
        "--claw-root",
        type=Path,
        default=Path("/home/izu/opt/clawpack-v5.14.0"),
    )
    parser.add_argument(
        "--petsc-dir",
        type=Path,
        default=Path("/home/izu/opt/petsc-3.25.3"),
    )
    parser.add_argument("--petsc-arch", default="arch-linux-c-opt")
    parser.add_argument(
        "--geoclaw-python",
        type=Path,
        default=Path("/usr/bin/python3"),
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.geoclaw_workers <= 0:
        parser.error("worker counts must be positive")
    if args.workers != 8:
        parser.error(
            "--workers must remain 8 because the reviewed Level A/H1/H2 "
            "execution contracts freeze eight workers and eight in-flight tasks"
        )
    if args.preflight:
        result = validate_prerequisites(
            output_root=args.output_root,
            claw_root=args.claw_root,
            petsc_dir=args.petsc_dir,
            petsc_arch=args.petsc_arch,
            geoclaw_python=args.geoclaw_python,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --preflight is used")
    summary = run_chain(
        output_root=args.output_root,
        workers=args.workers,
        geoclaw_workers=args.geoclaw_workers,
        claw_root=args.claw_root,
        petsc_dir=args.petsc_dir,
        petsc_arch=args.petsc_arch,
        geoclaw_python=args.geoclaw_python,
    )
    print(summary)


if __name__ == "__main__":
    main()
