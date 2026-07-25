#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.common_time_v2 import (  # noqa: E402
    code_state,
    validate_generation_contract_artifact,
)
from src.data_gen.generation_sequence import (  # noqa: E402
    STAGES,
    _load_yaml,
    execute_stage,
    execution_policy,
    freeze_generation_contract,
    preflight_stage,
    resolve_stage_config,
    validate_stage_prerequisites,
    verify_stage,
    write_stage_attestation,
)
from src.data_gen.simulate_dataset import TsunamiDatasetBuilder  # noqa: E402


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
DEFAULT_CONFIG = ROOT / "configs/data/dataset.yaml"
LEGACY_CONFIG = ROOT / "configs/data/legacy/dataset_saved_step_v1.yaml"


def _configure_threads(workers: int) -> None:
    if workers <= 1:
        return
    invalid = {
        key: value
        for key in THREAD_ENV_KEYS
        if (value := os.environ.get(key)) not in (None, "1")
    }
    if invalid:
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(invalid.items())
        )
        raise SystemExit(
            "Multiprocess generation requires single-thread numerical backends; "
            f"found {details}"
        )
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, "1")


def _add_direct_arguments(parser: argparse.ArgumentParser, *, legacy: bool) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=not legacy,
        default=LEGACY_CONFIG if legacy else None,
    )
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--split", choices=("train", "eval", "test"))
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument(
        "--solver-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cloud-provider")
    parser.add_argument("--cloud-zone")
    parser.add_argument("--machine-type")
    parser.add_argument("--storage-class")
    parser.add_argument("--hourly-cost-usd", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--bathymetry-dir")
    parser.add_argument("--source-dir")
    parser.add_argument("--manifest-path")
    parser.add_argument("--continue", dest="continue_from_last", action="store_true")
    parser.add_argument("--start-at", type=int)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--allow-override", action="store_true")
    parser.add_argument("--rebuild-manifests", action="store_true")
    parser.add_argument("--acknowledge-provisional", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single entrypoint for tsunami dataset generation. Production is "
            "common-time and staged; saved-step generation requires the explicit "
            "legacy-v1 command."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    production = subparsers.add_parser(
        "production",
        help="Fail-closed common-time production preflight or execution.",
    )
    production.add_argument("--stage", choices=tuple(STAGES), required=True)
    production.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    production.add_argument("--input-root", type=Path, required=True)
    production.add_argument("--run-root", type=Path, required=True)
    production.add_argument("--workers", type=int)
    production.add_argument("--max-in-flight", type=int)
    production.add_argument("--cloud-provider")
    production.add_argument("--cloud-zone")
    production.add_argument("--machine-type")
    production.add_argument("--storage-class")
    production.add_argument("--hourly-cost-usd", type=float)
    production_mode = production.add_mutually_exclusive_group()
    production_mode.add_argument("--execute", action="store_true")
    production_mode.add_argument("--verify-only", action="store_true")

    freeze = subparsers.add_parser(
        "freeze-contract",
        help="Recover contract freezing after a verified rehearsal completed.",
    )
    freeze.add_argument("--input-root", type=Path, required=True)
    freeze.add_argument("--run-root", type=Path, required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Direct provisional common-time generation for tests or diagnostics.",
    )
    _add_direct_arguments(generate, legacy=False)

    legacy = subparsers.add_parser(
        "legacy-v1",
        help="Explicitly reproduce archived saved-step v1 data.",
    )
    _add_direct_arguments(legacy, legacy=True)
    return parser


def _apply_direct_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("dataset", {})
    dataset = cfg["dataset"]
    for argument, key in (
        ("num_samples", "num_samples"),
        ("n_steps", "n_steps"),
        ("save_every", "save_every"),
        ("num_workers", "num_workers"),
    ):
        value = getattr(args, argument)
        if value is not None:
            dataset[key] = int(value)
    for argument, key in (
        ("output_dir", "output_dir"),
        ("bathymetry_dir", "bathymetry_dir"),
        ("source_dir", "source_dir"),
        ("manifest_path", "manifest_path"),
    ):
        value = getattr(args, argument)
        if value is not None:
            dataset[key] = str(value)
    if args.split is not None:
        cfg.setdefault("requested_output", {})["split"] = args.split

    cfg.setdefault("operations", {})
    operations = cfg["operations"]
    if args.max_in_flight is not None:
        operations["max_in_flight"] = int(args.max_in_flight)
    if args.progress_every is not None:
        operations["progress_every"] = int(args.progress_every)
    if args.solver_progress is not None:
        operations["solver_progress"] = bool(args.solver_progress)
    for key in ("cloud_provider", "cloud_zone", "machine_type", "storage_class"):
        value = getattr(args, key)
        if value is not None:
            operations[key] = str(value)
    if args.hourly_cost_usd is not None:
        operations["hourly_cost_usd"] = float(args.hourly_cost_usd)


def _run_direct(args: argparse.Namespace, *, legacy: bool) -> None:
    config_path = args.config.resolve()
    cfg = _load_yaml(config_path)
    requested = cfg.get("requested_output")
    requested_enabled = isinstance(requested, dict) and bool(
        requested.get("enabled", False)
    )
    if legacy and requested_enabled:
        raise SystemExit(
            "legacy-v1 refuses requested-output configs; use the generate or "
            "production command"
        )
    if not legacy and not requested_enabled:
        raise SystemExit(
            "generate requires common-time requested_output; saved-step configs "
            "require the explicit legacy-v1 command"
        )
    if not legacy and str(requested.get("status", "provisional")) == "accepted":
        raise SystemExit("accepted generation must use the production command")
    if legacy and args.split is not None:
        raise SystemExit("--split is not meaningful for saved-step legacy generation")

    _apply_direct_overrides(cfg, args)
    workers = int(cfg["dataset"].get("num_workers", 1))
    _configure_threads(workers)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temporary_config = Path(handle.name)
    try:
        builder = TsunamiDatasetBuilder(
            str(temporary_config), provenance_config_path=config_path
        )
        builder.run(
            continue_from_last=bool(args.continue_from_last),
            start_at=args.start_at,
            stop_at=args.stop_at,
            allow_override=bool(args.allow_override),
            rebuild_manifests=bool(args.rebuild_manifests),
            acknowledge_provisional=bool(args.acknowledge_provisional),
        )
        print(f"Dataset generation complete using {config_path}")
    finally:
        temporary_config.unlink(missing_ok=True)


def _output_dir(run_root: Path, stage_name: str) -> Path:
    folder = "train" if stage_name in {"train-1", "train-2"} else stage_name
    return run_root / folder / "raw"


def _manifest_path(run_root: Path, split: str) -> Path:
    return run_root / "manifests" / f"{split}_scenario_manifest.jsonl"


def _discover_one(paths: list[Path], *, label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {len(paths)}")
    return paths[0]


def _discover_contract(run_root: Path) -> tuple[Path, str, dict[str, Any]]:
    path = _discover_one(
        sorted((run_root / "generation_contracts").glob("*/generation_contract.json")),
        label="accepted generation contract",
    )
    artifact = validate_generation_contract_artifact(path)
    return path, str(artifact["contract_hash"]), artifact


def _discover_attestation(run_root: Path, stage_name: str) -> Path:
    output = _output_dir(run_root, stage_name)
    return _discover_one(
        sorted((output / "stage_attestations").glob(f"{stage_name}_*.json")),
        label=f"{stage_name} stage attestation",
    )


def _rehearsal_policy(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "workers": args.workers,
        "max_in_flight": args.max_in_flight,
        "cloud_provider": args.cloud_provider,
        "cloud_zone": args.cloud_zone,
        "machine_type": args.machine_type,
        "storage_class": args.storage_class,
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise SystemExit(
            "rehearsal requires the real execution policy; missing " + ", ".join(missing)
        )
    return execution_policy(
        workers=int(args.workers),
        max_in_flight=int(args.max_in_flight),
        cloud_provider=str(args.cloud_provider),
        cloud_zone=str(args.cloud_zone),
        machine_type=str(args.machine_type),
        storage_class=str(args.storage_class),
        hourly_cost_usd=args.hourly_cost_usd,
    )


def _production_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path, list[Path]]:
    stage = STAGES[args.stage]
    run_root = args.run_root.resolve()
    output = _output_dir(run_root, stage.name)
    manifest = _manifest_path(run_root, stage.split)
    contract_path: Path | None = None
    contract_hash: str | None = None
    prerequisites: list[Path] = []
    if stage.requires_generation_contract:
        contract_path, contract_hash, artifact = _discover_contract(run_root)
        policy = dict(artifact["execution_policy"])
        if any(
            value is not None
            for value in (
                args.workers,
                args.max_in_flight,
                args.cloud_provider,
                args.cloud_zone,
                args.machine_type,
                args.storage_class,
                args.hourly_cost_usd,
            )
        ):
            raise SystemExit(
                "accepted stages load the frozen execution policy automatically; "
                "do not repeat or override cloud/worker flags"
            )
        prerequisites = [
            _discover_attestation(run_root, required)
            for required in {
                "train-1": ("validation",),
                "train-2": ("validation", "train-1"),
                "test": ("validation", "train-1", "train-2"),
            }.get(stage.name, ())
        ]
    else:
        policy = _rehearsal_policy(args)
    resolved = resolve_stage_config(
        base_config=args.config,
        stage=stage,
        input_root=args.input_root,
        output_dir=output,
        manifest_path=manifest,
        policy=policy,
        generation_contract_path=contract_path,
        generation_contract_hash_value=contract_hash,
    )
    validate_stage_prerequisites(
        stage=stage,
        attestation_paths=prerequisites,
        generation_contract_hash_value=contract_hash,
    )
    return resolved, output, manifest, prerequisites


def _run_production(args: argparse.Namespace) -> None:
    stage = STAGES[args.stage]
    resolved, output, manifest, prerequisites = _production_context(args)
    if args.verify_only:
        result = verify_stage(
            resolved_config=resolved,
            stage=stage,
            input_root=args.input_root,
            output_dir=output,
        )
    else:
        preflight = preflight_stage(
            resolved_config=resolved,
            stage=stage,
            input_root=args.input_root,
            output_dir=output,
            manifest_path=manifest,
        )
        if not args.execute:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return
        state = code_state(ROOT)
        if bool(state["dirty"]):
            raise RuntimeError(
                "Refusing generation from a dirty checkout; commit and transfer "
                "the exact tested code first"
            )
        execute_stage(resolved_config=resolved, stage=stage)
        result = verify_stage(
            resolved_config=resolved,
            stage=stage,
            input_root=args.input_root,
            output_dir=output,
        )
    result["prerequisite_attestations"] = {
        path.name: str(path) for path in prerequisites
    }
    result["stage_attestation"] = str(
        write_stage_attestation(output_dir=output, verification=result)
    )
    if stage.name == "rehearsal" and args.execute:
        artifact_dir = freeze_generation_contract(
            rehearsal_config=resolved,
            rehearsal_verification=result,
            artifact_root=args.run_root / "generation_contracts",
        )
        contract = validate_generation_contract_artifact(
            artifact_dir / "generation_contract.json"
        )
        result["generation_contract"] = str(
            artifact_dir / "generation_contract.json"
        )
        result["generation_contract_hash"] = contract["contract_hash"]
        result["mass_generation_authorized"] = True
    print(json.dumps(result, indent=2, sort_keys=True))


def _freeze_contract(args: argparse.Namespace) -> None:
    stage = STAGES["rehearsal"]
    output = _output_dir(args.run_root.resolve(), stage.name)
    snapshot = output / "dataset_config.snapshot.yaml"
    resolved = _load_yaml(snapshot)
    verification = verify_stage(
        resolved_config=resolved,
        stage=stage,
        input_root=args.input_root,
        output_dir=output,
    )
    artifact_dir = freeze_generation_contract(
        rehearsal_config=resolved,
        rehearsal_verification=verification,
        artifact_root=args.run_root / "generation_contracts",
    )
    contract = validate_generation_contract_artifact(
        artifact_dir / "generation_contract.json"
    )
    print(
        json.dumps(
            {
                "generation_contract": str(
                    artifact_dir / "generation_contract.json"
                ),
                "generation_contract_hash": contract["contract_hash"],
                "mass_generation_authorized": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "production":
        _run_production(args)
    elif args.command == "freeze-contract":
        _freeze_contract(args)
    elif args.command == "generate":
        _run_direct(args, legacy=False)
    elif args.command == "legacy-v1":
        _run_direct(args, legacy=True)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
