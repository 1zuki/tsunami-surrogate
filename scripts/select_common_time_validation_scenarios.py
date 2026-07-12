#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.alignment import (
    SCHEMA_ID,
    stable_hash_payload,
    stable_hash_scenario_ids,
)
from src.utils.config import load_config
from src.utils.io import save_json


PRIMARY_POLICY_NAME = "primary-4-per-family-cell"
REDUCED_POLICY_NAME = "reduced-2-per-family-cell"
DEFAULT_SMOKE_COUNT = 12


def _repo_relative_text(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _stable_rank(
    *,
    audit_hash: str,
    seed: int,
    scenario_id: str,
    bathymetry_type: str,
    source_type: str,
    purpose: str,
) -> str:
    return stable_hash_payload(
        {
            "schema_id": SCHEMA_ID,
            "artifact_kind": "common-time-validation-ranking",
            "purpose": purpose,
            "audit_hash": str(audit_hash),
            "seed": int(seed),
            "scenario_id": str(scenario_id),
            "bathymetry_type": str(bathymetry_type),
            "source_type": str(source_type),
        }
    )


def _normalize_policy_name(policy_name: str) -> str:
    text = str(policy_name).strip().lower()
    if text in ("primary", PRIMARY_POLICY_NAME):
        return PRIMARY_POLICY_NAME
    if text in ("reduced", REDUCED_POLICY_NAME):
        return REDUCED_POLICY_NAME
    raise ValueError(
        f"Unsupported selection policy {policy_name!r}. "
        f"Expected one of: primary, {PRIMARY_POLICY_NAME}, reduced, {REDUCED_POLICY_NAME}."
    )


def _required_per_cell(policy_name: str) -> int:
    if policy_name == PRIMARY_POLICY_NAME:
        return 4
    if policy_name == REDUCED_POLICY_NAME:
        return 2
    raise ValueError(f"Unsupported policy {policy_name!r}")


def _smoke_selection(
    dense_validation_rows: list[dict[str, Any]],
    *,
    audit_hash: str,
    seed: int,
    smoke_count: int,
) -> list[dict[str, Any]]:
    if smoke_count <= 0:
        raise ValueError("smoke_count must be positive")
    if smoke_count > len(dense_validation_rows):
        raise ValueError("smoke_count cannot exceed dense validation count")

    remaining = []
    for row in dense_validation_rows:
        ranked = dict(row)
        ranked["smoke_rank_hash"] = _stable_rank(
            audit_hash=audit_hash,
            seed=seed,
            scenario_id=str(row["scenario_id"]),
            bathymetry_type=str(row["bathymetry_type"]),
            source_type=str(row["source_type"]),
            purpose="implementation-only-smoke",
        )
        remaining.append(ranked)

    selected: list[dict[str, Any]] = []
    covered_bathymetry: set[str] = set()
    covered_source: set[str] = set()

    while remaining and len(selected) < smoke_count:
        best = min(
            remaining,
            key=lambda row: (
                -(
                    int(str(row["bathymetry_type"]) not in covered_bathymetry)
                    + int(str(row["source_type"]) not in covered_source)
                ),
                -int(str(row["bathymetry_type"]) not in covered_bathymetry),
                -int(str(row["source_type"]) not in covered_source),
                str(row["smoke_rank_hash"]),
            ),
        )
        selected.append(best)
        covered_bathymetry.add(str(best["bathymetry_type"]))
        covered_source.add(str(best["source_type"]))
        remaining = [
            row
            for row in remaining
            if str(row["scenario_id"]) != str(best["scenario_id"])
        ]

    return [
        {
            "scenario_id": str(row["scenario_id"]),
            "bathymetry_type": str(row["bathymetry_type"]),
            "source_type": str(row["source_type"]),
            "source_strength": float(row["source_strength"]),
        }
        for row in selected[:smoke_count]
    ]


def select_common_time_validation_scenarios(
    audit_artifact: Mapping[str, Any],
    *,
    policy_name: str,
    seed: int,
    smoke_count: int = DEFAULT_SMOKE_COUNT,
) -> dict[str, Any]:
    if str(audit_artifact.get("schema_id", "")) != SCHEMA_ID:
        raise ValueError(f"Audit artifact schema_id must be {SCHEMA_ID!r}")
    if str(audit_artifact.get("artifact_kind", "")) != "paired-reference-audit":
        raise ValueError("Expected a paired-reference-audit artifact")
    if str(audit_artifact.get("status", "")) != "pass":
        raise ValueError("Selection requires a passing audit artifact")

    normalized_policy = _normalize_policy_name(policy_name)
    required_per_cell = _required_per_cell(normalized_policy)
    audit_hash = str(audit_artifact.get("audit_hash", ""))
    if not audit_hash:
        raise ValueError("Audit artifact is missing audit_hash")

    eligible_scenarios = list(audit_artifact.get("eligible_scenarios", []))
    if not eligible_scenarios:
        raise ValueError("Audit artifact has no eligible_scenarios")

    scenarios_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in eligible_scenarios:
        entry = {
            "scenario_id": str(row["scenario_id"]),
            "bathymetry_type": str(row["bathymetry_type"]),
            "source_type": str(row["source_type"]),
            "source_strength": float(row["source_strength"]),
        }
        scenarios_by_cell.setdefault(
            (entry["bathymetry_type"], entry["source_type"]),
            [],
        ).append(entry)

    undersized_cells = [
        {
            "bathymetry_type": bathymetry_type,
            "source_type": source_type,
            "eligible_count": len(rows),
            "required_count": required_per_cell,
        }
        for (bathymetry_type, source_type), rows in sorted(scenarios_by_cell.items())
        if len(rows) < required_per_cell
    ]
    if undersized_cells:
        raise ValueError(
            f"Selection policy {normalized_policy!r} cannot be satisfied: "
            f"{undersized_cells!r}"
        )

    dense_validation_rows: list[dict[str, Any]] = []
    family_cells: list[dict[str, Any]] = []
    for bathymetry_type, source_type in sorted(scenarios_by_cell):
        rows = scenarios_by_cell[(bathymetry_type, source_type)]
        ranked = []
        for row in rows:
            entry = dict(row)
            entry["rank_hash"] = _stable_rank(
                audit_hash=audit_hash,
                seed=seed,
                scenario_id=str(row["scenario_id"]),
                bathymetry_type=str(row["bathymetry_type"]),
                source_type=str(row["source_type"]),
                purpose=normalized_policy,
            )
            ranked.append(entry)
        ranked.sort(key=lambda row: str(row["rank_hash"]))
        selected = ranked[:required_per_cell]
        dense_validation_rows.extend(selected)
        family_cells.append(
            {
                "bathymetry_type": bathymetry_type,
                "source_type": source_type,
                "eligible_count": int(len(rows)),
                "selected_count": int(len(selected)),
            }
        )

    dense_validation_output = [
        {
            "scenario_id": str(row["scenario_id"]),
            "bathymetry_type": str(row["bathymetry_type"]),
            "source_type": str(row["source_type"]),
            "source_strength": float(row["source_strength"]),
        }
        for row in dense_validation_rows
    ]
    smoke_rows = _smoke_selection(
        dense_validation_output,
        audit_hash=audit_hash,
        seed=int(seed),
        smoke_count=int(smoke_count),
    )

    observed_bathymetry = sorted(
        {str(row["bathymetry_type"]) for row in dense_validation_output}
    )
    observed_source = sorted(
        {str(row["source_type"]) for row in dense_validation_output}
    )

    result = {
        "schema_id": SCHEMA_ID,
        "artifact_kind": "common-time-validation-scenarios",
        "audit_hash": audit_hash,
        "seed": int(seed),
        "selection_policy": {
            "name": normalized_policy,
            "required_per_family_cell": int(required_per_cell),
            "automatic_fallback": False,
            "label": "reduced"
            if normalized_policy == REDUCED_POLICY_NAME
            else "primary",
        },
        "observed_bathymetry_types": observed_bathymetry,
        "observed_source_types": observed_source,
        "family_cells": family_cells,
        "dense_validation": {
            "count": int(len(dense_validation_output)),
            "ordered_scenarios": dense_validation_output,
            "ordered_scenario_ids": [
                str(row["scenario_id"]) for row in dense_validation_output
            ],
            "list_hash": stable_hash_scenario_ids(
                [str(row["scenario_id"]) for row in dense_validation_output]
            ),
        },
        "smoke": {
            "label": "implementation_only_smoke",
            "count": int(len(smoke_rows)),
            "ordered_scenarios": smoke_rows,
            "ordered_scenario_ids": [str(row["scenario_id"]) for row in smoke_rows],
            "list_hash": stable_hash_scenario_ids(
                [str(row["scenario_id"]) for row in smoke_rows]
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select frozen common-time validation scenarios from a passing audit artifact."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/eval/common_time_alignment.yaml",
        help="Alignment/selection YAML config.",
    )
    parser.add_argument(
        "--audit-artifact",
        default="results/common_time_validation/audit/paired_reference_audit.json",
        help="Passing paired-reference audit artifact JSON.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Selection policy: primary or reduced.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional explicit ranking seed override.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path.",
    )
    args = parser.parse_args()

    config_path = (
        ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    )
    config = load_config(config_path)
    selection_cfg = dict(config.get("selection", {}))
    policy_name = (
        args.policy
        if args.policy is not None
        else str(selection_cfg.get("dense_validation", {}).get("policy", "primary"))
    )
    seed = int(
        args.seed if args.seed is not None else selection_cfg.get("seed", 20260711)
    )
    smoke_count = int(selection_cfg.get("smoke", {}).get("count", DEFAULT_SMOKE_COUNT))

    audit_path = (
        ROOT / args.audit_artifact
        if not Path(args.audit_artifact).is_absolute()
        else Path(args.audit_artifact)
    )
    with audit_path.open("r", encoding="utf-8") as handle:
        audit_artifact = json.load(handle)

    selected = select_common_time_validation_scenarios(
        audit_artifact,
        policy_name=policy_name,
        seed=seed,
        smoke_count=smoke_count,
    )
    selected["source_audit_artifact"] = _repo_relative_text(audit_path)

    default_output = selection_cfg.get(
        "output_path",
        "configs/eval/common_time_validation_scenarios.json",
    )
    output_path = (
        (
            ROOT / args.output
            if args.output is not None and not Path(args.output).is_absolute()
            else Path(args.output)
        )
        if args.output is not None
        else (ROOT / default_output)
    )
    save_json(selected, output_path)

    print(f"[selection] wrote {output_path}")
    print(
        f"[selection] policy={selected['selection_policy']['name']} "
        f"dense_count={selected['dense_validation']['count']} "
        f"smoke_count={selected['smoke']['count']}"
    )


if __name__ == "__main__":
    main()
