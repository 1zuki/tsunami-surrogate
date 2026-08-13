#!/usr/bin/env python
"""Export a checksum-bound index of accepted common-time-v2 numerical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_suite_preflight import load_suite_contract
from src.utils.io import save_json


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/eval/final_v2_suite.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = load_suite_contract(args.contract)
    rows = []
    for entry in contract.get("accepted_numerical_artifacts", []):
        if not isinstance(entry, Mapping):
            raise TypeError("accepted_numerical_artifacts rows must be mappings")
        root = ROOT / str(entry["root"])
        decision_path = root / str(entry["decision_path"])
        decision = _read_object(decision_path)
        observed = decision.get(str(entry["decision_key"]))
        expected = entry["expected_decision"]
        if observed != expected:
            raise ValueError(
                f"Numerical evidence decision mismatch for {entry['id']}: "
                f"{observed!r} != {expected!r}"
            )
        checksums = []
        for relative in entry.get("checksum_files", []):
            path = root / str(relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            checksums.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                }
            )
        rows.append(
            {
                "id": str(entry["id"]),
                "root": str(root.relative_to(ROOT)),
                "decision_path": str(decision_path.relative_to(ROOT)),
                "decision_sha256": _sha256(decision_path),
                "decision_key": str(entry["decision_key"]),
                "decision": observed,
                "checksum_files": checksums,
            }
        )

    result = {
        "evaluation_type": "v2_numerical_evidence_index",
        "contract_path": str(args.contract),
        "contract_sha256": _sha256(ROOT / args.contract),
        "rows": rows,
    }
    save_json(result, args.output)
    print(f"[v2-numerical-evidence] rows={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
