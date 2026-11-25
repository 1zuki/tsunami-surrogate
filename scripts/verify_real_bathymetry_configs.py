#!/usr/bin/env python
"""Check public GEBCO settings against saved configurations and published fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsunami_surrogate.data_gen.common_time import parse_requested_output_config


def verify(config_root: Path, data_root: Path) -> dict:
    rows = []
    for config in sorted(config_root.glob("*_dataset.yaml")):
        name = config.name.removesuffix("_dataset.yaml")
        current = yaml.safe_load(config.read_text())
        saved = data_root / "configs" / config.name
        snapshot = data_root / "raw" / name / "dataset_config.snapshot.yaml"
        if not (current == yaml.safe_load(saved.read_text())
                == yaml.safe_load(snapshot.read_text())):
            raise ValueError(f"Generation settings differ for {name}")
        requested = parse_requested_output_config(current["requested_output"])
        if requested is None or requested.status != "accepted":
            raise ValueError(f"Missing completed-run output settings for {name}")
        fields = sorted((data_root / "raw" / name / "hydrostatic/samples").glob("*/sample.npz"))
        if len(fields) != current["dataset"]["num_samples"]:
            raise ValueError(f"Incomplete saved fields for {name}")
        for field in fields:
            with np.load(field, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["timestamps"], requested.requested_times)
                if str(data["contract_hash"].item()) != requested.contract_hash:
                    raise ValueError(f"Output settings identity differs for {field}")
                eta = data["trajectory_eta"]
                if eta.shape != (50, 64, 64) or not np.isfinite(eta).all():
                    raise ValueError(f"Invalid published trajectory for {field}")
        preprocess = config.with_name(name + "_preprocess.yaml")
        settings = yaml.safe_load(preprocess.read_text())
        if settings["raw"]["generation_config"] != str(config):
            raise ValueError(f"Preprocessing does not name the public settings for {name}")
        original = yaml.safe_load((data_root / "configs" / preprocess.name).read_text())
        original["raw"]["generation_config"] = str(config)
        if settings != original:
            raise ValueError(f"Preprocessing differs beyond the generation-config path for {name}")
        rows.append({"suite": name, "samples_checked": len(fields),
                     "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                     "saved_config_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                     "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                     "requested_times_identity": requested.contract_hash})
    if len(rows) != 4:
        raise ValueError("Expected four GEBCO evaluation suites")
    return {"status": "verified", "suites": rows,
            "interpretation": "Settings match saved run snapshots and every stored output schedule and field. This check does not regenerate the GEBCO inputs or rerun the solver."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-root", type=Path, default=Path("configs/data/real_bathymetry"))
    p.add_argument("--data-root", type=Path, default=Path("data/real_bathymetry"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.config_root, a.data_root)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Verified four suites and {sum(r['samples_checked'] for r in result['suites'])} trajectories")


if __name__ == "__main__":
    main()
