#!/usr/bin/env python
"""Exercise generation, preprocessing, training, and evaluation on small splits."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsunami_surrogate.utils.config import load_config, save_config
from tsunami_surrogate.utils.hashing import sha256_file

REFERENCES = ("hydrostatic", "muscl_hr", "boussinesq")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _run(root: Path, label: str, script: str, *args: str) -> None:
    log = root / "logs" / f"{label}.log"
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    print(f"[smoke] {label}", flush=True)
    with log.open("w") as output:
        result = subprocess.run([sys.executable, script, *map(str, args)], cwd=ROOT,
                                env=env, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")


def _inventory(root: Path) -> dict[str, str]:
    return {_relative(path): sha256_file(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "logs" not in path.relative_to(root).parts
            and path.name not in {"smoke-report.json", "SHA256SUMS.json"}}


def verify_files(root: Path) -> int:
    inventory = json.loads((root / "SHA256SUMS.json").read_text())
    if not inventory or inventory != _inventory(root):
        raise RuntimeError("Smoke output inventory or SHA-256 differs")
    return len(inventory)


def run_pipeline(root: Path, args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from tsunami_surrogate.data.dataset import ShardedTsunamiDataset
    from tsunami_surrogate.data_gen.common_time import code_state, parse_requested_output_config, validate_publication
    from tsunami_surrogate.models import build_model
    from tsunami_surrogate.training.checkpointing import load_checkpoint

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a new empty smoke directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    configs = root / "configs"
    configs.mkdir()
    processed = root / "processed"
    report = {"purpose": "pipeline smoke test; not manuscript evidence",
              "source_code": code_state(ROOT),
              "horizon": args.horizon, "frames": 50, "references": {},
              "raw_publications": 0, "resume_preserved_hashes": True}
    normalization_hashes = {}
    for split, count, suffix in (("train", args.train_samples, ""),
                                 ("eval", args.eval_samples, "_eval"),
                                 ("test", args.test_samples, "_test")):
        cfg = load_config(ROOT / f"configs/data/dataset{suffix}.yaml")
        split_root = root / split
        cfg["dataset"].update(num_samples=count, num_workers=args.workers,
                              bathymetry_dir=_relative(split_root / "bathymetry"),
                              source_dir=_relative(split_root / "sources"),
                              output_dir=_relative(split_root / "raw"),
                              manifest_path=_relative(split_root / "synthetic/scenario_manifest.jsonl"))
        cfg["paired_inputs"]["inventory_path"] = _relative(split_root / "synthetic/native_input_inventory.jsonl")
        cfg["requested_output"].update(start=args.horizon / 50, step=args.horizon / 50,
                                        horizon=args.horizon, status="provisional",
                                        execution_scope="preparation-only", acknowledge_provisional=True)
        config = configs / f"generate_{split}.yaml"
        save_config(cfg, config)
        _run(root, f"generate-{split}", "scripts/make_dataset.py", "--config", _relative(config))
        requested = parse_requested_output_config(cfg["requested_output"])
        publications = list((split_root / "raw").glob("*/samples/sample_*/publication.json"))
        if len(publications) != count * len(REFERENCES):
            raise RuntimeError(f"Incomplete {split} publications")
        for publication in publications:
            validate_publication(publication.parent, expected_times=requested.requested_times)
            with np.load(publication.parent / "sample.npz", allow_pickle=False) as sample:
                eta = sample["trajectory_eta"]
                if eta.shape != (50, 64, 64) or not np.isfinite(eta).all():
                    raise RuntimeError(f"Invalid smoke trajectory: {publication.parent}")
        report["raw_publications"] += len(publications)
        # Re-enter the real resumable CLI and require existing publication bytes to survive.
        before = {p: sha256_file(p) for pub in publications for p in pub.parent.rglob("*") if p.is_file()}
        _run(root, f"resume-{split}", "scripts/make_dataset.py", "--config", _relative(config), "--continue")
        if any(sha256_file(p) != expected for p, expected in before.items()):
            raise RuntimeError("Resuming generation changed an existing publication")

        prep = load_config(ROOT / f"configs/data/preprocess_{split}.yaml")
        prep["raw"].update(generation_config=_relative(config),
                            scenario_manifest=_relative(split_root / "synthetic/scenario_manifest.jsonl"),
                            fde_manifests={ref: _relative(split_root / f"synthetic/{ref}_manifest.jsonl") for ref in REFERENCES},
                            raw_dirs={ref: _relative(split_root / f"raw/{ref}/samples") for ref in REFERENCES})
        prep["processed_dir"] = _relative(processed)
        if split != "train":
            prep["normalization"]["reference_stats_by_fde"] = {
                ref: _relative(processed / ref / "normalization_stats.json") for ref in REFERENCES}
        prep_path = configs / f"preprocess_{split}.yaml"
        save_config(prep, prep_path)
        _run(root, f"preprocess-{split}", "src/data_gen/preprocess.py", "--config", _relative(prep_path))
        current = {ref: sha256_file(processed / ref / "normalization_stats.json") for ref in REFERENCES}
        if split == "train":
            normalization_hashes = current
        elif current != normalization_hashes:
            raise RuntimeError("Validation/test preprocessing changed training normalization")
        for ref in REFERENCES:
            folder = processed / ref / ("val" if split == "eval" else split)
            ds = ShardedTsunamiDataset(folder)
            if len(ds) != count:
                raise RuntimeError(f"Wrong processed sample count: {folder}")
            for index in range(len(ds)):
                if not torch.isfinite(ds[index]["x"]).all() or not torch.isfinite(ds[index]["y"]).all():
                    raise RuntimeError(f"Nonfinite processed sample: {folder}")

    for ref in REFERENCES:
        cfg = deepcopy(load_config(ROOT / "configs/model/fno.yaml"))
        cfg.update(device=args.device, output_dir=_relative(root / f"models/{ref}/seed_18"))
        cfg["data"].update({f"{split}_path": _relative(processed / ref / split)
                             for split in ("train", "val", "test")})
        cfg["data"].update(batch_size=2, num_workers=0)
        cfg["model"].update(width=8, depth=2, modes1=4, modes2=4)
        cfg["train"].update(epochs=args.epochs)
        cfg["eval"].update(dataset_path=_relative(processed / ref / "test"), batch_size=1,
                            output_dir=_relative(root / f"metrics/{ref}"))
        config = configs / f"model_{ref}.yaml"
        save_config(cfg, config)
        _run(root, f"train-{ref}", "scripts/train.py", "--config", _relative(config))
        checkpoint = ROOT / cfg["output_dir"] / "best.pt"
        model = build_model(cfg)
        loaded = load_checkpoint(checkpoint, model, validate_training_data=True)
        if loaded["compatibility"]["training_data"] != "match":
            raise RuntimeError(f"Checkpoint is not bound to training data: {ref}")
        output = root / f"metrics/{ref}/accuracy.json"
        _run(root, f"evaluate-{ref}", "scripts/eval_accuracy.py", "--config", _relative(config),
             "--checkpoint", _relative(checkpoint), "--output", _relative(output))
        repeat = output.with_name("accuracy-repeat.json")
        _run(root, f"evaluate-repeat-{ref}", "scripts/eval_accuracy.py", "--config", _relative(config),
             "--checkpoint", _relative(checkpoint), "--output", _relative(repeat))
        metrics, repeated = json.loads(output.read_text()), json.loads(repeat.read_text())
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if not math.isfinite(value) or not math.isclose(value, repeated[key], rel_tol=1e-6, abs_tol=1e-8):
                    raise RuntimeError(f"Nonfinite or non-repeatable evaluation metric: {ref}/{key}")
        if metrics["num_samples"] != args.test_samples:
            raise RuntimeError(f"Wrong evaluation sample count: {ref}")
        report["references"][ref] = {"checkpoint_sha256": sha256_file(checkpoint),
                                      "training_data_identity": loaded["compatibility"]["training_data"],
                                      "normalization_sha256": normalization_hashes[ref], "metrics": metrics}

    # Corrupt a copy: verify the real reader rejects tampering without touching test data.
    source = processed / "hydrostatic/test"
    probe_root = root / "corruption-probe"
    probe = probe_root / "test"
    shutil.copytree(source, probe)
    shutil.copyfile(source.parent / "normalization_stats.json", probe_root / "normalization_stats.json")
    try:
        manifest = json.loads((probe / "shards_manifest.json").read_text())
        shard = probe / manifest["shards"][0]["file"]
        with shard.open("ab") as handle:
            handle.write(b"smoke-corruption-probe")
        try:
            ShardedTsunamiDataset(probe)[0]
        except RuntimeError as error:
            if "hash mismatch" not in str(error):
                raise
        else:
            raise RuntimeError("The reader accepted a corrupted processed shard")
    finally:
        # Only remove the small generated probe files listed by this command.
        for path in sorted(probe_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        probe_root.rmdir()
    report["corrupted_shard_rejected"] = True
    (root / "SHA256SUMS.json").write_text(json.dumps(_inventory(root), indent=2) + "\n")
    report["verified_files"] = verify_files(root)
    report["status"] = "passed"
    (root / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/smoke"))
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=8.4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = (ROOT / args.output_root).resolve()
    if root != ROOT / "data/smoke" and ROOT / "data/smoke" not in root.parents:
        parser.error("Use data/smoke or a new subdirectory beneath it")
    if min(args.train_samples, args.eval_samples, args.test_samples, args.epochs, args.workers) <= 0:
        parser.error("Sample counts, epochs, and workers must be positive")
    if not math.isfinite(args.horizon) or args.horizon <= 0:
        parser.error("horizon must be positive and finite")
    if args.verify_only:
        print(f"Verified {verify_files(root)} files")
    else:
        report = run_pipeline(root, args)
        print(f"Smoke pipeline passed: {report['raw_publications']} publications, "
              f"{len(report['references'])} trained/evaluated models, {report['verified_files']} verified files")
        print(root / "smoke-report.json")


if __name__ == "__main__":
    main()
