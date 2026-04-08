from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_config


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("[curriculum]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _stage_best_checkpoint(stage_cfg: Dict[str, Any], root: Path) -> Path:
    checkpoint_dir = Path(stage_cfg.get("paths", {}).get("checkpoint_dir", "results/default/checkpoints"))
    checkpoint_dir = checkpoint_dir if checkpoint_dir.is_absolute() else (root / checkpoint_dir)
    return checkpoint_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curriculum-by-resolution training stages.")
    parser.add_argument("--config", type=str, required=True, help="Path to the curriculum yaml file.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    curriculum_path = Path(args.config).resolve()
    root = curriculum_path.parents[1]
    config = load_config(curriculum_path)
    curriculum = config.get("curriculum", {})
    stages = curriculum.get("stages", [])
    if not stages:
        raise ValueError("No stages were found in the curriculum config.")

    generate_dataset = bool(curriculum.get("generate_dataset", True))
    resume_from_previous = bool(curriculum.get("resume_from_previous", True))
    run_final_accuracy_eval = bool(curriculum.get("run_final_accuracy_eval", False))

    previous_best: Path | None = None

    for index, stage in enumerate(stages, start=1):
        stage_name = str(stage.get("name", f"stage{index}"))
        stage_cfg_path = (root / str(stage.get("config"))).resolve()
        stage_cfg = load_config(stage_cfg_path)

        print(f"\n=== Curriculum stage {index}: {stage_name} ===")
        print(f"config: {stage_cfg_path}")

        if generate_dataset:
            _run([sys.executable, str(root / "src/data_gen/simulate_dataset.py"), "--config", str(stage_cfg_path)], dry_run=args.dry_run)

        train_cmd = [sys.executable, str(root / "src/training/train.py"), "--config", str(stage_cfg_path)]
        if resume_from_previous and previous_best is not None:
            train_cmd.extend(["--resume", str(previous_best)])
        _run(train_cmd, dry_run=args.dry_run)

        current_best = _stage_best_checkpoint(stage_cfg, root)
        if not args.dry_run and not current_best.exists():
            raise FileNotFoundError(f"Expected best checkpoint was not found: {current_best}")
        previous_best = current_best

    if run_final_accuracy_eval:
        final_stage_cfg = (root / str(stages[-1].get("config"))).resolve()
        _run([sys.executable, str(root / "src/evaluation/eval_accuracy.py"), "--config", str(final_stage_cfg)], dry_run=args.dry_run)

    print("\n[curriculum] Finished all stages.")


if __name__ == "__main__":
    main()
