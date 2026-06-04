#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import get_git_commit, save_json

DEFAULT_MODEL_CONFIGS = "configs/model/fno.yaml,configs/model/fno_muscl_hr.yaml,configs/model/fno_boussinesq.yaml"
DEFAULT_SOLVER_COMPARE_PAIRS = (
    "hydrostatic:muscl_hr,muscl_hr:hydrostatic,"
    "hydrostatic:boussinesq,boussinesq:hydrostatic,"
    "muscl_hr:boussinesq,boussinesq:muscl_hr"
)
DEFAULT_EMULATOR_CONFIGS = (
    "configs/eval/emulator_superiority_hydro_to_muscl_hr.yaml,"
    "configs/eval/emulator_superiority_muscl_hr_to_hydro.yaml"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return _parse_bool(value)


def _parse_bool(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, env_name: str, default: bool, help_text: str) -> None:
    parser.add_argument(name, type=_parse_bool, default=_env_bool(env_name, default), help=help_text)


def _split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _split_ints(value: str) -> list[int]:
    return [int(part) for part in _split_csv(value)]


def _command_text(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _write_stage_csv(stages: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "name",
        "status",
        "optional",
        "exit_code",
        "started_at",
        "ended_at",
        "duration_s",
        "log_path",
        "command",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stage in stages:
            writer.writerow({key: stage.get(key, "") for key in fieldnames})


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_root = Path(args.run_root)
        self.logs_dir = Path(args.logs_dir) if args.logs_dir else self.run_root / "logs"
        self.metrics_dir = Path(args.metrics_dir) if args.metrics_dir else self.run_root / "metrics"
        self.pipeline_json = self.metrics_dir / "full_pipeline_results.json"
        self.pipeline_csv = self.metrics_dir / "full_pipeline_stages.csv"
        self.stages: list[dict[str, Any]] = []
        self.metric_files: list[dict[str, str]] = []
        self.outputs: dict[str, Any] = {}

        self.run_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self) -> None:
        payload = {
            "created_at": self.outputs.get("created_at"),
            "updated_at": _utc_now(),
            "git_commit": get_git_commit(),
            "run_root": str(self.run_root),
            "logs_dir": str(self.logs_dir),
            "metrics_dir": str(self.metrics_dir),
            "dry_run": bool(self.args.dry_run),
            "outputs": self.outputs,
            "metric_files": self.metric_files,
            "stages": self.stages,
        }
        save_json(payload, self.pipeline_json)
        _write_stage_csv(self.stages, self.pipeline_csv)

    def skip_stage(self, name: str, reason: str, optional: bool = True) -> None:
        print(f"[pipeline][skip] {name}: {reason}", flush=True)
        self.stages.append(
            {
                "name": name,
                "status": "skipped",
                "optional": bool(optional),
                "reason": reason,
                "started_at": _utc_now(),
                "ended_at": _utc_now(),
            }
        )
        self.write_manifest()

    def run_stage(self, name: str, cmd: list[str], optional: bool = False) -> bool:
        log_path = self.logs_dir / f"{name}.log"
        stage = {
            "name": name,
            "status": "pending",
            "optional": bool(optional),
            "command": _command_text(cmd),
            "log_path": str(log_path),
            "started_at": _utc_now(),
        }
        self.stages.append(stage)
        self.write_manifest()

        print(f"[pipeline] {name}: $ {_command_text(cmd)}", flush=True)
        if self.args.dry_run:
            stage.update({"status": "dry_run", "exit_code": 0, "ended_at": _utc_now(), "duration_s": 0.0})
            self.write_manifest()
            return True

        start = time.time()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
            code = proc.wait()

        stage.update(
            {
                "status": "ok" if code == 0 else "failed",
                "exit_code": code,
                "ended_at": _utc_now(),
                "duration_s": round(time.time() - start, 3),
            }
        )
        self.write_manifest()
        if code != 0:
            print(f"[pipeline][error] {name} failed with exit code {code}", file=sys.stderr, flush=True)
            if optional and self.args.continue_on_optional_error:
                return False
            raise SystemExit(code)
        return True

    def collect_metric(self, source: Path | str, relative_dest: str | None = None) -> None:
        src = Path(source)
        rel = Path(relative_dest) if relative_dest else Path(src.name)
        dest = self.metrics_dir / rel
        record = {"source": str(src), "dest": str(dest)}

        if self.args.dry_run:
            record["status"] = "planned"
            self.metric_files.append(record)
            self.write_manifest()
            return

        if not src.exists():
            record["status"] = "missing"
            self.metric_files.append(record)
            self.write_manifest()
            print(f"[metrics][skip] missing {src}", flush=True)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        record["status"] = "copied"
        self.metric_files.append(record)
        self.write_manifest()
        print(f"[metrics] {src} -> {dest}", flush=True)


def _data_command(args: argparse.Namespace, config: str, samples: int | None = None) -> list[str]:
    cmd = [args.python, "scripts/make_dataset.py", "--config", config]
    if samples is not None:
        cmd.extend(["--num-samples", str(samples)])
    if args.n_steps is not None:
        cmd.extend(["--n-steps", str(args.n_steps)])
    if args.save_every is not None:
        cmd.extend(["--save-every", str(args.save_every)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.dataset_continue:
        cmd.append("--continue")
    if args.allow_override:
        cmd.append("--allow-override")
    return cmd


def _model_output_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("output_dir", "experiments/default")))


def _eval_output_dir(cfg: dict[str, Any]) -> Path:
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {})) or {}
    output_dir = str(eval_cfg.get("output_dir", "")).strip()
    if not output_dir or output_dir == "experiments/eval":
        output_dir = f"{cfg.get('output_dir', 'experiments/default')}/eval"
    return Path(output_dir)


def _checkpoint_for(model_cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    return _model_output_dir(model_cfg) / "best.pt"


def _target_from_model_config(config_path: str, cfg: dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    haystack = " ".join(
        str(value)
        for value in [
            config_path,
            cfg.get("output_dir", ""),
            data_cfg.get("train_path", ""),
            data_cfg.get("test_path", ""),
            eval_cfg.get("dataset_path", ""),
        ]
    ).lower()
    if "bouss" in haystack:
        return "boussinesq"
    if "muscl" in haystack:
        return "muscl_hr"
    return "hydrostatic"


def _model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = _split_csv(args.model_configs) or [args.model_config]
    checkpoints = _split_csv(args.checkpoints)
    specs: list[dict[str, Any]] = []
    for idx, config_path in enumerate(configs):
        cfg = load_config(config_path)
        output_dir = _model_output_dir(cfg)
        eval_dir = _eval_output_dir(cfg)
        override_checkpoint = checkpoints[idx] if idx < len(checkpoints) else None
        if override_checkpoint is None and len(configs) == 1 and args.checkpoint:
            override_checkpoint = args.checkpoint
        label = _safe_name(output_dir.name or Path(config_path).stem)
        target = _target_from_model_config(config_path, cfg)
        specs.append(
            {
                "config": config_path,
                "cfg": cfg,
                "label": label,
                "target": target,
                "output_dir": output_dir,
                "eval_dir": eval_dir,
                "checkpoint": _checkpoint_for(cfg, override_checkpoint),
            }
        )
    return specs


def _maybe_add_device(cmd: list[str], device: str) -> list[str]:
    if device:
        cmd.extend(["--device", device])
    return cmd


def _all_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _target_suffix(target: str) -> str:
    return {"hydrostatic": "hydrostatic", "muscl_hr": "muscl_hr", "boussinesq": "boussinesq"}[target]


def _short_solver_label(target: str) -> str:
    return {"hydrostatic": "hydro", "muscl_hr": "muscl_hr", "boussinesq": "boussinesq"}.get(target, _safe_name(target))


def _raw_dir_for_target(target: str) -> Path:
    return Path("data/raw") / target / "samples"


def _suite_config(kind: str, target: str) -> Path:
    suffix = _target_suffix(target)
    if kind == "ood_splits":
        return Path(f"configs/data/ood_splits_{suffix}.yaml")
    if kind == "ood_eval":
        return Path(f"configs/eval/ood_suites_{suffix}.yaml")
    if kind == "resolution_proxy":
        return Path(f"configs/eval/resolution_transfer_proxy_{suffix}.yaml")
    if kind == "true_resolution":
        return Path(f"configs/eval/resolution_{suffix}.yaml")
    raise ValueError(f"unknown suite config kind: {kind}")


def _stage_metric_output_dir(config_path: str | Path) -> Path:
    return _eval_output_dir(load_config(config_path))


def _true_res_suite_paths(config_path: str | Path) -> list[Path]:
    cfg = load_config(config_path)
    suites = cfg.get("eval", {}).get("real_resolution", {}).get("suites", [])
    paths = []
    for suite in suites:
        if isinstance(suite, dict) and suite.get("path"):
            paths.append(Path(str(suite["path"])))
    return paths


def _emulator_output_path(config_path: str) -> Path:
    cfg = load_config(config_path)
    return Path(str(cfg.get("output_path", "results/emulator_superiority.json")))


def _emulator_solver_compare_path(config_path: str) -> Path:
    cfg = load_config(config_path)
    return Path(str(cfg.get("solver_compare_path", "")))


def _solver_compare_pairs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    pairs: list[tuple[str, str, Path]] = []
    raw_pairs = _split_csv(args.solver_compare_pairs)
    if not raw_pairs:
        return [("custom_a", "custom_b", Path(args.solver_compare_output))]
    for raw_pair in raw_pairs:
        if ":" not in raw_pair:
            raise ValueError(f"Invalid solver pair {raw_pair!r}; expected left:right")
        left, right = [part.strip() for part in raw_pair.split(":", 1)]
        left_label = _short_solver_label(left)
        right_label = _short_solver_label(right)
        if left_label == "hydro" and right_label == "muscl_hr":
            out = Path(args.solver_compare_output)
        elif left_label == "muscl_hr" and right_label == "hydro":
            out = Path("results/solver_compare_muscl_hr_vs_hydro.json")
        else:
            out = Path("results") / f"solver_compare_{left_label}_vs_{right_label}.json"
        pairs.append((left, right, out))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin wrapper around the README tsunami surrogate pipeline.")
    parser.add_argument("--python", default=_env("PYTHON_BIN", sys.executable))
    parser.add_argument("--run-root", default=_env("RUN_ROOT", f"experiments/cloudrun/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--logs-dir", default=_env("LOG_DIR", ""), help="Console/stage log directory. Defaults to RUN_ROOT/logs.")
    parser.add_argument("--metrics-dir", default=_env("METRICS_DIR", ""), help="Copied metrics directory. Defaults to RUN_ROOT/metrics.")

    parser.add_argument("--data-config", default=_env("DATA_CONFIG", "configs/data/dataset.yaml"))
    parser.add_argument("--preprocess-config", default=_env("PREPROCESS_CONFIG", "configs/data/preprocess.yaml"))
    parser.add_argument("--model-config", default=_env("MODEL_CONFIG", "configs/model/fno.yaml"))
    parser.add_argument("--model-configs", default=_env("MODEL_CONFIGS", DEFAULT_MODEL_CONFIGS))
    parser.add_argument("--checkpoint", default=_env("CHECKPOINT", ""))
    parser.add_argument("--checkpoints", default=_env("CHECKPOINTS", ""), help="Comma-separated checkpoints matching --model-configs.")
    parser.add_argument("--dataset-samples", type=int, default=_env_optional_int("DATASET_SAMPLES"))
    parser.add_argument("--n-steps", type=int, default=_env_optional_int("N_STEPS"))
    parser.add_argument("--save-every", type=int, default=_env_optional_int("SAVE_EVERY"))
    parser.add_argument("--num-workers", type=int, default=_env_optional_int("NUM_WORKERS"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=_env("DEVICE", "auto"))
    parser.add_argument("--allow-tf32", default=_env("ALLOW_TF32", ""), help="Optional eval_speed TF32 override: true/false.")
    _add_bool_arg(parser, "--dataset-continue", "DATASET_CONTINUE", True, "Resume/skip completed raw samples during dataset generation.")
    _add_bool_arg(parser, "--allow-override", "ALLOW_OVERRIDE", False, "Regenerate existing raw sample folders instead of reusing them.")

    parser.add_argument("--true-resolutions", default=_env("TRUE_RESOLUTIONS", "32,64,128"))
    parser.add_argument("--true-res-samples", type=int, default=_env_optional_int("TRUE_RES_SAMPLES"))
    parser.add_argument("--true-res-shared-from64", type=_parse_bool, default=_env_bool("TRUE_RES_SHARED_FROM64", False))
    parser.add_argument("--sample-sizes", default=_env("SAMPLE_SIZES", "8,16,32,64,128"))
    parser.add_argument("--sample-scaling-output-root", default=_env("SAMPLE_SCALING_OUTPUT_ROOT", "experiments/sample_scaling"))

    parser.add_argument("--solver-compare-pairs", default=_env("SOLVER_COMPARE_PAIRS", DEFAULT_SOLVER_COMPARE_PAIRS))
    parser.add_argument("--solver-a-dir", default=_env("SOLVER_A_DIR", "data/raw/hydrostatic/samples"))
    parser.add_argument("--solver-b-dir", default=_env("SOLVER_B_DIR", "data/raw/muscl_hr/samples"))
    parser.add_argument("--solver-compare-output", default=_env("SOLVER_COMPARE_OUTPUT", "results/solver_compare_hydro_vs_muscl_hr.json"))
    parser.add_argument("--solver-compare-max-samples", type=int, default=_env_optional_int("SOLVER_COMPARE_MAX_SAMPLES"))
    parser.add_argument("--solver-speed-solvers", default=_env("SOLVER_SPEED_SOLVERS", "swe_hydrostatic,swe_muscl_hr,boussinesq"))
    parser.add_argument("--solver-speed-max-samples", type=int, default=_env_optional_int("SOLVER_SPEED_MAX_SAMPLES"))
    parser.add_argument("--solver-speed-repeats", type=int, default=_env_optional_int("SOLVER_SPEED_REPEATS"))
    parser.add_argument("--speed-warmup", type=int, default=_env_optional_int("SPEED_WARMUP"))
    parser.add_argument("--speed-repeats", type=int, default=_env_optional_int("SPEED_REPEATS"))
    parser.add_argument("--speed-max-batches", type=int, default=_env_optional_int("SPEED_MAX_BATCHES"))
    parser.add_argument("--emulator-configs", default=_env("EMULATOR_CONFIGS", DEFAULT_EMULATOR_CONFIGS))

    _add_bool_arg(parser, "--run-dataset", "RUN_DATASET", True, "Run raw dataset generation.")
    _add_bool_arg(parser, "--run-preprocess", "RUN_PREPROCESS", True, "Run preprocessing.")
    _add_bool_arg(parser, "--run-true-res-generation", "RUN_TRUE_RES_GENERATION", False, "Generate/preprocess native-resolution datasets.")
    _add_bool_arg(parser, "--run-train", "RUN_TRAIN", True, "Train all configured models.")
    _add_bool_arg(parser, "--run-eval-accuracy", "RUN_EVAL_ACCURACY", True, "Run accuracy evaluation.")
    _add_bool_arg(parser, "--run-eval-speed", "RUN_EVAL_SPEED", True, "Run model speed benchmark.")
    _add_bool_arg(parser, "--run-arrival", "RUN_ARRIVAL", True, "Run arrival-map evaluation.")
    _add_bool_arg(parser, "--run-ood", "RUN_OOD", True, "Build OOD splits and evaluate them.")
    _add_bool_arg(parser, "--run-resolution-proxy", "RUN_RESOLUTION_PROXY", True, "Run proxy/downsample resolution benchmark.")
    _add_bool_arg(parser, "--run-true-res-eval", "RUN_TRUE_RES_EVAL", False, "Run native true-resolution benchmark if data exists.")
    _add_bool_arg(parser, "--run-solver-compare", "RUN_SOLVER_COMPARE", True, "Compare raw solver outputs.")
    _add_bool_arg(parser, "--run-solver-speed", "RUN_SOLVER_SPEED", True, "Benchmark reference solver speed.")
    _add_bool_arg(parser, "--run-speed-table", "RUN_SPEED_TABLE", True, "Aggregate model/solver speed table.")
    _add_bool_arg(parser, "--run-emulator-superiority", "RUN_EMULATOR_SUPERIORITY", True, "Compute emulator-vs-solver superiority ratios.")
    _add_bool_arg(parser, "--run-sample-scaling", "RUN_SAMPLE_SCALING", False, "Run n-sample learning curve sweeps.")
    _add_bool_arg(parser, "--dry-run", "DRY_RUN", False, "Write logs/manifest but do not execute commands.")
    _add_bool_arg(parser, "--continue-on-optional-error", "CONTINUE_ON_OPTIONAL_ERROR", True, "Continue after optional stage failures.")
    _add_bool_arg(parser, "--skip-missing-optional", "SKIP_MISSING_OPTIONAL", True, "Skip optional stages whose input files are missing.")
    return parser


def _run_model_stages(args: argparse.Namespace, pipe: Pipeline, specs: list[dict[str, Any]]) -> list[Path]:
    model_speed_paths: list[Path] = []
    built_ood_targets: dict[str, bool] = {}

    for spec in specs:
        label = str(spec["label"])
        target = str(spec["target"])
        model_config = str(spec["config"])
        checkpoint = Path(spec["checkpoint"])
        eval_dir = Path(spec["eval_dir"])

        if args.run_train:
            pipe.run_stage(f"train_{label}", [args.python, "scripts/train.py", "--config", model_config])
        else:
            pipe.skip_stage(f"train_{label}", "RUN_TRAIN=false", optional=False)

        if args.run_eval_accuracy:
            pipe.run_stage(
                f"eval_accuracy_{label}",
                _maybe_add_device([args.python, "scripts/eval_accuracy.py", "--config", model_config, "--checkpoint", str(checkpoint)], args.device),
            )
            pipe.collect_metric(eval_dir / "metrics.json", f"model_eval/{label}/metrics.json")

        if args.run_eval_speed:
            model_speed_path = Path("results/speed") / f"model_speed_{label}_{_safe_name(args.device)}.json"
            cmd = [
                args.python,
                "scripts/eval_speed.py",
                "--config",
                model_config,
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--precision",
                "fp32",
                "--output",
                str(model_speed_path),
            ]
            if args.allow_tf32:
                cmd.extend(["--allow-tf32", args.allow_tf32])
            if args.speed_warmup is not None:
                cmd.extend(["--warmup", str(args.speed_warmup)])
            if args.speed_repeats is not None:
                cmd.extend(["--repeats", str(args.speed_repeats)])
            if args.speed_max_batches is not None:
                cmd.extend(["--max-batches", str(args.speed_max_batches)])
            if pipe.run_stage(f"eval_model_speed_{label}", cmd, optional=True):
                model_speed_paths.append(model_speed_path)
                pipe.collect_metric(model_speed_path, f"speed/{model_speed_path.name}")

        if args.run_arrival:
            arrival_ok = pipe.run_stage(
                f"eval_arrival_maps_{label}",
                _maybe_add_device([args.python, "scripts/eval_arrival_maps.py", "--config", model_config, "--checkpoint", str(checkpoint)], args.device),
                optional=True,
            )
            if arrival_ok:
                pipe.collect_metric(eval_dir / "arrival_map_model_vs_target.json", f"model_eval/{label}/arrival_map_model_vs_target.json")
                pipe.collect_metric(eval_dir / "arrival_map_model_vs_target.npz", f"model_eval/{label}/arrival_map_model_vs_target.npz")

        if args.run_ood:
            split_cfg = _suite_config("ood_splits", target)
            eval_cfg = _suite_config("ood_eval", target)
            if not split_cfg.exists() or not eval_cfg.exists():
                pipe.skip_stage(f"ood_eval_{label}", f"missing OOD configs for target={target}")
            else:
                if target not in built_ood_targets:
                    built_ood_targets[target] = pipe.run_stage(
                        f"ood_splits_{target}",
                        [args.python, "scripts/make_ood_splits.py", "--config", str(split_cfg), "--overwrite"],
                        optional=True,
                    )
                if built_ood_targets[target]:
                    ood_output_dir = _stage_metric_output_dir(eval_cfg)
                    ood_ok = pipe.run_stage(
                        f"ood_eval_{label}",
                        _maybe_add_device([args.python, "scripts/eval_generalization.py", "--config", str(eval_cfg), "--checkpoint", str(checkpoint)], args.device),
                        optional=True,
                    )
                    if ood_ok:
                        pipe.collect_metric(ood_output_dir / "ood_generalization.json", f"ood/{label}/ood_generalization.json")
                else:
                    pipe.skip_stage(f"ood_eval_{label}", f"OOD split generation failed for target={target}")

        if args.run_resolution_proxy:
            proxy_cfg = _suite_config("resolution_proxy", target)
            if not proxy_cfg.exists():
                pipe.skip_stage(f"resolution_proxy_{label}", f"missing resolution proxy config for target={target}")
            else:
                proxy_output_dir = _stage_metric_output_dir(proxy_cfg)
                proxy_ok = pipe.run_stage(
                    f"resolution_proxy_{label}",
                    _maybe_add_device([args.python, "scripts/eval_resolution_transfer.py", "--config", str(proxy_cfg), "--checkpoint", str(checkpoint)], args.device),
                    optional=True,
                )
                if proxy_ok:
                    pipe.collect_metric(proxy_output_dir / "resolution_transfer_proxy.json", f"resolution/{label}/resolution_transfer_proxy.json")

        if args.run_true_res_eval:
            true_cfg = _suite_config("true_resolution", target)
            if not true_cfg.exists():
                pipe.skip_stage(f"true_resolution_eval_{label}", f"missing true-resolution config for target={target}")
            else:
                required_paths = _true_res_suite_paths(true_cfg)
                if required_paths and not _all_exist(required_paths) and args.skip_missing_optional and not args.dry_run:
                    missing = [str(path) for path in required_paths if not path.exists()]
                    pipe.skip_stage(f"true_resolution_eval_{label}", f"missing native-resolution suite paths: {missing}")
                else:
                    true_output_dir = _stage_metric_output_dir(true_cfg)
                    true_ok = pipe.run_stage(
                        f"true_resolution_eval_{label}",
                        _maybe_add_device([args.python, "scripts/eval_full_resolution.py", "--config", str(true_cfg), "--checkpoint", str(checkpoint)], args.device),
                        optional=True,
                    )
                    if true_ok:
                        pipe.collect_metric(true_output_dir / "real_resolution.json", f"resolution/{label}/real_resolution.json")

    return model_speed_paths


def main() -> None:
    args = build_parser().parse_args()
    pipe = Pipeline(args)
    specs = _model_specs(args)
    pipe.outputs.update(
        {
            "created_at": _utc_now(),
            "data_config": args.data_config,
            "preprocess_config": args.preprocess_config,
            "model_configs": [spec["config"] for spec in specs],
            "normal_outputs_policy": "Use configs in configs/ directly; cloud run stores only logs plus copied metrics.",
        }
    )
    pipe.write_manifest()

    if args.run_dataset:
        pipe.run_stage("dataset_generate", _data_command(args, args.data_config, args.dataset_samples))
    else:
        pipe.skip_stage("dataset_generate", "RUN_DATASET=false", optional=False)

    if args.run_preprocess:
        pipe.run_stage("preprocess", [args.python, "src/data_gen/preprocess.py", "--config", args.preprocess_config])
    else:
        pipe.skip_stage("preprocess", "RUN_PREPROCESS=false", optional=False)

    if args.run_true_res_generation:
        resolutions = _split_ints(args.true_resolutions)
        for res in resolutions:
            dataset_cfg = f"configs/data/multires/dataset_{res}.yaml"
            if Path(dataset_cfg).exists():
                pipe.run_stage(f"true_res_dataset_{res}", _data_command(args, dataset_cfg, args.true_res_samples), optional=True)
            else:
                pipe.skip_stage(f"true_res_dataset_{res}", f"missing config {dataset_cfg}")

        if args.true_res_shared_from64 and Path("configs/data/multires/preprocess_64.yaml").exists():
            pipe.run_stage(
                "true_res_preprocess_64_reference_stats",
                [args.python, "src/data_gen/preprocess.py", "--config", "configs/data/multires/preprocess_64.yaml"],
                optional=True,
            )
        for res in resolutions:
            suffix = "_shared_from64" if args.true_res_shared_from64 else ""
            preprocess_cfg = f"configs/data/multires/preprocess_{res}{suffix}.yaml"
            if Path(preprocess_cfg).exists():
                pipe.run_stage(f"true_res_preprocess_{res}", [args.python, "src/data_gen/preprocess.py", "--config", preprocess_cfg], optional=True)
            else:
                pipe.skip_stage(f"true_res_preprocess_{res}", f"missing config {preprocess_cfg}")
    else:
        pipe.skip_stage("true_res_generation", "RUN_TRUE_RES_GENERATION=false")

    model_speed_paths = _run_model_stages(args, pipe, specs)

    solver_compare_outputs: list[Path] = []
    if args.run_solver_compare:
        for left, right, out in _solver_compare_pairs(args):
            if left == "custom_a":
                left_dir = Path(args.solver_a_dir)
                right_dir = Path(args.solver_b_dir)
                stage_label = "custom"
            else:
                left_dir = _raw_dir_for_target(left)
                right_dir = _raw_dir_for_target(right)
                stage_label = f"{_short_solver_label(left)}_vs_{_short_solver_label(right)}"
            required = [left_dir, right_dir]
            if not _all_exist(required) and args.skip_missing_optional and not args.dry_run:
                missing = [str(path) for path in required if not path.exists()]
                pipe.skip_stage(f"solver_compare_{stage_label}", f"missing solver raw dirs: {missing}")
                continue
            cmd = [
                args.python,
                "scripts/compare_solvers_physical.py",
                "--solver-a-dir",
                str(left_dir),
                "--solver-b-dir",
                str(right_dir),
                "--require-quality-ok",
                "--missing-quality-action",
                "include",
                "--save-arrival-maps",
                "--output",
                str(out),
            ]
            if args.solver_compare_max_samples is not None:
                cmd.extend(["--max-samples", str(args.solver_compare_max_samples)])
            if pipe.run_stage(f"solver_compare_{stage_label}", cmd, optional=True):
                solver_compare_outputs.append(out)
                pipe.collect_metric(out, f"solver_compare/{out.name}")
                pipe.collect_metric(out.with_name(out.stem + "_arrival_maps.npz"), f"solver_compare/{out.stem}_arrival_maps.npz")

    solver_speed_paths: list[Path] = []
    if args.run_solver_speed:
        for solver in _split_csv(args.solver_speed_solvers):
            out = Path("results/speed") / f"solver_speed_{solver}.json"
            cmd = [args.python, "scripts/eval_solver_speed.py", "--config", args.data_config, "--solver", solver, "--device", "cpu", "--output", str(out)]
            if args.solver_speed_max_samples is not None:
                cmd.extend(["--max-samples", str(args.solver_speed_max_samples)])
            if args.n_steps is not None:
                cmd.extend(["--n-steps", str(args.n_steps)])
            if args.solver_speed_repeats is not None:
                cmd.extend(["--repeats", str(args.solver_speed_repeats)])
            if pipe.run_stage(f"solver_speed_{solver}", cmd, optional=True):
                solver_speed_paths.append(out)
                pipe.collect_metric(out, f"speed/{out.name}")

    if args.run_speed_table:
        speed_table_csv = Path("results/speed/speed_table.csv")
        speed_table_json = Path("results/speed/speed_table.json")
        if (not solver_speed_paths or not model_speed_paths) and args.skip_missing_optional:
            pipe.skip_stage("speed_table", "missing model or solver speed JSON inputs")
        else:
            cmd = [args.python, "scripts/make_speed_table.py"]
            for path in solver_speed_paths:
                cmd.extend(["--solver", str(path)])
            for path in model_speed_paths:
                cmd.extend(["--model", str(path)])
            cmd.extend(["--output", str(speed_table_csv), "--output-json", str(speed_table_json)])
            if pipe.run_stage("speed_table", cmd, optional=True):
                pipe.collect_metric(speed_table_csv, "speed/speed_table.csv")
                pipe.collect_metric(speed_table_json, "speed/speed_table.json")

    if args.run_emulator_superiority:
        for emulator_config in _split_csv(args.emulator_configs):
            emulator_path = Path(emulator_config)
            label = _safe_name(emulator_path.stem)
            if not emulator_path.exists():
                pipe.skip_stage(f"emulator_superiority_{label}", f"missing config {emulator_config}")
                continue
            emulator_solver_compare = _emulator_solver_compare_path(emulator_config)
            emulator_output = _emulator_output_path(emulator_config)
            if not emulator_solver_compare.exists() and args.skip_missing_optional and not args.dry_run:
                pipe.skip_stage(f"emulator_superiority_{label}", f"missing solver comparison JSON: {emulator_solver_compare}")
                continue
            emu_ok = pipe.run_stage(
                f"emulator_superiority_{label}",
                _maybe_add_device([args.python, "scripts/eval_emulator_superiority.py", "--config", emulator_config], args.device),
                optional=True,
            )
            if emu_ok:
                pipe.collect_metric(emulator_output, f"solver_compare/{emulator_output.name}")

    if args.run_sample_scaling:
        output_root = Path(args.sample_scaling_output_root)
        for spec in specs:
            label = str(spec["label"])
            model_config = str(spec["config"])
            model_output_root = output_root / label
            sample_scaling_ok = pipe.run_stage(
                f"sample_scaling_{label}",
                [
                    args.python,
                    "scripts/run_sample_scaling.py",
                    "--config",
                    model_config,
                    "--samples",
                    args.sample_sizes,
                    "--output-root",
                    str(model_output_root),
                    "--device",
                    args.device,
                ],
                optional=True,
            )
            if sample_scaling_ok:
                pipe.collect_metric(model_output_root / "sample_scaling_results.json", f"sample_scaling/{label}/sample_scaling_results.json")
                pipe.collect_metric(model_output_root / "sample_scaling_results.csv", f"sample_scaling/{label}/sample_scaling_results.csv")

    pipe.write_manifest()
    print(f"[pipeline] wrote {pipe.pipeline_json}")
    print(f"[pipeline] wrote {pipe.pipeline_csv}")


if __name__ == "__main__":
    main()
