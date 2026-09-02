#!/usr/bin/env bash
# Fail-closed common-time-v2 evaluation runner.
#
# Default invocation performs read-only preflight. Evaluation requires
# --execute and a new immutable --run-id. All outputs stay beneath
# evaluation_runs/<run-id>.staging until consolidation validates every declared
# result; only then is the directory atomically promoted.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
CONTRACT="configs/eval/final_v2_suite.yaml"
EXECUTE=0
RUN_ID=""
INCLUDE_ENSEMBLE=0
INCLUDE_REAL_BATHYMETRY=1
INCLUDE_SPEED=0
INCLUDE_PAPER_EVIDENCE=0
RERUN_NUMERICAL_VALIDATION=0
DEEP_PAYLOAD_AUDIT=0
NUMERICAL_WORKERS="${NUMERICAL_WORKERS:-8}"
GEOCLAW_WORKERS="${GEOCLAW_WORKERS:-4}"
CLAW_ROOT="${CLAW_ROOT:-/home/izu/opt/clawpack-v5.14.0}"
PETSC_DIR="${PETSC_DIR:-/home/izu/opt/petsc-3.25.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt}"
GEOCLAW_PYTHON="${GEOCLAW_PYTHON:-$PY}"

usage() {
  sed -n '1,10p' "$0"
  cat <<'EOF'

Usage:
  bash scripts/run_eval_suite.sh
  bash scripts/run_eval_suite.sh --execute --run-id <immutable-id>

Options:
  --no-real-bathymetry       Skip the rebuilt v2 auxiliary suites.
  --include-ensemble         Require and evaluate all seven frozen members.
  --include-speed            Include model and production-matched solver timing.
  --include-paper-evidence   Regenerate all currently supported v2 paper metrics.
                             Implies --include-ensemble.
  --rerun-numerical-validation
                             Run a fresh isolated H0/A/B/H1/H2 regression chain.
  --deep-payload-audit       Re-hash/reopen all raw v2 payloads during preflight.
  --device cpu|cuda|auto     Override DEVICE for neural-model evaluation.

Numerical rerun environment overrides:
  NUMERICAL_WORKERS, GEOCLAW_WORKERS, CLAW_ROOT, PETSC_DIR, PETSC_ARCH,
  GEOCLAW_PYTHON
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --execute)
      EXECUTE=1
      ;;
    --run-id)
      shift
      RUN_ID="${1:-}"
      ;;
    --include-ensemble)
      INCLUDE_ENSEMBLE=1
      ;;
    --no-real-bathymetry)
      INCLUDE_REAL_BATHYMETRY=0
      ;;
    --include-speed)
      INCLUDE_SPEED=1
      ;;
    --include-paper-evidence)
      INCLUDE_PAPER_EVIDENCE=1
      INCLUDE_ENSEMBLE=1
      ;;
    --rerun-numerical-validation)
      RERUN_NUMERICAL_VALIDATION=1
      ;;
    --deep-payload-audit)
      DEEP_PAYLOAD_AUDIT=1
      ;;
    --device)
      shift
      DEVICE="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ "$DEVICE" != "cpu" ] && [ "$DEVICE" != "cuda" ] && [ "$DEVICE" != "auto" ]; then
  echo "--device must be cpu, cuda, or auto" >&2
  exit 2
fi
if ! [[ "$NUMERICAL_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUMERICAL_WORKERS must be a positive integer" >&2
  exit 2
fi
if [ "$NUMERICAL_WORKERS" != 8 ]; then
  echo "NUMERICAL_WORKERS must remain 8 under the frozen validation contracts" >&2
  exit 2
fi
if ! [[ "$GEOCLAW_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GEOCLAW_WORKERS must be a positive integer" >&2
  exit 2
fi

if [ -z "$RUN_ID" ]; then
  RUN_ID="preflight"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "Invalid run ID: $RUN_ID" >&2
  exit 2
fi
if [[ "$RUN_ID" == *.staging ]]; then
  echo "Run ID must not end with .staging" >&2
  exit 2
fi

STAGING_ROOT="evaluation_runs/${RUN_ID}.staging"
FINAL_ROOT="evaluation_runs/${RUN_ID}"
PREFLIGHT_ARGS=(
  "$PY" scripts/eval_suite_preflight.py
  --contract "$CONTRACT"
  --output-root "$STAGING_ROOT"
)
if [ "$INCLUDE_ENSEMBLE" = 1 ]; then
  PREFLIGHT_ARGS+=(--include-ensemble)
fi
if [ "$INCLUDE_PAPER_EVIDENCE" = 1 ]; then
  PREFLIGHT_ARGS+=(--include-paper-evidence)
fi
if [ "$RERUN_NUMERICAL_VALIDATION" = 1 ]; then
  PREFLIGHT_ARGS+=(--require-current-numerical-evidence)
fi
if [ "$INCLUDE_REAL_BATHYMETRY" = 0 ]; then
  PREFLIGHT_ARGS+=(--allow-missing-real-bathymetry)
fi
if [ "$DEEP_PAYLOAD_AUDIT" = 1 ]; then
  PREFLIGHT_ARGS+=(--deep-payload-audit)
fi

if [ "$EXECUTE" = 0 ]; then
  "${PREFLIGHT_ARGS[@]}"
  if [ "$RERUN_NUMERICAL_VALIDATION" = 1 ]; then
    "$PY" scripts/run_numerical_validation_chain.py \
      --preflight \
      --output-root "$STAGING_ROOT/numerical_validation" \
      --claw-root "$CLAW_ROOT" \
      --petsc-dir "$PETSC_DIR" \
      --petsc-arch "$PETSC_ARCH" \
      --geoclaw-python "$GEOCLAW_PYTHON"
  fi
  echo
  echo "Preflight passed. No files were created and no evaluation was run."
  echo "To execute, choose a permanent run ID:"
  echo "  bash scripts/run_eval_suite.sh --execute --run-id <run-id>"
  exit 0
fi

if [ "$RUN_ID" = "preflight" ]; then
  echo "--execute requires an explicit --run-id" >&2
  exit 2
fi
if [ -e "$STAGING_ROOT" ] || [ -e "$FINAL_ROOT" ]; then
  echo "Evaluation run already exists: $STAGING_ROOT or $FINAL_ROOT" >&2
  exit 1
fi

PREFLIGHT_TMP="evaluation_runs/.${RUN_ID}.preflight-${$}.json"
cleanup_preflight_tmp() {
  if [ -f "$PREFLIGHT_TMP" ]; then
    rm -f "$PREFLIGHT_TMP"
  fi
}
trap cleanup_preflight_tmp EXIT
mkdir -p evaluation_runs
"${PREFLIGHT_ARGS[@]}" --report "$PREFLIGHT_TMP"

mkdir "$STAGING_ROOT"
mv "$PREFLIGHT_TMP" "$STAGING_ROOT/preflight_report.json"
trap - EXIT

MANIFEST_ARGS=(
  "$PY" scripts/create_eval_run_manifest.py
  --contract "$CONTRACT"
  --run-id "$RUN_ID"
  --output "$STAGING_ROOT/run_manifest.json"
  --preflight-report "$STAGING_ROOT/preflight_report.json"
)
if [ "$INCLUDE_ENSEMBLE" = 1 ]; then
  MANIFEST_ARGS+=(--include-ensemble)
fi
if [ "$INCLUDE_REAL_BATHYMETRY" = 1 ]; then
  MANIFEST_ARGS+=(--include-real-bathymetry)
fi
if [ "$INCLUDE_SPEED" = 1 ]; then
  MANIFEST_ARGS+=(--include-speed)
fi
if [ "$INCLUDE_PAPER_EVIDENCE" = 1 ]; then
  MANIFEST_ARGS+=(--include-paper-evidence)
fi
if [ "$RERUN_NUMERICAL_VALIDATION" = 1 ]; then
  MANIFEST_ARGS+=(--rerun-numerical-validation)
fi
"${MANIFEST_ARGS[@]}"

run() {
  echo
  echo ">>> $*"
  "$@"
}

if [ "$RERUN_NUMERICAL_VALIDATION" = 1 ]; then
  echo "########## FRESH NUMERICAL-VALIDATION REGRESSION CHAIN ##########"
  run "$PY" scripts/run_numerical_validation_chain.py \
    --output-root "$STAGING_ROOT/numerical_validation" \
    --workers "$NUMERICAL_WORKERS" \
    --geoclaw-workers "$GEOCLAW_WORKERS" \
    --claw-root "$CLAW_ROOT" \
    --petsc-dir "$PETSC_DIR" \
    --petsc-arch "$PETSC_ARCH" \
    --geoclaw-python "$GEOCLAW_PYTHON"
fi

DIRECT_IDS=(
  fno ffno cnn unet convlstm ufno wno fno_modes8 fno_modes20
  fno_muscl_hr fno_boussinesq
)
DIRECT_CONFIGS=(
  configs/model/fno.yaml
  configs/model/ffno.yaml
  configs/model/cnn.yaml
  configs/model/unet.yaml
  configs/model/convlstm.yaml
  configs/model/ufno.yaml
  configs/model/wno.yaml
  configs/model/fno_modes8.yaml
  configs/model/fno_modes20.yaml
  configs/model/fno_muscl_hr.yaml
  configs/model/fno_boussinesq.yaml
)
DIRECT_CHECKPOINTS=(
  experiments/fno/best.pt
  experiments/ffno/best.pt
  experiments/cnn/best.pt
  experiments/unet/best.pt
  experiments/convlstm/best.pt
  experiments/ufno/best.pt
  experiments/wno/best.pt
  experiments/fno_modes8/best.pt
  experiments/fno_modes20/best.pt
  experiments/fno_muscl_hr/fno_muscl_hr_seed_18/best.pt
  experiments/fno_boussinesq/best.pt
)

echo "########## DIRECT COMMON-TIME-V2 EVALUATIONS ##########"
for i in "${!DIRECT_IDS[@]}"; do
  model="${DIRECT_IDS[$i]}"
  config="${DIRECT_CONFIGS[$i]}"
  checkpoint="${DIRECT_CHECKPOINTS[$i]}"
  out="$STAGING_ROOT/direct/$model"
  mkdir -p "$out"
  run "$PY" scripts/eval_accuracy.py \
    --config "$config" --checkpoint "$checkpoint" --device "$DEVICE" \
    --output "$out/metrics.json"
  run "$PY" scripts/eval_perframe.py \
    --config "$config" --checkpoint "$checkpoint" --device "$DEVICE" \
    --output "$out/perframe.json"
  run "$PY" scripts/eval_physics_diagnostics.py \
    --config "$config" --checkpoint "$checkpoint" --device "$DEVICE" \
    --output "$out/physics_diagnostics.json" \
    --per-sample-output "$out/physics_diagnostics_per_sample.csv"
done

echo "########## CONDITIONAL SEEDED WINDOW ROLLOUTS ##########"
WINDOW_IDS=(fno_window5_hydrostatic ffno_window5_hydrostatic)
WINDOW_CONFIGS=(
  configs/model/fno_window5_hydrostatic.yaml
  configs/model/ffno_window5_hydrostatic.yaml
)
WINDOW_CHECKPOINTS=(
  experiments/fno_window5_hydrostatic/best.pt
  experiments/ffno_window5_hydrostatic/best.pt
)
for i in "${!WINDOW_IDS[@]}"; do
  model="${WINDOW_IDS[$i]}"
  out="$STAGING_ROOT/window/$model"
  mkdir -p "$out"
  run "$PY" scripts/eval_window_rollout.py \
    --config "${WINDOW_CONFIGS[$i]}" \
    --checkpoint "${WINDOW_CHECKPOINTS[$i]}" \
    --device "$DEVICE" \
    --output "$out/metrics.json" \
    --perframe-output "$out/perframe.json"
done

echo "########## SAMPLE-SCALING EVALUATIONS ##########"
SCALING_IDS=(n_000100 n_000250 n_000500 n_001000 n_002500 n_005000)
for model in "${SCALING_IDS[@]}"; do
  out="$STAGING_ROOT/sample_scaling/$model"
  mkdir -p "$out"
  run "$PY" scripts/eval_accuracy.py \
    --config "experiments/sample_scaling/configs/fno_${model}.yaml" \
    --checkpoint "experiments/sample_scaling/${model}/best.pt" \
    --device "$DEVICE" \
    --output "$out/metrics.json"
done

echo "########## NATIVE MUSCL-HR DIAGONAL EVALUATIONS ##########"
for grid in 32 64 128; do
  out="$STAGING_ROOT/native_muscl/res${grid}"
  mkdir -p "$out"
  run "$PY" scripts/eval_accuracy.py \
    --config "configs/model/fno_res${grid}_muscl_hr.yaml" \
    --checkpoint "experiments/fno_res${grid}_muscl_hr/best.pt" \
    --device "$DEVICE" \
    --output "$out/metrics.json"
done

echo "########## STRICT FAMILY HOLDOUT EVALUATIONS ##########"
run env DEVICE="$DEVICE" PY="$PY" bash scripts/run_strict_holdout_evals.sh \
  --output-root "$STAGING_ROOT/strict_holdout"

if [ "$INCLUDE_REAL_BATHYMETRY" = 1 ]; then
  echo "########## V2 REAL-BATHYMETRY AUXILIARY ##########"
  mkdir -p "$STAGING_ROOT/real_bathymetry/direct"
  run "$PY" scripts/eval_full_resolution.py \
    --config configs/eval/real_bathymetry_hydrostatic.yaml \
    --checkpoint experiments/fno/best.pt \
    --device "$DEVICE" \
    --output "$STAGING_ROOT/real_bathymetry/direct/fno.json"
  run "$PY" scripts/eval_full_resolution.py \
    --config configs/eval/real_bathymetry_ffno_hydrostatic.yaml \
    --checkpoint experiments/ffno/best.pt \
    --device "$DEVICE" \
    --output "$STAGING_ROOT/real_bathymetry/direct/ffno.json"

  mkdir -p "$STAGING_ROOT/real_bathymetry/window"
  run "$PY" scripts/eval_window_suites.py \
    --config configs/eval/window5_real_bathymetry_hydrostatic.yaml \
    --checkpoint experiments/fno_window5_hydrostatic/best.pt \
    --device "$DEVICE" \
    --output "$STAGING_ROOT/real_bathymetry/window/fno_window5_hydrostatic.json"
  run "$PY" scripts/eval_window_suites.py \
    --config configs/eval/ffno_window5_real_bathymetry_hydrostatic.yaml \
    --checkpoint experiments/ffno_window5_hydrostatic/best.pt \
    --device "$DEVICE" \
    --output "$STAGING_ROOT/real_bathymetry/window/ffno_window5_hydrostatic.json"
fi

echo "########## DATASET AND MODEL INVENTORY ##########"
run "$PY" scripts/_summarize_datasets.py \
  --processed-root data/processed \
  --output "$STAGING_ROOT/dataset_summary.json"
run "$PY" scripts/export_parameter_counts.py \
  --output "$STAGING_ROOT/parameter_counts.json" \
  --csv-output "$STAGING_ROOT/parameter_counts.csv"

if [ "$INCLUDE_ENSEMBLE" = 1 ]; then
  echo "########## SEVEN-MEMBER ENSEMBLE ##########"
  ENSEMBLE_ARGS=()
  for seed in 11 22 33 44 55 66 77; do
    ENSEMBLE_ARGS+=(--checkpoint "experiments/ensemble/member_${seed}/best.pt")
  done
  mkdir -p "$STAGING_ROOT/ensemble"
  run "$PY" scripts/eval_uncertainty.py \
    --config configs/eval/uncertainty_indist_hydrostatic.yaml \
    "${ENSEMBLE_ARGS[@]}" \
    --device "$DEVICE" \
    --output "$STAGING_ROOT/ensemble/indist.json"
fi

if [ "$INCLUDE_PAPER_EVIDENCE" = 1 ]; then
  echo "########## CURRENT-V2 PAPER EVIDENCE ##########"
  PAPER_ROOT="$STAGING_ROOT/paper_evidence"
  PAPER_SUITES=(
    --suite "source_holdout_multi_gauss=source_type_in:multi-gauss"
    --suite "bathymetry_holdout_trench=bathymetry_type_in:trench"
    --suite "source_strength_extreme_high=source_strength_min:0.82"
  )
  GAUGES=(
    --gauge "16,16" --gauge "16,32" --gauge "16,48"
    --gauge "32,16" --gauge "32,32" --gauge "32,48"
    --gauge "48,16" --gauge "48,32" --gauge "48,48"
  )
  ENSEMBLE_ARGS=()
  ENSEMBLE_SLICE_ARGS=()
  for seed in 11 22 33 44 55 66 77; do
    checkpoint="experiments/ensemble/member_${seed}/best.pt"
    ENSEMBLE_ARGS+=(--checkpoint "$checkpoint")
    ENSEMBLE_SLICE_ARGS+=(--ensemble-checkpoint "$checkpoint")
  done

  run "$PY" scripts/export_v2_numerical_evidence.py \
    --contract "$CONTRACT" \
    --output "$PAPER_ROOT/numerical_evidence.json"

  PAPER_DIRECT_IDS=(fno_hydrostatic fno_muscl_hr fno_boussinesq)
  PAPER_DIRECT_CONFIGS=(
    configs/model/fno.yaml
    configs/model/fno_muscl_hr.yaml
    configs/model/fno_boussinesq.yaml
  )
  PAPER_DIRECT_CHECKPOINTS=(
    experiments/fno/best.pt
    experiments/fno_muscl_hr/fno_muscl_hr_seed_18/best.pt
    experiments/fno_boussinesq/best.pt
  )
  PAPER_DIRECT_DATASETS=(
    data/processed/hydrostatic/test
    data/processed/muscl_hr/test
    data/processed/boussinesq/test
  )
  for i in "${!PAPER_DIRECT_IDS[@]}"; do
    model="${PAPER_DIRECT_IDS[$i]}"
    config="${PAPER_DIRECT_CONFIGS[$i]}"
    checkpoint="${PAPER_DIRECT_CHECKPOINTS[$i]}"
    dataset="${PAPER_DIRECT_DATASETS[$i]}"
    run "$PY" scripts/eval_v2_slices.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --dataset "$dataset" \
      "${PAPER_SUITES[@]}" \
      --device "$DEVICE" \
      --output "$PAPER_ROOT/slices/direct/${model}.json"
  done

  for group in source_type bathymetry_type; do
    run "$PY" scripts/eval_v2_slices.py \
      --config configs/model/fno.yaml \
      --checkpoint experiments/fno/best.pt \
      --dataset data/processed/hydrostatic/test \
      --group-by "$group" \
      --device "$DEVICE" \
      --output "$PAPER_ROOT/slices/groups/${group}.json"
  done

  PAPER_WINDOW_IDS=(fno_window5_hydrostatic ffno_window5_hydrostatic)
  PAPER_WINDOW_CONFIGS=(
    configs/model/fno_window5_hydrostatic.yaml
    configs/model/ffno_window5_hydrostatic.yaml
  )
  PAPER_WINDOW_CHECKPOINTS=(
    experiments/fno_window5_hydrostatic/best.pt
    experiments/ffno_window5_hydrostatic/best.pt
  )
  for i in "${!PAPER_WINDOW_IDS[@]}"; do
    model="${PAPER_WINDOW_IDS[$i]}"
    run "$PY" scripts/eval_v2_slices.py \
      --config "${PAPER_WINDOW_CONFIGS[$i]}" \
      --checkpoint "${PAPER_WINDOW_CHECKPOINTS[$i]}" \
      --dataset data/processed/hydrostatic/test \
      "${PAPER_SUITES[@]}" \
      --window \
      --device "$DEVICE" \
      --output "$PAPER_ROOT/slices/window/${model}.json"
  done

  run "$PY" scripts/eval_resolution_transfer.py \
    --config configs/eval/resolution_transfer_proxy_hydrostatic.yaml \
    --checkpoint experiments/fno/best.pt \
    --device "$DEVICE" \
    --output "$PAPER_ROOT/resolution/proxy_hydrostatic.json"
  run "$PY" scripts/eval_v2_native_transfer.py \
    --contract "$CONTRACT" \
    --device "$DEVICE" \
    --output "$PAPER_ROOT/resolution/native_muscl_hr.json"

  run "$PY" scripts/eval_v2_reference_analysis.py \
    --contract "$CONTRACT" \
    --model "hydrostatic|configs/model/fno.yaml|experiments/fno/best.pt|data/processed/hydrostatic/test" \
    --model "muscl_hr|configs/model/fno_muscl_hr.yaml|experiments/fno_muscl_hr/fno_muscl_hr_seed_18/best.pt|data/processed/muscl_hr/test" \
    --model "boussinesq|configs/model/fno_boussinesq.yaml|experiments/fno_boussinesq/best.pt|data/processed/boussinesq/test" \
    --bootstrap-seed 20260813 \
    --bootstrap-resamples 2000 \
    --device "$DEVICE" \
    --output-dir "$PAPER_ROOT/reference_analysis"

  for i in "${!PAPER_DIRECT_IDS[@]}"; do
    model="${PAPER_DIRECT_IDS[$i]}"
    config="${PAPER_DIRECT_CONFIGS[$i]}"
    checkpoint="${PAPER_DIRECT_CHECKPOINTS[$i]}"
    dataset="${PAPER_DIRECT_DATASETS[$i]}"
    run "$PY" scripts/eval_arrival_maps.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --arrival-threshold-fraction 0.10 \
      --device "$DEVICE" \
      --output "$PAPER_ROOT/arrival_maps/${model}.json" \
      --maps-output "$PAPER_ROOT/arrival_maps/${model}.npz"
    run "$PY" scripts/eval_v2_wave_metrics.py \
      --contract "$CONTRACT" \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --dataset "$dataset" \
      "${GAUGES[@]}" \
      --arrival-threshold-fraction 0.10 \
      --peak-plateau-fraction 0.99 \
      --device "$DEVICE" \
      --output "$PAPER_ROOT/wave_metrics/${model}.json"
  done

  run "$PY" scripts/eval_v2_slices.py \
    --config configs/model/fno.yaml \
    "${ENSEMBLE_SLICE_ARGS[@]}" \
    --dataset data/processed/hydrostatic/test \
    "${PAPER_SUITES[@]}" \
    --device "$DEVICE" \
    --output "$PAPER_ROOT/ensemble/slices.json"
  run "$PY" scripts/eval_v2_calibration.py \
    --config configs/model/fno.yaml \
    "${ENSEMBLE_ARGS[@]}" \
    --val-dataset data/processed/hydrostatic/val \
    --test-dataset data/processed/hydrostatic/test \
    "${PAPER_SUITES[@]}" \
    --device "$DEVICE" \
    --output "$PAPER_ROOT/ensemble/calibration.json"
  run "$PY" scripts/eval_v2_seed_metrics.py \
    --config configs/model/fno.yaml \
    "${ENSEMBLE_ARGS[@]}" \
    --dataset data/processed/hydrostatic/test \
    --bootstrap-seed 20260813 \
    --bootstrap-resamples 2000 \
    --device "$DEVICE" \
    --output "$PAPER_ROOT/ensemble/seed_stability.json"
fi

if [ "$INCLUDE_SPEED" = 1 ]; then
  echo "########## OPTIONAL CONTROLLED SPEED EVALUATIONS ##########"
  mkdir -p "$STAGING_ROOT/speed"
  for i in "${!DIRECT_IDS[@]}"; do
    model="${DIRECT_IDS[$i]}"
    run "$PY" scripts/eval_speed.py \
      --config "${DIRECT_CONFIGS[$i]}" \
      --checkpoint "${DIRECT_CHECKPOINTS[$i]}" \
      --device "$DEVICE" \
      --batch-size 1 \
      --precision fp32 \
      --allow-tf32 false \
      --warmup 5 \
      --repeats 20 \
      --output "$STAGING_ROOT/speed/speed_${model}.json"
  done
  for solver in swe_hydrostatic swe_muscl_hr boussinesq; do
    run "$PY" scripts/eval_solver_speed.py \
      --config configs/data/dataset_test.yaml \
      --solver "$solver" \
      --device cpu \
      --sample-ids 1 2 3 4 5 6 7 8 \
      --repeats 3 \
      --output "$STAGING_ROOT/speed/solver_speed_${solver}.json"
  done
  run "$PY" scripts/_aggregate_speed_table.py \
    --results-dir "$STAGING_ROOT/speed" \
    --output "$STAGING_ROOT/speed/speed_table.json" \
    --csv-output "$STAGING_ROOT/speed/speed_table.csv"
fi

echo "########## FAIL-CLOSED CONSOLIDATION ##########"
run "$PY" scripts/_consolidate_results.py \
  --run-root "$STAGING_ROOT"

mv "$STAGING_ROOT" "$FINAL_ROOT"
echo
echo "Evaluation run validated and promoted atomically:"
echo "  $FINAL_ROOT"
