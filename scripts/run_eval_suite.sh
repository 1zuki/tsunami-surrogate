#!/usr/bin/env bash
# Finalized evaluation runner.
#
# Behaviour:
#   * Each eval script writes its own per-model JSON natively, exactly as when run
#     from the CLI (into experiments/<model>/eval/*.json) -- nothing is suppressed.
#   * After the runs, every per-eval JSON is mirrored into results/ with a flat,
#     descriptive name AND merged into a single results/all_results.json for easy reading.
#
# Run this on a QUIESCENT machine if you care about the speed numbers
# (timing is meaningless under background load). Pass --no-speed to skip timing.
#
# Usage:
#   bash scripts/run_eval_suite.sh              # full paper-facing suite
#   bash scripts/run_eval_suite.sh --quick      # cheap sanity subset
#   bash scripts/run_eval_suite.sh --no-speed   # skip model/solver timing
#   bash scripts/run_eval_suite.sh --no-solver-validation
#   bash scripts/run_eval_suite.sh --speed-only # rerun model/solver timing only
#   DEVICE=cpu bash scripts/run_eval_suite.sh   # force CPU
#
# Full mode includes strict held-out family evals. To rerun just that block:
#   DEVICE=cuda bash scripts/run_strict_holdout_evals.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE="${DEVICE:-cuda}"
RESULTS=results
# Raw-rollout base dir, only needed by evals that read raw samples (e.g. solver
# timing, arrival maps). Override with: RAW_DATA_BASE=/path/to/data bash ...
RAW_DATA_BASE="${RAW_DATA_BASE:-data}"
RAW_SPLIT="${RAW_SPLIT:-test}"
if { [ ! -d "$RAW_DATA_BASE/$RAW_SPLIT/raw/hydrostatic/samples" ] \
     || [ ! -d "$RAW_DATA_BASE/$RAW_SPLIT/bathymetry" ] \
     || [ ! -d "$RAW_DATA_BASE/$RAW_SPLIT/sources" ]; } \
   && [ -d "/mnt/Windows/Users/Izu/tsunami-surrogate/data/$RAW_SPLIT/raw/hydrostatic/samples" ]; then
  RAW_DATA_BASE="/mnt/Windows/Users/Izu/tsunami-surrogate/data"
fi
HYDRO_SAMPLES="$RAW_DATA_BASE/$RAW_SPLIT/raw/hydrostatic/samples"
MUSCL_HR_SAMPLES="$RAW_DATA_BASE/$RAW_SPLIT/raw/muscl_hr/samples"
BOUSSINESQ_SAMPLES="$RAW_DATA_BASE/$RAW_SPLIT/raw/boussinesq/samples"
BATHYMETRY_CACHE="$RAW_DATA_BASE/$RAW_SPLIT/bathymetry"
SOURCE_CACHE="$RAW_DATA_BASE/$RAW_SPLIT/sources"
mkdir -p "$RESULTS"

RUN_SPEED=1
RUN_SOLVER_VALIDATION=1
RUN_ONLY_SPEED=0
RUN_MODE=full
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-speed)
      RUN_SPEED=0
      ;;
    --no-solver-validation)
      RUN_SOLVER_VALIDATION=0
      ;;
    --speed-only)
      RUN_ONLY_SPEED=1
      ;;
    --quick)
      RUN_MODE=quick
      ;;
    --full)
      RUN_MODE=full
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done
if [ "$RUN_ONLY_SPEED" = 1 ] && [ "$RUN_SPEED" = 0 ]; then
  echo "--speed-only conflicts with --no-speed" >&2
  exit 2
fi

# Models that have a trained checkpoint. ConvLSTM excluded (training incomplete).
ACCURACY_MODELS=(fno ffno cnn unet fno_modes8 fno_modes20 ufno wno fno_muscl_hr fno_boussinesq)
HYDROSTATIC_DIRECT_MODELS=(fno ffno fno_modes8 fno_modes20 ufno wno)
PHYSICS_DIAGNOSTIC_MODELS=(fno ffno fno_modes8 fno_modes20 wno)
WINDOW_MODELS=(fno_window5_hydrostatic ffno_window5_hydrostatic)
WINDOW_SUITE_CONFIGS=(
  configs/eval/window5_ood_hydrostatic.yaml
  configs/eval/window5_crossres_hydrostatic.yaml
  configs/eval/window5_real_bathymetry_hydrostatic.yaml
  configs/eval/ffno_window5_ood_hydrostatic.yaml
  configs/eval/ffno_window5_crossres_hydrostatic.yaml
  configs/eval/ffno_window5_real_bathymetry_hydrostatic.yaml
)
WINDOW_SUITE_CHECKPOINTS=(
  experiments/fno_window5_hydrostatic/best.pt
  experiments/fno_window5_hydrostatic/best.pt
  experiments/fno_window5_hydrostatic/best.pt
  experiments/ffno_window5_hydrostatic/best.pt
  experiments/ffno_window5_hydrostatic/best.pt
  experiments/ffno_window5_hydrostatic/best.pt
)
REAL_BATHY_MODELS=(fno ffno)
REAL_BATHY_CONFIGS=(
  configs/eval/real_bathymetry_hydrostatic.yaml
  configs/eval/real_bathymetry_ffno_hydrostatic.yaml
)
UNCERTAINTY_CONFIGS=(
  configs/eval/uncertainty_indist_hydrostatic.yaml
  configs/eval/uncertainty_ood_hydrostatic.yaml
)
ENSEMBLE_CHECKPOINTS=(
  experiments/ensemble/member_11/best.pt
  experiments/ensemble/member_22/best.pt
  experiments/ensemble/member_33/best.pt
  experiments/ensemble/member_44/best.pt
  experiments/ensemble/member_55/best.pt
  experiments/ensemble/member_66/best.pt
  experiments/ensemble/member_77/best.pt
)
FNO_MODELS=(fno fno_muscl_hr fno_boussinesq)
FNO_TARGETS=(hydrostatic muscl_hr boussinesq)
SOLVERS=(swe_hydrostatic swe_muscl_hr boussinesq)

if [ "$RUN_ONLY_SPEED" = 1 ]; then
  echo "########## CLEAN (removing stale speed outputs only) ##########"
  for m in "${ACCURACY_MODELS[@]}"; do
    rm -f experiments/$m/eval/speed.json
  done
  rm -f "$RESULTS"/speed_*.json "$RESULTS"/solver_speed_*.json \
        "$RESULTS"/speed_table.json "$RESULTS"/speed_table.csv \
        "$RESULTS"/all_results.json
else
  # Full rerun: clear stale per-eval outputs so nothing old survives into the merged file.
  echo "########## CLEAN (removing stale eval outputs) ##########"
  for m in "${ACCURACY_MODELS[@]}"; do
    rm -f experiments/$m/eval/metrics.json \
	          experiments/$m/eval/speed.json \
	          experiments/$m/eval/ood_generalization.json \
	          experiments/$m/eval/resolution_transfer_proxy.json \
	          experiments/$m/eval/perframe.json \
	          experiments/$m/eval/physics_diagnostics.json \
	          experiments/$m/eval/physics_diagnostics_per_sample.csv
  done
  for m in "${WINDOW_MODELS[@]}"; do
    rm -f experiments/$m/eval/metrics.json \
          experiments/$m/eval/perframe.json \
          experiments/$m/eval_ood_suites/window_rollout_suites.json \
          experiments/$m/eval_crossres_native/window_rollout_suites.json \
          experiments/$m/eval_real_bathymetry/window_rollout_suites.json
  done
  for m in "${REAL_BATHY_MODELS[@]}"; do
    rm -f experiments/$m/eval_real_bathymetry/real_resolution.json
  done
  rm -f experiments/fno/eval_uncertainty_indist/uncertainty.json \
        experiments/fno/eval_uncertainty_ood/uncertainty_ood.json
  for m in "${FNO_MODELS[@]}"; do
    rm -f experiments/$m/eval_ood_suites/ood_generalization.json \
          experiments/$m/eval_resolution_proxy/resolution_transfer_proxy.json
  done
  rm -f "$RESULTS"/accuracy_*.json "$RESULTS"/speed_*.json \
	        "$RESULTS"/ood_generalization_*.json "$RESULTS"/resolution_transfer_proxy_*.json \
	        "$RESULTS"/ood_suites_*.json "$RESULTS"/solver_speed_*.json \
	        "$RESULTS"/perframe_*.json \
	        "$RESULTS"/physics_diagnostics_*.json \
	        "$RESULTS"/window_rollout_*.json "$RESULTS"/window_rollout_perframe_*.json \
	        "$RESULTS"/speed_table.json "$RESULTS"/speed_table.csv \
	        "$RESULTS"/parameter_counts.json "$RESULTS"/parameter_counts.csv \
	        "$RESULTS"/native_resolution_transfer_matrix_fno_hydrostatic.json \
	        "$RESULTS"/dataset_summary.json "$RESULTS"/all_results.json
  if [ -d "$HYDRO_SAMPLES" ] && [ -d "$MUSCL_HR_SAMPLES" ] && [ -d "$BOUSSINESQ_SAMPLES" ]; then
    rm -f "$RESULTS"/solver_compare_*.json \
          "$RESULTS"/solver_compare_*_arrival_maps.npz \
          "$RESULTS"/emulator_superiority_*.json
  else
    echo "preserving solver-compare/emulator-superiority JSONs (raw solver dirs not all present)."
  fi
  rm -f "$RESULTS"/solver_validation_full/phase3_solver_validation.json \
        "$RESULTS"/solver_validation_full/phase3_solver_validation_table.csv
fi
echo "cleaned."

run() {
  echo
  echo ">>> $*"
  local tmp
  tmp="$(mktemp)"
  if "$@" >"$tmp" 2>&1; then
    grep -vE "shard-aware|batch sampler" "$tmp" | tail -3 || true
    rm -f "$tmp"
  else
    cat "$tmp" >&2
    rm -f "$tmp"
    return 1
  fi
}

run_solver_compare() {
  local label="$1"
  local solver_a_dir="$2"
  local solver_b_dir="$3"
  local output="$4"
  if [ ! -d "$solver_a_dir" ] || [ ! -d "$solver_b_dir" ]; then
    echo "SKIPPED solver comparison $label (missing raw dirs: $solver_a_dir or $solver_b_dir)"
    return 0
  fi
  run $PY scripts/compare_solvers_physical.py \
        --solver-a-dir "$solver_a_dir" \
        --solver-b-dir "$solver_b_dir" \
        --require-quality-ok \
        --missing-quality-action include \
        --save-arrival-maps \
        --output "$output"
}

run_emulator_superiority() {
  local label="$1"
  local config="$2"
  local solver_compare="$3"
  if [ ! -f "$config" ]; then
    echo "SKIPPED emulator-superiority $label (missing config: $config)"
    return 0
  fi
  if [ ! -f "$solver_compare" ]; then
    echo "SKIPPED emulator-superiority $label (missing $solver_compare)"
    return 0
  fi
  run $PY scripts/eval_emulator_superiority.py --config "$config" --device "$DEVICE"
}

run_uncertainty() {
  local config="$1"
  local args=()
  for ckpt in "${ENSEMBLE_CHECKPOINTS[@]}"; do
    args+=(--checkpoint "$ckpt")
  done
  run $PY scripts/eval_uncertainty.py --config "$config" "${args[@]}" --device "$DEVICE"
}

if [ "$RUN_ONLY_SPEED" = 0 ]; then
  echo "########## ACCURACY ##########"
  for m in "${ACCURACY_MODELS[@]}"; do
    run $PY scripts/eval_accuracy.py --config configs/model/$m.yaml \
	          --checkpoint experiments/$m/best.pt --device "$DEVICE"
  done

  if [ "$RUN_MODE" = quick ]; then
    echo "########## OOD GENERALIZATION (quick: FNO, hydrostatic) ##########"
    run $PY scripts/eval_generalization.py --config configs/model/fno.yaml \
          --checkpoint experiments/fno/best.pt --device "$DEVICE"

    echo "########## CROSS-RESOLUTION PROXY (quick: FNO) ##########"
    run $PY scripts/eval_resolution_transfer.py --config configs/model/fno.yaml \
          --checkpoint experiments/fno/best.pt --device "$DEVICE"
  else
    echo "########## OOD SUITE CONSTRUCTION ##########"
    for target in "${FNO_TARGETS[@]}"; do
      run $PY scripts/make_ood_splits.py --config configs/data/ood_splits_$target.yaml --overwrite
    done

    echo "########## OOD SUITES (all FNO targets) ##########"
    for i in "${!FNO_MODELS[@]}"; do
      m="${FNO_MODELS[$i]}"
      target="${FNO_TARGETS[$i]}"
      run $PY scripts/eval_generalization.py --config configs/eval/ood_suites_$target.yaml \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## CROSS-RESOLUTION PROXY (all FNO targets) ##########"
    for i in "${!FNO_MODELS[@]}"; do
      m="${FNO_MODELS[$i]}"
      target="${FNO_TARGETS[$i]}"
      run $PY scripts/eval_resolution_transfer.py --config configs/eval/resolution_transfer_proxy_$target.yaml \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## NATIVE-RESOLUTION MATRIX (hydrostatic FNO only) ##########"
    run $PY scripts/eval_native_resolution_hydrostatic.py --device "$DEVICE" \
          --output "$RESULTS/native_resolution_transfer_matrix_fno_hydrostatic.json"

    echo "########## PER-FRAME ERROR CURVES (FNO targets) ##########"
    for m in "${FNO_MODELS[@]}"; do
      run $PY scripts/eval_perframe.py --config configs/model/$m.yaml \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## PER-FRAME ERROR CURVES (hydrostatic neural-operator baselines) ##########"
    for m in "${HYDROSTATIC_DIRECT_MODELS[@]}"; do
      if [ "$m" = fno ]; then
        continue
      fi
      run $PY scripts/eval_perframe.py --config configs/model/$m.yaml \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## PHYSICS DIAGNOSTICS (main hydrostatic operators) ##########"
    for m in "${PHYSICS_DIAGNOSTIC_MODELS[@]}"; do
      run $PY scripts/eval_physics_diagnostics.py --config configs/model/$m.yaml \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## SEEDED WINDOW ROLLOUTS ##########"
    for m in "${WINDOW_MODELS[@]}"; do
      run $PY scripts/eval_window_rollout.py --config configs/model/$m.yaml \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## SEEDED WINDOW SUITES (OOD / cross-res / real bathymetry) ##########"
    for i in "${!WINDOW_SUITE_CONFIGS[@]}"; do
      run $PY scripts/eval_window_suites.py --config "${WINDOW_SUITE_CONFIGS[$i]}" \
            --checkpoint "${WINDOW_SUITE_CHECKPOINTS[$i]}" --device "$DEVICE"
    done

    echo "########## REAL BATHYMETRY TRANSFER (direct models) ##########"
    for i in "${!REAL_BATHY_MODELS[@]}"; do
      m="${REAL_BATHY_MODELS[$i]}"
      run $PY scripts/eval_full_resolution.py --config "${REAL_BATHY_CONFIGS[$i]}" \
            --checkpoint experiments/$m/best.pt --device "$DEVICE"
    done

    echo "########## DATASET SUMMARY ##########"
    run $PY scripts/_summarize_datasets.py --output "$RESULTS/dataset_summary.json"

    echo "########## PARAMETER COUNTS ##########"
    run $PY scripts/export_parameter_counts.py \
          --output "$RESULTS/parameter_counts.json" \
          --csv-output "$RESULTS/parameter_counts.csv"

    echo "########## UNCERTAINTY (hydrostatic 7-member ensemble) ##########"
    for config in "${UNCERTAINTY_CONFIGS[@]}"; do
      run_uncertainty "$config"
    done

    echo "########## SOLVER-VS-SOLVER PHYSICAL GAP ##########"
    run_solver_compare hydro_vs_muscl_hr \
      "$HYDRO_SAMPLES" "$MUSCL_HR_SAMPLES" \
      "$RESULTS/solver_compare_hydro_vs_muscl_hr.json"
    run_solver_compare muscl_hr_vs_hydro \
      "$MUSCL_HR_SAMPLES" "$HYDRO_SAMPLES" \
      "$RESULTS/solver_compare_muscl_hr_vs_hydro.json"
    run_solver_compare hydro_vs_boussinesq \
      "$HYDRO_SAMPLES" "$BOUSSINESQ_SAMPLES" \
      "$RESULTS/solver_compare_hydro_vs_boussinesq.json"
    run_solver_compare muscl_hr_vs_boussinesq \
      "$MUSCL_HR_SAMPLES" "$BOUSSINESQ_SAMPLES" \
      "$RESULTS/solver_compare_muscl_hr_vs_boussinesq.json"
    run_solver_compare boussinesq_vs_hydrostatic \
      "$BOUSSINESQ_SAMPLES" "$HYDRO_SAMPLES" \
      "$RESULTS/solver_compare_boussinesq_vs_hydrostatic.json"
    run_solver_compare boussinesq_vs_muscl_hr \
      "$BOUSSINESQ_SAMPLES" "$MUSCL_HR_SAMPLES" \
      "$RESULTS/solver_compare_boussinesq_vs_muscl_hr.json"

    echo "########## EMULATOR-SUPERIORITY RATIO ##########"
    run_emulator_superiority hydro_to_muscl_hr \
      configs/eval/emulator_superiority_hydro_to_muscl_hr.yaml \
      "$RESULTS/solver_compare_hydro_vs_muscl_hr.json"
    run_emulator_superiority muscl_hr_to_hydro \
      configs/eval/emulator_superiority_muscl_hr_to_hydro.yaml \
      "$RESULTS/solver_compare_muscl_hr_vs_hydro.json"
    run_emulator_superiority hydro_to_boussinesq \
      configs/eval/emulator_superiority_hydro_to_boussinesq.yaml \
      "$RESULTS/solver_compare_hydro_vs_boussinesq.json"
    run_emulator_superiority muscl_hr_to_boussinesq \
      configs/eval/emulator_superiority_muscl_hr_to_boussinesq.yaml \
      "$RESULTS/solver_compare_muscl_hr_vs_boussinesq.json"
    run_emulator_superiority boussinesq_to_hydrostatic \
      configs/eval/emulator_superiority_boussinesq_to_hydrostatic.yaml \
      "$RESULTS/solver_compare_boussinesq_vs_hydrostatic.json"
    run_emulator_superiority boussinesq_to_muscl_hr \
      configs/eval/emulator_superiority_boussinesq_to_muscl_hr.yaml \
      "$RESULTS/solver_compare_boussinesq_vs_muscl_hr.json"

    if [ "$RUN_SOLVER_VALIDATION" = 1 ]; then
      echo "########## PHASE 3 SOLVER VALIDATION ##########"
      run $PY scripts/run_solver_validation.py \
        --lake-samples 24 \
        --conservation-samples 24 \
        --cfl-samples 12 \
        --conservation-solvers swe_hydrostatic swe_muscl_hr \
        --cfl-solvers swe_hydrostatic swe_muscl_hr \
        --output-dir "$RESULTS/solver_validation_full"
    else
      echo "########## PHASE 3 SOLVER VALIDATION SKIPPED (--no-solver-validation) ##########"
    fi

    echo "########## STRICT HELD-OUT FAMILY EVALS ##########"
    run bash scripts/run_strict_holdout_evals.sh
  fi
fi

if [ "$RUN_SPEED" = 1 ]; then
  echo "########## MODEL SPEED ##########"
  for m in "${ACCURACY_MODELS[@]}"; do
    run $PY scripts/eval_speed.py --config configs/model/$m.yaml \
          --checkpoint experiments/$m/best.pt --device "$DEVICE"
  done
  echo "########## SOLVER SPEED (CPU) ##########"
  for s in "${SOLVERS[@]}"; do
    run $PY scripts/eval_solver_speed.py --solver $s --device cpu \
	          --max-samples 8 \
            --bathymetry-dir "$BATHYMETRY_CACHE" \
            --source-dir "$SOURCE_CACHE" \
            --output "$RESULTS/solver_speed_$s.json"
  done
  echo "########## SPEED TABLE ##########"
  run $PY scripts/_aggregate_speed_table.py --output "$RESULTS/speed_table.json" \
        --csv-output "$RESULTS/speed_table.csv"
else
  echo "########## SPEED SKIPPED (--no-speed) ##########"
fi

echo "SKIPPED (pending trained checkpoint): ConvLSTM row"

echo
echo "########## CONSOLIDATE -> $RESULTS/ ##########"
$PY scripts/_consolidate_results.py
echo "DONE. Per-eval JSONs in $RESULTS/, merged view in $RESULTS/all_results.json"
