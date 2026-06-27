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
#   DEVICE=cpu bash scripts/run_eval_suite.sh   # force CPU

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE="${DEVICE:-cuda}"
RESULTS=results
# Raw-rollout base dir, only needed by evals that read raw samples (e.g. solver
# timing, arrival maps). Override with: RAW_DATA_BASE=/path/to/data bash ...
RAW_DATA_BASE="${RAW_DATA_BASE:-data}"
RAW_SPLIT="${RAW_SPLIT:-test}"
mkdir -p "$RESULTS"

RUN_SPEED=1
RUN_MODE=full
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-speed)
      RUN_SPEED=0
      ;;
    --quick)
      RUN_MODE=quick
      ;;
    --full)
      RUN_MODE=full
      ;;
    -h|--help)
      sed -n '1,17p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# Models that have a trained checkpoint. ConvLSTM excluded (training incomplete).
ACCURACY_MODELS=(fno cnn unet fno_muscl_hr fno_boussinesq)
FNO_MODELS=(fno fno_muscl_hr fno_boussinesq)
FNO_TARGETS=(hydrostatic muscl_hr boussinesq)
SOLVERS=(swe_hydrostatic swe_muscl_hr boussinesq)

# Full rerun: clear stale per-eval outputs so nothing old survives into the merged file.
echo "########## CLEAN (removing stale eval outputs) ##########"
for m in "${ACCURACY_MODELS[@]}"; do
  rm -f experiments/$m/eval/metrics.json \
	        experiments/$m/eval/speed.json \
	        experiments/$m/eval/ood_generalization.json \
	        experiments/$m/eval/resolution_transfer_proxy.json \
	        experiments/$m/eval/perframe.json
done
for m in "${FNO_MODELS[@]}"; do
  rm -f experiments/$m/eval_ood_suites/ood_generalization.json \
        experiments/$m/eval_resolution_proxy/resolution_transfer_proxy.json
done
rm -f "$RESULTS"/accuracy_*.json "$RESULTS"/speed_*.json \
	      "$RESULTS"/ood_generalization_*.json "$RESULTS"/resolution_transfer_proxy_*.json \
	      "$RESULTS"/ood_suites_*.json "$RESULTS"/solver_speed_*.json \
	      "$RESULTS"/solver_compare_*.json "$RESULTS"/solver_compare_*_arrival_maps.npz \
	      "$RESULTS"/emulator_superiority_*.json "$RESULTS"/perframe_*.json \
	      "$RESULTS"/speed_table.json "$RESULTS"/speed_table.csv \
	      "$RESULTS"/dataset_summary.json "$RESULTS"/all_results.json
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

  echo "########## PER-FRAME ERROR CURVES (FNO targets) ##########"
  for m in "${FNO_MODELS[@]}"; do
    run $PY scripts/eval_perframe.py --config configs/model/$m.yaml \
          --checkpoint experiments/$m/best.pt --device "$DEVICE"
  done

  echo "########## DATASET SUMMARY ##########"
  run $PY scripts/_summarize_datasets.py --output "$RESULTS/dataset_summary.json"

  echo "########## SOLVER-VS-SOLVER PHYSICAL GAP ##########"
  HYDRO_SAMPLES="$RAW_DATA_BASE/$RAW_SPLIT/raw/hydrostatic/samples"
  MUSCL_HR_SAMPLES="$RAW_DATA_BASE/$RAW_SPLIT/raw/muscl_hr/samples"
  BOUSSINESQ_SAMPLES="$RAW_DATA_BASE/$RAW_SPLIT/raw/boussinesq/samples"
  run_solver_compare hydro_vs_muscl_hr \
    "$HYDRO_SAMPLES" "$MUSCL_HR_SAMPLES" \
    "$RESULTS/solver_compare_hydro_vs_muscl_hr.json"
  run_solver_compare muscl_hr_vs_hydro \
    "$MUSCL_HR_SAMPLES" "$HYDRO_SAMPLES" \
    "$RESULTS/solver_compare_muscl_hr_vs_hydro.json"
  run_solver_compare hydro_vs_boussinesq \
    "$HYDRO_SAMPLES" "$BOUSSINESQ_SAMPLES" \
    "$RESULTS/solver_compare_hydro_vs_boussinesq.json"

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
	          --max-samples 8 --output "$RESULTS/solver_speed_$s.json"
  done
  echo "########## SPEED TABLE ##########"
  run $PY scripts/_aggregate_speed_table.py --output "$RESULTS/speed_table.json" \
        --csv-output "$RESULTS/speed_table.csv"
else
  echo "########## SPEED SKIPPED (--no-speed) ##########"
fi

echo "SKIPPED (pending trained checkpoint): ConvLSTM row"
echo "SKIPPED (pending ensemble checkpoints): ensemble uncertainty"
echo "SKIPPED (pending GEBCO preprocessing): real-bathymetry transfer"
echo "SKIPPED (pending native-resolution checkpoint matrix): native-resolution evaluation"

echo
echo "########## CONSOLIDATE -> $RESULTS/ ##########"
$PY scripts/_consolidate_results.py
echo "DONE. Per-eval JSONs in $RESULTS/, merged view in $RESULTS/all_results.json"
