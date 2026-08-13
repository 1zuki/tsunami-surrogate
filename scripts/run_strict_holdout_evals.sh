#!/usr/bin/env bash
# Run strict held-out-family evaluations.
#
# The final suite passes --output-root for isolated, complete outputs. The
# historical shared-output modes remain available for repository users:
#   DEVICE=cuda bash scripts/run_strict_holdout_evals.sh
#   DEVICE=cuda bash scripts/run_strict_holdout_evals.sh --full-model-only
#   bash scripts/run_strict_holdout_evals.sh --summary-only

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT=""
RESULTS_DIR="${RESULTS_DIR:-results/strict_holdout}"
RUN_ACCURACY=1
RUN_PERFRAME=1
RUN_PHYSICS=1
CLEAN=1
FULL_MODEL_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-root)
      shift
      OUTPUT_ROOT="${1:-}"
      ;;
    --summary-only)
      RUN_ACCURACY=0
      RUN_PERFRAME=0
      RUN_PHYSICS=0
      CLEAN=0
      ;;
    --no-perframe)
      RUN_PERFRAME=0
      ;;
    --no-physics)
      RUN_PHYSICS=0
      ;;
    --no-clean)
      CLEAN=0
      ;;
    --full-model-only)
      FULL_MODEL_ONLY=1
      RUN_PERFRAME=0
      RUN_PHYSICS=0
      ;;
    -h|--help)
      sed -n '1,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

ISOLATED=0
if [ -n "$OUTPUT_ROOT" ]; then
  ISOLATED=1
  if [ "$RUN_ACCURACY" = 0 ] || [ "$RUN_PERFRAME" = 0 ] || \
    [ "$RUN_PHYSICS" = 0 ] || [ "$FULL_MODEL_ONLY" = 1 ]; then
    echo "--output-root requires the complete strict-holdout matrix." >&2
    exit 2
  fi
  if [ -e "$OUTPUT_ROOT" ]; then
    echo "Strict-holdout output root already exists: $OUTPUT_ROOT" >&2
    exit 1
  fi
  mkdir -p "$OUTPUT_ROOT"
else
  mkdir -p "$RESULTS_DIR"
fi

LABELS=(
  bathymetry_trench
  bathymetry_continental
  source_rough
  source_okada_like
)
HELDOUT_CONFIGS=(
  configs/model/fno_holdout_bathymetry_trench.yaml
  configs/model/fno_holdout_bathymetry_continental.yaml
  configs/model/fno_holdout_source_rough.yaml
  configs/model/fno_holdout_source_okada_like.yaml
)
ID_CONFIGS=(
  configs/model/fno_holdout_bathymetry_trench_eval_id.yaml
  configs/model/fno_holdout_bathymetry_continental_eval_id.yaml
  configs/model/fno_holdout_source_rough_eval_id.yaml
  configs/model/fno_holdout_source_okada_like_eval_id.yaml
)
FULL_CONFIGS=(
  configs/model/fno_full_on_holdout_bathymetry_trench.yaml
  configs/model/fno_full_on_holdout_bathymetry_continental.yaml
  configs/model/fno_full_on_holdout_source_rough.yaml
  configs/model/fno_full_on_holdout_source_okada_like.yaml
)
CHECKPOINTS=(
  experiments/fno_holdout/bathymetry_trench/best.pt
  experiments/fno_holdout/bathymetry_continental/best.pt
  experiments/fno_holdout/source_rough/best.pt
  experiments/fno_holdout/source_okada_like/best.pt
)
FULL_CHECKPOINT=experiments/fno/best.pt

run() {
  echo
  echo ">>> $*"
  "$@"
}

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  checkpoint="${CHECKPOINTS[$i]}"
  if [ "$ISOLATED" = 1 ]; then
    base="$OUTPUT_ROOT/$label"
    id_dir="$base/eval_id"
    heldout_dir="$base/eval_heldout"
    full_dir="$base/full_on_heldout"
  else
    id_dir="experiments/fno_holdout/$label/eval_id"
    heldout_dir="experiments/fno_holdout/$label/eval_heldout"
    full_dir="experiments/fno_full_on_holdout/$label/eval_heldout"
  fi
  mkdir -p "$id_dir" "$heldout_dir" "$full_dir"

  if [ "$CLEAN" = 1 ] && [ "$ISOLATED" = 0 ]; then
    if [ "$FULL_MODEL_ONLY" = 1 ]; then
      rm -f "$full_dir/metrics.json"
    else
      rm -f "$id_dir/metrics.json" "$id_dir/perframe.json" \
        "$id_dir/physics_diagnostics.json" \
        "$id_dir/physics_diagnostics_per_sample.csv" \
        "$heldout_dir/metrics.json" "$heldout_dir/perframe.json" \
        "$heldout_dir/physics_diagnostics.json" \
        "$heldout_dir/physics_diagnostics_per_sample.csv" \
        "$full_dir/metrics.json"
    fi
  fi

  if [ "$RUN_ACCURACY" = 1 ]; then
    if [ "$FULL_MODEL_ONLY" = 0 ]; then
      run "$PY" scripts/eval_accuracy.py \
        --config "${ID_CONFIGS[$i]}" \
        --checkpoint "$checkpoint" \
        --device "$DEVICE" \
        --output "$id_dir/metrics.json"
      run "$PY" scripts/eval_accuracy.py \
        --config "${HELDOUT_CONFIGS[$i]}" \
        --checkpoint "$checkpoint" \
        --device "$DEVICE" \
        --output "$heldout_dir/metrics.json"
    fi
    run "$PY" scripts/eval_accuracy.py \
      --config "${FULL_CONFIGS[$i]}" \
      --checkpoint "$FULL_CHECKPOINT" \
      --device "$DEVICE" \
      --output "$full_dir/metrics.json"
  fi

  if [ "$RUN_PERFRAME" = 1 ] && [ "$FULL_MODEL_ONLY" = 0 ]; then
    run "$PY" scripts/eval_perframe.py \
      --config "${ID_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE" \
      --output "$id_dir/perframe.json"
    run "$PY" scripts/eval_perframe.py \
      --config "${HELDOUT_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE" \
      --output "$heldout_dir/perframe.json"
  fi

  if [ "$RUN_PHYSICS" = 1 ] && [ "$FULL_MODEL_ONLY" = 0 ]; then
    run "$PY" scripts/eval_physics_diagnostics.py \
      --config "${ID_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE" \
      --output "$id_dir/physics_diagnostics.json" \
      --per-sample-output "$id_dir/physics_diagnostics_per_sample.csv"
    run "$PY" scripts/eval_physics_diagnostics.py \
      --config "${HELDOUT_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE" \
      --output "$heldout_dir/physics_diagnostics.json" \
      --per-sample-output "$heldout_dir/physics_diagnostics_per_sample.csv"
  fi
done

if [ "$ISOLATED" = 1 ]; then
  run "$PY" scripts/summarize_strict_holdout_evals.py \
    --eval-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/strict_holdout_summary.json" \
    --csv-output "$OUTPUT_ROOT/strict_holdout_summary.csv"
  SUMMARY_ROOT="$OUTPUT_ROOT"
else
  if [ "$CLEAN" = 1 ]; then
    rm -f "$RESULTS_DIR/strict_holdout_summary.json" \
      "$RESULTS_DIR/strict_holdout_summary.csv"
  fi
  run "$PY" scripts/summarize_strict_holdout_evals.py \
    --output "$RESULTS_DIR/strict_holdout_summary.json" \
    --csv-output "$RESULTS_DIR/strict_holdout_summary.csv"
  SUMMARY_ROOT="$RESULTS_DIR"
fi

echo
echo "DONE. Strict-holdout outputs: $SUMMARY_ROOT"
