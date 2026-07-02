#!/usr/bin/env bash
# Run the strict held-out family evaluation matrix for the hydrostatic FNO.
#
# This evaluates each trained strict-holdout checkpoint on:
#   1. the matched ID test split;
#   2. the held-out family test split.
#
# Outputs:
#   experiments/fno_holdout/<label>/eval_id/{metrics,perframe,physics_diagnostics}.json
#   experiments/fno_holdout/<label>/eval_heldout/{metrics,perframe,physics_diagnostics}.json
#   experiments/fno_full_on_holdout/<label>/eval_heldout/metrics.json
#   results/strict_holdout/strict_holdout_summary.{json,csv}
#
# Usage:
#   DEVICE=cuda bash scripts/run_strict_holdout_evals.sh
#   DEVICE=cuda bash scripts/run_strict_holdout_evals.sh --no-physics
#   DEVICE=cuda bash scripts/run_strict_holdout_evals.sh --no-perframe
#   bash scripts/run_strict_holdout_evals.sh --summary-only

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="${RESULTS_DIR:-results/strict_holdout}"
RUN_ACCURACY=1
RUN_PERFRAME=1
RUN_PHYSICS=1
CLEAN=1

while [ "$#" -gt 0 ]; do
  case "$1" in
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
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

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
EVAL_DIRS=(
  experiments/fno_holdout/bathymetry_trench/eval_id
  experiments/fno_holdout/bathymetry_trench/eval_heldout
  experiments/fno_full_on_holdout/bathymetry_trench/eval_heldout
  experiments/fno_holdout/bathymetry_continental/eval_id
  experiments/fno_holdout/bathymetry_continental/eval_heldout
  experiments/fno_full_on_holdout/bathymetry_continental/eval_heldout
  experiments/fno_holdout/source_rough/eval_id
  experiments/fno_holdout/source_rough/eval_heldout
  experiments/fno_full_on_holdout/source_rough/eval_heldout
  experiments/fno_holdout/source_okada_like/eval_id
  experiments/fno_holdout/source_okada_like/eval_heldout
  experiments/fno_full_on_holdout/source_okada_like/eval_heldout
)

run() {
  echo
  echo ">>> $*"
  local tmp
  tmp="$(mktemp)"
  if "$@" >"$tmp" 2>&1; then
    grep -vE "shard-aware|batch sampler" "$tmp" | tail -5 || true
    rm -f "$tmp"
  else
    cat "$tmp" >&2
    rm -f "$tmp"
    return 1
  fi
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

mkdir -p "$RESULTS_DIR"

for i in "${!LABELS[@]}"; do
  require_file "${HELDOUT_CONFIGS[$i]}"
  require_file "${ID_CONFIGS[$i]}"
  require_file "${FULL_CONFIGS[$i]}"
  require_file "${CHECKPOINTS[$i]}"
done
require_file "$FULL_CHECKPOINT"

if [ "$CLEAN" = 1 ]; then
  echo "########## CLEAN STRICT-HOLDOUT EVAL OUTPUTS ##########"
  for d in "${EVAL_DIRS[@]}"; do
    rm -f "$d/metrics.json" \
          "$d/perframe.json" \
          "$d/physics_diagnostics.json" \
          "$d/physics_diagnostics_per_sample.csv"
  done
  rm -f "$RESULTS_DIR/strict_holdout_summary.json" \
        "$RESULTS_DIR/strict_holdout_summary.csv"
fi

if [ "$RUN_ACCURACY" = 1 ]; then
  echo "########## STRICT HOLDOUT ACCURACY ##########"
  for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    checkpoint="${CHECKPOINTS[$i]}"
    run "$PY" scripts/eval_accuracy.py \
      --config "${ID_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE"
    run "$PY" scripts/eval_accuracy.py \
      --config "${HELDOUT_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE"
    run "$PY" scripts/eval_accuracy.py \
      --config "${FULL_CONFIGS[$i]}" \
      --checkpoint "$FULL_CHECKPOINT" \
      --device "$DEVICE"
    echo "[strict-holdout] accuracy done: $label"
  done
fi

if [ "$RUN_PERFRAME" = 1 ]; then
  echo "########## STRICT HOLDOUT PER-FRAME CURVES ##########"
  for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    checkpoint="${CHECKPOINTS[$i]}"
    run "$PY" scripts/eval_perframe.py \
      --config "${ID_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE"
    run "$PY" scripts/eval_perframe.py \
      --config "${HELDOUT_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE"
    echo "[strict-holdout] per-frame done: $label"
  done
else
  echo "########## STRICT HOLDOUT PER-FRAME SKIPPED ##########"
fi

if [ "$RUN_PHYSICS" = 1 ]; then
  echo "########## STRICT HOLDOUT PHYSICS DIAGNOSTICS ##########"
  for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    checkpoint="${CHECKPOINTS[$i]}"
    run "$PY" scripts/eval_physics_diagnostics.py \
      --config "${ID_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE"
    run "$PY" scripts/eval_physics_diagnostics.py \
      --config "${HELDOUT_CONFIGS[$i]}" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE"
    echo "[strict-holdout] physics done: $label"
  done
else
  echo "########## STRICT HOLDOUT PHYSICS SKIPPED ##########"
fi

echo "########## STRICT HOLDOUT SUMMARY ##########"
run "$PY" scripts/summarize_strict_holdout_evals.py \
  --output "$RESULTS_DIR/strict_holdout_summary.json" \
  --csv-output "$RESULTS_DIR/strict_holdout_summary.csv"

echo
echo "DONE. Strict-holdout summary:"
echo "  $RESULTS_DIR/strict_holdout_summary.json"
echo "  $RESULTS_DIR/strict_holdout_summary.csv"
