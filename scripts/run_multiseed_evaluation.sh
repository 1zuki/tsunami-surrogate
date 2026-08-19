#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/multiseed_v2}"
PREFLIGHT_ONLY=0

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [--preflight-only]" >&2
  exit 2
fi
if [ "$#" -eq 1 ]; then
  if [ "$1" != "--preflight-only" ]; then
    echo "Unknown argument: $1" >&2
    echo "Usage: $0 [--preflight-only]" >&2
    exit 2
  fi
  PREFLIGHT_ONLY=1
fi

cd "$ROOT"

if [ ! -x "$PY" ]; then
  echo "Python environment is missing or not executable: $PY" >&2
  exit 1
fi

checkpoint() {
  local model="$1"
  local seed="$2"
  case "$model:$seed" in
    fno_hydrostatic:18) echo experiments/fno/best.pt ;;
    fno_muscl_hr:18) echo experiments/fno_muscl_hr/fno_muscl_hr_seed_18/best.pt ;;
    fno_boussinesq:18) echo experiments/fno_boussinesq/best.pt ;;
    ffno_hydrostatic:18) echo experiments/ffno/best.pt ;;
    unet_hydrostatic:18) echo experiments/unet/best.pt ;;
    convlstm_hydrostatic:18) echo experiments/convlstm/best.pt ;;
    *)
      echo "experiments/multiseed_v2/$model/${model}_seed_${seed}/best.pt"
      ;;
  esac
}

require_completed_checkpoint() {
  local model="$1"
  local seed="$2"
  local path
  path="$(checkpoint "$model" "$seed")"
  require_checkpoint "$path"
  "$PY" - "$path" "$seed" <<'PY'
import sys

from scripts.eval_suite_preflight import PreflightError, validate_completed_checkpoint

path = sys.argv[1]
seed = int(sys.argv[2])
try:
    summary = validate_completed_checkpoint(path, expected_seed=seed)
except PreflightError as exc:
    raise SystemExit(f"[multiseed-preflight] failed: {exc}") from exc
print(
    "[multiseed-preflight] "
    f"checkpoint={path} seed={seed} completion={summary['completion']} "
    f"best_epoch={summary['best_epoch']} last_epoch={summary['last_epoch']}"
)
PY
}

require_checkpoint() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing checkpoint: $path" >&2
    exit 1
  fi
}

run_seed_metrics() {
  local label="$1"
  local reference="$2"
  local config="$3"
  local dataset="$4"
  local batch_size="$5"
  shift 5
  local args=()
  local path
  for path in "$@"; do
    require_checkpoint "$path"
    args+=(--checkpoint "$path")
  done
  "$PY" -u scripts/eval_v2_seed_metrics.py \
    --config "$config" \
    "${args[@]}" \
    --dataset "$dataset" \
    --label "$label" \
    --reference "$reference" \
    --expected-seeds 18 36 67 \
    --batch-size "$batch_size" \
    --bootstrap-seed 20260813 \
    --bootstrap-resamples 2000 \
    --device "$DEVICE" \
    --output "$OUTPUT_ROOT/seed_metrics/${label}.json"
}

MODELS=(
  convlstm_hydrostatic
  fno_hydrostatic
  fno_muscl_hr
  fno_boussinesq
  ffno_hydrostatic
  unet_hydrostatic
)

for model in "${MODELS[@]}"; do
  for seed in 18 36 67; do
    require_completed_checkpoint "$model" "$seed"
  done
done

echo "[multiseed-preflight] status=passed models=${#MODELS[@]} checkpoints=18"
if [ "$PREFLIGHT_ONLY" = 1 ]; then
  exit 0
fi

run_seed_metrics \
  fno_hydrostatic hydrostatic configs/model/fno.yaml \
  data/processed/hydrostatic/test 64 \
  "$(checkpoint fno_hydrostatic 18)" \
  "$(checkpoint fno_hydrostatic 36)" \
  "$(checkpoint fno_hydrostatic 67)"
run_seed_metrics \
  fno_muscl_hr muscl_hr configs/model/fno_muscl_hr.yaml \
  data/processed/muscl_hr/test 64 \
  "$(checkpoint fno_muscl_hr 18)" \
  "$(checkpoint fno_muscl_hr 36)" \
  "$(checkpoint fno_muscl_hr 67)"
run_seed_metrics \
  fno_boussinesq boussinesq configs/model/fno_boussinesq.yaml \
  data/processed/boussinesq/test 64 \
  "$(checkpoint fno_boussinesq 18)" \
  "$(checkpoint fno_boussinesq 36)" \
  "$(checkpoint fno_boussinesq 67)"
run_seed_metrics \
  ffno_hydrostatic hydrostatic configs/model/ffno.yaml \
  data/processed/hydrostatic/test 64 \
  "$(checkpoint ffno_hydrostatic 18)" \
  "$(checkpoint ffno_hydrostatic 36)" \
  "$(checkpoint ffno_hydrostatic 67)"
run_seed_metrics \
  unet_hydrostatic hydrostatic configs/model/unet.yaml \
  data/processed/hydrostatic/test 64 \
  "$(checkpoint unet_hydrostatic 18)" \
  "$(checkpoint unet_hydrostatic 36)" \
  "$(checkpoint unet_hydrostatic 67)"
run_seed_metrics \
  convlstm_hydrostatic hydrostatic configs/model/convlstm.yaml \
  data/processed/hydrostatic/test 8 \
  "$(checkpoint convlstm_hydrostatic 18)" \
  "$(checkpoint convlstm_hydrostatic 36)" \
  "$(checkpoint convlstm_hydrostatic 67)"

for seed in 18 36 67; do
  "$PY" -u scripts/eval_v2_reference_analysis.py \
    --model "hydrostatic|configs/model/fno.yaml|$(checkpoint fno_hydrostatic "$seed")|data/processed/hydrostatic/test" \
    --model "muscl_hr|configs/model/fno_muscl_hr.yaml|$(checkpoint fno_muscl_hr "$seed")|data/processed/muscl_hr/test" \
    --model "boussinesq|configs/model/fno_boussinesq.yaml|$(checkpoint fno_boussinesq "$seed")|data/processed/boussinesq/test" \
    --training-seed "$seed" \
    --bootstrap-seed 20260813 \
    --bootstrap-resamples 2000 \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/reference_analysis/seed_${seed}"
done

"$PY" -u scripts/summarize_v2_multiseed_reference.py \
  --input "$OUTPUT_ROOT/reference_analysis/seed_18/cross_reference.json" \
  --input "$OUTPUT_ROOT/reference_analysis/seed_36/cross_reference.json" \
  --input "$OUTPUT_ROOT/reference_analysis/seed_67/cross_reference.json" \
  --output "$OUTPUT_ROOT/reference_analysis/multiseed_summary.json"

echo "[multiseed-evaluation] status=completed output_root=$OUTPUT_ROOT"
