#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

CONFIGS=(
  configs/model/multiseed/fno_hydrostatic.yaml
  configs/model/multiseed/fno_muscl_hr.yaml
  configs/model/multiseed/fno_boussinesq.yaml
  configs/model/multiseed/ffno_hydrostatic.yaml
  configs/model/multiseed/unet_hydrostatic.yaml
)

# ConvLSTM is intentionally excluded from this queue because its three
# checkpoints are complete and its recurrent training cost is substantially
# higher. Use configs/model/multiseed/convlstm_hydrostatic.yaml explicitly
# only when a local retrain or resume is actually required.

cd "$ROOT"

if [ ! -x "$PY" ]; then
  echo "Python environment is missing or not executable: $PY" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to start the overnight queue")
print(f"[multiseed] cuda_device={torch.cuda.get_device_name(0)}")
PY

echo "[multiseed] start=$(date --iso-8601=seconds)"
echo "[multiseed] python=$PY"
echo "[multiseed] configs=${#CONFIGS[@]} new_runs=10 seeds=36,67"
echo "[multiseed] existing seed 18 checkpoints are reused, not retrained"

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  echo
  echo "[multiseed] config=$((index + 1))/${#CONFIGS[@]} path=$config"
  "$PY" -u scripts/train.py --config "$config"
done

echo
echo "[multiseed] completed=$(date --iso-8601=seconds)"
