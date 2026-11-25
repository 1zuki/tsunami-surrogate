#!/bin/bash
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.venv}"
RUNS_FILE="${RUNS_FILE:?RUNS_FILE must be exported by the suite submitter}"
SUITE_ID="${SUITE_ID:?SUITE_ID must be exported by the suite submitter}"

cd "$PROJECT_ROOT"

if [ ! -f "$RUNS_FILE" ]; then
    echo "Missing suite run manifest: $RUNS_FILE"
    exit 1
fi

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
RUN_LINE=$(sed -n "${LINE_NUMBER}p" "$RUNS_FILE")
if [ -z "$RUN_LINE" ]; then
    echo "No run entry for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
IFS=$'\t' read -r RUN_LABEL CONFIG_PATH OUTPUT_DIR RUN_SEED <<< "$RUN_LINE"

if [ -z "$RUN_LABEL" ] || [ -z "$CONFIG_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$RUN_SEED" ]; then
    echo "Malformed run entry: $RUN_LINE"
    exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Missing generated config: $CONFIG_PATH"
    exit 1
fi

source "$PROJECT_ROOT/slurm/environment.sh"

export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing to requeue blindly"
    exit 1
fi


echo "Selected GPU  : ${CUDA_VISIBLE_DEVICES:-Slurm-managed}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('CUDA version', torch.version.cuda); print('device count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

set +e
RUN_STATE=$(python scripts/cluster_run_state.py \
    --output-dir "$OUTPUT_DIR" \
    --expected-seed "$RUN_SEED" \
    --config "$CONFIG_PATH")
STATE_EXIT=$?
set -e
if [ "$STATE_EXIT" -ne 0 ]; then
    echo "Run-state validation failed: $RUN_STATE"
    exit "$STATE_EXIT"
fi
IFS=$'\t' read -r RUN_ACTION RUN_REASON RESUME_PATH <<< "$RUN_STATE"
echo "Run action    : $RUN_ACTION"
echo "Run reason    : $RUN_REASON"
if [ "$RUN_ACTION" = "skip" ]; then
    echo "Completed run already exists; nothing to do"
    exit 0
fi
if [ "$RUN_ACTION" != "fresh" ] && [ "$RUN_ACTION" != "resume" ]; then
    echo "Unexpected run action: $RUN_ACTION"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/cluster_console.log"
RUN_ERR="$OUTPUT_DIR/cluster_console.err"
TRAIN_CMD=(python -u scripts/train.py --config "$CONFIG_PATH")
if [ "$RUN_ACTION" = "resume" ]; then
    TRAIN_CMD+=(--resume "$RESUME_PATH")
fi

START_EPOCH=$(date +%s)
set +e
"${TRAIN_CMD[@]}" 2> >(tee -a "$RUN_ERR" >&2) | tee -a "$RUN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
END_EPOCH=$(date +%s)

echo "End time      : $(date --iso-8601=seconds)"
echo "Elapsed sec   : $((END_EPOCH - START_EPOCH))"
echo "Training exit : $TRAIN_EXIT"
exit "$TRAIN_EXIT"
