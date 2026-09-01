#!/bin/bash
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:2
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
ENV_PREFIX="$PROJECT_ROOT/.venv"
RUNS_FILE="${RUNS_FILE:?RUNS_FILE must be exported by the suite submitter}"
SUITE_ID="${SUITE_ID:?SUITE_ID must be exported by the suite submitter}"
REQUIRED_VRAM="${REQUIRED_VRAM:-16384}"

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
if type module >/dev/null 2>&1; then
    module clear -f
fi

source /datastore/vit/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

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
if [ ! -x /usr/local/bin/gpu_check.sh ]; then
    echo "Missing compute-node helper: /usr/local/bin/gpu_check.sh"
    exit 1
fi
if grep -Fq "nvidia-smi-i " /usr/local/bin/gpu_check.sh; then
    echo "gpu_check.sh contains invalid command 'nvidia-smi-i'"
    echo "Ask the cluster administrator to change it to 'nvidia-smi -i'"
    echo "Refusing to call the helper because it would requeue this job forever"
    exit 1
fi

echo "Suite         : $SUITE_ID"
echo "Run label     : $RUN_LABEL"
echo "Array job     : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array task    : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Host          : $(hostname)"
echo "Start time    : $(date --iso-8601=seconds)"
echo "Working dir   : $(pwd)"
echo "Config        : $CONFIG_PATH"
echo "Output dir    : $OUTPUT_DIR"
echo "Training seed : $RUN_SEED"
echo "Git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Python        : $(which python)"
echo "Python ver.   : $(python --version)"
echo "nvidia-smi    : $(command -v nvidia-smi)"

unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(env PATH="/usr/bin:/bin:$PATH" \
    /usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$SLURM_JOB_ID")
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ "$EXIT_CODE" -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "gpu_check.sh failed with exit code $EXIT_CODE"
    echo "$CHECK_OUT"
    exit "$EXIT_CODE"
fi
if [[ ! "$CHECK_OUT" =~ ^[0-9]+$ ]]; then
    echo "gpu_check.sh returned an invalid GPU index: $CHECK_OUT"
    exit 1
fi

BEST_GPU="$CHECK_OUT"
export CUDA_VISIBLE_DEVICES="$BEST_GPU"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job${SLURM_JOB_ID}"

cleanup() {
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
trap cleanup EXIT

cleanup
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "Selected GPU  : $BEST_GPU"
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
