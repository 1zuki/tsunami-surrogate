#!/bin/bash
#SBATCH --job-name=tsunami-strict-eval
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:20
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
ENV_PREFIX="$PROJECT_ROOT/.venv"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must name a new evaluation directory}"
REQUIRED_VRAM="${REQUIRED_VRAM:-16384}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

cd "$PROJECT_ROOT"

if type module >/dev/null 2>&1; then
    module clear -f
fi

source /datastore/vit/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -e "$OUTPUT_ROOT" ]; then
    echo "Evaluation output already exists: $OUTPUT_ROOT"
    exit 1
fi
if ! [[ "$HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "HEARTBEAT_SECONDS must be a positive integer"
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing to run GPU evaluation"
    exit 1
fi
if [ ! -x /usr/local/bin/gpu_check.sh ]; then
    echo "Missing compute-node helper: /usr/local/bin/gpu_check.sh"
    exit 1
fi
if grep -Fq "nvidia-smi-i " /usr/local/bin/gpu_check.sh; then
    echo "gpu_check.sh contains invalid command 'nvidia-smi-i'"
    exit 1
fi

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

echo "Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "Host         : $(hostname)"
echo "Start time   : $(date --iso-8601=seconds)"
echo "Working dir  : $(pwd)"
echo "Output root  : $OUTPUT_ROOT"
echo "Selected GPU : $BEST_GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

set +e
env DEVICE=cuda PY="$ENV_PREFIX/bin/python" \
    bash scripts/run_strict_holdout_evals.sh \
    --output-root "$OUTPUT_ROOT" &
EVAL_PID=$!

while kill -0 "$EVAL_PID" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$EVAL_PID" 2>/dev/null; then
        FILE_COUNT=$(find "$OUTPUT_ROOT" -type f 2>/dev/null | wc -l)
        echo "[heartbeat] $(date --iso-8601=seconds) files=$FILE_COUNT"
    fi
done

wait "$EVAL_PID"
EVAL_EXIT=$?
set -e

echo "End time     : $(date --iso-8601=seconds)"
echo "Evaluation exit: $EVAL_EXIT"
exit "$EVAL_EXIT"
