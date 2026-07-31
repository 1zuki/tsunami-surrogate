#!/bin/bash
#SBATCH --job-name=tsunami-fno
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:2
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
ENV_PREFIX="$PROJECT_ROOT/.venv"
CONFIG_PATH="${CONFIG_PATH:-configs/cluster/fno_smoke.yaml}"
RESUME_PATH="${RESUME_PATH:-}"
REQUIRED_VRAM="${REQUIRED_VRAM:-16384}"

cd "$PROJECT_ROOT"

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

echo "Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "Job name     : ${SLURM_JOB_NAME:-N/A}"
echo "Host         : $(hostname)"
echo "Start time   : $(date --iso-8601=seconds)"
echo "Working dir  : $(pwd)"
echo "Config       : $CONFIG_PATH"
echo "Git commit   : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Python       : $(which python)"
echo "Python ver.  : $(python --version)"
echo "nvidia-smi   : $(command -v nvidia-smi || echo missing)"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Missing config: $CONFIG_PATH"
    exit 1
fi
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

echo "Selected GPU : $BEST_GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('CUDA version', torch.version.cuda); print('device count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

START_EPOCH=$(date +%s)
set +e
if [ -n "$RESUME_PATH" ]; then
    if [ ! -f "$RESUME_PATH" ]; then
        echo "Missing resume checkpoint: $RESUME_PATH"
        exit 1
    fi
    python -u scripts/train.py --config "$CONFIG_PATH" --resume "$RESUME_PATH"
else
    python -u scripts/train.py --config "$CONFIG_PATH"
fi
TRAIN_EXIT=$?
set -e
END_EPOCH=$(date +%s)

echo "End time     : $(date --iso-8601=seconds)"
echo "Elapsed sec  : $((END_EPOCH - START_EPOCH))"
echo "Training exit: $TRAIN_EXIT"
exit "$TRAIN_EXIT"
