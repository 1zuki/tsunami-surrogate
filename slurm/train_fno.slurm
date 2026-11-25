#!/bin/bash
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.venv}"
CONFIG_PATH="${CONFIG_PATH:-configs/cluster/training_preflight.yaml}"
RESUME_PATH="${RESUME_PATH:-}"

cd "$PROJECT_ROOT"


source "$PROJECT_ROOT/slurm/environment.sh"

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


echo "Selected GPU : ${CUDA_VISIBLE_DEVICES:-Slurm-managed}"
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
