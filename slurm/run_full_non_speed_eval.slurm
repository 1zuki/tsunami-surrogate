#!/bin/bash
#SBATCH --job-name=tsunami-full
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=mps:a100:20
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
ENV_PREFIX="$PROJECT_ROOT/.venv"
RUN_ID="${RUN_ID:?RUN_ID must name a fresh immutable evaluation run}"
REQUIRED_VRAM="${REQUIRED_VRAM:-16384}"

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
export CLAW_ROOT="$PROJECT_ROOT/.tools/clawpack-v5.14.0"
export PETSC_DIR="$PROJECT_ROOT/.tools/petsc-3.25.3"
export PETSC_ARCH="arch-linux-c-opt"
export GEOCLAW_PYTHON="$ENV_PREFIX/bin/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -e "evaluation_runs/$RUN_ID" ] || [ -e "evaluation_runs/$RUN_ID.staging" ]; then
    echo "Evaluation run already exists: $RUN_ID"
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

export CUDA_VISIBLE_DEVICES="$CHECK_OUT"
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
echo "Run ID       : $RUN_ID"
echo "Git commit   : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Selected GPU : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

bash scripts/run_eval_suite.sh \
    --execute \
    --run-id "$RUN_ID" \
    --device cuda \
    --include-ensemble \
    --include-paper-evidence \
    --deep-payload-audit \
    --rerun-numerical-validation

echo "End time     : $(date --iso-8601=seconds)"
