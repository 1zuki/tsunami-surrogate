#!/bin/bash
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --job-name=tsunami-full
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.venv}"
MPI_ROOT="${MPI_ROOT:-}"
GCC_ROOT="${GCC_ROOT:-}"
RUN_ID="${RUN_ID:?RUN_ID must name a fresh immutable evaluation run}"

cd "$PROJECT_ROOT"


source "$PROJECT_ROOT/slurm/environment.sh"

export PATH="$ENV_PREFIX/bin:${MPI_ROOT:+$MPI_ROOT/bin:}${GCC_ROOT:+$GCC_ROOT/bin:}$PATH"
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export CLAW_ROOT="${CLAW_ROOT:-$PROJECT_ROOT/.tools/clawpack-v5.14.0}"
export PETSC_DIR="${PETSC_DIR:-$PROJECT_ROOT/.tools/petsc-3.25.3}"
export PETSC_ARCH="arch-linux-c-opt"
export GEOCLAW_PYTHON="$ENV_PREFIX/bin/python"
export LD_LIBRARY_PATH="${MPI_ROOT:+$MPI_ROOT/lib:}$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"
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


echo "Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "Host         : $(hostname)"
echo "Start time   : $(date --iso-8601=seconds)"
echo "Working dir  : $(pwd)"
echo "Run ID       : $RUN_ID"
echo "Git commit   : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Selected GPU : ${CUDA_VISIBLE_DEVICES:-Slurm-managed}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

bash scripts/run_eval_suite.sh \
    --execute \
    --run-id "$RUN_ID" \
    --device cuda \
    --include-ensemble \
    --include-paper-evidence \
    --deep-payload-audit

echo "End time     : $(date --iso-8601=seconds)"
