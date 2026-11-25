#!/bin/bash
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --qos=mig1-18
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_real_bath_%j.out
#SBATCH --error=logs/slurm/%x_real_bath_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/slurm/environment.sh"

export PATH="$PROJECT_ROOT/.venv/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p logs/data
RUN_LOG="logs/data/real-bathymetry-rebuild_${SLURM_JOB_ID}.log"

echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Host        : $(hostname)"
echo "Start time  : $(date --iso-8601=seconds)"
echo "Git commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Run log     : $RUN_LOG"

set +e
python -u scripts/build_real_bathymetry_eval.py \
    --num-workers "${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" 2>&1 | tee "$RUN_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "End time    : $(date --iso-8601=seconds)"
echo "Exit code   : $EXIT_CODE"
exit "$EXIT_CODE"
