#!/bin/bash
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_real_bath_%j.out
#SBATCH --error=logs/slurm/%x_real_bath_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
cd "$PROJECT_ROOT"

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
    --num-workers "${NUM_WORKERS:-8}" 2>&1 | tee "$RUN_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "End time    : $(date --iso-8601=seconds)"
echo "Exit code   : $EXIT_CODE"
exit "$EXIT_CODE"
