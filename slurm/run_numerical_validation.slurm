#!/bin/bash
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --qos=mig1-18
#SBATCH --job-name=tsunami-numval
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="evaluation_runs/numerical_validation_final_v2_20260910-r1"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/slurm/environment.sh"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Start time : $(date --iso-8601=seconds)"
echo "Host       : $(command -v hostname >/dev/null 2>&1 && hostname || echo unknown)"
echo "Commit     : $(git rev-parse HEAD)"
echo "Python     : $PYTHON"
echo "MPI        : $(command -v mpiexec)"

"$PYTHON" -u scripts/run_numerical_validation_chain.py \
  --allow-unvalidated-contract \
  --output-root "$RUN_ROOT" \
  --workers "${SLURM_CPUS_PER_TASK:-4}" \
  --geoclaw-workers 2 \
  --claw-root "$PROJECT_ROOT/.tools/clawpack-v5.14.0" \
  --petsc-dir "$PROJECT_ROOT/.tools/petsc-3.25.3" \
  --petsc-arch arch-linux-c-opt \
  --geoclaw-python "$PYTHON"

echo "End time   : $(date --iso-8601=seconds)"
