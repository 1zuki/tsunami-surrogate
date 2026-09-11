#!/bin/bash
#SBATCH --job-name=tsunami-numval
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
RUN_ROOT="evaluation_runs/numerical_validation_final_v2_20260910-r1"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
export PATH="/cm/shared/apps/mvapich2/gcc/64/2.3.7/bin:/cm/shared/apps/openpbs/23.06.06/bin:/usr/bin:/bin:$PATH"
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
  --workers 8 \
  --geoclaw-workers 4 \
  --claw-root "$PROJECT_ROOT/.tools/clawpack-v5.14.0" \
  --petsc-dir "$PROJECT_ROOT/.tools/petsc-3.25.3" \
  --petsc-arch arch-linux-c-opt \
  --geoclaw-python "$PYTHON"

echo "End time   : $(date --iso-8601=seconds)"
