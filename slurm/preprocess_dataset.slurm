#!/bin/bash
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_preprocess_%j.out
#SBATCH --error=logs/slurm/%x_preprocess_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must name a preprocessing config}"
cd "$PROJECT_ROOT"

export PATH="$PROJECT_ROOT/.venv/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Missing preprocessing config: $CONFIG_PATH"
    exit 1
fi

echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Host        : $(hostname)"
echo "Start time  : $(date --iso-8601=seconds)"
echo "Config      : $CONFIG_PATH"
echo "Git commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Python      : $(command -v python)"

python -u -m src.data_gen.preprocess --config "$CONFIG_PATH"

echo "End time    : $(date --iso-8601=seconds)"
