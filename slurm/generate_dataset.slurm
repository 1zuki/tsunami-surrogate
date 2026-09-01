#!/bin/bash
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/%x_generate_%j.out
#SBATCH --error=logs/slurm/%x_generate_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
ENV_PREFIX="$PROJECT_ROOT/.venv"
SPLIT="${SPLIT:?SPLIT must be train, eval, or test}"
RESUME="${RESUME:-0}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-}"

if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "$ALLOCATED_CPUS" ]; then
    echo "This generator must run inside a Slurm allocation"
    exit 1
fi
if ! [[ "$ALLOCATED_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid Slurm CPU allocation: $ALLOCATED_CPUS"
    exit 1
fi

NUM_WORKERS="${NUM_WORKERS:-$ALLOCATED_CPUS}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-$NUM_WORKERS}"
if ! [[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_WORKERS must be a positive integer"
    exit 1
fi
if ! [[ "$MAX_IN_FLIGHT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_IN_FLIGHT must be a positive integer"
    exit 1
fi
if [ "$NUM_WORKERS" -gt "$ALLOCATED_CPUS" ]; then
    echo "NUM_WORKERS=$NUM_WORKERS exceeds allocated CPUs=$ALLOCATED_CPUS"
    exit 1
fi
if [ "$MAX_IN_FLIGHT" -lt "$NUM_WORKERS" ]; then
    echo "MAX_IN_FLIGHT must be at least NUM_WORKERS"
    exit 1
fi
if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
    echo "RESUME must be 0 or 1"
    exit 1
fi

case "$SPLIT" in
    train)
        CONFIG_PATH="configs/data/dataset.yaml"
        ;;
    eval)
        CONFIG_PATH="configs/data/dataset_eval.yaml"
        ;;
    test)
        CONFIG_PATH="configs/data/dataset_test.yaml"
        ;;
    *)
        echo "Unknown SPLIT=$SPLIT; expected train, eval, or test"
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"
if type module >/dev/null 2>&1; then
    module clear -f
fi
source /datastore/vit/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p logs/data
RUN_LOG="logs/data/${SPLIT}-generation_${SLURM_JOB_ID}.log"
GENERATION_CMD=(
    python -u scripts/make_dataset.py
    --config "$CONFIG_PATH"
    --num-workers "$NUM_WORKERS"
    --max-in-flight "$MAX_IN_FLIGHT"
)
if [ "$RESUME" = "1" ]; then
    GENERATION_CMD+=(--continue)
fi

echo "Job ID        : $SLURM_JOB_ID"
echo "Host          : $(hostname)"
echo "Start time    : $(date --iso-8601=seconds)"
echo "Working dir   : $(pwd)"
echo "Git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Git status    : $(git status --porcelain --untracked-files=no | wc -l) tracked changes"
echo "Python        : $(which python)"
echo "Split         : $SPLIT"
echo "Config        : $CONFIG_PATH"
echo "Allocated CPUs: $ALLOCATED_CPUS"
echo "Workers       : $NUM_WORKERS"
echo "Max in flight : $MAX_IN_FLIGHT"
echo "Resume        : $RESUME"
echo "Run log       : $RUN_LOG"
echo "Command       : ${GENERATION_CMD[*]}"

START_EPOCH=$(date +%s)
set +e
"${GENERATION_CMD[@]}" 2>&1 | tee "$RUN_LOG"
GENERATION_EXIT=${PIPESTATUS[0]}
set -e
END_EPOCH=$(date +%s)

echo "End time      : $(date --iso-8601=seconds)"
echo "Elapsed sec   : $((END_EPOCH - START_EPOCH))"
echo "Generator exit: $GENERATION_EXIT"
exit "$GENERATION_EXIT"
