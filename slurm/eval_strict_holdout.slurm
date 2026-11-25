#!/bin/bash
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --job-name=tsunami-strict-eval
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.venv}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must name a new evaluation directory}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

cd "$PROJECT_ROOT"


source "$PROJECT_ROOT/slurm/environment.sh"

export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -e "$OUTPUT_ROOT" ]; then
    echo "Evaluation output already exists: $OUTPUT_ROOT"
    exit 1
fi
if ! [[ "$HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "HEARTBEAT_SECONDS must be a positive integer"
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
echo "Output root  : $OUTPUT_ROOT"
echo "Selected GPU : ${CUDA_VISIBLE_DEVICES:-Slurm-managed}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

set +e
env DEVICE=cuda PY="$ENV_PREFIX/bin/python" \
    bash scripts/run_strict_holdout_evals.sh \
    --output-root "$OUTPUT_ROOT" &
EVAL_PID=$!

while kill -0 "$EVAL_PID" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$EVAL_PID" 2>/dev/null; then
        FILE_COUNT=$(find "$OUTPUT_ROOT" -type f 2>/dev/null | wc -l)
        echo "[heartbeat] $(date --iso-8601=seconds) files=$FILE_COUNT"
    fi
done

wait "$EVAL_PID"
EVAL_EXIT=$?
set -e

echo "End time     : $(date --iso-8601=seconds)"
echo "Evaluation exit: $EVAL_EXIT"
exit "$EVAL_EXIT"
