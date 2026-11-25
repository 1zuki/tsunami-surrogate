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
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.venv}"

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/slurm/environment.sh"
export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"

echo "Host          : $(hostname)"
echo "PATH          : $PATH"
echo "Python        : $(which python)"
echo "nvidia-smi    : $(command -v nvidia-smi || echo missing)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable inside the compute allocation"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

python - <<'CHECK_GPU'
import torch
assert torch.cuda.is_available(), "No CUDA device is visible in this allocation"
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("Allocated device:", torch.cuda.get_device_name(0))
CHECK_GPU
