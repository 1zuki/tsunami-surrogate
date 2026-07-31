#!/bin/bash
#SBATCH --job-name=tsu-gpu-probe
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=mps:1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
ENV_PREFIX="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"
if type module >/dev/null 2>&1; then
    module clear -f
fi
source /datastore/vit/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
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

if [ ! -r /usr/local/bin/gpu_check.sh ]; then
    echo "gpu_check.sh is unavailable on this compute node"
    exit 1
fi

echo "gpu_check.sh  : /usr/local/bin/gpu_check.sh"
echo "Syntax check  :"
bash -n /usr/local/bin/gpu_check.sh
echo "Helper lines 20-45:"
nl -ba /usr/local/bin/gpu_check.sh | sed -n '20,45p'

if grep -Fq "nvidia-smi-i " /usr/local/bin/gpu_check.sh; then
    echo "ERROR: gpu_check.sh contains 'nvidia-smi-i'; the administrator"
    echo "must change it to 'nvidia-smi -i' before training jobs are submitted"
    exit 1
fi

echo "Known command check: passed"
