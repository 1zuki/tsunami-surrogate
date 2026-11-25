#!/usr/bin/env bash
# Source from a job submitted at the repository root, or set PROJECT_ROOT.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
if [ ! -f requirements.txt ] || [ ! -d tsunami_surrogate ]; then
    echo "Submit from the repository root, or export PROJECT_ROOT" >&2
    return 1
fi
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_ROOT/.venv}"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "Missing project environment: $ENV_PREFIX" >&2
    return 1
fi
source "$ENV_PREFIX/bin/activate"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TMPDIR="$PROJECT_ROOT/.tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"
