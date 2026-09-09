#!/bin/bash
# Build the pinned GeoClaw/PETSc runtime under the project directory.
#
# This job is intentionally separate from dataset generation.  It installs
# source/build artifacts under .tools/, leaves system packages untouched, and
# refuses to reuse a tree at an unexpected revision.
#SBATCH --job-name=tsunami-geoclaw-setup
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
TOOLS_ROOT="$PROJECT_ROOT/.tools"
CLAW_ROOT="$TOOLS_ROOT/clawpack-v5.14.0"
PETSC_DIR="$TOOLS_ROOT/petsc-3.25.3"
PETSC_ARCH="arch-linux-c-opt"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

EXPECTED_CLAW_COMMIT="5ce53b4814a3e03536fa17e8ae05e65960194c56"
EXPECTED_GEOCLAW_COMMIT="11479f675dd8bb19066a2430c28206999cebf299"
EXPECTED_PETSC_COMMIT="fd2ca03442be2a78130b411057f3b922130f6b1b"

cd "$PROJECT_ROOT"
mkdir -p "$TOOLS_ROOT" logs/slurm

if type module >/dev/null 2>&1; then
    module clear -f
fi
module load gcc/14.2.0
module load openmpi/gcc/64/4.1.8

export PATH="$PROJECT_ROOT/.venv/bin:/usr/bin:/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PETSC_DIR
export PETSC_ARCH

if [ ! -f .git/info/exclude ] || ! grep -Fxq ".tools/" .git/info/exclude; then
    printf '%s\n' '.tools/' >> .git/info/exclude
fi

clone_or_check_clawpack() {
    if [ -d "$CLAW_ROOT/.git" ]; then
        local commit geoclaw_commit
        commit=$(git -C "$CLAW_ROOT" rev-parse HEAD)
        geoclaw_commit=$(git -C "$CLAW_ROOT/geoclaw" rev-parse HEAD)
        if [ "$commit" != "$EXPECTED_CLAW_COMMIT" ] ||
           [ "$geoclaw_commit" != "$EXPECTED_GEOCLAW_COMMIT" ]; then
            echo "Existing Clawpack tree has unexpected revisions" >&2
            echo "  top-level: $commit" >&2
            echo "  geoclaw:   $geoclaw_commit" >&2
            exit 1
        fi
    else
        local staging="$TOOLS_ROOT/.clawpack-v5.14.0.tmp-${SLURM_JOB_ID:-$$}"
        if [ -e "$staging" ]; then
            echo "Refusing to reuse staging path: $staging" >&2
            exit 1
        fi
        git clone --recurse-submodules --branch v5.14.0 \
            https://github.com/clawpack/clawpack.git "$staging"
        if [ "$(git -C "$staging" rev-parse HEAD)" != "$EXPECTED_CLAW_COMMIT" ] ||
           [ "$(git -C "$staging/geoclaw" rev-parse HEAD)" != "$EXPECTED_GEOCLAW_COMMIT" ]; then
            echo "Cloned Clawpack revision does not match the frozen contract" >&2
            exit 1
        fi
        mv "$staging" "$CLAW_ROOT"
    fi
}

clone_or_check_petsc() {
    if [ -d "$PETSC_DIR/.git" ]; then
        local commit
        commit=$(git -C "$PETSC_DIR" rev-parse HEAD)
        if [ "$commit" != "$EXPECTED_PETSC_COMMIT" ]; then
            echo "Existing PETSc tree has unexpected revision: $commit" >&2
            exit 1
        fi
    else
        local staging="$TOOLS_ROOT/.petsc-3.25.3.tmp-${SLURM_JOB_ID:-$$}"
        if [ -e "$staging" ]; then
            echo "Refusing to reuse staging path: $staging" >&2
            exit 1
        fi
        git clone --branch v3.25.3 \
            https://gitlab.com/petsc/petsc.git "$staging"
        if [ "$(git -C "$staging" rev-parse HEAD)" != "$EXPECTED_PETSC_COMMIT" ]; then
            echo "Cloned PETSc revision does not match the frozen contract" >&2
            exit 1
        fi
        mv "$staging" "$PETSC_DIR"
    fi
}

clone_or_check_clawpack
clone_or_check_petsc

if [ ! -f "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so" ]; then
    (
        cd "$PETSC_DIR"
        ./configure PETSC_ARCH="$PETSC_ARCH" \
            --with-debugging=0 \
            --with-cc=mpicc \
            --with-cxx=mpicxx \
            --with-fc=mpif90 \
            --download-fblaslapack
        make -j "$SLURM_CPUS_PER_TASK" \
            PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
    )
fi

if [ ! -f "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so" ]; then
    echo "PETSc build did not produce libpetsc.so" >&2
    exit 1
fi

cat > "$TOOLS_ROOT/geoclaw_env.sh" <<EOF
export CLAW_ROOT="$CLAW_ROOT"
export PETSC_DIR="$PETSC_DIR"
export PETSC_ARCH="$PETSC_ARCH"
export GEOCLAW_PYTHON="$PYTHON"
export PYTHONPATH="$CLAW_ROOT:\${PYTHONPATH:-}"
export PATH="$PROJECT_ROOT/.venv/bin:/usr/bin:/bin:\${PATH:-}"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
export CLAW_MPIEXEC="mpiexec"
export CLAW_MPIFC="mpif90"
export FC="gfortran"
EOF

PYTHONPATH="$CLAW_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    CLAW="$CLAW_ROOT" \
    "$PYTHON" - <<'PY'
from pathlib import Path
from src.evaluation.geoclaw_adapter import GeoClawEnvironment, validate_geoclaw_environment

root = Path("/datastore/vit/tsunami-surrogate")
environment = GeoClawEnvironment(
    claw_root=root / ".tools/clawpack-v5.14.0",
    petsc_dir=root / ".tools/petsc-3.25.3",
    petsc_arch="arch-linux-c-opt",
    python_executable=root / ".venv/bin/python",
)
print(validate_geoclaw_environment(environment))
PY

echo "GeoClaw/PETSc setup complete"
echo "Environment file: $TOOLS_ROOT/geoclaw_env.sh"
