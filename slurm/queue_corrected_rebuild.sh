#!/usr/bin/env bash
# Queue independent corrected-contract rebuild branches.
#
# The three native branches are independent. Each branch is ordered only as:
# generation -> preprocessing -> native MUSCL-HR training.
# Real bathymetry is independent of native training and runs concurrently.

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; initialize the Slurm module first" >&2
    exit 1
fi

submit_native() {
    local grid="$1"
    local generate_job preprocess_job train_job
    local generation_config="configs/data/multires/dataset_${grid}.yaml"
    local preprocess_config="configs/data/multires/preprocess_${grid}.yaml"
    local train_config="configs/model/fno_res${grid}_muscl_hr.yaml"

    generate_job=$(sbatch --parsable \
        --cpus-per-task=8 \
        --mem=20G \
        --export="ALL,SPLIT=train,CONFIG_PATH=${generation_config},RESUME=0,NUM_WORKERS=8,MAX_IN_FLIGHT=8" \
        slurm/generate_dataset.slurm)
    preprocess_job=$(sbatch --parsable \
        --dependency="afterok:${generate_job}" \
        --export="ALL,CONFIG_PATH=${preprocess_config}" \
        slurm/preprocess_dataset.slurm)
    train_job=$(sbatch --parsable \
        --dependency="afterok:${preprocess_job}" \
        --export="ALL,CONFIG_PATH=${train_config},REQUIRED_VRAM=16384" \
        slurm/train_fno.slurm)

    printf 'res%s generation=%s preprocess=%s train=%s\n' \
        "$grid" "$generate_job" "$preprocess_job" "$train_job"
}

submit_native 32
submit_native 64
submit_native 128

real_job=$(sbatch --parsable \
    --export="ALL,NUM_WORKERS=8" \
    slurm/build_real_bathymetry.slurm)
printf 'real_bathymetry rebuild=%s\n' "$real_job"

echo
echo "Queued corrected data/model branches. Final preflight/evaluation are not"
echo "queued here because they require the regenerated artifacts and current"
echo "numerical evidence; inspect each log before submitting those gates."
