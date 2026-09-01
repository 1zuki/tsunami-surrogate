#!/bin/bash
#SBATCH --job-name=tsunami
#SBATCH --output=logs/slurm/tsunami_%j.out
#SBATCH --error=logs/slurm/tsunami_%j.err
#SBATCH --chdir=/datastore/vit/tsunami-surrogate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_ROOT="/datastore/vit/tsunami-surrogate"
SUITE_ID="${SUITE_ID:?SUITE_ID must be exported by the suite submitter}"
ARRAY_JOB_ID="${ARRAY_JOB_ID:?ARRAY_JOB_ID must be exported by the suite submitter}"
GENERATED_DIR="${GENERATED_DIR:?GENERATED_DIR must be exported by the suite submitter}"
ARCHIVE_ROOT="$PROJECT_ROOT/logs/slurm/archives"
STAGING="$ARCHIVE_ROOT/${SUITE_ID}_${ARRAY_JOB_ID}"

cd "$PROJECT_ROOT"
if [ -e "$STAGING" ]; then
    echo "Refusing to overwrite existing finalizer staging: $STAGING"
    exit 1
fi
mkdir -p "$STAGING/generated" "$STAGING/slurm" "$STAGING/per-run"

cp -a "$GENERATED_DIR"/. "$STAGING/generated/"
find logs/slurm -maxdepth 1 -type f \
  \( -name "tsunami_${ARRAY_JOB_ID}_*.out" -o -name "tsunami_${ARRAY_JOB_ID}_*.err" \) \
  -exec cp -a {} "$STAGING/slurm/" \;

while IFS=$'\t' read -r run_label config_path output_dir seed; do
    mkdir -p "$STAGING/per-run/$run_label"
    for name in cluster_console.log cluster_console.err; do
        if [ -f "$output_dir/$name" ]; then
            cp -a "$output_dir/$name" "$STAGING/per-run/$run_label/"
        fi
    done
done < "$GENERATED_DIR/runs.tsv"

sacct -j "$ARRAY_JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode \
  > "$STAGING/sacct.txt" || true
git rev-parse HEAD > "$STAGING/git_commit.txt" 2>/dev/null || true
git status --short --branch > "$STAGING/git_status.txt" 2>/dev/null || true
git diff --binary > "$STAGING/git_diff.patch" 2>/dev/null || true
(
  cd "$STAGING"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

ARCHIVE_BASE="$ARCHIVE_ROOT/${SUITE_ID}_${ARRAY_JOB_ID}.tar"
if command -v zstd >/dev/null 2>&1; then
    tar -C "$ARCHIVE_ROOT" -cf - "${SUITE_ID}_${ARRAY_JOB_ID}" \
      | zstd -T1 -3 -o "$ARCHIVE_BASE.zst"
    sha256sum "$ARCHIVE_BASE.zst" > "$ARCHIVE_BASE.zst.sha256"
    echo "[cluster-finalizer] archive=$ARCHIVE_BASE.zst"
else
    tar -C "$ARCHIVE_ROOT" -czf "$ARCHIVE_BASE.gz" \
      "${SUITE_ID}_${ARRAY_JOB_ID}"
    sha256sum "$ARCHIVE_BASE.gz" > "$ARCHIVE_BASE.gz.sha256"
    echo "[cluster-finalizer] archive=$ARCHIVE_BASE.gz"
fi
