#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-experiments/cloudrun/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
METRICS_DIR="${METRICS_DIR:-${RUN_ROOT}/metrics}"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
exec > >(tee -a "${LOG_DIR}/console.log") 2>&1

export PYTHON_BIN RUN_ID RUN_ROOT LOG_DIR METRICS_DIR

export DATA_CONFIG="${DATA_CONFIG:-configs/data/dataset.yaml}"
export PREPROCESS_CONFIG="${PREPROCESS_CONFIG:-configs/data/preprocess.yaml}"
export MODEL_CONFIG="${MODEL_CONFIG:-configs/model/fno.yaml}"
export MODEL_CONFIGS="${MODEL_CONFIGS:-configs/model/fno.yaml,configs/model/fno_muscl_hr.yaml,configs/model/fno_boussinesq.yaml}"
export CHECKPOINT="${CHECKPOINT:-}"
export CHECKPOINTS="${CHECKPOINTS:-}"

# Empty means: let the YAML configs decide, exactly like the README commands.
export DATASET_SAMPLES="${DATASET_SAMPLES:-}"
export N_STEPS="${N_STEPS:-}"
export SAVE_EVERY="${SAVE_EVERY:-}"
export NUM_WORKERS="${NUM_WORKERS:-}"
export DEVICE="${DEVICE:-auto}"
export ALLOW_TF32="${ALLOW_TF32:-}"

export RUN_DATASET="${RUN_DATASET:-1}"
export RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
export RUN_TRUE_RES_GENERATION="${RUN_TRUE_RES_GENERATION:-0}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_EVAL_ACCURACY="${RUN_EVAL_ACCURACY:-1}"
export RUN_EVAL_SPEED="${RUN_EVAL_SPEED:-1}"
export RUN_ARRIVAL="${RUN_ARRIVAL:-1}"
export RUN_OOD="${RUN_OOD:-1}"
export RUN_RESOLUTION_PROXY="${RUN_RESOLUTION_PROXY:-1}"
export RUN_TRUE_RES_EVAL="${RUN_TRUE_RES_EVAL:-0}"
export RUN_SOLVER_COMPARE="${RUN_SOLVER_COMPARE:-1}"
export RUN_SOLVER_SPEED="${RUN_SOLVER_SPEED:-1}"
export RUN_SPEED_TABLE="${RUN_SPEED_TABLE:-1}"
export RUN_EMULATOR_SUPERIORITY="${RUN_EMULATOR_SUPERIORITY:-1}"
export RUN_SAMPLE_SCALING="${RUN_SAMPLE_SCALING:-0}"
export DATASET_CONTINUE="${DATASET_CONTINUE:-1}"
export ALLOW_OVERRIDE="${ALLOW_OVERRIDE:-0}"

export TRUE_RESOLUTIONS="${TRUE_RESOLUTIONS:-32,64,128}"
export TRUE_RES_SAMPLES="${TRUE_RES_SAMPLES:-}"
export TRUE_RES_SHARED_FROM64="${TRUE_RES_SHARED_FROM64:-0}"
export SAMPLE_SIZES="${SAMPLE_SIZES:-8,16,32,64,128}"
export SAMPLE_SCALING_OUTPUT_ROOT="${SAMPLE_SCALING_OUTPUT_ROOT:-experiments/sample_scaling}"

export SOLVER_COMPARE_PAIRS="${SOLVER_COMPARE_PAIRS:-hydrostatic:muscl_hr,muscl_hr:hydrostatic,hydrostatic:boussinesq,boussinesq:hydrostatic,muscl_hr:boussinesq,boussinesq:muscl_hr}"
export SOLVER_A_DIR="${SOLVER_A_DIR:-data/raw/hydrostatic/samples}"
export SOLVER_B_DIR="${SOLVER_B_DIR:-data/raw/muscl_hr/samples}"
export SOLVER_COMPARE_OUTPUT="${SOLVER_COMPARE_OUTPUT:-results/solver_compare_hydro_vs_muscl_hr.json}"
export SOLVER_COMPARE_MAX_SAMPLES="${SOLVER_COMPARE_MAX_SAMPLES:-}"
export SOLVER_SPEED_SOLVERS="${SOLVER_SPEED_SOLVERS:-swe_hydrostatic,swe_muscl_hr,boussinesq}"
export SOLVER_SPEED_MAX_SAMPLES="${SOLVER_SPEED_MAX_SAMPLES:-}"
export SOLVER_SPEED_REPEATS="${SOLVER_SPEED_REPEATS:-}"
export SPEED_WARMUP="${SPEED_WARMUP:-}"
export SPEED_REPEATS="${SPEED_REPEATS:-}"
export SPEED_MAX_BATCHES="${SPEED_MAX_BATCHES:-}"
export EMULATOR_CONFIGS="${EMULATOR_CONFIGS:-configs/eval/emulator_superiority_hydro_to_muscl_hr.yaml,configs/eval/emulator_superiority_muscl_hr_to_hydro.yaml}"
export CONTINUE_ON_OPTIONAL_ERROR="${CONTINUE_ON_OPTIONAL_ERROR:-1}"
export SKIP_MISSING_OPTIONAL="${SKIP_MISSING_OPTIONAL:-1}"
export DRY_RUN="${DRY_RUN:-0}"

cat <<INFO
[start] run_id=${RUN_ID}
[start] run_root=${RUN_ROOT}
[start] logs_dir=${LOG_DIR}
[start] metrics_dir=${METRICS_DIR}
[start] policy=wrapper only; configs/data/model/results stay at their normal README paths
[start] data_config=${DATA_CONFIG}
[start] preprocess_config=${PREPROCESS_CONFIG}
[start] model_configs=${MODEL_CONFIGS}
[start] checkpoints=${CHECKPOINTS:-<each model output_dir>/best.pt}
[start] yaml_overrides=samples=${DATASET_SAMPLES:-<config>} n_steps=${N_STEPS:-<config>} save_every=${SAVE_EVERY:-<config>} workers=${NUM_WORKERS:-<config>}
[start] dataset_continue=${DATASET_CONTINUE} allow_override=${ALLOW_OVERRIDE}
[start] optional=ood:${RUN_OOD} resolution_proxy:${RUN_RESOLUTION_PROXY} solver_compare:${RUN_SOLVER_COMPARE} emulator:${RUN_EMULATOR_SUPERIORITY}
[start] heavy_optional=true_res_generation:${RUN_TRUE_RES_GENERATION} true_res_eval:${RUN_TRUE_RES_EVAL} sample_scaling:${RUN_SAMPLE_SCALING}
[start] solver_compare_pairs=${SOLVER_COMPARE_PAIRS}
[start] solver_speed_solvers=${SOLVER_SPEED_SOLVERS}
[start] device=${DEVICE}
[start] console_log=${LOG_DIR}/console.log
INFO

"${PYTHON_BIN}" scripts/run_full_pipeline.py

cat <<INFO
[start] done
[start] console_log=${LOG_DIR}/console.log
[start] stage_table=${METRICS_DIR}/full_pipeline_stages.csv
[start] metrics_manifest=${METRICS_DIR}/full_pipeline_results.json
[start] copied_metrics=${METRICS_DIR}
INFO
