![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

# Tsunami Surrogate Modeling with Neural Operators

## 1) Project Idea

This repository builds a research benchmark for fast surrogate modeling of tsunami-like wave propagation in a controlled synthetic setting.

Instead of running a numerical PDE solver online for every scenario, we train neural surrogates to learn:

`(bathymetry, source/initial disturbance) -> future wave-height trajectory`

The scope is research and benchmarking, not an operational early-warning deployment system.

## 2) Research Questions

The current forward-surrogate benchmark focuses on:

1. Fidelity: how closely predictions match the shallow-water solver.
2. Speed: how much inference acceleration is gained over full numerical rollout.
3. Robustness: how performance changes under distribution shift (unseen bathymetry/source families) and cross-resolution transfer.
4. Uncertainty quality: whether predictive confidence tracks actual error.

Separate follow-up track (not part of the current forward-surrogate paper):

5. Inverse problem: recover source characteristics from observed wave signals/fields.

## 3) What Is Implemented vs Planned

- Implemented core: synthetic data generation, preprocessing, forward surrogate training, and benchmark evaluation.
- Implemented models: FNO (primary) with CNN/U-Net/ConvLSTM and ensemble paths for comparison.
- Implemented evaluations: accuracy, speed, generalization, resolution transfer, and uncertainty.
- Separate follow-up work: dedicated inverse-problem experiments and a separate paper track.

## 4) Canonical Workflow

The default full-module pipeline in this repo is:

1. Generate synthetic physics rollouts.
2. Preprocess into train/val/test tensors.
3. Train surrogate models.
4. Evaluate solver-fidelity, speed, robustness, and uncertainty.
5. Export plots/tables and map outputs into paper sections.
6. Keep inverse-problem workflow as a separate follow-up track (outside current forward-paper claims).

## 5) Setup

```bash
git clone https://github.com/1zuki/tsunami-surrogate.git
cd tsunami-surrogate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6) Run Commands

Core workflow order:
1. `6.1` generate forward raw dataset
2. `6.2` preprocess into train/val/test archives
3. `6.3` train model checkpoints
4. `6.4` evaluate those checkpoints on matching test splits

Everything after `6.4` is optional track work (OOD, resolution transfer, real-resolution benchmark, solver-vs-solver comparison, inverse scaffold).

### 6.0 Full Paper Pipeline (Manual Local Run)

This section is the complete local run that produces every metric, table, and figure source used in the paper. The individual commands in `6.1`-`6.15` are easy to run out of order or to miss a dependency (the runtime speedup table, in particular, needs both a model-speed JSON and a solver-speed JSON before it can be aggregated), so run them in the order below.

`scripts/run_full_pipeline.py` also exists as a one-call wrapper, but it was written for batch/server runs; for a local run it is not needed, and the explicit sequence below is the equivalent.

The commands assume hydrostatic + MUSCL-HR processed datasets from `6.1`-`6.2` already exist. They use `cuda` for model training and inference; if you have no local GPU, replace `--device cuda` with `--device cpu`. The reference-solver timing always runs on CPU by design, and you should keep the CPU model-speed run either way, because the speedup table compares CPU solver time against both CPU and GPU model inference.

Boussinesq is intentionally not in this sequence: it lives on a separate scenario regime, so it stays out of the main same-scenario tables and is run separately for the appendix (see `6.9`).

```bash
# --- Train (6.3) ---
python scripts/train.py --config configs/model/fno.yaml
python scripts/train.py --config configs/model/fno_muscl_hr.yaml

# --- Same-solver accuracy (6.4) ---
python scripts/eval_accuracy.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt
python scripts/eval_accuracy.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt

# --- Model inference speed: numerator of the speedup ratio (6.4) ---
# Records device, batch size, precision, and TF32 state into each JSON.
python scripts/eval_speed.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt          --device cpu  --precision fp32 --allow-tf32 false --output results/speed/model_speed_fno_cpu.json
python scripts/eval_speed.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt          --device cuda --precision fp32 --allow-tf32 true  --output results/speed/model_speed_fno_cuda.json
python scripts/eval_speed.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --device cpu  --precision fp32 --allow-tf32 false --output results/speed/model_speed_muscl_hr_cpu.json
python scripts/eval_speed.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --device cuda --precision fp32 --allow-tf32 true  --output results/speed/model_speed_muscl_hr_cuda.json

# --- Reference-solver speed: denominator of the speedup ratio (6.4) ---
# Re-runs the NumPy CPU solver rollout on cached scenarios (no dataset regeneration).
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_hydrostatic --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_swe_hydrostatic.json
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_muscl_hr    --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_swe_muscl_hr.json

# --- Aggregate speed table: speedup = solver rollout time / model inference time (6.4) ---
python scripts/make_speed_table.py \
  --solver results/speed/solver_speed_swe_hydrostatic.json \
  --solver results/speed/solver_speed_swe_muscl_hr.json \
  --model  results/speed/model_speed_fno_cpu.json \
  --model  results/speed/model_speed_fno_cuda.json \
  --model  results/speed/model_speed_muscl_hr_cpu.json \
  --model  results/speed/model_speed_muscl_hr_cuda.json \
  --output results/speed/speed_table.csv \
  --output-json results/speed/speed_table.json

# --- OOD generalization (6.5) ---
python scripts/make_ood_splits.py --config configs/data/ood_splits_hydrostatic.yaml --overwrite
python scripts/make_ood_splits.py --config configs/data/ood_splits_muscl_hr.yaml --overwrite
python scripts/eval_generalization.py --config configs/eval/ood_suites_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_generalization.py --config configs/eval/ood_suites_muscl_hr.yaml   --checkpoint experiments/fno_muscl_hr/best.pt

# --- Cross-resolution proxy: resize the 64 test set, no extra data (6.6) ---
python scripts/eval_resolution_transfer.py --config configs/eval/resolution_transfer_proxy_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_resolution_transfer.py --config configs/eval/resolution_transfer_proxy_muscl_hr.yaml   --checkpoint experiments/fno_muscl_hr/best.pt

# --- Native cross-resolution: real 32/64/128 grids (6.7) ---
# You generated native 32 and 128 pools, so this is the stronger result, not just the proxy.
# Preprocess each native grid first (raw -> processed), then evaluate one checkpoint across all of them.
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_32.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_64.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_128.yaml
python scripts/eval_full_resolution.py --config configs/eval/resolution_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_full_resolution.py --config configs/eval/resolution_muscl_hr.yaml   --checkpoint experiments/fno_muscl_hr/best.pt

# --- Solver-vs-solver physical gap (6.8) ---
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --require-quality-ok --save-arrival-maps \
  --arrival-maps-output results/solver_compare_hydro_vs_muscl_hr_arrival_maps.npz \
  --output results/solver_compare_hydro_vs_muscl_hr.json

# --- Emulator-superiority ratio (6.15) ---
# Requires the solver-vs-solver JSON above as its denominator.
python scripts/eval_emulator_superiority.py --config configs/eval/emulator_superiority_hydro_to_muscl_hr.yaml
python scripts/eval_emulator_superiority.py --config configs/eval/emulator_superiority_muscl_hr_to_hydro.yaml

# --- Arrival-time maps, model vs target solver (6.14) ---
python scripts/eval_arrival_maps.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt
python scripts/eval_arrival_maps.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
```

Uncertainty (ensemble) and the learning-curve sweep are separate because they need extra training runs:

```bash
# Deep ensemble for uncertainty (train 3 members, then evaluate) - see 6.4 / 6.13
python scripts/train_ensemble.py --config configs/model/fno.yaml
python scripts/eval_uncertainty.py --config configs/model/fno.yaml \
  --checkpoint experiments/ensemble/member_11/best.pt \
  --checkpoint experiments/ensemble/member_22/best.pt \
  --checkpoint experiments/ensemble/member_33/best.pt

# Learning curve / sample-scaling (one of the strongest figures) - see plan.md
python scripts/run_sample_scaling.py --config configs/model/fno.yaml          --samples 100,500,1000,2500,5000,10000 --output-root experiments/sample_scaling/fno          --device cuda
python scripts/run_sample_scaling.py --config configs/model/fno_muscl_hr.yaml --samples 100,500,1000,2500,5000,10000 --output-root experiments/sample_scaling/fno_muscl_hr --device cuda

# Qualitative truth/prediction/error maps for the paper figure (Fig 3) - see 6.12
python scripts/export_figures.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt          --out paper/figs/fno_hydrostatic
python scripts/export_figures.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --out paper/figs/fno_muscl_hr
```

#### 6.0.3 Where each paper table/figure comes from

| Paper artifact | Produced by | Output |
| --- | --- | --- |
| Table 2 — same-solver accuracy | `eval_accuracy.py` | `experiments/<model>/eval/metrics.json` |
| Table 3 / Fig 4 — runtime + speedup | `eval_speed.py` + `eval_solver_speed.py` -> `make_speed_table.py` | `results/speed/speed_table.{csv,json}` |
| Table 4 / Fig 5 — OOD generalization | `eval_generalization.py` | `experiments/<model>/eval_ood_suites/ood_generalization.json` |
| Fig 6 — cross-resolution | `eval_resolution_transfer.py` | `.../eval_resolution_proxy/resolution_transfer_proxy.json` |
| Table 5 / Fig 7 — solver gap | `compare_solvers_physical.py` | `results/solver_compare_*.json` |
| emulator-superiority ratio | `eval_emulator_superiority.py` | `results/emulator_superiority_*.json` |
| Fig 2 — learning curves | `run_sample_scaling.py` | `experiments/sample_scaling/*/sample_scaling_results.csv` |
| Table 7 — uncertainty | `eval_uncertainty.py` | `.../eval_uncertainty/uncertainty.json` |
| Fig 3 — qualitative maps | `export_figures.py` | `paper/figs/...` |

### 6.1 Step 1 - Generate Forward Raw Dataset (Required)

Main benchmark generation (hydrostatic + MUSCL-HR):

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml
```

`make_dataset.py` runs in three stages:
- stage 1: generate/cache all bathymetry samples first (default cache: `data/bathymetry`)
- stage 2: generate/cache all source samples first (default cache: `data/sources`)
- stage 3: load cached bathymetry + source pairs and run configured FDE rollouts from `fdes.enabled`

Raw rollouts are separated by solver under `data/raw/`:
- `data/raw/hydrostatic/samples/...`
- `data/raw/muscl_hr/samples/...`

Manifests are separated as:
- scenario-level: `data/synthetic/scenario_manifest.jsonl`
- solver-level: `data/synthetic/hydrostatic_manifest.jsonl`, `data/synthetic/muscl_hr_manifest.jsonl`, `data/synthetic/boussinesq_manifest.jsonl`

Runnable FDEs currently include `swe_hydrostatic`, `swe_muscl_hr`, and `boussinesq`.
Default `configs/data/dataset.yaml` enables only `swe_hydrostatic` + `swe_muscl_hr` for safer baseline runs.
Legacy alias `swe_muscl` is still accepted and automatically mapped to `swe_muscl_hr` for backward compatibility.

Experimental Boussinesq generation uses a dedicated config:

```bash
python scripts/make_dataset.py --config configs/data/dataset_boussinesq.yaml
```

This writes Boussinesq samples to `data/raw_bouss/boussinesq/samples/...` and scenario manifest to `data/synthetic/scenario_manifest_bouss.jsonl`.

Resume an interrupted run:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --continue
```

`--continue` rolls back to the last completed worker batch boundary before resuming (using `num_workers`) to reduce partial-batch holes after interruptions.
During resume, per-FDE sample folders that are already complete are reused as-is (not overwritten) unless `--allow-override` is set.

Resume from an explicit sample index (1-based):

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --start-at 142
```

Force regeneration in a range even if outputs already exist:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --start-at 142 --allow-override
```

Rebuild manifests from already-generated sample folders:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --rebuild-manifests
```

Quality guardrails (configured in `quality:` inside dataset YAML):
- `on_violation: warn|fail`
- `reject_nonfinite`
- `min_h_tolerance`
- `max_abs_eta_limit`
- `max_velocity_limit`
- Recommendation: keep `on_violation: fail` (now default in provided dataset configs) so unstable samples do not silently enter raw manifests.

### 6.2 Step 2 - Preprocess Forward Data (Required)

Main benchmark preprocessing:

```bash
python src/data_gen/preprocess.py --config configs/data/preprocess.yaml
```

`preprocess.yaml` supports FDE-aware modes:
- `fde.mode: single` with `fde.targets: [hydrostatic]` writes to `data/processed/hydrostatic/...`
- `fde.mode: separate_all` writes one processed dataset per solver (`hydrostatic`, `muscl_hr`, `boussinesq`) using the same scenario split
- `fde.mode: multifidelity` writes a combined dataset to `data/processed/multifidelity/...`
- For `multifidelity`, keep `input.use_solver_id: true` (or omit it, since it auto-enables by default) so the model can condition on solver identity instead of learning an ambiguous one-to-many mapping

Boussinesq-only preprocessing (separate manifests/paths + 50-step target horizon):

```bash
python src/data_gen/preprocess.py --config configs/data/preprocess_boussinesq.yaml
```

Main outputs used by training/eval:
- `data/processed/hydrostatic/{train,val,test}/eval_dataset.npz`
- `data/processed/muscl_hr/{train,val,test}/eval_dataset.npz`

### 6.3 Step 3 - Train Forward Models (Required Before Eval)

Train hydrostatic-label FNO:

```bash
python scripts/train.py --config configs/model/fno.yaml
```

Train MUSCL-HR-label FNO:

```bash
python scripts/train.py --config configs/model/fno_muscl_hr.yaml
```

Optional training tracks:
- Boussinesq model (experimental): `python scripts/train.py --config configs/model/fno_boussinesq.yaml`
- ConvLSTM baseline: `python scripts/train.py --config configs/model/convlstm.yaml`
- ConvLSTM (MUSCL-HR labels): `python scripts/train.py --config configs/model/convlstm_muscl_hr.yaml`
- Ensemble for uncertainty: `python scripts/train_ensemble.py --config configs/model/fno.yaml`

Native-resolution training tracks (P2 extension):
- Hydrostatic: `configs/model/fno_res{32,64,128}_hydrostatic.yaml`
- MUSCL-HR: `configs/model/fno_res{32,64,128}_muscl_hr.yaml`
- Shared-from64 normalization checkpoints:
  - `configs/model/fno_res64_shared_from64_hydrostatic.yaml`
  - `configs/model/fno_res64_shared_from64_muscl_hr.yaml`

### 6.4 Step 4 - Baseline Eval on Matching Test Split

After `6.3`, evaluate each model on its matching processed test set:

```bash
python scripts/eval_accuracy.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --device cpu --precision fp32 --allow-tf32 false
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --device cuda --precision fp32 --allow-tf32 true
python scripts/eval_generalization.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_accuracy.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_generalization.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
```

Uncertainty evaluation needs an ensemble (2+ checkpoints):

```bash
python scripts/eval_uncertainty.py --config configs/model/fno.yaml \
  --checkpoint experiments/ensemble/member_11/best.pt \
  --checkpoint experiments/ensemble/member_22/best.pt \
  --checkpoint experiments/ensemble/member_33/best.pt
```

Eval notes:
- `eval_accuracy.py`, `eval_generalization.py`, and `eval_resolution_transfer.py` report normalized metrics by default and add `*_physical` metrics automatically when target denormalization stats are available in the evaluation dataset archive
- `eval_generalization.py` supports explicit OOD suites via `eval.generalization.suites` (or top-level `generalization.suites`) in config
- Single-member uncertainty is blocked by design (degenerate variance)
- Train/eval entrypoints validate dataset-vs-model I/O channels early, so stale `model.in_channels` / `model.out_channels` mismatches fail fast
- Eval JSON outputs now include sample-count metadata (`num_samples` or `dataset_num_samples`) so paper tables can report support size explicitly
- `--device` now overrides config in eval entrypoints (`eval_accuracy`, `eval_generalization`, `eval_uncertainty`, `eval_arrival_maps`, `eval_emulator_superiority`)

Runtime fairness benchmark helper flow (CPU NumPy solver denominator + CPU/GPU surrogate inference):

```bash
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_hydrostatic --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/hydrostatic_cpu.json
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --device cpu --precision fp32 --allow-tf32 false --output results/speed/fno_hydrostatic_cpu.json
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --device cuda --precision fp32 --allow-tf32 true --output results/speed/fno_hydrostatic_cuda.json
python scripts/make_speed_table.py --solver results/speed/hydrostatic_cpu.json --model results/speed/fno_hydrostatic_cpu.json --model results/speed/fno_hydrostatic_cuda.json --output results/speed/speed_table.csv
```

### 6.5 Optional - OOD Suite Evaluation

Prerequisites:
- processed test archives from `6.2`
- trained checkpoints from `6.3`

Build OOD suite datasets from processed test archives:

```bash
python scripts/make_ood_splits.py --config configs/data/ood_splits_hydrostatic.yaml --overwrite
python scripts/make_ood_splits.py --config configs/data/ood_splits_muscl_hr.yaml --overwrite
```

Tip:
- `make_ood_splits.py` prints available `source_type` / `bathymetry_type` counts and `source_strength` range from your current test archive.
- If a suite selects zero samples, relax or change filters in `configs/data/ood_splits_*.yaml`, rebuild suites, then rerun evaluation.
- OOD split configs now support `min_samples` + `min_samples_action: warn|fail` to prevent accidentally evaluating tiny suites.

Run suite-based generalization evaluation:

```bash
python scripts/eval_generalization.py --config configs/eval/ood_suites_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_generalization.py --config configs/eval/ood_suites_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
```

Output file:
- `.../eval_ood_suites/ood_generalization.json`

### 6.6 Optional - Resolution Transfer (Proxy, No Extra Training)

This track uses one trained checkpoint from `6.3` and evaluates it on resized versions of one test archive.
It is a proxy study, not native re-simulation at each resolution.

```bash
python scripts/eval_resolution_transfer.py \
  --config configs/eval/resolution_transfer_proxy_hydrostatic.yaml \
  --checkpoint experiments/fno/best.pt
python scripts/eval_resolution_transfer.py \
  --config configs/eval/resolution_transfer_proxy_muscl_hr.yaml \
  --checkpoint experiments/fno_muscl_hr/best.pt
```

Output file:
- `.../eval_resolution_proxy/resolution_transfer_proxy.json`

### 6.7 Optional - Real-Resolution Benchmark (Native 32/64/128)

Prerequisites:
- generate native-grid forward data per resolution
- preprocess each resolution
- use a checkpoint from `6.3` for cross-resolution evaluation

Generate native-grid raw datasets:

```bash
python scripts/make_dataset.py --config configs/data/multires/dataset_32.yaml
python scripts/make_dataset.py --config configs/data/multires/dataset_64.yaml
python scripts/make_dataset.py --config configs/data/multires/dataset_128.yaml
```

Preprocess each native-grid dataset:

```bash
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_32.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_64.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_128.yaml
```

Evaluate one trained checkpoint across all real resolutions in one JSON table:

```bash
python scripts/eval_full_resolution.py \
  --config configs/eval/resolution_hydrostatic.yaml \
  --checkpoint experiments/fno/best.pt
python scripts/eval_full_resolution.py \
  --config configs/eval/resolution_muscl_hr.yaml \
  --checkpoint experiments/fno_muscl_hr/best.pt
```

Output file:
- `.../eval_resolution/real_resolution.json` (includes `evaluation_type: native_real_resolution_benchmark`)

Native-resolution normalization policy:
- `configs/eval/resolution_*.yaml` now defaults to `real_resolution.normalization_policy: require_target_stats_match`.
- This fails fast if suite target normalization does not match the configured training/reference dataset stats (`normalization_reference_path`), which avoids misleading cross-resolution claims.

True cross-resolution transfer option (shared normalization from res64 reference):

```bash
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_32_shared_from64.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_64_shared_from64.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_128_shared_from64.yaml

python scripts/train.py --config configs/model/fno_res64_shared_from64_hydrostatic.yaml
python scripts/train.py --config configs/model/fno_res64_shared_from64_muscl_hr.yaml

python scripts/eval_full_resolution.py \
  --config configs/eval/resolution_hydrostatic_shared_from64.yaml \
  --checkpoint experiments/fno_res64_shared_from64_hydrostatic/best.pt
python scripts/eval_full_resolution.py \
  --config configs/eval/resolution_muscl_hr_shared_from64.yaml \
  --checkpoint experiments/fno_res64_shared_from64_muscl_hr/best.pt
```

Important:
- `eval_full_resolution.py` now validates that checkpoint training normalization stats match the configured reference stats.  
- For native/shared-from64 claims, do not reuse generic `6.3` checkpoints; use dedicated shared-from64 checkpoints.

### 6.8 Optional - Solver-vs-Solver Physical Comparison

Compare raw hydrostatic vs raw MUSCL-HR labels on shared scenarios in physical eta units:

```bash
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --output results/solver_compare_hydro_vs_muscl_hr.json
```

The comparison now includes:
- pointwise physical metrics (`rmse`, `mae`, `max_abs`, `rel_l2`)
- spectral differences (`spectral_rmse`, `spectral_l1`, `spectral_js_divergence`)
- arrival-time differences in timestep units and seconds (when timestamps are available)

Arrival threshold is configurable:

```bash
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --arrival-threshold-fraction 0.05 \
  --output results/solver_compare_hydro_vs_muscl_hr.json
```

Arrival-map export (aggregated spatial maps):

```bash
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --save-arrival-maps \
  --arrival-maps-output results/solver_compare_hydro_vs_muscl_hr_arrival_maps.npz \
  --output results/solver_compare_hydro_vs_muscl_hr.json
```

Quality-filtered comparison (recommended when your raw samples include `quality_status` in `meta.json`):

```bash
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --require-quality-ok \
  --missing-quality-action skip \
  --output results/solver_compare_hydro_vs_muscl_hr_quality_ok.json
```

### 6.9 Optional - Boussinesq Propagation Diagnostic

Run a dedicated propagation diagnostic (metrics + plots) for one scenario:

```bash
python scripts/diagnose_boussinesq.py \
  --config configs/data/dataset_boussinesq.yaml \
  --sample-index 1 \
  --output-dir results/boussinesq_diagnostic/sample_000001
```

Saved artifacts include:
- `summary.json`
- `timeseries.npz`
- `frames_eta.npy`
- `timeseries.png`
- `frames_gallery.png`
- `fields.png`

Reference-use gate:
- treat Boussinesq labels as exploratory until `diagnose_boussinesq.py` outputs physically consistent propagation on your chosen scenarios.

### 6.10 Optional - Inverse Dataset Scaffold (Separate Follow-Up Track)

Prerequisite:
- forward processed outputs from `6.2`

Export inverse scaffold datasets (observation -> source field):

```bash
python scripts/make_inverse_dataset.py --config configs/data/inverse_hydrostatic.yaml --overwrite
python scripts/make_inverse_dataset.py --config configs/data/inverse_muscl_hr.yaml --overwrite
```

Current status:
- scaffold export is implemented
- inverse-model training/eval remains outside the current forward-surrogate paper scope

Sparse-gauge inverse scaffold:

```bash
python scripts/make_inverse_dataset.py --config configs/data/inverse_hydrostatic_sparse_gauges.yaml --overwrite
python scripts/make_inverse_dataset.py --config configs/data/inverse_muscl_hr_sparse_gauges.yaml --overwrite
```

Sparse exports include:
- `gauge_coords` (`[G,2]`)
- `gauge_mask` (`[N,H,W]`)
- `gauge_observations` (`[N,G,T]`)
- `gauge_summary` (`[N,H,W]`, sparse on gauge locations)

### 6.11 Quick Smoke Run

```bash
bash scripts/quickstart.sh
```

### 6.12 Visualize One Sample (Truth vs Prediction + Uncertainty)

```bash
python scripts/visualize_rollout.py \
  --config configs/model/fno.yaml \
  --checkpoint experiments/fno/best.pt \
  --processed-path data/processed/hydrostatic/test \
  --raw-dir data/raw/hydrostatic/samples \
  --sample-index 0
```

Optional visualization controls:
- `--wave-3d-mode eta|overlay` for eta-only 3D surfaces or bathymetry overlays
- `--wave-scale <float>` to control vertical exaggeration in 3D plots (auto if omitted)
- target/prediction frames are denormalized automatically when target stats exist in the processed archive

### 6.13 Optional - OOD Uncertainty Suites

Evaluate ensemble uncertainty metrics on OOD suite datasets:

```bash
python scripts/eval_uncertainty.py \
  --config configs/eval/uncertainty_ood_hydrostatic.yaml \
  --checkpoint experiments/ensemble/member_11/best.pt \
  --checkpoint experiments/ensemble/member_22/best.pt \
  --checkpoint experiments/ensemble/member_33/best.pt
```

For MUSCL-HR suites:

```bash
python scripts/eval_uncertainty.py \
  --config configs/eval/uncertainty_ood_muscl_hr.yaml \
  --checkpoint experiments/ensemble/member_11/best.pt \
  --checkpoint experiments/ensemble/member_22/best.pt \
  --checkpoint experiments/ensemble/member_33/best.pt
```

Output file:
- `.../eval_uncertainty_ood/uncertainty_ood.json`

Note:
- uncertainty outputs now include physical-unit calibration/correlation metrics with `_physical` suffix when target denormalization stats are available.

### 6.14 Optional - Arrival-Time Maps (Model vs Target Solver)

```bash
python scripts/eval_arrival_maps.py \
  --config configs/model/fno.yaml \
  --checkpoint experiments/fno/best.pt
```

Outputs:
- `.../eval/arrival_map_model_vs_target.json`
- `.../eval/arrival_map_model_vs_target.npz`

### 6.15 Optional - Emulator Superiority Ratio

Compute:
`error(FNO trained on A, solver B) / error(solver A, solver B)`

```bash
python scripts/eval_emulator_superiority.py \
  --config configs/eval/emulator_superiority_hydro_to_muscl_hr.yaml
python scripts/eval_emulator_superiority.py \
  --config configs/eval/emulator_superiority_muscl_hr_to_hydro.yaml
```

Safety notes:
- default numerator metric is now `rmse_physical_separate_denorm`, which denormalizes predictions using checkpoint-train stats and targets using eval-target stats.
- if normalization signatures mismatch, unsafe numerator metrics are blocked (`fail` by default) to avoid misleading emulator-superiority ratios.

## 7) Current Repository Structure

```text
tsunami-surrogate/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ configs/                        # all experiment/data/model/eval configs
│  ├─ data/                        # data-generation and preprocessing configs
│  │  ├─ dataset.yaml              # three-stage generation config + per-FDE raw outputs
│  │  ├─ dataset_boussinesq.yaml
│  │  ├─ multires/                 # native 32/64/128 forward-data configs
│  │  ├─ ood_splits_hydrostatic.yaml
│  │  ├─ ood_splits_muscl_hr.yaml
│  │  ├─ inverse_hydrostatic.yaml
│  │  ├─ inverse_muscl_hr.yaml
│  │  ├─ inverse_hydrostatic_sparse_gauges.yaml
│  │  ├─ inverse_muscl_hr_sparse_gauges.yaml
│  │  ├─ preprocess.yaml           # raw -> processed split/export config
│  │  ├─ preprocess_boussinesq.yaml
│  │  ├─ bathymetry.yaml           # bathymetry synthesis controls
│  │  ├─ bathymetry_boussinesq.yaml
│  │  ├─ source.yaml               # tsunami source family controls
│  │  ├─ source_boussinesq.yaml
│  │  └─ multires/preprocess_*_shared_from64.yaml
│  ├─ model/                       # model-centered train/eval configs
│  │  ├─ fno.yaml                  # primary FNO config
│  │  ├─ fno_muscl_hr.yaml         # FNO on MUSCL-HR processed labels
│  │  ├─ fno_boussinesq.yaml       # FNO on Boussinesq processed labels
│  │  ├─ fno_res32_hydrostatic.yaml
│  │  ├─ fno_res64_hydrostatic.yaml
│  │  ├─ fno_res128_hydrostatic.yaml
│  │  ├─ fno_res32_muscl_hr.yaml
│  │  ├─ fno_res64_muscl_hr.yaml
│  │  ├─ fno_res128_muscl_hr.yaml
│  │  ├─ fno_res64_shared_from64_hydrostatic.yaml
│  │  ├─ fno_res64_shared_from64_muscl_hr.yaml
│  │  ├─ cnn.yaml                  # CNN baseline config
│  │  ├─ unet.yaml                 # U-Net baseline config
│  │  ├─ convlstm.yaml             # ConvLSTM baseline config
│  │  └─ convlstm_muscl_hr.yaml    # ConvLSTM on MUSCL-HR labels
│  ├─ train/                       # shared/base + training variants
│  │  ├─ base.yaml                 # common seed/device/data/train defaults
│  │  ├─ physics_loss.yaml         # physics-regularized FNO variant
│  │  └─ train_32_to_64.yaml       # resolution-transfer training setup
│  └─ eval/
│     ├─ eval_template.yaml        # template for standalone eval scripts
│     ├─ ood_suites_hydrostatic.yaml
│     ├─ ood_suites_muscl_hr.yaml
│     ├─ uncertainty_ood_hydrostatic.yaml
│     ├─ uncertainty_ood_muscl_hr.yaml
│     ├─ resolution_transfer_proxy_hydrostatic.yaml
│     ├─ resolution_transfer_proxy_muscl_hr.yaml
│     ├─ resolution_hydrostatic.yaml
│     ├─ resolution_muscl_hr.yaml
│     ├─ resolution_hydrostatic_shared_from64.yaml
│     ├─ resolution_muscl_hr_shared_from64.yaml
│     ├─ emulator_superiority_hydro_to_muscl_hr.yaml
│     └─ emulator_superiority_muscl_hr_to_hydro.yaml
├─ scripts/                        # CLI entrypoints (generate/train/eval/export)
├─ src/                            # implementation modules
│  ├─ data_gen/                    # simulation + preprocess pipeline internals
│  ├─ data/                        # dataset loaders and multires dataset wrappers
│  ├─ solver/                      # shallow-water and related numerical solvers
│  ├─ models/                      # FNO/CNN/U-Net/ConvLSTM/ensemble/uncertainty models
│  ├─ training/                    # trainer, losses, metrics, callbacks, checkpoints
│  ├─ evaluation/                  # accuracy/speed/generalization/UQ evaluation utils
│  └─ utils/                       # config/io/logger/device/seed/visualization helpers
├─ data/                           # generated artifacts (raw, processed, manifests)
├─ experiments/                    # run outputs (checkpoints, history, eval json)
├─ figures/                        # exported figures/plots
├─ results/                        # aggregate result dumps
├─ tests/                          # unit/integration checks
├─ paper/                          # LaTeX manuscript workspace
│  ├─ main.tex                     # paper entrypoint
│  ├─ figs/                        # paper figures
│  ├─ build/                       # latex build artifacts
│  └─ sections/                    # section files (role-only naming)
└─ references-notes/               # literature notes for writing and framing
```

## 8) Paper Alignment

This README follows the same framing as the paper abstract/introduction:

- controlled synthetic benchmark setting;
- FNO-centered surrogate evaluation against CNN/U-Net/ConvLSTM baselines;
- emphasis on speed-accuracy-robustness trade-offs;
- explicit non-operational scope (research benchmark, not production warning stack);
- inverse-problem work kept as separate follow-up paper scope, not part of forward-surrogate claims here.

## 9) Notes

- Development note: Portions of the codebase were developed with AI-assisted programming support. All code should be treated as author-reviewed research software, with tests and validation required before use in reported experiments.
- Test split tip: quick CI/local smoke can use `pytest -q -m "not slow"`; full solver dynamics validation can use `pytest -q -m slow`.
