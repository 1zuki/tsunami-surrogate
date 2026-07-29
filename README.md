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
- Implemented models: FNO (primary) with CNN/U-Net and ensemble paths for comparison. (A ConvLSTM baseline exists in the code but is experimental and not part of the paper results.)
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

## 5a) Reproducibility notes

The intended runtime is Python 3.10, with dependencies listed in
`requirements.txt`. Paper CUDA timing rows use the recorded speed metadata from
runs with PyTorch 2.10.0+cu128 and CUDA 12.8. Trained checkpoints are not
redistributed in full; the release provides configs, seeds, training histories,
and checkpoint-selection information so reported runs can be reproduced. This
archive supports research benchmark reproducibility, not operational tsunami
prediction.

## 5b) Reproduce from the released benchmark data (recommended)

You do **not** need to regenerate the 300 GB raw rollouts to reproduce the paper.
The released benchmark bundle ships the model-ready *processed* arrays, so you can
go straight to training (Section 6.3) and evaluation (6.4+).

1. Download the dataset bundle from the archive (DOI: https://doi.org/10.5281/zenodo.20974604).
2. Verify integrity, then extract each archive into `data/processed/`:

```bash
# from the bundle directory
sha256sum -c SHA256SUMS.txt

# main 64x64 references (hydrostatic / MUSCL-HR / Boussinesq) + eval split
for f in main_processed/*.tar.zst; do
  tar --use-compress-program=unzstd -xf "$f" -C /path/to/tsunami-surrogate/data/processed/
done

# optional: OOD suites, cross-resolution, and real-bathymetry diagnostics
for f in ood_processed/*.tar.zst crossres_processed/*.tar.zst real_bathymetry_processed/*.tar.zst; do
  tar --use-compress-program=unzstd -xf "$f" -C /path/to/tsunami-surrogate/data/processed/
done
```

After extraction you should have `data/processed/hydrostatic/{train,val,test}`,
`data/processed/muscl_hr/...`, and `data/processed/boussinesq/...`, which is what the
training and evaluation configs expect. The exact archive layout, per-suite contents,
and citation are documented in the bundle's own `README.md`.

Skipping the bundle? Generate everything from scratch via Sections 6.1--6.2 instead.
All data paths in `configs/` are repo-relative (`./data/...`), so commands run from
the repository root without edits.

## 6) Run Commands

Core workflow order:
1. `6.1` generate raw physics rollouts
2. `6.2` preprocess all solver targets into train/val/test archives
3. `6.3` train target-specific FNO checkpoints
4. `6.4` evaluate same-target accuracy
5. `6.5` benchmark model speed, solver speed, and the speedup table
6. `6.6` run OOD suites
7. `6.7` run proxy resolution transfer
8. `6.8` run native 32/64/128 resolution experiments where configs exist
9. `6.9` compare solver-vs-solver physical gaps
10. `6.10` compute emulator-superiority ratios
11. `6.11`-`6.17` run optional diagnostics, inverse scaffold, smoke checks, uncertainty, arrival maps, learning curves, and figure exports

The manual commands below are the source of truth. Treat wrapper scripts as convenience helpers only if they match this order.

### 6.0 Full Paper Pipeline (Manual Local Run)

This is the condensed ordered run for the core paper-facing benchmark. It assumes `configs/data/dataset.yaml` is the shared-scenario dataset with hydrostatic, MUSCL-HR, and Boussinesq enabled. Use `--num-workers` and `--num-samples` as CLI overrides if the machine/run needs them; otherwise the YAML values are used. Extra diagnostics, uncertainty, arrival-map, learning-curve, and figure-export commands are listed in the detailed sections after `6.9`.

```bash
# 1. Raw data, same scenarios for all three solvers.
python scripts/make_dataset.py --config configs/data/dataset.yaml

# 2. Preprocess all three solver targets with one shared split.
python src/data_gen/preprocess.py --config configs/data/preprocess.yaml

# 3. Train the three target-specific FNOs.
python scripts/train.py --config configs/model/fno.yaml
python scripts/train.py --config configs/model/fno_muscl_hr.yaml
python scripts/train.py --config configs/model/fno_boussinesq.yaml

# 4. Same-target accuracy.
python scripts/eval_accuracy.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt
python scripts/eval_accuracy.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_accuracy.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt

# 5. Model inference speed. Keep CPU and CUDA rows if CUDA is available.
python scripts/eval_speed.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt          --device cpu  --precision fp32 --allow-tf32 false --output results/speed/model_speed_fno_cpu.json
python scripts/eval_speed.py --config configs/model/fno.yaml          --checkpoint experiments/fno/best.pt          --device cuda --precision fp32 --allow-tf32 true  --output results/speed/model_speed_fno_cuda.json
python scripts/eval_speed.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --device cpu  --precision fp32 --allow-tf32 false --output results/speed/model_speed_muscl_hr_cpu.json
python scripts/eval_speed.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --device cuda --precision fp32 --allow-tf32 true  --output results/speed/model_speed_muscl_hr_cuda.json
python scripts/eval_speed.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt --device cpu  --precision fp32 --allow-tf32 false --output results/speed/model_speed_boussinesq_cpu.json
python scripts/eval_speed.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt --device cuda --precision fp32 --allow-tf32 true  --output results/speed/model_speed_boussinesq_cuda.json

# 6. Reference-solver speed. This is the speedup denominator.
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_hydrostatic --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_swe_hydrostatic.json
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_muscl_hr    --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_swe_muscl_hr.json
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver boussinesq      --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_boussinesq.json

# 7. Aggregate speed table.
python scripts/make_speed_table.py \
  --solver results/speed/solver_speed_swe_hydrostatic.json \
  --solver results/speed/solver_speed_swe_muscl_hr.json \
  --solver results/speed/solver_speed_boussinesq.json \
  --model  results/speed/model_speed_fno_cpu.json \
  --model  results/speed/model_speed_fno_cuda.json \
  --model  results/speed/model_speed_muscl_hr_cpu.json \
  --model  results/speed/model_speed_muscl_hr_cuda.json \
  --model  results/speed/model_speed_boussinesq_cpu.json \
  --model  results/speed/model_speed_boussinesq_cuda.json \
  --output results/speed/speed_table.csv \
  --output-json results/speed/speed_table.json

# 8. OOD generalization.
python scripts/make_ood_splits.py --config configs/data/ood_splits_hydrostatic.yaml --overwrite
python scripts/make_ood_splits.py --config configs/data/ood_splits_muscl_hr.yaml --overwrite
python scripts/make_ood_splits.py --config configs/data/ood_splits_boussinesq.yaml --overwrite
python scripts/eval_generalization.py --config configs/eval/ood_suites_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_generalization.py --config configs/eval/ood_suites_muscl_hr.yaml   --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_generalization.py --config configs/eval/ood_suites_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt

# 9. Proxy cross-resolution transfer, no extra simulation.
python scripts/eval_resolution_transfer.py --config configs/eval/resolution_transfer_proxy_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_resolution_transfer.py --config configs/eval/resolution_transfer_proxy_muscl_hr.yaml   --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_resolution_transfer.py --config configs/eval/resolution_transfer_proxy_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt

# 10. Native cross-resolution. Currently configured for hydrostatic and MUSCL-HR.
python scripts/make_dataset.py --config configs/data/multires/dataset_32.yaml
python scripts/make_dataset.py --config configs/data/multires/dataset_64.yaml
python scripts/make_dataset.py --config configs/data/multires/dataset_128.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_32.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_64.yaml
python src/data_gen/preprocess.py --config configs/data/multires/preprocess_128.yaml
python scripts/eval_full_resolution.py --config configs/eval/resolution_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_full_resolution.py --config configs/eval/resolution_muscl_hr.yaml   --checkpoint experiments/fno_muscl_hr/best.pt

# 11. Solver-vs-solver physical gaps.
# This shows the main Hydro/MUSCL denominator; see 6.9 for all pair directions.
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --require-quality-ok --save-arrival-maps \
  --arrival-maps-output results/solver_compare_hydro_vs_muscl_hr_arrival_maps.npz \
  --output results/solver_compare_hydro_vs_muscl_hr.json
```

#### 6.0.3 Where each paper table/figure comes from

| Paper artifact | Produced by | Output |
| --- | --- | --- |
| Same-target accuracy | `eval_accuracy.py` | `experiments/<model>/eval/metrics.json` |
| Runtime + speedup | `eval_speed.py` + `eval_solver_speed.py` -> `make_speed_table.py` | `results/speed/speed_table.{csv,json}` |
| OOD generalization | `make_ood_splits.py` + `eval_generalization.py` | `experiments/<model>/eval_ood_suites/ood_generalization.json` |
| Proxy cross-resolution | `eval_resolution_transfer.py` | `.../eval_resolution_proxy/resolution_transfer_proxy.json` |
| Native 32/64/128 resolution | `eval_full_resolution.py` | `.../eval_resolution/real_resolution.json` |
| Solver physical gap | `compare_solvers_physical.py` | `results/solver_compare_*.json` |
| Emulator-superiority ratio | `eval_emulator_superiority.py` | `results/emulator_superiority_*.json` |
| Arrival maps | `eval_arrival_maps.py`, `compare_solvers_physical.py --save-arrival-maps` | `...arrival_map*.{json,npz}` |
| Learning curves | `run_sample_scaling.py` | `experiments/sample_scaling/*/sample_scaling_results.{csv,json}` |
| Uncertainty | `train_ensemble.py` + `eval_uncertainty.py` | `.../eval_uncertainty*/uncertainty*.json` |
| Qualitative maps | `export_figures.py` or `visualize_rollout.py` | `paper/figs/...` |

### 6.1 Step 1 - Generate Forward Raw Dataset (Required)

Main benchmark generation. The default paper-facing dataset is shared across hydrostatic, MUSCL-HR, and Boussinesq by sample ID:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml
```

For a larger server run, prefer CLI overrides rather than editing committed YAML:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --num-workers 64
```

`make_dataset.py` runs in three stages:
- stage 1: generate/cache all bathymetry samples first (default cache: `data/bathymetry`)
- stage 2: generate/cache all source samples first (default cache: `data/sources`)
- stage 3: load cached bathymetry + source pairs and run configured FDE rollouts from `fdes.enabled`

Raw rollouts are separated by solver under `data/raw/`:
- `data/raw/hydrostatic/samples/...`
- `data/raw/muscl_hr/samples/...`
- `data/raw/boussinesq/samples/...`

Manifests are separated as:
- scenario-level: `data/synthetic/scenario_manifest.jsonl`
- solver-level: `data/synthetic/hydrostatic_manifest.jsonl`, `data/synthetic/muscl_hr_manifest.jsonl`, `data/synthetic/boussinesq_manifest.jsonl`

Runnable FDEs currently include `swe_hydrostatic`, `swe_muscl_hr`, and `boussinesq`.
Default `configs/data/dataset.yaml` enables all three so the raw targets are comparable on the same bathymetry/source scenarios.
Legacy alias `swe_muscl` is still accepted and automatically mapped to `swe_muscl_hr` for backward compatibility.

Storage-limited server workflow:
- On the server, generate only hydrostatic + MUSCL-HR if storage is tight.
- Download `data/bathymetry`, `data/sources`, `data/raw/hydrostatic`, `data/raw/muscl_hr`, and `data/synthetic`.
- Locally, run the default all-three config with `--continue`; completed hydrostatic/MUSCL folders are reused and only missing Boussinesq folders are generated.
- Rebuild manifests after the local completion so the scenario manifest records all three solvers.
- If you pass `--num-samples` on the server, pass the same value again for the local `--continue` run.

One way to make the temporary server-only hydro/MUSCL config without committing another YAML file:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/data/dataset.yaml").read_text())
cfg["fdes"]["enabled"] = ["swe_hydrostatic", "swe_muscl_hr"]
cfg["fdes"]["primary"] = "swe_hydrostatic"
Path("/tmp/dataset_hydro_muscl.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

python scripts/make_dataset.py --config /tmp/dataset_hydro_muscl.yaml --num-workers 64
```

Then, after copying the generated folders down locally:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --continue
python scripts/make_dataset.py --config configs/data/dataset.yaml --rebuild-manifests
```

Do not use `configs/data/dataset_boussinesq.yaml` for the main same-scenario paper dataset. That file intentionally uses a separate diagnostic Boussinesq regime (`data/raw_bouss`, different bathymetry/source configs, and different depth/source scaling).

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
- `max_eta_over_depth`
- `require_cg_converged`
- Recommendation: keep `on_violation: fail` (now default in provided dataset configs) so unstable samples do not silently enter raw manifests.

### 6.2 Step 2 - Preprocess Forward Data (Required)

Main benchmark preprocessing:

```bash
python src/data_gen/preprocess.py --config configs/data/preprocess.yaml
```

Large paper-facing preprocess configs use bounded shards by default:
- `saving.sharded: true`
- `saving.shard_size: 128`
- `saving.write_legacy_eval_archive: false`

This keeps preprocessing and training RAM-bounded. Existing model config paths ending in `.../eval_dataset.npz` still work: when the monolithic archive is absent and the split is sharded, the loader falls back to that file's parent directory and reads `shards_manifest.json`. Paths may also point directly at the split folder, e.g. `data/processed/hydrostatic/train`.
For sharded training splits, the loader uses a shard-aware batch sampler: it shuffles shard order and sample order within each shard, but keeps each mini-batch inside one shard to avoid repeatedly reloading compressed shard files.

`preprocess.yaml` supports FDE-aware modes:
- `fde.mode: single` with `fde.targets: [hydrostatic]` writes to `data/processed/hydrostatic/...`
- `fde.mode: separate_all` writes one processed dataset per solver (`hydrostatic`, `muscl_hr`, `boussinesq`) using the same scenario split
- `fde.mode: multifidelity` writes a combined dataset to `data/processed/multifidelity/...`
- For `multifidelity`, keep `input.use_solver_id: true` (or omit it, since it auto-enables by default) so the model can condition on solver identity instead of learning an ambiguous one-to-many mapping

Boussinesq-only preprocessing for the separate diagnostic regime:

```bash
python src/data_gen/preprocess.py --config configs/data/preprocess_boussinesq.yaml
```

Use this only with `configs/data/dataset_boussinesq.yaml` outputs. For the main same-scenario dataset, `configs/data/preprocess.yaml` already exports Boussinesq together with hydrostatic and MUSCL-HR.

Main outputs used by training/eval:
- `data/processed/hydrostatic/{train,val,test}/shards_manifest.json`
- `data/processed/muscl_hr/{train,val,test}/shards_manifest.json`
- `data/processed/boussinesq/{train,val,test}/shards_manifest.json`
- per-split shards under `.../{train,val,test}/shards/shard_*.npz`

Small/debug configs can still use the legacy single-archive format by setting `saving.sharded: false`, which writes `eval_dataset.npz` as before.

### 6.3 Step 3 - Train Forward Models (Required Before Eval)

Train hydrostatic-label FNO:

```bash
python scripts/train.py --config configs/model/fno.yaml
```

Train MUSCL-HR-label FNO:

```bash
python scripts/train.py --config configs/model/fno_muscl_hr.yaml
```

Train Boussinesq-label FNO:

```bash
python scripts/train.py --config configs/model/fno_boussinesq.yaml
```

To train replicated models sequentially, add a top-level seed list to the model
config. The existing single `seed` behavior is unchanged when `seeds` is absent.

```yaml
seeds: [18, 36, 67, 72, 154]
```

For `output_dir: experiments/fno`, these runs are written to
`experiments/fno/fno_seed_18`, `experiments/fno/fno_seed_36`, and so on. Each
directory contains the complete run artifacts, including resolved config,
metadata, history, and checkpoints.

The headline Hydrostatic FNO and F-FNO configs use all five frozen seeds. Every
other ordinary model config uses the shared three-seed subset `[18, 36, 67]`;
the dedicated uncertainty-ensemble config retains its own member seeds. A seed
list does not make legacy holdout or native-resolution data compatible with the
common-time-v2 core, so run those configurations only under their own protocol.

Optional training tracks:
- Ensemble for uncertainty: `python scripts/train_ensemble.py --config configs/model/fno.yaml`
- Experimental (not reported in the paper): a ConvLSTM baseline exists in the code
  (`configs/model/convlstm.yaml`) but did not converge under the training budget and
  is excluded from all paper results.

Native-resolution training tracks (P2 extension):
- Hydrostatic: `configs/model/fno_res{32,64,128}_hydrostatic.yaml`
- MUSCL-HR: `configs/model/fno_res{32,64,128}_muscl_hr.yaml`
- Shared-from64 normalization checkpoints:
  - `configs/model/fno_res64_shared_from64_hydrostatic.yaml`
  - `configs/model/fno_res64_shared_from64_muscl_hr.yaml`

### 6.4 Step 4 - Same-Target Accuracy Eval

After `6.3`, evaluate each model on its matching processed test set:

```bash
python scripts/eval_accuracy.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_accuracy.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_accuracy.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt
```

Eval notes:
- `eval_accuracy.py`, `eval_generalization.py`, and `eval_resolution_transfer.py` report normalized metrics by default and add `*_physical` metrics automatically when target denormalization stats are available in the evaluation dataset archive
- `eval_generalization.py` supports explicit OOD suites via `eval.generalization.suites` (or top-level `generalization.suites`) in config
- Single-member uncertainty is blocked by design (degenerate variance)
- Train/eval entrypoints validate dataset-vs-model I/O channels early, so stale `model.in_channels` / `model.out_channels` mismatches fail fast
- Eval JSON outputs now include sample-count metadata (`num_samples` or `dataset_num_samples`) so paper tables can report support size explicitly
- `--device` now overrides config in eval entrypoints (`eval_accuracy`, `eval_generalization`, `eval_uncertainty`, `eval_arrival_maps`, `eval_emulator_superiority`)

### 6.5 Step 5 - Runtime and Speedup Benchmark

Runtime speedup uses CPU NumPy solver timing as the denominator and FNO inference timing as the numerator. Run CPU model timing for fairness and CUDA timing for the practical accelerator result.

```bash
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --device cpu --precision fp32 --allow-tf32 false --output results/speed/model_speed_fno_cpu.json
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --device cuda --precision fp32 --allow-tf32 true --output results/speed/model_speed_fno_cuda.json
python scripts/eval_speed.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --device cpu --precision fp32 --allow-tf32 false --output results/speed/model_speed_muscl_hr_cpu.json
python scripts/eval_speed.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --device cuda --precision fp32 --allow-tf32 true --output results/speed/model_speed_muscl_hr_cuda.json
python scripts/eval_speed.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt --device cpu --precision fp32 --allow-tf32 false --output results/speed/model_speed_boussinesq_cpu.json
python scripts/eval_speed.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt --device cuda --precision fp32 --allow-tf32 true --output results/speed/model_speed_boussinesq_cuda.json

python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_hydrostatic --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_swe_hydrostatic.json
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver swe_muscl_hr --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_swe_muscl_hr.json
python scripts/eval_solver_speed.py --config configs/data/dataset.yaml --solver boussinesq --device cpu --precision float64 --repeats 3 --max-samples 8 --output results/speed/solver_speed_boussinesq.json

python scripts/make_speed_table.py \
  --solver results/speed/solver_speed_swe_hydrostatic.json \
  --solver results/speed/solver_speed_swe_muscl_hr.json \
  --solver results/speed/solver_speed_boussinesq.json \
  --model results/speed/model_speed_fno_cpu.json \
  --model results/speed/model_speed_fno_cuda.json \
  --model results/speed/model_speed_muscl_hr_cpu.json \
  --model results/speed/model_speed_muscl_hr_cuda.json \
  --model results/speed/model_speed_boussinesq_cpu.json \
  --model results/speed/model_speed_boussinesq_cuda.json \
  --output results/speed/speed_table.csv \
  --output-json results/speed/speed_table.json
```

### 6.6 Step 6 - OOD Suite Evaluation

Prerequisites:
- processed test archives from `6.2`
- trained checkpoints from `6.3`

Build OOD suite datasets from processed test archives:

```bash
python scripts/make_ood_splits.py --config configs/data/ood_splits_hydrostatic.yaml --overwrite
python scripts/make_ood_splits.py --config configs/data/ood_splits_muscl_hr.yaml --overwrite
python scripts/make_ood_splits.py --config configs/data/ood_splits_boussinesq.yaml --overwrite
```

Tip:
- `make_ood_splits.py` prints available `source_type` / `bathymetry_type` counts and `source_strength` range from your current test archive.
- If a suite selects zero samples, relax or change filters in `configs/data/ood_splits_*.yaml`, rebuild suites, then rerun evaluation.
- OOD split configs now support `min_samples` + `min_samples_action: warn|fail` to prevent accidentally evaluating tiny suites.

Run suite-based generalization evaluation:

```bash
python scripts/eval_generalization.py --config configs/eval/ood_suites_hydrostatic.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_generalization.py --config configs/eval/ood_suites_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_generalization.py --config configs/eval/ood_suites_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt
```

Output file:
- `.../eval_ood_suites/ood_generalization.json`

### 6.7 Step 7 - Resolution Transfer (Proxy, No Extra Training)

This track uses one trained checkpoint from `6.3` and evaluates it on resized versions of one test archive.
It is a proxy study, not native re-simulation at each resolution.

```bash
python scripts/eval_resolution_transfer.py \
  --config configs/eval/resolution_transfer_proxy_hydrostatic.yaml \
  --checkpoint experiments/fno/best.pt
python scripts/eval_resolution_transfer.py \
  --config configs/eval/resolution_transfer_proxy_muscl_hr.yaml \
  --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_resolution_transfer.py \
  --config configs/eval/resolution_transfer_proxy_boussinesq.yaml \
  --checkpoint experiments/fno_boussinesq/best.pt
```

Output file:
- `.../eval_resolution_proxy/resolution_transfer_proxy.json`

### 6.8 Step 8 - Real-Resolution Benchmark (Native 32/64/128)

Prerequisites:
- generate native-grid forward data per resolution
- preprocess each resolution
- use a checkpoint from `6.3` for cross-resolution evaluation

The native-resolution configs use the common-time-v2 three-reference policy.
Every resolution deterministically regenerates the same seed-763 `128 x 128`
master bathymetry/source scenario and area-averages that master to the target
grid. Before the first rollout, `make_dataset.py` freezes and verifies the full
1,000-scenario input inventory under `data/res*/synthetic/`. Hydrostatic and
MUSCL-HR use radiation boundaries; Boussinesq uses the accepted open-boundary,
sparse-LU policy. All sponges remain outside the published crop.
Raw generation and preprocessing cover all three references; dedicated
Boussinesq native-resolution model/evaluation configs are not frozen yet.

Generate native-grid raw datasets:

```bash
python scripts/make_dataset.py --config configs/data/multires/dataset_32.yaml
python scripts/make_dataset.py --config configs/data/multires/dataset_64.yaml
python scripts/make_dataset.py --config configs/data/multires/dataset_128.yaml
```

On a new machine, run one fresh scenario first by adding `--stop-at 1`. After
checking its 50 timestamps, quality status, crop, provenance, and peak memory,
resume the exact same config with `--continue`. The conservative 128-grid
default is eight single-thread workers; the local full-horizon canary for its
actual `192 x 192` computational grid peaked near 208 MiB for one worker.

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

### 6.9 Step 9 - Solver-vs-Solver Physical Comparison

Compare raw solver labels on shared scenarios in physical eta units. The Hydro/MUSCL pair is the main denominator for the emulator-superiority experiment; Boussinesq pairs are useful for physical-gap reporting if the Boussinesq quality gates pass.

```bash
python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --require-quality-ok --missing-quality-action include --save-arrival-maps \
  --output results/solver_compare_hydro_vs_muscl_hr.json

python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/muscl_hr/samples \
  --solver-b-dir data/raw/hydrostatic/samples \
  --require-quality-ok --missing-quality-action include --save-arrival-maps \
  --output results/solver_compare_muscl_hr_vs_hydro.json

python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/hydrostatic/samples \
  --solver-b-dir data/raw/boussinesq/samples \
  --require-quality-ok --missing-quality-action include --save-arrival-maps \
  --output results/solver_compare_hydro_vs_boussinesq.json

python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/boussinesq/samples \
  --solver-b-dir data/raw/hydrostatic/samples \
  --require-quality-ok --missing-quality-action include --save-arrival-maps \
  --output results/solver_compare_boussinesq_vs_hydro.json

python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/muscl_hr/samples \
  --solver-b-dir data/raw/boussinesq/samples \
  --require-quality-ok --missing-quality-action include --save-arrival-maps \
  --output results/solver_compare_muscl_hr_vs_boussinesq.json

python scripts/compare_solvers_physical.py \
  --solver-a-dir data/raw/boussinesq/samples \
  --solver-b-dir data/raw/muscl_hr/samples \
  --require-quality-ok --missing-quality-action include --save-arrival-maps \
  --output results/solver_compare_boussinesq_vs_muscl_hr.json
```

The comparison includes pointwise physical metrics, spectral differences, and arrival-time differences in timestep units and seconds when timestamps are available. Use `--arrival-threshold-fraction 0.05` to change the arrival threshold.

### 6.10 Step 10 - Emulator Superiority Ratio

Compute:
`error(FNO trained on A, solver B) / error(solver A, solver B)`

These configs currently cover the Hydrostatic/MUSCL-HR pair only. Run the matching solver comparison JSONs in `6.9` first.

```bash
python scripts/eval_emulator_superiority.py \
  --config configs/eval/emulator_superiority_hydro_to_muscl_hr.yaml
python scripts/eval_emulator_superiority.py \
  --config configs/eval/emulator_superiority_muscl_hr_to_hydro.yaml
```

Safety notes:
- default numerator metric is now `rmse_physical_separate_denorm`, which denormalizes predictions using checkpoint-train stats and targets using eval-target stats.
- if normalization signatures mismatch, unsafe numerator metrics are blocked (`fail` by default) to avoid misleading emulator-superiority ratios.

### 6.11 Optional - Boussinesq Propagation Diagnostic

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

### 6.12 Optional - Inverse Dataset Scaffold (Separate Follow-Up Track)

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

### 6.13 Quick Smoke Run

```bash
bash scripts/quickstart.sh
```

### 6.14 Visualize One Sample (Truth vs Prediction + Uncertainty)

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

### 6.15 Optional - OOD Uncertainty Suites

Train an ensemble, then evaluate uncertainty metrics on OOD suite datasets:

```bash
python scripts/train_ensemble.py --config configs/model/fno.yaml

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
- Train separate ensemble member directories per solver target before comparing Hydrostatic and MUSCL-HR uncertainty. The default `ensemble.member_dir_template` from the base config is shared, so do not overwrite one target's ensemble with another by accident.

### 6.16 Optional - Arrival-Time Maps (Model vs Target Solver)

```bash
python scripts/eval_arrival_maps.py \
  --config configs/model/fno.yaml \
  --checkpoint experiments/fno/best.pt
python scripts/eval_arrival_maps.py \
  --config configs/model/fno_muscl_hr.yaml \
  --checkpoint experiments/fno_muscl_hr/best.pt
python scripts/eval_arrival_maps.py \
  --config configs/model/fno_boussinesq.yaml \
  --checkpoint experiments/fno_boussinesq/best.pt
```

Outputs:
- `.../eval/arrival_map_model_vs_target.json`
- `.../eval/arrival_map_model_vs_target.npz`

### 6.17 Optional - Learning Curves and Figure Exports

Learning curves train/evaluate one model family at several training-set sizes:

```bash
python scripts/run_sample_scaling.py --config configs/model/fno.yaml --samples 100,500,1000,2500,5000,10000 --output-root experiments/sample_scaling/fno --device cuda
python scripts/run_sample_scaling.py --config configs/model/fno_muscl_hr.yaml --samples 100,500,1000,2500,5000,10000 --output-root experiments/sample_scaling/fno_muscl_hr --device cuda
python scripts/run_sample_scaling.py --config configs/model/fno_boussinesq.yaml --samples 100,500,1000,2500,5000,10000 --output-root experiments/sample_scaling/fno_boussinesq --device cuda
```

Qualitative prediction figures for the paper:

```bash
python scripts/export_figures.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt --out paper/figs/fno_hydrostatic_prediction.png
python scripts/export_figures.py --config configs/model/fno_muscl_hr.yaml --checkpoint experiments/fno_muscl_hr/best.pt --out paper/figs/fno_muscl_hr_prediction.png
python scripts/export_figures.py --config configs/model/fno_boussinesq.yaml --checkpoint experiments/fno_boussinesq/best.pt --out paper/figs/fno_boussinesq_prediction.png
```

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
│  │  ├─ ood_splits_boussinesq.yaml
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
│     ├─ ood_suites_boussinesq.yaml
│     ├─ uncertainty_ood_hydrostatic.yaml
│     ├─ uncertainty_ood_muscl_hr.yaml
│     ├─ resolution_transfer_proxy_hydrostatic.yaml
│     ├─ resolution_transfer_proxy_muscl_hr.yaml
│     ├─ resolution_transfer_proxy_boussinesq.yaml
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
- FNO-centered surrogate evaluation against CNN/U-Net baselines;
- emphasis on speed-accuracy-robustness trade-offs;
- explicit non-operational scope (research benchmark, not production warning stack);
- inverse-problem work kept as separate follow-up paper scope, not part of forward-surrogate claims here.

## 9) Notes

- Development note: Portions of the codebase were developed with AI-assisted programming support. All code should be treated as author-reviewed research software, with tests and validation required before use in reported experiments.
- Test split tip: quick CI/local smoke can use `pytest -q -m "not slow"`; full solver dynamics validation can use `pytest -q -m slow`.
