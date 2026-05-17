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

Planned paper extension:

5. Inverse problem: recover source characteristics from observed wave signals/fields.

## 3) What Is Implemented vs Planned

- Implemented core: synthetic data generation, preprocessing, forward surrogate training, and benchmark evaluation.
- Implemented models: FNO (primary) with CNN/U-Net and ensemble paths for comparison.
- Implemented evaluations: accuracy, speed, generalization, resolution transfer, and uncertainty.
- Planned extension: dedicated inverse-problem experiments and paper section.

## 4) Canonical Workflow

The default full-module pipeline in this repo is:

1. Generate synthetic physics rollouts.
2. Preprocess into train/val/test tensors.
3. Train surrogate models.
4. Evaluate solver-fidelity, speed, robustness, and uncertainty.
5. Export plots/tables and map outputs into paper sections.
6. Add inverse-problem workflow and reporting as a separate extension track.

## 5) Setup

```bash
git clone https://github.com/1zuki/tsunami-surrogate.git
cd tsunami-surrogate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6) Run Commands

### 6.1 Generate Raw Dataset

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml
```

`make_dataset.py` now runs in three stages:
- stage 1: generate/cache all bathymetry samples first (default cache: `data/bathymetry`);
- stage 2: generate/cache all source samples (default cache: `data/source`);
- stage 3: load cached bathymetry + source pairs and run configured FDE rollouts from `fdes.enabled` in `configs/data/dataset.yaml`.

Raw rollouts are separated by solver under `data/raw/`:
- `data/raw/hydrostatic/samples/...`
- `data/raw/muscl/samples/...`
- `data/raw/boussinesq/samples/...`

Manifests are separated as:
- scenario-level: `data/synthetic/scenario_manifest.jsonl`
- solver-level: `data/synthetic/hydrostatic_manifest.jsonl`, `data/synthetic/muscl_manifest.jsonl`, `data/synthetic/boussinesq_manifest.jsonl`

Runnable FDEs currently include `swe_hydrostatic`, `swe_muscl`, and `boussinesq`.

Recommended Boussinesq configs:

```bash
python scripts/make_dataset.py --config configs/data/dataset_boussinesq.yaml
```

Resume an interrupted run:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --continue
```

`--continue` rolls back to the last completed worker batch boundary before resuming (using `num_workers`) to reduce partial-batch holes after interruptions.

Resume from an explicit sample index (1-based):

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --start-at 142
```

Force regeneration in a range even if outputs already exist:

```bash
python scripts/make_dataset.py --config configs/data/dataset.yaml --start-at 142 --allow-override
```

### 6.2 Preprocess

```bash
python src/data_gen/preprocess.py --config configs/data/preprocess.yaml
```

`preprocess.yaml` now supports FDE-aware modes:
- `fde.mode: single` with `fde.targets: [hydrostatic]` writes to `data/processed/hydrostatic/...`
- `fde.mode: separate_all` writes one processed dataset per solver (`hydrostatic`, `muscl`, `boussinesq`) using the same scenario split
- `fde.mode: multifidelity` writes a combined dataset to `data/processed/multifidelity/...`
- For `multifidelity`, keep `input.use_solver_id: true` (or omit it, since it auto-enables by default) so the model can condition on solver identity instead of learning an ambiguous one-to-many mapping.

### 6.3 Train

```bash
python scripts/train.py --config configs/model/fno.yaml
```

Train on MUSCL labels:

```bash
python scripts/train.py --config configs/model/fno_muscl.yaml
```

Train on Boussinesq labels:

```bash
python scripts/train.py --config configs/model/fno_boussinesq.yaml
```

Optional ensemble run:

```bash
python scripts/train_ensemble.py --config configs/model/fno.yaml
```

### 6.4 Evaluate

```bash
python scripts/eval_accuracy.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_speed.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_generalization.py --config configs/model/fno.yaml --checkpoint experiments/fno/best.pt
python scripts/eval_resolution_transfer.py --config configs/train/train_32_to_64.yaml --checkpoint experiments/fno_32_to_64/best.pt
python scripts/eval_uncertainty.py --config configs/model/fno.yaml \
  --checkpoint experiments/ensemble/member_11/best.pt \
  --checkpoint experiments/ensemble/member_22/best.pt \
  --checkpoint experiments/ensemble/member_33/best.pt
```

Notes:
- `eval_accuracy.py`, `eval_generalization.py`, and `eval_resolution_transfer.py` report normalized metrics by default and add `*_physical` metrics automatically when target denormalization stats are available in the evaluation dataset archive.
- `eval_generalization.py` supports explicit OOD suites via `eval.generalization.suites` (or top-level `generalization.suites`) in config, so you can evaluate separate unseen-regime datasets instead of a single grouped test split.
- For uncertainty evaluation, pass at least 2 checkpoints (or configure `uncertainty.ensemble_checkpoints` with at least 2 members). Single-member ensembles are blocked because they produce degenerate variance.
- Train/eval entrypoints now validate dataset-vs-model I/O channels early, so if multifidelity adds `solver_id`, a stale `model.in_channels` mismatch fails fast with a clear message.

### 6.5 Quick Smoke Run

```bash
bash scripts/quickstart.sh
```

### 6.6 Visualize One Sample (Truth vs Prediction + Uncertainty)

```bash
python scripts/visualize_rollout.py \
  --config configs/model/fno.yaml \
  --checkpoint experiments/fno/best.pt \
  --processed-path data/processed/hydrostatic/test \
  --raw-dir data/raw/hydrostatic/samples \
  --sample-index 0
```

## 7) Current Repository Structure

```text
tsunami-surrogate/
├─ README.md
├─ LICENSE
├─ requirement.txt
├─ configs/                        # all experiment/data/model/eval configs
│  ├─ data/                        # data-generation and preprocessing configs
│  │  ├─ dataset.yaml              # three-stage generation config + per-FDE raw outputs
│  │  ├─ dataset_boussinesq.yaml
│  │  ├─ preprocess.yaml           # raw -> processed split/export config
│  │  ├─ bathymetry.yaml           # bathymetry synthesis controls
│  │  ├─ bathymetry_boussinesq.yaml
│  │  ├─ source.yaml               # tsunami source family controls
│  │  └─ source_boussinesq.yaml
│  ├─ model/                       # model-centered train/eval configs
│  │  ├─ fno.yaml                  # primary FNO config
│  │  ├─ fno_muscl.yaml            # FNO on MUSCL processed labels
│  │  ├─ fno_boussinesq.yaml       # FNO on Boussinesq processed labels
│  │  ├─ cnn.yaml                  # CNN baseline config
│  │  └─ unet.yaml                 # U-Net baseline config
│  ├─ train/                       # shared/base + training variants
│  │  ├─ base.yaml                 # common seed/device/data/train defaults
│  │  ├─ physics_loss.yaml         # physics-regularized FNO variant
│  │  └─ train_32_to_64.yaml       # resolution-transfer training setup
│  └─ eval/
│     └─ eval_template.yaml        # template for standalone eval scripts
├─ scripts/                        # CLI entrypoints (generate/train/eval/export)
├─ src/                            # implementation modules
│  ├─ data_gen/                    # simulation + preprocess pipeline internals
│  ├─ data/                        # dataset loaders and multires dataset wrappers
│  ├─ solver/                      # shallow-water and related numerical solvers
│  ├─ models/                      # FNO/CNN/U-Net/ensemble/uncertainty models
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
- FNO-centered surrogate evaluation against convolutional baselines;
- emphasis on speed-accuracy-robustness trade-offs;
- explicit non-operational scope (research benchmark, not production warning stack);
- explicit plan to include inverse-problem analysis as an additional paper section.

## 9) Notes

- Development note: Portions of the codebase were developed with AI-assisted programming support. All code should be treated as author-reviewed research software, with tests and validation required before use in reported experiments.
