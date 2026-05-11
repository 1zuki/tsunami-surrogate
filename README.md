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
pip install -r requirement.txt
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

Runnable FDEs currently include `swe_hydrostatic`, `swe_muscl`, and `boussinesq`.

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

### 6.3 Train

```bash
python scripts/train.py --config configs/model/fno.yaml
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
python scripts/eval_uncertainty.py --config configs/model/fno.yaml
```

### 6.5 Quick Smoke Run

```bash
bash scripts/quickstart.sh
```

### 6.6 Visualize One Sample (Truth vs Prediction + Uncertainty)

```bash
python scripts/visualize_rollout.py \
  --config configs/model/fno.yaml \
  --checkpoint experiments/fno/best.pt \
  --processed-path data/processed/test \
  --raw-dir data/raw/samples \
  --sample-index 0
```

## 7) Repository Structure (with inline folder purpose)

```text
tsunami-surrogate/
├─ README.md
├─ LICENSE
├─ requirement.txt
├─ configs/                        # all experiment/data/model/eval configs
│  ├─ data/                        # data-generation and preprocessing configs
│  │  ├─ dataset.yaml              # two-phase generation config (bathymetry cache + FDE list)
│  │  ├─ preprocess.yaml           # raw -> processed split/export config
│  │  ├─ bathymetry.yaml           # bathymetry synthesis controls
│  │  └─ source.yaml               # tsunami source family controls
│  ├─ model/                       # model-centered train/eval configs
│  │  ├─ fno.yaml                  # primary FNO config
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

- Prefer `scripts/make_dataset.py` as the default data-generation path.
- Keep `make_toy_data.py` as a compatibility helper, not the primary workflow.
- Development note: Portions of the codebase were developed with AI-assisted programming support. All code should be treated as author-reviewed research software, with tests and validation required before use in reported experiments.
