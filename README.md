# Tsunami Surrogate Modeling with Neural Operators

A research-grade codebase for synthetic tsunami simulation, neural-operator surrogates, and paper-ready evaluation.

## Abstract

This repository studies whether neural operators can learn a fast surrogate for tsunami wave propagation. Instead of solving the governing PDEs from scratch for every event, we generate synthetic trajectories with a two-dimensional shallow-water solver and train models to approximate the mapping

`bathymetry + initial seafloor disturbance -> wave-height evolution over time`.

The repository is built around Fourier Neural Operators (FNOs), baseline convolutional models, controlled synthetic data generation, and evaluation protocols for accuracy, speed, generalization, and uncertainty. The project is intentionally compact enough for iteration on a laptop, while keeping a structure that scales to a stronger paper.

## Why this project matters

Tsunami simulation is physically rich but computationally expensive. Neural operators are especially attractive here because they learn operators between functions, not only fixed-dimensional vectors, and they are designed to transfer across discretizations more naturally than standard CNN baselines. That makes curriculum-by-resolution and cross-resolution evaluation natural research directions for this codebase. The original FNO work also highlighted mesh transfer and zero-shot super-resolution as key strengths, which directly motivates the curriculum scaffold included here. 

## Core contributions of this repository

- synthetic tsunami dataset generation with a shallow-water solver
- Fourier Neural Operator, U-Net, CNN, and ConvLSTM baselines
- unified train / evaluate / visualize workflow
- experiment folders aligned with paper sections
- uncertainty evaluation hooks for MC dropout and ensembles
- curriculum-by-resolution training scaffold
- manuscript skeleton under `paper/`

## Project overview

### Inputs

- bathymetry field
- initial disturbance / uplift field

### Targets

- wave-height maps over time

### Main question

Can a neural operator approximate tsunami propagation much faster than the simulator while keeping acceptable error on held-out bathymetry and event families?

## Repository layout

```text
tsunami-surrogate/
|- README.md
|- NEXT_STEPS.md
|- GITHUB_REPO_SETUP.md
|- requirements.txt
|- configs/
|- data/
|- experiments/
|- figures/
|- results/
|- src/
|  |- solver/
|  |- data_gen/
|  |- models/
|  |- training/
|  |- evaluation/
|  `- utils/
`- paper/
```

## Method summary

### 1. Physics simulator

`src/solver/shallow_water.py` generates trajectories on a structured grid. The generator uses random bathymetry and synthetic source fields to produce wave evolution data for supervised learning.

### 2. Surrogate models

`src/models/fno.py` is the main model. Baselines in `cnn.py`, `unet.py`, and `convlstm.py` help position the FNO result more convincingly.

### 3. Training

`src/training/train.py` handles optimization, checkpointing, early stopping, metric logging, and validation plots.

### 4. Evaluation

The evaluation suite covers:

- `eval_accuracy.py`: reconstruction quality on held-out data
- `eval_speed.py`: surrogate-vs-simulator runtime comparison
- `eval_generalization.py`: unseen bathymetry and harder source distributions
- `eval_uncertainty.py`: predictive spread and calibration-style analysis

## Installation

```bash
git clone https://github.com/Acceleratorer/tsunami-surrogate.git
cd tsunami-surrogate
pip install -r requirements.txt
```

## Quick start

### Generate a dataset

```bash
python src/data_gen/simulate_dataset.py --config configs/fno.yaml
```

### Train the FNO baseline

```bash
python src/training/train.py --config configs/fno.yaml
```

### Evaluate

```bash
python src/evaluation/eval_accuracy.py --config configs/fno.yaml
python src/evaluation/eval_speed.py --config configs/fno.yaml
python src/evaluation/eval_generalization.py --config configs/fno.yaml
python src/evaluation/eval_uncertainty.py --config configs/fno.yaml --method mc_dropout
```

## Curriculum by resolution

This repository now includes a clean curriculum scaffold for one of the strongest extension stories in the paper.

### Idea

Train progressively:

1. low resolution `32x32`
2. medium resolution `64x64`
3. high resolution `128x128`

The goal is not only better final performance, but a stronger scientific claim: whether an FNO transfers more gracefully across resolution than fixed-resolution convolutional baselines.

### Files added for this

- `configs/curriculum_resolution.yaml`
- `configs/curriculum_stage_32.yaml`
- `configs/curriculum_stage_64.yaml`
- `configs/curriculum_stage_128.yaml`
- `src/training/curriculum.py`
- `experiments/exp6_curriculum_resolution/README.md`

### Run the curriculum

```bash
python src/training/curriculum.py --config configs/curriculum_resolution.yaml
```

This launcher can optionally generate datasets stage-by-stage and resume each new stage from the previous stage's best checkpoint.

## Recommended paper experiments

### Exp 1. Same-resolution surrogate learning

Train and test at `32x32`.

### Exp 2. Unseen bathymetry

Hold out rougher or structurally different seabeds.

### Exp 3. Cross-resolution transfer

Train lower, test higher.

### Exp 4. Physics-loss ablation

Measure the effect of gradient, temporal, mass, or spectral losses.

### Exp 5. Uncertainty

Compare deterministic prediction error with ensemble or dropout spread.

### Exp 6. Curriculum by resolution

Show whether progressive resolution training improves stability, sample efficiency, or final high-resolution accuracy.

## Suggested headline result

A good final paper should be able to report a sentence of the form:

> Our neural operator achieves an X-fold speedup over the numerical simulator with Y error on held-out tsunami scenarios.

## Data format

Saved `.npz` files contain arrays such as:

- `bathymetry` with shape `[N, H, W]`
- `disturbance` with shape `[N, H, W]`
- `wave` with shape `[N, T, H, W]`

plus metadata for source family, roughness, center, amplitude, and related generation attributes.

## Reproducibility notes

- configuration is YAML-based
- training and evaluation paths are explicit
- seeds are centralized in `src/utils/seed.py`
- logs and checkpoints are written under `results/`
- the paper skeleton is already aligned with the code structure

## GitHub metadata

Suggested repo description, topics, and release text are in:

```text
GITHUB_REPO_SETUP.md
```

## Paper scaffold

The manuscript starter lives in:

```text
paper/main.tex
```

It is paired with section files and a starter bibliography so you can write the paper without reorganizing the project later.

## Practical scope

This is a research prototype, not an operational warning system. The shallow-water simulator is intentionally lightweight so the full data-to-model-to-evaluation loop stays easy to inspect and extend.

## Next directions

See `NEXT_STEPS.md` for stronger extensions such as sensor-to-wave inverse forecasting, probabilistic FNOs, multi-fidelity training, and real bathymetry adaptation.
