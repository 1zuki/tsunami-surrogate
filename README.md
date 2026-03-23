![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

# Tsunami Surrogate Modeling with Neural Operators

This project aims to approximate tsunami wave propagation using neural operators 
(e.g. Fourier Neural Operator) as a fast surrogate for traditional physics simulations.

Instead of solving PDEs directly, we train a model to learn the mapping:
bathymetry + initial disturbance → wave evolution over time.

## Motivation

Traditional tsunami simulations are computationally expensive.
This project explores whether neural operators can approximate them
orders of magnitude faster while maintaining acceptable accuracy.


## Model

We use the Fourier Neural Operator (FNO), which is designed to learn mappings between functions and is well-suited for PDE problems.

## Installation

```
bash
git clone https://github.com/xxx/tsunami-surrogate.git
cd tsunami-surrogate
pip install -r requirements.txt
```
## Usage
** Generate dataset **
```
python src/data_gen/simulate_dataset.py
```
** Train model **
```
python src/training/train.py --config configs/fno.yaml
```
** Evaluate model **
```
python src/evaluation/eval_accuracy.py
```

## Stage 1 — Define the mini-problem
**  Use: ** 

- 2D grid
- simple bathymetry
- simple earthquake source
- output = wave height over time

** Start small: ** 

- grid: 32x32
- timesteps: 20

## Stage 2 — Build the physics simulator

** Make a basic shallow water equation simulator in NumPy. **

** This is your data generator. ** 

** For each sample: ** 

- random seabed
- random quake location / strength
- simulate wave propagation

** Save: ** 

- input: bathymetry + initial disturbance
- target: wave maps over time

## Stage 3 — Generate dataset

** Create many simulated examples. ** 

** Example: ** 

- train: 5k–20k samples
- val/test: smaller split

** Important: ** 

- vary earthquake position
- vary strength
- vary seabed shapes

## Stage 4 — Build the model

Use FNO, not CNN.

** Why: ** 

- wave motion is global
- FNO works naturally with grids + PDE-like problems
- more research-worthy

** Model learns: ** 

- from spatial fields
- to future wave field-s

## Stage 5 — Train it

** Train model to minimize error between: ** 

- predicted wave evolution
- simulated wave evolution

** Use: ** 

- MSE loss
- PyTorch
## Stage 6 — Evaluate properly

** This part matters a lot. ** 
 
** Compare: ** 

- accuracy: how close to simulator?
- speed: how much faster than simulator?

## Main result should look like:

> “Our neural operator is X times faster with Y error.”

That’s your core contribution.

## Stage 7 — Make it paper-worthy

** Add one extra angle: ** 

** Pick one: ** 

- uncertainty estimation
- multi-resolution test
- different seabed generalization
- sensor-to-wave inverse version

## Results

| Model | Speedup | MSE |
|------|--------|-----|
| Simulator | 1x | 0 |
| FNO | 50x | 0.02 |

## Visualization

(Add gifs or plots here) //will make latter 

## Research Questions

- Can neural operators generalize to unseen bathymetry?
- How does resolution affect performance?
- What is the speed vs accuracy trade-off?

## Final structure of the project
1/ Build shallow-water simulator
2/ Generate synthetic tsunami dataset
3/ Train FNO surrogate
4/ Compare speed vs accuracy
5/ Write up results like a mini paper


```
tsunami-surrogate/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ base.yaml
│  ├─ fno.yaml
│  ├─ unet.yaml
│  ├─ cnn.yaml
│  ├─ physics_loss.yaml
│  └─ train_32_to_64.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ bathymetry/
│  └─ synthetic/
├─ src/
│  ├─ solver/
│  │  ├─ shallow_water.py
│  │  ├─ boussinesq.py
│  │  ├─ boundary_problems.py
│  │  └─ source_modelling.py
│  ├─ data_gen/
│  │  ├─ generate_bathymetry.py
│  │  ├─ generate_sources.py
│  │  ├─ simulate_dataset.py
│  │  └─ preprocess.py
│  ├─ models/
│  │  ├─ fno.py
│  │  ├─ unet.py
│  │  ├─ cnn.py
│  │  ├─ convlstm.py
│  │  └─ uncertainty.py
│  ├─ training/
│  │  ├─ train.py
│  │  ├─ losses.py
│  │  ├─ metrics.py
│  │  └─ callbacks.py
│  ├─ evaluation/
│  │  ├─ eval_speed.py
│  │  ├─ eval_accuracy.py
│  │  ├─ eval_generalization.py
│  │  └─ eval_uncertainty.py
│  └─ utils/
│     ├─ seed.py
│     ├─ logging.py
│     └─ visualization.py
├─ experiments/
│  ├─ exp1_same_resolution/
│  ├─ exp2_unseen_bathymetry/
│  ├─ exp3_cross_resolution/
│  ├─ exp4_physics_loss_ablation/
│  └─ exp5_uncertainty/
├─ figures/
├─ results/
└─ paper/
   ├─ main.tex
   ├─ references.bib
   └─ figs/
``` 
