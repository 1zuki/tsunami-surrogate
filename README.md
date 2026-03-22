# Tsunami Surrogates

## Struct

```text
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
│  │  ├─ bathymetry.py
│  │  ├─ boussinesq.py
│  │  └─ source_modeling.py
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