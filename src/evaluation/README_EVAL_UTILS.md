# Evaluation + Utils package for tsunami-surrogate

This package adds the files for:

- `src/evaluation/eval_speed.py`
- `src/evaluation/eval_accuracy.py`
- `src/evaluation/eval_generalization.py`
- `src/evaluation/eval_uncertainty.py`
- `src/utils/seed.py`
- `src/utils/logger.py`
- `src/utils/visualization.py`

It also includes:

- `src/evaluation/_common.py` for shared loading, batching, benchmarking, and metric logic
- `configs/eval_template.yaml` as a ready starting config

## What the code assumes

The evaluation code is built around the mapping:

`input field(s) -> rollout tensor [T, H, W]`

The most direct supported setup is:

- inputs per sample: `[C, H, W]`
- targets per sample: `[T, H, W]`
- model output: `[B, T, H, W]`

It supports dataset storage in either of these forms:

1. A single `.npz` file containing arrays like:
   - `inputs`: `[N, C, H, W]`
   - `targets`: `[N, T, H, W]`
2. A directory of per-sample `.npz` files containing `inputs` and `targets`
3. Separate `.npy` files via `inputs_path` and `targets_path`

## Typical usage

```bash
python src/evaluation/eval_accuracy.py --config configs/eval_template.yaml --checkpoint results/checkpoints/best.pt
python src/evaluation/eval_speed.py --config configs/eval_template.yaml --checkpoint results/checkpoints/best.pt
python src/evaluation/eval_generalization.py --config configs/eval_template.yaml --checkpoint results/checkpoints/best.pt
python src/evaluation/eval_uncertainty.py --config configs/eval_template.yaml --checkpoint results/checkpoints/best.pt
```

## Uncertainty modes

`eval_uncertainty.py` supports:

- `mc_dropout`
- `ensemble`
- `direct`

### `mc_dropout`
Your model should contain dropout layers.

### `ensemble`
Provide multiple checkpoints under `uncertainty.checkpoints`.

### `direct`
Your model should return either:

- a dict with `mean` and one of `variance`, `var`, `std`, or `logvar`
- or a tuple/list like `(mean, variance_or_logvar)`

## Simulator speed benchmarking

To compute speedup versus the physics simulator, set:

```yaml
speed:
  simulator_callable: src.solver.shallow_water:simulate_rollout
```

The callable should accept a single sample input tensor on CPU as the first argument and return the simulator rollout.

## Notes

- The package is intentionally config-driven because your final dataset format and model constructor are not yet fixed.
- Normalization is optional and can be turned on via the YAML config.
- All outputs are saved automatically under the configured evaluation output directory.
