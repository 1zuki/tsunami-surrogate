# Experiment 6: curriculum by resolution

## Question

Does progressive training from coarse to fine spatial grids improve final high-resolution performance, optimization stability, or sample efficiency?

## Stages

1. train on `32x32`
2. continue on `64x64`
3. continue on `128x128`

## Motivation

This experiment is a natural fit for neural operators because the scientific question is not only whether the model predicts well, but whether it scales gracefully across discretizations.

## Files

- `configs/curriculum_resolution.yaml`
- `configs/curriculum_stage_32.yaml`
- `configs/curriculum_stage_64.yaml`
- `configs/curriculum_stage_128.yaml`
- `src/training/curriculum.py`

## Run

```bash
python src/training/curriculum.py --config configs/curriculum_resolution.yaml
```

## Compare against

- direct training at `128x128`
- CNN / U-Net baselines trained only at the target resolution
- low-to-high curriculum vs high-resolution-from-scratch FNO

## Suggested metrics

- RMSE / MAE on held-out test data
- wall-clock training time
- convergence curves
- memory usage per stage
- cross-resolution transfer gap

## Suggested figure

A three-panel figure works well:

1. training curves by stage
2. final test error by method
3. qualitative wave maps at `128x128`
