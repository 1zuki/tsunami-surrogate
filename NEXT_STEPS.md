# Next steps and stronger research angles

If you want to push this project toward a stronger paper, these are the most promising upgrades.

## 1) Sensor-to-wave inverse forecasting

Instead of predicting the full wave field only from bathymetry and initial disturbance, predict tsunami evolution from sparse virtual sensors or invert source parameters from sparse observations.

Why it is interesting:
- closer to real warning workflows
- connects naturally to inverse problems and data assimilation
- stronger novelty than only forward surrogates

Suggested additions:
- `src/inverse/`
- `src/assimilation/`
- `configs/inverse_sensor.yaml`
- `experiments/exp7_sensor_inverse/`

## 2) Probabilistic FNO

Train the surrogate to output both mean and uncertainty, not only a deterministic prediction.

Ideas:
- heteroscedastic Gaussian head
- quantile regression
- deep ensembles
- evidential regression

Suggested additions:
- `src/models/probabilistic_fno.py`
- `src/training/calibration.py`
- `experiments/exp8_probabilistic_fno/`

## 3) Multi-fidelity training

Use many cheap shallow-water simulations and a smaller number of more accurate weakly dispersive runs.

Possible strategy:
- pretrain on shallow-water data
- finetune on Boussinesq-like data
- evaluate transfer and sample efficiency

## 4) Real bathymetry domain adaptation

Move from synthetic bathymetry to real gridded bathymetry.

Ideas:
- NOAA or GEBCO rasters
- patch extraction and normalization
- train on synthetic, adapt on real

## 5) Physics-aware latent constraints

Add losses that regularize:
- global mass consistency
- spectral content
- energy decay trend
- wavefront arrival consistency at sensor points

## 6) Curriculum by resolution

Train at `32x32`, finetune at `64x64`, then test at `128x128`.

This is a strong story if you want to show that FNO scales more gracefully than CNN baselines.

## 7) Event-family generalization

Create train/test splits where test events use source shapes not seen during training.

Example:
- train on Gaussian + dipole
- test on ring + Okada-like

This is a clean paper experiment because it tests structural generalization rather than only interpolation.

## 8) Hybrid digital-twin angle

A longer-term direction is:
- estimate source parameters from sparse offshore pressure or wave sensors
- then roll forward a surrogate or reduced-order model in real time

That would move the project much closer to modern probabilistic tsunami forecasting and digital-twin research.
