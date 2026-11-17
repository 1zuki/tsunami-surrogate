# Nested-grid self-convergence diagnostic

The 2026-11-17 production replay supersedes the legacy dense-replay section
below for the manuscript. The legacy replay used a separate 64-cell,
0.004--0.08 normalized configuration and is retained only as historical
implementation evidence. The production results are recorded in
`paper/notes/production_dense_validation_20261117.md`.

This diagnostic addresses the reviewer request for a small, reproducible refinement check. It is an empirical self-convergence study, not an exact-solution proof.

The run uses the repository's scripts/run_solver_convergence_minimal.py on twelve deterministic smooth variable-bathymetry scenarios (seed=20260703). Each master scenario is generated at 128 x 128 and restricted by cell averaging to 64 x 64 and 32 x 32. All rollouts reach t=0.02 with adaptive stepping, reflective boundaries, and the sponge disabled so that grid contraction is not mixed with production boundary damping. The sweep evaluates both solvers at both CFL values; the main rows use CFL=0.1125 for Hydrostatic and CFL=0.225 for MUSCL-HR, while the cross-CFL rows provide a sensitivity check.

~~~
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=/tmp/tsunami-surrogate-pythonpath \
.venv/bin/python -u scripts/run_solver_convergence_minimal.py \
  --output-dir results/reviewer_revision/solver_convergence_production \
  --num-scenarios 12 --seed 20260703 --grids 32 64 128 \
  --cfl-values 0.1125 0.225 --final-time 0.02 \
  --solvers swe_hydrostatic swe_muscl_hr \
  --boundary reflective
~~~

The run completed all 144 rollouts without a non-finite state or failed case. The maximum recorded CFL was 0.2254 for the requested 0.225 value. For each scenario and solver, the observed order is

~~~
p = log2(E_32/64 / E_64/128)
~~~

where the fine-grid field is restricted by the same cell-mean operator before comparison. The Hydrostatic median orders were 0.457 for relative L2 and 0.357 for RMSE, with minimum relative-L2 contraction ratio 1.317. MUSCL-HR gave 1.456 and 1.372, with minimum contraction ratio 2.533. The alternate-CFL rows gave Hydrostatic 0.495/0.393 with minimum contraction 1.346 and MUSCL-HR 1.454/1.373 with minimum contraction 2.536. These numbers support a contracting finite-resolution hierarchy for this smooth benchmark. They do not establish exact-solution accuracy, convergence on every production scenario, or physical validity.

## Exact-solution periodic linear-wave diagnostic (2026-11-17)

The same current solver code was tested against a small-amplitude right-going sinusoid on a constant-depth periodic domain. The grid sequence was 32, 64, 128, 256, and 512 cells in the propagation direction, with a fixed transverse width of 32 cells, final time 0.5, and target CFL values 0.1125, 0.225, and 0.45. All 30 rollouts were finite.

At the finest grid, Hydrostatic relative-L2 errors were 0.0526, 0.0462, and 0.0334 for the three CFL values; MUSCL-HR errors were 0.007394, 0.007394, and 0.007396. Hydrostatic observed spatial orders increased from about 0.72--0.82 on the 32-to-64 pair to 0.95 on the 256-to-512 pair. MUSCL-HR gave about 1.33--1.35, 1.53--1.54, 1.04--1.05, and 0.285--0.289 on the successive pairs, showing an error floor at the finest level. The finest-grid temporal comparisons were spatial-error dominated. These are solver-specific empirical accuracy results, not formal production-solver orders.

## Full current-code dense interpolation replay (2026-11-17)

The mounted raw bundle did not contain natural trajectories for the full preregistered selection, so the missing trajectories were regenerated with the current solver factories and archived as dense replay data under `/mnt/Windows/Dataset/current_dense_replay_full_20261117`. The bundle contains 120 complete scenarios for each of Hydrostatic, MUSCL-HR, and Boussinesq, with dense diagnostic rows and a 51-frame stride-5 extraction from each 2500-step rollout. The full replay was then audited on the 0.004--0.08 common-time grid.

The audit had zero reproduction failures and zero extrapolation failures. Aggregate interpolation RMSE was 0.00672 of trajectory RMS and 0.06396 of the smallest dense solver gap. The scenario gap-ratio median and 95th percentile were 0.0598 and 0.1156; the scenario RMS-ratio median and 95th percentile were 0.00584 and 0.0165. Every aggregate gate passed. The only failed guardrail was the island--Gaussian family cell, with median ratio 0.1137 against the preregistered 0.10 limit. The result is therefore a full 120-case temporal-extraction audit with one family-cell guardrail failure, not a blanket solver-accuracy pass.
