# Reference-sensitivity extension

This post-review study was planned on 25 November 2026 and launched at 04:49 Vietnam time that day. The run completed successfully at 06:10 Vietnam time, with all ninety numerical rollouts and all twelve model/seed evaluations verified. Its scenario roster is in [roster.csv](roster.csv), and the unchanged pre-execution selection settings and runtime calculations are in [plan.json](plan.json). Local run status is recorded in `/mnt/Windows/Dataset/reference_sensitivity_30_20261125_044858/status.json`; predictions, numerical checkpoints, and the final score file are written under that run directory.

## Question and selection

For unchanged neural predictions, how much do reported errors and model comparisons change when the Hydrostatic reference core is refined from 128 to 256 cells?

The sample contains thirty additional scenarios: one for each of the five bathymetry families crossed with the six source families. It excludes the six previously reported cases (indices 1, 4, 5, 13, 28, and 34). Selection uses family labels, resolved source strengths, and a fixed hash-based ordering; it reads no model errors. Lower, middle, and higher strengths each contribute ten cases. The tercile boundaries are approximately 0.19508 and 0.24250, calculated from the remaining test-manifest strengths.

This is a family-balanced exploratory extension of an already examined test distribution. It is not a new untouched test set. One scenario per family combination cannot establish within-combination variability, and thirty cases cannot establish reliable uncertainty estimates for every subgroup. High-strength undercoverage remains a separate question.

## Computation and analysis

- Generate Hydrostatic trajectories at solver cores 64, 128, and 256: **90 rollouts**, each through time 420 with 50 requested outputs, reduced to the same 64-by-64 evaluation grid.
- Preserve the original fixed-raster construction: coarsen the original 128-cell input by block means and refine it by repeating parent cells. These results do not establish convergence for smoothly resampled terrain.
- Hold the trained ConvLSTM, U-FNO, FNO, and F-FNO checkpoints fixed at seeds 18, 36, and 67. Do not tune, retrain, or select checkpoints from the new results.
- The primary comparison is the paired change in denormalized per-scenario relative L2 between reference cores 128 and 256. Report both absolute and percentage changes, along with adjacent-grid RMSE.
- Average seeds within each scenario for the primary model summary. Report the fraction of scenarios with model-pair ordering reversals, with seed-specific results secondary. Seeds and frames are repeated measurements, not additional independent scenarios.
- Use equal weights for the thirty family combinations. A paired scenario bootstrap may describe variation across these selected cases, retaining all models, grids, and seeds together. It must not be presented as a design-based population interval: each family combination has only one selected scenario.
- Keep the original six cases separate from the new thirty. Record every numerical failure; do not replace a failed case based on its error or silently omit it.

The optional MUSCL-HR extension adds another 90 rollouts. It is useful for checking solver dependence, but is not required to extend the Hydrostatic fixed-prediction result.

## Runtime estimate

The estimates use the measured elapsed times of the completed six-case study on this machine, rather than theoretical grid scaling. Four workers are assumed, with one numerical-library thread each.

| Scope | Sum of projected task times | Four-worker planning estimate, including prediction evaluation |
|:--|--:|--:|
| Hydrostatic, all three grids | 6.79 hours | **2.5–3.5 hours** |
| Both SWE solvers, all three grids | 15.78 hours | **5–7.5 hours** |

The calculation divides the sum of measured task times by four, adds a 25–75% allowance for different cases, memory pressure, scheduling, and I/O, then allows 15–30 minutes for prediction evaluation. That last allowance has not been benchmarked for this roster. These are planning ranges, not guaranteed completion times or statistical confidence intervals. With one worker, expect roughly 7–10 hours for Hydrostatic and 16–22 hours for both solvers, including overhead.

Before launching, time one selected baseline/fine pair, benchmark prediction evaluation on a small batch, and confirm that four fine-grid jobs fit comfortably in memory. Preserve this roster even if the timing estimate changes. The original six-case tools retain their original scope. The separate execution path is now implemented in `scripts/run_reference_sensitivity_30.py`, with checkpoint-preserving recovery in `scripts/resume_reference_sensitivity_30.py`. The completed run reused the fixed predictions across every reference grid.

Recreate the selection and estimate from the local source data into a new output directory:

```bash
python scripts/plan_reference_sensitivity.py \
  --manifest /path/to/test/synthetic/scenario_manifest.jsonl \
  --completed-study /path/to/benchmark_numerical/benchmark_numerical_study.json \
  --output-dir /path/to/new-plan-directory
```

The canceled 120-scenario reproduction is not part of this study.

## Small numerical checks

The bounded checks use existing solver implementations and do not rerun the main dataset:

1. **Amplitude sweep:** compare amplitudes 0.001, 0.0001, and 0.00001 on 128, 256, and 512 cells, with eight transverse cells, CFL 0.1125, periodic boundaries, no sponge, and final time 0.5. Reuse the matching six rows from the completed 0.001-amplitude study and compute twelve new rollouts at the smaller amplitudes. Compare total, phase, fundamental-amplitude, and nonfundamental errors against the same linearized wave. Residual error at small amplitude must not be labeled pure spatial error.
2. **Output-time sensitivity:** on the original continental/Gaussian scenario (index 1), compare natural-step interpolation with integration steps shortened to land on all 50 output times. Use both SWE solvers at the 128-cell core and common CFL 0.1125: four new rollouts. Preserve the original inputs, boundaries, damping, and output reduction. Report errors against the corresponding same-case reference gap. Since shortening steps also changes integration, this contrast does not isolate interpolation error.

These are narrow diagnostic checks. They do not establish full-distribution numerical accuracy or a formal convergence order. Completed values and source hashes are recorded in `numerical-checks.json` when available.

Both checks are complete: twelve new amplitude runs and four new temporal-output runs. On the 512-cell periodic grid, reducing amplitude from 0.001 to 0.00001 lowers MUSCL-HR relative error from 0.007394 to 0.001722, while Hydrostatic remains near 0.0522. The amplitude-sensitive component decreases, but a residual floor remains. On the continental/Gaussian case, natural-step interpolation versus output-aligned integration differs by 0.0134% relative L2 for Hydrostatic and 0.00182% for MUSCL-HR. These small temporal differences apply only to this case and setup. The original 0.001-amplitude rows were reused, not rerun.

## GEBCO configuration repair

The reported GEBCO experiments used the newer saved configurations. The mismatch was in the public regeneration templates: they still contained an older output schedule, source-strength range, solver grid, buffer, and damping reference interval.

The four public generation files now match the saved run snapshots exactly. The four preprocessing files retain the saved processing settings while naming the corresponding public generation file. Verification compares the snapshots, all thirteen archived output schedules and setting identities, finite output fields, and preprocessing settings. The thirteen fields comprise ten morphology-test crops plus three separate illustrative/stress trajectories.

This repair does not regenerate GEBCO crops, retrain a model, change a reported result, or make the selected checkpoints publicly available. A reader still needs the saved crops, initial inputs, inventories, normalization data, and checkpoints. See [gebco-config-verification.json](gebco-config-verification.json) for the completed configuration check.

The run resumed at 05:15 Vietnam time with ten numerical workers, retaining 32 completed rollouts and all twelve saved prediction sets. Workers now load only the numerical libraries. The current log is `/mnt/Windows/Dataset/reference_sensitivity_30_20261125_044858.resume10.log`; the shared `status.json` remains the current phase record. Worker-count changes are recorded separately from the unchanged scientific execution plan.

The completed extension increased aggregate mean relative L2 by 2.82% (ConvLSTM), 6.60% (F-FNO), 3.68% (U-FNO), and 4.48% (FNO) when refining the reference core from 128 to 256. F-FNO and U-FNO exchange aggregate positions; seven scenarios have at least one seed-averaged pairwise reversal. All thirty cases have contracting adjacent-grid RMSE. Total elapsed wall time was about 81 minutes, including the interruption and switch from four to ten workers. The estimates above remain the original planning estimates. The main-paper table and portable evidence are now in `paper/sections/results.tex` and `paper/data/reference_sensitivity_30.json`.
