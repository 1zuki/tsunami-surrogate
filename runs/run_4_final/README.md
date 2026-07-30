# Run 4 Final Archive

This directory is the preserved result/log snapshot for the legacy-v1 Run 4
evaluation campaign. Files inside this archive should not be silently replaced
by later development evaluations, even when a later file uses the same relative
path under the working `results/` directory.

## Archive policy

- `results/` is the ignored working-output directory used by evaluation scripts.
- `runs/run_4_final/results/` is the tracked Run 4 archive.
- Existing archive files are immutable by default.
- A working result may be added only when a Run 4 log explicitly identifies it
  as an output and the destination path does not already exist.
- Paths embedded inside JSON/CSV files retain their original working
  `results/...` values; they describe where the command wrote the artifact.

## July 30, 2026 cleanup

Twenty previously unarchived files were moved here without overwriting any
existing file. Their provenance is recorded by:

- `logs/validation_20260703_summary.log`
  - `results/solver_convergence_minimal/`
  - `results/swe_standard_validation/`
  - `results/reviewer_validation/boussinesq_dispersion/`
  - `results/reviewer_validation/failure_cases/`
  - `results/reviewer_validation/ensemble_calibration/`
- `logs/boussinesq_dispersion_full_20260703.log`
  - `results/reviewer_validation/boussinesq_dispersion_full/`
- `logs/ensemble_calibration_full_20260703.log`
  - `results/reviewer_validation/ensemble_calibration_full/`

The cleanup intentionally did **not** archive:

- common-time-v2 audit and dense-validation outputs created after Run 4;
- `solver_convergence_minimal_smoke/`;
- the later development solver-speed output under `results/speed/`;
- newer root files whose relative paths already exist here but whose hashes
  differ, including `all_results.json` and the strict-holdout summaries.

This preserves Run 4 as a historical snapshot rather than turning it into a
mixture of Run 4 and later development evidence.
