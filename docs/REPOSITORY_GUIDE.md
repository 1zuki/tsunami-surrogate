# Repository guide

The source repository contains the supported workflow, optional research studies, the current manuscript, and compact evidence needed to interpret the reported results.

| Location | Purpose |
| --- | --- |
| `src/`, exposed as `tsunami_surrogate` | Solvers, data preparation, models, training, and evaluation |
| `scripts/` | Commands for generation, training, evaluation, plotting, and release preparation |
| `configs/` | Current workflow and explicitly separate study settings |
| `slurm/` | Cluster job templates and environment setup |
| `tests/` | Regression tests and historical configuration fixtures |
| `paper/` | Current manuscript source, figures, and portable result summaries |
| `artifacts/` | Selected frozen input inventories and historical numerical evidence |
| `docs/reference-sensitivity-30-20261125/` | Thirty-scenario selection and numerical-check records |

Generated datasets, weights, and full evaluation outputs belong in the ignored `data/`, `experiments/`, `evaluation_runs/`, `results/`, and `results_archive/` locations. They are not downloaded by cloning the source repository. Dataset release links and checkpoint availability are described in the README.

Old manuscript copies under `paper-backup/`, historical working outputs under `runs/`, unused preview media, and LaTeX build intermediates are excluded from the current source tree. Earlier committed versions remain accessible through Git history. Required scientific inventories, current paper data, and archived numerical controls are retained. Removing files from the current tree does not rewrite Git history or reduce existing historical pack files.

## Study boundaries

The main workflow uses the same requested output times for all three references. The native-resolution, GEBCO morphology, pooled-reference, strict-holdout, and uncertainty studies have separate settings because they answer different questions. Sharing a naming convention does not make their data interchangeable.

Historical numerical controls retain their original sampling and damping settings where needed for comparisons. Released-data readers support the stored formats and identifiers. These compatibility paths are not alternative recommendations for generating the current benchmark.

For the reported paper metrics, use the selected checkpoints and the recorded evaluation inputs. A fresh generation or training run has a new code identity and may produce different numerical artifacts. Preserve the manifests and normalization statistics that accompany each dataset; do not relabel old outputs as newly generated data.

## Development checks

Run focused tests for the subsystem being changed, then the practical regression suite with `python -m pytest -q -m "not slow"`. Slow numerical tests, actual cluster submission, full data generation, model training, and manuscript compilation are separate checks. Passing Python tests alone does not establish that every research experiment has been rerun.
