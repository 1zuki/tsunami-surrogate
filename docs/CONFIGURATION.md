# Configuration guide

Run project commands from the repository root. Paths such as `data/train`, `data/processed`, `experiments/fno`, and `results/` refer to locations beneath that root. They do not depend on the author's home directory. Command-line path overrides can point to data or tools stored elsewhere.

The shared YAML loader merges the files listed under `defaults:` in order, then applies the current file's settings. Each defaults path is relative to the file containing it. Nested mappings merge; lists are replaced. Inheritance cycles and non-mapping configuration files are rejected. Generation, preprocessing, and training use this loader.

## Main workflow

| Stage | Configuration |
| --- | --- |
| Training data generation | `configs/data/dataset.yaml` |
| Validation data generation | `configs/data/dataset_eval.yaml` |
| Final test data generation | `configs/data/dataset_test.yaml` |
| Preprocessing | `configs/data/preprocess_{train,eval,test}.yaml` |
| Model and training settings | `configs/model/`, with shared settings in `configs/train/` |
| Complete paper evaluation | `configs/eval/benchmark_suite.yaml` |
| Multi-seed cluster training | `configs/cluster/final_rebuild_training_suite.yaml` |

The benchmark publishes 50 shared times from 8.4 to 420. Run training preprocessing first. Validation and test reuse its normalization statistics. Keep the final test split closed until model and analysis choices are fixed.

## Separate studies

| Study | Settings |
| --- | --- |
| Native resolution transfer | `configs/data/multires/`, `configs/model/fno_res*`, `configs/eval/resolution_*` |
| GEBCO-derived morphology transfer | `configs/data/real_bathymetry/` |
| Pooled-reference ablation | `configs/data/pooled_reference_ablation/`, `configs/model/pooled_reference_ablation/` |
| Thirty-scenario reference sensitivity | `configs/eval/reference_sensitivity_30.yaml` |
| Numerical verification | Named study files under `configs/eval/` |
| Isolated Boussinesq diagnostic | `configs/diagnostics/` |

The diagnostic Boussinesq settings intentionally differ from the main benchmark and must not replace its generation settings. Numerical controls and tests can still read saved-step data to compare earlier methods.

`scripts/run_generation_pilot.py` uses the main dataset and master-field settings directly. The former `configs/data/rebuild/` copies are no longer needed. Its historical numerical checks remain covered by regression tests.

## Names and existing archives

Current entrypoint and configuration filenames omit redundant version suffixes. Examples are `benchmark_suite.yaml`, `common_time_h2.yaml`, `run_common_time_h2.py`, and `eval_seed_metrics.py`. The current GEBCO settings live in `configs/data/real_bathymetry/`.

Existing dataset directories, stored schema identifiers, recorded run names, and evidence hashes can still contain `v2`. These identify released data and remain unchanged so existing archives and checkpoints remain usable. A filename rename does not imply that old data were regenerated or that their numerical settings changed. Superseded configuration variants needed by regression tests live under `tests/fixtures/configs/`.

## Slurm resources

The supplied training suite uses an allocated MIG GPU profile. Its `resources` section supplies the scheduler account, partition, QoS, GPU type/count, CPU count, memory, time limit, account job limit, and maximum simultaneous array tasks. The submitter passes these values to `sbatch`; change them to match your cluster. The configured account limit is a local planning setting, not a live query of available slots. Other users' jobs and cluster policy can leave work pending.

Create `logs/slurm` before submission. Submit from the repository root, or export `PROJECT_ROOT`. Jobs activate `.venv` unless `ENV_PREFIX` selects another environment. Slurm controls GPU visibility. CPU-only jobs require the site's permission and suitable queue settings. No MPS helper or personal filesystem path is required.
