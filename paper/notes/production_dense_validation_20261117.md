# Production dense replay and boundary diagnostic

The current mounted test bundle was audited directly from its `sample.npz`,
`meta.json`, `publication.json`, and paired 128-cell input caches. All 2,500
scenario records were complete across Hydrostatic, MUSCL-HR, and Boussinesq;
the shared inputs, family labels, source strengths, requested times, finite
states, and quality records matched across references. The regenerated
primary selection contains 120 scenarios with four cases per current
bathymetry/source family cell.

The selected scenarios were replayed with the production 192-cell buffered
configuration: a 128-cell solver core, 32-cell buffer, requested times
8.4--420, Hydrostatic CFL 0.1125, MUSCL-HR CFL 0.225, Boussinesq CFL 0.35,
radiation boundaries for the shallow-water references, an open boundary for
Boussinesq, and the elapsed-time-consistent cosine sponge. All 360 solver--
scenario replays remained finite, and every natural-step count and archived
natural-step history agreed to floating-point precision.

Relative $L_2$ differences between the regenerated requested outputs and the
stored production publications were:

| Solver | median | 95th percentile | maximum |
|---|---:|---:|---:|
| Hydrostatic | $1.289\times10^{-5}$ | $5.423\times10^{-5}$ | $1.572\times10^{-4}$ |
| MUSCL-HR | $1.168\times10^{-5}$ | $4.808\times10^{-5}$ | $1.454\times10^{-4}$ |
| Boussinesq | $3.314\times10^{-8}$ | $3.382\times10^{-8}$ | $3.425\times10^{-8}$ |

The replay comparison is a production consistency check against the archived
requested outputs. It is not an exact-solution accuracy test and does not
separate code-state drift from solver discretization error.

The production boundary diagnostic used the same natural-step replay. The
95th-percentile maximum interface mass flux was $0.1125$ for Hydrostatic and
$0.1337$ for MUSCL-HR. The corresponding 95th-percentile absolute cumulative
residual, normalized by initial domain mass, was $2.813\times10^{-3}$ and
$2.903\times10^{-3}$. The residual includes sponge damping, projection, and
the remaining discrete mass balance; it is a boundary-activity diagnostic,
not a dimensional conservation budget. All 240 shallow-water replays were
finite, and the Boussinesq replays had zero conjugate-gradient failures.

Artifacts: `/mnt/Windows/Dataset/current_production_dense_validation_20261117/`.
