# Verification follow-ups (2026-11-17)

This note records the legacy normalized diagnostics. The production-configured
replay and boundary results supersede the flux values below for manuscript use;
they are recorded in `paper/notes/production_dense_validation_20261117.md`.

## Full 120-case open-boundary flux diagnostic

The current open/radiation boundary and sponge configuration was rerun for all 120 dense-replay scenarios at final time 0.08 and target CFL 0.45. Both solvers completed 120/120 finite rollouts.

| Solver | median max boundary flux | 95th percentile | maximum | median absolute cumulative residual | 95th percentile residual |
|---|---:|---:|---:|---:|---:|
| Hydrostatic | 0.04243 | 0.12372 | 0.17824 | 0.00270 | 0.01468 |
| MUSCL-HR | 0.04601 | 0.13381 | 0.19135 | 0.00272 | 0.01506 |

These are benchmark flux units. The cumulative residual includes sponge damping and positivity/projection effects, so it is a boundary-activity diagnostic rather than a dimensional conservation budget.

## Island--Gaussian family-cell diagnosis

The preregistered four-case island--Gaussian cell remains a genuine guardrail failure, with ratios 0.09599, 0.10456, 0.12939, and 0.12290 (median 0.11373). To test whether four cases were representative, twelve additional eligible cases were selected post hoc by evenly spanning source strength from 0.4500 to 0.8462, excluding the preregistered four. All twelve extra cases passed the dense validation decision. Their median ratio was 0.07028 and their 95th percentile was 0.11835. Combining the original and extra cases gives a 16-case median of 0.07639 and a 95th percentile of 0.12902.

This follow-up suggests selection sensitivity rather than a universal island--Gaussian failure, but it cannot change the preregistered four-case decision. The original guardrail remains reported as failed.

## Phase/amplitude decomposition

The 32--512 periodic linear-wave diagnostic was decomposed at the finest grid. Hydrostatic relative-$L_2$ error is dominated by amplitude dissipation: amplitude relative errors are approximately $-0.0522$, $-0.0458$, and $-0.0328$ for CFL values 0.1125, 0.225, and 0.45. MUSCL-HR amplitude errors are approximately $-2.69 \times 10^{-4}$ for all three CFL values and phase errors remain below $1.7 \times 10^{-4}$ radians, while its full-field relative-$L_2$ error remains about $0.007394$--$0.007396$. The result supports a MUSCL-HR finest-grid error floor and does not justify a blanket order claim.

Artifacts are stored under `/mnt/Windows/Dataset/current_boundary_flux_full_20261117`, `/mnt/Windows/Dataset/island_gaussian_extra_dense_20261117`, `/mnt/Windows/Dataset/island_gaussian_extra_validation_20261117`, and `/mnt/Windows/Dataset/periodic_linear_convergence_20261117`.
