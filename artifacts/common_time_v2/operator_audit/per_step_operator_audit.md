# Per-step production-operator audit

**Scope.** This is a source audit of the three production solver implementations as they exist for the common-time-v2 preparation slice. It is not a convergence result and changes no solver equation or coefficient. Source references are repository-relative.

## Effective-rate rule

A fixed per-natural-step damping multiplier `m` applied `N` times gives

```text
q(T) = m^N q(0),     N approximately T / dt,
lambda_eff(dt) = -log(m) / dt.
```

Therefore lowering CFL/`dt` increases the number of applications over a fixed elapsed benchmark time and changes the effective damping problem. For a Boussinesq filter Fourier mode with per-step amplification `G(k)`, the corresponding elapsed-time rate is `log|G(k)| / dt`; it also changes when `dt` changes if `G(k)` is fixed per step.

## Hydrostatic SWE

| Operation | Active source | Frequency | `dt` consistency / interpretation |
|---|---|---:|---|
| Hydrostatic reconstruction with nonnegative face depth | `src/solver/hydrostatic_swe.py` `_hydro_face_x`, `_hydro_face_y` | Every face, every natural step | Nonlinear reconstruction/projection; no standalone continuous-time coefficient. |
| Rusanov numerical flux | `src/solver/hydrostatic_swe.py` `_rusanov_flux_x`, `_rusanov_flux_y` and active face helpers | Every face, every natural step | Flux divergence is multiplied by `dt/dx` or `dt/dy`; this is numerical dissipation, not physical viscosity. |
| Positivity projection and dry momentum reset | `update`, lines around the final `np.maximum(h_new, 0)` and dry mask | Once after each update, with another dry cleanup in `apply_boundary_conditions` | Event/projection operation; application path and threshold crossings may change under timestep refinement. |
| Boundary ghost-face treatment | `_boundary_state_x`, `_boundary_state_y` | Every boundary face and step | Part of the finite-volume update. “Open” is a numerical boundary rule, not demonstrated radiation physics. |
| Sponge of momentum and elevation relative to rest depth | `apply_sponge_layer` | Once per completed natural step | Fixed mask, no `dt` factor: **not timestep-consistent** by the effective-rate rule above. |
| Velocity cap helper | `_stabilized_conserved` | **Not called by the active first-order `update` path** | The configured `max_velocity` must not be claimed as an active Hydrostatic clipping safeguard. |

No separate physical viscosity or bottom-friction term was found.

## MUSCL-HR SWE

| Operation | Active source | Frequency | `dt` consistency / interpretation |
|---|---|---:|---|
| Minmod reconstruction of depth, surface and velocity | `src/solver/muscl_hr_swe.py` reconstruction helpers | Twice per natural step (two Euler stages) | Nonlinear limiter; activation can change with timestep/state path. |
| Hydrostatic face reconstruction and inherited Rusanov flux | `_euler_step_from_state` plus Hydrostatic helpers | Every face in both Euler stages | Conservative/source increments are `dt`-scaled; dissipation is numerical, not physical viscosity. |
| Cell and face velocity clipping | reconstruction path | Twice per natural step | Projection/clipping with no `dt` coefficient; frequency/path dependent. |
| Positivity and dry-cell projection | `_euler_step_from_state`, inherited `set_state` | Each Euler stage and final state | Event/projection operation, not a continuous `dt`-scaled term. |
| `nan_to_num(..., nan=0, posinf=0, neginf=0)` | each Euler output and final RK average | Three calls per natural step | Silent stabilization. A finite final state does not prove no intermediate nonfinite occurred; no counter is exposed. |
| SSPRK/Heun-style two-stage update | `update` | One completed natural step | The Euler increments and bathymetric source are `dt`-scaled. |
| Inherited sponge | Hydrostatic `step` / `apply_sponge_layer` | Once per completed natural step | Fixed mask, no `dt` factor: **not timestep-consistent**. |

No separate physical viscosity or bottom-friction term was found.

## Boussinesq

| Operation | Active source | Frequency | `dt` consistency / interpretation |
|---|---|---:|---|
| Velocity-Verlet update | `src/solver/boussinesq.py` `step` | One natural step | Kinematic/acceleration increments use `dt` and `dt^2`. |
| Matrix-free preconditioned CG acceleration solve | `solve_acceleration` | Twice per natural step | Iterative approximation controlled by tolerance/cap, not a damping term. A failed approximate acceleration is still consumed; health policy must reject failures. |
| Fixed effective-depth floor | initial depth construction | Once at setup | Model regularization, not dynamic wet/dry handling. |
| Laplacian filter on `eta` and `eta_t` | `apply_filter` | Once per completed natural step | Coefficient has no `dt`; each mode has fixed `G(k)`: **not timestep-consistent**. Values above 0.25 are silently clamped at application. |
| Sponge on `eta` and `eta_t` | `apply_sponge_layer` | Once per completed natural step | Fixed mask, no `dt` factor: **not timestep-consistent**. |
| Scalar boundary padding | `src/solver/boundary_conditions.py` | Every spatial operator call | Periodic wraps; “open” and reflective scalar fields both use edge padding. Their physical distinction is not established. |
| Final finite-state check | end of `step` when enabled | Once after filter and sponge | Detects only final post-operator state, not all CG intermediates. |

No separate physical viscosity or bottom-friction term was found.

## Consequences and deferred correction

1. **Clean temporal refinement:** disable sponge and Boussinesq filtering, or first reformulate them with an elapsed-time rate (for example `m(dt)=exp(-lambda dt)` and a filter coefficient derived from a frozen rate). Verify the reformulation separately before adopting it.
2. **Frozen production comparison:** with current sponge/filter enabled, call the result **total temporal-discretization and production-operator sensitivity**, not pure timestep convergence.
3. Add output-neutral counters before relying on claims about clipping, limiter activation, `nan_to_num`, or per-operator state changes; those events are not fully observable from final states today.
4. Production-like Boussinesq evidence remains blocked by unresolved `depth_scale`, open-boundary, sponge-width, filter and long-horizon benchmark-distribution choices. A periodic filter/sponge-disabled `0.1750` pilot establishes mechanics only.
5. This audit proposes no solver modification in the preparation slice. Any future timestep-consistent correction needs dedicated operator, lake-at-rest, dispersion, boundary, convergence and legacy-reproduction tests.
