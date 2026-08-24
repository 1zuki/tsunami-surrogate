# GeoClaw production-like discrepancy ablation

- Frozen GeoClaw bundle: `3eb1afd1653a3d5dbbd12a381c0ab1eccdc40920d98f6b503249698d5cd62460`
- Cases: 3 frozen production canaries
- Status: descriptive controlled ablation; GeoClaw is not physical truth

## Mean GeoClaw trajectory relative L2

| Variant | Hydrostatic | MUSCL-HR |
|---|---:|---:|
| Production 96, radiation + sponge | 0.534348 | 0.238176 |
| Production 96, radiation, no sponge | 0.533271 | 0.235024 |
| Production 96, open, no sponge | 0.533754 | 0.235345 |
| Extended 192, open, no sponge | 0.533016 | 0.233893 |
| 2x refined production domain, open, no sponge | 0.388686 | 0.134344 |

## Controlled changes

Negative delta means the changed setup moved closer to GeoClaw.

| Solver | Tested dimension | Mean delta | Improved cases |
|---|---|---:|---:|
| swe_hydrostatic | sponge | -0.001077 | 2/3 |
| swe_hydrostatic | outer_boundary | +0.000483 | 1/3 |
| swe_hydrostatic | domain_extent_and_exterior | -0.000738 | 2/3 |
| swe_hydrostatic | spatial_resolution | -0.145068 | 3/3 |
| swe_muscl_hr | sponge | -0.003153 | 2/3 |
| swe_muscl_hr | outer_boundary | +0.000322 | 1/3 |
| swe_muscl_hr | domain_extent_and_exterior | -0.001452 | 2/3 |
| swe_muscl_hr | spatial_resolution | -0.101001 | 3/3 |

## Solver-formulation signal

| Variant | Hydrostatic minus MUSCL-HR gap | MUSCL-HR closer |
|---|---:|---:|
| production_96_radiation_sponge | +0.296172 | 3/3 |
| production_96_radiation_no_sponge | +0.298247 | 3/3 |
| production_96_open_no_sponge | +0.298409 | 3/3 |
| extended_192_open_no_sponge | +0.299123 | 3/3 |
| refined_192_open_no_sponge | +0.254342 | 3/3 |

## Evidence-backed conclusion

- Removing the sponge, changing radiation to open boundaries, and moving the open boundary outward changed mean trajectory relative L2 by only 0.0003--0.0032 in these canaries.
- Halving the in-house cell size changed the mean gap by -0.1451 for Hydrostatic and -0.1010 for MUSCL-HR, moving closer to GeoClaw in all six solver-case comparisons. This is the dominant tested sensitivity.
- A material residual remains after refinement (mean 0.3887 and 0.1343), while MUSCL-HR is closer than Hydrostatic in every tested setup. The evidence therefore does not isolate reconstruction or any other single method component as the cause.

## Interpretation boundary

- The 96-versus-192 GeoClaw comparison is not a coarse-versus-fine grid comparison: both use `dx = dy = 1/64`; 192 moves the boundary farther away.
- The nested 2x refinement halves `dx` on the same physical in-house domain and preserves coarse cell averages. It is a numerical sensitivity check, not new high-resolution physical bathymetry.
- The extended 192/open/no-sponge residual uses the same frozen domain, central state, cell size, requested times, and broad open boundary class as GeoClaw. Its remaining difference is attributable to the combined numerical-method/configuration package, not to one isolated reconstruction component.
- The Hydrostatic-versus-MUSCL-HR contrast changes reconstruction, topographic-source treatment, time integration, and solver-specific CFL together; it must not be described as reconstruction order alone.
