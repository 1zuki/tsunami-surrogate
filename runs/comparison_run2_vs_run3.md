# Eval Runs Comparison — run_2 vs run_3

_Analysis only. run_1 is the failed run (Windows partition not mounted → only `dataset_summary.json` produced; the solver-fidelity / emulator evals had no raw data). run_2 and run_3 are the two full comparable runs._

## TL;DR

- **run_2 and run_3 are effectively identical.** Every accuracy, OOD, solver-fidelity, per-frame, and emulator-superiority number matches to machine precision.
- The **only** differences are in timing (model speed, solver speed), and those are pure measurement noise: all within ±1.5%.
- This is the expected outcome — model inference + metric computation are deterministic. Re-running confirms the pipeline is stable and reproducible.
- **Two issues surfaced, identical in both runs** (so not a run-to-run difference, but worth knowing): the speed-table aggregation is empty (a path bug), and ConvLSTM is absent (training never completed).

---

## 1. Accuracy — IDENTICAL

Bit-for-bit identical between runs (deterministic eval). Physical-unit MAE/RMSE, rel-L2, max-error all match exactly.

| Model | rel-L2 | max-err | run2 vs run3 |
|---|---|---|---|
| FNO (hydrostatic) | 0.2350 | 22.92 | identical |
| U-Net | 0.4754 | 27.38 | identical |
| CNN | 0.6074 | 29.79 | identical |
| FNO (MUSCL-HR) | 0.2632 | 22.13 | identical |
| FNO (Boussinesq) | 0.2096 | 15.94 | identical |

Ranking holds: FNO best on every target; Boussinesq is the easiest target (0.210), CNN the weakest model (0.607).

---

## 2. Model inference speed — NOISE ONLY (±1.5%)

| Model | run2 | run3 | Δ |
|---|---|---|---|
| FNO | 0.906 ms | 0.906 ms | −0.1% |
| CNN | 0.175 ms | 0.173 ms | −1.5% |
| U-Net | 0.240 ms | 0.241 ms | +0.2% |
| FNO MUSCL-HR | 0.906 ms | 0.906 ms | −0.0% |
| FNO Boussinesq | 0.905 ms | 0.906 ms | +0.1% |

The three FNO variants are ~identical (0.905–0.906 ms) — same architecture, as expected. CNN fastest, FNO slowest among models.

---

## 3. Solver rollout speed (CPU) — NOISE ONLY (±1.3%)

| Solver | run2 | run3 | Δ |
|---|---|---|---|
| hydrostatic | 33.21 s | 33.30 s | +0.3% |
| MUSCL-HR | 72.09 s | 72.31 s | +0.3% |
| Boussinesq | 16.60 s | 16.38 s | −1.3% |

MUSCL-HR is the slowest solver (~2× hydrostatic, second-order Heun + slope limiting). Boussinesq is fastest despite the CG solve.

---

## 4. Speedup (model vs its own solver) — derived, stable

Computed manually from the raw JSONs (see issue #1 below — the auto speed-table is empty). Using run_2 numbers:

| Model | Solver | Speedup |
|---|---|---|
| FNO | hydrostatic | ~36,650× |
| U-Net | hydrostatic | ~138,000× |
| CNN | hydrostatic | ~189,000× |
| FNO MUSCL-HR | MUSCL-HR | ~79,600× |
| FNO Boussinesq | Boussinesq | ~18,300× |

All ~10⁴–10⁵×. (Reminder: these are vs unoptimized CPU NumPy solvers — implementation-level, not vs optimized/GPU solvers.)

---

## 5. OOD held-out suites — IDENTICAL

All three FNO targets, all three held-out suites, rel-L2 matches between runs.

| Target | source-holdout (multi-gauss) | bathy-holdout (trench) | strength-extreme-high |
|---|---|---|---|
| FNO (hydro) | 0.2322 (n=414) | 0.2666 (n=481) | 0.2777 (n=43) |
| FNO (MUSCL-HR) | 0.2590 | 0.3141 | 0.2998 |
| FNO (Boussinesq) | 0.1963 | 0.2045 | 0.2831 |

Note: the MUSCL-HR `source_strength_extreme_high` suite now returns n=43 samples — confirming the `source_strength_min` 1.6→0.82 bug fix worked (it would have been 0 before).

---

## 6. Per-frame error growth — IDENTICAL, and revealing

rel-L2 across the 50-frame rollout (frame 0 → mid → last):

| Target | frame 0 | mid | last | growth |
|---|---|---|---|---|
| FNO (hydro) | 0.214 | 0.325 | 0.309 | +0.095 |
| FNO (MUSCL-HR) | 0.217 | 0.385 | 0.469 | **+0.251** |
| FNO (Boussinesq) | 0.207 | 0.206 | 0.208 | **+0.001** |

This is the most interesting scientific signal:
- **Hydrostatic & MUSCL-HR drift over the horizon** (error accumulates), MUSCL-HR worst — it nearly doubles by the last frame.
- **Boussinesq is essentially flat** (0.207→0.208) — no temporal drift at all. The model tracks the weakly-dispersive field uniformly across the whole rollout. This is why Boussinesq has both the best overall rel-L2 AND the lowest max-error.

---

## 7. Solver-fidelity (raw solver-vs-solver gaps) — IDENTICAL

All compared on 2500 shared samples. Aggregate rel-L2 (mean):

| Pair | rmse.mean | mae.mean | rel-L2.mean |
|---|---|---|---|
| hydro vs MUSCL-HR | 0.00173 | 0.00084 | 0.164 |
| MUSCL-HR vs hydro | 0.00173 | 0.00084 | 0.164 |
| hydro vs Boussinesq | 0.01632 | 0.00651 | (much larger) |

Key insight for the paper: the **raw gap between the two shallow-water solvers (hydro vs MUSCL-HR) is rel-L2 ≈ 0.164** — smaller than the surrogate's own error (~0.235). The hydro-vs-Boussinesq gap is ~10× larger (different physics), as expected.

---

## 8. Emulator-superiority ratios — IDENTICAL

| Direction | solver denom | emulator numer | ratio |
|---|---|---|---|
| hydro → MUSCL-HR | 0.00173 | 0.00343 | **1.983** |
| MUSCL-HR → hydro | 0.00173 | 0.00305 | **1.760** |

Both ratios > 1 → the surrogate's error is ~1.8–2.0× the raw solver-vs-solver gap. Interpretation: the emulator is NOT yet "superior" (ratio < 1 would mean the surrogate is closer to the cross-solver reference than the solvers are to each other). It's within ~2× of that bar.
