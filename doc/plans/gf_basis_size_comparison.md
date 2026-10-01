# How large a basis does the self-energy need? Per-frequency vs Lanczos, and importance-ranked admission

**Status: measured on three fixtures (F-NiO, F-pos, F-metal); the geometry (F-geom) and real-workload tiers are
not done.** Branch `gf-basis-size`. Plan: `~/.claude/plans/swirling-cooking-spark.md`.

## The questions

1. Does a per-frequency Green's-function solve (`gf_method="bicgstab"`) need a smaller maximum basis than
   block Lanczos, which solves the whole mesh with one recurrence?
2. Can the per-frequency basis growth do better than admitting every determinant the solver produces?
3. Can the basis and G be built together, one frequency at a time, by something CIPSI-like? (The retired
   `gf_method="cipsi"`, `doc/plans/gf_cipsi_frequency_truncation.md`, was exactly this; this work does not revive it.)

"Needs" is defined operationally: the smallest **maximum retained determinant count over the work units** that
brings the self-energy error below a tolerance, found on a ladder of caps (freeze-growth) or admission thresholds.

## What was built (commits on `gf-basis-size`)

| | |
|---|---|
| C1 | `stats["points"]`: per-solve seed / rebuilt / solve basis; `GF_BICGSTAB_WARM_HISTORY` (0 = true cold start); a root-side `gf_unit_basis` trace note for both methods (the Lanczos kernel's own note is per colour and showed rank 0 only 2 of 4 units at `-n 3`) |
| C2 | `GF_BICGSTAB_RESIDUAL_CHECK`: one cutoff-0 matvec after each solve splits the true residual at the solve basis into its in-basis part and the **boundary residual** `(1-P)HX` the solver never sees, with a bound on `|G - G_exact|` built from them (2-5x the real error on SIAM-6; boundary residual exactly 0 uncapped) |
| C3 | `GF_BICGSTAB_ADMISSION=outer`: solve on a frozen basis, score the boundary, admit above a threshold, re-solve. Admission only *between* solves: dropping rows inside a BiCGSTAB recurrence breaks its recursive residual |
| C4 | `GF_LANCZOS_ADMIT_TOL`: the same pruning applied to the Lanczos recurrence, with a permanent **ban** on rejected rows. The ban is what keeps the recurrence the exact Lanczos of `P_m H P_m`: without it 13 of 80 tested (threshold, reort) combinations on SIAM-6 were wrong by up to 0.13 |
| C5 | Fixtures with an exact dense reference (`test/support/aim_fixtures.py`), the harness (`basis_size_harness.py`), smoke tests and the opt-in benchmark (`RUN_BASIS_SIZE_BENCH=1`) |

`GF_ADMIT_FIRST_SHELL_TOL` (C5) relaxes the "seeds and their first H-shell are never pruned" contract; see the F-pos
result below for why it exists.

## Setup

**F-NiO.** Two e_g-like orbitals (four spin-orbitals), Kanamori `U = 8`, `J = 1`, each impurity spin-orbital
hybridized with its own star of 9 filled bath levels of width 3. The impurity level is set so that the
charge-transfer energy `E(d9L) - E(d8) = 4` at `V = 0`; the hybridization `V_eff` is **calibrated to the `d9L`
weight** of the ground state (5 %, 15 %, 27 %; 27 % is the production NiO value), not quoted as a nominal `V/Delta`.
Ground state: the `S_z = 1` member of the high-spin triplet. Sectors: 190 determinants (ground state), 4,940
(removal, the cap-relevant unit), 20 (addition). Energies are in eV-like units; `delta` = 0.06 and 0.4 are the
`delta/W` = 0.02 (NiO-like) and 0.13 (FCC-Ni-like) ratios of the width-3 bath.

**F-pos.** The same, with two of the nine levels per spin-orbital made *spectators*: far below the bath (-15, -16)
and coupled at `1e-5`. They enlarge the connectivity closure without changing G: the positive control for
importance-ranked admission.

**Reference.** One dense `eigh` per sector, evaluated at any frequency. It follows the driver's own convention
`G = (G_add - G_rem^T)/Z` and matches the production driver (BiCGSTAB and Lanczos, both axes) to 1e-7; a flipped
removal sign fails that test. The non-interacting limit gives `Sigma = 0` exactly.

**Metric.** `dSigma = |Sigma - Sigma_ref|_F / max(|Sigma_ref|_F, 0.1 U)`, worst over the selected points, reported
per axis (Matsubara; real axis). The axes are kept apart because they disagree. Per-frequency methods run on 6
Matsubara + 12 real points (the highest-spectral-weight points plus an even spread); the Lanczos methods solve
the whole mesh and are scored on the same points. Per-frequency methods run **cold** (warm-start history 0).

**Sizes** are read from the traced work units, never from a basis object handed to a solver: an uncapped
Lanczos recurrence does not track its support, so "uncapped" is measured under a finite cap that never binds.

## Results

### Uncapped: every method needs the whole closure (measured)

Every method reaches the full 4,940-determinant closure. The seeds alone are 702-704 determinants (14 %).
**Every removal-unit solve of the cold, uncapped per-frequency method reaches the full closure**, at every
frequency (Matsubara and real), for every weight and broadening: per-point support equals the union even with no
warm start. The earlier "per-point = union" result was therefore not an artifact of the warm start; it is
structural. A cap below the seed support changes nothing (the recurrence freezes on the seeds).

### At equal cap the two methods keep the same number of determinants but not the same answer (measured)

F-NiO, `d9L` 15 %, `delta` 0.4, cold per-frequency BiCGSTAB vs freeze-growth Lanczos:

| cap | Lanczos size | Lanczos dΣ Matsubara | Lanczos dΣ real | BiCGSTAB size | BiCGSTAB dΣ Matsubara | BiCGSTAB dΣ real |
|---|---|---|---|---|---|---|
| 247 | 702 | 5.5e-02 | 9.8e-02 | 702 | 5.5e-02 | 9.8e-02 |
| 494 | 702 | 5.5e-02 | 9.8e-02 | 702 | 5.5e-02 | 9.8e-02 |
| 741 | 730 | 4.6e-02 | 8.3e-02 | 722 | 4.5e-02 | 8.6e-02 |
| 988 | 982 | 3.8e-02 | 7.1e-02 | 974 | 7.7e-03 | 1.9e-02 |
| 1,482 | 1,468 | 3.8e-02 | 7.1e-02 | 1,474 | 2.9e-03 | 8.6e-03 |
| 1,976 | 1,973 | 3.8e-02 | 6.7e-02 | 1,958 | 1.1e-03 | 4.6e-03 |
| 2,717 | 2,711 | 1.6e-03 | 3.6e-03 | 2,708 | 8.7e-04 | 2.9e-03 |
| 3,458 | 3,458 | 7.2e-04 | 6.0e-03 | 3,447 | 9.7e-04 | 1.1e-02 |
| 4,199 | 4,199 | 2.5e-04 | 3.2e-03 | 4,190 | 1.2e-03 | 6.3e-03 |

Lanczos's error is flat at 3.8e-2 from cap 988 to 1,976 and drops 25x at 2,717: the extra determinants admitted
in between do not help. BiCGSTAB improves smoothly to ~1e-3 and then stops improving (it is slightly *worse* than
Lanczos at 70-85 % of the closure, on both axes). The prior that the two methods keep the same basis for the same
answer (P1) is refuted for caps of 20-40 % of the closure on the Matsubara axis. **Hypothesis, not tested:** the overflow step
ranks candidates by residual amplitude, and a per-frequency residual is ranked at the frequency it serves while
the Lanczos Krylov residual is ranked for all frequencies at once.

### Smallest basis (determinants) reaching a Sigma tolerance (measured)

`—` means no cell of that method reached the tolerance. Cells with unconverged per-frequency solves are
excluded (they occur for `bicgstab-swm` everywhere and for `bicgstab-cap` at 15 %, delta 0.06).
The admission-threshold ladder is coarse (decades of eta, sizes jump ~800 -> ~3,500), so differences of less than
a rung between `bicgstab-outer` and the other methods are not resolved.

#### F-NiO

| d9L | δ | axis | tol | lanczos-cap | bicgstab-cap | lanczos-pruned | bicgstab-outer |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.06 | Matsubara | 1e-02 | 731 | 722 | 722 | 835 |
| 0.05 | 0.06 | Matsubara | 1e-03 | 2,702 | 974 | 1,122 | 3,984 |
| 0.05 | 0.06 | real | 1e-02 | 3,458 | 4,940 | — | 2,112 |
| 0.05 | 0.06 | real | 1e-03 | 4,199 | 4,940 | — | 3,984 |
| 0.05 | 0.4 | Matsubara | 1e-02 | 731 | 722 | 722 | 722 |
| 0.05 | 0.4 | Matsubara | 1e-03 | 2,702 | 974 | 1,122 | 4,204 |
| 0.05 | 0.4 | real | 1e-02 | 2,702 | 974 | 866 | 2,580 |
| 0.05 | 0.4 | real | 1e-03 | 4,199 | 4,940 | — | 4,204 |
| 0.15 | 0.06 | Matsubara | 1e-02 | 2,711 | 974 | 1,064 | 3,564 |
| 0.15 | 0.06 | Matsubara | 1e-03 | 3,458 | 3,447 | — | 4,468 |
| 0.15 | 0.06 | real | 1e-02 | 4,940 | 4,940 | — | 3,564 |
| 0.15 | 0.06 | real | 1e-03 | 4,940 | 4,940 | — | 4,468 |
| 0.15 | 0.4 | Matsubara | 1e-02 | 2,711 | 974 | 1,064 | 3,550 |
| 0.15 | 0.4 | Matsubara | 1e-03 | 3,458 | 2,708 | — | 4,566 |
| 0.15 | 0.4 | real | 1e-02 | 2,711 | 1,474 | 2,978 | 3,550 |
| 0.15 | 0.4 | real | 1e-03 | 4,940 | 4,940 | — | 4,566 |
| 0.27 | 0.06 | Matsubara | 1e-02 | 2,702 | 1,950 | 3,114 | 4,110 |
| 0.27 | 0.06 | Matsubara | 1e-03 | 4,940 | 4,940 | — | 4,556 |
| 0.27 | 0.06 | real | 1e-02 | 4,940 | 4,940 | — | 4,110 |
| 0.27 | 0.06 | real | 1e-03 | 4,940 | 4,940 | — | 4,556 |
| 0.27 | 0.4 | Matsubara | 1e-02 | 2,702 | 1,950 | 3,114 | 4,037 |
| 0.27 | 0.4 | Matsubara | 1e-03 | 4,940 | 4,940 | — | 4,646 |
| 0.27 | 0.4 | real | 1e-02 | 4,199 | 4,190 | — | 4,037 |
| 0.27 | 0.4 | real | 1e-03 | 4,940 | 4,940 | — | 4,646 |

#### F-pos

| d9L | δ | axis | tol | lanczos-cap | bicgstab-cap | lanczos-pruned | bicgstab-outer |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.06 | Matsubara | 1e-02 | 739 | 718 | 718 | 856 |
| 0.05 | 0.06 | Matsubara | 1e-03 | 2,715 | 988 | 886 | 1,558 |
| 0.05 | 0.06 | real | 1e-02 | 3,457 | — | — | 1,558 |
| 0.05 | 0.06 | real | 1e-03 | 4,940 | — | — | 2,396 |
| 0.05 | 0.4 | Matsubara | 1e-02 | 739 | 718 | 718 | 718 |
| 0.05 | 0.4 | Matsubara | 1e-03 | 2,715 | 988 | 886 | 1,754 |
| 0.05 | 0.4 | real | 1e-02 | 2,715 | 988 | 886 | 1,754 |
| 0.05 | 0.4 | real | 1e-03 | 4,199 | 4,940 | — | 1,754 |

### Reading the tables

* **Matsubara axis.** Per-frequency BiCGSTAB at a cap needs a smaller basis than Lanczos at a moderate tolerance:
  for `dSigma <= 1e-2` the ratio (Lanczos / BiCGSTAB basis) is 1.0x at 5 % weight, 2.8x at 15 % and 1.4x at
  27 %; at `1e-3` it is 2.8x at 5 %, 1.0-1.3x at 15 % and 1.0x at 27 %. The advantage shrinks as the tolerance
  tightens and as the hybridization grows, which is what a flat importance distribution predicts.
* **Real axis.** The advantage appears only at the larger broadening (delta = 0.4, 0.13 of the bath width), where
  it is 2.8x at 5 % and 1.8x at 15 % weight at `1e-2` and absent at 27 %. At the NiO-like delta = 0.06
  per-frequency BiCGSTAB needs the full closure at 15 % and 27 % weight, and at 5 % weight it is *worse* than
  Lanczos (4,940 vs 3,458). The real axis is where a hard cap fails near a pole.
* **`bicgstab-outer`** (importance-ranked admission) is **not** better than the plain cap on the Matsubara axis
  (it needs 3.7x more at 15 %, `1e-2`). Its one win is the real axis at small delta, where the plain
  BiCGSTAB cap fails: at `1e-2` 2,112 against 4,940 (BiCGSTAB cap) and 3,458 (Lanczos) at 5 %, and 3,564
  against 4,940 for both at 15 %.
* **`lanczos-pruned`** beats freeze-growth Lanczos at weak hybridization (1,122 determinants for
  `dSigma_mats <= 1e-3` where `lanczos-cap` needs 2,702) and fails at tight tolerance and strong hybridization: its permanent ban makes the
  first-shell cut a floor on the error, so it plateaus (~1e-3 Matsubara, ~1e-2 real at 15 %).
* **`bicgstab-swm`** (the existing amplitude cutoff inside the matvec, the control) never reduces the basis
  below ~95 % of the closure at any accuracy it reaches: amplitude pruning cannot shrink this support.

### F-pos: the positive control, and a contract that had to change (measured)

With the seeds' first H-shell kept whole (the original contract, which keeps the `Sigma` tail exact), importance
admission excluded **nothing** on F-pos: a spectator coupled at 1e-5 still generates its determinant in the first
shell, so the whole closure is in the start set. The control failed, and the failure was in the design, not the
harness. `GF_ADMIT_FIRST_SHELL_TOL` (default 0, which keeps the strict behaviour) lets the first shell be cut by
amplitude; the H^2 moment then errs by at most the summed squared amplitudes of what was dropped. With it,
at full size the pruned methods reach `dSigma_mats <= 1e-3` with 886 (Lanczos) and 1,558 (outer) determinants
against a 4,940 closure, where freeze-growth Lanczos needs 2,715. On the real axis at delta 0.06 the outer
admission reaches `1e-2` with 1,558 determinants against 3,457 for freeze-growth Lanczos, while plain
per-frequency BiCGSTAB has no converged cell that reaches it.


### F-metal: a strongly hybridized metal with a truncated ground state (measured)

A scaled-down `examples/semicircular_siam`: a half-filled single-orbital Anderson model on a 7-level semicircular
star per spin (Gauss-Chebyshev discretization, half-bandwidth D = 0.5 Ry, U = 0.5 Ry, impurity level at -U/2, the
particle-hole symmetric point), 16 spin-orbitals, removal closure 7,840 determinants. V = 1.0 Ry is the
example's value (twice D); V = 0.5 and 0.25 weaken it. The fixture is pinned by particle-hole symmetry:
`Re Sigma(i w_n) = U/2` exactly on the exact ground state, and the check fails if the impurity level is moved off
-U/2.

**The ground state is truncated** to its 150 determinants of largest weight (the lowest eigenstate of H projected
onto them, as a CIPSI ground state is). With the exact ground state the seeds alone cover almost the whole
sector, and no method can go below the seed support; with 150 determinants the seeds are 1.9 % of the closure
(14 % in F-NiO), which is the regime of the production calculations. The state is not an eigenstate, but `G` for
fixed seeds and energy is a well-defined resolvent, so the dense reference stays exact (and matches the
production driver on it). Its energy error against the exact ground state grows with V (4.5e-4, 3.7e-3, 1.7e-2 Ry
at V = 0.25, 0.5, 1.0) because K is held fixed.

The Matsubara axis does not depend on the broadening and is run once per V (6 points); the real axis once per
delta (8 points, 46-300 mesh points for the Lanczos methods). delta = 0.02 and 0.13 Ry are delta/W = 0.02 and 0.13
for the bandwidth W = 2D = 1 Ry; 0.02 is the broadening of the example.

Smallest basis (determinants) reaching a Sigma tolerance, closure 7,840:

| V (Ry) | axis | tol | lanczos-cap | bicgstab-cap | lanczos-pruned | bicgstab-outer |
|---|---|---|---|---|---|---|
| 0.5 | Matsubara | 1e-02 | 2,352 | 2,352 | 3,716 | 1,362 |
| 0.5 | Matsubara | 1e-03 | 4,312 | 4,312 | — | 1,362 |
| 0.5 | Matsubara | 1e-05 | 5,488 | 4,312 | — | 2,608 |
| 0.5 | real δ=0.02 | 1e-02 | 5,488 | 5,488 | — | 5,487 |
| 0.5 | real δ=0.02 | 1e-03 | 6,664 | 6,664 | — | 5,487 |
| 0.5 | real δ=0.02 | 1e-05 | 7,840 | 7,840 | — | 6,977 |
| 0.5 | real δ=0.13 | 1e-02 | 2,352 | 1,176 | 3,716 | 1,081 |
| 0.5 | real δ=0.13 | 1e-03 | 4,312 | 2,352 | 4,905 | 2,979 |
| 0.5 | real δ=0.13 | 1e-05 | 5,488 | 5,488 | — | 5,378 |
| 1.0 | Matsubara | 1e-02 | 2,352 | 2,352 | 5,786 | 1,854 |
| 1.0 | Matsubara | 1e-03 | 4,312 | 4,312 | — | 1,854 |
| 1.0 | Matsubara | 1e-05 | 5,488 | 5,488 | — | 3,302 |
| 1.0 | real δ=0.02 | 1e-02 | 4,312 | 4,312 | — | 5,246 |
| 1.0 | real δ=0.02 | 1e-03 | 5,488 | 4,312 | — | 5,246 |
| 1.0 | real δ=0.02 | 1e-05 | 6,664 | 6,664 | — | 6,731 |
| 1.0 | real δ=0.13 | 1e-02 | 2,352 | 2,352 | 5,786 | 3,217 |
| 1.0 | real δ=0.13 | 1e-03 | 4,312 | 4,312 | — | 3,217 |
| 1.0 | real δ=0.13 | 1e-05 | 5,488 | 5,488 | — | 5,708 |
| 0.25 | Matsubara | 1e-02 | 784 | 784 | 2,157 | 358 |
| 0.25 | Matsubara | 1e-03 | 2,351 | 1,176 | 2,157 | 910 |
| 0.25 | Matsubara | 1e-05 | 2,351 | 2,350 | — | 1,792 |
| 0.25 | real δ=0.02 | 1e-02 | 2,351 | 1,568 | 2,157 | 1,170 |
| 0.25 | real δ=0.02 | 1e-03 | 2,351 | 2,350 | — | 2,475 |
| 0.25 | real δ=0.02 | 1e-05 | 6,664 | 6,664 | — | 5,232 |
| 0.25 | real δ=0.13 | 1e-02 | 784 | 392 | 424 | 516 |
| 0.25 | real δ=0.13 | 1e-03 | 2,351 | 784 | 2,157 | 1,556 |
| 0.25 | real δ=0.13 | 1e-05 | 4,312 | 4,312 | — | 3,454 |

**Matsubara axis: the outer admission is the best method at every V and every tolerance down to 1e-5**, which it
was not on F-NiO. Against the best plain cap it needs 1.3x (V = 0.25), 3.2x (0.5) and 2.3x (1.0) fewer determinants
at `1e-3`, and 1.3x, 1.7x and 1.7x fewer at `1e-5` (1,792 / 2,608 / 3,302 against 2,350 / 4,312 / 5,488). Freeze-growth
Lanczos and capped BiCGSTAB need the same basis for the same Matsubara accuracy here in most cells; they differ
at `1e-3` for V = 0.25 (2,351 vs 1,176) and at `1e-5` for V = 0.5 (5,488 vs 4,312), by at most 2x, so the
equal-cap prior P1 roughly holds in the metal. Pruned Lanczos is again the weakest: it reaches `1e-3` only at
V = 0.25 (2,157) and, on the real axis at delta = 0.13, at V = 0.5 (4,905), and never at V = 1.0.

**Real axis: no method is consistently better.** At delta = 0.13 per-frequency BiCGSTAB needs about half the basis of
Lanczos at moderate tolerance (V = 0.25: 392 vs 784 at `1e-2`; V = 0.5: 1,176 vs 2,352), ties it at `1e-5`, and the
outer admission is about equal to the plain cap (sometimes worse: V = 0.5 at `1e-3`, 2,979 vs 2,352). At the
narrow delta = 0.02 and V = 0.5 or 1.0 every method needs 55-100 % of the closure for `1e-3` and tighter; at
V = 0.25 it is 30 % at `1e-3` and 67-85 % at `1e-5`. The outer admission saves at most 18-21 % there (V = 0.5 at `1e-3`: 5,487 vs 6,664; V = 0.25 at `1e-5`: 5,232 vs 6,664) and is
worse at V = 1.0.

**Cost.** Wall time of the cheapest cell that reaches the tolerance (per-frequency methods solve 6 Matsubara or
8 real points; the Lanczos methods solve the whole mesh):

| V | axis | tol | Lanczos cap | BiCGSTAB cap | outer admission |
|---|---|---|---|---|---|
| 0.5 | Matsubara | 1e-03 | 4,312 dets, 0 s | 4,312 dets, 41 s | 1,362 dets, 1 s |
| 0.5 | Matsubara | 1e-05 | 5,488 dets, 1 s | 4,312 dets, 41 s | 2,608 dets, 2 s |
| 0.5 | real δ=0.02 | 1e-03 | 6,664 dets, 35 s | 6,664 dets, 88 s | 5,487 dets, 58 s |
| 0.5 | real δ=0.02 | 1e-05 | 7,840 dets, 48 s | 7,840 dets, 105 s | 6,977 dets, 72 s |
| 0.5 | real δ=0.13 | 1e-03 | 5,488 dets, 3 s | 2,352 dets, 24 s | 2,979 dets, 9 s |
| 0.5 | real δ=0.13 | 1e-05 | 5,488 dets, 3 s | 5,488 dets, 49 s | 5,378 dets, 13 s |
| 1.0 | Matsubara | 1e-03 | 4,312 dets, 0 s | 4,312 dets, 43 s | 1,854 dets, 2 s |
| 1.0 | Matsubara | 1e-05 | 5,488 dets, 1 s | 5,488 dets, 53 s | 3,302 dets, 4 s |
| 1.0 | real δ=0.02 | 1e-03 | 5,488 dets, 12 s | 4,312 dets, 81 s | 6,731 dets, 130 s |
| 1.0 | real δ=0.02 | 1e-05 | 7,840 dets, 17 s | 6,664 dets, 103 s | 6,731 dets, 130 s |
| 1.0 | real δ=0.13 | 1e-03 | 4,312 dets, 2 s | 4,312 dets, 39 s | 3,217 dets, 19 s |
| 1.0 | real δ=0.13 | 1e-05 | 5,488 dets, 3 s | 5,488 dets, 48 s | 5,708 dets, 25 s |
| 0.25 | Matsubara | 1e-03 | 2,351 dets, 0 s | 1,176 dets, 4 s | 910 dets, 1 s |
| 0.25 | Matsubara | 1e-05 | 2,351 dets, 0 s | 2,350 dets, 5 s | 1,792 dets, 1 s |
| 0.25 | real δ=0.02 | 1e-03 | 2,351 dets, 5 s | 2,350 dets, 25 s | 2,475 dets, 23 s |
| 0.25 | real δ=0.02 | 1e-05 | 6,664 dets, 24 s | 6,664 dets, 65 s | 5,232 dets, 36 s |
| 0.25 | real δ=0.13 | 1e-03 | 2,351 dets, 1 s | 784 dets, 6 s | 1,556 dets, 2 s |
| 0.25 | real δ=0.13 | 1e-05 | 4,312 dets, 2 s | 4,312 dets, 28 s | 3,454 dets, 6 s |

The outer admission is not slow next to the other per-frequency method: on the Matsubara axis it takes 1-4 s against
4-53 s for capped BiCGSTAB, because it stops at a small basis, and on the real axis 2-130 s against 6-105 s
(Lanczos: 1-48 s). It is
slower than Lanczos (1-4 s against 0-1 s on the Matsubara axis; up to 8x on the real axis at V = 1.0, delta = 0.02),
and **per-frequency cost scales with the number of points**: these times are for 6-8 points, so a 300-point
real-axis mesh would cost roughly 40 times more, where Lanczos solves it in one recurrence.

**Why F-NiO and F-metal differ is not established.** The seeds are a much smaller part of the closure here
(1.9 % vs 14 %), the model is a single orbital with a delocalized bath, and the ground state is truncated; any of
these could account for the outer admission winning on the Matsubara axis. No experiment separated them.

## Answers

1. **Does per-frequency need a smaller maximum basis than Lanczos?** *Sometimes, and it depends on the axis.* Uncapped:
   no, both need the full closure. At a cap, on the Matsubara axis and at moderate accuracy: yes, 1.4-2.8x on this
   fixture. On the real axis: only at large broadening; at the NiO-like broadening no. At tight accuracy or strong
   hybridization the advantage is gone.
2. **Does importance ranking help the per-frequency expansion?** *It depends on the model.* On the NiO-like fixture
   rarely: it beats the plain cap in one corner (real axis, small delta) and is worse on the Matsubara axis. On the
   metal it is the best method on the Matsubara axis at every V and tolerance down to 1e-5 (1.3-3.2x fewer
   determinants than the best cap) and only ties on the real axis. It helps greatly where the closure contains
   determinants G does not need (F-pos), but only with the first shell relaxed. It helps the Lanczos recurrence at
   weak hybridization only.
3. **A per-frequency CIPSI-like build of G and the basis?** This work's `outer` admission is that idea in its
   principled form (residual-driven, between exact solves). It did not beat freeze-growth at equal budget on the
   Matsubara axis, consistent with the retired kernel's verdict.

## Limits of these conclusions

* Two model families (two e_g orbitals with a 9-level star; a single-orbital SIAM with a 7-level semicircular
  star), one size each, closures of 4,940 and 7,840. Real workloads have far larger ones, and the metal's bath
  is half the 14 levels of the example it is scaled from.
* The seeds are 14 % of the closure here. Where the seed support saturates the cap (FCC Ni) no admission rule
  can act (prior P3, not re-measured here).
* Sigma errors are measured against the exact reference; the recorded G error bound of C2 is not used in the
  tables. The 1e-3 and 1e-2 tolerances are not physical thresholds.
* The per-frequency methods run cold. Production warm-starts, which carries extrapolated support into each
  point's basis. Sizes under a warm start were not measured.
* The BiCGSTAB error stops improving near 1e-3 at 70-85 % of the closure while Lanczos keeps improving. Cause
  not investigated.
* The ladder resolution for eta is one decade; thresholds between rungs were not tried.
* Wall time: a per-frequency cell costs 5-48 s against 0.5-8 s for a Lanczos cell on the same mesh (about 10x).

## Not done

F-geom (the same model in star, chain and natural-orbital bases), the real-workload tier (FCC Ni is the metal there) (NiO archives,
geometries regenerated from the star archive with `rspt2spectra.edchain`), the `memory_estimate` term for
`outer`, and Matsubara/real-axis tolerances propagated from a physical requirement.

## Reproducing

```bash
RUN_BASIS_SIZE_BENCH=1 BENCH_N_B=9 BENCH_OUT=out.json \
  python -m pytest -s -m benchmark src/impurityModel/test/gf/test_gf_basis_size_comparison.py -k "full_size and F-NiO"
```

About 50 minutes for F-NiO (three weights x two broadenings) and 30 for F-pos, at 4 BLAS threads. The metal:

```bash
RUN_BASIS_SIZE_BENCH=1 BENCH_OUT=metal.json \\
  python -m pytest -s -m benchmark src/impurityModel/test/gf/test_gf_basis_size_comparison.py -k full_size_metal
```

(`BENCH_METAL_N_B`, `BENCH_METAL_K`, `BENCH_METAL_V`, `BENCH_DELTAS`; 1 hour 42 minutes for the three V and two
broadenings at n_b = 7, K = 150. The JSON is rewritten after each configuration.)
