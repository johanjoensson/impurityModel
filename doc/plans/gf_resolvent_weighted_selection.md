# Resolvent-weighted determinant selection for the sparse GF recurrence — verdict

**Status: step-1 gate FAILED (2026-10-14). Nothing was implemented.** As a per-row selector (policy B) the
criterion is dead. As a freeze rule (policy A) it is *looser* than the shipped leak ≤ cut freeze wherever the cut is
tiny, it was safe on this fixture, and only a trickle-regime fixture can say whether it shrinks production supports. The scripts were scratch numpy simulations (dense `H`, scalar Lanczos,
full reort); they are reproduced in the appendix so the numbers can be regenerated.

## Question

`GF_SUPPORT_FREEZE` (PR #4) decides all at once, in *matvec* units. It freezes the support once the weight
reaching new rows (the **leak**) over 16 steps is no more than the weight the `slaterWeightMin` prunes cut. Can we
instead decide each step which new determinants to admit, keeping only those that noticeably affect the converged
`G`, and so build a GF-targeted selected-CI basis?

The per-step mechanism already exists:
- `_PrunedBasisProxy` / `GF_LANCZOS_ADMIT_TOL` admits by amplitude and bans rejected rows for good.
- `gf_method="cipsi"` and `GF_BICGSTAB_ADMISSION=outer` do the per-frequency version.

Measured verdicts (`gf_basis_size_comparison.md`, `gf_cipsi_frequency_truncation.md`):
- NiO-like: importance is flat over the closure.
- Metal, Matsubara axis: amplitude admission needs 1.3–3.2x fewer determinants.

So the only new ingredient worth testing was the **criterion**: measure what a row costs in units of `G`.

## The criterion

`G(z) = ||s||² [(z − T)⁻¹]_{00}`. Let `c_k(z) = [(z − T)⁻¹]_{k0}`. Dropping the out-of-support part `d_k` of
`H q_k` perturbs `H` by `−(d_k q_k† + h.c.)`. Since `d_k ⟂` Krylov, the first-order term vanishes, so:

    |δG_k(z)| ≲ ||s||² |c_k(z)|² · leak_k · ||R_⊥(z)||,   ||R_⊥|| ≤ 1/|Im z|

This gives two policies:
- **A**, a resolvent-weighted freeze: freeze when `Σ_window max_z est_k ≤ gf_tol`.
- **B**, a depth-dependent ban threshold: `eta_k² ∝ gf_tol / max_z |c_k|²`.

## Measurements

**Fixture:** `build_semicircle_siam(7, v=0.5, gs_keep=150)`, removal unit, one seed column; a 3,920-determinant
sector.

**Meshes:**
- Matsubara, `T = 0.005`, 12 points;
- real axis, δ = 0.13 and 0.02, over `[-1.5, 1.5]`.

The NiO-like `n_b=3` sectors (56 and 8 determinants) are already filled by their seeds, so there is nothing to
select there.

### 1. Estimate vs. truth (ban step k's new rows; exact `G` on the complement)

| axis | k = 1..5: estimate/true ratio, `c_k` from final `T_m` | `c_k` from online `T_k` |
|---|---|---|
| Matsubara | 6, 63, 73, 293, 5e4 | 0.64, 18, 21, 60, 1.4e4 |
| real δ=0.13 | 1.0, 18, 3.8, **0.83**, 78 | 2.9, 25, 6.3, 0.96, 146 |
| real δ=0.02 | 3.4, 31, 8.2, 2.2, 199 | 11, 27, 14, 2.1, 300 |

*(ratio = estimate / true δG; below 1 means the estimate under-predicts)*

- **Not a bound.** It under-predicts in two cells (0.64, 0.83). Rows first reached by round-off at steps 6–8
  (leak ~1e-40) cost up to 3e-9 at δ=0.02, an under-prediction of about 30 orders of magnitude. A row's
  first-encounter weight does not bound what banning it costs later.
- **Loose.** Up to 5e4x over (1.4e4x with the online `T_k`), against the gate's ~10x.
- **No ranking information beyond the leak.** Spearman(true, estimate) = 1.00, and Spearman(true, leak alone) =
  1.00, because the leak falls by orders of magnitude per step.
- **Growth ends by step ~8–14,** while `|c_k|²` is still O(1).

### 2. Where each rule stops (online `T_k`, `||R_⊥|| ≤ 1/|Im z|`, swm 1e-6 and 1e-8)

- The support saturates the sector (3,920) at step 14.
- The shipped freeze (16-step window) fires at step 29–30. In this fixture it is a no-op; it only hands the
  saturated tail to the CSR.
- The convergence monitor stops at:
  - step 14 on Matsubara at 1e-6, step 20 at 1e-9;
  - steps 70 / 114 on real δ=0.13;
  - step 292 or never on real δ=0.02.
- **A** with the 16-step window fires only after saturation, so it changes nothing.
- **A** with window 1 fires at steps 6–8, freezing at 3,471–3,785 of 3,920 (3–11% fewer).
  - Freeze truth was computed only for Matsubara and real δ=0.02 at swm 1e-4 and 1e-6 (not for δ=0.13 or swm
    1e-8). On those cells the true error is within tolerance, except at real δ=0.02 and tol 1e-9: true 3.9e-9, a 4x
    overshoot.
  - Window 1 is fragile: the leak fluctuates (step 8 = 2.5e-9, step 9 = 1.9e-8), so a single dip triggers it.
- `max_z |c_k|²` decays geometrically on Matsubara (9 → 1e-6 by step 14 → 1e-20 by step 40). On the real axis at
  δ=0.02 it stays O(1–100) to step ~100.
- **A is looser than the shipped rule when the cut is tiny.** At real δ=0.02, swm 1e-6, step 10, A's windowed
  estimate is 9e-8 relative (under a 1e-6 tol), while the shipped rule compares a leak of 1.2e-9 with a cut of
  6.8e-12 and is nowhere near firing. With the same 16-step window, A fires at steps 21–28, before the shipped
  29–30. Wherever freeze truth exists, it falls much faster than A's estimate.

## Verdict

- **Policy B is dead.** All rows reached in one step share `c_k`, so B ranks rows within a step by amplitude, which
  is `GF_LANCZOS_ADMIT_TOL`. Growth ends while `c_k` is O(1), so B ≈ constant-eta pruning. B does not escape the
  measured "importance is flat over the closure" verdict either.
- **Policy A is a looser freeze, and was safe here, but this fixture cannot show a large gain.**
  - On Matsubara, with window 1, it froze at step 6 (3,471 determinants) while the monitor stopped at step 14
    (3,920), with a true error of 2e-15. It acts before convergence, not as a relabelled monitor.
  - On the narrow-δ real axis, which DMFT consumes (the real-axis Σ feeds the next hybridization), `|c_k|²` does
    not decay. But A still compares against a G tolerance rather than the cut, so where the cut is tiny it is the
    looser rule. That is every production cutoff (1.5e-8), and it is exactly the CrI3 case, where the shipped
    trickle never meets its cut.
  - Measured gain: ≤ 11% of a closure that saturates by step 14. Short windows are fragile, and there was one 4x
    tolerance overshoot.
- **Open, and it leans favourable: the trickle regime.** Production supports keep admitting determinants at the `slaterWeightMin` floor
  for thousands of blocks (CrI3: 1,000–5,000 per 16 blocks at ~5.14M). This fixture saturates its sector and cannot
  reproduce that.
  - A would need a much larger sector to be tested there: metal `n_b=9`, removal sector ≈ 53k, with a sparse
    per-z truth.
  - Alternatively, log `max_z |c_k|²` beside the existing `GF support:` leak/cut lines on the next cluster round,
    then read off where A would have frozen. The round-9 arrhenius A/B, which reports how often the shipped freeze
    fires at 1.5e-8, decides whether this is needed.

"Optimal basis" framing: what this would build is a greedy, Krylov-adapted selection, not an optimal basis. On
this fixture, the closure the recurrence reaches is nearly the basis; the trickle regime is where A could differ.

## Prior art

- **A. Go and A. J. Millis**, "Adaptively truncated Hilbert space based impurity solver for dynamical mean-field
  theory", arXiv:1703.04928.
  - Builds the GF space by particle-hole excitation order around amplitude-ranked ground-state seeds, motivated by
    the Hamiltonian's own action.
  - Converges the excitation order until Σ is causal.
  - Uses no resolvent weighting.
- **C. Mejuto-Zaera, N. Tubman, K. B. Whaley**, ASCI as a DMFT impurity solver, Phys. Rev. B 100, 125165 (2019):
  selects determinants for the ground state.
- **Lu et al.** (Haverkort group), PRB (2014), and **Bi, Huang, Tong**, arXiv:1802.02340: Lanczos in a
  natural-orbital basis. This matches our own finding that the bath basis is a bigger lever than the solver
  (`gf_basis_size_comparison.md`, F-geom).

## Follow-ups (independent of this verdict)

1. **Move `_PrunedBasisProxy`'s admission into `finish_step`.** Its `begin_step` opts out of the deferral, which is
   what forces `GF_APPLY_ROW_CHUNKS=1`.
2. **Ban the rows the after-sum prune drops.** Coupled to #1, not an existing bug.
   - In `_lanczos_step.pxi`, `prune_rows(slaterWeightMin)` runs before `finish_step`. Today the pruned proxy decides
     per chunk, before that cut, so it sees those rows. Once #1 moves its decision after the cut, the dropped rows
     escape the ban and could re-enter, so the decision must take the new-row set before the cut and ban the
     sub-`slaterWeightMin` ones.
   - Together #1 and #2 let `GF_LANCZOS_ADMIT_TOL` run at production cutoffs instead of raising.
   - Plan: `doc/gf_support_pipeline.md`.
3. **Trickle-regime test of policy A** (above), gated on the round-9 A/B.

## Appendix: reproduce

The scripts are kept in `debug/gf_resolvent_weighted/` (gitignored), with their outputs. They used `aim_fixtures.build_semicircle_siam`, `AIM.sector`/`AIM.dense`, and `_seeds_for`:
- `oracle_ck.py`: per-step ban truth and the freeze table;
- `killcheck.py`: monitor / A / shipped stop steps.

The simulation:
1. A scalar Lanczos on `Hs = E0 − H` with full reorthogonalization.
2. After-sum and next-block prunes at `swm`, not banned (as production does).
3. Truth: an `eigh` of `Hs` restricted to the kept rows.

A full run takes ~15 min serially.
