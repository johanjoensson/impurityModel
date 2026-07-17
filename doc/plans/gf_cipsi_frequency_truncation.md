# Frequency-targeted CIPSI truncation of the per-frequency Green's function

**Status: experimental — `gf_method="cipsi"` shipped; NiO accuracy-vs-budget verdict below.**

## The question

Can the interacting Green's function be computed per frequency on a many-body basis that is
kept *small* by choosing which Slater determinants to keep, instead of letting the solver's
connectivity closure decide? Prior campaigns established the negative results this work
starts from:

- Blindly running the sparse per-point solvers does not shrink the basis: the live support
  of an uncapped BiCGSTAB/GMRES solve is the H-connectivity closure of the seed support,
  invariant under seed filtering (`spectrum_slicing.md`), and on FCC Ni the per-point
  support equals the union support (`bicgstab_per_frequency_gf.md`).
- The freeze-growth cap (`_CappedBasisProxy`) bounds memory but retains determinants in
  essentially *discovery order* (everything admitted until the cap, one importance-ranked
  boundary step at the overflow, then frozen). At a binding cap the retained set is not the
  important set: on FCC Ni at cap 400k, every method's self-energy failed causality.

The old `CIPSI_Basis.expand_at` (deleted in `582bb0f`/`f219ca6`, last at
`f219ca6^:src/impurityModel/ed/cipsi_solver.py`) was intended for exactly this: expand the
basis around a caller-supplied reference energy using the CIPSI second-order importance.
This work revives it in modern form and benchmarks it as a per-frequency truncation policy.

## Why iterate-referenced CIPSI selection is the principled criterion

For the resolvent linear system `(z - H) X = s` solved on a subspace `P` (with
`supp(s) ⊆ P`), the true residual at an out-of-basis determinant `D` is exactly
`-<D|H|X>` — the solver residual inside `P` is driven to `atol`, so the *entire* remaining
error lives on the boundary. First-order inversion of the diagonal gives the leading-order
weight of `D` in the exact solution:

```
w_D = sum_i |<D|H|X_i>|^2 / |z - E_D|^2
```

That is precisely the CIPSI Epstein–Nesbet importance with the reference energy replaced by
the (complex) resolvent frequency and the reference state replaced by the current iterate.
Two consequences:

- `expand_at` with the *iterate* as reference is residual-driven greedy selection — not a
  heuristic transplant from ground-state CIPSI but the exact leading-order error metric of
  the linear system being solved.
- The energy denominator is what makes it *frequency-targeted*: near-resonant determinants
  (`E_D ≈ Re z`) are the most important, and `Im z` (the broadening `iδ`, or the Matsubara
  distance) regularizes the divergence — no clamp needed, unlike the ground-state scorer.

## Survey of dynamic-truncation policies

| Policy | Status | Notes |
| --- | --- | --- |
| Residual-driven CIPSI loop (`GF_CIPSI_SCORER=de2`) | **implemented, primary** | solve frozen → score boundary of iterate → admit top → re-solve; stops on the boundary residual, the budget, or the round cap |
| Bare-coupling scorer (`GF_CIPSI_SCORER=amplitude`) | **implemented, baseline** | same loop, `w_D = sum_i |<D|H|X_i>|^2` without the denominator — what the frequency targeting must beat |
| PT2 downfolding of the discarded boundary (`GF_CIPSI_PT2`) | **implemented** | Löwdin/Feshbach at second order: `dG_ij = sum_D <D|H|X_i>(z-E_D)^{-1}<D|H|X_j>` over the unadmitted candidates (complex-symmetric approximation for the bra); magnitude always recorded as the per-point truncation-error bar |
| Warm-start top-K amplitude pruning | **implemented (always on)** | between frequency points the warm-start-only support is pruned to fit the budget by collective amplitude bisection; seeds are never truncated (the bicgstab seed-overflow contract) |
| Freeze-growth cap (`gf_method="bicgstab"` + `truncation_threshold`) | pre-existing comparator | discovery-order retention; exact on the retained `P H P` |
| Chebyshev spectrum slicing | ruled out | `spectrum_slicing.md`: the live basis is the connectivity closure of the seed support, invariant under filtering — no memory win |
| dN occupation windows | pre-existing, orthogonal | bounds *which sectors* exist, not how many determinants; null on metals (FCC Ni gate) |

## Implementation

- `CIPSISolver.select_at(z, psi_ref, H, de2_min, max_new, scorer)`
  (`src/impurityModel/ed/cipsi_solver.py`): one resolvent-targeted selection round on the
  shared candidate machinery (`_candidate_overlaps_and_energies`: hash-owned candidate
  enumeration from the redistributed `H|psi_ref>`, coupling matrix, hash-phase diagonal
  probe for `E_D`) and the shared fixed-budget collective admission (`_admit_top`). Returns
  the admitted set plus the global per-column boundary residual norms and the rank-local
  PT2 ingredients. Fully collective; rank-count invariant by construction (sorted
  candidates + collective bisection).
- `block_Green_cipsi` (`src/impurityModel/ed/gf_solvers.py`): the per-point driver. Same
  systems, unit contract and warm-start chain as `block_Green_bicgstab`; per point it
  rebuilds the basis from the seed + (budget-pruned) warm-start support, then runs the
  solve→select→admit loop with every solve *frozen* on the current basis (a
  `_CappedBasisProxy` capped at the current size — exact BiCGSTAB/GMRES of `P H P`), so
  growth belongs to the selection, never to the solver's connectivity closure.
- Routing: `get_Greens_function(gf_method="cipsi")` reuses the bicgstab
  distribution/assembly verbatim (the point kernel is a parameter of
  `_get_greens_function_bicgstab`); the memory model maps `"cipsi"` onto the bicgstab
  live-vector estimate; the diagnostics report gains `cipsi_boundary` (worst per-point
  boundary residual vs `GF_CIPSI_BOUNDARY_TOL`).
- Knobs (`config.py`, all `GF_CIPSI_*`): `BUDGET` (per-point determinant cap; defaults to
  `truncation_threshold`), `MAX_NEW` (per-round admission stage), `DE2_MIN` (importance
  floor), `MAX_ROUNDS`, `BOUNDARY_TOL` (defaults to the solver `atol`), `SCORER`, `PT2`.
- Tests: `src/impurityModel/test/test_gf_cipsi_driver.py` on the SIAM-6 dense-resolvent
  oracle (uncapped exactness, monotone budget decay, boundary residual as error bar, PT2
  improvement, scorer knob, driver-vs-Lanczos, MPI lock-step and capped rank-count
  invariance).

### The boundary residual is the honest error bar

Freeze-growth reports only that the cap was hit; the truncation error itself is invisible.
The CIPSI loop *measures* it every point: `sum_D |<D|H|X_i>|^2` over the boundary is the
squared norm of the true residual outside `P`. It is reported per block
(`cipsi_boundary`), drives the stop decision, and bounds the observed `G` error in every
benchmark below.

## NiO benchmark

Workload: `impmod_tests/NiO/impmod/5_BathStates_linked_chainGeometry.../impurityModel_data.h5`
(58 spin-orbitals: 10 impurity + 48 chain-linked bath, `chain_restrict`, `delta=0.01`,
`tau=0.0025`, ground state 72 determinants — thermal triplet). Real-axis mesh subsampled to
5 points (each an independent resolvent solve), `GF_BICGSTAB_ATOL=1e-6`, 4 MPI ranks.
Reference: uncapped block Lanczos (`reort="partial"`, converged to ~1e-9).

Scale anchors: the uncapped per-point connectivity closure is **~2.9–3.1k determinants**
(uncapped bicgstab `max_solve_basis`), so budgets of 300–3000 probe genuine truncation.
The ground state (72 determinants) is untruncated in every run.

### Accuracy vs budget (max relative `|dG(w)|`, max absolute `|dSigma(w)|`)

| run | wall (4 ranks) | max rel dG | max dSigma | boundary residual (reported) |
| --- | --- | --- | --- | --- |
| lanczos ref | 50 s | — | — | — |
| bicgstab uncapped | 227 s | 2.1e-7 | 4.3e-7 | — |
| bicgstab cap 3000 | 244 s | 2.3e-7 | 4.3e-7 | — |
| **cipsi 3000** | 492 s | 5.8e-4 | 6.6e-4 | 2e-2 / 1e-1 |
| **cipsi uncapped** | 481 s | 5.8e-4 | 6.6e-4 | 2e-2 / 1e-1 |
| bicgstab cap 1000 | 97 s | 1.5e-2 | 2.8e-2 | (cap_hit only) |
| **cipsi 1000** | 150 s | 1.6e-2 | 2.9e-2 | 4.1e-1 / 4.8e-1 |
| **cipsi 1000, amplitude scorer** | 145 s | 1.6e-2 | 3.1e-2 | 2.6e-1 / 4.3e-1 |
| **cipsi 1000 + PT2** | 157 s | 3.0e-2 | 5.7e-2 | — |
| bicgstab cap 300 | 24 s | 8.6e-2 | 1.3e-1 | (cap_hit only) |
| **cipsi 300** | 25 s | 7.7e-2 | 1.2e-1 | 8.6e-1 / 1.1e0 |
| **cipsi 300 + PT2** | 27 s | 2.0e-1 | 2.7e-1 | — |

Per-frequency, the error of *both* methods concentrates on the two low-`omega` points
carrying the spectral weight and is statistically indistinguishable between the policies at
every point and budget. Self-energy causality holds in every run (this workload does not
reproduce the FCC Ni cap-400k causality failure). The `cipsi 3000/uncapped` residual of
5.8e-4 sits at one hard point and is *round-limited*, not budget-limited: `MAX_ROUNDS=8`
H-shell expansions from the 72-determinant seed support stop short of the converged
resolvent tail there (the basis reached only ~2.2–2.8k), while uncapped bicgstab's
connectivity closure reaches it.

## Verdict

**Importance-ranked per-frequency truncation matches — but does not beat — freeze-growth
retention at equal determinant budget on NiO, at 1.5–2x the wall cost.** Concretely:

1. *Selection quality*: at budgets 300 and 1000 the CIPSI-selected basis and the
   discovery-order freeze-growth basis give the same error to within ~10%. The resolvent's
   importance distribution over the chain-restricted closure is evidently flat beyond the
   near-seed shell both policies retain identically: there is no concentrated "important
   subset" for the selection to find. This is the third face of the closure-invariance
   verdicts (slicing invariance; per-point support == union).
2. *Frequency targeting is a no-op*: the `de2` resolvent scorer and the bare-coupling
   `amplitude` scorer are indistinguishable. The energy denominator never discriminates,
   because the candidate energies within the occupation-restricted space are broad compared
   to the evaluation window — nothing is near-resonant enough to stand out at `delta=0.01`.
3. *PT2 downfolding hurts at binding budgets*: with boundary residuals of 0.3–1.1 the
   discarded space is not perturbative, and the correction (validated helpful on the SIAM-6
   oracle at moderate truncation) overshoots. Useful only when the boundary is already
   small — where the plain result is already good.
4. *What the method genuinely adds*: the **measured boundary residual**. Freeze-growth
   reports only that the cap was hit; the CIPSI loop reports a conservative (~10–25x) upper
   bound on the actual `G` error per block, every run, and stops itself when a tolerance is
   met. That turns silent truncation into a quantified one.

**Recommendation**: keep `gf_method="cipsi"` as the diagnostic/error-bar variant and for
budget regimes where an application demands a certified truncation error; use plain
`bicgstab` + cap when speed at a fixed memory budget is the goal. Do not expect a
memory-at-equal-accuracy win from determinant selection on workloads whose importance
distribution is closure-flat — on the evidence of NiO (and the prior FCC Ni gates), that is
the norm for these impurity models, not the exception. If a future workload shows a
concentrated importance profile (e.g. core-level XAS at large `delta`, where near-resonant
selection has something to bite on), the machinery is in place and rank-count invariant.

### Follow-ups worth considering

- `GF_CIPSI_MAX_ROUNDS` is the binding limit for tight boundary tolerances (each round is
  one H-shell): raising it trades solves for reach. A shell-doubling schedule (admit
  boundary-of-boundary candidates within a round) would decouple reach from solve count.
- The boundary-residual reporting is independent of the selection policy; it could be
  grafted onto the plain bicgstab path as a cheap one-shot `select_at(max_new=0)` after
  the final solve of each point, giving freeze-growth the same error bar without the extra
  rounds.
