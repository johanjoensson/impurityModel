# RIXS R2 (final-state resolvent) performance: proposed tweaks

**Status (2026-07-14): proposal, not started.** Written up following the NiO L3
validation of the RIXS R1 solver chain (see [[rixs-r1-solver-chain]] in memory /
commits `48e001a`/`0c43d93`/`3466450`), which measured where the remaining wall time
goes on a real workload. Nothing here is implemented; this is the analysis the next
session should start from, not an execution plan with resolved checkboxes (per the
`doc/plans` convention in `README.md`, an execution plan of this must be free of open
choices — the choices below are still open, so treat them as **design options**, not
tasks, until one is picked and a proper checkbox plan is written for it).

## The measured problem

The R1 solver chain (dense `SectorResolventCache` -> `KrylovShiftedResolvent` ->
per-point BiCGSTAB -> GMRES) is validated and, on a forced-recycler NiO L3 run, fully
solved: all 75 R1 solves served by the shift-recycled Krylov recurrence, worst relative
residual 9.9e-16, zero declines to the per-point fallback.

The wall time on that same run was **2679 s vs 581 s** with the dense sector caches
enabled (4.6x) -- and **all of that difference is R2**, the final-state (energy-loss)
resolvent evaluated once per `(eigenstate, wIn, in-component)` point (309 calls on this
workload). Each is a fresh `block_Green` call: its own basis rebuild, its own
block-Lanczos recurrence, run to whatever the convergence monitor demands. 309 of those
recurrences ran to 218-536 blocks and *still* warned about not reaching the monitor's
target -- yet the resulting map matched the dense truth at 1.45e-4, comfortably inside
the 1e-3 the adaptive sampler was asked for. The monitor was chasing an accuracy nobody
needed.

R1 is therefore solved; R2 is where the next win has to come from. The tweaks below are
ordered by expected effort-to-payoff, not dependency (1 does not require 2/3/4).

## Tweak 1 -- align the R2 convergence target with the map's actual accuracy

**The cheapest tweak and the one the numbers most directly support.**

The R2 `block_Green` call in both `getRIXSmap_new`'s and `getRIXSmap_tensor`'s
`eval_out` passes no `eval_meshes`, so `_make_gf_convergence_monitor` falls back to its
spectral-edge mesh and converges to `_gf_rel_tol(slaterWeightMin)` -- effectively the
`_GF_REL_TOL_FLOOR` of 1e-9 (`gf_convergence.py`) unless `slaterWeightMin` is looser.
But the map only needs to land within `_RIXS_R1_ATOL` (1e-6) of the dense truth in the
non-adaptive case, or the caller's own `adaptive_wIn_tol` (measured 1e-3 on NiO L3) in
the adaptive case -- three to six orders of magnitude looser than what the monitor is
silently targeting.

This is exactly the failure mode the monitor-axis fix (`_gf_eval_meshes`, see
[[gf-monitor-was-converging-the-wrong-axis]]) already fixed once, for a different
mismatch (Matsubara-only self-energies converging a real-axis fallback mesh they never
evaluate): 3.6-4.1x wasted blocks, fixed by testing convergence *where the caller
evaluates*. This is the same class of bug -- converging to the wrong *tolerance*, not
the wrong *mesh* -- with a comparable-or-larger blast radius (order-of-magnitude gap,
not a small constant factor).

**Proposed change:**

1. Plumb `eval_meshes=[wLoss + E_e]` (the real mesh `calc_G`/`calc_thermally_averaged_G`
   evaluates R2 on) into both `eval_out`'s `gf.block_Green(...)` calls, mirroring how
   `get_Greens_function` already builds `eval_meshes` for its own block-Lanczos calls
   (`_gf_eval_meshes`, `greens_function.py`).
2. Give `block_Green` (and `block_green_impl`) an overridable relative-change tolerance,
   derived from `_RIXS_R1_ATOL` on the RIXS path -- not a new hardcoded literal (the
   repo's single-source-of-truth convention, see [[no-duplicated-tolerance-literals]]).
   The natural seam is a `rel_tol=None` parameter on `block_Green`/`block_green_impl`
   that defaults to today's `_gf_rel_tol(slaterWeightMin)` when unset, and is threaded
   into `_make_gf_convergence_monitor` in place of that computed value.
3. RIXS's `eval_out` passes `rel_tol=_RIXS_R1_ATOL` (or, on the adaptive path, something
   derived from `adaptive_wIn_tol` -- needs a decision, see Open question below).

**Expected effect:** fewer Lanczos blocks per R2 evaluation, roughly in proportion to how
far past the useful accuracy the monitor currently overshoots -- on the order the
monitor-axis fix measured for an analogous mismatch (single digit multiples), though R2's
gap here is target-vs-target rather than mesh-vs-mesh and hasn't been benchmarked
directly. **Must re-validate against the dense truth after the change** (same NiO L3
comparison harness already used, `scratchpad/rixsmpi/compare.py`): the accuracy budget is
being spent deliberately looser, so confirm the map still lands inside its tolerance
before trusting the speedup.

**Open question:** what `rel_tol` to pass on the *adaptive* wIn path. `_RIXS_R1_ATOL` is
the right bound for a dense (non-adaptive) map, but the adaptive sampler's own tolerance
(`adaptive_wIn_tol`, e.g. 1e-3 on NiO L3) is looser still and arguably the tighter of the
two should win only where they actually interact (R2 error compounds into the AAA fit,
which then reconstructs the *unsolved* points -- an R2 error above the AAA tolerance could
poison the reconstruction, not just the solved point). Needs a decision before this
becomes a checkbox plan, not an on-the-spot implementation call.

## Tweak 2 -- Krylov reuse across wIn (R2's own shift-recycling)

R2's Hamiltonian and shift set (`wLoss + i*delta2 + E_e`, fixed for a given eigenstate)
are identical across every wIn point of a chunk; only the seed block (the R1 solution,
which varies smoothly with wIn) changes. The existing framing --"R2 has no recycled
analogue by design: its seeds change with wIn (no shared RHS)" (see
[[rixs-r1-solver-chain]]) -- only rules out the *exact* `KrylovShiftedResolvent`
mechanism (one recurrence, one seed block, many shifts). It does not rule out reusing
Krylov *work* across a smoothly-varying seed.

Two concrete mechanisms, both research-grade (neither has a working prototype):

- **Seed-deflated continuation.** Keep the previous wIn point's semi-orthogonal Krylov
  basis `Q`; project the new point's seed onto it; extend the recurrence only in the
  orthogonal complement of that projection. Pays off to the extent consecutive R2 seeds
  are linearly close (true for a fine wIn grid, by construction of the adaptive sampler
  choosing nearby points when the map is smooth there).
- **GCRO-DR-style deflation.** Retain a handful of near-pole Ritz vectors from the
  previous solve and use them to deflate the next one's initial residual (the standard
  "recycled Krylov subspace" idea from GCRODR/GMRES-DR, adapted to block Lanczos).

Both inherit `KrylovShiftedResolvent`'s existing constraints if built the same way:
`reort="partial"` (reconstruction needs semi-orthogonality), the store bounded by
`GF_KRYLOV_RECYCLE_MAX_BYTES`, and no complex64/tail-only retention (reconstruction reads
Q back, unlike the plain continued-fraction path).

**This is the higher-risk, higher-payoff option** -- it could in principle collapse R2
from "one recurrence per wIn point" to "one recurrence plus cheap corrections," which is
the same shape of win R1 already banked. It needs a throwaway prototype behind an env
knob (e.g. `GF_R2_RECYCLE=1`) measured against the dense truth before it goes anywhere
near the production path, not a direct implementation.

## Tweak 3 -- union-seed chunk block

Simpler and lower-risk than Tweak 2: instead of one `block_Green` call per wIn point,
build ONE call per chunk over the flattened union of that chunk's R2 seed blocks (all
`(eigenstate, wIn, in-component)` seeds in the chunk stacked as one wide block), the same
pattern R2's own `try_eval`/tensor contraction already uses to serve every polarization
cross-term from one block-Lanczos (see `getRIXSmap_tensor`'s `eval_out`). The existing
rank deflation inside `block_Green`/`block_bicgstab` trims the redundancy automatically
when seeds are near-parallel (adjacent wIn points' R1 solutions are close for a smooth
map), same guarantee already relied on for the Cartesian polarization seeds.

**Trade-off to size before committing:** one recurrence over a block of width `W` costs
more per Lanczos step (`O(W^2)` overlaps) than `W` separate width-1 recurrences, so this
only wins if the deflation collapses the effective rank well below `W` -- true when the
chunk's seeds are nearly co-linear (smooth wIn dependence), false when they are not
(near a resonance, or a chunk spanning a rapid feature). Needs a measurement of R1's own
seed-block rank deflation on a representative chunk (that machinery already exists and
is exercised daily) before estimating whether R2's seeds deflate similarly.

## Tweak 4 -- distributed dense sector cache

`SectorResolventCache.try_eval`/`try_solve` decline whenever `basis.comm.size > 1`
(serial/single-color only). The NiO L3 validation's own unit-color discovery narrows
where this actually matters: `run_units_distributed` splits ranks into per-unit serial
colors whenever units >= ranks, so the dense cache already serves under MPI in the
common case (measured: both ranks at `-n 2` eigendecomposed their own 636-det sector
normally). This tweak only pays off when a single unit's sector must span multiple
ranks -- more MPI ranks than RIXS work units, or a sector too large for one rank's
memory budget even though the *unit* itself isn't distributed.

Two shapes, in order of effort:

- **Rank-0 `eigh` + broadcast**, when the sector fits comfortably in one rank's memory
  budget but the unit's basis happens to be split across ranks anyway. Cheapest: no new
  linear algebra, just a collective broadcast of the eigenvectors instead of a
  replicated per-rank compute.
- **ScaLAPACK/ELPA distributed Hermitian eigensolve**, for sectors too large for any
  single rank -- real infrastructure work (a new dependency), only worth it if a
  production workload is actually found where the sector genuinely doesn't fit one
  rank's share of memory. No such workload is known yet.

**Recommendation: defer this one.** It unblocks a case ("more ranks than work units, or
an outsized sector") that hasn't been observed on a real workload -- the NiO L3 run's
"declined" path was reached via `GF_SECTOR_DENSE_MAX=0`, a forced test condition, not an
organic decline. Revisit if a real run actually hits it.

## Already in place (so nobody re-proposes it)

The R2 dense sector cache (`SectorResolventCache`, shared with R1) already has
`GF_SECTOR_CACHE_DIR` disk persistence for its one-time `eigh` cost: measured 150 s warm
vs 589 s cold serial on the NiO L3 workload. That part of "cache the R2 sector" is
shipped; the remaining cost is the *fallback* path (declined sectors, or -- per this
doc's main finding -- an over-converged monitor even on served-by-cache-adjacent solves
that still fall through to `block_Green`).

## Recommended order

1. **Tweak 1** (convergence-target alignment) first: smallest change, most directly
   supported by the measured data (the 309 non-convergence warnings were all chasing an
   unneeded tolerance), and independently valuable regardless of whether 2/3 ever land.
2. **Tweak 4** (distributed dense cache) next, but only in its cheap rank-0-broadcast
   form, and only if/when a real workload is found where a unit's sector actually spans
   multiple ranks -- don't build it speculatively.
3. **Tweak 3** (union-seed chunk block) before Tweak 2: same rank-deflation mechanism
   already in production use elsewhere, lower implementation risk, and its measurement
   (chunk seed-rank deflation) is a useful input to deciding whether Tweak 2 is worth
   the research effort at all.
4. **Tweak 2** (Krylov reuse across wIn) last: the highest potential payoff, but
   genuinely research-grade -- prototype behind an env knob, measure against the dense
   truth, and expect it to take real iteration before it is production-ready.
