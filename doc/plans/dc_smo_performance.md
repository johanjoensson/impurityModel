# SrMnO3 (SMO) gap-DC performance: Phase 0 measurements

Companion to `~/.claude/plans/crystalline-waddling-knuth.md` (streamlining the Gap/Occupation
double-counting search). Workload: `smo` in `restriction_diagnostics.WORKLOADS`
(`impmod_tests/SMO/cubic/impmod/impurityModel_data.h5`, iteration 1). All measurements below are
serial (`comm=None`) unless noted; `fixed_gap_dc` with `allow_charge_state_change=True`,
`verbosity=1`, cap as given.

## The bug that started this campaign

Two production runs (local `impmod_tests/SMO/cubic/impmod`, Arrhenius
`~/Dokument/arrhenius/SMO/cubic/impmod`) OOM'd inside the first DC evaluation. Root cause:
`memory_estimate`'s cap sizing assumes `block_width=4` at every call site, but the array-kernel
Lanczos block actually used is `len(psi_refs) + 1` -- the thermal manifold width, which grows
with the determinant cap through a feedback loop (`expand`'s `num_wanted = 2 * len(psi_refs)`
warm-starting the next solve). On the crashed Arrhenius run this reached `p ~ 105` (sweep) and
`k_ret` up to `3p ~ 315` (TRLM's retained Ritz block after a restart), against the model's
assumed 4 -- a ~26-79x under-prediction on the dominant term, which the bisection in
`_suggest_for_budget` answered with a 55.7M-determinant cap (should have been ~1-3M).

The actual crash site, traced from the Arrhenius traceback (`BlockLanczosCore.so`, not
`BlockLanczosArray.so` as first suspected): `block_apply`'s array branch in
`src/cython/_block_ops.pxi`, called from `trlm.py`'s restart-continuation loop **every block of
every restart** (not just the one-time initial sweep in `BlockLanczosArray.pyx`, which has the
same defect but fires far less often). Both materialized the full `(global_N, w)` matvec
product on every rank before an `Allreduce` and a 1/size slice.

## Measurement 1: block widths (`p`, `w`) scale with cap, confirming the mechanism

Instrumented via `solver_trace.note` in `cipsi_solver.get_eigenvectors` (`eigensolve_block_width`)
and `trlm.py`'s two `block_apply` call sites (`block_apply_width`, `site=continuation` /
`site=restart_rebuild`).

| cap | eigensolve p (sweep) | block_apply[continuation] w | block_apply calls | walltime |
|---|---|---|---|---|
| 500  | below `dense_cutoff` (dense eigensystem path, no TRLM) | -- | -- | 49.9 s |
| 2000 | n=388, min 2, max **16**, mean 4.7 | n=13,573, min 1, max **16**, mean 2.8 | 13,573 | 450.8 s |

Even at this modest cap, `block_apply` fires **13,573 times** in one DC search (5 evaluations) --
confirming it, not the one-time sweep, is the dominant call-count site. Width at cap 2000 (max
16) is far below the crashed run's p=105/k_ret=315 -- those numbers come from the ~1M-determinant
production cap, not reproduced here (would take hours; not needed to validate the fix, since the
fix's correctness does not depend on the exact width reached, only on peak memory scaling with
`max(w)` and shrinking with rank count instead of growing with `global_N`).

No `restart_rebuild` events fired in this run (`orth_err` stayed under `RESTART_ORTH_TOL` at every
restart, so the cheap textbook-coefficient arm handled all of them; the expensive Rayleigh-Ritz
rebuild arm -- the one that can reach `w = k_ret` up to `3p` -- was not exercised at this cap).

## Measurement 2: `nnz_per_state` -- the CSR term is a much smaller number than assumed

Instrumented via `solver_trace.note("h_matrix_nnz", ...)` at both `H_mat = build_sparse_matrix(...)`
sites in `cipsi_solver.py`.

| cap | n_builds | min | max | mean |
|---|---|---|---|---|
| 500 | 468 | 1.0 | 11.3 | **5.6** |

The `memory_estimate.py` default (`nnz_per_state=100`) **over-predicts by ~18x** on this
workload -- opposite direction from the block-width error, and on the CSR term specifically (a
minor contributor next to the replicated-buffer term at any nontrivial rank count, but worth
using a measured default rather than a guess since it is free to derive from the same matrix
build). Sample from one cap; re-check at a larger cap before trusting the number in production
(a denser cluster or different excitation budget could change the ratio).

## Measurement 3: `occupation_spread` is real on SMO -- Phase 3 cannot default here

Instrumented via `solver_trace.note("sector_occupation_spread", ...)` in
`_SectorContext.sector_solve`.

| cap | n_states (manifold) | max occupation_spread | manifold_spread (gap_report) |
|---|---|---|---|
| 500  | 6/2/4 (N+1/N/N-1) | 2.058e-2 | 0.011 |
| 2000 | 12/2/4            | 6.540e-2 | 0.058 |

**Not machine-zero**, and growing with cap. Unlike every fixture checked in
`gap-criterion-adversarial-review` (F4: 2e-16, 0.0, 5e-15), SMO's retained thermal manifold
genuinely does not share `N_imp` across states. **Conclusion for Phase 3: the
`occupation_ground` (T=0) substitution for the DC search's per-sector occupation changes the
answer on this workload and must ship as an opt-in knob, not a default** -- exactly the outcome
the plan's measurement gate was designed to catch.

## Measurement 4: measured vs predicted peak RSS -- serial probe under-exercises the real bug

| cap | measured peak RSS | predicted (ground state) |
|---|---|---|
| 500  | 335.5 MiB | 2.2 MiB |
| 2000 | 422.1 MiB | 9.0 MiB |

Both serial (`comm=None`), so `mpi=False` throughout -- **this does not exercise the
replicated-matvec bug at all** (that branch only runs under `mpi=True`). The ~40-150x gap here is
from elsewhere: the Python/import floor (`_PY_BASIS_OVERHEAD_BYTES`'s calibration constant),
`nnz_per_state` (measurement 2 shows the model's CSR assumption is 18x too high, which should
partly offset this gap once fixed, not explain it), and unmodeled transients (CIPSI candidate
generation, restriction bookkeeping). A serial run cannot validate Phase 2's real target (the
`mpi=True` replicated term); that needs an `-n>1` measurement, deferred to the Phase 1/2
end-to-end verification since it requires the fix to already be in place to run at a survivable
cap.

## Gap centre in `mu` (for judging later phases' savings)

| cap | mu | gap_center | chi | tol | tol/\|chi\| |
|---|---|---|---|---|---|
| 500  | 0.068424 | -0.001360 | -0.5011 | 2.76e-3 | 5.51e-3 |
| 2000 | 0.139038 | -0.000962 | -0.4905 | 2.50e-3 | 5.10e-3 |

Per `dc-perf-campaign-measured-levers`: judge every later saving against this resolution
(`tol/|chi|` in `mu`), never against a raw energy difference. Note `mu` itself moves by ~0.07
between these two caps -- larger than the ~5e-3 resolution -- so **SMO's cap ladder has not yet
settled at cap 2000**; unlike the NiO baseline this campaign's memory (`dc-perf-campaign-measured-
levers`) describes, this workload needs the calibrated cap ladder (Phase 5) for a trustworthy
answer, not just for speed.

## Open question, not blocking

The walltime figures here (49.9 s at cap 500, 450.8 s at cap 2000) are markedly slower than an
earlier memory note's SMO baseline (`smo-gap-dc-is-the-real-bottleneck`: 63.4-76.6 s at cap 2000
post-fix). Not reconciled -- possibly the physics/sector-mismatch issue (nominal occupation 3 vs.
achieved sector 5-7, tracked as out-of-scope in the campaign plan) drives more sector solves here
than on whatever fixture that note measured, or the referenced TRLM fix (`num_converge` narrowing
the eigensolver gate off the eigenstate pad) is not the current behavior. Current measurements
supersede the stale note; flagged rather than chased further, since it does not block Phase 1's
correctness gate.

## Phase resequencing: Phase 2's block-width threading needs Phase 4 first

Found while implementing Phase 2's "make the cap honest" item 2 (thread a real block width
through `memory_estimate`'s five call sites), before writing that code: the plan's original
order (Phase 2, then Phase 3, then Phase 4) assumed a cheap probe solve at a small cap could
stand in for the real production block width. Measurement 1 above refutes that directly: `p` at
cap 2000 is 2-16, against 105-315 at the ~1M-determinant production cap that actually crashed --
the width grows with the cap itself (the `num_wanted = min(2*len(psi_refs), len(self.basis))`
feedback loop, `cipsi_solver.py:852`, is bounded only by the *current* basis size, not by
`_EIGENSTATE_PAD`/`_MAX_EIGENSTATE_DOUBLINGS`/`_size_subspace` in isolation). A probe at a small
cap would under-report `p`, and since `block_width` still dominates
`estimate_gs_peak_bytes` post-Phase-1 (now via `krylov_bytes`, not `replicated_bytes` -- verified:
at the Arrhenius point, `p=315` alone gives `krylov_bytes` on the order of a GiB against tens of
MiB for `replicated_bytes`), threading an under-measured width through the five call sites would
*raise* the suggested cap, reproducing the OOM this campaign exists to fix.

The width only stops being a moving, unmeasurable target once a Lanczos-block-width cap (Phase
4's `GS_MAX_BLOCK_WIDTH`, **not yet implemented**) pins it to a config constant -- and that
phase's own gate must be measured against Phase 3's manifold-shrinking output, not today's (Phase
4's text says so explicitly), so the real dependency order is **Phase 3 -> Phase 4 -> Phase 2's
remainder -> Phase 5**, not the original 2-3-4-5 listing. What still landed from Phase 2
independent of that dependency (`estimate_gs_peak_bytes`'s `replicated_bytes` chunked-bound fix)
is safe regardless of `block_width`'s value, since it only lowers a term that no longer matched
Phase 1's code.

### The Krylov-store term needed a second look: no flat constant is safe

The first attempt at this section replaced the pre-Phase-1 `n_blocks=30` literal with a
documented constant of 4, derived from `k_blocks = ceil(num_wanted / p) ~ 2` in the regime where
`num_wanted` tracks `2p` (true today, since nothing decouples them yet). That derivation has a
real hole: it only holds while `num_wanted` and `block_width` move together. Once a block-width
cap exists, `num_wanted` is explicitly *not* touched (Phase 4's own text: "Keep `num_wanted`
untouched. The manifold that gets *returned* must not shrink") while `block_width` is capped --
so `k_blocks = ceil(num_wanted / p)` can exceed the coupled-regime asymptote by a wide margin.

Instrumented `cipsi_solver._size_subspace`'s three call sites directly (`solver_trace.note`,
kind `size_subspace`) and re-ran the cap-2000 SMO probe:

| site | n events | blocks range |
|---|---|---|
| initial | 388 | 26-60 |

`blocks/width` ratio: min 1.62, max **30.00**, mean 14.31 -- the worst case (`blocks=60` at
`width=2`, `num_wanted=20`) comes from an early, tiny-basis CIPSI iteration where `cap =
len(self.basis)` itself binds `_size_subspace`'s `max_subspace`, not the `2*num_wanted` term. At
production scale (a large basis, so `cap` never binds) the same shape of blowup recurs whenever
`num_wanted` is large relative to a capped `block_width` -- exactly the post-Phase-4 regime, e.g.
`num_wanted ~ 600` (an uncapped manifold request, historically observed) against
`block_width = 20` (a hypothetical `GS_MAX_BLOCK_WIDTH`) gives `blocks = ceil(1200/20) = 60`,
matching the empirical worst case's magnitude by the same mechanism.

No flat constant covers this: the ratio is unbounded as `num_wanted/block_width` grows, since the
two are no longer coupled. Fixed by replacing the constant with
`memory_estimate._gs_krylov_columns(block_width, num_wanted)`, which mirrors
`_size_subspace`'s own formula (`max_subspace = max(2*nw, nw+10)`, `nw = num_wanted +
_EIGENSTATE_PAD`) rather than assuming a ratio. Its default (`num_wanted=None` -> `2*block_width`)
reproduces today's coupled-regime behavior; Phase 2's remainder must pass the real `num_wanted`
once Phase 3/4 land, or this will under-predict again.
