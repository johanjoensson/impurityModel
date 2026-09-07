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
4's `GS_MAX_BLOCK_WIDTH`) pins it to a config constant -- and that phase's own gate must be
measured against Phase 3's manifold-shrinking output, not today's (Phase 4's text says so
explicitly), so the real dependency order is **Phase 3 -> Phase 4 -> Phase 2's remainder ->
Phase 5**, not the original 2-3-4-5 listing. What still landed from Phase 2 independent of that
dependency (`estimate_gs_peak_bytes`'s `replicated_bytes` chunked-bound fix) is safe regardless
of `block_width`'s value, since it only lowers a term that no longer matched Phase 1's code.

**Phase 2's remainder, done:** `GS_MAX_BLOCK_WIDTH` is implemented (Phase 4) and now threaded
through every `estimate_gs_peak_bytes` call site via `memory_estimate.resolve_gs_block_width`:
`groundstate.calc_gs` and `dc_criteria.py`'s two sites read it directly (falling back to the
historical `4` when unset); `selfenergy.py`/`susceptibility.py` combine it with their own
GF-derived block width via `resolve_sizing_block_width`'s `max()`, since
`suggest_truncation_threshold`/`log_memory_budget` size both paths through one `block_width`
parameter and a real refactor to two separate parameters was judged not worth it for this
campaign. **The knob defaults to unset** (Phase 4's own gate -- the SMO width sweep against
`0.25*tol/|chi|` in `mu` -- has not run), so production still sizes the ground-state term with
`block_width=4` unless an operator sets `GS_MAX_BLOCK_WIDTH` by hand; `log_memory_budget` now
prints a warning on that path rather than letting the placeholder look like a measured bound.
**The original failure mode -- `suggest_truncation_threshold` returning ~118M determinants at
the Arrhenius point, predicting 139 GiB/rank at the real `p=315` -- is therefore still reachable
after this commit.** Running the width sweep and setting the knob from its result is what
actually closes it; see the Verification section.

A review of the first version of this threading caught a second hole in it: capping
`block_width` alone is not enough, because `num_wanted` is explicitly *not* capped (Phase 4's
own text -- "Keep `num_wanted` untouched") and keeps growing with the manifold independent of
the block-width cap. `_gs_krylov_columns`'s default (`num_wanted=None` assumes `2*block_width`)
is only valid in the pre-Phase-4 coupled regime; once `GS_MAX_BLOCK_WIDTH` decouples the two,
that default silently under-counts the retained Krylov store by the same ratio Phase 0 measured
between `num_wanted` and `p` (up to ~30x).

**The first fix attempt was itself wrong, caught by a second review round.** It passed the
bisection candidate `n` itself as `num_wanted` whenever the block width was knob-capped, on the
reasoning that this forces `_gs_krylov_columns`'s own `min(..., n_dets)` clamp to bind -- an
"invariant-subspace worst case" in the same style `estimate_gf_peak_bytes` already uses for its
Krylov store. The parity claim doesn't hold: `estimate_gf_peak_bytes` only takes that worst case
when `reort != "none"`, not the GF production default; `estimate_gs_peak_bytes` has no `reort`
parameter and models `reort="full"` unconditionally, so the same substitution is *unconditional*
on GS sizing -- exactly the path `GS_MAX_BLOCK_WIDTH` exists to enable. Worse, the substitution
makes the term scale like `n_dets^2/ranks` (verified: `blocks -> n_dets/p`, `columns -> n_dets`),
astronomically more pessimistic than the real `num_wanted` Phase 0 measured (105-315 at the ~1M
production cap). Setting `GS_MAX_BLOCK_WIDTH` under this fix made `suggest_truncation_threshold`
recommend a cap 5-30x *smaller* than the real target at realistic rank counts -- inverting the
knob's purpose.

**Reverted.** There is no analytically-safe worst case for `num_wanted` other than `n_dets`
itself, and that bound is useless in practice. `num_wanted` is a measured quantity, exactly like
`nnz_per_state` already is (Phase 2 item 3): `estimate_gs_peak_bytes`'s `num_wanted=None` default
keeps the pre-Phase-4 `2*block_width` assumption unconditionally (a no-op whether or not the knob
is set), and a new `gs_num_wanted` parameter (`suggest_truncation_threshold`/`log_memory_budget`/
`_suggest_for_budget`/the `_main()` CLI probe's `--gs-num-wanted`) lets a caller supply the real
value once measured. `log_memory_budget` now warns specifically on the dangerous configuration --
`GS_MAX_BLOCK_WIDTH` set but `gs_num_wanted` not supplied -- rather than silently assuming the
coupled default or silently over-correcting to the useless quadratic bound. None of the five
production call sites supply `gs_num_wanted` yet: doing so needs the same width sweep that sets
`GS_MAX_BLOCK_WIDTH`, which also has to record the resulting `num_wanted` (unaffected by the
block-width cap -- Phase 4 explicitly preserves the returned manifold, so this is an independent
measurement, not a derived one).

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

`blocks/width` ratio: min 1.62, max **30.00**, mean 14.31 -- worst case `blocks=60` at
`width=2`, recorded (already-padded) `num_wanted=20`.

No flat constant covers this: the ratio is unbounded as `num_wanted/block_width` grows once the
two are decoupled. Fixed by replacing the constant with a function,
`memory_estimate._gs_krylov_columns(n_dets, block_width, num_wanted)`, meant to mirror
`_size_subspace` exactly.

**The first version of that function got the mirror wrong, caught by `/code-review`:** it
reproduced only `max_subspace = max(2*nw, nw+10)` and dropped `_size_subspace`'s own
`blocks = min(2*ceil(max_subspace/p) + 20, max(2, cap//p - 1))` -- both the factor of 2 and the
flat `+20` block headroom, replacing it with a bare `ceil(max_subspace/p)`. Checked against the
measured worst case above (`width=2`, internal `num_wanted=20`, i.e. unpadded `num_wanted=10` in
this function's own convention): the first version gave `blocks=20` against the real 60, a 3x
undercount, in the same direction as the OOM this campaign exists to fix. The `2*` and `+20` are
not a rounding nicety -- they are most of the formula at small-to-moderate width.

Fixed by transcribing `_size_subspace`'s body verbatim rather than its docstring's prose summary
(the earlier version was written from the *prose*, which itself omits the `2*` and `+20` --
they are only visible in the actual return statement). Cross-checked by direct comparison: 2000
randomized `(num_wanted, width, cap)` triples fed to both `_size_subspace` (real) and
`_gs_krylov_columns` (mirror, with the internal/unpadded `num_wanted` convention translated
between them), zero mismatches. The corrected function also reproduces the measured worst case
above exactly (`_gs_krylov_columns(n_dets=2000, block_width=2, num_wanted=10)` returns 120
columns = 60 blocks).

Its default (`num_wanted=None` -> internally `2*block_width`) reproduces today's coupled-regime
behavior (e.g. at `block_width=315`, `n_dets=1e6`: 9450 columns, 30 blocks -- coincidentally close
to the pre-Phase-1 `n_blocks=30` literal, for an unrelated reason: that literal was never
derived, this is a converged asymptote of the real formula for large `p` in the coupled regime).
Phase 2's remainder must pass the real `num_wanted` once Phase 3/4 land and decouple it from
`block_width`, or this will under-predict again.

## Phase 5 — the calibrated determinant cap, ported from `DC_gap_perf`

Ported from branch `DC_gap_perf` (`f61bfaa` + `91109b8`, itself 119 commits stale against
`master`), re-derived rather than cherry-picked as the plan requires. That branch's commits mix
three concerns: the cap ladder itself, a verbosity/output overhaul (`Reporter`/`SILENT`, a
`-v`/`-vv`/`-vvv` ladder collapsing `Basis.verbose` into a separate `solver_verbose`), and a
post-search cap re-verification/retry mechanism. Only the first is this phase's scope; the second
is out of scope entirely (today's `dc_criteria.py` already has its own plain `verbose`/`rank`
convention, and duplicating a second verbosity abstraction on top of it is not this phase's job);
the third is scoped out below.

**Landed:** `dc_search.calibrate_truncation_threshold` (the cap ladder itself, its four
constants), ported with `narrate=SILENT`/`Reporter` replaced by this module's existing
`verbose: bool, rank: int` convention (matching `_find_nominal_sector_point`'s pattern: an
informational line gated on `verbose`, an unconditional `WARNING:` when the ladder exhausts its
rung budget without settling — matching `_report_unattainable_target`'s convention that a result
worth distrusting is never hidden behind a verbosity flag). Its three documented invariants
(returning `rungs[-1][0]`, never a further-doubled cap past the last measured rung; the SPAN over
a trailing window rather than one pairwise step; a `None` rung resetting the window rather than
being skipped) are pinned by `test/gf/test_dc_search_cap_ladder.py`, each checked against the
specific wrong behaviour it guards (a synthetic staircase whose every step sits exactly on the
pairwise gate; an undefined rung sitting between two agreeing pairs) rather than only against the
correct implementation, which is how the original bugs got past four tests on `DC_gap_perf`.

Also landed: the `dc_record.py` vocabulary this phase needs — `_FIELDS` (`dc_cap`, `dc_cap_drift`,
`dc_cap_check`, `slope`, `mu_tol_effective`) and `_ANNOTATIONS` (`dc_cap_parity`, `dc_cap_mu`,
`dc_cap_retried`), plus the five new `_annotate` cases (one per new `_FIELDS` key above;
`dc_cap_parity`/`dc_cap_mu`/`dc_cap_retried` are sidecar keys those five read via `.get()`, not
separate cases), ported as one group per the review of Phase 5 (1/3) even though
`dc_cap_mu`/`dc_cap_retried`/`slope` belong to the retry mechanism deferred below — `dc_cap`'s own
annotation reads `dc_cap_parity`/`dc_cap_retried`, so the group does not fragment across commits.
No caller writes any of these fields yet.

**Not yet landed (next commits in this phase):** wiring `calibrate_truncation_threshold` into
`fixed_gap_dc`/`fixed_occupation_dc` (only recalibrating a cap that defaulted from the memory
probe — an explicit `truncation_threshold` stays the caller's instruction); the cache-reuse fix
for the gap criterion (`91109b8`: the ladder's last-evaluated rung is the one about to be
consumed, so keeping its caches instead of clearing them unconditionally saves a full re-solve at
`mu = 0` — verified safe only because `calibrate_truncation_threshold`'s contract guarantees
`cap == rungs[-1][0]`, checked by an explicit `cached_cap != cap` guard rather than assumed); and
actually filling the `dc_record.py` fields already landed above (`dc_cap`, `dc_cap_drift`,
`dc_cap_parity`, `mu_tol_effective`) from the two criteria's own state — three separable pieces of
work, expected to land as separate commits per this repo's small-single-concern convention.

**Note on Phase 2's mid-search-cap-change hazard.** Phase 2's remainder section above argues that
changing `block_width` mid-search is hazardous because a populated `sector_at`/`n_center_at` cache
would then describe a different width than the context claims. The cap ladder changes
`ctx.truncation_threshold` mid-search in exactly the same sense — but it is not the same hazard:
every rung's evaluation clears every cache first (`ctx.sector_at.clear()`,
`ctx.n_center_at.clear()`, and the criterion's own local `sectors_at`/`width_at`), and the accept
path either keeps the last rung's caches (verified via the `cached_cap` guard) or clears them
too. No cache is ever read against a cap it was not built at. The two situations differ in that
respect: the cap ladder was designed around this hazard from the start, while a hypothetical
mid-search `block_width` change (Phase 2's concern) would need the same discipline added.

**Deferred, deliberately, past this phase: the post-search cap re-verification/retry
mechanism** (`_cap_holds_at`/`_run_gap_search` on the `DC_gap_perf` branch, gap criterion only).
The ladder above certifies the cap at `mu = 0` — the double-counting *guess* — but the cap is
*consumed* at whatever `mu` the search actually returns, which measured 0.325 away from the guess
on the workload the branch measured it on. The branch's own measurement: transporting a cap-2000
run's answer to a cap-500 run's `mu` along the measured slope leaves ~5.1e-3 in the gap centre —
60% of the acceptance band `tol/|chi|` — from the cap alone, at the point the answer is actually
returned. That is a systematic, not noise that averages out across charge-self-consistency
iterations. The mechanism re-solves the two off-centre sectors at double the accepted cap, at the
converged `mu`, and re-runs the whole search (once, not to convergence) if the answer moved more
than the calibration's own target — a second, separate correctness surface (saving and restoring
individual `sector_at` entries around the probe, a bounded retry loop) that deserves its own
adversarial review rather than riding in on this phase's first commits. Tracked here exactly as
Phase 1b is tracked above; the record field it would populate (`dc_cap_check`) is named in this
document's own earlier phase-5 sketch and its absence here is this deferral, not an oversight.
