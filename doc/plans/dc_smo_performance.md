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

> **Superseded in part -- see "Phase 1b / Side-finding" at the end of this document.** This
> heading rests on a single TRLM data point (cap 2,000; cap 500 is below `dense_cutoff` and runs
> the dense path). A later measurement at cap 8,000 finds `p`'s *maximum* unchanged at 16 and its
> *mean* grown only 4.7 -> 5.06, so "scale with cap" is not established between these caps and
> `p` should not be interpolated across them in either direction.

Instrumented via `solver_trace` in `cipsi_solver.get_eigenvectors` (`eigensolve_block_width`)
and `trlm.py`'s two `block_apply` call sites (`site=continuation` / `site=restart_rebuild`;
originally a `block_apply_width` note, now the timed `block_apply` kind -- see Phase 1b).

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
No caller wrote any of these fields yet at this point in the phase — see Phase 5 (3/3) below.

**Landed (Phase 5, 3/3):** `calibrate_truncation_threshold` wired into `fixed_gap_dc`/
`fixed_occupation_dc` (only recalibrating a cap that defaulted from the memory probe — an
explicit `truncation_threshold` stays the caller's instruction, gated by the broadcast
`ctx.cap_from_memory` on `_SectorContext`/`_OccupationContext`); the cache-reuse fix for the gap
criterion (`91109b8`: the ladder's last-evaluated rung is the one about to be consumed, so keeping
its caches instead of clearing them unconditionally saves a full re-solve at `mu = 0` — verified
safe via an explicit `cached_cap != cap` guard rather than assumed, since `calibrate_truncation_
threshold`'s contract guarantees `cap == rungs[-1][0]` but a caller should check that itself
rather than trust it blindly); and the `dc_record.py` fields already landed in Phase 5 (2/3)
(`dc_cap`, `dc_cap_drift`, `dc_cap_parity`, `mu_tol_effective`) now actually filled from the two
criteria's own state. Landed together in one commit rather than three, once it became clear the
three pieces share one small block of code in each criterion and splitting them would mean an
intermediate commit that calibrates a cap and then discards it unrecorded.

A review of that commit found `mu_tol_effective`'s write sites in both criteria used truthiness
(`if dc_rec.get("delta_sum"):` / `if occ_chi:`) to guard against dividing by a zero slope —
correct against a crash, but it silently omitted the field on a genuinely measured zero slope
(a real charge-transfer level crossing, or occupation plateau) instead of recording it, which is
exactly the case `dc_record.py`'s own annotation of this field exists to flag and had already been
fixed twice to tell apart from a missing measurement (Phase 5 (2/3)'s own review rounds). Fixed to
`is not None`, writing `float("inf")` when the slope is exactly zero rather than dividing by it —
`dc_record`'s annotation recomputes the same zero-slope test independently and prints "not a
meaningful bound" beside it. Extracted into a shared `_mu_tol_effective(tol, per_mu)` helper, since
the two criteria had computed this independently and picked up the identical truthiness bug in
both places — one formula, one place to get it right, rather than two copies free to drift apart.

That same review also found the ~20-line cap-calibration block (the ladder call, the
`dc_cap`/`dc_cap_drift`/`dc_cap_parity` bookkeeping, the `cached_cap` tracking a cache-reuse guard
verifies against) duplicated near-verbatim between the two criteria — a separate duplication from
`mu_tol_effective`'s, not its cause, despite an inaccurate claim to that effect in the first draft
of this fix (corrected in `_calibrate_cap`'s own docstring). Extracted into a shared
`_calibrate_cap` helper, parameterized by each criterion's own evaluate-at-`mu=0` callback and
cache-clearing callback; each criterion still supplies its own keep-vs-clear cache policy
explicitly (the one genuine difference between them — see `_calibrate_cap`'s docstring for why the
occupation criterion has nothing to keep) rather than duplicating the boilerplate around it.

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

---

# Verification: the campaign's acceptance tests, run locally

The plan's four end-to-end acceptance tests, run on the local desktop (8 cores, 15 GiB RAM,
8 GiB swap) against the same archive the production run crashed on. Item 3 is done; item 1 is
done only in its local, sensitivity-checking half (the plan specifies it "under `mpiexec -n 128`
conditions", which this machine cannot supply); item 2 (the `RUN_DC_DIAG` cap-ladder comparison)
and item 4 (the Arrhenius resubmission) are not.

## Item 1: the suggested cap's sensitivity to the block width

`memory_estimate._suggest_for_budget(budget, 58, block_width, "none", 1, 100, 1)` at **one
pinned** per-rank budget of 4.750 GiB (`safety 0.5` of a 9.5 GiB probe). Pinned deliberately:
the obvious way to run this — two `python -m impurityModel.ed.memory_estimate` invocations —
re-probes live `MemAvailable` each time and is not a controlled comparison, which is the same
defect this document rejects the pinned-cap A/B for below. (Those two CLI runs gave 962,476 and
88,906 off 9.5 and 9.4 GiB probes respectively; the controlled numbers are these.)

| `block_width` | suggested `truncation_threshold` |
|---|---|
| 4 (the stale default every call site used to pass) | 963,770 |
| 105 (the block width the crashed Arrhenius run reached) | **90,283** |

The answer moves **10.7x** on this axis alone. That is what this measurement establishes, and
all it establishes. The CLI additionally emits the Phase 2/4 caveat that `GS_MAX_BLOCK_WIDTH` is
unset, so the ground-state figure is a placeholder rather than a bound — and the 10.7x confirms
that caveat is not decoration.

**Three things this is *not* evidence of**, against the temptation to read it as a pass:

- **Not a pass of the plan's acceptance band.** The plan asks that the suggested cap land in
  1e4-1e6 rather than 5.6e7. Both rows satisfy that on this desktop, the stale one included, so
  clearing the band here says nothing. 5.6e7 is only reachable in the Arrhenius 128-rank memory
  configuration — which is item 1's own outstanding half (the plan specifies it "under
  `mpiexec -n 128` conditions"), not item 4. Item 4 is the resubmission plus the
  `log_peak_vs_predicted` cgroup check, a different criterion.
- **Not a prediction of the crash point.** An earlier draft of this section claimed 90,283 lands
  "within ~1 % of the crash's first expansion". That comparison is void: it sets a memory-derived
  *cap* against a CIPSI *expansion size*. The two pinned-cap runs below reach the identical
  88,164-determinant first expansion at two caps 6 % apart, neither of which bound — so within
  that narrow range the expansion size is invariant to the cap, which is as much as those two
  runs can show (the ladder run at 64,000 obviously never reached 88,164, so nothing here claims
  cap-independence in general). The category error voids the comparison on its own; no stronger
  claim is needed. Both pinned runs also *survived* 88,164 and died at 517,759, 5.87x higher.
- **Not a validation of the production budget.** This is a serial probe of a whole 15 GiB
  desktop, not 128 ranks inside a Slurm `--mem` cgroup. It validates the formula's *sensitivity*
  to `block_width`, nothing about the budget itself. Item 4 closes that.

(The campaign's headline 26x is a third quantity again — Phase 0's block-width under-prediction
ratio, 105/4 — not the 10.7x cap ratio here. Keep them apart.)

## Item 3: the local SMO DC search completes

Three `mpiexec -n 6` runs of `fixed_gap_dc` (`offset=0.0`, `allow_charge_state_change=True`,
`verbosity=1`) against `impmod_tests/SMO/cubic/impmod/impurityModel_data.h5`, loaded directly
via `model.load_selfenergy_archive` — the archived Hamiltonian the production run died on,
without re-running the multi-hour RSPt pipeline in front of it. Cross-check that this is the
same workload: the first expansion reaches 88,164 determinants here against the crash log's
88,298, and `n_spin_orbitals=58` matches.

Peak RSS below is from an external watcher sampling `/proc` every 5 s across all six ranks, not
from `log_peak_vs_predicted` — coarse, and the peak of a run the same watcher then terminated.

| run | cap | outcome | peak RSS (all 6 ranks / max rank) |
|---|---|---|---|
| ladder (`truncation_threshold=None`, Phase 5 live) | ladder **stopped** at its last rung, 64,000, without settling; the memory budget would have allowed 1,059,144 | **COMPLETED, 3290.3 s**, `sector=4`, 3 evaluations | ~2 GiB / ~0.5 GiB |
| no ladder, cap pinned to the memory probe's own answer | 1,067,592 | killed by the host's memory guard, inside the *second* expansion | ~10.6 GiB / 1.9 GiB |
| no ladder + `GS_MAX_BLOCK_WIDTH=8` | 1,132,656 | killed, same place | ~11.5 GiB / 2.2 GiB |

**Item 3 passes: the search completes where it previously died in the first evaluation.** The
lever that does it is the cap ladder — at 64,000 determinants the whole search fits in ~2 GiB
across six ranks and finishes in 55 minutes.

**What the two pinned-cap runs establish, and what they do not.** Both grew the basis along the
identical path — 88,164 determinants at the first expansion, 517,759 at the second, the same
numbers in both logs — and both died in the second. The two runs differ in exactly two settings
(the cap, and `GS_MAX_BLOCK_WIDTH`), and the growth path was **insensitive to both**. That is
the whole of it. Two earlier drafts of this sentence each claimed more than the data carry —
first that growth is "cap-driven" (refuted: neither cap bound), then that it is "not
width-driven" (also unsupported: that needs a premise about the width, and `p` was never
recorded — see the next bullet but one). Insensitivity to the two variables that differed is
what was measured; which variable *does* drive the growth is not established here.

The two runs are **not** a controlled A/B of `GS_MAX_BLOCK_WIDTH`, and the ~1 GiB peak difference
between them is not evidence the knob did anything:

- The **cap** difference is explained; the **peak** difference is not. These two experiment
  scripts sized their own cap with a direct `suggest_truncation_threshold(n_spin_orbitals,
  comm=comm)` call, passing no `block_width`, so they took that function's own `block_width=4`
  default. (That is a property of *these scripts*, not of the code path production uses:
  `dc_criteria._prepare_sector_context`, `fixed_occupation_dc` and `groundstate.py` all call
  `resolve_gs_block_width()` and pass the result, so in production the knob *does* move the
  memory-derived cap. The ladder run in row 1 went through that production path, where
  `resolve_gs_block_width()` returned 4 because the knob was unset.) The `block_width=4` premise
  is checkable from the recorded caps themselves: inverting `_suggest_for_budget` at 6 ranks and
  `safety 0.5`, cap 1,132,656 implies node `MemAvailable` of 11.16 GiB at `bw=4` and **15.49 GiB
  at `bw=8` — impossible on a 15 GiB machine**. So the `=8` run's cap was sized at width 4, and
  the two runs' caps imply 10.52 and 11.16 GiB of node `MemAvailable`: a quieter machine for the
  second, which is the whole of the 6 % difference. Note this is inverted from the caps, not read
  from a recorded probe — neither run logged `MemAvailable` itself.

  The peak RSS difference is a different matter: neither run reached its cap, so the cap cannot
  have sized their peaks either, and nothing recorded explains the gap. An earlier draft
  attributed it to the cap, then to sampling coarseness — neither better than the other. It has
  no measured cause; that is the entry.
- `p` was never recorded: `solver_trace.tracing()` was not open in either run, and TRLM's
  `rank k_ret/nkeep` lines cannot stand in for it. `nkeep = k_blocks * p` with
  `k_blocks = ceil(num_wanted / p)`, so `nkeep` tracks `num_wanted` — precisely the quantity
  Phase 4 deliberately leaves uncapped. The two runs' retained-block ranks are near-identical
  (20/20 x31, 28/28 x13, 21/21 x3 in both; 33/33 vs 36/36 the only difference), which is what
  Phase 4 predicts either way and therefore says nothing about whether the warm block was
  truncated.

So `GS_MAX_BLOCK_WIDTH`'s effect on this workload is **unmeasured**. What is measured is that
setting it to 8 does not rescue a pinned ~1.1M cap.

## Why the ladder stopped at 64,000: the rung budget, not memory

Worth stating plainly, because it was initially mis-diagnosed here as a memory limit and that
mis-diagnosis pointed at the wrong fix. `CAP_LADDER_START = 500` and `CAP_LADDER_MAX_RUNGS = 8`
put the last rung at `500 * 2^7 = 64,000` **whenever `memory_cap >= 64,000`** — which it was, by
16x (`dc_cap_parity = 1,059,144`). Memory never entered into it. The ladder ran out of rungs.

That refutes the width-cap hypothesis this document carried in its first draft ("a rung above
64,000 is unaffordable at uncapped width, so pinning `GS_MAX_BLOCK_WIDTH` might let the ladder
reach a settling cap"). Pinning the width cannot move a bound that is `2^rungs * start`; only
raising `CAP_LADDER_MAX_RUNGS` or `CAP_LADDER_START` can.

**The affordability half of that hypothesis is undetermined**, and three drafts of this paragraph
have now each asserted an answer the numbers do not support — first "rung 9 is unaffordable at
uncapped width", then "rung 9 is affordable at any width, no cap needed". Neither is established.
What can actually be said:

Back-solving this run's budget from the cap it reported (`dc_cap_parity = 1,059,144` at
`block_width=4`, 6 ranks) gives ~0.870 GiB/rank. At that budget, `_suggest_for_budget` says rung 9
— 128,000 — is affordable up to `block_width` **80** (bw 80 → 128,280; bw 81 → 126,798). Two
things stop that 80 from settling the question:

- **The threshold moves 5x with an assumption the estimator makes about a quantity Phase 4
  deliberately left uncapped.** With `gs_num_wanted=None` the model assumes `num_wanted = 2 *
  block_width`; `memory_estimate`'s own docstring says that under-counts by up to ~30x at
  production scale, precisely because `GS_MAX_BLOCK_WIDTH` caps the width and *not* `num_wanted`.
  Measured at the same budget: the largest width affording rung 9 is 80 at `num_wanted = 2p`,
  58 at 5p, 40 at 10p, and **17 at 30p** — below the `p = 16` already measured at cap 2,000.
- **`p(128,000)` is unrecorded, and cannot be interpolated.** ~~Log-interpolating the two
  recorded points — `p` maxing at 16 at cap 2,000, and ~105 at the ~1M production cap — gives
  `p(128,000) ≈ 56`; Measurement 1's other production figure (`k_ret` up to 315) interpolates to
  ~118.~~ **Withdrawn** — see "Phase 1b / Side-finding" at the end of this document: `p`'s
  maximum is *still* 16 at cap 8,000, so the curve those two points were fitted through is not
  smooth and neither number means anything. All that stands is that `p(128,000)` is unmeasured
  and the threshold it has to clear is somewhere between 17 and 80 depending on `num_wanted`.

So rung 9 may or may not need a width cap; deciding it needs `p` and `num_wanted` measured at
that cap, not another estimator call. (The often-quoted 90,283 is at item 1's 4.750 GiB/rank
pinned budget, a different budget entirely — at this run's 0.870 GiB/rank the production-width
figure is 99,216. Don't read the two as one estimate at two rank counts.)

**A caution that applies to every `_suggest_for_budget` number in this document.** The one place
in this campaign where the estimator meets a realized determinant count and a measured RSS, it
under-predicts badly: `estimate_gs_peak_bytes(517_759, 58, block_width=4, ranks=6)` gives
435.5 MiB/rank against the 1.9 GiB max-rank peak both pinned runs reached at that basis size —
4.5x low, and the runs were *killed*, so the true peak was higher still. That is not a refutation
of the model (feeding it `block_width=4` when the real `p` is much larger is exactly the campaign's
diagnosis), but it does mean an estimator call at an assumed width is a hypothesis, not a budget.

Corollary, and the reason this belongs in the record rather than a footnote: **"survivability:
fixed" is currently contingent on a hard-coded rung budget happening to land at a size this
machine can afford**, not on a measured budget. The grid is geometric and coarse — a factor of
two between neighbouring rungs, except for the last one when `memory_cap` binds. (It is *not*
restricted to powers of two: the loop steps `cap = min(2 * cap, memory_cap)`, so a binding
`memory_cap` always gets its own final rung, at whatever ratio that lands on. An earlier draft
claimed a machine that cannot fit 64,000 would fall back to 32,000; it would in fact evaluate its
own `memory_cap` as the last rung.) What the code genuinely cannot do is tell
"stopped because the answer settled" from "stopped because it ran out of rungs" — nothing marks
the difference except the warning below.

## The convergence criterion is *not* met, even though the run completes

The ladder run finished, but it exhausted all eight rungs without the answer settling, and said
so (verbatim, including the final sentence naming its own remedy):

```
WARNING: the determinant cap ladder reached 64000 without the answer settling (it still varies
by 1.060e-01 over the last 3 rungs, against a target of 6.250e-04). The double counting is
truncation-limited here, not search-limited. The reported dc inherits that drift, and a cap
ladder (test/support/dc_diagnostics.py) is the only honest error bar.
```

| quantity | value |
|---|---|
| `dc_cap` | 64,000 (`dc_cap_parity` = 1,059,144, the budget it stayed under) |
| `dc_cap_drift` | 1.06e-01 |
| ladder target (`0.25 * tol`, with `tol` = 2.5e-03) | 6.25e-04 |
| `delta_sum` (the measured slope `mu_tol_effective` divides by, as `delta_sum/2`) | 0.4441 |
| `mu_tol_effective` = `tol / (delta_sum/2)` | 1.13e-02 |
| truncation's own contribution, `drift / (delta_sum/2)` | 4.77e-01 |
| `mu`, `gap_center`, `chi` | -0.0230, 1.16e-04, -0.2213 |

(`dc_record` annotates this field with `0.5*delta_sum` when `delta_sum` is present and `chi`
otherwise — two estimators with deliberately different error structure. The 4.77e-01 above is
the `delta_sum` one; `drift/|chi|` would read 4.79e-01. Both are in the table so the reader can
check which.)

The plan's success criterion is "gap centre in `mu` stable across the cap ladder to within the
criterion's own `tol/|chi|`". **It is not satisfied**: truncation moves the answer 42x more than
the search tolerance does (`drift/tol`; note this is *not* the 170 you get against the ladder's
own `0.25*tol` target — the two ratios answer different questions and the campaign notes have
conflated them once already).

This is the same behaviour Phase 0 saw and flagged (`mu` moved ~0.07 between caps 500 and 2000,
against a ~5e-3 resolution) — SMO had not settled at cap 2,000 and it still has not at 64,000.
Understating it further would be easy: `mu` has not merely moved but **changed sign** since Phase
0, 0.139038 at cap 2,000 against -0.0230 here, a swing of ~0.16. A negative returned `mu` is
also the region the campaign plan flags as its out-of-scope-but-suspect sector question (nominal
d³ against an achieved d⁴/d⁵, with `_find_nominal_sector_point` walking `mu` negative while the
gap root was expected positive). Whether these are the same problem is not established here.

### Against the plan's three success criteria

| criterion | verdict |
|---|---|
| SMO DC search completes **on both machines** | **half done** — passes locally (item 3), Arrhenius untested (item 4) |
| per-rank peak RSS within the predicted budget | **cannot be adjudicated yet** — see below |
| gap centre in `mu` stable to within `tol/\|chi\|` | **fails**, by 42x |

(The first row is graded the same way item 1 is above: the plan says "on both machines", so one
machine is half, not a pass.)

The middle row was omitted from an earlier draft of this section, which is a worse error than
getting it wrong, since the runs did record RSS. The model, for reference:
`estimate_gs_peak_bytes(64_000, 58, ranks=6)` predicts **53.8 MiB/rank** at `block_width=4`,
**116.3 MiB** at 16, **194.5 MiB** at 32 and 574.7 MiB at 105 (that last is the ~1M-cap width and
has no business being applied to a 64,000-determinant solve; an earlier draft anchored the
bracket there and read the measurement as sitting "at the top" of it, which was wrong twice over).

Against that, the two recorded figures are ~2 GiB across six ranks and ~0.5 GiB on the max rank —
and **they do not leave room for a useful subtraction.** Six ranks are six Python processes, so
Phase 0's serial floor of 335.5 MiB/rank is already 1.97 GiB of the ~2 GiB total on its own. A
second draft tried netting that floor off the max rank and quoting "roughly 90-210 MiB" of solve;
the arithmetic is 512 − 335.5 = 176.5 and 512 − 422.1 = 89.9, so the range was mis-stated (210 has
no derivation), and calling 335-420 MiB "the Python/import floor" also mislabels Measurement 4,
whose own text attributes that gap to the floor **plus** the 18x `nnz_per_state` over-prediction
**plus** cap-dependent transients — not a constant, and 422.1 MiB is itself a cap-2,000 peak, so
using it as the floor's ceiling double-counts growth with cap.

**Verdict: cannot be adjudicated, and this measurement cannot be rescued by arithmetic.** The
totals are too coarse and the floor too poorly separated to place the solve's own share anywhere
useful; `p` at cap 64,000 was never recorded either. What this row needs is
`log_peak_vs_predicted` at a run with `solver_trace` open — the hook Phase 0 item 4 named for
exactly this — not another subtraction by eye.

Read together, that splits the campaign's outcome:

- **Survivability: fixed**, subject to the rung-budget caveat above. The run no longer dies; it
  produces a DC value in 55 minutes where it previously OOM'd in the first evaluation.
- **Accuracy at the cap it can afford: not established.** The returned DC inherits a 4.77e-01
  truncation drift in `mu`, and the code says so unprompted rather than hiding it — which is the
  behaviour the Phase 5 warning was written for, working as intended on the first real workload
  it met.

**What would actually resolve this**, since item 2 on its own will not: it varies caps only
(2,000/8,000/32,000), and every one of those is a rung the ladder already ran and already
reported as unsettled, so it cannot arbitrate between the deferred post-search cap
re-verification and a width cap. What it *does* deliver is the cost scaling, the per-kind
timings, and an independent `mu`-drift measurement (below). It does **not** deliver `p(cap)`,
the quantity missing from every measurement above: `run_dc_search`'s returned row carries
per-kind seconds and counts but no block widths, so that needs its own probe. Deciding the width
question needs its own A/B at one pinned cap with `solver_trace` open, at a cap large enough for
`GS_MAX_BLOCK_WIDTH` to bind (at cap 2,000, Phase 0 measured `p` maxing at 16, so a cap of 8
barely binds).

Deciding the accuracy question needs a rung above 64,000, and the cheapest experiment that could
is **raising `CAP_LADDER_MAX_RUNGS` and re-running**. Whether that run also needs a width cap to
afford rung 9 is genuinely open (above); the experiment should measure `p` and `num_wanted` at
that cap rather than assume either answer.

No prediction is offered here about whether rung 9 would settle. An earlier draft argued "the
drift has not shrunk over the last three rungs" — that claim has no source: `dc_cap_drift` is a
*single* span over the trailing window, and `_calibrate_cap` discards the ladder's `rungs` list
(`cap, cap_drift, _rungs = calibrate_truncation_threshold(...)`), so no per-rung history reaches
`dc_record` and no trend is recoverable after the fact. Plumbing `rungs` through would make that
question answerable from a run that already happened, which is probably worth doing before
spending another ladder on it.

## Item 2: the `RUN_DC_DIAG` cap ladder

`RUN_DC_DIAG=1 DC_DIAG_CRITERION=gap DC_DIAG_WORKLOAD=smo DC_DIAG_CAPS=2000,8000,32000` under
`mpiexec -n 6`, `OPENBLAS_NUM_THREADS=1`, 28 min total. The harness pins `iteration=1`; the SMO
archive holds only one iteration (the cluster group carries the datasets directly, no iteration
subgroups), so this is the same data the item-3 runs loaded through the loader's `last` default
— the confound `archives-differ-on-whether-h0-contains-dc` warns about does not arise here.

```
      cap    seconds  evals  solves   hits     dets   build_s   expand_s   eigen_s         mu      value
     2000      178.4      6      51     47     2000       1.4      162.1      14.2   0.141796   -0.00207
     8000      354.9      4      35     31     8000       1.5      324.6      27.6   0.208290   -0.00106
    32000     1168.8      3      26     21    32000       2.3     1050.4     112.7  -0.046748    0.00034
achieved value across the ladder: spread 0.0024 (STABLE)
mu across the ladder: spread 0.255037
scaling: seconds ~ cap**0.68
production cap on this machine: 1085040 determinants -> projected 12742 s (3.5 h) per DC search
chi = d(value)/dmu per cap: -1.2797, -0.4730, -0.2384
```

### The `STABLE` verdict was measuring the wrong axis — since fixed

**The output above is the historical run, printed by the pre-fix harness.** The
`achieved value across the ladder: spread 0.0024 (STABLE)` line was graded on `value`, which for
`criterion="gap"` is **the gap centre** — the quantity the search itself drives to zero at every
cap. Its spread is therefore bounded by the search tolerance, not by truncation, and a "STABLE"
verdict was close to vacuous for this criterion. The campaign plan says exactly this ("judge
every later saving on this [the centre converted to `mu`], never on one sector's `e0`"), and
`dc-perf-campaign-measured-levers` records it as a lesson already learned once — the same failure
class as `gf-monitor-was-converging-the-wrong-axis`.

**Fixed.** `print_ladder` now grades `mu` against the criterion's own resolution
(`_mu_verdict`/`_row_resolution`), prints the per-cap resolutions it used, and prints the value
spread unlabelled as a verdict. Re-running the ladder above today would report `DRIFTS`. The
"24x" in the table below is on the `tol/|chi|` estimator, which is *not* what the fixed code
grades a gap ladder against — it prefers the criterion's own `mu_tol_effective`, and the two
disagree by a factor 3.7 on a measured SMO rung (5.19e-03 against 1.41e-03 at cap 500). Kept
here as the run that motivated the fix, not as current behaviour.

On the right axis the ladder is emphatically not stable:

| cap | `mu` | `chi` | resolution `tol/\|chi\|` |
|---|---|---|---|
| 2,000 | 0.141796 | -1.2797 | 1.95e-03 |
| 8,000 | 0.208290 | -0.4730 | 5.29e-03 |
| 32,000 | -0.046748 | -0.2384 | 1.05e-02 |

`mu` spans **0.255**, between 24x (against the loosest per-cap resolution) and 130x (against the
tightest) the band the criterion claims to deliver. It also changes sign between 8,000 and
32,000, independently reproducing the sign change the item-3 ladder run showed at 64,000 — on a
different code path, at a pinned iteration, with the cap ladder bypassed. The non-settling is
not an artifact of the Phase 5 ladder.

`chi` collapsing by 5.4x across the ladder is the mechanism, and it is worth stating separately
because it makes the two axes move in opposite directions: `mu` is recovered from the returned
`dc`, and the centre is driven to zero, so a nearly-constant `value` divided by a collapsing
slope produces a wandering `mu`. It also means the criterion's own resolution *degrades* with
cap (1.95e-03 → 1.05e-02): spending more determinants buys a looser bound, not a tighter one.

### Cost: the scaling exponent is not the whole story

`seconds ~ cap**0.68` is a real improvement on the `cap**0.98` this campaign started from
(`smo-gap-dc-is-the-real-bottleneck`) — but most of it is an artifact of the evaluation count
falling with cap (6 → 4 → 3), not of any evaluation getting cheaper. Per evaluation:

| cap | seconds/evaluation |
|---|---|
| 2,000 | 29.7 |
| 8,000 | 88.7 |
| 32,000 | 389.6 |

which fits `cap**0.93` — essentially the original 0.98 within the noise of three points. The
harness's own projection to this machine's production cap (1,085,040 determinants → 3.5 h)
inherits the 0.68 and is correspondingly optimistic; at 0.93 per evaluation and three
evaluations it is **~8.5 h**, against the campaign's opening estimate of ~12 h. Quote the
per-evaluation number when projecting, since nothing guarantees a harder DC surface will keep
converging in three evaluations.

`expand` is 90-91 % of wall-clock at every cap, `eigensolve` 8-10 %, `build` under 1 %. Note
these are *siblings* under `sector_solve` (`groundstate.py` times `solver.expand(...)` and
`solver.get_eigenvectors(...)` as two consecutive blocks), so they do not overlap — but
`expand`'s internal CIPSI iterations call `get_eigenvectors` themselves and those calls are
**not** separately timed, so the Lanczos work is spread across both columns and `eigensolve_s`
is only the final solve. Do not read 8-10 % as the eigensolver's share of the run.

### Against Phase 0's baseline

At cap 2,000 Phase 0 recorded `mu = 0.139038`, `gap_center = -0.000962`; this run gives
`0.141796` / `-0.00207`. The `mu` difference is 2.8e-03, at the edge of that cap's own 1.95e-03
resolution — **Phases 1-5 did not move the answer** at this cap, which is what the campaign
needed to show.

Wall-clock is **not** comparable between the two: Phase 0's 450.8 s at cap 2,000 was serial
(`comm=None`, as its header states) and this is `-n 6`. Reading the 2.5x as a campaign speedup
would be wrong; most or all of it is parallelism.

## Phase 1b: measured, and not worth doing

Phase 1's review deferred two changes to `_block_ops.pxi`'s `block_apply` (`Phase 1 follow-up`
in the campaign plan): hoisting the per-call re-derivation of an invariant (the `Allgather` of
the row partition, and the `size` fresh CSR row-slices, both identical on every call across a
whole TRLM restart loop), and replacing the `size` sequential blocking `Reduce` calls with a
depth-bounded `Ireduce` pipeline. Both were accepted-and-deferred on plausibility, never
measured. They are measured now.

### How this was measured, after a first attempt that was not a measurement

`trlm.py`'s two `block_apply` call sites are now wrapped in `solver_trace.timed("block_apply",
site=..., w=...)` rather than only noted, so a traced run reports the call's own wall time
alongside its width. That instrumentation exists because the first version of this section did
something else: it took per-call costs from a synthetic-CSR benchmark and folded them through the
recorded width histogram. That was wrong by 1.74x (1.38 ms/call synthetic against 2.40 ms/call
measured), and wrong in *two* directions at once with neither bound — the benchmark sizes every
call at the cap while the real calls run over CIPSI bases still growing toward it, and it used
Phase 0's `nnz_per_state = 5.6`, a cap-500 sample whose own Measurement 2 says to re-check it at
a larger cap before trusting it. Both flaws are avoidable by reading the clock at the call site,
which is what the trace was already open for.

### The per-call cost split, measured in situ

One full gap DC search at cap 8,000, `mpiexec -n 6`, `solver_trace` open. Three runs:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| walltime | 349.3 s | 352.2 s | 355.8 s |
| `block_apply`, rank 0 | 33.06 s (**9.46 %**) | 33.16 s (9.42 %) | 33.87 s (9.52 %) |
| calls | 13,802 | 13,802 | 13,802 |

`solver_trace` is rank-local, so those are **rank 0's** numbers, and `block_apply` ends in
`Allgather`/`Reduce` — one rank's total is its own work *plus* its wait for the slowest peer.
Run 3 gathered every rank to separate them:

```
per-rank totals (s): 33.87, 33.61, 23.75, 24.37, 24.17, 25.97
  max 33.87 s = 9.52 % of walltime      min 23.75 s = 6.68 %
  spread 10.12 s = 2.84 % of the search = 30 % of the max
```

**Nearly a third of `block_apply`'s apparent cost is collective wait**, and no amount of
pre-slicing removes wait. The honest work figure is the rank that waits least: **6.68 % of the
search**, not the 9.4-9.5 % a single-rank trace reports. (An earlier draft of this section quoted
the rank-0 number as if it were the workload; a review round then argued upward from it to
5-6 %. Both were reading the same confound.)

Widths: min 1, max 16, mean 3.23, all from `site=continuation` (the rebuild arm never fired at
this cap, as in Phase 0). Per-width mean cost on rank 0, and the call-count-weighted
least-squares split into a `w`-independent term and a `w`-proportional one:

```
w=1:1.665(n=4674)  w=2:1.951(n=1944)  w=3:2.337(n=3850)  w=4:2.647(n=160)   w=5:2.871(n=1184)
w=6:3.027(n=600)   w=7:4.240(n=582)   w=10:2.623(n=124)  w=11:4.639(n=380)  w=16:7.880(n=304)

cost(w) ~ 1.257 + 0.354*w  ms      (run 2; run 3 gives 1.310 + 0.354*w)
```

Only `w`-independent work is reachable: the CSR row-slice is `O(nnz)` while the matmul and
reduce are `O(nnz * w)`. The intercept is **52 % of mean per-call cost** — and the synthetic
benchmark's pre-slice-only column agrees closely with that structural share (46.8 % measured at
cap 8,000/`w=4` against the model's 47.0 %; 19.4 % against 18.2 % at `w=16`), which says
pre-slicing removes close to the whole intercept rather than part of it.

So the reachable saving is **~3 % of the DC search**: 52 % of the wait-free 6.68 %. Bounds on
that, since the intercept is a fitted quantity and the width distribution is very uneven
(74 % of calls at `w <= 3`):

| | share of the search |
|---|---|
| rank-0 intercept, weighted fit (what an earlier draft called a "hard ceiling") | 5.08 % |
| same, unweighted | 4.31 % |
| same, fit over `w <= 6` only | 5.44 % |
| `w=1`'s own measured mean — a real upper bound on `w`-independent work, rank 0 | 6.5 % |
| **net of collective wait (the min rank), which is the removable one** | **3.1-3.5 %** |

The "hard ceiling of 4.9 %" claimed earlier was neither hard nor a ceiling: two of the fits above
exceed it, and all of them sit on rank 0's wait-inflated total.

### The synthetic benchmark, kept for the one thing only it can answer

`bench_block_apply.py` (scratchpad), `mpiexec -n 6`, 100 reps per point, each variant a faithful
copy of the shipped code with exactly one thing changed and cross-checked against it for
agreement. Its *absolute* numbers are superseded by the in-situ measurement above; what it can
still answer is the relative ranking of variants, which the in-situ clock cannot. Each column is
that variant's own increment, expressed as a percentage of the shipped code's total, so the
columns add:

```
   cap    w | hoist Allgather  + preslice CSR  + Ireduce   hoist+preslice   pre-Phase-1 vs shipped
  8000    4 |            10.0            46.8      -31.8             56.8       29.3
  8000   16 |             4.2            19.4      -52.1             23.6      -38.7
  8000   64 |            -0.6            12.7      -57.1             12.2      -45.6
 32000    4 |            -3.6            37.4      -38.4             33.9       -8.2
 32000   16 |             1.3            14.8      -40.9             16.1      -14.6
 32000   64 |            -1.0             3.7      -21.9              2.7      -12.4
 64000    4 |            10.8            23.7      -37.3             34.5      -20.4
 64000   16 |            -3.2             6.5      -23.9              3.2       -4.1
 64000   64 |             1.2             1.6      -25.0              2.7      -22.8
```

- **The `Ireduce` pipeline is a pessimization at this rank count**, by 22-57 %, at every one of
  the nine points — not the "correct fix" Phase 1's review deferred it as. Caveat before calling
  it dead: the latency argument for it was always about large `size`, and 6 round trips is not
  128, so it deserves a high-rank-count check. But nothing should be implemented on the strength
  of the original reasoning.
- **Hoisting the `Allgather` is noise**, -3.6 % to +10.8 %, straddling zero.
- **Pre-slicing the CSR is the only real saving**, and it decays with `w` exactly as the in-situ
  intercept predicts it must, since it removes fixed work from a growing denominator.
- Incidentally: **Phase 1 was not a pure speed win, and the sign depends on the width.** The
  pre-Phase-1 `Allreduce` is 29 % faster at cap 8,000/`w=4` and 46 % slower at `w=64`. That is
  the memory-for-latency trade Phase 1 made deliberately, now measured rather than assumed.

### Verdict: do not do Phase 1b

**~3 % of a DC search**, once collective wait is netted out, does not justify a
correctness-sensitive change to a function three call sites share (`trlm.py`'s continuation and
rebuild arms, and `BiCGSTAB.pyx`), in a campaign whose own history includes a rank-local early
return in an extracted helper deadlocking a collective. Doing it properly means hoisting the
partition and the row-slices out to the caller and threading them through that shared signature,
or caching them inside `block_apply` on a key that correctly invalidates when `H` changes — the
second is the cache-staleness hazard this document already records twice. Half of the deferred
work is a measured pessimization anyway.

**The opportunity-cost argument, stated correctly this time.** An earlier draft wrote "`expand`
is 91 % of this search, so a 3-4 % lever is not where the next effort belongs" — which compares a
part against its own whole: `cipsi_solver.expand` calls `get_eigenvectors` internally, so almost
all of `block_apply`'s 33 s sits *inside* `expand`'s 322 s. Speeding up `block_apply` **is**
speeding up `expand`. The valid form of the claim is that `expand`'s **other** ~289 s — candidate
generation, PT2 scoring, the restriction bookkeeping — is 82 % of the search and untouched by
anything in Phase 1b. That is where a lever of any size has room to be found.

What would change this verdict: a rank count where the `size` sequential `Reduce`s actually hurt
(the 128-rank Arrhenius configuration is the case the deferral was written for, and is
untested), or a workload whose widths are large enough for `block_apply` to dominate — the
opposite of this one, where the mean width is 3.2 and the intercept carries half the cost.

**A better target than pre-slicing, visible in the same data.** The per-rank spread is 10.12 s —
30 % of `block_apply`'s cost on the slowest-observed rank, 2.8 % of the whole search — and it is
load imbalance across the row partition, not slicing. It is also larger than the ~3 % pre-slicing
could win. Nothing here investigates it; noted because the measurement fell out of the same run
and points at partitioning rather than at the per-call code path.

**Not measured, despite an earlier claim here that it was:** how the share moves with cap. The
in-situ probe was run at cap 8,000 only. An earlier draft asserted "at cap 32,000 the same
arithmetic lands near 3 %", then a correction said "flat near 5 %" — the first had no derivation,
and the second silently reused the withdrawn synthetic arithmetic's 5 % while the measured share
is 9.4 % (rank 0) or 6.7 % (wait-free). Both are withdrawn. The call count, width histogram and
wall time at cap 32,000 are not recorded anywhere; "flat" is what the verdict rests on and is
itself a reconstruction from item 2's per-evaluation scaling, not a measurement.

### Side-finding: `p`'s *maximum* did not grow between cap 2,000 and cap 8,000

`eigensolve_block_width` at cap 8,000: n=327 solves, `p` min 2, **max 16**, mean 5.06, with
`num_wanted` min 12, max 40, mean 18.39.

Phase 0 measured `p` max **16**, mean 4.7 at cap 2,000, and `block_apply` width mean 2.8. So
across this 4x range the *maximum* is pinned at 16 while the *means* grow modestly (4.7 → 5.06,
and 2.8 → 3.23). "`p` did not grow" is true only of the maximum, and the growth that is there is
far too slow to reach the `p ≈ 105` recorded at the ~1M production cap.

That makes `p` a poor candidate for interpolation across caps, and this document should stop
doing it in both directions. Withdrawn accordingly: the log-interpolated `p(128,000) ≈ 56` from
an earlier commit, and the attempt before that to assert `p` is nowhere near 80 — both were
reading a curve off two points that this measurement shows is not smooth. Also withdrawn: using
this cap's `p` as an anchor for the RSS row at cap 64,000, for the same reason.

**One correction that runs the other way from what an earlier draft claimed.** `num_wanted / p`
here is **3.63** (18.39 / 5.06), not the 2.2 an earlier version of this section stated with no
derivation. Against `memory_estimate._GS_COUPLED_NUM_WANTED_RATIO = 2` that is an *under*-count
in `estimate_gs_peak_bytes`'s Krylov term — the opposite of the "the estimator's `num_wanted`
assumption is sound" conclusion drawn from the mis-derived number.

Its size is **1.17-1.22x, not the ~1.8x a second draft claimed** by taking 3.63/2 directly.
`_gs_krylov_columns` does not scale linearly in the ratio: it computes
`nw = num_wanted + _GS_EIGENSTATE_PAD` (10) and `blocks = 2*ceil(max_subspace/p) + 20`, and both
the additive pad and the flat +20 dilute it. Evaluated at `n_dets = 8,000`: `p=4` gives 152
columns assumed against 184 measured (1.21x), `p=5` 180 vs 220 (1.22x), `p=6` 216 vs 252 (1.17x),
`p=16` 512 vs 608 (1.19x).

Two caveats on the 3.63 itself. It is a ratio of means; the estimator wants the mean of the
per-solve ratio, and both quantities are recorded per solve but only the means were kept. And it
is a cap-8,000 number, so it says nothing about `memory_estimate`'s own ~30x warning, which is
explicitly about production scale. The rung-9 width threshold moves *down* from 80 on this
correction, but by ~20 %, not by the factor a 1.8x under-count would imply.
