# Block Lanczos / TRLM / IRLM invariants

Landing place for the incident knowledge stripped out of inline comments during the
R2 comment-triage pass of the Lanczos/IRLM/TRLM redesign (see the plan at
`i-want-a-thorough-agile-fern.md`). The source still carries a one-line pointer at each
site this document expands on; this is where the "why", the numbers and the history
live. Covers `BlockLanczos.pyx`, `_lanczos_step.pxi`, `_trlm.pxi`, `_irlm.pxi`,
`BlockLanczosArray.pyx`, `_reort.pxi`.

## Representation dispatch

Every block-Krylov primitive (`block_apply`/`block_combine`/`block_inner`/
`block_orthogonalize`/`block_normalize`/`block_tsqr`, all in `_reort.pxi`) dispatches on
the block's representation: a dense `(N, p)` ndarray (the array kernel), a single
shared-support `ManyBodyState` (the current block representation — a block of `p`
many-body vectors stored once over their union support, so the matvec/Gram/axpy
primitives run once per determinant *row* instead of once per `(determinant, vector)`
pair), or (only at IRLM/TRLM boundaries that predate the block-native restart
bookkeeping) a bare `list[ManyBodyState]`.

**Rule: get the block branch right BEFORE the list branch, in every dispatcher.** A
`ManyBodyState`'s own `len()` is its row count (dict-like, matching `len(dict)`), not
its column/width count — silently wrong for "how many Krylov columns are here", not a
raise. `_q_cols` (`_irlm.pxi`) is the canonical column-count helper that gets this
right for array / `ManyBodyState` / plain list uniformly; every restart-loop width
computation goes through it rather than a bare `len()`.

The bare-`list[ManyBodyState]` branches in `_reort.pxi`'s dispatchers
(`block_add_scaled`, `block_combine`, `block_orthogonalize`) are now defensive-only
`raise TypeError`s: every production caller converts to a block at its entry point
before ever reaching them (TRLM/IRLM's `psi0` seed, `BiCGSTAB.pyx`/`GMRES.pyx`). They
are not dead code to delete — a future caller that reintroduces a bare list would hit
them immediately instead of silently misbehaving.

## MPI collective gating

**The rule, everywhere in this stack:** an MPI collective must never be gated on a
rank-local read of a `ManyBodyState`/`ManyBodyOperator`-derived value; gate only on
values that are provably rank-invariant (a global column count, a value already
broadcast), or `bcast` the decision first.

**The `-n 3` deadlock this rule exists because of** (confirmed via
`test_block_lanczos_mbs_empty_rank.py`): `block_lanczos_step_cy`'s locking-deflation
branch used to gate on `if locked` — a `ManyBodyState` defines no `__bool__`, so
truthiness falls back to `len()`, which is its LOCAL ROW count. An empty rank (owns
zero determinants under the hash-distribution) can have `width > 0` but `rows == 0`, so
`if locked` reads `False` there while every other rank reads `True` — and the two
`Allreduce`s inside the branch then execute on some ranks but not others. Fixed by
gating on `_q_cols(locked) > 0` instead, which is the rank-invariant column count,
replicated identically everywhere.

**Defense-in-depth pattern** (`_irlm.pxi`, the `res_width < p` early-stop before
`purge_restart`): `total`/`res_width` are rank-invariant *by construction* — they
descend from `tsqr`'s globally-reduced `active_k` (an `Allgather` across ranks in
`TSQR.pyx`) and from `_q_cols`/`SparseKrylovDense.n_cols`, which an empty-local-partition
rank still reports in full. The `comm.bcast(take_break, root=0)` guarding the early
`break` is not needed for correctness today — it is there so that *if* this invariant
were ever violated by a future change, the two branches' next collectives mismatch (a
1x1 `Allreduce` in the early-stop path vs. an `Allgather` in the purge/restart path
below) and the run deadlocks loudly instead of silently corrupting the restart. Mirrors
`_trlm_core`'s `done = comm.bcast(done, root=0)` for the same reason.

## Deflation vs breakdown scales

Two different questions, deliberately kept apart throughout this stack (see
`_cholesky_or_deflate`'s contract, `BlockLanczosArray.pyx`, and `TSQR.tsqr` which
inherited it):

- **Breakdown** — is the block numerically zero, i.e. is the block-Krylov space closed?
  An absolute statement needing a reference: a block is zero relative to *something*.
  `k = 0` when `‖β‖₂ = √λmax(M) ≤ BREAKDOWN_TOL * scale`. Callers that normalize
  near-unit-norm vectors leave `scale = 1`; the Lanczos sweeps pass `~‖H‖`, because a
  *residual* block is zero when negligible against `H`, not against `1`.
- **Rank deficiency** — are some column *directions* linearly dependent? A statement
  about `λmin/λmax`, judged relative to the block's own largest singular value: a
  direction deflates when `σ_k < EPS^(1/3) σmax`, i.e. `λ_k < EPS^(2/3) λmax`. This
  bounds the retained block's condition number to `≲ EPS^(-1/3)` (~1.7e5), comfortably
  inside the regime where a CholeskyQR2 second pass restores orthonormality to machine
  precision (needs `κ ≲ EPS^(-1/2)`), and is what stops the `O(κ)` amplification of
  per-step `Allreduce` rank-order rounding from diverging the `reort=NONE` recurrence.

**History: the two used to be fused.** `evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)`
had a `1.0` clamp that turned the *rank* test absolute for any block below
`DEFLATE_TOL = EPS**(1/3)` (~6.06e-6): a small but perfectly well-conditioned block was
declared rank 0. Every warm-started Krylov solve was handed back its own input and
reported success — `block_bicgstab` returned `x0` unrefined once `‖R0‖ < 6e-6`, and
TRLM warm-started from `CIPSISolver.expand`'s eigenvectors returned them unimproved at
`‖r‖ = 2.2e-9`, capping ground-state accuracy whatever `tol` asked for. `BREAKDOWN_TOL`
existed but was referenced nowhere; invariant-subspace detection rode entirely on the
clamp.

**`f_plus`'s breakdown reference must be `tnorm`, not `1.0`** (`_irlm.pxi`, the residual
block after re-banding): `f_plus` is a *residual* block — `O(‖H‖)`, not `O(1)` — so its
breakdown reference must be the operator norm, like the two Lanczos sweeps and unlike
`block_normalize`. `tnorm` (the largest-magnitude Ritz value including locked ones) is
the proxy already in hand. This is a consistency fix, not a live bug today: eq. (15)
locks a Ritz pair as soon as `res ≤ u*tnorm + |cntl2|`, so a residual block cannot
survive to this branch once it has shrunk that far, and an anisotropic one is caught by
the relative rank test first — instrumented over the restart/Lanczos/CIPSI suite plus a
warm-start probe: 734 hits, 0 decisions changed, closest approach 1127x above the
branch. Kept anyway: an isotropic residual block numerically zero against `‖H‖` would
otherwise deflate to nothing yet still be normalized, amplifying its rounding noise by
`‖H‖/eps`.

**Why `res_width < p` must be tested BEFORE `purge_restart`** (`_irlm.pxi`): the
trailing residual block can be rank-deficient when the sweep reaches a (near-)invariant
subspace without shrinking a *diagonal* block, so the alpha-width guard doesn't fire.
The extreme form is an exact breakdown (`res_width == 0`, `beta_last` exactly zero):
every Ritz residual is then zero, eq. (15) accepts every active pair into the locked
set, and if the caller asked for more pairs than the space holds (`num_wanted > dim`,
which `cipsi_solver` does routinely — it caps `num_wanted` against `len(basis)`, not
against the reachable Krylov space) `n_need` stays positive with no unlocked candidate
left. `select_restart_indices` then legitimately returns an empty `kept_idx`, and
`purge_restart` — which needs at least one full block of retained pairs — used to die
on an empty `np.concatenate`. The guard now runs first and locks-and-stops cleanly.

## PRO estimator honesty

`estimate_orthonormality` (`BlockLanczosArray.pyx`) tracks the Paige-Simon partial
reorthogonalization (PRO) estimate `W` through the signed three-term recurrence, plus
two additive terms: a rounding-injection magnitude and a local-rounding noise floor.
The full derivation, with the exact measured numbers, is kept verbatim in that
function's comments (`BlockLanczosArray.pyx:266-395`) rather than duplicated here —
read it there when touching the estimator. Three incidents worth knowing before you do:

1. **The magnitude-over-signed-propagation trap.** The three-term propagation term is
   *signed* on purpose: the `O(‖β‖)` structural terms cancel to `O(eps‖β‖)` exactly as
   the true overlaps do, and that cancellation IS the physics. A magnitude version (sum
   of `|terms|`, tried as an "upper bound") destroys it — the estimate jumps to
   `O(‖β‖/σmin) ~ O(1)` after one step and compounds exponentially: measured
   **1e15–1e62x over-prediction** on the NiO ground state, flagging every block on every
   iteration and silently degrading PARTIAL to FULL's cost.
2. **The noise-floor sign/scale trap.** An earlier floor `eps*(β_i + β_j)` omitted the
   `1/σmin(β_i)` factor and *shrank* with `β_i` — exactly backwards, since a small `β_i`
   (near-invariant block) is where the new vector is most rounding-dominated. The floor
   vanished exactly when the true loss was worst, the bad-block trigger never fired, and
   PARTIAL degenerated to no reorthogonalization at all.
3. **The `omega_{i+1,i}` seed / warm-start trap.** The seed term must bound `‖H‖`, not
   `‖β_0‖` — they coincide only for a cold start (`‖H q_0‖ ~ ‖H‖` for a random `q_0`).
   Warm-started from converged eigenvectors, `‖β_0‖` is the eigenpair residual (measured
   2.2e-9 on the NiO ground state against `‖H‖ ~ 1.1e2`); at `i=0` the old expression
   collapsed to `β_0^{-H} @ β_0 ~ I`, estimating `eps` where the truth is
   `eps*‖H‖/‖β_0‖ ~ 1e-5`. The trigger never fired, PARTIAL silently did nothing, and
   the recurrence diverged ~30 steps later (measured `‖Q^H Q - I‖ = 11.3`, `‖β‖` 8x
   FULL's). Fixed by seeding from `max(‖α_0‖, ‖β_0‖)` directly.

**The "HONEST reset" fix** (`apply_reort`, `_reort.pxi`): after a bad-block
reorthogonalization pass, the post-reort `W` entry used to be written as `EPS` — a lie.
Against a Krylov set whose own mutual orthogonality has degraded to `delta`, CGS2
leaves a residual `~ delta * |overlap| >> EPS`. Writing `EPS` blinded the estimator
right when it mattered (both live `W` rows get chopped on consecutive acted steps), and
the true loss regrew geometrically from the un-modeled residual until the recurrence
diverged (measured: estimate `1e-9` while the true overlap was `1e-2`). Fixed by
writing the final CGS pass's measured (Allreduced) overlap `O_last` instead — a
measured, conservative bound on the residual that is `~EPS`-scale in the healthy
regime, so the trigger cadence there is unchanged.

## Warm-start / resume protocol

`block_lanczos_cy` resumes from block `k0 = len(alphas_init)` when `alphas_init`,
`betas_init` and `Q_init` are all given together (`None` for all three starts fresh —
partial combinations are a caller error). The warm-start `Q` blocks are sliced out of
`Q_init` and the existing W-estimator (`W_init`) is reused as-is. If `W_init` is `None`
but `reort` is `'partial'`/`'selective'`, `W` is *exactly* initialized instead of left to
grow from an empty estimate: the exact overlaps of the starting blocks against every
prior block are computed once (Exact Overlap Restart, EOR) rather than trusting the
Paige-Simon recurrence to reconstruct history it never saw — the same "estimate from
nothing" gap that under-predicted on a cold start would otherwise repeat on every resume.

`alphas_buf`/`betas_buf` are allocated once, sized `(k0 + max_iter, p, p)`, before the
loop starts; the returned `alphas`/`betas` are *views* into these buffers
(`alphas_buf[:it_abs + 1]`), not copies, so no per-iteration `np.array()` rebuild sits
on the hot path.

## Krylov store

`SparseKrylovDense` (`_krylov_store.pxi`, out of scope for this redesign — only its
call sites are touched) is the columnar dense-over-support Krylov basis store used
whenever `store_krylov=True`. It doubles as the dense reorthogonalization mirror for
every reort mode: `apply_reort` slices its columns via `store.reort()` rather than
gathering `Q[:, cols]` into a fresh dense buffer. Its `reort()` method is **collective**
(an unconditional per-pass `Allreduce` on the small `(n_sel, p)` overlap matrix `O`) —
`n_rows` is not rank-identical (an empty-partition rank contributes zero-row gemms) but
every rank must still call it. Covered under MPI (`-n 2`/`-n 3`, including the
empty-rank case) by `test_krylov_store.py::test_store_reort_mpi_matches_serial_and_handles_empty_rank`
(added in R0c).

**`store_krylov=False`** (requires `reort == 'none'` and no `locked` set) skips this
store entirely: the accumulated basis is not retained, and the returned `Q_basis` holds
only the last two blocks (the previous block plus the residual) — exactly the tail the
warm-start protocol above needs to resume. `alphas`/`betas` are bit-identical to a
`store_krylov=True` run of the same problem; only the `O(N_det * p * k)` dead retention
is dropped. On resume with `store_krylov=False`, `Q_init` is interpreted as that
two-block tail, split by `block_widths_init[-1]`.

## Thick-restart coefficient validity

Both TRLM restart-coefficient shortcuts (`_trlm_core`, `BlockLanczosArray.pyx`) require
`Q^H Q = I` on the retained Ritz block, and cost nothing when it holds:

- **Healthy case** (`‖Q^H Q - I‖ ≤ RESTART_ORTH_TOL` at full rank): the textbook
  coefficients follow from the recurrence identity
  `H Q = Q T + q_m β_res E_last^H` *together with* `Q^H Q = I` — the retained block's
  projected operator is `diag(theta_keep)`, the spike is `β_res @ Y_last`, and the
  carried-over residual block `q_m` is already the whole residual (rank `≤ p`).
- **Otherwise**: the retained block lost semi-orthogonality (and possibly rank).
  `Q^H Q ≠ I`, so *neither* textbook coefficient is valid — both derive from it — and
  rescaling them by the orthonormalizing factor only amplifies the error by
  `1/σmin`. The fallback is an explicit Rayleigh-Ritz step on the orthonormalized
  retained basis (`T_lead = Q_ret^H H Q_ret`, `q_m` the orthonormalized residual,
  `cross = q_m^H H Q_ret`), which assumes nothing about `Q_basis`, at the cost of
  `k_ret` matvecs. `reort=NONE` needs this routinely (measured `‖Q^H Q - I‖ = 1.0` on a
  spectrum with a 1e-9 cluster); the semi-orthogonal modes rarely take it.

Without `Q^H Q = I`, the residual is no longer rank `≤ p`, so `q_m` can be up to
`k_ret` wide — `T_full` is sized off `p_resid` (the actual residual width), never off
the constant `p`, or the deflating branch overruns `T_full`.

## complex64 measurements

`krylov_dtype='complex64'` is rejected outright (not just warned) when combined with
`reort='partial'/'selective'`. The measurements behind that:

| Quantity | complex128 | complex64 |
|---|---|---|
| FULL reort steady-state `‖Q^H Q - I‖` | 1.1e-15 | 6.0e-8 |
| PARTIAL/SELECTIVE trigger target (`REORT_TOL = √EPS`) | ~1.5e-8 | unreachable (4x tighter than the store's own rounding floor) |
| Bad-block selection floor (`BAD_BLOCK_TOL = EPS^0.75`) | ~1.8e-12 | ~5 orders below the fp32 noise floor |

Once the trigger *does* fire under complex64, `BAD_BLOCK_TOL` flags every block (its
threshold sits far below fp32 rounding), so PARTIAL degenerates to FULL's cost while
still delivering *worse* orthogonality than FULL at complex128. Paying FULL's cost for
a worse answer is strictly dominated, hence reject rather than warn.

The estimator's own reading of the situation was not measured end to end (this guard
fires first and blocks it): `O_last` tracks the true residual within ~1.5x when there
is a real projection to do, but is measured against the *stored* (rounded) basis — on a
near-no-op step it reads rounding-level while the true loss sits at `~u32` (fp32 unit
roundoff). That is the same under-prediction failure mode as the "HONEST reset"
incident above, so the combination is unusable regardless.

## Compiler directives (boundscheck / wraparound)

`BlockLanczos.pyx` compiles `boundscheck=False, wraparound=False, cpow=True`;
`BlockLanczosArray.pyx` and the shared `BlockLanczosCore.pyx` both compile the opposite,
`boundscheck=True, wraparound=True`. This is a deliberate, audited split, not drift:

- `_lanczos_step.pxi` (included into `BlockLanczos.pyx`) already carries the workaround
  `lst[len(lst) - 1]` instead of `lst[-1]` on its cdef-typed lists — `wraparound=False`
  segfaults on negative indexing into a typed buffer rather than raising, so that pattern
  is load-bearing for the file it's compiled in.
- `BlockLanczosArray.pyx` has exactly one `[-1]` site (`calculate_thermal_gs`, indexing a
  plain Python/numpy array via `__getitem__`, unaffected by either directive) and no
  negative indexing into any `cdef`-typed memoryview or C array.

Unifying either direction has a cost with no offsetting benefit: turning `wraparound`/
`boundscheck` off on the array kernel buys nothing today (no negative-index site exists
to speed up) while silently removing the safety net around its heavier nogil buffer
arithmetic; turning them on for `BlockLanczos.pyx` would require rewriting the
`len(lst) - 1` workaround back to `[-1]` and re-auditing every other typed access in
`_lanczos_step.pxi` for a directive it was never written against. Left as-is: strict/fast
for the single-step MBS kernel, permissive/safe for the array kernel and the primitives
both kernels share.

## Known open issues

**1. PARTIAL orthogonality degrades with horizon length on a clustered/near-degenerate
spectrum (found 2026-08-18, R0b).** On an 8-cluster near-degenerate spectrum (N=2000,
p=2, `reort=PARTIAL`), final orthogonality `‖Q^H Q - I‖` degrades *monotonically* as
the run horizon grows, while a same-matrix FULL run holds `~1e-14` throughout:

| max_iter | PARTIAL `‖Q^HQ-I‖` | FULL `‖Q^HQ-I‖` |
|---|---|---|
| 80 | 1.4e-10 | 4.7e-15 |
| 160 | 8.3e-7 | 7.5e-15 |
| 240 | 6.0e-5 | 1.0e-14 |
| 320 | **0.13** | 1.3e-14 |

Confirmed via the `max_iter × {PARTIAL, FULL}` sweep (not a fixture-saturation
artifact — FULL stays flat at every horizon on the identical matrix/seed). PARTIAL also
acted on 189/320 calls (59%) at the 320-iteration horizon — a high trigger rate that
still failed to hold semi-orthogonality, meaning the trigger *fires* but the
reorthogonalization it performs (or the estimate driving block selection) is not
enough at long horizon. This sharpens (is materially worse than) the previously
documented "~1e-4, opt-in-benchmark-only" PARTIAL orthogonality failure. On a broad
quasi-continuum spectrum ("gf_style" fixture) at the same horizon, PARTIAL holds
`7.3e-11` with only a 4% trigger rate — so the defect is spectrum-dependent, not
universal. Reproducible via `.lanczos_golden/lanczos_partial_probe.py`. **This is R8's
baseline to fix** (`R8a` instruments the three additive `estimate_orthonormality` terms
against this exact regression before any change).

**2. MBS `test_no_ghost_bands` floor invariant is seed/mode-dependent, not
root-caused (found 2026-08-18, R0c).** `test_no_ghost_bands_ritz_floor`
(`test_no_ghost_bands.py`) asserts "no Ritz value below the true spectral minimum"
without the module's usual xfail, but is currently restricted to the `path="array"`
parametrization. On the `ManyBodyState` path, on the same 12-state tight-restart
fixture:

- With the fixture's own all-positive `np.random.rand()` seed (cosine between the two
  p=2 seed columns ≈ 0.75, condition ≈ 2.6), IRLM/`Reort.NONE` on `exact_degenerate`
  diverges outright: `|β|=4.5e4` at iteration 4, Ritz value **-1775** (the true minimum
  is 1.0).
- Swapping to a signed `randn` seed (matching the array path and
  `lanczos_golden.py`'s `_mbs_psi0`) removes that divergence.
- Even with the signed seed, `near_degenerate` under `Reort.NONE`/`PERIODIC` still
  lands Ritz values at **-72.4** and **-0.21**; `PARTIAL`/`FULL`/`SELECTIVE` settle to
  benign near-machine-zero artifacts (`~1e-15`) instead — consistent with this module's
  documented "spurious zero Ritz value from a partially-filled T_full when an exact
  degeneracy exceeds the block size" limitation, not a new divergence.

Net: the MBS path's story on this fixture is seed-condition-dependent *and*
reort-mode-dependent, narrower than a blanket bug but not yet root-caused. Not blocking
R1-R7 (pure code motion); revisit before or during R7, which touches the shared step
policy this floor test exercises.
