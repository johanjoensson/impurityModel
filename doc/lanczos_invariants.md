# Block Lanczos / TRLM invariants

> **Status (2026-10 core redesign):** IRLM, PARTIAL/SELECTIVE/PERIODIC reorthogonalization, the
> PRO machinery, `ea16.py`, `_reort.pxi` and the separate array kernel were deleted; there is one
> `lanczos_step` (`_lanczos_step.pxi`) and `reort` is a boolean (full reorthogonalization or none).
> Remaining mentions of the deleted pieces are kept as a record of *why* the surviving code looks as it does.


Landing place for the incident knowledge stripped out of inline comments during the
R2 comment-triage pass of the Lanczos/IRLM/TRLM redesign (see the plan at
`i-want-a-thorough-agile-fern.md`). The source still carries a one-line pointer at each
site this document expands on; this is where the "why", the numbers and the history
live. Covers `BlockLanczos.pyx`, `_lanczos_step.pxi`, `BlockLanczosArray.pyx`,
`_reort.pxi`, and the restart layer in `ed/trlm.py` / `ed/irlm.py`.

**Note on file names.** The restart-layer business logic named below used to live in
`_trlm.pxi` / `_irlm.pxi`. Those files no longer exist: `_trlm_core` / `_irlm_core`,
locking, purging and result assembly are plain Python in `ed/trlm.py` and `ed/irlm.py`,
which `BlockLanczos.pyx` re-exports. References here have been updated; older plan
documents under `doc/plans/` still use the `.pxi` names and are left as historical
record.

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
raise. `block_cols` (`BlockLanczosCore`) is the canonical column-count helper that gets this
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

**Defense-in-depth pattern** (`irlm.py`, the `res_width < p` early-stop before
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

**`f_plus`'s breakdown reference must be `tnorm`, not `1.0`** (`irlm.py`, the residual
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

**Why `res_width < p` must be tested BEFORE `purge_restart`** (`irlm.py`): the
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

## Deflation is not closure (2026-09-01)

**Rule: a rank-deficient residual narrows the block; it does not close the space.** Only
two things close a block-Krylov space, and both are about the *sweep*, not about widths:
the sweep returned fewer blocks than asked (`m_actual < m`), or it stored no trailing
residual block at all (`q_m is None`, which only happens on breakdown). A sweep that
deflates at step 1 and then runs all `m` blocks at the reduced width ends with a
perfectly good residual block and has closed nothing.

`trlm._extract_invariant_subspace` used to accept `deflated = total < m_actual * p` as a
third condition, on the stated warrant that "the spanned space is (near-)invariant, so its
Ritz pairs are accurate eigenpairs". Measured on NiO 7-bath: one solve took that exit with
widths `[3, 2, 2, …]`, `m_actual == m == 48` and `q_m` present, and returned 20 states of
which **only the lowest 2 were converged** — the rest carrying true residuals from 3.6e-5
up to **4.0e-2** against a requested `tol` of 1e-8, two of them inside the thermal cut and
so inside the returned manifold. It had skipped the restart loop, which is what converges
the Ritz pairs. On a minimal fixture (diagonal `H`, two-fold ground state, start block
spanning both plus one generic vector) it returned `[0, 0, 1.93, 6.69, 15.93, 24.32,
31.34, 37.57]` for a true lowest eight of `[0, 0, 1, 2, 3, 4, 5, 6]`.

**Corollary, and why this survived so long: the TRLM restart loop had never once run on a
width-deflated sweep.** Removing the short-circuit surfaced a latent shape crash
immediately — `beta_res = betas[-1][:p_resid, :]` sliced off the trailing coupling's padded
*rows* but not its padded *columns*. By `_build_full_T`'s convention that block is
`(width coupled to, width coupled from)`, and a sweep that narrows its diagonal blocks
leaves the second below `p` as well. Any guard that makes a code path unreachable also
makes it untested.

The same trap sits in `ea16.ritz_residual_norms`, whose `Z[nrows - p:, :]` slice assumes
uniform width. `_irlm_core` establishes that before calling it; the two width-varying sites
form the residual against the real widths instead.

## Never report a Ritz pair that failed the acceptance test (2026-09-01)

**Rule: locking is what certifies a pair. Any exit that returns pairs it did not lock must
apply the same eq. (15) test, and must stop at the first failure rather than skipping it.**

IRLM's `total < m_act * p` break is *correct* — unlike TRLM's width-aware thick restart,
the uniform-width purge/restart genuinely cannot continue a narrowed factorization, so
there is nothing to fall through to. What was wrong is where the pairs then came from. Two
exits returned Ritz pairs with no residual test at all:

- `_assemble_results` took the lowest *active* Ritz pairs of whatever the last
  factorization held. An active pair is by construction one that never converged well
  enough to be locked.
- `lock_remaining_and_stop` locked its whole `order`. Its rationale was that an exact
  breakdown zeroes every Ritz residual — true, but it fires on `res_width < p`, which also
  covers a residual block that merely *narrowed* and leaves the residuals at `O(‖H‖)`.

Measured: eigenvalues wrong by 31.6 with residuals of 5.3 against `tol = 1e-10` on the
deflating fixture, and a pair at 3.1e-6 on a generic Hermitian.

**Stop, do not skip.** Callers read the result as the lowest `len(eigvals)` eigenvalues —
`cipsi_solver._energy_cut_indices` measures its thermal window from `eigvals[0]` and trims
the top — so accepting a converged pair after rejecting a lower unconverged one produces a
set with a hole, which is a wrong answer where a short list is an honest one. Returning
fewer than `num_wanted` is the documented contract, and `get_eigenvectors` treats a short
return as an exhausted start block and retries once from the cold vector.

**At final acceptance, `EPS*‖T‖` is the floor, not a threshold.** This belongs beside
"Deflation vs breakdown scales" above: it is the same lesson one level up. Eq. (15)'s
`u = EPS` term is right for *locking*, a commitment made while the iteration can still
improve a pair, so conservatism there costs only work. At final acceptance the question is
whether a pair is fit to report, and there `EPS*‖T‖` is the residual floor of double
precision itself — the estimate accumulates over the whole `m*p`-dimensional
factorization, so converged pairs land at a small, arithmetic-dependent multiple of it
rather than under it. A gate at ten times that floor kept 3 of 3 wanted pairs in serial and
**1 of 3 at `-n 2`** on the same fixture (‖H‖ = 2.5e5), decided by reduction order alone.
`FINAL_ACCEPT_SCALE = BREAKDOWN_TOL` makes the threshold operator-relative —
`BREAKDOWN_TOL*‖T_k‖ + |CNTL(2)|`, the same construction as TRLM's `stop_beta`, not a
second literal — and still leaves five to eleven orders of margin on what the gate exists
to reject.

**How both of these were found, which is the reusable part.** Not by reading the code:
by probing the *true* residual `max_i ‖H v_i − e_i v_i‖` at the `get_eigenvectors` call
site on a real workload. Every tolerance in the stack above the kernels — including
`cipsi_solver._degeneracy_tol`, whose whole job is to be no tighter than the achieved
accuracy — floors on the tolerance the solver was *asked* for. Nothing checked what it
achieved, and the full pytest suite caught neither bug.

## Block width bounds the manifold a solve can resolve (2026-09-01)

**Rule: a block-Krylov space cannot hold more of a degenerate eigenspace than its block is
wide.** For any `m`, restarts included, `dim(K(H, V) ∩ E_λ) ≤ rank(P_λ V) ≤ p`, because
`P_λ H^j V c = λ^j P_λ V c`: every Krylov power lands in `range(P_λ V)`. This is the same
fact `test_no_ghost_bands.py` documents (multiplicity 3 against block size 2) and it
constrains *width*, never depth — enlarging `m` cannot cure it and shrinking `m` cannot
cause it.

It bears on the thermal-manifold completeness guarantee, because
`cipsi_solver.get_eigenvectors` runs at `p = 1` on two paths: `expand`'s first round
(`psi_refs = getattr(self, "psi_refs", None)` is `None` on a fresh solver) and the
warm-start cold retry, which deliberately *replaces* a wide warm block with the single
full-support hash vector. Read literally, the bound says such a solve can return one member
of a 3-fold manifold, whereupon `_energy_cut_indices` sees a state beyond the cut, reports
`need_more = False`, and certifies a manifold holding a third of its states.

**Measured, it does not do that** — recorded here so the next reader does not re-derive the
worry and act on it. Cold (`p = 1`) versus warm (`p = 7`) on the cubic d-shell fixture, same
basis, `max_energy = 1e-7`:

| basis | cold `p=1` | warm `p=7` | ground splittings |
|---|---|---|---|
| capped at 800 determinants | 3 states | 3 states | 0, 1.5e-9, 1.4e-8 |
| uncapped, 1771 determinants | 3 states | 3 states | 0, 1.8e-15, 8.4e-10 |

Two reasons, both worth knowing. On a *capped* basis the truncation breaks the symmetry, so
the triplet is split at 1.4e-8 and its members are distinct eigenvalues that any block
resolves. Uncapped, where two members are degenerate to 1.8e-15, floating-point Lanczos
under PARTIAL reorthogonalization reproduces near-degenerate copies anyway — the classic
ghost mechanism, tolerated up to `REORT_TOL ~ sqrt(EPS)`. The exact-arithmetic bound is a
statement about exact arithmetic.

So: the constraint is real and worth respecting when reasoning about a *new* code path, but
a floor on `p` is **unmotivated pending evidence**, not merely unmeasured. If it is ever
revisited, the measurement to beat is the table above.

## Warm-start / resume protocol

`block_lanczos` resumes from block `k0 = len(alphas_init)` when `alphas_init`,
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

**`store_krylov=False`** (requires `reort=False` and no `locked` set) skips this
store entirely: the accumulated basis is not retained, and the returned `Q_basis` holds
only the last two blocks (the previous block plus the residual) — exactly the tail the
warm-start protocol above needs to resume. `alphas`/`betas` are bit-identical to a
`store_krylov=True` run of the same problem; only the `O(N_det * p * k)` dead retention
is dropped. On resume with `store_krylov=False`, `Q_init` is interpreted as that
two-block tail, split by `block_widths_init[-1]`.

## Thick-restart coefficients

TRLM (`ed/trlm.py`, array blocks only) fully reorthogonalizes every new vector against the stored basis
(two CGS passes), so `Q^H Q = I` holds to working precision on the retained Ritz block and the textbook
thick-restart coefficients are exact up to roundoff and cost nothing: the retained block's projected
operator is `diag(theta_keep)`, the spike is `β_res @ Y_last`, and the carried-over residual block `q_m` is
already the whole residual (rank `≤ p`) by the recurrence identity `H Q = Q T + q_m β_res E_last^H`. There is
no Rayleigh-Ritz rebuild arm; a retained block that lost rank raises. `T_full` is still sized off `p_resid`
(the actual residual width), never off the constant `p`, so a deflating residual cannot overrun it.

The `locked` projection (restricting a solve to the complement of converged vectors) runs *after* the
full reorthogonalization in `lanczos_step` and last before normalization in the restart loop: the stored
basis carries an O(eps) locked component, and any projection done before the CGS passes is undone by them.

**Window mode** (`trlm.Window`) retains as many Ritz vectors as an energy window needs and converges the
in-window pairs plus one bracket pair above the edge; see `doc/plans/gs_redesign.md`.

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
- `BlockLanczosArray.pyx` has no negative indexing into any `cdef`-typed memoryview or C array.

Unifying either direction has a cost with no offsetting benefit: turning `wraparound`/
`boundscheck` off on the array kernel buys nothing today (no negative-index site exists
to speed up) while silently removing the safety net around its heavier nogil buffer
arithmetic; turning them on for `BlockLanczos.pyx` would require rewriting the
`len(lst) - 1` workaround back to `[-1]` and re-auditing every other typed access in
`_lanczos_step.pxi` for a directive it was never written against. Left as-is: strict/fast
for the single-step MBS kernel, permissive/safe for the array kernel and the primitives
both kernels share.

## Removed material

The PRO-estimator honesty notes and the R7-R9 "known open issues" ledger described partial/selective
reorthogonalization, which no longer exists; they are in git history (`git log -- doc/lanczos_invariants.md`).
