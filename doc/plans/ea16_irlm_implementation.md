# Plan: EA16-faithful Implicitly Restarted Block Lanczos (IRLM)

Reference: K. Meerbergen & J. Scott, *The design of a block rational Lanczos code
with partial reorthogonalization and implicit restarting*, RAL-TR-2000-011 (EA16).

---
## STATUS — IMPLEMENTED (2026-06-24)

Stages 0–3 complete. Array + ManyBodyState IRLM converge to machine precision on the
8-site/4-particle tight-binding oracle at `m=6` for `num_wanted ∈ {2,3,4}` (incl. the
degenerate pair and non-multiple-of-`p` counts), all five reort modes, serial and MPI
(np=1/2/4): final suite **81 passed serial / 88 passed at np=4**, no ghost bands, no
deadlocks. Key deliverables:

* `src/impurityModel/ed/ea16.py` — shared path-agnostic numerics (residual norms,
  eq.15 acceptance, retain/purge selection, `purge_restart` = eq.6 explicit purge with
  reverse block-Lanczos re-banding, `_block_tridiagonalize`).
* `src/impurityModel/ed/irlm.py` — single `_irlm_core` driving both paths via `block_*`
  helpers + a path-specific `sweep`; locking (`_lock_block`, collapse-skipping),
  boundary reorth-vs-locked (`_orth_against_locked`), eq.15 stop, restart-PRO.
* `src/cython/BlockLanczos.pyx` — the old standalone MBS IRLM loop retired; the symbol
  now delegates to `_irlm_core` (single source of the algorithm).
* Tests: `test_irlm_small_subspace_locking` (new regression); IRLM `xfail`s removed
  from `test_block_lanczos_reort_matrix.py`.

**Root-cause note:** the pre-existing restart was correct only for `k_blocks=1`. For
`k_blocks≥2` the block bulge-chase (`givens_qr.implicit_qr_step_block`) does not yield
the staircase `U` the Sorensen residual formula needs (verified: restart factorization
residual ≈ 1.07 for k=2), so the old code silently converged to wrong eigenvalues
(`-4.4987` vs true `-4.7588`). Replaced by the eq.6 Ritz-basis purge (residual ≈5e-14
for all k). The deferred items below (Chebyshev/Leja shifts §1F, `ω_max` W-reseed
§2.6.3, full MPI operator-deflation) are not needed for correctness on the target
problem.

---

This plan rebuilds the IRLM drivers — first the **array** path
(`irlm.py::_implicitly_restarted_block_lanczos_array` +
`BlockLanczosArray.pyx`), then the **ManyBodyState** path
(`BlockLanczos.pyx::implicitly_restarted_block_lanczos_cy`) — so they mirror the
EA16 standard-eigenproblem, regular-mode algorithm as closely as practical.

---

## 0. Scope: which parts of EA16 we implement

EA16 is a general code (standard + generalized + buckling, spectral transform,
rational Krylov, pole selection, harmonic Ritz, singular-mass breakdown). This
project solves the **standard real-symmetric/Hermitian problem `A x = λ x`** with
`OP = A` (regular mode), wanting the algebraically smallest eigenvalues. So:

**IN SCOPE (faithful to EA16):**
- §2.1 Block Lanczos process (Algorithm 2.1) — already present.
- §2.6.1 Partial reorthogonalization (Algorithm 2.6, ω-recurrence eq. 14) — present
  as a block-matrix generalization (`estimate_orthonormality`); align thresholds.
- §2.3.1 Implicitly shifted QR restart (Theorem 2.1, Algorithm 2.2) — present but
  needs correction (see Roadblock R1).
- §2.3.2 Exact shifts — present.
- **§2.2.2 Locking of converged Ritz pairs — MISSING. This is the central fix.**
- **§2.2.1 Purging — MISSING as an explicit step. Implicit restart already purges
  unwanted directions, but we add an explicit lock-aware purge (eq. 6) as a
  first-class part of the restart, not an opt-in.**
- §2.6.2 Partial reorthogonalization against locked Ritz vectors — MISSING (depends on locking).
- §2.6.3 Reorthogonalization and implicit restarting (ω_max re-seed) — present but heuristic; align.
- §3.2.4 Stopping criterion (eq. 15) — replace the ad-hoc `tol` test.
- Shrinking-block deflation (rank-deficient β) — present in the forward recurrence.

**OUT OF SCOPE (document as such; leave interface seams):**
- §2.3.3 Chebyshev / §2.3.4 Leja shifts — *optional stretch goal* (Stage 1F).
- §2.4 Spectral transformation, generalized/buckling problem, §2.4.3 rational
  Lanczos, §2.4.2 trust intervals / pole selection, §2.5 harmonic Ritz,
  §2.7 singular-mass breakdown. These are only meaningful with `OP ≠ A`.

---

## 1. Diagnosis: why the current IRLM is "broken"

Both drivers (array and ManyBodyState) are structurally identical:

```
run Lanczos to m blocks
loop:
    eigh(T) ; check max wanted residual < tol ; break if converged
    shifts = all unwanted Ritz values
    apply single-shift implicit QR for each shift, accumulate U
    keep first k_blocks ; reconstruct residual via Sorensen formula
    Cholesky/deflate new trailing block ; resume Lanczos for m-k blocks
```

Confirmed defects, ranked:

1. **No locking, no purging (the killer).** The loop tries to converge *all*
   `num_wanted` Ritz pairs simultaneously inside a tiny subspace (e.g. `m=6`,
   `k=2` blocks ⇒ only 4 shift-blocks of filtering per restart). Once a pair nearly
   converges it is neither frozen nor removed, so it keeps being recomputed and
   re-polluted; the not-yet-converged pairs never get enough subspace room. This is
   exactly the symptom captured by the test suite's `xfail`: *"IRLM doesn't reach
   1e-8 on the tight-binding system within the subspace size — a known IRLM limit."*
   It is **not** an inherent IRLM limit — it is the absence of EA16 §2.2 locking.

2. **Shift count is block-misaligned (Roadblock R1).** Both drivers use
   `shifts = eigvals_T[argsort][num_wanted:]`, i.e. `m·p − num_wanted` scalar shifts,
   then keep `k_blocks = ceil(num_wanted/p)` blocks. To reduce a block factorization
   from order `m` to order `k` you must apply **exactly `(m − k)·p`** shifts. These
   agree only when `num_wanted` is an exact multiple of `p`; otherwise the QR sweep
   over-/under-shifts and the retained `T_k`/`Q_k`/residual no longer satisfy the
   Lanczos recurrence — silent corruption.

3. **Stopping criterion is ad-hoc.** `max_res < tol` on `‖β_last · z[-p:]‖`. EA16
   eq. (15) accepts `(x, θ)` when `‖OP·x − θx‖ ≤ u‖T_k‖ + |CNTL(2)| + |CNTL(3)|·|θ|`
   (absolute + relative backward-error blend, scaled by the actual operator norm
   estimate `‖T_k‖`). The current test ignores `‖T_k‖`, so `tol` is not comparable
   across spectra.

4. **Restart residual reconstruction is correct-but-fragile.** The Sorensen formula
   `f⁺ = V⁺_{k+1-block}·β̂ + f_m·σ̂` *is* EA16 Theorem 2.1 (good), but: (a) it is only
   valid when the shift count is block-aligned (defect 2); (b) the
   `active_k < p` deflation branch silently bails (array) or recomputes via a fresh
   matvec (ManyBodyState), injecting an O(1) orthogonality loss the W-recurrence
   cannot model. Needs to be lock-/deflation-aware.

5. **Reorth re-seeding after restart is heuristic.** Both seed the Paige–Simon
   estimator `W` uniformly at `REORT_TOL = √eps`. EA16 §2.6.3 prescribes re-seeding
   the loss estimate at **`ω_max`** (the largest pre-restart `ω_{jl}`), because the
   restart's orthogonal map `Q` preserves global orthogonality only up to `‖Q E_k Q‖`.
   Uniform `√eps` is usually fine but is not the documented EA16 rule and can
   under-trigger after an aggressive purge.

6. **Two divergent copies.** Every fix must land twice (array + ManyBodyState),
   which is how the kernels drifted before (see the sibling plan
   `blocklanczos_reort_reliability.md`). We extract shared, path-agnostic helpers.

---

## 2. Target architecture

Keep the existing **path-agnostic block helpers** (`block_apply`, `block_inner`,
`block_combine`, `block_orthogonalize`, `block_normalize`, `_cholesky_or_deflate`)
so a single IRLM control routine drives both paths. New shared, numerics-only
helpers (operate on the small `T`, `Z`, `U`, and on `block_*` abstractions):

| Helper | Role | EA16 ref |
|---|---|---|
| `ea16_select_wanted(evals, which, n_want)` | pick wanted/unwanted indices | §3.2.2 WHICH |
| `ea16_exact_shifts(evals, keep_idx, n_shift_blocks, p)` | block-aligned exact shifts | §2.3.2 |
| `ea16_implicit_restart(T, p, shifts) -> (T⁺, U)` | (m−k)·p shifts, accumulate U | Thm 2.1 |
| `ea16_restart_residual(Q, U, T⁺, f_m, β_m, k, p)` | Sorensen residual block f⁺ | Thm 2.1 |
| `ea16_residual_norms(T, β_last, p)` | per-Ritz residual `‖β_last·z[-p:]‖` | §2.1 |
| `ea16_converged(res, theta, Tnorm, cntl2, cntl3, u)` | eq. (15) accept test | §3.2.4 |
| `ea16_lock(...)` | move converged pairs out; shrink T,Q,β | §2.2.2 |
| `ea16_purge(...)` | explicit lock-aware purge (eq. 6) | §2.2.1 |
| `ea16_reseed_W(omega_max, ...)` | ω_max re-seed of W after restart | §2.6.3 |

The two public entry points stay as today
(`irlm.py::implicitly_restarted_block_lanczos_cy` dispatching to the array driver,
and `BlockLanczos.pyx::implicitly_restarted_block_lanczos_cy`); both call the same
control flow, differing only through the `block_*` abstraction layer.

The EA16 main loop (Algorithm 2.2 + locking) the drivers must realize:

```
choose V1 (p columns), k wanted-blocks, m max-blocks  (m > k)
run p... actually run m steps of block Lanczos  -> order-m factorization
repeat (restart = 0..max_restarts):
    Z, Θ = eigh(T_active)                       # active = unlocked part
    res  = ea16_residual_norms(T_active, β_last, p)
    lock all wanted pairs with res ≤ EA16-tol   # §2.2.2  (grow locked set)
    if (#locked ≥ num_wanted): break            # done
    select k−#lockedBlocks shifts from unwanted Ritz values  # §2.3.2
    T⁺, U = ea16_implicit_restart(T_active, p, shifts)        # Thm 2.1
    f⁺    = ea16_restart_residual(...)                        # Thm 2.1
    β_k, … = cholesky_or_deflate(<f⁺,f⁺>)                     # shrinking block
    reseed W at ω_max                                         # §2.6.3
    resume block Lanczos for (m − k) blocks, reorth also vs locked Ritz (§2.6.2)
final: assemble locked + best remaining Ritz pairs
```

---

## STAGE 0 — Oracle, baseline, scaffolding  *(½ day)*

**Goal:** lock in a numerical reference and a red/green signal *before* touching math.

- 0.1 Add a tiny dense reference solver in the test module: full `numpy.linalg.eigh`
  of the explicit dense `A` (the test systems already build it via
  `build_dense_matrix_from_manybody`). This is the oracle.
- 0.2 Build the **failing-cell matrix**: `(reort_mode) × (path: array | MBS) ×
  (ranks: 1, 2, 4) × (num_wanted not multiple of p, e.g. 3 with p=2)`. Record which
  currently pass/`xfail`/hang. Convert the blanket IRLM `xfail` into per-cell marks.
- 0.3 Add a `cntl` parameters object (CNTL(2), CNTL(3) defaults `0, √u`) plumbed
  through both drivers, unused for now — pure scaffolding for eq. (15).

**Checkpoint C0:** dense oracle test exists; a documented red baseline of exactly
which `(mode, path, ranks, num_wanted)` cells fail today. *No production code changed.*

**Roadblock R0:** MPI hangs in the baseline (cf. the int32 empty-rank deadlock in
memory). Mitigation: run the baseline matrix with a per-test `timeout`, mark hangs
distinctly from wrong-answer failures; fix hangs only if they block Stage 1 tests.

---

## STAGE 1 — Array interface: EA16-faithful IRLM

Implement entirely in `BlockLanczosArray.pyx` (new `cpdef` numeric helpers) +
`irlm.py` (`_implicitly_restarted_block_lanczos_array` control flow). All steps
keep the existing dense/CSR + MPI-row-block matvec untouched.

### 1A. Correct the restart mechanics (Theorem 2.1) — *fixes Defect 2 & 4*
- Replace the shift list with a **block-aligned** count: `n_shift_blocks = m − k`,
  `n_shift = n_shift_blocks · p`; take the `n_shift` algebraically-largest Ritz
  values as exact shifts (`which = smallest` ⇒ shift away the largest).
- Move the implicit-QR sweep + Sorensen residual into `ea16_implicit_restart` and
  `ea16_restart_residual`; add an assertion that `U[:, :k·p]` is orthonormal to
  `√eps` and that `f⁺ ⟂ Q_k` to `√eps` (cheap, gated behind a debug flag).
- Audit `givens_qr.implicit_qr_step_block`: the bulge-chase loop runs
  `for j in range(1, m−1)`, which can leave a residual bulge in the final block row.
  Verify against a dense `(T−σI)=QR; T⁺=QᵀTQ` reference for random symmetric block
  tridiagonals; fix the loop bound / final rotation if the reconstruction error
  exceeds `√eps`.

**Checkpoint C1A:** with a large subspace (`m·p ≥ Krylov dim`, no real restart) and
with `num_wanted` *not* a multiple of `p`, array IRLM reproduces the dense oracle to
`1e-10`. The current `num_wanted=3, p=2` corruption is gone.

**Roadblock R1:** complex arithmetic — Ritz values of a Hermitian `T` are real, but
`T` is stored complex; ensure shifts are taken as real (`evals.real`) so the QR step
stays a real shift and `T⁺` retains its Hermitian band structure. Symptom if wrong:
slowly growing imaginary parts on the diagonal of `T⁺`.

### 1B. Locking of converged Ritz pairs — *fixes Defect 1 (core)*
Implement EA16 §2.2.2. Maintain a growing set of **locked Ritz pairs** `(Θ_lock,
X_lock)` (vectors in the `block_*` representation):
- After `eigh` each restart, compute residuals; any *wanted* pair with
  `res ≤ ea16-tol` is locked: append its Ritz vector to `X_lock`, record `Θ_lock`,
  and **deflate it out of the active recurrence** by zeroing rows `1..q` of the
  `(kb+1)…(kb+b)` block of `S_k` (eq. 7) → the active `T̃` shrinks to
  `(kb − qb)` while the locked pairs are held fixed.
- The active Lanczos basis from here on is kept orthogonal to `X_lock` (see 1D/§2.6.2).
- Stop when `#locked ≥ num_wanted`.

**Checkpoint C1B:** array IRLM converges all `num_wanted` to `1e-8` on the
tight-binding system with the *small* subspace (`m=6, k=2`) that currently `xfail`s.
Remove that `xfail`.

**Roadblock R2:** a locked Ritz vector is only an *approximate* eigenvector, so the
deflated recurrence's RHS is nonzero at the `O(tol)` level (eq. 7). Mitigation: keep
locking against *all* locked vectors in reorth (§2.6.2), and re-check locked
residuals at the final extraction; if a locked pair drifts above `10·tol`, unlock and
continue (EA16 does not unlock, but we add it as a safety net for the tiny subspaces
this project uses).

**Roadblock R3 (MPI):** the lock/unlock decision is a thresholded comparison; if
computed independently per rank with any reduction drift it can diverge → deadlock.
Mitigation: compute the lock mask on rank 0 from the (already broadcast) `eigvals_T`
/ residuals and `bcast` the integer mask (mirrors the existing SELECTIVE pattern).

### 1C. Explicit purging — §2.2.1  *(now first-class, not opt-in)*
Implement EA16 purging as a standard step of the restart. Exact-shift implicit
restart implicitly purges unwanted directions, but EA16 also defines an explicit
purge that compresses the order-`k` factorization to order `p` in the **Ritz basis**:
- Order the Ritz pairs wanted-first (locked pairs always retained), then form the
  purged recurrence from eq. (6): `A X_p = [X_p  V_{k+1}] · [Λ_p ; B_{k+1}E_kᵀZ_p]`,
  transformed back to band form by the unitary `Y_p` of eq. (6) so `T_p` keeps half
  bandwidth `b+1`. This removes the `(k−p)·b` lowest-priority directions while
  preserving the locked block exactly.
- Run purging together with locking each restart: lock pushes converged pairs out of
  the active recurrence (§1B); purge removes the least-wanted *unconverged*
  directions so the rebuilt subspace concentrates on the wanted, not-yet-converged
  region. The two are complementary (EA16 uses both).
- `ea16_purge` operates on the small `Λ_p`, `Z_p`, `Y_p` plus the `block_*` basis
  transform, so it is shared across both paths like the other `ea16_*` helpers.

**Roadblock R3b (purge ↔ restart consistency):** purging and the implicitly-shifted
QR restart both reduce the order and both must leave a *valid* Lanczos recurrence
(`A V_p = V_{p+1} T_p`, `T_p` band, half-bandwidth `b+1`). Applying both in one
restart can double-compress or desynchronize `T`, `Q`, `β_last`, the W estimator,
and `block_widths`. Mitigation: pick **one** compression mechanism per restart
region — purge the unconverged-but-unwanted directions via eq. (6), then let the
implicit-QR shifts (§1A) supply the *filtering* polynomial on what remains — and
assert the post-step recurrence residual `‖A V_p − V_{p+1} T_p‖ ≤ √eps` before
resuming.

**Checkpoint C1C:** (a) the post-purge factorization satisfies the band-recurrence
invariant to `√eps`; (b) locked eigenvalues are unchanged by the purge to `1e-12`;
(c) on a spectrum with many unwanted interior Ritz values, purge+lock reaches the
EA16 tolerance in strictly fewer restarts than locking alone.

### 1D. PRO integration — reorth vs locked Ritz + ω_max re-seed — Defects 5, §2.6.2/2.6.3
- During the resumed Lanczos sweep, after the standard PRO bad-block reorth, also
  reorthogonalize the new block against `X_lock` when the §2.6.2 estimate `ξ_{j+1}`
  exceeds `ω_TOL` (reuse the existing double-pass `block_orthogonalize`).
- Replace the uniform `REORT_TOL` re-seed of `W` with the EA16 `ω_max` rule: after
  the restart's orthogonal map, set the new `ω_{jl} ∼ ω_max` (largest pre-restart
  entry of `W`). Keep the uniform `√eps` as a floor.

**Checkpoint C1D:** across **all** reort modes (NONE/PARTIAL/FULL/PERIODIC/SELECTIVE),
array IRLM on the small subspace matches the dense oracle to `1e-8` and produces
**no ghost bands** (reuse `test_no_ghost_bands.py`).

**STATUS (done):** §2.6.2 reorth-vs-locked-Ritz is implemented as boundary
reorthogonalization (`_orth_against_locked` on the restarted basis, the new residual,
and every locked column; `_lock_block` skips columns that collapse against the locked
set). Empirically this holds machine precision with no ghost bands for all modes on
both paths, serial and MPI (np=1/2/4) — validated by disabling operator deflation
entirely. The §2.6.3 `ω_max` W-reseed is **intentionally deferred**: the conservative
uniform `REORT_TOL` seed already triggers sufficient reorthogonalization (no observed
loss), and extracting a correct `ω_max` from the kernel's W layout (which carries
identity self-overlap blocks) risks regression for marginal benefit. Revisit only if a
system shows post-restart orthogonality loss above `REORT_TOL`.

### 1E. EA16 stopping criterion (eq. 15) — Defect 3
- Compute `Tnorm = ‖T_k‖` (largest |Ritz value|; cheap from the `eigh`).
- Accept `(x, θ)` when `res ≤ u·Tnorm + |CNTL(2)| + |CNTL(3)|·|θ|`.
- Keep the user `tol` as a convenience that maps onto `CNTL(2)` (absolute) unless
  the caller sets `cntl` explicitly. Update docstrings to state the exact criterion.

**Checkpoint C1E:** convergence behavior is scale-invariant — scaling `A → 10·A`
scales accepted residuals proportionally and returns the same eigenvectors.

### 1F. Shift strategies (stretch) — §2.3.3 / §2.3.4
- Add `shifts="exact" | "chebyshev" | "leja"` to `ea16_exact_shifts`'s call site.
- Chebyshev: zeros of the degree-`(m−k)p` Chebyshev poly scaled to the unwanted
  interval `[α, β]` (estimate `α,β` from current Ritz spread). Leja: fast-Leja
  points seeded with the unwanted interval endpoints, reused across restarts.
- Default remains `exact` (EA16's advocated choice for well-separated extremal eigs).

**Checkpoint C1F:** on a clustered spectrum, Leja/Chebyshev shifts converge in
≤ the restart count of exact shifts; default path unchanged.

**STAGE 1 EXIT:** the array IRLM converges every Stage-0 baseline cell to the EA16
tolerance, serial and MPI (1/2/4 ranks), with locking+purging, no ghost bands, and a
documented eq.(15) stop. The `xfail` in `test_block_lanczos_reort_matrix.py` is
deleted.

---

## STAGE 2 — ManyBodyState interface

The ManyBodyState driver (`BlockLanczos.pyx::implicitly_restarted_block_lanczos_cy`)
is line-for-line parallel to the array driver, differing only in the `block_*`
abstraction (lists of `ManyBodyState` vs ndarray columns, hash-distributed MPI).

- 2.1 Port each Stage-1 helper call: because `ea16_*` helpers operate on the small
  dense `T`/`U`/`Z` and on `block_*` abstractions, they are **reused unchanged**.
  Only the basis-carrying steps (`block_combine_sparse`, `inner_multi`,
  `add_scaled_multi`, `redistribute_psis`) differ.
- 2.2 Locking: `X_lock` becomes a list of `ManyBodyState`; reorth-vs-locked uses
  `inner_multi`/`add_scaled_multi` with the `redistribute_psis` MPI reduction.
- 2.3 Residual reconstruction: replace the `add_scaled_multi(f_plus, …)` block with
  the shared `ea16_restart_residual` driving `block_combine`/`block_add_scaled`.
- 2.4 Keep the existing `slaterWeightMin` pruning at every basis-touching step.

**Checkpoint C2:** the MBS IRLM is **bit-for-bit identical** to the array IRLM on a
shared Hamiltonian for all reort modes, serial + MPI — extend the existing
`test_array_irlm.py::test_array_irlm_matches_mbs` to the locking/`num_wanted=3` cases.

**Roadblock R4:** hash-distributed MPI redistribution after the restart matvec
(`redistribute_psis`) must run on the *new* basis vectors; the locked vectors live in
the same distribution. Ensure `X_lock` is redistributed consistently or kept
replicated-small. Symptom if wrong: inner products against locked vectors mismatch
across ranks → lock mask divergence (R3 again).

---

## STAGE 3 — Pipeline finish (orchestration Stages 2–5)

Per the `hpc-orchestration` pipeline, after the core code (Stages 1–2):

- 3.1 **Docstrings** (`python-docstring` standards): document every `ea16_*` helper
  with `Args`/`Returns`, the `nogil`/BLAS context, MPI collective dependencies, and
  LaTeX for the recurrence (eq. 3,4), Thm 2.1 restart, eq. (14) ω-recurrence, eq. (15)
  stop. Cite RAL-TR-2000-011 section numbers inline.
- 3.2 **Tests** (`hpc-test-writer`): orthogonality via `assert_allclose(QᴴQ, I)`;
  recurrence residual `‖A V_k − V_{k+1} T_k‖`; restart invariants (U orthonormal,
  f⁺⟂Q_k); locking correctness (locked residuals stay ≤ tol); eq.(15) scale
  invariance; multi-rank `mpirun -np {1,2,4}` synchronization; toy matrices for speed.
- 3.3 **Review** (`python-review`): Black, 88-col, both core + test files.
- 3.4 **Build & verify**:
  ```bash
  python3 setup.py build_ext --inplace
  pytest src/impurityModel/test/test_array_irlm.py \
         src/impurityModel/test/test_irlm_divergence.py \
         src/impurityModel/test/test_no_ghost_bands.py
  mpirun -np 4 pytest --with-mpi src/impurityModel/test/test_mpi_block_lanczos_cy.py
  ```

**Checkpoint C3:** full suite green serial + MPI; IRLM `xfail`s removed; the
`irlm.py` driver delegates to shared `ea16_*` helpers with no duplicated math.

---

## Cross-cutting roadblocks (watchlist)

- **W-recurrence growth / cost.** `W` is `(2, m+1, p, p)`; SELECTIVE rebuilds
  `T_full`+`eigh` each step. Locking shrinks the active part — make sure `W`,
  `block_widths`, and the locked set stay index-consistent after each shrink, or PRO
  silently mis-maps bad blocks (this is the established failure mode in the sibling
  reort plan).
- **No `git checkout` to undo A/B experiments** (memory: the whole reort branch is
  uncommitted). Back up files before each stage; never `git checkout`.
- **Deflation × restart interaction.** A rank-deficient trailing block (`active_k<p`)
  during a restart currently aborts. With locking this is the *normal* near-invariant
  signal; route it to "lock remaining + finish" rather than "bail".
- **MPI determinism.** Every thresholded branch (lock, purge, convergence, bad-block)
  must derive from rank-0 + `bcast`, never independent per-rank reductions.

## Definition of done
Pick any `Reort` mode and any `num_wanted ≤ p·(m−1)`; both array and ManyBodyState
IRLM return the `num_wanted` algebraically smallest eigenpairs to the EA16 eq.(15)
tolerance, in serial and under MPI (1/2/4 ranks), with locking + purging, no ghost
bands, no deadlocks — and the two paths agree to machine precision on shared inputs.
```
```
