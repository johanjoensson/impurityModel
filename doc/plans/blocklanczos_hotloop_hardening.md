# Implementation Plan: Block Lanczos Hot-Loop Hardening

**Scope:** correctness and numerical stability of the *sparse* distributed core
(`src/cython/BlockLanczos.pyx`) and the *array* core
(`src/cython/BlockLanczosArray.pyx`). These loops are the hot computational core
(ground state via CIPSI, and every interacting Green's function / spectrum). The
goal is **no silent failure modes and no MPI deadlocks**, matching the EA16 block
Lanczos with partial/selective reorthogonalization (Meerbergen & Scott, RAL-TR-2000-011,
eq. 14 / Alg. 2.6).

Each item has a verification checkbox. "Serial" = `pytest`; "MPI" =
`mpirun -n {2,3,4} pytest --with-mpi`.

> **Execution conventions:** see "Implementation conventions for weak-model execution"
> in `README.md` (anchor on quoted snippets, rebuild + test after each checkbox, all
> decisions pre-made). Most of this plan is **done** or **relocated** — the only item
> still open here is §6.

---

## 1. Eliminate MPI deadlock in SELECTIVE reorthogonalization (CRITICAL) ✅

**Fixed.** Changes in `BlockLanczos.pyx` and `BlockLanczosArray.pyx`:
- Replaced non-deterministic ARPACK `eigsh` with dense `scipy.linalg.eigh`/`la.eigh`
  in both files; removed the `N_T <= 10` fork (always dense now).
- Sparse path: `ritz_to_project` list built on rank 0, `bcast` before the projection
  loop so every rank enters/skips `Allreduce` consistently.
- Removed `try/except: pass` from both SELECTIVE blocks (Items 2 merged here).
- Fixed W-index in array path: `W[j, it+1]` → `W[1, j]` (Item 3 merged here).
- Removed dead W-update code guarded by `W.shape[0] > 2` (always False).

- [x] `ritz_to_project` decided on rank 0 and `bcast`
- [x] Dense `eigh` used; `N_T <= 10` fork removed
- [x] `Allreduce` reached by all ranks unconditionally
- [x] **Verified:** `mpirun -n 3 pytest test_selective_reort.py test_restarted_lanczos.py` — 7/7 pass

## 2. Remove blanket `try/except: pass` around reorthogonalization ✅

Merged into Item 1 — both `try/except: pass` blocks removed.

- [x] Bare excepts removed; errors now propagate

## 3. Fix the array-path W-index bug ✅

Merged into Item 1 — `W[j, it+1]` → `W[1, j]` in `BlockLanczosArray.pyx`.

- [x] Array path uses `W[1, j]` (current row), matching sparse path `W[-1, j]`

## 3b. Eliminate the empty-local-partition (int32 CSR) MPI deadlock (CRITICAL) ✅

**Fixed.** A second, independent MPI deadlock class beyond Item 1's SELECTIVE one.

**Root cause.** In the array path the Hamiltonian is sliced to each rank's local
columns (`H_mat[:, basis.local_indices]`). When more ranks exist than the basis
distributes onto, some rank owns **0 local basis states**; `tocsr()` on the empty
`(global_N, 0)` slice yields **int32** `indices`/`indptr`, whereas non-empty ranks
get **int64**. The kernel binds those to `cdef long[:]` (int64) memoryviews, so the
empty rank raised a buffer-dtype-mismatch `ValueError`, left the routine, and the
other ranks blocked forever on the in-loop `Allreduce` → deadlock. Manifested as
`test_groundstate.py` hanging at 4/5/6 ranks (CIPSI `expand` → `trlm` →
`block_lanczos_array`).

**Fix applied** (`BlockLanczosArray.pyx`, `is_sparse` setup after `h_op.tocsr()`):
coerce both index arrays to int64 before binding the memoryviews —
`np.ascontiguousarray(h_op.indices, dtype=np.int64)` / `...(h_op.indptr, ...)`.

- [x] CSR `indices`/`indptr` coerced to int64 regardless of local partition size
- [x] **Regression test:** `test_block_lanczos_array_empty_rank.py` — two
      `@pytest.mark.mpi` tests partition a known Hermitian matrix with the last
      rank deliberately empty (independent of the hash distribution) and assert the
      distributed run completes and reproduces the dense eigenvalues. Verified to
      fail (empty-rank `ValueError`, exit 1) with the fix reverted, pass with it.
- [x] **Verified:** `test_groundstate.py` + lanczos suites — 16 pass at 6 ranks.

**General lesson:** any rank can have an empty local partition; distributed kernels
must stay dtype/shape robust to `local_N == 0`. When debugging an MPI hang, dump all
ranks' Python stacks (`PYTHONFAULTHANDLER=1` + `timeout -s ABRT`); the rank whose
stack differs (e.g. parked in `conftest.py` teardown `Barrier`) left the collective
early.

## Items 4–8: relocated (single worklist lives elsewhere)

To avoid two plans carrying overlapping, drifting worklists for the same hot loop, the
formerly-open items here have been **moved** and given concrete, decision-free,
weak-model-ready instructions in their new homes. Implement them there, not here:

| Old item | Where it lives now | Status of detail |
|----------|--------------------|------------------|
| **4.** EA16 block-shrink deflation (no zero-padding) | `blocklanczos_reort_reliability.md` **Phase 1** | Resolved to *full EA16 shrinking block*; literal pseudocode + `test_deflation_shrinking_block`. |
| **5.** Cholesky-first fast path | `blocklanczos_reort_reliability.md` **Phase 1** (deflation item) | Folded in: try `cholesky` first, fall back to `eigh` shrink; delete unused `_cholesky_beta`. |
| **6.** Incremental convergence check (pass buffer *views* to `converged_fn`) | **stays a TODO — see below** | Still open; hardened in place. |
| **7.** Real BLAS for the array path | `blocklanczos_blas_acceleration.md` | Tracked there (the `matmul_nogil` triple-loop + commented-out `cython_blas`). |
| **8.** Regression guardrails (orthonormality, matches-dense, mode-consistency, no rank-0-gated collectives) | `blocklanczos_reort_reliability.md` **Phase 0** | Became the cross-product oracle + ghost-band suite. |

### 6. Incremental convergence check (the one item still owned here)

**Problem.** `block_lanczos_cy` rebuilds `np.array(alphas_list)` / `np.array(betas_list)`
**every** iteration and passes growing arrays to `converged_fn`; the GF `converged`
callback then runs a full continued-fraction recursion over all blocks each step →
O(k²) extra work in the hot loop.

- [ ] Pass *views* into the already-present pre-allocated buffers — `alphas_buf[:it_abs+1]`
      / `betas_buf[:it_abs+1]` — to `converged_fn`, instead of rebuilding lists→arrays.
      Anchors: `grep -n "np.array(alphas_list)\|alphas_buf" src/cython/BlockLanczos.pyx`.
      Do **not** also touch the GF callback's continued-fraction caching (out of scope).
- [ ] **Verification:** `test_converged_views_equivalence` — identical convergence
      iteration count and eigenvalues vs the current list-rebuild version.
      Checkpoint: rebuild, `pytest -q src/impurityModel/test/test_restarted_lanczos.py`.

## Out of scope / cleanup observed while reviewing
- `_cholesky_beta` removal is now handled in the reort plan's Phase 1 deflation item
  (single conditioning policy).
- ✅ Debug artefacts removed: the `cipsi_solver.py` `H_mat_rank{rank}.txt` dump
  (was gated on `self.basis.size == 252`) and the `BOGUS …` / `DEBUG …` prints in
  `thick_restart_block_lanczos_cy` and `trlm.py` are all stripped. One
  **verbose-gated** dump remains by design: `trlm.py` prints the final `T_full`
  matrix when `verbose=True` on rank 0 (handled in reort plan Phase 3 cleanup).
