# Implementation Plan: Real BLAS in the Array Block Lanczos Kernel

**Scope:** `src/cython/BlockLanczosArray.pyx`. The dense/CSR path is used by
`greens_function.py` (dense branch) and by CIPSI's `irlm`/`trlm` solvers. Today its
core kernel `matmul_nogil` is a **hand-written triple loop** and the
`scipy.linalg.cython_blas` import is **commented out** (line ~257). For block sizes
`p > 1` and Krylov widths this leaves most of the FLOPs off the BLAS-3 fast path —
exactly the regime the block method exists to exploit (cf. EA16's motivation for
blocksize > 1: "take advantage of BLAS 3 kernels").

> **Execution conventions:** see "Implementation conventions for weak-model execution"
> in `README.md`. This is *performance* work gated on the reort-reliability Phase-0
> oracle being green — do Item 0 first and keep it green after every change. **Item 3
> is explicitly out of weak-model scope** (it is a distribution redesign, not a local
> edit); a small/fast model should implement Items 1, 2, 4 only.

## 0. Baseline & safety net (do first)
- [ ] Add `test_array_lanczos_matches_dense` (eigenvalues to 1e-8) and
      `test_array_lanczos_orthonormality` (‖Q†Q−I‖ < √eps), serial + MPI, before
      touching the kernel. These are the regression oracle.
- [x] **Already present:** `test_block_lanczos_array_empty_rank.py` exercises the
      MPI path with one rank owning **0 local basis states** (empty `(global_N, 0)`
      CSR slice) and asserts the distributed eigenvalues match dense. Any refactor
      below (esp. Item 3's reduce-scatter) must keep this passing — `local_N == 0`
      is a real, hit-in-practice case, not a corner. See the hot-loop hardening
      plan §3b.
- [ ] Micro-benchmark harness: time `block_lanczos_array_cy` on a synthetic dense
      Hermitian matrix for `p ∈ {1,2,4,8}`, record current ns/iteration.

## 1. Swap `matmul_nogil` for `cython_blas.zgemm`
- [ ] Enable the import (anchor: `grep -n "cython_blas\|matmul_nogil" src/cython/BlockLanczosArray.pyx`):
      `from scipy.linalg.cython_blas cimport zgemm`.
- [ ] Add **one** row-major helper and route every `matmul_nogil` call through it. BLAS
      is column-major; our arrays are `double complex[:, ::1]` (C/row-major). A C-order
      `C = op(A) @ op(B)` is computed by asking BLAS for `Cᵀ = op(B)ᵀ · op(A)ᵀ` with
      operands swapped — no explicit transpose/copy. Concrete helper to copy in:
      ```cython
      cdef void zgemm_c(double complex* A, double complex* B, double complex* C,
                        int m, int k, int n,
                        bint conjA, bint conjB) noexcept nogil:
          # C (m x n, row-major) = op(A) (m x k) @ op(B) (k x n), op = conj-transpose if flag set.
          # Computed as Cᵀ = op(B)ᵀ @ op(A)ᵀ in column-major terms (operands swapped).
          cdef double complex one = 1.0, zero = 0.0
          cdef char ta = b'C' if conjB else b'N'   # note: flags map to the *swapped* operands
          cdef char tb = b'C' if conjA else b'N'
          # leading dims are the row-major row lengths
          zgemm(&ta, &tb, &n, &m, &k, &one, B, &n, A, &k, &zero, C, &n)
      ```
      Validate the exact `trans`/`ld` mapping against `numpy @` in `test_zgemm_helper`
      **before** wiring it into the loop (random complex `A,B`; assert `zgemm_c` ≡ `A@B`
      and the conj-transpose variants to `1e-13`). If the helper test fails, fix the
      helper — do not proceed.
- [ ] Replace each `matmul_nogil(...)` call site with `zgemm_c(...)`, mapping the four
      cases the loop special-cases (`N·N`, `Cᴴ·N`, `N·Cᴴ`, `Cᴴ·Cᴴ`) to the
      `conjA/conjB` flags. Checkpoint after each site: rebuild, run
      `test_zgemm_helper` + the reort Phase-0 array cells.

## 2. Pre-allocate / reuse workspaces (no allocation in the loop)
- [ ] The loop currently does `np.append(alphas, ...)` and `np.append(betas, ...)`
      **every iteration** — each is a full reallocation+copy (O(k²) total). Replace
      with pre-allocated `(max_iter, n, n)` buffers and slice at return (the sparse
      `block_lanczos_cy` already does this — copy that pattern).
- [ ] Reuse `wp`, `alpha_i`, `M`, `beta_*` scratch across iterations.
- [ ] **Verification:** `test_no_realloc_in_loop` (optional: assert via a counting
      allocator or just confirm equivalence + benchmark improvement).

## 3. Reduce the distributed memory footprint — **STRONG-MODEL ONLY (skip if you are a small model)**

This is **not** a local edit and is deliberately out of weak-model scope: it changes
the distribution model, not a kernel call. A small/fast model should leave it and move
to Item 4.

- Problem: the MPI branch allocates `wp_global` of shape `(global_N, p)` on **every**
  rank and `Allreduce`s it — the full vector is replicated (the Rank-0/all-rank OOM
  risk symmetry-plan Phase 5.2 worries about). Anchor:
  `grep -n "wp_global" src/cython/BlockLanczosArray.pyx`.
- Why it's hard: a true fix requires each rank to compute **only the rows it owns**
  (so the full `(global_N, p)` partial product is never formed), which needs the
  Hamiltonian distributed by rows for the matvec — a different data layout than the
  current column-slice. A naive `Reduce_scatter` still transiently forms the full
  partial on each rank, so it does not actually lower peak memory. This needs a design
  decision by a human/strong model, not a mechanical edit.
- When done (by whoever does it): `test_array_lanczos_mpi_memory` — peak per-rank
  allocation scales like `local_N`, not `global_N` (`tracemalloc`); and
  `test_block_lanczos_array_empty_rank.py` stays green (the `local_N == 0` /
  same-collectives-on-every-rank constraint from hot-loop hardening §3b).

## 4. Decide the long-term relationship between the two kernels
- [ ] Document when callers should use the **sparse hash-distributed**
      `block_lanczos_cy` (large Hilbert space, matrix never formed) vs the **array**
      `block_lanczos_array_cy` (small/dense sector, BLAS-friendly). The symmetry
      plan's Phase 5 leans on the array path for multi-tOp; that is only viable after
      items 1–3 here.
- [ ] **Verification:** a short section in `doc/architecture_overview.md` stating the
      selection rule, cross-linked from both `.pyx` module docstrings.

## Acceptance
- [ ] All Section-0 oracle tests still pass (serial + `mpirun -n {2,3,4}`).
- [ ] Benchmark shows speedup growing with `p` (BLAS-3 effect); no regression at `p=1`.
- [ ] No Python-level allocation inside the iteration body.
