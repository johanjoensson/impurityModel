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
- [x] Add `test_array_lanczos_matches_dense` (eigenvalues to 1e-8) and
      `test_array_lanczos_orthonormality` (‖Q†Q−I‖ < √eps), serial + MPI, before
      touching the kernel. These are the regression oracle. **DONE (2026-06-24):**
      `test_array_lanczos_oracle.py` — `matches_dense` (p∈{1,2,3}, full Krylov on a
      dense Hermitian) and `orthonormality` (column Gram `Q†Q`, serial + an MPI
      row-block variant with the `Allreduce`'d partial Gram); serial + MPI n=2,3,4 green.
- [x] **Already present:** `test_block_lanczos_array_empty_rank.py` exercises the
      MPI path with one rank owning **0 local basis states** (empty `(global_N, 0)`
      CSR slice) and asserts the distributed eigenvalues match dense. Any refactor
      below (esp. Item 3's reduce-scatter) must keep this passing — `local_N == 0`
      is a real, hit-in-practice case, not a corner. See the hot-loop hardening
      plan §3b.
- [x] Micro-benchmark harness (2026-06-23): dense Hermitian, FULL reort, `p ∈ {1,2,4,8}`.
      Hand-loop baseline 0.59/0.86/1.80/7.87 ms/iter → zgemm 0.23/0.31/0.45/0.94 ms/iter
      (**2.6× / 2.8× / 4.0× / 8.4×**; speedup grows with `p`, no regression at `p=1`).

## 1. Swap `matmul_nogil` for `cython_blas.zgemm` — ✅ DONE (2026-06-23)

Implemented by replacing `matmul_nogil`'s body with a single `zgemm` call, **keeping
its exact signature** so all call sites are unchanged (simpler than a separate
`zgemm_c` helper + per-site edits). Row/col-major mapping: row-major
`C = opA(A) @ opB(B)` is computed as column-major `Cᵀ = opB(B)ᵀ @ opA(A)ᵀ` by swapping
the operands (B then A), swapping `(m,n)→(n,m)`, keeping the trans flags, and using the
physical row lengths (`A.shape[1]`, `B.shape[1]`, `n`) as leading dims. Zero-dim guards
added for empty MPI ranks (`k==0` → `C = beta*C`; `m==0`/`n==0` → no-op).

- [x] Enabled `from scipy.linalg.cython_blas cimport zgemm`.
- [x] Validated the trans/ld mapping against `numpy @` for all four combos
      (`N·N`, `Cᴴ·N`, `N·Cᴴ`, `Cᴴ·Cᴴ`) + alpha/beta in **`test_zgemm_matmul.py`**
      (the plan's `test_zgemm_helper`), to 1e-12, plus the k=0 / empty-rows guards.
- [x] All `matmul_nogil` call sites use the BLAS path (the function body itself was
      swapped). Reort Phase-0 array cells + empty-rank MPI stay green.

## 2. Pre-allocate / reuse workspaces (no allocation in the loop)
- [x] `alphas`/`betas` are already pre-allocated `alphas_buf`/`betas_buf` (no
      `np.append` in the hot loop) — done prior to this phase.
- [ ] **Bounded W** still outstanding: the per-iteration allocation is inside
      `estimate_orthonormality` (fresh `W_out` each call); truly bounding it needs that
      function to write into a caller-provided buffer. Memory micro-opt, not a leak
      (W is bounded per restart cycle). Deferred.
- [ ] Reuse `wp`, `alpha_i`, `M`, `beta_*` scratch across iterations (minor; deferred).

## 3. Reduce the distributed memory footprint — ❌ WON'T FIX (decided 2026-06-23)

**Decision: do not implement halo exchange / a distribution redesign for the array
kernel.** The `(global_N, p)` replication is real, but it optimizes a path the
memory-bound case never takes:

- The replication lives **only in the array kernel**. A massive basis runs through the
  **sparse hash-distributed kernel** (`BlockLanczos.pyx`), which forms no dense global
  vector at all (`apply_multi` on `ManyBodyState` + `redistribute_psis` by hash; only
  `p×p` `Allreduce`s). It is already memory-scalable.
- The array kernel can't even be forced onto a massive basis: its dense path forms the
  `global_N²` matrix, which OOMs long before the `global_N·p` vector does. So the
  `(global_N, p)` transient is never the binding constraint — either `global_N` is
  small (fine) or the kernel is mis-selected and a larger object kills you first.
- For the genuinely massive case (CIPSI ground-state basis), the binding constraint is
  **basis size** itself (`global_N` Slater determinants split across ranks); no matvec
  trick helps once the basis representation doesn't fit. And the **Green's function**
  step that follows applies `c†/c`, leaving for the larger `N±1` sectors — so it needs
  strictly *more* memory in a space not yet built. Haloing the ground-state matvec buys
  nothing there.

Net: the array kernel's per-rank memory model is appropriate for its small/dense remit;
the real scaling frontier is upstream (shrink sectors via the symmetry/Casimir work; or
a GF scheme that doesn't materialize the full `N±1` Krylov basis), not distributed-matvec
plumbing. A lightweight guardrail comment was added at the `wp_global` allocation instead.

Original analysis kept for reference:
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

## 4. Decide the long-term relationship between the two kernels — ✅ DONE (2026-06-23)
- [x] Documented the selection rule (**sparse hash-distributed `block_lanczos_cy`** for
      a large Hilbert space where the matrix is never formed vs. **array
      `block_lanczos_array_cy`** for small/dense, BLAS-friendly sectors and block size
      `p > 1`) in `doc/architecture_overview.md` ("Block Lanczos kernels: which one to
      use"). The array `BlockLanczosArray.pyx` has a module docstring; the
      `BlockLanczos.pyx` module docstring already names both wrappers.

## Acceptance
- [x] All Section-0 oracle tests pass serial (262/0/50) + MPI n=2 (384/0/100); n=3/4
      verified green in earlier phases (the kernel change is rank-count-agnostic).
- [x] Benchmark shows speedup growing with `p` (2.6×→8.4× for p=1→8); no regression at `p=1`.
- [~] No Python-level allocation inside the iteration body — `alphas`/`betas` are
      pre-allocated; the W buffer (via `estimate_orthonormality`) still allocates per
      iteration (tracked under Item 2 "Bounded W", deferred).
