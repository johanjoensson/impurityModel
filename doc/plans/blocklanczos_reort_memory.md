# Bounding the memory of Block-Lanczos reorthogonalization (GF / self-energy path)

**Scope:** the sparse (`ManyBodyState`) Block-Lanczos kernel `src/cython/BlockLanczos.pyx`
and its Krylov retention `SparseKrylovDense` (`src/cython/ManyBodyUtils.pyx`) — i.e. the
Green's-function / self-energy path. The array kernel's Krylov growth is capped by the
TRLM/IRLM restarts and is out of scope.

**Problem.** Reorthogonalization retains the whole Krylov basis in rank-local RAM. That
forces a choice between capping `truncation_threshold` ~30x lower than the
un-reorthogonalized run allows, or setting `reort="none"` and accepting spectra and
self-energies of unverified reliability. FCC Ni with a fitted bath state above E_F is the
reference failure: 833 block-Lanczos iterations, the GF-change monitor stuck near 1e-4
against a 1e-6 target, and a divergence-guard trip at iteration 833.

**Policy.** Measure first. Pure refactors are A/B'd against a file backup (never
`git checkout` — the working tree carries uncommitted experiments). Algorithmic reorderings
may be tolerance-equivalent rather than bit-for-bit and must say so in the commit message.
Test gate per commit: `python -m pytest` + `mpiexec -n 2 python -m pytest --with-mpi`, plus
`-n 3` for anything touching the reort collectives.

Prior campaigns (do not re-plan): `blocklanczos_partial_perf_memory.md` (the columnar store
and the PARTIAL estimator calibration), `blocklanczos_reort_reliability.md` (the reort
oracle, which is this campaign's safety net).

## Phase 0 — why the store dominates (established 2026-07-09)

`SparseKrylovDense` holds `n_rows x (p*m)` complex128 over the rank-local determinant
support, `m` the block count and `p` the block width. Per retained determinant:

| term | bytes per determinant | at `p=1, m=833` |
|---|---|---|
| excited `Basis` bookkeeping | 232 | 232 |
| 3 live `ManyBodyBlockState` blocks | `3*(16p + 56)` | 216 |
| **Krylov store** | **`16*p*m`** | **13 464** |

A factor **30** at the Ni operating point (`p=1`, `m=833` — read off `impurityModel-Ni.out`);
**43x** at `p=4, m=400`. `memory_estimate.estimate_gf_peak_bytes` models this correctly,
which is exactly why `suggest_truncation_threshold` collapses the cap as soon as
`reort != "none"`. The coupling is not a bug — it is the sizing model faithfully reporting a
real `O(N_local * p * m)` allocation.

Four structural facts that shape the fix:

1. **The GF path never reads `Q` downstream.** `greens_function.block_Green_sparse` returns
   only `(alphas, betas, r)`; the store serves reorthogonalization and the warm-start resume.
   Anything producing the same `T` is a drop-in — no eigenvector reconstruction constrains us.
2. **Every access to `Q` is a monotone forward column sweep** — `apply_reort` (FULL: all
   columns; PARTIAL/SELECTIVE: ascending `bad_cols`), `selective_orthogonalize` via
   `combine`, and the EOR resume slice. Nothing needs random access.
3. **Loss of orthogonality lives in converged Ritz directions** (Paige). That set is already
   computed in `selective_orthogonalize`, which rebuilds each Ritz vector from `Q` per call
   and discards it. Retaining them (Parlett–Scott) is what makes the raw Lanczos blocks
   droppable — the Phase-3 lever.
4. `BAD_BLOCK_TOL = EPS**0.75` **stays**. It is Simon's selection threshold (project against
   `omega_j > eps^(3/4)` so post-projection overlap falls to `~eps` and the next trigger is
   far off). Raising it makes PARTIAL fire *more* often, not less.

### Instrumentation added

`SparseKrylovDense.stats()` — `rows`, `cols`, `chunks` (per-chunk `(rows, cols, used)`),
`buffer_bytes` (allocated capacity), `payload_bytes` (coefficients actually addressed),
`slack_bytes` split into `unused_col_bytes` / `unused_row_bytes`, `support_bytes`,
`total_bytes`. `payload_bytes` is what `memory_estimate` predicts; the ratio to
`buffer_bytes` calibrates the chunk growth policy.

## Phase 1 — transient elimination and chunk right-sizing (DONE)

Measured on a synthetic store, `n_rows=4000`, `n_cols=240`, `p=2` (buffer 14.6 MiB,
`memory_bytes` 16.1 MiB), peak transient via `tracemalloc`:

| reort sweep | before | after | |
|---|---|---|---|
| FULL (`cols=None`) | 29.8 MiB (**1.85x the whole store**) | 0.4 MiB (0.03x) | **74x** |
| PARTIAL (46% of columns) | 13.9 MiB (0.86x) | 0.4 MiB (0.03x) | 35x |

Root cause: `reort` called `_gather_columns(cols)` — a dense `(n_rows x len(cols))` copy —
and then `np.conj(Qsel.T)`, which is *not* a view: **two full-size copies live at once**.

1. **Streaming `reort`.** `_plan_selection` maps the selection onto the column chunks once;
   both gemms then stream chunk by chunk. A contiguous ascending run inside a chunk slices as
   a zero-copy view; anything else copies at most one chunk's columns. The overlap is
   accumulated as `conj(Q_c^T conj(Wd))` so the *large* operand enters the gemm as a plain
   (possibly strided) view — `Q_c^H` would materialize a conjugated chunk copy, whereas
   `conj(Wd)` is only `n_rows x p`. Transient is now bounded by `n_rows*p` plus one chunk.
   `_gather_columns` is deleted.
   - *MPI*: the per-pass `Allreduce(O)` stays unconditional. `p`, `n_cols` and `n_passes` are
     rank-identical (`cols` is broadcast by the caller); `n_rows` is not, so a rank owning
     zero determinants contributes zero-row gemms and still joins every collective.
   - *Numerics*: chunk-partitioned summation reorders the accumulation — tolerance-equivalent,
     not bit-for-bit. Verified against the independent `reorth_cgs2_dense` reference at 1e-15
     for full / contiguous / chunk-straddling-gapped / single-column selections.

2. **Retired chunks are trimmed.** On a growing support (the GF regime, where `reserve_rows`
   only ever sees the seed basis) it is *row* growth, not column exhaustion, that ends a
   chunk's life — so each retired chunk kept most of its 32 reserved columns. That dead
   reservation measured **95% of all chunk slack**. `_chunk_for_append` now trims a chunk to
   the columns it received before opening the next one (one chunk-wide copy, once per chunk).

   Growing-support store (100 appends, `p=2`, support 64 -> 19 864 rows, 200 columns):

   | | buffer | payload | slack | vs flat `n_rows*n_cols` |
   |---|---|---|---|---|
   | before | 54.5 MiB | 33.9 MiB | 20.6 MiB (37.8%) | 0.90x |
   | after | 37.6 MiB | 33.9 MiB | 3.6 MiB (9.7%) | 0.62x |

   The remaining slack is the single still-open chunk — `32/n_cols` ≈ 4% of an 833-column
   production store, so no adaptive column sizing is warranted.

3. **Batched Ritz projection.** `selective_orthogonalize` projected one Ritz vector at a time,
   each costing a full store sweep. The flagged Ritz vectors are mutually orthonormal
   (eigenvectors of the Hermitian banded `T` over an orthonormal `Q`), so projecting them as a
   block via CGS2 is equivalent up to rounding. Batched `RITZ_BATCH = 8` at a time: one sweep
   per 8 vectors instead of one per vector, with the materialized Ritz block bounded at
   `n_rows x 8` rather than `n_rows x k`.

Regressions: `test_krylov_store.py` gains the `reort`-vs-`reorth_cgs2_dense` equivalence
matrix, a `tracemalloc` bound asserting the transient is not store-sized, and the per-chunk
trim invariant (`used == cols` for every retired chunk). Each was confirmed to fail against
the pre-change implementation.

Gate: serial 798 passed / `-n 2` 961 passed / `-n 3` 961 passed.

## Phase 2 — single-precision Krylov store (DONE, opt-in)

`SparseKrylovDense(dtype=...)` holds the coefficient chunks in `complex64` or `complex128`.
The scatter is compiled for both widths through the `krylov_t` fused type; every arithmetic
consumer (`combine`, `reort`) promotes to complex128 through numpy, so the overlaps, the
residual and the live recurrence blocks never narrow — only the *stored* basis does.

Measured on a 4-bath SIAM with levels straddling E_F (200-determinant sector, 82 blocks):

| mode | dtype | buffer | true ‖QᴴQ − I‖ | max\|G − G_ref\| |
|---|---|---|---|---|
| FULL | complex128 | 786 432 B | 1.1e-15 | 7.7e-15 |
| FULL | complex64 | **393 216 B** | 6.0e-08 | 1.8e-07 |
| PARTIAL | complex128 | 786 432 B | 1.9e-09 | 3.7e-14 |

Two findings, both of which changed the design:

1. **`PARTIAL`/`SELECTIVE` must reject `complex64`.** A basis stored to `u32 ~ 6e-8` cannot be
   projected against more accurately than that (measured: FULL + complex64 settles at
   `‖QᴴQ − I‖ = 6.0e-08`), while those modes steer to `REORT_TOL = sqrt(EPS) ~ 1.5e-8`. The
   target is unreachable. And once the trigger fires, block selection uses
   `BAD_BLOCK_TOL = EPS**0.75 ~ 1.8e-12`, five orders *below* the fp32 noise floor — so every
   block is flagged and PARTIAL degenerates into FULL while landing at a worse orthogonality
   than FULL at complex128. Paying FULL's cost for a worse answer is strictly dominated, so
   `block_lanczos_cy` raises. `FULL`/`PERIODIC` hold no estimator and simply settle at ~6e-8.

   *Not measured end to end* (the guard fires first): what the estimator itself would do. Its
   reading is regime dependent — `O_last` tracks the true residual within ~1.5x when there is a
   real projection to do (`1.71e-09` vs `2.49e-09`), but it is measured against the *stored,
   rounded* basis, so on a near-no-op step it reads `1.09e-16` while the true loss sits at
   ~`u32`. That under-prediction is the failure mode recorded in
   `partial-reort-estimator-underprediction`. Either way the combination is unusable; the
   floor argument above is the one that is established.

2. **It cannot be the default.** `test_gf_truncation` asserts that a capped recurrence
   reproduces the dense `P H P` resolvent to `atol=1e-9` — an exactness guarantee, not a
   tolerance. complex64 lands at ~1e-7 and breaks it. So `krylov_dtype` is opt-in
   (`complex128` everywhere by default), and `block_Green_sparse` passes an explicit request
   straight through to the kernel's guard.

Whether complex64-FULL or complex128-PARTIAL is the more orthogonal basis is workload
dependent (on a 5-bath variant PARTIAL drifts to 4.4e-07, above its own `sqrt(EPS)` target),
so the regressions in `test_krylov_dtype.py` assert absolute bounds rather than an ordering.

Aside, from the same sweep: **`PERIODIC` diverges on this workload at both dtypes**
(‖QᴴQ − I‖ = 1.0 after ~90 blocks). It reorthogonalizes only every `reort_period` steps,
which is not enough on a dense spectrum. Pre-existing; recorded here, not fixed.

The real payoff of the dtype machinery is Phase 3: the compressed basis is projected against
*exactly* (no estimator to blind) and is small, so `complex64` there is both sound and a
further 2x on an already 8x-smaller store.

## Phase 3 — Ritz compression is unsound (NEGATIVE RESULT, implemented and removed)

The plan was to bound the store at `krylov_budget` columns by periodically replacing it with
the converged Ritz vectors (the only directions along which orthogonality can be lost —
Paige), a residual-ordered buffer of not-yet-converged pairs, and the last few raw Lanczos
blocks. **It does not work, and the failure is structural rather than a tuning problem.**

The implementation carried a coordinate map `M`, the replicated column-orthonormal matrix
with `store == Q_full @ M`. A Ritz vector is then `y_k = Q_full s_k ≈ store (M^H s_k)`, and
because `M`'s columns are orthonormal the part the store *cannot* represent is exactly
`sqrt(1 - ||M^H s_k||^2)` — computable from replicated data, for free. That diagnostic is
what killed the idea. Traced on the 4-bath SIAM (`p = 2`, 200-determinant sector):

| budget | ritz_buffer | compressions | `d_before → d_after` (first) | `n_converged` (last) | `max_lost` (last) |
|---|---|---|---|---|---|
| 64 | 16 | 45 | 66 → 53 | 120 / 166 | **1.00** |
| 64 | 64 | 51 | 66 → 66 | 144 / 166 | **1.00** |
| 128 | 64 | 19 | 130 → 130 | 158 / 166 | **0.95** |
| 200 | 200 | 0 | — | — | — |

Two things to read off it:

1. **Compression barely shrinks anything while it is still lossless.** `d_after ≈ d_before`,
   because `n_converged` grows nearly as fast as the Krylov dimension itself. The only
   losslessly droppable directions are the unconverged non-buffer ones, worth ~20%.
2. **Forcing the budget below `n_converged` destroys the retained span.** `max_lost` → 1.0:
   the converged Ritz vectors are no longer representable at all, so they cannot be
   projected out, so orthogonality is simply lost.

The reason: a Ritz vector that converges at step `m` has `y_k = sum_j Q_j s_k[j]` with weight
on **every** block, including the ones an earlier compression discarded. It cannot be
reconstructed after the fact, and therefore cannot be projected out. Keeping a buffer only
helps for directions that were already near-converging when the blocks were dropped. There is
no bounded subspace that contains all *future* converged Ritz vectors except the whole Krylov
space.

This is precisely why Parlett–Scott selective orthogonalization, and Simon's own PRO codes
(LANSO), keep `Q` on **secondary storage**: the Lanczos vectors must remain available to
*form* the Ritz vectors, even though only a few of them are ever projected against. The
original instinct to page `Q` to disk was the correct one; there is no clever basis that
avoids it.

(Note for anyone re-reading the raw traces: `sqrt(1 - repr2)` has a noise floor of
`sqrt(EPS) ≈ 1.5e-8` when `repr2 ≈ 1`, so the early `max_lost ≈ 3e-8` readings are rounding,
not loss. Only the O(1) values are real.)

Phases 1 and 2 turn out to be the *prerequisites* for the out-of-core route rather than
consolation prizes: every access to the store is now a monotone chunk-local streaming sweep
(Phase 1), and `complex64` halves the bytes that have to move (Phase 2).

## Phase 4 — make the sizing model pay out (DONE)

The cap is what forces truncation, so a saving the sizing model does not know about never
reaches the user. `memory_estimate` gains `krylov_dtype` on `estimate_gf_peak_bytes`,
`suggest_truncation_threshold`, `max_colors_within_budget` and `log_memory_budget`, routed
through a new `_krylov_itemsize(reort, krylov_dtype)`. That helper mirrors the kernel's guard —
it *raises* on `complex64` + `PARTIAL`/`SELECTIVE` rather than predicting a peak for a
combination the solver refuses to run.

The old `(n_rows x n_cols)` gather transient is no longer modelled: Phase 1 deleted it.

## Phase 5 — out-of-core spill (now the primary route, not a backstop)

Phase 3 established that `Q` must be retained in full. The remaining lever is *where*.
`SparseKrylovDense` would gain a spill policy: chunks beyond a RAM budget backed by a per-rank
`np.memmap`, unlinked on open, LRU-evicted. Phase 1 already made every access a chunk-local
forward sweep, so this is a transparent chunk accessor — no algorithm change.

What makes it plausible: **PARTIAL acts rarely.** After the estimator calibration
(`blocklanczos_partial_perf_memory.md`) it fired on 5 of 60 iterations, projecting against a
subset of blocks. Reads are the exception; writes are sequential appends.

What makes it risky, and must be measured first:

* The target clusters expose only a shared parallel FS. Hundreds of ranks streaming Krylov
  columns through Lustre/GPFS may cost more than the matvec the reorthogonalization protects.
* The modes that can spill most cheaply in bytes (FULL/PERIODIC, which may use `complex64`)
  are exactly the ones that read the *whole* store every step. PARTIAL reads little but
  cannot narrow. That tension decides the policy and is not yet measured.

Probe `/proc/mounts`, refuse-with-warning on a network FS unless forced, and measure achieved
per-rank bandwidth before making it a default.

## Alternatives

- **`block_bicgstab` per frequency** — already implemented in `cg.py` and used by the RIXS
  intermediate-state resolvent. Memory-flat (O(1) vectors), and *no orthogonality to lose*: it
  sidesteps this entire problem class. The cost is one solve per mesh point instead of one
  recurrence for all of them, plus poor conditioning at small `delta` on the real axis. For a
  Matsubara self-energy — where the mesh is a few hundred points and `delta` is effectively the
  Matsubara spacing — this deserves a direct benchmark against the Lanczos path before more
  effort goes into paging `Q`. It is the only option here that is both memory-flat and exactly
  as reliable as a linear solve.
- **Recompute `Q` from `(alphas, betas)` on demand** — rejected. O(m) matvecs per reort event;
  at `m ~ 800` and ~70 acted steps, a ~50x slowdown of the dominant kernel.
- **Restarted Lanczos for `f(A)b`** (Frommer/Güttel/Schweitzer) — rejected as scoped. Would
  rewrite the continued fraction, the frozen-mesh convergence monitor and the thermal average.
- **More ranks.** The store is rank-local and scales as `1/ranks`; `estimate_gf_peak_bytes`
  already models it. Mundane, and it composes with everything above.
