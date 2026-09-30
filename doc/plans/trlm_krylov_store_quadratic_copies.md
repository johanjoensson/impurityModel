# TRLM / array Lanczos: quadratic Krylov-basis copies in the ground-state eigensolver

Status: **fixed** (2026-09-30, branch `trlm-fix`); diagnosed 2026-09-29. Found while profiling the
GF-review baseline (`doc/reviews/gf_review.md`, row X1). This is a ground-state issue, outside the GF
campaign. The sections from "Symptom" on are the original diagnosis; the outcome comes first.

## Outcome

Four commits, plus the benchmark that measures them (`test_trlm_continuation_cost_vs_subspace_size` in
`test/lanczos/test_block_lanczos_perf.py`, `-m benchmark`). Serial, N=20000 sparse Hermitian, p=4,
`reort="full"`, `OPENBLAS_NUM_THREADS=1`, at `max_subspace_blocks=160` (D up to 640):

| step | numerics | initial sweep | continuation, per block |
|---|---|---|---|
| before | | 20.4 s (20.0 ns/(row·col)) | 110.8 ms (17.1 ns/(row·col)) |
| A: FULL reort unwraps the one-element list | bit-identical | 14.9 s | 112.8 ms |
| B: `KrylovColumnStore`, array arm | bit-identical | 15.1 s | 99.4 ms |
| C: `KrylovColumnStore`, ManyBodyState arm | array path bit-identical | — | — |
| D: `adjoint_product` conjugates the small operand | equal to rounding | **7.9 s (7.7)** | **56.5 ms (8.7)** |

**The NiO replay** (the protocol below: 15-bath archive, `CAP=50000`, `N_IW=N_W=128`, `-n 1`,
`OPENBLAS_NUM_THREADS=1`) prints its first occupation sector (N_imp = 5, the HF seed, E = -63.664293
in both runs) after **282 s on the pre-fix code** (`src/` at `3d7277ce`, i.e. master) and **155 s
after the fix**, a 1.8x speedup. The time includes the basis build. This is a controlled A/B, same
machine and environment, with the build verified live each time. The diagnosis run's "more than 10
minutes" was on `gf-review`, a different code base and cap policy, and is not a comparable baseline.
py-spy over 30 s in the next sectors, after the fix (1499 samples):

| share | leaf | what it is |
|---|---|---|
| 61.6% | `block_view.py:214-215` (`KrylovColumnStore.project`) | the CGS2 passes against the basis, `adjoint_product` inside |
| 12.3% | `trlm.py` `sweep` | the initial sweep (was 14.0%) |
| 2.3% | scipy CSR matmul | the sparse matvec |
| 2.2% | `block_view.py:181` (`KrylovColumnStore.append`) | the one-block copy into the store |

`concat_cols` and `column_stack` are gone from the leaves. What remains is the reorthogonalization
itself; see "twice is enough" below.

Overall: 2.6x on the sweep, 2.0x on the continuation. The normalized constant was flat across
m = 20…160 before and after, as it must be: every term here is O(N D) per block, so the size sweep shows
what a fix removes as a lower constant, not as a change of slope.

**Mechanism 3, found while fixing the other two: the conjugated copy.** `np.conj(Q.T) @ wp`, in
`block_inner` and in `block_orthogonalize_array` whenever `overlaps` is not passed in, materializes a
conjugated copy of the whole basis before the GEMM, on every inner product. That is two per continuation
block and two per sweep step. The diagnosis profile (Symptom) books it as "CGS2". Measured at N=50000, D=200, p=4:
`np.conj(Q.T)` alone takes 37 ms of a 51 ms inner product, while `conj(conj(wp)^T Q)^T` takes 17 ms. It
cost more than the concatenation did. `adjoint_product` (`BlockLanczosCore.pyx`) conjugates whichever
operand has fewer columns, the form `SparseKrylovDense.reort` already used. The GEMM runs with its
operands swapped, so it is not bitwise: fixture eigenvalues agree to 2.4e-14, and eigenvectors differ
only by phase (|<old|new>| = 1).

**C (ManyBodyState).** The arm is backed by `SparseKrylovDense`, with `project()` running the
continuation's CGS passes through `reort(n_passes=1)`. The overlaps come back pre-projection, so their
last rows are alpha. An adversarial review of the design before implementation changed four things:
- the store is exposed as `columns` and read only through `slice_cols`/`block_cols`, because
  `block_combine` on a `SparseKrylovDense` silently switches to per-row pruning;
- it is always complex128, since these columns rebuild the eigenvectors;
- `project()` raises when `reort` returns no overlaps;
- the row reservation is a hint, not a bound.

The results equal the old path to rounding: eigenvalues to 1.5e-14, eigenvectors differ by phase plus
~1e-6 mixing inside a 1e-9-spaced cluster. End to end, ManyBodyState TRLM on 4560 determinants (p=4,
m=30, 3 restarts) went from 5.72 s to 1.27 s.

**Not done:**
- **IRLM (`irlm.py` :284, :455, :843).** None of these appends grows with D per block. `Xl` and
  `accepted` take at most `num_wanted` single columns per solve; `:455` concatenates once per restart.
  There is nothing quadratic to remove. The review also warned against handing the ManyBodyState kernel
  a store it would keep appending into (`Q_init` is adopted as-is at `_lanczos_step.pxi:628`).
- **The "twice is enough" CGS criterion** (below) is still unmeasured.
- **The memory model.** `memory_estimate` never budgeted the concatenation's old+new transient, so it
  needs no change. The fix makes it more accurate.

## Symptom

The replay of the NiO production archive (15 bath states per impurity orbital, 144 spin-orbitals,
`impmod_tests/NiO/impmod/15_BathStates_peeledGeometry_noneReorthonormalization_6_processors_`) spent
more than 10 minutes inside the first occupation sector of the ground-state search, at `-n 1` with
`CAP=50000`. `py-spy record` over 30 s (1484 samples, `OPENBLAS_NUM_THREADS=1`):

| share | leaf | what it is |
|---|---|---|
| 25.6% | `trlm.py:536` | second CGS pass: `block_orthogonalize(wp, Q_basis)` |
| 20.6% | `trlm.py:528` | `block_inner(Q_basis, wp)`: overlaps for alpha and the first pass |
| 15.2% | `block_view.py:99` ← `trlm.py:585` | **`concat_cols` → `np.concatenate`, re-copying `Q_basis` every block** |
| 14.0% | `trlm.py:631` (`sweep` → `block_lanczos_array`) | the initial sweep (Cython, opaque to py-spy) |
| 7.3% | `trlm.py:535` | first CGS pass (reuses the overlaps) |
| 6.6% | `column_stack` ← `sweep` | **`np.column_stack` of the whole Krylov basis in the sweep's FULL reort** |
| 2.2% | scipy CSR matmul | the actual sparse matvec |

About 22% of the eigensolver goes to copying the Krylov basis, 53% to the (inherent) CGS2
reorthogonalization, and 2% to the matvec. Both copies grow quadratically with the Krylov dimension,
so the share gets worse as `max_subspace_blocks * p` grows.

This is consistent with `eigenstate_expansion_tightening.md`, which measured the restart
continuation at 85% of the eigensolver cost and concluded that shrinking the Krylov subspace only
moves work into it. The copies below are a large part of *why* the continuation is that expensive,
and removing them is orthogonal to the subspace size.

## Mechanism 1: TRLM continuation re-concatenates `Q_basis` every block

`_trlm_core` (`src/impurityModel/ed/trlm.py`) is shared by the array and ManyBodyState paths. After
each thick restart it rebuilds the Krylov basis one block at a time:

```python
Q_basis = concat_cols(Q_ret, copy_block(q_m))                  # trlm.py:511, once per restart
for i in range(k_blocks, m):                                    # continuation, up to m - k_blocks blocks
    ...
    Q_basis = concat_cols(Q_basis, copy_block(q_next))          # trlm.py:585, EVERY block
```

`concat_cols` (`block_view.py:91`) is `np.concatenate([A, B], axis=1)` on the array path. That
allocates a fresh `(N_local, D + w)` array and copies all `D` existing columns, so a continuation of
`b = m - k_blocks` blocks of width `p` starting from `D0 = k_ret` columns copies

```
sum_{j=0}^{b-1} (D0 + j p) * N_local  ~  N_local * (b D0 + p b^2 / 2)
```

complex numbers per restart, plus the same number of fresh allocations. This is O(N D²/p) memory
traffic per restart, against the O(N D) the basis actually needs, and there are up to `max_restarts`
restarts per solve and one solve per CIPSI cycle and per occupation sector.

`copy_block(q_next)` is a second, small copy (one block) that exists only because `concat_cols` might
alias. With a preallocated store it disappears.

**The ManyBodyState path is worse.** For a `ManyBodyState` block, `concat_cols` is
`ManyBodyState.from_states(A.to_states() + B.to_states())`. That materializes every column as a
Python-level state and re-merges the union support on every block. `thick_restart_block_lanczos_cy`
goes through the same `_trlm_core` lines.

`irlm.py:284`, `:455` and `:843` use the same append-by-concatenate pattern. `:284` and `:843` append
one accepted column at a time.

## Mechanism 2: the sweep's FULL reorthogonalization column-stacks a one-element list

The initial sweep calls `block_lanczos_array` (`src/cython/BlockLanczosArray.pyx`). The ground state
runs it with `reort="full"` (`groundstate.py:313`, `:441`, `:578`). Its Krylov store is already
done right: a doubling growth buffer, amortized O(1) (`BlockLanczosArray.pyx:1006-1014`, commented at
`:423`). But the store is handed to the reorthogonalization wrapped in a one-element list:

```python
Q_list = [Q_buf]                       # BlockLanczosArray.pyx:451
...
Q_list[0] = Q_buf[:, :q_cols]          # :1014, a view, no copy
...
apply_reort(wp_arr, Q_list, ..., Reort.FULL, ...)          # :801
```

and `apply_reort` → `block_orthogonalize` → `np.column_stack(Q)` because `Q` is a list
(`_block_ops.pxi:242`, and `:60/62` in `block_inner`). `np.column_stack` of a one-element list
**copies the entire stored basis**. FULL reort runs two passes per step (`_block_ops.pxi` `apply_reort`,
`for _ in range(2)`), so the whole basis is copied twice per Lanczos step. That is O(N D²/p) again,
and it is the 6.6% `column_stack` under `sweep` in the profile.

The growth buffer exists precisely to avoid per-step copies (`:423`: "amortized O(1) copies instead of
one full np.concatenate reallocation per step"). The list wrapper silently reintroduces one.

## Why the reorthogonalization itself is not the target (yet)

The 53% CGS2 is the intended cost of `reort="full"`: three passes over `Q_basis` per block (the
overlaps, then two projections), each O(N D p). It is not a bug. One possible follow-up, to be
measured before adopting, is the Kahan/Parlett "twice is enough" criterion: skip the second pass when
`||wp||` after the first pass kept more than about `1/sqrt(2)` of its norm. That would remove up to a
third of it. It changes the numerics (still to working precision), so it needs its own
ghost-eigenvalue and rank-invariance validation (`test_no_ghost_bands`, the TRLM suites at
`-n 1/2/3`), and it is **not** part of the fix below.

## Fix design

Three independent commits, in order of payoff per risk.

### A. Unwrap single-element lists instead of column-stacking them (Mechanism 2)

In `_block_ops.pxi`, `block_inner` (`:57`), `block_combine` (`:203`) and `block_orthogonalize`
(`:239`) treat `isinstance(Q, list)` as "list of column vectors or blocks → `np.column_stack`". Add the
fast path first:

```python
if isinstance(Q, list) and len(Q) == 1 and is_array(Q[0]) and Q[0].ndim == 2:
    Q = Q[0]           # a wrapped store: use the view, do not copy it
```

- **Numerics:** the result is bit-identical, since it is the same data and the same BLAS call. BLAS
  reads the non-contiguous view `Q_buf[:, :q_cols]` of a C-ordered buffer through its leading
  dimension. Confirm that numpy does not insert a hidden `ascontiguousarray` for this layout. If it
  does, store `Q_buf` in Fortran order (see B) so column prefixes are contiguous.
- **Where:** `src/cython/_block_ops.pxi`. This is a Cython change: re-run the pip install, grep the
  output for "Successfully installed", and run the `IMPURITYMODEL_BUILD=debug` gate.

### B. Preallocated column store in `_trlm_core` (Mechanism 1, array path)

At each restart, `dim = k_ret + (m - k_blocks) * p_resid` is already computed for `T_full`
(`trlm.py:485-508`, bounded as recorded there). Allocate the Krylov store alongside it, write each new
block in place, and keep `Q_basis` a view of the filled prefix:

```python
Q_store = np.empty((n_local, dim), dtype=complex, order="F")   # column prefixes contiguous
Q_store[:, :k_ret] = Q_ret
Q_store[:, k_ret:k_ret + p_resid] = q_m
filled = k_ret + p_resid
...
Q_store[:, filled:filled + w_next] = q_next     # replaces concat_cols(Q_basis, copy_block(q_next))
filled += w_next
Q_basis = Q_store[:, :filled]                   # view, no copy
```

- **Order:** Fortran order makes `Q_store[:, :filled]` a contiguous block, which BLAS consumes with no
  hidden copy. Check that `block_inner`, `block_orthogonalize_array` and `block_combine` accept
  F-ordered input without `ascontiguousarray` round trips (they call `np.conj(V.T) @ W`, which is
  fine). Measure C versus F; the sweep's own buffer is C-ordered, so the handoff at the first restart
  either copies once (acceptable, once per restart) or the sweep can be asked for F order.
- **Where the change sits:** keep it in a small `KrylovColumnStore` helper in `block_view.py` with
  `append(block)`, a `view` property and `ManyBodyState` dispatch. `_trlm_core` then stays
  path-agnostic, as its docstring promises, and `irlm.py`'s three sites can use it too.
- **Extraction:** `_trlm_extract(T_full, Q_basis, D, ...)` receives the view. Check that nothing holds
  a view across the next restart's reallocation. The old `Q_basis` is dead after `block_combine` at
  `trlm.py:444`, so it is safe, but assert it in a test.
- **Memory:** the store is allocated once at `dim` columns. That is the final size of the concatenated
  basis, so the peak does not grow. It *drops*, because concatenation briefly held the old and new
  arrays at the same time, a ~2x transient at every block. Update `memory_estimate`'s ground-state
  model if it budgets that transient.
- **Rank invariance:** `cipsi_solver.py` relies on bitwise-identical results across ranks
  (`CLAUDE.md`, build modes). Writing into a preallocated array instead of concatenating changes no
  arithmetic, but a C→F layout change *can* change BLAS blocking and therefore the last bits of every
  inner product. Run the rank-invariance tests (`test/mpi_infra/test_rank_independence.py`) and the
  TRLM suites at `-n 1/2/3`. If a layout change moves the last bits, keep C order and accept
  BLAS reading through `lda`.

### C. The ManyBodyState path

`concat_cols` on `ManyBodyState` round-trips through `to_states()`. The MBS Lanczos kernel already
keeps a block-native dense Krylov store (`SparseKrylovDense`, `_krylov_store.pxi`) that appends columns
without re-merging. Route `KrylovColumnStore`'s `ManyBodyState` arm through it (or through
`ManyBodyState.insert_rows` / a column-append primitive, if one exists by then). Lower priority: the
production ground state runs the array path (`cipsi_solver.get_eigenvectors` builds a CSR `H_mat`),
which is what the profile shows.

## Verification protocol

1. **Before and after profile.** Use the same NiO archive, `CAP=50000`, `N_IW=N_W=128`, `-n 1`,
   `OPENBLAS_NUM_THREADS=1`, and `py-spy record -d 30 -r 50 --format raw` attached once the log shows
   `Impurity occupation search:`. After A plus B, `concat_cols` and `column_stack` must be gone from
   the top leaves. Report the wall time of the first occupation sector (it is printed as the first
   `N_imp` row) before and after.
2. **A size sweep that shows the quadratic term.** `test/lanczos/test_block_lanczos_perf.py` style:
   time `thick_restart_block_lanczos` at fixed `N` for `max_subspace_blocks ∈ {20, 40, 80, 160}`
   ([[measure-the-block-you-are-modelling]]). Before the fix the per-block continuation cost grows
   linearly in the block index; after it, it is flat apart from the CGS2 term. A measure-first gate
   needs a benchmark that varies the scaling parameter, not a single operating point.
3. **Numerics.** Keep the TRLM suites green (`test/lanczos/test_trlm*`, `test_no_ghost_bands`,
   `test_irlm_locking_deflation`, `test_warm_restart_refines`) and the ground-state and CIPSI
   suites, at `-n 1/2/3`. Fix A should be bit-identical (assert it: compare eigenpairs before and
   after on a fixture with `np.array_equal`). Fix B is bit-identical in C order.
4. **The full gate** at `-n 1` and `-n 2`, plus `-n 3`, because the TRLM restart loop issues
   collectives.

## Critical files

- `src/impurityModel/ed/trlm.py` — `_trlm_core` (`:511`, `:585`), `_thick_restart_block_lanczos_array`
- `src/impurityModel/ed/block_view.py` — `concat_cols` (`:91`), `copy_block`; new `KrylovColumnStore`
- `src/cython/_block_ops.pxi` — `block_inner` (`:57`), `block_combine` (`:203`),
  `block_orthogonalize` (`:239`), `apply_reort` (`:491`)
- `src/cython/BlockLanczosArray.pyx` — `Q_list` wrapper (`:451`, `:1014`), `apply_reort` call (`:801`)
- `src/impurityModel/ed/irlm.py` — `:284`, `:455`, `:843` (same append pattern)
