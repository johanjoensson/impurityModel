# Per-frequency BiCGSTAB Green's function

**Status (2026-07-09):** Phases 0–2 done. BiCGSTAB is consolidated into `src/cython/` and is now
**matvec-bound** (93% of a solve). Phase 3 — the per-frequency Green's-function driver — is the
open work.

## Why

`blocklanczos_reort_memory.md` establishes that the block-Lanczos Green's function must retain the
Krylov basis `Q` in full to reorthogonalize: Ritz compression of `Q` is unsound, so the store
cannot be shrunk, only paged to disk. That plan names one alternative as *both* memory-flat and
exactly as reliable as a linear solve, and asks for a direct benchmark before more effort goes into
paging:

> **`block_bicgstab` per frequency** — already implemented in `cg.py` and used by the RIXS
> intermediate-state resolvent. Memory-flat (O(1) vectors), and *no orthogonality to lose*.

This plan is that work. A BiCGSTAB solve carries seven blocks; a converged Lanczos recurrence at
`m ~ 800` blocks retains ~100x more. The trade is one solve per mesh point instead of one
recurrence for all of them, plus poor conditioning at small `delta` on the real axis.

## Phase 0 — Baseline  ✔

`src/impurityModel/test/test_bicgstab_perf.py` (`pytest -m benchmark`, `RUN_BICGSTAB_BENCH=1`).
Times the solve the future driver will run per mesh point,

    (w + i*delta + E0 - H) X = c^dag_j |psi0>,     G[i, j] = <seed_i | X_j>

on the NiO d-shell ground state, at the production real-axis broadening and at the first fermionic
Matsubara point. Two traps, both fixed in the harness and worth remembering:

* **`de2_min`, not `nBaths`, sets the ground-state basis size.** At the CIPSI default (`1e-6`) the
  anchor selects ~600 determinants no matter how many baths, and the excited basis never leaves the
  toy regime. It is now a knob on `_nio_workload.build_ground_state_workload`.
* **The NiO d8 ground state fills one spin channel of the d shell completely**, so `c^dag`
  annihilates those orbitals outright. Seeds must be ranked by (globally reduced) seed norm; a
  prefix of the impurity orbitals hands the solver a zero right-hand side.

Note also that the excited basis **saturates** — at 50 baths it is 3863 determinants whether the
ground-state basis is 644 or 12126. The restrictions bound it, not the ground state.

## Phase 1 — Consolidate into Cython  ✔

`src/cython/BiCGSTAB.pyx` → `impurityModel.ed.BiCGSTAB`; `cg.py` is a thin re-export, the same
arrangement `irlm.py` / `trlm.py` have over `BlockLanczos.pyx`. Verbatim translation, bit-identical.

Only scalar locals are typed. The block objects stay `object`, deliberately: `ManyBodyBlockState`
is a `cdef class` in `ManyBodyUtils.pyx`, which has no `.pxd`, and **a top-level one would not
work** — Cython resolves a cross-module type import by the pxd's dotted name, so it would emit
`PyImport_ImportModule("ManyBodyUtils")` and fail against the installed
`impurityModel.ed.ManyBodyUtils`. Making it work means reshaping `src/cython/` into a
package-mirroring tree plus `cythonize(include_path=...)`. Not worth it: ~15 block-primitive calls
per iteration at ~0.1 µs of call overhead each, against a matvec measured in hundreds of
milliseconds.

## Phase 2 — Hot path  ✔

### 2a. The `local_basis` membership scan (the whole game)

`state not in basis.local_basis` — and `local_basis` is a plain `list`. Run once per determinant of
the support, twice per iteration: **O(support × basis) per iteration**, quadratic in the basis the
solver is itself growing. It was 62% of a width-5 solve, more than twice what every matvec cost
together. And it was redundant — `add_states` already dedups against `_index_dict`.

Fixed with `Basis.contains_local` (an O(1) `_index_dict` lookup; *not* `__contains__`, which runs a
global index query when distributed).

    width 1:  25.4 -> 12.6 ms/iter   (2.0x)
    width 5: 177.9 -> 53.7 ms/iter   (3.3x)

### 2b. Width-0 key masks for the determinant bookkeeping

The `seen_states` Python hash set and the two `support_keys()` calls per matvec are replaced by two
width-0 `ManyBodyBlockState` masks (`keys_new_above` / `merge_keys`, nogil):

* `seen_mask` — every determinant touched, cutoff 0; feeds the `it * n` exhaustion bound.
* `offered_mask` — every determinant handed to `add_states`, cutoff `slaterWeightMin`.

**Two masks, not one.** A determinant can enter the support below `slaterWeightMin` (seen, not
offered) and grow above it later; one mask would record it on first sighting and never offer it.
They coincide only at `slaterWeightMin == 0`.

Measured at 24723 determinants: resident bookkeeping 2906 → 2704 KiB (**a wash** — a width-0 mask
entry costs about what a boxed hash does, because `SlaterDeterminant` keys heap-allocate), but the
Python objects materialized over a solve drop 494460 → 24723 (**20x**, 18.9 → 0.9 MiB of transient
allocation), and each determinant is now built at most once per solve instead of once per
iteration. A few percent of wall time.

### 2c / 2d — dropped, measured

The plan called for fusing the three-term `axpy` updates (`block_add_scaled2`) and an in-place
`block_add_scaled_into` for the common case where the source support is already contained.

**There is no headroom left.** At 100 baths (24723 determinants, width 5) the profile is

    2.937 s   93%   ManyBodyOperator.apply_block   (12 matvecs)
    0.080 s  2.5%   add_states + contains_local + heapq.merge
    0.130 s  4.2%   the compiled block algebra (6x block_add_scaled_cy, 5x block_inner_cy, prune)

2c targets a slice of that 4.2%; fusing two of six `axpy`s buys ~1%. The transient blocks it would
remove do not raise process peak RSS either — the allocator recycles them (measured RSS delta
across a solve: 0.2 MiB).

2d additionally carries a silent-corruption hazard worth recording. At entry to
`_block_bicgstab_core` the sparse path sets `ri = r0_t = pi = rhs` — three names for one object,
deliberately ("blocks are never mutated: rebinding replaces copying"). `r0_t` is read every
iteration to the end (`b_inner(r0_t, vi)`), so an in-place write through `ri` or `pi` on iteration 1
corrupts the shadow residual and the solver converges to the wrong answer *quietly*. Any future
attempt needs a copy-on-first-use or an `is`-identity guard against `r0_t`, plus a regression test
asserting `r0_t` is unchanged after the first iteration.

**Conclusion: BiCGSTAB's own bookkeeping is done.** Its time now lives entirely in
`ManyBodyOperator::apply` — see `manybodyoperator_apply_performance.md`, which is where any further
speedup of this path has to come from.

## Phase 3 — The driver (open)

The shape already exists in `spectra.py:_rixs_driver`'s kernel: sweep a contiguous frequency chunk
in order, warm-starting each `block_bicgstab` from the previous frequency's solution, on a
`tmp_basis` rebuilt per point.

- [ ] New kernel in `greens_function.py` alongside `block_Green` / `block_Green_sparse`, selected
      by a `gf_method` in `{"lanczos", "bicgstab"}` threaded through `_block_green_group` — not by
      overloading the existing `sparse` bool.
- [ ] For addition (`c^dag`) it solves `(z + E0 - H) X = seeds` over the seed block and forms
      `G[i,j] = <seed_i | X_j>`; removal flips the sign. This yields the **full block `G_ij`
      directly**, which makes the `_gf_operator_split` pairwise / polarization-identity machinery
      unnecessary on this path.
- [ ] Frequencies become a work-unit axis: `(block, side, eigenstate, frequency-chunk)`, through
      the existing `enumerate_gf_units` / `unit_cost_weights` / `run_units_distributed` engine —
      exactly the RIXS `wIn`-chunk scheme.
- [ ] **Benchmark Matsubara first.** A few hundred points, spacing `2 pi T`, so consecutive
      `i w_n` give excellent warm starts and the shift keeps the system well conditioned. The real
      axis at small `delta` is where the conditioning gets hard; sweep ordering, and whether a
      shift-and-invert or a preconditioner is needed, are open questions there.
- [ ] Two things need a per-frequency analogue rather than a port: `gf_diagnostics.py`'s
      convergence report is expressed in `(alphas, betas)` and must become residual-based, and
      `memory_estimate.estimate_gf_peak_bytes` models a Krylov store this path does not have. The
      `_CappedBasisProxy` truncation oracle is Lanczos-specific too.
- [ ] The success criterion is the one `blocklanczos_reort_memory.md` asks for: a direct wall-time
      and peak-RSS comparison against the Lanczos path on a Matsubara self-energy, at fixed accuracy.

Cost model to beat: at the operating point above, one solve is 5 iterations = ~11 matvecs. A
Lanczos recurrence covering the whole mesh at `m` blocks is `m` matvecs. Per-frequency BiCGSTAB
therefore wins on memory unconditionally and loses on time once the mesh exceeds roughly `m / 11`
points — **unless warm starts across the mesh cut the iteration count**, which is the thing to
measure first.
