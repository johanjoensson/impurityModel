# Per-frequency BiCGSTAB Green's function

**Status (2026-07-09):** Phases 0–2 done. BiCGSTAB is consolidated into `src/cython/` and is now
**matvec-bound** (93% of a solve). Phase 3a (warm starts, measured) is done and settles the
question this plan exists to answer: **per-frequency BiCGSTAB is not a faster Green's function.
It is the memory escape hatch.** On the production 375-point Matsubara mesh it costs ~10x the wall
time of a single block-Lanczos recurrence and ~7x less memory, at identical accuracy. Whether to
build the driver (Phase 3b) is therefore a decision about memory, not speed — see
[Phase 3a](#phase-3a--warm-starts-measured).

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

## Phase 3a — Warm starts, measured  ✔

The plan said: *"Per-frequency BiCGSTAB wins on memory unconditionally and loses on time once the
mesh exceeds roughly `m / 11` points — unless warm starts across the mesh cut the iteration count,
which is the thing to measure first."* Measured. They cut it by 4.1x, and it still loses on time.

### The blocker found first

`block_bicgstab` **could not refine a warm start at all.** `_cholesky_or_deflate`'s rank floor is
absolute — `evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)` with `DEFLATE_EVAL_TOL = EPS**(2/3)` —
so any initial residual block with `||R0|| < sqrt(EPS**(2/3)) ~ 6.06e-6` deflated to rank 0 and the
solver returned `x0` **unrefined and silently**, whatever `atol` asked. A frequency-swept warm start
lives exactly there: linear extrapolation across the Matsubara mesh lands at `||R0||/||Y|| ~ 1.8e-7`.

Worse, even above the floor a warm start bought *nothing*: the core applies `atol` to the deflated,
**normalized** system, so the true residual delivered is `atol * ||R0||`. Shrinking `||R0||`
silently tightened the target instead of ending the solve sooner — cold and warm both took 5
iterations, warm just landed at a smaller residual.

Fixed in `block_bicgstab` only (the shared `_cholesky_or_deflate` keeps its floor, which is
load-bearing for the `reort=NONE` Lanczos recurrence): deflate the *normalized* Gram, rescale
`beta_j`/`beta_inv`, scale the inner tolerance by `||Y||/||R0||` so `atol` means "residual relative
to `||Y||`", and early-exit explicitly when `||R0|| <= atol * ||Y||`. A cold start has `R0 = Y`, a
ratio of exactly 1, and is unchanged. Regression tests in `test_cg.py`.

> Also noted, not fixed: the dense (`is_arr`) path has **no iteration bound** —
> `global_seen_size = [inf]` and `max_iter` defaults to `np.inf`, so a non-converging array solve
> hangs forever. Only the sparse path is bounded (by `it * n < global_seen_size`).

The one production caller, the RIXS resolvent (`spectra._rixs_driver`), is warm-started and so was
relying on the old accidental behaviour: its `atol=1e-5` was being applied to a warm-start residual
roughly 10x smaller than `||Y||`. Under the corrected semantics that loosened the RIXS map's
accuracy-vs-dense from 2.6e-7 to 8.2e-7 (still far inside the benchmark's 1e-4 assertion). Its
`atol` is now `1e-6`, which reproduces 2.6e-7 exactly at unchanged wall time; `1e-7` would buy
6.4e-9 if a map ever needs it.

### Warm-start cost

NiO addition GF, 50 bath, `de2_min = 1e-12`, GS basis 5071 dets, excited basis 2332 → 4000, block
width 5, `atol = 1e-8`. Block matvecs per mesh point; every mode converged to `<1e-8` relative, so
these are directly comparable:

| axis | cold | warm | linear extrap | quadratic extrap | cubic extrap |
|---|---|---|---|---|---|
| Matsubara (375 pts, `T = 0.002`) | 12.0 | 7.10 | 4.19 | **2.90** | 4.00 |
| real axis (48 pts, `delta = 0.2`) | 12.0 | 8.29 | 5.92 | **4.29** | 4.08 |

* Sweep **direction is irrelevant** (warm-down = warm-up = 7.10 on Matsubara).
* **Quadratic extrapolation in `z` is the optimum.** Cubic is worse: it amplifies the `atol`-level
  noise in the very solutions it extrapolates through.
* Cost per point scales with the mesh spacing, so a denser mesh is cheaper *per point* and the
  total stays ~linear in the number of points. That is the asymmetry against Lanczos, whose cost is
  mesh-independent.

### Head-to-head vs one block-Lanczos recurrence

Same excited basis, same seeds, one `(block, side, eigenstate)` unit, 375-point Matsubara mesh.
Both paths use the same second-quantized block matvec (`apply_multi` / `apply_block`), so the
matvec counts are comparable. Lanczos gets its **production** convergence monitor `delta = 0.2`
(the real-axis broadening); passing `w_0 = 0.0063` instead forces it to resolve the real-axis
resolvent at that broadening and over-charges it 2x (154 blocks instead of 80).

| | matvecs | wall | retained working set | peak RSS |
|---|---|---|---|---|
| block Lanczos (`PARTIAL`) | 80 blocks | 3.15 s | `Q = m*p*N*16` = **14.2 MiB** | 20.8 MiB |
| BiCGSTAB (quadratic warm start) | 1071 (2.86/pt) | 31.85 s | 7 live blocks = **2.1 MiB** | 1.4 MiB |

The two Green's functions agree to **7.8e-9** relative. (Report peak RSS with care: whichever path
runs second is served from warm allocator pools. The analytic working sets are the honest figures.)

Note the two do not span the same subspace: BiCGSTAB's `add_states` grows the excited basis to 4000
determinants while the Lanczos recurrence stays at the 2332 it started with. They agree anyway, so
the extra determinants carry less than `1e-8` of weight — but the growth does dilute BiCGSTAB's
memory advantage (measured 6.8x rather than the `m/7 = 11x` the block counts imply).

### The trade law

With `m` Lanczos blocks, `M` mesh points, `c` matvecs per point, block width `p`, `N` local
determinants:

    time_ratio   = M * c / m            (BiCGSTAB / Lanczos matvecs)
    memory_ratio = m / 7                (Lanczos retained Q vs BiCGSTAB's 7 live blocks)
    breakeven    = M = m / c            (~28 points here, at c = 2.86, m = 80)

Their product `M * c / 7` is independent of `m`. **BiCGSTAB's footprint is constant in both `m` and
`M`; Lanczos's grows linearly in `m`.** That is the whole trade, and it is the one
`blocklanczos_reort_memory.md` asked to price: per-frequency BiCGSTAB is a better memory escape
hatch than paging `Q` to disk (10x wall time, no I/O, no new failure modes), and it is *not* a
faster Green's function on any production mesh (375 Matsubara points, 2000 real-axis points).

## Phase 3b — The driver (open, and now a memory decision)

Given the above, the driver should be an **opt-in / auto-selected fallback**, not the default:
select `gf_method="bicgstab"` when `memory_estimate.estimate_gf_peak_bytes` predicts the retained
`Q` will not fit. Its one structural advantage over Lanczos is that the frequency axis is
embarrassingly parallel, while a Lanczos recurrence is strictly sequential — so on a machine with
more ranks than the determinant-parallel matvec can absorb, BiCGSTAB converts spare ranks into wall
time that Lanczos cannot.

The shape already exists in `spectra.py:_rixs_driver`'s kernel: sweep a contiguous frequency chunk
in order, warm-starting each `block_bicgstab` from the previous frequency's solution, on a
`tmp_basis` rebuilt per point. Note that chunking caps the warm-start win — the first point of every
chunk is cold, so a chunk of `C` points costs `(12 + (C-1) * 2.9) / C` matvecs per point.

**Before writing it, price shifted BiCGSTAB.** All `M` systems share one Krylov space
(`K(z - H, b) = K(H, b)`, and the RHS is `z`-independent), which is exactly why Lanczos gets the
whole mesh from one recurrence. A shifted BiCGSTAB (Frommer 2003) keeps that property *and*
BiCGSTAB's flat memory: one seed solve plus `O(1)` vector work per shift. If the seed system's
Krylov space suffices for every shift, that is ~12 matvecs for the whole mesh — better than Lanczos
on both axes, and it would make the per-frequency driver below obsolete before it is written.

- [ ] New kernel in `greens_function.py` alongside `block_Green` / `block_Green_sparse`, selected
      by a `gf_method` in `{"lanczos", "bicgstab"}` threaded through `_block_green_group` — not by
      overloading the existing `sparse` bool.
- [ ] For addition (`c^dag`) it solves `(z + E0 - H) X = seeds` over the seed block and forms
      `G[i,j] = <seed_i | X_j>`; removal flips the sign. This yields the **full block `G_ij`
      directly**, which makes the `_gf_operator_split` pairwise / polarization-identity machinery
      unnecessary on this path.
- [ ] Frequencies become a work-unit axis: `(block, side, eigenstate, frequency-chunk)`, through
      the existing `enumerate_gf_units` / `unit_cost_weights` / `run_units_distributed` engine —
      exactly the RIXS `wIn`-chunk scheme. Sweep each chunk in mesh order and warm-start from a
      **quadratic** extrapolation in `z` of the previous two solutions (measured optimum; keep the
      chunk long, since its first point is cold).
- [ ] Two things need a per-frequency analogue rather than a port: `gf_diagnostics.py`'s
      convergence report is expressed in `(alphas, betas)` and must become residual-based, and
      `memory_estimate.estimate_gf_peak_bytes` models a Krylov store this path does not have (it is
      instead the thing that should *select* this path). The `_CappedBasisProxy` truncation oracle
      is Lanczos-specific too.
- [ ] Sweep ordering and conditioning on the real axis at small `delta` remain open; the 48-point
      `delta = 0.2` sweep above is the only real-axis data, and the production mesh is 2000 points.
- [x] ~~Benchmark Matsubara first~~ — done, Phase 3a. The success criterion
      `blocklanczos_reort_memory.md` asked for (wall time + peak RSS vs Lanczos at fixed accuracy)
      is answered: 10x slower, 7x leaner, `|dG|` agrees to 7.8e-9.
