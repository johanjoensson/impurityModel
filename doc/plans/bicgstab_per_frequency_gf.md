# Per-frequency BiCGSTAB Green's function

**Status (2026-07-10): the memory case is gone.** Phases 0–2 done; BiCGSTAB is consolidated into
`src/cython/` and is matvec-bound (93% of a solve). Phase 3a priced it against a block-Lanczos
recurrence whose convergence monitor sampled the whole Ritz band on the real axis — even for a
Matsubara-only mesh. With that fixed (`81a3c75`, `_gf_eval_meshes`), the same Lanczos recurrence
converges in **10 blocks instead of 74**, and per-frequency BiCGSTAB comes out **136x slower and
only 1.15x leaner**. It is not the memory escape hatch; the memory requirement it was meant to
escape was a monitor bug. See [Phase 3a-ter](#phase-3a-ter--the-re-price-2026-07-10).

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
`M`; Lanczos's grows linearly in `m`.** That is the whole trade.

> **Superseded.** Every number in this section was measured against a Lanczos recurrence whose
> convergence monitor was resolving the real-axis resolvent for a Matsubara-only mesh. With the
> monitor fixed the same recurrence needs `m = 10`, not 80, and the memory ratio falls from 6.8x to
> **1.15x** while the wall-time ratio rises from 10x to 136x. See
> [Phase 3a-ter](#phase-3a-ter--the-re-price-2026-07-10). The trade law itself is unchanged; only
> `m` was wrong, and `m` is the whole argument.

## Phase 3a-bis — Auditing the Lanczos convergence monitor  ✔

A hand-rolled width-5 seed block suggested the recurrence was over-converging by ~10x (80 blocks
run, 8 needed). **That was an artifact of the harness**, not a property of the code: the driver
decomposes the d-shell into width-1 symmetry blocks, and the error norm there is a scalar rather
than the max over a 5x5 matrix. Re-measured *through* `calc_selfenergy`, per `(block, side)` unit —
`m` blocks produced against the minimum `k` for which the same continued fraction truncated to `k`
blocks reproduces `G` on the mesh the driver actually evaluates:

| unit | width | blocks `m` | `k` for 1e-8 | over-convergence |
|---|---|---|---|---|
| real axis | 1 | 114–116 | 82–97 | **1.2–1.4x** |
| Matsubara | 1 | 114–116 | 28–32 | **3.6–4.1x** |

So the monitor is close to right on the real axis, which is what actually drives the depth. It
over-converges only when a real-axis mesh was **not requested**: it converges the real-axis
resolvent regardless. For a Matsubara-only self-energy (the DMFT case, `w=None`), capping the
recurrence at the measured requirement gives, end to end through `calc_selfenergy`:

    matsubara-only, cap 35 vs 114:  3.65x wall time, ~3.3x less retained Q
                                    sigma      rel diff 1.3e-09
                                    sigma_static  bit-identical
    both meshes,   cap 100 vs 114:  1.37x wall time
                                    sigma      rel diff 2.6e-16
                                    sigma_real rel diff 1.5e-07
                                    sigma_static  bit-identical

`Sigma(iw_max) - Sigma_static` is unchanged to all printed digits in every case, so a truncated
continued fraction does not damage the high-frequency tail.

**Actionable:** make the monitor mesh-aware — test convergence of `G` on the meshes the caller
passed, keeping the two-consecutive-steps gate. Worth ~3.6x wall and ~3.3x memory on Matsubara-only
runs, nothing on runs that also want the real axis. Not the 10x it first appeared to be.

### The FCC-Ni operating point is a *divergent* run

`blocklanczos_reort_memory.md` sizes its memory table at "`p=1`, `m=833` — read off
`impurityModel-Ni.out`". The last line of that file is:

> warning: block Green's function did not reach the convergence tolerance 1.0e-06; the
> block-Lanczos recurrence was truncated after 833 block(s) (**divergent tail**).

So `m=833` is the `BETA_BLOWUP_FACTOR` divergence guard truncating a runaway recurrence, **not the
monitor asking for 833 blocks**. The store that motivates paging `Q` to disk is being sized from a
recurrence that never converged. Whether a *converged* Ni Green's function needs anything like 833
blocks is unknown, and should be established before that plan's cost model is trusted.

That run could not be reproduced here: `impurityModel-Ni.out` was not produced by this codebase
(neither `"Bath geometry"` nor `"Hybridization fit"` appears anywhere in `src/`), and `h0_106.dat`
is a dense 106x106 matrix that `read_h0_dict` cannot parse. Reproducing the Ni run's *structural*
settings on NiO instead (`chain_restrict=True`, no `dN` window) changes none of the numbers above.

> Two incidental bugs found while trying: `selfenergy.get_selfenergy` still mis-calls
> `get_noninteracting_hamiltonian_operator` with a stale positional argument order (already noted
> in `_nio_workload`), so the CLI entry point is dead; and `op_parser.skip_whitespaces` raises
> `UnboundLocalError` rather than a parse error on an all-whitespace line.

## Phase 3a-ter — The re-price (2026-07-10)

Phase 3a-bis called the monitor "close to right on the real axis" and worth ~3.6x on Matsubara.
It was worth far more than that, and the reason is worse than over-convergence.

`_make_gf_convergence_monitor` never saw the caller's frequency mesh. It built its own from the
resolved Ritz band on the line `w + i*delta`, so it converged the **real-axis** resolvent at
broadening `delta` no matter which axis had been requested, and across the whole band rather than
the window asked for. `_gf_eval_meshes` (`81a3c75`) now hands it the frequencies `calc_G` will
actually be given.

Same workload as Phase 3a — NiO 50 bath, `de2_min = 1e-12`, excited basis 3232 -> 4000, block
width 5, 375-point Matsubara mesh at `T = 0.002`, `atol = 1e-8`, `reort = PARTIAL`:

| Lanczos monitor | rel-tol | blocks `m` | status | retained `Q` | wall | memory ratio | breakeven |
|---|---|---|---|---|---|---|---|
| band-wide (old) | 1e-6 | 74 | converged | 18.25 MiB | 2.54 s | 8.54x | 26.4 pts |
| band-wide (old) | 1e-9 | 193 | **diverged** | 47.59 MiB | 12.61 s | 22.28x | 68.9 pts |
| caller's mesh | 1e-6 / 1e-9 / 1e-14 | **10** | converged | **2.47 MiB** | 0.26 s | **1.15x** | 3.6 pts |

BiCGSTAB, quadratic warm-start extrapolation in `z`: 1051 matvecs (2.80/pt), 35.8 s, 2.14 MiB of
live blocks. `|dG|` between the two paths is 7.0e-9 in every row — which is BiCGSTAB's own `atol`,
not a Lanczos error.

Three things follow, and the first two invalidate Phase 3a's conclusion rather than refine it.

* **Phase 3a's row is reproduced exactly, and it was measuring the monitor.** The band-wide monitor
  at the then-current 1e-6 floor gives 74 blocks / 2.54 s / 18.25 MiB / 8.54x / 26.4 points against
  Phase 3a's reported 80 / 3.15 s / 14.2 MiB / 6.8x / ~28 points. The harness agrees; the baseline
  was wrong.
* **`m = 10` is not premature.** `G` at 10 blocks equals `G` at 108 blocks to **3.0e-16**, and the
  block count is unchanged as the tolerance is driven from 1e-6 to 1e-14 — the monitor cannot ask
  for fewer than the recurrence needs. The Matsubara resolvent is simply converged: every point
  `i*w_n` sits a distance `sqrt(E_k^2 + w_n^2)` from every pole.
* **The band-wide monitor could not reach 1e-9 at all.** Asked for it, the recurrence ran to 193
  blocks and tripped the `BETA_BLOWUP_FACTOR` divergence guard. This is the same failure the
  FCC-Ni `m = 833` figure records (below), on a workload we *can* reproduce.

So the trade law stands as algebra but its inputs collapse:

    time_ratio   = M * c / m            = 375 * 2.80 / 10   = 105x  (matvecs), 136x wall
    memory_ratio = m * p / 7            = 10 * 5 / 7        = 7.1x columns, 1.15x measured
    breakeven    = M = m / c            = 3.6 points

The measured memory ratio (1.15x) falls short of the column ratio (7.1x) because BiCGSTAB's
`add_states` grows the excited basis from 3232 to 4000 determinants while the Lanczos recurrence
needs only the 3232 it started with. That growth was always there; it simply had nothing to hide
behind when `m` was 74.

**Conclusion.** Per-frequency BiCGSTAB is not a faster Green's function *and it is not a leaner
one*. `blocklanczos_reort_memory.md`'s premise — that the retained `Q` is large enough to need an
escape hatch — rests on `m` in the hundreds, and every observation of that regime so far has been
a monitor that was converging a resolvent nobody asked for, or a recurrence that was diverging.
Before any further work on this path, someone must exhibit a **converged** Green's function whose
retained `Q` does not fit. That has not been done.

## Phase 3b — The driver (blocked: it has no case)

**Do not build this until Phase 3a-ter's open question is answered.** At the only operating point
we can reproduce, the driver would be 136x slower for a 1.15x memory saving. Its remaining
structural advantage over Lanczos is that the frequency axis is embarrassingly parallel while a
Lanczos recurrence is strictly sequential — so on a machine with more ranks than the
determinant-parallel matvec can absorb, BiCGSTAB converts spare ranks into wall time that Lanczos
cannot. That is a *scaling* argument, not a memory one, and it has not been measured.

What would revive the memory case: a **converged** Green's function (not a diverging one, and not
one whose monitor is chasing an axis the caller never evaluates) whose retained `Q` does not fit in
RAM. If such a workload exists, `m` is large there and the rest of this section applies as written.
If it does not, the honest answer is that `reort=PARTIAL` block Lanczos is simply the right kernel.

The sketch below is kept because it is correct *if* the premise is ever established.

The shape already exists in `spectra.py:_rixs_driver`'s kernel: sweep a contiguous frequency chunk
in order, warm-starting each `block_bicgstab` from the previous frequency's solution, on a
`tmp_basis` rebuilt per point. Note that chunking caps the warm-start win — the first point of every
chunk is cold, so a chunk of `C` points costs `(12 + (C-1) * 2.9) / C` matvecs per point.

**Before writing it, price shifted BiCGSTAB.** All `M` systems share one Krylov space
(`K(z - H, b) = K(H, b)`, and the RHS is `z`-independent), which is exactly why Lanczos gets the
whole mesh from one recurrence. A shifted BiCGSTAB (Frommer 2003) keeps that property: fix the
stabilizing `omega_i` on a seed system and transplant them as `omega_i^z = omega_i / (1 + omega_i z)`,
which restores the collinearity `r_m^z = r_m / Phi_m(-z)` that BiCGSTAB's local minimal-residual step
otherwise destroys. Seed the *hardest* shift (smallest `|Im z|`); the rest then converge at least as
fast.

It does **not** keep BiCGSTAB's flat memory. Textbook shifted BiCGSTAB stores `x_z` and `p_z` per
shift — `2M` blocks, 750 here against Lanczos's 80 — and chunking the mesh into groups of `C` only
buys a Pareto curve (`C = 40` -> 87 blocks and ~112 matvecs, i.e. parity with Lanczos on both).
It dominates *per-frequency* BiCGSTAB everywhere, but it is not free. The one way to get flat
memory is to exploit that the driver needs only `G_ij = <y_i | X_j>`, never `X_j` itself, and
project onto the seed Krylov space — which is what the continued fraction already does.

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
