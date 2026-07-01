# Benchmarking `calc_selfenergy` — Profiling Plan + Improvement Framework

## Context

`selfenergy.calc_selfenergy` (`src/impurityModel/ed/selfenergy.py:232`) is the top-level
DMFT impurity solve: build the interacting Hamiltonian, find the thermal ground state,
compute the interacting Green's function, and invert to the self-energy. It is the wall-clock
hotspot of the whole package and is MPI-collective end to end. We want a **reproducible,
attributable benchmark** that pinpoints where the time goes, and a clear rubric for reading the
numbers. Only *after* the benchmark data is in hand do we commit to a performance-improvement
plan (Part B below is the framework that data will populate/prune — it is explicitly not yet
scheduled work).

This plan honors three confirmed choices: **(1)** serial attribution first, then an MPI
strong-scaling sweep; **(2)** a committed, house-style benchmark harness (cf. the oracle-first
Phase-0 convention in `doc/plans/manybodyoperator_apply_performance.md` and
`src/impurityModel/test/test_apply_perf.py`); **(3)** the anchor workload is
`h0/h0_NiO_100bath.pickle` (NiO d-shell, `ls=2`, 100 bath states).

## Cost map (from reading the call graph)

`calc_selfenergy` decomposes into:

1. **Hamiltonian build** — `ManyBodyOperator(h0) + ManyBodyOperator(u)` (`selfenergy.py:342`). Cheap, one-off.
2. **`calc_gs`** (`groundstate.py:409`) — almost certainly a top-2 cost. Sub-phases:
   - `find_ground_state_basis` — CIPSI prescan / sector discovery.
   - `CIPSISolver.expand` (`cipsi_solver.py:89`) — iterative basis growth; each iteration does a
     `ManyBodyOperator.apply` matvec + an eigensystem solve (dense below `dense_cutoff`, else
     IRLM/TRLM block-Lanczos).
   - `CIPSISolver.get_eigenvectors` (`cipsi_solver.py:218`) — final eigensolve.
   - `build_density_matrices` (distributed: apply→`redistribute_psis`→local inner→`Allreduce`).
   - `compute_gs_statistics` + optional observable report (collective).
3. **`get_Greens_function`** (`greens_function.py:104`) — the other top-2 cost. Block-Lanczos
   continued-fraction GF (`block_Green_sparse`/`block_green_impl`), one Lanczos run per thermal
   state per inequivalent block; the `apply_multi` matvec is the inner hot loop.
4. **`get_sigma`** (`selfenergy.py:589`) — dense `hyb` + `np.linalg.inv` over the frequency mesh.
   Expected cheap (small block × mesh); to be confirmed, not assumed.
5. **`get_Sigma_static`** (`selfenergy.py:637`) — small dense contraction. Negligible.

The single second-quantized matvec `ManyBodyOperator::apply` underlies CIPSI expand, both
eigensolves, **and** the GF Lanczos — so it is the common denominator the benchmark must
attribute time to (an apply-perf plan already exists: `doc/plans/manybodyoperator_apply_performance.md`).

---

## Part A — The benchmark (execute now)

### A1. Harness (committed, non-invasive)

Add `src/impurityModel/test/test_selfenergy_perf.py`, modeled on `test_apply_perf.py`:

- **Driver:** call `selfenergy.get_selfenergy(...)` with `h0_filename="h0/h0_NiO_100bath.pickle"`,
  `ls=2`, `nBaths=100` (set `nValBaths` to match the pickle's bath layout), `tau=0.002`,
  `delta=0.2`, `verbose=False`. `get_selfenergy` already wires the NiO d-shell, reads the pickle,
  and calls `calc_selfenergy` — reuse it rather than re-deriving the setup.
- **Coarse phase timers (non-invasive):** wrap the five phase entry points
  (`calc_gs`, `get_Greens_function`, `get_sigma`, `get_Sigma_static`, and the `ManyBodyOperator`
  Hamiltonian build) with `time.perf_counter` via `unittest.mock.patch` wrappers that call through
  and accumulate elapsed time per phase. No edits to production code. Print a per-phase table
  under `pytest -s` (same style as `spectra.py`'s `time(PS)=…` prints).
- **Fine attribution:** run the driver under `cProfile`, dump `*.pstats` to the scratchpad, and
  print the top ~30 entries sorted by cumulative and by tottime. This gives function-level
  attribution (`apply`, `block_lanczos_step_cy`, `selective_orthogonalize`, `redistribute_psis`,
  `Allreduce`, `np.linalg.inv`) for free.
- **CI guard:** the 100-bath full solve is far too heavy for default CI. Gate the body behind an
  opt-in env var (e.g. `RUN_SELFENERGY_BENCH=1`) and `pytest.mark.skipif` otherwise — exactly the
  "small in CI, large under `-s`/opt-in" split `test_apply_perf.py` uses. The file is committed and
  re-runnable so later speedups can be validated against the same harness.

### A2. Serial run (attribution)

`RUN_SELFENERGY_BENCH=1 pytest -s src/impurityModel/test/test_selfenergy_perf.py`

Collect: (a) the per-phase wall-clock split; (b) the cProfile cumulative/tottime tables;
(c) reported basis sizes (number of Slater determinants after CIPSI), number of thermal states
considered (`num_wanted`/`len(es)`), and Lanczos iteration counts if surfaced. Run 2–3 reps to
confirm stability (the matvec is deterministic; wall-time variance should be small).

### A3. MPI strong-scaling sweep

Re-run the same harness under `mpirun -n {1,2,4} pytest -s --with-mpi …` (rank 0 prints the
per-phase table; cProfile per-rank dumps to distinct files). Record per-phase wall-clock at each
rank count. The fixed problem (100-bath) shrinks per-rank as ranks grow → strong scaling.

### A4. How to interpret the results

**Step 1 — phase split (from A2).** Which of {`calc_gs`, `get_Greens_function`, `get_sigma`}
dominates? Expected: the two Lanczos-driven phases dwarf `get_sigma`. If `get_sigma` is
non-trivial, that is a surprise worth a dedicated look (mesh size × block × inv).

**Step 2 — within the dominant phase (cProfile cumulative).** Attribute to one of:
- **Matvec-bound** — `ManyBodyOperator.apply`/`apply_multi` dominates tottime. → optimization
  family **B1** (and the existing apply-perf plan) is the highest-leverage lever, because it speeds
  CIPSI, eigensolve, and GF simultaneously.
- **Lanczos-kernel-bound** — `block_lanczos_step_cy`, `selective_orthogonalize`,
  `estimate_orthonormality`, CholeskyQR show up large *outside* `apply`. → family **B2**
  (BLAS acceleration / reort cost).
- **Basis-growth-bound** — many CIPSI `expand` iterations, basis size explodes, `apply` called an
  excessive number of times. → family **B3** (CIPSI growth/truncation tuning).
- **Comm/redistribution-bound** — `redistribute_psis`, `Allreduce`, `Allgather`,
  `build_density_matrices`, `compute_gs_statistics` are large. → family **B4**.

**Step 3 — scaling (from A3).** Compute speedup `S(n) = T(1)/T(n)` per phase.
- Near-linear (`S(4)≈3–4`): phase is compute-bound and parallel-healthy; optimize its serial cost
  (B1/B2/B3).
- Sub-linear / flat: comm- or load-balance-bound. Cross-check with the per-rank cProfile spread —
  large variance across ranks ⇒ load imbalance from the splitmix64 hash distribution; uniform but
  high `Allreduce`/`bcast` time ⇒ comm volume. → family **B4**.
- Anti-scaling (slower with more ranks): a serial section or per-rank fixed cost is being paid
  redundantly (e.g. dense fallback below `dense_cutoff`, replicated dense buffers). Flag the
  specific call.

**Step 4 — sanity cross-checks.** Confirm basis size and `num_wanted` are representative (no
runaway retry loop at `selfenergy.py:364`); confirm the GF mesh (2000 real-axis points) is the
intended production size; confirm no time lost in the unphysical-GF `check_greens_function`
re-save paths.

### A5. Deliverable of Part A

A short results section appended to **this file**: the per-phase table (serial + per rank count),
the top cProfile entries, and a one-line verdict per Step-2/Step-3 category naming the dominant
bottleneck(s). That verdict selects which Part-B families get scheduled.

---

## Part B — Improvement plan framework (populated after Part A)

Candidate optimization families, with interdependencies marked. **Which of these become real
work — and in what order — is decided by the A4 verdict; do not pre-commit.**

- **B1 — Matvec throughput (`ManyBodyOperator::apply`).** Foundational and *independent* of the
  Lanczos kernels (different code: C++ operator vs Cython drivers). Already has a detailed,
  oracle-gated plan (`doc/plans/manybodyoperator_apply_performance.md`). Speeding `apply` benefits
  CIPSI + eigensolve + GF at once. **Depends on:** its own Phase-0 golden oracle (exists).
  **Independent of:** B2, B3, B4.

- **B2 — Block-Lanczos kernel.** Real BLAS in the array kernel
  (`doc/plans/blocklanczos_blas_acceleration.md`) and reort cost/reliability
  (`doc/plans/blocklanczos_reort_reliability.md`). **Depends on:** the reort-reliability Phase-0
  oracle (per `doc/plans/README.md` reading order — BLAS work is gated on it). **Independent of:**
  B1 (different code). Relevant only if Step 2 shows Lanczos-kernel-bound (not matvec-bound).

- **B3 — CIPSI basis growth / truncation.** Tune `de2_min`, `slaterWeightMin`, `dense_cutoff`,
  prescan, and the `num_wanted` retry policy to cut the number of matvecs / basis size.
  Algorithmic. **Interacts with B1** (fewer matvecs vs faster matvec — measure B3's effect *after*
  B1 lands, or hold B1 fixed). Numerically sensitive → must be guarded by a results-equivalence
  check (self-energy within tolerance of the baseline harness output).

- **B4 — MPI comm / load balance.** Reduce `redistribute_psis` / `Allreduce` volume; address hash
  distribution skew; avoid redundant per-rank dense work. **Only scheduled if A3 shows sub-/anti-
  scaling.** Largely **independent of** B1/B2/B3 (orthogonal axis), but its benefit is measured on
  the *same* harness so it must be sequenced so its before/after is attributable (don't land B4 and
  B1 in the same measurement window).

- **B5 — GF/`get_sigma` dense path.** Only if Step 1 shows `get_sigma` non-trivial: vectorize
  `hyb`/`inv` over the mesh, or reduce mesh density. Self-contained, independent of all the above.

**Sequencing rule of thumb (to be confirmed by data):** land B1 first if matvec-bound (broadest
leverage, independent, oracle already exists); B2 needs its reort oracle first; treat B3 and B4 as
separate measurement windows so each speedup is independently attributable on the committed harness.

---

## Files

- **New:** `src/impurityModel/test/test_selfenergy_perf.py` (the harness; the only code added in Part A).
- **Read/driver (unchanged):** `selfenergy.py` (`get_selfenergy`/`calc_selfenergy`),
  `groundstate.py` (`calc_gs`), `greens_function.py` (`get_Greens_function`),
  `cipsi_solver.py` (`expand`/`get_eigenvectors`).
- **Input:** `h0/h0_NiO_100bath.pickle`.
- **Existing perf plans referenced by Part B:** `doc/plans/manybodyoperator_apply_performance.md`,
  `doc/plans/blocklanczos_blas_acceleration.md`, `doc/plans/blocklanczos_reort_reliability.md`.

## Verification

- Harness runs end-to-end serially: `RUN_SELFENERGY_BENCH=1 pytest -s src/impurityModel/test/test_selfenergy_perf.py`
  prints a per-phase table and cProfile top-N without error.
- Harness runs under MPI: `RUN_SELFENERGY_BENCH=1 mpirun -n 2 pytest -s --with-mpi src/impurityModel/test/test_selfenergy_perf.py`
  (and `-n 4`) completes; rank-0 table + per-rank pstats produced.
- Default CI is unaffected: plain `pytest src/impurityModel/test/test_selfenergy_perf.py` skips
  (guard env var unset).
- Results (per-phase split serial + at n=1,2,4, cProfile top entries, bottleneck verdict) appended
  to this file as the Part-A deliverable, which then selects the Part-B families to schedule.

---

# 100-BATH GF CONVERGENCE — DIAGNOSED + ENABLED (2026-06-30)

**Goal:** make the 100-bath GF converge so calc_selfenergy benchmarks at scale.

**Two distinct failures (both had to be addressed):**

1. **Reort fragility → divergence.** With `Reort.NONE` (the GF default) *and* `Reort.PARTIAL`, the
   block-Lanczos loses orthogonality at 100-bath: ghost components appear, the matvec spreads them
   to spurious determinants (a positive-feedback **basis explosion**), and `|beta|` blows up
   (~1e6 by block ~45) → the divergence guard truncates with a "divergent tail" → wrong/garbage GF.
   PARTIAL's *selective* reort (W-estimator-gated) doesn't catch the loss fast enough at this scale.
   **`Reort.FULL` is robust** — orthogonalising the residual against the whole basis every step keeps
   `|beta|` bounded (~10) with **zero truncations**, breaking the ghost feedback. (B2's dense reort
   makes FULL affordable.)

2. **Unbounded excited sector → memory explosion.** Independently, the GF excited-state restrictions
   are effectively absent with `dN=None` (calc_selfenergy's default): `build_excited_restrictions`
   gets `imp/val/con_change=None` → no occupation window → the reachable N±1 sector at 100-bath is
   enormous, so even with FULL reort one GF seed's Krylov space grows past 100 blocks / >8 GB toward
   OOM. **Setting `dN` bounds the sector** to the physical N±1 window. Tradeoff: `dN=1` is tractable
   (~1 GB, completes) but the self-energy is **unphysical** (sector too tight → `Σ=G0⁻¹−G⁻¹` gets a
   positive-Im diagonal); `dN=2` is physical-er but one seed still grinds toward OOM (~8 GB, did not
   complete in 15 min). So a *physically accurate* 100-bath Σ is memory-bound; a *performance*
   benchmark is fine at `dN=1`.

**Enabled (harness):** `SELFENERGY_BENCH_{REORT,DN,OCC_CUTOFF,SWMIN,ALLOW_UNPHYSICAL}` env knobs.
With `REORT=full DN=1 ALLOW_UNPHYSICAL=1` the **100-bath solve completes**:

| | wall | GF phase | max\|beta\| | truncations |
|--|----:|---------:|----------:|------------:|
| 100-bath, FULL, dN=1 | **375 s** | 99.7% | ~10 (bounded) | 0 |

Per-op split: matvec ~48%, recurrence(+FULL reort) ~51%, monitor 1.6%. So at scale the GF is the
whole cost and matvec ≈ recurrence/reort — pointing at **B1 (`ManyBodyOperator::apply`)** and the
FULL-reort cost as the next levers. (`ALLOW_UNPHYSICAL` swallows the dN=1 Σ-physicality failure so
the *timing* is reported; it is a performance benchmark, not a converged-physics result.)

**Verdict / recommendations:**
- The GF should **not** default to `Reort.NONE` at scale — it silently diverges. A robust default
  (`FULL`, or auto-escalate to FULL when the divergence guard trips) is the correctness fix; B2 made
  it affordable. (Policy change — left for sign-off; not imposed.)
- `calc_selfenergy`/`get_Greens_function` should set a **sensible `dN`** for large bath counts to
  bound the excited sector (production's `dN=None` is a latent scaling bug).
- A *physically accurate* 100-bath Σ is intrinsically memory-heavy (large N±1 sector); routine
  accurate work at that scale needs methodology changes (tighter restriction that stays physical, a
  Krylov method that doesn't grow the determinant basis, or a smaller fitted bath) — beyond a
  convergence fix.

---

# PART A2 — REORT.NONE vs REORT.PARTIAL BENCHMARK (post-fix, run 2026-06-30)

Done with the committed harness (`SELFENERGY_BENCH_REORT` knob), production `dense_cutoff=500`,
`NW=200`. Self-energy is **identical across every cell** (NONE vs PARTIAL and serial vs MPI agree to
~1e-13 relative — `‖σ‖=1957.617940220…`), so reort mode and rank count change only *performance and
the convergence path*, never the physics.

## Anchor caveat: 100-bath is GF-bound and a clean 6-cell matrix is not feasible

With the fixes, 100-bath `calc_selfenergy` **runs to completion**, but the Green's-function phase is
so expensive that a full NONE×PARTIAL×{1,2,4} matrix doesn't finish in a reasonable window:
- **NONE @ 100-bath:** the block-Lanczos needs *enormous* Krylov spaces — one GF invocation passed
  **923 blocks** with `|beta|`≈1900 still climbing and had not converged after ~15 min (serial).
- **PARTIAL @ 100-bath:** blocks stay bounded (~49-77), but some invocations suffer a **basis
  explosion** (each Lanczos step ~100 s as the discovered-determinant set blows up).

So the clean, *complete, comparable* matrix below is at **10-bath**, with **20- and 100-bath used
for the block-count scaling trend**. The headline (GF dominates; NONE's cost explodes with system
size while PARTIAL's per-block count stays bounded) is consistent across all three sizes.

## Phase split (robust at every cell): the GF is everything

`get_Greens_function` is **97.7-99.7%** of wall time in all 6 cells; `calc_gs` 0.3-2.3%;
`get_sigma`/`get_Sigma_static` ~0%. The self-energy bottleneck *is* the block-Lanczos Green's
function. (B5 — get_sigma — remains correctly dropped.)

## Wall-time matrix — NiO 10-bath, NW=200 (seconds)

| ranks | REORT.NONE | REORT.PARTIAL |
|------:|-----------:|--------------:|
| n=1   |     **43.6** |       **294.0** |
| n=2   |      139.9 |         235.1 |
| n=4   |       69.7 |         257.7 |

**Serial (n=1) is the only apples-to-apples cell** (see MPI caveat below). There, **NONE is 6.7×
faster than PARTIAL** at 10-bath.

## Why — measured per-step cost (serial n=1)

Counting total block-Lanczos steps across all GF invocations and the wall time:

| mode | total Lanczos steps | wall (s) | **ms / step** |
|------|--------------------:|---------:|--------------:|
| NONE    | 691 | 43.6  | **63**  |
| PARTIAL | 427 | 294.0 | **688** |

The decomposition is the opposite of "PARTIAL just does fewer iterations": **PARTIAL converges in
~38% fewer steps** (better-conditioned recurrence) **but each step is ~11× more expensive**, and the
per-step penalty wins → 6.7× slower overall at 10-bath.

## Per-step split (measured — env-gated Cython timers, `BLOCKLANCZOS_PROFILE=1`)

cProfile lumps everything into `block_Green_sparse` (the inner kernels are Cython, no Python
frames), so I added per-op `perf_counter` accumulators in `block_lanczos_step_cy` /
`block_lanczos_cy` (`get_block_lanczos_profile()`). Serial 10-bath, fraction of GF kernel time:

| op | NONE | PARTIAL |
|----|-----:|--------:|
| matvec (`apply_multi`)            | **25.9%** (16.2 s) | 3.3% (9.3 s) |
| recurrence LA (α, M, CholeskyQR)  | 20.6% (12.9 s) | 3.7% (10.4 s) |
| CholeskyQR2 (conditional)         | 0.2% | 0.0% |
| W-estimator (`estimate_orthonormality`) | — (unused) | 0.3% (0.9 s) |
| **triggered reort** (`apply_reort` projections) | — (unused) | **91.1% (253 s)** |
| **convergence monitor** (`converged_fn`)        | **53.3% (33.2 s)** | 1.4% (4.0 s) |

**NONE is monitor-bound: 53%.** The convergence monitor (`_gf_converged_mesh` +
`_greens_function_change` → `_block_cf_inverse`) runs every step and costs ~34 ms/call —
*more* than matvec (26%) + recurrence (21%) combined. (cProfile only saw the `np.linalg.solve`
sliver, ~24%; the full monitor incl. mesh/`_gf_diag_range`/cache is 53%.)

**PARTIAL is reort-bound: 91%** — and the split confirms the correction. The W-estimator *bookkeeping*
is trivial (0.3%); the cost is the **block projections** `apply_reort` performs against the
flagged blocks. Crucially the trigger fires on **431/443 steps (97%)**: on this near-degenerate GF
spectrum orthogonality degrades almost every step, so the "selective" reort effectively reorthogonalizes
nearly constantly (close to FULL behavior) — *not* because it projects against the whole basis, but
because it is triggered nearly every step. The monitor is cheap for PARTIAL (1.4%) because it runs
on fewer/shorter continued fractions (fewer blocks).

So the ~11× per-step gap (NONE 63 ms vs PARTIAL 688 ms) is **entirely the triggered reort
projections**, which the matvec/recurrence (nearly identical between modes) are dwarfed by.

## Block-count scaling with system size (the crossover)

Max blocks reached by a single GF invocation:

| bath | NONE max-blocks | PARTIAL max-blocks |
|-----:|----------------:|-------------------:|
| 10   |   142           |  96                |
| 20   |   241           |  ~49-82            |
| 100  |   **923+** (not converged) | ~49-77 |

NONE's Krylov requirement **grows steeply** with bath size (and `|beta|` with it: ~90 @10-bath →
~1900 @100-bath), while PARTIAL's stays **roughly flat**. NONE's cheap-per-step advantage is
eaten by block growth as the system grows — so the modes **cross over**: NONE wins small, PARTIAL is
the only viable mode at production size (where NONE effectively does not converge).

## MPI sweep — measured but NOT representative (two confounds)

The 10-bath MPI numbers (n=2/n=4 above) do **not** give clean strong-scaling:
1. **Too small** — at 10-bath the per-rank compute is dwarfed by per-step `Allreduce`/redistribute
   latency, so adding ranks adds overhead (NONE: 44→140→70 s, non-monotonic).
2. **Different work per rank count** — the per-rank CIPSI random seed (`cipsi_solver.expand`,
   `seed(42+rank)`) changes the adaptively-built basis trajectory and hence the thermal-state set,
   so each rank count solves a *different* number of GF invocations (12 @n=1, 6 @n=2, 4 @n=4) with
   different block counts. The converged self-energy is identical, but the *work* is not — timings
   aren't comparable across ranks.

A meaningful strong-scaling sweep needs the production (100-bath) size, which is currently GF-bound
(above). **Deferred** until the GF cost is reduced (B6/B1/B2).

## Verdict (refines Part A)

- **GF phase is the whole game** (~98%+). The per-step split (measured) makes the two levers precise
  and **mode-specific**:
  - **B6 — the convergence monitor — is the #1 lever for NONE (53% of its kernel time).** It runs
    every step at ~34 ms/call (`_gf_converged_mesh` + `_greens_function_change` → `_block_cf_inverse`).
    Make it incremental (extend the resolvent rather than re-inverting) and/or check every *k* steps
    instead of every step, and terminate earlier. Helps NONE enormously and helps PARTIAL a little.
  - **B2 — the triggered reort projections — is the #1 lever for PARTIAL (91%).** The W-estimator
    bookkeeping is negligible (0.3%); the cost is `apply_reort`'s block projections, which fire on
    **97% of steps** here. Two angles: (i) reduce the per-projection cost (BLAS / fewer flagged
    blocks), and (ii) **why does it trigger ~every step?** — the near-degenerate GF spectrum trips
    `REORT_TOL` constantly, so PARTIAL is doing near-FULL work; a smarter trigger or a
    better-conditioned seed/spectrum handling could cut the trigger rate.
- **Reort choice is a real trade-off, not a free default:** NONE step 63 ms but step count explodes
  with system size (max blocks 142→241→923+ for bath 10→20→100); PARTIAL step 688 ms (reort) but a
  roughly flat step count. They **cross over** — NONE wins small, PARTIAL is the only mode that
  converges at 100-bath.
- **Highest-leverage single fix: B6.** The monitor is per-step in *both* modes; cutting its cost and
  frequency shrinks NONE's dominant cost directly and lets *both* modes terminate sooner (fewer steps
  → less reort for PARTIAL too).
- `calc_gs` and `get_sigma` remain non-priorities.

---

# B6 — GF CONVERGENCE MONITOR (IMPLEMENTED, 2026-06-30)

**Goal:** cut the convergence monitor, which the per-step split measured at **53% of the GF kernel
for reort=NONE** (it rebuilds an O(k)-level block continued fraction every block → O(k²) per GF
invocation).

**Failed approaches (measured, discarded):** simply checking every N blocks, or switching the
measure to a stricter "drift since last check," both *added Lanczos steps* — the monitor also
*terminates* the recurrence, so checking it less often delays convergence, and the extra (more
expensive) Lanczos steps cancelled the monitor saving (net **slower**: e.g. drift+gate ran 1148
steps vs 691 baseline and was 45→47 s vs 43.6 s).

**Shipped: adaptive sampling.** Sample the test only every `_GF_CHECK_EVERY` (=8) blocks during the
long "building" plateau — where the relative change still sits a decade+ above tolerance and
convergence is impossible — then switch to **every block** once a check lands within
`_GF_NEAR_FACTOR` (=2) × tol, so the exact convergence point (and the `_GF_CONSEC_CONVERGED` gate)
is caught with **no added steps**. Same measure, same tolerance. Both knobs are env-overridable
(`GF_CHECK_EVERY`, `GF_NEAR_FACTOR`; `GF_CHECK_EVERY=1` restores the old every-block behavior).
(`greens_function.py` `_make_gf_convergence_monitor`.)

**Result — controlled A/B on a pinned basis (`PYTHONHASHSEED=0`, NiO 10-bath, reort=NONE):**

| | wall | Lanczos steps | monitor share |
|--|----:|--------------:|--------------:|
| baseline (every block) | 66.5 s | 987 | 53.2% |
| **B6 (check=8, near=2)** | **50.2 s** | 1017 (+30) | **33.6%** |

**~24% faster, monitor 53→34%, +30 steps, and the self-energy is bit-identical** (rel ≈ 7e-18 —
convergence triggers at the same block because the near-convergence phase is sampled densely).

**Validation:** self-energy unchanged (bit-identical on a pinned basis; rel ~1e-12 run-to-run);
**serial == MPI(n=2) preserved** (sigma rel 6.6e-13 — the gate decision uses only Allreduced data,
so it is collective-safe and does not reintroduce the B0.3 divergence); GF/eigensolver/groundstate
suite green (54 serial + 11 MPI). The benefit **scales with system size**: at 100-bath the building
plateau spans most of ~900 blocks and the per-call CF is O(900) levels, so the sparse phase saves
far more there than at 10-bath.

*Note on measurement:* run-to-run wall comparison at 10-bath is unreliable — `PYTHONHASHSEED`
nondeterminism in the CIPSI determinant sets swings the basis trajectory (687–1072 steps for the
"same" config, ~40% wall swing). All A/B numbers above pin `PYTHONHASHSEED=0` so the two cells do
identical Lanczos work; the monitor share and step count are the robust quantities.

**Next lever:** **B2** — PARTIAL's reort (91% of its kernel) — now done (dense-BLAS projection,
5.1× on the reort); see the B2 section below.

---

# B2 — PARTIAL REORTHOGONALIZATION (INVESTIGATED, 2026-06-30)

**Constraint (from the user):** the Paige–Simon W-estimator must remain an *upper* bound on the
inter-block overlap (it must never under-predict, or needed reorth is skipped → lost orthogonality →
ghosts/divergence; cf. the "partial reort estimator under-prediction" history). So B2 may only make
the *projection* cheaper, not weaken the estimator or the trigger.

**Diagnosis (measured, env-gated `apply_reort` counters → `get_reort_profile()`):** on the
near-degenerate GF spectrum the reort is **barely selective** — per acting call it flags **20.6 of
35.7 blocks (58%)**, i.e. ~206 columns, and fires on **97% of steps**. So it is effectively *full*
reorthogonalization: the block *count* cannot be reduced without weakening the estimator (those
blocks genuinely lost orthogonality). The cost is the sheer volume of the map-based projection:
`inner_multi(Q_bad, wp)` (≈206×10 `flat_map` inner products) + `add_scaled_multi` (10×206 map
merges), ×2 (CGS2), per step. (`inner()` already iterates the smaller map — it is not the lever.)

**Tried and rejected — OpenMP threading of the block primitives.** `inner_multi` / `add_scaled_multi`
are `nogil` loops over independent outputs, so threading them with `prange` is race-free and
bit-identical. But the per-call OpenMP fork/join overhead on these medium loops (and especially
`add_scaled_multi`'s 10-iteration target loop) *exceeded* the gain: controlled A/B on a pinned basis
(PARTIAL 10-bath) gave **OMP=1: reort 230.5 s vs OMP=8: 252.9 s — slower** (result rel 1.2e-12).
Fine-grained per-call parallelism is the wrong tool here; **reverted.** (The env-gated reort
fan-out instrumentation is kept for the next attempt.)

**Shipped: dense-BLAS projection.** New `ManyBodyUtils.reorth_cgs2_dense(wp, Q, n_passes, comm)`
materializes `wp` and the flagged `Q_bad` columns onto their merged (local) determinant support,
runs the two CGS2 passes as `O = Qᴴ wp` (Allreduced) / `wp -= Q O` via BLAS `zgemm`, and scatters
`wp` back to `ManyBodyState`s. `apply_reort` (`BlockLanczosArray.pyx`) calls it for the sparse path
(both the FULL and PARTIAL/SELECTIVE branches); the array path is unchanged. The **W-estimator and
bad-block selection are untouched** — only the projection arithmetic changes (hash-map inner/merge →
dense BLAS).

**Result — controlled A/B on a pinned basis (`PYTHONHASHSEED=0`, PARTIAL 10-bath):**

| | reort time | wall |
|--|----------:|-----:|
| map-based (before) | 230.5 s (92.8%) | 252 s |
| **dense BLAS (after)** | **45.4 s (60.8%)** | **79.3 s** |

**reort 5.1× faster, total 3.2× faster.** Bit-identical gate: vs the map-based PARTIAL on the *same*
pinned basis, self-energy **rel 1.18e-12** (max|diff| 7e-11); vs the NONE baseline rel 5.2e-14.
serial == MPI(n=2) preserved (rel 6.5e-13 — the per-pass `Allreduce` mirrors the map path, so it
stays collective-safe). Suite green: 82 serial + 13 MPI + the `apply` golden oracle. Empty-rank /
empty-column / empty-support cases short-circuit to a no-op.

**Tried first and rejected: OpenMP threading** of `inner_multi`/`add_scaled_multi` — bit-identical
but net *slower* (per-call fork/join overhead; OMP=8 reort 253 s vs OMP=1 231 s). Reverted; the
algorithmic dense-BLAS win is far larger and single-threaded.

**Follow-up shipped: dense Krylov basis (removes the per-call gather).** New Cython
`SparseKrylovDense` (`ManyBodyUtils.pyx`) keeps a rank-local dense copy of the block-Krylov basis
(`det→row` `std::map` support + a doubling `(rows×cols)` buffer); `block_lanczos_cy` seeds it and
`append`s each block in lockstep with `Q_basis`, and threads it through `block_lanczos_step_cy` →
`apply_reort`, which now *slices* `Q[:, bad_cols]` instead of re-materializing `Q_bad` from
`flat_map`s every step. Only `wp` (10 cols) is materialized/scattered per call; the W-estimator /
selection are still untouched; collective footprint unchanged (the per-pass `(n_cols×p)` overlap
`Allreduce`).

**Result (pinned basis, PARTIAL 10-bath), reort time across the three implementations:**

| reort impl | reort time | wall |
|--|----------:|-----:|
| map-based | 230.5 s | 252 s |
| dense BLAS, per-call gather | 45.4 s | 79.3 s |
| **dense BLAS + dense Krylov** | **21.8 s** | **57.7 s** |

**Cumulative: reort 230.5 → 21.8 s (10.6×), wall 252 → 58 s (4.4×).** The dense-Krylov result is
**bit-identical** to the per-call dense on the same basis (rel 2.97e-17) and matches NONE to 5.2e-14;
serial == MPI(n=2) preserved; suite green (95 serial + 13 MPI + apply golden). The reort is no longer
the dominant GF cost (41.9% now; matvec 28%).

---

# GF PARALLELIZATION — EIGENSTATE-GROUPING KNOB (IMPLEMENTED + SWEPT, 2026-06-30)

**Context.** The GF parallelization runs one block-Lanczos recurrence per thermal eigenstate at
width = n_ops (the block's transition operators), split over two nested communicators
(blocks → eigenstates). The open question was whether **wider blocks** (stacking eigenstates) and/or
splitting the transition operators help. Off-diagonal `G_ij` forces all n_ops orbital columns into
one block, so the only tunable axis that preserves the off-diagonals is **how many eigenstates share
one recurrence**. Wider blocks share the Krylov/matvec build but grow per-step reort with width², so
the optimum is workload-dependent — hence a knob + benchmark rather than a fixed choice.

**Shipped (the knob).** `GF_EIGENSTATE_GROUP=g` (`greens_function.py`, `_gf_eigenstate_group` +
`calc_Greens_function_with_offdiag`). `g=1` is the historical per-eigenstate behavior (default,
exactly reproduced). `g>1` stacks `g` co-located eigenstates' seeds into one width-`g·n_ops`
block-Lanczos recurrence sharing a Krylov space; the shared `(alphas, betas)` are reused per
eigenstate while each keeps its own `n_ops` columns of the seed projection `r` (`r[:, p*n_ops:(p+1)*n_ops]`)
and its own energy shift, so `calc_G` reconstructs each eigenstate's `n_ops×n_ops` block exactly.
Grouping happens *after* the eigenstate split, on the eigenstates already co-located on a color, so it
needs **no extra communication** and **no kernel change** (`block_lanczos_cy`/`block_lanczos_array`
already accept arbitrary seed width).

**Correctness (oracle).** Two new equivalence tests in `test_greens_function.py`
(`..._eigenstate_grouping`, `..._eigenstate_grouping_with_bath`) assert `g=2` reproduces `g=1` — a
mathematical identity — on a closed no-bath system (dense + sparse paths) and on a hybridizing-bath
system whose excited basis grows (sparse, the production path), to ~1e-6/1e-5. Full GF suite green
serial (25) and MPI n=2 (9); `test_selfenergy` green serial default + `g=2` (16) and MPI n=2 `g=2`
(16). Black-clean (120 col).

**Phase-3 sweep (NiO 10-bath, NW=200, serial, `PYTHONHASHSEED=0` so the 10 thermal states are
identical across cells).** σ is **FP-identical** across every converged cell (‖σ‖=1957.6179; rel
≤1.4e-13 vs the g=1 baseline) — grouping never changes the physics, only performance and stability.

| reort | g | wall (s) | GF (s) | matvec calls / time | recurrence(+reort) (s) | monitor (s) | converged? |
|------:|--:|---------:|-------:|--------------------:|-----------------------:|------------:|:-----------|
| NONE  | 1 | **28.1** | 26.1 | 714 / 6.8 | 11.0 | 4.3 | yes (712 blk) |
| NONE  | 2 | >900 (killed) | — | — | — | — | **diverges** (\|β\|→1.3e4 @1384 blk) |
| FULL  | 1 | 149.2 | 85.0 | 580 / 19.1 | 52.1 | 4.7 | yes |
| FULL  | **2** | **86.2** | 62.0 | 170 / 6.6 | 45.6 | 2.3 | yes |
| FULL  | 4 | 141.6 | 111.7 | 123 / 9.4 | 84.7 | 4.0 | yes |
| FULL  | 6 | 337.8 | 271.2 | 46 / 12.4 | 155.1 | 24.0 | yes |

**Two regimes, opposite answers:**

1. **NONE reort (only viable for small/cheap systems): keep `g=1`.** Stacking *destabilizes* the
   knife's-edge `Reort.NONE` recurrence — orthogonality loss grows with block width, so the width-20
   `g=2` block loses orthogonality far faster than the width-10 `g=1` block, `|β|` blows up to ~1.3e4
   and it never converges, even at 10-bath where `g=1`/NONE converges cleanly in 712 blocks. (Same
   mechanism as B0.3, amplified by width.) So with NONE the cheap narrow recurrence is the *only*
   usable one; wider blocks are not merely slower, they are unstable.

2. **FULL reort (mandatory at production scale, where even `g=1`/NONE diverges — see the 100-bath
   section): grouping wins, with a clear optimum at `g=2` (1.73×).** Once full reort is paid for, the
   shared Krylov space is pure upside: matvec **calls** fall 580→170→123→46 as `g` rises (fewer,
   fatter, shared recurrences), and at `g=2` *every* component drops (matvec 19.1→6.6 s, recurrence+
   reort 52.1→45.6 s, monitor 4.7→2.3 s). Beyond `g=2` the per-step width²/width³ LA (FULL reort,
   Gram/CholeskyQR, and the monitor's continued-fraction inversion — 24 s at width-60!) overtakes the
   diminishing step-count savings, so wall turns back up (g=4 142 s, g=6 338 s). The 3.4× step-count
   drop g=1→g=2 vs only 1.4× g=2→g=4, against a 4× per-step reort growth, puts the minimum at g=2.

**Verdict / recommended default policy (data-backed Phase-4 adaptive rule):**
- Use **`g=1`** whenever the GF runs with `Reort.NONE` (small systems where NONE converges) — it is
  both fastest (28 s) and the only stable choice there.
- Use **`g≈2`** whenever `Reort.FULL` is in force — i.e. at the scale where NONE diverges and full
  reort is already mandatory. There grouping is a free ~1.7× (serial), reusing the reort you must pay
  anyway. `g≳4` is counter-productive; `g=n_states` (here g=10, width-100) is pathological.
- These are **serial** numbers. Grouping also cuts the matvec **call** count 3.4× (580→170 at g=2),
  i.e. 3.4× fewer per-step `Allreduce` rounds, so the wide-block benefit should *grow* under MPI
  (latency amortization) — the serial 1.7× is a conservative lower bound on the distributed gain.
  (A clean MPI sweep still needs the 100-bath anchor; 10-bath MPI is latency-confounded, see above.)

**Still open (not done):** the optional pairwise-scalar operator split (the narrow end) remains
future work; the eigenstate-group knob already answers the headline "are wider blocks worth it?" —
yes, moderately, in the FULL-reort regime that production runs in. *(Now shipped — see the
operator-split section below.)*

---

# GF PARALLELIZATION — UNIFIED SINGLE SPLIT (Phase 1, IMPLEMENTED, 2026-06-30)

**Context.** `get_Greens_function` used a *nested* two-level communicator split: first over
Green's-function blocks (`split_basis_and_redistribute_psi([len(block)**2 …])`), then a *second*
split over thermal eigenstates inside each block's color (in `calc_Greens_function_with_offdiag`).
With many small symmetry blocks (the typical production case — block sizes ~1–4), the hierarchical
rank carving balances poorly and pays two rounds of comm/intercomm creation.

**Shipped.** `get_Greens_function` now does **one** global split over the full
`(block × addition/removal × eigenstate-group)` work-unit cross-product. It applies the transition
operators for every block and both spectral sides up front (collective on the full basis), enumerates
the units, weights each by `log10(excited seed length)+1` (the old eigenstate-split heuristic, now
applied globally), and calls `split_basis_and_redistribute_psi` **once**; each color runs its units'
block-Lanczos recurrences and the results are gathered and reassembled per block. The excited-sector
restrictions are built **once** (they are block- and side-independent — the dN window is symmetric
over all impurity orbitals) instead of per block. Three shared helpers
(`_build_excited_restrictions`, `_apply_transition_ops`, `_block_green_group`) are factored out so
`calc_Greens_function_with_offdiag` (still used by `spectra.py` and its own tests) and the new
`get_Greens_function` share identical seed-building and block-Green logic. The eigenstate-grouping
knob (`GF_EIGENSTATE_GROUP`) composes with the unified split unchanged.

**Correctness.** New `test_get_Greens_function_split_threshold_invariant_mpi` asserts the per-block GF
is identical under maximal split (`split_threshold=1e9`) vs a unified communicator
(`split_threshold=0`) for a multi-block / multi-eigenstate system — the invariant the refactor must
preserve. Suites green: GF + selfenergy serial (34) and broader GF serial (43); MPI **n=2, 3, 4** (26
each), including the new invariant test, the `calc_Greens_function_with_offdiag` MPI tests, and
selfenergy MPI; grouping (`g=2`) green under the unified split serial + MPI. At the 10-bath benchmark
the unified-split self-energy is **FP-identical** to the pre-refactor nested split (‖σ‖=1957.6179,
rel 1.1e-13) with serial wall unchanged (26.9 s) — the load-balance benefit shows only with many
small blocks (the production regime), not the single-block anchor.

**Found (pre-existing, out of scope):** `block_Green`'s **dense** basis-expansion loop
(`greens_function.py:603`) raises `ValueError: operands could not be broadcast` (e.g. `(1,3,3)` vs
`(1,2,2)`) on a **multi-orbital** block whose dense basis expands across iterations — the padded
block width differs between two `block_green_impl` calls. It reproduces at `g=1` (the unmodified
path), so it predates this work; the production `sparse_green=True` path is unaffected. Flagged for a
separate fix.

---

# GF PARALLELIZATION — OPERATOR SPLIT / PAIRWISE-SCALAR (Phase 2 narrow end, IMPLEMENTED, 2026-07-01)

**Context.** The eigenstate-group knob explores the *wide* end of the block-width spectrum; the
operator split explores the *narrow* end. For a block of `n` transition operators it replaces the one
shared width-`n` block-Lanczos recurrence with `n` width-1 (scalar) recurrences for the diagonal
seeds `v_i = c_i|ψ⟩` plus, per off-diagonal pair `i<j`, two more scalar recurrences for the
polarization seeds `v_i+v_j` and `v_i+i·v_j`. The off-diagonal `G_ij` is recovered exactly from the
four scalar resolvents via the polarization identity
`M_ij = ½[S(v_i+v_j) − i·S(v_i+i v_j) − (1−i)(M_ii+M_jj)]` (and its mirror for `M_ji`). This maximizes
the number of independent, communication-free work units — the regime where ranks vastly outnumber
the `block × eigenstate` units — at the cost of building `n + n(n−1)` Krylov spaces per eigenstate
instead of one shared width-`n` space.

**Shipped.** `GF_OPERATOR_SPLIT=1` (`greens_function.py`, `_gf_operator_split`). It rides the *same*
unified single-split machinery: each scalar seed is one width-1 work unit in the global
`(block × side × eigenstate × seed)` decomposition, so load-balancing, redistribution and gather are
unchanged. A per-eigenstate `PairwiseGF` collects the scalar continued fractions on rank 0;
`calc_thermally_averaged_G` dispatches on it to `calc_G_pairwise`, which assembles the `n×n` block via
the identity above. Mutually exclusive with `GF_EIGENSTATE_GROUP` (operator split takes precedence,
forces `g=1`). In pairwise mode only the G-derived diagnostics (thermal cutoff, mesh density,
causality) run — the seed-projection- and scalar-tridiagonal-based checks (sum rule, integrated
weight, Lanczos convergence) do not apply to the reassembled block.

**Correctness.** New `test_calc_G_pairwise_polarization_identity` (single-pole synthetic, 1e-12) pins
the assembly math; `test_get_Greens_function_operator_split_matches_block` (serial) and
`test_get_Greens_function_operator_split_matches_block_mpi` (**n=2, 3, 4**) assert the pairwise GF
reproduces the shared-Krylov block GF on a 2-orbital hybridizing-bath system (off-diagonal pairs
exercised) to ~1e-5. Full GF suites green serial (23) + MPI n=2 (11).

**Benchmark verdict (NiO 10-bath anchor, NW=200, serial).** The anchor is a **single 10×10 block**
(the NiO d-shell, 5 orbitals × 2 spin) — the *worst case* for the operator split: it explodes one
shared width-10 recurrence into `10 + 45×2 = 110` redundant width-1 Krylov builds per eigenstate. The
block path completes in **27.5 s** (GF 24.9 s); the operator-split path takes **280.2 s (10.2×
slower)**, exactly as the cost model predicts (no shared Krylov, `O(n²)` recurrences). The self-energy
is **FP-identical** across the two paths (‖σ‖=1957.618 both; max|Δσ|=5.2e-11, rel 1.1e-12) — the
polarization reassembly is exact, so the 10.2× is pure redundant-Krylov overhead, not a different
answer.

**Verdict / Phase-4 default policy.** Keep `GF_OPERATOR_SPLIT` **off by default.** It is the right
knob only in the opposite corner from this anchor: **many small blocks (`n = 1–2`, the typical
production symmetric GF) with ranks ≫ units**, where the wide recurrence cannot fill the machine and
the redundant-Krylov penalty is small (`n=1` → 1 unit, no pairs; `n=2` → 2 diag + 2 polarization = 4
units, vs 1). For `n=1` blocks it is *free* (a width-1 block already is the scalar recurrence) and
purely adds parallel granularity. It is **counter-indicated for any block with `n≳3`**, and
pathological on a large dense block like the NiO anchor. The composed default rule now reads:
- `n=1` blocks: operator split is a no-op reorganization — enable freely when units < ranks.
- small blocks (`n=2`) with idle ranks: enable to expose 4× the units.
- large blocks (`n≳3`) or units ≥ ranks: leave off; prefer `g=1` (or `g=2` under FULL reort per the
  eigenstate-group verdict).

---

# FIXES APPLIED (B0 — calc_gs blockers)

## B0.1 + B0.2 — FIXED (one root cause, one-line change)

Both `calc_gs` blockers were the **same** bug: `CIPSISolver.truncate` (`cipsi_solver.py:42`) called
`self.basis.local_basis.clear()`, which empties only the `local_basis` *list* while leaving
`self.size` and `_index_dict` stale. The subsequent `add_states(trimmed subset)` then deduped every
state against the still-populated `_index_dict` → `unique_new == []` → `local_basis` never
repopulated, leaving `size > 0` with an empty `local_basis`. Downstream that desync surfaced two
ways: the dense `get_eigenvectors` path crashed in `Basis.build_state` (`IndexError`, B0.2), and the
Lanczos path built a **zero-norm seed** from the empty `local_basis` → "Block collapsed to zero
rank" (B0.1).

**Fix:** `self.basis.local_basis.clear()` → `self.basis.clear()` (the container reset that also
zeroes `size` and rebuilds `_index_dict`, so `add_states` repopulates correctly).

**Verified:** 20-bath `calc_selfenergy` now completes on **both** paths (dense: wall 177 s, `calc_gs`
2.2%; Lanczos eigensolver path: past `calc_gs` into a converging GF). Regression: `95 passed`
serial (`test_groundstate`/`test_manybody_basis`/`test_selfenergy`) and `79 passed` under
`mpirun -n 2`.

## B0.3 — DIAGNOSED (MPI GF non-convergence); fix is a numerical-methods decision

**Symptom:** at 10-bath the GF block-Lanczos (`block_Green_sparse`) converges in ~12 sweeps serially
but under `mpirun -n>1` `|beta|` grows unbounded and it never converges.

**What it is NOT:** the distributed matvec is correct. A controlled single `H@psi` (NiO-10bath
interacting H) is **bit-identical across n=1, 2, 3** — 139 outputs, zero missing/extra,
max |amp diff| = 1.4e-14 (pure FP). Ownership is hash-consistent (`redistribute_psis` and
`add_states` both route by splitmix64), and `unpack_psis_fused` sums duplicate contributions.

**Root cause:** the GF recurrence runs with `Reort.NONE`. Per-step `alpha`/`M`(→`beta`) are formed
by local inner products + `Allreduce(SUM)`. The MPI reduction sums partial results in a
**rank-count-dependent order**, so each coefficient differs from the serial sum at the ~1e-13
level. With no reorthogonalization these differences are injected every step and **amplified at
near-rank-deficient blocks** — `beta = chol(M)` of a near-singular Gram matrix (retained condition
up to ~`1/sqrt(EPS) ≈ 7e7` under the current `DEFLATE_TOL = sqrt(EPS)` floor). Matched serial/MPI
GF invocations agree to ~4 sig figs through it=3 then branch macroscopically at the ill-conditioned
it=4 block (|beta| 5.8 vs 33), after which the trajectories diverge. Serial happens to satisfy the
convergence monitor (~block 12) **before** the recurrence degrades; MPI's slightly different
coefficients miss that window and run on into the divergent regime. The divergence guard
(`1e3 × ‖H‖`) is too loose to catch the slow growth (|beta|→3628 ≪ ~2.5e5).

It is therefore a **floating-point reproducibility** problem (serial and MPI are not bit-identical
because `Allreduce` is not bit-reproducible vs a serial sum), not a logic bug — the `Reort.NONE`
recurrence is on a knife's edge and the two reduction orders fall to opposite sides.

`Reort.PARTIAL` (tested, n=2) **bounds** `|beta|` (no blow-up) but is much slower and still
truncates without converging on this workload — so reort alone is not a complete fix; this overlaps
the in-progress `doc/plans/blocklanczos_reort_reliability.md` work (same shared kernel).

**Fix applied (option a — robust deflation):** in `BlockLanczosArray.pyx` raised the block rank
floor from `DEFLATE_TOL = sqrt(EPS)` to `EPS**(1/3)` (and `DEFLATE_EVAL_TOL = EPS` → `EPS**(2/3)`,
the eigenvalue equivalent). This deflates a near-rank-deficient residual block *before* `chol(M)`
amplifies it, bounding the retained block condition number to ~`EPS**(-1/3)` (~1.7e5) instead of
~`EPS**(-1/2)` (~6.7e7) — comfortably inside the CholeskyQR2 recovery regime. The ~1000× smaller
per-step amplification keeps the `Reort.NONE` recurrence on the *same* convergent trajectory
serially and under MPI, so the rank-order rounding of the Allreduce can no longer tip MPI into
divergence. Single point of change: both the recurrence `beta` and `block_normalize_array` route
through `_cholesky_or_deflate`.

**Verified:**
- Eigensolver oracle/correctness suite unaffected: `99 passed` serial
  (`test_array_lanczos_oracle`, `test_block_lanczos_cy`, `test_restarted_lanczos`, `test_lanczos`,
  `test_block_lanczos_reort_matrix`, `test_block_lanczos_blowup`, …) + `13 passed` under
  `mpirun -n 2`. Two `test_block_lanczos_blowup.py` oracle tests that hardcoded the old `sqrt(EPS)`
  floor were updated to derive from the `DEFLATE_*TOL` constants (single source of truth) and to
  exercise CholeskyQR2 on the worst *retained* (now milder) conditioning.
- The 10-bath GF that previously diverged under `mpirun -n 2` (|beta|→3628, "divergent tail",
  never converged) now **converges with zero divergence warnings**, `|beta|` bounded, `1 passed`.
- **Serial-vs-MPI self-energy now identical to FP:** dumped `sigma_real` (shape `(1,200,10,10)`)
  from a serial run and an `mpirun -n 2` run — `‖σ‖` 1957.617940220945 (serial) vs 1957.617940220987
  (n=2), **max |diff| = 3.97e-11, relative = 6.6e-13** (~13 significant figures). The "serial and
  MPI should be identical" requirement is met.

**Considered but not needed:** (b) mandatory GF reort (couples to the reort-reliability plan;
PARTIAL alone bounded but didn't fully converge), (c) deterministic reduction (highest effort),
(d) tighter divergence guard (mitigation only).

---

# PART A — RESULTS (run 2026-06-30, this machine, Python 3.14 venv)

## Headline finding: `calc_selfenergy` does not complete at the 100-bath anchor

The intended anchor (NiO 100-bath) **cannot be benchmarked** — `calc_selfenergy` crashes inside
`calc_gs` for any workload whose CIPSI basis grows past `dense_cutoff` (empirically ≈20 bath and
up), on **both** eigensolver paths, **serial and MPI**:

1. **Lanczos path** (production `dense_cutoff=500`): `ValueError: Block collapsed to zero rank`
   at `BlockLanczosArray.pyx:564`, raised from the IRLM/TRLM **seed** normalization
   (`cipsi_solver.py` `expand`/`get_eigenvectors` → `restarted_lanczos` → `block_normalize` of the
   initial block). Both `irlm` and `trlm` collapse. Reproduced at 20- and 100-bath.
2. **Dense path** (forced via large `dense_cutoff`): `IndexError: list index out of range` at
   `manybody_basis.py:1201` in `Basis.build_state`. Probe at the crash:
   `vs.shape=(1439,1439)  size=1439  len(local_basis)=0`. The basis reports global `size=1439`
   but its `local_basis` is **empty**. Root cause is in the truncation path:
   `CIPSISolver.truncate` (`cipsi_solver.py:42`) does `self.basis.local_basis.clear()`, and the
   `expand` ">truncation_threshold → truncate → break" path (`cipsi_solver.py:208-212`) leaves the
   basis with `size` set but `local_basis` desynced before the next `get_eigenvectors`.

Both are **correctness bugs in `calc_gs` at scale**, not performance issues. They are the true
current blocker for the target workload. The committed harness re-runs unchanged once they are
fixed (`SELFENERGY_BENCH_NBATH=100`).

Two pre-existing latent bugs in the intended driver `get_selfenergy` were also found (the harness
works around both by calling `calc_selfenergy` directly): its `get_noninteracting_hamiltonian_operator`
call uses a **stale positional arg order** (the `nValBaths` parameter was added later, so `SOCs`
now receives the 3-tuple `hField` and fails to unpack), and it passes `nominal_occ` as the legacy
**3-tuple** where `calc_gs`/`find_ground_state_basis` now expect the impurity-occupation **dict**.

## Attribution at the largest runnable size (NiO 10-bath, serial, NW=2000)

Total wall = **40.9 s**. Phase split:

| phase                 | time (s) |     % |
|-----------------------|---------:|------:|
| `get_Greens_function` |    38.78 | 94.8% |
| `calc_gs`             |     2.11 |  5.2% |
| `get_sigma`           |     0.01 |  0.0% |
| `get_Sigma_static`    |     0.00 |  0.0% |

GF phase is ~95% of runtime and is **insensitive to the output mesh** (NW=50 → 35 s vs NW=2000 →
41 s): the cost is the Lanczos recurrence + convergence machinery, not the resolvent evaluation on
the requested `omega_mesh`. cProfile (by `tottime`) splits the GF phase into two co-equal centers:

| function (file)                                              | tottime | ncalls | note |
|-------------------------------------------------------------|--------:|-------:|------|
| `block_Green_sparse` (`greens_function.py:887`)             |  19.98s |     12 | block-Lanczos sweeps (matvec + reorth + QR) |
| `np.linalg.solve`                                           |  12.50s | 38 884 | called from the **convergence monitor** |
| `_block_cf_inverse` (`greens_function.py:1049`)             |   3.26s |    752 | continued-fraction block resolvent |
| `_greens_function_change` (`greens_function.py:1132`)       | 11.67s* |    680 | per-block rel-change stopping test (*cumtime) |
| `build_local_operator_list` (`manybody_basis.py:1211`)      |   1.23s |     15 | GF seed operator build |
| `eigh` / `inv` / `wI`                                       |  ~1.3s  |     —  | small dense ops |

`calc_gs` at 10-bath is cheap (dense, small basis); `get_sigma`/`get_Sigma_static` are negligible.

## MPI strong-scaling sweep — not obtainable

At 10-bath under `mpirun -n 2`, the **GF block-Lanczos diverges**: `|beta|` grows without bound
(896 at it 371 → 3628 at it 957, alpha-diagonals ~150–175) where the *serial* run converged in 12
sweeps. The run never completes, so no clean `S(n)` data exists. This is a third scale/parallel
issue (an MPI-specific GF-Lanczos divergence, cf. the "Block GF Lanczos CholeskyQR divergence"
memory). The A3 sweep is **deferred** until (a) the `calc_gs` bugs unblock a production-size
workload and (b) the MPI GF divergence is resolved — only then is strong scaling at 100-bath
meaningful.

## Bottleneck verdict

- **#0 (blocking): correctness at scale.** `calc_selfenergy` must be made to *complete* at ≥20
  bath before any perf work on it is measurable. Three distinct defects: GS Lanczos seed collapse,
  `build_state`/`truncate` `local_basis` desync, MPI GF-Lanczos divergence.
- **#1 (perf, once unblocked): the Green's-function phase** (~95% at 10-bath). Within it, two
  co-equal levers: the **block-Lanczos sweeps** (matvec + reorth) and the **GF convergence
  monitor**, which spends ~12.5 s in 38 884 `np.linalg.solve` calls re-inverting the continued
  fraction on a frozen mesh after *every* Lanczos block.
- `calc_gs` (~5%) and `get_sigma`/static (~0%) are **not** current perf priorities → **B5 is
  dropped**.

This verdict reorders Part B (below).

---

# PART B — REVISED (data-driven)

**B0 — Unblock `calc_selfenergy` at scale (PREREQUISITE FOR ALL PERF WORK).** Independent of every
other family; nothing downstream is measurable until this lands. Three sub-items, each with the
10-bath harness as a working oracle and a chosen larger size as the regression target:
- **B0.1** GS Lanczos seed collapse (`block_normalize` "Block collapsed to zero rank" on the IRLM/
  TRLM initial block) — relates to the active IRLM reort/deflation work
  (`doc/plans/blocklanczos_reort_reliability.md`).
- **B0.2** `Basis.build_state` / `CIPSISolver.truncate` `local_basis`-vs-`size` desync
  (`cipsi_solver.py:42,208-212`; `manybody_basis.py:1199-1201`).
- **B0.3** MPI GF block-Lanczos divergence (`block_Green_sparse`) — `|beta|` blow-up under ranks>1.
- *(Optional cleanup)* fix `get_selfenergy`'s stale `get_noninteracting_hamiltonian_operator` arg
  order and `nominal_occ` tuple so the documented CLI driver works again.

**B6 — GF convergence-monitor cost (NEW, highest single perf lever found).** The frozen-mesh
relative-change test (`_greens_function_change` → `_block_cf_inverse` → 38 884 `np.linalg.solve`,
~30% of total runtime at 10-bath) recomputes the block continued-fraction resolvent **from scratch
after every Lanczos block**. Make it incremental (extend the previous block's resolvent rather than
re-inverting), batch the per-mesh-point `solve` into one stacked solve, and/or check convergence
less frequently. Self-contained in `greens_function.py`; guard with a GF-equivalence check against
the 10-bath baseline. **Independent of** B1–B4. *(Promoted above B2/B3/B4 by the data.)*

**B1 — Matvec throughput (`ManyBodyOperator::apply`).** Unchanged from the original framework:
foundational, independent, oracle exists (`doc/plans/manybodyoperator_apply_performance.md`). Drives
the other ~half of the GF phase (the `block_Green_sparse` sweeps) plus CIPSI/eigensolve. Schedule
after B0.

**B2 — Block-Lanczos kernel (BLAS + reort).** Unchanged; addresses the `block_Green_sparse` sweep
cost. Gated on the reort-reliability Phase-0 oracle. Note B0.1/B0.3 overlap this code, so B2 and
those bug-fixes should be coordinated (same kernels).

**B3 — CIPSI basis growth / truncation.** Deprioritized: `calc_gs` is only ~5% at 10-bath. Revisit
only if it grows materially at the (unblocked) production size. Note B0.2 lives in this code.

**B4 — MPI comm / load balance.** Cannot be assessed until B0.3 makes an MPI run complete; folded
into "schedule after B0, then re-measure with the A3 sweep."

**B5 — GF/`get_sigma` dense path.** **Dropped** — measured at ~0% of runtime.

**Revised sequencing:** B0 (all three sub-items) → re-run the harness at 100-bath to refresh
attribution at production scale → then B6 and B1 in parallel (independent, both target the GF
phase) → B2 (gated on its oracle) → re-run the A3 MPI sweep for B4. Each perf step re-uses the
committed harness as its before/after gate.

## Artifacts

- Harness: `src/impurityModel/test/test_selfenergy_perf.py` (committed).
- Raw logs + per-rank `.pstats`: scratchpad `serial_10bath.log`, `mpi_10bath_n{2,4}.log`,
  `pstats_10bath_serial/` (regenerable via the harness).
