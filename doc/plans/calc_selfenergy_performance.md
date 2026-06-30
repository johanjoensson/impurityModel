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

*Correction to an earlier draft:* PARTIAL is **selective**, not blanket. `apply_reort`
(`BlockLanczosArray.pyx`) only reorthogonalizes when the Paige–Simon W-estimate exceeds `REORT_TOL`,
and then only against the **flagged "bad" blocks** (`max|W[-1,j]| > BAD_BLOCK_TOL`) — *not* the whole
Krylov basis. So the per-step cost is **data-dependent**: on this near-degenerate GF spectrum the
estimate trips often enough (and flags enough blocks) that the per-step `estimate_orthonormality`
bookkeeping + the triggered block projections (full inner products over the determinant vectors) +
the extra CholeskyQR2 pass sum to ~11× NONE's bare matvec+QR step.

cProfile attributes nearly everything to `block_Green_sparse` (286.9 s) because the inner kernels
(`block_lanczos_cy`, `estimate_orthonormality`, `apply_reort`, `inner_multi`) are Cython and don't
get their own Python frames — so cProfile **cannot** split matvec vs W-estimator vs triggered-reort
vs CholeskyQR2. That split needs a Cython line-profile / explicit counters (not yet done). What *is*
measured: the per-step cost and step count above.

For **NONE** the same lumping applies, but its Python-level convergence monitor *is* visible and
sizeable: `_block_cf_inverse`→`np.linalg.solve` is **10.6 s over 37 637 calls** (~24% of NONE
runtime), re-inverting the continued fraction on the frozen mesh every block. NONE needs more steps
(691) but each is cheap (63 ms).

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

- **GF phase is the whole game** (~98%+). Levers:
  **(B6)** the convergence monitor — for NONE it is a *measured* ~24% of runtime
  (`_block_cf_inverse`/`np.linalg.solve`, re-inverting the continued fraction every block); making it
  incremental + terminating earlier shrinks the step count *both* modes pay for. **(B2)** the
  PARTIAL per-step reort cost (~11× NONE's bare step here) — BLAS-accelerate the triggered block
  projections / `estimate_orthonormality`, and tune the trigger so it fires less on near-degenerate
  GF spectra. A Cython line-profile is needed to split matvec vs W-estimator vs triggered-reort
  before optimizing B2 (cProfile lumps them).
- **Reort choice is a real trade-off, not a free default:** NONE has a cheap step (63 ms) but its
  step count explodes with system size (max blocks 142→241→923+ for bath 10→20→100); PARTIAL has an
  expensive step (688 ms) but a roughly flat step count (selective reort keeps the recurrence
  conditioned). They **cross over** — NONE wins small, PARTIAL is the only mode that converges at
  100-bath. **B6** is the highest-leverage fix because reducing the block count helps both.
- `calc_gs` and `get_sigma` remain non-priorities.

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
