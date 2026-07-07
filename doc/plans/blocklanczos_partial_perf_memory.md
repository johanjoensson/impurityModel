# Block Lanczos PARTIAL-mode performance & memory campaign

**Scope:** the Block Lanczos hot loops — sparse hash-distributed kernel
(`src/cython/BlockLanczos.pyx`), array kernel (`src/cython/BlockLanczosArray.pyx`) — and,
only where the profile demands it, the `ManyBodyOperator::apply` matvec. PARTIAL
reorthogonalization is the production mode (CIPSI ground state); memory is the binding
constraint on complicated problems.

**Policy:** measure first (Phase 0 harness), then land optimizations incrementally.
Pure refactors must be bit-for-bit A/B (file-backup, never `git checkout`); algorithmic
reorderings may be tolerance-equivalent with the reort oracle green and must say so in
their commit message. Test gate per commit: `python -m pytest` + `mpiexec -n 2 python -m
pytest --with-mpi` (+ `-n 3` for distribution/reort-collective changes).

Prior campaigns (do not re-plan): apply P0–P5
(`manybodyoperator_apply_performance.md`; 2-body masked sign + NBX deferred in
`apply_perf_deferred_designs.md`), array-kernel BLAS (`blocklanczos_blas_acceleration.md`;
bounded W + scratch reuse deferred there, now Phase 1 here), reort reliability
(`blocklanczos_reort_reliability.md`; its oracle is this campaign's safety net).

## Key structural facts (established 2026-07-07)

- **The production CIPSI PARTIAL path runs the *array* kernel**: `cipsi_solver.expand` /
  `get_eigenvectors` build a CSR `H_mat` (column-sliced under MPI) and call the
  dispatching TRLM/IRLM → `block_lanczos_array_cy`. The hash-distributed
  `ManyBodyState` kernel (`block_lanczos_cy`) is the memory-scalable alternative
  (`cipsi_solver.py:264-270` documents the `(global_N, n)` per-rank replication of the
  array path); the GF path runs it with `reort=none` + `store_krylov=False`.
- The GS-occupation search (`find_ground_state_basis` → `calc_energy`) runs its CIPSI
  expansions with `reort="full"` (the `calc_energy` default) — only `calc_gs`'s
  follow-up `expand`/`get_eigenvectors` use the `Reort.PARTIAL` default.
- `ManyBodyState` storage cost: `flat_map<SlaterDeterminant(std::vector<uint64>), complex>`
  ⇒ per entry 40 B inline pair + a separate ≥32 B heap block per key ≈ **72 B per
  determinant-coefficient** (measured via the new `memory_bytes()`), vs 16 B dense
  payload → the Phase-3 columnar store bound is ~4.5x (confirmed by the harness'
  "columnar equivalent" line at fill ratio ~1.0).

## Phase 0 — baseline harness (DONE, this commit)

`src/impurityModel/test/test_block_lanczos_perf.py` (`-m benchmark`, sizing via
`BLBENCH_*` env knobs; see its docstring). NiO d-shell workload from
`_nio_workload.build_selfenergy_inputs` through the production preamble
(symmetry rotation → bath classification → orbital grouping →
`find_ground_state_basis` + one `CIPSISolver.expand`), then two benchmarks on the
frozen basis:

- `test_partial_trlm_array_bench` — production path (TRLM PARTIAL on CSR).
- `test_partial_sparse_kernel_bench` — `block_lanczos_cy` PARTIAL, fixed iteration
  budget, per-phase timers (`BLOCKLANCZOS_PROFILE` counters, now runtime-toggleable via
  `enable_*_profile()`), memory breakdown (new `ManyBodyState.memory_bytes()`,
  `support_stats()` fill ratio, `reorth_cgs2_dense` transient counters, per-rank RSS
  sampler), and built-in correctness assertions (`‖QᴴQ−I‖ < 1e-6`, E0 agreement with
  the array kernel to 1e-4).

New instrumentation added for this phase (all ~zero cost when off):
`matvec` timer split into `matvec_apply` / `matvec_redistribute` (BlockLanczos.pyx),
`ManyBodyState.memory_bytes()` + `support_stats()` + `get/reset/enable_manybody_profile`
(ManyBodyUtils.pyx), `enable_block_lanczos_profile` / `enable_reort_profile` setters.

### Baseline (serial, this machine, 2026-07-07)

NiO d-shell, mixed-valence window 1, `TRUNC=30000`, sparse kernel `p=2`:

| workload | basis | TRLM/array median | sparse kernel ms/iter | matvec_apply | reort | recurrence | w_estimate | Q_basis (72 B/entry) | columnar equiv | fill |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 bath, 40 its | 670 | 0.35 s | 10.1 | 4.75 ms | 2.99 ms | 0.74 ms | 0.70 ms | 4.1 MiB | 0.9 MiB (4.5x) | 1.00 |
| 50 bath, 60 its | 5848 | 15.6 s | 125.2 | 62.2 ms | 44.6 ms | 8.2 ms | 1.3 ms | 52.0 MiB | 11.6 MiB (4.5x) | 1.00 |

50-bath detail: reort acted 59/60 iterations (846/1830 blocks flagged, 46%);
`reorth_cgs2_dense` transient peak 9.1 MiB/call, 180 MB total churn over the run;
peak RSS delta 64 MiB ≈ Q_basis (52) + transients; `‖QᴴQ−I‖ = 9.1e-12`.

MPI n=2, 50 bath (basis 5858 — the CIPSI trajectory is rank-count-dependent, so sizes
differ slightly from serial): sparse kernel 84.7 ms/iter = matvec_apply 38.2 +
reort 30.3 + redistribute 2.9 + recurrence 6.4; same over-trigger (59/60); local
Q_basis 28.4/23.5 MiB per rank; assertions green distributed. TRLM/array median 1.97 s
(restart count differs from serial via the per-rank random start — not a scaling
comparison).

### Findings so far

1. **matvec_apply dominates** the sparse-kernel iteration (~50% at both scales);
   redistribute is nil serially (measure under MPI).
2. **PARTIAL over-triggers**: `apply_reort` acted on ~every iteration (59/60 at 50
   bath), flagging ~46% of Krylov blocks each time, and the resulting
   `‖QᴴQ−I‖ ≈ 9e-12` is *far* below the PARTIAL target (√ε ≈ 1.5e-8). The magnitude
   upper-bound W recurrence (the under-prediction fix) now appears to over-predict,
   so PARTIAL does near-FULL projection work (36% of iteration time at 50 bath) *plus*
   estimator overhead. Needs a dedicated look: check the W-estimate vs true overlaps
   on this workload, and re-tune the noise floor / `BAD_BLOCK_TOL` interplay so
   PARTIAL fires rarely while keeping semi-orthogonality. → top Phase-2 item (it is
   simultaneously a speed and a correctness-calibration issue).
3. Fill ratio is ~1.0 on the GS workload → the Phase-3 columnar store gives the full
   ~4.5x Krylov-memory cut and retires the `reorth_cgs2_dense` transient
   (9.1 MiB peak per call at basis 5848, 180 MB total churn over 60 iterations).

## Phase 2.1 — PARTIAL estimator calibration (DONE, second commit)

Root cause of finding 2, established with a truth-vs-estimate replay
(`block_lanczos_step_cy` driven directly with `reort=NONE`, exact `<Q_j|q_next>`
overlaps vs the W recurrence on the same trajectory):

- The **magnitude** W recurrence (sum of `|terms|`, introduced by "HARDEN Block
  Lanczos" `48308d1` together with the 1/σ_min noise-floor fix) destroys the exact
  structural cancellation of the O(‖β‖) identity-adjacent terms in the three-term
  recurrence. The estimate jumps to O(‖β‖/σ_min) ≈ O(1) after **one** step and
  compounds exponentially — measured over-prediction 1e15→1e62 on the NiO ground
  state. Every block was flagged every iteration: PARTIAL silently did FULL work
  (this also explains the "B2 investigation" conclusion in `0b8bfff` that reort is
  "barely selective" — it was the estimator, not the spectrum).
- Fix: **signed propagation restored** (pre-HARDEN form; the cancellation is the
  physics) **+ the 1/σ_min-amplified magnitude noise floor kept** (that part of
  HARDEN was the genuine under-prediction fix) **+ √N scaling** of the floor
  (rounding accumulates as a √N random walk over an N-dimensional state; same
  convention as the drivers' locked-reort floor `EPS*p*sqrt(N)`; the historical
  `N=1` default under-scaled by exactly the measured ~10x gap at N=670). Call sites
  now pass the global dimension N.
- Validated: estimate/truth ratio stable at 2–10x (upper bound, never under at
  trigger onset) across the full run; the estimator crosses REORT_TOL exactly when
  the true loss reaches the √ε semi-orthogonality boundary.

Effect (serial):
- 20 bath: reort acted 4/40 (was 39/40), reort 2.99 → 0.93 ms/it, iteration
  10.1 → 8.1 ms, `‖QᴴQ−I‖ = 1.49e-8`.
- 50 bath: **iteration 125.2 → 68.9 ms (1.8x)**, reort 44.6 → 10.6 ms/it acting
  5/60 (was 59/60), `reorth_cgs2_dense` churn 180 → 42 MB per run,
  `‖QᴴQ−I‖ = 1.9e-8`, E0 identical to the array TRLM (-69.361029).

The final orthogonality now sits exactly at the PARTIAL design point
(semi-orthogonality at √ε with minimal projection work; the previous 1e-12 was
FULL-grade orthogonality bought at FULL-grade cost). Tolerance-equivalent change by
policy (the reort firing pattern differs ⇒ different rounding trajectory); full gate
green serial + n=2 + n=3 with baseline-identical pass/xfail counts, reort oracle and
restarted-Lanczos suites green.

## Phase 1 — bit-for-bit wins (planned)

1. Bounded W buffer (callers pass views; kills the per-step `(2, i+2, n, n)` realloc in
   `estimate_orthonormality`).
2. Cache per-`j` β norms + reuse the driver's β SVD (three factorizations of the same
   β_i per step today).
3. Audit / drop the defensive `st.copy()` at `Q_basis.extend` (BlockLanczos.pyx:851).

## Phase 2 — profile-ranked time optimizations (planned)

Ranked by the Phase-0 profile; current candidate order:
1. PARTIAL trigger calibration (finding 2 above).
2. Cross-state parallelism / batching in `apply_multi` (the p block matvecs run
   serially, `ManyBodyOperator.h:110-130`).
3. Apply cost center D (map→vector→sort→flat_map rebuild) — also a 2x output-memory
   transient.
4. 2-body masked sign (deferred design §A) only if the general-path share justifies it.

## Phase 3 — columnar Krylov store (planned, gated on Phase 0 numbers)

Promote a `SparseKrylovDense`-like structure to the *primary* Krylov storage
(shared determinant→row support + one geometric complex buffer, `get_block`/`combine`
accessors); rewire warm-start, TRLM/IRLM extraction, `selective_orthogonalize`,
`apply_reort`; FULL/PERIODIC stop double-storing; `reorth_cgs2_dense` fallback retires.
Recurrence stays bit-for-bit; the bad-block projection changes summation order
(tolerance-equivalent, flag in commit).

## Phase 4 — transient-spike reductions (planned)

`SparseKrylovDense` realloc doubling (column chunking), apply output double-pass,
redistribute buffer reuse.
