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

## Phase 2 — block-state matvec: `ManyBodyBlockState` (approved 2026-07-07)

Item 1 (estimator calibration) is done above. Items 2–4 of the old list are
superseded/absorbed by the block-state design: instead of parallelizing p
independent matvecs, restructure the *hot-loop* state container so a block of p
vectors is stored as ONE shared determinant support + a row-major `(n_dets x p)`
amplitude array. `ManyBodyOperator::apply` then does the term loop, bit/sign work,
restriction checks and — the dominant cost — the accumulator hash operation **once
per (determinant, term)**, emitting p scaled amplitudes with p FMAs; threading stays
over determinants (the existing PARALLEL partitioning, still opt-in per the
MPI-oversubscription decision). matvec_apply is ~77% of the iteration post-Phase-3,
so this is the main remaining lever; it also gemm-ifies `inner_multi`/
`add_scaled_multi` and shrinks live-state memory by the same 72→16 B/coefficient
argument as the Krylov store (fill ratio 1.00 measured).

Design constraints: `ManyBodyState` is NOT retrofitted (it stays the scalar boundary
type everywhere outside the hot loop); block width p is runtime (no compile-time
width); row pruning keeps a row if ANY column survives the cutoff (deliberate,
flagged semantic difference vs per-column prune); deflation narrows p via column
subset. The Lanczos-loop conversion is contained because the kernel already funnels
all block ops through `apply_multi`/`inner_multi`/`add_scaled_multi`/
`redistribute_psis`/`prune`.

Stages (each committable, gate-green):
- **2.0 oracle + baseline — DONE (2026-07-07)**: `test_apply_block_width_scaling`
  in both harnesses. Measured (serial): hopping fixture 81 ms/state flat →
  1.00/2.00/4.03/9.69x at p=1/2/4/8; NiO 50-bath H (restrictions set) 18.7 ms/state
  flat → 1.00/1.95/3.97/7.88x. Cost is exactly linear in p on shared-support blocks,
  i.e. everything except the per-column FMAs is repeated work — the amortizable
  fraction is ~all of a single apply. Golden equality (block == p independent
  applies, bit-for-bit) lands with the 2.2 API.
- **2.1 container — DONE**: C++ `ManyBodyBlockState` (sorted shared support +
  row-major amplitudes, header-only) + Cython wrapper with buffer protocol
  (zero-copy `np.asarray`, export guard against mutation-under-view); conversions
  to/from `list[ManyBodyState]` bit-exact; row-prune-any-survives.
- **2.2 block apply (serial) — DONE**: `ManyBodyOperator::apply(const
  ManyBodyBlockState&, cutoff)` mirrors the scalar hot loop with a
  determinant→row block accumulator (one hash op per (det, term), p FMAs per
  emission); the density/1-body fast paths ported unchanged. Golden: bit-for-bit
  equal to p independent applies (parametrized p ∈ {1,2,3,5}, density + 1-body +
  general 2-body terms); whole-row cutoff semantics tested. **Measured**: block
  apply is near-flat in p — NiO 50-bath H 18.96/20.09/22.30/23.52 ms at
  p=1/2/4/8 → speedup 0.99/1.89/3.43/6.45x vs apply_multi; hopping fixture
  2.12/3.89/6.79x at p=2/4/8; no p=1 regression.
- **2.3 block redistribute — DONE**: `pack_block_fused`/`unpack_block_fused`
  (MpiUtils) + `graph_alltoall_block` (mpi_comm) + `Basis.redistribute_block`.
  One wire entry per shared-support row — `state_bytes + 16p` bytes per
  determinant instead of `p*(state_bytes + 20)` — same `routing_hash` ownership,
  same cached dist-graph + one fused `Neighbor_alltoallv(BYTE)`. Cross-rank
  duplicate rows are summed in arrival order (stable sort), bit-identical to the
  scalar unpack; verified against `graph_alltoall_psis` at n=2/3 including an
  empty-contributor rank.
- **2.4 Lanczos loop switch — DONE**: q_prev/q_curr/wp/q_next are
  ManyBodyBlockStates inside `block_lanczos_step_cy`/`block_lanczos_cy`; the
  recurrence runs on the bit-exact block primitives (2.4a commit), the store
  appends block rows directly, and the rare paths (FULL/PERIODIC reort, PARTIAL
  trigger, SELECTIVE cadence, EOR seed, global truncation) convert at the
  boundary only when they actually run (`apply_reort` converts inside the
  acted branch). The whole-row prune (any-column-survives) is the one flagged
  semantic difference. **Measured, NiO 50 bath (basis 5848, 60 its, serial)**:
  p=2 iteration 65.5 → **33.9 ms** (matvec 50.2 → 27.4, recurrence 6.6 → 1.4);
  p=4 iteration **44.7 ms** (matvec 29.6 — near-flat in p); `‖QᴴQ−I‖` and E0
  unchanged to all printed digits; reort trigger pattern identical (5/60); loop
  peak RSS delta 0.4 MiB. **Cumulative vs the campaign baseline: 125.2 → 33.9
  ms/iter (3.7x) at p=2** plus the 3.1x Krylov memory cut.
- **2.4c spectra paths — DONE (user-requested)**: the block matvec now powers the
  whole spectra layer, not just the Lanczos kernel:
  - `cg.block_bicgstab` / `_block_bicgstab_core`: the sparse branch runs on
    `ManyBodyBlockState` end to end — `apply_block` matvecs (2 per iteration),
    fused block redistribute, per-column norms from `col_norm2`, Gram products via
    `block_inner_cy`, axpy rebinds via `block_add_scaled_cy`. Callers keep the
    `list[ManyBodyState]` interface (conversion at the solver boundary). The dense
    (ndarray) branch is untouched. The reachable-support bookkeeping uses the new
    `ManyBodyBlockState.support_keys(min_amp)` (row-max filter == the old union of
    per-column filters).
  - `greens_function._apply_transition_ops`: transition operators applied to the
    whole thermal-eigenstate block at once.
  - `greens_function` excited-basis reachability probe: the 5x repeated H
    application runs on the block.
  - The Lanczos-based GF path (`block_green_impl`) already used the block kernel
    since 2.4b.
  - Tests: the five obsolete mock-based cg tests (patching pre-block internals on
    plain dicts) are replaced by four real end-to-end sparse solver tests (dense
    reference, rank-deficient RHS linearity, warm start, max_iter); the four real
    pre-block tests (array deflation cases, sparse rank-deficient linearity,
    break-on-active-mask) are kept verbatim.
- **2.5 threaded path re-tune**: the block accumulator changes memory-per-entry;
  re-measure the MIN_SD_PER_THREAD workload scaling; threading stays opt-in.

Parked pending re-measurement after 2.2: apply cost center D (map→sort→rebuild —
the block accumulator changes its weight) and the 2-body masked sign (§A design).

## Phase 3 — columnar Krylov store (IN PROGRESS; gate criteria met)

Gate confirmed by Phase 0: Q_basis is 52 of 75 MiB peak per rank (50 bath) at fill
ratio 1.00 → the columnar store cuts Krylov memory ~4.5x (72 B/coeff flat_map → 16 B
dense over the shared support).

**Design (fixed 2026-07-07).** The TRLM/IRLM restart machinery is already
path-agnostic through a small dispatch surface — `_q_cols`/`_q_slice`/`_q_concat`/
`_copy_block` (BlockLanczos.pyx) and `is_array`/`block_combine`/`block_inner`/
`block_orthogonalize`/`block_add_scaled` (BlockLanczosArray.pyx) — so the store lands
as a *third dispatch path*, not a rewrite:

1. **Commit A**: extend `SparseKrylovDense` (ManyBodyUtils.pyx) into the primary
   store: sequence protocol (`__len__` = n_cols, `__getitem__` int→`ManyBodyState`,
   slice→list, `__iter__`), `combine(Y, a, b, slaterWeightMin)` (zgemm over
   `Qbuf[:, a:b] @ Y`, scatter only the output states — the dense analogue of
   `block_combine_sparse`), `memory_bytes()`. Unit tests: bit-exact
   materialize/append round-trip; `combine` vs `block_combine_sparse` on
   materialized states.
2. **Commit B**: `block_lanczos_cy` maintains the store as the ONLY retention for
   `store_krylov=True` (all reort modes; the FULL/PERIODIC list+mirror double
   storage disappears, PARTIAL's `reorth_cgs2_dense` transient path retires —
   `apply_reort` gets `krylov=store` always). Returns the store as `Q_basis`;
   accepts `Q_init` as store (resume round-trip: greens_function works unchanged)
   or legacy list. Dispatch helpers + `selective_orthogonalize` learn the store
   (a lightweight `(store, a, b)` view from `_q_slice` so `block_combine` can gemm
   without materializing). Tests that do `inner_multi(Q, Q)` materialize via
   `list(Q)`. The EOR warm-start W-seed (rare path) materializes block slices.

Recurrence stays bit-for-bit (q_prev/q_curr untouched); the bad-block projection's
summation order changes (insertion-ordered store rows vs merged-sorted transient
support) → tolerance-equivalent, flag in commit B.

**Commit B landed (2026-07-07). Measured, NiO 50 bath (basis 5848, p=2, 60 its):**

|  | baseline (pre-campaign) | after estimator fix | after store |
|---|---|---|---|
| ms/iteration (serial) | 125.2 | 68.9 | **65.5** |
| ms/iteration (n=2) | 84.7 | — | **49.8** |
| reort ms/it (serial) | 44.6 | 10.6 | **2.2** |
| Q_basis / rank (serial) | 52.0 MiB | 52.0 MiB | **16.8 MiB** (3.1x) |
| loop peak RSS delta | 64.2 MiB | 74.6 MiB | **1.0 MiB** |
| cgs2_dense churn / run | 180 MB | 42 MB | **0 (path retired)** |

E0 identical to the array TRLM (-69.361029), `‖QᴴQ−I‖ = 1.9e-8` unchanged. The
16.8 MiB store vs the 11.6 MiB theoretical minimum is geometric capacity slack
(122 columns in a 128-capacity buffer + row slack) — acceptable; column-chunked
growth (Phase 4) can shave it if it ever matters. TRLM/IRLM restart extraction
gets the store's gemm `combine` automatically through `block_combine`; restart
boundaries still transiently materialize slices (`_q_slice` → list) — measure
before optimizing further (a `truncate_cols` + store-view would remove it).

## Phase 4 — transient-spike reductions (planned)

`SparseKrylovDense` realloc doubling (column chunking), apply output double-pass,
redistribute buffer reuse.
