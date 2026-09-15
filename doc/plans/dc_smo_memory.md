# SrMnO3 double-counting OOM: measurement and fix

A 24 h SrMnO3 (cubic) DMFT job on Arrhenius (SLURM 2307501, 256 ranks × 2 cores / 2 nodes)
was OOM-killed inside the `Gap` double-counting search, before a single self-energy was
produced. This records what was measured against the crash's own archive
(`impurityModel_data.h5`, reproducible offline via `ImpurityModel.from_hdf5` /
`load_selfenergy_archive`), what changed as a result, and — this document's most important
correction — which of those changes actually address the crash's dominant cost, and which
address a real but secondary one.

## The crash

| fact | source |
|---|---|
| 13 of 256 ranks OOM-killed, all on one node (128 ranks/node) | `slurm-*.out` |
| the solver believed it had 5.0 GiB/rank, budgeted half | `impurityModel-*-dc.out` |
| it chose `truncation_threshold = 119,555,328` — 126× the basis that killed it | same line |
| the code warned the number was unsafe (`gs_num_wanted` unset) and used it anyway | same log |
| CIPSI ran to PT2 exhaustion at 949,834 determinants, then died | last log line |
| `job.rspt` sets `GS_MAX_BLOCK_WIDTH=5` | production job script |

## Where the memory actually goes: a per-step RSS ledger

Two earlier passes each named a single culprit from a formula and were each refuted by
measurement (`build_sparse_matrix`, then the selection round, then the eigensolver — see
the refuted-claims section below). This section replaces all of them with a direct
**per-step peak-RSS ledger**, which needs no attribution argument at all: `peak_rss_bytes()`
is sampled immediately before and after every step, so the first step that raises the
high-water mark *is* the culprit. The method matters as much as the answer -- a single
before/after pair around one suspected callee is what produced both wrong attributions, because it cannot distinguish "this allocated it" from "this ran just before whatever did".

Reproduction: the crash's own archive
(`/home/johan/Dokument/arrhenius/SMO/cubic/impmod/impurityModel_data.h5` — **not**
`restriction_diagnostics.WORKLOADS["smo"]`, which points at a different SrMnO3 run in
`impmod_tests` with `tau=0.0025` against this one's `0.025`), `GS_MAX_BLOCK_WIDTH=5` as
`job.rspt` sets it, `de2_min=GS_DE2_MIN`, serial, `OPENBLAS_NUM_THREADS=1`, 20,000-determinant
cap.

| cycle | step | peak RSS | increment |
|---|---|---|---|
| 0 | eigensolve (dense, 120 dets) | 298.4 → 300.7 MiB | +2 MB |
| 0 | selection | 300.7 → 327.3 MiB | +27 MB |
| 1 | eigensolve | 327.3 → 450.0 MiB | +123 MB |
| 1 | selection | 450.0 → 686.0 MiB | +236 MB |
| 2 | eigensolve | 686.0 → 925.8 MiB | **+240 MB** |
| 2 | **selection** | 925.8 → **1932.6 MiB** | **+1007 MB** |
| 3–12 | either | flat at 1932.6 MiB | 0 |

**The peak is one cycle, and inside it the selection round is 81% and the eigensolver 19%.**
Cycles 3 onward never raise the mark at all, despite running the same operations on the same
20,000-determinant basis — so nothing about the *converged* expansion matters; only the
transient at cycle 2 does.

What distinguishes cycle 2 is not `num_wanted` (88 there, against 182 at cycle 1) but the
selection round's size: `H·ψ_ref` rows go 67,344 → **350,250** and candidates 63,192 →
**312,172**, while the block width `p` barely moves (78 → 68).

Bracketing each sub-step of `determine_new_Dj` at that cycle splits the +1007 MB three ways,
all comparable:

| sub-step | peak RSS | increment |
|---|---|---|
| entering the selection round | 925.6 MiB | — |
| after `_apply_block_and_redistribute` | 1341.4 MiB | **+416 MB** |
| after `_candidate_overlaps_and_energies` | 1730.5 MiB | **+389 MB** |
| after `_score_candidates` | 1932.3 MiB | **+202 MB** |
| after `_admit_top` | 1932.3 MiB | 0 |

416 + 389 + 202 = 1007 MB, matching the coarse bracket exactly. **There is no single dominant
term inside the selection round either** — it is three arrays of comparable size, so no one
fix here is worth more than about a third.

This also settles why Phase 2a (below) was inert. Its premise was that the pre-fix
`to_states()` round trip held `p` width-1 states each carrying a full copy of the keys, which
at 350,250 × 68 pairs would be ~1.3 GB. The measurement says
`_apply_block_and_redistribute` costs **416 MB in total** — about one 381 MB block, not two
and certainly not three. So `to_states()`' columns must share key storage with the source
block rather than copying it, the raw block is released as the merged one is built, and there
never was a 1.3 GB duplication to remove. Phase 2a is still a simplification that removes a
real round trip; it is not a memory fix, and this document's earlier "~72 B/pair" arithmetic
for it was wrong.

### What the measured fixes buy

Direct A/B, `git HEAD` vs the working tree, same caps, same configuration, run back to back
in one session (file-backup swap, never `git checkout`):

| configuration | cap 5,000 | cap 20,000 |
|---|---|---|
| `git HEAD` (production today) | 799.9 MiB | 1967.9 MiB |
| + selection fixes + eigenstate-request fixes | 685.7 MiB | 1933.3 MiB |
| + `GS_SELECTION_CHUNK=8` | — | **1730.4 MiB** |

`e0` is bit-identical down the columns — `−16.940022` at cap 20,000 in all three rows — so none of
this changes the answer. (An earlier version of this table carried one shared `e0` column, which was
wrong: `e0` differs *between* caps, and not even monotonically — `−16.966995` / `−16.880057` /
`−16.940022` at 5,000 / 10,000 / 20,000. Fixed-budget CIPSI drops determinants when the cap binds,
so a larger-cap subspace need not contain a smaller one and the energies are not variationally
ordered. The comparison that matters here is down a column, at fixed cap.)

- **At cap 5,000 the code changes are worth 14.3%**; **at cap 20,000 they are worth 1.8%** —
  within noise. They cut `num_wanted`-proportional terms, and cap 20,000's peak is not
  `num_wanted`-proportional.
- **`GS_SELECTION_CHUNK` is worth 202 MB (10.5%) at cap 20,000 and costs nothing.** It bounds
  the `de2` score stack, the third of the three terms above, and it already ships (unset).
  Measured as a plateau rather than a tuned point, and free in wall-clock:

  | `GS_SELECTION_CHUNK` | peak RSS | total run | selection rounds |
  |---|---|---|---|
  | unset | 1932.6 MiB | 254.9 s | 12 rounds, 31.7 s |
  | 4 | 1730.x MiB | — | — |
  | 8 | 1730.7 MiB | 254.6 s | 12 rounds, 31.2 s |
  | 16 | 1730.x MiB | — | — |

  4, 8 and 16 all land on the same peak, so this is not tuned to one workload's edge, and the
  wall-clock difference is 0.1% — the Python-level chunk loop costs nothing against the block
  apply it wraps. `e0` bit-identical throughout. **This is the one change measured to help at
  the production cap, and it needs no code.**
- The remaining ~805 MB is `_apply_block_and_redistribute` (416 MB) plus `overlaps` (389 MB),
  in comparable halves. Neither is touched, and neither is worth more than ~20% of the peak on
  its own.

### Settled, so it need not be carried further

- **`thick_restart_block_lanczos` does not have a hidden memory explosion.** The suspected
  unbounded allocation (`T_full` sized `dim × dim` where `dim = k_ret + (m − k_blocks) ·
  p_resid`, and `p_resid` can reach `k_ret` on the Rayleigh-Ritz rebuild arm — 6.9 GB per
  rank at this workload's numbers) **never fires**: over 115 restarts at three caps, `dim`
  never exceeded `m · p`, `p_resid` never exceeded `p` (6), and `orth_err` sat at ~2e-12
  against a `RESTART_ORTH_TOL` of 1.5e-8. A trace note (`trlm_restart`) now records this so a
  future caller at a weaker `reort` would be noticed.
- **CORRECTED: the thermal manifold *does* grow with basis size.** An earlier version of this
  section claimed it did not, from caps 5,000 / 10,000 / 20,000 giving 10 / 58 / 44 — bases 47×
  smaller than the one that crashed. The crash's own log settles it. `trlm.py:450` prints
  `rank {k_ret}/{nkeep}` where `nkeep = k_blocks · p` is the *requested* width, and the log's
  sequence of `nkeep` across the expansion's cycles is **96 → 162 → 192 → 210 → 222**, then
  flat. Inverting `nkeep = p·⌈(2·n_kept + pad)/p⌉` at `p = 6`, `pad = 10` gives a kept manifold
  of **~42 → 75 → 90 → 99 → 105, saturating at ~105**. So it grows, and it does saturate — but
  at 105, not at the 44 the small caps suggested. `GS_MAX_NUM_WANTED` is still not needed (the
  growth saturates on its own), but the sizing consequence is the opposite of what was claimed.

  **This makes the shipped `_manifold_request` fix worth far more than the 1.8% measured at cap
  20,000.** At a saturated manifold of 105 the old `2·n_kept` rule requests `nkeep = 222`; the
  new rule requests `105 + max(10, 2·0) = 115`, i.e. `nkeep = 126`. That is a **1.76× cut on
  every `num_wanted`-proportional term at the scale that actually crashed** — the Krylov store,
  the `build_state` transient and the selection round's reference width alike. The 1.8% figure
  is what the fix is worth at cap 20,000, where the manifold is 44 and the peak is not
  `num_wanted`-proportional; it is not what the fix is worth in production.

- **`GS_MAX_BLOCK_WIDTH=5` actually yields a 6-wide block.** `get_eigenvectors` truncates
  `warm_block` to the knob's value and then appends the cold full-support start vector, so
  `len(psi0) = 6`. Confirmed from the crash log: every `nkeep` and every subspace `dim` in it
  (96, 162, 192, 210, 222, 432, 624, 654, 702, 750) is divisible by 6 and none is divisible by
  5. Worth knowing before anyone reasons from the knob's value as if it were `p`.
- **Why the request was ever 182.** Cycle 0 runs the *dense* branch on the 120-determinant
  seed basis, and the dense branch ignores `num_wanted` entirely when given a cut: 86 of
  those 120 states lie inside the 0.23 eV window, because a 120-determinant basis is too
  small to resolve the window. `expand`'s old `num_wanted = 2 · len(psi_refs)` then sized
  cycle 1's solve on a 35× larger basis off that 86. The converged manifold there is 10–78.
  So the "100+ converged eigenvalues" was a seed-basis artifact amplified by a doubling, not
  a physical requirement — but it was also never the dominant cost.

## Phase 2: the right cost centre, the wrong terms inside it

Everything below was measured and implemented before the `GS_MAX_BLOCK_WIDTH` correction
above. None of it is wrong — every change is bit-identical-verified against the code it
replaced — but none of it is what caused this crash to OOM. It stays in because it is a
genuine, tested improvement to a real cost center (the CIPSI selection round does allocate
real, previously-uncounted memory — just not enough to dominate once the eigensolver's own
footprint is counted correctly), and because the instrumentation it adds is what made the
correction above possible to find at all.

Reproducing the crash's exact solver basis offline (58 spin-orbitals, groups
`[[0,1,5,6]]`/`[[2,3,4,7,8,9]]`, `tau=0.025`) and running the same uncapped (no
`GS_MAX_BLOCK_WIDTH`) CIPSI expansion serially at a 20,000-determinant cap gives, at the
peak cycle:

| quantity | measured |
|---|---|
| basis size | 20,000 |
| Lanczos block width `p` | 68 |
| `H·ψ_ref` shared-support rows | 350,250 |
| new (out-of-basis) candidates | 330,250 |
| stored Hamiltonian nnz/state | 6.4 |
| raw fan-out (`build_local_operator_list`) | 25.2 (mean), 47 (max) |
| measured VmHWM (no `GS_MAX_BLOCK_WIDTH`) | 2.51 GiB |
| measured VmHWM (`GS_MAX_BLOCK_WIDTH=5`, production-matched) | **2.0 GiB, unchanged pre/post Phase 2** |

Two things came out of the (still valid) measurement work regardless:

- **`nnz_per_state`'s old default (100) was ~16× too high.** `build_sparse_matrix` only
  keeps images that land back *in* the basis after pruning; measured nnz/state is 6.4.
  `build_sparse_matrix` is not an OOM contributor at this scale (~570 kB/rank at production
  size) and needed no changes.
- **The selection round's own arrays were genuinely uncounted, and are real memory** —
  just not the dominant term once the eigensolver is measured honestly alongside it.

### `cipsi_solver.py`

- **`_apply_block_and_redistribute`**: the per-column cutoff prune used to split the shared
  block into `p` separate width-1 `ManyBodyState` objects (`to_states()`), prune each, and
  reassemble (`from_states()`) — a round trip that copies the full determinant key for every
  (row, column) pair twice (~72 B/pair on this workload's 58-spin-orbital keys). It now zeros
  pruned entries in place via the block's own buffer-protocol view (`np.asarray`, zero-copy,
  writable), matching `ManyBodyBlockState::prune_rows`'s exact `|amp|² <= cutoff²` C++
  criterion (not `abs(v) <= cutoff`, which takes a sqrt first and is not guaranteed
  bit-identical at the cutoff boundary). Verified bit-for-bit identical to the old round trip
  at 1, 2, and 3 ranks (`test_cipsi_selection_block.py`).
- **`determine_new_Dj` / `_calc_de2`**: the manifold-summed Epstein-Nesbet score used to
  materialize the whole `(p, n_Dj)` de2/mask/stack array at once. `_score_candidates`
  computes the identical quantity (verified bit-for-bit against the old formula) in
  group-aligned batches over the reference axis — a batch never splits a degenerate
  manifold, so an elementwise running max over batches equals stacking every group and
  maxing once at the end. Batch width is `GS_SELECTION_CHUNK` (new knob, default unset =
  unchunked = today's behaviour exactly). `_calc_de2`'s `de2` array also had a real dtype
  bug: `np.zeros_like(overlaps)` inherited `overlaps`' complex128 dtype for a value that is
  always real, doubling that array's footprint for nothing; now `dtype=float`.
- **Per-cycle instrumentation**: `last_selection` (candidates, admitted, discarded and
  *sub-threshold* PT2 mass, `H·ψ_ref` row count) is now surfaced every cycle under
  `verbose`, not only once a cap has bound; a `solver_trace.timed("cipsi_selection", ...)`
  frame records it when a trace is open; per-rank VmHWM and local-determinant-count
  min/max are sampled every cycle via unconditional collectives (CLAUDE.md: never gate a
  collective on rank-local state — the print is gated on `verbose`, the Allreduce is not).
  **This instrumentation is what found the eigensolver as the real culprit above** — it
  remains valuable independent of Phase 2's original hypothesis.
- **`expand`'s `memory_budget_bytes` guard** (opt-in, default `None` = no-op): an uncapped
  expansion checks its own measured peak RSS — already sampled for the diagnostic log —
  against a caller-supplied budget every cycle. The first cycle that reaches it retroactively
  adopts a fixed-budget cap at the *current* basis size, handing off to the existing,
  independently-tested fixed-budget CIPSI machinery rather than growing until a kernel OOM
  kill, which the existing `DC_CAP_STRATEGY=max` retreat (a catchable `MemoryError`) cannot
  see. **This guard is the one Phase-2 deliverable that still directly helps**: it samples
  RSS after the whole cycle, including the eigensolve step, so it catches a TRLM-driven
  blowup exactly as well as a selection-round one — it does not depend on which term
  actually dominates. Deliberately not yet wired to a production call site (see below).

### `memory_estimate.py`

`estimate_gs_peak_bytes` gained a `selection_bytes` term:
`local * selection_fanout * block_width * _SELECTION_BYTES_PER_PAIR`. This is a real,
measured cost (see above) and stays in — but it must **not** be read as "the model now
correctly predicts this workload's peak." It doesn't: the eigensolver's own footprint
(`krylov_bytes`, already an existing term) is the one still under-provisioned by roughly an
order of magnitude once `num_wanted` is supplied honestly, and that gap is not yet closed.
Both `_SELECTION_FANOUT_DEFAULT = 40` and `_SELECTION_BYTES_PER_PAIR = 50` are derived from
the array-size arithmetic in their own code comments, not fitted to noisy VmHWM deltas.

### `config.py`

`GS_SELECTION_CHUNK` (group `groundstate`, default unset = unchunked). Registered in `KNOBS`;
`doc/configuration.md` regenerated from `config.dump()`.

## Claims from earlier in this investigation that measurement refuted

- **Fan-out is ~25-41, not the ~710 first guessed from a raw term count.**
- **The selection block width is `p ≈ 104` at the thermal cut, not 206** (production, at
  `tau=0.025`; the 78/87 seen in the un-representative no-`GS_MAX_BLOCK_WIDTH` reproduction
  above is a different, non-production configuration).
- **`build_sparse_matrix` was never the OOM site.** Measured nnz/state (6.4) puts it at
  ~570 kB/rank at production scale.
- **(This document's own earlier claim) "the model now predicts 902.7 MiB/rank at the
  crashed basis size" was true only for the `selection_bytes` term in isolation, and did not
  account for `krylov_bytes` being simultaneously wrong by roughly an order of magnitude at
  the `num_wanted` production actually reaches. Retracted as a claim about the crash; the
  code (the new term itself) is unaffected and still correct on its own terms.**
- **(This document's own previous headline) "the CIPSI selection round is not the dominant
  cost — the eigensolver is." Refuted by the per-step RSS ledger above: at the peak cycle the
  selection round is 81% and the eigensolver 19%.** That claim came from instrumenting *one*
  TRLM call and reading the whole cycle's RSS jump as belonging to it, in a cycle where a large
  selection round ran immediately afterwards and was never separately sampled. The lesson is
  the one this file keeps relearning: sample both sides of every step, and let the first step
  that raises the high-water mark name itself. A single before/after pair around one callee
  cannot distinguish "this allocated it" from "this ran just before whatever did".
- **"`num_wanted` reaches 104–222 and the thermal manifold grows with the basis."** Measured
  kept-manifold size is 10 / 58 / 44 at caps 5,000 / 10,000 / 20,000 — bounded and
  non-monotonic. The 104–222 figures were the *request*, inflated by the old doubling, and the
  86 it was derived from is a property of the 120-determinant seed basis (see above).
- **"`GS_MAX_BLOCK_WIDTH` caps `krylov_bytes`."** It does not, or barely: the retained column
  count is `p · blocks ≈ 4 · num_wanted + 20p`, and `_size_subspace` grows `blocks` as `p`
  shrinks, so capping `p` only shrinks the `20p` headroom term. `get_eigenvectors`' own comment
  at the truncation site claims otherwise and should be corrected when that code is next
  touched.

### Structural explanations for the 250× per-rank gap, refuted by reading the source

Before measuring anything multi-rank, four candidate explanations for "3,710 determinants/rank
cannot cost 5 GiB" were each killed statically. Recorded because every one of them looks like
the obvious answer, and re-deriving them costs an afternoon:

- **The basis is not secretly replicated.** `manybody_basis.py:172`:
  `is_distributed = comm is not None and comm.size > 1`, true at 256 ranks.
- **There is no replicated global index map.** `_index_dict` is `{state: self.offset + i}` over
  `local_basis` only (`manybody_basis.py:317`), and `_index_sequence` resolves out-of-rank keys
  by hash-routed collective (`:656`) rather than from a local table. Measured per-determinant
  cost of the basis holdings: ~117 B/det (48 B key object + ~69 B dict entry), so ~434 kB/rank
  at production's local size.
- **`add_states` does not gather the new determinants.** It routes through
  `distribute_determinants`, a hash-partitioned `Neighbor_alltoallv`
  (`mpi_comm.py:349-394`); `all_received` is this rank's share, not everyone's.
- **No communicator leak.** Every `Create_dist_graph_adjacent` is matched by a `Free()`
  (`mpi_comm.py:331`, `:394`) or goes through the keyed cache with eviction (`:92-98`). This was
  the most promising candidate — it is invisible to every serial measurement, because
  `distribute_determinants` returns early at `comm.size <= 1` — and it is not there.

What these leave standing: flat per-rank terms, terms that grow with *rank count* rather than
with local work, accumulation across the double-counting search's many trials
(`SectorCache(max_size=3)` holds whole solvers), and costs outside impurityModel entirely
(RSPt co-residency; 128 siblings against one node's memory).

### MPI runtime state per rank saturates — refuted as the missing term

One surviving hypothesis was that MPI's own per-rank state grows with communicator size, which
no 1–8 rank solver sweep could see. Measured directly with the workload stripped away (import,
`MPI_Init`, then `Barrier`/`Allreduce`/`Alltoall`/`Allgather` so lazily-created endpoints are
actually allocated), oversubscribed well past this box's 6 cores:

| ranks | after import | after collectives | collective delta |
|---|---|---|---|
| 1 | 206.9 MiB | 207.3 MiB | 0.4 MiB |
| 2 | 218.8 | 223.3 | 4.5 |
| 4 | 218.6 | 231.4 | 12.8 |
| 8 | 219.1 | 248.5 | 29.4 |
| 16 | 219.1 | 248.5 | 29.4 |
| 32 | 211.3 | 244.9 | 33.5 |
| 64 | 209.7 | 243.3 | **33.6** |

**It saturates.** The per-peer cost grows only to n≈8 and is flat from 16 to 64 at ~34 MiB, so
Open MPI is not allocating per-peer state linearly in rank count. Extrapolating the plateau,
MPI runtime state at 256 ranks is tens of MiB per rank, not GiB. That removes it as a candidate
for the missing ~4.7 GiB/rank.

## The rank sweep: what actually exhausted 5 GiB/rank

Everything before this section was measured on one rank. The sweep runs the same SMO DC sector
solve at 1, 2 and 4 ranks on the crash's own archive, `GS_MAX_BLOCK_WIDTH=5`,
`OPENBLAS_NUM_THREADS=1`, one process per data point (`peak_rss_bytes()` is a process-lifetime
high-water mark, so two caps in one process make the second report the first's peak).

**Strong scaling, global cap 20,000 fixed.** Kept manifold per cycle, selection width per cycle
and `e0` are *identical* at every rank count (`78,68,39,47,42,44x8`; `e0 = -16.940022122` to 1e-14,
and bit-identical across ranks within every run), so only the rank count varies:

| ranks | local dets (max) | floor | peak/rank | above floor |
|---|---|---|---|---|
| 1 | 20,000 | 280.6 MiB | 1938.1 MiB | 1657.6 |
| 2 | 10,017 | 292.7 | 1622.7 | 1330.0 |
| 4 | 5,387 | 292.8 | 1128.8 | 836.0 |
| 6 | 3,388 | 292.2 | 924.5 | 632.4 |
| 8 | 2,731 | 292.6 | 795.7 | 503.1 |

**Distribution is far from ideal: 8x the ranks buys 3.3x less memory, not 8x.** A log-log fit gives
`above ∝ local^0.62` (R² = 0.973) along this path, with the flat setup cost removed. Note the n=6 row: its local count (3,388)
is essentially production's mean of 3,710, and it costs **924.5 MiB/rank** against production's
~5 GiB — the clearest statement of the gap, at matched local work.

### `routing_hash` is deliberately non-uniform, and the cap is sized off the mean

`SlaterDeterminant::routing_hash` (`src/cython/SlaterDeterminant.h:31-64`) is **by design** a
locality-preserving linear hash over GF(2^64), not a dispersing one — its own comment says why:
hopping terms change only a few bytes, so each rank talks to a bounded number of peers and the
communication graph stays sparse "to 100,000+ ranks". The implementation sums popcount-weighted
words per byte. Because a charge sector has **fixed electron number**, those popcounts sum to a
constant (53 here), so the hash lives on a restricted lattice and `hash % size` cannot be
uniform.

Measured on 20,000 real determinants from this workload, against an exact Monte-Carlo null
(the same number of items thrown uniformly into the same number of buckets, 2000 trials):

| ranks | measured max/mean | uniform-null mean | null p95 | null max | p-value |
|---|---|---|---|---|---|
| 8 | 1.092 | 1.028 | 1.046 | 1.073 | < 0.0005 |
| 16 | 1.378 | 1.050 | 1.075 | 1.101 | < 0.0005 |
| 64 | 1.808 | 1.135 | 1.181 | 1.248 | < 0.0005 |
| 128 | 1.901 | 1.213 | 1.274 | 1.421 | < 0.0005 |
| **256** | **2.790** | 1.333 | 1.421 | 1.626 | **< 0.0005** |

At 256 ranks the busiest rank owns **2.79x the mean and 14.5x the lightest** (min/mean = 0.192).
The skew **grows with rank count**, and it is not small-sample noise: every measured value lies
outside the null's full 2000-trial range.

This cross-validates against the sweep itself, which measured the same thing a completely
different way: the solver reported `local[min,max]` of 10,017/10,000 at 2 ranks (ratio 1.002)
and 5,387/5,000 at 4 ranks (1.077) — matching the hash test's 1.002 and 1.077 exactly.

**Why this is the crash.** `suggest_truncation_threshold` sizes the cap from
`local = n_global / ranks` — the *mean* — while an OOM happens on the *heaviest* rank. At 256
ranks that under-budgets the binding rank by 2.79x, and it explains the one production fact
nothing else did: **only 13 of 256 ranks died** because only the heavy tail of the distribution
was over budget.

### A forward prediction, and the one thing that stops it from being sharp

Production's heaviest rank held `2.79 x 3,710 = 10,351` determinants, and the crash log's `nkeep`
sequence puts its kept manifold at `p ~ 105` (against 44 here). Extrapolating from the n=8 anchor
by `local` and `p`, with the flat setup cost subtracted so it is not inside a term whose scaling is
being fitted:

| assumption about the local-count response | predicted peak/rank |
|---|---|
| linear (`alpha = 1`) | **4.59 GiB** |
| the measured strong-scaling exponent (`alpha = 0.62`) | **2.88 GiB** |
| observed in production | **~5 GiB** |

Production sits at or just above the linear end. **The factors are identified; the exponent is
not** — and that is the whole remaining uncertainty, down from a 250x gap to a factor of 1.6.

(Subtracting `after_setup` rather than the bare interpreter floor barely matters here — the
exponent moves 0.603 → 0.618 and the band 2.91-4.73 → 2.88-4.59 GiB — but a scaling fit is only
defensible with the flat part removed, and checking was cheap.)

**Why this data cannot pin the exponent.** The strong-scaling path ties local count to rank count
(`local = 20,000/n`), so `above ∝ local^0.62` is a *joint* law, not a local-count law. The deficit
from the ideal `1/n` can be read two ways, and they diverge wildly when extrapolated:

- as a pure local-count effect — giving the 2.88 GiB row above;
- or as a separate rank-count penalty `g(n)`, in which case `c·local·g(n) ∝ n^-0.62` implies
  `g ∝ n^0.38`, a ~3.7x penalty at 256 ranks relative to 8, and a prediction far **above** the 5 GiB
  observed — so that reading is wrong too, probably because the penalty saturates the way the MPI
  floor does.

Both readings fit the five measured points; neither survives a 32x extrapolation in rank count. The
pairwise exponents say the same thing less formally: 0.599, 0.748, 0.748, 1.061 against the n=8
anchor — not a constant, so no power law is being measured over a wide enough range.

Separating the two requires varying local count **at fixed rank count**, which needs a basis far
larger than this box holds — or varying rank count far enough that the penalty's saturation shows.
**One 256-rank job does the first directly:** at that rank count the within-run local spread is
14.5x (against 1.17x at 4 ranks), so a log-log fit over the per-rank `(local, RSS)` pairs measures
the local-count response in a single run. The prepared probe does exactly that fit and prints it.

### What skew does *not* explain: "all 13 on one node"

Tested directly, since it is the one production fact skew seemed to account for. Production laid
256 ranks over 2 nodes, so node 0 held ranks 0-127 and node 1 held 128-255. If `routing_hash % 256`
put systematically more determinants in one index half, that asymmetry would land on one node:

| quantity | measured |
|---|---|
| determinants in buckets 0-127 vs 128-255 | ratio **1.0113** |
| uniform null: `abs(ratio-1)` at least this large | **42.9%** of 5000 trials |
| the 13 heaviest buckets, how many in the first half | **6 of 13** |

**No node asymmetry.** The skew is spread evenly across node halves, so it does not explain why the
deaths were concentrated. The likely explanation needs no asymmetry in the hash: both nodes were
equally loaded on average and *both* were near the edge, so whichever node had slightly less
headroom tipped first, and its OOM killer took 13 victims there. That is consistent with every
measured fact, but it is an inference, not a measurement — the prepared probe reports per-node RSS
totals, which is what would confirm it.

### UNEXPLAINED, and possibly more important than the memory result: `e0` gets *worse* with a bigger cap

The sweep's rank-count correctness guard passes cleanly — cap 20,000 gives `e0 = -16.940022122` at
1, 2, 4, 6 and 8 ranks, agreeing to 1e-14, bit-identical across ranks within every run. But across
*caps*, on the same archive, the same 120-determinant seed basis and the same sector
(51 electrons, impurity occupation 3):

| cap | 5,000 | 10,000 | 20,000 |
|---|---|---|---|
| `e0` | **−16.966995** | −16.880057 | −16.940022 |

Non-monotonic, spanning **87 meV**, and the *best* (lowest) energy comes from the *smallest* basis.
Independent of the `num_wanted` pin: pinned and unpinned cap-5,000 runs both give −16.966995027.

**It is not truncation discarding determinants the eigenvector needs.** That was the obvious
explanation and it is refuted by the logs: at cap 20,000 the per-restart `MinEigval` improves
monotonically from −16.286487 to −16.940022 and **never once goes below it**. The run does not find
−16.967 and lose it; it never finds it at all.

So the two caps converge to **different states**, and since each result is a Rayleigh quotient on its
own subspace, the true ground state satisfies `E0 <= -16.966995`. **The cap-20,000 run therefore
returns a state ~27 meV above the ground state** — a larger basis giving a worse answer because the
greedy PT2 expansion walked into a different basin.

This is adjacent to, but not the same as, what `smo-dc-is-truncation-limited` already records (that
the *DC answer* `mu` moves by 0.255 with the cap). That is a difference of sector energies; this is a
single sector's ground state being wrong by 27 meV in a way that gets worse with more resources. It
matters for the double-counting search specifically, because the criterion **differences** sector
energies and a basin-dependent 27-87 meV error does not cancel.

**Not chased here** — it is an accuracy question, and this investigation is about memory. Flagged
because it was found by the sweep's own correctness guard and because no amount of memory
engineering fixes it. The obvious next probe is whether the cap-20,000 expansion's *selected
determinants* ever contain the cap-5,000 run's support.

## The 256-rank probe: skew confirmed exactly, and refuted as the cause

One 30-minute 256-rank job on Arrhenius (`memprobe-2319502.out`), three caps, one sector solve each.
**It ran the deployed HEAD code, not the working tree** — zero `local[min,max]` lines and an empty
manifold list prove it — which is the right thing for reproducing the crash (the old
`num_wanted = 2·len(psi_refs)` rule and HEAD's estimator are what production used), but it cost the
manifold trace. That was recovered from TRLM's `rank k_ret/nkeep` lines instead.

**Confirmed exactly.** `ranks_on_node = [128,128]`; `MemAvailable` 667.93 GiB; `available_bytes_per_rank`
**5.22 GiB**, reproducing the crash log's "5.0 GiB/rank". Startup floor **213.2 MiB/rank** (26.65 GiB
of a node before any work). And the load skew, against the off-cluster prediction of 2.79 / 0.19 / 14.5:

| cap | local dets min / mean / max | max/mean | min/mean | max/min |
|---|---|---|---|---|
| 20,000 | 15 / 78 / 218 | **2.790** | **0.192** | **14.5** |
| 50,000 | 45 / 195 / 402 | 2.058 | 0.230 | 8.9 |
| 100,000 | 160 / 391 / 727 | 1.861 | 0.410 | 4.5 |

The cap-20,000 row matches the offline Monte-Carlo prediction to three decimal places. The skew is
real, it is exactly as large as computed, and it **shrinks as local counts grow** — so at production's
mean of 3,710 it would be milder than 2.79, not worse.

### And skew is not the cause — the exponent is ~0, not ~1

The probe's whole purpose was to fit the local-count response at fixed rank count, where the lever is
14.5x instead of 1.17x:

| cap | lever | fitted exponent | R² |
|---|---|---|---|
| 20,000 | 14.5x | **0.025** | 0.24 |
| 50,000 | 8.9x | 0.035 | 0.15 |
| 100,000 | 4.5x | 0.111 | 0.42 |

**Per-rank memory is essentially independent of local determinant count.** A rank owning 218
determinants and one owning 15 use within 10% of the same memory (RSS min 594.5, max 656.0 MiB).
Neither of the two readings this document offered — `alpha = 1` (4.59 GiB) or `alpha = 0.62`
(2.88 GiB) — is right; the truth is `alpha ~ 0`.

**So the "skew x manifold" explanation is refuted.** Skew is measured and exactly as predicted, but it
cannot be the mechanism: scaling a quantity that does not depend on local work by a load-imbalance
factor predicts nothing. This is the fourth attribution in this investigation to be killed by
measurement, and the second to be killed by a measurement this document itself proposed.

### What the probe says the cost actually is

Per-rank peak, and the part above the floor:

| cap | global dets | local mean | peak/rank (max) | above floor | HEAD model | under-prediction |
|---|---|---|---|---|---|---|
| 20,000 | 20,000 | 78 | 656.0 MiB | 447.0 | 0.4 MiB | **1028x** |
| 50,000 | 50,000 | 195 | 864.3 | 657.1 | 1.1 MiB | 609x |
| 100,000 | 100,000 | 391 | 1012.4 | 801.4 | 2.2 MiB | 372x |

447 MiB above the floor for **78 determinants** is ~5.7 MB per determinant, which is by itself proof
that the dominant term is not per-determinant. Above-floor memory grows only `~global^0.3` while local
work grows 5x, and within a run it is flat in local work. **The dominant per-rank term is set by
something collective or global, not by this rank's own share.** The model, which sums only
local-proportional terms, is therefore wrong by 372-1028x — and the direction of its error is
structural, not a missing coefficient.

Per-node RSS totals were 77.6-113.5 GiB against 667.93 GiB available, so this probe never approached
the edge — as intended, it is 9.5x smaller than the basis that died and runs one solve, not a search.

### A measured candidate, of the right shape and magnitude

`distribute_determinants` and `graph_alltoall_block` build a **distributed-graph communicator** whose
neighbour set is "every rank I have data for". Measured locally (Open MPI 5.0.9), a complete-graph
`Create_dist_graph_adjacent` + `Neighbor_alltoallv` costs, per rank:

| ranks (complete graph) | 2 | 8 | 32 | 64 |
|---|---|---|---|---|
| delta | 4.7 MiB | 29.4 | 128.8 | 261.2 |
| per neighbour | 2.35 | 3.68 | 4.02 | **4.08** |

Linear in neighbour count at ~4 MiB each, and — unlike the plain collectives, which saturate at
33.6 MiB — it **does not saturate**. Extrapolated to a complete graph at 256 ranks: ~1 GiB/rank.

Those three figures are each a separate process measuring one *complete* (and therefore valid,
symmetric) graph via `VmHWM`, so they stand — but they include a **one-off MPI first-touch cost**
since isolated by a warm-up measurement (12.9 MiB at 6 ranks). Netting it out puts the marginal cost
nearer **3.6-4.0 MiB/neighbour**, which does not change the conclusion.

Two methodology traps found while building the degree-ladder version of this probe, both worth
remembering: a dist-graph whose neighbour set is **asymmetric** (`{r+1..r+deg}` as both sources and
destinations) is an invalid graph, and `Neighbor_alltoallv` on it **hangs forever** rather than
erroring — four "timeouts" that looked like oversubscription slowness were deadlocks. And a
per-configuration delta must be read from **`VmRSS`, not `VmHWM`**: a high-water mark never
decreases, so measuring the largest graph first makes every later row report exactly zero.

This fits every observation: independent of local determinant count (so `alpha ~ 0`); invisible to
every serial measurement (`distribute_determinants` returns early at `comm.size <= 1`); and growing
with global basis size only weakly, through graph density.

It also undercuts the hash's rationale. `routing_hash` is locality-preserving so that each rank talks
to few peers — but locality bounds the targets of *one determinant's* images, and a rank owns many.
At cap 20,000 each rank generates ~1,950 candidate rows into 256 buckets (~7.6 per bucket), so the
probability a given peer receives nothing is `e^-7.6` ~ 0.05%: **the graph is complete.** The design's
cost (load skew) is paid in full while its benefit (sparsity) is not realised at this rank count.

*Correction (round 5 measurement, and the reason behind it).* That estimate assumed the ~1,950
images scatter *uniformly* over the buckets. They do not: `routing_hash` is exactly linear in the
occupied orbitals (one 64-bit weight per occupied orbital, `SlaterDeterminant.h`), so an operator
term that flips a fixed set of bits shifts the hash by a constant and every image of a rank's
determinants lands on `(rank + Δ_term) mod size`. The reachable set is the term set's Δ-shifts, not
a Poisson draw -- measured at 11% of rank pairs, ~28 sources per rank (37 max), on the real
Hamiltonian at 256 buckets (below), and independent of how many determinants a rank owns. The
`matvec_exchange` trace note (`GS_MATVEC_EXCHANGE=graph`, `doc/plans/dc_smo_performance.md`)
reports that degree from every run. The candidate redistribution and the matvec reduce-scatter
are the same Δ-shift graph, so neither needs an offline count any more.

**Not yet proven**, and the gap is named: 4 MiB/neighbour is Open MPI 5.0.9 on a laptop, while
Arrhenius runs a different MPI under `mpprun`. The cluster's 447-801 MiB above floor is the same order
as a 256-neighbour graph at 1.7-3.1 MiB each, which is supportive, not conclusive.
`arrhenius_handover/graph_cost.py` + `graphcost.sbatch` settle it in a **5-minute job with no archive
and no solver** — and they also distinguish per-neighbour cost from per-rank cost, which decides
whether the fix is "make the graph sparse again" or "stop using neighbourhood collectives".

### The dist-graph candidate: REFUTED on the cluster

`graphcost-2319919.out`, 256 ranks, one launch, no archive and no solver:

| neighbours | 2 | 4 | 8 | 16 | 32 | 64 | 128 | **255 (complete)** |
|---|---|---|---|---|---|---|---|---|
| RSS delta | 3.7M | 0.2M | 0.3M | 0.4M | 0.8M | 6.4M | 5.6M | **7.6M** |
| per neighbour | 1.869M | 0.062 | 0.033 | 0.027 | 0.025 | 0.100 | 0.044 | **0.030M** |

**0.030 MiB per neighbour, 7.6 MiB for a complete 256-rank graph** — against ~4 MiB per neighbour
measured locally on Open MPI 5.0.9, a factor of 130. The whole ladder moved peak RSS by 18 MiB.
The distributed-graph communicator is **not** the missing term; the local measurement was an Open
MPI artefact that does not transfer to this cluster's MPI, and extrapolating it was wrong.

A follow-up killed the obvious repair too: the real code performs thousands of redistributions per
cycle, so MPI buffer pools might grow with *traffic* rather than topology. Measured over 2000
successive `Neighbor_alltoallv` calls on a complete graph: **+17.02 MiB after 1 exchange, +17.02
after 2000.** Flat. No accumulation.

### Where that leaves it: six hypotheses down, and the method that has never failed

| hypothesis | verdict |
|---|---|
| local / per-determinant work | **no** — exponent ~0 against a 14.5x lever |
| load skew as the mechanism | **no** — confirmed at 2.790x, but memory ignores local work |
| dist-graph topology state | **no** — 7.6 MiB at 256 ranks on this cluster |
| MPI buffer-pool growth with traffic | **no** — flat over 2000 exchanges |
| model / solver setup | **no** — ~22 MiB (+18.3 archive, +3.5 solver basis) |
| plain collective state | **no** — saturates at ~34 MiB by 16 ranks |

447 MiB/rank above the floor at 78 local determinants remains unattributed. **Every one of the six
came from reasoning about the code; every refutation came from a measurement.** The per-step RSS
bracket is the only instrument in this investigation that has ever produced a surviving answer, and
it has never been run at production's rank count — that is the gap, not another hypothesis.

`arrhenius_handover/bracket_probe.py` closes it: `build_basis_and_solver` / `expand` /
`get_eigenvectors` called separately (what `solve_sector` does internally), peak RSS MAX-reduced
between phases, **HEAD APIs only so nothing needs pushing**. Validated at 2 ranks, where it puts
185.7 of 188.5 MiB in `expand` and 0.0 in the final eigensolve.

## ANSWER: `expand` holds a non-distributed term that scales with the GLOBAL basis

The per-phase bracket at 256 ranks (`bracket-2319968.out`), run against HEAD so nothing needed
pushing:

| phase | cap 2,000 | cap 20,000 |
|---|---|---|
| 0. import + MPI_Init | 213.5 MiB | — |
| 1. `load_selfenergy_archive` | +12.3 | — |
| 2. `prepare_solver_basis` | +3.4 | — |
| 3. `build_basis_and_solver` | +3.8 | +0.0 |
| 4. **`expand`** | **+205.0** | **+253.0** |
| 5. `get_eigenvectors` | +0.0 | +0.0 |

**The entire cost is `expand`.** Setup is 15.7 MiB, basis construction 3.8, the final eigensolve
exactly zero. And `expand` cost 205 MiB at a local mean of **8 determinants per rank** (min 0, max
68) — nothing algorithmic scales that way, since at 8 local rows the Krylov store is ~60 kB.

### The same bracket at 1, 2 and 256 ranks settles what it depends on

Memory above the setup mark, at a **fixed global basis of 2,000**:

| ranks | local dets/rank | above setup |
|---|---|---|
| 1 | 2,000 | 197.9 MiB |
| 2 | 1,000 | 188.5 MiB |
| 256 | **8** | 208.9 MiB |

**A 250x range in local work, and the memory does not move.** Distribution buys nothing at this
basis size. That is the same fact the exponent fit reported (`local^0.025`), now localised to a
single function and reproduced at one rank.

At a global basis of 20,000 there *is* a distributed component — 1 rank needs ~1,640 MiB above
setup against 256 ranks' ~462 MiB, a 3.5x reduction for 256x the ranks — but underneath it sits a
**non-distributed floor of ~460 MiB**. So:

| global basis | non-distributed floor, per rank |
|---|---|
| 2,000 | ~200 MiB |
| 20,000 | ~460 MiB |

~`global^0.36`. Extrapolated to production's 949,834 determinants: **~2.0 GiB/rank** of
non-distributed cost, before any distributed term, on every one of 256 ranks. That is the shape of
the crash, and at global 2,000 it is **~100 kB per global determinant**, which is absurd enough to
be a defect rather than a design cost.

### Why this matters more than the number

**The investigation comes home.** This is reproducible at **one rank** (197.9 MiB at global 2,000,
in ~30 seconds), so localising it further needs no cluster time, no 256-rank job, and no MPI at all.
Every earlier pass looked for the cost in the wrong place because `solve_sector` hides the phase
boundary: the serial ledger bracketed *cycles within* `expand` and so measured only the part that
does scale, while the non-distributed floor sat underneath, constant and invisible.

Six hypotheses were eliminated getting here (see above); none was needed. The bracket found it in
one run, at the rank count where the problem lives, by measuring instead of reasoning.

### Bisecting `expand` at 1 rank — and why that was the wrong rank count

Temporary sub-step probes inside `expand` (reverted byte-identically afterwards; the gate result
attaches to the tree without them), 1 rank, global basis 2,000, 196.6 MiB total:

| cycle | eigensolve | `determine_new_Dj` | `add_states` |
|---|---|---|---|
| 0 | +2.6 MiB | +27.1 MiB | +0.0 |
| 1 | **+51.9** | **+114.8** | +0.0 |
| 2-8 | 0 | 0 | 0 |

Selection 141.9 MiB (72%), eigensolve 54.5 MiB (28%), `add_states` exactly zero, and **everything
in the first two cycles** — seven further cycles at the same basis size add nothing. Reproduced
independently at 2 ranks by monkeypatching the two methods from outside (143.9 / 41.5 MiB).

**CORRECTION to this document's own conclusion above.** At 256 ranks with 8 determinants per rank,
`determine_new_Dj`'s arrays are ~0.25 MB, not 142 MiB — so the 256-rank 205 MiB **cannot** have this
composition. The claim that `expand` holds a single *non-distributed* term scaling with the global
basis conflated two different costs which happen to sum to ~200 MiB at every rank count tried
(196.6 at 1 rank, 188.5 at 2, 208.9 at 256). That coincidence is what produced the wrong reading.

What is actually established: the cost is entirely inside `expand`; at 1-2 ranks it is
local-proportional selection plus eigensolve arrays, concentrated in the first two cycles; at 256
ranks something of similar magnitude is present that cannot be those arrays. The composition at
production's rank count is **not yet measured**.

### What is left

Run the same sub-step bracket at 256 ranks. `arrhenius_handover/subbracket_probe.py` does it by
**monkeypatching** `determine_new_Dj` and `get_eigenvectors` from outside, so it needs nothing
pushed, and it prints an explicit *unattributed* remainder — if the two methods account for only a
fraction of the above-setup total, the rest is outside both, and that gap is the finding. Validated
at 2 ranks.

**A note on method, since this investigation has now corrected itself seven times.** Every wrong
answer came from reasoning about which array is biggest; every correct one came from bracketing and
reading the increment. The bracket has never yet been wrong — but it is only ever right *about the
configuration it ran in*, which is the mistake made here: a 1-rank bisection cannot describe a
256-rank cost, however cleanly it localises.

## ROUND 3 (256 ranks): it is the EIGENSOLVER, and the serial ledger had it backwards

`subbracket-2324132.out` — `determine_new_Dj` and `get_eigenvectors` wrapped from outside
(monkeypatched, HEAD, nothing pushed), 256 ranks, cap 2,000:

| | 256 ranks | 1-2 ranks (local) |
|---|---|---|
| `get_eigenvectors` | **+197.4 MiB (94.9%)** | +38.4 MiB (20.7%) |
| `determine_new_Dj` | +10.5 MiB (5.1%) | +147.0 MiB (79.3%) |
| attributed | 207.9 of 215.9 MiB above setup (**96%**) | 185.4 of 192.6 (96%) |

**The composition is exactly inverted between 2 ranks and 256.** Serially the selection round
dominates; at production's rank count the eigensolver is 95% and selection is noise. Both
measurements are correct about their own configuration, and neither generalises — which is the
single most important methodological result in this document.

That also settles an old score. This document's *first* headline was "the eigensolver is the
bottleneck"; a serial per-step ledger refuted it and put selection at 81%. **Both were right.** The
serial refutation was sound serially, and the original claim named the component that actually
dominates at 256 ranks — for reasons neither pass had measured.

### Localised to one call

| call | basis | local_max | RSS | delta |
|---|---|---|---|---|
| `get_eigenvectors` | 120 | 32 | 235.0 -> 239.1 | +4.1 (dense branch) |
| `get_eigenvectors` | **2,000** | **75** | 251.4 -> **442.6** | **+191.1** |
| `get_eigenvectors` | 2,000 | 67 | 442.6 -> 442.6 | +0.0 |
| ... every later call | 2,000 | 66-75 | 442.6 | +0.0 |

**191.1 MiB on the first distributed Lanczos solve, with 75 local determinants** — and nothing
afterwards. The 120-determinant call took the *dense* branch and cost 4.1 MiB, so this is specific
to the distributed path.

**Caveat on reading that as "one-off":** `peak_rss_bytes()` is a high-water mark, so an allocation
that recurs at the same size every cycle also shows +0.0 after the first. The measurement proves the
peak is set by the first call at that basis size; it does **not** prove the allocation happens only
once. Recorded because the distinction changes which mechanisms are candidates.

### The leading candidate, and why it is not yet a conclusion

`BlockLanczosArray.pyx:609` implements the matvec reduction as

```cython
for dest in range(size):
    comm.Reduce(chunk_buf[:dest_count, :], wp_arr, op=MPI.SUM, root=dest)
```

— `size` separate collectives with `size` different roots, per matvec. The code comment calls this
a "row-chunked reduce-scatter"; it is one, hand-rolled as 256 collectives. At 75 local rows and
width 6 every Python-level array in that kernel is under ~1 MiB, so 191 MiB is not algorithmic, and
MPI internal state allocated per root fits the shape (~0.75 MiB per peer x 256).

**Tested locally and inconclusive.** Comparing the loop against a single `Reduce_scatter` over
2-64 ranks gives a non-monotonic 0.3 / 8.6 / 8.6 / 12.8 / 16.9 / 0.3 MiB — no clean scaling, and it
collapses at 64. This document has already recorded that local MPI measurements do not transfer to
this cluster: the dist-graph probe read ~4 MiB/neighbour on Open MPI 5.0.9 against **0.030** on
Arrhenius, a factor of 130. So the local null is not evidence either way.

The test that would settle it is `reduce_pattern.py` at 256 ranks — seconds of compute, no archive,
no solver — comparing the two spellings of the same reduction. It doubles as a test of the fix:
replacing 256 collectives with one `Reduce_scatter` would be a large latency win regardless of what
it does for memory.

## ROUND 4 (256 ranks): it is `restarted_lanczos`, and the Reduce-loop hypothesis is dead

`eigenbracket-2327908.out`, 256 ranks, cap 2,000. Part A wrapped the four callees of
`get_eigenvectors` without presupposing which one mattered:

| callee | cost | share of `get_eigenvectors` |
|---|---|---|
| **`restarted_lanczos`** | **+118.6 MiB** | **89.7%** |
| `build_sparse_matrix` | +0.8 | 0.6% |
| `build_distributed_vector` | +0.1 | 0.1% |
| `build_state` | +0.1 | 0.0% |
| `get_eigenvectors` TOTAL | +132.3 | 88% of the 150.3 MiB above setup |
| unattributed inside it | +12.7 | 10% |

**The TRLM driver is ~79% of the entire per-rank cost at production rank count**, and the
attribution is 90% complete, so it is not hiding elsewhere. The cost is spread across successive
calls (+80.2, +20.8, +8.7, +2.9, +2.5), not a single allocation.

### Part B: the per-destination Reduce loop is NOT the cost

`BlockLanczosArray.pyx:609`'s `for dest in range(size): comm.Reduce(..., root=dest)` was the leading
hypothesis. Measured at 256 ranks against the single collective that would replace it:

| spelling | cost |
|---|---|
| per-destination `Reduce` loop (256 roots) | **5.0 MiB** (0.020 MiB/peer) |
| one `Reduce_scatter` | **13.5 MiB** |

The loop costs 5 MiB, and the "fix" costs *more*. **Eighth hypothesis refuted by measurement**, and
the reason Part A was built hypothesis-free in the same job: had the probe only tested the
mechanism, it would have returned a clean negative and no answer.

### What does not yet fit

`trlm.py:508` allocates `T_full = np.zeros((dim, dim))` — dense, replicated on every rank,
independent of local rows, which is exactly the shape a cost flat in local determinants should
have. But `restarted_lanczos` costs **23.1 MiB at 2 ranks and 118.6 MiB at 256**, at the *same*
global basis and therefore the same `dim`. A replicated `dim x dim` matrix is identical at both, so
`T_full` alone cannot explain a 5x difference. Something inside the driver scales with rank count
and is not the Reduce loop.

The `trlm_restart` trace note added earlier (`restart`, `k_ret`, `p_resid`, `dim`, `budget`,
`orth_err`) is now deployed on the cluster and reads out `dim` directly — the cheapest way to
confirm or kill the `T_full` branch at 256 ranks, and the note whose comment claims the concern was
"retired". It was retired *serially*; at 256 ranks it is unverified.

### Local rank scan of `restarted_lanczos`: no law, and a coincidence that nearly became a claim

Same probe at 1/2/4/6 ranks locally, fixed global basis 2,000, against the cluster's 256:

| ranks | 1 | 2 | 4 | 6 | **256** |
|---|---|---|---|---|---|
| `restarted_lanczos` | 52.2 | 22.8 | 37.4 | 35.5 | **118.6** |

It falls 1 -> 2 (distribution working), turns between 2 and 4, then sits flat near 36. A two-point
log fit on the rising branch, `22.8 + 14.6 log2(ranks/2)`, predicts **125 MiB at 256** against the
measured 118.6 — and is **wrong**: it predicts 45.9 at 6 ranks where the measurement is 35.5. The
agreement at 256 was coincidence, and 1-6 ranks are too few and too closely spaced to establish any
scaling. Recorded because it was one sentence away from becoming this document's ninth wrong claim.

**What is established, and what is not.** Established: the cost at production rank count is
`restarted_lanczos`, ~79% of the whole solve, and it is none of the four callees around it nor the
per-destination `Reduce` loop. Not established: which allocation inside the driver, or how it scales.
Nothing inside it *looks* like a candidate — `chunk_buf` is `max(counts) x n` and shrinks as ranks
grow, `Q_basis` is `(N_local, dim)` and shrinks too, and `T_full` is `dim x dim` and is
rank-independent — which is precisely why the next step must be a bracket and not a reading.

**Marginal value, stated honestly.** Going further is a bracket inside `trlm.py` at 256 ranks
(`_build_full_T`, `_trlm_extract`, `_trlm_core` are all module-level and patchable, and the deployed
`trlm_restart` note reads out `dim` directly). But that answers an *efficiency* question. The
*survivability* question is already answered by the corrected trip-wire, which reacts to measured
RSS and does not care which line allocates it. Two unfinished items outrank a fifth cluster run:
committing the guard fix, and wiring `gs_num_wanted`.

## The node view: replicated memory costs `ranks_per_node` x, and that is a free lever

Prompted by the right question — isn't `T_full` replicated, so shouldn't it scale with rank count?

**Per rank it does not.** Each rank allocates one `T_full` of `dim**2 * 16` bytes regardless of P.
Measured with the deployed `trlm_restart` trace note at cap 2,000: `dim` is **identical** at 1, 2 and
6 ranks (min 54, max 552), `p_resid` never exceeds 6, and `dim > budget` never occurs. So
`T_full` = **4.65 MiB/rank**, flat in rank count — far too small to be the 118.6 MiB in
`restarted_lanczos`, and it cannot produce the 23 -> 118.6 MiB rise from 2 to 256 ranks.

**Per node it does.** `ranks_on_node` copies of every replicated object sit on one node, and the OOM
killer acts at node level. `available_bytes_per_rank` already divides by `ranks_on_node` for exactly
this reason, but the consequences are worth stating in node units:

| replicated per rank | per rank | x 128 ranks/node |
|---|---|---|
| interpreter + numpy/scipy/mpi4py + impurityModel | 213 MiB | **26.6 GiB** |
| model + solver-basis setup | ~20 MiB | 2.6 GiB |
| `T_full` | 4.65 MiB | 0.58 GiB |
| **production total** | **~5 GiB** | **~640 GiB of 667.93 available** |

The last row is the crash, essentially exactly.

### `ranks_per_node` is a job-script lever, and the current setting is close to the worst choice

`job.rspt` requests `-n 256 -c 2`, which on 256-core nodes puts **128 ranks on each of 2 nodes** and
yields 5.22 GiB/rank. Alternatives, with no code change:

| request | nodes | ranks/node | budget/rank |
|---|---|---|---|
| `-n 256 -c 2` (today) | 2 | 128 | 5.22 GiB |
| `-n 256 -c 4` | 4 | 64 | **10.4 GiB** |
| `-n 128 -c 4` | 2 | 64 | **10.4 GiB** |

And the trade is unusually favourable, because **per-rank memory is almost independent of local
work**: the 256-rank probe fitted `above_floor ~ local^0.025` against a 14.5x spread in local
determinants. Packing more ranks onto a node therefore divides the memory budget while buying very
little, since the dominant per-rank cost does not shrink when each rank's share of the basis does.
Halving `ranks_per_node` is the cheapest single change available to this workload, and it needs no
code at all.

## The trip-wire as first merged would NOT have saved the crashed run

Found immediately after `16257fc` was merged and Arrhenius updated to it. The guard read

```python
if memory_budget_bytes is not None and not capped and peak_rss >= memory_budget_bytes:
```

with `capped = np.isfinite(threshold)`. Production set `truncation_threshold = 119,555,328` — a
number produced by `suggest_truncation_threshold`, not by a human — so `capped` was **True**, so the
memory check **never evaluated**. The basis died at 949,834 determinants, **0.79% of its own cap**:
the cap never bound, but its existence disabled the guard.

The real failure mode was never an uncapped expansion. It is a cap that is finite and absurdly too
large, which is precisely what an estimator that under-predicts by 372-1028x produces.

**Fixed** by giving the trip-wire its own latch (`budget_tripped`) instead of riding on `capped`,
and by tightening with `min(threshold, basis.size)` so it can never *loosen* a caller's cap. The
latch matters independently: `peak_rss` is a high-water mark, so once it exceeds the budget it stays
above it forever and an unlatched guard would re-fire every cycle and walk the threshold down to the
seed determinant.

This is a **deliberate behaviour change**, and the test that pinned the old rule
(`test_guard_never_engages_when_a_real_cap_is_already_set`) was replaced rather than deleted, by
three that keep what was genuinely right about it: the guard never loosens a caller's cap, it does
tighten one memory cannot afford, and it fires exactly once. A cap is a target; a memory budget is a
physical constraint.

## GS_SELECTION_CHUNK is demoted: it tunes 5% of the production cost, not 81%

`GS_SELECTION_CHUNK` was this document's "one measured, already-implemented, free win" — 202 MB
(10.5%) at cap 20,000. **That measurement is serial**, and the 256-rank bracket shows the selection
round is **5.1%** of the per-rank cost at production rank count, against 94.9% for the eigensolver.

So its real value in production is of order **0.5%, not 10.5%**. It remains free and correct, and
there is no reason to turn it off — but it must stop being described as the first thing to do. Every
recommendation in this document that was ordered by the serial ledger inherits the same correction:
the serial cost centre and the production cost centre are different components.

## WIRED: the measured-RSS trip-wire is now connected

`memory_budget_bytes` shipped implemented, tested and **unwired** — which is why an uncapped
production expansion grew to 949,834 determinants and was killed. Now connected at both
`CIPSISolver.expand` call sites (`groundstate.py:488` in `_solve_sector_core`, `:1218` in
`solve_ground_state`) through one shared helper:

- **`groundstate.expand_memory_budget(comm)`** — returns `safety * available_bytes_per_rank(comm)`.
  Shared by both sites deliberately: the budget is a policy, and two copies of a policy expressed
  as arithmetic is how they drift. Documented as collective (it splits a shared-memory
  sub-communicator and min-reduces), and called unconditionally at both sites for that reason.
- **`GS_MEMORY_BUDGET_SAFETY`** (new `config.Knob`, group `groundstate`, registered in `KNOBS`,
  `doc/configuration.md` regenerated). Default `None` = use `memory_estimate.DEFAULT_MEMORY_SAFETY`
  rather than repeating `0.5` — one source of truth for "what fraction of RAM is safe to hold".
  `0` disables the guard and restores the pre-2026-09 behaviour. Negative values clamp to 0 via
  `minimum=0.0`, so a typo cannot become a negative budget that trips on cycle 0.

**Why this is first rather than fourth.** It reacts to *measured* RSS, MAX-allreduced across ranks
(`cipsi_solver.py:1189`), so it trips on the rank that would actually OOM and is indifferent to
every modelling error this document has catalogued — the 4-5x under-prediction on a laptop, the
372-1028x at 256 ranks, and the fact that the dominant term is not proportional to a rank's own
determinant count at all. Nothing else here is correct independently of knowing what the memory is.

**Blast radius is confined to the dangerous configuration.** The guard fires only when the
expansion is *uncapped*; with a `truncation_threshold` set, the fixed-budget machinery governs and
the trip-wire never evaluates, so every capped run is bit-identical. When it does fire it adopts a
fixed-budget cap at the current basis and warns, handing control to the same code path a pre-chosen
threshold would have taken — not a new one.

Tests: `test_expand_memory_budget_wiring.py` (7 passed) pins the derivation, the disable path, the
negative-value clamp, and — source-level, deliberately — that **both** call sites pass the argument,
since what regressed before was the argument and not the behaviour. The guard's own semantics stay
covered by `test_cipsi_memory_budget_guard.py`.

## What to do — reordered by what the rank sweep found

The first four items were ordered by the serial ledger, which measured the wrong axis. Production's
problem is **not** which array inside a cycle is biggest; it is that the cap was sized for the
average rank under an assumed manifold of ~10. Reordered accordingly:

1. **Wire `memory_budget_bytes` into `groundstate.py`'s two `expand` call sites.** Implemented,
   tested, unwired — and **already skew-aware**: `cipsi_solver.py:1189` MAX-allreduces `peak_rss`,
   so the guard trips on the heaviest rank's *measured* RSS. It needs no model, no skew factor and
   no estimate, which is decisive given that the model's error grows with scale. It would have
   adopted a cap mid-expansion and the run would have continued instead of being killed.
2. **Supply `gs_num_wanted`** — now wired, as the `GS_NUM_WANTED` knob read by
   `memory_estimate.resolve_gs_num_wanted` at both `dc_criteria` sizing sites. The parameter had
   existed on `suggest_truncation_threshold` and `log_memory_budget` all along and **no caller ever
   passed it**.

   **Corrected value claim.** An earlier version of this list said supplying it moves the prediction
   at production's own cap from 2.51 to 8.43 GiB, "from fits to refused". That was measured against
   the estimator *before* `selection_bytes` was added; with `selection_bytes` in the model that cap
   is refused on every path (6.86 GiB unset, 12.78 GiB at 222, against 5.0 GiB available). What the
   knob still buys is a smaller *chosen* cap — at 256 ranks and a 2.5 GiB budget,
   `suggest_truncation_threshold` returns 43,570,432 unset, 31,447,552 at 105 and 23,396,096 at 222,
   a **1.4-1.9x reduction**.

   **It does not make the cap safe**, and the test suite asserts that it doesn't: all of those remain
   24-46x above the 949,834 determinants that actually exhausted memory, because the estimate is
   structurally low by 372-1028x at that rank count. This narrows the gap the measured-RSS trip-wire
   has to cover; it does not replace it. Use ~105 for this workload (`2·block_width` ≈ 10 is the
   unset assumption).
3. **Make `estimate_gs_peak_bytes` size the heaviest rank, not the mean.** `local = n_global/ranks`
   under-budgets the binding rank by the skew factor, which *grows with rank count* (1.09 at 8 →
   2.79 at 256). Prefer re-validating after the first cycle against the already-measured
   `local_max` over hard-coding a table: the skew depends on the determinant set, not on the code.
4. **Set `GS_SELECTION_CHUNK`.** Still free and still real — 202 MB (10.5%) at cap 20,000, a plateau
   over 4/8/16, 0.1% wall-clock, `e0` bit-identical, already shipped. Demoted from first place only
   because 10% does not save a run that is over budget by 2-4×.
5. **The other two selection terms, in either order — neither dominates.**
   `_apply_block_and_redistribute` is 416 MB and `overlaps` is 389 MB at cap 20,000. A row-chunked
   destination would bound the first; for the second, naive chunking over reference columns is *not*
   exact, because `e_Dj` derives from the whole block's candidate support. Expect about a fifth of
   the peak from each at best.

### Still open, and what would close it

- **Whether per-rank memory is linear in local determinant count.** The skew is measured and large;
  its memory consequence assumes linearity, and the local lever (1.17× at 4 ranks) is too small and
  too contaminated by rank-0 asymmetry to test it. At 256 ranks the lever is 14.5×. The prepared
  probe (`arrhenius_handover/`) reports the per-rank RSS min/max against the local-count min/max,
  which settles it from one 30-minute job.
- **Why the deaths concentrated on one node.** Not hash asymmetry (refuted above). Most likely both
  nodes sat near the edge and one tipped; the probe's per-node RSS totals would confirm.
- **The n=8 sweep rows**, which this box refused without `--oversubscribe` (6 physical cores).

## Verification

- `python -m pytest`: 2162 passed, 0 failures (uncontended run).
- `mpiexec -n 2` (2391 passed) and `-n 3` (2395 passed) `python -m pytest --with-mpi`: green,
  0 failures.
- New test files, all exactness-checked against a hand-rebuilt reference of the exact
  pre-fix algorithm (`np.testing.assert_array_equal`/`np.array_equal`, not `allclose`):
  `test_cipsi_selection_block.py` (Phase 2a), `test_cipsi_selection_chunking.py` (Phase 2b),
  `test_cipsi_memory_budget_guard.py` (Phase 4).
- `black`/`ruff` clean on every changed file.
- **This pass**: `python -m pytest` 2162 passed; the per-step RSS ledger and the pre/post
  A/B above; `test_manifold_request.py` (57 passed) pinning the eigenstate-request
  arithmetic and checking end-to-end that `expand` reaches the *same kept manifold* under
  the new rule as under the old doubling, on a fixture whose manifold actually grows.
- **`thick_restart_block_lanczos`'s memory scaling is now measured** (115 restarts, three
  caps) and is *not* the dominant cost: 19% of the peak cycle against the selection round's
  81%. The `trlm_restart` trace note records the bound it stays inside.
- **`GS_SELECTION_CHUNK` is measured on both axes**: the memory plateau over 4/8/16 and the
  wall-clock (254.9 s unset vs 254.6 s at 8; selection rounds 31.7 s vs 31.2 s).
- **Not yet done**: any multi-rank measurement at all (item 5 above). Every number in this
  document is single-rank, and the production residual is a multi-rank question.

## ROUND 5 (256 ranks): memory by CATEGORY, and the ratchet is MPI shared memory

Every earlier probe sampled only `VmHWM`, which cannot separate numpy arrays from Cython-owned C++
objects, glibc heap retained after frees, or MPI shared-memory pages. `arrhenius_handover/memtax_probe.py`
brackets every step of one sector solve with rank-local samples of `RssAnon`/`RssShmem`/`RssFile`/
`VmHWM`, the `tracemalloc` peak, `mallinfo2()` (outer frames only -- 14 ms/call on a fragmented heap)
and `malloc_trim(0)`, resets `VmHWM` between caps via `/proc/self/clear_refs`, and A/B-tests a
`mallopt`-tuned allocator in the same job. Log: `from_arrhenius/memtax-2331040.out` (Intel MPI 2021.16,
128 ranks/node).

| cap | HWM above start, binding rank | `RssShmem` growth | numpy-visible | C/C++-owned live | glibc-retained |
|---|---|---|---|---|---|
| 2,000 (first solve) | 144 MiB | 80 | 0 | 14 | 44 |
| 20,000 | 74 | 32 | 16 | 0 | 0 |
| 100,000 | 169 | 97 | 0 | 37 | 0 |
| 300,000 | 489 | **296** | 0 | 32 | 9 |

**`RssShmem` ratchets and is never released.** At the *start* of each successive solve the binding
rank holds 1.8 -> 90 -> 112 -> 125 -> 178 -> 276 MiB of shared memory; anonymous memory over the same
six solves is flat (192 -> 226 MiB). At cap 300,000 it is 85% of the growth, and it grows inside the
all-pairs collectives: the matvec's per-destination `Reduce` loop (110 MiB in `_block_ops.pxi`'s
`block_apply`, 37 in the sweep kernel), TSQR's `Allgather` (12) and the packed `Neighbor_alltoallv`
(74). Round 4's "the Reduce loop costs 5 MiB" was measured with 7 KB messages; production chunks are
megabytes, and message size was the axis that measurement did not vary.

**The selection round is 86% of the binding rank's peak at cap 300,000** (420 of 489 MiB:
`_apply_block_and_redistribute` 228 MiB of its own plus 192 in the exchange), the eigensolver 69.
Round 3/4's "eigensolver 95%" at cap 2,000 was the shared-memory first-touch -- connection setup,
80 MiB -- landing in the first collective-heavy step of a workload too small to show anything else.
The "composition inverts with rank count" reading was really "first-touch lands wherever the first
all-pairs collective runs".

`mallopt` tuning does nothing at 256 ranks (the glibc-retained share is already small there; it was
worth 9% at 4 ranks). `HWM above start ~ global^0.68`, about 0.93 GiB/rank for a single solve at
949,834 determinants -- production runs many solves with `num_wanted` ~222 against ~100 here.

**H's coupling graph is sparse under the routing hash.** Measured offline at cap 20,000 with 256
buckets: 11% of (source, destination) rank pairs carry a nonzero block, ~28 sources per rank (37 max).
A sparse `Neighbor_alltoallv` can therefore replace the 256 dense `Reduce`s per matvec, and 89% of
rank pairs would never open a shared-memory connection in the matvec at all.

### The open question: physical, or accounting?

`RssShmem` counts every shared page a process has touched. Intel MPI's shm transport is a node-wide
pool (forward/backward cells per rank, one extended-cell pool per node), so a page written by a sender
and read by a receiver counts in both ranks' RSS and a pool page touched by many ranks counts many
times. Summed per-rank RSS can overstate physical use by a large factor, and the OOM killer fires on
physical exhaustion. `arrhenius_handover/shm_pattern.py` measures `/proc/meminfo` `Shmem` and
`/dev/shm` usage per node against the summed `RssShmem` around the kernel's `Reduce` loop at
production message sizes, a `Reduce_scatter`, and a 28-neighbour sparse exchange, then around a real
solve, under default settings, Intel's cell-size knobs, and `I_MPI_SHM=off`. The sparse-exchange fix
is not implemented until that log says the shared memory is physical.

## ROUND 6: the shared memory was accounting, and the crash log names the mechanism itself

**Shared memory is not the OOM.** `from_arrhenius/shmpattern-2331617.out` measured, on each node,
`/proc/meminfo` `Shmem` and `/dev/shm` usage next to the sum of the ranks' `RssShmem`: after a real
cap-20,000 solve the ranks summed to **19.0 GiB per node while the node physically held 0.39 GiB**.
The round-5 ratchet is each rank touching pages of one node-wide Intel MPI pool -- a 48x overcount.
`I_MPI_SHM=off` removes it and makes the matvec's `Reduce` loop 5x slower, so it is not a
recommendation. Two side results: a 28-neighbour sparse `Neighbor_alltoallv` is 4x faster than the
256-root `Reduce` loop at production message sizes (1.7 s vs 7.0 s at 10,000 x 110), and
`block_apply` is 62% of the solve's wall-clock at cap 300,000 -- a performance lever, not a memory one.

**A second crash, with the mechanism in its own log.** Job 2327163 (killed 14:12 on 2026-09-12,
before the trip-wire fix `0beab43` was deployed, so the guard never ran) prints the MAX-over-ranks
`VmHWM` on every CIPSI cycle line:

| sector | cycle | basis | p | Hpsi_rows | admitted | VmHWM |
|---|---|---|---|---|---|---|
| N_imp 5 | 12 (saturated) | 949,834 | 104 | 2.43M | 0 | 2.0 GiB |
| N_imp 3 | 2 | 67,344 | 82 | 703k | 613k | 2.0 GiB |
| N_imp 3 | 3 | 680,774 | 100 | 5.0M | 2.94M | 2.4 GiB |
| N_imp 3 | 4 | 3,625,002 | 150 | 21.1M | 5.41M | **5.8 GiB** |
| N_imp 3 | 5 | ~9M | | | | OOM-killed |

The first-cycle mark of 939 MiB is the in-process RSPt plus Python floor. Under a 43.75M cap the
N_imp 3 sector admits everything, the basis grows 5-10x per cycle, and the selection round's memory
follows `Hpsi_rows x p`.

**The memory hog is `CIPSISolver._apply_block_and_redistribute`.** Measured at 4 ranks, cap 20,000
(`dup_probe`): on the growth cycle the owned candidate block is **92 MiB and the step's high-water
mark is 572 MiB -- 6.2x**. The pre-redistribution row duplication is only 1.6-1.8x; the factor is the
number of *simultaneous copies*: the raw apply output (~1.6x the owned rows), its packed send buffer,
the receive buffer and the merged block are all alive at once. Scaled to the crash's cycle 4
(21M rows x 150 columns, skew ~2) that is ~0.4 GiB owned on the binding rank, x6, plus the overlaps
step -- the 4.9 GiB of growth observed. The eigensolver is minor at every size that matters (69 of
489 MiB at cap 300,000 on 256 ranks).

**The trip-wire cannot catch this, even as fixed.** It compares a high-water mark *after* the round
that set it, and one cycle grows the basis 5x, so a 2x safety margin is overrun in a single step
(2.4 GiB under a 2.5 GiB budget, then 5.8). When it fires it still admits the already-selected
candidates (3.6M + 5.4M = 7.2M). And the global `VmHWM` never resets, so once an earlier solve set a
higher mark a round's own peak is invisible: sector 2's cycles 0-2 all read "2.0 GiB". Writing `5` to
`/proc/self/clear_refs` resets `VmHWM` unprivileged (`memtax_probe` relies on it), so a per-round
transient is measurable.

### Fixes this points to

1. **A look-ahead admission cap in `expand`.** *Implemented (2026-09-12).* Measured on the
   SrMnO3 archive at 2 ranks, cap 20,000: with the default budget the bound never binds and `e0`
   is bit-identical to the guard-disabled run (`-16.940022121819`); with a budget of 0.12 x
   available (666 MiB) it binds once, at the 120-determinant seed ("peaked 20.4 MiB above its
   319.7 MiB resident set ... the next round can afford 1,705 of the 4,032 candidates"), adopts a
   cap of 1,825 and hands over to fixed-budget refinement, and `truncation_report["memory_bound"]`
   records it. Two things the first cut got wrong and the workload run caught: the baseline must
   be the RSS the round *started* from (sampled inside the round, the candidate arrays are still
   resident and the transient is counted twice -- 953 MiB "now" against a ~600 MiB start), and
   the bound only counts as binding when it is tighter than an existing cap's own admission
   target (otherwise it warned "tightening a cap of 20,000 to 20,000" on every cycle). Reset the high-water mark before the selection round
   and read the round's transient `T_k = HWM_after - RSS_before`. Bound the next basis
   `b_{k+1} = b_k + n_new` by `T_k x (b_{k+1}/b_k) x (p_next/p_k) <= budget - RSS_now`, re-truncating
   `new_Dj` through a second `_admit_top` (which needs the scores kept from `determine_new_Dj`); when
   no growth is affordable, adopt a fixed-budget cap at the current size. Measured RSS only, no model.
2. **Chunked `_apply_block_and_redistribute`.** *Implemented (2026-09-12) as the
   `GS_APPLY_ROW_CHUNKS` knob; default 4 since the same day, `1` recovers the one-shot path.*
   Measured on the SrMnO3 archive at 4 ranks, cap
   20,000, on the growth cycle (basis 4,152 -> 20,000, owned candidate block 92 MiB):

   | chunks | step peak | step time | `e0` |
   |---|---|---|---|
   | unset | 673 MiB | 1.43 s | -16.940022121819 |
   | 4 | 243 MiB | 2.27 s | -16.940022121819 |
   | 8 | 248 MiB | 2.25 s | -16.940022121819 |

   2.8x less on the step that killed the production job, a plateau from 4 chunks on (the
   accumulating merged block and its `+=` reallocation are the floor), `e0` bit-identical here,
   and about +1 s per growth cycle. The approach: apply H to row chunks of the reference block,
   redistribute each chunk and accumulate with the in-place `+=` (`add_scaled` over the union
   support), so three of the four copies are bounded to chunk size: ~1.4x the owned block instead
   of 6x. Row chunks prune partial sums at the `slater_weight_min` cutoff (1.5e-8 here), so they are
   not bit-identical to today's path at that boundary; column chunks would be, but `concat_cols` on
   block states goes through the `to_states`/`from_states` round trip this function was rewritten
   to avoid. Knob-gated, measured on the SrMnO3 workload before any default changes.

## Round 7: GF unit memory -- a job-wide cap inherited onto a 5-rank color

A second production job (128 ranks x 4 threads, 2 nodes, ~9.4 GiB/rank) finished the Gap
double-counting section and the ground-state solve (`VmHWM=2.4-2.7 GiB` every cycle, cap never
bound: `truncation = None` in `ground_state_statistics.json`), then was OOM-killed inside the
interacting Green's function:

```
[2026-09-13T02:00:00] error: Detected 2 oom_kill events in StepId=2339929.0
srun: error: n367: tasks 24,43,45: Out Of Memory
```

**3 of 128 ranks died -- the signature of a skewed per-rank term, not a global over-allocation.**
The log's own two lines name the mechanism:

```
Mn: truncation_threshold=40,234,112: predicted per-rank peak 4.7 GiB (ground state)
    / 221.8 MiB (Green's function), 9.4 GiB/rank available.
...
25 simultaneous unit bases at truncation_threshold=40,234,112:
    predicted per-rank GF peak 4.1 GiB if a unit fills its cap.
```

`suggest_truncation_threshold` sizes `truncation_threshold` so the *ground state* fits `0.5 x
available_bytes_per_rank` **on all 128 ranks** (reproduced exactly:
`estimate_gs_peak_bytes(40_234_112, 58, 5, ranks=128) = 4.7 GiB`). `basis_split.py` then handed
that same number, verbatim, to each split unit basis -- but each unit basis runs on
`128 // 25 = 5` ranks, not 128. `_CappedBasisProxy` enforces the cap it is given; nothing enforced
one sized for the ranks a unit actually landed on.

**The crash archive (`~/Dokument/arrhenius/SMO/cubic/impmod/`) is no longer available on this
machine.** The closest surviving substitute is `impmod_tests/SMO/cubic/impmod/impurityModel_data.h5`
-- the same Mn cluster and 58 spin-orbitals, but `WORKLOADS["smo"]`'s `tau=0.0025`, not the crash's
`tau=0.025` (see `arrhenius-smo-crash-archive-is-not-the-workloads-key`). A local repro at 2 ranks
against this substitute answers the one question the crash log could not: does a GF unit's basis
actually run toward the cap, or does it plateau below it regardless (in which case a per-unit cap
would be a no-op)?

```
cap=20,000:
  unit 1: n0=1,061 -> retained=20,000  cap_hit=True   n_blocks=317
  unit 2: n0=  698 -> retained=20,000  cap_hit=True   n_blocks=266
  unit 3: n0=1,068 -> retained=20,000  cap_hit=True   n_blocks=259
  unit 4: n0=  263 -> retained= 4,570  cap_hit=False  n_blocks=114
  unit 5: n0=1,339 -> retained=19,999  cap_hit=True   n_blocks=916
```

Both halves of the plan's premise hold, in the same run: most units genuinely grow toward
whatever cap they are given (4 of 5 above), so a cap sized for the wrong rank count really does
starve them; one unit (4) plateaus naturally below the cap on its own H-connectivity closure, so
a correctly-sized cap costs it nothing (it never binds there either way). Caps 2,000 and 5,000
also cap-hit and, as expected at that small a size, made `calc_selfenergy`'s own causality check
raise (`UnphysicalGreensFunctionError`) -- an unrelated, correctly-firing guard, not evidence
against the fix.

**The arithmetic — and the first version of it, which was wrong.** An adversarial review of this
round found the original write-up's budget table unsupportable and the fix it described inert.
Both corrections are below; the superseded table is not reproduced, because nothing in it was
worth keeping.

*What was claimed:* that the GF phase sees `9.4 - 2.6 = 6.8 GiB` available, that
`max_unit_dets_within_budget` therefore derives `30,790,637` against the job's `40,234,112`, and
that this 23% recovers the margin that was lost.

*Why it does not hold.* The run's own log contains

```
Memory budget caps the unit split at 25 simultaneous unit bases (truncation_threshold=40,234,112).
```

That line prints **only** when `max_colors_within_budget` actually bound, i.e. when it returned
from inside its loop having verified `estimate_gf_peak_bytes(40.2M, ranks=128//25=5) <= 0.5 *
available`. On master (pre-skew) that estimate is 4.107 GiB, so `available >= 8.21 GiB` at split
time — not 6.8. The write-up's budget contradicts the log it was derived from.

*The deeper problem: the fix as first written could not bind at all.* Whenever
`max_colors_within_budget` returns `n_colors >= 2` it has already verified the cap fits at that
color's rank count; the split can only reduce the color count, which only raises `ranks`; and
`estimate_gf_peak_bytes` is monotone non-increasing in `ranks` (verified, 1030 points, zero
inversions). So `unit_cap >= cap` identically and `min(cap, unit_cap) == cap`. Measured over a
400-cell grid of rank count x unit count x cap x budget: **385 no-op, 15 binding, and all 15
binding cells had `n_colors == 1`** — which is precisely the case where `basis_split` returns the
caller's own basis object, so the only cases where it did anything were the cases where it
corrupted the caller (see "Two defects the review found", below). In the actual crash geometry
(128 ranks, 40 units, cap 40.2M) it was a no-op at *every* plausible budget from 6.8 to 12 GiB.

The branch's own test `test_max_colors_and_max_unit_dets_compose_without_double_counting`
asserted `unit_cap >= cap` and read it as evidence the composition was sound. It was in fact
proof the feature was inert. That test is now renamed
`test_max_unit_dets_without_residency_is_structurally_a_no_op` and says so.

*What actually makes it bind.* The two inversions used the same budget, so the second could add
nothing. The per-unit cap now takes `resident_bytes` — this rank's measured RSS entering the GF
phase, MAX-reduced over the communicator — and budgets
`safety * (available + resident) - resident`: the process may occupy at most `safety` of its
total per-rank share, it already holds `resident`, so the GF phase may *add* only the difference.
That is strictly tighter than `safety * available` (by `resident * (1 - safety)`) and it is the
constraint the color inversion structurally cannot express, because that one runs against
*remaining* headroom — a snapshot taken before every rank on the node grows its unit basis at
once. In the crash geometry, at `resident = 2.6 GiB`:

| available at split | colors | ranks/color | unit cap before | unit cap now | binds |
|---|---|---|---|---|---|
| 6.8 GiB | 18 | 7 | 42,818,434 | 26,446,682 | no -> **yes** |
| 8.2 GiB | 21 | 6 | 44,394,057 | 30,317,895 | no -> **yes** |
| 9.4 GiB | 25 | 5 | 42,563,530 | 30,790,637 | no -> **yes** |
| 12.0 GiB | 32 | 4 | 43,663,127 | 34,202,781 | no -> **yes** |

(The 9.4 GiB row reproduces the original write-up's 30,790,637 — the same number, now reached
because the budget accounts for residency rather than because `available` was assumed wrong.)

**What this does *not* establish.** The model still predicts a 4.4 GiB peak against ~6.8 GiB of
headroom on the crashing configuration, which would not OOM. So the per-unit cap is a real bound
that was missing, but it is **not** a demonstrated explanation of this crash, and this round
should not be read as having found the cause. The unexplained remainder is still the unmodelled
pack/send/receive transient, the tail of the rank-skew distribution, and whatever the model does
not capture about the excited basis. On this branch the change that actually moves the 128-rank
behaviour is the **skew factor inside `max_colors_within_budget`** (it lowers the color count,
raising ranks per unit), not the per-unit cap.

### Two defects the review found in this round's own code

1. **`run_units_distributed` rewrote the caller's basis.** `basis_split` returns the caller's
   `basis` object itself as `split_basis` whenever the split collapses to one color, so
   `split_basis.truncation_threshold = ...` leaked out of the GF phase. Confirmed at 2 ranks:
   `1,000,000,000 -> 7,636`, persisting after the call. `spectra.simulate_spectra` reuses one
   basis across IPS/PS/XAS/NIXS/RIXS, so each single-color call would have ratcheted the next
   spectrum's cap down (`min`), an order-dependent accuracy loss with no opt-out and a floor of
   one determinant. The cap is now scoped to the GF phase with a `finally` restore, and
   `test_run_units_distributed_does_not_mutate_the_callers_basis_cap_mpi` pins it.
2. **The chunked matvec allocated a full copy of the block per chunk.** `copy()` + `keep_rows(mask)`
   duplicates the whole block and then shrinks only its logical length — `keep_rows`' `resize`
   does not release `std::vector` capacity — so each "bounded" chunk ran alongside a full-size
   copy, and the mask cost one Python key object per row per step. That is the mechanism behind
   the measured "chunking is slower with a higher peak". Replaced by a new C++/Cython
   `row_slice(lo, hi)` primitive that allocates exactly the chunk's rows. The identical pattern
   was in the already-shipped `cipsi_solver._apply_block_and_redistribute`, running by default
   since 2026-09-12, and is fixed with it.

### What shipped

1. **Per-unit cap, sized for the unit's own rank count.** `memory_estimate.max_unit_dets_within_budget`
   inverts `estimate_gf_peak_bytes` at a fixed `ranks` (the complement of the existing
   `max_colors_within_budget`, which inverts it at a fixed cap). `gf_units.run_units_distributed`
   calls it right after the split, at `ranks_per_color = comm.size // n_colors`, and tightens
   (never loosens) `split_basis.truncation_threshold` to `min(inherited_cap, unit_cap)`.
2. **A measured rank-skew factor in `estimate_gf_peak_bytes`.** `routing_hash`'s max/mean skew was
   already measured at 2, 4, 8, 16, 64, 128, 256 ranks (see "`routing_hash` is deliberately
   non-uniform" above); `_routing_skew_factor` log-log interpolates between those anchors (clamped
   outside `[2, 256]`, never extrapolated) and multiplies `local_rows`. One factor, shared by both
   inversions, so they compose (a tighter color count implies more ranks per color, hence a larger
   affordable per-unit cap) rather than double-count. `estimate_gs_peak_bytes` is deliberately left
   alone -- the GS solve spans the whole communicator, its measured skew there is smaller (1.62x on
   this archive's own final cycle) and already inside `DEFAULT_MEMORY_SAFETY`, and moving it would
   re-calibrate every DC-search iteration, a much larger change than this round's scope.
3. **Per-unit memory reporting.** The GS phase prints `VmHWM` every CIPSI cycle; the GF phase
   printed only the two split-time predictions above and then nothing until it finished or was
   killed. `_block_green_group` now resets `peak_rss_bytes()` on entry and, on return, MAX-allreduces
   it over the unit's own color and reports it (`_trace_note` + an optional print) alongside the
   retained basis size and Lanczos block count -- the number this round needed arithmetic to
   reconstruct.
4. **`GF_APPLY_ROW_CHUNKS` (default 4, on since 2026-09-14).** Mirrors `GS_APPLY_ROW_CHUNKS` at the
   GF unit's own matvec (`_lanczos_step.pxi`'s `wp = h_op.apply_block(q_curr, ...)`): row-chunked
   apply + redistribute + accumulate bounds the pack/send/receive transient
   `estimate_gf_peak_bytes`'s docstring already documents as unmodelled. Shipped off by default;
   turned on (at `GS_APPLY_ROW_CHUNKS`'s own measured plateau of 4, not a value separately measured
   for the GF matvec) on explicit instruction. The numerical caveat did not go away: unlike the
   CIPSI selection round's chunked output (feeds a `slater_weight_min` prune and a candidate
   ranking), this sum feeds the Lanczos recurrence directly, so a summation-order change can in
   principle move a deflation or iteration-count decision. Verified against the same strong oracle
   `test_gf_truncation.py` holds the one-shot path to (a capped recurrence's continued fraction must
   equal the dense resolvent of `H` projected on whatever it actually retained) across
   cap/reort/chunk-count combinations, plus tight numerical agreement with the one-shot path above
   the reachable space (no admission boundary to perturb there) -- but not against a step-peak/
   wall-time plateau sweep at production GF scale the way `GS_APPLY_ROW_CHUNKS`'s 4 was; see
   "GF_APPLY_ROW_CHUNKS default flip" below for the local sweep run after turning this on.
   `GF_APPLY_ROW_CHUNKS=1` recovers the one-shot path bit-for-bit with a pre-2026-09 run if a
   workload needs it.
5. **Two-strike look-ahead, memory-cap==0 excepted.** `CIPSISolver.expand`'s look-ahead guard used
   to adopt a permanent fixed budget the first time one round's measured transient predicted the
   next round would not fit -- a single noisy reading (allocator jitter, a GC pause inside the
   measured window) then pinned the basis at that cycle's size for the rest of the run. Now it
   waits for two consecutive binding rounds, *except* when the affordable growth is exactly zero:
   a round that admits nothing leaves the basis unchanged, so the loop's own termination
   (`cap_cycles == 0 and self.basis.size == old_size: break`) would exit before a second round
   could ever confirm the reading -- and zero is a floor, not a noisy estimate a second sample
   could revise upward. `determine_new_Dj` already applies the round's own `memory_cap` to its
   admission regardless of the streak, so the guard's safety property is unchanged either way;
   only *when* it locks into permanent fixed-budget mode moved.

Not shipped: a production-scale repro against the actual crash archive (gone) or a cluster job.
The substitute-archive repro above and the corrected arithmetic are the evidence; the next
production job is the first real test at scale.

### `GF_APPLY_ROW_CHUNKS` default flip (2026-09-14) and the 2-rank sweep against it

Turned on by default (4, mirroring `GS_APPLY_ROW_CHUNKS`'s own measured plateau) on explicit
instruction, ahead of a GF-specific measurement. A same-day 2-rank sweep on the substitute
archive (one ground-state solve at `cap=5,000`, `GF_APPLY_ROW_CHUNKS` varied 1/2/4/8 on the
*same* basis so only the GF phase's own cost is compared) came back the opposite of a win at
this rank count:

| `n_chunks` | GF-phase wall | GF-phase MAX `VmHWM` |
|---|---|---|
| 1 (one-shot) | 46.9 s | 403.9 MiB |
| 2 | 71.2 s | 407.5 MiB |
| 4 | 74.8 s | 407.5 MiB |
| 8 | 143.4 s | 407.5 MiB |

Chunking was **both slower and no smaller** here: 1.6-3.1x the wall time and a slightly *higher*
peak, not lower.

**The adversarial review then found why, and it was a bug, not a law of nature.** Each chunk was
built as `q_curr.copy()` + `keep_rows(mask)` — a full-size duplicate of the block whose logical
length is then shrunk, without releasing `std::vector` capacity — so every "bounded" chunk apply
ran alongside a full copy of the very block it was chunking, and the mask cost one Python
`SlaterDeterminant` per row of `q_curr` on every step. Replaced by a `row_slice(lo, hi)` C++
primitive that allocates exactly the chunk's rows. Re-measured, same harness:

| `n_chunks` | wall before | wall after | GF-phase MAX `VmHWM` after |
|---|---|---|---|
| 1 (one-shot) | 46.9 s | 47.0 s | 423.7 MiB |
| 2 | 71.2 s | 54.2 s | 423.7 MiB |
| 4 | 74.8 s | 58.4 s | 423.7 MiB |
| 8 | 143.4 s | 64.2 s | 423.7 MiB |

The overhead collapses (2.2x faster at 8 chunks, 1.28x at 4) but does not vanish: at 2 ranks
chunking is still ~1.24x slower than one-shot and the peak does not move at all — every chunk
count reports the identical 423.7 MiB, and the one-shot figure itself drifts 403.9 -> 423.7 MiB
between runs, so ~5% is run-to-run noise and the peak is simply insensitive to the knob here.
That insensitivity is the mechanism the knob doc always predicted at the low end: a 2-rank
`redistribute_block` has almost no pack/send/receive transient to bound in the first place, so
there is nothing for chunking to save. `GS_APPLY_ROW_CHUNKS`'s own plateau was measured at 4
*ranks*, not 2, for the same reason.

Whether 4 is a net win at the rank counts a real GF unit color runs at (5-10 in the crash's own
geometry) is **still open** — the default stands on the explicit instruction that shipped it, not
on this measurement, which shows only a cost at 2 ranks. `GF_APPLY_ROW_CHUNKS=1` recovers the
one-shot path.

Same-run side observation, single sample so treat it as indicative: the ground-state phase's peak
went 752.1 -> 694.8 MiB across the copy fix. The GS path has run chunked by default
(`GS_APPLY_ROW_CHUNKS=4`) since 2026-09-12 and carried the identical duplicate-per-chunk, so the
fix applies there too — which means the measured "6x -> 1.4x" that justified the GS default was
itself understating what the mechanism can do.

**The whole-suite timing corroborates that this was one bug, not a property of chunking.** The
serial test gate ran 114 s before the GF default flip, 519 s after it (4.5x, which is what made
the cost look structural — hundreds of small tests using a finite cap, each paying a full block
duplicate per chunk), and **105 s** once `row_slice` replaced the copy. Chunking on by default now
costs the serial suite nothing measurable. The `-n2` / `-n3` legs were never slower (350 s / 546 s,
in line with their pre-flip 370 s / 551 s) — consistent with the duplicate being proportional to
block size, which those legs spread over more ranks.

## Round 8: a second SrMnO3 GF OOM, and the matvec fanout was the missing term

A second cubic-SrMnO3 self-energy run (slurm 2399956, 14 Sep 14:59→19:30, 128 ranks / 2 nodes) was
OOM-killed in the same place as the round-7 crash: inside the interacting Green's function, ~2 min
after the ground state finished (`n347: tasks 16,25,32-33: Out Of Memory`).

**First, the thing that mattered most: this run predated round 7's fix.** Its split line is the
pre-fix message (`25 simultaneous unit bases ...: predicted per-rank GF peak 4.1 GiB if a unit
fills its cap`), and `git log -S "if a unit fills its cap"` shows that string was *deleted* by
`c4c3cf7`. The cluster install was stale. So this was a second, independent sample of the
*original* bug, not evidence the round-7 fix failed — valuable precisely because it closes the
question round 7 left open (why ranks died when the model said they should not).

### The geometry, pinned

| quantity | value | source |
|---|---|---|
| job-wide cap | 40,340,864 | log |
| GS final basis | 329,632 dets (cap never bound) | log, cycle 22 |
| resident at GF entry | ~2.2 GiB | log, `VmHWM` last GS cycle |
| available | 9.5 GiB/rank | log |
| units / colors | 40 / 25 | log |
| ranks per color | 6,6,6,6,6, 5×8, 4,4, 5×10 | `np.diff(unit_roots + [128])` |
| block width | 1 | `impurityModel_data.h5`'s `vs_star`: two inequivalent 1×1 blocks × 2 sides × 10 eigenstates = 40 units |

### What killed it: the stale install, and nothing more exotic

**This section originally claimed the model predicted these colors would survive. That claim was
wrong, and an adversarial review caught it.** It is recorded here rather than deleted, because the
error is instructive and is the second time this project has made it.

The original arithmetic asked whether `5.54/4.45/3.73 GiB` (the modelled peak at 4/5/6 ranks) plus
`2.2 GiB` resident fits in `9.5 GiB` available — leaving `1.76-3.57 GiB` of apparent headroom, and
so "the model says every color survives." But `9.5 GiB` is not the budget the code enforces.
`_resident_adjusted_budget(safety=0.5, available=9.5 GiB, resident=2.2 GiB)` is **3.65 GiB**. The
table had silently used `safety = 1.0`. Against the real budget, with the fanout term set to zero
— i.e. exactly round 7's shipped model:

| ranks in color | modelled peak | exceeds the 3.65 GiB budget? | `max_unit_dets_within_budget` | below the 40,340,864 cap? |
|---|---|---|---|---|
| 4 | 5.54 GiB | **yes** | 26,561,734 | **yes** |
| 5 | 4.45 GiB | **yes** | 33,054,656 | **yes** |
| 6 | 3.73 GiB | **yes** | 39,521,539 | **yes** |

Round 7's mechanism already refuses all three colors and already tightens all three caps, with no
fanout term at all. **The stale install is the whole explanation for this crash** — as the section
above established independently, and as is sufficient on its own. There is no residual unexplained
kill for a new term to account for.

What survives from this round is narrower and worth stating precisely: `estimate_gf_peak_bytes` was
genuinely missing a term. `block_lanczos_step_cy` calls `h_op.apply_block(q_curr)`, and
`ManyBodyOperator::apply` returns a **fully materialized** `ManyBodyBlockState` by value
(`ManyBodyOperator.h:114`, no streaming) — not hash-partitioned, everything the rank's rows reach
under H, before `redistribute_block` routes and the cap prunes. That allocation is real, it is
unmodelled, and the function's own docstring conceded as much. Closing a known model gap is a
legitimate reason to ship the term. Claiming it explains a crash that round 7's own mechanism
already refuses is not, and the first draft of this round did exactly that.

**A defect found along the way, but NOT this crash's cause:** `run_units_distributed` and
`max_colors_within_budget` both sized a color's per-rank cap on `comm.size // n_colors` — the
*mean* — while `_pack_units` apportions ranks to colors proportionally to bin mass with a floor of
1, so colors genuinely differ (two of this crash's own colors sat on 4 ranks while the mean said
5). The dead ranks (16, 25, 32, 33) sit in colors with 6, 6 and 5 ranks — the *larger*, not
smaller, colors — so this defect predicts the opposite of what killed this run. Real, and fixed
(below), but a second finding, not the explanation. Do not let a found defect become a claimed
explanation twice in one project.

### Measuring the fanout: pairs vs. rows, and why `_SELECTION_FANOUT_DEFAULT` was the wrong number

The first draft of this round reached for `_SELECTION_FANOUT_DEFAULT = 40` (CIPSI's own measured
raw connectivity) to size the new term. **That was caught before it shipped.** `40` counts
source-determinant/candidate *pairs* from a list of per-state output states
(`build_local_operator_list`); the GF matvec's `wp = h_op.apply_block(q_curr)` returns one *row*
per determinant it reaches, keyed — multiple source rows in `q_curr` landing on the same target
determinant collapse into a single row there. Pricing the new term at the pair count would have
been off by roughly the dedup factor: at width 1, `local_rows * 40 * row_bytes` is ~8.7x the basis
term alone, which would have cut `max_unit_dets_within_budget`'s returned cap by roughly the same
factor — the wrong direction for a workload already described as truncation-limited
(`smo-dc-is-truncation-limited`).

So the row fanout was measured directly instead, cheaply, in plain Python — no cluster, no
Cython rebuild: `h_op.apply_block(q, 0)` is callable on any width-1 block built directly from a
basis's own determinants. Built the real solver Hamiltonian from the **crash archive itself**
(`/home/johan/Dokument/arrhenius/SMO/cubic/impmod/impurityModel_data.h5`, `tau=0.025` — **not**
`restriction_diagnostics.WORKLOADS["smo"]`, a different SrMnO3 archive at `tau=0.0025`, see
`arrhenius-smo-crash-archive-is-not-the-workloads-key`), ran the real ground-state CIPSI solve
(`prepare_solver_basis` + `calc_gs`, no MPI) at three basis sizes, and on each converged basis
measured `len(h_op.apply_block(q, 0)) / len(q)` over four disjoint width-1 chunks of the real
determinant set:

| cap requested | GS basis built | build time | fanout per chunk | mean |
|---|---|---|---|---|
| 5,000 | 5,000 | 83.2 s | 20.68 / 22.12 / 20.74 / 15.32 | **19.72** |
| 20,000 | 20,000 | 377.8 s | 18.60 / 19.88 / 17.67 / 12.45 | **17.15** |
| 100,000 | 100,000 | 1206.1 s | 14.77 / 16.43 / 14.35 / 9.34 | **13.72** |

**Those numbers are the fanout of a QUARTER-block, and shipping them as a whole-step constant was
the second error of this round.** The probe chunked the basis into four before applying, so each
measurement is `len(apply(N/4 dets)) / (N/4)`. Fanout falls as the block grows — a bigger block's
reachable set overlaps itself more — so a quarter-block figure over-counts a whole step by exactly
the dedup the chunking destroyed. `_SELECTION_FANOUT_DEFAULT` was rejected for conflating pairs
with rows; this conflated *block sizes*, which is the same class of mistake one level down.
Measure the block you are modelling.

### The corrected measurement: whole-step fanout, and what chunking actually saves

Re-measured on the same three bases, applying H to a width-1 block spanning the **whole** basis,
and separately to contiguous `row_slice`-shaped chunks of it (the same shape
`_lanczos_step.pxi` builds, since `ManyBodyState` keys are sorted and `row_slice` takes a
contiguous span of them). The peak a chunked step pays is the **largest single chunk's** raw
output, because each `_raw` is freed before the next chunk runs (`del _raw`), so the saving is
`whole / max_over_chunks`:

| basis | whole-step rows | fanout | `whole/max` @2 | @4 | @8 |
|---|---|---|---|---|---|
| 5,000 | 66,525 | **13.30** | 1.48 | 2.41 | 3.99 |
| 20,000 | 213,650 | **10.68** | 1.41 | 2.15 | 3.83 |
| 100,000 | 805,871 | **8.06** | 1.35 | 1.96 | 3.49 |

Two results, both of which contradict a position this round had already shipped:

1. **Whole-step fanout is 8.06 at the largest basis, not 20.** Monotone decreasing, log-log slope
   −0.167. Extrapolated to a 10⁷-determinant block it would be ~3.7, but this module does not
   extrapolate past its measured range (`_routing_skew_factor` clamps for exactly this reason), so
   `_GF_MATVEC_ROW_FANOUT_DEFAULT = 8.1` clamps at the largest anchor — which over-predicts
   relative to the trend, the safe direction.
2. **Chunking does bound this transient — by ~1.9x at the default of 4, not 1x and not 4x.**
   Both previously-held positions were wrong and neither was measured. The mechanism is plain in
   the code (`row_slice` → `apply_block` → `del _raw` per chunk), so "no credit" was never
   defensible on mechanism; and chunks reach heavily overlapping sets, so the full chunk count was
   never available either. `_GF_CHUNK_DIVISOR_ANCHORS` carries the measured values rounded down
   (the divisors themselves decline with basis size — 2.41 → 2.15 → 1.96 at 4 chunks — so
   production is likely lower still, and under-crediting over-predicts, which is safe).

The earlier justification for taking no credit cited the 2-rank sweep in
"`GF_APPLY_ROW_CHUNKS` default flip" above. That sweep measured *total process VmHWM* at
`cap=5,000` on 2 ranks, where this term is a rounding error against the ~213 MiB Python floor — it
could not have resolved a change in it either way. Absence of a signal in an instrument that
cannot see the quantity is not evidence of absence.

Net effect of the two corrections, on the crash's own geometry: the effective coefficient falls
20 → **4.26** per determinant (8.1 ÷ 1.9), per-determinant GF cost 1988 → **855 B** (548 B with no
term at all), and the affordable color count recovers **5 → 16** (25 with no term). The first
shipped version of this round would have cost roughly two thirds of the GF phase's concurrency to
a term it had over-priced ~4.7x.

### What shipped

1. **`_GF_MATVEC_ROW_FANOUT_DEFAULT` = 8.1, divided by `_GF_CHUNK_DIVISOR_ANCHORS`**
   (`memory_estimate.py`): the measured whole-step row fanout, added to `estimate_gf_peak_bytes`
   as `ceil(local_rows * 8.1 / divisor(GF_APPLY_ROW_CHUNKS)) * row_bytes`, inherited identically
   by both `max_colors_within_budget` and `max_unit_dets_within_budget` (same composition property
   the skew factor already has). The model reads the chunking knob, so `GF_APPLY_ROW_CHUNKS=1`
   correctly prices the larger one-shot transient. Pinned by
   `test_estimate_gf_peak_bytes_scales_local_rows_by_the_skew` (exact byte formula) and
   `test_gf_chunk_divisor_credits_chunking_but_never_the_full_chunk_count`.
2. **Per-color rank sizing** (`gf_units.py`): `run_units_distributed` sizes each color's cap on
   `split_basis.comm.size` — the color's real rank count — instead of the job-wide mean.
   `max_colors_within_budget`'s own mean-based `ranks_per_color` is documented, not fixed: it runs
   *before* `_pack_units` (deciding one of that function's own inputs) and sits below
   `basis_split` in the layering, so it cannot see the true spread; the real per-color bound lives
   in `gf_units.py`, which never loosens what the coarser color count allows.
3. **Split-time diagnostics** (`gf_units.py`): the split print now records block width, resident
   set, job-wide available bytes, and the rank-count spread across colors (derived from
   `unit_roots`, no extra collective) before any unit runs — reconstructing this crash needed
   inverting `max_colors_within_budget`'s return to recover a number the process had in hand the
   whole time.
4. **`build_vector`'s full-size gather on the seed-QR path** (`gf_primitives.py`):
   `_distributed_seed_qr` allocated the full `(n, basis.size)` array on every rank via
   `build_vector(..., root=0)` before reducing into rank 0's copy. Switched to
   `build_distributed_vector` (local-shaped) plus a new `_gather_qr_rows` `Gatherv` — the mirror
   of the existing `_scatter_qr_columns`. Only rank 0 now holds a global-shaped array, which is
   unavoidable since the QR itself runs there. Not this crash's cause (~16 MB at width 1 on this
   workload) but a standing model gap (`build-vector-is-a-full-dense-state-gather`) that was never
   fixed. `build_vector` itself is untouched — still used, and tested, elsewhere.
5. **One resident-adjusted budget policy** (`memory_estimate.py`): `max_colors_within_budget` and
   `max_unit_dets_within_budget` used two different budget expressions (`safety * available` vs.
   `safety * (available + resident) - resident`). Factored into `_resident_adjusted_budget`;
   `max_colors_within_budget` gained the same optional `resident_bytes` parameter (default `None`,
   so no call site's behavior changed).

**Before the next cluster launch: verify the installed package actually contains the current
work.** A stale install is what cost this run — the same install-verification step round 7 closed
with should be standard practice before every production launch.

---

## Round 9: the guard budgeted against free memory, and pinned every sector at its seed

A cubic-SrMnO3 DFT+DMFT run with the gap DC (slurm 2483188, 15 Sep 14:33→14:39, 128 ranks all on
one node `n325`, `-n 128 -c 2`, `GS_MAX_BLOCK_WIDTH=5`) died six minutes in. It is the first crash
in this series that is **not** an OOM.

### It was not an OOM

All 128 ranks print `MPI_Abort(…, -1)` (`slurm-2483188.out:11-138`) *before* the `SIGNAL Killed`
line (`:139`); exit 137 is srun's teardown after the abort. There are no OOM strings anywhere. The
terminating event is `UnphysicalGreensFunctionError` from `selfenergy.py:141` — 469 of 2001
real-frequency Σ points (23.4%) with positive `Im Σ`.

Two traps in reading that log, both of which cost time:

* `impurityModel-Mn-<N>.out` are **per-rank** files (0-127), not per-iteration. There was one
  iteration: `slurm:4,10` says fresh start, no `sig` file.
* The `p=108` in the CIPSI cycle lines is `len(psi_refs)` (`cipsi_solver.py:1441`) — the
  *manifold*, **not** the Lanczos block width. `GS_MAX_BLOCK_WIDTH=5` had already sliced the warm
  block to 5 and appended the cold column, so the run executed at width **6**: every subspace
  dimension in the log (24, 42, 66, 84, 126, 132, 150, 162, 222, 462) is divisible by 6 and none
  by 108. This repeats the warning in `smo-thermal-manifold-is-a-seed-basis-artifact`.

### What the log shows

| line | event |
|---|---|
| 286–347 | sector `N_imp=5` expands 252 → 4,860 → 53,894 → 322,837 dets, all at **VmHWM 1.2 GiB** |
| 383–392 | → 839,148 dets; VmHWM 1.4 → 2.3 GiB |
| 403 | guard fires at 934,289 dets: *"peaked 381.1 MiB above its 2.3 GiB resident set against a 2.5 GiB budget"*. PT2 mass 1.4e-3 — a good solve |
| 507→742 | **every later sector pinned at its seed basis** — 10, 45, 120, 210, 252 dets — each reporting a **2.5 GiB resident set** against a round transient of **4–68 KiB** |

A 120-determinant basis reporting 2.5 GiB resident, in a process that held a 322,837-determinant
basis at 1.2 GiB. The main solver in the same process inherited it
(`impurityModel-Mn.out:247-303`).

### Root cause: two quantities that are not the same quantity

`groundstate.expand_memory_budget` returned `safety * available_bytes_per_rank`, and `available`
is `MemAvailable / ranks_on_node` — memory that is **free**, already net of what the process
holds. That is an *increment* allowance. Both guards in `CIPSISolver.expand` compare it against
**absolute** process RSS.

At 2.5 GiB resident against a `0.5 * 4.9 = 2.45 GiB` budget, headroom was negative before any work
was done. `_memory_growth_bound.affordable` returns `0` from the `headroom <= 0` branch **without
ever consulting the round's own measured transient**, and `cipsi_solver.py:1384` locks that in on
the first strike. Hence a 4–68 KiB round refused all growth, permanently.

### Why the resident set was 2.5 GiB — and why `malloc_trim` is the wrong instinct

Not glibc-retained heap. Round 5's `memtax` probe already measured this cluster's geometry (256
ranks, 128/node): glibc-retained 44/0/0/9 MiB, while `RssShmem` ratchets 1.8 → 276 MiB with
`shm ~ global^0.81`, extrapolating to ~700 MiB at 934k determinants. `current_rss_bytes()` reads
`VmRSS`, **which includes `RssShmem`** — MPI shared pages touched by 128 co-resident ranks inflate
each rank's RSS without moving `MemAvailable`. `malloc_trim` was measured at **0%** at 256 ranks.
The remainder is the in-process RSPt Fortran side.

### The chain to the crash, and why the bath fit is *not* the culprit

An adversarial review argued the causality violation is "carried by `G₀`" because
`Im(G₀⁻¹) = 2.968` against `Im(G⁻¹) = 1.98e-2`, and pointed at the `-0.6699` bath-fit
discretization error at `:230`. That reading is wrong. **`G₀` and `G` are both functionals of the
same fitted bath, so a fit error is common-mode and cancels in `Σ = G₀⁻¹ − G⁻¹`.** Causality of
`Σ` is guaranteed by `G` being the *exact* Green's function of that Hamiltonian, not by the bath
being accurate — a poor bath gives a wrong-but-causal `Σ`. What breaks it is `G` and `G₀` computed
to *different fidelity*, which is exactly what a truncated determinant basis does.

Every step is in the log:

| step | evidence |
|---|---|
| the guard freezes the GS at 120 dets | `impurityModel-Mn.out:301-304` |
| that pins the job-wide cap | `:1079` — *"unit basis capped at 120 determinants (**job-wide truncation_threshold=120**)"* |
| the GF support freezes | `:1241`, `:1405` — 113 / 118 / 120 determinants |
| `G` itself stays causal | `:1422`, `:1435` — as a truncated Lehmann sum must: positive weights, too few poles |
| `G` has no weight where the bath does | `Im(G⁻¹) = 1.98e-2`, i.e. nearly real — no pole nearby |
| and that is exactly where the bath lives | violating window **[−0.551, −0.11]** contains **every** bath level: block 0 at −0.309, −0.286, −0.272, −0.257, −0.169, −0.145 and block 2 at −0.297, −0.259, −0.227, −0.183 (`impurityModel-Mn-dc.out:48`, `:61`) |

### What shipped

1. **`memory_estimate.absolute_rss_budget(safety, available, resident)`** = `safety * (available +
   resident)`, the rank's whole share, which is what an absolute RSS reading may be compared
   against. `_resident_adjusted_budget` — round 8's headroom form — is redefined in terms of it so
   the policy keeps one definition. **Note for anyone re-deriving this:** round 8's helper is
   *already* the right formula; it returns the head*room* (`absolute − resident`), and comparing
   that against the budget number instead of against RSS is the same units confusion as the bug.
2. **`memory_estimate.resident_bytes_per_rank(comm)`**, a MAX-reduce of `current_rss_bytes()`.
   `available` stays MIN-reduced. Worst case on both, deliberately — which means the sum is *not*
   any single rank's quantity and must not be reasoned about as one.
3. **`expand_memory_budget` re-samples both per expansion.** The budget stays a plain `int`, so
   all five consumers in `cipsi_solver` are unchanged.
4. **`_memory_growth_bound` reports *why* it capped**: `(cap, reason)` with reason in
   `{"unmeasured", "budget", "transient"}`, and the warning names the side and the margin. The
   crashed run would now print *"the resident set alone is over the 100.0 MiB budget by 400.0 MiB,
   so the round's own cost did not enter into it"* — absurd on sight, which is the point.
5. **`GS_MEMORY_BUDGET_INCLUDE_RESIDENT=0`** rolls the arithmetic back without disabling the guard.
6. **A memory-bound DC no longer reports `converged`**: `truncation_report["memory_bound"]`
   propagates onto `_SectorSolution` and `ctx.memory_bound_at`, and `_reject_if_memory_bound`
   raises `DoubleCountingUnreachable` at all three criteria unless `DC_ALLOW_MEMORY_BOUND=1`.

### Two things deliberately *not* done, and why

* **A two-round streak for budget-side zeros.** Unreachable: a `"budget"` zero means
  `rss_base >= budget`, the after-the-fact trip-wire fires on `peak_rss >= budget`, and
  `peak_rss >= rss_base` by construction (VmHWM is reset to the round's starting RSS), so the
  backstop has always capped first. Also unnecessary — `available` is `MemAvailable /
  ranks_on_node`, so a neighbouring rank allocating a whole GiB moves the budget by ~8 MiB at 128
  ranks/node. The argument depends on both reductions staying as they are.
* **Gating the DC on `dc_cap_drift / tol`** (22.6x on this run). Drift is the *cap-calibration
  ladder's* spread, and it is silent in exactly this failure: a run pinned at its seed returns the
  same frozen answer on every rung, so the rungs agree and drift → 0. It is also unpopulated when
  `DC_CAP_STRATEGY` resolves the cap without a ladder. Raw `discarded_de2_mass` is likewise wrong
  — unnormalized `sum(scores[discarded])` (`cipsi_solver.py:889`), so it grows with how *many*
  candidates were dropped, and a healthy large-cap run would trip it.

### The gate was half-wired, and only an end-to-end test could see it

Worth recording as a method point. The DC gate reads `truncation_report["memory_bound"]`, and six
unit tests over `_reject_if_memory_bound` plus a source-level count of its three call sites all
passed -- but the **after-the-fact trip-wire never set that flag**. Only the look-ahead half did.
So a run stopped by the OOM backstop produced a report saying `cap_hit` with `memory_bound` False,
indistinguishable from a `truncation_threshold` the caller chose, and the gate let it through.

It surfaced only on driving a real `fixed_gap_dc` at `GS_MEMORY_BUDGET_SAFETY=1e-9`: the basis was
held at 1-2 determinants, every "GS basis cap hit" warning fired, and `dc_memory_bound` still came
back `no`. Fixed by setting `memory_bound_observed` (a pure reporting flag) in the trip-wire, never
`memory_bound`, which also gates whether the look-ahead may adopt a tighter cap later.

The same run is now two tests: one asserting a healthy search records `dc_memory_bound = no` (which
proves the call site is *reached*, where the source-level count only proves it exists), and an MPI
one asserting the rejection is rank-invariant at `-n 2` and `-n 3` -- a split verdict would turn a
memory problem into a hang inside a collective.

### Part 1b, CLOSED: the block width was never the constraint

A width-`w` block Lanczos spans at most `w` vectors per eigenvalue, so a manifold of multiplicity
`m > w` comes back with `w` copies and the rest silently missing. The crashed run executed at
`w = 6`. If the bottom multiplicity were >= 6, every number it produced would rest on an
under-resolved manifold -- a correctness question, not a memory one.

**Measured, and the answer is no.** The valid experiment builds **one** basis and diagonalizes it
at several widths from an identical warm block, so width is the only thing that varies. On the
crash archive at cap 20,000 (20,000 determinants, 44 reference states):

| block width | `e0` | max abs dE vs p=6, all 44 states |
|---|---|---|
| 6  | -13.291093624866 | — |
| 16 | -13.291093624866 | 6.6e-14 |
| 32 | -13.291093624866 | 4.3e-14 |

`e0` bit-identical, the whole retained spectrum agreeing to machine precision, the same 44 states
and the same group structure at every width. Width 6 is sufficient for this workload.

**Two ways to get this wrong, both of which happened here:**

1. **Reading the multiplicity off a single dump.** It is basis-dependent -- 1 at cap 2,000, 1 at
   cap 20,000, three doublets (split 4e-8) in the production 934k dump. The first attempt at this
   question read the **252-determinant seed** dump, saw "-17.026 x6", and concluded the ceiling
   might be binding. A seed basis says nothing about the converged manifold
   (`smo-thermal-manifold-is-a-seed-basis-artifact`, again).
2. **Sweeping the width across separate CIPSI runs.** The basis CIPSI grows depends on the
   reference states, hence on the width, so the runs end on different bases. At cap 50,000 that
   gave 47,356 determinants at p=6 against 50,000 at p=16, and a 1.1e-4 difference in `e0` that
   is entirely the basis, not the width. (The cap-20,000 sweep agreed only because both runs hit
   the cap exactly and so shared a basis -- luck, not design.) Note in passing that p=6 reached
   the *lower* variational energy on the *smaller* basis.

Probe: `width_controlled.py` (scratch). It needs `psi_refs` passed in -- a cold start builds a
width-1 block whatever the knob says, so a naive "call `get_eigenvectors` at two widths" measures
nothing.

### Step 0 result: the peak is `_candidate_overlaps_and_energies`, and Tier 1 buys nothing

Per-site ledger (`from_arrhenius/site_ledger.py`, round 6's method: reset `VmHWM` immediately
before a site, read it immediately after, difference from the site's entry RSS). SrMnO3 archive,
cap 20,000, **1 rank**, ten selection rounds, maxima over rounds:

| site | max ABS peak | max growth | max d(anon) |
|---|---|---|---|
| `_candidate_overlaps_and_energies` | **1378.9 MiB** | 708.5 MiB | 269.2 MiB |
| `_score_candidates` | 1335.6 MiB | 397.4 MiB | **16.7 MiB** |
| `_apply_block_and_redistribute` | 980.8 MiB | 526.6 MiB | 272.7 MiB |

The **growth** column is each site's peak measured from its own entry RSS, in its own worst round.
Those are not disjoint windows over a common baseline, so they must not be summed: 708 + 527 + 397
= 1632 MiB against a 1376 MiB process peak is a meaningless total, not evidence of double-counting.
What identifies the peak-setter is the **absolute** column, and `_candidate_overlaps_and_energies`
reaches 1378.9 MiB against the 1376.0 MiB measured in an *uninstrumented* run -- so it is the site
that sets the process high-water mark, and `_apply_block_and_redistribute`, despite the second
largest growth, never comes near it.

`e0` is unchanged at -13.291093624866 with the instrumentation in place.

**`GS_SELECTION_CHUNK` is not the cheap win the ranking claimed.** The knob does exactly what it
documents -- at production shape (p=44, 137k candidates) the score stack goes 332 -> 111 -> 60 MiB
for chunk off/8/1, checksums identical, so the exactness claim holds too. But in a full solve
`e0` was bit-identical and the process peak moved 1376.0 -> 1375.6 MiB: **0.03%**. The `d(anon)`
column says why. `_score_candidates` retains 16.7 MiB; it is almost pure transient, and it runs
*after* `_candidate_overlaps_and_energies` has already set the mark. Cutting 221 MiB out of a site
that is not the high-water mark moves the high-water mark by nothing.

So Tier 1's "cheapest possible win, no new code" is **refuted**: flipping this default buys
nothing until whatever sets the peak is brought below ~332 MiB. Third predicted lever in this
campaign to evaporate on measurement (`dc-perf-campaign-measured-levers`).

**Where the campaign should go instead:** `_candidate_overlaps_and_energies`
(`cipsi_solver.py:772-832`), which both sets the peak and retains the most. The survey had already
flagged its shape -- `overlaps = np.ascontiguousarray(amps[new_mask].T)` is a boolean-index *copy*
at `p x n_Dj x 16 B`, alongside `psi_all_Dj` materialized by dict comprehension over every local
candidate, plus `H_psi_all` and a redistribute inside it -- but ranked it below the two knobs.
(Step 1 below decomposes it, and finds the survey pointed at the wrong term inside it.)

**Two caveats on these numbers, before they drive anything.** They are **1-rank**, where `n_Dj` is
the *global* candidate count; at 128 ranks each rank holds ~1/128 of it times the ~2.79 routing
skew. All three sites scale with the local candidate count so the ordering should carry, but
`_apply_block_and_redistribute` also carries communication buffers that scale differently. The
ordering is a hypothesis for the cluster, not a measurement of it.

**A note on the instrument itself:** the first version of the ledger reported two sites, not
three, because `_score_candidates` is a module-level function and the probe looked for it on
`CIPSISolver` with `hasattr` -- finding nothing and saying nothing. It now asserts. A probe that
silently measures less than it claims is the same failure class as a sanitizer that is not
running.

### Step 1 result: the peak inside that site is `applyOp`, and the overlaps copy buys nothing either

Step 0 named `_candidate_overlaps_and_energies`; it did not say *which part* of it. The survey's
suspect was the boolean-index copy. Decomposing the function one level further
(`from_arrhenius/overlap_decompose.py`, a line-for-line copy that reads RSS at every internal
boundary) at the worst of ten rounds -- a 316,093 x 44 `Hpsi_ref` block, 296,093 of whose rows are
new candidates:

| boundary | RSS since entry | that step's OWN transient | term |
|---|---|---|---|
| `keys()` / `new_mask` / `local_Djs` | 4.6 MiB | 2.3 MiB | 5.3 MiB |
| `amps[new_mask]` | 205.6 MiB | 201.0 MiB | 198.8 MiB |
| `ascontiguousarray(.T)` | 404.4 MiB | 198.8 MiB | a *second* 198.8 MiB |
| `del` the copy | 205.6 MiB | — | — |
| `phases` / probe dict / `ManyBodyState` | 244.3 MiB | 39.6 MiB | 14.5 MiB |
| `applyOp` | 488.7 MiB | **503.1 MiB** | 2,936k rows |
| `e_Dj` | 489.0 MiB | 0.6 MiB | 2.3 MiB |

**The suspect was real and it still bought nothing.** `np.ascontiguousarray(amps[new_mask].T)`
genuinely holds two full `(p, n_Dj)` buffers at once -- the boolean index copies `(n_Dj, p)`, then
`ascontiguousarray` copies its transpose into a second one -- so the line peaks at 400 MiB to
retain 199. Replacing it with a row-tiled gather that fills one preallocated destination is
bit-identical (md5 of the result buffer matches across five shapes including empty and width 0)
and, measured standalone at exactly that shape, strictly better on both axes:

| form | peak | time |
|---|---|---|
| `ascontiguousarray(amps[mask].T)` | 399.6 MiB | 113.7 ms |
| column-at-a-time `np.compress(..., out=)` | 210.0 MiB | 235.6 ms |
| **row-tiled (2 MiB tile)** | **199.5 MiB** | **58.7 ms** |

And in the full solve it moved the process peak 1376.0 -> 1375.6 MiB and the site's own growth
708.7 -> 708.4 MiB. **Zero.** Reverted.

**Why, and this is the reusable part:** `applyOp` runs *after* the duplicate has been freed, and
carries a 503.1 MiB transient of its own -- larger than the whole two-buffer overlaps sequence.
The site's peak is therefore `RSS on entry to applyOp` + 503 MiB, and RSS on entry to `applyOp` is
insensitive to anything freed before it. Removing a term that is neither live at the peak nor
larger than the peak-setter moves nothing, however large the term is. That is the same shape as
the `GS_SELECTION_CHUNK` refutation one level up, and it is now the **fourth** predicted lever in
this campaign to evaporate on measurement.

**A methodological correction that caused this to be nearly missed.** The first decomposition reset
`VmHWM` once, at function entry, and reported the running maximum at each boundary. That column
cannot distinguish *"this step allocated 300 MiB"* from *"this step ran after 300 MiB was already
allocated"* -- and that distinction is the entire question. It read `applyOp` as +300 MiB on top of
a 408 MiB mark. Resetting at **every** boundary shows its own cost is 503 MiB and the mark it
inherits is irrelevant. Per-step resets, not a cumulative column.

**Where step 2 goes.** `applyOp(H, psi_all_Dj)` on the 296k-determinant probe produces a 2,936k-row
state, retaining ~244 MiB (~87 B/row) and transiently ~503 MiB (~180 B/row). Note that **chunking
the probe is not bit-identical**: `H psi_all = sum_k phase_k H|D_k>`, so splitting the probe and
summing the partial states changes the floating-point accumulation order per determinant. Any fix
here has to reduce the apply's own working set, not partition its input -- or else give up the
bit-identity the rest of this campaign has held to, which needs the user's call, not mine.

### Corrections to earlier claims in this document

* **`reort="partial"` saves projection FLOPs, not store bytes.** Retention is mode-independent
  (`_lanczos_step.pxi:850-853` appends unconditionally); the mode only selects which columns are
  *read*. The "30x/43x the store dominates" multipliers in
  `blocklanczos_reort_memory.md` Phase 0 came from a run that hit the divergence guard and never
  converged; `memory_estimate.py:337-343` already walks that framing back.
* **`GS_SELECTION_CHUNK` is worth ~54 MB at 128 ranks, not 1.5 GB.** `_score_candidates` runs on
  **rank-local** candidates, while the `n_candidates` in the cycle log is `_allreduce_sum`
  (`cipsi_solver.py:869`) — a global count. The ledger's "+202 MB" came from a few-rank run where
  local ≈ global.

### Still open

* ~~Whether the width-6 block under-resolved the manifold.~~ **Closed above: it did not.**
* **Whether unfreezing the GS clears the causality error.** The chain above predicts it does. Re-run
  and check; if `Σ` is still acausal on a healthy basis, the common-mode argument says look at GF
  convergence or the Lanczos band (`:1426` warns `lanczos_band 1.028e-03`, and `weight_add
  1.380e-01` is lost off-mesh) — **not** at the bath fit.
* **`GS_NUM_WANTED` is unset** and `impurityModel-Mn.out:241` says so: the GS peak estimate
  under-counts by up to ~30x, which is why the 20,358,272 cap was optimistic. One line in
  `job.rspt`, next to raising `-c`.
* **The broader memory campaign.** Two full-repo surveys found ~40 candidate sites across DC / GS /
  GF. The ranking is not yet trustworthy — see the `GS_SELECTION_CHUNK` correction above — so the
  next step is a per-site RSS ledger that **attributes `RssShmem` separately from anonymous RSS**,
  since conflating them is what produced the wrong diagnosis here.
