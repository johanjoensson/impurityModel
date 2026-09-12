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
