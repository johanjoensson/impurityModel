# Implementation Plan: `ManyBodyOperator::apply` Performance

**Scope:** raise the throughput of `ManyBodyOperator::apply`
(`src/cython/ManyBodyOperator.cpp`) — the second-quantized matvec that dominates
every Block Lanczos sweep (`BlockLanczos.pyx::apply_multi` →
`block_lanczos_step_cy`), CIPSI expansion, and Green's-function / spectrum builds.
The goal is **faster `apply` with bit-for-bit-equal results** (up to key reordering),
preserving the class's ability to represent **any** second-quantized operator:
1-/2-body number-conserving Hamiltonians (the hot case), **unpaired** transition
operators (self-energy / spectra), and **constant shifts** (zero elementary
operators).

Each task has a verification checkbox. "Serial" = `pytest`; "MPI" =
`mpirun -n {2,3,4} pytest --with-mpi`.

> **Execution conventions:** see "Implementation conventions for weak-model
> execution" in `README.md` (anchor on quoted snippets, rebuild + test after each
> checkbox, all decisions pre-made). After editing any `.cpp`/`.h`/`.pyx`, rebuild
> with `pip install -e . --no-build-isolation` (set `BOOST_ROOT` or
> `pip install boost-headers` first if Boost headers are missing), then run the
> test named in that task.

---

## Workload constraints (decide nothing that violates these)

- **`n_orbs` is runtime-determined with NO upper bound.** Typical = 100–200 (→ 2–4
  `uint64` chunks in the `SlaterDeterminant` key), but the hot path must **never**
  assume a compile-time or bounded chunk count. Key length is fixed within a run,
  arbitrary across runs. No `std::array<uint64_t, NCHUNK>` templating; use a reused
  scratch key or `boost::container::small_vector<uint64_t, 4>` (small-buffer
  optimization: typical case on the stack, larger spills to heap and still works).
- **Dominant operators are 1-/2-body number-conserving Hamiltonian terms.** Unpaired
  / transition operators and constants exist but are rare → they keep a **generic
  fallback kernel** for correctness, not speed.
- **SD ownership rule (MPI):** rank `det.get_hash() % comm.size` via the C++
  splitmix64 hash (`SlaterDeterminant.h`), *not* Python `hash()`. See README
  cross-cutting facts.

## Current hot-loop cost centers (anchor: `ManyBodyOperator::apply`, the `for (const auto &[slater, amp] : state)` loop)

- **(C)** `std::unordered_map<key_type, mapped_type> map_res` accumulator — node-per-
  entry allocation, full-vector rehash per insert. Largest single cost.
- **(A)** `out_slater_determinant = slater` — a heap `std::vector<uint64_t>` copy per
  (SD, term): `N_sd × M` allocations.
- **(B)** `create` / `annihilate` recompute the fermion sign by popcount over **all**
  preceding chunks on every elementary operator; consecutive ops redo prefix work.
- **(D)** final `std::sort(local_res)` + flat_map rebuild, and the **serial**
  per-thread `local_maps` merge under `PARALLEL`.

---

## Phase 0 — Benchmark + golden-output regression gate (PREREQUISITE FOR ALL) 

**Goal:** a reproducible baseline and a correctness oracle that every later phase must
keep green. Two artifacts: a Python regression/bench test (durable, CI-runnable,
exercises the real `ManyBodyUtils` path) and an upgraded C++ microbench for local
profiling.

- [x] **0.1 — Representative fixtures.** Added `src/impurityModel/test/test_apply_perf.py`
  building three `ManyBodyOperator` + `ManyBodyState` fixtures via the Python API
  (`ManyBodyOperator(dict[(orbital,'c'|'a'),…])`, `ManyBodyState(dict[SlaterDeterminant, complex])`):
  (a) **Hamiltonian** — random 1-body `((i,'c'),(j,'a'))` + 2-body
  `((i,'c'),(j,'c'),(k,'a'),(l,'a'))` number-conserving terms at `n_orbs=160`;
  (b) **transition** — unpaired single-`'c'` / single-`'a'` terms; (c) **constant** —
  the empty-tuple term `(): shift`. Small (oracle) and large (timing) sizes split so the
  committed golden stays small. *Checkpoint:* ✅ `pytest test_apply_perf.py -q` — 7 passed.
- [x] **0.2 — Golden oracle.** Computes `op.apply_multi([psi], 0.0)` per fixture,
  serializes the **sorted** `(key_chunks, re, im)` list, asserts it matches the committed
  `apply_perf_golden.json` (162 KB; regenerate via `REGEN_APPLY_GOLDEN=1`).
  *Checkpoint:* ✅ loads committed golden & passes; perturbing one amplitude raises
  `AssertionError` (verified).
- [x] **0.3 — Timing harness.** `test_apply_timing` reports median ms over 7 reps under
  `-s`; `test_apply_is_deterministic` asserts repeats agree. *Checkpoint:* ✅ baselines
  (Python `apply_multi`, this machine): hamiltonian ≈ **128 ms** (n_out 188 056),
  transition ≈ **187 ms** (n_out 270 170), constant ≈ **0.7 ms** (n_out 2 000).
- [x] **0.4 — C++ microbench.** `src/cython/perf.cpp::setup()` rewritten to the
  Hamiltonian-shaped fixture (fixed seed); `main()` times `apply` (median/best over 10
  reps) and reports `n_out`; build command documented in the file header. *Checkpoint:*
  ✅ builds & runs. Baselines (n_in 2000, n_out 187 639): **serial median 110.7 ms**,
  **`-DPARALLEL` median 127.4 ms** — parallel is *slower* here (thread-spawn + serial
  merge overhead), corroborating Phase 5's premise.

---

## Phase 1 — Allocation / accumulator / sign micro-opts (after Phase 0; 1a–1d independent)

Low-risk, keep the generic kernel, attack (A)(B)(C)(D) directly. Each sub-task is one
checkbox + golden-oracle re-run.

- [x] **1a — Remove the per-term SD copy (A).** `out_slater_determinant = slater` is now
  done **once per input SD**; each term applies its operators in place and undoes the
  `[start_idx, i)` bits afterward via a new `toggle_bit` helper (bit-toggle is its own
  inverse, order-independent; a failing `s==0` op leaves the key untouched and is
  excluded). Both serial and `PARALLEL` paths. *Checkpoint:* ✅ golden oracle + 29
  tests green; C++ microbench serial 110.7→81.9 ms, PARALLEL 127.4→78.3 ms (combined
  with 1b; PARALLEL now beats serial).
- [x] **1b — Faster accumulator (C).** Swapped `std::unordered_map` for
  `boost::unordered_flat_map` (Boost 1.90, `<boost/unordered/unordered_flat_map.hpp>`)
  with a `SlaterKeyHash` wrapping the `std::hash<SlaterDeterminant>` specialization
  (boost::hash does not pick it up), `reserve(state.size())`. Both serial `map_res` and
  `PARALLEL` `local_maps`. *Checkpoint:* ✅ golden oracle green; **largest single win** —
  Python `apply_multi`: hamiltonian 128→89 ms (−30%), transition 187→120 ms (−36%),
  constant 0.71→0.28 ms.
- [x] **1c — Restriction fast-path.** Hoisted a `check_restrictions` flag
  (`!masks.empty() || !weighted.empty()`) out of the per-output loop; the no-restriction
  case skips `state_is_within_restrictions` entirely. *Checkpoint:* ✅ golden oracle +
  `test_weighted_restrictions` green (36 tests); restricted path unchanged.

---

## Phase 2 — Term classification + masked kernels (this is the core win; depends on 1b)

Classify each term **once** in `build_flat_representation()`; dispatch in `apply`.

- [x] **2a — Classifier.** `build_flat_representation` now tags each term: **diagonal**
  (created-orbital multiset == annihilated-orbital multiset ⟹ occupation conserved;
  includes constants) and, more tightly, **density** (a pure number-operator product —
  diagonal + all-distinct orbitals + a build-time *probe* on the all-involved-occupied
  determinant returns a nonzero, occupancy-independent sign). Stored as per-term
  `m_flat_diagonal` / `m_flat_density` + precomputed `m_density_mask` (occupancy mask)
  and `m_density_coeff` (sign-folded). *Checkpoint:* ✅ golden oracle green.
- [x] **2b — Diagonal / density fast path (the real win).** Per input SD, diagonal terms
  accumulate into one scalar and emit a **single** insert at `slater` instead of M
  colliding inserts; **density** terms further skip `create`/`annihilate` entirely —
  one occupancy AND-test against `m_density_mask` plus a constant-signed add. *Why the
  two tiers:* after 1b the flat-map insert is cheap, so single-insert alone was only
  ~3% on the diagonal fixture — the dominant cost was the per-term sign machinery, which
  the density mask-test removes. *Correctness trap handled:* balanced terms also include
  `c_i c†_i = 1 − n_i` (nonzero when *empty*, expands to a contraction); these fail the
  all-occupied probe and fall back to the general diagonal path, so the fast path only
  ever fires on provable pure-`n` products. *Checkpoint:* ✅ golden oracle +
  `test_diagonal_independent` (occupancy-only oracle, no fermion sign — a true
  cross-check) green; **diagonal fixture 29→12 ms (−59%)** vs the Phase-1 binary.
- [x] **2c — Off-diagonal one-body masked sign (done; measured).** `c^d_i c_j` (i≠j) now
  has its own kernel: occupancy/vacancy bit tests + the fermion sign as
  `(-1)^popcount(state & between_mask(i,j))` with a precomputed between-mask, plus two
  `toggle_bit`s for the output — no `create`/`annihilate`, no prefix scans. *Checkpoint:*
  ✅ golden oracle green (the hamiltonian fixture's 300 hops cross-check the sign against
  the general path) + broader pipeline. **A/B on the hopping microbench (½ one-body, ½
  two-body terms): serial 83.5→79.6 ms (~4.6%), parallel 62.1→60.7 ms (~2.3%)** — a real
  but modest win (one-body is half the terms; a hop-heavier H gains more).
- [x] **2d — GENERAL fallback + two-body deferred.** Off-diagonal **two-body** and
  unpaired/transition terms keep the Phase-1 in-place `create`/`annihilate` path (which
  already early-rejects on the occupancy/vacancy bit test before any popcount). The
  two-body masked sign is **deferred**: unlike the one-body case its sign is
  occupancy-dependent and order-sensitive (each operator sees the state the previous one
  modified), so a correct fused mask needs the incremental between-mask treatment — higher
  risk for a gain bounded by the two-body *producing* fraction (~6% of applications on a
  half-filled determinant). The general path is the correctness fallback for any operator.
  *Checkpoint:* ✅ transition + constant golden green; `test_finite`/`h0`/`density_matrix`/
  `selfenergy` + MPI `test_block_lanczos_cy_mpi` green.

---

## Phase 3 — Build-time normal ordering (this is the Q2 idea; depends on Phase 2)

- [x] **3a — Canonicalize at build.** `collect_flat_terms` rewrites each term to
  canonical order (creations before annihilations, each group ascending) via a recursive
  anticommutator expansion: `c_p c†_q = δ_pq − c†_q c_p` (emits a contraction term when
  `p==q`), `c†_p c†_q = −c†_q c†_p` / `c_p c_q = −c_q c_p`, and drops Pauli-vanishing
  equal-orbital pairs. The expansion is merged (sum equal terms, drop ~zero coeffs) and
  feeds the flat arrays + Phase-2 classification. Gated by `m_normal_order` (default
  **on**), exposed via `set_normal_ordering` / `normal_ordering` / `num_flat_terms` on the
  Cython wrapper. *Checkpoint:* ✅ the Phase-2 golden (generated **without** normal
  ordering) is reproduced with default-on within tolerance — i.e. normal ordering is a
  true identity transform on every fixture's apply action.
- [x] **3b — Confirm it feeds Phase 2 / measure expansion.** `test_normal_ordering_*`
  cover: (i) the `c_i c†_i = 1 − n_i` contraction (2 terms; 0 occupied / 1 empty,
  occupancy oracle); (ii) **A/B invariance** — apply identical with normal ordering on vs
  off, max diff **0.0**, on a contraction-heavy non-normal-ordered operator; (iii) the
  per-fixture **multiplier**, asserted ≤ 1.5. *Checkpoint:* ✅ multipliers on the
  already-normal-ordered fixtures: hamiltonian **0.98** (a Pauli `i==j` term dropped),
  transition / constant / diagonal **1.00** — no blow-up; contraction expansion occurs
  only for genuinely non-normal-ordered input. Broader pipeline (`test_finite`, `h0`,
  `density_matrix`, `selfenergy`, `natural_orbitals`, `greens_function`, `spectra`,
  `gf_autorouting`) + MPI block-Lanczos all green with default-on.

**Phase 3 ROI note:** real impurity Hamiltonians are already normal-ordered, so this is a
performance no-op on them (build-time only, apply unchanged). Its value is (a) robustness
— any operator input order now yields the same canonical apply path — and (b) it lets the
Phase-2 density fast path absorb `1 − n`-style contractions that it otherwise defers.

---

## Phase 4 — MPI: efficient redistribute communication (RESCOPED for 100s–1000s of ranks)

**Why rescoped.** The original premise (avoid Python `pickle` by emitting per-rank
buckets) was wrong on inspection: `redistribute_psis` → `graph_alltoall_psis` already
packs via C++ (`pack_psis_cy`) into raw `uint64`/`complex`/`int32` buffers and exchanges
them with `Neighbor_alltoallv` over a sparse `dist_graph` — **no pickle**. And the local
pack pass is O(n_out per rank), *independent of rank count*, so it is not a scaling wall
(~13% fixed local overhead; fusing it into apply would save ~6–7% single-process only).

The actual scaling limiter at 100s–1000s of ranks is the **communication pattern**: with
hash ownership (`hash % size`) each rank's output scatters across *all* ranks, so the
`dist_graph` neighbourhood is effectively **dense** and the exchange is a latency-bound
personalized all-to-all of small messages, redone every matvec — plus a per-matvec
`Create_dist_graph_adjacent`/`Free` and a dense size-`P` count `Alltoall`. So Phase 4 now
targets the *message pattern*, not the local loop.

- [x] **4a — Fuse the 3 `Neighbor_alltoallv` into 1.** New C++ `pack_psis_fused` /
  `unpack_psis_fused` (`MpiUtils`) serialize each entry as one interleaved record
  `[state | amp | psi_idx]` (`bytes_per_entry = chunks*8 + 20`) into a single rank-ordered
  byte buffer; `graph_alltoall_psis` now does **one** `Neighbor_alltoallv(MPI.BYTE)`
  instead of three. Same total bytes, **3× fewer latency-bound message rounds** — the
  dominant cost of the dense small-message all-to-all at scale. *Checkpoint:* ✅ MPI
  `test_mpi_comm` (round-trip, ownership, ring-exchange) + `test_block_lanczos_cy_mpi` +
  `test_no_ghost_bands` green at **n=3/4/6**; serial regression green.
- [x] **4b — Reuse the `dist_graph` communicator** across matvecs instead of
  `Create_dist_graph_adjacent` + `Free` every call. `_cached_dist_graph` (in `mpi_comm.py`)
  caches the graph comm keyed by `id(parent comm)` (parent pinned to keep the id valid) and
  rebuilds only when an `Allreduce(LOR)` over each rank's
  `(sources, destinations) != cached` says any rank's topology changed — so the collective
  `Create_dist_graph_adjacent` stays consistent across ranks (deadlock-free by construction).
  *Checkpoint:* ✅ MPI `test_mpi_comm` + `test_block_lanczos_cy_mpi` + `test_no_ghost_bands`
  + `test_mpi_block_lanczos_cy` green at n=3/4/6; instrumented check confirms the graph comm
  is built once and reused across repeated redistributes (1 cache entry, stable id).
- [ ] **4c — Sparse count exchange (NBX). DEFERRED — needs real-scale validation.**
  Replace the dense size-`P` `Alltoall` of `send_counts` with a nonblocking-consensus
  (NBX: `Issend` to destinations + `Ibarrier` + `Iprobe`/`Recv` loop) so the count/source
  discovery is O(neighbours) not O(P). This is the deepest remaining scaling lever, but
  (i) it's secondary to 4a/4b (the count message is small — `P` int64, ~8 KB at P=1000 —
  vs the data exchange those fixed), (ii) it's a research-grade rewrite of neighbourhood
  discovery that interacts with the 4b cache, and (iii) its benefit is **unobservable at
  the n≤6 test scale available here** — it must be validated on a real 100s–1000s-rank
  allocation. Recommend doing it as a dedicated, separately-benchmarked effort rather than
  shipping it blind.

---

## Phase 5 — Threading (parallel track; depends on 1b's accumulator choice)

- [x] **5a — Parallel merge.** Each compute thread now partitions its output into
  `num_buckets` (= thread count) sub-maps by key hash; the merge then runs **one
  lock-free thread per bucket** (disjoint key sets), each also applying the cutoff while
  collecting its survivors — replacing the serial `for (auto &tmp_map : local_maps)` merge
  that re-did every insert on one thread. *Checkpoint:* ✅ `IMPURITYMODEL_PARALLEL=1` build:
  golden + independent diagonal + A/B-invariance + broader pipeline + MPI block-Lanczos
  green. C++ microbench (8 cores): hopping fixture serial 85→parallel **63 ms** (merge-
  bound, n_out = 90×n_in); the realistic diagonal fixture **12.4→3.1 ms** (~4×).
- [x] **5b — Workload-scaled thread count (instead of a persistent pool).** A persistent
  pool is high-risk for ~nil gain (per-call spawn is ~0.4% of a large apply); the real
  hazard is small/under-MPI applies. So `apply` now scales the thread count to the input
  (`>= 256` SDs/thread; tiny states run single-threaded), avoiding spawn overhead on small
  matvecs and node oversubscription. The persistent thread pool is **deferred**.
  *Checkpoint:* ✅ small (oracle, 40-SD) and large (2000-SD) fixtures both green on the
  parallel build.

**Phase 5 shipping policy.** The threaded path is **opt-in, off by default**:
`IMPURITYMODEL_PARALLEL=1 pip install -e . --no-build-isolation`. It must NOT be on for
the usual one-MPI-rank-per-core runs (every rank would spawn its own threads and
oversubscribe the node); it is for single-process / few-rank-many-core use. The shipped
default build is serial and unaffected.

---

## Dependency graph

```
Phase 0 (bench + golden oracle) ── blocks everything
   ├── Phase 1 (1a,1b,1c)                  [1b before 5; accumulator co-designed with 4]
   │     └── Phase 2 (classify + diagonal + masked)
   │           └── Phase 3 (normal ordering)
   ├── Phase 4 (MPI bucketed return)        ∥ parallel with 1/2/3
   └── Phase 5 (threading)                  ∥ parallel (needs 1b decided)
```

**Critical path for single-rank speed:** 0 → 1 → 2 → 3. **Parallel tracks:** 4, 5.
**Highest value / lowest risk first:** 1b, 1a, 1c, then 2b/2c.

## Investigation answers (the two ideas in the original ask)

- **Pair create/annihilate to speed the bitmasking (Q1):** *worth it* — realized as
  Phase 2's classification + between-mask O(1)-ish sign and the single-insert diagonal
  fast path. The win comes from the dominant 1-/2-body number-conserving terms.
- **Always normal-order via commutators (Q2):** do it as a **build-time, once-per-
  operator** canonicalization (Phase 3), not at apply time. It mainly *enables* Phase 2
  rather than winning on its own; watch the shared-index contraction expansion (benign
  for 1-/2-body, dangerous for general operators — hence the flag + multiplier guard).
- **MPI restructuring:** Phase 4 — `apply` returns one local state + per-rank send
  buckets, removing the pickle round-trip from `redistribute_psis`.
