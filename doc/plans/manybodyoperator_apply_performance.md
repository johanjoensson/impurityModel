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

- [ ] **2a — Classifier.** In `build_flat_representation`, tag each term as one of:
  `CONSTANT` (0 ops), `DIAGONAL` (creations and annihilations on the identical orbital
  set → key unchanged: `c†_i c_i`, `c†_i c†_j c_j c_i`), `OFFDIAG_1BODY`
  (`c†_i c_j`, i≠j), `OFFDIAG_2BODY`, `GENERAL` (everything else incl. unpaired /
  transition). Store the tag + precomputed orbital indices/masks in parallel arrays.
  Assert the partition is **total** (every term lands in a class or GENERAL).
  *Checkpoint:* golden oracle green (classifier inert until 2b–2d wire kernels).
- [ ] **2b — Diagonal fast path (top-tier for number-conserving H).** For each input
  SD, sum **all** `CONSTANT` + `DIAGONAL` term contributions into one scalar via bit
  tests, then do a **single** accumulator insert at `out == slater` — instead of M
  colliding inserts on the same key. *Checkpoint:* golden oracle green; large speedup
  on the Hamiltonian fixture.
- [ ] **2c — Off-diagonal masked sign.** For `OFFDIAG_1BODY`/`OFFDIAG_2BODY`: occupancy
  + vacancy bit tests for early rejection, fermion sign as
  `(-1)^popcount(state & between_mask)` with a **precomputed** between-orbital mask
  (loops over actual `key_size` chunks). *Checkpoint:* golden oracle green; speedup.
- [ ] **2d — Keep GENERAL fallback.** Route `GENERAL` (and any unclassified) term
  through the existing sequential `create`/`annihilate` path. *Checkpoint:* transition
  + constant fixtures' golden oracles green.

---

## Phase 3 — Build-time normal ordering (this is the Q2 idea; depends on Phase 2)

- [ ] **3a — Canonicalize at build.** In `build_flat_representation`, reorder each
  term's elementary operators to grouped (all creations, then all annihilations),
  tracking the anticommutation sign; emit contraction terms when a creation crosses an
  annihilation of the **same** orbital (`c_i c†_i = 1 − c†_i c_i`). Behind a flag
  defaulting **on** (expansion is bounded for 1-/2-body). *Checkpoint:* golden oracle
  green (canonical form is an identity transform on the operator's action); record the
  term-count multiplier per fixture in the checkpoint note.
- [ ] **3b — Confirm it feeds Phase 2.** Verify the canonical form makes diagonal-vs-hop
  classification unambiguous and de-duplicates order-equal terms; if the expansion
  multiplier on any fixture exceeds ~1.5, **stop and report** (plan bug — revisit the
  no-shared-index-crossing variant). *Checkpoint:* oracle green; multipliers logged.

---

## Phase 4 — MPI: `apply` returns per-rank bucketed states (parallel track; depends on Phase 0, co-design with 1b)

Today `apply_multi` builds one full local state, then `redistribute_psis` →
`graph_alltoall_psis` **pickles whole `ManyBodyState`s** before `Neighbor_alltoallv`.

- [ ] **4a — Bucketed accumulator.** Make `apply` key each output SD into bucket
  `det.get_hash() % comm.size` during accumulation and return a
  `std::vector<ManyBodyState>` of length `size` (`result[rank]` = local keep). Serial
  (`size==1`) → single bucket, no behavior change. *Checkpoint:* golden oracle green
  serial.
- [ ] **4b — Wire to C++ pack/unpack.** Replace the pickle round-trip in the
  `apply_multi` → redistribute path with the existing `MpiUtils` `pack_psis` /
  `unpack_psis` on the raw buckets + `Alltoallv`. *Checkpoint:*
  `mpirun -n 3 pytest --with-mpi src/impurityModel/test/test_block_lanczos_cy_mpi.py`
  green; per-rank ownership identical to before.

---

## Phase 5 — Threading (parallel track; depends on 1b's accumulator choice)

- [ ] **5a — Parallel merge.** Replace the serial `for (auto &tmp_map : local_maps)`
  merge with a tree reduction (or per-output-bucket disjoint maps so threads never
  collide). *Checkpoint:* `PARALLEL` build golden oracle green; thread-scaling improves.
- [ ] **5b — Reuse a thread pool.** Stop spawning `std::thread` per `apply` (currently
  one spawn+join set per matvec); reuse a persistent pool. *Checkpoint:* oracle green;
  per-call overhead drops in the C++ microbench.

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
