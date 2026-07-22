# Unifying `ManyBodyState` and `ManyBodyBlockState`

Working notes for the refactor tracked at
`/home/johan/.claude/plans/i-want-to-change-calm-shannon.md` (session plan file, not
checked in). This doc holds the durable, checked-in record: the measurements, the
decisions, and the before/after. See `blocklanczos_partial_perf_memory.md` Phase 2 for
the original design that introduced `ManyBodyBlockState` deliberately *without*
retrofitting `ManyBodyState` — **that constraint is superseded by this effort.**

## Why

Two containers do overlapping jobs: `ManyBodyState` (`compat::flat_map<SlaterDeterminant,
complex>`, one vector) and `ManyBodyBlockState` (sorted key vector + row-major
`(rows x width)` amplitude matrix, `p` vectors over a shared support). The split costs
~50 boundary conversions (`from_states`/`to_states`/`from_keys_and_amps`) scattered through
the Krylov/reort/MPI layers, duplicated kernels (`ManyBodyOperator::apply` exists four
times; MPI pack/unpack, `redistribute_psis`/`redistribute_block`, `inner_multi`/
`block_inner_cy`, etc. each exist twice), and a third de-facto representation
(`list[ManyBodyState]`) that most solver APIs actually pass around.

**Target**: one state type — `ManyBodyState`, keeping its flat_map map semantics but
storing `p` vectors over a shared determinant support as one flat contiguous matrix.
`p == 1` is an ordinary block, not a special case.

## Approach: migrate consumers first, rename last

Renaming the class first and migrating callers after cannot keep every commit green —
`ManyBodyOperator`, `MpiUtils`, `_krylov_store.pxi` and ~200 Python call sites are bound to
the flat_map class today, so deleting it up front would force several phases into one
unreviewable commit. Instead: `ManyBodyBlockState` grows the full container surface under
its own name first (this is what Phase 1 below did), each consumer module migrates onto it
independently while the flat_map class stays as a bit-for-bit oracle, and the rename +
deletion of the flat_map class happens last, as a mechanical no-op cleanup.

## Memory model

flat_map storage costs ~72 B per stored nonzero (the entry pair plus a 32-B key heap
block per determinant key vector, per `ManyBodyState.memory_bytes`). Shared-support block
storage costs `16*p + 56` B per row (16 B/column times width, plus the row's own key heap
block), independent of how many columns are actually nonzero in that row. Equating the two
gives the break-even fill ratio

```
f* = (16 + 56/p) / 72
```

| p | f* |
|---|---|
| 1 | 1.000 |
| 2 | 0.611 |
| 4 | 0.417 |
| 8 | 0.319 |

At `p == 1` a block and a flat_map cost exactly the same — there is no memory argument for
converting a width-1 list at all; the case for merging list-of-1 sites is entirely about
code simplicity, not memory. `f* = 1.000` there is a correct fixed point, not a rounding
artifact.

## Fill measurements

Method: a probe hooks `Basis.redistribute_psis` (pure Python; `ManyBodyBlockState.from_states`
is a cdef staticmethod and cannot be monkeypatched from outside the extension) and records
`(caller site, p, union support size, total stored nonzeros)` per call, using the existing
`support_stats(states)` (`src/cython/_mpi_pack.pxi`). Verdict per site is `CONVERT` only if
the **worst** (minimum) observed fill across all calls clears `f*` — one bad call sets the
peak memory footprint, so the average/median fill is not the right statistic to gate on.

Real workload used: `impmod_tests/NiO/impmod/verify_fixes` (`nio_20` in
`restriction_diagnostics.WORKLOADS`), via `real_workload.load_workload` +
`run_selfenergy(..., gf_method="lanczos", n_iw=4, n_w=0)`.

| site | p | union | fill min/med | f* | verdict |
|---|---|---|---|---|---|
| `cipsi_solver.py:741 (expand)` | 9 | 104 | 0.504/0.867 | 0.309 | **CONVERT** |
| `groundstate.py:577 (calc_gs)` | 9 | 80 | 0.717/0.717 | 0.309 | **CONVERT** |
| `cipsi_solver.py:427 (determine_new_Dj)` | 9 | 80 | 0.237/0.444 | 0.309 | MIXED |
| `gf_primitives.py:310`, `greens_function.py:1016` | 1 | 10 | 1.000/1.000 | 1.000 | n/a (p=1) |

**Metal tier (`fcc_ni_5`) not measured.** `run_selfenergy` on that workload is expensive
enough that two attempts (at `n_iw=2` and `n_iw=4`) ran for 30+ minutes of wall time without
completing even the ground-state stage in this environment, and were killed rather than
left running further. This is a real gap: FCC Ni is the metal-tier reference workload and
its GF seeds / thermal-eigenstate lists are exactly the kind of heterogeneous, possibly-low-
fill site the measure-gate exists to catch (see `fccni-decision-gate-verdict` in memory — Ni
support sizes there run into the hundreds of thousands). **Before any Phase 6 site on the
metal path is converted, re-run this probe against `fcc_ni_5` with a longer budget (or a
background/scheduled run) and update this table.** Until then, treat any metal-tier
conversion as unverified against the measure-gate.

`determine_new_Dj`'s MIXED verdict means: convert if a later, more careful pass confirms the
worst-case call across production runs (not just this one probe run) clears `f*`; otherwise
leave it a list. It should not be converted on the strength of this single measurement.

## Phase 1 — the block state's container surface

Landed in three commits on `block_state_default`:

1. **`cdef14e`** — gave `ManyBodyBlockState` the row-valued container surface the flat_map
   class already had: `RowSpan` (a `std::span` stand-in for pre-C++20 builds), `row()`
   returning a span instead of a raw pointer, a row-entry iterator with the standard arrow
   proxy, the mapping surface (`find`/`contains`/`at`/`operator[]`/`erase`/`clear`/
   `reserve`/`swap`), the vector space (`norm2`/`+=`/`-=`/`*=`//=`/unary `-`/`add_scaled`/
   `max_norm2`/`count_above`/`truncate`), and the Python `Row` view + dict-like surface.
2. **`15e0764`** — fixed the code review of (1): two CRITICAL findings (a width mismatch
   inside `with nogil` aborted the process via `std::terminate` instead of raising, for
   want of `except +`; `ManyBodyBlockState({})` gave width 1 instead of the polymorphic
   zero width 0, which triggered exactly that abort in the natural accumulator idiom) and
   two SERIOUS findings (a `Row` caught in a Python reference cycle could permanently wedge
   its owning state, because `tp_clear` nulls `Row._owner` before `__dealloc__` runs and the
   old design decremented an export counter there; a rejected `__setitem__` left a spurious
   zero row in the support). The `Row` lifetime model changed from an export-counter block
   (mutation forbidden while any row view is reachable) to a generation counter (mutation
   always allowed; a stale read through an old `Row` raises `RuntimeError`) — the Python
   dict/list iterator-invalidation model.
3. **`e4496fd`** — Phase 1.2, additive surface for the migration phases ahead: `column`/
   `select`, `block_inner_scalar`, bulk `insert_rows` (dict-update semantics), and the four
   in-place operators (`+=`/`-=`/`*=`//=`) closing an oracle-parity gap against the flat_map
   class. Review found and fixed one SERIOUS finding (`insert_rows` didn't resolve a
   duplicate key repeated within one call to "last wins") and two MINOR ones.

Every commit's diff went through an independent subagent code review before landing;
CRITICAL and SERIOUS findings were fixed inline and re-verified against the built
extension, not just reasoned about.

### Gate history

| commit | serial | `-n 2 --with-mpi` |
|---|---|---|
| `207a149` (pre-refactor baseline) | 1162 passed, 230 skipped, 30 xfailed, 99.4 s | 1362 passed, 60 xfailed, 120.2 s |
| `cdef14e` | 1162 passed, ... | 1383 passed, ... |
| `15e0764` | 1189 passed | 1389 passed |
| `e4496fd` | 1199 passed | 1399 passed |

All test-count growth is new tests (`test_state_row_api.py`, 37 as of `e4496fd`); no
pre-existing test was removed or weakened.

## Phase 2 — `ManyBodyOperator`: one `apply`

Phase 2 is deliberately thin, not a caller migration. A caller can only switch to
`apply(block)` once its own data is already a block; the actual apply callers
(`applyOp`, `op(psi)`, `.apply_multi`) live in the basis + solver/physics layer, i.e.
Phases 3–6's territory. Migrating them here would inject exactly the boundary-conversion
churn (`from_states`/`to_states` wrapping) this refactor exists to delete, and double-count
Phase 6. So "move every caller onto `apply(block)`" is the end state Phases 3–6 reach
collectively, not a Phase-2 action; discriminator for future phases: a site belongs in the
phase that converts its *data* to a block, not earlier.

What Phase 2 actually did, landed as `703da53`:

- Fixed the stale `ManyBodyState build_restriction_mask(const restrictions&)` declaration in
  `ManyBodyOperator.pxd:43` (true C++ signature is `void`, `ManyBodyOperator.h:105`). Latent,
  not actually broken — the one call site (`_operator.pxi:281`) discards the result, so no
  temporary was ever fabricated — but wrong on its face. Reviewed independently (verified the
  true signature, the implementation's `noexcept` contract, every call site, and that the fix
  can't change generated code at the discard site); no residual concerns.
- Recorded the Phase-2 acceptance benchmark, doubling as the still-open Phase-0 baseline:
  `pytest -m benchmark` on `test_apply_perf.py::test_apply_block_width_scaling` (the 2000-SD,
  n_orbs=160 hamiltonian fixture) gives, at `p == 1`, block apply **faster** than the
  list-based `apply_multi` it will eventually replace (two runs: 1.06x and 1.21x speedup) —
  comfortably inside the "~5% regression" acceptance bar, in the winning direction. Width
  scaling is preserved: p=2/4/8 speedups of 2.3–2.5x / 3.9–4.3x / 7.1–7.7x over the
  independent-solves baseline, consistent with prior measurements.
  `test_block_lanczos_perf.py`'s apply p-scaling (tiny NiO 124-basis fixture) gives p=1
  block/multi = 0.97x — within tolerance, size too small to read further into.
  `test_bicgstab_perf.py` benchmarks all skip in this environment (missing real-workload
  fixture) — not run, not a Phase 2 regression signal either way.
  `test_block_lanczos_perf.py::test_partial_sparse_kernel_bench` fails
  (`|Q^H Q - I| = 1.449e-04` vs `1e-6`) — this is the pre-existing, already-documented
  `preexisting-partial-bench-orth-failure` (predates TSQR, toy NiO fixture, opt-in only),
  unrelated to this change.

Gate: serial 1199 passed (matches Phase 1 exactly), `-n 2 --with-mpi` 1399 passed (matches
Phase 1 exactly) — the correct outcome for a declaration-only fix plus a benchmark run with
no production code touched.

## Phases 3–4 — traced, found hollow (no code change)

Both phases were scoped against the current tree before writing anything, using the Phase-2
discriminator ("does this stay gate-green without adding a `from_states`/`to_states` a later
phase deletes?"). Neither has a real move available yet:

- **Phase 3's one substantive item, `apply_global_truncation`**, has exactly one caller:
  `_lanczos_step.pxi:220-226`, which does `to_states()` → per-column
  `apply_global_truncation` → `from_states()`. Porting it to operate on a block directly is
  not a mechanical port: the caller today truncates **per-column**, but the `prune_rows`
  comment three lines above it explicitly warns that per-column pruning desyncs the block's
  shared support — a block-native version has to choose whole-row vs per-column, and that
  choice changes the Krylov recurrence. That is exactly the kind of decision the plan
  reserves for Phase 5's highest-effort review, not a Phase-1.2-style additive port. Left
  alone; the call-site sweep and file merge stay correctly deferred to Phase 7.
- **Phase 4's `redistribute_psis`/`redistribute_block` unification** can't collapse while
  callers still pass both types — confirmed by tracing every basis object that reaches
  `block_lanczos_step_cy` (production `Basis`, `_CappedBasisProxy`, and the two test mocks
  that actually exercise this path, `MockBasis` in `test_restarted_lanczos.py` and
  `_FakeBasis` in `test_gf_truncation.py`): all four already implement **both** methods
  today. One incidental finding from that trace: the `hasattr(basis, "redistribute_block")`
  duck-typing fallback at `_lanczos_step.pxi:138-144` (with its `from_states`/`to_states`
  round-trip) appears to have no reachable caller in the tree as it stands — worth deleting
  when Phase 4 lands, but not touched now. It sits inside the MPI hot loop CLAUDE.md flags
  for deadlock history, its value is a 4-line readability trim, and confirming "no caller"
  by grep is weaker evidence than the review Phase 4 will get anyway — the risk/reward
  doesn't clear the bar for touching it opportunistically outside that reviewed phase.

## Phase 6 candidates — traced, both larger than a boundary tweak

The two Phase-0 CONVERT-verdict sites are not simple boundary conversions once traced to
their actual data flow:

- **`cipsi_solver.py:741`** (`expand`, `psi_refs`): threads through `determine_new_Dj`,
  `self.truncate`, and `self.psi_refs` (read by external callers, e.g. `groundstate.py`).
  Converting the `redistribute_psis` call alone would add a conversion at that one line while
  every neighboring consumer stays list-typed — the wasteful pattern the Phase-2 discriminator
  exists to catch. A real conversion means `determine_new_Dj` and `truncate` becoming
  block-native too.
- **`groundstate.py:577`** (`calc_gs`, `psis`): same shape — flows into
  `build_density_matrices`, `compute_gs_statistics`, `compute_entanglement_entropy`, each in a
  different module. Converting the site means converting all three signatures together.

Both are legitimate Phase 6 work, sized the way Phase 6 was always scoped (a whole consumer
module moves together, with its own review pass) — not something to fold into an odd moment
between phases.

## Phase 6a — `groundstate.py`'s `calc_gs` observable pipeline (COMPLETE)

User chose `groundstate.py:calc_gs` over `cipsi_solver.py:expand` as the first Phase 6 site
(smaller-looking at a glance; that estimate itself grew twice as the trace went deeper — see
below). Decision: convert fully, keeping `calc_gs`'s external return contract as
`list[ManyBodyState]`, so every external caller (`susceptibility.py`, `get_spectra.py`,
`selfenergy.py`, 5 test files) is unaffected. In the event the return needed no explicit
`.to_states()` conversion: `psis` (the list) was never discarded — it survives alongside
`psis_blk` for its own genuinely list-only uses (`add_states`, `redistribute_psis`), so the
function's existing return statement already produces the right type with no round trip.

Landed as 5 commits (`24a5146`, `0bccfad`, `ea1eaea`, `9014fc0`, `e8891b5`), each independently
reviewed, gates green throughout (serial 1199 / `-n 2` 1399 unchanged end to end — a pure
internal-representation refactor, no test added or removed). One deliberate scope revision
mid-way (`build_density_matrices` stayed a thin shim rather than going block-native — see item
3b below) after an advisor review flagged that its per-orbital annihilators are too cheap to
amortize a shared apply, while reading per-state values back out would cost a p-fold Gram-matrix
diagonal with no cheap primitive to avoid it. The two genuine algorithmic wins
(`manifold_observable_values`'s shared-apply, `_local_partials`/`compute_impurity_rdm`'s
one-pass-per-determinant) landed as designed; the crc32-keyed `graph_alltoall` in
`compute_impurity_rdm` (the checklist's highest MPI-review-priority item) came through review
clean.

**Why this got bigger twice, on the record** (so a future session doesn't have to re-derive
it): the first pass found 3 downstream functions (`build_density_matrices`,
`compute_gs_statistics`, `compute_entanglement_entropy`). Reading `calc_gs` end to end found
4 more calls to `manifold_observable_values` (S²/L²/J² Casimirs, `<S_imp.S_bath>`, its z-only
variant) plus `compute_static_susceptibilities` and the `try` block around
`compute_correlation_diagnostics`/`compute_screening_diagnostics` — all of which *also* call
`manifold_observable_values` internally, several times each (`compute_screening_diagnostics`
loops over every bath pair, up to `max_bath_correlation_levels=200`). Tracing
`compute_entanglement_entropy` one level deeper found it bottoms out in
`compute_impurity_rdm` (`gs_statistics.py`) — a third file, with its own MPI collective
(`crc32`-keyed `graph_alltoall`, not a plain `Allreduce`) and block-diagonal RDM construction
keyed on impurity electron count. Also: `psis` is part of `calc_gs`'s **return tuple**, so its
type is a public contract, not just an internal detail.

**Conversion checklist** (✅ = landed and reviewed; the rest not started):

1. ✅ **`manifold_observable_values`** (`observables.py:1295`, commit `24a5146`) — the common
   leaf, called ~10 times. Was: `eigenstates: list[ManyBodyState]`, `apply_op(psi) ->
   ManyBodyState` (called once per state), `o_matrix` built via a Python-level `n x n` list
   comprehension of scalar `inner()` calls. Now: `eigenstates: ManyBodyBlockState`,
   `apply_op(block) -> ManyBodyBlockState` (one `apply_block` covering all `p` states — the
   actual win, cutting the redundant term/sign work `p`-fold), `redistribute` block-based,
   `o_matrix` via one `block_inner_cy` call. `calc_gs` builds `psis_blk =
   ManyBodyBlockState.from_states(psis)` once, right after the existing `redistribute_psis`
   call, keeping the original `psis` list alongside it for items (3)-(5) below. Reviewed at
   high effort (MPI-collective + physics correctness): `block_inner_cy`'s summation order
   confirmed bit-for-bit-equal to the old scalar path; every per-loop closure confirmed to
   capture its operator correctly (no late-binding bug); the bra ket redistribute routing
   (`redistribute_psis` for build_density_matrices etc. vs `redistribute_block` for the
   converted calls) confirmed to hash identically per determinant, so nothing is dropped from
   a Gram matrix built from two different redistribute calls. No CRITICAL/SERIOUS findings;
   fixed 2 MINOR (black formatting, one orphaned import). **Follow-up noted, not fixed**: no
   test exercises the `comm != None` distributed path's VALUES directly (only that it doesn't
   crash/deadlock, via the `-n 2` `calc_gs` test, which doesn't assert the Casimir/sisb/
   susceptibility numbers since they're wrapped in a degrade-to-`None` `try/except`) — a
   worthwhile follow-up test, not blocking.
2. ✅ `compute_correlation_diagnostics`, `compute_static_susceptibilities`,
   `compute_screening_diagnostics` (`observables.py`) — their own `psis` parameter is now a
   block; landed in the same commit as (1).
3. `build_density_matrices` (`basis_transcription.py:195`) — independent of (1)/(2), own
   commit. Today: `for psi_n in psis: phi = [op_orb(psi_n, 0) for orb in ...]` (`p * n_orb`
   scalar applies). Target: one `apply_block` per orbital (`n_orb` block-applies of width
   `p`), `rhos[n]` extracted from `block_inner_cy`'s full Gram diagonal per orbital pair.
3b. ✅ **`build_density_matrices`** (`basis_transcription.py:195`, commit `0bccfad`) — NOT
    rewritten block-native (initial plan target revised after advisor review). Accepts a
    `ManyBodyBlockState` via a thin `isinstance` + `to_states()` shim on entry; body
    unchanged. Rejected the block-native rewrite: this function applies many trivial
    single-term annihilators (little shared term/sign work an `apply_block` would amortize),
    while reading per-state values back out would need the diagonal of a full `width x width`
    Gram matrix per orbital pair — `p`-fold more inner-product work than the existing
    per-state loop, with no cheap diagonal-only primitive to avoid it. `calc_gs` passes the
    already-built `psis_blk`, so its call needs no conversion.
4. ✅ **`_local_partials` / `compute_gs_statistics`** (`gs_statistics.py:108,146`, commit
   `ea1eaea`) — pure support iteration, no inner products, so this one *is* a genuine
   algorithmic win (unlike 3b): was `for n, psi in enumerate(psis): for state, amp in
   psi.items(): ...` (`p` separate dict traversals); now one pass over the block's rows
   (`for det, row in blk.items(): for n in range(width): amp = row[n]`) — each determinant's
   config/occupation info computed once instead of redundantly per state. Missing
   determinants of a column are exact zeros in the block (documented `from_states`
   contract), so summing over every column reproduces the old sparse accumulation exactly.
   `compute_gs_statistics`'s `[... for _ in psis]` idiom (relied on iterating `psis` for its
   width) fixed to `[... for _ in state_configs]`, since iterating a block yields one entry
   per row, not per column. Manually verified (the review subagent hit a session token limit
   mid-run): `from_states`'s zero-padding contract, `items()`'s one-row-per-determinant
   iteration, `state_configs`'s length invariant under either input type, and that no MPI
   collective downstream changed.
5. ✅ **`compute_impurity_rdm`** (`gs_statistics.py:388`, commit `9014fc0`) — same shape as
   (4): a determinant's impurity config / bath key depend only on its own bit pattern, so
   the local pass visits each row once instead of once per `(state, determinant)` pair.
   Zero-padded columns are skipped via `if amp != 0`, reproducing the old sparse traversal's
   exact entry set (a zero-amplitude outer product contributes nothing either way) while
   avoiding shipping `p`-fold more entries through the crc32-keyed `graph_alltoall` than the
   sparse states actually held. Reviewed at high effort (the highest MPI-review-priority item
   in this checklist — an alltoall, not a plain `Allreduce`): the alltoall's participation
   stays unconditional regardless of how the filter thins a rank's local groups, the
   `observed_n` allgather and the final per-`(n,n_e)` `Allreduce` loop are untouched, and the
   hoisted per-row computation is bit-for-bit identical to recomputing it per state. No
   findings. `compute_entanglement_entropy` needed no change (only forwards `psis` through).
6. ✅ **`calc_gs` itself** (`groundstate.py`, commit `e8891b5`) — no functional change needed:
   steps 1-4 already left `calc_gs` in the target end state as a side effect of building
   `psis_blk` once and reusing it everywhere. The plain `psis` list survives only where it's
   genuinely still a list (`add_states`, `redistribute_psis`, the function's own
   `list[ManyBodyState]` return contract) — converting the return via `.to_states()` would
   have been a pointless round trip since the list was never discarded. Landed as a stale-
   comment fix only.

Each step got its own gate run (serial + `-n 2 --with-mpi`) and, per the plan's review
process, an independent code review before landing — (5) at the highest effort level (MPI
alltoall collective).

## Phase 6b — `cipsi_solver.py`

Traced with a discriminator Phase 6a's `build_density_matrices` decision didn't need: the
Phase-0 fill gate is a MEMORY verdict per call site, not a blanket verdict on the file, and it
does not always agree with which sites have a compute win. Three sites in this file, three
different outcomes:

- **`determine_new_Dj`'s / `select_at`'s `Hpsi_ref` apply+redistribute path** (the
  `Hpsi_ref = [applyOp_test(H, psi_i, ...) for psi_i in psi_ref]` then
  `self.basis.redistribute_psis(Hpsi_ref)` pattern) has the same "few expensive operators,
  many independent per-state applies" shape that made `manifold_observable_values` a clean
  win in Phase 6a — but its own fill measurement (`cipsi_solver.py:427`, see "Fill
  measurements" above) came back **MIXED** on `nio_20` (0.237–0.444 vs f*=0.309) and was
  explicitly left unconverted "until a later, more careful pass confirms the worst-case call
  across production runs clears f*". The metal-tier (FCC Ni) measurement that would settle
  this is the one this environment can't complete (30+ min, killed twice). **Left
  unconverted** — the compute win alone does not license overriding a memory-gate
  precondition the plan itself wrote down.
- **`truncate`** (commit `a0c9743`) operates on the *clean* reference/eigenvector manifold —
  the same fill profile as the `expand:741` CONVERT-verdict site, not the wider H-applied
  candidate space `determine_new_Dj` builds — so it isn't subject to that MIXED verdict.
  Landed: reuses the `ManyBodyBlockState` it already built for `row_max_norms2` (previously
  discarded) via `keep_rows` instead of a second per-state dict-comprehension pass, and one
  `redistribute_block` instead of `len(psis)` separate `redistribute_psis` transfers.
  Reviewed at high effort (this is the core CIPSI selection loop): `keep_rows` only ever
  reads the mask's key set (row amplitudes untouched), determinant ownership routes
  identically through `redistribute_psis`/`redistribute_block` (both hash on
  `routing_hash() % comm_size`), the empty-`retained` edge case matches old behavior on
  every rank. No findings.
- **`determine_new_Dj`'s symmetry-generator closure** (commit `a0c9743`) had a genuinely
  pre-existing, block-independent O(n²) hazard: `next_chunk_state[state] = amp` inserting
  one key at a time into a flat_map (`operator[]` on a missing key is a sorted-vector
  insert). Not a block-conversion question at all — width-1, no `p` dimension — fixed by
  batching into a plain dict and constructing once (one bulk range-insert), safe because
  each determinant can only be written once across the whole pass (the existing
  `state not in new_Dj` guard).

`expand`'s own outer `psi_refs`/`psi0` were traced and deliberately **left as a list**: the
actual eigensolve in the `dense_cutoff` branch runs through the *array* IRLM kernel
(`build_distributed_vector`/`build_state` boundary conversions to/from a dense matrix, per
`cipsi-partial-runs-array-kernel` in memory) — there is no `ManyBodyOperator.apply` in that
hot path for a block to speed up, and `block_normalize`'s existing array/list dispatch
(`_reort.pxi`) is Phase 5's territory, not this one's. Converting `psi_refs` end-to-end would
also force `determine_new_Dj`'s parameter to become a block, which forces exactly the
MIXED-verdict conversion above through propagation, or a wasteful `to_states()` round-trip at
its entry — neither is worth it for zero compute win.

## Phase 6c — GF/spectra/RIXS/susceptibility modules (TRACED, hollow — no code change)

Swept `gf_solvers.py`, `gf_shift_recycling.py`, `gf_units.py`, `gf_primitives.py`,
`greens_function.py`, `spectra.py`, `rixs.py`, `susceptibility.py`, `basis_split.py` for the
same three things Phase 6b looked for: a redundant list↔block round trip on an already-formed
union (fill-neutral, convertible now), a genuine per-operator/per-state union not yet formed
(fill-gated, same class as `Hpsi_ref`), and any leftover O(n²) single-key-insert hazard. Every
site resolved into one of five buckets, and none of them is "convert now":

1. **Already block-native, nothing to do.** `gf_units.py`'s `_apply_transition_ops` builds
   `psi_blk = ManyBodyBlockState.from_states(list(psis))` once (the thermal eigenstates
   genuinely share support) and runs `tOp.apply_block(psi_blk, ...)` per operator — exactly
   the Phase 6a win shape, already landed (not part of this session's work, but confirmed
   correct and not needing further change).
2. **Fill-gated, same class as `Hpsi_ref` — left alone pending the FCC Ni measurement.** Every
   per-frequency/per-shift seed union in this file family has the same shape: independent
   per-(eigenstate, operator) applies get stacked into one block only at the point they're
   redistributed onto a rebuilt-and-discarded `tmp_basis` (`gf_solvers.py`'s
   `block_Green_bicgstab` loop at `:689-696`/`:741-744`, `gf_shift_recycling.py`'s
   `_expand_to_closure` seed union at `:174-175`, `rixs.py`'s R1 fallback at `:328-338`,
   `greens_function.py`'s spectrum-slicing `unit_seeds[u]` redistribute at `:822`). None of
   these forms a genuinely new union beyond what `Hpsi_ref` already measured MIXED on — they
   are the same seed-construction pattern reappearing at each call site, not new evidence — so
   the same held-pending-measurement verdict applies without a separate number for each.
3. **Structurally not a shared-support block.** `greens_function.py`'s `get_greens_function_moments`
   (`side_krylov_moments`) applies `n_corr` *different* creation/annihilation operators to *one*
   reference state (`s0 = [op(psi_n, 0) for op in creation]`) — the inverse shape of Phase 6a's
   win (many states, one operator), so there is no shared term/sign work to amortize by
   blocking, and the cross term is already computed as one batched Gram matrix via
   `inner_multi`, not a loop of scalar `inner()` calls. Nothing to convert.
4. **Width-1, excluded by the plan's own rule (`f* = 1.000`).** `gf_primitives.py:310`,
   `greens_function.py:1016`/`:1033`, `susceptibility.py`'s per-seed
   `redistribute([seed])[0]` at `:226`, and `gf_solvers.py`'s `block_green_impl` residual
   probe (`last_q`/`q_last = Q_list[:, -1:]`, `:301` — verified this is a *single* Krylov
   column regardless of the block width `n`, so the `ManyBodyBlockState.from_states(list(last_q))`
   at `:85`/`:102` is already a degenerate width-1 wrap, not a `p > 1` site as it first looked).
5. **Already scoped to a later phase — converting now would fragment it.** `gf_solvers.py:743`'s
   `ManyBodyBlockState.from_states(list(X))` looks like a redundant round trip (`X` comes back
   from `solve_shifted_block` → `block_bicgstab`), and it is one: `BiCGSTAB.pyx:222-223`
   already converts its inputs to a block internally and runs the whole core iteration on it,
   only converting back via `.to_states()` at `BiCGSTAB.pyx:318`. But `block_bicgstab` and
   `block_gmres` are shared kernels (`cg.py`/`gmres.py`) with several callers beyond this one
   site, and removing their list boundary is exactly what **Phase 5** ("`ChebyshevFilter.pyx`,
   `BiCGSTAB.pyx`, `GMRES.pyx`, `cg.py`, `gmres.py`: block end to end; the list boundary at
   entry/exit disappears") already commits to doing properly, across every caller at once.
   Patching this one caller ahead of Phase 5 would just split that future diff without a
   present compute or memory win (`X` is still handed to `hist_x`/`_warm_start_extrapolation`
   as a list a few lines later regardless). `greens_function.py`'s spectrum-slicing seeds
   feeding `chebyshev_apply` (`:839`) are blocked the same way — `ChebyshevFilter.pyx` is
   list-based until Phase 5. `basis_split.py:241`'s cross-color `psis` transfer is its own
   hand-rolled byte-serialization protocol over MPI intercomms (not the standard
   pack/unpack path Phase 4 unifies) — converting it safely needs the `-n 3` multi-rank gate
   Phase 4 already flags for this file, so it stays with Phase 4/7 rather than a one-off 6c
   edit.

No O(n²) single-key-insert hazard turned up in this batch (the one instance, in
`cipsi_solver.py`, was Phase 6b's).

## Phase 5 step 1 — `block_bicgstab`/`block_gmres` go block end to end (DONE, commit `e47ad5c`)

Before writing anything, read `BiCGSTAB.pyx`, `GMRES.pyx`, `ChebyshevFilter.pyx`, `_trlm.pxi`
and `_irlm.pxi` in full, because the plan's "delete the dual dispatch" framing covers two
structurally different situations that must not be sized the same way:

- **BiCGSTAB and GMRES were already internally block-native.** Their sparse-path cores
  (`_block_bicgstab_core`, the Arnoldi loop in `block_gmres`) never touch
  `list[ManyBodyState]` — every operation already ran on the `ManyBodyBlockState` the
  `is_arr` branch builds on entry. The `ManyBodyBlockState.from_states(list(x0))` /
  `.to_states()` at entry/exit were pure boundary ceremony around an already-block
  computation — a **relocation**, not a conversion.
- **`_trlm.pxi`/`_irlm.pxi` are genuinely list-based internally.** `_q_slice`/`_q_concat`/
  `_copy_block` (the shared `_irlm_core`/`_trlm_core` basis helpers) operate on
  `list[ManyBodyState]` via plain Python slicing/concatenation for the ManyBodyState path,
  mirroring the array path's `(N, k)` ndarray column-slicing. Converting `Q_basis` in these
  to a `ManyBodyBlockState` needs real new block-native slicing/concat primitives, not a
  boundary move. **Not started** — this is the bulk of what's left in Phase 5.

Shipped this step: moved `block_bicgstab`/`block_gmres`'s sparse-path list boundary out to
every caller, so the two functions now accept and return `ManyBodyBlockState` directly.
The real win isn't just deleted ceremony: `gf_solvers.solve_shifted_block`'s restart loop
used to round-trip list->block->list on *every* BiCGSTAB attempt (up to
`1 + GF_BICGSTAB_RESTARTS` of them) and again into the GMRES escalation; now the same block
carries through the whole chain with zero conversions in between. Updated every caller:
`gf_solvers.py` (`solve_shifted_block`, `block_Green_bicgstab`, `block_Green_cipsi` — the
last of these found by a full-repo grep for `solve_shifted_block(`, not just
`block_bicgstab(`/`block_gmres(`, after the first gate run caught a caller the narrower grep
had missed), `rixs.py`'s R1 fallback, and the BiCGSTAB/GMRES test suites (`test_cg.py`,
`test_gmres.py`, `test_gf_bicgstab_driver.py`, `test_bicgstab_perf.py`).

Reviewed at high effort (subagent, focused on the MPI collective rules and the new
pass-by-reference aliasing surface — the old `from_states` boundary conversion incidentally
built a fresh copy, which is now gone). No correctness findings: every sparse-path
operation on `x0`/`y` was traced and confirmed non-mutating (`apply_block`,
`redistribute_block`, `block_add_scaled_cy`, `combine_columns` all return fresh blocks;
the one in-place mutator, `prune_rows`, only ever runs on freshly-built results). One
low-severity note acted on: `test_block_bicgstab_sparse_info_and_rhs_untouched` asserted
only on the *original list* of states, which the `from_states` copy at the test's own call
boundary made blind to block-level mutation — the actual new hazard surface. Strengthened
to snapshot the passed block's own amplitudes (`np.asarray(y_blk)`) before/after.

Gates green, matching baseline exactly: serial 1199/230/30, `-n 2` 1399/60.

**Traced and explicitly NOT converted: `ChebyshevFilter.pyx`.** `chebyshev_apply`'s
recurrence is also already block-native internally (`block_add_scaled_cy`/`combine_columns`,
no `is_array` dispatch at all). But unlike BiCGSTAB/GMRES, its list boundary is not
redundant: the entry conversion is tied to `redistribute_psis` (list-based; unifying it is
Phase 4's job), and the exit conversion is inherent to the caller
(`greens_function.py`'s spectrum-slicing loop unpacks each filtered window into per-state
`prune`/concatenation-with-seeds to build a fresh ad-hoc list per window — Phase 6c's
already-deferred fill-gated seed-union shape). Relocating the boundary here would not
eliminate any round trip the way it did for BiCGSTAB/GMRES, so there is nothing to do until
either Phase 4 or the FCC Ni fill measurement unblocks the caller side.

## Still open

- The FCC Ni fill measurement (above) — blocks `determine_new_Dj`'s/`select_at`'s `Hpsi_ref`
  conversion, every per-frequency seed union found in Phase 6c, and (per the note just
  above) `ChebyshevFilter.pyx`'s caller in `greens_function.py`.
- Phase 5's remaining, larger piece: `_trlm.pxi`/`_irlm.pxi`'s `Q_basis` bookkeeping (real
  list-to-block conversion, needs new block-native slice/concat primitives), and then the
  `_reort.pxi` collapse itself (`is_array` 3 arms -> 2, delete `block_combine_sparse` /
  `block_orthogonalize_sparse` / `block_normalize_sparse` / `_block_inner_mpi`, drop the
  `as_list` branch in `block_tsqr` and `was_block` in `apply_reort`) once nothing feeds a
  bare list into those dispatchers — grep-checkable: unblocked exactly when no caller passes
  `list[ManyBodyState]` into `block_normalize`/`block_combine`/`block_orthogonalize`/
  `block_tsqr`/`apply_reort`.
- Phase 7 (the rename, the flat_map class's deletion) — see the session plan file for the
  phase breakdown; each remaining phase is its own multi-commit body of work across a large
  fraction of `src/cython/` and `src/impurityModel/ed/`.
- No further Phase 6 sites identified — 6a, 6b and 6c between them covered every module the
  plan's Phase 6 list named.
