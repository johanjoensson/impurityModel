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

## Still open

- The FCC Ni fill measurement (above).
- Benchmark baselines (`pytest -m benchmark`: `test_apply_perf.py`,
  `test_block_lanczos_perf.py`, `test_bicgstab_perf.py`) not yet recorded — needed before
  Phase 2 (`ManyBodyOperator`: one `apply`) can claim "no regression at width 1".
- Phases 2–7 (the actual consumer migration, the rename, the flat_map class's deletion) —
  see the session plan file for the phase breakdown; each is its own multi-commit body of
  work across a large fraction of `src/cython/` and `src/impurityModel/ed/`.
