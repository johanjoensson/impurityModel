# Flat key store: one `vector<uint64_t>` per block, `SlaterDeterminant` as a view

**Status: IMPLEMENTED 2026-09-22 on branch `flat-key-store`, and it cleared the gate.** The
case and hazards below stand as written; the measured outcome is at the end, under "What
actually happened". One hazard (5, the byte constants) is deliberately left open and is the
only part of the design not carried out.

## The idea

`ManyBodyBlockState` today stores `std::vector<Key> m_keys`, where every `Key` is itself a
heap-owning `std::vector<uint64_t>` (`SlaterDeterminant.h:20-21` — `SlaterDeterminant` *inherits*
`std::vector<CHUNK>`). So an N-determinant block holds **N separate small heap allocations**, one
per determinant, plus the pointer array indexing them.

Replace that with a single flat buffer — a `vector<uint64_t>` of `N * n_chunks`, i.e. a row-major
matrix of determinants — and make `SlaterDeterminant` a **view** (pointer + length) into it. The
amplitudes are already stored exactly this way (`m_amps`, one contiguous `vector<Value>` with
`RowSpan` views over it), so this makes the key storage match the amplitude storage rather than
introducing a new idea.

## Why: both open problems have this as their common cause

**Memory.** Measured today, per determinant at nso=124 (one cold process per point, marginal slope
over N = 200k -> 400k; see `memory_footprint_audit.md` 0a/0h):

| `Basis` backing | B/det |
|---|---|
| `_index_dict` (original) | 172.8 |
| Python `bisect` over a list | 88.2 |
| width-0 `ManyBodyState` (today) | **52.3** |
| **flat key store (predicted)** | **~16-24** |

The 52.3 is ~24 B of `std::vector` control block plus a ~32 B allocator-rounded heap block, for
**16 B of actual determinant**. A flat buffer stores the 16 B and nothing else. That is a further
~2-3x on the dominant per-determinant term, which is the term multiplied by a combinatorially
growing determinant count — i.e. it is a ceiling-setter in the campaign's ranking sense.

**Speed, which is the part that was not predicted.** Moving the index lookup into C++
`std::lower_bound` bought only 1.1-1.5x over a Python `bisect` and remains **~2.3x slower than the
dict it replaced**, even batched (`find_rows`, 830 ns vs the dict's 358 ns at N = 1M). The reason
is this same structure: a binary search over N determinants dereferences ~log2(N) *separate malloc
blocks* scattered across the heap. A flat buffer makes the search walk one contiguous array with a
stride, which is what the hardware wants.

Phase 0b of the audit demoted the per-key heap allocation as a *memory* lever — correctly, it is
flat in `n_orb` and shrinks as a share of cost. This note is the correction to the scope of that
demotion: it is a first-order **speed** liability for ordered lookup, and the flat store is the
one change that addresses both axes at once.

**Correctness, as a side effect.** With a fixed width per block, the variable-length key ambiguity
found during this work disappears by construction: today `from_bytes(b"\x80")` and
`from_bytes(b"\x80" + b"\0"*8)` are two *unequal* keys with different hashes for the same physical
occupation, because `std::vector::operator<` ranks a proper prefix below its zero-extended twin.

## Hazards this design has to clear

1. **View invalidation.** Any growth of the flat buffer reallocates and dangles every outstanding
   view. `ManyBodyBlockState` already rebuilds rather than mutates in place for most operations
   (`add_scaled` and friends build fresh and `std::move`), and the class comment already warns
   that key-by-key insertion is O(n^2) — but `insert_row` / `operator[]` do mutate. The generation
   counter `ManyBodyState` already maintains for `Row` views is the existing precedent for
   handling exactly this, and should be reused rather than reinvented.
2. **`SlaterDeterminant` is currently an owning value, used standalone.** It is not only a block
   key: `ManyBodyOperator::apply` builds them as scratch (`out_sd`), `MpiUtils` unpacks into them,
   restriction masks are `SLATER`s. A view type cannot serve those. The change almost certainly
   needs an explicit split — an owning determinant and a non-owning view over a flat buffer, with
   the comparison/hash/`routing_hash` logic shared. `routing_hash` reads `data()`/`size()` only, so
   it works unchanged over either.
3. **Uniform width becomes a hard invariant, and that is fine** — it is already assumed.
   `MpiUtils.cpp:16` takes `chunks_per_state` from `dets[0].size()` for a whole set, and `Basis`
   now normalizes every determinant to `n_bytes` on input. Make it explicit and checked rather
   than inherited.
4. **The ordering contract is load-bearing across ranks.** `Basis` derives global indices from
   `offset + position` with `offset` from an `allgather` of local lengths, so *every rank must
   compute the same total order*. Comparison must stay element-wise over chunks in the same chunk
   order. Verified during this work that byte order, chunk order and padded-lexicographic order
   currently agree; a flat store must preserve that, and the regression test must pin comparison
   order on keys differing only in their high chunks, not merely membership.
5. **Every byte-accounting constant becomes wrong.** `_key_heap_bytes`, `_FLAT_MAP_ENTRY_BYTES`,
   `bytes_per_determinant`, `ManyBodyState.memory_bytes`, `SparseKrylovDense.memory_bytes` and
   `_PY_BASIS_OVERHEAD_BYTES` all encode the per-key-allocation model. They feed RAM-derived
   `truncation_threshold` defaults, so recalibration **changes which determinants are admitted on a
   capped run** — a tier-3 effect requiring its own commit and its own before/after numbers, not a
   silent follow-on. (`_PY_BASIS_OVERHEAD_BYTES` is *already* stale from the Step A/B work and is
   an outstanding item independent of this note.)
6. **The `debug` build is the only one that catches an out-of-bounds read here.** A flat buffer
   indexed by `row * n_chunks` is exactly the shape where an off-by-one returns a plausible
   neighbouring determinant instead of crashing. Develop this under
   `IMPURITYMODEL_BUILD=debug`, and note every kernel carries `wraparound=False`, so a negative
   index is not a safety net.

## What would confirm it

Same instruments as this campaign used, no new ones: the cold-process per-determinant slope
(`scratchpad/one_point.py` shape), the lookup benchmark against the dict baseline
(`lookup_bench3.py` shape, which must now *beat* 358 ns/lookup at N = 1M rather than lose to it),
and the four-leg test gate. The memory prediction above is arithmetic, not a measurement, and
should be labelled as such until a cold-process slope says otherwise.

### The cheaper middle path, priced and REFUTED

Before committing to the view split, the obvious cheaper variant was measured: give
`SlaterDeterminant` fixed **inline** storage (capacity 4 chunks, covering nso <= 256) instead
of a heap block. `std::vector<Key>` then *is* a contiguous strided array — the cache win
without a view type, without touching `key(r)`'s return type, and without a single call site
changing, since `size()`/`data()`/`operator[]` all still work.

Same harness, same probes, N = 1M, three runs:

| representation | ns/lookup | key store | packed-path prediction | vs the 286 ns bar |
|---|---|---|---|---|
| `vector<vector<uint64_t>>` (today) | 595-618 | 53.41 MiB | 628 (measured) | 2.2x slower |
| **inline, `vector<InlineKey>`** | 304-316 | 38.15 MiB | **344** | **still loses** |
| **flat `N x n_chunks`** | 174-185 | 15.26 MiB | **216** | **beats by 1.33x** |

The middle path buys 2x on speed and only 1.4x on memory, and lands at 344 ns — **above the
dict it has to beat**. It would be a real improvement that fails the gate, which is precisely
the shape this campaign exists to avoid shipping: four predicted levers were each standalone
2x better and each moved the peak 0.03%.

Why it falls short is structural, not incidental: `sizeof(InlineKey)` is 40 B against the flat
store's 16 B stride, so a binary search touches ~2.5x the cache lines. Shrinking the inline
capacity to 2 chunks gives 24 B, still 1.5x the stride and still short.

**So the design as written is the one to build.** This is recorded because the middle path is
the natural thing for the next reader to propose, and the answer is a number rather than an
argument.

## Scope note

This is a substrate change touching `SlaterDeterminant.h`, `ManyBodyBlockState.h`,
`ManyBodyOperator.cpp`, `MpiUtils.cpp` and every `.pxi` that constructs a determinant. It should
be staged behind its own equivalence tests and land on its own, not bundled with anything else.


## What actually happened

Three commits on `flat-key-store`, off `master` at `02731c3`, each four-leg green on **both**
`IMPURITYMODEL_BUILD=debug` and `release`:

- `510a40c` pins the determinant total order, before any call site moved.
- `c6fb80d` replaces `std::vector<Key>` with `FlatKeyStore` (an `N x n_chunks` buffer plus a
  non-owning `View`), and converts the three `apply` loops and the MPI routing to a scratch
  key hoisted out of the loop.
- `d7c1274` moves `from_states`' union support into C++ (`from_columns`).

### The gate, re-measured on the implementation (release, N = 1M, nso = 124)

| | before | after | |
|---|---|---|---|
| `find_rows_packed` | 617-641 ns | **252-265** | 2.45x, **beats the 300-307 ns dict** |
| `find_rows` (object path) | 732-734 ns | **407-425** | 1.76x, still above the dict |
| `Basis` peak | 244.2 B/det | **180.2** | 1.35x |
| `Basis` retained | 130.4 B/det | **72.1** | 1.81x |
| a 40k-row width-32 block | 68.45 MiB | **19.54** | 3.5x, against a 20.14 MiB payload |

**The bar is met on the path that decides it.** The screening decomposition predicted 218 ns
for the packed path and it came in at 252-265 -- within 15%, so the method of measuring the
C++ core in isolation and adding back the measured Python overhead held up. The object path
is nearly halved but stays above the dict, and that is structural rather than unfinished:
~143 ns of it is handling one Python `SlaterDeterminant` per query, which no storage change
removes.

### Corrections to the prediction

- The design predicted **~16-24 B/det** for the `Basis` backing. Retained came out at 72.1
  B/det on this campaign's probe. Those numbers are not comparable: this probe's retained
  column never reproduced the 52.3 B/det the design quotes either (it reads 130.4 on the
  pre-change code, against ~56 structurally), because it counts construction transients glibc
  does not return. The **ratio**, measured with one probe on both sides, is 1.81x.
- The first measurement was taken on a `debug` build against `release` baselines. That
  understates the change rather than flattering it, but it is not a comparison; everything
  quoted above is release-on-release.

### Hazard 5 is the one part not carried out, on purpose

`_key_heap_bytes`, `bytes_per_determinant` and `ManyBodyState.memory_bytes` all still model
one heap block per key. Both now **over-state**, which is the safe direction -- smaller caps,
and a Krylov-store guard that declines earlier -- and both are annotated at their definitions
saying so. Correcting them is tier 3 by effect: `bytes_per_determinant` feeds RAM-derived
`truncation_threshold` defaults, so it changes which determinants are admitted and therefore
energies on capped runs, and `memory_bytes` gates the recycled Krylov store at
`gf_shift_recycling.py:497`, so correcting it loosens that guard. Both want the Phase-3
error-budget harness, which does not exist yet.

### Still open

- ~18 Cython `key(r)` sites in `_block_state.pxi`, `_mpi_pack.pxi` and `_krylov_store.pxi`
  still materialize a determinant per call. They are correct and they compile because `key(r)`
  returns by value; converting them is where the object path's remaining overhead lives.
- `SlaterDeterminant` remains an owning value everywhere outside the block (shape (a) of the
  two the design offered). The full owning/view split was not needed to clear the bar.
