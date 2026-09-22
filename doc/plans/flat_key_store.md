# Flat key store: one `vector<uint64_t>` per block, `SlaterDeterminant` as a view

**Status: designed, deliberately NOT implemented.** This is a major change to the storage
substrate and is deferred. Everything below is the case for it and the hazards it has to clear,
recorded while the measurements that motivate it are fresh.

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

## What the 2026-09-21 campaign round adds to the case

Three measurements from `memory_footprint_audit.md` that were not available when this was
written. None of them changes the design; two strengthen it and one is a caution.

**1. The vector-of-vectors cost more than 52.3 B/det — it also over-allocated by `width`.**
P1-10 measured `m_keys` on a 40k-row block built by `from_states`: 0.92 MiB of live keys
against **48.00 MiB of allocated key slots at width 32** (52.4x), because the support is built
by pushing `width * rows` keys and then `erase`-ing, which keeps the capacity. A
`shrink_to_fit` fixed it. But the *class* of defect exists only because `m_keys` is a growable
vector of heap-owning vectors: a flat `N x n_chunks` buffer sized once from a known row count
cannot have it. This is the second defect in this campaign whose root cause is the same
structure.

**2. Building the block is expensive in a way the per-determinant slope does not show.** P1-9
compared three representations for a retained basis snapshot, on VmHWM:

    list(local_basis)             80.2 B/det
    ManyBodyState.from_keys(...) 115.2 B/det   <- the C++ key block, WORSE
    packed (n, n_bytes) array      9.8 B/det

The key block loses to the Python list it was supposed to replace, because `from_keys` builds
a `vector<SlaterDeterminant>` — N separate mallocs — and copies it into the block, and glibc
does not return the freed transient. A flat store writes the buffer directly and has no such
intermediate. **The retained slope is not the whole cost of this representation; the
construction path is the other half**, and it is the half that sets VmHWM.

**3. The instrument now exists.** `ManyBodyBlockState::row_capacity`/`amp_capacity`, exposed as
`ManyBodyState.capacity_stats()`, report allocated against logical size. RSS cannot see vector
capacity — shrinking returns nothing to the allocator — so before this there was no way to
measure the flat store's win on the axis where it is largest. The confirmation bar below should
gain a row: `capacity_stats()` on a freshly built block must show allocated == logical, which
the flat store gets by construction rather than by remembering to shrink.

The caution: **the speed claim is still the one that decides this.** The memory case was
already demoted at 0b (the key representation is a falling share as `nso` grows, and low single
digits at production block widths), and nothing above changes that. What the round adds is
evidence about *construction* and *capacity*, not about the per-row bytes. The bar stays what
it was — beat 358 ns/lookup at N = 1M — and it is not met by arguing.

## Scope note

This is a substrate change touching `SlaterDeterminant.h`, `ManyBodyBlockState.h`,
`ManyBodyOperator.cpp`, `MpiUtils.cpp` and every `.pxi` that constructs a determinant. It should
be staged behind its own equivalence tests and land on its own, not bundled with anything else.
