# Memory-footprint audit: raising the system-size ceiling

Campaign plan: `~/.claude/plans/reflective-weaving-platypus.md`.
Predecessor: `dc_smo_memory.md` (9 rounds, SrMnO3 double counting, peak-RSS ledgers).

## Why this is not that campaign

`dc_smo_memory.md` ran **two full-repo memory surveys, ~40 candidate sites, none ranked**; its
step 6 measured the four best-argued and none set the peak. Four predicted fixes were each exact,
each standalone ~2x better, and each moved the process peak by 0.03%.

This campaign ranks by **scaling class**, not by today's measured peak, because the objective is
the reachable system size, not the SrMnO3 high-water mark. The ranking rule, the finding schema
and the phase structure are in the plan file. One rule is restated here because it is the rule
that nearly deleted this campaign's headline finding:

> A distributed `O(N_local)` term is "buy more nodes" **only when its constant is
> near-irreducible.** The per-node ceiling is `bytes_per_det x dets_per_rank x ranks_per_node`,
> so a reducible constant factor on the dominant distributed term is itself a ceiling-setter.

## Phase 0 results

> **Which numbers are authoritative.** Sections 0a, 0e, 0f and 0g were written as the work
> happened and quote per-determinant figures measured with **three N values in one process**.
> Section 0a itself later established that this method is history-dependent (the same sweep
> re-run gives 166.7 -> 191.8 -> 198.6 B/det with physically impossible negative intercepts),
> so those figures are internally comparable but **not** trustworthy in absolute terms. The
> authoritative per-determinant numbers are the cold-process marginal slopes in **0h**:
> 172.8 (dict) -> 88.2 (bisect) -> 52.3 (C++ key block) B/det. Where an earlier section says
> 166.7 or 82.1, read 172.8 and 88.2. The *ratios and conclusions* in those sections are
> unchanged; only the absolute constants moved.

### 0a. The per-determinant price — measured, then corrected by adversarial review

**Final numbers.** Per *local* determinant, nso = 124, `RssAnon` slopes:

| | B/local-det | note |
|---|---|---|
| `Basis`, **distributed** (`comm.size` 2 and 4) | **~285** | identical at 2 and 4 ranks, so structural, not an MPI-init artifact |
| `Basis`, serial (`COMM_SELF`) | ~172-192 | the branch production never takes |
| `ManyBodyState` width 1 (C++ block) | **72.0** | matches `bytes_per_determinant(124)` exactly |

**The production-relevant ratio is therefore ~4x, not the 2.3x first reported** — the first
measurement understated the finding.

**Three defects in the first measurement, found by adversarial review, all real:**

1. **Wrong branch.** The probe built its `Basis` with `comm=MPI.COMM_SELF`, exercising the
   non-distributed path (`manybody_basis.py:278-289`). Every production run is `comm.size > 1`
   and takes `:291-329`, which adds O(N) transients (`list(set(...))`, `all_received`,
   `sorted(set(...))`) plus `distribute_determinants`. Measured under weak scaling (50k dets per
   rank, disjoint candidates): 192 B/local-det serial against **285 at both 2 and 4 ranks**.
2. **The slope is history-dependent.** Re-running the same 3-point sweep in one process gave
   166.7 -> 191.8 -> 198.6 B/det with intercepts +0.97, **-3.72, -5.24 MB**. A negative intercept
   is physically meaningless; it arises because small-N points get cheaper as the allocator
   reuses pages freed by the previous pass. Only the never-before-touched high-water point
   (N=200k) is stable, at ~172.2 across all passes. **So `166.7 +- 15%` was one realization of an
   unstable fit, and its third digit and its intercept were never defensible.**
3. **`malloc_trim` cannot reach the allocator that holds the transients.** The O(N) temporaries in
   `add_states` are `SlaterDeterminant` cdef instances from CPython's **pymalloc**, not glibc.
   `malloc_trim` releases only the glibc arena; pymalloc returns an arena to the OS only when it
   is 100% empty. Measured: building and freeing one 100k-determinant `Basis` leaves a
   **permanent +3.9 MB (39 B/det) residue** that five settle-and-repeat cycles never reclaim. The
   probe's `settle()` does not restore a clean baseline, which is precisely what defeats the
   small-N points in defect 2.

This is the same mechanism my own instrument control had already bounded (exact for numpy/C++
heap, ±10-30% for Python-object-heavy allocations) — the review identified *why*, and that the
error is systematic rather than random.

**The reachable-reduction figure is a projection and is now labelled as one.** No contiguous
fixed-width key store has been built or benchmarked here; "~3x-7x" was arithmetic from assumed
constants, printed next to measured slopes as though equally solid. What *is* verified is the
structural premise: `_index_dict`'s value is exactly `offset + array_position` in both branches
(`manybody_basis.py:286,317`), so the dict is genuinely redundant with a sorted array plus binary
search. The magnitude against the distributed baseline is **unmeasured** and could be larger or
smaller than the projection.

**Secondary, minor:** incremental growth (many `add_states` calls, as real expansion does) costs
~5-10% more per determinant than one bulk call (182.3 -> 200.8 -> 189.1 B/det for 1 / 10 / 50
chunks at N=100k).

**A correction to this section's first version, kept because the error is instructive.** The
first probe priced a `list` of `bytes` plus a `{bytes: int}` dict and reported 149.6 B/det. But
`add_states` converts every input via `self.type.from_bytes(...)` (`manybody_basis.py:277`), so
`local_basis` holds **`SlaterDeterminant` wrapper objects, not `bytes`** — the mock measured the
wrong container. It also reported a single-N ratio rather than a slope. Both failures are ones
this campaign's own plan warns about, and the probe now asserts its own row count.

**What the campaign takes from this.** The qualitative finding is unchanged and strengthened: the
Python-side `Basis` costs several times the C++ block per determinant, and its index is
structurally redundant. Every *number* attached to it moved. Any future per-determinant figure in
this campaign must state its branch (distributed or serial), be taken on a cold process, and
carry no more precision than a pymalloc-bounded instrument supports.

### 0b. CLOSED — the determinant-key representation does not scale with system size

The obvious framing of `SlaterDeterminant` inheriting `std::vector<uint64_t>` (one malloc per
determinant, `SlaterDeterminant.h:20-21`) is that it is the big structural lever. It is not, and
the repo's own model says so. `_key_heap_bytes` (`memory_estimate.py:235-239`) is
**flat at 32 B from 64 through 192 spin-orbitals**, 48 B at 256:

**Independently cross-checked**: real `ManyBodyState` objects built at nso in {64,124,192,256,512}
measured the predicted slope at **every point, ratio 1.000** (72/72/72/88/120). The formula is
empirically exact across the claimed range, not merely self-consistent.

| nso | chunks | key_heap | B/det **at block width 1** | payload | overhead |
|---|---|---|---|---|---|
| 64 | 1 | 32 | 72 | 24 | 66.7% |
| 124 | 2 | 32 | 72 | 32 | 55.6% |
| 192 | 3 | 32 | 72 | 40 | 44.4% |
| 256 | 4 | 48 | 88 | 48 | 45.5% |
| 512 | 8 | 80 | 120 | 80 | 33.3% |

The overhead **falls** as spin-orbitals grow — the 24 B vector header plus 16-byte malloc
rounding dominate a small payload, and the payload is what grows. So this term does not scale
with the parameter the campaign exists to raise, and the whole C++ key representation is ~22% of
per-determinant basis cost.

**Two caveats on the table.** (a) Every percentage above is **conditioned on block width
`p = 1`**; `bytes_per_determinant()` takes no width argument, but the real per-row cost is
`16p + key_heap + 24` (measured 90/114/317 B/det at p = 1/4/16). At production widths — 105 to
315 at the ~1M cap — the amplitude term dwarfs the 32-80 B key heap and the key representation's
share falls to low single digits, which *reinforces* the demotion. (b) `_FLAT_MAP_ENTRY_BYTES`'s
comment (`memory_estimate.py:52-53`) is stale: `ManyBodyState` is backed by two separate vectors
(`m_keys`, `m_amps`), not a flat_map of pairs. The arithmetic coincides with the old model only
at width 1 — correct by coincidence, not because the described structure still exists.

**Not closed outright, but demoted and re-scoped**: its surviving claim is replication across
containers (`m_keys`, masks, caches, `SparseKrylovDense`'s *double* key storage) and allocator
fragmentation from millions of tiny blocks — not the per-row bytes. It is ranked below the
Python-side store and is gated on that claim being priced separately.

### 0c. CLOSED — `build_vector` / `build_dense_matrix` are not production findings

Both live behind `dense = len(basis) < 500` (`gf_solvers.py:184,185,211`). `Basis.__len__`
returns `self.size` (`manybody_basis.py:544-552`), which the distributed branch sets from an
`allgather` sum (`:311-313`), so it is the **global** count — verified, not assumed. A production basis is far larger than 500 — the documented real workloads put FCC Ni's GF
excited basis at ~3k determinants, not the "millions" an earlier draft of this section claimed —
so the branch is never taken and the allocation is bounded where it is. Additionally,
`block_Green_sparse` (`gf_solvers.py:314`), the production self-energy path, is a **separate
function that never calls these at all**; it goes through `_CappedBasisProxy` directly. The gated
`block_Green` *is* reached in production (`greens_function.py:1235`, `rixs.py:795,950`), but every
other caller of `build_vector`/`build_dense_matrix` in the repo is a test. Recorded as non-findings rather than ranked, per
the plan's close-out rule. (`basis_transcription.build_vector`'s shape is unchanged and its
contract is what the tests assert — the fix, historically, belonged at call sites.)

### 0d. The `p` sweep — the ordering reproduces locally, on the growth column only

The pre-registration asked whether a local rig reproduces the 128-rank ordering of the three
CIPSI sites. Reaching `p ~ 320` locally turned out to be time-bound, not memory-bound (the solve
cost grows sharply with the thermal window), so the experiment was reframed to test the
*mechanism*: sweep `p` via `tau` at fixed cap and watch whether the ordering moves. One cold
process per point. SMO cubic, occ 3, cap 10,000, 1 rank:

| tau | `p` | apply growth | overlaps growth | score growth | ABS ordering |
|---|---|---|---|---|---|
| 0.0025 | 4 | **5.7 MiB** | 2.1 | 1.2 | flat (floor-dominated) |
| 0.01 | 4 | **9.2 MiB** | 1.6 | 1.3 | flat (floor-dominated) |
| 0.025 | 32 | **149.2 MiB** | 39.7 | 35.2 | score 559.9 > apply 528.2 > overlaps 524.7 |

**Result: the growth column reproduces the 128-rank ordering
(`_apply_block_and_redistribute` > `_candidate_overlaps_and_energies` > `_score_candidates`) at
every `p` measured, while the ABS column does not** — at `p = 32` the ABS ordering puts
`_score_candidates` first, which is *not* the production ordering.

**An incidental observation worth more than the point it came from.** The `tau = 0.04` point
never finished: it was still running after ~12 minutes at **5.7 GB RSS** on a basis capped at
**10,000 determinants**, and was killed. Against the `tau = 0.025` point (`p = 32`, peak ~560 MiB)
that is roughly 10x the memory for ~3x the manifold width, on a basis one-thousandth the size of
production. Treat the 5.7 GB as a *lower bound* on that point's peak, not a measurement — the run
was killed mid-flight. But the direction is unambiguous and it is the campaign's thesis in one
data point: **at fixed determinant count, memory is set by `p`, and it grows faster than
linearly.** `p` is set by the thermal window, which is a convergence parameter — the user's
opening observation that tightening convergence inflates memory, showing up on the `p` axis
rather than the basis-size axis.

This is a direct local confirmation of the predecessor campaign's hardest-won method rule: **rank
on growth, not on absolute peak.** At small `p` the ABS column is pure floor (~380 MiB of
interpreter and imports) and carries no signal at all; by `p = 32` it carries signal, and the
signal is *wrong*. Per the plan's asymmetry rule a match is weak evidence — 3 ranks have neither
the routing skew nor a rank-local `n_Dj` — so this licenses local *mechanism* work, not
cluster-scale claims.

### 0e. Sizing the first `Basis` increment: the index dict is 41% of it

*(Figures below use the superseded one-process method; see the banner. The 41% share is a ratio
within one measurement and is unaffected.)*

Measured on a real `Basis`, cold process, N = 200,000, nso = 124:

| component | B/det | share |
|---|---|---|
| `Basis` total | 177.6 | 100% |
| **`_index_dict`** | **73.1** | **41.2%** |
| remainder (`local_basis` list + `SlaterDeterminant` objects) | 104.5 | 58.8% |

`_index_dict`'s value is exactly `offset + array_position` and `local_basis` is already sorted, so
the dict is fully derivable by binary search. **Dropping it alone cuts `Basis` memory by 41% with
no change to `local_basis`'s representation** — so no call-site churn for the ~10 production files
that iterate or index it, and no Cython work. The remaining 104.5 B/det needs the contiguous
fixed-width key array, which is the invasive change.

That splits the headline finding into two increments of very different risk:

* **Step A** — delete `_index_dict`, derive the index by binary search. -41% of `Basis`. Tier 1.
  Requires the miss-sentinel fix (see below) and a lookup benchmark, since `build_sparse_matrix`
  does O(nnz) lookups that move from O(1) to O(log n).
* **Step B** — contiguous key store replaces the list and the wrapper objects. Attacks the
  remaining 58.8%. Invasive; deferred behind Step A's benchmark.

### 0f. Step A's price, measured before shipping it

Dropping `_index_dict` moves every index lookup from an O(1) dict to an O(log n) Python-level
`bisect_left`, whose comparisons are `SlaterDeterminant.__lt__` calls. Measured directly over
sorted determinants (200k probes each):

| N | dict | bisect | ratio |
|---|---|---|---|
| 10,000 | 67 ns | 321 ns | 4.8x |
| 100,000 | 429 ns | 661 ns | 1.5x |
| 1,000,000 | 622 ns | 1,785 ns | **2.9x** |

`build_sparse_matrix` performs one lookup per stored nonzero, so this is the price on the hottest
assembly path. Two mitigations are already in the change: the column index was being *looked up*
(`_index_dict[ket]`) while `ket` walks `local_basis` in order — it is simply `offset + position`,
so those N lookups are now an enumerate and cost nothing; and the distributed branch already
routed rows through `_index_sequence`. But the per-nonzero row lookups remain, and with fan-out
~40 they dominate the N column lookups removed.

**So Step A as implemented trades ~41% of `Basis` memory for a real slowdown in matrix assembly.**
The predecessor campaign measured that build at 4.5% of CIPSI time serial and 13.5% at `-n 2`, so
the wall-clock cost is bounded but not negligible.

**What this argues.** The C++ layer does exactly this lookup with `std::lower_bound` over a
contiguous key array (`ManyBodyBlockState.h:259-265`), without per-comparison Python call
overhead. So the measurement is less an argument against dropping the dict than an argument that
**the lookup belongs in C++ — i.e. Step B — rather than in Python `bisect`**. Step A should not
ship standalone on a workload where matrix assembly matters until that trade is accepted
deliberately; it is reported here as a measured trade, not a recommendation.

### 0g. Step A: full gate green (per-determinant figures superseded by 0h)

Same probe, same geometry (real `Basis`, nso=124, slope over N in {50k,100k,200k}, cold process),
before and after removing `_index_dict`:

| | B/det | vs C++ block |
|---|---|---|
| `Basis` before | 166.7 | 2.31x |
| **`Basis` after** | **82.1** | **1.14x** |
| `ManyBodyState` width 1 (C++) | 72.0 | 1.00x |

**A 50.7% reduction — better than the 41.2% the dict-share measurement predicted.** The
difference is itself informative: `dict_share.py` measured the dict's marginal cost by emptying it
*after* construction, which captures the table and its `int` values but not the allocator pressure
its construction leaves behind. The predicted number was a lower bound.

The Python-side `Basis` now costs **10 B/det more than the C++ block it indexes**, against 95 B/det
more before. What remains is the `local_basis` list and its `SlaterDeterminant` wrapper objects —
Step B's target.

**Caveat, and it is the same one that caught this campaign out already:** this is the *serial*
branch. The distributed branch measured ~285 B/det before the change and has **not** been
re-measured after it. The distributed path has extra O(N) transients of its own, so the saving
there is expected but unverified.

**Test gate, run on an untouched tree after the change** (an earlier gate run was invalidated by
editing the tree underneath it — the failure mode `CLAUDE.md` warns about, and it cost a full
re-run):

| leg | result |
|---|---|
| serial | 2361 passed |
| `mpiexec -n 1` | 2610 passed |
| `mpiexec -n 2` | 2626 passed |
| `mpiexec -n 3` | 2631 passed (ranks 1 and 2: 2630 each, checked in `.pytest_mpi_rank*.out`) |

black clean at line length 120.

**What shipped in this change**, beyond the dict removal itself:
* `col = _index_dict[ket]` in `build_sparse_matrix` was looking up an answer the loop already had
  — `ket` walks `local_basis` in order, so the column is `offset + position`. Those N lookups are
  gone entirely (a pure win, independent of the dict question).
* Two `state not in basis.local_basis` O(n) list scans replaced with `contains_local`
  (`gf_shift_recycling.py`, `gf_solvers.py`) — the exact anti-pattern `contains_local`'s own
  docstring warns about.
* Determinant width normalization in `Basis`, which immediately caught a test feeding a 64-byte
  determinant into a 1-byte-wide basis (see 0-defects above).

### 0h. Step B: the key store moved into C++, and what the lookup benchmark actually said

`Basis` now stores its determinants in a **width-0 `ManyBodyState`** — a sorted C++ key vector —
instead of a Python list of `SlaterDeterminant` wrapper objects. `local_basis` became a
non-materializing sequence view (`_LocalBasisView`): `len`, `in` and `[i]` go straight to the
block via new `find_row` / `key_at` accessors, and only iteration materializes, because a caller
that iterates is asking for the objects themselves.

**Memory, re-measured properly.** All three versions, one cold process per point, marginal slope
over N = 200k -> 400k (the earlier 166.7 / 82.1 figures came from three points in *one* process,
which this campaign had already shown to be history-dependent — they were internally comparable
but not trustworthy in absolute terms):

| version | B/det (marginal) | |
|---|---|---|
| `_index_dict` (original) | **172.8** | baseline |
| Python `bisect` | 88.2 | -49% |
| **C++ key block** | **52.3** | **-70% overall, 3.3x** |

The `Basis` is now *below* the 72 B/det of the `ManyBodyState` block it indexes — as it should be,
since it stores keys and no amplitudes.

**The lookup benchmark refused to cooperate, and that is the interesting part.**

| N | dict | Python bisect | C++ `find_row` |
|---|---|---|---|
| 10,000 | 80 ns | 269 | 181 |
| 100,000 | 438 ns | 521 | 479 |
| 1,000,000 | 594 ns | 1,378 | 1,123 |

Moving the search into C++ bought only **1.1-1.5x over Python `bisect`, and it remains slower than
the dict it replaced**. So the premise behind "do the lookup in C++" was wrong: at one lookup per
call the cost is dominated by the **Python-to-Cython call boundary and argument conversion**, not
by the comparison loop. `std::lower_bound` was never the bottleneck.

**Batching helps and still does not close the gap.** `find_rows(keys)` runs the whole loop inside
the extension and is wired into both `_index_sequence` branches:

| N | dict | bisect | `find_row` | **`find_rows` batched** |
|---|---|---|---|---|
| 10,000 | 82 ns | 255 | 183 | **149** |
| 100,000 | 183 ns | 449 | 427 | **334** |
| 1,000,000 | 358 ns | 1,150 | 1,059 | **830** |

Batching buys ~1.3x over per-call, and the result is still **~2.3x slower than the dict**.

**The diagnosis, and it turns on the structure this campaign earlier demoted.** `m_keys` is a
`std::vector<Key>` where every `Key` is itself a heap `std::vector<uint64_t>`
(`SlaterDeterminant.h:20-21`). So a binary search over N determinants dereferences ~log2(N)
*separate malloc blocks* scattered across the heap, while a dict hashes once and touches one
bucket. The search is cache-hostile by construction.

Phase 0b demoted the per-key heap allocation as a **memory** lever, correctly: it is flat in
`n_orb` and shrinks as a share of cost. It is, however, a first-order **speed** liability for
ordered lookup — and a genuinely contiguous fixed-width key array (`n_chunks * N` in one
allocation) would fix both at once: ~16 B/det instead of 52, and a binary search that walks one
buffer instead of chasing pointers. That is the finding this phase actually produced, and it is
now the best-supported candidate for further work.

This is the fifth time in this campaign's history that a predicted mechanism turned out not to be
the mechanism. The memory result stands on its own measurement; the speed result is reported as
measured, including the part that did not work.

### 0i. The peak was the number that mattered, and the storage work had been aimed at the other one

Everything up to 0h measured **retained** RSS. The cap model, the memory guard and the OOM killer
all act on **VmHWM**. Measured properly, the storage work looked very different:

| | retained (marginal) | peak VmHWM (marginal) |
|---|---|---|
| original (`_index_dict`) | 172.8 | 250.5 |
| C++ key block as first committed | 52.3 | **284.4** — *worse* |
| + `ManyBodyState.from_keys` | 52.3 | 233.1 |

The first version built the block through `dict.fromkeys(states, ())`, an N-entry throwaway dict,
so a change that cut retained cost 3.3x made the peak 13.5% **worse than the code it replaced** —
and that was invisible in the retained column, which is identical either way.

**Rule this earns: measure a storage change on the metric its guard uses, not the one that
flatters it.** This caught two of this campaign's own commits within an hour.

### 0j. ...and then the probe measured the wrong scenario, which inverted the verdict again

With the peak identified, ~180 B/det of it was construction transient in `add_states`: a
`list(merge(self.local_basis, unique_new))` that — now `local_basis` is a view — materialized the
*entire existing basis* as Python objects on every call, plus a sortedness assert that allocated
2N wrapper objects per call while guarding a property the C++ block cannot violate. Replacing the
merge with `ManyBodyBlockState::merge_keys` (in-place, `nogil`, O(n+m)) and dropping the assert
measured **worse** on the one-shot probe: 233.1 -> 248.6 B/det peak.

That probe builds a basis in a single `add_states` from empty — a case production never runs.
CIPSI *grows* a basis over many cycles, and the whole point of `merge_keys` is not having to
rebuild what is already there. Measured on that pattern instead (N determinants added in 10
calls):

| | peak VmHWM | steady |
|---|---|---|
| `list(merge(...))` | 282.5 B/det | 244.8 |
| **`merge_keys`** | **66.3 B/det** | **50.6** |

**4.3x lower peak**, and the one-shot regression (233 -> 249) is real but confined to a shape
production does not use.

This is the second time in one session that the *scenario* rather than the instrument decided the
answer, and it is the campaign's oldest recorded lesson — "measure the block you are modelling" —
re-earned. Both probes were correct; one of them was answering a question nobody asked.

## The ranked table

Ranked lexicographically by the plan's keys — replicated-per-rank > superlinear > grows with
`de2_min` > never-evicted > simultaneous copies — with bytes at the stated geometry as the
tiebreak *within* a class. Two filters run first: a distributed `O(N_local)` term with no
reducible constant is dropped rather than ranked, and a row whose absolute peak is out of reach
of the process high-water mark is marked **blocked**.

Sections are numbered in the order they were written, which is not rank order and by now is not
even numeric order in the file (`P1-3` was used twice). This table is the index; the numbers
below are anchors, not a ranking.

| # | finding | class | tier | at the stated geometry | state |
|---|---|---|---|---|---|
| [P1-6](#p1-6) | `cartan_subalgebra`'s never-read left-singular block | superlinear, `O(n^4)`, replicated | 1 | **OOM-killed at n=150** -> 623.7 MiB | FIXED |
| [P1-7](#p1-7) | the fused unpack held the amplitudes twice | simultaneous copies on the peak-setting path | 1 | 517.4 -> 224.0 MiB at width 320 | FIXED |
| [P1-4](#p1-4) | `component_symmetry_reduction`'s residual block | superlinear, replicated | 2 | 1950.9 -> 244.9 MiB at n_orb=151 | FIXED |
| [P1-5](#p1-5) | `compute_impurity_rdm`'s guard stage and entry packing | replicated block, `2^n_imp` | 2 | 462.6 -> 0.0 (refused), 420.1 -> 142.1 MiB | FIXED |
| [P1-1](#p1-1) | `build_sparse_matrix` grew with rank count | anti-scaling with `comm.size` | 1 | 810.4 -> 414.8 MiB at 2 ranks | FIXED |
| [P1-2](#p1-2) | every walk of `local_basis` materialized it | constant on the dominant distributed term | 1 | 83.2 -> 0.0 B/det | FIXED |
| [P1-8](#p1-8) | the basis split's wire payloads | simultaneous copies, `x n_colors` | 1 | 158.3 -> 16.1 B/entry | FIXED |
| [P1-9](#p1-9) | `best_basis` doubled the basis | simultaneous copies, retained across cycles | 1 | 80.2 -> 9.8 B/det | FIXED |
| — | the `Basis` determinant store (Phase 0e-0j) | constant on the dominant distributed term | 1 + 3 | 282.5 -> 66.3 B/det | FIXED |
| — | replicated `ManyBodyOperator` + flat caches | replicated-per-rank | — | 8.3 MiB/rank at n_orb=124 | **blocked** (0.2% of peak) |
| — | `SectorResolventCache._index` | never-evicted | — | 0.06% of the `(N,N)` it sits beside | **blocked** |
| — | `ManyBodyOperator::apply`'s `num_threads^2` accumulators | replicated per thread | 1 | 1148 -> 596-685 MiB at width 320 | FIXED |
| — | no `shrink_to_fit` anywhere in the C++ layer | capacity slack, monotone | — | unmeasured | open, needs a capacity accessor |
| — | the split's *construction* transient (the replication itself is counted — claim refuted) | simultaneous copies | — | unmeasured | open |
| — | path C (double counting) | — | — | scalars only; no array in 5,333 lines | **no findings** (read) |

**An `unmeasured` row may never outrank a measured one**, which is why the four open rows sit
below every fixed one regardless of their class.

## The promotion chain, written as a prediction

The record's one reliable predictive pattern is that cutting the peak-setter promotes a
previously-refuted knob or a previously-invisible site. It has held three times. **Writing the
successor down before the next measurement is the only version of this that is a prediction**,
and the five fixes shipped on 2026-09-21 were all landed without one — so these are registered
now, before the cluster round, and the round either confirms them or does not.

1. **P1-7 cut `unpack_block_fused` 2.3x at production width.** `redistribute_block` is what the
   128-rank ledger named as the CIPSI peak-setter, so the peak should now move *within* the
   apply round trip rather than out of it — most likely to `pack_block_fused`'s send buffer,
   which is `total x bpe` and still built in full before the exchange. Prediction: at 128 ranks
   the apply remains the top site, and its residue is dominated by the send buffer plus the
   received buffer being alive simultaneously.
2. **`GS_SELECTION_CHUNK` becomes live for the third time.** It was refuted twice because the
   apply, not the selection stack, set the peak. P1-7 cut the apply. Prediction: it measures
   non-zero at 128 ranks now, and the correct response is still to re-measure rather than to
   assume — the first two refutations were also predictions.
3. **P1-2 and P1-9 removed Python-object materialization from the ground-state path.** What is
   left there is the `psi_refs` block itself (`p x N_local x 16`), which no packing can shrink.
   Prediction: the ground-state path's residue is now genuinely `p`-bound, so it responds to
   `GS_MAX_BLOCK_WIDTH` and to nothing else.

A prediction that fails is the useful outcome here; it is recorded so that it *can* fail.

## Phase 1 findings

### P1-1. `build_sparse_matrix`'s distributed branch grows with rank count

Measured, VmHWM, one cold process per point, synthetic basis closed under a hopping
Hamiltonian (see caveats):

| | 1 rank, N ~393k | 2 ranks, N ~484k |
|---|---|---|
| `build_local_operator_list` | 230.7 MiB | **136.8 MiB** |
| `build_sparse_matrix` | 291.2 MiB | **810.4 MiB** |

The image build halves from 1 to 2 ranks, which is what a distributed structure should do. The
matrix build nearly **triples**, adding ~670 MiB on top of a 137 MiB image. It is called from
`get_eigenvectors` (`cipsi_solver.py:1790,1961`), i.e. **once per CIPSI cycle**, plus once per
eigenstate-doubling retry.

**Why.** The distributed branch (`basis_transcription.py:198-215`) keeps `columns`, `bras` and
`values` (each nnz-long, `bras` holding nnz *boxed* `SlaterDeterminant` objects), then builds
`global_rows` (nnz) while those are still alive, then `rows`, `cols`, `vals` (three more nnz
lists) while all four are still alive — up to **seven nnz-sized Python containers concurrently**,
none deleted before the function returns. `basis._index_sequence(bras)` adds its own `list(s)`
copy, a `comm.size`-way send bucketing and a result array on top, which is why the term grows
with rank count rather than shrinking.

It also reads the applied image through `.items()` (`:192,204`), allocating a `SlaterDeterminant`
and a `Row` per (row, column) pair. The CIPSI selection round eliminated exactly this by reading
through the buffer protocol instead, and measured it at **33-43% of a selection round**
(`cipsi_solver.py:784-789`). `build_sparse_matrix` was never migrated.

**Caveats, stated because the absolute numbers are not production values.** The probe's operator
gives a fan-out of ~6.9 rows per determinant against a production figure nearer 40, and the basis
is the closure of a random determinant set rather than a physical charge sector. So the *shape*
(seven concurrent nnz containers; growth with rank count) is the finding; the MiB figures
indicate scale, and a production-geometry measurement belongs in the cluster handover.

**A probe bug worth recording**, because it is the failure mode this campaign keeps hitting: the
first version of this measurement reported `nnz = 0` and a meaningless B/nnz, because a random
determinant set is not closed under hopping and `build_sparse_matrix` drops every bra that is not
in the basis. It measured an empty matrix and would have said nothing had the derived figure not
come out absurd. The probe now asserts the closure added determinants.

### P1-3. `component_symmetry_reduction` held its residual block twice — FIXED

The top-ranked finding of Phase 1 by the campaign's own rule: **replicated per rank**,
**superlinear in `n_orb`**, and **unconditionally live** on the default XAS path
(`spectra.py:429-430`, reached from `scripts/run_cmd.py` -> `get_spectra.run_spectra` ->
`spectra.simulate_spectra`, with no rank guard anywhere on the chain).

`lie_algebra.component_symmetry_reduction` built its residual block as a Python list of
per-generator column vectors and then did `np.array(columns).T`, so the whole `(m n_orb^2, n_gen)`
block existed **twice at once**. With `m = 3` (the Cartesian components) and `n_gen ~ n_orb` for a
generic non-degenerate spectrum, that block is O(n_orb^3). Measured, VmHWM, cold process:

| n_orb | before | after |
|---|---|---|
| 62 | 23.1 MiB | 12.4 MiB |
| 124 | **177.6 MiB** | **90.8 MiB** |

Doubling `n_orb` costs 7.7x, confirming the cubic scaling. Filling the array in place removes the
duplicate; the resulting matrix is **bit-identical** (verified by array equality against the old
spelling at two sizes), so this is tier 1. 127 symmetry tests pass.

Per rank, so it multiplies by ranks per node: ~87 MiB/rank of pure duplicate removed at 124
spin-orbitals, and the term itself grows as `n_orb^3` — at 192 spin-orbitals the block is ~660
MiB/rank even after this fix. **The duplicate is fixed; the cubic term is not**, and it remains
the largest known replicated-per-rank allocation on a production spectra path.

This site already had memory history: its own comment records that the SVD below it once
materialized a `(m n^2)^2` left-singular block, *"~21 GiB at n = 112"*, fixed by switching to an
economy SVD. The duplicate above it survived that pass.

**A note on how this was found, because it corrects an earlier claim in this campaign.** An
earlier message in this session flagged `lie_algebra`'s `(n_orb,)**4` tensor as a top finding and
then retracted it: every production caller passes `two_body=False`, and the parameter named
`n_orb` is bound to the *impurity* count at the only `two_body=True` caller. That retraction was
correct. The module was nonetheless the right place to look — just for a different object, three
functions away. Retracting a wrong reason is not the same as clearing the area.

### P1-4. `component_symmetry_reduction`'s residual block is no longer formed — FIXED

The follow-on to P1-3, which removed the *duplicate* and left the cubic term standing. Two
corrections to what P1-3 recorded, both from measurement rather than argument:

**`n_gen` is not `~n_orb`.** P1-3 assumed one generator per non-degenerate eigenvalue.
`discover_one_body_symmetries` actually returns one per *pair* within each degenerate group, so
`n_gen = sum over groups of (group size)^2`. Measured on the real archive `h_solver` matrices
(`test/support/restriction_diagnostics.WORKLOADS`):

| workload | n_orb | groups | max group | n_gen | n_gen / n_orb |
|---|---|---|---|---|---|
| `fcc_ni_5` | 59 | 23 | 3 | 157 | 2.7 |
| `fcc_ni_15` | 145 | 56 | 3 | 389 | 2.7 |

**The peak was understated by ~3x.** P1-3 extrapolated ~660 MiB/rank at 192 spin-orbitals.
Measured at XAS shape (the archive `h_solver` under a degenerate core-p block, the layout
`spectra.py:428-430` builds), VmHWM growth in a cold process, both sides through the same probe:

| n_orb | n_gen | before | after | wall time |
|---|---|---|---|---|
| 65 | 193 | 170.5 MiB | **80.2 MiB** | 0.40 s -> 0.20 s |
| 151 | 425 | **1950.9 MiB** | **244.9 MiB** (-87%) | 8.17 s -> 3.95 s |

1951 MiB/rank, replicated, is the largest single allocation this campaign has measured. The
`(m n_orb^2, n_gen)` block is 443.6 MiB of it; the rest is LAPACK's left-singular block of the
same shape (never read) plus `gesdd` workspace.

*The first version of this table read 157.0 -> 66.8 and 1800.2 -> 298.8, and was wrong in the
same direction on both sides.* The probe reported `n_gen` by calling
`discover_one_body_symmetries` **before** taking its baseline, so the heap had already absorbed
one generator list and every growth figure was understated. It was caught by the end-to-end
number coming out smaller than one of its own stages -- an arithmetic impossibility that a
plausible-looking figure would have hidden. `n_gen` is now read off after the measurement.

**The fix.** Only the *null space* of the residual block `B` is ever used, and `B = QR` gives
`B^dagger B = R^dagger R` — so `R`, which is `(n_gen, n_gen)`, carries the same singular values
and the same right singular vectors. `_residual_r_factor` streams `B` one row block at a time
into `R` and never holds the whole thing. Two passes, because a row block needs every column
while the projection needs whole columns: pass 1 takes the projection coefficients through the
trace identity `<q_j, [C,T]> = tr([T, q_j^dagger] C)`, which drops an `O(n^3)` commutator per
generator to an `O(n^2)` contraction — which is why the function also got *faster* rather than
paying the usual streaming tax.

**Tier 2, not tier 1, and this is a correction to how P1-3 framed the area.** The null-space
basis is arbitrary, so `R` returns a different — equally valid — basis than the SVD of `B` did,
and `Q` changes (measured `||Q_old - Q_new|| ~ 2.8`). Bit-identity, which is what P1-3 earned by
array equality, is *not* available here and was not claimed. What was verified instead, against
the pre-change code from `git HEAD` on both synthetic and real geometries:

- identical `diagonalizable` flag and identical set *partition* of components into groups
  (compared as a partition, not as labels — the labelling may permute);
- `Q` unitary to 1e-10;
- **cross-reconstruction both ways**: a symmetry-respecting `chi` built in one implementation's
  basis is diagonalised (off-diagonal < 4.2e-12) and rebuilt (rel. err < 2.6e-12) by the other.
  This is the non-circular form of the existing soundness test, which builds `chi` from the same
  `Q` it then checks.

The caller tolerates the change by construction: `Q` reaches `calc_spectra_tensor` only through
`_combine_component_ops(component_ops, Q[:, c])` and
`einsum("wa,pa,qa->wpq", chi_diag, Q, Q.conj())`, both invariant under a per-column phase. That
was checked before the rewrite, not after.

**The tolerance is deliberately not simplified.** `cut` still scales with `m * n_orb^2`, the row
count of the block that was conceptually solved, not `R.shape[0]`. That expression reconstructs
the row count instead of reading it off the block, so it holds only while `m == len(Ts)` and
every `T` is `(n_orb, n_orb)`; the comment at the site states that invariant, because an
arithmetic expression correct only under an unstated invariant is how the `n_bytes // 8` hang
got in.

**The row-block budget is measured, not guessed.** The transient runs ~5.7x the budget
(`vstack` copies the block beside the carried factor, then `np.linalg.qr` copies again), so the
constant matters. Swept at `n_orb=151` -- 64 MiB: 295.5 MiB / 3.31 s, 32: 193.1 / 3.26,
16: 91.5 / 3.52, 8: 52.2 / 4.95, 4: 32.7 / 8.06. The knee is at **16 MiB**, which is the
default; below it the per-block `O(n_gen^3)` re-triangularisation dominates. Every budget
returned a leading singular value identical to 12 digits.

**Four injected bugs confirm the tests discriminate** (chunk-offset slice, missing transpose in
the trace contraction, missing conjugation, transposed commutator block) — each turns the new
tests red. Written down because a test that is green for the wrong reason is this campaign's
most repeated failure.

**What is left standing here, by measurement rather than assertion.** Per-stage VmHWM with a
`clear_refs` reset at every boundary, at `n_orb=151`: component tensors 0.4 MiB, `q_t` 4.1,
**`discover_one_body_symmetries` 150.2**, the streamed residual 91.6, `svd(R)` and downstream
14.3 -- which sums to the 244.9 above, so the total is accounted for. The dense generator list
is now the dominant term at 61% of what remains: `n_gen` matrices of `n_orb^2` complex
(147.9 MiB of payload at `n_orb=151`), still `O(n_orb^3)` and still replicated.
Since every generator is a rank-1 outer product `u_a u_b^dagger` of eigenvectors, a factored
representation would make it `O(n_orb^2)` — but that changes `discover_one_body_symmetries`'
public contract, so it is a separate change and is **not** made here.

### P1-5. `compute_impurity_rdm`'s guard protects the wrong stage — MEASURED, FIXED

`gs_statistics.compute_impurity_rdm` takes `max_bytes=256 MiB`. The guard is evaluated at
`gs_statistics.py:470`, **after** the local pass has built `local_groups` and after the whole
thing has been `graph_alltoall`'d. It bounds `state_blocks` only, which is the one allocation
in the function that is not the problem.

Measured with a stand-in basis carrying a real communicator (the function reads only
`num_spin_orbitals`, `impurity_spin_orbital_indices`, `comm`, `is_distributed`; the state is a
real `ManyBodyState`), `n_orb=124`, VmHWM growth per rank, sampled at the stage boundaries by
wrapping the `graph_alltoall` name **in `gs_statistics`'** namespace — patching the defining
module would not have reached the call site:

| N_local | width | ranks | total | local pass | alltoall | guard + blocks | what the guard bounds |
|---|---|---|---|---|---|---|---|
| 5,000 | 54 | 2 | 126.6 MiB | 28.7 | 80.1 | 17.8 | 52.3 MiB |
| 10,000 | 54 | 2 | 211.2 | 57.3 | 153.8 | 0.0 | 52.3 |
| 20,000 | 54 | 2 | **420.1** | 111.2 | 308.9 | 0.0 | 52.3 |
| 40,000 | 54 | 2 | **858.7** | 199.3 | 659.4 | 0.0 | 52.3 |

Linear in `N_local` and linear in `width` (at `N_local=20,000`: p=1 -> 8.7 MiB, p=8 -> 62.2,
p=27 -> 212.9, p=54 -> 420.3), and **flat in rank count** at fixed per-rank load (2/3/4 ranks:
419.8 / 408.9 / 391.1 MiB). So the transient is set by `N_local x width` and every rank pays it
at once — on a node it multiplies by ranks per node.

**The sharpest form of the finding: when the guard fires, it has already paid.** At `n_imp=14`
(an f shell) the blocks would be 9,705 MiB, so the guard trips and returns `None` — after
spending **462.6 MiB/rank**, 1.8x its own budget, to reach the decision. A guard that exists to
avoid an allocation cannot be placed after the allocation it is avoiding is already dwarfed.

**Where the ~400 B/entry goes**, per `(determinant x nonzero column)` entry — the measured
constant is 397-413 B across every point above:

| | bytes/entry |
|---|---|
| `local_groups` tuple `(n, n_e, m, amp)` + list slot | 120 |
| pickled send buffer | 33 |
| pickled receive buffer | 33 |
| received tuples, materialised again as Python objects | 120 |
| dict/list overhead, allocator slack | remainder |

**Nothing is freed until the function returns**: `local_groups`, `send` (which holds the *same*
list objects, so it is a re-indexing and not a copy), `received` and `groups` are all still
bound while the blocks are accumulated and Allreduced.

**The wire is not the problem, the objects are.** Pickle puts a distinct entry on the wire in
33 B against 23 B for a packed record — only 1.4x. The 16x is in materialising Python tuples
and complex objects twice. Every field is fixed-width (`n < width`: uint16, `n_e <= n_imp`:
uint8, `m < C(n_imp, n_e)`: uint32, `amp`: complex128 = 23 B packed), so a record array would
carry the same information at ~23 B/entry live *and* on the wire, taking the ~400 B/entry to
roughly 50-70 including both copies. This is the same shape as the `_index_sequence` pickle
peak (P1-1b), and the same fix applies.

**Ranking.** `O(N_local x width)`, distributed, flat in rank count — which the plan's
`O(N_local)` filter would drop, *except* that the filter explicitly does not drop a dominant
distributed term whose constant is reducible. `N_local` grows as `de2_min` tightens, so it is
also convergence-sensitive.

### Both halves fixed

**Staging** (`03e2d0f`). A crude bound over *every* impurity count needs no pass at all; when
it fits, the exact decision cannot differ, so the common case pays nothing. Only when it fails
is the exact `observed_n` worth one allocation-free pass, and the refusal is then free. The
original late guard stays as the authority.

| | before | after |
|---|---|---|
| guard trips (`n_imp=14`) | 462.6 MiB / 1.94 s | **0.0 MiB / 0.09 s** |
| guard passes (`n_imp=10`) | 420.1 MiB | 420.6 MiB (unchanged) |

Both stages decide from an **allreduced** `guard_width`. The first draft gated the new
`allgather` on the rank-local `width`, which is the asymmetric-deadlock shape this repo already
has on record for a width-0 block on one rank.

**Representation** (`55ee8e9`). Entries became flat typed arrays.

| N_local (width 54, 2 ranks) | before | after |
|---|---|---|
| 20,000 | 420.1 MiB / 3.10 s | **142.1 MiB / 1.61 s** |
| 40,000 | 858.7 MiB / 6.79 s | **275.6 MiB / 3.08 s** |

~400 -> ~138 B/entry, and about 2x faster. Tier 2: the accumulation order changes, though the
old code already varied with rank count through `defaultdict` iteration order. Verified against
the pre-change implementation at 1/2/3/4 ranks, an empty rank, widths 1-27 and several impurity
counts: worst elementwise and worst block-trace deviation both 4.4e-16.

**The prediction that was wrong, recorded because it was the obvious one.** This looked like
the `_index_sequence` pickle peak and the expectation was that the wire format dominated. It
does not: a distinct entry pickles to 33 B against 23 B packed, only 1.4x. The 16x was in
materialising Python objects on both sides. *Measuring the serialized size before choosing the
fix is what separated these two cases*, which otherwise present identically.

**A test that was green for the wrong reason, found by injection.** Four plausible bugs were
injected; three turned the new tests red and "segment boundary forgets the state index" did
not. Two rounds of reasoning about why were both wrong. The reason, from direct experiment:
entries sort as `(bath group, state, N_imp, config)`, so a group with several `N_imp` values
resets the count at each state boundary and accidentally marks it. Only a bath group whose
configurations all share one `N_imp` leaves consecutive entries differing by state alone. The
oracle fixture now contains such a family.

### P1-1b. `build_sparse_matrix` FIXED for serial; the distributed peak relocated to `_index_sequence`

| | before | after |
|---|---|---|
| serial, N ~393k | 291.2 MiB | **171.7 MiB** (-41%) |
| 2 ranks, N ~484k | 810.4 MiB | 783.1 MiB (-3.4%) |

Streamed the operator image instead of listing it, read amplitudes through the buffer protocol
instead of `.items()`, and built numpy arrays instead of Python lists. Verified bit-identical
against a dense matrix built directly from the images.

**The distributed branch barely moved, and attributing it relocates the target.** At two ranks,
building the `bras` list costs **113.9 MiB** and `Basis._index_sequence` costs **596.0 MiB** of the
~783. The peak-setter is the routed all-to-all that resolves bras to global row indices, which
pickles `SlaterDeterminant` objects -- not this function's own accumulation. **Further work inside
`build_sparse_matrix` buys nothing distributed;** the next increment is `_index_sequence` /
`mpi_comm`, and that is where the rank-count growth lives.

**An intermediate version of this fix made serial worse** -- 291 -> 520 MiB -- by unifying the two
branches, so the serial path stopped filtering inline and accumulated the whole image before
masking. About 76% of image rows are dropped, so *where* the filter runs is most of the cost. The
distributed branch cannot filter early: its lookup is collective and must run exactly once per
rank.

**Fourth silent-zero of the campaign, caught before it fired.** Switching this function to a
generator would have silently zeroed the perf harness's "apply" leg: it patches by module-global
name, and timing a generator *call* measures only its creation. The harness now wraps the
generator and charges each image as produced.

### P1-2. Materialization sites created by the `local_basis` view — MEASURED, FIXED

Making `local_basis` a non-materializing view left every site that *iterates* it paying for
objects it did not previously allocate. Measured before ranking, on a 400k-determinant basis,
cold process, `VmHWM` reset at the stage boundary:

| stage | peak before | peak after |
|---|---|---|
| `len(local_basis)` (control) | 0.0 B/det | 0.0 |
| walk the basis, keep nothing | 83.2 | **0.0** |
| `list(local_basis)` | 95.7 | 95.7 (the caller keeps them; unchanged by design) |
| `build_distributed_vector` | 200.5 | **13.4** |
| the restriction scan (`basis_restrictions.py:114`) | 83.1 | **0.0** |
| the GF unit split (`basis_split.py:219,223`) | 222.3 | 187.2 — *payload, not walk; still open* |

**The probe had to be rewritten before any of this was true.** Its first version built the
basis from one N-element list and reported `list(local_basis)` at **29 B/det** — impossible
for an object that cannot cost under ~80. The setup had left N determinants' worth of freed
pymalloc arenas behind, and the stage allocated into them. Growing the basis in chunks, the
way CIPSI actually grows one, gave 95.7. *A setup must not pre-warm the heap to the size of
the thing being measured* — and the tell was an arithmetically impossible figure, the same
shape of tell that caught the biased baseline in P1-4.

**Two fixes, two commits.**

1. `_LocalBasisView.__iter__` was `iter(self._keys.keys())` — a Python list of every
   determinant built before the first one is yielded. It now streams `key_at(i)`. This is
   *also faster*: 26.1 ms against 30.9 ms per 400k walk, because building the list is itself
   work. The memory win and the time win point the same way, which is unusual enough in this
   campaign to be worth stating.

2. `build_distributed_vector` kept `itertools.product`, which **materializes each argument to
   a tuple when it is constructed** — so it defeats a streaming iterator by construction, and
   fix 1 does nothing for it. Nested loops: 200.5 -> 13.4 B/det, against a 16 B/det output.
   The visit order is identical (product yields its last argument fastest), so the values are
   bit-identical.

Both are tier 1 on values — every amplitude is bit-identical. Fix 1 carries one qualifier
that a bare "exact by construction" claim would hide: code that grew the basis mid-walk used
to get silently wrong results and now raises. The tier-1 claim therefore rests on the
**audit** below finding no such caller, not on the change being incapable of altering
behaviour. The guard that makes fix 1 safe is the interesting part: `add_states` merges
into the key block **in place** (`ManyBodyBlockState::merge_keys`), so growing the basis
mid-walk shifts the positions of determinants not yet yielded, and a streaming walk would
silently skip or repeat them — wrong answers, no crash, the class this plan named as its top
risk. The size is therefore checked *before* each step and iteration raises, the way a `dict`
does. Every `local_basis` use in the repo was audited first, including the Cython ones
(`_lanczos_step.pxi:631,673` take only `len()`; `ChebyshevFilter.pyx:74` consumes the walk
once); no production site mutates the basis while iterating it.

Of the four new tests, two fail against the pre-fix code, plus the `build_distributed_vector`
bound — verified by reverting each, not assumed.

**One eager path remains on the view, unfixed and deliberately so.** `__getitem__` still calls
`self._keys.keys()[index]` for a *slice* — the same materialization `__iter__` just lost. No
production or test site slices `local_basis` (checked: no `local_basis[...:...]` anywhere in
`src/`), so it costs nothing today and a lazy replacement would be a code path with no caller
to exercise it. Recorded here rather than changed.

**Still open, and deliberately not folded into these numbers**: the split walk's remaining
187.2 B/det is not iteration at all. `basis_split.py:219-223` retains a list of key `bytes`
*and* `set(basis.local_basis)` simultaneously — that is the wire payload, and it needs its own
finding rather than credit from this one.

### P1-9. `best_basis` doubled the basis representation during refinement — MEASURED, FIXED

(Numbered 9 rather than 3: an earlier pass gave two findings the number P1-3, and renumbering
them now would break the references already written above.)

`cipsi_solver.py:1315` held `best_basis = list(self.basis.local_basis)` across every subsequent
refinement cycle of a capped expansion, alongside the live basis it is a copy of. Measured on a
400k-determinant basis, peak equal to steady in all three cases:

| snapshot representation | cost |
|---|---|
| `list(local_basis)` — what it was | 80.2 B/det |
| `ManyBodyState.from_keys(...)` — the obvious fix | **115.2 B/det, worse** |
| packed `(n, n_bytes)` key array — what shipped | **9.8 B/det** |

**The obvious fix is the wrong one, and the reason is 0i's rule applied to a prediction of this
campaign's own.** The C++ key block is the representation the `Basis` store itself moved to, so
it looked certain to win. It loses: `from_keys` builds a `vector<SlaterDeterminant>` and copies
it into the block, and glibc does not return the freed transient, so `VmHWM` — the metric the
guard and the OOM killer act on — keeps both. On retained RSS it would have looked fine. The
measurement was taken before the change was written, which is the only reason this was a
refuted prediction rather than a shipped regression.

The restore goes through `add_states`, which normalizes `bytes` one determinant at a time, so
nothing is materialized as a list on the way back either.

**The second test exists because the first could pass for the wrong reason.** Packing only the
first chunk is not a crash and not an exception — `_as_determinant` zero-pads a short key back
to the basis width — it is a *different determinant*. So the fixture asserts it actually
occupies the second chunk before the round trip means anything, and it needed its own `Basis`:
the shared helper in that test file is pinned at 64 spin-orbitals and rejects an over-wide key
rather than storing it.

### P1-6. `cartan_subalgebra` asked LAPACK for a 16 GiB block nothing reads — MEASURED, FIXED

Found by pricing an unmeasured path, not from either open-items list. The intent was to measure
the dense generator list of `discover_one_body_symmetries` (the top-ranked open item); the first
stage above it turned out to be four orders of magnitude worse.

`cartan_subalgebra` solves `sum_k c_k [X, H_k] = 0` as the null space of a `(2 n^2, m)` real
system with `m ~ 2n`, and reads only `s` and `vt`. The default `full_matrices=True` makes LAPACK
produce the `(2 n^2, 2 n^2)` left-singular block as well, and it is discarded on the next line.
`_matrix_commutant` stacks a `(len(mats) n^2, n^2)` system and does the same.

Measured on a spin-degenerate `h` (group size 2, so `n_gen = 2n`), VmHWM reset at the boundary:

| n | before | after `full_matrices=False` | after `del cols` |
|---|---|---|---|
| 80 | 2605.0 MiB / 30.30 s | 129.7 MiB / 0.79 s | **114.2 MiB / 0.73 s** |
| 150 | **OOM-killed** (exit 137) | 726.7 MiB / 12.91 s | **623.7 MiB / 9.19 s** |
| 200 | unreachable | 1716.9 MiB / 35.76 s | **1473.2 MiB / 36.28 s** |

The discarded block is 1.3 GiB at `n = 80` and 16 GiB at `n = 150`. This is a **production
path** — `solver_basis.py:286,363` -> `symmetry_adapted_basis` /
`symmetry_adapted_transformation` — so a symmetric `h` at these sizes was OOM-killed inside
symmetry discovery, not inside the many-body machinery.

**Tier 1 by structure**, and the distinction matters: the left block is never read, and for a
tall system `vt` is `(m, m)` under either form, so `s` and `vt` are the same mathematical
objects. Verified bit-identical on the shapes this code builds (the whole Cartan and the rotated
one-body diagonal agree exactly at `n = 12, 30, 50`; `s` and `vh` agree exactly on `(1800, 60)`,
`(5000, 100)`, `(12800, 160)`). It is *not* a general LAPACK guarantee — on one synthetic
`(2000, 40)` real matrix gesdd's two paths differ by 8.3e-16 — so the claim is "same object,
verified identical here", not "bit-identical by construction".

**The uncomfortable part.** This is the *third* instance of this pattern in this one file. Two
were fixed in `002a971`, and the note written at the time says in as many words: *"Watch
`np.linalg.svd` defaults: `full_matrices=True` on tall matrices allocates M x M."* The rule was
recorded and not applied exhaustively. A grep for `linalg.svd` without `full_matrices` across
`src/` takes seconds and would have found both; it has now been run, and the only remaining
hit is the square `(n^2, n^2)` superoperator at `lie_algebra.py:405`, where the two forms are
the same array.

**The test asserts the shape LAPACK is asked for**, not a memory figure — that is the mechanism,
and it does not drift with the machine. Its first version passed against the unfixed
`_matrix_commutant`: with a single matrix the system is `(n^2, n^2)`, square, where both forms
are the same array. It needs two matrices to be tall. Both parametrizations now fail against
the pre-fix code.

**Open, measured, and the reason the generator list was demoted.** `cartan_subalgebra` still
holds five simultaneous copies of the same `O(m n^2)` content: `generators`, `herm`, `cols`,
`real_sys` and the economy left block. Dropping `cols` the moment `real_sys` exists removes one
of them — predicted 15.6 / 103.0 / 244.1 MiB at `n = 80 / 150 / 200`, measured 15.5 / 103.0 /
243.7, which is as close as this campaign's predictions have come. `generators` cannot be freed
from inside, because the caller holds the list; `herm` is read after the SVD. Those remain.

The dense generator list itself is **103.0 MiB of a 623.7 MiB peak at n=150, i.e. 17%** — not
the 61% recorded in the handover, which was measured against the post-P1-4
`component_symmetry_reduction` peak, a different call path. Each generator is a rank-1
`u_a u_b^dagger`, so a factored form would be `O(n_orb^2)`; it changes a public contract and is
now worth less than it looked.

### P1-7. The fused unpack held the amplitudes twice — MEASURED, FIXED

Plan item 3, and the first finding from path **D** (the Cython/C++ storage, operator and
MPI-packing layer), which no earlier pass had looked at.

`unpack_block_fused` (`MpiUtils.cpp`) parsed every received entry into `keys` — a vector of
vectors, one heap allocation per determinant — and `amps`, `total x width` complex, then
deduplicated those into `out_keys`/`out_amps`. So the coefficients were resident twice at the
worst instant, on top of the MPI receive buffer, which is alive for the whole call.

| total entries | width | peak before | peak after | time |
|---|---|---|---|---|
| 60k | 32 | 56.2 MiB | **26.3** | 0.03 -> 0.02 s |
| 60k | 128 | 209.6 | **92.3** | 0.10 -> 0.04 |
| 60k | **320** (production `p`) | 517.4 | **224.0** | 0.24 -> 0.09 |
| 100k | 64 | 178.6 | **80.1** | 0.09 -> 0.05 |
| 100k | 64, no duplicates | 203.9 | **105.8** | 0.10 -> 0.05 |

The excess tracked `amps` exactly (`60000 x 320 x 16` = 293 MiB). Only the keys come out now,
into one flat `uint64` array `chunks/width` times smaller and with no per-key allocation; each
amplitude row is read once from `recv_buf` at emit time. The distinct determinants are counted
before reserving, so the long-lived result carries no capacity slack either. **Peak now equals
steady** — 224.0 MiB for a 220.4 MiB result, 1.6% overhead — and the call is 2.4x faster.

This is `redistribute_block`, which the 128-rank ledger names as the CIPSI peak-setter.

**Tier 1, measured rather than argued.** The previous implementation was rebuilt from a file
backup and its keys and amplitudes compared against this one's: `array_equal` on both, max
difference 0.0. The property that makes that hold is that the comparison is element-wise over
`uint64` — what `std::vector<uint64_t>::operator<` does. A byte compare would disagree on a
little-endian machine, and key order is what every rank's `offset`-based global index
arithmetic depends on.

**The rebuild silently did not happen, and the arithmetic is what caught it.** The first
attempt failed to compile (`SlaterDeterminant` inherits `std::vector` without
`using vector::vector`, so it has no iterator-range constructor), `pip` printed
`failed-wheel-build-for-install`, and the shell still exited 0 — so the old `.so` stayed
loaded and five measurements of unchanged code were taken as the result. The tell was that
they reproduced the baseline to 0.1%, which a 2x change cannot do. After editing
`src/cython/`, assert the rebuild positively (`grep -c '^Successfully installed'`); an exit
code and a log tail both miss it.

**One test discriminates and one does not, deliberately.** The memory bound fails against the
previous implementation (2.32x the payload against a 1.5x bound). The oracle test passes
against both, because the old code was also correct — it pins the wire contract going forward.
Saying which is which matters: a suite where every test passes against the pre-fix code is the
shape of a suite that tests nothing.

**Also worth recording**: the oracle initially failed against *correct* code because it decoded
the output keys through `to_bytearray()`, which writes each chunk big-endian and over-allocates
8x (the latent defect recorded below). The chunk accessor `k[c]` is the way in.

### P1-8. The basis split's wire payloads were Python objects — MEASURED, FIXED

The half of P1-2 that was carved out as payload rather than walk, plus the seed vectors beside
it. `split_basis_and_redistribute_psi` replicates the basis into every color, so both payloads
are built once per determinant per other color and materialized again on arrival.

| payload | built as Python objects | packed as arrays |
|---|---|---|
| `det_send` (a `bytes` per determinant) | 87.2 B/entry | **0.0** |
| `psi_send` (an `(int, bytes, complex)` tuple per entry) | 158.3 B/entry | **16.1** |

At 200k determinants and 4 seeds that is 120.8 -> 12.3 MiB on the seed payload alone, 9.8x.
The reasoning is P1-5's, reused: `graph_alltoall` pickles a numpy array as a raw buffer and a
Python object as objects on **both** sides, and in `compute_impurity_rdm` the wire format was
only 1.4x of the packed one while materializing the objects was the other 16x.

**The risk was routing, not packing**, and that is where the tests went. The scalar path used
pure-Python ints deliberately — `routing_hash()` is a `uint64` that does not fit a C long, and
`big_int % np.int64` overflows on the numpy coercion — so the vectorized modulus is taken in
`uint64` throughout. Getting it wrong scatters determinants to the wrong color: wrong answers,
no crash, the class this plan ranks first among risks. It is pinned against the scalar
spelling over hashes spanning the top of the range, where the `int64`-coerced form gives
`[1, 2, 2]` for the correct `[2, 0, 0]`.

Grouping is a stable argsort, so each destination's rows keep their enumeration order — the
order duplicates are summed in on arrival. That is pinned as an **ordering** property rather
than a numerical one, because an unstable sort would still deliver every row and still total
correctly to rounding; a tolerance test would not have noticed. Verified: the same grouping
with `kind="quicksort"` fails it.

**Re-measured before assuming**: the carve-out was 187.2 B/det when P1-2 recorded it, and
186.7 after the P1-7 unpack rewrite — unchanged, because the split goes through
`graph_alltoall` and not the fused path. That check cost one probe run; the assumption it
replaced is the one that produced the stale 61% figure corrected in P1-6.

## Verification: benchmarks and the calibrated constant

### `pytest -m benchmark` — green, and the apply golden holds

The plan requires this because a memory win must not cost wall time silently. Default set:
8 passed, 11 skipped (all env-gated `RUN_*_BENCH`, not missing data). The two that cover paths
touched here were run with their gates set — `test_matrix_build_perf` and
`test_symmetry_golden`, both pass. The block-apply golden is unmoved:

    [apply-block] p=1  block=  98.94 ms  speedup=0.99x
    [apply-block] p=8  block=  99.57 ms  speedup=6.94x   n_out=188056

### `_PY_BASIS_OVERHEAD_BYTES` — re-measured, NOT changed, and the reproduction is partial

Re-run by the method the constant's own comment specifies: VmHWM, nso=124, one cold process
per point, marginal slope over N = 200k -> 400k. Three repeats, and the slope is stable to
0.5 B/det.

| | recorded 2026-09-18 | re-measured 2026-09-21 |
|---|---|---|
| marginal **peak** slope | 233.1 B/det | **243.7 - 244.2** |
| marginal **retained** slope | 52.3 B/det | **130.4** |

The peak column — the one the constant is derived from — reproduces to ~5%, implying 172
against the stored 161. **The retained column does not reproduce at all**, and that is the
part worth stating rather than smoothing over: the C++ key block should retain ~56 B/det at
`n_bytes = 16` structurally, which matches the recorded 52.3 and not the 130.4 measured here.
So this probe is inflated by construction transients glibc never returned, which makes it a
*partial* reproduction of the original instrument, not a replacement for it.

**The constant is therefore left at 161**, for two reasons. The probe cannot be shown to
reproduce the instrument, and changing it is **tier 3 by effect** — it resizes RAM-derived
truncation caps, hence which determinants are admitted, hence energies on capped runs — which
the plan routes through the Phase-3 harness that does not yet exist. The direction is recorded
because it is the unsafe one: if 172 is right, 161 is optimistic, and optimistic here means
*larger* caps.

Nothing shipped in this campaign touched that path. `add_states`' serial branch still builds a
list, a `set` and a sorted list of N determinants before `merge_keys`, and that is what sets
the 244 B/det peak — the next thing to attack if this term is ever worth attacking.

### P1-10. `from_states` left `width x` the key capacity — MEASURED, FIXED

Plan item 5 ("no `shrink_to_fit` anywhere in this layer"), and the waste was **not where the
item predicted it**. It is at construction, not after a shrink.

`ManyBodyState.from_states` builds the union support by pushing every column's keys —
`width * rows` of them — then sorts, deduplicates and `erase`s. `erase` shrinks the logical
length and keeps the allocation, so the block carries `width` times the key capacity it needs
for its whole life. Measured on a 40k-row block, key slots live -> allocated:

| width | live | allocated | factor |
|---|---|---|---|
| 1 | 0.92 MiB | 1.50 MiB | 1.6x (ordinary geometric growth) |
| 8 | 0.92 | 12.00 | **13.1x** |
| 32 | 0.92 | 48.00 | **52.4x** |

One `shrink_to_fit` makes every one exact, taking a width-32 block from 68.45 to 20.45 MiB —
**3.3x** — and the Lanczos and GF recurrences hold several blocks at once. The amplitude
vector was always exact (it is `resize`d once) and is now pinned so that cannot silently
change.

**It needed an instrument, and that is the transferable part.** `RSS` cannot see this:
shrinking a `std::vector` returns nothing to the allocator, so the slack is invisible from
outside the process. `memory_bytes()` reports the logical size and is blind to it too. This
finding only exists because `row_capacity`/`amp_capacity` were added to the C++ block and
exposed as `capacity_stats()` first — an instrument commit *before* a number, which is the
shape the plan asked for and the reason item 5 sat unpriced for the whole campaign.

**The predicted half is real and deliberately left alone.** After a 10% `keep_rows`
projection the block keeps 10x the capacity. Shrinking there reallocates and copies on the
capped Green's-function recurrence's hot path, and that cost is unmeasured. A test pins the
current behaviour, so it stays a known quantity and adding the shrink has to be a deliberate
act rather than a drive-by.

## Closed as not worth it

The prior campaign's most valuable output was its refutations, so each row here carries the
measurement that closed it rather than an argument.

### The replicated `ManyBodyOperator` — CLOSED at ~8 MiB/rank

Plan item 7's remaining half, and the item the plan's own schedule said to start with: a
replicated cost is independent of `comm.size`, so one rank settles it authoritatively.

Measured at production shape — the two-body Coulomb block on an f shell (14 spin-orbitals),
the one-body block spanning every orbital:

| n_orb | terms | `m_ops` retained | flat caches retained | total/rank |
|---|---|---|---|---|
| 124 | 23,657 | 3.8 MiB (168.5 B/term) | 4.5 MiB (198.4 B/term) | **8.3 MiB** |
| 248 | 69,785 | 9.8 MiB (146.9 B/term) | 16.8 MiB (251.7 B/term) | 26.6 MiB |

**8.3 MiB/rank at the f-shell geometry is ~0.2% of the per-rank peak**, so by the plan's own
"blocked, not ranked" filter this is not a finding, however good its scaling class looks. The
per-node figure (~1.1 GiB at 128 ranks/node) is real, and the remedy for it is already on
record and needs no code: halving ranks per node doubles the per-rank budget.

Recorded so the next pass does not re-derive it: of the flat caches' 198 B/term, 48 B is two
`std::vector` headers per term — `m_density_mask` and `m_onebody_between` are pushed for
*every* term, empty when that term's flag is unset. Construction peaks at ~1.7x the retained
size (`collect_flat_terms` copies the whole term list, and eleven vectors grow geometrically
with no `reserve`). Both are real and both are worth nothing at this scale.

**The shape matters more than the size here.** The same probe with the two-body block over
*all* orbitals gives 610k terms and ~267 MiB/rank. That is not a production shape — it is the
`n_orb^4` blow-up already closed by `extract_tensors(..., two_body=False)` — and measuring it
first would have promoted this item by 30x on a geometry nothing runs.

### Path C (double counting) — reviewed, NO findings. Evidence: read, not measured.

The last wholly unexamined path. Reviewed against the hazard checklist across
`dc_criteria.py`, `dc_search.py`, `dc_static.py`, `dc_reference.py`, `dc_record.py`,
`dc_frozen.py` (5,333 lines).

1. **Replication across ranks** — `_SectorSolution` is explicitly rank-replicated, and that is
   the point: it is broadcast. It holds six scalars.
2. **Simultaneous copies** — none. There is not a single `np.zeros`/`np.empty`/`np.ones`/
   `np.full` in any of the six modules; nothing here allocates an array sized by determinants,
   orbitals or frequencies.
3. **Superlinear growth** — none; the accumulators are `rungs`, `attempts`, `edges`, `spreads`,
   `sizes` and `lines`, all one scalar or tuple per search step.
4. **Never-evicted caches** — `_SectorContext.sector_at` is keyed by `(mu, n_trial)` and never
   bounded, but it stores `_SectorSolution`s (~200 B) and is cleared at `dc_criteria.py:1799`
   and `:2435`. A search of a hundred evaluations holds ~20 KB.
5. **Peak-vs-steady transients** — the DC layer's peak is entirely the ground-state solves it
   drives, which is the machinery the rest of this campaign audited.

**Someone already applied this campaign's reasoning here, before the campaign.**
`sector_solve`'s docstring records the decision not to retain eigenvectors across a search in
exactly these terms: *"~20 states over a sector basis is ~2 GB across a search at the
400k-determinant caps this stack runs at, where a float pair is nothing."* The cache keeps the
two floats and pays one extra `build_density_matrices` per solve instead.

**Stated as a read, deliberately.** The ranking schema admits `read` as an evidence level, and
this is one: no probe was run, because there is no term here whose size depends on anything
this campaign varies. What would overturn it is a DC path that starts retaining a `Basis`, a
`psis` list or a density matrix across evaluations — so that is the thing to re-check, not the
line count.

### Plan item 8's second half — the claim is REFUTED by reading the function

The plan listed "`basis_split` replicating the full basis into every color, uncounted by
`max_colors_within_budget`". It is counted. The function estimates the per-rank GF peak with
`ranks=ranks_per_color = comm.size // n_colors` (`memory_estimate.py:1046`), and that argument
*is* the replication: after the split each color holds the whole `n_dets` on
`comm.size / n_colors` ranks, so the per-rank share is `n_dets * n_colors / comm.size`, which
is exactly what passing the reduced rank count models. Its own docstring says so — "each
color's unit basis may fill the same `truncation_threshold` on only `comm.size / n_colors`
ranks, so per-rank memory grows with the color count".

**What is genuinely not modelled** is the split's own *construction* transient, which is a
different thing from the steady replication: inside `split_basis_and_redistribute_psi` a rank
briefly holds its pre-split share, the received keys, the `new_states` set and the new
`Basis` being built from it. That is of order the post-split basis, and P1-8 has just removed
its payload half. Unmeasured — it needs a real multi-color split — so it is recorded as a row,
not ranked.

The ranked table's row for this item is updated accordingly: the term is real but it is the
transient, not the replication.

### Plan item 6's three never-evicted caches — CLOSED, none of them is one

- **`SectorCache` (`groundstate.py:107-143`)** is not never-evicted: it is an LRU bounded at
  `max_size=3`, with an explicit `clear(keep=...)` that also nulls the MPI communicator on the
  losing bases to break a reference cycle (a split communicator freed by the cyclic GC can be
  collected after `MPI_Finalize` and crash). Three whole `(Basis, CIPSISolver)` pairs is a real
  3x multiplier on the basis, and it is the documented trade for not re-walking a sector.

- **`SectorResolventCache._index` (`gf_shift_recycling.py:148`)** *is* the `state -> index`
  Python dict this campaign removed from `Basis`, resurrected in the GF path. It is
  nevertheless **blocked, not ranked**: it sits beside `_evecs`, a dense `(N, N)` complex
  eigenvector matrix, and `_sector_dense_max` derives the admissible `N` from exactly that
  array. At the campaign's own measured 73.1 B/det (0e), the dict is 0.3% of `_evecs` at
  `N = 1000` and **0.06% at `N = 8000`**; the bound puts `N` in that range by construction.

  *A note on how this was measured, because the first attempt produced nonsense.* Building the
  dict on a freshly-built `Basis` reported 3.1 and 1.5 B/det at `N = 4000` and `8000` — below
  the 8 B of a bare pointer, so not a measurement of anything. The basis construction had left
  arenas the dict allocated into, the same contamination that made P1-2's first probe report
  29 B/det. At these sizes the term is far too small to measure against a warm heap at all,
  which is itself the answer; the 73.1 B/det figure above comes from 0e, where it was measured
  on a geometry that could carry it.

  The multi-cache hazard the plan flagged separately — `rixs.calc_map_cartesian` keeping one
  cache per thermal eigenstate — is already handled: `_sector_dense_max` takes `n_live_caches`
  and sizes `N = sqrt(0.25 * available / ((n_live + 2) * 16))`.

- **`_rixs_map_adaptive`'s `cols` (`rixs.py:125`)** is not a cache. It accumulates one map
  column per *solved* `wIn` point, and the assembled map is the function's return value — every
  column in it is needed to produce the result. At the sampler's measured 28 solves of 121 it
  is also the smaller of the two possible shapes.

## Phase 0 review — feasibility of the headline fix

Reviewed read-only against the whole call-site surface. **Verdict: feasible, no fundamental
blocker**, but three findings change the plan.

### The ordering risk is de-risked, and for a better reason than expected

The plan flagged key comparison order as the highest-risk item (if it flips, `offset`-based
global indices disagree between ranks — wrong answers, no crash). It is **definitional, provided
the new store reuses the existing key type**: `SlaterDeterminant.__lt__` is `self.s < other.s`
(`_slater_state.pxi:71-72`) and `ManyBodyBlockState::Key` is *the same C++ type*
(`ManyBodyBlockState.h:154`), searched with the same `operator<` / `std::lower_bound`
(`:259-265`). Verified empirically as well: `from_bytes`'s little-endian/reverse round trip
(`_slater_state.pxi:30-37`) is algebraically identical to zero-padding at the tail, and
`sorted(bytes) == sorted(by chunks)` over random draws. Three orders agree.

**So the rule is: reuse the key type and comparator, never re-derive a byte comparison.**

### NEW, and not on anyone's list: variable-length keys are already ambiguous

`std::vector::operator<` treats a proper prefix as strictly less than its zero-extended twin, so
the *same physical occupation* built from byte strings of different length is two distinct,
unequal keys today:

    from_bytes(b"\x80")            -> 1 chunk
    from_bytes(b"\x80" + b"\0"*8)  -> 2 chunks     # unequal, and the 1-chunk one sorts first

This is latent in the current dict-based `Basis` too. A fixed-width contiguous array either
forecloses it by construction or, implemented carelessly, reads past a shorter vector. It must
become an **enforced precondition** (every determinant entering one `Basis` has exactly
`n_bytes`), asserted before any storage change — a standalone hardening commit.

### NEW, and the one place a mechanical patch produces silently wrong answers

`_index_dict.get(val, self.size)` (`manybody_basis.py:659,677`) returns the **global** size as a
miss sentinel. A naive `offset + bisect_pos` substitution returns `offset + local_len` instead —
which, for any rank but the last non-empty one, **is a valid global index owned by another
rank**. `_index_sequence`'s repair loop (`:680-690`) only re-queries when the result is `> size`
or `< 0`, so a bogus-but-in-range miss sails through and lands as a fabricated row/column index
in `build_sparse_matrix` (`basis_transcription.py:205`). Needs a deliberate sentinel translation
and a cross-rank-miss regression test at `-n 2`/`-n 3`.

### Two latent defects verified directly (independent of the memory work)

**1. Same occupation, two unequal keys.** Reproduced:

    a = SlaterDeterminant.from_bytes(b"\x80")            # 1 chunk
    c = SlaterDeterminant.from_bytes(b"\x80" + b"\0"*8)  # 2 chunks
    a == c   -> False        a < c -> True        hash(a) == hash(c) -> False

`add_states` does **not** pad inputs to `self.n_bytes` (`manybody_basis.py:277`), so mixed-length
inputs to one `Basis` would produce duplicate, unequal entries for one physical state. No
production path is known to mix lengths today, so this is latent, not live — but it is the
precondition the contiguous-array work must enforce rather than inherit.

**2. `SlaterDeterminant.to_bytearray()` over-allocates 8x.** `_slater_state.pxi:44` allocates
`8 * n_bytes * len(self)` where `n_bytes` is already 8 (bytes per chunk), then fills only
`n_bytes * len(self)`. Measured: a 1-chunk determinant returns **64 B with 56 trailing zeros**; a
2-chunk one returns 128 B.

*Harmless today, and the reason matters*: both callers compensate — `basis_split.py:219` slices
`[: basis.n_bytes]` before it goes on the wire, and `basis_restrictions.py:114` bounds bit
extraction by `num_spin_orbitals`. So this is a per-determinant **transient** over-allocation, not
an 8x wire or storage cost, and it must not be reported as one. It is worth a one-line fix plus a
test mainly because it is a trap that compounds defect 1: anyone round-tripping
`from_bytes(to_bytearray())` gets 8x the chunks and a key that compares unequal to the original.

### The exactness tier was wrong, and this is a correction to the plan

The plan classified the `Basis` store as tier 1, gated by equality rather than by the
error-budget harness. That holds for the **representation swap** itself. It does **not** hold for
the follow-on: `_PY_BASIS_OVERHEAD_BYTES` is calibrated to the current representation and feeds
RAM-derived `truncation_threshold` defaults, so recalibrating it changes the cap, which changes
**which determinants are admitted**, which moves energies on capped runs. That is tier 3 by
effect, not tier 1.

**Resolution: two commits, two gates.** The representation swap ships tier-1 under equality
(`e0` bit-identical, identical basis contents, cap held fixed). The recalibration ships
separately, tier-3, through the Phase-3 harness with before/after numbers. Folding them into one
commit would launder a selection-changing effect through an "exact by construction" claim.

### Cost and surface, recorded honestly

- **Not encapsulated today**: ~10 production files reach past the public API into `local_basis`
  or `_index_dict` (`basis_transcription.py`, `cipsi_solver.py`, `basis_restrictions.py`,
  `basis_split.py`, `gf_primitives.py`, `gf_shift_recycling.py`, `gf_solvers.py`, plus
  `BiCGSTAB.pyx` and `ChebyshevFilter.pyx`), and ~30 test files. The public map/sequence surface
  itself *is* representation-independent and already has an oracle
  (`test/basis/test_basis_storage.py`) that should pass unmodified.
- **The real performance risk**: `build_sparse_matrix` does O(nnz) index lookups per matrix build
  (`basis_transcription.py:184-199`). Moving those from O(1) average to O(log n) is a measurable
  cost on the hottest assembly path and must be benchmarked before shipping, not argued.
- **Insertion is not a regression**: `add_states` already merges two sorted sequences
  (`:281-282,300,309`), and the C++ layer has the matching O(n+m) bulk primitive (`merge_keys`,
  `ManyBodyBlockState.h:634-651`). Reuse that idiom, not `insert_row`'s O(n^2) one.
- **Precedent exists**: `_CappedBasisProxy` (`gf_primitives.py:284-449`) already does all its
  bookkeeping in the C++ sorted-key layer in production.

## Status

Phase 0 complete. 0a was re-measured after review found it priced the wrong container; the
pre-registered ordering check ran in reframed form at 0d (reaching production `p` locally is
time-bound, not memory-bound, so the experiment tested the mechanism instead), and the `Basis`
key store shipped at 0e-0j.

Phase 1: P1-1 (both halves), P1-3 (the residual block), P1-4 (the residual block is no longer
formed), P1-5 (the impurity-RDM guard and its entry packing), P1-2 (the `local_basis` iteration
sites) P1-6 (the never-read left-singular block, which turned an OOM into a 623.7 MiB run)
P1-7 (the fused unpack's duplicate amplitude copy, plan item 3), P1-8 (the basis split's wire
payloads) and P1-9 (`best_basis`) are fixed. Open and unmeasured: the split payload carved out of P1-2, and
`best_basis` (P1-3 in the refinement sense, `cipsi_solver.py:1315`). Path **D** is now open rather than unexamined: P1-7 was its first finding, and plan items 4
(`ManyBodyOperator::apply`'s `num_threads^2` accumulators) and 5 (no `shrink_to_fit` anywhere in
the layer — confirmed absent by grep, unpriced) remain. Path **C** (double counting) has still
had no pass. Plan item 8's other half -- `max_colors_within_budget` not accounting for the split
replicating the full basis into every color -- is untouched; P1-8 cut the payload, not the
replication. Also open: the remaining simultaneous copies in `cartan_subalgebra` (`generators` and `herm`, measured in P1-6), and the
dense generator list in `discover_one_body_symmetries` — `O(n_orb^3)` replicated per rank, but
**17%** of the symmetry path's peak rather than the 61% previously recorded (P1-6 corrects that
figure and the call path it was measured on).
