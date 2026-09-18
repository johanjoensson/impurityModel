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

Phase 0 items 0a-0c complete, 0a re-measured after review found it priced the wrong container.
Remaining Phase 0: the high-`p` local rig and the pre-registered ordering check. Nothing
implemented yet; no production code changed.
