# Large-support block apply: measurements and decisions (2026-10-26)

Question: why is the MBS Green's-function matvec so much slower than the CSR path once the support
exceeds ~1e6 rows, and what can be sped up? Rig: `test/support/freeze_replay.py run ... --freeze off` at
`-n 1` with `IMPMOD_MATVEC_PROF=1` (stage table to stderr every 30 s, `_block_ops.pxi`), plus
`py-spy record --native` on the running rank.

## Where a step goes (single rank)

| archive | support | apply | chunk merge | decide/admission | redistribute |
|---|---|---|---|---|---|
| SMO cubic, iter 1 (first GF unit) | 87k | 96.6% | 2.0% | 0.8% | 0.4% |
| NiO 10 bath linked chain, iter 1 | 1.2-1.5M | 96.0% | 2.5% | 1.2% | 0.0% (one rank) |

Inside the apply (py-spy native, symbols of the inlined loop are not resolved further):

| | SMO 87k | NiO 1.3M |
|---|---|---|
| `ManyBodyOperator::apply` (term loop, create/annihilate, hash accumulate) | 69% | 73% |
| output sort (`introsort_loop`) | 9% | 7.7% |
| restriction check (`state_is_within_restrictions`) | 7.5% | (removed) |
| unresolved libc symbol | 7% | 12% |

Hypothesis (i), sorting, is real but small (8-9%) even at 1.3M rows. Hypothesis (ii), finding determinants
outside the basis, is not a cost: admission is 1% (linear merges over sorted flat buffers).
The cost is the per-row term loop and the accumulator.

## What was done

| change | effect |
|---|---|
| `from_keys_and_amps` adopts the flat key store | removes N heap keys x2 per step (not separately visible) |
| flat key arena + permutation sort in `apply` (serial and threaded) | synthetic 2.8M-row output: 2.48 s -> 1.31 s |
| k-way merge of sorted sender segments in `unpack_block_fused` | 8% on the unpack microbench (sort is not what dominates the unpack) |
| restriction check as row value + per-term delta | removes the 7.5% bucket |
| create/annihilate only where an occupancy mask says the term fires | with the above: apply 2.42 -> 1.65 us/row on SMO (1.47x) |

## Not done, and why

* 1d (fuse admission passes), 1e (chunk merge): 1.2% and 2.5% of a step; below the 10% bar.
* Pattern-indexed term table (Phase 2): SMO has only 100-270 terms and the firing mask removed most of the
  per-term cost; revisit only if a profile with the new build still shows the term loop dominating.
* Multi-rank exchange cost is not measured here (one rank). The unpack at many senders is the only
  place the merge matters.

# Part 2: inside the apply (2026-10-26)

Part 1 left the apply at ~96% of a step. This part looks inside it, on the real NiO 10-bath solver
Hamiltonian (495 flat terms: 132 impurity off-diagonal 2-body, 55 impurity density, 96 bath density,
172 bath hops, 40 hybridisation hops) under the excited-sector restrictions the run logged, applied to the
103k-row support the matvec reaches from the d8 seed. `test/operators/test_apply_nio_bench.py` (opt-in,
`-m benchmark`) reproduces every number below.

## Where the time was

Applying subsets of the operator showed ~4.4-5.8 ns per term *visited*, whether or not it fires, at 103k and
at 1.3M rows alike (so not memory latency). Per row only ~146 of 495 terms fire (128 density terms that only
add to the diagonal, 12 hops and 6 impurity 2-body terms that emit), so ~350 visits did nothing. A one-term
operator costs 0.06 us/row, i.e. the fixed per-row cost is small.

## What was done (block widths 1 / 2 / 4, serial release build, us per input row)

| step | p = 1 | p = 2 | p = 4 |
|---|---|---|---|
| start of part 2 | 2.27 | 2.52 | 2.84 |
| firing bitmap over all terms (structure-of-arrays masks, padded to 64 terms), visit only set bits; one row kernel for serial and threaded | 1.70 | - | - |
| arena: row counter, inline compare, one-multiply hash; emissions queued and prefetched per row | 1.44 | 1.57 | 1.84 |
| density diagonal as one coefficient sum per row | 1.21 | 1.23 | 1.29 |

Whole step, NiO iteration 1 at one rank, 1.16M-row supports (`IMPMOD_MATVEC_PROF=1`, same rig as part 1):
3.15 -> 1.37 us per input row for the matvec, apply 3.08 -> 1.26 us/row, i.e. 2.3x. Apply is now 91.7% of the
step, chunk merge 5.0%, decide/admission 2.6%.

Threaded build (4 threads): 0.99 us/row at p = 1. The `safe` build (scalar flag packing, no AVX2) gives the
same output and 1.66 us/row.

Output is bitwise identical to the old code on the same input for the first two steps; the third reassociates
the diagonal sum (3e-13 absolute on amplitudes up to ~1e2). `diagonal()` accumulates the density terms the
same way, so it still equals `apply({D: 1})` bit for bit (checked on 1500 NiO determinants).

## Findings along the way

* **The tests compared the code with itself.** `apply`, `__call__` and `apply_multi` all forward to the one C++
  apply, so every "block == scalar" test was a self-comparison. `test/operators/apply_reference.py` is a
  pure-Python reference (set-of-orbitals determinants, operator-by-operator) and the old tests now use it;
  `test_apply_reference.py` adds fixtures for term counts around the 64-term word boundary, orbitals across
  the 63/64 chunk boundary, raw strings that revisit an orbital, the constant term, and restrictions that
  reject some input rows.
* **A latent out-of-bounds write.** A one-body hop whose created orbital lies beyond the determinant width
  passed the old `!bit_set(i)` test (false out of range) and then toggled a bit past the end of the key; under
  ASan it is a heap-buffer-overflow in `toggle_bit`, in release it silently emitted a wrong row. In the mask
  form such a hop is dead. Not reachable from the shipped solvers (their keys cover every orbital), but a
  caller-built operator on a narrow block could hit it.
* **A false alarm worth recording.** Comparing outputs of two builds on blocks generated by each build is
  invalid: rounding residues of ~1e-56 survive `cutoff = 0` or not depending on summation order, which changes
  the generated support and everything after it. Compare on one fixed set of keys.

## Not done, and why

* **Output sort** (comparison sort of ~197k keys, ~0.12 us/row of the ~1.2): sorting contiguous 24-byte
  key+row records instead of a permutation measured no gain, so it was reverted. A radix sort could save at
  most ~5% of the apply.
* **Threaded sort/merge, ratio-based arena sizing:** not measured to matter; the threaded path scales
  1.25x on 4 threads here, bound by the merge and the final sort.
* What remains is the emission itself (~12 hops + 6 two-body terms per row, ~50 ns each: hash, probe, two
  random reads) and the sort. Both are inherent to a hash-accumulate-then-sort apply.

# Part 3: the emission and the sort (2026-10-26)

Target: supports that keep growing (NiO: 202 Lanczos steps, 1.6M -> 2.4M rows, never settles) and units whose
frozen CSR does not fit; there the MBS apply is the whole unit. NiO fixed-key bench (103k rows, p = 1 / 2 / 4,
us per input row, release, idle machine), baseline `oracle_base2` (bitwise reference for every row below).

| step | p = 1 | p = 2 | p = 4 | outcome |
|---|---|---|---|---|
| start of part 3 | 1.13 | - | - | |
| C1: input keys pre-inserted in the arena, sort only the new rows, merge the two runs | 1.08 | 1.03 | 1.15 | kept, bitwise identical |
| C2: emission queue carried across rows (source row + diagonal snapshot per entry) | 1.13 | 1.21 | 1.15 | rejected, no gain |
| C5: key and amplitudes in one record per arena row | 1.04-1.09 | 1.07-1.11 | 1.16-1.19 | rejected, ~2% at 552k rows |
| C3: one k-way merge of the chunk outputs | not built | | | rejected, see below |

* **The emission is not memory-latency bound.** At 552k input rows (841k output rows) C1 costs 1.05-1.07
  us/row and C5 1.04-1.05, the same per-row cost as at 103k rows, so merging the three dependent touches into
  one record and overlapping more probes (C2) buy nothing measurable. What is left is arithmetic and branches
  of the ~18 emissions per row, not cache misses.
* **C3 trades memory for 3%.** The chunk merge is 4-5% of a step; one k-way merge would save about two thirds
  of it, but it keeps all `APPLY_ROW_CHUNKS` pieces alive until the end (today: `W` plus one piece). The chunking
  exists to bound that peak, so it is not worth ~3%.
* **Multi-rank exchange is not a cost.** 3-rank NiO replay (`--freeze on`, 76-82 steps, up to 1.96M rows/step):
  apply 90.5-91.7%, chunk merge 4.2-4.9%, decide 3.7-4.2%, `chunk_redistribute` 0.0%. Lever D dropped.
* Rejected patches are kept out of the tree; the apply is now ~1.08 us/row, ~4% under part 2. No further
  apply-internal lever above ~5% is known. The remaining several-fold lever for growing supports is E (an
  incremental matrix during the growth phase), which changes the Lanczos step rather than the apply.
