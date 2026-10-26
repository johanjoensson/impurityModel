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
