# The GF support pipeline: where a determinant may enter, and where amplitude is cut

Every truncation in the GF/spectra solvers belongs to one of two categories. Keeping them apart is what makes the
interactions predictable.

## Category S — support decisions

A support decision says which determinants may carry amplitude at all. For the recurrences (block Lanczos,
BiCGSTAB/GMRES matvec) there are exactly two entry points on the policy object:

- `admit_seed(block)`: the seed block. No cut, and it does not add a record to the settle window.
- `decide(wp, cutoff)`: once per matvec, on the **summed** output of that matvec.

Row chunking (`APPLY_ROW_CHUNKS`) only bounds the transient memory of the apply. Per chunk the policy does
`transport(raw)`: redistribute, then the work that commutes with the sum (the frozen `keep_rows`, the RSS
memory-guard check). Nothing that depends on the *total* amplitude of a row is decided per chunk.

### Order inside `decide`

1. **Measure and cut.** Take `new = keys_new_above(mask, 0)`, then `prune_rows(slaterWeightMin)` on the summed
   rows. The weight the cut drops is recorded here.
2. **Frozen:** `keep_rows(mask)`.
3. **Ban filter** (importance pruning only).
4. **Admission rule.** Pruning: new rows above `eta` are candidates; the first shell is judged by
   `first_shell_tol`; below-eta rows, and sub-`slaterWeightMin` new rows, are banned.
5. **Cap:** top-K by maximum column `|amp|^2`, then freeze.
6. **Support settle:** not in `decide` itself but in the same step, one call later: the Lanczos step hands the policy the
   normalised next block before its cut (`SupportPolicy.note_next_block`), which completes the step's record and
   takes the `FreezeRule` verdict (below). A verdict settles the unit from the *next* step.
7. **Project** onto the mask.

Steps 3–6 are skipped once the support is frozen or settled.

Plain bases (no cap, no pruning) get a null policy that only does step 1.

### The settle rule (`gf_primitives.FreezeRule`)

A unit **settles** when, over `SETTLE_WINDOW` = 16 steps, the largest `|amp|^2` the normalised next block `q_{k+1}`
carries on a row that was new at step `k` is at most `(ratio x slaterWeightMin)^2` *and* the least-squares slope of
`log10 max(|amp|^2, (1e-3 swm)^2)` over the window is not positive (a heuristic against a monotonically growing
trickle). Measured on the TSQR output before the cut acts, so `ratio 1` means "every excluded row would have been
pruned from the next block anyway" and `ratio 0` (or `swm = 0`) is closure. From the settle on the unit is the exact
Lanczos of `P H P` on the CSR (reort none): it drops the rows whose next-block amplitude stayed at or below the bound
over the window. That is not an error bound on `G`; the default ratio is chosen by measurement. Nothing checks the frontier
afterwards except the end-of-unit probe (`gf_solvers._settled_frontier`, opt-in with `GF_SUPPORT_TRACE`). `gf_settle_ratio` /
`GF_SETTLE_RATIO` (default 3) sets the ratio; `gf_settle` switches the rule off. The records are sums over ranks and
chunks, so a window within rounding of the threshold can settle a step earlier or later when the layout changes.

The default and its evidence: `doc/plans/gf_support_freeze_redesign.md` (acceptance rule: Matsubara change <= 1e-7,
pointwise change of Sigma_ii <= 0.1 x the causality guard's headroom; the chosen ratio is the largest measured whose
change stays below 1e-6 of the headroom). If a self-energy still fails the causality check, `calc_selfenergy`
attributes it per block (settled units, cap or guard state, the nearest bath pole, weighted by the block's own
hybridisation). With `gf_settle_retry on` (default off) and units that settled with a non-zero ratio, none held by a cap or
the guard, and a real-axis failure, it solves those units again once closure-only (`selfenergy._settle_retry`) and accepts
the result only if no re-solved unit was capped, held or unconverged. The closure-only re-solve can take as long as an
unsettled run (3,271 s against 280 s on the small SMO case), which is why it is opt-in.

## Category A — amplitude cuts inside an already-decided support

These never add rows; they are left in place on purpose:

- the 5a prune of the orthonormalised `q_next`;
- the TSQR `Q` prune;
- the BiCGSTAB/GMRES `axpy`/residual/solution prunes;
- the resolvent's per-column `combine`.

## What crosses a step boundary, by design

- The 5a cut acts on the next block, so its weight is recorded with the *next* step (`note_step_cut`, a trace column).
- A settle verdict applies from the next step.
- Outer admission (`GF_BICGSTAB_ADMISSION=outer`) selects between solves, not inside one.
- BiCGSTAB's switch to the frozen CSR (`_FrozenShiftedSystem`) is a solver switch, not a truncation.

## Growth registered outside the policy

BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states` and the post-solve `psi2.prune`, and the raw-basis residual
probes (`residual_blocks`, `boundary_residual`, `_next_shell`). `gf_shift_recycling._expand_to_closure` calls
`apply_block` directly; it is serial-only (the caller declines `comm.size > 1`), where cutting before and after the
sum agree.

## Mutual exclusions

Importance pruning (`GF_LANCZOS_ADMIT_TOL`) and the support settle cannot both be on: pruning bans every new row
below `eta` for good, so the rows the settle rule measures (new rows in the next block) are already decided by another
rule, and settling on them would stack two different admission criteria. Under pruning the settle stays off.

## Where it lives

`gf_primitives.SupportPolicy` (the old `_CappedBasisProxy` name is an alias) holds the three entry points; the one
recurrence matvec is `apply_and_redistribute` (`src/cython/_block_ops.pxi`), which the block-Lanczos step, the
BiCGSTAB/GMRES matvec all call. A plain basis is the null policy: redistribute, then cut.
The policy advertises `tracks_support` and `prunes` for callers to read; nothing tests its class.

| site | category | when | notes |
|---|---|---|---|
| `apply_and_redistribute` -> `SupportPolicy.decide` | S | every recurrence matvec | cut, ban filter, admission, cap |
| `SupportPolicy.note_next_block` | S | every recurrence step (settle on) | next-block frontier record, settle verdict |
| `SupportPolicy.admit_seed` | S | seed block | no cut, not a window step |
| `SupportPolicy.transport` | S (memory only) | per chunk | frozen projection, RSS guard |
| `_lanczos_step` 5a `q_next.prune_rows` | A | after the step | weight reaches the next window (`note_step_cut`) |
| TSQR `Q` prune, BiCGSTAB/GMRES `axpy`/residual/solution prunes | A | inside a step | never add rows |
| `KrylovShiftedResolvent` `combine` | A | per column | deliberate |
| BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states`/`psi2.prune` | growth outside the policy | between solves | |
| `residual_blocks`, `boundary_residual`, `_next_shell` | probes on the raw basis | | read, never register |
| `gf_shift_recycling._expand_to_closure` | direct `apply_block` | serial only | cut before and after the sum agree |

Importance pruning (`GF_LANCZOS_ADMIT_TOL`) decides inside `decide` like the cap does, from the new rows taken
*before* the `slaterWeightMin` cut, so the cut rows are banned too and row chunking cannot move the decision.
