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
3. **Admission rule.** New rows are admitted while the cap allows.
4. **Cap:** top-K by maximum column `|amp|^2`, then freeze.
5. **Support settle:** not in `decide` itself but in the same step, one call later: the Lanczos step hands the policy the
   normalised next block before its cut (`SupportPolicy.note_next_block`), which completes the step's record and
   takes the `FreezeRule` verdict (below). A verdict settles the unit from the *next* step.
6. **Project** onto the mask.

Steps 3–5 are skipped once the support is frozen or settled.

Plain bases (no cap) get a null policy that only does step 1.

### The settle rule (`gf_primitives.FreezeRule`)

A unit **settles** when, over `SETTLE_WINDOW` = 16 steps, the largest `|amp|^2` the normalised next block `q_{k+1}`
carries on a row that was new at step `k` is at most `(ratio x slaterWeightMin)^2` *and* the least-squares slope of
`log10 max(|amp|^2, (1e-3 swm)^2)` over the window is not positive (a heuristic against a monotonically growing
trickle). Measured on the TSQR output before the cut acts, so `ratio 1` means "every excluded row would have been
pruned from the next block anyway" and `ratio 0` (or `swm = 0`) is closure. From the settle on the unit is the exact
Lanczos of `P H P` on the CSR (reort none): it drops the rows whose next-block amplitude stayed at or below the bound
over the window. That is not an error bound on `G`; the default ratio is chosen by measurement. Nothing checks the frontier
afterwards except the end-of-unit probe (`gf_recurrence.settled_frontier`, opt-in with `GF_SUPPORT_TRACE`). `gf_settle_ratio`
(default 3) sets the ratio; `gf_settle` switches the rule off. The records are sums over ranks and
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

Importance pruning (`gf_prune_tol`, below) and the support settle cannot both be on: the rows the settle rule
measures (new rows in the next block) are already decided by the pruning rule, and settling on them would stack two
admission criteria. Under pruning the settle stays off; a settle option the run asks for explicitly (`gf_settle`
on, `gf_settle_ratio`, `gf_settle_retry`) turns the default pruning off instead.

## Where it lives

`gf_primitives.SupportPolicy` (the old `_CappedBasisProxy` name is an alias) holds the three entry points; the one
recurrence matvec is `apply_and_redistribute` (`src/cython/_block_ops.pxi`), which the block-Lanczos step, the
BiCGSTAB/GMRES matvec all call. A plain basis is the null policy: redistribute, then cut.
The policy advertises `tracks_support` for callers to read; nothing tests its class. (The Green's-function
recurrence's own policy, below, also advertises `prunes`.)

| site | category | when | notes |
|---|---|---|---|
| `apply_and_redistribute` -> `SupportPolicy.decide` | S | every recurrence matvec | cut, admission, cap |
| `SupportPolicy.note_next_block` | S | every recurrence step (settle on) | next-block frontier record, settle verdict |
| `SupportPolicy.admit_seed` | S | seed block | no cut, not a window step |
| `SupportPolicy.transport` | S (memory only) | per chunk | frozen projection, RSS guard |
| `_lanczos_step` 5a `q_next.prune_rows` | A | after the step | weight reaches the next window (`note_step_cut`) |
| TSQR `Q` prune, BiCGSTAB/GMRES `axpy`/residual/solution prunes | A | inside a step | never add rows |
| `KrylovShiftedResolvent` `combine` | A | per column | deliberate |
| BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states`/`psi2.prune` | growth outside the policy | between solves | |
| `residual_blocks`, `boundary_residual`, `_next_shell` | probes on the raw basis | | read, never register |
| `gf_shift_recycling._expand_to_closure` | direct `apply_block` | serial only | cut before and after the sum agree |

## The Green's-function recurrence on the discovered-key matrix

The block-Lanczos Green's function (`gf_method = "lanczos"`) does not use the policy above for its sparse units. It
runs `gf_recurrence.block_Green_discovered`: one recurrence step (`lanczos_step`) over a matrix `M` that holds, per
admitted determinant `D` (the retained set `P`), the **complete** restricted image `H|D>` over every determinant any
admitted one has reached (`K ⊇ P`). `M V` for `V` on `P` is exactly `H V`; the rows in `F = K \ P` are the step's
frontier amplitudes on which admission decides. Admitting `D` costs one apply of `H` to `|D>`, so the apply work of a
unit is one apply per determinant ever admitted, not one per step per retained determinant.

The matvec mode is decided once per unit, collectively, from the priced memory (`choose_matrix_mode`, printed with
`-v`): **discovered** (the matrix above; needed by swap refinement), **retained** (the `P H P` matrix built at the
freeze), or **none** (the operator is applied every block). Admission, the cut semantics, TSQR and the convergence
monitor are the same code in all three.

### Pruned growth (`gf_prune_tol`, default 1e-3)

A step admits a new determinant only if its largest amplitude on the normalised step block exceeds `gf_prune_tol`
(`gf_recurrence._PRUNE_RULE = "mass"` reads it as a fraction of the frontier weight instead). The seeds' first H-shell is kept whole
(`gf_first_shell_tol` relaxes it), which keeps the moments of `G` through `H^2`, hence the self-energy tail, exact.
A deferred row is **not banned**: it stays in `F` with its matrix entries and enters a later step when a stronger path
reaches it. After the growth the recurrence restarts from the seeds on the final support, so the result is the exact
block Lanczos of `P H P`. Pruning is exclusive with the settle. `0` switches it off.

It changes the retained set only where the cap leaves a choice: when the cap overflows in the first growth step (the
seeds plus part of the first shell fill it), the cap's own top-K selection is already amplitude-ordered and pruning
selects the same set. The synthetic metal of `doc/plans/gf_relevant_support.md` section 10 shows this at every cap
where breadth-first fails there.

**The growth pass and its tolerance (`gf_growth_tol`, default off).** Under pruning the settle is off, so nothing
freezes the support: the growth pass runs the operator matrix-free until the Green's-function monitor converges at the
full tolerance, and that `G` is then discarded by the restart (which runs on the stored `P H P`). On a unit where the
support is found in the first few blocks this is most of the unit's wall (Mn8, bulk LMO: 2380 growth blocks against
2181 restart blocks, the growth admitting 2.4 % of the final support after block 1). `gf_growth_tol` gives the growth
pass its own, looser relative-change tolerance (never tighter than the run's own, per axis); a pass that stopped on it
always restarts, so the returned `G` is still the exact block Lanczos of `P H P` on the final support, to the full
tolerance. What the looser tolerance can change is the support: a determinant first reached after the early stop is
not admitted. The unit line prints `growth=<blocks> blk/<s> s, restart=<blocks> blk/<s> s` so a run shows the split.

Local sweeps (`freeze_replay run --prune-tol 1e-3 --growth-tol X`, synthetic metals from `write_metal`, U = 0.5,
uncapped, one rank). Blocks and seconds are summed over the units and split by pass; dSigma is the largest absolute
difference to the run with the option off, on the Matsubara / real axis. No row has an acausal point.

1 orbital, 6 bath orbitals (8 units):

| `gf_growth_tol` | growth blocks / s | restart blocks / s | wall | dSigma |
|---|---|---|---|---|
| off | 1666 / 1.2 | 1536 / 0.4 | 3.3 s | 0 / 0 |
| 1e-3 | 320 / 0.0 | 1536 / 0.4 | 2.0 s | 0 / 0 |
| 1e-4 | 436 / 0.0 | 1536 / 0.4 | 2.1 s | 0 / 0 |
| 1e-5 | 750 / 0.6 | 1536 / 0.4 | 2.4 s | 0 / 0 |
| 1e-6 | 947 / 0.6 | 1536 / 0.4 | 2.5 s | 0 / 0 |
| 1e-7 | 1237 / 0.8 | 1536 / 0.4 | 2.7 s | 0 / 0 |

2 orbitals, 4 bath orbitals each (32 units):

| `gf_growth_tol` | growth blocks / s | restart blocks / s | wall | dSigma |
|---|---|---|---|---|
| off | 4954 / 100.2 | 4613 / 19.6 | 229 s | 0 / 0 |
| 1e-3 | 1122 / 23.0 | 4807 / 18.0 | 151 s | 3.8e-9 / 3.5e-7 |
| 1e-4 | 1646 / 31.8 | 4593 / 18.3 | 162 s | 3.5e-9 / 1.2e-7 |
| 1e-5 | 2113 / 43.1 | 4521 / 18.3 | 171 s | 1.2e-11 / 1.7e-7 |
| 1e-6 | 2977 / 60.7 | 4634 / 19.6 | 191 s | 8.6e-12 / 4.1e-8 |
| 1e-7 | 3467 / 70.4 | 4569 / 19.4 | 201 s | 1.6e-13 / 4.4e-8 |

A growth block costs about 20 ms against about 4 ms for a restart block here, so the saving follows the growth block
count. The 1-orbital support is the same at every tolerance (Sigma bit-identical); in the 2-orbital model a looser
tolerance moves the support slightly (restart blocks change, dSigma up to 3.5e-7). Wall includes the ground state and
the other phases (about 110 s of the 2-orbital runs). These are not measurements of the LMO production workload,
which is cap-bound locally.

### Swap refinement (`gf_refine_rounds`, default 3)

For a unit the cap froze, up to `gf_refine_rounds` rounds of scoring the retained determinants (what their removal
costs `G`) and the discovered ones (what admitting them gains, Epstein-Nesbet) on the matrix, swapping the least
important retained for the most important discovered at the fixed cap, then one recurrence on the final support. A
round stops when the estimated remaining gain falls below `gf_refine_tol` of `max|G|`. A unit the cap does not bind, a
unit the memory guard froze, and a unit that is not in discovered mode are left alone (and the log says which). Cost:
the scoring solves are the expense (about 2.7x the breadth-first Green's-function phase on the capped NiO units).
`0` switches it off.

### Defaults

`SolverOptions` leaves both options unset, and an unset option is on for `gf_method = "lanczos"`
(`model.DEFAULT_GF_PRUNE_TOL`, `DEFAULT_GF_REFINE_ROUNDS`); `0` is the explicit off. An archive written before the
options existed reads as off, so a replay of an old run reproduces it.
