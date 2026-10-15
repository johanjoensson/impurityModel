# The GF support pipeline: where a determinant may enter, and where amplitude is cut

Every truncation in the GF/spectra solvers belongs to one of two categories. Keeping them apart is what makes the
interactions predictable.

## Category S — support decisions

A support decision says which determinants may carry amplitude at all. For the recurrences (block Lanczos,
BiCGSTAB/GMRES matvec, Chebyshev) there are exactly two entry points on the policy object:

- `admit_seed(block)`: the seed block. No cut, and it does not advance the support-freeze window.
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
6. **Support freeze:** update the window of (leak, cut). A verdict freezes from the *next* step.
7. **Project** onto the mask.

Steps 3–6 are skipped once the support is frozen.

Plain bases (no cap, no pruning) get a null policy that only does step 1.

## Category A — amplitude cuts inside an already-decided support

These never add rows; they are left in place on purpose:

- the 5a prune of the orthonormalised `q_next`;
- the TSQR `Q` prune;
- the BiCGSTAB/GMRES `axpy`/residual/solution prunes;
- the Chebyshev `t_next` prune;
- the resolvent's per-column `combine`.

## What crosses a step boundary, by design

- The 5a cut acts on the next block, so its weight reaches the *next* step's support-freeze window (`note_step_cut`).
- A support-freeze verdict applies from the next step.
- Outer admission (`GF_BICGSTAB_ADMISSION=outer`) selects between solves, not inside one.
- BiCGSTAB's switch to the frozen CSR (`_FrozenShiftedSystem`) is a solver switch, not a truncation.

## Growth registered outside the policy

BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states` and the post-solve `psi2.prune`, and the raw-basis residual
probes (`residual_blocks`, `boundary_residual`, `_next_shell`). `gf_shift_recycling._expand_to_closure` calls
`apply_block` directly; it is serial-only (the caller declines `comm.size > 1`), where cutting before and after the
sum agree.

## Mutual exclusions

Importance pruning (`GF_LANCZOS_ADMIT_TOL`) and the support freeze cannot both be on: under
pruning, banned rows always land outside the support, so the leak no longer measures growth.

## Where it lives

`gf_primitives.SupportPolicy` (the old `_CappedBasisProxy` name is an alias) holds the three entry points; the one
recurrence matvec is `apply_and_redistribute` (`src/cython/_block_ops.pxi`), which the block-Lanczos step, the
BiCGSTAB/GMRES matvec and the Chebyshev filter all call. A plain basis is the null policy: redistribute, then cut.
The policy advertises `tracks_support` and `prunes` for callers to read; nothing tests its class.

| site | category | when | notes |
|---|---|---|---|
| `apply_and_redistribute` -> `SupportPolicy.decide` | S | every recurrence matvec | cut, ban filter, admission, cap, support freeze |
| `SupportPolicy.admit_seed` | S | seed block | no cut, not a window step |
| `SupportPolicy.transport` | S (memory only) | per chunk | frozen projection, RSS guard |
| `_lanczos_step` 5a `q_next.prune_rows` | A | after the step | weight reaches the next window (`note_step_cut`) |
| TSQR `Q` prune, BiCGSTAB/GMRES `axpy`/residual/solution prunes, Chebyshev `t_next` prune | A | inside a step | never add rows |
| `KrylovShiftedResolvent` `combine` | A | per column | deliberate |
| BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states`/`psi2.prune` | growth outside the policy | between solves | |
| `residual_blocks`, `boundary_residual`, `_next_shell` | probes on the raw basis | | read, never register |
| `gf_shift_recycling._expand_to_closure` | direct `apply_block` | serial only | cut before and after the sum agree |

Importance pruning (`GF_LANCZOS_ADMIT_TOL`) decides inside `decide` like the cap does, from the new rows taken
*before* the `slaterWeightMin` cut, so the cut rows are banned too and row chunking cannot move the decision.
