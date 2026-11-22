# The GF support pipeline: where a determinant may enter, and where amplitude is cut

Every truncation in the GF/spectra solvers belongs to one of two categories. Keeping them apart is what makes the
interactions predictable.

## For users: what the solver does and what its messages mean

Each Green's-function calculation is split into *units* (one orbital block, one side -- adding or removing an
electron -- and one group of thermal states). A unit explores the many-body states the added or removed electron can
reach, keeps the ones that matter and discards the rest. When the run finishes, one line per unit tells you how many
determinants it kept and how it ended:

| What the line says | What it means | What to do |
|---|---|---|
| a number of determinants, nothing else | the unit ended normally: it kept what mattered and the result converged | nothing |
| `frozen at the cap` | the unit wanted more determinants than the determinant cap allows; its result is truncated | raise the cap, or keep refinement on (`gf_refine_rounds`) so the cap is spent on the determinants that matter |
| `stopped at the memory limit` | the unit ran out of the memory the run was allowed; its size is what fitted | give the run more memory (more nodes, or fewer ranks per node: replicated data is paid once per rank on a node), then re-run |

If the self-energy is not causal (a spectral function with negative weight), the run stops and prints which orbital
block fails, at which frequency and by how much, followed by a *likely cause* and *what to change*. Units that hit a
limit are named first because they are the usual cause; otherwise the determinant sets were cut too coarsely near a
weakly coupled bath level, and lowering `gf_prune_tol` (or `slater_min`) restores them. A larger broadening `delta`
always makes the problem easier at the price of resolution.

The sections below describe the mechanism in the terms a developer needs.

## Category S — support decisions

A support decision says which determinants may carry amplitude at all. For the recurrences (block Lanczos,
BiCGSTAB/GMRES matvec) there are exactly two entry points on the policy object:

- `admit_seed(block)`: the seed block. No cut.
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
5. **Project** onto the mask.

Steps 3–4 are skipped once the support is frozen.

Plain bases (no cap) get a null policy that only does step 1.

### There is no support settle any more

Earlier versions stopped a sparse unit's growth once the frontier amplitude stayed below a threshold for 16 blocks
(`gf_settle`, `gf_settle_ratio`, `gf_settle_retry`, `FreezeRule`). It was removed in 2026-11: without pruning it
degenerated into breadth-first filling of whatever room the cap left, and with pruning it never ran. Callers that
used it (spectra, off-diagonal and susceptibility Green's functions, any run that did not
ask for pruning) now get pruned growth at `DEFAULT_GF_PRUNE_TOL` (1e-3), which stops where the amplitude rule says
the frontier no longer carries weight. `gf_prune_tol = 0` runs the unpruned growth to the cap or to closure. An input
that still carries a removed option fails with the reader's unknown-key error.

If a self-energy fails the causality check, `calc_selfenergy` attributes it per block (cap or guard state, the
nearest bath pole weighted by the block's own hybridisation) and says which knob to turn. Before that, a block that fails although some of its units were not limited by the cap or the memory is re-solved once with a finer selection (`RefineSpec.grow_tol` x0.1, twice the rounds) and re-judged.

## Category A — amplitude cuts inside an already-decided support

These never add rows; they are left in place on purpose:

- the 5a prune of the orthonormalised `q_next`;
- the TSQR `Q` prune;
- the BiCGSTAB/GMRES `axpy`/residual/solution prunes;
- the resolvent's per-column `combine`.

## What crosses a step boundary, by design

- The 5a cut acts on the next block (`q_next.prune_rows(slaterWeightMin)`) and never adds rows.
- BiCGSTAB's switch to the frozen CSR (`_FrozenShiftedSystem`) is a solver switch, not a truncation.

## Growth registered outside the policy

BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states` and the post-solve `psi2.prune`, and the raw-basis residual
probes (`residual_blocks`, `boundary_residual`, `_next_shell`). `gf_shift_recycling._expand_to_closure` calls
`apply_block` directly; it is serial-only (the caller declines `comm.size > 1`), where cutting before and after the
sum agree.

## Where it lives

`gf_primitives.SupportPolicy` (the old `_CappedBasisProxy` name is an alias) holds the three entry points; the one
recurrence matvec is `apply_and_redistribute` (`src/cython/_block_ops.pxi`), which the block-Lanczos step, the
BiCGSTAB/GMRES matvec all call. A plain basis is the null policy: redistribute, then cut.
The policy advertises `tracks_support` for callers to read; nothing tests its class. (The Green's-function
recurrence's own policy, below, also advertises `prunes`.)

| site | category | when | notes |
|---|---|---|---|
| `apply_and_redistribute` -> `SupportPolicy.decide` | S | every recurrence matvec | cut, admission, cap |
| `SupportPolicy.admit_seed` | S | seed block | no cut |
| `SupportPolicy.transport` | S (memory only) | per chunk | frozen projection, RSS guard |
| `_lanczos_step` 5a `q_next.prune_rows` | A | after the step | never adds rows |
| TSQR `Q` prune, BiCGSTAB/GMRES `axpy`/residual/solution prunes | A | inside a step | never add rows |
| `KrylovShiftedResolvent` `combine` | A | per column | deliberate |
| BiCGSTAB `_SupportTracker.add_states`, RIXS `add_states`/`psi2.prune` | growth outside the policy | between solves | |
| `residual_blocks`, `boundary_residual`, `_next_shell` | probes on the raw basis | | read, never register |
| `gf_shift_recycling._expand_to_closure` | direct `apply_block` | serial only | cut before and after the sum agree |

## The Green's-function recurrence on the discovered-key matrix

The block-Lanczos Green's function does not use the policy above for its sparse units. It
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

A step admits a new determinant only if its largest amplitude on the normalised step block exceeds `gf_prune_tol`. The seeds' first H-shell is kept whole
(`gf_first_shell_tol` relaxes it), which keeps the moments of `G` through `H^2`, hence the self-energy tail, exact.
A deferred row is **not banned**: it stays in `F` with its matrix entries and enters a later step when a stronger path
reaches it. After the growth the recurrence restarts from the seeds on the final support, so the result is the exact
block Lanczos of `P H P`. `0` switches it off.

It changes the retained set only where the cap leaves a choice: when the cap overflows in the first growth step (the
seeds plus part of the first shell fill it), the cap's own top-K selection is already amplitude-ordered and pruning
selects the same set. The synthetic metal of `doc/plans/gf_relevant_support.md` section 10 shows this at every cap
where breadth-first fails there.

**The growth pass.** Only the cap and the memory guard freeze the support (the frozen continuation
then runs and is discarded by the restart): the growth pass runs the
operator until the Green's-function monitor converges at the full tolerance, and that `G` is then discarded by the
restart on the final support (the exact block Lanczos of `P H P`). The growth pass has no early stop of its own: the
options `gf_growth_tol` (a looser tolerance for the growth monitor), `gf_growth_quiet` (end the growth after a window
of blocks that admitted nothing, default 64) and the internal stop at the freeze were removed in 2026-11 as part of the
redesign to one path. They were a measured speedup (metal: 474 -> 48 growth blocks at the same support with a window of
16; LMO Mn8: 2380 growth blocks against 2181 restart blocks, the growth admitting 2.4 % of the final support after
block 1); the speedup is given up on purpose and a replacement stop rule is to be tuned against the monitor. The unit
line prints `growth=<blocks> blk/<s> s, restart=<blocks> blk/<s> s` so a run shows the split.

**The admission rule.** One rule: a frontier determinant is admitted when one block's amplitude squared clears
`gf_prune_tol**2`. The `accumulated` rule (the weight summed over the blocks that reach a determinant) and the `mass`
rule (a fraction of the step's frontier weight), and the option `gf_prune_rule` that chose between them, were removed
in 2026-11, as part of the redesign to one admission path. `accumulated` found the support earlier (so a short quiet
window trimmed less) but is ω-blind like `amplitude`, so it did not address the resonance defect, and it cost a
deferred-weight store per unit. A deferred determinant that matters at a pole of the bath is recovered by the grow mode
of the refinement below.

### Second-order growth (`gf_pt2_tol`, opt-in)

`gf_pt2_tol` replaces the amplitude rule (setting both is an error). After the `slaterWeightMin` cut and the whole first
shell, a frontier determinant `f` gets

    s_f = max_k  sum_cols |rho_f(z_k)|^2 / |z_k - H_ff| / ||G_P(z_k)||,    rho = H X(z),  X = (z - H_PP)^-1 seed,

over 31 uniform points `z_k` across the unit's own real-frequency window at the run's own broadening, and the growth
admits rows in descending score until the score mass left out is at most `gf_pt2_tol` (one collective tail cut; at the
cap the rows are ranked by score instead). There is no hand-placed frequency. `rho` costs no extra application of `H`:
a `ShiftedGalerkin` over the K rows is stepped with the matvec output `H v`, which is linear in the block (module
`gf_growth_score`). The frontier is scored on every 4th block (`PT2_SCORE_EVERY`); a row first reached at a block
scores zero until the next scored block and is never banned. A mesh point that carries under 1e-3 of the score mass on
each of the first 10 scored blocks is dropped from the accumulator. Growth ends at twice the block of the last
admission, and at least 16 quiet blocks later; the restart on the final `P` is unchanged, so the retained set does not
depend on that stop.

Measured meaning of the budget: the relative error `max|dG| / max|G|` of the unit. Scoring every block it stayed inside
the budget on everything measured; every 4th block reaches 1.6x it on paramagnetic NiO. It is the better rule per row
where a resonant minority of frontier rows exists (SMO, NiO) and neutral in rows on a metal (FCC Ni, where most of the
frontier is resonant in the window), at a higher cost per row: on FCC Ni, 1.4-2.3x the wall at equal rows. See
`doc/plans/pt2_growth_rule.md` for the measurements.

Limits. A unit with no real-frequency evaluation mesh (Matsubara only), or one whose matrix mode is not `discovered`,
falls back to the amplitude rule with `gf_prune_tol = 1e-3` and says so in its log. The re-solve of a failing block
(`gf_prune_tol`) does not run under `gf_pt2_tol`. Refinement acts on cap-frozen units only. Unmeasured: complex
Hamiltonians, chunks wider than 2 states, cap-bound units other than SMO unit 7, Sigma on NiO and FCC Ni, NiO PM above
about 50k rows, rank-count invariance on real archives, memory on NiO and FCC Ni.

### Swap refinement (`gf_refine_rounds`, default 3)

For a unit the cap froze, up to `gf_refine_rounds` rounds of scoring the retained determinants (what their removal
costs `G`) and the discovered ones (what admitting them gains, Epstein-Nesbet) on the matrix, swapping the least
important retained for the most important discovered at the fixed cap, then one recurrence on the final support. A
round stops when the estimated remaining gain falls below `gf_refine_tol` of `max|G|`. A unit the cap does not bind, a
unit the memory guard froze, and a unit that is not in discovered mode are left alone (and the log says which). Cost:
the scoring solves are the expense (about 2.7x the breadth-first Green's-function phase on the capped NiO units).
`0` switches it off.

### Defaults

`SolverOptions` leaves both options unset, and an unset option is on
(`model.DEFAULT_GF_PRUNE_TOL`, `DEFAULT_GF_REFINE_ROUNDS`); `0` is the explicit off. An archive written before the
options existed reads as off, so a replay of an old run reproduces it.

## The `exhausted` status

A frozen, unreorthogonalized recurrence on a support `P` cannot reach a tolerance below its rounding floor
(about `eps*|H|*k/eta^2`, growing with the block count `k`). It used to run until the divergence guard fired, at
2.5-5x `|P|` vectors. All three recurrence sites (the Python loop of `block_lanczos_discovered`,
`_continue_on_array_kernel`, `gf_solvers._array_block_lanczos`) now stop with status `"exhausted"` once
`sum(widths) >= EXHAUSTED_VECTORS_OVER_P * |P|` (`gf_convergence.krylov_exhausted`, replicated on every rank).
The status reaches `check_lanczos_convergence` as a WARN naming the rounding floor and the last relative change;
compare that with `gf_tol` / `gf_real_tol`. The array continuation budgets in vectors, not blocks.
