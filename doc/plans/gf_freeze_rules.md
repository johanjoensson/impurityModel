# GF support freeze: tolerance-referenced rules

**Status (2026-10-16):** Phase 0 (diagnostics) and the trial-solve freeze prototype are on branch `gf-freeze-rules`,
default off. Phase 1 (numpy oracle study) is done. **Phase 2 (the local SMO replay) and Phase 2b (a whole-Sigma
workload) are NOT done**, so nothing here is evidence about `Sigma_real` on a production model, and the default is
unchanged. The cluster A/B decides.

## Problem

The shipped freeze compares the weight reaching new rows (leak) with the weight `slaterWeightMin` cuts. At a
production cutoff of 1.5e-8 the cut is ~2e-16 per row, so the rule waits for the support to close, while the stopping
tolerances (`gf_real_tol` 1e-4) already allow errors five orders larger than the freeze costs (<= 4e-8 in G).

## Candidates

1. **Trial solve** (`GF_FREEZE_TRIAL_WINDOW`): every W blocks solve `P_k H P_k` to convergence from the live tail; freeze
   when consecutive trials agree to `GF_FREEZE_BUDGET` x tolerance per axis. The accepted trial is the answer.
2. **Ban rows cut by `slaterWeightMin`**: `GF_LANCZOS_ADMIT_TOL = slaterWeightMin` (a row reached and not admitted is never admitted later).
3. **Amplitude eta** (10 and 100 x swm), same mechanism.

## Oracle

`debug/gf_freeze_rules/oracle.py` (gitignored): scalar Lanczos with the production cut, admission, ban, freeze and
monitor semantics. Checked against the real code on the 14-orbital chain: leak and cut histories agree to 1e-13, the
freeze step (30) and the retained sizes (shipped 1528, ban 617) are identical. On the 7-bath SIAM, with no cut and
no freeze, the summed `G` matches `reference_G` to 1e-13 and `Sigma` to 1e-12.

## Results

**Metal SIAM, 7 and 9 baths** (52,920-determinant sectors at 9): the sector fills completely before any rule can
save anything; shipped and trial both end at the closure. Ban and eta shrink the support by 20-30% and cost
`Sigma_real` 4e-2 to 1e-1 (n_b=9, swm 1e-8) against an unfrozen error of 2e-3. **Banning at first encounter is not safe on a metal.**

**Chain, 48,620 determinants** (the shipped rule freezes before closure; G only), truncation error of G with the
stopping tolerance tightened so only the support matters:

| cell | shipped | trial W8 f0.3 c1 | trial W4 f0.1 c2 | ban | eta10 |
|---|---|---|---|---|---|
| delta 0.1, tol 1e-4, swm 1e-8 | 32.4k | 7.0k | 4.9k | 14.5k | 5.9k |
| delta 0.01, tol 1e-4, swm 1e-6 | 14.9k | 7.5k | 7.5k | 5.3k | 1.7k |
| delta 0.01, tol 1e-6, swm 1e-8 | 32.4k | 14.4k | (not run) | 14.5k | 5.9k |

Trial, measured configurations: W4/f0.1/c2 (the knob defaults) and W8/f0.3/c1 stay inside f*tol in every cell run
(<= 8e-6 at tol 1e-4 against 3e-5, <= 3e-7 at tol 1e-6). W4/f0.3/c1, the most aggressive setting, did not: 3.7e-5 against
3e-5 at delta 0.01, tol 1e-4, swm 1e-6. Savings and margin trade against each other; the defaults are the safe column.
Ban and eta: 3e-6 to 3e-4 at tol 1e-6 and narrow delta, up to 1000x the budget. (The narrow-delta reference did not
converge in 6000 steps; errors below ~1e-7 there include its floor.)

**Trial rules end more accurate than the unit tolerance** (G error 5e-6 against 8e-5 unfrozen) because the trial is
converged to 0.1 x tolerance. Without that tightening two trials never agree.

## Verdict so far

- Ban and eta: no-go as a default (unsafe on the metal, over budget at tight tolerance).
- Trial solve (default knobs): inside the budget in every cell run, and saves 1.2-6x against the shipped rule on a
  trickling support (oracle, G only).
  Whether it saves anything on SMO is open: the production logs show 75-80% of the final support exists at block 16, so
  the ceiling is ~20-40% on heavy units, and W=8 cannot beat the shipped freeze at blocks 22-25 there.
- Not tested: `Sigma_real` on a non-saturating production model; SMO unit 20; the cost of rejected trials at millions of determinants.

## One real unit (small archive, one rank)

SMO unit 20 on the repository's older SMO archive (7,969-determinant ground state, unit of ~80k determinants), trial
rule W4 / budget 0.1 / 2 agreeing trials, run through the kit's new unit mode:

- accepted after 36 blocks at 80,862 determinants; the shipped rule froze the same unit at 81,518 (about 1% more);
- 9 trials of ~12 s each (a full ~1250-block solve on the frozen matrix) took about 110 s of a 129 s unit;
- the real-axis change between successive trials sat at 0.06-0.13 of the tolerance, around the 0.1 budget, which is
  why it needed 9 trials to find two agreeing ones. The support was already 95% complete at block 16.

So on a unit whose support saturates early the rule saves almost nothing and roughly doubles the wall time. Whether
it pays on the heavy cluster units (growth to 10-17M determinants) is what the unit legs of the kit measure.

## Local replay of the production unit: not possible on this machine

The 277,107-determinant ground state of the cluster archive was killed by the kernel's out-of-memory killer at one,
two and four ranks (15.6 GB machine; one rank peaked at 9.2 GiB in the CIPSI selection round). No local number for the
production unit exists.

## Dropped from the plan

Phase 0b/0c (tracing and cut-noting on the importance-pruned path, and `_eta2 = eta * eta`) were not done: ban and eta
are a no-go, so no pruned-path diagnostics are needed.

## Library checks

Rejected trials build and drop a `P H P` matrix and a basis clone each. The matvec exchange creates its graph
communicators either freed on the spot or held in a bounded cache with deterministic collective eviction
(`mpi_comm._cached_dist_graph`), so a rejection leaks no communicator and leaves nothing to the garbage collector.

## Cluster verdict (SrMnO3 it.1, 128 ranks, tree 917b3f9e, jobs 3951526 / 3964546 / 3981432): no-go

Single-unit legs: unit 20 has an unfrozen reference and unit 30 does not. **Correction (2026-10-19): the reference was
not converged to 1e-9 on the real axis.** It set `GF_REAL_TOL=1e-9` as a knob, but the replay passes the archive's
`gf_real_tol=1e-4` as an explicit option, which wins (its manifest records `gf_real_tol=0.0001`, and it stopped after
447 blocks, fewer than ctrl's 454). The kit's guard against that conflict had been disabled for the run. So every
leg, the reference included, stopped the real axis at 1e-4, and the error column below is the difference between
two such runs: it bounds nothing about the freeze itself. The kit now passes a reference's tolerances as explicit
options. On this
tree the thermal ensemble has 68 states (see below) and every unit carries one of them. Unit 20 is the state at
E0 + 136.5 meV, with Boltzmann weight about 1e-3. So these legs grade the rule only on light, low-weight units, and
the round-9 numbers in the plan (751,857 dets, 110 s) do not apply.

| leg | unit | retained | blocks | unit wall | trials | real-axis change vs ref (max-relative / pointwise), both stopped at 1e-4 |
|---|---|---|---|---|---|---|
| shipped (ctrl) | 20 | 3,479,840 | 454 | 16.2 s | 0 | 2.8e-5 / 5.6e-4 |
| trial W4 / 0.1 / 2 | 20 | 3,479,840 | 454 | 84.2 s | 7, none accepted | 2.8e-5 / 5.6e-4 (bitwise the ctrl G) |
| trial W8 / 0.3 / 1 | 20 | 1,778,561 | 638 | 15.6 s | 2, accepted | 8.6e-5 / 2.7e-3 |
| unfrozen ref | 20 | 7,173,876 | 447 | 146.8 s | - | - |
| shipped (ctrl) | 30 | 2,364,027 | 608 | 14.1 s | 0 | no reference |
| trial W4 / 0.1 / 2 | 30 | 2,109,474 | 816 | 56.9 s | 6, accepted | no reference |
| trial W8 / 0.3 / 1 | 30 | 2,109,474 | 816 | 32.6 s | 3, accepted | no reference |

The legs ran in two jobs on different nodes, and the comparison is still fair. All ten legs on this tree
report the same ground state (2,963,444 determinants, 68 states), and the Matsubara G of ctrl agrees with the ref's
to exactly 0, so the seeds are bitwise identical. Units 20 and 30 are both addition units of block 0. The heavy
removal units, where round 9 spent most of its GF wall, were never exercised, so the shipped rule's accuracy below
holds for light addition units only.

Matsubara changes are zero or round-off on every leg. The real axis is the binding axis, and its error sits in the
top quarter of the mesh, where |G| is small.

- **Default setting:** it fails the support bar (0.89x and 1.00x against 0.8x) and the wall bar (4.0x and 5.2x
  against 1.5x).
- **Aggressive setting:** it halves the support on unit 20 at the control's wall, but on unit 30 it saves only 0.89x
  at 2.3x the wall, which fails the wall bar. Its accuracy is not measured: the 8.6e-5 change against the
  reference (3x the shipped rule's 2.8e-5) is between runs both stopped at 1e-4, so it cannot be split into freeze
  error and stopping noise. Being a single-agreement setting (consec=1), it could not have passed the plan's
  false-positive bar anyway.
- **No Sigma:** Sigma_real was never graded (the whole-run legs could not run, below). That alone keeps the rule from
  shipping.
- **What holds:** a rejected trial leaves the live run bitwise intact at 128 ranks. W4 ctrl and trial give identical G
  and an identical block count. The shipped rule keeps 0.49x the unfrozen support at 9x less unit wall, and its G
  differs from the unfrozen run's by 0.28 `gf_real_tol`, within the tolerance both were stopped at; how much of that
  is the freeze is not measured.
- **Defaults defect:** on unit 20 the trial-to-trial real-axis change stays flat at 0.09-0.14 tol over trials 3-7
  while the support grows 2.6x. The same plateau appeared on the small local unit (0.06-0.13). It is the trials'
  own stopping noise. `GF_FREEZE_TRIAL_TOL` (0.1) equals `GF_FREEZE_BUDGET` (0.1), so two trials can only agree by
  chance. Its doc string claims the opposite. A trial tolerance of about 0.01x would lift the floor, but each trial
  would then cost more.

**Decision (2026-10-19):** the trial-solve code was removed: the `GF_FREEZE_TRIAL_*` / `GF_FREEZE_BUDGET` knobs, `_TrialFreeze`, `SupportPolicy.freeze_support_now` and the `trial_frozen` / `freeze_trials` cap stats. The per-step diagnostics (`support_trace`, `GF_SUPPORT_TRACE`) stay. The shipped leak-versus-cut rule is unchanged. Rebuilding the trial means reverting that removal commit. Change `GF_FREEZE_TRIAL_TOL` well below the budget first.

**Whole-run legs (auto75-ctrl/-trial/-trial8, jobs 3996733 / 4007411 / 4007413 / 4007414):** all four were OOM-killed
about 100 s into the GF phase, before `run_units_distributed` printed "New unit roots". The control died the same
way, so the freeze rule is not the cause. The unit legs ran the same 68-state `enumerate_gf_units` and survived,
because the shim passes on one unit's seeds. That puts the kill inside `run_units_distributed` before its print, in
`_colors_affording` or `split_basis_and_redistribute_psi`. On this tree that step sees all units' seeds on a
2.97M-determinant ground state with 68 states, against 10 states on 277k in round 9.

**Why 68 states:** the redesigned GS (window solve) finds a 4-fold ground state and the multiplets at
+112 / +114 / +136 meV:

- Degeneracies run 4, 4+6, 12, 6, 12, and the pairs agree to 1e-8.
- Round 9 found 2 of the 4 ground states and nothing below +223 meV, even in its own 30-root solve. Its basis never
  held these states. E0 agrees to 1e-6.
- The four ground states carry 95.4% of the thermal weight and the other 64 carry 4.6%.

Both trees keep the same energy window: the highest kept state is +227.4 meV in round 9 and +227 meV here. So the
step from 10 to 68 states is all completeness, not a wider window. Round 9's thermal ensemble was incomplete.
Earlier rounds on the old GS are probably affected too, but they were not checked. `gf_min_weight` would keep 44
states (99.93% of the weight) at 3e-4 and 26 states (99.14%) at 1e-3, but it changes Sigma.

## Next (superseded by the cluster verdict above)

Replay SMO it.1 unit 20 (4 ranks) with `GF_FREEZE_TRIAL_WINDOW=4 GF_FREEZE_BUDGET=0.1 GF_FREEZE_TRIAL_CONSEC=2` against the
shipped rule, graded on G per axis; then a whole-`Sigma` archive; then the cluster A/B on SMO and CrI3
(`-v` prints `GF trial freeze after ...` lines; `GF_SUPPORT_TRACE=1` prints per-block leak/cut).
