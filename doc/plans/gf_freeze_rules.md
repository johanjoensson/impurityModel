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

## Dropped from the plan

Phase 0b/0c (tracing and cut-noting on the importance-pruned path, and `_eta2 = eta * eta`) were not done: ban and eta
are a no-go, so no pruned-path diagnostics are needed.

## Library checks

Rejected trials build and drop a `P H P` matrix and a basis clone each. The matvec exchange creates its graph
communicators either freed on the spot or held in a bounded cache with deterministic collective eviction
(`mpi_comm._cached_dist_graph`), so a rejection leaks no communicator and leaves nothing to the garbage collector.

## Next

Replay SMO it.1 unit 20 (4 ranks) with `GF_FREEZE_TRIAL_WINDOW=4 GF_FREEZE_BUDGET=0.1 GF_FREEZE_TRIAL_CONSEC=2` against the
shipped rule, graded on G per axis; then a whole-`Sigma` archive; then the cluster A/B on SMO and CrI3
(`-v` prints `GF trial freeze after ...` lines; `GF_SUPPORT_TRACE=1` prints per-block leak/cut).
