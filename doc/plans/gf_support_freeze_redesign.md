# GF support settle: investigation and redesign

**Status (2026-10-25): shipped on branch `gf-support-settle`; the failing 1e-8 run that started it could not be
reproduced.** The live description is `doc/gf_support_pipeline.md` (step 6); this file is the measurement record.

## Problem

A NiO 10-bath linked-chain run (`impmod_tests/NiO/impmod/10_BathStates_linked_chainGeometry_noneReorthonormalization_3_processors_`)
printed, for the eg block, `GF unit support frozen at 1,014 determinants (support freeze)` while the largest eg unit
reached 22,559. At `slater_min 1e-10` the run was fine; at `1e-8` the user saw acausal self-energies (that run is
overwritten) and suspected the eg basis froze too early. Asked: a thorough look at the freeze condition and the
window logic, and better criteria.

## What was measured (and what was not)

Rig: `test/support/freeze_replay.py` (tracked). One run = the archived `calc_selfenergy` call of one iteration,
`slater_weight_min` overridden, `gf_tol`/`gf_real_tol` from the archive, `SIGMA_CAUSALITY_TOL=1e9` so Sigma comes out
whatever it is, 3 ranks, one BLAS thread per rank. Grading (`freeze_replay.grade`): (A) max change of Sigma on the
Matsubara mesh; (B) per orbital and mesh point, the change of Sigma_ii against the causality guard's headroom
`h(w) = -Im Sigma_ref,ii(w) + SIGMA_CAUSALITY_TOL * max_w |Im Sigma_ref,ii|`, so (B) = 1 means "the change alone would
use up the room the guard leaves". Archive fingerprint of the NiO runs: `slater_min 1e-10`, `tau 0.0025`, `delta 0.01`,
`excitation_budget 6`, `chain_restrict`, `gf_method lanczos`, `dN None`, `truncation_threshold None`.

### 1. The 1,014-determinant unit is not a premature freeze

Iteration 1 reproduces production (E0 -65.767282, 470 ground-state determinants, units of 1,014 and 5,499).

| run | GF wall | eg removal units, states Sz = -0.89 / +0.89 / 0 | t2g removal units | max Im Sigma_real | Sigma_real vs 1e-10 settled |
|---|---|---|---|---|---|
| 1e-10, old rule (production) | 17.6 s | 24,224 / **1,014** / 24,456 | 30,875 / 5,499 / 32,142 | -2.1e-4 | - |
| 1e-8, old rule | 15.2 s | 16,550 / 1,014 / 17,853 | 18,663 / 5,499 / 14,830 | -2.1e-4 | 1.1e-5 |
| 1e-8, freeze off | 385 s | 36,947 / 1,014 / 36,573 | 33,789 / 5,499 / 33,638 | -2.1e-4 | 1.1e-5 |
| 1e-10, freeze off | 460 s | 37,832 / 1,014 / 37,792 | 34,066 / 5,499 / 34,096 | -2.1e-4 | 4.8e-10 |

- Units are (block, side, eigenstate). The Green's function is solved for one orbital per symmetry class (orbital 0,
  eg-down; orbital 2, t2g-down) and copied. Unit 4 is `c_eg,down` on the Sz = +0.89 state: it lands in the stretched
  Sz = +3/2 sector of N-1 (10 impurity configurations against 50 for the other two states) and is 1,014 with the
  freeze off too. The diagnostics row named the *smallest* unit of the block beside a `maximum ... 22,559` line.
- The old freeze's own error is 1e-9 in Sigma_real and 1e-8 in G_real at 1e-8 (5e-10 / 5e-9 at 1e-10), for a 26x
  shorter GF phase. Matsubara differs by below 1e-13: the blocks that carry G(i omega) are the first ones, bitwise
  the same in both runs; the continued fraction has converged there before the settle acts.
- The difference between the two cutoffs (1.1e-5 in Sigma_real) is the cutoff itself (ground-state basis and
  recurrence prune together; the two were not separated).
- **No acausal Sigma was reproduced, and these replays cannot reproduce it in principle.** The guard trips at 1e-3
  relative to `max |Im Sigma_ii|`: five orders above the freeze error, two above the cutoff effect. Iterations 2, 5,
  10 and 28 replayed at 1e-8 reuse the 1e-10 run's bath fit and are causal (|dSigma_real| <= 5.4e-5, max Im Sigma
  equal to the archive's). A real 1e-8 loop has its own fit trajectory. `debug/gf_freeze/README_0a_rerun_1e-8.md`
  tells what to rerun; until that run exists the attribution path has no true positive.

### 2. A bug in the window logic: chain interiors went to the wrong group

`symmetries.group_orbitals_by_blocks` placed each bath orbital with the impurity orbital it couples to most strongly
*directly*. In a chain geometry 76 of the 96 bath sites have no direct coupling; `argmax` of an all-zero row is the
first impurity orbital, so every chain-interior site went to the eg group: **84 / 12 instead of 48 / 48**. The 36
t2g chain-interior sites `[34-39, 42-47, 50-55, 82-87, 90-95, 98-103]` then sat at distance infinity from eg
orbital 0, counted as "far", and carried the `<= 2 holes` window, while the eg chains' own interiors (distance
< 6.91) carried none. The ground-state group occupation windows (`group 0 valence 82-84`) counted t2g sites too.

Fixed (commit "grouping: a chain's interior stays with the impurity orbital it hangs from"): sites without a direct
coupling follow the nearest impurity orbital through the hopping graph (`-log` hopping weights, multi-source
Dijkstra); directly coupled sites (every star) are unchanged. A bath component that hops directly to impurity orbitals
of two groups joins those groups into one (printed at every verbosity): no per-group window can hold across it. No
such component exists on NiO-10 (eg 4 + 48, t2g 6 + 48 at iterations 1 and 28); the rule makes no assumption about
how many bath sites each orbital has. Effect on the
archive, iteration 1 (the per-iteration window table is `debug/gf_freeze/0b_windows_nio10.txt`, old grouping):

| | old grouping | new grouping |
|---|---|---|
| groups | 84 / 12 | 48 / 48 |
| excited window | 36 sites, `(34, 36)` (iterations 1-2, 21), 40 sites from iteration 3 | none at iteration 1; at iteration 28 the four eg chain-end sites `[10, 22, 58, 70]`, `(2, 4)` |
| GS | E0 -65.767282, 470 dets | identical |
| Sigma_real, 1e-8 / 1e-10 | - | changes by 1.7e-6 / 3.4e-7 (8e-6 / 2e-6 of the headroom); Matsubara 5e-16 |

The windows are identical between `slater_min` 1e-8 and 1e-10 at every one of the 28 archived iterations (old and
new grouping): the deep boundary `a < slater_min**0.5` (1e-4 / 1e-5) is never reached on this model (smallest
`a(o)` 3.8e-4), so the cutoff-dependent part of the window logic is inert here.

### 3. The old criterion, and why it was replaced

`leak <= cut` over 16 blocks (`support_window_freezes`): the weight on new rows against the weight `slaterWeightMin`
pruned.

1. The leak counts only rows that survived the matvec cut, so the rule excludes rows up to `sqrt(sum cut)`, and the
   cut grows with the unit: about 45 x `slater_min` per row at 1e-8, 70 x at 1e-10 on NiO, ~1000 x on a 10M-row
   SMO unit. The user accepted a per-determinant cutoff and got a size-dependent frontier.
2. It compares unlike quantities: the cut is dominated (30-3000x) by the prune of the *next normalised block*, a sum
   of many negligible rows; the leak is measured on `H q` before the TSQR, scaled by `beta_k`.
3. The verdict is permanent and nothing looks at the frontier afterwards.
4. The diagnostics row stitched the smallest support to the largest leak of different units and called a lossless
   event "frozen", the word the cap uses for a lossy one.

### 4. The new criterion

`FreezeRule` (`gf_primitives.py`): the unit **settles** when, over 16 steps, the largest amplitude the normalised next
block `q_{k+1}` carries on a determinant reached for the first time is at most `ratio x slaterWeightMin` and the
log-slope of those amplitudes over the window is not positive (a heuristic against a monotonically growing
trickle). Measured on the TSQR output before the cutoff acts, so `ratio 1` reads "every excluded row would have been
pruned from the next block anyway" and `ratio 0` is closure. It differs from the first-encounter ban of
`doc/plans/gf_freeze_rules.md` (a recorded no-go on a metal): nothing is banned, a row above the frontier still
enters while the unit is unsettled.

Sweeps against the unsettled run at the same cutoff, 3 ranks. **Two groupings**: the first table was run before the
grouping fix of section 2 (old 84/12 grouping, the accidental 36-site window; `debug/gf_freeze/sweep_it1.txt`), the
second on the final tree (48/48 grouping; `debug/gf_freeze/sweep_final.txt`). The cliff between ratios 10 and 30 moved
with the grouping, which is why the default was fixed only from the second.

Old grouping, NiO iteration 1 (the unsettled run took 189-310 s):

| slater_min | ratio | wall | (B) change / headroom |
|---|---|---|---|
| 1e-8 | 0 / 1 / 2 | 310 / 284 / 263 s | 3.7e-9 |
| 1e-8 | 3 / 5 / 10 / 30 | 57 / 24 / 15.6 / 11.8 s | 5.0e-9 / 8.6e-9 / 1.9e-7 / 2.2e-2 |
| 1e-10 | 0 / 1 / 2 | 189 / 235 / 187 s | 2.0e-9 / 2.0e-9 / 7.2e-9 |
| 1e-10 | 3 / 5 / 10 / 30 | 46.5 / 31.9 / 19.3 / 16.6 s | 7.2e-9 / 4.9e-9 / 3.1e-9 / 1.7e-7 |

At ratios 0-2 only the units whose support closes settle (the 1,014 and 5,499 ones); the 20-35k units run to their own
end, hence 190-310 s against the old rule's 15-18 s.

**Final tree (48/48 grouping)**, iterations 1 and 28 (ratio 3: `debug/gf_freeze/sweep_final_r3.txt`):

| iteration | slater_min | ratio | wall (unsettled) | (A) Matsubara | (B) change / headroom | dG / max G (real) |
|---|---|---|---|---|---|---|
| 1 | 1e-8 | **3** | 55.0 s (303 s) | 0 | 5.4e-9 | 1.7e-10 |
| 1 | 1e-10 | **3** | 46.3 s (270 s) | 0 | 5.6e-9 | 1.4e-10 |
| 28 | 1e-8 | **3** | 42.9 s (275 s) | 0 | 9.3e-9 | 1.8e-10 |
| 28 | 1e-10 | **3** | 52.8 s (289 s) | 0 | 8.4e-9 | 1.6e-10 |
| 1 | 1e-8 | 5 / 10 / 30 | 22.1 / 13.4 / 11.0 s (303 s) | 0 | 8.7e-9 / 7.9e-6 / 2.7e-2 | 2.3e-10 / 6.5e-7 / 2.4e-3 |
| 1 | 1e-10 | 5 / 10 / 30 | 25.1 / 20.0 / 14.5 s (270 s) | 0 | 6.6e-9 / 3.2e-9 / 3.0e-5 | 2.3e-10 / 1.0e-10 / 2.2e-6 |
| 28 | 1e-8 | 5 / 10 / 30 | 30.6 / 19.0 / 15.2 s (275 s) | 0 | 8.6e-9 / **7.1e-4** / 6.3e-2 | 2.9e-10 / 3.0e-5 / 4.9e-3 |
| 28 | 1e-10 | 5 / 10 / 30 | 22.5 / 17.4 / 14.6 s (289 s) | 0 | 6.7e-9 / 8.5e-9 / 6.2e-3 | 1.3e-10 / 1.6e-10 / 4.7e-4 |

Matsubara changes are zero or below 1e-13 at every ratio. SMO (28 spin-orbitals, `slater_min 1.49e-8`,
`debug/gf_freeze/sweep/smo_r*.npz`; old grouping is irrelevant there, a star), against closure (ratio 0, 3,271 s): ratio 5
(431 s) 4.3e-8 of the headroom, ratio 10 (280 s) 8.7e-8, ratio 30 (300 s) 7.9e-8, the old rule (310 s) 5.5e-8. Metal SIAM
(`examples/semicircular_siam`, 14-level star): the settle never fires (GF wall 185 s with it off, 229 s at ratio 10, 177 s at
ratio 30) and Sigma is identical to the precision of the output files.

**Default `GF_SETTLE_RATIO = 3`** (decided 2026-10-25). Ratio 5 is the largest ratio of the grid whose change stays below
1e-6 of the guard's headroom on every run (NiO at most 9e-9, SMO 4e-8); 3 keeps a step of margin below it, at 43-55 s per
iteration against 22-31 s at ratio 5 and 270-303 s unsettled (NiO change at most 9.3e-9 of the headroom). Ratio 10, the
first choice, was taken from the old-grouping table and fails the rule on the final tree (7.1e-4 at iteration 28, 1e-8);
ratio 30 reaches 6e-2. SMO was **not** re-run at ratio 3 (its 5 / 10 / 30 results are not monotone in the ratio, so 3 is
not implied to be better there, only expected to be). The acceptance rule used: (A) Matsubara change <= 1e-7, (B) pointwise change of Sigma_ii <= 0.1 x
headroom; `max|dG|/max|G|` is reported and not gated, because at the default tolerance (1e-9 relative) it lies below the
reference's own stopping noise. The reading of RSPt's `convergency 1e-6` as a bound on Sigma(i omega) was **not**
confirmed from RSPt's documentation.

### 4b. The retry's error estimate was loose by five orders (removed)

The estimate and its gate below are **gone** from the shipped code; this section records why. The retry is off by
default (`gf_settle_retry on` enables it): it re-solves only units that settled with a non-zero ratio, none capped or
held, on a real-axis failure, and it can cost as much as an unsettled run.

`selfenergy._settle_estimate` (the 10% gate of the re-solve) was calibrated on the one case with a real settle effect:
NiO iteration 1 at 1e-8, ratio 30 against unsettled (`debug/gf_freeze/calibrate_estimate.py`).

| block | settled units | measured max `|dSigma_ii|` | estimate at that point |
|---|---|---|---|
| 0 | 3 | 5.8e-3 | 326 |
| 1 | 3 | 3.3e-3 | 536 |

It overshoots by 6e4-2e5 (the bound carries 1/delta^2 and a sum of non-cancelling terms), so the "estimate >= 10% of the
violation" condition essentially never skips a re-solve. What does gate the retry is: units settled, none held by a cap or
the guard, `gf_settle_retry` on. A bath pole near the worst point is reported (weighted by the block's own hybridisation),
not used to veto the re-solve.

### 5. Symmetry copy

The eg-down solve stands in for eg-up. Measured premise: the retained ensemble is spin-symmetric. The thermal density
matrix has `max |rho_dn - rho_up| = 1.8e-9` over the impurity block (eg 0.554362 / 0.554362, t2g 1 / 1), the three
states of the manifold are degenerate to below 1e-9 and equally weighted, at both cutoffs. **An eg-up Green's
function was not solved**; the number above is the density-matrix mirror, not a comparison of G.

## What changed (commits on `gf-support-settle`)

harness and window diff; frontier amplitude in the trace; end-of-unit frontier probe; `get_Greens_function` as plan /
solve / assemble; the rule (`FreezeRule`, `note_next_block`, `ManyBodyState.max_row_norm2_in`); consumers and the
settle verdict; per-unit records and the "settled" wording; options `gf_settle` / `gf_settle_ratio` /
`gf_settle_retry`; default ratio; the BiCGSTAB seam rename; the grouping fix; Sigma-causality attribution with one
gated closure-only re-solve; chain zones as data. After the adversarial review: the probe made opt-in (settled units only,
on the true last block); the retry off by default, without the estimate gate, Matsubara-only or ratio-0 re-solves; default
ratio 3 and archive booleans parsed as words; log lines that say settled / held / frozen at the cap and no "lossless"
claim; impurity groups that one bath component reaches merged (none on NiO-10); chain-zone records read off the emitted
windows.

Deviations from the plan, and why: the retry's magnitude gate is ineffective (section 4b); `chain_deep_cutoff` and the occupation-gated deep zone were **not** done (the deep
zone is empty on NiO-10 at every iteration and the 1e-8 case that would motivate them is unavailable); the retry gate
has no "excluded rows are of excitation order <= 1" test and no frozen-matrix pre-check (the probe does not enumerate
excluded rows; the re-solve is accepted only if uncapped, unguarded and converged); the unit record has no (N, Sz)
sector label (the state index and E-E0 identify the unit; the sector reading above is from the ground-state report);
`impmod_interface`: the solver-line tokens are on its local branch `gf-settle-options` (commit 20777d5, off `development`
5f74b43, unmerged and unpushed); its `development` working tree holds uncommitted changes of its own to the same `lib.py`,
so the merge is left to its owner. Its test suite has 53 failures with or without the branch, none in the parser.

## What this does not show

- That the freeze is not the cause of the user's acausal Sigma at 1e-8: only that it moves Sigma by 1e-9 on every
  replayed iteration and that the 1e-8 replays are causal. The failing run is the missing evidence.
- That the chain windows are not the cause: they are identical at the two cutoffs on every archived iteration, but
  a 1e-8 run has a different fit trajectory.
- The ratio sweep covers NiO iteration 1, SMO iteration 1 and the metal SIAM. CrI3 and the heavy production units
  (10M determinants) were not run; the retry's cost and its memory gate are untested at that scale.
- Whether the trend term of the rule refused any window in these runs (it was not logged separately): it is a
  heuristic against a monotonically growing trickle, pinned by synthetic tests only.
