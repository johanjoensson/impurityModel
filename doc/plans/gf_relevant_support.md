# Relevant determinants at a fixed cap: measurement record

**Status (2026-10-27): Phase 1 of the plan (the discovered-key matrix and the recurrence on it, three matvec
modes) is on master through `b65081f2`; the 1.6 gate (delete today's GF growth/frozen path) was measured like
for like on 2026-10-27 and accepted by the user (section 3).** The live description of the pipeline is
`doc/gf_support_pipeline.md`; this file is the measurement record of the series. The plan itself is kept
outside the repository (revision 7, 2026-10-26); its mechanisms are E (the discovered-key matrix), A (pruned
growth), B (swap refinement) and C (the hopping-cost budget).

## 1. The rig

One run = the archived `calc_selfenergy` call of NiO it.1
(`impmod_tests/NiO/impmod/10_BathStates_linked_chainGeometry_noneReorthonormalization_3_processors_.bak/outdir-2026-10-24/outdir-1/impurityModel_data.h5`),
replayed by `test/support/freeze_replay.py run`: 3 ranks, one BLAS thread per rank, `slaterWeightMin` 1e-10,
support settle on, `IMPMOD_MATVEC_PROF=1`, under `/usr/bin/time -v` (VmHWM = the largest resident set of any
rank over the whole run, ground state included). `--matrix-mode` sets `gf_solvers._MATRIX_MODE`: `off` is
today's path (the ManyBodyState growth kernel and its frozen `P H P` CSR), `auto` prices the mode per unit,
`discovered`/`retained`/`none` force one. `--cap N` pins `truncation_threshold` for the ground state and the
Green's function alike (the ground state stops at 379k and never reaches it, so only the GF cap pins).
Grading (`freeze_replay grade`): (A) max change of Σ on the Matsubara mesh and on the real axis; (B) the change of
Σ_ii against the causality guard's headroom per mesh point and orbital; and, as context, max|dG|/max|G| and the
worst Im Σ_ii on the real axis.

## 2. The first series (1.5, 2026-10-27, auto-sized cap): not like for like

Every mode forced, machine otherwise idle except where noted.

| mode | GF wall | whole run | VmHWM | GF cap (auto) | guard stops | blocks [0]/[2] | worst Im Σ |
|---|---|---|---|---|---|---|---|
| off (today) | 1557 s | 1908 s | 1.24 GiB | 1,898,707 | 0 | 2008 / 1933 | +6.7e-3 |
| retained | 3129 s | 3496 s | 1.59 GiB | 1,953,561 | 1 (at 1.85M) | 1993 / 1925 | +5.0e-3 |
| discovered | 2707 s | 3071 s | 1.48 GiB | 1,948,280 | 3 (983k, 1.50M, 1.78M) | 1978 / 1929 | causal |
| none | killed at 7 h 44 m | — | — | — | — | — | — |

Confounds, named: the GF cap was auto-sized from free RAM at the GF split and moved by 2.9 % between runs; the
guard budget likewise (1.3-1.4 GiB); another session's job overlapped the start of the series; the G/Σ
differences therefore come from different retained sets (caps and guard stops), not from the kernel (block
counts agree to 1 %). What the series did establish: the new loop's frozen recurrence was ~2x slower per block
than the array kernel (py-spy: 79 % in the scatter, 18 % in allocation traffic of the dense step) and its growth
step held ~0.3 GiB more (the whole applied image materialized as Python key objects every step). `discovered`
cannot fit this unit at 3 ranks on this machine (priced ~1.1 GiB on top of 0.46 GiB resident against 1.3 GiB
headroom); `none` is unusable as run (3.2 s per matvec at 2.1M rows).

The 1.5b series answered this before the second measurement (`6f0b0384` stage timers with RSS high-water and
`--cap`/`--units`; `b973ccf1` growth step without key objects and an allocation-free dense step; `416d0556`
int32 packed rows, 20 B per element; `cb707f88` the retained mode's frozen phase on the array kernel;
`93b8bf1e` P slots numbered in key order before the freeze build; `a7ac8bff` one copy of each block at the
handoff; `b65081f2` `auto` takes the discovered matrix only for a consumer of the frontier rows).

## 3. The like-for-like measurement (1.5b.4, 2026-10-27, cap pinned at 1,898,707)

`off` against `auto`; `auto` chose `retained` on every unit (the stage table has `build_columns`,
`to_array_operator` and `frozen_kernel` and no `matvec_all`/`matvec_P`). Scratch files
`.../scratchpad/nio15/f_off.*` and `f_auto2.*`.

| metric | off (today) | auto (chose retained) |
|---|---|---|
| GF wall | 1558.9 s | 1493.9 s |
| whole run wall | 1947.6 s | 1875.7 s |
| VmHWM | 1,237,900 kB | 1,255,124 kB (+17.2 MB, +1.4 %) |
| minor page faults | 7.72M | 5.82M |
| worst Im Σ_ii, real axis | +2.902e-03 | +2.902e-03 (identical) |
| max\|dΣ_mats\| / max\|dΣ_real\| vs off | — | 6.493e-11 / 3.868e-05 |
| max\|dG\|/max\|G\| real / mats vs off | — | 5.399e-06 / 5.041e-11 |
| headroom ratio (B) | — | inf at mesh point 350, orbital 0: the 4 worst reference points are already inside the guard, i.e. the same worst points |

**Gate as written: wall, VmHWM and grade of `auto` ≤ `off`.** Wall ✓ (4 % faster on the GF phase and on the
whole run), grade ✓ (G and Σ agree to numerical noise; the causality verdict is identical), VmHWM ✗ by 17.2 MB.

**Where the 17 MB is.** The resident set at the GF split, before any mode-dependent code has run (the ground
state is the same code in both runs), was 445.6 MiB in `off` and 460.3 MiB in `auto`: 14.7 MiB = 15.4 MB, i.e.
~90 % of the VmHWM difference is already in the floor the GF phase starts from. Over the fifteen runs of this
rig that share that pre-GF code the floor spread 442.6-460.3 MiB with no dependence on the mode (two `retained`
single-unit runs: 445.1 and 460.2 MiB). The two `off` runs at this cap are 3.3 MB apart in VmHWM; on the
single-unit rig the `retained` run (`u_ret6`) is 8 MB *below* the `off` run (`u_off`). In both modes the peak
sits in the frozen phase on the same array kernel (`frozen_kernel` rss ≤ 1.29-1.30 GB per unit in `auto`,
matching its VmHWM). Verdict: the VmHWM difference is the run-to-run floor, not the new loop. The user accepted
the gate on this evidence (2026-10-27) and 1.6 proceeds.

## 4. The three-way at cap 1,000,000 (where `discovered` fits)

Measured 2026-10-27, `--cap 1000000`, `-n 3`, swm 1e-10, freeze on, machine idle (scratch `c1m_{off,ret,dis}`).

| mode | GF wall | whole run | VmHWM | worst Im Σ (real) |
|---|---|---|---|---|
| off (today) | 767.5 s | 1157 s | 864,556 kB | +6.887e-2 |
| retained | 762.9 s | 1149 s | 899,536 kB | +6.887e-2 |
| discovered | 1367.6 s | 1758 s | 975,920 kB | +6.887e-2 |

Grade against `off`: `retained` and `discovered` both give max|dΣ_mats| 2.3e-10, max|dΣ_real| 8.6e-5,
max|dG|/max|G| 5.4e-6 (real) and 1.3e-10 (Matsubara). The reference is acausal at this cap in all three (12 mesh
points already inside the guard, the headroom ratio is `inf` by that definition), so the modes agree on the Σ,
including its sign problem; the cap, not the engine, makes it. `retained` is at wall parity with `off`
(-0.6 %); `discovered` costs 1.78x the GF wall (its frozen matvec is the scatter over P rows of a matrix that
also carries the F rows and the 24 B/entry packing) and +111 MB VmHWM over `off`. |P| and |K| were not
recorded by this rig (the unit line prints neither); the 1.5 pricing measured |K| ~ 2.1 |P|.

## 5. What this leaves for Phases 2-3

The retained mode is now today's path in cost (slightly faster, same memory within noise) and the discovered
mode is the substrate the importance probe (2.1) and swap refinement (3) need: it costs more memory (the F rows
and the K store) and its frozen matvec is ~0.33 s per block on this unit against the array kernel's. The user's
rule (2026-10-27): a refinement run may be slower than the retained path if it yields a causal Σ, bounded by
"still converges a self-energy in a reasonable timeframe"; refinement features are not gated on matching the
retained wall, but their per-unit and full-run cost is reported.

## 6. Pruned growth (Phase 2.6): eta sweep at cap 1,000,000

Measured 2026-10-28, NiO it.1 replay, `--cap 1000000`, `-n 3`, swm 1e-10, `--matrix-mode discovered`, rule `eta`,
first shell kept whole (`first_shell_tol` unset). Reference: the breadth-first discovered run of section 4
(`c1m_dis`), which is **acausal** (worst Im Σ +6.887e-2), so `dG` against it is a distance from a wrong answer,
not an error. Scratch: `scratchpad/p2/eta*.{log,time,npz}`.

| eta | units at the cap | GF+run wall | VmHWM | worst Im Σ (real) | max\|dΣ_mats\| vs breadth |
|---|---|---|---|---|---|
| breadth (reference) | all 12 | 51 min | 1,476,292 kB | +6.887e-2 | -- |
| 1e-3 | none (closure-like) | 30 min | 812,240 kB | -2.0e-4 (causal) | 2.1e-4 |
| 1e-4 | all heavy units | 91.6 min | 1,271,048 kB | -2.0e-4 (causal) | 2.2e-4 |
| 1e-5 | all heavy units | 87.6 min | 1,020,212 kB | +3.2e-2 (acausal) | 2.2e-4 |

Reading:
- At eta 1e-3 no unit reaches the 1M cap: the pruned support closes at ~0.45-0.54M determinants per unit (heaviest
  three: 445k / 480k / 536k) and Σ is causal, in 30 min against 51 and 0.81 GB against 1.48. The cap was not the
  limit; the breadth-first order of admission was.
- At eta 1e-4 the units reach the cap again (deeper growth) and Σ stays causal, but the wall is 3x the 1e-3 leg.
- At eta 1e-5 the support is breadth-like again and the acausal feature returns, at half its size (+3.2e-2).
- Matsubara Σ agrees to 2e-4 across the sweep: the problem is the real-axis feature at the bath pole, as diagnosed.

**Go/no-go for B (2.2)**: pruning alone already gives a causal Σ at lower cost than the breadth reference, with the
cap not binding at eta 1e-3. B (swap refinement at a binding cap) is therefore not needed for the failing unit;
it stays relevant only where a unit's *pruned* closure exceeds the cap. The importance-probe pass on a real
capped unit (the 2.2 swap-fraction criterion) was not run: the capped legs that would feed it (eta <= 1e-4) are
the ones that are already causal at 1e-4. Decision: Phase 3 is deferred, not dropped; revisit if a production unit
is acausal with pruning on at its cap.

Not yet measured: the mass rule (tau) legs, and `first_shell_tol > 0`. The first-shell guard of the plan (dropped
weight <= 1e-6 of ||Hs||^2) is still not enforced in code; a nonzero `gf_first_shell_tol` is the user's
responsibility.

## 7. Open item: swap refinement for complex H

`gf_refinement.refine_frozen_unit` declines a unit whose matrix has a non-real element (`matrix_is_real`, 1e-12), because
the removal identity `dG_ij = -X_i(D) X_j(D) / R_DD` needs `R = (z - H_P)^-1` complex-symmetric, i.e. real symmetric `H`.
For complex Hermitian `H` (spin-orbit coupling, complex hoppings, a complex basis rotation) `R(z)^T = R(z*)^*`: the
retained score needs a second solve at `conj(z)` (`dG_ij = -X_i(D) Y_j(D) / R_DD` with `Y = R(z*)^* s`), and `R_DD`
then comes from the same first-order `Sigma_D` with complex amplitudes. To do: the second solve per shift, complex
`|M|^2` stays as is, tests against the dense complex oracle at -n 1/2/3, and drop the decline. (User, 2026-10-28:
"we will have to implement the complex H handling at a later stage.")

**Scheduled (2026-10-28, user: fold it into the plan):** plan step 3.7, after the 12-unit NiO grade and before Phase 5.
The scoring side is done (`39e240a8`: the second solve at `conj(z)` whenever the seeds or `H` are not real); what is
left is removing the `"complex H"` decline in `refine_frozen_unit` and proving the loop end to end on a complex
Hermitian chain fixture (dense complex oracle, `-n 1/2/3`, rank invariance, injected `X`-for-`Y` bug).

## 8. Swap refinement (Phase 3): measurements

Code: `gf_refinement.py` (`importance_scores`, `refine_frozen_unit`), on the discovered matrix; real H only (§7).
Rig: NiO it.1 replay, `-n 3`, swm 1e-10, `--cap 1000000`, `--matrix-mode discovered`, breadth-first round 0,
reference `c1m_dis` (§4, acausal, worst Im Σ +6.887e-2). Scratch: `scratchpad/p2/refine2hb.*`, `u10cheap.*`,
`refine12b.*`.

**First pass (3.4, units 4/9/10 only, 2 rounds, tol 1e-3, swap 0.1, rtol 1e-6, maxit 500).** Every importance
solve hit `maxit = 500` (the final residual was not recorded). A pass cost 745-2509 s per unit whatever the block
width (block BiCGSTAB, `a79d586d`): ~500 matvecs per shift. Unit 10 settled (gain 14.2, 0.17, 0.008), units 4 and 9
were still swapping the full 10 % at round 2. That run's wall (7185 s) includes a machine hibernation, so CPU time
per rank (~5000 s) is the honest figure; later runs are under `systemd-inhibit --what=sleep:idle`.

**Cheap, honest solves (3.5a, `a4acc508`).** `RefineSpec.rtol/maxit/broaden` (module constants `REFINE_*`, set by
the rig, not user options), the real-axis scoring shifts at `Re z + i·broaden·Im z` (Matsubara shifts and the
holdout unchanged), the worst final residual per pass in the record. Unit 10 alone at rtol 1e-3, maxit 150,
broaden 4: passes 196 / 159 / 99 s (against 1594 / 1530 / 1862), same settle, holdout dG 0.183 (0.181 before),
residuals 1e-2 → 1e-3; unit total 695 s.

**The decisive run (3.5b): all 12 units, 3 rounds, rtol 1e-3, maxit 150, broaden 4.**

| unit | swapped per round | settled | dG scoring / holdout | refinement s | unit wall s |
|---|---|---|---|---|---|
| 4 | 100k ×4 | yes | 2.6e-4 / 3.3e-2 | 859 | 1259 |
| 10 | 50k ×3 | yes | 9.8e-2 / 1.8e-1 | 984 | 1371 |
| 9 | 100k ×4 | yes | 3.8e-3 / 1.7e-1 | 1122 | 1611 |
| 3 | 100k ×3 | yes | 4.5e-4 / 2.7e-1 | 910 | 1288 |
| 5 | 100k ×4 | yes | 7.6e-4 / 1.6e-1 | 1050 | 1618 |
| 11, 1, 0, 6, 7, 2, 8 | nothing (gain below tol at round 0) | yes | -- | 174-271 | 383-627 |

Graded against `c1m_dis` (`freeze_replay grade`): **worst Im Σ_ii on the real axis -2.005e-4 (causal; reference
+6.887e-2)**, max|dΣ_mats| 2.2e-4, max|dΣ_real| 0.19 (the acausal feature moved), max|dG|/max|G| 7.1e-2 real /
1.5e-3 Matsubara. GF wall 3729 s against 1368 s (2.7x), whole run 1:12:28, VmHWM 1.56 GB against 0.98 GB. The
causal value equals the eta 1e-3 leg's (§6): both remove the same real-axis feature.

Reading and caveats:
- Refinement turns the breadth-first acausal Σ causal at the same cap, at ~2.7x the GF wall and +0.6 GB.
  Pruning (§6) reaches the same Σ at 0.6x the wall: where pruning closes below the cap it is the cheaper route;
  refinement is for units whose pruned closure still exceeds the cap.
- The solves stop at residuals of 1e-3 to 1e-1 (units 5, 4: 8e-2 to 1e-1). Every unit still settles, so the ranking
  is good enough for the estimate; the rankings are not certified.
- Deviations from the plan: the round-0 fallback and the settle test the *estimated* gain (no full-mesh grade per
  round); the final recurrence is run once, and the full-mesh grade is the offline `grade` above.
- Open: one run of the 12-unit leg (`refine12`, before the fault handler was on) died with a segmentation fault in
  `SparseScatter` on rank 0 after 41 min; the identical rerun with `PYTHONFAULTHANDLER=1` completed. Not explained.

### 8.1 The Phase 3 review (Opus, read-only) and what was done

Fixed (commit after `a4acc508`, with tests that go red under the old code):
- **Retire ranking resolution.** `low = 1/(1 + s_ret/smax)` lies in (0.5, 1] and the selections bin scores to 32
  mantissa bits, so every row below ~2e-10 `smax` fell into one bin and was retired in key order (plus a pickled
  allgather of that bin). Now `low = 1/s_ret` (relative resolution). Test: 40 decades of scores, the retired rows
  are exactly the k lowest.
- **Exchange plans held across graph-cache insertions.** `importance_scores` built four plans up front and held them
  for every shift; `mpi_comm._cached_dist_graph` evicted first-in-first-out and a hit did not refresh an entry, so
  a later plan's construction could `Free()` a graph a live plan still used (a collective on a freed
  communicator is a segmentation fault). History-dependent, which fits "one crash, not reproduced". Plans are now
  built right before their matvecs and the cache is least-recently-used. The review could not tie this to a
  fault inside `scatter_discovered`; the crash site in the first 12-unit run was `SparseScatter` and is
  **not proven** to be this.
- Complex seeds: the removal identity needs seeds real up to a phase per column as well as real `H`; a column
  with a relative phase is now declined (`|s^T s| = s^H s`).
- The record's `swapped` lists the swaps made (it used to include the one the last pass would have made);
  a `-1` key slot can no longer wrap onto the last K slot; one copy of the solutions through `set_p_keys`;
  `queue_progress()` in the solve loops.

Open (not changed; each is a design question, not a bug fix):
1. **The fallback and the settle test the estimated remaining gain** (sum of the top-k `s_cand`), which is not an
   error measure: K only grows, so the last candidate pool is a superset of round 0's, and a retired row
   re-enters F with a tiny score. The design's full-mesh grade of the rerun needs the self-energy, which exists
   only outside the unit; `dG_holdout` is computed and not used. The 12-unit grade above is the offline check.
2. **Hermitian, not complex-symmetric.** The scores use `X_i(D) = s_i^H R e_D`, i.e. real `H`; the Hermitian form
   needs `Y = R(z*) s` (a second solve per shift). Declined for complex `H` and, now, complex seeds. Per the
   user's rule, new code must not add to this (§7).
3. **Solve accuracy.** Residuals of 1e-3 to 1e-1 leave noise in `X(D)` of the size of the smallest `s_ret`. A
   noise floor (refuse swaps inside it) or a multi-shift block Lanczos on `P H P` (Hermitian, no breakdown, all
   shifts from one Krylov sequence) is the principled fix.
4. Memory: `|M|^2` scatter copies `data` once per shift after the first compaction; four plans' buffers; no
   budget check before `set_p_keys` (design B.2 asks for one); K is never trimmed.
5. Rank-count invariance of the refined P is a tolerance on G, not exact (columns stop on different iterations).

## 9. Phase 4 (hopping-cost budget): what was measured, what is left

- **NiO it.1 (§6 rig), cost budget α = 0.3 / 0.5 on top of eta 1e-3: no effect.** Σ identical to the eta leg (worst
  Im Σ -2.005e-4, max|dΣ_mats| 2.0e-4 / 2.1e-4), wall 1405 / 1415 s against 1377 s, VmHWM 0.80 / 0.82 GB against
  0.81, ground state 624,576 in every leg, heaviest unit 525,194 / 536,037 against 536,037. Cause:
  `B_GS = round(8·α·(−log slater_weight_min))` is 55 hops at α 0.3 and swm 1e-10, and the Green's-function floor is
  `C_max` of a 624k-determinant ground state, so the budget never binds. C is a lever for a much larger
  `slater_weight_min`, a metal or a long chain, not for this workload. It stays opt-in (`BasisOptions.cost_budget`,
  default off); the soft tail (4.8/4.9) was not started (no measured gain).
- **Per-row form cost** (`test_apply_nio_bench.py`, 103k-row block, width 1, one build, 6 reps): the cost form
  (18 weight groups on NiO it.1) adds 1.09 → 1.16 µs per input row, +6.3 %, one run on an otherwise idle machine;
  the plan's rule is "coarsen κ if ≥ 5 %". Not coarsened: the user decided "leave kappa" (κ = 8), the form is off
  by default and the measurement is a single run.
- **Mask seams (4.6).** `z - hOp` is a fresh operator without hOp's weighted mask: the RIXS R1 solve and
  `_FrozenShiftedSystem.boundary` now install it (test: a violating row never appears in the R1 solution on the
  sparse and the frozen tier; the frozen leak check no longer counts out-of-window rows as leaks). The RIXS
  kernel installs the masks from `tmp_basis` before `chain.solve` and from `green_basis` before `eval_out`.
  Correction to the b318f4c7 message: `gf_shift_recycling`'s `hOp` does *not* come masked from the RIXS tiers (it
  is installed by the kernel now), and the `gs_connectivity` change was a no-op and is reverted. `gf_admission`'s
  caller sets the weighted mask: audited, unchanged.
- `weighted_parts` (4.1): `symmetries.weighted_parts` is the one unpacker; tails ride through `widen`, the floor and
  the symmetry check; the operator refuses a tail it cannot honour.
- Tests added: rank-count invariance of a cost-restricted ground-state basis at `-n 1/2/3`; the causality floor admits
  every single transition (dropping the floor turns it red).

### 9.1 Open items from the Phase 4 review

- **H2 (open, core class).** GS seeds outside the weighted window make the masked ground-state `H` non-Hermitian
  (a seed row the mask rejects is dropped from `H s` but `s` itself is kept). The fix is a floor or filter at
  `Basis` construction, which changes a core class; not done here. The causality floor protects the Green's-function
  side only.
- **M1 (done).** `get_effective_weighted_bounds` is cached on the basis (key: `generation` and the entries) and
  expands the determinants to bits in chunks of 2^16, so the transient is bounded. The floor is on the default
  path of every Green's-function call, not only with a cost budget.
- **M2 (open).** The plan's "partially occupied sites weight 0" is not implemented: the cost weights see only the
  valence/conduction classes, not occupations.
- **M3.** `widen_weighted_restrictions` adds `max|w|` on top of the floor, so the Green's-function budget is close
  to non-binding; a second reason, beside §9's `B_GS` arithmetic, for the NiO null result.
- **M4.** The "first shell is admitted" claim holds only for separate valence and conduction chains.
