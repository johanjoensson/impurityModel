# Swap refinement of capped GF units: review (2026-10-30)

**Scope.** The excited-state basis refinement as it stands at `f6c10738`:
- `gf_refinement.py`;
- `ShiftedGalerkin` in `gf_shift_recycling.py`;
- the shifted hooks and the refine orchestration in `gf_recurrence.py`;
- the scoring tables in `selfenergy.py`;
- `refine_for` in `greens_function.py`.

The measurement record is `doc/plans/gf_relevant_support.md` §7–§14. The review changes no library code. Every finding
is marked confirmed, refuted or not run, with its evidence. Proposed fixes are at the end, for approval.

**Evidence.** The scripts and raw outputs are in `debug/refinement_review/` (gitignored).
- **Dense oracle (`oracle.py`).**
  - Model: a single-orbital SIAM with a 7-level star bath per spin; U = 3, half filling. The ground state is truncated
    to its 300 largest amplitudes, standing in for the GS cap and swm. The addition sector `c†_up |GS⟩` holds 3920
    determinants.
  - P: the seed support plus breadth-first H-shells, the overflowing shell cut by amplitude. Caps are 12 %, 20 %, 30 %
    and 50 % of the sector.
  - Scores: the library's score definitions, replicated densely as `test_gf_refinement._dense_scores` does (that test
    proves the library equals them).
  - Comparisons: against the *exact* removal dG (`|X_D|²/|R_DD|`, Sherman–Morrison) and the exact addition dG (the
    bordered Schur complement). Swap rounds use the library's acceptance rule, without the gain fallback, and are
    graded against the exact full-sector G. The shifts follow `g0_zero_shifts`: the bath-pole maxima of |G0⁻¹|,
    ω = 0, and two Matsubara points, at δ = 0.05 and 0.01.
- **`galerkin_residual.py`.** The library's `ShiftedGalerkin`, driven by a width-2 block Lanczos *without*
  reorthogonalisation on the same H, for 1200 blocks. Compared: the recursive residual, the true residual
  ‖Q₁ − (z − H)X‖, and X against the dense solve.
- **Production logs.** CrI3 SOC at 60k, SMO 2-bath at 40k/80k, and NiO it.1 at 1M, from the §8–§14 sessions.
- **Instrumented replays (`instrument.py`).** In-process monkeypatches around `freeze_replay`, on every rank and
  collective-safe. They log, per scoring pass:
  - the exempt and swappable counts;
  - the best candidate score against the worst swappable retained score;
  - which shift attains each row's maximum.

  They also override `FIRST_SHELL_EXEMPT_TAIL`. Runs: SMO 2-bath at 40k and M1 (12 baths) at 65k, each at tails 1e-6,
  1e-2 and 1. They are graded with `freeze_replay grade` against the uncapped references of §10 and §11. The tail-1e-6
  arm reproduces the earlier uninstrumented runs bit for bit, so the instrumentation does not perturb the solve. The
  SMO walls are **not** cost evidence: those runs shared the machine with the oracle.

## Verdicts at a glance

**Severity:**
- `!!`: a plausible silent wrong or worse result.
- `!`: the feature does not do its job.
- `·`: performance or quality.

| id | finding | verdict | sev |
|---|---|---|---|
| F4 | The acceptance rule measures an estimate, not the error: rounds can regress G, and the gain can fall while the real axis degrades | **confirmed** (oracle: non-monotone, once worse than breadth; M1 with a loosened tail: real axis 7x worse while the gain fell) | ! at the default tail, !! once the tail is loosened |
| P1 | The first-shell exemption (tail 1e-6) leaves nothing swappable exactly where the cap binds; loosening it helps SMO and hurts M1's real axis 7x | **confirmed; the fix is conditional on R1** | ! |
| F1 | Thermal columns scored at other states' shifts, with no Boltzmann weight | **confirmed in the oracle; unreachable at the default `GF_EIGENSTATE_GROUP = 1`** | · (latent behind a knob) |
| B3 | The scoring accumulator (8 shifts × 2 conjugates on complex H) is carried even when nothing is swappable | **confirmed in kind; the CrI3 size is not separated from the pruning restart** | · |
| S1 | `REFINE_STOP_RTOL = 1e-4` is stricter than the ranking needs | confirmed down to 2e-3; 2e-3–1 untested | · |
| B5 | The last round's score-only recurrence is followed by a separate GF recurrence on the same P | confirmed (code) | · |
| F2 | Scores in the orthonormal Q₁ frame, stop threshold in the r-weighted frame | confirmed; ranking unaffected at near-scalar r only | · |
| B4 | `_g_of_solutions`, `matrix_is_real` dead | confirmed (grep incl. .pyx/.pxi) | · |
| N2 | The recursive Galerkin residual is dishonest without reorthogonalisation | **refuted for a fixed P** (caveats in §2) | — |
| N4 | The first-order `R_DD`/Epstein–Nesbet scores misrank against the exact dG | **refuted** | — |
| P2 | Matsubara shifts dominate the max with unit weights | **refuted** (the opposite holds) | — |
| F3 | A score-only rerun that hits `max_blocks` returns an unjudged G | refuted (benign) | — |
| B1 | Round 0 is lengthened by the scoring residual | refuted (CrI3, SMO) | — |
| B2 | `importance_scores` matvecs are a bottleneck | refuted (0.3–13 s per pass) | — |
| — | The array-kernel handoff bypasses the accumulator | refuted (`array_operator` exists only on `_ApplyEngine`) | — |
| F6 | The final GF rerun (`gf_refinement.py:498`) is not checked for `"diverged"`; the scoring reruns are (`:479`) | confirmed (code); unlikely to fire | · |
| N3, F5, N5, P3 | deflation floor, ping-pong, halving search, Σ-vs-G objective | not exercised / not run | open |

## 1. Correctness and physics

### F4 — the acceptance rule measures an estimate, not the error (`!` at the default tail, `!!` once loosened; confirmed)

`refine_frozen_unit` (`gf_refinement.py:451-494`) works as follows:
- It swaps up to `swap · |P|` rows per round, on single-row first-order scores.
- It settles or falls back on `gain`, the summed `s_cand` of the next k_max candidates
  (`_select_swap`, `gf_refinement.py:348-349`).
- `dG_holdout` is computed and never used (`:501-502`).

Nothing compares G before and after a round.

**Oracle.** The oracle's swap loop is *not* the library's trajectory:
- it has no gain, so it models neither the settle stop nor the fallback;
- it does not exempt the previous round's admissions, so a row can be retired right after it was admitted.

Its "round n" is the state after the n-th swap; the library's round 0 is breadth. Real-axis relative error of G
against the exact full sector:

| cap | breadth | round 0 | round 1 | round 2 |
|---|---|---|---|---|
| 30 %, δ 0.05 | 9.09e-2 | 7.79e-2 | 7.63e-2 | **8.25e-2** |
| 20 %, δ 0.05 | 2.56e-1 | 2.43e-1 | 2.46e-1 | **2.76e-1 (worse than breadth)** |
| 30 %, δ 0.01 | 2.91e-1 | 2.46e-1 | 2.54e-1 | 2.54e-1 |

The Matsubara error follows the same pattern, for example 1.41e-3 → 6.5e-4 → 6.5e-4 → 7.0e-4.

The arm that swaps on the *exact* single-row dG makes exactly the same swaps and gets exactly the same errors, so the
scores are not the cause. What the oracle shows is non-monotone behaviour:
- at 30 %, δ 0.05, round 1 still improves;
- at 30 %, δ 0.01, nothing regresses;
- at 20 %, round 2 ends worse than breadth.

Part of the regression may be the ping-pong the oracle allows and the library forbids.

**Production:**
- SMO 40k, the worst unit: 3 rounds of 3 swaps each, with the gain rising 0.188 → 0.194, so it fell back.
- NiO and SMO refined units: the held-out |dG| is 10–600x the scoring-shift |dG| (NiO unit 4: 2.6e-4 against 3.3e-2).
  The swaps move G mostly where it is not scored.

At the default tail, production shows no degradation. On SMO the fallback fired (the safeguard acting), and
default-tail refinement improved G (dG real 3.4e-2 → 1.6e-2). `dG_holdout` is a change from round 0, not an error.

**Where production shows the problem: M1 with a loosened tail (§P1).** The estimated gain fell from 0.29 to 0.058
while the real-axis Σ went from +0.56 to +4.1. The fallback compares estimates, so it cannot see a real-axis
degradation. That is the case that makes F4 `!!`.

### P1 — the first-shell exemption disables refinement where the cap binds (`!`, confirmed on SMO; M1 pending)

`_exempt_pslots` (`gf_refinement.py:304-322`) exempts the seeds plus the heaviest first-shell rows that cover
`1 − FIRST_SHELL_EXEMPT_TAIL` (1e-6) of `|H s|²`, which is in practice the whole first shell.

SMO 2-bath, 40k, instrumented:
- About 35,100 of the 40,000 slots are **seed rows**. The first shell that fits under the cap is about 4,900 rows, and
  tail 1e-6 exempts all of them.
- The two cap-bound units had 3 and 0 swappable rows, while 244–26,160 candidates beat the worst retained row.

| arm (SMO 40k, graded against the uncapped reference, worst Im Σ_ref −1.15e-3) | worst Im Σ_ii real | dΣ/headroom | max dΣ_mats | dG real / mats |
|---|---|---|---|---|
| breadth | +6.77e-2 | 16.8 | 1.24e-3 | 3.4e-2 / 5.9e-3 |
| refine, tail 1e-6 (default) | +6.69e-2 | 16.6 | 5.6e-4 | 1.6e-2 / 2.7e-3 |
| refine, tail 1e-2 | **+1.73e-3** | **2.1** | 3.5e-4 | 8.0e-3 / 1.7e-3 |
| refine, tail 1 (seeds only) | +1.52e-3 | 2.0 | 3.6e-4 | 7.7e-3 / 1.7e-3 |

Loosening the tail moves Σ about 40x closer to causal; it is still slightly acausal (dΣ/headroom 2.1). The two
cap-bound units then swap 933/355/217 and 567/453/124 rows, and the estimated gain falls 0.19 → 0.09 and 0.48 → 0.06.

The oracle shows the other side of this:
- At 12 % (cap inside the first shell; the default leaves 0 swappable), tail 1e-3 helps: real-axis 0.438 → 0.393.
- Tails ≥ 1e-2 make the real-axis error worse (up to 0.533) while improving Matsubara (5.8e-3 → 2.8e-3).
- At 20–30 % the tail barely matters, because the first shell is small against the cap.

**Reading.** P1 and F4 are one problem. More swappable rows means more exposure to an acceptance rule that never
measures the error. The tail should not be loosened without an error-measuring acceptance (fix R1).

M1 (12 baths) at 65k (cap inside the seeds' first H-shell, §10):
- **Default.** 64,921–64,957 of the 65,000 slots are exempt, leaving 43–79 swappable rows while 7,579–9,297 candidates
  beat the worst retained row. It swaps 32/21/16, the §10.5 sequence.
- **Seeds only** (tail 1). Still about 60,000 rows exempt.

| arm (M1 65k, graded against the uncapped reference, worst Im Σ_ref −2.16e-3) | worst Im Σ_ii real | dΣ/headroom | max dΣ_mats | dG real / mats |
|---|---|---|---|---|
| breadth | +0.536 | 322 | 1.95e-2 | 2.5e-4 / 2.3e-2 |
| refine, tail 1e-6 (default) | +0.564 | 335 | 1.17e-2 | 2.3e-4 / 1.4e-2 |
| refine, tail 1e-2 | **+4.08** | 1648 | 2.85e-3 | **1.5e-3** / 2.6e-3 |
| refine, tail 1 | +4.00 | 1621 | 2.91e-3 | 1.4e-3 / 2.6e-3 |

On M1, loosening the tail improves the Matsubara side 4–5x (dΣ_mats, dG_mats). It degrades the real axis about 7x:
the worst Im Σ goes from +0.56 to +4.1, and dG_real from 2.3e-4 to 1.5e-3. This is the oracle's 12 % pattern again,
where the cap lies inside the first shell. The estimated gain still falls (0.29 → 0.058), so neither the settle nor
the fallback sees the damage.

**Reading.** P1 is real on both workloads: the default leaves essentially nothing swappable. Loosening it is not a
safe default, though:
- SMO (a cap above the first shell) improves on every axis;
- M1 (a cap inside the first shell) trades the real axis for Matsubara.

P1 and F4 are one problem: more swappable rows means more exposure to an acceptance rule that never measures the
error. **R2 must not ship without R1, and R1's measured error must include held-out real-axis points.** First-shell rows
carry the moments, and their removal shows up on the real axis first.

### F1 — thermal cross-pairing and no Boltzmann weight (`·`, latent behind `GF_EIGENSTATE_GROUP > 1`)

`unit_refine_spec` (`gf_refinement.py:170`) builds the unit's shifts as every thermal energy × every table entry.
`importance_scores` (`:72-92`) then maximises over *all* columns × *all* shifts. So the seed column of state k is
scored at state k′'s frame `z + E_k′`. That is a shift with no physical meaning for that column, and it can sit on one
of its poles. There is also no Boltzmann weight.

**Oracle: two stacked states, E1 − E0 = 0.319.**

| δ | top-k overlap with the correctly paired ranking (candidates / retained) | with the Boltzmann-weighted ranking (T = 0.05) | max s_cand, code / correctly paired |
|---|---|---|---|
| 0.05 | 0.84 / 0.88 | 0.71 / 0.71 | 1.5e-3 / 1.4e-3 |
| 0.01 | **0.29 / 0.38** | 0.35 / 0.35 | **1.33 / 2.2e-3 (600x)** |

At small δ the swaps are driven by meaningless shifts, and the inflated `gain` also keeps `settled` from firing.

**Production exposure: none at the defaults.** `GF_EIGENSTATE_GROUP` defaults to 1 (`config.py:422`), and
`gf_units.py:500-503` then builds one unit per thermal state. `unit_refine_spec` therefore receives a single energy,
and cross-pairing cannot happen. CrI3 confirms it: 3 blocks × 2 sides × 2 states = 12 units, and the log has 12 unit
lines. F1 bites only when that knob stacks states that are split by about δ or more.

Caveats on the oracle numbers:
- A cross shift is still a real-axis point of that state's own mesh, at ω − ΔE.
- The 600x at δ 0.01 is the near-pole conditioning of N1, solved exactly here. In production that shift would mostly
  fail to converge.
- The code arm QRs both columns jointly (r₀₁ = 0.12); the paired arm QRs each one separately. The overlaps therefore mix
  frame mixing with cross-pairing.
- "Pair each column with its state" is only defined in the r-weighted frame, so R3 depends on R7.

### F2 — frame mismatch (`·`, confirmed; ranking unaffected)

The scores use the orthonormal seed block `q0` (`gf_recurrence.py:1100`, `seeds_initial[1]`). The stop threshold uses
`tol · max|G|` with `G = g_cf(...)`, which carries the seed's R factor (`gf_recurrence.py:1209-1211`;
`gf_refinement.py:443, 465`). The oracle has |r|² = 0.5:
- the ranking is unchanged (Spearman 1.000, top-k overlap 0.98–1.00);
- the scores are 2x the physical-frame dG.

So the threshold is off by about 1/|r|². That is mild for well-filled orbitals and large for nearly empty or nearly
full ones. The ranking was tested only at a near-scalar r (0.707 / 0.696). With strongly unequal occupations in one
block, the maximum over the columns can reorder.

### F3, F5 and the array-kernel handoff

- **F3** (refuted as harmful). A score-only rerun that hits `max_blocks = 4·n_first + 100` (`gf_recurrence.py:1206`)
  returns `"max_iter"`, which skips the final GF rerun (`gf_refinement.py:496`). But the monitor's `converged_flag` is
  sticky (`gf_convergence.py`, `converged_flag[0] or is_conv`), and that run is longer than GF convergence needed. The
  unit reports honestly.
- **Array-kernel handoff** (refuted). `array_operator` exists only on `_ApplyEngine` (`gf_recurrence.py:593`), and
  refinement runs only in the discovered mode, on `_MatrixEngine`.
- **F5** (ping-pong: only the previous round's admissions are exempt) and **N5** (the `k //= 2` search finds a
  power-of-two-ish k, not the largest feasible one): not exercised.

## 2. Numerics

- **N2 (refuted): recursive against true residual.** Over 1200 blocks with orthogonality completely lost
  (‖VᴴV − I‖ ≈ 90), `ShiftedGalerkin.residuals` equals the true residual ‖Q₁ − (z − H)X‖ to every printed digit until
  the true residual reaches its attainable floor (3e-15). Only below that floor does the recursive value keep falling.
  X agrees with the dense solve to the same accuracy. The D-Lanczos update is sound without reorthogonalisation
**for a fixed P**.

Not exercised by this check:
- the width-change p-buffer path in `step`;
- `note_deflation`;
- `pad_rows` mid-recurrence;
- conjugate shifts;
- complex H.

There is also a production case it does not cover: the carried round-0 accumulator runs while P still grows, because
`_Admission` defers rows below swm that can enter later. `_needs_restart` checks only `prunes`. There, the recursive
residual is the residual of the recurrence as it ran, not of `P_final H P_final`.
- **N1 (confirmed): conditioning at the pole shifts.** At a bath-pole shift:
  - δ = 0.05: about 1000 blocks to reach 1e-4;
  - δ = 0.01: still 0.16 after 1200 blocks, in a 3920-dimensional space;
  - Matsubara shift: 1e-11 by block 400.

  The scoring shifts sit on the bath poles at the production δ (`selfenergy.py:218`); the BiCGSTAB-era `broaden`
  factor is gone.
- **S1 (confirmed down to 2e-3): how converged must X be?** Lanczos on H_PP, scored from the Galerkin X after m steps,
  with the top-k candidate and bottom-k retained sets compared against the exact-X sets:

  | δ | residual | candidate / retained overlap |
  |---|---|---|
  | 0.05 | 1e-3–2e-4 (m = 25) | 0.99–1.00 |
  | 0.01 | 2.6 (m = 25) | 0.33 / 0.24 |
  | 0.01 | 2.1e-3 (m = 50) | 1.00 / 1.00 |

  `REFINE_STOP_RTOL = 1e-4` (`gf_refinement.py:128`) is therefore stricter than the ranking needs by at least 20x.
  Residuals between 2e-3 and 1 (which includes the NiO §8.2 regime of 1e-2–1e-1) were not resolved by these step
  counts.
- **N4 (refuted): score quality.** Against the exact single-row dG, `s_ret` has Spearman 1.000 and top-k overlap
  0.96–1.00 (median ratio 1.00–1.01), and `s_cand` has Spearman 1.000 and top-k overlap 1.00 (median ratio 0.999). This
  holds at every cap and at both δ. The first-order `R_DD ≈ 1/(z − H_DD − Σ_D)` is not the weak link.
- **P2 (refuted): which shift decides.**
  - Oracle: no retained row ever takes its maximum at a Matsubara shift. 65–70 % take it at the strongest pole and the
    rest at ω≈0.
  - SMO: candidates win at the pole shifts; rows retired in the larger rounds win mostly at ω≈0; Matsubara shifts win
    only for 3–81 of the top 200 in a few passes.
- **N3 (not exercised).** The deflation floor was 0.0 in every log, so its drop above rtol
  (`gf_recurrence.py:805-807`) never ran.

## 3. Bottlenecks

- **B3 (confirmed in kind; size not separated).**
  - The accumulator is `(rows × n_z × w)`, stepped every block. At the default of one state per unit, `n_z` ≤ table size
    (8) × 2 conjugates when `need_y` (complex H such as CrI3).
  - CrI3's default arm is prune + refine, and its 1.5x is *not* the accumulator alone:
    - with `support.prunes`, every capped unit runs a second full recurrence from the seeds (`_needs_restart`,
      `gf_recurrence.py:966-969`, `:1161-1183`);
    - `n_blocks` reports only the last recurrence, so "identical block counts" hides the doubled run;
    - the `csr=` time is the whole recurrence wall.
  - Rough split of a heavy unit: 206 s ≈ 76 s growth + 76 s restart + ~55 s accumulator/scoring. Light units go from
    15 s to 40 s while settling at round 0.
  - The arm that would separate the two (`--prune-tol 0 --refine-rounds 3` on CrI3) was not run.
  - What is confirmed: every CrI3 unit ends with 0 swaps, so whatever the accumulator costs there is discarded. The
    exempt and swappable sets can be known when the cap freezes P, before any accumulator work, but nothing checks them.
  - §14's "the 1.5x is the unused scoring accumulator" inherits this confound.
- **B1 (refuted).** The `done and cap_hit` scoring gate (`gf_recurrence.py:798`) does not lengthen round 0:
  - CrI3 block counts of the final recurrence are identical;
  - SMO round 0 runs 1122–1543 blocks, which is GF convergence;
  - the score-only rounds after it take 101–131 blocks.
- **B5 (confirmed by reading).** Whenever the loop breaks after a score-only rerun (the settle test, `n_swap == 0`, or
  the last round), that rerun's scores feed only the `gain` record and the fallback test (`gf_refinement.py:465, 490`). It then runs a separate GF recurrence on the same P
  (`:496-498`). One combined run (stop when both are met, as round 0 does) saves about one scoring recurrence per
  refined unit. That is small on SMO (about 110 blocks) and larger on NiO (§12: the scoring rounds did not stop early at
  900–1500 blocks). Caveat: a merged run is GF-length, which is wasted if the fallback then fires.
- **B2 (refuted as a bottleneck).** The refine passes (scores + exempt + select) take 0.3–13 s per unit on SMO and CrI3,
  and 25–46 s on swapping units, against minutes of recurrence. The per-shift `exchange_plan` and matvecs are not worth
  batching yet.
- **B4.** `_g_of_solutions` (`gf_refinement.py:370`) and `matrix_is_real` (`:203`) have no callers in `src/`, including
  `.pyx`/`.pxi`.

## 4. Not run

- FCC Ni (the user's decision: skipped).
- CrI3 with a loosened tail.
- The NiO δ/4δ shift legs and the NiO profile.

- CrI3 with `--prune-tol 0 --refine-rounds 3`, the arm that would separate the accumulator from the pruning restart in
  B3.

These cost hours locally. The oracle answers the numerical question of the NiO legs: the ranking holds down to a
residual of 2e-3, and the pole shifts are what is slow.

## 5. Proposed fix series (not executed; each with a test red under the old code, gate -n 1/2/3)

| # | change | addresses | expected effect |
|---|---|---|---|
| R1 | Accept a round only if an *error proxy* improves, otherwise revert to the previous round's P (not round 0's) and stop. A proxy is needed: dG between rounds measures change, not error, and summing single-row identities is the additivity F4 indicts. Candidate: the frontier residual ‖H_FP X‖ (the true K-space residual of the P solution) at held-out real-axis and Matsubara shifts. It costs a held-out-quality solve per round, which eats into R5/R6 | F4 | stops regressing rounds; makes P1 safe to loosen (to be demonstrated on M1) |
| R2 | Loosen `FIRST_SHELL_EXEMPT_TAIL` (e.g. 1e-2) **only behind R1**, with R1 grading held-out real-axis G; keep the seeds exempt | P1 | SMO 40k worst Im Σ +6.7e-2 → +1.7e-3; on M1, R1 must reject the swaps that take +0.56 to +4.1 |
| R3 | When `GF_EIGENSTATE_GROUP > 1`: pair each column with its own state's shifts (needs R7's r-weighted frame), weight by Boltzmann, and dedupe shifts within a tolerance (≪ δ) | F1 | correct ranking for stacked split states; no effect at the defaults |
| R4 | Compute the exempt set at the freeze; if nothing is swappable, drop the accumulator before stepping | B3 | removes the accumulator share (~55 s per heavy CrI3 unit); the pruning restart stays |
| R5 | `REFINE_STOP_RTOL` 1e-4 → 1e-3 (supported down to 2e-3; looser needs its own check), or a block cap on score-only rounds | S1, N1 | shorter scoring rounds on NiO |
| R6 | Merge the last scoring round with the final GF run | B5 | one recurrence fewer per refined unit |
| R7 | Score in the r-weighted frame, or scale `tol` by the r factor | F2 | an honest settle threshold |
| R8 | Delete `_g_of_solutions` and `matrix_is_real` | B4 | — |
| R9 | Check the final GF rerun for `"diverged"` and fall back as the scoring reruns do | F6 | — |

Priority: R1 first (it gates R2); R4, R5 and R6 are independent cost cuts; R7 before R3.

The review was itself reviewed (Opus, read-only). Its corrections are folded in above:
- F1 is unreachable at the defaults;
- B3 is confounded by the pruning restart;
- R1's summed-identity option is unsound;
- the oracle loop differs from the library's;
- the N2 and F2 scope caveats;
- F6.

## 6. Implementation status (2026-10-30)

Landed, one commit each, gf directory green at `-n 1` (1060 passed), refinement tests green at `-n 1/2/3`:

| # | status |
|---|---|
| R8 | done: both functions deleted |
| R9 | done: a diverged final recurrence falls back to round 0 (test red without the check) |
| R5 | done: `REFINE_STOP_RTOL = 1e-3` |
| R4 | done: the scoring tail is skipped, and the unit declined with the reason, when the exempt set covers P; on the pruned path the accumulator is never created. Test covers both paths |
| R6 | done: the last permitted round also runs to GF convergence; early-stopping units still need the final run |
| R7 | done: the settle threshold uses |G| in the orthonormal seeds' frame (`record["scale_gain"]`) |
| R3 | **guard form only**: a unit stacking thermal states of distinct energies is declined with the reason. Per-state column pairing and Boltzmann weighting are not implemented (they need per-state column maps through the QR); the case is unreachable at `GF_EIGENSTATE_GROUP = 1` |
| R1 | **implemented, did not pass its M1 test** (below) |
| R2 | **not done**: `FIRST_SHELL_EXEMPT_TAIL` stays 1e-6 |

**R1 on M1 (65k, tail loosened to 1e-2, graded against the uncapped reference).** The acceptance test compares the
frontier error `sum_F |(H X)_f|^2/|z-H_ff|^2` summed over up to four held-out shifts. It accepted every round
(568, 498, 31 swaps) because the error fell monotonically (summed, e.g. 1.8e-3 → 3.6e-4), while the graded worst Im Σ
on the real axis stayed at +4.08, exactly the unguarded result. Two held-out placements were tried and both failed the same way:
the scoring shifts with doubled broadening, and with the production broadening displaced along the real axis by 5|Im z|.
Logs: `debug/refinement_review/m1_65k_t1e-2*.log`.

So the frontier residual at held-out shifts of this kind does not see the damage. On the fixture it does reject an
adversarial swap (test), so it protects against gross regressions only. It also costs up to four accumulator shifts per
refined unit that nothing yet shows to be worth it, which is a reason to reconsider `verify=True` as the default.
What R1 needs is an error measure that tracks the real-axis G or Σ itself, or a different view of why the real axis worsens
while every first-order estimate improves; that is open.
