# Pruned GF growth that keeps resonant rows, at a growth cost LMO can afford (design rev. 4)

Status: rev. 1 went through adversarial review R2 (physics, numerics/MPI). Rev. 2 is below; it overrides sections 3–6
where they conflict. Nothing implemented. Evidence: `debug/smo_causality/FINDINGS.md`.

## Rev. 4 after review of rev. 3 (physics + numerics). This section overrides rev. 3.

### What the review established about C5 (post-growth completion rounds)

1. **Reach.** Both reviewers measured independently how far the 207k chain-end graft rows sit from P(1e-3), through
   one-body moves (upper bounds; the seeds were not captured). Within 1 / 2 / 3 / 4 hops: **44 / 77 / 91 / 96 %**.
   - The parents along the chain have |z − H_ff| ≈ 3–7δ, so Epstein–Nesbet ranks them low. The rows behind them
     become visible only after the parents are admitted.
   - One round therefore reaches less than half, and 90 % needs R ≥ 3–4 rounds, each with a restart.
2. **The cause is collective.** The 207k rows cut ΔG7 by 2.7e-4 together, about 1e-9 per row on average.
   - A per-row threshold, or a pre-cut on max|HX|, is the amplitude rule again.
   - Admission must be cumulative: sort, then admit until the residual summed score is ≤ τ·tol (the CIPSI e_pt2
     budget), or take the top k.
3. **Cost.**
   - SMO needs 66–240 shifts at δ/2, and C5 costs ⌈N_z/C⌉ restart passes plus one wide apply per round.
   - `shifted_update` work grows as n_z·w(w+p), and the frontier at width C is about 15 GB globally (unpriced).
   - At R = 3 the completion costs about 96 min, against about 41 min for today's full growth.
   - `REFINE_MAX_SHIFTS` = 8 today, and a 2048-shift accumulator has OOM-killed ranks before.
4. **Plumbing.** `ShiftedGalerkin` is never stepped on the retained restart: it hands off to
   `_continue_on_array_kernel` at block 0 (gf_recurrence.py:926–939). Refinement is gated to discovered mode. The
   dressed Σ_f cannot be computed with K = P, since there is no squared operator apply.
5. **Accuracy of X at δ = 0.01.**
   - With residual rtol 1e-3 the error bound is ‖r‖/δ ≈ 10 % of ‖s‖, concentrated in exactly the resonant directions.
   - The deflation floor is amplified 100×.
   - Ghost behaviour was validated only for δ ≥ 0.03.
6. **Acceptance.**
   - `sigma_error` sees only the one-hop frontier, and is uncalibrated in this regime. It must not be called an
     honest error bar ("first-shell frontier estimate", calibrated against references).
   - Per-unit weights drop ρ_u and use G0⁻¹, not G⁻¹, and eight per-unit tolerances can add up.
   - Σ is causal only after assembly, which no unit sees. Per-unit admission can create the spin asymmetry
     ((4,7) against (5,6)).
7. **Where the danger is.**
   - |G0⁻¹|² is 9.2 at −0.28, against a median of 0.88. The worst |G0⁻¹|²·ΔG points are all at −0.280 to −0.284.
   - |G0⁻¹| is large at the bath poles of Δ(ω), so the dangerous frequencies are known a priori from G0, which the
     unit already receives (`g0_inv_of`). An earlier idea, a window from −Im G_unit, would have excluded −0.28 at a
     5 % threshold.
   - Separately, the largest ΔG7 overall (1.1e-3) is at −0.81, and needs both row sets.
8. **The energy signature is inferred, not measured.** "H_ff within δ of z" rests on the one-body ε = −0.278. The
   many-body diagonal energies of the graft rows have not been looked at.

### Consequence for the design

- C5 as a post-growth completion is **demoted to a backup**. Growth is the only mechanism that already provides reach
  (it propagates holes along the chain over about 1,750 steps).
- The resonance information has to act **inside growth's admission**, where rows are seen at every step.
- **Candidate K1: G0-weighted energy-aware admission in growth.**
  - At each step the rejected frontier rows (about 2e5 per step on SMO) get H_rr from `H.diagonal` (rank-local,
    cheap).
  - Their weight is boosted by L(E_r) = max_ω ŵ(ω)·δ²/((E_r − E0 ∓ ω)² + δ²), with ŵ = |G0⁻¹(ω)|² normalised by
    its mesh median, and the removal and addition sign frames as in `unit_refine_spec`.
  - The admission test is |w_k(r)|²·L(E_r) > η², per step, with no accumulation, so it has no ghost or
    growth-length drift.
  - On SMO the boost at −0.28 is about 10. That lifts the 1e-7 rows over η² = 1e-6. Away from bath resonances it is
    about 1, so selection is unchanged there.
  - Cost: one `H.diagonal` call per step on the cut-passing frontier rows, and no extra passes, shifts or memory.
  - Open questions:
    - whether to use the bare L, or to also cap it (L ≤ L_max);
    - which ω window to use;
    - whether a parent row (3–7δ off resonance) also needs boosting. If it does, L must have tails wider than δ.
      E0b and E1 answer this.
- **K2** is C5 with the review fixes: a cumulative budget, G0-weighted shifts, ρ_u, an oracle-checked X, and
  streaming. It is kept as the backup.
- **K3** is Ritz scoring: Ritz vectors in the window, one wide apply, rows scored analytically at z = H_ff + iδ, no
  mesh. A control.

### Experiments (debug rigs, in this order)

- **E0a. Energy signature.**
  - Histogram of H_ff − E0 (library `H.diagonal`, via a replay hook) for:
    - the 207k graft rows, by chain-end hole count;
    - the 501k other rows;
    - P(1e-3).
  - Overlay ŵ(ω). If the graft rows do not sit at the high-ŵ energies, K1 is dead.
- **E0b. Reach graft.** P(1e-3) plus only the hop-1 graft rows, seeds included. If ΔG7(−0.28) still falls to about
  3e-6, one shell suffices, and the parent-boost question is answered.
- **E1. K1 prototype on unit 7** (rig patch of `_pruned`). Judged by:
  - ΔG7 and ΔΣ per ω against the reference on the whole mesh, not on [−0.32, −0.24] alone;
  - |P|, growth wall, and the overlap with the graft set (secondary);
  - held-out ω (−0.81).

  Controls:
  - L ≡ 1, which must reproduce 1e-3;
  - L with ŵ ≡ 1, energy-only, which tests whether G0 weighting is needed;
  - growth forced to 2× length, which must not change P much.
- **E1-stop (added at the user's request): can K1 also end growth early?** Record, per growth step:
  - the boosted rejected weight, Σ over rejected rows of |w|²·L;
  - its maximum;
  - the admitted boosted weight.

  Offline, find the steps k* where the boosted rejected weight first drops below candidate tolerances. Freeze P at
  those steps (rows admitted ≤ k*) through the allowlist rig and restart. ΔG and ΔΣ per ω against the full K1 run and
  the reference tell whether a K1-based stop can replace most of the about 1,750 growth steps. Growth wall saved =
  (steps after k*) × the measured per-step cost. The criterion must use replicated quantities only (an allreduced
  sum or max), so it would be rank-invariant. This is the only route by which K1 could speed up LMO; it is judged on
  SMO, and LMO is checked later from its log diagnostics.
- **E2. All 8 units + assembly.** Σ per ω, the spin linearity G5 = (2G4 + G7)/3 (baseline 8.4e-5), and the
  cap-bound run at 618,681.
- **E3. Rank invariance** (`-n 2` against `-n 6`) and a NiO 5-bath it.8 regression (Σ unchanged where there is no
  bath resonance, cost).
- **E4.** The fixture for the eventual test: a 5-site weak-link chain with a dense-Lehmann oracle. Amplitude must
  fail and K1 must pass at the same η, with held-out shifts.

### Reporting regardless (C4)

- Assembled-Σ causality, with anti-Hermitian eigenvalues against the headroom.
- The spin-linearity check per block, where the GS is a multiplet.
- The rejected weight per step.
- Cap-bound units marked.
- Any frontier estimate labelled "first-shell, uncalibrated" until it is calibrated.

## Rev. 3: energy-aware completion (reopened by the user, 2026-11-18). This section overrides everything below.

### Why reopen

- **The graft intervention identifies the cause.** On SMO unit 7, adding only the 207k chain-end rows to P(1e-3)
  cuts ΔG7 at −0.28 from 2.7e-4 to 3.0e-6. Adding the other 501k rows changes nothing.
- **These rows have a common signature.** Their diagonal energy lies within δ of the frequency they spoil, and they are
  fed weakly. A per-step amplitude test cannot see the first property. The earlier rejection of
  resolvent-weighted admission (`gf-resolvent-weighted-selection-verdict`) was measured on a metal fixture with no
  isolated resonance. It also used per-step c_k weights, and every row reached in one step shares the same c_k.
- **The cap results add a second requirement.** No capped rule is accurate: the best misses Im Σ(−0.28) by 43 %. So the
  criterion must also decide *which* rows to keep at the cap. A first-come freeze cannot do that.

### What exists

`gf_refinement.py`: the importance probe and swap refinement, discovered mode only.
- **Scores**, for X = (z − PHP)⁻¹s from `ShiftedGalerkin`, weighted by |G0⁻¹(z)|² per shift:
  - a candidate D: s_cand(D) = max_z |(H X)_D|²/|z − H_DD − Σ_D|, with Σ_D the dressed first-order denominator;
  - a retained row: the exact removal score s_ret.
- **The error estimate:** `sigma_error` = |G0⁻¹|² max|dG2|.
- **Swap rounds** accepted on `_round_worse`.
- `ManyBodyOperator.diagonal(block)` gives H_DD for arbitrary determinants, rank-local.

Why none of it acted on SMO:
1. **The probe reads the frontier from the stored K.** SMO ran `retained`, where K = P, so every score is 0.
2. **The 16 real check shifts are 0.19 apart** (`g0_check_shifts`). A resonance of width δ = 0.01 falls between them.
3. **The probe and refinement only act on a capped unit.** Arm A was uncapped.

### Proposal C5: energy-aware completion rounds after growth (all matvec modes)

Per unit, after the restart has converged G on P:

1. **Shifts.** Real shifts z_j = ω_j + iδ at spacing ≤ δ/2 over the unit's spectral window, meaning the part of the
   evaluation mesh where the unit's own −Im G_unit exceeds a small fraction of its maximum. Add the existing
   Matsubara check shifts. Process the shifts in chunks of C, by memory.
2. **Solutions.** X_j for a chunk comes from one more recurrence pass on the stored P H P with `ShiftedGalerkin`, as
   the refinement already does. Cost: one restart pass per chunk (SMO unit 7: about 1,500 blocks at 0.155 s, about 4
   min).
3. **Frontier without K.** One unfiltered H apply of the width-C block X_chunk, as a growth step does. It gives
   (H X_j)_f on the rows outside P, then H_ff by `H.diagonal` on the rows that pass a pre-cut on max_j |(HX_j)_f|. The
   dressing Σ_f needs |H_fD'|² against P. To start with, use the bare denominator (`first_order=False`) and measure
   whether the dressing matters here.
4. **Score:** s(f) = max_j w_j |(H X_j)_f|²/|z_j − H_ff| (the `importance_scores` form), with w_j = |G0⁻¹(z_j)|²/Z.
   - Admit the rows whose predicted ΔΣ contribution exceeds τ × (the Σ tolerance).
   - At the cap, swap: evict by s_ret using the same shifts, as refinement does.
   - Accept or reject the round on `sigma_error` evaluated on the dense shifts, with `_round_worse` semantics.
5. **Restart** on the new P and repeat, for at most R rounds. The unit stops when max_j (predicted ΔΣ) is below the
   tolerance, or when nothing is admitted.
6. **Report:** the rounds, the rows admitted and swapped, and the final predicted ΔΣ per unit. That final ΔΣ is the
   honest error bar the diagnostics need, and it measures the remaining truncation in an uncapped unit too.

### Cost lever for LMO

- **Shorter growth.** Growth no longer has to find the resonant rows by itself, so it can stop on the existing
  `gf_growth_tol` or quiet stop, and the completion rounds supply what matters.
- **Price.** Growth trickles for about 1,750 steps at 9× a restart step. A completion round costs one restart pass per
  shift chunk, plus one wide apply.
- **Whether this wins** depends on R, C and the number of shifts. That must be measured (E2), not assumed.

### Experiments, before any library change (debug rigs)

- **E1, selection quality.** On SMO unit 7 from P(1e-3), one C5 round with about 32 shifts on [−0.32, −0.24] plus a
  coarse cover of the rest:
  - the overlap of the admitted set with the 207k graft set;
  - ΔG7 against the reference after the restart;
  - the bare against the dressed denominator.

  Control: the same number of rows admitted by plain |HX|², without the energy denominator.
- **E2, cost.** Growth stopped early (`gf_growth_tol` 1e-4, and quiet 64) plus C5 rounds, against today's full growth,
  on:
  - SMO unit 7;
  - the NiO 5-bath it.8 replay;
  - the metal fixture.

  Compare wall time and Σ against the references.
- **E3, cap.** Units 4, 5 and 7 at 618,681: today's freeze plus C5 swap rounds, against Σ_ref per ω.
- **E4, rank invariance.** E1 at `-n 2` against `-n 6`: ΔP and ΔG.

### Known risks, stated up front

- **The EN denominator is non-perturbative at the pole.** |z − H_ff| ~ δ makes the first-order gain an overestimate.
  That is acceptable for ranking, since rows are solved exactly after admission, but `sigma_error` as an acceptance
  measure was not validated where the support is acausal everywhere.
- **The shift count drives the cost.** It scales with spectral-window width / δ.
- **The diagonal-energy signature may be flat** on NiO-like workloads (the CIPSI-GF verdict). C5 must then cost
  little and change nothing, not hurt.
- **Rank invariance.** Scores are reduced from rank-local rows, and admission thresholds and cap-time ranking must be
  decided on replicated quantities (`rank_boundary`, `first_keys`).

## Rev. 2 after R2

**What R2 broke in rev. 1:**
- **C1's justification is wrong in the code that runs.** Growth uses `reort=False` (gf_recurrence.py:1295). After
  orthogonality is lost, ghost directions recur, so Σ_k |⟨r|Hq_k⟩|² is unbounded and grows with the step count. The
  effective threshold becomes η²/N_visits: accumulated at 1e-3 acts like amplitude at about 2e-5. P then depends on
  how long the G monitor lets growth run, which is the same lack of control charged to the amplitude rule. The rule
  is also ω-blind, and favours rows coupled to band-edge ghosts.
- **C3 will cut the same rows.** `_top_and_freeze` ranks only the current step's candidates, and earlier admissions
  stay, so a capped P favours rows admitted early. The late, slowly reached chain-end rows are exactly the ones lost.
  At 619k against 1–3.6M, no first-come rule at a fixed cap is both deep and accurate.
- **The diagnosis is correlational.** The composition shows which rows are missing, not which ones make Σ acausal.
- **C2 is overclaimed and incomplete:**
  - Emission dominates the apply: 96 %, against 1–2 % for the redistribution and lookup (apply_large_support_perf.md).
  - On NiO 5-bath, growth is only 2× the restart, and a stored-matrix growth pass was only 1.3× faster.
  - Coast steps pay for a rebuild of the exchange tables after each admitting probe (`_reached_owners`: an
    `np.unique` over nnz), a fresh exchange plan per `matvec` call, and unsorted slots.
  - Row extensions do not fit the contiguous `(c0, indptr)` segment format. `_permute_columns` would corrupt
    overlapping segments.
  - The resident matrix breaks both the memory guard and `_fits_now`: it is double-counted, the restart is pushed to
    `"none"` (about 6× slower), and P shrinks.
  - The ×m estimator aliases: |⟨r|Hq_k⟩|² oscillates like cos(kθ(ε)), which biases energy-selected rows.
- **Further points:**
  - The memory fallback (store above 25 % of the budget → amplitude) is likely to be the default path on large runs.
  - `amp2` is a max over the block columns.
  - swm cuts before accumulating.
  - The acceptance thresholds must be per ω on |ΔG|·|G0⁻¹|² against −Im Σ_ref, not |ΔG|/max|G|.
  - Rank invariance is unmeasured on a leaky unit.

**Rev. 2 plan: measure before choosing.**

*Phase M (debug rigs, no library change):*
1. **Graft intervention**, queued. Unit 7 at 3e-4 with admission restricted to P(1e-3) plus the 207k chain-end rows,
   and a control with P(1e-3) plus the other 501k rows. ΔG against the reference at −0.28 and over the mesh settles
   whether the chain-end rows are the cause.
2. **Cap-bound selection**, running. Units 4, 5 and 7 at cap 618,681, with accumulated 1e-3, amplitude 3e-4 and
   amplitude 1e-4. Σ is judged per ω against Σ_ref, and the growth blocks before the cap are recorded.
3. **Stop-rule sensitivity of `accumulated`.** Unit 7 with growth forced to about 2× the blocks, by a tighter monitor
   tolerance. If ΔP and ΔG are large, accumulated is rejected as it stands.
4. **Rank invariance.** Unit 7 accumulated at `-n 2` against `-n 6`: ΔP and ΔG.

*Candidate criteria to compare in M (selection), all with a finite, growth-length-independent measure:*
- (a) a sliding-window sum over the last L steps, with L tied to the energy resolution needed (L ≈ W/δ_res);
- (b) the exact lagged probe: stack the last m blocks and apply H once at width m·p. This gives the exact sum with no
  aliasing, and also serves as C2's cost lever if emission per row is width-insensitive. Measure that first;
- (c) amplitude at a tighter η, as the baseline.

  Judged on Σ against Σ_ref per ω, |P|, growth wall, and rank invariance.

*Cap-bound units:* swap at the cap is the only option consistent with depth: keep growing past the cap and evict by
a score with no early-admission bias. It is measured in M before it is proposed, and reported cap-bound.

*Spin symmetry, independent and cheap (the R2 physics finding):*
- G(M) = a + bM for the S = 3/2 quartet (Wigner–Eckart), validated offline to 8.4e-5.
- Solving the M = ±3/2 units of each side and completing the rest halves the removal cost, and removes the
  symmetry-breaking source of acausality.
- Prerequisite: the truncated GS quartet must be exactly S₋-related. Check this in the GS report (n₀ steps are
  linear to 4e-11).
- Design separately, as its own proposal to the user.

*C2 is deferred* until M shows which selection is right. Its prerequisites are a measured unsorted-slot matvec on
SMO unit 7, and the plan and owner-flag caching. Any C2 version must subtract the resident matrix in the guard and in
`_fits_now`, and re-price on every probe with one fixed-shape MAX.

*Tests (when code comes):*
- Fixture: a 5-site chain with two weak links (0.02, 0.017); the resonant level only in the 2-hole sector; a
  dense-Lehmann oracle; ε at an aliasing energy.
- Controls:
  - the new rule passes where the amplitude rule at the same η fails;
  - doubling the growth length leaves P unchanged within tolerance.

---

*(Rev. 1 follows, kept for the record.)*

## 1. The defect, as measured

SMO "Mn 1", iteration 1, δ = 0.01. Arm A was uncapped (max unit 521k against a cap of 619k), and Σ₀₀ was acausal at
ω = −0.28 (+6.8e-3). A reference Σ built from pruning-converged units is causal at all 3,001 points (−3.0e-3 at
−0.28).

- **The cause is pruning at η = 1e-3, and only pruning.** These are cleared:
  - the chain window (window off at the converged level: ΔG ≤ 2.8e-6);
  - the excitation bound;
  - the continued-fraction termination;
  - the restart;
  - the mesh.

  The ground state is not tested (margin 3e-3 against an estimated 8.5e-4).
- **What η = 1e-3 drops.** On unit 7, P(3e-4) ⊇ P(1e-3), adding 708k rows (+125,746 seed rows in both).
  - The missing rows are reached at every depth, at the same growth depth (about 1,760 blocks).
  - Each visit carries |w|² ≈ 1e-7, below η² = 1e-6, so they are seen and rejected at every visit.
  - They carry holes on the chain-end sites:
    - 1 hole: ×10;
    - 2 holes: ×840;
    - deepest hole at the end of a 5-site chain: ×12.
  - Those sites sit at ε = −0.278, within δ of the acausal ω. The chain is head → 13 → 12 → 11 → end, with hoppings
    0.061, **0.021**, 0.058, **0.017**.
- **Mechanism.** The per-step amplitude measures how fast weight flows into a row. A resonant row with weak links
  receives little weight per step, but its resolvent weight at ω ≈ ε is large and builds up over many steps.
- **P never closes under the amplitude rule.** Every growth step rejects about 5e-3 of the unit-normalised block's
  weight, 89–99.8 % of the leak, for 1,770 steps. "Converged" is the G monitor on a leaking P.
- **It is not a controlled approximation.** At η = 1e-3 the result moved by 4.3e-4 in G between two commits that only
  changed the growth-stop rules. It also breaks spin symmetry: the removal pair (5,6) against (4,7) gives Im Σ
  +1.35e-2 against +7e-5.

### Constraints (user decisions on file)

- Pruning exists to give the growth enough H applications to carry holes and electrons deep into the bath. Filling
  the cap quickly, breadth-first or with the mass rule, is not a fix. The mass rule reached 4M on unit 5 after 11
  growth blocks and was 1e-3 off.
- No reuse of P between SCF iterations. No growing breadth-first and pruning only once the cap binds.
- No per-row resolvent- or importance-weighted admission. It was measured on the metal and ranked no better than
  the leak.
- No quiet-window probe schedules. The heavy units never go quiet.
- Pruning and refinement stay on by default.
- LMO growth phases are already too slow. Nothing may make them slower, and the target is faster.

### Cost facts

- Growth wall ≈ steps × |P|. The weight store is not the cost. Per growth block:
  - unit 4: 1.0 s at |P| = 407k (1e-3), 3.2 s at 1.45M (3e-4), 4.0 s at 3.64M (accumulated);
  - unit 7: 0.51 s at 337k, 1.42 s at 1.05M.
- Every rule grows for about the same number of blocks (1,750–2,200), because the G monitor decides that.
- **A retained-mode growth block costs 9× a restart block** (SMO unit 7: 1.42 s against 0.155 s; metal: 13×). A
  growth step is a full matrix-free apply: it generates every image, sends it to its owner rank and looks up every
  key. The restart instead runs the stored P H P columns. In LMO-like runs, this is where the time goes.

## 2. Requirements

1. An uncapped unit's P is accurate enough that |ΔG|·|G0⁻¹|²/Z stays below the causality headroom. In practice,
   resonant rows reached slowly are admitted.
2. A cap-bound unit spends the cap on the rows that matter, after enough H applications, and is reported as cap-bound.
3. Growth wall is no worse on LMO, NiO or the metal fixture, and is better where growth dominates.
4. MPI rules:
   - decisions on replicated data only;
   - fixed-shape collectives, and no new collective under rank-local state;
   - every rank owning zero rows handled;
   - `config.assert_uniform` for any new knob.

## 3. Design

### C1. Admit by accumulated weight (the default rule becomes `accumulated`)

- A frontier row is admitted when Σ_k |⟨r|H|q_k⟩|² over the growth steps so far crosses η². The rule exists and is
  merged opt-in (`gf_prune_rule = accumulated`, `_KeyWeights` in the matrix-free modes, K-slot array in discovered
  mode).
- **Why this quantity.** The q_k are orthonormal, so by Bessel's inequality the sum is bounded by ‖Π_K H|r⟩‖², the
  row's coupling to the Krylov space explored so far. It is finite and convergent, and it counts a row reached slowly
  as fully as one reached at once. It is not the resolvent weight, but it no longer discards a row because its weight
  arrives slowly, which is exactly the SMO failure.
- **Evidence.** On SMO unit 4 at η = 1e-3, accumulated gives 3.64M rows, uncapped. It agrees with 3e-4 to 8e-5 and with
  the mass rule at 4M to 1.6e-4.
- **Its cost on SMO is the size of the support the physics needs.** It is 4× the 1e-3 amplitude run, at about 1.1 µs
  per row and block against 2.5 µs. C2 addresses that cost.
- **Caveat.** The bound needs orthonormal q_k. Retained growth runs without reorthogonalization. Over about 2,000
  steps orthogonality is lost and ghost copies of converged Ritz directions recur, so the sum can exceed the bound
  and grow with the step count. The rule's admissions keep trickling to the end, which is consistent with this.
  Whether ghosts inflate admissions measurably is open.
- **Defaults kept:** η = 1e-3 and `first_shell_tol`. The memory fallback (store > 25 % of the budget → amplitude rule)
  must be **reported**, today it only prints. It must also be visible in the diagnostics, since it silently brings
  the defect back.
- **Open item:** `rule_fallback` is not in `cap_stats` (known review item).

### C2. Strided frontier probes over an incrementally kept P H P

- **Step types.** A growth step is either:
  - a **probe**: the full matrix-free apply as today, which gives the frontier, the admission, and w on P; or
  - a **coast**: a stored-matrix `matvec(rows="P")`, as the restart uses.
- **Schedule.** A probe runs every m-th step. The first S steps are all probes, which covers the first shell and the
  early fan-out. m and S are constants; whether they become knobs is open, and a knob needs `assert_uniform`.
- **Accumulated weight under striding.** The weight is taken at probe steps only and multiplied by m. That estimates
  Σ_k over all steps if |⟨r|H|q_k⟩|² varies slowly with k, which is an assumption to validate (section 4). Rows are
  admitted at probe steps, so admission is delayed by at most m − 1 steps. The growth coefficients are discarded
  anyway: the restart recomputes G on the final P.
- **Keeping P H P exact while P grows (new matrix capability):**
  - Admitting rows builds their columns with `keep_frontier=False`. That stores entries to every row already in P,
    so H(j_old, i_new) is present.
  - The missing entries are H(i_new, j_old) = conj(H(j_old, i_new)). They sit in the new column at row j, owned by
    the column owner of i, and must move to the owner of column j: one `Alltoallv` of (j_slot, packed i, conj v) per
    admission batch.
  - The receiver appends them as a row-extension segment. Segments are merged log-structured (merge when the
    unmerged extension nnz exceeds half the merged nnz), so the merge cost stays O(nnz log) over the run, not O(nnz)
    per probe.
  - `sort_p_slots` and `build_columns` at the freeze become a no-op or a renumbering.
- **Memory.** P H P is resident during growth, at the size the freeze already prices with `_fits_now`. On a probe
  step it coexists with the apply's peak. The mode decision prices it at the start against the real |P| measured so
  far (the "price discovered mode at the real |P|" lead). If it does not fit, the unit runs m = 1, which is today's
  path.
- **Expected growth cost.** (c_probe + (m−1)·c_coast)/m, with c_probe/c_coast = 9–13. At m = 8 that is about
  2–2.5·c_coast against 9–13, so growth is about 4–5× faster at equal |P|. This holds for every rule and every
  workload, including LMO, where growth dominates. It needs a measurement on LMO-shaped data (from the existing
  logs, no local LMO run).
- **Rank invariance.** m, S and the schedule are replicated. A probe or coast decision depends only on the step
  index. The admission is as today.
- **Discovered mode** (store the frontier rows K as well) is rejected for these sizes. The frontier is tens of millions
  of rows per unit on SMO: the 1e-3 run rejected about 4e8 row-visits.

### C3. Cap-bound units (pending the cap measurement, section 4.1)

- **Today.** `_top_and_freeze` ranks the candidates by `norms2`, which are the totals under `accumulated`, then
  freezes. `stop_at_freeze` ends growth there, and the restart runs on the frozen P.
- **Question.** How many H applications has an accumulated-rule unit made when it reaches 619k, and is its Σ causal
  and accurate against Σ_ref? Arm B (amplitude 1e-4, capped after 36–93 growth blocks) was causal, but that is not a
  result: it differs from A in three ways at once.
- **Decision rule:**
  - **(a)** If the capped accumulated rule is within the headroom of Σ_ref at −0.28, keep (a) and report the unit
    as cap-bound.
  - **(b)** Otherwise, the cap must be spent later. Options for the user:
    - **swap at the cap**: keep growing after the cap, evicting rows by accumulated weight, which is close to the
      existing refinement machinery;
    - **a larger cap** for such units (memory).

  Not decided before the data.

### C4. Honest reporting (plan phase 5, unchanged)

- A `pruning` diagnostic row per block: the rule (and any fallback), the rejected weight per step (the phase 3
  diagnostic), and the store's total undelivered weight. A capped unit is marked cap-bound.
- `basis_cap OK` must no longer read as "complete".
- `doc/known_issues.md` is rewritten.
- The `_pole_note` and `_attribute_block` texts name pruning.

## 4. Validation (before any default flips)

1. **Cap-bound selection, SMO** (running): removal units 4, 5 and 7 at cap 618,681, with three rules: accumulated
   1e-3, amplitude 3e-4 and amplitude 1e-4. Measures: Σ against Σ_ref, growth blocks before the cap, and composition.
2. **C2 stride sweep.** m ∈ {1, 4, 8, 16} on:
   - SMO unit 7 (accumulated, uncapped);
   - the NiO 5-bath it.8 replay;
   - the metal fixture.

   Measure ΔP (symmetric difference), ΔG against m = 1, and growth wall. Accept m when ΔG < 1e-5·max|G| and ΔΣ is
   below the headroom everywhere.
3. **Regression arms on default flips:**
   - NiO 5-bath it.8: Σ unchanged, cost reported;
   - metal fixture;
   - LMO: none locally; the next LMO log carries the new diagnostics.
4. **End to end:** SMO it.1 at default settings. Σ is causal at δ = 0.01, or the units are reported cap-bound. Wall
   is recorded against A's 9.5k s.

## 5. Tests

- **C1:** a fixture where a weakly linked resonant site is lost under `amplitude` and admitted under `accumulated`. A
  model chain with a 0.02 link and the end level at the probe frequency; the test checks ΔG, not counts. The fallback
  is reported in `cap_stats`.
- **C2:**
  - the incremental P H P equals a fresh `build_columns` (dense comparison) after every admission batch, at `-n 1/2/3`,
    with an empty rank at `-n 3`;
  - a coast step equals a probe step's w on P bit for bit up to summation order;
  - parity m = 1 against today's code: bit-identical;
  - the stride's G against m = 1 within tolerance.
- **MPI:**
  - the extension `Alltoallv` runs on every rank, every probe step, with fixed dtypes, including empty sends;
  - `-n 3` targeted runs;
  - the full gate once at the end of the series.
- Inject bugs to check that the tests discriminate: drop the adjoint fill, stride without the ×m.

## 6. Order of work (single-concern commits)

1. The phase 3 diagnostic: rejected weight in the packed allreduce. Reporting only.
2. C2 matrix capability, the row-extension exchange, with its own tests. No behaviour change.
3. C2 schedule (m = 1 default, bit-identical), then validation 4.2, then the default m.
4. C1 default flip and fallback reporting, then validation 4.3.
5. C3 per the cap data.
6. C4 docs and diagnostics. Review R3, then the gate.

## 7. Open questions for the user

1. Make `accumulated` the default rule? This changes every pruned run's support.
2. C3 (b), if needed: swap at the cap, or a larger cap?
3. m and S as constants, or as knobs?
