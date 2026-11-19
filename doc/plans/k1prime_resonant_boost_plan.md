# K1′: resonance-boosted admission in pruned GF growth — reasoning and implementation plan

Status 2026-11-18: **superseded as a plan. The user chose a GF workflow redesign** (see `gf_redesign_handover.md`); this file is kept as input to that research, not for implementation. Original status: design gated, not settled. Part A explains how the design was reached. Part B is the plan.
Step 0 of Part B is a gate that two open measurements decide (E0c, running; the offline screen below already
narrowed the candidates). Part B's design is conditional on that gate. Companion documents:
`doc/plans/gf_resonant_admission_design.md` (rev. 1–4) and `debug/smo_causality/FINDINGS.md` (every number quoted here).

---

## Part A — how the design was reached

### A1. The physical requirement: an untruncated Σ is Herglotz

Σ(z) = G0⁻¹(z) − G⁻¹(z). For the exact G of H = H0 + U and any stationary density matrix (here the exactly
degenerate S = 3/2 quartet, thermal weights 1/4), Σ(ω + iδ) is a Herglotz function: Im Σ ≤ 0 on the whole real axis
for every δ > 0. G and G0 are built with the same h0 − dc and the same δ (`solver_basis.py`, `sigma.py:312`). A
mismatched double counting or chemical potential only shifts Σ by a real constant. So **every point with Im Σ > 0 is
a truncation.** Raising δ lowers the sensitivity to the truncation and does not remove it. It is not a fix.

### A2. Why the error lands where it does: |G0⁻¹|² is a magnifier

A change δG in G moves Σ by G⁻¹ δG G⁻¹. Near a pole of the hybridisation Δ(ω) = Σ_m |V_m|²/(ω − ε_m + iδ),
|G0⁻¹| ≈ |V_m|²/δ is large. Wherever Σ is small, |G⁻¹| ≈ |G0⁻¹| too. On SMO orbital 0, |G0⁻¹|² is 9.2 at ω = −0.28,
against a median of 0.88 over the mesh. The acausal window, [−0.283, −0.277], sits on the strongest bath mode of
orbital 0's chain (ε_m = −0.2822, |V_m|² = 2.1e-2). Even a tiny error in G there becomes a sign error in Im Σ:
Im G0⁻¹ ≈ 2.906 and Im G⁻¹ ≈ 2.899 differ by +6.8e-3.

### A3. What was cleared, and what the uncapped run truncates

Arm A (production settings, no unit at the cap) has Im Σ = +6.8e-3 at −0.28. All of these were cleared by
measurement:
- continued-fraction termination (dropping the last 25 % of blocks changes G by exactly 0);
- the restart, the mesh edge, off-diagonal elements, `lanczos_band`;
- the excitation bound (holes ≤ 8 is the physical maximum here);
- the chain window `[10,15,29,34]: (2,4)` (ΔG ≤ 2.8e-6 with the window off at the converged level).

The ground state is untested but has a large margin. **Pruning at η = 1e-3 alone explains the violation**:
the pruning-converged reference Σ_ref is causal at all 3,001 points, with Im Σ_ref(−0.28) = −3.0e-3.

### A4. What pruning drops: deep, weakly coupled chain-end holes

Unit 7 (removal, block 0), P(1e-3) against P(3e-4):
- P(3e-4) ⊇ P(1e-3), at the same depth (about 1,770 growth blocks). The 708k extra rows were *seen* by the 1e-3
  run at every depth, with |w|² ≈ 1e-7 (below η² = 1e-6 at every visit).
- They carry holes on the chain ends. With one hole on the four ε = −0.278 end sites, there are 10× more rows in
  P(3e-4); with two holes, 840× more.
- **Graft intervention.** Restrict admission at 3e-4 to an allowlist:
  - P(1e-3) + the 207k chain-end rows gives ΔG7(−0.28) = 3.0e-6 (from 2.7e-4);
  - P(1e-3) + the other 501k rows gives 2.75e-4, no change.
  - So the chain-end rows *cause* the −0.28 error and the rest do nothing there.
- **E0b.** Only the 101k chain-end rows one H-move away from P(1e-3) ∪ seeds are needed, giving 3.02e-6. The
  needed rows are on the frontier and get *rejected* there. They are not deeper parents that pruning never reached.

Geometry: orbital 0 → site 14 (V = 0.232) → 13 → 12 → 11 → 10, with hoppings 0.061, **0.021**, 0.058, **0.017**.
The end site 10 is reached through two weak bonds.

**The mechanism.** The admission rule tests the per-step Krylov amplitude |w_k(r)|², the amount transferred into
row r by one application of H. For a weakly coupled site that is small at every step. Its contribution to G at a
resonant frequency comes from the resolvent: ⟨r|(z − H)⁻¹|s⟩ ~ Σ_k w_k(r)·(stuff)/(z − E), and it accumulates
over many steps and is enhanced by 1/δ on resonance. A per-step amplitude threshold measures the *transfer rate*,
not the *resonant weight*. Weak coupling is exactly what makes a bath level a sharp pole of Δ, so the rule's
blind spot coincides with Σ's most fragile frequencies.

### A5. What this rules out, and why

| candidate | why it was dropped |
|---|---|
| raise δ | hides the truncation (A1) |
| mass rule / fill the cap fast (breadth-first) | user decision: pruning exists so growth gets enough H applications to carry holes and electrons deep into the bath; a seed-dense P stops after a few applications. A mass run agreed with the reference only at a 4M cap. |
| tighter fixed η | 3e-4 costs about 3.6× the support and is still acausal when capped (+2.1e-4 at the production cap of 618,681) |
| successive difference η → η/√10 | about 2× the cost of the accumulated rule; at the production cap round 2 caps anyway; an opt-in verifier at best |
| PT2 / `sigma_error` trigger | returns 0 in retained mode (K = P) and is non-perturbative at a pole |
| capped rules in general | the best capped rule (1e-4) misses Im Σ(−0.28) by 43 %; growth length does not order accuracy |
| **K1: boost by the row's diagonal energy** (EN-style \|z − H_ff\|) | **E0a: dead.** Rows sit at H_ff − E0 = +2.5 to +6.8 (medians), against the removal resonance at +0.28; none of 700k within 0.02. Correlation offsets swamp the one-body ε. This also kills Epstein–Nesbet ranking and C5 completion rounds on this model. |

What survives: the information that a row matters is **one-body** (which bath orbital holds the hole), not
many-body (the row's energy). That is K1′.

### A6. K1′, and a correction to what was first reported

**Idea.** Give each bath orbital i a weight B_i ≥ 1 that measures how strongly a hole (removal) or particle
(addition) on i feeds the frequencies where |G0⁻¹|² is large. At each growth step a frontier row r is admitted if
|w_k(r)|² · B(r) > η², with B(r) = max over the bath orbitals where r deviates from the reference of B_i. The
reference is valence full and conduction empty.

**Correction.** The first version evaluated ŵ = |G0⁻¹|²/median at each bath *site's* diagonal energy ε_i. It
reported a sharp contrast: 9.6 at the chain ends (ε = −0.278) against 0.06–0.48 elsewhere. That is the wrong basis.
The poles of Δ are the bath *eigenmodes*. In the mode basis, orbital 0's chain has three resonant modes:

| mode ε_m | \|V_m\|² | ŵ(ε_m) |
|---|---|---|
| −0.3045 | 5.6e-3 | 5.4 |
| −0.2822 | 2.1e-2 | 11.1 |
| −0.2706 | 1.5e-2 | 7.2 |
| −0.1757 | 4.4e-3 | 1.0 |
| −0.1521 | 9.8e-3 | 1.4 |

The per-site weight r_i = Σ_m |U_im|² ŵ(ε_m) is 8.7 on site 10, the end, and 3.0 / 4.3 / 4.6 / 5.6 on sites 11–14.
That is a contrast of about 2, not about 20. The site-energy contrast was partly a coincidence of the end site
sitting near the strongest mode.

**Offline screen** (`debug/smo_acausal/screen_tables.py`). For each candidate table, what fraction of the needed
101k rows and of the 501k do-nothing rows is lifted over η² = 1e-6? It uses the 3e-4 run's admission |w|², which is
about 1e-7 for both sets. It is a screen, not a confirmation.

| table | needed rows lifted | do-nothing rows lifted |
|---|---|---|
| site energy L(ε_i), every chain | 0.75 | 0.000 |
| mode-projected r_i, own chain 10–14 only | 0.23 | 0.012 (6k rows) |
| mode-projected, own chain + spin partner 29–33 | 0.30 | 0.016 |
| oracle: 10 on the four chain ends | 0.72 | 0.000 |
| oracle: 10 on orbital 10 only | 0.32 | 0.000 |

Two findings:
1. The needed rows hold their chain-end hole on all four ends: 45k on 10 (orbital 0's own chain, spin up),
   25k on 15 (orbital 1's chain), 17k on 29 (orbital 5, spin partner of 0), and 17k on 34 (orbital 6). A table built
   from block 0's own G0 can name only site 10.
2. A table that lifts 75 % of the needed rows and none of the rest exists, but it is the site-energy table, whose
   physical justification was just corrected.

So the open question is physical. **Does unit 7's G at −0.28 need the other chains' end holes (15, 29, 34),
which couple to orbital 0 only through U?** If it does, the boost has to use every block's resonances (the
resonant bath orbitals of the *whole* model), not the unit's own block. If it doesn't, the block's own
mode-projected table is the principled choice, but it lifts only about a quarter of the needed rows at a boost of
about 9, and the boost would have to be stronger or shaped differently.

E0c, the graft split, measures this directly (Part B, step 0).

### A7. The physics argument for a model-wide table — REFUTED by E0c (2026-11-18)

E0c: rows with a hole on orbital 10 alone fix −0.28 (3.03e-6); rows with holes only on 15/29/34 do nothing. The
unit's own chain suffices. The argument below is kept for the record.


A removal excitation at ω = −0.28 creates a hole that propagates from orbital 0 into its chain. The orbital-0
spectral weight at −0.28 belongs to many-body states in which the hole sits mostly in the −0.28 mode of chain 10–14.
In a correlated ground state, though, that excitation dresses itself through U with particle–hole pairs on the
other e_g orbitals (1, 5, 6). Chain 29–33, the spin partner, has an identical spectrum (verified to 5e-16). The ends of
chains 15–19 and 34–38 sit at the same ε = −0.278, but their mode weights in orbitals 1 and 6's G0 are not computed
yet. A pair whose hole sits in a resonant mode of *another*
chain is nearly degenerate with the excitation itself, so its amplitude is resonantly enhanced in exactly the same
way. That predicts the other ends do matter, and that the right table is model-wide: B_i from the resonant modes of
the block that owns orbital i, evaluated against the *unit's* frequency window. E0c tests this prediction.

### A8. Why K1′ can be cheap and safe

- **One-body data only:** h_bath, the impurity–bath hoppings, G0⁻¹ on the mesh. These are computed once per
  Σ evaluation, on rank 0, the way `_refine_tables` does it. No extra H applications, no growth passes.
- **Per row it costs bit tests:** mask the key against the resonant orbitals' bits. Only rows in the band
  η²/B_max < |w|² ≤ η² need testing, a small slice of the frontier.
- **Floored at 1:** where no bath level is a sharp pole (metals, NiO), every B_i is ≈ 1 and the rule reduces to the
  amplitude rule. A boost can only *add* rows, so accuracy cannot get worse; the risk is cost.
- **Keeps depth:** it does not fill the cap with near-seed rows, which respects the user's constraint. It changes
  *which* frontier rows are kept, not how many H applications growth gets.
- **Rank-invariant:** B(r) depends only on the determinant and replicated tables.

Cost on SMO unit 7: about +100k rows on a P of 337k (+30 %). The per-step growth cost scales with |P|.

---

## Part B — implementation plan

### Step 0 — gate (measurements, no library code)

1. **E0c graft split (running).** Unit 7 at 3e-4, allowlist = P(1e-3) ∪ seeds plus one of:
   - (a) the 45,210 hop-1 rows with a hole on orbital 10;
   - (b) the 55,910 hop-1 rows with chain-end holes only on 15, 29, 34.

   ΔG7(−0.28) against `p1/e_u7_3e-4.pkl`, compared with 3.0e-6 for both together. Outcomes:
   - (a) alone fixes it: own-block table, then step 0.2;
   - (b) is needed: model-wide table (A7);
   - neither alone works: model-wide table, and the max over orbitals stays.
2. **Table shape, offline.** With the outcome of 0.1, compute the candidate table from first principles:
   B_i = max over the blocks b that i couples into of Σ_m |U_im|² ŵ_b(ε_m), with ŵ_b = ‖G0,b⁻¹‖²/median. Then
   repeat the screen. Shaping options, judged on the screen:
   - exponent: B_i^γ;
   - max vs sum over a row's deviated orbitals;
   - resolving ŵ_b on the modes' own Lorentzians rather than the nearest mesh point.

   **Reject** any shape with a free parameter tuned to SMO that cannot be stated from physics.
3. **E1 prototype** (debug monkeypatch in `unit_replay.py`, `--boost TABLE`). It wraps `_Admission._pruned`,
   multiplying `norms2` by B(r) before the selection. Unit 7 at 1e-3, plus controls:
   - B ≡ c, a uniform boost with the same mean admitted count. This tells K1′ apart from "just lower η";
   - the own-block table vs the model-wide table;
   - growth forced to 2× length (P must not change much);
   - E1-stop: record the boosted rejected weight per step, and find where a stop criterion would have fired.

   Pass: ΔG7 ≤ 1e-5 on [−0.32, −0.24]; |P| well below P(3e-4)'s 1.05M; the uniform control does not pass at the
   same |P|.
4. **E2: all removal units 4–7 plus addition units 0–3**, and assemble Σ. Pass: Σ causal at all 3,001 points; the
   spin linearity G5 = (2G4 + G7)/3 holds to the reference's 8.4e-5. Then the production cap of 618,681: report
   whether units cap, and Σ there.

**Decision point.** The library work below starts only after step 0 passes, and after an adversarial physics +
numerics review of this plan with the step 0 numbers filled in. If step 0 fails, K1′ is dropped, and the
fallback is rev. 4's K2/K3 list.

### Step 1 — the table (library, `gf_refinement.py` + `selfenergy.py`)

- `resonance_boost_table(h_bath, v_full, g0_inv_of, omega_mesh, delta, blocks, n_orb, side)` in
  `gf_refinement.py` returns a float array of length n_orb (1 on the impurity orbitals).
- Computed in `selfenergy._refine_tables`, which already builds G0⁻¹ per block on rank 0 and broadcasts. The table
  rides in the same broadcast, so it stays replicated: a rank-local `eigh` would break rank invariance (the reason
  `_refine_tables` exists).
- Signed by side:
  - removal boosts holes in valence bath orbitals;
  - addition boosts particles in conduction bath orbitals;
  - each frequency is mapped by the same sign convention as `unit_refine_spec` (the removal frame is ω ↔ −ω).
- Floor at 1, cap at `B_MAX`, one named module constant with a docstring (no duplicated tolerance literals).
- Bits follow the MSB-first orbital convention (`orbital i = bit 7 − i % 8 of byte i // 8`). The table becomes a
  short list of (level, byte mask) pairs, so a row's boost is the largest level whose mask hits a deviation.

### Step 2 — the admission (library, `gf_recurrence.py`)

- `_Admission._pruned`: before `sel`, rows with η²/B_max < norms2 ≤ η² get norms2 ← norms2 · B(r).
  - B(r) comes from `keys_of(slots[band])`, only for that band.
  - Applied to the amplitude and accumulated rules alike (under `accumulated`, to the totals). Not under the first
    shell, which uses `first_shell_tol`.
  - The boosted norms2 also feeds `_top_and_freeze`, so at the cap the ranking prefers resonant rows. This is the
    cap-bound case's selection, which no capped rule got right before (A5).
- The collective structure stays the same: no new collectives, and the same `allreduce_sum(len(slots))`.
- Retained and discovered modes both go through `_Admission`, so both get the boost.
- Diagnostics: count the rows admitted only because of the boost, packed into the existing reductions of
  `cap_stats` (no new collective). Add a log line per run listing the resonant bath orbitals with their B_i and the
  modes behind them, so a user can see which bath levels are being protected.

### Step 3 — option and defaults

- `gf_prune_boost: {"auto", "off"}` in `options.py`, `auto` by default while pruning is on. That follows the user
  decision that pruning and refinement stay default ON. A knob read from the environment, if added, is registered
  uniform and checked by `config.assert_uniform`.
- `off` reproduces today's admission bit for bit (a test asserts it).

### Step 4 — tests

1. **Table unit tests.**
   - A synthetic star bath with one weakly coupled level at ω₀: that level's B_i is maximal and the others are ≈ 1.
   - A flat, metallic bath: every B_i is ≈ 1 (≤ 1 + ε).
   - The side sign convention.
   - Mode vs site basis: a rotated bath gives the same table after rotating back.
2. **Mask tests:** bits for orbitals 0, 7, 8, 63 (MSB-first), and multi-byte keys.
3. **Fixture test** (inject-a-bug discipline): a small impurity + weak-end chain model where the 1e-3 amplitude
   rule leaves a measurable ΔG at the resonance against an unpruned oracle, and the boost removes it. With the boost
   disabled inside the test, the assertion must fail.
4. **`off` parity:** P and G bitwise equal to the current rule.
5. **MPI:** targeted `-n 1`, `-n 2 --with-mpi`, `-n 3`, since admission feeds collectives and an empty rank must
   still take part. Parity of the admitted count across rank counts, within the layout noise the current rule
   already has.
6. **Regressions:** the NiO 5-bath and metal fixtures (`doc/gf_support_pipeline.md`) keep Σ, with |P| growth
   reported. The full gate (`-n 1/2/3`) runs once, at the end of the series.

### Step 5 — validation (local only)

- End to end: `freeze_replay run debug/smo_causality/smo.h5 --iteration 1` at defaults. Σ causal at δ = 0.01, wall
  time against A's 9.5k s.
- NiO 5-bath iteration 8 (Σ and wall time against 6.1 s / 20.7 s).
- LMO: no local run. The new log line (resonant orbitals, rows admitted by the boost) is the check on the next
  user run.
- Adversarial review (physics + numerics, Opus) of the diff and the end-to-end numbers.

### Commit series (single concern each, after step 0 and its review)

1. `gf refinement: resonance boost table from the bath modes and G0^-1` (+ table tests)
2. `gf selfenergy: the boost table rides the replicated refine-table broadcast`
3. `gf growth: resonant frontier rows admitted by boosted amplitude` (+ fixture, mask, parity tests)
4. `gf growth: rows admitted by the boost reported in cap_stats and the unit timing line`
5. `options: gf_prune_boost (auto|off)` + docs (`doc/known_issues.md`: the SMO violation was a truncation)

### Not addressed by K1′ (stays open)

- **The −0.81 error.** ΔG7 = 8.8e-4 at −0.81 persists in every graft, and it needs both row sets. It is a second
  truncation, not near a chain resonance of orbital 0. It is causal today, but it is the largest remaining ΔG.
- **The cap-bound case.** K1′ improves the cap-time ranking, but whether the production cap holds enough rows is
  measured in E2, not assumed.
- **LMO's slow growth.** K1′ adds rows, so it does not make growth cheaper. Only the E1-stop result could.
