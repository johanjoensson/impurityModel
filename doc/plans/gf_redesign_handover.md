# GF workflow redesign — findings handover (2026-11-18)

**Purpose.** This session measured *why* pruned GF growth gives an acausal Σ on SMO when no unit reaches the cap.
It gathered data and wrote it down. A separate session will research how to redesign the GF workflow to fix this
without a large amount of code clutter. **This document holds no design decision.** The K1′ plan
(`k1prime_resonant_boost_plan.md`) is one candidate direction, recorded as input for that session, not something to
implement.

Detailed numbers: `debug/smo_causality/FINDINGS.md`. The rev. 1–4 design history: `gf_resonant_admission_design.md`.
The K1′ reasoning (physics, the site-vs-mode correction, the offline screen): `k1prime_resonant_boost_plan.md` Part A.
`debug/` is gitignored; everything below is local to this machine.

---

## 1. The problem, in one paragraph

Σ = G0⁻¹ − G⁻¹ is Herglotz for the exact model, so every Im Σ > 0 is a truncation. On SMO ("Mn 1", archive
iteration 1, δ = 0.01), the production run had no unit at the cap and still gave Im Σ = +6.8e-3 at ω = −0.28,
orbital 0. The cause is **pruned GF growth at η = 1e-3**. It rejects, at every step, frontier rows whose per-step
Krylov amplitude is small (|w|² ≈ 1e-7 < η² = 1e-6), but which carry holes on weakly coupled chain-end bath sites.
Those rows hold the resolvent weight at a sharp pole of Δ, where Σ magnifies any ΔG by |G0⁻¹|² (about 10× the
median). The per-step amplitude measures the *transfer rate* into a row, not its *resonant weight*.

## 2. Established facts (measured; see FINDINGS.md for the tables)

**Cause**
1. Σ_ref, from pruning-converged units (3e-4 / accumulated rule, deep references), is causal at all 3,001 points.
   Im Σ_ref(−0.28) = −3.0e-3, against +6.8e-3 at the 1e-3 default.
2. Cleared as causes:
   - continued-fraction termination;
   - the restart;
   - the mesh;
   - off-diagonal elements;
   - the excitation bound;
   - the chain window (ΔG ≤ 2.8e-6 with it off);
   - layout noise (2.8e-14).
   The ground state is untested, but its margin is fine.
3. **The rows that matter**, on unit 7 (removal, block 0, M = +3/2):
   - P(3e-4) ⊇ P(1e-3), at the same depth (about 1,770 growth blocks).
   - The 708k extra rows were seen at every depth, with |w|² ≈ 1e-7.
   - Grafting only the 207k chain-end-hole rows fixes −0.28: ΔG7 goes from 2.7e-4 to 3.0e-6. The other 501k rows
     change nothing there.
   - Only the 101k chain-end rows **one H-move** from P(1e-3) ∪ seeds are needed (E0b: 3.02e-6). They sit on the
     frontier and are rejected there.
   - E0c: the 45k of those with a hole on orbital 10 (the end of the unit's own chain) suffice: 3.03e-6. The other
     chains' ends do nothing (section 6).

**Reference and spin checks**
4. A second, independent truncation remains at ω ≈ −0.81 (ΔG7 ≈ 8.8e-4 in every graft). It needs both row sets.
   It is causal today, but it is the largest remaining ΔG.
5. Pruning **breaks the spin symmetry** of an exact S = 3/2 multiplet: the removal spread (4,7) vs (5,6) is
   1.35e-2 in Im Σ at 1e-3. The exact G is linear in M, G5 = (2G4 + G7)/3, which holds to 8.4e-5 with deep
   references. This is a free, built-in truncation diagnostic, and the current workflow does not use it.
6. Addition units are pruned too: units 1 and 2 roughly double from 1e-3 to 3e-4, with ΔG up to 7e-4 at ω ≈ 1.13.
   This affects accuracy, not causality, there.

**Behaviour of the current workflow**
7. The pruned result at η = 1e-3 shifts by 4.3e-4 in G between HEAD and pre-HEAD growth-stop rules. That is as
   large as the error that flips the sign of Im Σ: the result depends on stopping heuristics.
8. **Cap-bound selection (cap 618,681):**

   | rule | Im Σ(−0.28) | result |
   |---|---|---|
   | reference | −3.0e-3 | — |
   | 1e-4 | −1.70e-3 | best capped rule, still off by 43 % |
   | accumulated | −1.21e-3 | |
   | 3e-4 | +2.1e-4 | acausal |

   No capped rule is accurate, and growth length does not order accuracy.
9. The mass rule (breadth-first, fills the cap fast) is shallow: unit 5 capped after 11 blocks and missed by 1e-3
   at −0.81. **User decision: cap-filling is not a fix; pruning exists to give growth depth.**
10. Diagonal energies don't locate the resonance (E0a). H_ff − E0 of all relevant rows is +2.5 to +6.8, against
    +0.28. Epstein–Nesbet / energy-denominator ranking, K1, and CIPSI-style completion rounds are dead on this model.
11. One-body (bath-orbital) information does separate the needed rows: a boost on the four chain-end orbitals lifts
    72 % of them and none of the 501k do-nothing rows. But the principled per-block, mode-projected weight reaches
    only 23 % (contrast about 2 between the end site and its chain neighbours, not about 20). The first
    "site-energy" table that looked selective used the wrong basis. Details in `k1prime_resonant_boost_plan.md` A6.
12. Growth costs: a growth step is a full matrix-free apply, about 9× a restart step on SMO and about 13× on LMO.
    Growth runs about 1,750–2,200 blocks for every η tried, because the G monitor decides when it stops, not the
    admission rule. Admission keeps trickling until the last block. P never closes. Growth's own G is discarded and
    the restart recomputes it. SMO arm A: 9.5k s makespan, of which growth is about 95 %.

## 3. Structural observations about the code (inputs for the redesign, not proposals)

- GF workflow modules: about 12.4k lines in `gf_*.py` + `greens_function.py` + `selfenergy.py` + `sigma.py`.
  `gf_recurrence.py` (1.6k) holds growth admission (`_Admission`, `_pruned`, `_top_and_freeze`, `_KeyWeights`).
  `gf_refinement.py` (0.7k) holds post-growth refinement (`importance_scores`, `sigma_error`, `g0_zero_shifts`).
  `gf_admission.py` (0.3k) is the per-frequency BiCGSTAB outer loop.
- **Three admission mechanisms are separate code paths:**
  - growth pruning (rules `amplitude` / `accumulated` / `mass`);
  - post-growth refinement (swap rounds scored at G0⁻¹-zero shifts, discovered mode only);
  - the BiCGSTAB per-point outer loop.

  Each has its own threshold, scoring and stop logic.
- **The knowledge that Σ is fragile at the zeros of G0 already exists**, but only in refinement:
  `g0_zero_shifts` computes the |G0⁻¹| peaks and `_refine_tables` broadcasts them. Growth admission and its
  cap-time ranking never see it.
- Refinement's `sigma_error` returns 0 in retained mode (K = P). `ShiftedGalerkin` is never stepped on the
  retained restart (array-kernel hand-off, `gf_recurrence.py` about l.926–939). So in the default matvec mode there
  is no Σ-aware check of growth's output at all.
- `_top_and_freeze` ranks only the current step's candidates at the moment the cap binds, not the whole P.
- Stopping is decided by a G monitor on growth's own (discarded) G: quiet stop, plateau checkpoints every
  √2× blocks, freeze at the restart. Point 7 shows these heuristics move the answer by the size of the defect.
- Every Σ-relevant criterion must be replicated: allreduced quantities only, tables computed on rank 0 and
  broadcast (`_refine_tables` exists for exactly this reason).

## 4. Constraints the redesign must respect (user decisions on file)

- If no unit reaches the cap, Σ must be causal, and accurate to the causality headroom. Raising δ is not a fix.
- Pruning exists for **depth**. No cap-filling / breadth-first rule as the production fix.
- Pruning and refinement stay **default ON**.
- No P reuse across SCF iterations; no "prune only when the cap binds".
- LMO growth is already too slow; the redesign must not make it slower and should make it faster.
- The ground state keeps every H symmetry (orbit-closed selection). The spin breaking of point 5 is a GF-side
  defect worth fixing for the same reason.
- Asymmetric baths are the norm: don't design around particle–hole symmetry. Complex Hermitian H must work, so no
  complex-symmetric shortcuts.
- Local runs only: no cluster runs, no big local LMO replays, at most one replay at a time on this machine.
  Commit only when asked, never push upstream.
- MPI rules in CLAUDE.md: no collectives gated on rank-local state, no full state gathers, uniform knobs.

## 5. Questions for the research session

1. What should decide *which* frontier rows growth keeps: per-step amplitude, accumulated amplitude, a one-body
   resonance weight, or a Σ-weighted resolvent estimate? Can one criterion serve growth, cap-time ranking and
   refinement, so that three code paths become one?
2. Can the restart's G (which is kept) drive admission and stopping, instead of growth's discarded G, without
   paying growth's 9–13× per-step cost?
3. Is there a cheap, replicated, rank-invariant error estimate of G (or Σ) at the G0⁻¹ zeros in retained mode,
   where PT2 is identically 0? Candidates seen: successive difference (expensive), spin-linearity (free when an
   exact multiplet exists), closure bounds ‖QHx‖ (needs x stored).
4. What does an honest cap-bound unit look like (which rows at the cap, what is reported), given that no capped
   rule is accurate here?
5. Where does the −0.81 truncation come from? It is not a chain resonance of orbital 0.
6. E0c settled that the unit's own chain suffices (section 6). Open: a *principled* measure that ranks the needed
   own-chain rows high enough. The mode-projected one-body weight gets about half the needed lift.

## 6. E0c — graft split

Unit 7 at 3e-4, allowlist = P(1e-3) ∪ seeds plus:

| leg | added rows | ΔG7(−0.28) | max on [−0.32, −0.24] | mesh max |
|---|---|---|---|---|
| all 101k hop-1 chain-end rows (E0b) | 101,120 | 3.02e-6 | 1.51e-5 | 8.8e-4 at −0.81 |
| hole on orbital 10 (own chain) | 45,210 | **3.03e-6** | 1.51e-5 | 9.0e-4 at −0.81 |
| ends 15/29/34 only | 55,910 | 2.72e-4 (no change) | 2.74e-4 | 9.8e-4 at −0.81 |

**Verdict: the block's own chain end is enough.** The 45k rows with a hole on orbital 10, the end of the chain that
couples to orbital 0, fix −0.28 completely. The 56k rows with holes on the other chains' ends do nothing. The
prediction in `k1prime_resonant_boost_plan.md` A7 (dressing through U makes the other chains' resonances matter) is
refuted. Whatever criterion is chosen, it can be built from the unit's own block's G0 and bath.

Screen restricted to these 45k rows (lifted over η² = 1e-6 with the 3e-4 run's |w|²):

| table | lifted |
|---|---|
| site energy | 0.735 |
| mode-projected, own chain | 0.433 |
| oracle: 10 on orbital 10 | 0.704 |

So the principled mode-projected weight (8.7 on site 10) is about half strong enough. It would take
|w|² ≈ 1e-7 × 8.7 < 1e-6 for most rows. Whether a principled shape exists (a per-step vs accumulated measure, a
resolvent rather than a per-step amplitude) is open.

Rows are also one H-move from P(1e-3) ∪ seeds (E0b). Combined: **the rows that decide Σ at the resonance are the
frontier rows, rejected at the threshold, that put a hole on the weakly coupled end site of the unit's own
chain.**

## 7. Data and tools inventory (local, `debug/`, gitignored)

- **Workload:** `debug/smo_causality/smo.h5`; arm logs/npz `A`, `B`, `D` (A = production settings);
  `A.npz` gives G_A, Σ_A, so G0⁻¹ = 1/G_A + Σ_A for orbital 0.
- **Rig:** `debug/smo_acausal/unit_replay.py`. It replays chosen GF units of the archive. Flags:
  - `--units`, `--prune-tol`, `--prune-rule`, `--gf-cap`, `--no-chain`, `--swm`, `--e-pt2-tol`;
  - `--capture` (per-step admitted keys, |w|², rejected histogram);
  - `--allow FILE` (graft: restrict admission to a key set);
  - `--diag IN OUT` (H_ff of key lists);
  - it prints `PEAKRSS`.
  Run under `systemd-run --user --scope -p MemoryMax=12G mpiexec -n 6`, with `OPENBLAS_NUM_THREADS=1`. The ground
  state takes about 9 min per launch.
- **Analysis:**
  - `offline_sigma.py`: per-unit continued fractions → G, Σ, G0⁻¹ for orbital 0;
  - `comp_analyse.py`: key decoding (MSB-first) and composition tables;
  - `cap_sigma.py`: Σ from capped units;
  - `hop1.py`: one-H-move neighbourhood;
  - `screen_tables.py`: the boost-table screen.
- **Unit pickles:**
  - `debug/smo_acausal/units/` (unit 4 at many η, accumulated);
  - `p1/` (all 8 units at 1e-3, deep references `e_u7_3e-4`, `j_u5_3e-4`, mass-rule legs);
  - `cap/` (capped legs);
  - `comp/` (captures, grafts, allowlists, `diag_out.pkl`).
- **Known rig traps:**
  - `SlaterDeterminant.from_bytes` pads to 512 bytes; build from `int.from_bytes(k[:8], "big")`.
  - Pickles written at the replay's exit must use `with open(...)`, or they are truncated.
  - `pgrep -f`/`kill` patterns must not match the calling shell's own command line.
- **Uncommitted library change:** `src/impurityModel/test/support/freeze_replay.py` (+7 lines, a `--delta`
  override). Still to verify with `-n 2 --with-mpi` on `test_gf_freeze_replay.py`, or revert.
