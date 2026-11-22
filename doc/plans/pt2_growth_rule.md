# PT2 growth rule for the Green's-function basis: gate verdict and implementation plan

Status: prototype measured (gitignored `debug/pt2_growth/FINDINGS.md`, Steps 3g–3q); independent Opus review done
2026-11-22 (verdict: proceed with conditions). No library code written yet. Branch `pt2-growth-prototype`.

## The rule

During pruned growth, after the `slaterWeightMin` cut and the whole first shell, a frontier determinant `f` gets

    s_f = max_k  sum_cols |rho_f(z_k)|^2 / |z_k - H_ff| / ||G_P(z_k)||,     rho = H X(z),  X = (z - H_PP)^-1 seed,

maximised over a uniform mesh `z_k` across the unit's own real-frequency evaluation window at the run's own broadening.
Rows are admitted in descending score until the remaining score mass is at most `tau` (one collective tail cut,
`collective_mass_cutoff`). The rule **replaces** the amplitude rule (`gf_prune_tol`); it has no hand-placed frequency.

`X` costs no extra H application: a K-row `ShiftedGalerkin` is stepped with the matvec output `w_j = H v_j`, so it
accumulates `rho` directly. `||G_P(z_k)||` is the seed block of `(z - T)^-1` from the recurrence's own alpha/beta.

Measured meaning of `tau`: a budget on the relative error of the unit's G, `max|dG| / max|G|`. Scoring every block the worst
unit is 0.08–0.87 tau (NiO, FCC Ni); scoring every 4th block it reaches 1.6 tau on paramagnetic NiO; on SMO the rule
delivers 100x better than tau.

## Gate verdict

| material | result at equal rows | cost |
|---|---|---|
| SMO (units 4/5/7) | PT2 tau 1e-3, 1.81M rows: max dSigma <= 1e-4 (reference floor), production 7e-4 at 1.72M: 3.6e-4, production 1e-3 at 1.25M: 7.6e-4 | wall 2501 s vs 1981 s (production 1e-3), peak RSS +40 % |
| NiO AFM 8-bath | tau 1e-4 (every 4th block): 15x lower error than amplitude 1e-3 at 8 % fewer rows | growth 19 s vs 25 s |
| NiO PM 10-bath | tau 1e-4: 4x lower error at 22 % fewer rows; tau 1e-3: 2.4x worse than amplitude 1e-3 at 42 % fewer rows; above ~50k rows unmeasured | growth 14 s vs 9 s |
| FCC Ni 4-bath | parity in rows vs error | growth 2.4–3.3x slower, wall 1.4–2.3x at equal rows |
| cap-bound (SMO u7, cap 230k) | PT2 1.5e-3 (dG at -0.28: 3.3e-5) vs amplitude rule 3.6e-3 (2.2e-3) | — |

Why FCC Ni shows no gain: more than half of all frontier determinants are resonant inside the window (min |z - H_ff| ~
broadening), so the denominator does not discriminate; the two rules admit 30–50 % different rows at equal accuracy.

Verdict: the rule is sound and better per row where a resonant minority exists (SMO, NiO), neutral in rows on a metal, and
never needed more rows than the amplitude rule for the same accuracy. It is **not** cost-neutral at equal nominal setting:
on FCC Ni tau 1e-3 keeps 1.6x the rows of amplitude 1e-3 at 3.1–3.6x the wall (for 8–15x lower error). Therefore: implement
as an opt-in rule first; the default switches only after the open decisions below.

## Open decisions (the user's)

1. **Default tau.** Data: tau 1e-3 under-delivers on NiO PM (every 4th block); tau 1e-4 is never worse than amplitude 1e-3 but
   costs 3x rows / 5.7x wall on FCC Ni. Recommendation: ship opt-in with no default change; decide after the library
   implementation's per-block cost is measured.
2. **Scoring cadence.** Every block keeps the error inside tau everywhere measured; every 4th block is ~2x cheaper in growth
   and exceeds tau by up to 1.6x. Recommendation: every 4th block, documented as such.
3. **Refinement.** With PT2 uncapped, refinement changed nothing on SMO u7 (same rows, same G). Recommendation: leave the
   refinement machinery untouched; it keeps acting on cap-frozen units only.

## Implementation series (each commit green on the targeted tests; full -n 1/2/3 gate once at the end)

1. **Option plumbing.** `gf_pt2_tol` (float, `None` = off) through `options.py`, the input schema, `selfenergy.py`,
   `greens_function._gf_solve_units`, `block_Green_discovered`. Mutually exclusive with an explicit `gf_prune_tol`
   (error names both). No behaviour yet. Tests: registry, schema, exclusivity.
2. **Scoring accumulator.** In `gf_recurrence`: a growth-only helper that owns the K-row `ShiftedGalerkin`, the alpha/beta
   needed for `||G_P||`, the mesh built from `eval_meshes` (the real-axis mesh with the smallest |Im z|, 31 points), and the
   mesh-point drop (share of frontier mass < 1e-3 on each of the first 10 scored blocks that carry mass). `ShiftedGalerkin`
   gets a public `keep(mask)` instead of the rig's private-buffer slicing. Tests: accumulated `rho` equals an explicit
   `H X` on a small model, before and after a drop (1e-10).
3. **Admission.** `_Admission._pruned` gains the PT2 branch: first shell unchanged; on unscored blocks admit nothing; on
   scored blocks one `collective_mass_cutoff`; at the cap `_top_and_freeze` ranked by score. All collectives unconditional
   and rank-invariant (empty ranks included). Tests: dense small-model oracle (tail left out <= tau, relative dG <= tau
   when scored every block), cap fill order, `-n 1/2/3` identical bases.
4. **Growth stop.** End growth at 2x the block of the last admission (at least 16 quiet blocks), decided from replicated
   state. Tests: stop leaves the basis identical to an unstopped run on a model that keeps running.
5. **Reporting.** One user-readable line per unit (rows from the first shell / from the rule, mesh points kept, block of
   the last admission, stop reason), and the option in the settings table.
6. **Replay parity.** `freeze_replay` / `real_workload` pass `gf_pt2_tol`; rerun SMO u7, NiO AFM, FCC Ni member N_imp=6
   against the prototype pickles: identical row counts expected, wall and peak RSS recorded.
7. **Docs.** Option docstring and the pruning section of the GF documentation; then the default-tau decision with the
   numbers from step 6.

## Known limits to state in the docs

- Small or low-weight units stop at the first shell once their estimated tail is inside tau (FCC Ni addition units: relative
  error 4e-5 at tau 1e-3); the amplitude rule is more exact there for a few hundred rows.
- SMO errors near 1e-4 are at the references' own accuracy; Im Sigma at -0.28 is not shown to improve.
- Unmeasured: complex Hamiltonians, multi-state chunks beyond width 2, cap-bound units other than SMO u7, Sigma on NiO and
  FCC Ni, NiO PM above ~50k rows, rank-count invariance on real archives, memory on NiO / FCC Ni.
