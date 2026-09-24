# CIPSI: converge the ground-state energy on the residual PT2, not on `de2_min`

Status: **implemented** on branch `cipsi-pt2-residual-convergence` (2026-09-24).

## The defect

`de2_min` was a threshold on **each candidate** determinant's Epstein-Nesbet importance
`|<Dj|H|psi>|^2 / |E_ref - E_Dj|`. `CIPSISolver.expand` stopped once a selection round admitted
nothing, which meant no *single* candidate cleared the threshold. Nothing bounded the **sum** of the
refused contributions, and that sum is what the energy error follows. A thousand candidates at 1e-8
each already make 1e-5, and the number of weak candidates grows with the bath.

The sum was computed in every round (`_admit_top`'s `subthreshold_de2_mass`) and printed, but it
never fed into the stopping decision.

## Measurements

**SIAM against exact diagonalization** (star geometry, 5 valence + 4 conduction levels,
hybridizations spread over 10^-1.5..10^-0.3; the fixture in
`test/restrictions/test_cipsi_pt2_convergence.py`; exact reference 22,356 determinants):

| `de2_min` only | basis | E − E_exact | terminating residual | ratio |
|---|---|---|---|---|
| 1e-4 | 180 | 7.60e-6 | 1.49e-5 | 0.51 |
| 1e-6 | 182 | 6.29e-6 | 1.23e-5 | 0.51 |
| 1e-7 | 212 | 2.51e-6 | 4.93e-6 | 0.51 |
| **1e-8** | 336 | **4.28e-7** | 8.47e-7 | 0.50 |
| 1e-9 | 526 | 3.61e-8 | 7.15e-8 | 0.51 |
| 1e-10 | 678 | 7.35e-9 | 1.46e-8 | 0.50 |

The ground state is a Kramers doublet, and the residual is summed over the manifold. So per state
the residual predicts the true error **to within 2% at every threshold**. At `de2_min = 1e-8` the
error is 43x the threshold.

New rule on the same model: `e_pt2_tol = 1e-8` gives 744 determinants and an error of 5.0e-9;
`e_pt2_tol = 1e-5` gives 187 determinants and an error of 4.7e-6.

**SrMnO3 cubic archive** (`~/Dokument/arrhenius/SMO/cubic/impmod/impurityModel_data.h5`), ground
state only, 4 ranks, excitation budget 8, production `slater_weight_min = 1.49e-8` and chain window:

| run | E0 | determinants | refine | GS phase |
|---|---|---|---|---|
| production, `de2_min = 1e-8` | −10.154666729 | 32,404 | 31 s | 912 s |
| `causality_check` "tight" (de2 1e-10, slater 0, chain off) | −10.15471587 | 95,678 | | |
| **`e_pt2_tol = 1e-8`** (production pruning + chain) | **−10.154716927** | 244,017 | 474 s | 1540 s |

- The production refine stopped with a terminating residual of **9.2e-5**. Per state (doublet)
  that predicts ~4.6e-5; the actual gap to the converged energy is **5.02e-5**.
- The 4.9e-5 drop in `causality_check/README.md` was attributed to three truncations at once.
  `de2_min` accounts for all of it. With the chain window and Slater pruning left at production
  values, the converged energy is still 1.1e-6 *below* the "tight" run.
- The cost is 7.5x the determinants and a refine 15x slower, 1.7x on the whole ground-state phase.
- The sector walk is **not** converged by this rule, deliberately. Its expansions stop with a
  residual of 1.3e-3..6.1e-2 (N_imp = 1..5, up to 144,555 determinants) against sector gaps of
  ~0.1 eV. Converging them would cost far more than the refinement the winning sector gets anyway.

## The design

- **Selection:** `collective_mass_cutoff(scores, e_pt2_tol, comm)` (`manybody_basis.py`) finds the
  largest cutoff whose tail `sum(score <= t)` fits `e_pt2_tol`. The same geometric bisection as
  `collective_amplitude_cutoff`, on allreduced partial sums, so every rank gets an identical
  cutoff. `determine_new_Dj` admits `score > t` (and `>= de2_min`, and `> 0`).
- **Stopping:** a round admits nothing exactly when the total residual is ≤ `e_pt2_tol`. The loop
  exit that already existed became the convergence test.
- **Scope:** scores are the max over reference manifolds of the manifold-summed de2, so the tail
  covers every reference state of the expansion, conservatively by up to the degeneracy. It is a
  second-order *estimate*, not a rigorous bound. States found only after `expand` (the
  `num_wanted` widening in `solve_ground_state`) were never references.
- **Report:** `CIPSISolver.convergence_report` = `{residual_pt2, residual_is_current, e_pt2_tol,
  converged, limited_by}`. It is surfaced as `gs_info["convergence"]`, in
  `ground_state_statistics.json`, and as `calc_selfenergy`'s `gs_convergence`. If unconverged, a
  rank-0 `WARNING` print (not `warnings.warn`, because `filterwarnings = error`).
- **Defaults:**
  - `GS_E_PT2_TOL = 1e-8`, and `GS_DE2_MIN = 0` (no floor).
  - The sector walk keeps `SECTOR_WALK_DE2_MIN = 1e-6`, with `SECTOR_WALK_E_PT2_TOL = None`.
  - DC sector solves: `DC_E_PT2_TOL` (unset = `GS_E_PT2_TOL`). `DC_DE2_MIN` and the RSPt line's
    `de2_min X` remain as a floor that loosens a solve.

## Review findings (adversarial review, same day)

- **Fixed:** after a binding cap the report could say `converged=True`. The last capped cycle's
  tail is within tolerance by construction, but `truncate` then drops determinants, so that
  residual described a basis that no longer existed. Measured: reported 1.0e-8 at cap 640,
  true residual of the returned basis 1.7e-8.
  - `residual_is_current` now tracks whether the last round left the basis untouched. A capped
    run is never reported converged.
  - Regression test at caps 640/700; it goes red when the fix is reverted.
- **Fixed:** `last_selection` was not reset per `expand` call, so an expansion that ran no round
  could read a previous call's residual.
- **Fixed:** stale "de2_min is the limit" wording in `dc_search`, `dc_record` and the
  `DC_CAP_STRATEGY` doc.
- **Open, by design:** the DC sector solves now converge to 1e-8 as well. On SMO the N−1 sector was
  already memory-bound, so expect more memory-bound sectors. `_reject_if_memory_bound` then refuses
  the answer unless `DC_ALLOW_MEMORY_BOUND` is set. `DC_E_PT2_TOL` (e.g. `1e-5`) is the lever for
  loosening it.

## Not in scope

- The cap-5k-vs-20k basin finding (`smo-cipsi-bigger-cap-worse-energy`) is a different failure mode.
- `select_at` / the GF selection is unchanged.
