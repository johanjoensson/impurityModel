# Known issues

Defects that are understood but deliberately not fixed here, each with what it breaks, why it
has not bitten yet, and what fixing it would involve. A defect with a known shape belongs in
writing rather than in someone's memory.

## `ImpurityModel.impurity_indices` counts blocks, not orbitals

`impurity_indices` (`ed/model.py`) flattens `impurity_orbitals` exactly one level:

```python
return sorted(orb for orbs in self.impurity_orbitals.values() for orb in orbs)
```

That is right for the models whose `impurity_orbitals` is `{group: [index, ...]}` —
`from_solver_matrix`, `from_hdf5`, `from_blocks`, which all store `{0: [0, 1, ..., n-1]}`.

It is wrong for `from_shells`, which stores a *nested* per-shell block list,
`{1: [[0..5]], 2: [[6..15]]}`. One level of flattening then yields the **blocks**, not the
orbitals, so a 2p + 3d model with sixteen impurity spin-orbitals reports:

```python
>>> len(model.impurity_indices)
2
```

**What it would break.** Every consumer treats the result as a list of orbital indices:

| consumer | use |
| --- | --- |
| `dc_reference._noninteracting_impurity_rho` | `np.ix_(impurity_indices, impurity_indices)` — slices the one-body matrix |
| `dc_reference` saturation warning | `len(model.impurity_indices)` as the number of spin-orbitals |
| `dc_static._model_u4_dense` | `n_imp = len(model.impurity_indices)` for the Coulomb tensor |
| `dc_criteria` | inherits the same `n_imp` |

So a fully-localized-limit double counting evaluated on a multi-shell model would build a
2x2 matrix for a 16-orbital impurity and slice the wrong rows — silently, with no exception.

**Why it has not bitten.** Those consumers are all in the double-counting layer, and the only
constructor producing the nested shape is `from_shells`, which serves the spectroscopy path —
where `run_spectra` never reads `model.dc` at all. The TOML reader's applicability matrix now
makes the combination unreachable by construction: `[double_counting]` on a `[spectroscopy]`
run accepts only `mlft` (folded into `h0`, never a matrix) and `none`. The bug is latent,
not active.

**Why it is not fixed here.** The one-line fix — flatten recursively — changes the value
`impurity_indices` returns for every `from_shells` model, and therefore the behaviour of four
double-counting entry points, on a path none of them is currently exercised from. That is a
change to the DC layer wearing the clothes of a typo fix, and it wants its own branch, its own
before/after on a real workload, and a decision about whether `impurity_orbitals` should be
uniformly shaped at the source instead (the better fix, and a larger one: `from_shells`'
nested blocks carry the per-shell grouping the solver basis relies on).

**If you fix it**, the test to write first is the property that a model's
`impurity_indices` and its `n_spin_orbitals` agree with what the constructor was asked for,
across *all* the constructors — not a golden list for one of them. That is the shape of test
that would have caught the sibling bug in the bath layout, where the conduction block was
indexed over the valence one and stayed invisible for years because every shipped workload but
one had an empty conduction list (fixed; see `test/spectra/test_bath_layout.py`).

## Σ causality violations at sharp hybridization resonances are a discretization artifact

A production self-energy run can hit `sigma.check_greens_function`'s causality check
(`Im Sigma_ii(omega) > 0`) at a real-frequency point sitting on a sharp hybridization
resonance. `config.SIGMA_CAUSALITY_TOL` (default `1e-3`, relative, per diagonal orbital) warns
and continues below that threshold instead of raising — sized against a real production run
(see below) that violated at `7.6e-3` — but the mechanism itself is a genuine finite-resolution
artifact of the excited-state basis and real-axis broadening, not a solver defect.

**Evidence.** Reproduced serially (no MPI) from a real archived Mn production run
(`impurityModel_data.h5`, loaded via `test/support/real_workload.load_workload` +
`calc_selfenergy(..., comm=None)` on a narrow `omega` slice around the violation) — the
serial rerun reproduces the original 128-rank run's violation to four significant figures
(`Im Sigma = 5.146e-3` at `omega ~ -0.266`, vs. the archived `5.144e-3`), exactly where
`Im Delta` (the hybridization function) peaks. Four bounded experiments on that reproduction,
each isolating one variable:

| experiment | change | worst `Im Sigma` (orbital 0) | verdict |
| --- | --- | --- | --- |
| baseline | production settings | `+5.146e-3` | reference |
| tighten GF Lanczos tolerance | `_GF_REL_TOL_FLOOR` 1e-9 -> 1e-12 (1000x) | `+5.146e-3` (7th digit) | **rules out** Lanczos non-convergence |
| `u4 = None` (non-interacting) | switches off the interaction entirely | `+1.2e-12` (roundoff) | **rules out** a Sigma-construction bug (index alignment, `delta` convention, DC handling) — those do not depend on `U` and would still show up |
| `excitation_budget` 4 -> 3 (tighter) | shrinks the excited-sector many-body space | `+3.094e-1` (60x worse; the whole window turns unphysical) | **implicates** excited-sector basis resolution |
| `delta` (broadening) 0.010 -> 0.020 (2x) | smooths the real-axis resolvent | `-2.04e-3` (causal) | **implicates** finite broadening of a coarse pole structure |

Loosening `excitation_budget` past the production default of 4 (to 6, 8, or disabled
entirely, via `BasisOptions.excitation_budget`) was also attempted, to see the violation shrink
from the other direction; each attempt was killed as measured-infeasible on a single core
(disabled: >1h with the ground state still not converged; budget 6: >25 min past a converged
ground state with the Green's function still unconverged) rather than run to a result — the
`excitation_budget=3` and `delta=0.020` points above already establish the direction and
magnitude without it.

**Conclusion.** The excited-state many-body basis (bounded by `excitation_budget` and
`slaterWeightMin`) is a discrete approximation to the continuous bath spectral function; near
a sharp `Im Delta` resonance, evaluated at a broadening `delta` too small to smooth over that
discreteness, the resulting `G` can be locally narrower than the `G0` built from the full
one-body `h0` — producing `Im Sigma > 0` in a thin window around the resonance. Both directions
move it exactly as that picture predicts: shrinking the excited-sector basis makes it sharply
worse, widening `delta` removes it. This is the standard finite-discretization artifact of an
ED-based impurity solver, cured by more excited-sector budget or more broadening — both cost
knobs the user already owns, not a code path to patch.

**Why it is not fixed here.** There is no code defect to fix: `Sigma = G0^-1 - G^-1` is exact
only in the limit of an unrestricted excited-sector basis and infinitesimal broadening, and
production runs use neither for tractability. `SIGMA_CAUSALITY_TOL` accepts that trade
explicitly instead of aborting a multi-hour/multi-rank job over an artifact whose only cure is
"spend more compute" or "smooth more."

**If you need it gone for a specific run**, raise `delta` (broadens every other spectral
feature too) or `excitation_budget` (cost grows combinatorially — budget 6 did not converge
within 25 minutes serially on this workload, so plan for the full production MPI allocation,
not a quick local check).
