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
