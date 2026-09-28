# The many-body basis and occupation restrictions

The variational basis — which Slater determinants are in play — is the other axis (besides the
solvers) where the code is subtle. Determinants are enumerated from occupation windows, grown
by selected-CI, and constrained by *restrictions* that keep the basis physical and finite. This
document explains the pieces and the failure modes they guard against.

## The `Basis` object

`manybody_basis.Basis` is the distributed set of Slater determinants plus its MPI bookkeeping:

- **Storage/lookup** — a rank-local sorted determinant list, a state→global-index dict, and
  hash-routed distributed lookups (`index`, `contains`, `__getitem__`).
- **Growth** — `expand(op, ...)` applies an operator and adds the newly reached determinants
  (the connectivity closure), the primitive under CIPSI and the GF excited-basis build.
- **Redistribution** — `redistribute_psis` (see [`mpi_model.md`](mpi_model.md)).
- **Lifecycle** — `clone`, `copy`, `clear`, and the collective `free_comm`.

Determinant storage lives directly in `Basis` (the old container indirection was dissolved).
The `Basis` is created around an **occupation window**: a nominal impurity occupation plus the
per-group and per-bath tolerances that bound how far the occupation may deviate.

## Enumeration and seeding

- `basis_generation.generate_initial_basis(...)` — pure enumeration of the initial determinant
  set from the occupation windows. No MPI.
- **Hartree–Fock seeding** — `groundstate.find_ground_state_basis(use_hf_seed=True)` picks the
  nominal occupation `N0` via a cheap memory-bounded UHF (`hartree_fock.py`) before enumeration.
  It chooses `N0`; it does **not** search — and it must never accept an occupation it could not
  converge (an unconverged `d10` seed leaks electrons and collapses the basis; see below).
- **CIPSI** — `cipsi_solver.CIPSISolver` grows the basis iteratively: score candidate
  determinants by their Epstein–Nesbet second-order energy contribution, add the top ones,
  repeat. `groundstate.calc_gs` drives HF-seed → CIPSI → solve → observables.

## Restrictions

A *restriction* bounds the occupation of a subset of orbitals: `{frozenset(orbitals):
(n_min, n_max)}`. Restrictions are what keep the excited/spectral basis finite and physical.
`basis_restrictions.py` builds them (it contains collectives — call from all ranks):

- `get_effective_restrictions(basis)` — the observed restrictions of the current basis.
- `build_initial_restrictions(...)` — the ground-state restrictions, including the
  connectivity/coupling-distance logic (`_impurity_coupling_distance`) that decides which bath
  sites can hybridize.
- `build_excited_restrictions(...)` — the widened windows for a spectral sector (electron
  addition/removal shifts the occupation by one).

Two orthogonal flavors layer on top, from `symmetries.py`:

- **Weighted restrictions** (`weighted_restriction`, `widen_weighted_restrictions`,
  `sz_weighted_restriction`) — bound a weighted sum of occupations, used to pin `Sz` or an
  eg/t2g ratio without pinning the total.
- **Frozen-shell generation pins** — the core shell (e.g. the 2p in an L-edge model) is
  pinned at generation time so it cannot drain into the valence. Which shell that is comes
  from the model's `core_l`, not from "the shell with no bath states": those coincide for
  every RSPt-produced workload, but a core shell that does carry bath states would otherwise
  be left unpinned.

### The rule that keeps covalency

Restrict the **whole impurity as one subset**, not per orbital-symmetry group. A per-group
occupation window pins the Sz / eg-t2g ratio and kills covalency (it forced a spurious triplet
in NiO). The exception is a genuinely frozen shell, which *does* stay pinned.

## Failure modes worth knowing (the "d10 collapse" family)

Several distinct bugs all present the same way — the basis collapses to a closed shell and the
spectrum goes to zero (`max|XAS| == 0`). They are worth recognizing:

1. **Unconverged HF seed** — `use_hf_seed=True` accepting a non-converged `3d10` occupation
   leaks two electrons and lands ~8.7 eV above the true `d8`, giving a one-determinant basis and
   `XAS == 0`. Workaround: `use_hf_seed=False`, or reject unconverged seeds.
2. **Total-only window draining the core** — without a frozen-shell generation pin, a
   total-only occupation window lets `2p → 3d` drain (a `2p4d10` configuration wins by 22.7 eV
   but is H-disconnected). Fix: pin the core with `generate_initial_basis(frozen_occupations)`.
3. **Per-group window blowing open** — summing per-group `total_slack` opened the window until
   `d2` won, while per-group nominal pinning removed covalency. Fix: a single max
   `total_slack` over the whole impurity plus multi-group generation widening.

The common diagnostic: **check `max|XAS| > 0` (or the occupation) before trusting any spectral
result.** A collapsed basis is silent otherwise.

## Truncation (fixed-budget)

`truncation_threshold` caps the *global* determinant count. The ground-state truncation is
fixed-budget (refine to a budget, then top-K amplitude truncate); the GF excited basis caps via
`_CappedBasisProxy`, after which the recurrence is exact Lanczos of the projected `PHP`. See
`doc/plans/truncation_reliability.md`.

### How the determinant cap is decided

`truncation_threshold` takes the same three spellings on the command line, in the TOML input
(`[many_body_basis]`) and on the RSPt solver line:

| you write | what the ground state gets | what each Green's-function unit gets | memory guard |
|---|---|---|---|
| a positive integer, e.g. `2000` or `2e6` | exactly that cap | exactly that cap | **warns only** -- your cap is never lowered |
| `auto` (the default) | sized from available memory for the ground state alone | one cap for every unit, sized at GF entry for the Green's-function path (the largest unit basis the smallest color can afford); units are grouped onto fewer colors -- more of them run in series -- until that is at least the ground-state cap, and the GF memory guard may hold a unit lower if measured memory runs short | may hold a ground-state expansion or a Green's-function unit lower when measured memory runs short |
| `unlimited` (or `inf`) | no cap | no cap | may hold a ground-state expansion lower when measured memory runs short (the Green's-function units are not guarded) |

A cap you set is **final**. If the run's measured memory reaches the budget
(`GS_MEMORY_BUDGET_SAFETY` of the rank's share of node RAM) you get one line,
`WARNING determinant cap: ...`, on stdout *and* stderr at any verbosity, and the job carries on
-- it may then be killed by the kernel. Lower the cap, use `auto`, or run fewer ranks per node.

An `auto` cap is sized the same way by every driver (`calc_selfenergy`, `calc_gs`, the
double-counting search, spectra, susceptibility), from the ground-state path alone
(`memory_estimate.suggest_gs_truncation_threshold`), so the double counting is found at the same
determinant budget the self-energy then solves with. The Green's-function units never inherit a
cap the ground-state guard lowered: they are sized from the resolved policy
(`gf_units.gf_cap`). Likewise the ground-state refinement starts from the resolved cap, not from
one the guard lowered during the occupation walk (which keeps several sectors alive at once).

The byte model behind `auto` is a starting point, not a guarantee -- it has under-predicted by
~50x at 256 ranks. The measured-RSS guard is what actually stops an `auto` or `unlimited` run
before an OOM kill.
