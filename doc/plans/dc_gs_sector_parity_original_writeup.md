# Fixed-occupation / fixed-peak DC must pin the observable of the ground state the selfenergy actually uses

Original root-cause writeup and change plan for the `fixed_occupation_dc` / `fixed_peak_dc` /
`find_ground_state_basis` sector-parity rework. Preserved from an untracked repo-root scratch file
(`fix_dc_calc.txt`) — this content was in git nowhere. See `.claude/plans/prancy-juggling-cerf.md` (or its
successor) for the tiered repair plan that followed the adversarial review of the first implementation
attempt; this document is the earlier, pre-review problem statement.

## Context

`fixed_occupation_dc` / `fixed_peak_dc` tune a uniform shift µ so that a target observable (impurity
occupation / spectral peak) is "fixed" — but they measure that observable on a ground state determined
completely differently from the one `calc_selfenergy` (and spectra) later use. In
`impmod_tests/NiO/impmod/15_BathStates_peeled_linked_chainGeometry_noneReorthonormalization_6_processors_`
the DC search reports "achieved N = 8.8411" (target n0 = 8.8484), yet the subsequent selfenergy ground
state has impurity occupation ≈ 7. Reproduced and root-caused (script: `scratchpad/repro_hf_seed.py`
rebuilds the model from `impurityModel_data.h5`):

Three independent inconsistencies between the two paths:

1. **Different ground-state sector determination.** The DC search builds one basis pinned at the input
   nominal occupation (`{0: 8}`, whole impurity as a single group), expands it once with the guess dc, and
   re-solves on that frozen basis for every trial µ. `calc_selfenergy` → `calc_gs` →
   `find_ground_state_basis` instead re-derives the sector at the found dc via the Hartree-Fock seed. In
   the NiO run HF returned `{0: 2, 1: 3}` (impurity 5) at the found dc — a different total-charge sector.
2. **The HF-seed path has no correction step.** `find_ground_state_basis(use_hf_seed=True)` does a single
   accurate solve at the seeded sector. The docstring assumes HF misses by ≤ ±1 ("absorbed by the
   mixed-valence window"); here it missed by ≥ 2 (MF pushed 3 electrons into conduction baths), and the
   accurate GS ended pinned at the hard edge of its occupation window (99.96% of the thermal weight in the
   edge config `[7, 124, 0]`) with no walk to correct it. The legacy `use_hf_seed=False` dN scan does walk
   (±1 while energy decreases).
3. **Different valence/conduction split ⇒ different total electron number.** The DC search uses
   `model.bath_states` (the hybridization-fit split: 122 valence / 12 conduction → total N = 130), while
   `calc_selfenergy` re-derives the split from the sign of the bath on-site energies in
   `_prepare_solver_basis` (126 / 8 → total N = 131 with the HF seed). The two searches don't even agree
   on the electron count of the "same" nominal sector.

Decisions taken with the user (at the time):

- Cheaper hybrid for the DC observable (re-run the cheap HF seed per µ; rebuild the basis only when the
  seeded sector changes) — not a full `find_ground_state_basis` per µ.
- Include the ±1 walk after the HF seed in `find_ground_state_basis` (changes GS determination for all
  selfenergy/spectra runs — that is the point; the NiO selfenergy GS is itself wrong).
- Both routines (`fixed_occupation_dc` and `fixed_peak_dc`) in this change.

## Changes (as originally proposed)

### 1. `groundstate.py` — walk after the HF seed (fixes the selfenergy GS)

In `find_ground_state_basis`, after the HF-seeded `get_energy(winning_N0)`:

- probe each group ±1 (`get_energy` on `N0[i] ± 1`, bounds-checked — `get_energy` already returns inf out
  of bounds and caches), and for any direction that lowers the energy, keep walking that direction while
  the energy strictly decreases — same loop structure as the legacy scan's walk; factor the walk into a
  small local helper shared by both paths rather than duplicating it.
- Rank-invariant decisions: in `get_energy`, broadcast `e_trial` from rank 0 before caching/comparing
  (`comm.bcast`), mirroring the guard in `double_counting._lowest_energy_and_thermal_rho`. The walk (and
  the existing scan) branch on these energies; roundoff-divergent ranks would deadlock the next collective
  solve.
- Keep the verbose per-trial prints (like the scan's) so runs remain auditable.

### 2. Move `_prepare_solver_basis` to a shared module

`double_counting` must derive the same solver basis (symmetry rotation, eg/t2g grouping, valence/
conduction split, per-group nominal occupation) as `calc_selfenergy`, but `selfenergy` sits above
`double_counting` (it imports it) — so move `_SolverBasis` + `_prepare_solver_basis` (and its two module
constants `_ROTATION_TRIM_TOL`, `_MAX_ROTATION_FILL`) verbatim into a new module
`src/impurityModel/ed/solver_basis.py` (imports only `symmetries` + numpy — layering-clean). Keep
`from impurityModel.ed.solver_basis import _prepare_solver_basis, _SolverBasis  # noqa: F401` re-exports in
`selfenergy.py`: `susceptibility.py` and `test/support/restriction_diagnostics.py` import it from there.

### 3. `double_counting.py` — hybrid sector-tracking DC searches

Shared plumbing for both searches:

- Replace `_normalize_dc_orbitals` + `model.bath_states` with the derived layout: call
  `_prepare_solver_basis(model.h0, model.dc, model.u4, model.impurity_orbitals, N0, mixed_valence,
  model.rot_to_spherical, verbosity)` once. The µ shift is a uniform impurity shift (commutes with the
  impurity block), so the rotation/grouping/split derived at the guess dc are valid for every µ; per-µ
  Hamiltonian is `sb.h - mu * N_imp_op` (the impurity identity is rotation-invariant). `model.bath_states`
  is no longer used by these two functions (`_require_bath_states` calls go away; the model no longer
  needs the fit split here — note this in the docstrings as a behavior change).
- HF seeding per µ: `hartree_fock_seed_occupation(h(mu), sb.impurity_orbitals, sb.bath_states,
  sb.nominal_occ)` (dense, deterministic, replicated — no collective). If HF does not converge, keep the
  previous sector (and warn once on rank 0).
- Sector-keyed cache: a small helper that owns `{sector_key: (Basis, CIPSISolver)}`; on a new sector it
  builds the basis the same way `calc_energy` does (delta occs 0, mixed_valence, weighted restrictions,
  chain_restrict, `build_excited_restrictions` after `truncate_initial`) and expands with the current
  `h(mu)` (not the guess dc, unlike today). Extract the basis+solver construction at the top of
  `groundstate.calc_energy` into a reusable helper in `groundstate.py` so DC and `calc_energy` share it
  verbatim (`double_counting` importing `groundstate` introduces no cycle). Evict non-current sectors'
  bases (keep 1-2) to bound memory; halve the memory safety fraction as `fixed_peak_dc` already does when
  more than one basis is held.

`fixed_occupation_dc`:

- `occupation_observable(mu)`: HF-seed the sector, get the (possibly rebuilt) basis/solver, solve, thermal
  rho → n. Solve with `num_wanted=10` + the existing `energy_cut` to match `calc_gs` (the NiO GS is 3-fold
  quasi-degenerate — Boltzmann weights 0.337/0.333/0.330 — a single-state rho misrepresents the thermal
  occupation).
- Edge-pinning guard (the hybrid's stand-in for the walk): after the solve, if n sits within a small ε of
  the basis's reachable impurity-occupation window edge (bounds from `get_effective_restrictions`),
  re-center the sector one step toward that edge, rebuild, re-solve; repeat while pinned (bounded by the
  shell size). Decisions from Allreduced/broadcast data only.
- The DFT reference n0 target and `_noninteracting_impurity_occupation` are unchanged (raw-h0 contract,
  per the DC-rework campaign).

`fixed_peak_dc`:

- Per µ: HF-seed the central sector N(µ); upper/lower sector bases at N(µ)±1 (through the same sector-keyed
  cache — sector changes rebuild both). The single-impurity-group restriction stays, but on the derived
  grouping the impurity has ≥ 2 groups — so for the ±1 sectors keep the whole-impurity ±1 semantics: seed
  per-group N(µ) from HF, and build the ±1 sector by adding/removing the electron in the group HF ranks as
  lowest-addition/highest-removal (the mean-field aufbau order); document this. Energies compared/branched
  on are already broadcast (`_lowest_energy_and_thermal_rho`).
- Drop the now-stale "expand once with the guess dc" step and comment in both searches; update both
  docstrings (the module docstring's "found on a different variational space than the solve that will use
  it" caveat becomes the description of the new parity).

### 4. Reporting

At the end of both searches (rank 0, always — not verbosity-gated for the warning case): print the final
sector, achieved observable, and target; warn loudly if `|achieved - target| > tol` (plateau path already
does this — extend to the sector-jump case).

## Files (as originally proposed)

- `src/impurityModel/ed/groundstate.py` — HF-seed walk, rank-invariant `get_energy`, extract basis+solver
  construction helper from `calc_energy`.
- `src/impurityModel/ed/solver_basis.py` — new; `_SolverBasis` + `_prepare_solver_basis` moved verbatim
  from `selfenergy.py`.
- `src/impurityModel/ed/selfenergy.py` — re-export shim.
- `src/impurityModel/ed/double_counting.py` — both searches reworked as above.
- `src/impurityModel/test/gf/test_fixed_dc.py` — existing 21 tests must pass unchanged (trivial sectors;
  HF seeds them correctly); add new tests below.

## Tests (as originally proposed)

1. Walk rescues a bad seed (`test/gf/` or `test/basis/`): monkeypatch
   `groundstate.hartree_fock_seed_occupation` to return a sector off by 2 on a small model; assert
   `find_ground_state_basis` lands on the same sector/energy as `use_hf_seed=False`.
2. DC ⇄ GS consistency (extend `test_fixed_dc.py`): analytic 2+2 model with a dc guess far enough off that
   the sector changes across the µ scan; after `fixed_occupation_dc`, run `find_ground_state_basis` +
   thermal rho at the found dc and assert the occupation matches the achieved value within `occ_tol`. Same
   for `fixed_peak_dc`: peak position of the sector-searched GS at the found dc equals the request.
3. Gate: `python -m pytest` and `mpiexec -n 2 python -m pytest --with-mpi` (with `OPENBLAS_NUM_THREADS=1`);
   also `-n 3` once — the walk and the sector cache add energy-gated branching around collective solves
   (the exact deadlock class from the project memory).

## End-to-end verification (real workload), as originally proposed

Extend `scratchpad/repro_hf_seed.py` into a driver that loads the archived NiO 15-bath model, runs
`fixed_occupation_dc` (self-consistent target, n0 = 8.8484) under `mpiexec -n 2`, then runs the
selfenergy-style GS at the found dc, and checks `|Tr rho_imp − 8.8484| ≤ occ_tol`. Use a reduced
`truncation_threshold` if runtime is excessive. Success criterion: the achieved occupation of the
selfenergy ground state (not the DC search's own basis) hits the target, and the GS is not
window-edge-pinned.

## Outcome note (added when this file was archived out of the repo root)

The first implementation attempt at this plan (the one that produced `solver_basis.py` and the rest of the
diff referenced above) was itself found to be substantially broken by adversarial review — see
`.claude/plans/prancy-juggling-cerf.md` for the detailed tiered repair that followed, including the
discovery that the edge-pinning guard proposed in item 3 above is physically inconsistent (it silently
changes the total electron count of the reference Slater determinant) and was removed rather than
restored; `fixed_occupation_dc`'s per-µ HF re-seeding was constrained to same-total-N sectors only, to keep
the function's own conservation contract intact, rather than adopting `find_ground_state_basis`'s
grand-canonical (varying-total-N) sector search. That distinction — and whether it should eventually be
unified — is recorded as an open question in the repair plan's progress log.
