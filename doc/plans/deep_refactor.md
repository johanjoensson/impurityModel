# Deep refactor + documentation overhaul

**Status (2026-07-15):** in progress. Phases 0, 1, 2 (a–e), 3 complete; config reference doc
written. Remaining: Phase 4 Cython `.pxi` splits; Phase 5 developer/user docs + Sphinx polish.

## Motivation

The codebase grew through successive feature/perf campaigns (the symmetry program, the RIXS
R1 solver chain, the Block-Lanczos reorthogonalization campaigns). Earlier refactor waves
dissolved `finite.py`, decomposed `Basis` into the `basis_*` modules, and split
`greens_function.py` into `gf_primitives`/`gf_convergence`/`gf_shift_recycling`. What remains:

- Four modules still bundle several concerns: `greens_function.py` (2281 lines),
  `spectra.py` (1930), `symmetries.py` (1702), `selfenergy.py` (1446).
- The Cython kernels are 1600–2200-line single files (`BlockLanczos.pyx` 2162,
  `ManyBodyUtils.pyx` 2096, `BlockLanczosArray.pyx` 1629).
- ~21 `GF_*` environment variables act as scattered, undocumented configuration, with
  defaults duplicated across modules (`memory_estimate.py` re-hardcodes `GF_GMRES_RESTART`
  and `GF_SLICES` defaults — a single-source-of-truth violation).
- Naming mixes camelCase and `_new` suffixes (`getSpectra_new`, `getRIXSmap_tensor`) with
  the snake_case used everywhere else.
- Documentation lags: no user guide, no configuration reference, a 7-line `intro.rst`,
  a `doc/plans/` index with stale statuses, and an architecture overview that documents
  three modules that no longer exist.

Goal: one obvious home per concern, a single documented configuration surface, consistent
naming, readable Cython kernels, and documentation for developers *and* users.

## Scope

- **Internal APIs may change** — names, signatures, module boundaries.
- **Stable surface:** CLI entry points and flags (`get_spectra`, `selfenergy`,
  `plot_spectra`, `plot_RIXS`), input file formats, and the `spectra.h5` layout.
- The Cython layer is in scope (restructure the `.pyx` files, not just their docstrings).
- Documented-failure code paths (`gf_method="sliced"`, the per-polarization RIXS path)
  are **kept** — they are deliberately retained alongside their verdict docs.

Every commit stays green on the full gate: `python -m pytest` and
`mpiexec -n 2 python -m pytest --with-mpi` (plus `-n 3` when touching `basis_split.py` or
`run_units_distributed`). Splits move code verbatim; renames are mechanical and separate
from moves.

---

## Phase 0 — Probe & inventory ✅

### 0a. Stale documentation found

`doc/architecture_overview.md` documents a "Bath construction (used by the `build_h0`
script)" section naming three modules that **do not exist** in the tree:

| Documented | Reality |
| --- | --- |
| `ed/edchain.py` | does not exist |
| `ed/natural_orbitals.py` | does not exist |
| `scripts/build_h0.py` (console script `build_h0`) | does not exist; no `build_h0` entry in `[project.scripts]` |

`CLAUDE.md` likewise advertises a `.[rspt]` extra "only for the `build_h0` script" — there
is no `rspt` extra in `pyproject.toml`. Both were corrected in Phase 0.

The only surviving module of that group is `ed/bath_fitting.py`, which has no production
importer (see below).

### 0b. Orphan audit

Importer counts (greps over `.py`, `.pyx`, `.pxd` in `ed/`, `scripts/`, `cython/`;
`ea16.py` taught us that `.pyx` files import Python modules at runtime, so they are
included):

| Module | Production importers | Test importers | Verdict |
| --- | --- | --- | --- |
| `ed/double_chain_haverkort/double_chains.py` (825 lines) | 0 | 0 | **dead** — no importer anywhere, no `main`, not a script |
| `ed/givens_qr.py` | 0 | 1 | test-only |
| `ed/density_matrix.py` | 0 | 1 | test-only |
| `ed/bath_fitting.py` | 0 | 1 | test-only (orphaned with `build_h0`) |
| `ed/rational_sampling.py` | 1 (`spectra.py`) | 1 | live |
| `ed/chebyshev_filter.py` | 2 | 1 | live (`gf_method="sliced"`) |

**Resolved (2026-07-15, with user sign-off):** all of these were deleted — the dead
`double_chain_haverkort/double_chains.py` (825 lines, zero importers) and the three
test-only modules `density_matrix.py`, `givens_qr.py`, `bath_fitting.py` together with
their tests (`test_density_matrix.py`, `test_givens_qr.py`, `test_bath_fitting.py`). None
had a production importer; all remain recoverable from git history.

Untracked build droppings present in the working tree (`src/cython/scratch.*.so`,
`src/impurityModel/ed/ManyBodyUtils.so`) are gitignored and left alone.

### 0c. Execution-path map

The dispatch surfaces a reader has to hold in their head. These anchors seed
`doc/gf_solver_architecture.md` (Phase 5).

**Green's-function resolvent kernel** — `gf_method`, dispatched in
`greens_function.get_Greens_function`:

| `gf_method` | Driver | Kernel | Notes |
| --- | --- | --- | --- |
| `"lanczos"` (default) | `get_Greens_function` | `_block_green_group` → `block_green_impl` / `block_Green_sparse` | one recurrence per unit, serves the whole mesh |
| `"bicgstab"` | `_get_greens_function_bicgstab` | `block_Green_bicgstab` | one linear solve per frequency, basis rebuilt-and-discarded |
| `"sliced"` | `_get_greens_function_sliced` | Chebyshev spectral windows | needs a real-axis mesh, else falls back to bicgstab; **documented failure**, retained |

Orthogonal switches on the Lanczos path: `sparse` (`ManyBodyState` kernel vs CSR array
kernel), the pairwise operator split (`GF_OPERATOR_SPLIT`), the reort mode
(NONE/PARTIAL/FULL/PERIODIC/SELECTIVE), and `truncation_threshold` capping
(`_CappedBasisProxy`).

**Spectra** — `spectra.simulate_spectra` chooses per spectrum:
PS/XPS/NIXS always go through `calc_spectra` (per-operator spectra); XAS and RIXS take
the *projected* path (`calc_spectra` / `calc_map`, per-polarization) when a
projector file is given, and otherwise the *tensor* path (`calc_spectra_tensor` /
`calc_tensor_map`), which stores a Cartesian tensor that `polarization.py` contracts at
plot time.

**RIXS solver tiers** — `spectra._R1SolverChain.solve`, in order:
`SectorResolventCache` (dense spectral cache on a closed sector) → `KrylovShiftedResolvent`
(one block-Lanczos recurrence serving every shift) → `block_bicgstab` (with restarts) →
`block_gmres` (escalation). Plus the adaptive incoming-energy sampler
(`_rixs_map_adaptive`, set-valued AAA).

**Distribution engine** — every GF/spectra/RIXS driver funnels through
`enumerate_gf_units` → `unit_cost_weights` → `run_units_distributed`
(→ `basis_split.split_basis_and_redistribute_psi`).

---

## Phase 1 — Central configuration module ✅

- [x] New `ed/config.py` (Layer 0): one declaration per knob — env-var name, type, default,
      docstring — with typed accessors and a `dump()` rendering the full table.
- [x] Migrate every `os.environ.get` in `greens_function.py`, `gf_convergence.py`,
      `gf_shift_recycling.py`, `spectra.py`, `memory_estimate.py` to it; this deletes the
      duplicated defaults in `memory_estimate.py`.
- [x] Env-var names and defaults stay identical, so existing run scripts keep working.
- [x] `test_config.py` covers defaults, lazy override, parsers/clamps, derived knobs.

Also fixed a pre-existing MPI-gate failure discovered here: `test_simulate_spectra`
asserted rank-0-only h5 writes on every rank (the gate was red on rank 1 at HEAD).

## Phase 2 — Python module splits (verbatim moves)

- [x] **`spectra.py`** → `ed/transition_operators.py` (dipole/NIXS/PES/IPS operator
      builders, pure physics) + `ed/rixs.py` (the whole RIXS half: adaptive sampler,
      `_R1SolverChain`, `_rixs_map_flat`, both map drivers) + `spectra.py`
      (XAS/PS/XPS/NIXS drivers + the `simulate_spectra` orchestrator). **Done:** 1930 → ~860
      lines; both new modules re-exported from `spectra.py` for backward compat.
- [x] **`selfenergy.py`** → `ed/double_counting.py` (`fixed_peak_dc`,
      `fixed_occupation_dc`, DC solver plumbing) + `ed/sigma.py` (`get_hcorr_v_hbath`,
      `hyb`, `get_sigma`, `get_Sigma_static`, causality check) + `selfenergy.py`
      (`calc_selfenergy` + CLI). **Done** (re-exported for backward compat). The
      `calc_selfenergy` body was further decomposed into named stage helpers
      (`_prepare_solver_basis` returning a `_SolverBasis`, `_check_gf_physical`,
      `_self_energy_on_mesh`), shrinking its logic by ~108 lines.
- [x] **`symmetries.py`** → `ed/lie_algebra.py` (the algebraic half: tensor extraction/
      rotation, symmetry discovery, structure constants, Cartan subalgebra, joint
      diagonalization, Casimirs) + `symmetries.py` (conserved charges, restrictions, block
      structure). **Done:** 1702 → 944 (symmetries) + 801 (lie_algebra); the algebraic
      primitives are re-exported from `symmetries.py` for backward compat.
- [x] **`greens_function.py`** → `ed/gf_units.py` (the distribution engine: `GFUnit`,
      `enumerate_gf_units`, `unit_cost_weights`, `run_units_distributed`, plus the
      `_gf_operator_split`/`_union_restrictions`/`_apply_transition_ops` helpers) +
      `ed/gf_solvers.py` (the per-unit kernels: `block_Green*`, `block_green_impl`,
      `solve_shifted_block`, the BiCGSTAB warm-start helpers) + `greens_function.py`
      (top-level drivers and assembly). **Done:** 2258 → 1223 lines; gf_units and
      gf_solvers are independent lower layers (neither imports the other, nor
      `greens_function`), verified by an AST call-graph. No explicit re-export shim was
      added for the two new modules — direct `from greens_function import <moved-name>`
      importers were repointed to the owning module; the `gf.X` facade accesses in
      `spectra.py`/`rixs.py` keep working because `greens_function` genuinely imports the
      names its own drivers use. Dead imports pruned from `greens_function.py`; the
      pre-existing `gf_primitives`/`gf_convergence`/`gf_shift_recycling` re-export block is
      kept.
- [x] **CLIs**: group `get_spectra.main`'s 30 positional parameters (and
      `selfenergy.get_selfenergy`'s 20) into dataclasses built by the argparse layer
      (`HamiltonianParameters`/`OccupationParameters`/`SpectrumParameters`/`SolverParameters`).
      **Done:** CLI flags unchanged; the function bodies are verbatim (the dataclasses are
      unpacked into the existing local names at the top). Verified `get_spectra.main` (no test
      coverage) with a mocked end-to-end run; `get_selfenergy` is covered by `test_selfenergy`.

## Phase 3 — Naming & signature cleanup

- [x] `getSpectra_new` → `calc_spectra`, `getSpectra_tensor` → `calc_spectra_tensor`,
      `getRIXSmap_new` → `rixs.calc_map`, `getRIXSmap_tensor` → `rixs.calc_tensor_map`.
- [x] Operator builders lose the `get` prefix and move to snake_case in
      `transition_operators.py`: `getDipoleOperator(s)` → `dipole_operator(s)`,
      `getDaggeredDipoleOperators` → `daggered_dipole_operators`, `getNIXSOperator(s)` →
      `nixs_operator(s)`, `get{Inverse,}PhotoEmissionOperators` →
      `{inverse_,}photoemission_operators`. Re-exported from `spectra.py` unchanged.
- [x] Physics-domain argument names (`nBaths`, `Fdd`, …) stay — they mirror the literature
      and the CLI surface.

## Phase 4 — Cython layer

Split via `.pxi` textual includes: same compiled extension modules, no build-system or
import changes, purely file-level readability.

- [ ] `BlockLanczos.pyx` → `_lanczos_step.pxi` + `_trlm.pxi` + `_irlm.pxi`.
- [ ] `ManyBodyUtils.pyx` → `_slater_state.pxi` + `_operator.pxi` + `_mpi_pack.pxi` +
      `_krylov_store.pxi` + `_block_state.pxi`.
- [ ] `BlockLanczosArray.pyx` → `_reort.pxi` + the array kernel.
- [ ] Kernel documentation pass: the invariants (reort-estimator honesty, deflation's two
      scales, seed ownership in the recurrences) documented at the code that owns them.
- [ ] `setup.py`: `depends=` on the `.pxi` files so edits trigger recythonization.

## Phase 5 — Documentation ✅ (developer + user docs done)

- [x] `doc/architecture_overview.md` — kept current throughout; reflects config/
      transition_operators/rixs/double_counting/sigma.
- [x] `doc/gf_solver_architecture.md` — the distribution engine, `gf_method` kernels, the
      RIXS four-tier solver chain, spectra dispatch, and a reading guide.
- [x] `doc/basis_and_restrictions.md` — `Basis` lifecycle, hash distribution, restriction
      flavors, CIPSI/HF seeding, the d10-collapse case studies.
- [x] `doc/mpi_model.md` — the distribution model and the MPI ground rules.
- [x] `doc/configuration.md` — every knob from `ed/config.py` (generated from `dump()`).
- [x] `doc/user_guide.md` — install, input formats, running the CLIs, `spectra.h5` layout,
      plotting, a worked NiO example.
- [x] Sphinx: rewrote `intro.rst`, sectioned `index.rst` toctree with all new pages,
      warning-free build on the new pages.
- [x] `doc/plans/README.md` — status block updated; slicing marked a documented failure,
      rixs_r2 marked a proposal.

## Remaining (follow-up sessions)

- Phase 4: Cython `.pxi` splits of the three large kernels + a kernel documentation pass.
