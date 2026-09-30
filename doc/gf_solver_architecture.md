# Green's-function & spectra solver architecture

The interacting Green's function is the computational heart of the package, and it is where
the execution paths are hardest to follow: two resolvent kernels, two Lanczos-kernel backends,
a work-unit decomposition, a four-tier RIXS solver chain, and a shared MPI distribution engine
underneath all of it. This document is the map. It names the entry points and traces the
dispatch so you can find the code that actually runs for a given call.

For the physics (why a block-tridiagonal continued fraction gives `G`), see
[`greens_function_theory.md`](greens_function_theory.md). For the module layering, see
[`architecture_overview.md`](architecture_overview.md). The 2026-09 review that shaped the
current structure, including every defect it found and fixed, is
[`reviews/gf_review.md`](reviews/gf_review.md).

## The one pipeline

Every Green's-function, spectra and RIXS driver runs the same pipeline:

```
enumerate_gf_units(...)            gf_units    flatten the work into independent units
        |                                      unit = (operator group x eigenstate chunk);
        v                                      an operator group is a block of transition
unit_cost_weights(unit_seeds)      gf_units    operators on one spectral side
        |                                      (estimated cost, for load balancing)
        v
run_units_distributed(...)         gf_units    split basis.comm into colors (basis_split),
        |                                      redistribute seeds, run kernel(split_basis, u,
        v                                      seeds) per color, gather to global rank 0
lanczos_unit_kernel(...)           gf_engine   _block_green_group: one block-Lanczos
        |                                      recurrence per unit; r split per stacked state
        v
states_by_group(...)               gf_engine   per operator group, per eigenstate:
        |                                      (alphas, betas, r)
        v
calc_thermally_averaged_G + ThermalEnsemble + combine_sides (G+ - G-^T)/Z
```

The modules split the work cleanly. `gf_units` owns distribution: units, weights, the
determinant caps and memory guard, and the split. `gf_engine` owns what a unit computes and
how results are put back together. `average.ThermalEnsemble` holds the Boltzmann weights and
`Z`.

A **unit** is the atom of parallel work. `run_units_distributed` splits the basis's
communicator into colors sized to fit the memory budget (`basis_split.py`), gives each color
its own sub-communicator and its own clone of the basis, redistributes the seed states onto
the rebuilt basis, and calls the caller's `kernel`. The distribution is identical for a
self-energy run, an XAS spectrum and a RIXS map; only the kernel and the assembly differ. RIXS
enumerates its own units, (eigenstate x chunk of incoming energies), and uses its own kernel
(the R1 chain below), but runs through the same `run_units_distributed`.

> **Why this matters:** determinants are hash-distributed (`hash(sd) % size`, one owner per
> determinant), so no rank ever holds a full state vector. The engine is where that invariant
> is maintained across the split/redistribute boundary. See [`mpi_model.md`](mpi_model.md).

**Occupation windows.** Each unit runs under an excited-sector occupation window, built by
`basis_restrictions.build_excited_restrictions` and combined with
`union_windows`/`intersect_windows`. `None` means unrestricted everywhere, and
`Basis.clone(restrictions=...)` inherits the parent's window only on its explicit `INHERIT`
default. Every consumer states the mask it needs on the shared Hamiltonian (`None` clears), and
the moments clear it: masks are sticky on the operator object and cannot be read back.

## Self-energy estimators (`sigma_estimators.py`)

`calc_selfenergy` asks a `SelfEnergyEstimator` (picked by `SolverOptions.sigma_method`) three
things:

- **`operator_families(block, solver_basis)`** returns `(X^dag, X)`, the operators the
  pipeline resolves for one block. `get_Greens_function(operator_families=...)` builds one
  operator group per family side, so each block's `G` is the Green's function of `X`, as wide
  as the family. By contract the family's leading `len(block)` operators are the block's own
  `c`, and the anticommutator sum rule is checked on those columns.
- **`impurity_gf(g_family, block)`** returns the impurity `G` inside the family's `G`: the
  leading block.
- **`sigma(mesh, g_families, ...)`** returns the self-energy read off the family's `G`.

`DysonEstimator` is the production estimator: `X = c` and `Σ = G0^-1 - G^-1` (`sigma.get_sigma`).
The symmetric improved estimator plugs in here. It resolves `X = [c, q]` with
`q = [c, H_int]`, a `2n`-wide family, taking `H_int` from `SolverBasis.h_int` (`h - h0_solve`,
built lazily, so the Dyson path never pays for a copy of the Coulomb operator). The
test-only stub in `test/gf/test_gf_operator_families.py` checks two things:
- the stub family's full `2n x 2n` `G` against the exact Lehmann oracle, on every kernel;
- its seed-Gram cross block `<{q_a, c_b^dag}>` against production's `sigma_static`.

## Resolvent kernel: the `gf_method` switch

`get_Greens_function(...)` in `greens_function.py` picks the resolvent kernel with the
`gf_method` argument:

| `gf_method` | Driver | Per-unit kernel | When |
| --- | --- | --- | --- |
| `"lanczos"` *(default)* | `get_Greens_function` | `_block_green_group` → `block_green_impl` (array) / `block_Green_sparse` (state) | One block-Lanczos recurrence per unit builds a continued fraction serving the **whole frequency mesh** at once. The workhorse. |
| `"bicgstab"` | `_get_greens_function_bicgstab` | `block_Green_bicgstab` | One linear solve **per frequency point**, basis rebuilt-and-discarded each point. Wins on memory (the live basis never exceeds one point's support) at a time cost. |

`"sliced"` and `"cipsi"` are retired (`config.RETIRED_GF_METHODS`: rejected with the reason,
replayed as `"lanczos"` from an old archive).

Orthogonal switches on the default Lanczos path:

- **`sparse`** — `block_Green_sparse` operates on the `ManyBodyState` representation (the
  matrix is never formed; the sparse hash-distributed Lanczos kernel `BlockLanczos.pyx`), vs
  `block_green_impl` which forms a CSR/dense sector and runs the BLAS-3 array kernel
  (`BlockLanczosArray.pyx`). Rule of thumb: array kernel for small/dense sectors, sparse when
  the matrix cannot be formed. See `architecture_overview.md` ("which one to use").
- **Eigenstate grouping** (`config.GF_EIGENSTATE_GROUP`) — stack several thermal eigenstates
  into one wide block recurrence sharing a Krylov space. (The former operator split,
  `GF_OPERATOR_SPLIT`, is retired: `doc/reviews/gf_review.md`, row S3.)
- **Truncation capping** (`truncation_threshold`) — `_CappedBasisProxy` freezes basis growth
  at a global determinant cap; the post-freeze recurrence is exact Lanczos of the projected
  `PHP` (see `doc/plans/truncation_reliability.md`).

## Convergence monitoring

The Lanczos path does not run a fixed number of blocks; it watches the resolvent stop moving.
`gf_convergence.py` owns this:

- `_make_gf_convergence_monitor(...)` — the runtime gate. It rebuilds the block continued
  fraction on a subsampled mesh and stops when the relative change falls below
  `slaterWeightMin²` (floored). This rebuild is the single largest cost of the recurrence
  (~53% at `reort=NONE`), so it is sampled every `config.GF_CHECK_EVERY` blocks during the
  long approach and every block once within `config.GF_NEAR_FACTOR × tol` of convergence.
- `_lanczos_convergence_summary(...)` — the post-hoc counterpart feeding the diagnostics.

The monitor converges `G` **where the caller will evaluate it** (`_gf_eval_meshes`): a
Matsubara-only self-energy does not pay to resolve the real-axis resolvent at broadening
`delta`.

## The RIXS solver chain (`rixs.py`)

RIXS is the deepest stack. For each incoming photon energy `wIn`, an intermediate
core-excited resolvent (R1) is solved and then projected through the emission step (R2). The
R1 solve is the cost target, and `_R1SolverChain.solve` tries four tiers in order, each
falling back to the next when it declines:

```
1. SectorResolventCache.try_solve   # dense spectral cache over a closed H-sector:
   (gf_shift_recycling.py)          #   eigendecompose once, serve every shift by projection.
        │  declines if sector too large (config.GF_SECTOR_DENSE_MAX)
        ▼
2. KrylovShiftedResolvent.solve     # one distributed block-Lanczos recurrence serves every
   (gf_shift_recycling.py)          #   shift of a fixed right-hand side.
        │  declines under the memory cap (config.GF_KRYLOV_RECYCLE_MAX_BYTES)
        ▼
3. cg.block_bicgstab                # per-point solve, restarted while still making progress
   (via gf.solve_shifted_block)     #   (config.GF_BICGSTAB_RESTARTS / _RESTART_PROGRESS)
        │  stagnates near a pole
        ▼
4. gmres.block_gmres                # minimizes the residual; has no r0-stagnation mode
   (via gf.solve_shifted_block)     #   (config.GF_GMRES_RESTART / _MAX_RESTARTS)
```

Every tier is collective-consistent: the decline/escalation decisions derive from
allreduce'd norms so every rank takes the same branch. `SectorResolventCache` can persist its
eigendecompositions to disk across runs (`config.GF_SECTOR_CACHE_DIR`) — the one-time `eigh`
is the dominant cost, so it is paid once per material rather than once per run.

On top of the per-`wIn` chain sits the **greedy adaptive sampler** (`_rixs_map_adaptive`,
enabled by `config.GF_RIXS_ADAPTIVE_TOL`): a set-valued AAA rational interpolant predicts the
map and only the incoming energies it cannot yet predict are actually solved (measured 28 of
121 solves on NiO L3 at 1e-4 relative error).

## Spectra dispatch (`spectra.py`)

`simulate_spectra(...)` is the orchestrator. Per spectrum:

- **PS / XPS / NIXS** — per-operator spectra via `calc_spectra`.
- **XAS** — the *projected* path (`calc_spectra`, per polarization) when an XAS projector
  file is given; otherwise the *tensor* path (`calc_spectra_tensor`) storing a Cartesian tensor
  under `spectra.h5:XAS/tensor` that `polarization.py` contracts with concrete polarizations
  at plot time.
- **RIXS** — the projected path (`rixs.calc_map`) with a projector file, else the
  tensor path (`rixs.calc_tensor_map`) storing the Kramers-Heisenberg tensor under
  `spectra.h5:RIXS/tensor`.

The transition operators themselves come from `transition_operators.py` (pure physics
builders); `rixs.py` owns the whole RIXS half.

## Diagnostics

`gf_diagnostics.py` runs representation-independent checks on the assembled `G` (thermal
cutoff, mesh density, causality) plus solver-specific records (Lanczos convergence, BiCGSTAB
residuals, basis truncation). Thresholds are derived from the value actually used
(`config` / `slaterWeightMin`), never re-hardcoded, so a diagnostic can never disagree with
the solve it is judging.

## Where to start reading

- Adding a spectrum type → `spectra.simulate_spectra` and `transition_operators.py`.
- Tuning GF performance/memory → `config.py` (every knob), then `get_Greens_function`.
- A RIXS map is slow or wrong → `rixs._R1SolverChain.solve` and the tier modules
  (`gf_shift_recycling.py`).
- A new distribution/parallelism concern → `gf_units.run_units_distributed` and
  `basis_split.py`; read [`mpi_model.md`](mpi_model.md) first.
- A new self-energy estimator → `sigma_estimators.py` (the protocol, `DysonEstimator`, the
  registry); the family reaches the engine through `get_Greens_function(operator_families=...)`.
- A new unit kind → `gf_engine.py` (the unit kernel and reassembly) and
  `gf_units.enumerate_gf_units` (operator groups of any width).
