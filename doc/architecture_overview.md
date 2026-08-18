# ImpurityModel Architecture Overview

This document describes the architecture of the `impurityModel` codebase: the C++/Cython
kernels, the Python module layering, and the execution flow of a calculation.

## Cython Extensions (`src/cython/`)

The performance-critical operations are implemented in C++ and exposed to Python via
Cython. This allows high-performance manipulation of quantum many-body states and
operators.

### Key Classes
1. **`SlaterDeterminant`**
   - **Role:** Represents a many-body Slater determinant using 64-bit integer chunks to track spin-orbital occupations.
   - **Details:** Wraps `std::vector<uint64_t>`. Bit manipulation compactly represents fermion occupation numbers, allowing very fast application of creation/annihilation operators and comparison of basis states.

2. **`ManyBodyState`**
   - **Role:** Represents a quantum many-body state — or a shared-support block of `p` states sharing the same set of nonzero Slater determinants (e.g. the `p` columns of a Block Lanczos iteration) — as a superposition of Slater determinants.
   - **Details:** Wraps a custom C++ block container (`ManyBodyBlockState`, sorted-unique determinant keys with `p` amplitudes per row): `psi[det]` / `.get()` / `.items()` / `.values()` return length-`p` `Row` views. A single ordinary state is `p == 1`, not a distinct class or code path — the block storage is the only representation, and every boundary kernel (`apply`, inner products, MPI redistribution) is written block-native. This replaced an earlier two-class design (a `std::flat_map`-backed single-vector class alongside the block class); the flat class was fully retired in the `ManyBodyState`/`ManyBodyBlockState` unification campaign (see `doc/plans/manybodystate_block_unification.md`).

3. **`ManyBodyOperator`**
   - **Role:** Represents a many-body operator as creation/annihilation sequences with amplitudes.
   - **Details:** Maps tuples of integer-indexed creation/annihilation operators (e.g. $c^\dagger_i c_j$) to complex amplitudes. Its `__call__` (equivalently `apply()`) applies the operator to a `ManyBodyState`, returning a new `ManyBodyState` of the same width: the term loop, fermion sign and restriction check run once per (determinant, term), emitting all `p` columns with `p` fused multiply-adds — bit-for-bit identical to `p` independent single-column applies at cutoff 0, and with no measurable overhead at `p == 1` (see the unification campaign doc's Phase 7 step-4 perf check). The sparse operator-state product is heavily optimized in C++.
   - **Term keys:** a term is keyed by its process tuple in *product* (left-to-right) order, so `((i, 'c'), (j, 'a'))` is $c^\dagger_i c_j$. The empty tuple `()` keys the **constant** (identity) term — a constant is just the zero-length operator string, and needs no orbitals of its own. `ManyBodyOperator()` is the **zero** operator; the identity is `ManyBodyOperator.identity()`.
   - **Canonical form:** stored terms are always in canonical normal order — creations before annihilations, each group ascending in orbital, Pauli-vanishing terms dropped, terms equal up to ordering merged. Constructors and all algebra maintain this, so `to_dict()` reports the canonical strings rather than the terms as written (`{((0,'a'),(0,'c')): 1}` reads back as $1 - n_0$). Only `__setitem__` can break the invariant; `canonicalize()` restores it and `is_canonical()` reports it. This is what makes the algebra simplify: without it `A*B - B*A` would never cancel.

   - **Algebra:** `+`, `-`, unary `-`, scalar `*` and `/`, and a scalar on either side of `+`/`-` (so `z - hOp` is the resolvent shift). `A * B` — equivalently `A @ B` — is composition, `(A*B)(psi) == A(B(psi))`; `A ** n` is the n-fold product. `commutator(A, B)` and `anticommutator(A, B)` are available as module-level functions and as methods; both skip term pairs on disjoint orbitals exactly, which is what makes `[H, c_i]` cost a pass over the terms touching orbital `i` rather than `len(H)` products. Also `adjoint()`/`dagger()`, `is_hermitian()`, `hermitian_part()`, `prune(tol)`, `approx_equal(other, tol)`, `orbitals()` and `body_rank()`.

     Products cost `len(A) * len(B)` term pairs before cancellation, so compose small operators — squaring a full Hamiltonian is not tractable. The two-body observables are built this way: `observables.casimir_operator` ($J^2 = J_-J_+ + J_z^2 + J_z$), `observables.spin_correlation_operator` ($\mathbf S_A\cdot\mathbf S_B$) and `lie_algebra.reconstructed_casimir_operator` ($\sum_a \hat O_a^2$). Build them once and reuse across states; the `apply_*` wrappers rebuild per call.

   - **Restrictions are not algebraic:** the occupation masks set by `set_restrictions` / `set_weighted_restrictions` belong to the operator *object*, not to the operator, and are **not** propagated through `+`, `-`, `*` or any bracket. A derived operator must have its restrictions set explicitly — which is what `gf_solvers` and `manybody_basis.Basis.set_restrictions` do.

4. **MPI Utilities**
   - **Role:** Efficient parallelization across ranks.
   - **Details:** Functions like `pack_determinants_cy` and `pack_block_fused_cy`/`unpack_block_fused_cy` serialize `ManyBodyState` blocks into contiguous byte buffers for a single-round `Neighbor_alltoallv` via `mpi4py`. Determinants are hash-distributed: each Slater determinant is owned by rank `hash(sd) % size`, and no rank ever holds a full state vector.

### Block Lanczos kernels: which one to use

There are two Block Lanczos kernels with identical reorthogonalization semantics
(they share the deflation, W-recurrence, FULL/PARTIAL/SELECTIVE reort, and threshold
logic — see `BlockLanczosArray.pyx`):

- **`BlockLanczos.pyx` — sparse / hash-distributed** (`block_lanczos_cy`,
  `thick_restart_block_lanczos_cy`, `implicitly_restarted_block_lanczos_cy`). Operates
  directly on `ManyBodyState`/`ManyBodyOperator`; the Hamiltonian matrix is **never
  formed**; MPI parallelism distributes Slater determinants by `hash(sd) % size`. Use
  this for a **large Hilbert space** where the dense/CSR matrix would not fit.

- **`BlockLanczosArray.pyx` — array / dense-or-CSR** (`block_lanczos_array_cy`).
  Operates on NumPy arrays / SciPy sparse operators; MPI parallelism is by row-block.
  The hot path uses BLAS-3 (`zgemm`), so it is fastest for **small/dense sectors** and
  for **block size `p > 1`** (the BLAS-3 speedup grows with `p`). Use this when the
  sector matrix is small enough to form, e.g. the Green's-function continued fraction
  and CIPSI reference solves.

Rule of thumb: **array kernel for small/dense, BLAS-friendly sectors; sparse kernel
when the matrix cannot be formed.** Both are driven through the same `Reort` modes and
the same TRLM/IRLM drivers, so switching is a matter of the input type.

The IRLM/TRLM restart logic lives inside `BlockLanczos.pyx` (`_irlm_core`,
`_trlm_core`); the shared EA16 numerics (residual norms, acceptance tolerances,
restart compression, locked-overlap recurrence) live in the Python module
`ed/ea16.py`, which both Cython kernels import at runtime. `ed/irlm.py` and
`ed/trlm.py` are thin re-export wrappers around the compiled entry points.

### Block orthonormalization: `TSQR.pyx`

Every block-Krylov routine — both Lanczos kernels, the starting/restart block
normalizations, the IRLM restart block, `block_bicgstab` and `block_gmres` — factors its
tall-skinny block through one leaf module, `TSQR.pyx` (`impurityModel.ed.TSQR`). It computes
the triangular factor from the block itself (LAPACK `zgeqrf` over local row panels, a Givens
sweep over flat packed triangles to merge them, one `Allgather` to combine the ranks) and
then forms `Q = A R^{-1}` by back substitution, instead of going through the Gram matrix
`A^H A`, which squares the condition number. Consequences worth knowing:

- The global `R` is **bitwise identical on every rank** by construction (the same merges are
  replayed in rank order everywhere), which is what lets each rank decide the same deflated
  block width without a broadcast.
- Deflation is decided from true singular values; the contract callers see is `k > 0`
  (retained rank), `k == 0` (numerically zero block relative to a caller-supplied `scale` —
  invariant subspace), `k == -1` (non-finite factor — a *corrupted* recurrence, not a closed
  one).
- `TSQR.pyx` owns `EPS`, `DEFLATE_TOL`, `DEFLATE_EVAL_TOL` and `BREAKDOWN_TOL`;
  `BlockLanczosArray` re-exports the ones its callers read from it.
- `block_tsqr` (in `_reort.pxi`) is the representation-dispatching entry point, so array,
  single-`ManyBodyState` and `list[ManyBodyState]` callers all run the same factorization.

`_cholesky_or_deflate` / `_cholesky_qr2` remain in `BlockLanczosArray.pyx` (next to the array
kernel's other private helpers) as the reference implementation the CholeskyQR2-era regression
tests are written against; no production path calls them. See `doc/lanczos_invariants.md`
("deflation vs breakdown scales") for the derivation their docstrings used to carry.

### File organization (`.pxi` includes)

The three large kernels are split into `.pxi` textual includes for readability (same
compiled modules; `setup.py` lists them in each Extension's `depends=` so edits trigger a
recompile). Each `.pxi` opens with a reading-map header:

- `BlockLanczos.pyx` = `_lanczos_step.pxi` (core recurrence) + `_trlm.pxi` (thick-restart) +
  `_irlm.pxi` (implicitly-restarted / EA16).
- `ManyBodyUtils.pyx` = `_slater_state.pxi` (`SlaterDeterminant`) + `_operator.pxi`
  (`ManyBodyOperator`) + `_mpi_pack.pxi` + `_krylov_store.pxi` (`SparseKrylovDense`) +
  `_block_state.pxi` (`ManyBodyState`).
- `BlockLanczosArray.pyx` keeps the array kernel and includes `_reort.pxi` (the
  `ManyBodyState`-path block primitives + `selective_orthogonalize`/`apply_reort`).

## Python Codebase (`src/impurityModel/ed/`)

The Python modules are layered; a module only imports from layers below it, and the
CLIs sit strictly on top. **Physics/operator-algebra modules never import solvers.**

```
Layer 0: average, utils, config, polarization, product_state_representation, op_parser,
         h0_format, mpi_comm, solver_trace, dc_record, ManyBodyUtils (Cython)
Layer 1: operator_algebra, dc_search
Layer 2: atomic_physics, eigensolvers, lie_algebra, symmetries, block_structure,
         transition_operators
Layer 3: observables, spin_pairs, dc_reference
Layer 4: manybody_basis (+ basis_generation, basis_restrictions,
         basis_transcription, basis_split), solver_basis
Layer 5: gf_primitives, gf_convergence, gf_shift_recycling, gf_units, gf_solvers,
         greens_function, spectra, rixs,
         cg, cipsi_solver, groundstate, hartree_fock, hamiltonian_io, gf_diagnostics,
         gs_statistics, dc_static, dc_criteria, dc_frozen, sigma, model
Layer 6: drivers: get_spectra, selfenergy, susceptibility
Layer 7: CLIs: scripts/cli (umbrella), scripts/{spectra,selfenergy,susceptibility},
         scripts/{plot_spectra,plot_RIXS}; entry points impurityModel / python -m impurityModel
```

`model.py` is the single construction point for the *physics* of a problem (the `ImpurityModel`
dataclass plus the `Meshes`/`BasisOptions`/`SolverOptions`/`SpectraOptions` option groups). It
imports only `atomic_physics`/`operator_algebra`/`hamiltonian_io`, so it sits below the drivers
and is what both the CLIs and embedded callers (the RSPt interface) build to pass into a driver.
`impurityModel.api` re-exports it together with `calc_selfenergy` and the save helpers.

### Foundations (Layer 0–2)
- **`average.py`** — thermal averaging (`thermal_average`, `thermal_average_scale_indep`, `k_B`).
- **`utils.py`** — small numerics/printing helpers (`rotate_matrix`, `matrix_print`, …).
- **`config.py`** — the central registry of the `GF_*` environment-variable tuning knobs: one `Knob` declaration per knob (name, type, default, clamp, rationale), a lazy `.get()` accessor, and `dump()` (which generates `doc/configuration.md`). Every solver/spectra module reads its knobs through this, so a default lives in exactly one place. Depends on nothing.
- **`product_state_representation.py`** — conversions between bit/bytes/tuple/string encodings of product states.
- **`op_parser.py`** — parsing of operator files for the CLIs.
- **`mpi_comm.py`** — the MPI communication primitives: sparse graph-alltoall of determinants and states, chunked broadcast/allgather of dicts, task partitioning (`get_job_tasks`).
- **`solver_trace.py`** — opt-in accounting of where a solve spends its wall clock: `tracing()` opens a collection block, `timed(kind)` / `note(kind)` record events, `labelled(**fields)` tags everything recorded underneath (which is how a sector solve three frames deep knows which trial `mu` it belongs to). Imports nothing from the package, so any layer may write into it — `groundstate.calc_energy` records the build/expand/eigensolve split of one sector solve, `dc_criteria` records one `dc_evaluation` per trial shift. Every hook is a single `is None` test when no block is open, so the production path pays nothing. Kinds **nest**, so summing across them double-counts. Traces are rank-local and issue no collective; a caller wanting a cross-rank comparison reduces the count itself, having first broadcast the decision to trace.
- **`polarization.py`** — numpy-only polarization vectors and tensor contractions (`contract_spectra_tensor`, `contract_rixs_tensor`, dichroism/isotropic helpers) that turn the tensor quantities `spectra.py` computes into polarization-resolved intensities; used both by `spectra.py`'s projector code paths and by the `plot_spectra`/`plot_RIXS` CLIs as a post-processing step, so no MPI or solver imports.
- **`operator_algebra.py`** — algebra on second-quantized operator dicts (`addOps`, `daggerOp`, `combineOp`, …) and the `(l, s, m)` label ↔ flat-index conversions (`c2i`, `i2c`). These serve the *pre-conversion* path only: operators keyed by `(l, s, m)` labels cannot be `ManyBodyOperator`s, which need integer orbital indices. Once an operator is integer-indexed, use the `ManyBodyOperator` algebra instead. Note `combineOp` is a single-particle *matrix* product, not `ManyBodyOperator.__mul__`.
- **`atomic_physics.py`** — single-shell atomic physics: Slater–Condon Coulomb integrals (`getU*`), spin-orbit coupling (`getSOCop`), Zeeman field (`gethHfieldop`), spherical↔cubic transforms, the MLFT double-counting correction (`dc_MLFT`), and the average Coulomb repulsion/exchange extractor (`uj_from_u4`) the `dc_static` FLL/AMF schemes derive `U`/`J` from.
- **`eigensolvers.py`** — eigensolver drivers for the low-energy spectrum: dense (`numpy.linalg.eigh`) and ARPACK (`scipy.sparse.linalg.eigsh`), behind the `eigensystem` driver and the MPI-aware `HermitianOperator` wrapper. No connection to the Block Lanczos stack.
- **`lie_algebra.py`** — the *algebraic half* of the symmetry machinery: tensor extraction/rotation (`extract_tensors`, `rotate_hamiltonian`), one-body symmetry discovery (the single-particle commutant null space), the Cartan reduction and joint diagonalization, and the reconstructed-Casimir observables. Depends only on `ManyBodyUtils`; `symmetries.py` builds its conserved charges and rotations on top of it.
- **`symmetries.py`** — the consumer half built on `lie_algebra`: conserved-charge classification, occupation-window restrictions (`S_z`-weighted and frozen-shell flavors), impurity/bath occupation classification, and the impurity/Green's-function block structures used to deduplicate and sectorize GF/RIXS solves. Re-exports the `lie_algebra` primitives for backward compatibility.
- **`block_structure.py`** — the `BlockStructure` type: detection of identical/transposed/particle-hole-related orbital blocks and matrix↔block conversions.
- **`transition_operators.py`** — pure second-quantized transition-operator builders for the spectroscopy drivers: dipole (`dipole_operator(s)`, `daggered_dipole_operators`), the plane-wave NIXS operator (`nixs_operator(s)`), the bare photo-emission/inverse-photo-emission ladder operators (`get{,Inverse}PhotoEmissionOperators`), and the `sph_harm` helper. Depends only on `atomic_physics` and `operator_algebra`; `spectra.py` builds its transition operators through these.
- **`dc_search.py`** — `_solve_dc_shift`, the root find all three ED double-counting criteria share, over the uniform shift `dc(mu) = dc_guess + mu * I`. It knows nothing about what the observable is, so it carries no ED machinery and imports no solver -- only `config` and `solver_trace` (Layer 0). **What is monotone is the charge sector, not the residual**: on a fixed space `E_N(mu) = E_N(0) - mu*n_N` is affine, so the winning sector is the argmin of a family of lines and hence a non-decreasing step function of `mu`. Stage A steps one-sidedly on that integer to find a point in the nominal sector; Stage B solves the criterion inside it, where the charge state is fixed and the residual is smooth with a known sign of slope (safeguarded secant + `_refine_bracket`, Newton-seeded when the caller vouches for a slope). The bidirectional geometric scan the search used to do came from the opposite premise and spent half its evaluations on the side no root could lie on; it survives only as a fallback after the informed direction fails. The residual -- the value of a *collective* observable, replicated only to roundoff -- is **broadcast once, in one place**, together with the sector, because every branch it gates decides whether the next collective call happens. An unreachable sector propagates as `None` ("the criterion is undefined here"), never as a finite penalty that could flip a sign and manufacture a bracket. Also the search's cost accounting (`_report_dc_trace`, the `DC_DIAGNOSTICS` knob): the per-`mu` build/expand/eigensolve split, and `_dc_chi`, the slope that converts a residual into an error on the answer (`delta_mu = tol / chi`), measured only over an evaluated pair that *straddles the returned mu*, is wider than the search's own bracket resolution, and stays inside one charge sector.
- **`dc_record.py`** — the single delimited `key = value` block every criterion emits on rank 0, searches and closed forms alike (`recording()` is a context manager that times the call and emits from a `finally`, so a search that raises `DoubleCountingUnreachable` still records how far it got). Unconditional -- not gated on `verbosity`, because the double counting *is* the answer of the call -- and a **closed vocabulary**: `_solve_dc_shift`'s internal report is projected through an explicit key list (`project_search_report`), so working state cannot leak into, and then silently change, a published format. `dc_levels()` reports `Tr(dc)` and `Tr(dc)/n_imp` separately (the literature's single eV number is the per-orbital level, and that distinction is a factor of `n_imp` in any comparison against a published value) and `dc_spread()` reports the diagonal's peak-to-peak, present only when the `dc` is *not* uniform -- `sigma_inf` is the scheme whose whole content is the orbital dependence an average destroys. The emit runs inside a `try`, because a diagnostic printed from a `finally` must never replace the exception it exists to document. Imports `numpy` and `mpi4py` only -- nothing from the package, so it is a Layer 0 leaf like `solver_trace` -- and issues no collective (the rank-0 gate is on printing alone).

### Observables (Layer 3)
- **`observables.py`** — occupations and angular-momentum expectation values from single-particle density matrices in the spherical basis, many-body spin/orbital/Casimir operator builders, and (thermally averaged) expectation-value reporting for degenerate manifolds.
- **`spin_pairs.py`** — derivation of the `(down, up)` spin-orbital pairings of impurity and bath consistent with a given one-body Hamiltonian (used for spin-flip basis completion and weighted restrictions).
- **`dc_reference.py`** — the DFT reference filling every double-counting scheme measures against: the Fermi occupation (`_noninteracting_impurity_occupation`) and density matrix (`_noninteracting_impurity_rho`) of the **raw** `h0`. `model.h0` is the KS/DFT Hamiltonian of the `h0 - dc + U` contract, so its raw filling *is* the DFT occupation; filling `h0 - dc` instead sinks the impurity levels below E_F and saturates the reference at the full shell (NiO: `n0` pinned at 10 instead of ~8.6 for a ~4 Ry dc). Because the reference is a property of the *discretized* bath it saturates whenever the fit places no impurity weight across E_F, which for FLL is an O(1)-electron, several-eV error made silently -- so the saturation check lives here, with the reference, and every scheme that defaults to it routes through `_warn_if_reference_saturated`. Deterministic NumPy on the replicated `h0`, depending only on `lie_algebra` and `ManyBodyUtils`: no MPI collective, identical on every rank.

### The many-body basis (Layer 4)
- **`manybody_basis.py`** — the `Basis` class: the distributed set of Slater determinants and its MPI bookkeeping. Storage/lookup (rank-local sorted determinant list, state → global-index dict, hash-routed distributed lookups), `redistribute_psis`, operator-driven `expand`, and lifecycle (`clone`, `copy`, `clear`, `free_comm`).
- **`basis_generation.py`** — pure enumeration of the initial determinant basis from occupation windows, and spin-flip completion of determinant sets. No MPI.
- **`basis_restrictions.py`** — occupation-restriction construction: effective (observed) restrictions of the current basis, connectivity-derived ground-state restrictions, and widened restrictions for excited/spectral sectors. Contains collectives; call from all ranks.
- **`basis_transcription.py`** — transcription between the distributed basis and dense/sparse linear algebra: wavefunction vectors (`build_vector`, `build_state`, …), operator matrices (`build_sparse_matrix`, `build_dense_matrix`), density matrices (`build_density_matrices`).
- **`basis_split.py`** — adaptive splitting of a `Basis` over MPI colors (`split_basis_and_redistribute_psi`) with the pure packing math in `_pack_units`; the distribution backbone of `gf_units.run_units_distributed`.
- **`solver_basis.py`** — `prepare_solver_basis`: assembles the interacting Hamiltonian `h0 - dc + U(u4)`, adaptively rotates into the impurity-diagonalising basis when that does not densify the Coulomb tensor, and derives the bath valence/conduction split, GF block structure, and per-group impurity orbital grouping/occupation from it (`SolverBasis`). Also `get_symmetry_generators`, the one-body symmetry discovery `cipsi_solver.expand` falls back to when not given generators explicitly. Imports only `symmetries` + `ManyBodyUtils`, so it sits below every one of its callers (`groundstate`, `cipsi_solver`, `dc_criteria` in Layer 5; `selfenergy`, `susceptibility` in Layer 6) rather than being reached through any of them. Split out of `selfenergy.py` (which keeps a re-export shim for its old private names).

### Solvers and spectra (Layer 5)
- **`groundstate.py`** — the ground-state driver `calc_gs`: builds the variational basis (CIPSI + Hartree-Fock occupation seeding), solves for the low-energy states, and reports observables.
- **`cipsi_solver.py`** — selected-CI (CIPSI) iterative basis expansion.
- **`hartree_fock.py`** — mean-field occupation seeding for the basis generation.
- **`gf_primitives.py`** — dependency-free GF building blocks: QR/state-vector plumbing (`build_qr`, `_distributed_seed_qr`), the block-tridiagonal continued fraction (`calc_G`, `calc_continuants`, `_block_cf_inverse`, `calc_thermally_averaged_G`, `PairwiseGF`/`calc_G_pairwise`), and the `truncation_threshold`-capping `_CappedBasisProxy`. Imports nothing from the other two below or from `greens_function`.
- **`gf_convergence.py`** — the runtime block-Lanczos convergence monitor (`_make_gf_convergence_monitor`) and its post-hoc counterpart (`_lanczos_convergence_summary`), plus the shared frequency-mesh helpers. Depends only on `gf_primitives`.
- **`gf_shift_recycling.py`** — `SectorResolventCache` (dense spectral cache over a closed H-sector) and `KrylovShiftedResolvent` (one distributed block-Lanczos recurrence serving every shift of a fixed right-hand side): the two tiers ahead of the per-point BiCGSTAB/GMRES fallback in the RIXS R1 solver chain. Depends only on `gf_primitives`.
- **`gf_units.py`** — the GF *distribution engine*: enumerate the independent GF work units a spectrum needs (`enumerate_gf_units`, `GFUnit`), weight their relative cost (`unit_cost_weights`), and drive them across a color-split communicator with per-unit basis rebuild + seed redistribution (`run_units_distributed`). Depends on `gf_primitives`, `memory_estimate`, `basis_split`; does not import the resolvent kernels.
- **`gf_solvers.py`** — the per-unit GF *resolvent kernels*: the block-Lanczos recurrence serving the whole mesh (`block_green_impl`/`block_Green_sparse`, wrapped by `block_Green`) and the per-frequency BiCGSTAB driver (`block_Green_bicgstab` on `solve_shifted_block`). Depends on `gf_primitives`/`gf_convergence`/`gf_shift_recycling` and the Lanczos/BiCGSTAB/GMRES kernels; does not import `gf_units` or `greens_function`.
- **`greens_function.py`** — interacting Green's functions via block Lanczos continued fractions: the top-level drivers (`get_Greens_function`, `calc_Greens_function_with_offdiag`, the bicgstab/sliced routers) and assembly (`build_full_greens_function`, `save_Greens_function`), built on the `gf_units` distribution engine and the `gf_solvers` kernels. Still re-exports the `gf_primitives`/`gf_convergence`/`gf_shift_recycling` symbols that other modules/tests reach via `greens_function.X` / `gf.X`.
- **`spectra.py`** — the `simulate_spectra` orchestrator and the XAS/XPS/PS/NIXS drivers on top of `greens_function`. PS/XPS/NIXS and the projector-driven XAS path return per-operator spectra directly; the default (unprojected) XAS path returns the polarization *tensor* (`calc_spectra_tensor`) rather than a polarization-contracted spectrum -- `simulate_spectra` stores the tensor as-is (`spectra.h5`: `XAS/tensor`), and `polarization.py` contracts it with concrete polarizations as a cheap post-processing step. Re-exports `rixs.calc_map`/`calc_tensor_map` so `simulate_spectra` and existing `spectra.getRIXSmap_*` callers reach them unchanged.
- **`rixs.py`** — the RIXS (resonant inelastic x-ray scattering) map half, split out of `spectra.py`: incoming-energy work-unit sizing, the greedy adaptive incoming-energy sampler, the per-tier R1 solver chain (`_R1SolverChain`), the flat-unit distribution driver `_rixs_map_flat`, and the two public drivers `calc_map` (per-polarization) / `calc_tensor_map` (Kramers-Heisenberg tensor stored under `spectra.h5:RIXS/tensor`). Sits on `greens_function` like `spectra.py`.
- **`cg.py`** — block BiCGSTAB solver (used by the RIXS tensor path).
- **`gf_diagnostics.py`** — convergence/consistency diagnostics for computed Green's functions.
- **`gs_statistics.py`** — ground-state statistics computation, printing, and saving.
- **`hamiltonian_io.py`** — construction and file I/O of the **labelled** `(l,s,m)`/`(l,b)` impurity Hamiltonian: readers for the pickled operator dict, the `.dat` operator-*term list* (via `op_parser`; it is not a dense matrix) and the `.json` crystal-field parameters, plus the builders combining h0 with SOC, magnetic field, Coulomb, and double counting. The flat single-index `.h0` interchange format is read by `h0_format.py` and relabelled into `(l,s,m)`/`(l,b)` here (`flat_h0_to_labelled`) -- valid only when the header guarantees `basis: "spherical"` and `spin_ordering: "down_first"` (each checked, each raises otherwise), because `i2c` on a single-shell `nBaths` dict is the exact inverse of the flat layout precisely under those two guarantees.
- **`h0_format.py`** — reader/writer for the `.h0` format written by `rspt2spectra`'s `build_h0` (spec: `doc/h0_file_format.md`): a JSON header plus flat `i j re im` terms, with the impurity block first. Flat single-particle indices, *not* the `(l,s,m)` labels `hamiltonian_io` reads and `c2i` remaps — the two orderings differ, so reading one as the other is silently wrong rather than an error; `hamiltonian_io.flat_h0_to_labelled` is the one sanctioned crossing, gated on the header. numpy + stdlib only, no MPI, so the reader is rank-invariant by construction. `model.load_model` is the single dispatch point over both families, for all three drivers (`spectra` included, via `ImpurityModel.from_shells`).
- **`dc_static.py`** — the closed-form schemes, each evaluated at `dc_reference`'s filling unless the caller supplies one: `fll_dc` (Fully Localized Limit), `amf_dc` (Around Mean Field), `sigma_inf_dc` (K. Held's `Sigma(infinity)`, the full static Hartree-Fock self-energy matrix at the actual density matrix, `sigma.get_Sigma_static`) and `nominal_dc` (FLL at the nominal integer occupation, which cannot inherit the reference filling's bath-discretization sensitivity). No ED solve and no MPI collective. Each emits a `dc_record` block of its own, `status = closed_form`; `nominal_dc` reuses FLL's formula through a private helper precisely so it emits *one* record, naming itself. `atomic_physics.uj_from_u4` derives the average Coulomb repulsion/exchange FLL and AMF need from `model.u4`.
- **`dc_criteria.py`** — the ED-based criteria built on that search: `fixed_peak_dc` pins a spectral peak (the sector-energy difference `E[N+1] - E[N]`, or `E[N] - E[N-1]` for a removal peak), `fixed_gap_dc` pins the midpoint of the two, `(E[N+1] - E[N-1])/2`, after Karolak et al. (arXiv:1004.4569), where the occupation criterion is flat and so determines nothing — note those are *total-cluster* charge sectors with no impurity projection, so the edges can be bath levels; the record's `delta_sum` reports how much of the transferred charge actually reached the impurity and warns when it is small; `fixed_occupation_dc` pins the impurity occupation, and with no explicit target pins it to `dc_reference`'s DFT occupation -- the criterion for CSC DFT+DMFT of wide-window p-d models, where only the d electrons enter the ED solve but the p electrons sit inside the DFT window. The peak and gap criteria share `_SectorContext`, so the setup, the centre-sector walk and the fixed-sector solves have one definition between them. At every trial shift all three determine the ground-state sector the *identical* way `groundstate.calc_gs` does (`find_ground_state_basis`'s HF-seed-then-walk search, not a search pinned at the input occupation): a dc measured on a different sector than the one `calc_selfenergy` later finds locks the downstream calculation onto the wrong charge state, which is worse than not fixing anything. Memory-limited like `calc_selfenergy`/`find_ground_state_basis` -- a `None` `truncation_threshold` derives the determinant cap from available per-rank memory (`memory_estimate.suggest_truncation_threshold`, at the same safety fraction `calc_gs` uses: the three sector solves are sequential, so the live peak is the centre walk's own `SectorCache` — halving it, as the peak and gap criteria once did, measured the double counting on half the budget the self-energy run would use at that same `dc`) -- and both honor `BasisOptions.excitation_budget`/`chain_restrict` via `basis_restrictions.build_weighted_restrictions`, so the double counting is found on the same variational space the production solve uses. Also `build_union_space`: constructs the growth-only, state-averaged `FrozenSpaceSweep` a seeded search brackets on, widening the occupation window to `sector_radius` and expanding once per bracket-end `mu` so the frozen space spans every candidate charge sector rather than whichever one a single solve happened to reach. Its only caller today is `test_dc_frozen.py`; the seeded search itself is not on the default path (`DC_SEEDED_SEARCH`, default off).
- **`dc_frozen.py`** — `FrozenSpaceSweep`: `energy(mu)` / `occupation(mu)` / `chi(mu)` over a *fixed* determinant space, built from a basis and operator handed to it -- it does not construct one itself (`dc_criteria.build_union_space` does that, and depends on `dc_frozen` rather than the reverse, so this module imports no `dc_*` sibling). Because the double counting is a uniform impurity shift, `H(mu) = H(0) - mu*N_imp` with `N_imp` **diagonal**, so the whole family is one matrix build plus a diagonal shift — the CIPSI expansion, 96.5% of a real DC search, is skipped entirely. Measured on NiO-20 at cap 400: a true re-expanding evaluation 3.41 s, one frozen solve **0.003 s** (0.08%, ~1240x). That is what makes `chi = dn/dmu` affordable, and `chi` is what turns an occupation residual into the error on the answer (`delta_mu = delta_n / chi`; on a plateau `chi -> 0` and `dc` is unbounded). On a frozen space the theory is exact — `E_0(mu)` is a min of affine functions, hence concave, with `n(mu)` non-decreasing, and on a *pure* sector the shift is affine to machine precision — none of which holds for the re-expanding observable, which is why the production search assumes no monotonicity. The frozen space is a cheap *model* of the answer, not the answer: it bounds the reachable occupation by its own charge window (`occupation_is_interior` reports that), so a root located here must be confirmed against the true observable. Solves route through `CIPSISolver.get_eigenvectors(h_matrix=...)` rather than around it, so the warm-start cold-retry guard still applies.
- **`sigma.py`** — self-energy extraction downstream of `G`: the static (Hartree-Fock) and dynamic self-energies (`get_sigma`, `get_Sigma_static`), the hybridization function (`hyb`), the correlated/bath splitting (`get_hcorr_v_hbath`), and the physicality check (`check_greens_function`, `UnphysicalGreensFunctionError`). Split out of `selfenergy.py`.

### Drivers (Layer 6)
These are library modules — importable functions, no `__main__` — that the CLIs (and embedded
callers) invoke with an `ImpurityModel` + option groups.
- **`get_spectra.py`** — `build_spectra_model` is a thin wrapper over `model.load_model`'s multi-shell path (`ImpurityModel.from_shells`), assembling the full interacting model (2p core + 3d correlated shell) from an `h0` file -- labelled or flat, the file supplies the correlated shell and its bath only; `run_spectra` finds the lowest eigenstates and calculates the spectra (PS, XPS, XAS, NIXS, RIXS), writing `spectra.h5`.
- **`selfenergy.py`** — impurity self-energy for DMFT-style workflows: `calc_selfenergy(model, meshes, basis, solver, ...)` on top of `dc_criteria`/`dc_static` and `sigma` (all re-exported). Note that **a re-exported name is a new binding**: patching `selfenergy.fixed_occupation_dc` does not affect the call site inside `dc_criteria`, which reads its own. Tests stubbing a dependency must patch the module that looks the name up at call time, and prove the stub ran -- a dead patch leaves an assertion like `assert not calls` passing vacuously.
- **`susceptibility.py`** — dynamical local susceptibilities of the impurity (`chi_spin_zz`, `chi_orb_zz`, `chi_charge`, transverse `chi_+-`) on a real mesh and the bosonic Matsubara mesh, via `spectra.calc_spectra` resolvent branches with the elastic (Curie) weight projected out per degenerate manifold; writes `chi.h5` and prints a Van Vleck/Curie/screening-scale summary (the Hund's-metal diagnostic).

### CLIs (Layer 7, `scripts/`)
- **`scripts/cli.py`** (console script `impurityModel`, and `python -m impurityModel`) — the umbrella argparse dispatcher over the sub-commands below.
- **`scripts/spectra.py`** (`impurityModel spectra`) — builds the model via `build_spectra_model` and runs `run_spectra`; the radial file is optional (NIXS is skipped without it).
- **`scripts/selfenergy.py`** (`impurityModel selfenergy`) — builds/loads the model, runs `calc_selfenergy`, and saves Σ/G (RSPt `.dat`), the static Σ, and a per-cluster HDF5 archive; `--from-archive` reproduces a recorded run.
- **`scripts/susceptibility.py`** (`impurityModel susceptibility`) — runs `calc_susceptibility_workflow`; also accepts `--from-archive`.
- **`scripts/_plot_common.py`** — shared CLI plumbing for the plot scripts (input/output/figure-style arguments, `spectra.h5` loading, orbital-selection parsing, `.dat` export), ported from `pyRSPthon.cli._common`.
- **`scripts/plot_spectra.py`** (console script `plot_spectra`, or `impurityModel plot-spectra`) — plots PS/XPS/NIXS from `spectra.h5`, and XAS by contracting the stored spectral tensor with the requested polarizations (`--pol`, default x/y/z + isotropic; `--xmcd`/`--xld` dichroism; `--tensor-components`) via `polarization.py`; also overlays the RIXS-tensor fluorescence yield when both are present.
- **`scripts/plot_RIXS.py`** (console script `plot_RIXS`, or `impurityModel plot-rixs`) — plots the RIXS map from `spectra.h5`'s `RIXS/tensor`, contracting with `--pol-in`/`--pol-out` polarization pairs, `--mcd` circular dichroism, `--fy` fluorescence yield, and `--cuts`/`--emission` energy-loss line cuts, all as post-processing (no solver re-run).

## Test Suite (`src/impurityModel/test/`)

- **Framework:** `pytest` + `pytest-mpi`. Serial run: `pytest`. MPI run: `mpiexec -n 2 python -m pytest --with-mpi` (CI runs serial, 1 rank, 2 ranks, and 3 ranks — a rank can only own zero local determinants at 3+ ranks for small hash-distributed test fixtures, so `-n 2` alone cannot catch an empty-rank MPI deadlock).
- **MPI tests** are marked `@pytest.mark.mpi`; `conftest.py` redirects non-root-rank output to `.pytest_mpi_rank*.out`, adds a per-test watchdog, and synchronizes teardown. Never name a test subdirectory `mpi` (or any other registered marker name) — pytest folds every path component into `item.keywords`, and pytest-mpi's skip check (`MPIMarkerEnum.mpi in item.keywords`) then matches unmarked tests too; the MPI-comm-plumbing group lives in `mpi_infra/` for this reason.
- **Benchmarks** are marked `benchmark` and skipped by default; run with `pytest -m benchmark`.
- **Layout:** grouped by topic under subpackages, each with `__init__.py`: `lanczos/`
  (block-Lanczos recurrence, TRLM/IRLM restart+locking, TSQR, reorthogonalization,
  Krylov storage, deflation), `linear_solvers/` (CG/BiCGSTAB/GMRES), `gf/` (Green's
  functions, self-energy, susceptibility), `spectra/` (XAS/RIXS/polarization),
  `basis/` (ManyBodyState/basis representation), `operators/` (operator algebra,
  apply kernels), `symmetry/` (symmetry observables, ground-state statistics),
  `restrictions/` (CIPSI, truncation, occupation restrictions), `block_structure/`,
  `mpi_infra/` (MPI communication plumbing, rank independence), `misc/` (model/
  config/CLI/Hartree-Fock/utils). Perf/benchmark files stay with their topic group
  (the `benchmark` marker is the cross-cutting view, via `pytest -m benchmark`).
  Shared fixtures/oracles live in `support/` (not itself a test module): `testtol.py`
  (derived numerical tolerances), `lanczos_fixtures.py`, `gf_oracles.py`,
  `real_workload.py`/`_nio_workload.py` (realistic HDF5/pickle workload loaders),
  `restriction_diagnostics.py`/`restriction_sweep.py` (opt-in sweep scripts).
  `conftest.py` stays at the `test/` root; it applies suite-wide.

## Execution Flow
1. **Define model:** the non-interacting Hamiltonian is read/built (`hamiltonian_io`), Coulomb/SOC/field terms added (`atomic_physics`, `operator_algebra`).
2. **Construct basis:** `Basis` enumerates determinants from the occupation windows (`basis_generation`), optionally seeded by Hartree–Fock occupations and grown by CIPSI.
3. **Diagonalize:** iterative solvers repeatedly apply the Hamiltonian (`ManyBodyOperator` on `ManyBodyState`) through the Lanczos kernels to find the low-energy states.
4. **Observables:** density matrices, occupations, and angular-momentum/Casimir expectation values are computed (`observables`) and reported.
5. **Spectra / self-energy:** excitation operators are applied to the eigenstates and Green's functions are built from block-Lanczos continued fractions, distributed over MPI colors via `run_units_distributed`.

## MPI ground rules

These invariants have bitten before; hold them when changing code:
- Never gate an MPI collective on rank-local state (e.g. a `verbose` flag that differs per rank).
- No full state-vector gathers: determinants are hash-distributed, one owner per determinant. Observables use apply-local → redistribute → local-inner → `Allreduce`.
- `MPI_Comm_free` is collective: free communicators/intercomms at synchronized points (see `basis_split.py`), never from the garbage collector.
