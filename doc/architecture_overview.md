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
   - **Role:** Represents a quantum many-body state as a superposition of Slater determinants.
   - **Details:** Wraps a custom C++ `flat_map<SlaterDeterminant, std::complex<double>>` — essentially a highly optimized dictionary mapping basis states to complex amplitudes, with vectorized addition, scalar multiplication, and inner products.

3. **`ManyBodyOperator`**
   - **Role:** Represents a many-body operator as creation/annihilation sequences with amplitudes.
   - **Details:** Maps tuples of integer-indexed creation/annihilation operators (e.g. $c^\dagger_i c_j$) to complex amplitudes. Its `__call__` applies the operator to a `ManyBodyState`, returning a new `ManyBodyState`; the sparse operator-state product is heavily optimized in C++.

4. **MPI Utilities**
   - **Role:** Efficient parallelization across ranks.
   - **Details:** Functions like `pack_determinants_cy` and `pack_psis_fused_cy` serialize `ManyBodyState` objects into contiguous NumPy arrays for fast communication via `mpi4py`. Determinants are hash-distributed: each Slater determinant is owned by rank `hash(sd) % size`, and no rank ever holds a full state vector.

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

## Python Codebase (`src/impurityModel/ed/`)

The Python modules are layered; a module only imports from layers below it, and the
CLIs sit strictly on top. **Physics/operator-algebra modules never import solvers.**

```
Layer 0: average, utils, polarization, product_state_representation, op_parser, mpi_comm,
         ManyBodyUtils (Cython)
Layer 1: operator_algebra
Layer 2: atomic_physics, eigensolvers, symmetries, block_structure
Layer 3: observables, spin_pairs
Layer 4: manybody_basis (+ basis_generation, basis_restrictions,
         basis_transcription, basis_split)
Layer 5: gf_primitives, gf_convergence, gf_shift_recycling, greens_function, spectra,
         cg, cipsi_solver, groundstate, hartree_fock, hamiltonian_io, gf_diagnostics,
         gs_statistics
Layer 6: CLIs: get_spectra, selfenergy
```

### Foundations (Layer 0–2)
- **`average.py`** — thermal averaging (`thermal_average`, `thermal_average_scale_indep`, `k_B`).
- **`utils.py`** — small numerics/printing helpers (`rotate_matrix`, `matrix_print`, …).
- **`product_state_representation.py`** — conversions between bit/bytes/tuple/string encodings of product states.
- **`op_parser.py`** — parsing of operator files for the CLIs.
- **`mpi_comm.py`** — the MPI communication primitives: sparse graph-alltoall of determinants and states, chunked broadcast/allgather of dicts, task partitioning (`get_job_tasks`).
- **`polarization.py`** — numpy-only polarization vectors and tensor contractions (`contract_spectra_tensor`, `contract_rixs_tensor`, dichroism/isotropic helpers) that turn the tensor quantities `spectra.py` computes into polarization-resolved intensities; used both by `spectra.py`'s projector code paths and by the `plot_spectra`/`plot_RIXS` CLIs as a post-processing step, so no MPI or solver imports.
- **`operator_algebra.py`** — algebra on second-quantized operator dicts (`addOps`, `daggerOp`, `combineOp`, …) and the `(l, s, m)` label ↔ flat-index conversions (`c2i`, `i2c`).
- **`atomic_physics.py`** — single-shell atomic physics: Slater–Condon Coulomb integrals (`getU*`), spin-orbit coupling (`getSOCop`), Zeeman field (`gethHfieldop`), spherical↔cubic transforms, and the MLFT double-counting correction (`dc_MLFT`).
- **`eigensolvers.py`** — eigensolver drivers for the low-energy spectrum: dense (`numpy.linalg.eigh`), ARPACK (`scipy.sparse.linalg.eigsh`), and the block-Lanczos TRLM path, behind the `eigensystem` driver and the MPI-aware `HermitianOperator` wrapper.
- **`symmetries.py`** — automated symmetry discovery for second-quantized Hamiltonians: tensor extraction, conserved-charge classification, symmetry-adapted rotations, restriction widening, Hamiltonian rotation.
- **`block_structure.py`** — the `BlockStructure` type: detection of identical/transposed/particle-hole-related orbital blocks and matrix↔block conversions.

### Observables (Layer 3)
- **`observables.py`** — occupations and angular-momentum expectation values from single-particle density matrices in the spherical basis, many-body spin/orbital/Casimir operator builders, and (thermally averaged) expectation-value reporting for degenerate manifolds.
- **`spin_pairs.py`** — derivation of the `(down, up)` spin-orbital pairings of impurity and bath consistent with a given one-body Hamiltonian (used for spin-flip basis completion and weighted restrictions).

### The many-body basis (Layer 4)
- **`manybody_basis.py`** — the `Basis` class: the distributed set of Slater determinants and its MPI bookkeeping. Storage/lookup (rank-local sorted determinant list, state → global-index dict, hash-routed distributed lookups), `redistribute_psis`, operator-driven `expand`, and lifecycle (`clone`, `copy`, `clear`, `free_comm`).
- **`basis_generation.py`** — pure enumeration of the initial determinant basis from occupation windows, and spin-flip completion of determinant sets. No MPI.
- **`basis_restrictions.py`** — occupation-restriction construction: effective (observed) restrictions of the current basis, connectivity-derived ground-state restrictions, and widened restrictions for excited/spectral sectors. Contains collectives; call from all ranks.
- **`basis_transcription.py`** — transcription between the distributed basis and dense/sparse linear algebra: wavefunction vectors (`build_vector`, `build_state`, …), operator matrices (`build_sparse_matrix`, `build_dense_matrix`), density matrices (`build_density_matrices`).
- **`basis_split.py`** — adaptive splitting of a `Basis` over MPI colors (`split_basis_and_redistribute_psi`) with the pure packing math in `_pack_units`; the distribution backbone of `greens_function.run_units_distributed`.

### Solvers and spectra (Layer 5)
- **`groundstate.py`** — the ground-state driver `calc_gs`: builds the variational basis (CIPSI + Hartree-Fock occupation seeding), solves for the low-energy states, and reports observables.
- **`cipsi_solver.py`** — selected-CI (CIPSI) iterative basis expansion.
- **`hartree_fock.py`** — mean-field occupation seeding for the basis generation.
- **`gf_primitives.py`** — dependency-free GF building blocks: QR/state-vector plumbing (`build_qr`, `_distributed_seed_qr`), the block-tridiagonal continued fraction (`calc_G`, `calc_continuants`, `_block_cf_inverse`, `calc_thermally_averaged_G`, `PairwiseGF`/`calc_G_pairwise`), and the `truncation_threshold`-capping `_CappedBasisProxy`. Imports nothing from the other two below or from `greens_function`.
- **`gf_convergence.py`** — the runtime block-Lanczos convergence monitor (`_make_gf_convergence_monitor`) and its post-hoc counterpart (`_lanczos_convergence_summary`), plus the shared frequency-mesh helpers. Depends only on `gf_primitives`.
- **`gf_shift_recycling.py`** — `SectorResolventCache` (dense spectral cache over a closed H-sector) and `KrylovShiftedResolvent` (one distributed block-Lanczos recurrence serving every shift of a fixed right-hand side): the two tiers ahead of the per-point BiCGSTAB/GMRES fallback in the RIXS R1 solver chain. Depends only on `gf_primitives`.
- **`greens_function.py`** — interacting Green's functions via block Lanczos continued fractions; `run_units_distributed` is the one distribution primitive shared by every GF driver (self-energy and spectra). Re-exports every symbol of the three modules above that other modules/tests reach via `greens_function.X` / `gf.X`, so it stayed a drop-in for existing callers when it was split into them.
- **`spectra.py`** — XAS/XPS/PS/NIXS/RIXS spectra drivers on top of `greens_function`. PS/XPS/NIXS and the projector-driven XAS/RIXS paths return per-operator spectra directly; the default (unprojected) XAS/RIXS paths return the polarization *tensor* (`getSpectra_tensor`, `getRIXSmap_tensor`) rather than a polarization-contracted spectrum -- `simulate_spectra` stores the tensor as-is (`spectra.h5`: `XAS/tensor`, `RIXS/tensor`), and `polarization.py` contracts it with concrete polarizations as a cheap post-processing step (in the projector paths, or at plot time).
- **`cg.py`** — block BiCGSTAB solver (used by the RIXS tensor path).
- **`gf_diagnostics.py`** — convergence/consistency diagnostics for computed Green's functions.
- **`gs_statistics.py`** — ground-state statistics computation, printing, and saving.
- **`hamiltonian_io.py`** — construction and file I/O of the impurity Hamiltonian: readers for pickled/`.dat`/`.json` h0 formats and the builders combining h0 with SOC, magnetic field, Coulomb, and double counting.

### CLIs (Layer 6)
- **`get_spectra.py`** (`python -m impurityModel.ed.get_spectra`) — find the lowest eigenstates, then calculate spectra (PS, XPS, XAS, NIXS, RIXS).
- **`selfenergy.py`** (`python -m impurityModel.ed.selfenergy`) — impurity self-energy calculation (for DMFT-style workflows).

### Bath construction (used by the `build_h0` script)
- **`edchain.py`** — transformation of star-geometry baths into chain geometries (Wilson chain, double chains, linked double chains).
- **`natural_orbitals.py`** — hybridization fitting in a natural-orbital basis.
- **`bath_fitting.py`** — hybridization-function bath fitting helpers.
- **`scripts/build_h0.py`** (console script `build_h0`) — builds a non-interacting Hamiltonian from RSPt output; requires the `rspt` extra (`pip install -e '.[rspt]'`).

### Plotting (post-processing, `scripts/`)
- **`scripts/_plot_common.py`** — shared CLI plumbing for the plot scripts (input/output/figure-style arguments, `spectra.h5` loading, orbital-selection parsing, `.dat` export), ported from `pyRSPthon.cli._common`.
- **`scripts/plot_spectra.py`** (console script `plot_spectra`) — plots PS/XPS/NIXS from `spectra.h5`, and XAS by contracting the stored spectral tensor with the requested polarizations (`--pol`, default x/y/z + isotropic; `--xmcd`/`--xld` dichroism; `--tensor-components`) via `polarization.py`; also overlays the RIXS-tensor fluorescence yield when both are present.
- **`scripts/plot_RIXS.py`** (console script `plot_RIXS`) — plots the RIXS map from `spectra.h5`'s `RIXS/tensor`, contracting with `--pol-in`/`--pol-out` polarization pairs, `--mcd` circular dichroism, `--fy` fluorescence yield, and `--cuts`/`--emission` energy-loss line cuts, all as post-processing (no solver re-run).

## Test Suite (`src/impurityModel/test/`)

- **Framework:** `pytest` + `pytest-mpi`. Serial run: `pytest`. MPI run: `mpiexec -n 2 python -m pytest --with-mpi` (CI runs serial, 1 rank, and 2 ranks).
- **MPI tests** are marked `@pytest.mark.mpi`; `conftest.py` redirects non-root-rank output to `.pytest_mpi_rank*.out`, adds a per-test watchdog, and synchronizes teardown.
- **Benchmarks** are marked `benchmark` and skipped by default; run with `pytest -m benchmark`.

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
