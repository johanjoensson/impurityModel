# ImpurityModel Architecture Overview

This document provides an overview of the architecture of the `impurityModel` codebase, detailing the roles of the Cython extension classes, the Python classes, and the various routines.

## Cython Extensions (`src/cython/`)

The core of the performance-critical operations in `impurityModel` is implemented in C++ and exposed to Python via Cython. This allows for high-performance manipulation of quantum many-body states and operators.

### Key Classes:
1. **`SlaterDeterminant`**
   - **Role:** Represents a many-body Slater determinant using 64-bit integer chunks to track spin-orbital occupations.
   - **Details:** This class wraps `std::vector<uint64_t>`. It uses bit manipulation to compactly represent fermion occupation numbers, allowing for very fast operations like applying creation/annihilation operators and comparing basis states.

2. **`ManyBodyState`**
   - **Role:** Represents a quantum many-body state as a superposition of Slater determinants.
   - **Details:** Wraps a custom C++ `flat_map<SlaterDeterminant, std::complex<double>>`. It acts essentially as a highly-optimized dictionary mapping basis states (Slater determinants) to their complex probability amplitudes. It implements vectorized operations like addition, scalar multiplication, and inner products.

3. **`ManyBodyOperator`**
   - **Role:** Represents a quantum many-body operator composed of creation and annihilation sequences with corresponding amplitudes.
   - **Details:** Maps tuples of integers representing creation/annihilation operators (e.g., $c^\dagger_i c_j$) to complex amplitudes. It provides a `__call__` method to apply the operator to a `ManyBodyState`, returning a new `ManyBodyState`. It's highly optimized in C++ to compute these sparse matrix-vector products quickly.

4. **MPI Utilities**
   - **Role:** Facilitate efficient parallelization across nodes.
   - **Details:** Functions like `pack_determinants_cy` and `unpack_psis_cy` serialize and deserialize `ManyBodyState` objects into contiguous NumPy arrays (`uint64`, `double`), allowing fast communication via `mpi4py`.

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

## Python Codebase (`src/impurityModel/ed/`)

The Python codebase builds upon the Cython extensions to implement exact diagonalization (ED) algorithms, solvers, and physics-specific logic.

### Key Modules and Classes:
1. **State Containers (`manybody_state_containers.py`)**
   - **Role:** Higher-level abstractions for state vectors.
   - **Details:** Defines classes that wrap `ManyBodyState` objects to interface seamlessly with iterative eigensolvers (like SciPy's ARPACK or custom Lanczos solvers). They handle block structure logic, allowing the Hamiltonian to be diagonalized block-by-block.

2. **Basis and Block Structure (`manybody_basis.py`, `block_structure.py`)**
   - **Role:** Define the Hilbert space and exploit symmetries.
   - **Details:** `manybody_basis.py` manages the generation of the Fock space basis. `block_structure.py` implements logic to partition the basis into non-interacting blocks (e.g., by particle number or $S_z$ sectors, or whatever symmetries it can find) to reduce the size of the matrices that need to be diagonalized.

3. **Eigensolvers (`lanczos.py`, `finite.py`, `cipsi_solver.py`, `trlm.py`, `cg.py`)**
   - **Role:** Find the ground state and excited states.
   - **Details:**
     - `finite.py`: Contains driver routines `eigensystem` and `dense_eigensystem`. Uses `scipy.sparse.linalg` (using `scipy.sparse.linalg.eigsh`) and dense matrix diagonalization (using `scipy.linalg.eigsh`).
     - `lanczos.py`: Implement the (Block) Lanczos algorithm. This is the workhorse of this repository, this is used in the non-`scipy` eigensolvers, in generating the interacting Greens functions for calculating spectra and self-energies.
     - `trlm.py`: Thick-Restart Lanczos Method for sparse symmetric/Hermitian matrices, in theory an efficient way to find extreme eigenvalues of large systems. This does not currently work correctly, so the code falls back to the finit.py implementations.
     - `irlm.py`: Implicitly Restarted Lanczos Method for sparse symmetric/Hermitian matrices, in theory an efficient way to find extreme eigenvalues of large systems. This does not currently work correctly, so the code falls back to the finit.py implementations.
     - `cipsi_solver.py`: Implements Configuration Interaction using a Perturbative Selection Iteratively (CIPSI) to selectively expand the active Hilbert space. Used for finding the ground state in an efficient manner.
     - `cg.py`: Implements a Conjugate Gradient solver for iterative numerical solutions.Used e.g. when calculating the RIXS spectra.

4. **Spectra and Green's Functions (`greens_function.py`, `spectra.py`, `get_spectra.py`, `selfenergy.py`)**
   - **Role:** Calculate observable physical quantities.
   - **Details:** After finding the ground state, these modules compute dynamical properties like the single-particle Green's function, density of states, self-energy (using Dyson's equation logic), or X-ray absorption spectra (XAS) by applying relevant operators to the ground state and computing continued fractions or using the Krylov space.

5. **Impurity Models and Chains (`edchain.py`)**
   - **Role:** Construct specific Hamiltonian models (e.g., Anderson Impurity Model).
   - **Details:** Defines bath geometries. Contains logic for transforming models from star geometries into various chain geometries (a single (Wilson) chain, double chains for nominally occupied/unoccupied bath states, or a linked double chain geometry) in order to hopefully reduce the size of the manybody basis required in the calculations.

## Test Suite Architecture (`src/impurityModel/test/`)

The testing framework utilizes `pytest`, `pytest-cov`, and `pytest-mpi` to ensure the codebase's mathematical and functional integrity across different platforms and compilation environments.
- **Coverage:** High global coverage (>80%) across standard ED solvers and physics quantities (Spectra, Self Energy, Green's Functions).
- **Parallel Testing:** Handled by utilizing `@pytest.mark.mpi` for all functions that interact via `MPI.COMM_WORLD`, ensuring isolation and preventing test-collection deadlocks when run under `mpirun`.
- **Mocks:** Broad use of dependency mocking (e.g., mocking ED eigenvalue returns) to verify integration logic for derived quantities like `get_sigma` without solving full dense eigenproblems in standard testing.

## Overview of Execution Flow
1. **Define Model:** A user defines an impurity model (orbitals, hoppings, interactions).
2. **Construct Basis:** The codebase generates the relevant `SlaterDeterminant` basis states and splits them into decoupled blocks.
3. **Build Hamiltonian:** `ManyBodyOperator` objects are instantiated for the Hamiltonian.
4. **Diagonalization:** A state vector is initialized, and iterative solvers repeatedly apply the Hamiltonian operator (`H(psi)`) to find the ground state (`ManyBodyState`).
5. **Observables:** Physical observables and spectral functions are calculated using the ground state and relevant excitation operators.
