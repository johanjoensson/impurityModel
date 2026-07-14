Introduction
============

``impurityModel`` solves finite-temperature Anderson impurity models by exact
diagonalization. An impurity shell (with a full multiplet Coulomb interaction, spin--orbit
coupling, and a crystal/ligand field) is coupled to a discretized bath; the code builds the
many-body ground state, then computes spectroscopic and thermodynamic observables from it.

What it computes
----------------

- **Ground state and observables** --- the low-energy eigenstates via block-Lanczos
  diagonalization, plus occupations and angular-momentum / Casimir expectation values.
- **Spectra** --- photoemission (PS), x-ray photoemission (XPS), x-ray absorption (XAS),
  non-resonant inelastic x-ray scattering (NIXS), and resonant inelastic x-ray scattering
  (RIXS), each as an interacting Green's function built from block-Lanczos continued
  fractions.
- **Self-energy** --- the impurity self-energy for DMFT-style workflows.

How it scales
-------------

The many-body Hilbert space is enormous, so the code never forms a full state vector.
Slater determinants are hash-distributed across MPI ranks (one owner per determinant), the
basis is grown adaptively by selected-CI (CIPSI) and bounded by occupation restrictions, and
the Hamiltonian is applied matrix-free through C++/Cython kernels. A single distribution
engine splits the work across ranks for every ground-state, spectra, and self-energy run.

Where to go next
----------------

- New here? Start with the :doc:`user_guide` --- install, inputs, running, and plotting.
- Want to understand the code? Read the :doc:`architecture_overview`, then the
  :doc:`gf_solver_architecture` (the most intricate part) and :doc:`basis_and_restrictions`.
- Working on parallelism? Read :doc:`mpi_model` first.
- Tuning a run? The :doc:`configuration` reference lists every environment-variable knob.

See also the ``README.md`` at the repository top level.
