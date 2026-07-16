# User guide

This guide is for running the package as a tool: installing it, feeding it a model, computing
spectra or a self-energy, and plotting the result. For how the code is organized internally see
[`architecture_overview.md`](architecture_overview.md); for the physics see
[`greens_function_theory.md`](greens_function_theory.md).

## Install

```bash
pip install numpy scipy cython "setuptools>=77.0.3" setuptools-scm
pip install --no-build-isolation -e .
```

`--no-build-isolation` requires the build prerequisites (first line) to already be present. The
C++/Cython layer needs Boost headers (set `BOOST_ROOT` for a non-standard location) and a
C++17 compiler (`CXX` / `CXXFLAGS` are respected). Editing anything under `src/cython/` requires
re-running the `pip install` to recompile.

Optional extras: `.[dev]` (pytest, pytest-mpi, black, ruff, mypy, cython-lint), `.[doc]`
(Sphinx). A threaded operator-apply is a compile-time option — install with
`IMPURITYMODEL_PARALLEL=1` (intended for few-rank-many-core nodes; do **not** combine it with
one MPI rank per core, which oversubscribes).

## The CLI

Everything is driven by one umbrella command with sub-commands:

```bash
impurityModel spectra         <h0> [radial] [options]   # PS/XPS/NIXS/XAS/RIXS -> spectra.h5
impurityModel selfenergy      <h0>          [options]   # Sigma(w)/Sigma(i nu), G_imp
impurityModel susceptibility  <h0>          [options]   # chi(w)/chi(i nu) -> chi.h5
impurityModel plot-spectra    spectra.h5    [options]   # plot from spectra.h5 (no re-solve)
impurityModel plot-rixs       spectra.h5    [options]
```

`python -m impurityModel <subcommand> ...` is an equivalent entry point when the console script
is not on `PATH`. Every sub-command runs identically under MPI (`mpiexec -n N ...`). See
`impurityModel <subcommand> --help` for the full option set.

## Inputs

1. **The non-interacting Hamiltonian `h0`** — a single-particle Hamiltonian read by
   `hamiltonian_io.py`. Supported formats: a pickled operator dict, a `.dat` matrix, or `.json`.
   The impurity orbitals come first, then the bath orbitals. The `spectra` sub-command combines
   it with the interaction (Slater–Condon `Fdd`/`Fpp`/`Fpd`/`Gpd`), spin–orbit coupling (`xi_2p`,
   `xi_3d`), a magnetic field, and the double-counting correction at runtime.
2. **The radial file** *(optional)* — the radial mesh and the radial part of the correlated
   orbitals, used by the NIXS excitation. Omit it and NIXS is skipped; every other spectrum is
   independent of it.

Alternatively, `selfenergy`/`susceptibility` can take `--from-archive impurityModel_data.h5`
(the file the RSPt interface writes) instead of an `h0` file — the whole model comes from the
archive.

## Running a spectra calculation

The command is `impurityModel spectra <h0_filename> [radial_filename] [options]`.
The key options (all have defaults; see `--help` for the full list):

| Option | Meaning |
| --- | --- |
| `--ls 1 2` | Angular momenta of the correlated orbitals (here p and d). |
| `--nBaths 0 10` | Bath states per angular momentum. |
| `--nValBaths 0 10` | Valence (occupied) bath states per angular momentum. |
| `--n0imps` | Nominal impurity occupation. |
| `--Fdd --Fpp --Fpd --Gpd` | Slater–Condon Coulomb parameters. |
| `--xi_2p --xi_3d` | Spin–orbit coupling strengths. |
| `--chargeTransferCorrection` | Double-counting parameter. |
| `--T 300` | Temperature (Kelvin). |
| `--energy_cut` | How many `k_B·T` above the ground state to keep. |
| `--delta --deltaRIXS --deltaNIXS` | Broadenings (HWHM). |
| `--nPsiMax` | Maximum number of eigenstates. |
| `--truncation_threshold` | Global cap on determinants per basis (memory control). |
| `--no-auto-block-structure` | Keep the hand-coded block structure instead of deriving it. |

Run it under MPI for anything nontrivial. A worked NiO L-edge invocation (from
`scripts/run_Ni_NiO_Xbath.sh`):

```bash
h0_filename=h0/h0_NiO_10bath.pickle
radial_filename=radialOrbitals/Ni3d.dat

mpiexec -n 8 impurityModel spectra \
    "$h0_filename" "$radial_filename" \
    --nBaths 0 10 --nValBaths 0 10
```

The runtime `GF_*` tuning knobs (solver method, memory caps, sector cache, adaptive RIXS
sampling) are environment variables — see [`configuration.md`](configuration.md). For example,
persist the RIXS sector eigendecompositions across runs with `GF_SECTOR_CACHE_DIR=/scratch/...`.

## Outputs: `spectra.h5`

A run writes a single HDF5 file, `spectra.h5`, one group per spectrum:

| Dataset | Contents |
| --- | --- |
| `PS/spectra` | Photoemission, per operator. |
| `XPS/spectra` | X-ray photoemission, per operator. |
| `NIXS/spectra` | Non-resonant inelastic x-ray scattering, per momentum transfer. |
| `XAS/tensor` *or* `XAS/projected` | XAS as the Cartesian polarization **tensor** (default) or per-projector spectra (with a projector file). |
| `RIXS/tensor` *or* `RIXS/projected` | RIXS as the Kramers–Heisenberg **tensor** (default) or per-projector maps. |

The tensor form stores the polarization-independent quantity; contracting it with a concrete
polarization is a cheap post-processing step, so you can re-plot with any polarization without
re-running the solve. No quick-look `.dat`/`.bin` files are written.

## Plotting

Two sub-commands post-process `spectra.h5` (no solver re-run); the standalone `plot_spectra` /
`plot_RIXS` console scripts are equivalent:

```bash
impurityModel plot-spectra spectra.h5 --pol x y z          # PS/XPS/NIXS, and XAS by contracting the tensor
impurityModel plot-rixs    spectra.h5 --pol-in x --pol-out y   # the RIXS map from RIXS/tensor
```

`plot-spectra` contracts the stored XAS tensor with the requested polarizations (`--pol`,
`--xmcd`/`--xld` for dichroism, `--tensor-components`) and can overlay the RIXS fluorescence
yield. `plot-rixs` contracts `RIXS/tensor` with `--pol-in`/`--pol-out` pairs and supports
`--mcd`, `--fy` (fluorescence yield), and `--cuts`/`--emission` energy-loss line cuts. Both
export `.dat` on request. See `--help` on each for the full option set.

## Self-energy (DMFT workflows)

`impurityModel selfenergy <h0>` computes the impurity self-energy for DMFT-style workflows. It
shares the ground-state and Green's-function machinery with `spectra`. Choose the frequency axes
with `--w_min/--w_max/--w_n/--delta` (real) and `--n_matsubara` (fermionic Matsubara); the run
writes the frequency-dependent Σ and impurity G in RSPt `.dat` format, the static Σ to a `.dat`
file, and a per-cluster HDF5 archive. `--from-archive impurityModel_data.h5 [--cluster L]`
reproduces a recorded DFT-embedded run (model, meshes and solver options all come from the
archive). The double-counting is fixed by pinning either a spectral peak or the impurity
occupation (`double_counting.py`).

## Susceptibilities

`impurityModel susceptibility <h0>` computes the dynamical impurity susceptibilities
(spin / orbital / charge / transverse) on a real mesh and the bosonic Matsubara mesh, writing
`chi.h5` and a Curie/Van-Vleck/screening-scale summary (the Hund's-metal diagnostic). It also
accepts `--from-archive`.

## Python API

The CLIs are thin wrappers over a small API (`impurityModel.api`). Build an `ImpurityModel` plus
option groups and call the solver directly — this is how the RSPt interface embeds the solver
in memory, no files involved:

```python
from mpi4py import MPI
from impurityModel.api import ImpurityModel, Meshes, BasisOptions, SolverOptions, calc_selfenergy

model = ImpurityModel(h0=h0_operator, u4=u4_tensor, impurity_orbitals={0: list(range(10))},
                      rot_to_spherical=rot)
result = calc_selfenergy(
    model,
    Meshes(iw=1j * matsubara_nu, w=real_mesh, delta=0.1),
    BasisOptions(nominal_occ={0: 8}, tau=0.002),
    SolverOptions(gf_method="lanczos"),
    comm=MPI.COMM_WORLD,
)
```

`ImpurityModel.from_h0_file(...)` builds the model from a file, and
`ImpurityModel.from_hdf5(archive)` / `load_selfenergy_archive(archive)` rebuild it (and the
option groups) from an `impurityModel_data.h5` archive.

## Sanity check

A collapsed basis is silent — it produces a zero spectrum, not an error. After a run, confirm
the spectrum is nonzero (e.g. `max|XAS| > 0`) and the impurity occupation is what you expect
before trusting the result. See the "d10 collapse" discussion in
[`basis_and_restrictions.md`](basis_and_restrictions.md).
