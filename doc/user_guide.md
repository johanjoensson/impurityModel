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

## Inputs

A calculation needs two files (positional arguments to `get_spectra`):

1. **The non-interacting Hamiltonian `h0`** — a single-particle Hamiltonian read by
   `hamiltonian_io.py`. Supported formats: a pickled operator dict, a `.dat` matrix, or `.json`.
   The impurity orbitals come first, then the bath orbitals. `get_spectra` combines it with the
   interaction (Slater–Condon `Fdd`/`Fpp`/`Fpd`/`Gpd`), spin–orbit coupling (`xi_2p`, `xi_3d`), a
   magnetic field, and the double-counting correction at runtime.
2. **The radial file** — the radial mesh and the radial part of the correlated orbitals, used by
   the NIXS excitation.

Optionally, XAS and RIXS **projector files** select specific transitions; without them the
default paths compute the full polarization tensor (see Outputs).

## Running a spectra calculation

The CLI is `python -m impurityModel.ed.get_spectra <h0_filename> <radial_filename> [options]`.
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

Run it under MPI for anything nontrivial. A worked NiO L-edge invocation (from
`scripts/run_Ni_NiO_Xbath.sh`):

```bash
h0_filename=h0/h0_NiO_10bath.pickle
radial_filename=radialOrbitals/Ni3d.dat

mpiexec -n 8 python -m impurityModel.ed.get_spectra \
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

Two console scripts post-process `spectra.h5` (no solver re-run):

```bash
plot_spectra spectra.h5 --pol x y z         # PS/XPS/NIXS, and XAS by contracting the tensor
plot_RIXS   spectra.h5 --pol-in x --pol-out y   # the RIXS map from RIXS/tensor
```

`plot_spectra` contracts the stored XAS tensor with the requested polarizations (`--pol`,
`--xmcd`/`--xld` for dichroism, `--tensor-components`) and can overlay the RIXS fluorescence
yield. `plot_RIXS` contracts `RIXS/tensor` with `--pol-in`/`--pol-out` pairs and supports
`--mcd`, `--fy` (fluorescence yield), and `--cuts`/`--emission` energy-loss line cuts. Both
export `.dat` on request. See `--help` on each for the full option set.

## Self-energy (DMFT workflows)

`python -m impurityModel.ed.selfenergy` computes the impurity self-energy for DMFT-style
workflows. It shares the ground-state and Green's-function machinery with `get_spectra`; the
double-counting is fixed by pinning either a spectral peak or the impurity occupation
(`double_counting.py`). The `--help` output lists its arguments.

## Sanity check

A collapsed basis is silent — it produces a zero spectrum, not an error. After a run, confirm
the spectrum is nonzero (e.g. `max|XAS| > 0`) and the impurity occupation is what you expect
before trusting the result. See the "d10 collapse" discussion in
[`basis_and_restrictions.md`](basis_and_restrictions.md).
