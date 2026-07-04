# Impurity model

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/johanjoensson/impurityModel/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/johanjoensson/impurityModel/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/johanjoensson/impurityModel/badge.svg)](https://coveralls.io/github/johanjoensson/impurityModel)

# Introduction

Calculate many-body states of an impurity Anderson model and a various spectra, e.g. photoemission spectroscopy (PS), x-ray photoemission spectroscopy (XPS), x-ray absorption spectroscopy (XAS), non-resonant inelastic x-ray scattering (NIXS), and resonant inelastic x-ray scattering (RIXS), using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm).

This package is a pure many-body solver: it takes a non-interacting Hamiltonian h0 (impurity + bath), a Coulomb interaction and a temperature, and produces self-energies, Green's functions, double countings and spectra. Constructing h0 from DFT output (hybridization fitting, bath geometries) is the job of upstream tooling such as [rspt2spectra](https://github.com/johanjoensson/rspt2spectra); this package has no dependency on it. External consumers should import from the stable `impurityModel.api` module only — everything under `impurityModel.ed.*` is internal and may change without notice.


## Getting started

### Installation
If you have downloaded the github archive, install using pip
```bash
pip install .
```

If you are a developer, you can do an editable installation
```bash
pip install -e .
```
this way you can make changes to the code, and they will immediately be included in your Python environment.

Optional extras:
```bash
pip install -e '.[dev]'   # test + lint toolchain (pytest, pytest-mpi, black, ruff, mypy, cython-lint)
pip install -e '.[doc]'   # Sphinx documentation toolchain
```
You can also use pip to install directly from github, without cloning the repository (and without being able to edit the code)
```bash
pip install git+https://github.com/johanjoensson/impurityModel
```

#### Cython & C++
The performance critical part of this code is the (Block) Lanczos method. In order to make the simulations fast (or as fast as possible at least)
this code uses classes and methods written in C++, this is then wrapped using Cython to make everything callable from within Python.
The C++ code needs to be compiled, and you can pass compilation flags, your preferred C++ compiler, etc. to Cython.
The compiler to use can be specified in the `CXX` environment variable (e.g. `export CXX=g++` for the GCC C++ compiler).
Any compilation flags you want to specify can be passed using the `CXXFLAGS` environment variable. The C++ code depends on the Boost library, but only the headers.
If your system has a Boost installation in any of the "standard" location, you probably don't need to do anything. If you have Boost installed in a custom location
you can let the compiler know via the `BOOST_ROOT` environment variable (i.e. `export BOOST_ROOT="<PATH_TO_CUSTOM_BOOST>"`), or you can install
the headers using `pip install boost-headers`. With all this set up, `pip install` will be able to compile the C++/Cython sources for you.

##### Thread parallel execution
The code can use multiple threads to speed up the C++ code further (the application of Hamiltonian to state mainly), to turn this on set the `IMPURITYMODEL_PARALLEL=1` environment variable when installing (this adds `-DPARALLEL -pthread` to both the compile and link steps):
```bash
IMPURITYMODEL_PARALLEL=1 pip install --no-build-isolation -e .
```
For small systems this might actually hurt performance a bit, but larger systems can benefit greatly (if you have multiple threads available).

Note: `--no-build-isolation` reuses the packages already installed in your environment instead of provisioning the build requirements declared in `pyproject.toml`. If you use it, make sure the build prerequisites are installed first, otherwise the Cython compilation (`cimport numpy`, `from scipy.linalg.cython_blas cimport zgemm`) and the setuptools-scm version lookup will fail:
```bash
pip install numpy scipy cython "setuptools>=77.0.3" setuptools-scm
```
A plain `pip install -e .` (without `--no-build-isolation`) installs these automatically and needs no such preparation.

##### C++ standards and their implications
Also, you can specify what
C++ standard you want to compile with (the code requires at least C++17), for the GCC and Clang compilers the flag is `-std=c++17` (set by default). C++20 includes some
useful potential optimizations that are turned on if you specify `-std=c++20` (or whatever flag your compiler requires). Finally, if you have access to C++23 the code
will try to use the `std::flat_map` container (by default it uses the boost flat_map implementation).

For more detailed information of the code architecture please see [the architecture overiew](doc/architecture_overview.md)

### Testing
The code comes with a test suite, to ensure that everything is running properly. To run the serial tests the command is simply `pytest`.
The code also contains tests that verify that the MPI parallelization is working correctly. To run the MPI tests, as well as the serial ones, use
```bash
mpiexec -n 2 python -m pytest --with-mpi
```
(replace 2 with however many MPI ranks you want to use; CI runs the suite serially, with 1 rank, and with 2 ranks).

Performance benchmarks (timing/profiling, not correctness) are marked `benchmark` and skipped by default. Run them explicitly with `pytest -m benchmark`. The self-energy benchmark additionally requires `RUN_SELFENERGY_BENCH=1` (and accepts `SELFENERGY_BENCH_*` env vars to size the workload).


# First X-ray spectra simulations

- To perform a simulation, first create a directory somewhere on your computer.
Move to that directory and then execute one of the example scripts in the `scripts` folder. E.g. type:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_Xbath.sh
```
This will start a simulation with 10 bath states and one MPI rank.
To have e.g. 20 bath states instead of 10, instead type:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_Xbath.sh 20
```
To have e.g. 20 bath states and 3 MPI ranks, instead type:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_Xbath.sh 20 3
```
These examples will read an non-interacting Hamiltonian from file.

A simpler non-interacting Hamiltonian can instead be constructed from crystal-field parameters.
This is done for NiO by typing:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_CF.sh
```
(similar crystal-field examples exist for MnO, FeO and CoO in the same folder).
#### Output files
The input parameters to the simulation are saved in `.npz` format.
Calculated spectra are saved to the file `spectra.h5`.
Some small size spectra are also stored in `.dat` and `.bin` format, for easy and fast plotting with e.g. gnuplot.
For plotting all generated spectra (using matplotlib), type:
```bash
python -m impurityModel.plotScripts.plotSpectra
```
For only plotting the RIXS map, type:
```bash
python -m impurityModel.plotScripts.plotRIXS
```
Using Gnuplot, instead type:
```bash
path/to/folder/impurityModel/impurityModel/plotScripts/plotRIXS.plt
```


### Documentation
The documentation of this package is found in the directory `doc`.

To build the manual, install the documentation toolchain (`pip install -e '.[doc]'`) and type:

```bash
make -s -C doc/sphinx clean
make -s -C doc/sphinx html
```
Open the generated `doc/sphinx/generated_doc/html/index.html` in a web browser.

### Contributors

Call for contributions: The impurityModel project welcomes your expertise and enthusiasm!

Contributors (from the original impurityModel repo):
- Johan Schött (@JohanSchott): Implemented many of the functionalities needed to solve the impurity problem.
- Johan Jönsson (@johanjoensson): Implementented the entire DMFT cycle in the fork https://github.com/johanjoensson/impurityModel (This repo). Also developed the related repo: https://github.com/johanjoensson/rspt2spectra
- Felix Sorgenfrei (@fesorg): First author in [publication](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.115126) using the impurityModel repo. Has also made the related repo: https://github.com/fesorg/Tutorial-X-ray-from-RSPt
- Patrik Thunström (@patrikthunstrom): Involved in discussions about computational algorithms, reported bugs, and has provided theoretical knowledge and inspiration.
- Petter Säterskog (@PetterSaterskog): Written some of the initial key functionalities.
- Christian Häggström (@kalvdans): Has provided valuable reviews on PRs.
- Mébarek Alouani: Has provided theoretical knowledge and inspiration.
- Olle Eriksson: Has provided theoretical knowledge and inspiration.
- Igor Di Marco (@igordimarco): Has provided theoretical knowledge and inspiration.

Please note that the list and the contribution information are incomplete.

### Note
Please note that this fork has diverged significantly from the original version of the code. The two versions are not compatible, or interchangeable.
