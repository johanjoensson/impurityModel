# Impurity model

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/johanjoensson/impurityModel/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/johanjoensson/impurityModel/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/johanjoensson/impurityModel/badge.svg)](https://coveralls.io/github/johanjoensson/impurityModel)

### Introduction

Calculate many-body states of an impurity Anderson model and a various spectra, e.g. photoemission spectroscopy (PS), x-ray photoemission spectroscopy (XPS), x-ray absorption spectroscopy (XAS), non-resonant inelastic x-ray scattering (NIXS), and resonant inelastic x-ray scattering (RIXS), using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm).


<figure>
<div class="row">
  <div class="column">
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/ps.png" alt="Photoemission (PS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/xps.png" alt="X-ray photoemission (XPS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/xas.png" alt="X-ray absorption spectroscopy (XAS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/nixs.png" alt="Non-resonant inelastic x-ray scattering (NIXS)" width="150"/>
  <img src="impurityModel/test/referenceOutput/Ni_NiO_50bath/rixs.png" alt="Resonant inelastic x-ray scattering (RIXS)" width="150"/>  </div>
</div>
<figcaption>Spectra of NiO. Simulated using 50 bath orbitals coupled to the Ni 3d orbitals.</figcaption>
</figure>

### Get started
If you have downloaded the github archive, install using pip
```bash
pip install .
```

If you are a developer, you can do an editable installation
```bash
pip install -e .
```
this way you can make changes to the code, and they will immediately be included in your Python environment.

You can also use pip to install directly from github, without cloning the repository (and without being able to edit the code)
```bash
pip install git+https://github.com/johanjoensson/impurityModel
```

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

A simpler non-interacting Hamiltonian can instead be constructed by crystal-field parameters.
This is done for NiO by typing:
```bash
path/to/folder/impurityModel/scripts/run_Ni_NiO_CFparam.sh
```
Although using a crystal-field approach is a bigger approximation, it is convinient when doing fitting to experimental spectra.
But for more accurate simulations it is better to read in a non-interacting Hamiltonian from file, that has been constructed from e.g. DFT or DFT+DMFT simulations.
The non-interacting Hamiltonians read from file by the scripts `run_Ni_NiO_Xbath.sh` and `run_Ni_NiO_Xbath.sh` have been constructed using non-spin polarized DFT calculations.

- The bash-scripts in the `scripts`-folder act as templates and can easily be modified. For example, to set the temperature to 10 Kelvin in `get_spectra.py`, add `--T 10` as input when calling the python-script.

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

### Optimization notes

#### Computational cost and memory usage
The exact diagonalization method scales poorly with the number of manybody basis states included in the calculations, and the number of manybodybasis states included in the calculation scales exponentially with the number of single particle (impurity and bath) orbitals included in the calculation.
In order to maintain the manybody basis as small as possible we use the CIPSI method for determining the ground state, and the minimal basis required.
When calculating spectra we need excited states (typically electrons added to or removed from the valence band). The description of these excited states is much harder, and generally requires a significantly larger basis than the groundstate does. In order to keep the basis size from exploding the
code ignores basis states with small weights, and it tries to build the basis and converge the excited state Greens functions at the same time. This increases computational time, but usually keeps the memory usage as low as possible.

The major bottleneck of the method is the repeated application of the Hamiltonian to manybody states, this is done in the CIPSI method, as well as when calculating the Greens functions for the excited states. This code uses Cython to implement ManyBodyOperator and ManyBoddyState, in order to
speed up the application of the Hamiltonian. The Cython code can use threads, by means of a parallel C++ standard library.

To specify a specific C++ compiler, and compiler flags, to use when building the Cython code use the environment variables CXX and CXXFLAGS.

In addition the code uses MPI to divide the computational cost over many CPUs. The code splits the work over the inequivalent block it identifies and the number of eigenstates included in the thermal ground state. In addition the manybody basis states are divided evenly over the MPI ranks.

### Tests
Type
```bash
pytest
```
and
```bash
pytest impurityModel/test/test_comparison_with_reference.py
```
to run all python unit tests in the repository.

### Documentation
The documentation of this package is found in the directory `doc`.

To update the manual type:

```bash
make -s -C doc/sphinx clean
make -s -C doc/sphinx html
```
Open the generated `doc/sphinx/generated_doc/html/index.html` in a web browser.

### Contributors

Call for contributions: The impurityModel project welcomes your expertise and enthusiasm!

Contributors:
- Johan Schött (@JohanSchott): Implemented many of the functionalities needed to solve the impurity problem.
- Johan Jönsson (@johanjoensson): Implementented the entire DMFT cycle in the fork https://github.com/johanjoensson/impurityModel. Also developed the related repo: https://github.com/johanjoensson/rspt2spectra
- Felix Sorgenfrei (@fesorg): First author in [publication](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.115126) using the impurityModel repo. Has also made the related repo: https://github.com/fesorg/Tutorial-X-ray-from-RSPt
- Patrik Thunström (@patrikthunstrom): Involved in discussions about computational algorithms, reported bugs, and has provided theoretical knowledge and inspiration.
- Petter Säterskog (@PetterSaterskog): Written some of the initial key functionalities.
- Christian Häggström (@kalvdans): Has provided valuable reviews on PRs.
- Mébarek Alouani: Has provided theoretical knowledge and inspiration.
- Olle Eriksson: Has provided theoretical knowledge and inspiration.
- Igor Di Marco (@igordimarco): Has provided theoretical knowledge and inspiration.

Please note that the list and the contribution information are incomplete.


### Publications using impurityModel

[Theory of x-ray absorption spectroscopy for ferrites](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.115126)

[Re-Dichalcogenides: Resolving Conflicts of TheirStructure–Property Relationship](https://onlinelibrary.wiley.com/doi/epdf/10.1002/apxr.202200010)

### Note
Please note that this fork has diverged significantly from the original version of the code. The two versions are not compatible, or interchangeable.
