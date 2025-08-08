"""
A package dealing with many-body impurity models using exact diagonalization.

    Examples of functionalities:
        - Calculate spectra, e.g. XAS, XPS, PS.
        - Calculate static expectation values

"""

from os import environ

environ["OMP_NUM_THREADS"] = "1"
