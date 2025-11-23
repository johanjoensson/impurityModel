"""
Module containing functions for performing averaging.
"""

import numpy as np
import scipy as sp

k_B = sp.constants.physical_constants["Boltzmann constant in eV/K"][0]


def thermal_average(energies, observable, T=300):
    """
    Returns thermally averaged observables.

    Assumes all relevant states are included.
    Thus, no not check e.g. if the Boltzmann weight
    of the last state is small.

    Parameters
    ----------
    energies - list(N)
        energies[i] is the energy of state i.
    observable - list(N,...)
        observable[i,...] are observables for state i.
    T : float
        Temperature

    """
    return thermal_average_scale_indep(energies, observable, k_B * T)


def thermal_average_scale_indep(energies, observable, tau):
    if isinstance(energies, float):
        energies = np.array(energies)
    elif not isinstance(energies, np.ndarray):
        energies = np.array(energies)
    if energies.shape[0] != observable.shape[0]:
        raise RuntimeError("Passed array is not of the right shape")
    e0 = np.min(energies)
    weights = np.exp(-(energies - e0) / tau)
    o_average = np.sum(np.expand_dims(weights, tuple(range(1, observable.ndim))) * observable, axis=0)
    return o_average / np.sum(weights)
