"""
Module containing functions for performing averaging.
"""

import numpy as np

# Boltzmann constant. Unit: eV/K. E = k_B * T,
# energy in eV and temperature in Kelvin.
# k_B = 8.61701580807947e-05
k_B = 6.3334e-06


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
    if len(energies) != np.shape(observable)[0]:
        raise ValueError("Passed array is not of the right shape")
    e0 = np.min(energies)
    weights = np.exp(-(energies - e0) / tau)
    o_average = np.sum([w * o for w, o in zip(weights, observable)], axis=0)
    return o_average / np.sum(weights)
