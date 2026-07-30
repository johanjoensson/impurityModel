"""
Module containing functions for performing averaging.
"""

import numpy as np
import scipy as sp

k_B = sp.constants.physical_constants["Boltzmann constant in eV/K"][0]

#: Boltzmann-weight threshold defining "negligible" occupation of a thermal state relative to
#: the ground state, used to bound how far above ``E0`` the retained eigenstate manifold reaches.
BOLTZMANN_DESIGN_WEIGHT = 1e-4


def energy_cut(tau, design_weight=BOLTZMANN_DESIGN_WEIGHT):
    """The energy above ``E0`` at which a thermal state's Boltzmann weight falls to ``design_weight``.

    Single source of truth for ``-tau * log(design_weight)``, which every eigensolver call that
    retains "the low-energy manifold at scale ``tau``" derives independently otherwise.

    Parameters
    ----------
    tau : float
        The characteristic energy scale (e.g., ``k_B * T``).
    design_weight : float, optional
        The Boltzmann weight, relative to the ground state, considered negligible.

    Returns
    -------
    float
    """
    return -tau * np.log(design_weight)


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
    """
    Return the thermal average of an observable, using energy scale tau.

    Parameters
    ----------
    energies : array_like of shape (N,)
        The energies of the states.
    observable : array_like of shape (N, ...)
        The observable values for each state.
    tau : float
        The characteristic energy scale (e.g., k_B * T).

    Returns
    -------
    o_average : ndarray
        The thermally averaged observable.
    """
    if isinstance(energies, float) or not isinstance(energies, np.ndarray):
        energies = np.array(energies)
    if energies.shape[0] != observable.shape[0]:
        raise RuntimeError("Passed array is not of the right shape")
    e0 = np.min(energies)
    weights = np.exp(-(energies - e0) / tau)
    o_average = np.sum(np.expand_dims(weights, tuple(range(1, observable.ndim))) * observable, axis=0)
    return o_average / np.sum(weights)
