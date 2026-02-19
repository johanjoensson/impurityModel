from itertools import product
import numpy as np
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, inner
from impurityModel.ed.finite import thermal_average_scale_indep as thermal_average


def calc_density_matrix(psi: ManyBodyState, orbital_indices: list[int]):
    n_orb = len(orbital_indices)
    rho = np.zeros((n_orb, n_orb), dtype=complex)
    for (i, orb_i), (j, orb_j) in product(enumerate(orbital_indices), repeat=2):
        op = ManyBodyOperator({((orb_j, "c"), (orb_i, "a")): 1.0})
        psi_p = op(psi)
        amp = inner(psi, psi_p)
        if np.abs(amp) > np.finfo(float).eps:
            rho[i, j] = amp

    return rho


def calc_density_matrices(psis: list[ManyBodyState], orbital_indices: list[int]):
    n_psi = len(psis)
    n_orb = len(orbital_indices)
    rhos = np.empty((n_psi, n_orb, n_orb), dtype=complex)
    for i, psi in enumerate(psis):
        rhos[i] = calc_density_matrix(psi, orbital_indices)
    return rhos
